import torch.nn as nn
from model import *
import torch
import numpy as np
import os
import random
import argparse
import csv
import json
import math
from dataset import TRANSFORMER_ALL_DATA, TRANSFORMERDATA
from torch.utils.data import DataLoader
from loss import advLoss, masked_mse, latent_regularization, masked_monotonic_loss, masked_degradation_ranking_loss
import itertools
import time

LOG_DIR = None
TRAIN_LOG_PATH = None
EPOCH_LOG_PATH = None
VALID_UNIT_LOG_PATH = None


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def append_csv(path, fieldnames, row):
    is_new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if is_new:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def unique_dir(path):
    if not os.path.exists(path):
        return path
    idx = 1
    while os.path.exists("{}_{}".format(path, idx)):
        idx += 1
    return "{}_{}".format(path, idx)


def log_print(message):
    print(message)
    if TRAIN_LOG_PATH is not None:
        with open(TRAIN_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(str(message) + "\n")


def init_train_logging(args, option_str):
    global LOG_DIR, TRAIN_LOG_PATH, EPOCH_LOG_PATH, VALID_UNIT_LOG_PATH
    run_id = args.run_id if args.run_id else time.strftime("%Y%m%d-%H%M%S")
    run_name = "train_{}_to_{}_type{}_{}".format(args.source, args.target, args.type, run_id)
    LOG_DIR = unique_dir(os.path.join(args.log_dir, run_name))
    os.makedirs(LOG_DIR, exist_ok=True)
    TRAIN_LOG_PATH = os.path.join(LOG_DIR, "train.log")
    EPOCH_LOG_PATH = os.path.join(LOG_DIR, "train_epoch.csv")
    VALID_UNIT_LOG_PATH = os.path.join(LOG_DIR, "train_validate_units.csv")
    write_json(
        os.path.join(LOG_DIR, "config.json"),
        {
            "mode": "train",
            "option": option_str,
            "args": vars(args),
        },
    )
    log_print("logs saved to {}".format(LOG_DIR))


def ramp_weight(epoch, warmup, ramp_epoch, max_factor=1.0):
    if epoch < warmup:
        return 0.0
    if ramp_epoch <= 0:
        return max_factor
    return min(max_factor, float(epoch - warmup + 1) / float(ramp_epoch))


def resolve_latent_split_args(args):
    if args.type != 3 or not args.latent_split:
        return
    if args.deg_latent_dim <= 0 and args.fault_latent_dim <= 0:
        args.deg_latent_dim = max(1, int(round(args.latent_dim * 0.75)))
        args.fault_latent_dim = max(1, args.latent_dim - args.deg_latent_dim)
    elif args.deg_latent_dim <= 0:
        args.deg_latent_dim = max(1, args.latent_dim - args.fault_latent_dim)
    elif args.fault_latent_dim <= 0:
        args.fault_latent_dim = max(1, args.latent_dim - args.deg_latent_dim)
    args.latent_dim = args.deg_latent_dim + args.fault_latent_dim


def clone_state_dict(state_dict):
    return {k: v.detach().clone() for k, v in state_dict.items()}


def clone_model_state(model):
    return clone_state_dict(model.state_dict())


def snapshot_rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def cpu_state_dict(state_dict):
    return {k: v.detach().cpu() for k, v in state_dict.items()}


def update_ema_state(ema_state, model, decay):
    current = model.state_dict()
    for key, value in current.items():
        if torch.is_floating_point(value):
            ema_state[key].mul_(decay).add_(value.detach(), alpha=1.0-decay)
        else:
            ema_state[key].copy_(value)


def validate_with_state(state_dict, epoch=None, preserve_rng=True):
    rng_state = snapshot_rng_state() if preserve_rng else None
    current = clone_model_state(net)
    was_training = net.training
    try:
        net.load_state_dict(state_dict)
        rmse, details = validate(epoch)
        return rmse, details
    finally:
        net.load_state_dict(current)
        net.train(was_training)
        if rng_state is not None:
            restore_rng_state(rng_state)


def validate_units_with_state(state_dict, names, epoch=None, preserve_rng=True):
    global target_test_names
    old_names = target_test_names
    try:
        target_test_names = names
        return validate_with_state(state_dict, epoch, preserve_rng=preserve_rng)
    finally:
        target_test_names = old_names


def save_best_state(state_dict):
    if args.type == 1:
        torch.save(cpu_state_dict(state_dict), "save/final/dann_"+source[-1]+target[-1]+".pth")
    elif args.type == 0:
        torch.save(cpu_state_dict(state_dict), "save/final/out_"+source[-1]+target[-1]+".pth")
    elif args.type == 2:
        torch.save(cpu_state_dict(state_dict), "online/"+source[-1]+target[-1]+"_net.pth")
        torch.save(D1.state_dict(), "online/"+source[-1]+target[-1]+"_D1.pth")
        torch.save(D2.state_dict(), "online/"+source[-1]+target[-1]+"_D2.pth")
    elif args.type == 3:
        torch.save(cpu_state_dict(state_dict), "save/final/latent_"+source[-1]+target[-1]+".pth")
        torch.save(cpu_state_dict(state_dict), "online/"+source[-1]+target[-1]+"_latent_net.pth")


def set_lr(optimizer, value):
    for group in optimizer.param_groups:
        group["lr"] = value


def reduce_lr_on_plateau(optimizer, factor, min_lr):
    old_lr = optimizer.param_groups[0]["lr"]
    new_lr = max(old_lr*factor, min_lr)
    set_lr(optimizer, new_lr)
    return old_lr, new_lr


def validate(epoch=None):
    net.eval()
    tot = 0
    details = []
    with torch.no_grad():
        for i in target_test_names:
            pred_sum, pred_cnt = torch.zeros(800), torch.zeros(800)
            valid_data = TRANSFORMERDATA(i, seq_len)
            data_len = len(valid_data)
            valid_loader = DataLoader(valid_data, batch_size=1000)
            valid_iter = iter(valid_loader)
            d = next(valid_iter)
            input, lbl, msk = d[0], d[1], d[2]
            input, msk = input.cuda(), msk.cuda()
            _, out = net(input, msk)
            out = out.squeeze(2).cpu()
            for j in range(data_len):
                if j < seq_len-1:
                    pred_sum[:j+1] += out[j, -(j+1):]
                    pred_cnt[:j+1] += 1
                elif j <= data_len-seq_len:
                    pred_sum[j-seq_len+1:j+1] += out[j]
                    pred_cnt[j-seq_len+1:j+1] += 1
                else:
                    pred_sum[data_len-seq_len+1-(data_len-j):data_len-seq_len+1] += out[j, :(data_len-j)]
                    pred_cnt[data_len-seq_len+1-(data_len-j):data_len-seq_len+1] += 1
            truth = torch.tensor([lbl[j,-1] for j in range(len(lbl)-seq_len+1)], dtype=torch.float)
            pred_sum, pred_cnt = pred_sum[:data_len-seq_len+1], pred_cnt[:data_len-seq_len+1]
            pred = pred_sum/pred_cnt
            if args.clip_rul:
                pred = pred.clamp(0.0, 1.0)
            mse = float(torch.sum(torch.pow(pred-truth, 2)))
            rmse = math.sqrt(mse/data_len)
            tot += rmse
            details.append({
                "epoch": epoch,
                "unit": i,
                "data_len": data_len,
                "rmse": rmse*Rc,
                "bias": float(torch.mean(pred-truth))*Rc,
                "pred_mean": float(torch.mean(pred))*Rc,
                "truth_mean": float(torch.mean(truth))*Rc,
                "pred_min": float(torch.min(pred))*Rc,
                "pred_max": float(torch.max(pred))*Rc,
            })
    mean_rmse = tot*Rc/len(target_test_names)
    return mean_rmse, details


def train():
    minn = 999
    train_start = time.perf_counter()
    bad_epochs = 0
    plateau_bad_epochs = 0
    best_epoch = -1
    best_state = None
    best_source = "raw"
    eval_best = 999
    eval_best_epoch = -1
    eval_best_source = ""
    ema_state = clone_model_state(net) if args.type == 3 and args.ema_decay > 0 else None
    epoch_fields = [
        "epoch", "epochs", "mode", "lr", "batch_count", "loss_rul",
        "adv_loss", "fea_loss", "out_loss", "latent_loss", "recon_loss",
        "smooth_loss", "decor_loss", "mono_loss", "rank_loss", "latent_factor",
        "mono_factor", "rank_factor", "rmse", "ema_rmse", "selected_rmse", "best_rmse",
        "eval_all_rmse", "eval_best_rmse", "eval_best_epoch", "eval_best_source",
        "is_best", "best_source",
        "bad_epochs", "epoch_time_sec", "elapsed_sec",
    ]
    valid_unit_fields = ["epoch", "unit", "data_len", "rmse", "bias", "pred_mean", "truth_mean", "pred_min", "pred_max"]
    for e in range(epochs):
        epoch_start = time.perf_counter()
        net.train()
        random.shuffle(source_list)
        random.shuffle(target_list)
        loss2_sum, loss1_sum = 0.0, 0.0
        bkb_sum, out_sum = 0.0, 0.0
        latent_sum, recon_sum, smooth_sum, decor_sum, mono_sum, rank_sum = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        latent_factor = ramp_weight(e, args.latent_warmup, args.latent_ramp_epoch, args.latent_max_factor) if args.type == 3 else 0.0
        mono_factor = ramp_weight(e, args.mono_warmup, args.mono_ramp_epoch) if args.type == 3 else 0.0
        rank_factor = ramp_weight(e, args.rank_warmup, args.rank_ramp_epoch, args.rank_max_factor) if args.type == 3 else 0.0
        cnt = 0
        s_iter = iter(DataLoader(s_data, batch_size=args.batch_size, shuffle=True))
        t_iter = iter(DataLoader(t_data, batch_size=args.batch_size, shuffle=True))
        l = min(len(s_iter), len(t_iter))
        for _ in range(l):
            s_d, t_d = next(s_iter), next(t_iter)
            s_input, s_lb, s_msk = s_d[0], s_d[1], s_d[2]
            t_input, t_msk = t_d[0], t_d[2]
            s_input, s_lb, s_msk = s_input.cuda(), s_lb.cuda(), s_msk.cuda()
            t_input, t_msk = t_input.cuda(), t_msk.cuda()
            if args.type == 3:
                s_features, s_out, s_latent = net(s_input, s_msk, return_latent=True)
                t_features, t_out, t_latent = net(t_input, t_msk, return_latent=True)
            else:
                s_features, s_out = net(s_input, s_msk)
                t_features, t_out = net(t_input, t_msk) # [bts, seq_len, feature_num]
            s_out.squeeze_(2)
            t_out.squeeze_(2)
            loss1 = masked_mse(s_out, s_lb, s_msk)
            loss1_sum += loss1.item()
            cnt += 1
            if args.type == 3:
                loss = loss1
                if latent_factor > 0:
                    s_lat_loss, s_recon, s_smooth, s_decor = latent_regularization(
                        s_latent,
                        s_msk,
                        args.latent_recon_weight,
                        args.latent_smooth_weight,
                        args.latent_decor_weight,
                    )
                    t_lat_loss, t_recon, t_smooth, t_decor = latent_regularization(
                        t_latent,
                        t_msk,
                        args.latent_recon_weight,
                        args.latent_smooth_weight,
                        args.latent_decor_weight,
                    )
                    latent_loss = latent_factor * (
                        args.latent_weight*s_lat_loss + args.target_latent_weight*t_lat_loss
                    )
                    latent_sum += latent_loss.item()
                    recon_sum += (s_recon + t_recon).item()*0.5
                    smooth_sum += (s_smooth + t_smooth).item()*0.5
                    decor_sum += (s_decor + t_decor).item()*0.5
                    loss = loss + latent_loss
                if mono_factor > 0:
                    s_mono = masked_monotonic_loss(s_out, s_msk)
                    t_mono = masked_monotonic_loss(t_out, t_msk)
                    mono_loss = mono_factor * (args.mono_weight*s_mono + args.target_mono_weight*t_mono)
                    mono_sum += mono_loss.item()
                    loss = loss + mono_loss
                if rank_factor > 0:
                    s_rank = masked_degradation_ranking_loss(
                        s_latent["deg_score"],
                        s_msk,
                        args.rank_margin,
                    )
                    t_rank = masked_degradation_ranking_loss(
                        t_latent["deg_score"],
                        t_msk,
                        args.rank_margin,
                    )
                    rank_loss = rank_factor * (args.rank_weight*s_rank + args.target_rank_weight*t_rank)
                    rank_sum += rank_loss.item()
                    loss = loss + rank_loss
            elif args.type == 1 or args.type == 0:
                if args.type == 1:
                    s_domain = D2(s_features)
                    t_domain = D2(t_features)
                else:
                    s_domain = D1(s_out)
                    t_domain = D1(t_out)
                loss2 = advLoss(s_domain.squeeze(1), t_domain.squeeze(1), 'cuda')
                loss2_sum += loss2.item()
                loss = loss1 + a*loss2
            elif args.type == 2:
                s_domain_bkb = D2(s_features)
                t_domain_bkb = D2(t_features)
                s_domain_out = D1(s_out)
                t_domain_out = D1(t_out)
                if e>=5:
                    fea_loss = advLoss(s_domain_bkb.squeeze(1), t_domain_bkb.squeeze(1), 'cuda')
                    out_loss = advLoss(s_domain_out.squeeze(1), t_domain_out.squeeze(1), 'cuda')
                    bkb_sum += fea_loss.item()
                    out_sum += out_loss.item()
                    loss = loss1 + a*fea_loss + b*out_loss
                else:
                    loss = loss1
            opt.zero_grad()
            loss.backward()
            if args.type == 3:
                torch.nn.utils.clip_grad_norm_(net.parameters(), 2)
            elif args.type == 0:
                torch.nn.utils.clip_grad_norm_(itertools.chain(net.parameters(), D1.parameters()), 2)
            elif args.type == 1:
                torch.nn.utils.clip_grad_norm_(itertools.chain(net.parameters(), D2.parameters()), 2)
            else:
                torch.nn.utils.clip_grad_norm_(itertools.chain(net.parameters(), D1.parameters(), D2.parameters()), 2)
            opt.step()    
            if ema_state is not None:
                update_ema_state(ema_state, net, args.ema_decay)

        rmse, valid_details = validate(e)
        ema_rmse, ema_details = None, None
        selected_rmse, selected_details = rmse, valid_details
        selected_state = clone_model_state(net)
        selected_source = "raw"
        if ema_state is not None and e >= args.ema_start_epoch:
            ema_rmse, ema_details = validate_with_state(ema_state, e, preserve_rng=True)
            if ema_rmse < selected_rmse:
                selected_rmse = ema_rmse
                selected_details = ema_details
                selected_state = clone_state_dict(ema_state)
                selected_source = "ema"
        avg_loss1 = loss1_sum/cnt
        avg_loss2 = loss2_sum/cnt
        avg_bkb = bkb_sum/cnt
        avg_out = out_sum/cnt
        avg_latent = latent_sum/cnt
        avg_recon = recon_sum/cnt
        avg_smooth = smooth_sum/cnt
        avg_decor = decor_sum/cnt
        avg_mono = mono_sum/cnt
        avg_rank = rank_sum/cnt
        if args.type == 3:
            ema_msg = "none" if ema_rmse is None else "{:.5f}".format(ema_rmse)
            log_print("{}/{}| loss1={:.5f}, latent={:.5f}, mono={:.5f}, rank={:.5f}, recon={:.5f}, smooth={:.5f}, decor={:.5f}, lf={:.3f}, mf={:.3f}, rf={:.3f}, rmse={:.5f}, ema_rmse={}, selected={}:{:.5f}".\
                format(e, args.epoch, avg_loss1, avg_latent, avg_mono, avg_rank, avg_recon, avg_smooth, avg_decor, latent_factor, mono_factor, rank_factor, rmse, ema_msg, selected_source, selected_rmse))
        elif args.type == 2:
            log_print("{}/{}| loss1={:.5f}, fea_loss={:.5f}, out_loss={:.5f}, rmse={:.5f}".\
                format(e, args.epoch, avg_loss1, avg_bkb, avg_out, rmse))
        else:    
            log_print("{}/{}| 1={:.5f}, 2={:.5f}, rmse={:.5f}".format(e, args.epoch, avg_loss1, avg_loss2, rmse))
        previous_best = minn
        is_best = selected_rmse < minn
        is_meaningful_best = selected_rmse < previous_best - args.min_delta
        if is_best:
            minn = selected_rmse
            best_epoch = e
            best_state = clone_state_dict(selected_state)
            best_source = selected_source
            if is_meaningful_best:
                bad_epochs = 0
                plateau_bad_epochs = 0
            else:
                bad_epochs += 1
                plateau_bad_epochs += 1
            save_best_state(best_state)
            log_print("min={} at epoch={} from {}".format(minn, best_epoch, best_source))
        else:
            bad_epochs += 1
            plateau_bad_epochs += 1

        eval_all_rmse = ""
        should_eval_all = (
            args.type == 3
            and all_target_names
            and (
                (args.eval_all_every > 0 and (e+1) % args.eval_all_every == 0)
                or (args.eval_all_on_best and is_best)
            )
        )
        if should_eval_all:
            eval_all_rmse, _ = validate_units_with_state(
                selected_state,
                sorted(all_target_names),
                e,
            )
            if eval_all_rmse < eval_best:
                eval_best = eval_all_rmse
                eval_best_epoch = e
                eval_best_source = selected_source
                torch.save(cpu_state_dict(selected_state), "save/final/latent_evalbest_"+source[-1]+target[-1]+".pth")
                log_print("eval_all_min={} at epoch={} from {}".format(eval_best, eval_best_epoch, eval_best_source))

        epoch_time = time.perf_counter() - epoch_start
        append_csv(
            EPOCH_LOG_PATH,
            epoch_fields,
            {
                "epoch": e,
                "epochs": args.epoch,
                "mode": type[args.type],
                "lr": opt.param_groups[0]["lr"],
                "batch_count": cnt,
                "loss_rul": avg_loss1,
                "adv_loss": avg_loss2,
                "fea_loss": avg_bkb,
                "out_loss": avg_out,
                "latent_loss": avg_latent,
                "recon_loss": avg_recon,
                "smooth_loss": avg_smooth,
                "decor_loss": avg_decor,
                "mono_loss": avg_mono,
                "rank_loss": avg_rank,
                "latent_factor": latent_factor,
                "mono_factor": mono_factor,
                "rank_factor": rank_factor,
                "rmse": rmse,
                "ema_rmse": "" if ema_rmse is None else ema_rmse,
                "selected_rmse": selected_rmse,
                "best_rmse": minn,
                "eval_all_rmse": eval_all_rmse,
                "eval_best_rmse": "" if eval_best >= 999 else eval_best,
                "eval_best_epoch": "" if eval_best >= 999 else eval_best_epoch,
                "eval_best_source": eval_best_source,
                "is_best": int(is_best),
                "best_source": best_source,
                "bad_epochs": bad_epochs,
                "epoch_time_sec": epoch_time,
                "elapsed_sec": time.perf_counter() - train_start,
            },
        )
        for row in selected_details:
            append_csv(VALID_UNIT_LOG_PATH, valid_unit_fields, row)
        
        if args.scheduler and args.scheduler_type == "step":
            sch.step()

        if args.scheduler and args.type == 3 and args.scheduler_type == "plateau" and e >= args.plateau_start_epoch and plateau_bad_epochs >= args.lr_patience:
            old_lr, new_lr = reduce_lr_on_plateau(opt, args.lr_factor, args.min_lr)
            plateau_bad_epochs = 0
            if best_state is not None and args.restore_best_on_plateau:
                net.load_state_dict(best_state)
                if ema_state is not None:
                    ema_state = clone_state_dict(best_state)
            log_print("plateau: lr {:.6g} -> {:.6g}, restored_best={}".format(
                old_lr,
                new_lr,
                int(best_state is not None and args.restore_best_on_plateau),
            ))
            if new_lr <= args.min_lr and e >= args.early_stop_start_epoch and bad_epochs >= args.patience:
                log_print("early stop at epoch {}: best_epoch={}, best_rmse={:.5f}".format(e, best_epoch, minn))
                break

        if args.type == 3 and e >= args.early_stop_start_epoch and bad_epochs >= args.patience:
            log_print("early stop at epoch {}: best_epoch={}, best_rmse={:.5f}".format(e, best_epoch, minn))
            break

    return {
        "best_rmse": minn,
        "best_epoch": best_epoch,
        "best_source": best_source,
        "eval_best_rmse": None if eval_best >= 999 else eval_best,
        "eval_best_epoch": eval_best_epoch,
        "eval_best_source": eval_best_source,
        "best_state": best_state,
    }


def get_score(pred, truth):
    """input must be tensors!"""
    x = pred-truth
    score1 = torch.tensor([torch.exp(-i/13)-1 for i in x if i<0])
    score2 = torch.tensor([torch.exp(i/10)-1 for i in x if i>=0])
    return int(torch.sum(score1)+torch.sum(score2))


if __name__ == "__main__":
    Rc = 130

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument("--epoch", type=int, default=240)
    parser.add_argument("--batch_size", type=int, default=128, help="batch_size")
    parser.add_argument("--seq_len", type=int, default=70)
    parser.add_argument("--source", type=str, default="FD003", help="decide source file", choices=['FD001','FD002','FD003','FD004'])
    parser.add_argument("--target", type=str, default="FD002", help="decide target file", choices=['FD001','FD002','FD003','FD004'])
    parser.add_argument("--a", type=float, default=0.1, help='hyper-param α')
    parser.add_argument("--b", type=float, default=0.5, help='hyper-param β')
    parser.add_argument("--scheduler", type=int, default=1, choices=[0,1], help="1 for using sheduler while 0 for not")
    parser.add_argument("--scheduler_type", type=str, default="step", choices=["step", "plateau"])
    parser.add_argument("--type", type=int, default=3, choices=[0,1,2,3], help="0:out only | 1:DANN | 2:backbone+output | 3:latent no-align")
    parser.add_argument("--train", default=1, type=int)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["sgd", "adamw"])
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--latent_dim", type=int, default=16)
    parser.add_argument("--latent_split", type=int, default=1, help="1 splits latent into degradation and fault factors")
    parser.add_argument("--deg_latent_dim", type=int, default=0, help="0 uses about 75 percent of latent_dim")
    parser.add_argument("--fault_latent_dim", type=int, default=0, help="0 uses the remaining latent dimensions")
    parser.add_argument("--latent_hidden", type=int, default=64)
    parser.add_argument("--latent_warmup", type=int, default=5)
    parser.add_argument("--latent_ramp_epoch", type=int, default=40)
    parser.add_argument("--latent_max_factor", type=float, default=1.0)
    parser.add_argument("--latent_weight", type=float, default=0.05)
    parser.add_argument("--target_latent_weight", type=float, default=0.01)
    parser.add_argument("--latent_recon_weight", type=float, default=1.0)
    parser.add_argument("--latent_smooth_weight", type=float, default=0.05)
    parser.add_argument("--latent_decor_weight", type=float, default=0.005)
    parser.add_argument("--mono_warmup", type=int, default=0)
    parser.add_argument("--mono_ramp_epoch", type=int, default=20)
    parser.add_argument("--mono_weight", type=float, default=0.02)
    parser.add_argument("--target_mono_weight", type=float, default=0.005)
    parser.add_argument("--rank_warmup", type=int, default=5)
    parser.add_argument("--rank_ramp_epoch", type=int, default=40)
    parser.add_argument("--rank_max_factor", type=float, default=1.0)
    parser.add_argument("--rank_weight", type=float, default=0.01)
    parser.add_argument("--target_rank_weight", type=float, default=0.005)
    parser.add_argument("--rank_margin", type=float, default=0.001)
    parser.add_argument("--ema_decay", type=float, default=0.0)
    parser.add_argument("--ema_start_epoch", type=int, default=5)
    parser.add_argument("--patience", type=int, default=60)
    parser.add_argument("--lr_patience", type=int, default=20)
    parser.add_argument("--plateau_start_epoch", type=int, default=45)
    parser.add_argument("--early_stop_start_epoch", type=int, default=80)
    parser.add_argument("--lr_factor", type=float, default=0.5)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--min_delta", type=float, default=0.02)
    parser.add_argument("--restore_best_on_plateau", type=int, default=0)
    parser.add_argument("--validate_all_target", type=int, default=0, help="1 validates on every target unit when saving best")
    parser.add_argument("--final_eval_all_target", type=int, default=1, help="1 evaluates the saved best on every target unit after training")
    parser.add_argument("--eval_all_every", type=int, default=5, help="periodically save the checkpoint with best all-target RMSE; 0 disables")
    parser.add_argument("--eval_all_on_best", type=int, default=1, help="1 evaluates all target units whenever validation gets a new best")
    parser.add_argument("--clip_rul", type=int, default=1, help="1 clips normalized RUL predictions to [0, 1] during validation/evaluation")
    parser.add_argument("--log_dir", type=str, default="logs", help="directory for training and validation logs")
    parser.add_argument("--run_id", type=str, default="", help="optional run id used in log directory names")
    args = parser.parse_args()
    set_seed(args.seed)
    if args.type != 3 and args.scheduler_type == "plateau":
        args.scheduler_type = "step"
    if args.lr is None:
        args.lr = 0.001 if args.type == 3 else 0.02
    resolve_latent_split_args(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu 
    source, target = args.source, args.target
    data_root = "CMAPSS/units/"
    label_root = "CMAPSS/labels/"
    type = {0:"out_only", 1:"DANN", 2:"backbone + output", 3:"latent no-align"}
    seq_len, a, epochs, b = args.seq_len, args.a, args.epoch, args.b
    option_str = "source={}, target={}, a={}, b={}, epochs={}, type={}, lr={}, scheduler={}".\
        format(source, target, a, b, epochs, type[args.type], args.lr, args.scheduler_type if args.scheduler else "off")
    if args.type == 3:
        option_str += ", optimizer={}, weight_decay={}, latent_dim={}, latent_split={}, deg_latent_dim={}, fault_latent_dim={}, latent_weight={}, target_latent_weight={}, latent_ramp_epoch={}, latent_max_factor={}, mono_weight={}, target_mono_weight={}, rank_weight={}, target_rank_weight={}, rank_warmup={}, rank_ramp_epoch={}, rank_max_factor={}, rank_margin={}, ema_decay={}, patience={}, plateau_start_epoch={}, early_stop_start_epoch={}, validate_all_target={}, final_eval_all_target={}, eval_all_every={}, eval_all_on_best={}, clip_rul={}, no source-target alignment".format(
            args.optimizer,
            args.weight_decay,
            args.latent_dim,
            args.latent_split,
            args.deg_latent_dim,
            args.fault_latent_dim,
            args.latent_weight,
            args.target_latent_weight,
            args.latent_ramp_epoch,
            args.latent_max_factor,
            args.mono_weight,
            args.target_mono_weight,
            args.rank_weight,
            args.target_rank_weight,
            args.rank_warmup,
            args.rank_ramp_epoch,
            args.rank_max_factor,
            args.rank_margin,
            args.ema_decay,
            args.patience,
            args.plateau_start_epoch,
            args.early_stop_start_epoch,
            args.validate_all_target,
            args.final_eval_all_target,
            args.eval_all_every,
            args.eval_all_on_best,
            args.clip_rul,
        )
    init_train_logging(args, option_str)
    log_print(option_str)

    net = mymodel(
        max_len=seq_len,
        dropout=args.dropout,
        use_latent=(args.type == 3),
        latent_dim=args.latent_dim,
        latent_hidden=args.latent_hidden,
        latent_split=bool(args.latent_split),
        deg_latent_dim=args.deg_latent_dim,
        fault_latent_dim=args.fault_latent_dim,
    ) 
    D1 = Discriminator(seq_len) if args.type in [0, 2] else None
    D2 = backboneDiscriminator(seq_len) if args.type in [1, 2] else None
    if args.type == 0:
        opt = torch.optim.SGD(itertools.chain(net.parameters(), D1.parameters()), lr=args.lr)
    elif args.type == 1:
        opt = torch.optim.SGD(itertools.chain(net.parameters(), D2.parameters()), lr=args.lr)
    elif args.type == 2:
        opt = torch.optim.SGD(itertools.chain(net.parameters(), D1.parameters(), D2.parameters()), lr=args.lr)
    elif args.type == 3:
        if args.optimizer == "adamw":
            opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        else:
            opt = torch.optim.SGD(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    net = net.cuda()
    if D1 is not None:
        D1 = D1.cuda()
    if D2 is not None:
        D2 = D2.cuda()
    sch = torch.optim.lr_scheduler.StepLR(opt, 80, 0.5)

    source_list = np.loadtxt("save/"+source+"/train"+source+".txt", dtype=str).tolist()
    target_list = np.loadtxt("save/"+target+"/train"+target+".txt", dtype=str).tolist()
    valid_list = np.loadtxt("save/"+target+"/test"+target+".txt", dtype=str).tolist()
    a_list = np.loadtxt("save/"+target+"/valid"+target+".txt", dtype=str).tolist()
    target_test_names = valid_list + a_list
    all_target_names = [
        name for name in os.listdir(data_root)
        if name.startswith(target + "-") and name.endswith(".txt")
    ]
    if args.validate_all_target:
        if all_target_names:
            target_test_names = sorted(all_target_names)
            log_print("validate_all_target=1, validation units={}".format(len(target_test_names)))
        else:
            log_print("validate_all_target=1 requested, but no all-target units found under {}; fallback to test+valid".format(data_root))
    s_data = TRANSFORMER_ALL_DATA(source_list, seq_len)
    t_data = TRANSFORMER_ALL_DATA(target_list, seq_len)
    if not os.path.exists('./online'):
        os.makedirs('./online')

    if args.train:
        train_time1 = time.perf_counter()
        train_result = train()
        train_time2 = time.perf_counter()
        train_time = train_time2-train_time1
        final_all_target_rmse = None
        if args.final_eval_all_target and all_target_names and train_result["best_state"] is not None:
            final_all_target_rmse, _ = validate_units_with_state(
                train_result["best_state"],
                sorted(all_target_names),
                train_result["best_epoch"],
            )
            log_print("final_all_target_rmse = {}".format(final_all_target_rmse))
            if args.type == 3 and (
                train_result["eval_best_rmse"] is None
                or final_all_target_rmse < train_result["eval_best_rmse"]
            ):
                torch.save(
                    cpu_state_dict(train_result["best_state"]),
                    "save/final/latent_evalbest_"+source[-1]+target[-1]+".pth",
                )
                train_result["eval_best_rmse"] = final_all_target_rmse
                train_result["eval_best_epoch"] = train_result["best_epoch"]
                train_result["eval_best_source"] = train_result["best_source"] + ":final_all"
                log_print("promoted final_all_target checkpoint to eval_best: rmse={}, epoch={}, source={}".format(
                    train_result["eval_best_rmse"],
                    train_result["eval_best_epoch"],
                    train_result["eval_best_source"],
                ))
        log_print(option_str)
        log_print("best = {}, best_epoch = {}, best_source = {}, train time = {}".format(
            train_result["best_rmse"],
            train_result["best_epoch"],
            train_result["best_source"],
            train_time,
        ))
        write_json(
            os.path.join(LOG_DIR, "summary.json"),
            {
                "best_rmse": train_result["best_rmse"],
                "best_epoch": train_result["best_epoch"],
                "best_source": train_result["best_source"],
                "eval_best_rmse": train_result["eval_best_rmse"],
                "eval_best_epoch": train_result["eval_best_epoch"],
                "eval_best_source": train_result["eval_best_source"],
                "final_all_target_rmse": final_all_target_rmse,
                "train_time_sec": train_time,
                "source": source,
                "target": target,
                "type": type[args.type],
            },
        )


