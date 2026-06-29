import argparse
import math
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import TRANSFORMERDATA
from model import *
import os
import random
import csv
import json
import time

LOG_DIR = None
EVAL_LOG_PATH = None
EVAL_UNIT_LOG_PATH = None


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
    if EVAL_LOG_PATH is not None:
        with open(EVAL_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(str(message) + "\n")


def init_eval_logging(args, checkpoint):
    global LOG_DIR, EVAL_LOG_PATH, EVAL_UNIT_LOG_PATH
    run_id = args.run_id if args.run_id else time.strftime("%Y%m%d-%H%M%S")
    run_name = "eval_{}_to_{}_type{}_{}".format(args.source, args.target, args.type, run_id)
    LOG_DIR = unique_dir(os.path.join(args.log_dir, run_name))
    os.makedirs(LOG_DIR, exist_ok=True)
    EVAL_LOG_PATH = os.path.join(LOG_DIR, "eval.log")
    EVAL_UNIT_LOG_PATH = os.path.join(LOG_DIR, "eval_units.csv")
    write_json(
        os.path.join(LOG_DIR, "config.json"),
        {
            "mode": "eval",
            "checkpoint": checkpoint,
            "args": vars(args),
        },
    )
    log_print("logs saved to {}".format(LOG_DIR))


def infer_latent_dim(state_dict, fallback=16):
    if "deg_projector.0.weight" in state_dict and "fault_projector.0.weight" in state_dict:
        return int(state_dict["deg_projector.0.weight"].shape[0] + state_dict["fault_projector.0.weight"].shape[0])
    if "latent_encoder.4.weight" in state_dict:
        return int(state_dict["latent_encoder.4.weight"].shape[0])
    if "decoder.weight" in state_dict and len(state_dict["decoder.weight"].shape) == 2:
        return int(state_dict["decoder.weight"].shape[1])
    return fallback


def infer_latent_split(state_dict):
    if "deg_projector.0.weight" not in state_dict or "fault_projector.0.weight" not in state_dict:
        return 0, 0, 0
    deg_dim = int(state_dict["deg_projector.0.weight"].shape[0])
    fault_dim = int(state_dict["fault_projector.0.weight"].shape[0])
    return 1, deg_dim, fault_dim


def score(pred, truth):
    """input must be tensors!"""
    x = pred-truth
    score1 = torch.tensor([torch.exp(-i/13)-1 for i in x if i<0])
    score2 = torch.tensor([torch.exp(i/10)-1 for i in x if i>=0])
    return int(torch.sum(score1)+torch.sum(score2))


def get_pred_result(data_len, out, lb, clip_rul=False):
    pred_sum, pred_cnt = torch.zeros(800), torch.zeros(800)
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
    truth = torch.tensor([lb[j,-1] for j in range(len(lb)-seq_len+1)], dtype=torch.float)
    pred_sum, pred_cnt = pred_sum[:data_len-seq_len+1], pred_cnt[:data_len-seq_len+1]
    pred2 = pred_sum/pred_cnt
    if clip_rul:
        pred2 = pred2.clamp(0.0, 1.0)
    pred2 *= Rc
    truth *= Rc
    return truth, pred2 


def test():
    truth, tot, tot_sc = [], 0, 0
    start = time.perf_counter()
    unit_fields = ["unit", "data_len", "rmse", "score", "bias", "pred_mean", "truth_mean", "pred_min", "pred_max"]
    net.eval()
    with torch.no_grad():
        for k in range(test_len):
            i = next(list_iter)
            dataset = TRANSFORMERDATA(i, seq_len)
            data_len = len(dataset)
            dataloader = DataLoader(dataset, batch_size=800, shuffle=0)
            it = iter(dataloader)
            d = next(it)
            input, lb, msk = d[0], d[1], d[2]
            if fake:
                input = torch.zeros(input.shape)
            input, msk = input.cuda(), msk.cuda()
            #uncertainty(input, msk, data_len, lb, i)
            _, out = net(input, msk)
            out = out.squeeze(2).cpu()
            truth, pred = get_pred_result(data_len, out, lb, args.clip_rul)
            mse = float(torch.sum(torch.pow(pred-truth, 2)))
            rmse = math.sqrt(mse/data_len)
            tot += rmse
            sc = score(pred, truth)
            tot_sc += sc
            append_csv(
                EVAL_UNIT_LOG_PATH,
                unit_fields,
                {
                    "unit": i,
                    "data_len": data_len,
                    "rmse": rmse,
                    "score": sc,
                    "bias": float(torch.mean(pred-truth)),
                    "pred_mean": float(torch.mean(pred)),
                    "truth_mean": float(torch.mean(truth)),
                    "pred_min": float(torch.min(pred)),
                    "pred_max": float(torch.max(pred)),
                },
            )
            log_print("for file {}: rmse={:.4f}, score={}".format(i, rmse, sc))
            log_print('-'*80)

    mean_rmse = tot/test_len
    mean_score = int(tot_sc/test_len)
    elapsed = time.perf_counter() - start
    log_print("tested on [{}] files, mean RMSE = {:.4f}, mean score = {}".format(test_len, mean_rmse, mean_score))
    write_json(
        os.path.join(LOG_DIR, "summary.json"),
        {
            "test_len": test_len,
            "mean_rmse": mean_rmse,
            "mean_score": mean_score,
            "elapsed_sec": elapsed,
        },
    )


if __name__ == "__main__": 
    Rc = 130
    fake = 0
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument("--seq_len", type=int, default=70)
    parser.add_argument("--source", type=str, default="FD002", help="file name the model trained on")
    parser.add_argument("--target", type=str, default="FD004", help="test domain")
    parser.add_argument("--sem", type=int, default=1)
    parser.add_argument("--type", type=int, default=3, choices=[2,3], help="2:backbone+output | 3:latent no-align")
    parser.add_argument("--model_path", type=str, default="", help="optional explicit checkpoint path")
    parser.add_argument("--eval_best", type=int, default=0, help="1 loads latent_evalbest_ST.pth for type 3")
    parser.add_argument("--clip_rul", type=int, default=1, help="1 clips normalized RUL predictions to [0, 1]")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--latent_dim", type=int, default=0, help="0 infers latent dimension from checkpoint")
    parser.add_argument("--latent_split", type=int, default=-1, help="-1 infers split mode from checkpoint")
    parser.add_argument("--deg_latent_dim", type=int, default=0, help="0 infers degradation latent dimension")
    parser.add_argument("--fault_latent_dim", type=int, default=0, help="0 infers fault latent dimension")
    parser.add_argument("--latent_hidden", type=int, default=64)
    parser.add_argument("--log_dir", type=str, default="logs", help="directory for evaluation logs")
    parser.add_argument("--run_id", type=str, default="", help="optional run id used in log directory names")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    seq_len = args.seq_len
    model_name = args.source
    test_name = args.target
    if args.model_path:
        checkpoint = args.model_path
    elif args.type == 3 and args.eval_best:
        checkpoint = "save/final/latent_evalbest_"+args.source[-1]+args.target[-1]+".pth"
    elif args.type == 3:
        checkpoint = "save/final/latent_"+args.source[-1]+args.target[-1]+".pth"
    else:
        checkpoint = "save/final/both_"+args.source[-1]+args.target[-1]+".pth"
    init_eval_logging(args, checkpoint)
    log_print("checkpoint={}".format(checkpoint))
    x=torch.load(checkpoint, map_location='cuda:0')
    latent_dim = args.latent_dim
    if args.type == 3 and latent_dim <= 0:
        latent_dim = infer_latent_dim(x)
        log_print("inferred latent_dim={}".format(latent_dim))
    latent_split = bool(args.latent_split)
    deg_latent_dim = args.deg_latent_dim
    fault_latent_dim = args.fault_latent_dim
    if args.type == 3:
        inferred_split, inferred_deg_dim, inferred_fault_dim = infer_latent_split(x)
        if args.latent_split < 0:
            latent_split = bool(inferred_split)
        if inferred_split:
            if deg_latent_dim <= 0:
                deg_latent_dim = inferred_deg_dim
            if fault_latent_dim <= 0:
                fault_latent_dim = inferred_fault_dim
            log_print("inferred latent_split={}, deg_latent_dim={}, fault_latent_dim={}".format(
                int(latent_split),
                deg_latent_dim,
                fault_latent_dim,
            ))
    net = mymodel(
        max_len=seq_len,
        dropout=args.dropout,
        use_latent=(args.type == 3),
        latent_dim=latent_dim,
        latent_hidden=args.latent_hidden,
        latent_split=latent_split,
        deg_latent_dim=deg_latent_dim,
        fault_latent_dim=fault_latent_dim,
    ).cuda()

    missing_keys, unexpected_keys = net.load_state_dict(x, strict=False)
    if missing_keys or unexpected_keys:
        log_print("checkpoint load warning: missing_keys={}, unexpected_keys={}".format(
            missing_keys,
            unexpected_keys,
        ))
    data_root = "CMAPSS/units/"
    label_root = "CMAPSS/labels/"
    lis = os.listdir(data_root)
    test_list = [i for i in lis if i[:5] == test_name]
    random.shuffle(test_list)
    test_len = len(test_list)
    list_iter = iter(test_list)
    test()
    
