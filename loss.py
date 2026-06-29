import torch
import torch.nn as nn
import torch.nn.functional as F


def advLoss(source, target, device):

    sourceLabel = torch.ones(len(source))
    targetLabel = torch.zeros(len(target))
    Loss = nn.BCELoss()
    if device == 'cuda':
        Loss = Loss.cuda()
        sourceLabel, targetLabel = sourceLabel.cuda(), targetLabel.cuda()
    #print("sd={}\ntd={}".format(source, target))
    loss = Loss(source, sourceLabel) + Loss(target, targetLabel)
    return loss*0.5


def masked_mse(pred, target, padding=None):
    """
    MSE on real timesteps only. padding uses the dataset convention:
    1 means ignore, 0 means valid.
    """
    loss = (pred - target).pow(2)
    if padding is None:
        return loss.mean()
    valid = (1 - padding.float()).clamp(min=0, max=1)
    return (loss * valid).sum() / valid.sum().clamp_min(1.0)


def masked_smoothness_loss(z, padding=None):
    """Encourage a smoother health trajectory inside each domain."""
    if z.size(1) < 2:
        return z.new_tensor(0.0)
    dz = z[:, 1:] - z[:, :-1]
    loss = dz.pow(2).mean(dim=2)
    if padding is None:
        return loss.mean()
    valid = (1 - padding[:, 1:].float()) * (1 - padding[:, :-1].float())
    return (loss * valid).sum() / valid.sum().clamp_min(1.0)


def masked_monotonic_loss(pred, padding=None):
    """
    RUL should not increase as cycles advance. This is a degradation-shape prior,
    not a source-target alignment loss.
    """
    if pred.size(1) < 2:
        return pred.new_tensor(0.0)
    rising = F.relu(pred[:, 1:] - pred[:, :-1]).pow(2)
    if padding is None:
        return rising.mean()
    valid = (1 - padding[:, 1:].float()) * (1 - padding[:, :-1].float())
    return (rising * valid).sum() / valid.sum().clamp_min(1.0)


def masked_degradation_ranking_loss(score, padding=None, margin=0.0):
    """
    A soft prior on the degradation latent: later valid timesteps should have
    no smaller degradation score than earlier timesteps inside the same window.
    """
    if score.size(1) < 2:
        return score.new_tensor(0.0)
    violation = F.relu(score[:, :-1] - score[:, 1:] + margin).pow(2)
    if padding is None:
        return violation.mean()
    valid = (1 - padding[:, 1:].float()) * (1 - padding[:, :-1].float())
    return (violation * valid).sum() / valid.sum().clamp_min(1.0)


def masked_reconstruction_loss(recon, features, padding=None):
    """
    Reconstruct transformer features from the latent code per domain.
    This preserves fault evidence without matching source and target distributions.
    """
    loss = F.mse_loss(recon, features.detach(), reduction='none').mean(dim=2)
    if padding is None:
        return loss.mean()
    valid = (1 - padding.float()).clamp(min=0, max=1)
    return (loss * valid).sum() / valid.sum().clamp_min(1.0)


def decorrelation_loss(z, padding=None):
    """
    Reduce redundant latent dimensions within a batch/domain, but do not align domains.
    """
    if padding is not None:
        valid = (1 - padding.float()).bool()
        z = z[valid]
    else:
        z = z.reshape(-1, z.size(-1))

    if z.size(0) <= 1:
        return z.new_tensor(0.0)

    z = z - z.mean(dim=0, keepdim=True)
    z = z / z.std(dim=0, keepdim=True).clamp_min(1e-6)
    corr = torch.matmul(z.transpose(0, 1), z) / (z.size(0) - 1)
    eye = torch.eye(corr.size(0), device=corr.device, dtype=torch.bool)
    return corr.masked_select(~eye).pow(2).mean()


def cross_decorrelation_loss(z1, z2, padding=None):
    """
    Make degradation and fault latents carry different information.
    This is an intra-sample disentanglement penalty, not domain alignment.
    """
    if z1 is None and z2 is None:
        return torch.tensor(0.0)
    if z1 is None:
        return z2.new_tensor(0.0)
    if z2 is None:
        return z1.new_tensor(0.0)
    if padding is not None:
        valid = (1 - padding.float()).bool()
        z1 = z1[valid]
        z2 = z2[valid]
    else:
        z1 = z1.reshape(-1, z1.size(-1))
        z2 = z2.reshape(-1, z2.size(-1))

    if z1.size(0) <= 1:
        return z1.new_tensor(0.0)

    z1 = z1 - z1.mean(dim=0, keepdim=True)
    z2 = z2 - z2.mean(dim=0, keepdim=True)
    z1 = z1 / z1.std(dim=0, keepdim=True).clamp_min(1e-6)
    z2 = z2 / z2.std(dim=0, keepdim=True).clamp_min(1e-6)
    corr = torch.matmul(z1.transpose(0, 1), z2) / (z1.size(0) - 1)
    return corr.pow(2).mean()


def latent_regularization(latent_info, padding=None, recon_weight=1.0, smooth_weight=0.1, decor_weight=0.01):
    if latent_info is None:
        raise RuntimeError("latent_regularization requires mymodel(use_latent=True)")
    z = latent_info["z"]
    z_deg = latent_info.get("z_deg", z)
    z_fault = latent_info.get("z_fault")
    recon = latent_info["recon"]
    features = latent_info["features"]
    recon_loss = masked_reconstruction_loss(recon, features, padding)
    smooth_loss = masked_smoothness_loss(z_deg, padding)
    if z_fault is None:
        decor_loss = decorrelation_loss(z, padding)
    else:
        decor_loss = cross_decorrelation_loss(z_deg, z_fault, padding)
    total = recon_weight*recon_loss + smooth_weight*smooth_loss + decor_weight*decor_loss
    return total, recon_loss, smooth_loss, decor_loss


if __name__ == "__main__":
    pass
