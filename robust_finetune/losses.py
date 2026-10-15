"""Masked WAFT mixture likelihood and paired Clean non-regression objective."""
import math
import torch


def epe_map(flow, target):
    return torch.linalg.vector_norm(flow.float() - target.float(), dim=1)


def masked_mean(values, mask):
    if values.ndim == 4:
        values = values.mean(1)
    good = mask >= .5
    if not good.any():
        raise ValueError("no valid pixels in loss")
    return values[good].mean()


def sequence_loss(output, target, valid, gamma=.85):
    with torch.autocast(device_type=target.device.type, enabled=False):
        total = target.new_zeros((), dtype=torch.float32)
        for i, (flow, info) in enumerate(zip(output["flow"], output["info"])):
            info = info.float()
            weight, raw = info[:, :2], info[:, 2:]
            log_b = torch.stack((raw[:, 0].clamp(0, 10), raw[:, 1].clamp(0, 0)), 1)
            error = (target.float() - flow.float()).abs().unsqueeze(2)
            term = weight - math.log(2) - log_b
            nf = torch.logsumexp(weight, 1, keepdim=True) - torch.logsumexp(term.unsqueeze(1) - error * (-log_b).exp().unsqueeze(1), 2)
            total = total + gamma ** (len(output["flow"]) - i - 1) * masked_mean(nf, valid)
        return total


def clean_regret(prediction, teacher, target, valid):
    # Pixelwise excess EPE; permits improvements, unlike unconditional imitation.
    return masked_mean((epe_map(prediction, target) - epe_map(teacher.detach(), target)).clamp_min(0), valid)
