"""One micro-batch of stage-2 training.

router:   BCE between gate logits and the expert-group target of the (possibly corrupted) triplet.
adapters: clean pass (flow loss + clean regret vs the frozen 2-frame parent) and corrupted pass (flow loss).
Gradient sync happens only on the last pass of the last micro-batch.
"""
from contextlib import nullcontext

import torch
from torch.nn import functional as F

from .losses import clean_regret, sequence_loss
from .runtime import autocast


def backward_microbatch(ddp, model, batch, stage, device, dtype, scaler, accumulation, sync, regret_weight=1.):
    context = ddp.no_sync if hasattr(ddp, "no_sync") else nullcontext
    stats = torch.zeros(3, device=device)
    frames = batch["corrupt"].unbind(1)
    if stage == "router":
        with (nullcontext() if sync else context()):
            with autocast(device, dtype):
                result = ddp(*frames, router_only=True)
            loss = F.binary_cross_entropy_with_logits(result["logits"].float(), batch["router"])
            scaler.scale(loss / accumulation).backward()
            stats[2] = loss.detach()
        return stats / accumulation
    # Do not retain both full1080p Clean/Corrupt graphs at the same time.
    clean_frames = batch["clean"].unbind(1)
    with torch.no_grad(), autocast(device, dtype):
        teacher = model(*clean_frames, baseline=True)["flow"][-1]
    with context():
        with autocast(device, dtype):
            clean = ddp(*clean_frames)
        clean_loss = sequence_loss(clean, batch["flow"], batch["valid"])
        regret = clean_regret(clean["flow"][-1], teacher, batch["flow"], batch["valid"])
        loss = clean_loss + regret_weight * regret
        scaler.scale(loss / accumulation).backward()
        stats[0] = clean_loss.detach()
    del clean, teacher, loss, clean_loss, regret
    with (nullcontext() if sync else context()):
        with autocast(device, dtype):
            corrupt = ddp(*frames)
        loss = sequence_loss(corrupt, batch["corrupt_flow"], batch["corrupt_valid"])
        scaler.scale(loss / accumulation).backward()
        stats[1] = loss.detach()
    return stats / accumulation
