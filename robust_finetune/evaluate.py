"""In-training diagnostic (robust train.py --eval-every): clean + all 20 corruptions at severity 3
on the held-out first `--prefix` triplets of every Spring train sequence (never reads val/ or test/)."""
import torch
import torch.distributed as dist

from .corruptions import ROBUSTSPRING_CORRUPTIONS
from .losses import epe_map
from .runtime import autocast


@torch.no_grad()
def evaluate_model(model, dataset, device, dtype, rank=0, world=1):
    conditions = ("clean", *ROBUSTSPRING_CORRUPTIONS)
    sums = torch.zeros(len(conditions), 4, dtype=torch.float64, device=device)
    was_training = model.training
    model.eval()
    for ci, condition in enumerate(conditions):
        for ri in range(rank, len(dataset), world):
            item = dataset.fetch(ri, condition, severity=3, corruption_seed=19001 + ri * 1009 + ci * 7919)
            frames = item["corrupt"].unsqueeze(0).to(device)
            target = item["corrupt_flow"].unsqueeze(0).to(device)
            valid = item["corrupt_valid"].to(device) >= .5
            with autocast(device, dtype):
                out = model(*frames.unbind(1), final_only=True)
            errors = epe_map(out["flow"][-1], target)[0][valid]
            if not errors.numel() or not torch.isfinite(errors).all():
                raise ValueError(f"invalid prediction for {item['id']}/{condition}")
            sums[ci, 0] += errors.double().sum()
            sums[ci, 1] += errors.numel()
            sums[ci, 2] += out.get("gate", torch.zeros(1, device=device)).double().mean()
            sums[ci, 3] += 1
            del out, frames, target
        if rank == 0:
            print(f"evaluation rank0 complete: {condition} ({len(dataset)} pairs globally)", flush=True)
    if world > 1:
        dist.all_reduce(sums)
    model.train(was_training)
    metrics = {c: {"epe": float(sums[i, 0] / sums[i, 1]), "gate_mean": float(sums[i, 2] / sums[i, 3]), "pairs": int(sums[i, 3])} for i, c in enumerate(conditions)}
    return {"protocol": "train-prefix native1080p, severity3, online temporal_v1; NOT official RobustSpring",
            "inherited_parent_trainval_contamination": True, "input_hw": [1080, 1920], "input_frames": 3,
            "layout_sha256": dataset.layout_sha256, "record_ids": [r["id"] for r in dataset.records],
            "clean_epe": metrics["clean"]["epe"], "corrupt_epe": sum(metrics[c]["epe"] for c in ROBUSTSPRING_CORRUPTIONS) / 20,
            "conditions": metrics}

