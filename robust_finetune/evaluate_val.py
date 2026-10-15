"""Spring val: EPE of the frozen 2-frame parent vs the 3-frame SPEAR model, clean and under the 20 corruptions.

Corruptions are synthesized with the training pipeline (NativeSpring.fetch) at a fixed severity.  Sample budget:
--heavy-samples clean triplets, --light-samples triplets per corruption, spread evenly over val/.
val/ is held out from stage 2 but is part of the parent's training data (clean-step4-spring).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .corruptions import (CORRUPTION_TO_EXPERT_GROUP, NUM_EXPERT_GROUPS, ROBUSTSPRING_CORRUPTIONS,
                          ROUTER_EXPERT_GROUPS, impulse_median_gate)
from .data import NativeSpring, WEATHER
from .losses import epe_map
from .model import DEFAULT_PARENT, build_model
from .runtime import atomic_json, autocast, load_training, precision


class ValTriplets(NativeSpring):
    """Enumerate val/ triplets; reuse the parent fetch()/corruption logic."""
    def __init__(self, root, cameras=("left",), directions=("FW",), seed=91):
        self.root = Path(root).resolve()
        vroot = self.root / "val"
        if not vroot.is_dir():
            raise FileNotFoundError(f"expected {vroot}")
        self.seed = seed
        self.records = []
        for sequence in sorted(vroot.iterdir()):
            if not sequence.is_dir() or not sequence.name.isdigit():
                continue
            for camera in cameras:
                directory = sequence / f"frame_{camera}"
                images = {int(p.stem.rsplit("_", 1)[1]): p for p in directory.glob(f"frame_{camera}_*.png")}
                if len(images) < 2:
                    continue
                numbers = sorted(images)
                for direction in directions:
                    delta = 1 if direction == "FW" else -1
                    for t in numbers:
                        if t + delta not in images:
                            continue
                        gt = sequence / f"flow_{direction}_{camera}" / f"flow_{direction}_{camera}_{t:04d}.flo5"
                        if not gt.is_file():
                            continue
                        previous = t - delta if t - delta in images else t
                        self.records.append({"id": f"{sequence.name}/{camera}/{direction}/{t:04d}",
                                             "images": [images[previous], images[t], images[t + delta]],
                                             "gt": gt, "times": (previous, t, t + delta)})
        if not self.records:
            raise ValueError("empty val triplet set")


@torch.no_grad()
def _epe(model, frames, target, valid, *, baseline):
    out = model(*frames, baseline=True) if baseline else model(*frames, final_only=True)
    e = epe_map(out["flow"][-1], target)[0][valid]
    if not e.numel() or not torch.isfinite(e).all():
        raise ValueError("invalid prediction")
    gate = None if baseline else out.get("gate")
    gate_mean = gate.float().mean(0).tolist() if gate is not None else None
    return e.double().sum().item(), e.numel(), gate_mean


def _median_gate(corrupt, tau):
    """impulse_median_gate on a (3, 3, H, W) uint8 triplet tensor."""
    images = [np.ascontiguousarray(frame.numpy().transpose(1, 2, 0)) for frame in corrupt]
    images, fired = impulse_median_gate(images, tau)
    if not fired:
        return corrupt, False
    return torch.from_numpy(np.stack([image.transpose(2, 0, 1) for image in images])), True


def evaluate(model, dataset, device, dtype, conditions, severity, has_adapter, heavy, light, median_tau=None):
    acc = {c: {"parent": [0.0, 0], "adapted": [0.0, 0], "n": 0, "fired": 0,
               "gate": [0.0] * NUM_EXPERT_GROUPS, "gate_n": 0} for c in conditions}
    model.eval()
    for ci, condition in enumerate(conditions):
        count = min(len(dataset), heavy if condition == "clean" else light)
        indices = [i * len(dataset) // count for i in range(count)]  # spread across full val
        for ri in indices:
            item = dataset.fetch(ri, condition, severity=severity, corruption_seed=19001 + ri * 1009 + ci * 7919)
            corrupt = item["corrupt"]
            if median_tau > 0:
                corrupt, fired = _median_gate(corrupt, median_tau)
                acc[condition]["fired"] += int(fired)
            frames = corrupt.unsqueeze(0).to(device).unbind(1)
            target = item["corrupt_flow"].unsqueeze(0).to(device)
            valid = item["corrupt_valid"].to(device) >= .5
            with autocast(device, dtype):
                s, n, _ = _epe(model, frames, target, valid, baseline=True)
                acc[condition]["parent"][0] += s; acc[condition]["parent"][1] += n
                if has_adapter:
                    s, n, g = _epe(model, frames, target, valid, baseline=False)
                    acc[condition]["adapted"][0] += s; acc[condition]["adapted"][1] += n
                    if g is not None:
                        for k in range(NUM_EXPERT_GROUPS):
                            acc[condition]["gate"][k] += g[k]
                        acc[condition]["gate_n"] += 1
        acc[condition]["n"] = count
        print(f"done: {condition} ({count} samples)", flush=True)
    def mean(pair):
        return pair[0] / pair[1] if pair[1] else None
    def gate_mean(v):
        return [x / v["gate_n"] for x in v["gate"]] if v["gate_n"] else None
    return {c: {"parent_epe": mean(v["parent"]), "adapted_epe": mean(v["adapted"]) if has_adapter else None,
                "samples": v["n"], "median_fired": v["fired"], "router_gate": gate_mean(v)} for c, v in acc.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--spring-root", required=True)
    p.add_argument("--parent", default=DEFAULT_PARENT)
    p.add_argument("--checkpoint", help="adapter delta checkpoint (.pt); omit for parent-only")
    p.add_argument("--output", required=True)
    p.add_argument("--cameras", nargs="+", choices=("left", "right"), default=["left"])
    p.add_argument("--directions", nargs="+", choices=("FW", "BW"), default=["FW"])
    p.add_argument("--conditions", choices=("clean", "all20"), default="all20")
    p.add_argument("--severity", type=int, default=3)
    p.add_argument("--median-gate", type=float, default=10.0, help="impulse median-gate threshold in percent (RobustSpring test: impulse >= 10.99, speckle <= 8.99); 0 disables")
    p.add_argument("--heavy-samples", type=int, default=50, help="clean triplets")
    p.add_argument("--light-samples", type=int, default=30, help="triplets per corruption")
    p.add_argument("--precision", choices=("auto", "bf16", "fp16", "fp32"), default="auto")
    p.add_argument("--attention", choices=("auto", "chunked"), default="auto")
    args = p.parse_args()

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    _, dtype = precision(args.precision, device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    dataset = ValTriplets(args.spring_root, tuple(args.cameras), tuple(args.directions))
    model = build_model(args.parent, args.attention, False).to(device).eval()
    has_adapter = bool(args.checkpoint)
    if has_adapter:
        load_training(args.checkpoint, model)

    conditions = ("clean",) if args.conditions == "clean" else ("clean", *ROBUSTSPRING_CORRUPTIONS)
    per = evaluate(model, dataset, device, dtype, conditions, args.severity, has_adapter, args.heavy_samples, args.light_samples, args.median_gate)

    weather = [c for c in conditions if c in WEATHER]
    other = [c for c in conditions if c != "clean" and c not in WEATHER]
    def avg(names, kind):
        vals = [per[c][kind] for c in names if per[c][kind] is not None]
        return sum(vals) / len(vals) if vals else None
    def group_conditions(gname):
        return [c for c in conditions if c != "clean" and CORRUPTION_TO_EXPERT_GROUP.get(c) == gname]
    expert_group_means = {g: {"parent_epe": avg(group_conditions(g), "parent_epe"),
                              "adapted_epe": avg(group_conditions(g), "adapted_epe")} for g in ROUTER_EXPERT_GROUPS}
    other_group = [c for c in conditions if c != "clean" and CORRUPTION_TO_EXPERT_GROUP.get(c) is None]
    report = {"split": "val", "note": "parent=frozen 2-frame baseline; adapted=robust 3-frame",
              "cameras": args.cameras, "directions": args.directions, "severity": args.severity,
              "heavy_samples": args.heavy_samples, "light_samples": args.light_samples,
              "router_groups": list(ROUTER_EXPERT_GROUPS),
              "clean": {"parent_epe": per["clean"]["parent_epe"], "adapted_epe": per["clean"]["adapted_epe"]},
              "weather_mean": {"parent_epe": avg(weather, "parent_epe"), "adapted_epe": avg(weather, "adapted_epe")},
              "other_mean": {"parent_epe": avg(other, "parent_epe"), "adapted_epe": avg(other, "adapted_epe")},
              "expert_group_means": expert_group_means,
              "other_group_mean": {"parent_epe": avg(other_group, "parent_epe"), "adapted_epe": avg(other_group, "adapted_epe")},
              "per_condition": per, "parent": args.parent, "checkpoint": args.checkpoint}
    atomic_json(Path(args.output), report)
    print(json.dumps({k: report[k] for k in ("clean", "expert_group_means", "other_group_mean")}, indent=2), flush=True)
    if has_adapter:
        print("router gate (mean sigmoid per condition):", flush=True)
        for c in conditions:
            g = per[c].get("router_gate")
            if g is not None:
                print("  %-18s %s" % (c, "  ".join("%s=%.3f" % (n, v) for n, v in zip(ROUTER_EXPERT_GROUPS, g))), flush=True)


if __name__ == "__main__":
    main()
