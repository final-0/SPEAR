"""RoCo test inference with the final SPEAR model: clean Spring test + the 20 RobustSpring corruptions.

Produces flo5 predictions in the devkit layout:
  <output>/<branch>/test/<scene>/flow_<dir>_<cam>/flow_<dir>_<cam>_<t>.flo5
where branch is 'clean' (from spring/test) or one of the 20 corruptions
(from robust_spring/<corruption>/test). The 3-frame robust model predicts the
center->target flow (FW: t->t+1, BW: t->t-1), matching training semantics.
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from utils import frame_utils
from .corruptions import ROBUSTSPRING_CORRUPTIONS, impulse_median_gate
from .model import DEFAULT_PARENT, build_model
from .runtime import autocast, load_training, precision

BRANCHES = ("clean",) + ROBUSTSPRING_CORRUPTIONS


def branch_root(branch, spring_root, robust_root):
    if branch == "clean":
        return Path(spring_root) / "test"
    return Path(robust_root) / branch / "test"


def list_frames(scene_dir, cam):
    d = scene_dir / f"frame_{cam}"
    out = {}
    for p in d.glob(f"frame_{cam}_*.png"):
        out[int(p.stem.rsplit("_", 1)[1])] = p
    return out


def load_arr(path):
    a = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    if a.shape != (1080, 1920, 3):
        raise ValueError(f"expected native 1080x1920, got {a.shape}: {path}")
    return np.ascontiguousarray(a)


def arr_to_tensor(a, device):
    return torch.from_numpy(np.ascontiguousarray(a.transpose(2, 0, 1)))[None].to(device).float()


def work_items(branches, roots):
    items = []
    for branch in branches:
        root = branch_root(branch, *roots)
        if not root.is_dir():
            raise FileNotFoundError(f"missing branch root: {root}")
        for scene in sorted(root.iterdir()):
            if scene.is_dir() and scene.name.isdigit():
                items.append((branch, scene))
    return items


@torch.no_grad()
def run(args):
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    _, dtype = precision(args.precision, device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    model = build_model(args.parent, args.attention, False).to(device).eval()
    if args.checkpoint:
        load_training(args.checkpoint, model)

    branches = args.branches or list(BRANCHES)
    items = work_items(branches, (args.spring_root, args.robust_root))
    mine = [it for i, it in enumerate(items) if i % args.num_shards == args.shard_id]
    print(f"[shard {args.shard_id}/{args.num_shards}] {len(mine)}/{len(items)} (branch,scene) items", flush=True)
    print(f"[shard {args.shard_id}] median-gate tau={args.median_gate}", flush=True)

    written = 0
    gated = 0
    for branch, scene in mine:
        for cam in ("left", "right"):
            frames = list_frames(scene, cam)
            if len(frames) < 2:
                continue
            for direction in ("FW", "BW"):
                delta = 1 if direction == "FW" else -1
                out_dir = Path(args.output) / branch / "test" / scene.name / f"flow_{direction}_{cam}"
                out_dir.mkdir(parents=True, exist_ok=True)
                for t in sorted(frames):
                    target = t + delta
                    if target not in frames:
                        continue
                    opp = t - delta if (t - delta) in frames else t  # boundary duplicates center
                    arrs = [load_arr(frames[opp]), load_arr(frames[t]), load_arr(frames[target])]
                    arrs, fired = impulse_median_gate(arrs, args.median_gate)
                    gated += int(fired)
                    prev_img, center, follow = (arr_to_tensor(a, device) for a in arrs)
                    with autocast(device, dtype):
                        out = model(prev_img, center, follow, final_only=True)
                    flow = out["flow"][-1][0].permute(1, 2, 0).float().cpu().numpy()  # HxWx2
                    frame_utils.writeFlo5File(flow, str(out_dir / f"flow_{direction}_{cam}_{t:04d}.flo5"))
                    written += 1
        print(f"[shard {args.shard_id}] done {branch}/{scene.name} (total {written})", flush=True)
    print(f"[shard {args.shard_id}] COMPLETE, wrote {written} flo5 (median-gated triplets: {gated})", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent", default=DEFAULT_PARENT)
    p.add_argument("--checkpoint", help="adapter delta .pt (omit for parent-only)")
    p.add_argument("--spring-root", required=True, help="spring root containing test/")
    p.add_argument("--robust-root", required=True, help="robust_spring root containing <corruption>/test/")
    p.add_argument("--output", required=True)
    p.add_argument("--branches", nargs="+", help="subset of branches; default all 21")
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-id", type=int, default=0)
    p.add_argument("--precision", choices=("auto", "bf16", "fp16", "fp32"), default="auto")
    p.add_argument("--attention", choices=("auto", "chunked"), default="auto")
    p.add_argument("--median-gate", type=float, default=10.0, help="impulse median-gate threshold in percent (calibrated on RobustSpring test); 0 disables")
    args = p.parse_args()
    if not (0 <= args.shard_id < args.num_shards):
        p.error("shard_id must be in [0, num_shards)")
    run(args)


if __name__ == "__main__":
    main()
