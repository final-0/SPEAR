"""Clean 2-frame EPE on Spring train/val (left camera, forward flow, n_pairs pairs per sequence).

The 1080p prediction is compared against the 4K ground truth: each predicted pixel takes the
minimum EPE over its 2x2 ground-truth block.

    python scripts/evaluate_spring.py --ckpt checkpoints/clean-step4-spring/waft-a2/dav2/42/20000.pth --split all
"""
import argparse
import json
import os
from glob import glob
from types import SimpleNamespace

import cv2
import numpy as np
import torch

from model import fetch_model
from utils import frame_utils
from utils.utils import load_ckpt


def load_model(ckpt_path, cfg_path, device):
    with open(cfg_path) as f:
        args = SimpleNamespace(**json.load(f))
    args.pretrained_backbones = False  # every tensor comes from ckpt_path
    model = fetch_model(args)
    load_ckpt(model, ckpt_path, strict=True)
    return model.to(device).eval()


@torch.inference_mode()
def evaluate_sequence(model, seq_dir, cam, n_pairs, device):
    frame_dir = os.path.join(seq_dir, f"frame_{cam}")
    flow_dir = os.path.join(seq_dir, f"flow_FW_{cam}")
    if not os.path.exists(frame_dir) or not os.path.exists(flow_dir):
        return None

    frames = sorted(glob(os.path.join(frame_dir, "*.png")))
    flows = sorted(glob(os.path.join(flow_dir, "*.flo5")))
    if len(frames) < 2 or len(flows) == 0:
        return None

    frame_nums = sorted([int(os.path.basename(f).split("_")[-1].split(".")[0]) for f in frames])
    n_eval = min(n_pairs, len(frame_nums) - 1)
    step = max(1, (len(frame_nums) - 1) // n_eval)
    indices = list(range(0, len(frame_nums) - 1, step))[:n_eval]

    epe_list = []
    for idx in indices:
        f1_num = frame_nums[idx]
        f2_num = frame_nums[idx + 1]

        img1_path = os.path.join(frame_dir, f"frame_{cam}_{f1_num:04d}.png")
        img2_path = os.path.join(frame_dir, f"frame_{cam}_{f2_num:04d}.png")
        flow_path = os.path.join(flow_dir, f"flow_FW_{cam}_{f1_num:04d}.flo5")

        if not all(os.path.exists(p) for p in [img1_path, img2_path, flow_path]):
            continue

        img1 = torch.from_numpy(np.array(cv2.imread(img1_path))[..., :3]).permute(2, 0, 1).float()[None].to(device)
        img2 = torch.from_numpy(np.array(cv2.imread(img2_path))[..., :3]).permute(2, 0, 1).float()[None].to(device)

        output = model(img1, img2)
        pred = output['flow'][-1][0].cpu().numpy().transpose(1, 2, 0)

        gt_4k = frame_utils.read_gen(flow_path).astype(np.float32)
        H, W = pred.shape[:2]
        if gt_4k.shape[0] == 2 * H and gt_4k.shape[1] == 2 * W:
            gt_blocks = gt_4k.reshape(H, 2, W, 2, 2)
            gt_cand = gt_blocks.transpose(0, 2, 1, 3, 4).reshape(H, W, 4, 2)
            diff = pred[:, :, None, :] - gt_cand
            epe_4 = np.sqrt((diff ** 2).sum(axis=3))
            epe = epe_4.min(axis=2)
        else:
            gt = gt_4k[::2, ::2]
            epe = np.sqrt(np.sum((pred - gt) ** 2, axis=2))

        valid = np.isfinite(epe) & (epe < 1000)
        if valid.sum() > 0:
            epe_list.append(epe[valid].mean())

    if not epe_list:
        return None
    return {"epe": np.mean(epe_list), "n": len(epe_list)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--cfg", default="config/clean-step4-spring.json", help="model config")
    parser.add_argument("--split", default="all", choices=["train", "val", "all"])
    parser.add_argument("--n_pairs", type=int, default=5)
    parser.add_argument("--cam", default="left")
    parser.add_argument("--spring_root", default=None)
    args_cli = parser.parse_args()

    spring_root = args_cli.spring_root or os.environ.get("SPRING_ROOT", "datasets/spring")
    device = "cuda"

    print(f"Loading: {args_cli.ckpt}", flush=True)
    print(f"Config: split={args_cli.split} n_pairs={args_cli.n_pairs} cam={args_cli.cam} cfg={args_cli.cfg}", flush=True)

    model = load_model(args_cli.ckpt, args_cli.cfg, device)

    splits = []
    if args_cli.split in ("train", "all"):
        splits.append(("train", os.path.join(spring_root, "train")))
    if args_cli.split in ("val", "all"):
        splits.append(("val", os.path.join(spring_root, "val")))

    print(f"\n{'Split':<6} {'Seq':<6} {'EPE':<8} {'Pairs':<6}", flush=True)
    print("-" * 28, flush=True)

    all_results = []
    for split_name, split_dir in splits:
        if not os.path.exists(split_dir):
            print(f"{split_name}: directory not found", flush=True)
            continue
        for seq in sorted(os.listdir(split_dir)):
            seq_dir = os.path.join(split_dir, seq)
            if not os.path.isdir(seq_dir):
                continue
            r = evaluate_sequence(model, seq_dir, args_cli.cam, args_cli.n_pairs, device)
            if r is None:
                continue
            all_results.append(r)
            print(f"{split_name:<6} {seq:<6} {r['epe']:<8.4f} {r['n']:<6}", flush=True)

    print("-" * 28, flush=True)
    if all_results:
        avg = np.mean([r["epe"] for r in all_results])
        total = sum(r["n"] for r in all_results)
        print(f"{'AVG':<6} {'':6} {avg:<8.4f} {total:<6}", flush=True)


if __name__ == "__main__":
    main()
