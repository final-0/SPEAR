"""Extract the RobustSpring frost / rain / snow layers used by SPEAR stage 2.

These corruptions are content-independent additive layers:
  corrupted = clip(clean + layer)  ->  layer ~= corrupted - clean
frost is static, so one layer per sequence is kept; rain/snow keep --num random layers.

  python3 scripts/extract_weather_overlays.py --corruption snow --num 120 \
      --clean-root $SPRING_ROOT/test --robust-root $ROBUST_SPRING_ROOT --out-dir $WEATHER_OVERLAY_DIR
"""
import argparse
import os
import random
import numpy as np
from PIL import Image

IMG_EXT = ".png"


def _list_pairs(clean_root, target_root):
    pairs = []
    for dirpath, _, files in os.walk(clean_root):
        for f in files:
            if not f.lower().endswith(IMG_EXT):
                continue
            cpath = os.path.join(dirpath, f)
            rel = os.path.relpath(cpath, clean_root)
            tpath = os.path.join(target_root, rel)
            if os.path.exists(tpath):
                pairs.append((cpath, tpath))
    pairs.sort()
    return pairs


def load(p):
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--corruption", required=True, choices=["frost", "rain", "snow"])
    p.add_argument("--clean-root", required=True, help="e.g. .../spring/test")
    p.add_argument("--robust-root", required=True, help="e.g. .../robust_spring")
    p.add_argument("--out-dir", default=os.environ.get("WEATHER_OVERLAY_DIR", "datasets/weather_overlays"))
    p.add_argument("--num", type=int, default=120,
                   help="number of overlays to sample (frost: one per sequence)")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    target_root = os.path.join(args.robust_root, args.corruption, "test")
    pairs = _list_pairs(args.clean_root, target_root)
    if not pairs:
        raise RuntimeError("no pairs found; check --clean-root / --robust-root")

    rng = random.Random(args.seed)
    if args.corruption == "frost":
        seen, chosen = set(), []
        for c, t in pairs:
            seq = os.path.relpath(c, args.clean_root).split(os.sep)[0]
            if seq not in seen:
                seen.add(seq)
                chosen.append((c, t))
    else:
        chosen = rng.sample(pairs, min(args.num, len(pairs)))

    out = os.path.join(args.out_dir, args.corruption)
    os.makedirs(out, exist_ok=True)
    for i, (c, t) in enumerate(chosen):
        layer = np.clip(load(t) - load(c), 0.0, 1.0)
        Image.fromarray((layer * 255.0 + 0.5).astype(np.uint8)).save(
            os.path.join(out, f"{args.corruption}_{i:04d}.png"))
    print(f"[{args.corruption}] saved {len(chosen)} overlays to {out}")


if __name__ == "__main__":
    main()
