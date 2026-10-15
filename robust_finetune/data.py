"""Train-only Spring triplets; no resizing/cropping or silent bad-file retries."""
import hashlib
import json
import random
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from .corruptions import CORRUPTION_GROUPS, ROBUSTSPRING_CORRUPTIONS, RobustSpringCorruptor, expert_group_label, sample_severity

WEATHER = tuple(CORRUPTION_GROUPS["weather"])
OTHER = tuple(c for c in ROBUSTSPRING_CORRUPTIONS if c not in WEATHER)
NATIVE = (1080, 1920)


def check_native(image):
    if image.shape != (*NATIVE, 3) or image.dtype != np.uint8:
        raise ValueError(f"expected native 1080x1920 RGB uint8, got {image.shape}/{image.dtype}; NO resize/crop fallback")


def read_flow(path):
    with h5py.File(path, "r") as f:
        ds = f["flow"]
        if ds.shape == (2160, 3840, 2):
            flow = np.asarray(ds[::2, ::2], dtype=np.float32)
        elif ds.shape == (*NATIVE, 2):
            flow = np.asarray(ds, dtype=np.float32)
        else:
            raise ValueError(f"unexpected GT shape {ds.shape}: {path}")
    # Spring's 4K label grid stores displacement in ORIGINAL IMAGE pixels.
    # Sampling the grid is NOT an input resize; do NOT divide flow by two.
    valid = np.isfinite(flow).all(-1) & (np.abs(flow) < 1000).all(-1)
    flow[~valid] = 0
    if not valid.any():
        raise ValueError(f"no valid flow pixels: {path}")
    return flow, valid.astype(np.float32)


def choose_condition(rng):
    probability = rng.random()
    if probability < .25:
        return "clean"
    return rng.choice(WEATHER if probability < .75 else OTHER)


def tensor_image(image):
    return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))


class NativeSpring(Dataset):
    def __init__(self, root, partition="train", cameras=("left", "right"), directions=("FW", "BW"), prefix=3, seed=91):
        if partition not in ("train", "diagnostic") or prefix < 1:
            raise ValueError("only train-derived partitions with a positive prefix are supported")
        if not cameras or not set(cameras) <= {"left", "right"} or not directions or not set(directions) <= {"FW", "BW"}:
            raise ValueError("invalid cameras/directions")
        self.root = Path(root).resolve()
        train = self.root / "train"  # never enumerate val/ or test/
        if not train.is_dir():
            raise FileNotFoundError(f"expected {train}")
        self.seed = seed
        self.records = []
        self.partition = partition
        for sequence in sorted(train.iterdir()):
            if not sequence.is_dir() or not sequence.name.isdigit():
                continue
            for camera in cameras:
                directory = sequence / f"frame_{camera}"
                images = {int(p.stem.rsplit("_", 1)[1]): p for p in directory.glob(f"frame_{camera}_*.png")}
                if len(images) < prefix + 2:
                    raise ValueError(f"missing/too short requested camera: {directory}")
                numbers = sorted(images)
                if numbers != list(range(numbers[0], numbers[-1] + 1)):
                    raise ValueError(f"non-contiguous frames: {directory}")
                for direction in directions:
                    delta = 1 if direction == "FW" else -1
                    centers = [t for t in numbers if t + delta in images]
                    diagnostic = set(centers[:prefix])
                    for t in centers:
                        if (t in diagnostic) != (partition == "diagnostic"):
                            continue
                        previous = t - delta if t - delta in images else t
                        gt = sequence / f"flow_{direction}_{camera}" / f"flow_{direction}_{camera}_{t:04d}.flo5"
                        if not gt.is_file():
                            raise FileNotFoundError(f"missing requested GT (no silent skip): {gt}")
                        self.records.append({"id": f"{sequence.name}/{camera}/{direction}/{t:04d}", "images": [images[previous], images[t], images[t+delta]], "gt": gt, "times": (previous, t, t+delta)})
        if not self.records:
            raise ValueError("empty train-derived partition")
        # Portable layout/size digest, not a claim of byte-hashing the dataset.
        layout = [{"id": r["id"], "files": [(str(p.relative_to(self.root)), p.stat().st_size) for p in (*r["images"], r["gt"])]} for r in self.records]
        self.layout_sha256 = hashlib.sha256(json.dumps(layout, sort_keys=True).encode()).hexdigest()

    def __len__(self):
        return len(self.records)

    def fetch(self, record_index, condition="clean", severity=3, corruption_seed=0):
        record = self.records[record_index]
        images = []
        for path in record["images"]:
            with Image.open(path) as im:
                rgb = np.array(im.convert("RGB"))
            check_native(rgb)
            images.append(rgb)
        flow, valid = read_flow(record["gt"])
        if condition == "clean":
            corrupted, target, mask = images, flow, valid
        else:
            if condition not in ROBUSTSPRING_CORRUPTIONS:
                raise ValueError(condition)
            # Only forward-in-triplet GT is consumed. Dummy opposite GT is never
            # supervised; apply_triplet still transforms the real target for elastic.
            corrupted, _, _, target, mask = RobustSpringCorruptor.apply_triplet(
                images, np.zeros_like(flow), np.zeros_like(valid), flow, valid,
                condition, severity, corruption_seed, frame_indices=record["times"],
            )
        if record["times"][0] == record["times"][1]:
            # Duplicate the ALREADY corrupted source at a missing-context boundary.
            # In particular elastic's extrapolated previous field is not a real frame.
            corrupted = list(corrupted)
            corrupted[0] = corrupted[1].copy()
        for image in corrupted:
            check_native(image)
        return {"clean": torch.stack([tensor_image(i) for i in images]),
                "corrupt": torch.stack([tensor_image(i) for i in corrupted]),
                "flow": torch.from_numpy(flow.transpose(2, 0, 1).copy()),
                "valid": torch.from_numpy(valid.copy()),
                "corrupt_flow": torch.from_numpy(target.transpose(2, 0, 1).copy()),
                "corrupt_valid": torch.from_numpy(mask.copy()),
                "router": torch.tensor(expert_group_label(None if condition == "clean" else condition), dtype=torch.float32),
                "condition": condition, "id": record["id"], "severity": severity}

    def __getitem__(self, logical_index):
        # Stateless per-global-sample randomness: rank/worker count cannot change it.
        rng = random.Random(self.seed + int(logical_index) * 1000003)
        row = rng.randrange(len(self.records))
        condition = choose_condition(rng)
        severity = sample_severity(condition, rng)
        return self.fetch(row, condition, severity, rng.randrange(2**31))


def sample_indices(start, stop, global_batch, micro_batch, rank, world):
    if global_batch % (micro_batch * world):
        raise ValueError("global batch must be divisible by micro batch * world size")
    for step in range(start, stop):
        for accumulation in range(global_batch // (micro_batch * world)):
            first = step * global_batch + accumulation * micro_batch * world + rank * micro_batch
            yield from range(first, first + micro_batch)
