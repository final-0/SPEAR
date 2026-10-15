"""Portable checkpoint/resume helpers; parent weights remain immutable."""
import hashlib
import json
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from .model import PARENT_SHA256


def precision(request, device):
    native_bf16 = device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 8 and torch.cuda.is_bf16_supported()
    if request == "auto":
        request = "bf16" if native_bf16 else "fp16" if device.type == "cuda" else "fp32"
    if request == "bf16" and device.type == "cuda" and not native_bf16:
        raise ValueError("native BF16 is unavailable; use FP16 + GradScaler on V100")
    return request, {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[request]


def autocast(device, dtype):
    return torch.autocast(device.type, dtype=dtype, enabled=dtype != torch.float32)


def source_sha256():
    root = Path(__file__).resolve().parents[1]
    paths = sorted((root / "robust_finetune").glob("*.py"))
    paths += sorted((root / "model").rglob("*.py"))
    paths += sorted((root / "thirdparty").rglob("*.py"))
    paths += sorted((root / "utils").rglob("*.py"))
    paths += [root / "config/clean-step4-spring.json"]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    with os.fdopen(fd, "w") as f:
        json.dump(value, f, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(name, path)


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None}


def save_training(path, model, optimizer, scaler, step, signature, rank, world):
    states = [None] * world
    if world > 1:
        dist.all_gather_object(states, rng_state())
    else:
        states[0] = rng_state()
    if rank == 0:
        path = Path(path)
        if path.exists():
            raise FileExistsError(f"refusing checkpoint overwrite: {path}")
        payload = {"format": "waft-large-3f-native-v1", "parent_sha256": PARENT_SHA256,
                   "delta": model.delta_state(), "optimizer": optimizer.state_dict(),
                   "scaler": scaler.state_dict(), "step": step, "signature": signature,
                   "rng": states}
        fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
        os.close(fd)
        torch.save(payload, name)
        os.replace(name, path)
        atomic_json(path.parent / "latest.json", {"checkpoint": path.name, "step": step})
    if world > 1:
        dist.barrier()


def load_training(path, model):
    # Resume contains Python/NumPy RNG state. Only load this user's own generated
    # checkpoints; unlike the pinned bundled parent these require weights_only=False.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "waft-large-3f-native-v1" or payload.get("parent_sha256") != PARENT_SHA256:
        raise ValueError("not a compatible WAFT-large three-frame checkpoint")
    model.load_delta(payload["delta"])
    return payload


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"])
