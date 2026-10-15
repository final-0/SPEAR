"""SPEAR stage 2 (torchrun entry point): --stage router, then --stage adapters --init <router step>."""
import argparse
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from .data import NativeSpring, sample_indices
from .evaluate import evaluate_model
from .engine import backward_microbatch
from .model import DEFAULT_PARENT, PARENT_SHA256, build_model
from .runtime import atomic_json, load_training, precision, restore_rng, save_training, source_sha256


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("router", "adapters"), required=True)
    p.add_argument("--spring-root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--parent", default=DEFAULT_PARENT)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--init", help="previous stage checkpoint; reset stage optimizer")
    g.add_argument("--resume", help="same stage checkpoint; restore optimizer, samples and RNG")
    p.add_argument("--steps", type=int, default=3000, help="target optimizer updates, not micro-batches; can extend with resume")
    p.add_argument("--global-batch", type=int, default=8)
    p.add_argument("--micro-batch", type=int, default=1)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4, help="constant after warmup")
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--clean-regret", type=float, default=1.0)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--eval-every", type=int, default=3000, help="0 disables in-training all20 diagnostics")
    p.add_argument("--prefix", type=int, default=3)
    p.add_argument("--seed", type=int, default=91)
    p.add_argument("--cameras", nargs="+", choices=("left", "right"), default=["left", "right"])
    p.add_argument("--directions", nargs="+", choices=("FW", "BW"), default=["FW", "BW"])
    p.add_argument("--precision", choices=("auto", "bf16", "fp16", "fp32"), default="auto")
    p.add_argument("--attention", choices=("auto", "chunked"), default="auto")
    args = p.parse_args()
    if min(args.steps, args.global_batch, args.micro_batch, args.save_every, args.prefix) < 1 or min(args.workers, args.warmup, args.eval_every, args.clean_regret) < 0:
        p.error("invalid nonpositive/negative option")
    if args.stage == "adapters" and not (args.init or args.resume):
        p.error("adapters must start from the router stage via --init (or resume an adapters run)")
    if args.lr <= 0:
        p.error("lr must be positive")
    return args


def main():
    args = arguments()
    rank, world, local = [int(os.environ.get(k, d)) for k, d in (("RANK", 0), ("WORLD_SIZE", 1), ("LOCAL_RANK", 0))]
    if args.global_batch % (args.micro_batch * world):
        raise ValueError("global batch must be divisible by micro batch * world size")
    accum = args.global_batch // (args.micro_batch * world)
    plan = {"stage": args.stage, "world_size": world, "micro_batch": args.micro_batch, "accumulation": accum, "global_batch": args.global_batch, "input_frames": 3, "input_hw": [1080, 1920], "resize": False, "crop": False, "target_steps": args.steps, "lr": args.lr}
    device = torch.device("cuda", local)
    torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group("nccl")
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed)  # identical zero/residual initialization across ranks
    precision_name, dtype = precision(args.precision, device)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    output = Path(args.output)
    if rank == 0:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError("use a NEW output directory, including for resume (preserves history)")
        output.mkdir(parents=True, exist_ok=True)
    if world > 1:
        dist.barrier()
    data = NativeSpring(args.spring_root, cameras=args.cameras, directions=args.directions, prefix=args.prefix, seed=args.seed)
    model = build_model(args.parent, args.attention).to(device)
    payload = load_training(args.resume or args.init, model) if (args.resume or args.init) else None
    if args.init and payload["signature"]["stage"] != "router":
        raise ValueError("--init must be a router-stage checkpoint")
    model.set_stage(args.stage)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-5, eps=1e-8)
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16, init_scale=128)
    # Constant LR after warmup: extending a run does not reset or redefine a cosine schedule.
    signature = {"stage": args.stage, "parent_sha256": PARENT_SHA256, "source_sha256": source_sha256(),
                 "layout_sha256": data.layout_sha256, "seed": args.seed, "global_batch": args.global_batch,
                 "world_size": world, "micro_batch": args.micro_batch, "precision": precision_name,
                 "attention": args.attention, "lr": args.lr, "warmup": args.warmup,
                 "clean_regret": args.clean_regret,
                 "input_hw": [1080, 1920], "frames": 3, "torch": torch.__version__}
    start = 0
    if args.resume:
        if payload["signature"] != signature:
            raise ValueError("resume signature differs (including world size); do not silently restart")
        optimizer.load_state_dict(payload["optimizer"])
        scaler.load_state_dict(payload["scaler"])
        start = payload["step"]
        restore_rng(payload["rng"][rank])
    if start >= args.steps:
        raise ValueError("target steps must exceed restored step")
    model.train()
    ddp = DDP(model, device_ids=[local], broadcast_buffers=False, find_unused_parameters=False) if world > 1 else model
    sampler = list(sample_indices(start, args.steps, args.global_batch, args.micro_batch, rank, world))
    loader = DataLoader(data, batch_size=args.micro_batch, sampler=sampler, num_workers=args.workers, pin_memory=True, **({"prefetch_factor": 1} if args.workers else {}))
    batches = iter(loader)
    if rank == 0:
        plan.update(signature)
        plan.update({"dataset_records": len(data), "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad), "parent": args.parent, "resume": args.resume, "init": args.init, "gpu": torch.cuda.get_device_name(device)})
        atomic_json(output / "run.json", plan)
        print(json.dumps(plan, indent=2), flush=True)
    begin = time.monotonic()
    for step in range(start, args.steps):
        optimizer.zero_grad(set_to_none=True)
        lr = args.lr * min(1.0, (step + 1) / max(1, args.warmup))
        for group in optimizer.param_groups:
            group["lr"] = lr
        stats = torch.zeros(3, device=device)
        for ai in range(accum):
            batch = next(batches)
            batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
            stats += backward_microbatch(ddp, model, batch, args.stage, device, dtype,
                                         scaler, accum, ai == accum - 1, args.clean_regret)
            del batch
        scaler.unscale_(optimizer)
        parameters = [p for p in model.parameters() if p.requires_grad]
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        if not torch.isfinite(grad_norm) or not torch.isfinite(stats).all():
            # Stop, rather than calling a skipped AMP update a completed training step.
            raise FloatingPointError("non-finite loss/gradient; checkpoint not advanced, inspect precision/loss")
        scaler.step(optimizer)
        scaler.update()
        completed = step + 1
        if world > 1:
            dist.all_reduce(stats)
            stats /= world
        if rank == 0 and (completed % 10 == 0 or completed == start + 1):
            record = {"step": completed, "lr": lr, "clean_nf": stats[0].item(), "corrupt_objective": stats[1].item(), "router_bce": stats[2].item(), "amp_scale": scaler.get_scale(), "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30, "elapsed_seconds": time.monotonic()-begin}
            print(json.dumps(record), flush=True)
            with (output / "train.jsonl").open("a") as f:
                f.write(json.dumps(record) + "\n")
        if completed % args.save_every == 0 or completed == args.steps:
            save_training(output / f"step-{completed:06d}.pt", model, optimizer, scaler, completed, signature, rank, world)
        if args.stage != "router" and args.eval_every and completed % args.eval_every == 0:
            diagnostic = NativeSpring(args.spring_root, "diagnostic", cameras=("left",), directions=("FW",), prefix=args.prefix)
            report = evaluate_model(model, diagnostic, device, dtype, rank, world)
            if rank == 0:
                atomic_json(output / f"all20-step-{completed:06d}.json", report)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
