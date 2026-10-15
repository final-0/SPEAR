"""Clean training step (TartanAir / FlyingThings3D / Spring); one process per visible GPU."""
import argparse
import os
import random
import time

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.optim as optim
from torch.amp import GradScaler, autocast

from config.parser import parse_args
from criterion.loss import sequence_loss
from dataloader.flow.spring import Spring
from dataloader.loader import fetch_dataloader
from model import fetch_model
from utils.ddp_utils import init_ddp, setup_ddp
from utils.utils import load_ckpt

VAL_FREQ = 500
VAL_SAMPLES = 50
# refine_freeze trains only these modules (new feature projection); everything else keeps its warm-start weights.
REFINE_FREEZE_TRAINABLE = ('fnet.', 'fmap_conv.', 'encoder.dpt_head.')


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0

    def update(self, val):
        self.sum += val
        self.count += 1

    @property
    def avg(self):
        return self.sum / max(self.count, 1)


@torch.no_grad()
def validate_spring(model, step):
    """EPE on the first 50 Spring val pairs (fp32, rank 0 only)."""
    model.eval()
    dataset = Spring(root=os.environ.get('SPRING_ROOT', 'datasets/spring'), split='val')
    epes = []
    for i in range(min(len(dataset), VAL_SAMPLES)):
        image1, image2, flow, valid = [x[None].cuda() for x in dataset[i]]
        pred = model(image1, image2)['flow'][-1]
        epe = ((flow - pred) ** 2).sum(dim=1).sqrt()
        if valid.sum() > 0:
            epes.append((epe * valid).sum().item() / valid.sum().item())
    if epes:
        print(f"[VAL] step={step} epe={sum(epes) / len(epes):.4f} (n={len(epes)})", flush=True)
    model.train()


def save_checkpoint(model, args, step):
    save_dir = os.path.join('checkpoints', args.name, args.algorithm, args.feature_encoder, str(args.seed))
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.module.state_dict(), os.path.join(save_dir, f"{step}.pth"))


def train(args, rank, world_size):
    model = fetch_model(args).to(rank)
    if args.restore_ckpt is not None:
        load_ckpt(model, args.restore_ckpt)
        print(f"restore ckpt from {args.restore_ckpt}")
    if args.refine_freeze:
        for name, param in model.named_parameters():
            param.requires_grad = name.startswith(REFINE_FREEZE_TRAINABLE)
    if rank == 0:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"refine_freeze={args.refine_freeze}, trainable parameters: {trainable}", flush=True)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank], static_graph=True)
    model.train()
    train_loader = fetch_dataloader(args, rank=rank, world_size=world_size)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params, lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps + 100,
                                              pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')
    scaler = GradScaler('cuda', enabled=args.use_bf16)
    checkpoint_freq = getattr(args, 'checkpoint_freq', 50000)
    avg_loss, avg_epe = AverageMeter(), AverageMeter()
    if rank == 0:
        print(f"Mixed precision: {'bf16' if args.use_bf16 else 'fp32'}", flush=True)
        validate_spring(model, 0)

    start_time = time.time()
    step = 0
    epoch = 0
    while step < args.num_steps:
        train_loader.sampler.set_epoch(epoch)
        epoch += 1
        for data_blob in train_loader:
            optimizer.zero_grad()
            image1, image2, flow, valid = [x.cuda(non_blocking=True) for x in data_blob]
            with autocast('cuda', dtype=torch.bfloat16, enabled=args.use_bf16):
                output = model(image1, image2, flow_gt=flow)
                loss = sequence_loss(output, flow, valid, args.gamma)

            if rank == 0:
                if valid.sum() > 0:
                    avg_loss.update(loss.item())
                    epe = ((flow - output['flow'][-1]) ** 2).sum(dim=1).sqrt()
                    avg_epe.update(((epe * valid).sum() / valid.sum()).item())
                if step % 100 == 0:
                    print(f"[TRAIN] step={step}/{args.num_steps} loss={avg_loss.avg:.4f} epe={avg_epe.avg:.3f} "
                          f"lr={scheduler.get_last_lr()[0]:.2e} elapsed={(time.time() - start_time) / 60:.1f}min", flush=True)
                    avg_loss.reset()
                    avg_epe.reset()

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, args.clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            step += 1

            if rank == 0 and step % VAL_FREQ == 0:
                validate_spring(model, step)
            if rank == 0 and (step % checkpoint_freq == 0 or step == args.num_steps):
                save_checkpoint(model, args, step)
            if step >= args.num_steps:
                break


def main(rank, world_size, args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    setup_ddp(rank, world_size)
    train(args, rank, world_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', help='experiment config (config/clean-step*.json)', required=True, type=str)
    parser.add_argument('--seed', help='seed', default=42, type=int)
    parser.add_argument('--restore_ckpt', help='overrides restore_ckpt of the config', default=None, type=str)
    args = parse_args(parser)
    world_size = init_ddp()
    mp.spawn(main, nprocs=world_size, args=(world_size, args))
