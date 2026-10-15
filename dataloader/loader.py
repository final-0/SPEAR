import json
import os

import torch.utils.data as data
from torch.utils.data.distributed import DistributedSampler

from dataloader.flow.things import FlyingThings3D
from dataloader.flow.spring import Spring
from dataloader.flow.tartanair import TartanAir
from utils.ddp_utils import calc_num_workers


def fetch_dataset(args):
    """Training set of a clean step; `args.scale` is the log2 resize offset of the augmentation."""
    if args.dataset == 'things':
        aug_params = {'crop_size': args.image_size, 'min_scale': args.scale - 0.4, 'max_scale': args.scale + 0.8, 'do_flip': True}
        return FlyingThings3D(aug_params, root=os.environ.get('THINGS_ROOT', 'datasets/FlyingThings3D'), dstype='frames_cleanpass')
    if args.dataset == 'spring':
        root = os.environ.get('SPRING_ROOT', 'datasets/spring')
        aug_params = {'crop_size': args.image_size, 'min_scale': args.scale, 'max_scale': args.scale + 0.2, 'do_flip': True}
        dataset = Spring(aug_params, root=root, split='train')
        if args.spring_include_val:
            dataset = dataset + Spring(aug_params, root=root, split='val')
        return dataset
    if args.dataset == 'tartanair':
        aug_params = {'crop_size': args.image_size, 'min_scale': args.scale - 0.2, 'max_scale': args.scale + 0.6, 'do_flip': True}
        with open(args.tartan_trajectories) as f:
            trajectories = json.load(f)
        return TartanAir(aug_params, root=os.environ.get('TARTAN_ROOT', 'datasets/TartanAir'), trajectories=trajectories)
    raise ValueError(f"unsupported dataset: {args.dataset}")


def fetch_dataloader(args, rank, world_size):
    train_dataset = fetch_dataset(args)
    sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    loader = data.DataLoader(train_dataset, batch_size=args.batch_size // world_size,
                             num_workers=calc_num_workers(), sampler=sampler)
    print('Training with %d image pairs' % len(train_dataset))
    return loader
