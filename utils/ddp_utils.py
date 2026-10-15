import os

import numpy as np
import torch
import torch.distributed as dist


def calc_num_workers():
    return len(os.sched_getaffinity(0)) // dist.get_world_size()


def setup_ddp(rank, world_size):
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def init_ddp():
    """Single-node rendezvous; one process per visible GPU."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(11451 + np.random.randint(100))
    world_size = torch.cuda.device_count()
    assert world_size > 0, "You need a GPU!"
    return world_size
