import random

import numpy as np
import torch
import torch.utils.data as data

from utils import frame_utils
from dataloader.augmentor import FlowAugmentor


class FlowDataset(data.Dataset):
    """Base class: subclasses fill image_list / flow_list (/ extra_info) and implement read_flow()."""
    def __init__(self, aug_params=None):
        self.augmentor = FlowAugmentor(**aug_params) if aug_params is not None else None
        self.is_test = False
        self.flow_list = []
        self.image_list = []
        self.extra_info = []

    def __getitem__(self, index):
        # An unreadable sample is replaced once by a random other sample (WAFT behaviour).
        try:
            return self.fetch(index)
        except Exception:
            return self.fetch(random.randint(0, len(self) - 1))

    def read_flow(self, index):
        raise NotImplementedError

    def fetch(self, index):
        if self.is_test:
            img1 = np.array(frame_utils.read_gen(self.image_list[index][0])).astype(np.uint8)[..., :3]
            img2 = np.array(frame_utils.read_gen(self.image_list[index][1])).astype(np.uint8)[..., :3]
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            return img1, img2, self.extra_info[index]

        index = index % len(self.image_list)
        flow, valid = self.read_flow(index)
        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])
        flow = np.array(flow).astype(np.float32)
        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)
        if img1.ndim == 2:  # grayscale
            img1 = np.tile(img1[..., None], (1, 1, 3))
            img2 = np.tile(img2[..., None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        if self.augmentor is not None:
            img1, img2, flow, valid = self.augmentor(img1, img2, flow, valid)

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()
        valid = torch.from_numpy(valid)
        valid = (valid >= 0.5) & ((~torch.isnan(flow)).all(dim=0)) & ((~torch.isinf(flow)).all(dim=0))
        flow[torch.isinf(flow)] = 0
        flow[torch.isnan(flow)] = 0
        return img1, img2, flow, valid.float()

    def __len__(self):
        return len(self.image_list)
