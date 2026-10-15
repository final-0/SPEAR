import numpy as np
import os
from glob import glob

from utils import frame_utils
from dataloader.template import FlowDataset

class Spring(FlowDataset):
    """Spring split (train / val / test) with FW and BW pairs of both cameras.

    train/val items: image1, image2, flow, valid.  test items: image1, image2, (frame, scene, cam, direction).
    subsample_groundtruth=True takes every second pixel of the 4K ground truth (1920x1080 labels).
    """
    def __init__(self, aug_params=None, root='datasets/spring', split='train', subsample_groundtruth=True):
        super().__init__(aug_params)
        assert split in ["train", "val", "test"]
        seq_root = os.path.join(root, split)
        if not os.path.exists(seq_root):
            raise ValueError(f"Spring {split} directory does not exist: {seq_root}")
        self.subsample_groundtruth = subsample_groundtruth
        self.split = split
        self.seq_root = seq_root
        self.data_list = []
        if split == 'test':
            self.is_test = True
        for scene in sorted(os.listdir(seq_root)):
            for cam in ["left", "right"]:
                images = sorted(glob(os.path.join(seq_root, scene, f"frame_{cam}", '*.png')))
                # Extract frame numbers from filenames
                frame_nums = sorted([int(os.path.basename(p).split('_')[-1].split('.')[0]) for p in images])
                # forward: pair (frame_i, frame_i+1)
                for i in range(len(frame_nums) - 1):
                    self.data_list.append((frame_nums[i], scene, cam, "FW"))
                # backward: pair (frame_i, frame_i-1)
                for i in range(1, len(frame_nums)):
                    self.data_list.append((frame_nums[i], scene, cam, "BW"))

        for frame_data in self.data_list:
            frame, scene, cam, direction = frame_data

            img1_path = os.path.join(self.seq_root, scene, f"frame_{cam}", f"frame_{cam}_{frame:04d}.png")

            if direction == "FW":
                img2_path = os.path.join(self.seq_root, scene, f"frame_{cam}", f"frame_{cam}_{frame+1:04d}.png")
            else:
                img2_path = os.path.join(self.seq_root, scene, f"frame_{cam}", f"frame_{cam}_{frame-1:04d}.png")

            self.image_list += [[img1_path, img2_path]]
            self.extra_info += [frame_data]

            if split != 'test':
                flow_path = os.path.join(self.seq_root, scene, f"flow_{direction}_{cam}", f"flow_{direction}_{cam}_{frame:04d}.flo5")
                self.flow_list += [flow_path]

    def read_flow(self, index):
        flow = frame_utils.read_gen(self.flow_list[index])
        if self.subsample_groundtruth:
            flow = flow[::2, ::2]
        valid = (np.abs(flow[..., 0]) < 1000) & (np.abs(flow[..., 1]) < 1000)
        return flow, valid

