import os
from glob import glob

import numpy as np

from dataloader.template import FlowDataset


class TartanAir(FlowDataset):
    """TartanAir image_left + flow, restricted to `trajectories` = {scene: {"Easy": [...], "Hard": [...]}}."""
    def __init__(self, aug_params=None, root='datasets/TartanAir', trajectories=None):
        super().__init__(aug_params)
        for scene in sorted(trajectories):
            for difficulty in ('Easy', 'Hard'):
                for traj in sorted(trajectories[scene].get(difficulty, [])):
                    seq_dir = os.path.join(root, scene, difficulty, traj)
                    flow_dir = os.path.join(seq_dir, 'flow')
                    img_dir = os.path.join(seq_dir, 'image_left')
                    if not os.path.isdir(flow_dir) or not os.path.isdir(img_dir):
                        raise FileNotFoundError(f"TartanAir trajectory missing (image_left/ or flow/): {seq_dir}")
                    for flow_path in sorted(glob(os.path.join(flow_dir, '*_flow.npy'))):
                        f1_num, f2_num = os.path.basename(flow_path).replace('_flow.npy', '').split('_')[:2]
                        img1 = os.path.join(img_dir, f'{f1_num}_left.png')
                        img2 = os.path.join(img_dir, f'{f2_num}_left.png')
                        if os.path.exists(img1) and os.path.exists(img2):
                            self.image_list.append([img1, img2])
                            self.flow_list.append(flow_path)

    def read_flow(self, index):
        flow = np.load(self.flow_list[index]).astype(np.float32)
        valid = np.ones(flow.shape[:2], dtype=np.float32)
        return flow, valid
