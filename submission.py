"""Clean 2-frame Spring test inference: writes <output>/<scene>/flow_<dir>_<cam>/flow_<dir>_<cam>_<frame>.flo5."""
import argparse
import os

import torch

from config.parser import parse_args
from dataloader.flow.spring import Spring
from model import fetch_model
from utils import frame_utils
from utils.utils import load_ckpt


@torch.no_grad()
def create_spring_submission(model, output_path, num_shards=1, shard_id=0):
    test_dataset = Spring(root=os.environ.get('SPRING_ROOT', 'datasets/spring'), split='test')
    scenes = sorted({info[1] for info in test_dataset.extra_info})
    my_scenes = scenes[shard_id::num_shards]
    for scene in my_scenes:
        ids = [i for i, info in enumerate(test_dataset.extra_info) if info[1] == scene]
        for test_id in ids:
            frame, _, cam, direction = test_dataset.extra_info[test_id]
            image1, image2, _ = test_dataset[test_id]
            flow = model(image1[None].cuda(), image2[None].cuda())['flow'][-1]
            output_dir = os.path.join(output_path, scene, f"flow_{direction}_{cam}")
            os.makedirs(output_dir, exist_ok=True)
            frame_utils.writeFlo5File(flow[0].permute(1, 2, 0).cpu().numpy(),
                                      os.path.join(output_dir, f"flow_{direction}_{cam}_{frame:04d}.flo5"))
        print(f"[shard {shard_id}/{num_shards}] done {scene} ({len(ids)} flows)", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', help='model config, e.g. config/clean-step4-spring.json', required=True, type=str)
    parser.add_argument('--ckpt', help='checkpoint path', required=True, type=str)
    parser.add_argument('--output_path', help='output directory', required=True, type=str)
    parser.add_argument('--num_shards', help='number of scene shards for multi-GPU submission', default=1, type=int)
    parser.add_argument('--shard_id', help='shard index handled by this process (0..num_shards-1)', default=0, type=int)
    args = parse_args(parser)
    args.pretrained_backbones = False  # every tensor comes from --ckpt
    model = fetch_model(args)
    load_ckpt(model, args.ckpt, strict=True)
    model = model.cuda().eval()
    create_spring_submission(model, args.output_path, args.num_shards, args.shard_id)


if __name__ == '__main__':
    main()
