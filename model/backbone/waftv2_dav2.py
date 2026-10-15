import torch
import torch.nn as nn
import torch.nn.functional as F

from model.backbone.head import DPTHead
from thirdparty.DepthAnythingV2.depth_anything_v2.dpt import DepthAnythingV2

DEPTH_ANYTHING_CONFIGS = {
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
}
INTERMEDIATE_LAYERS = {'vits': [2, 5, 8, 11], 'vitb': [2, 5, 8, 11], 'vitl': [4, 11, 17, 23]}


class DepthAnythingFeature(nn.Module):
    """Frozen DAv2 ViT encoder + a trainable DPT head producing 1/2-resolution features."""
    def __init__(self, model_name='vits', pretrained=True, lvl=-3):
        super().__init__()
        config = DEPTH_ANYTHING_CONFIGS[model_name]
        self.model_name = model_name
        self.intermediate_layer_idx = INTERMEDIATE_LAYERS
        depth_anything = DepthAnythingV2(**config)
        if pretrained:
            depth_anything.load_state_dict(torch.load(f'depth-anything-ckpts/depth_anything_v2_{model_name}.pth', map_location='cpu'))
        self.encoder = depth_anything.pretrained.eval().requires_grad_(False)
        self.embed_dim = self.encoder.embed_dim
        self.output_dim = config['features']
        self.dpt_head = DPTHead(self.embed_dim, features=self.output_dim, out_channels=config['out_channels'], lvl=lvl)

    def forward(self, x):
        h, w = x.shape[-2:]
        features = self.encoder.get_intermediate_layers(x, self.intermediate_layer_idx[self.model_name], return_class_token=True)
        patch_size = self.encoder.patch_size
        outs = self.dpt_head(features, h // patch_size, w // patch_size)
        return F.interpolate(outs[0], size=(h // 2, w // 2), mode='bilinear', align_corners=True)
