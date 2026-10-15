import math

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.backbone.patch_embed import PatchEmbed
from thirdparty.DepthAnythingV2.depth_anything_v2.dpt import DPTHead

MODEL_CONFIGS = {
    'vits': {'encoder': 'vit_small_patch16_224', 'features': 64, 'out_channels': [48, 96, 192, 384]},
}
INTERMEDIATE_LAYERS = {'vits': [2, 5, 8, 11]}


class VisionTransformer(nn.Module):
    """ViT refinement operator: timm ViT blocks on `input_dim`-channel patches, decoded by a DPT head."""
    def __init__(self, model_name, input_dim, patch_size=16, pretrained=True):
        super().__init__()
        model = timm.create_model(MODEL_CONFIGS[model_name]['encoder'], pretrained=pretrained, num_classes=0)
        self.idx = INTERMEDIATE_LAYERS[model_name]
        self.blks = model.blocks
        self.embed_dim = model.embed_dim
        self.patch_size = patch_size
        self.pos_embed = nn.Parameter(torch.zeros(1, 64, self.embed_dim))
        self.patch_embed = PatchEmbed(img_size=(224, 224), patch_size=patch_size, in_chans=input_dim, embed_dim=self.embed_dim)
        self.dpt_head = DPTHead(self.embed_dim, MODEL_CONFIGS[model_name]['features'], out_channels=MODEL_CONFIGS[model_name]['out_channels'])

    def interpolate_pos_encoding(self, x, h, w):
        previous_dtype = x.dtype
        npatch = x.shape[1]
        N = self.pos_embed.shape[1]
        if npatch == N and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        sqrt_N = math.sqrt(N)
        sx, sy = float(w0) / sqrt_N, float(h0) / sqrt_N
        pos_embed = nn.functional.interpolate(
            pos_embed.reshape(1, int(sqrt_N), int(sqrt_N), dim).permute(0, 3, 1, 2),
            scale_factor=(sy, sx),
            mode="bicubic",
            antialias=False
        )
        assert int(w0) == pos_embed.shape[-1]
        assert int(h0) == pos_embed.shape[-2]
        pos_embed = pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return pos_embed.to(previous_dtype)

    def forward(self, x):
        _, _, h, w = x.shape
        x = self.patch_embed(x)
        x = x + self.interpolate_pos_encoding(x, h, w)
        outputs = []
        for i, blk in enumerate(self.blks):
            x = blk(x)
            if i in self.idx:
                outputs.append([x])
        out = self.dpt_head(outputs, h // self.patch_size, w // self.patch_size, return_intermediate=True)[0]
        return {'out': F.interpolate(out, (h, w), mode="bilinear", align_corners=True)}
