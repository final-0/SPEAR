"""DPT-style head that turns DAv2 encoder tokens into the 1/2-resolution feature map used by WAFT."""
import torch.nn as nn


class ResidualConvUnit(nn.Module):
    def __init__(self, features, activation):
        super().__init__()
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.activation = activation

    def forward(self, x):
        out = self.conv1(self.activation(x))
        out = self.conv2(self.activation(out))
        return out + x


class FeatureFusionBlock(nn.Module):
    def __init__(self, features, activation):
        super().__init__()
        self.out_conv = nn.Conv2d(features, features, kernel_size=1, stride=1, padding=0, bias=True)
        self.resConfUnit1 = ResidualConvUnit(features, activation)
        self.resConfUnit2 = ResidualConvUnit(features, activation)

    def forward(self, *xs, size):
        output = xs[0]
        if len(xs) == 2:
            output = output + self.resConfUnit1(xs[1])
        output = self.resConfUnit2(output)
        output = nn.functional.interpolate(output, size=size, mode="bilinear", align_corners=True)
        return self.out_conv(output)


class DPTHead(nn.Module):
    def __init__(self, in_channels, features=256, out_channels=(256, 512, 1024, 1024), lvl=-2):
        super().__init__()
        self.projects = nn.ModuleList([
            nn.Conv2d(in_channels, out_channel, kernel_size=1, stride=1, padding=0)
            for out_channel in out_channels
        ])
        # Level i is resampled by 2**(i + lvl): transposed conv upsamples, strided conv downsamples.
        resize_layers = []
        for i, channels in enumerate(out_channels):
            if i + lvl < 0:
                k = 2 ** (-i - lvl)
                resize_layers.append(nn.ConvTranspose2d(channels, channels, kernel_size=k, stride=k, padding=0))
            else:
                k = 2 ** (i + lvl)
                resize_layers.append(nn.Conv2d(channels, channels, kernel_size=k, stride=k, padding=0))
        self.resize_layers = nn.ModuleList(resize_layers)
        self.scratch = nn.ModuleList([
            nn.Conv2d(channels, features, kernel_size=3, stride=1, padding=1, bias=False)
            for channels in out_channels
        ])
        self.refine = nn.ModuleList([FeatureFusionBlock(features, nn.ReLU(False)) for _ in out_channels])

    def forward(self, out_features, patch_h, patch_w):
        out = []
        for i, x in enumerate(out_features):
            x = x[0]
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            out.append(self.resize_layers[i](self.projects[i](x)))

        out_rn = [self.scratch[i](out[i]) for i in range(len(out))]
        out_rn[-1] = self.refine[-1](out_rn[-1], size=out_rn[-1].shape[2:])
        for i in range(2, len(out_rn) + 1):
            up_feat = nn.functional.interpolate(out_rn[-i + 1], scale_factor=2, mode='bilinear', align_corners=True)
            out_rn[-i] = self.refine[-i](out_rn[-i], up_feat, size=out_rn[-i].shape[2:])
        return out_rn
