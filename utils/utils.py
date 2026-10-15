import torch
import torch.nn.functional as F


def load_ckpt(model, path, strict=False):
    """Load a state dict. Non-strict mode skips missing keys and shape mismatches (warm start)."""
    ckpt = torch.load(path, map_location="cpu")
    if strict:
        model.load_state_dict(ckpt, strict=True)
        print(f"  Loaded all {len(ckpt)} keys from {path}", flush=True)
        return
    model_dict = model.state_dict()
    loaded, skipped = [], []
    for k, v in ckpt.items():
        if k in model_dict and v.shape == model_dict[k].shape:
            model_dict[k] = v
            loaded.append(k)
        else:
            skipped.append(k)
    model.load_state_dict(model_dict, strict=False)
    print(f"  Loaded {len(loaded)}/{len(ckpt)} keys, skipped {len(skipped)}", flush=True)
    if skipped:
        print(f"  Skipped: {skipped[:10]}{'...' if len(skipped) > 10 else ''}", flush=True)


class Padder:
    """Zero-pads images so that height and width are divisible by `factor`."""
    def __init__(self, dims, factor):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht + 8) // factor) + 1) * factor - self.ht
        pad_wd = (((self.wd + 8) // factor) + 1) * factor - self.wd
        self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, pad_ht // 2, pad_ht - pad_ht // 2]

    def pad(self, x):
        return F.pad(x, self._pad, mode='constant', value=0)

    def unpad(self, x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]


def bilinear_sampler(img, coords):
    """grid_sample with pixel coordinates (B, H, W, 2)."""
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1, 1], dim=-1)
    xgrid = 2 * xgrid / (W - 1) - 1
    ygrid = 2 * ygrid / (H - 1) - 1
    grid = torch.cat([xgrid, ygrid], dim=-1)
    return F.grid_sample(img, grid, align_corners=True)


def coords_grid(batch, ht, wd, device):
    coords = torch.meshgrid(torch.arange(ht, device=device), torch.arange(wd, device=device), indexing='ij')
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)
