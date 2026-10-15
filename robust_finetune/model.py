"""SPEAR stage-2 model: the frozen 2-frame parent extended to three frames, with a token router and expert adapters."""
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from model.waft_a2 import WAFTv2
from utils.utils import Padder, bilinear_sampler, coords_grid
from .adapters import GatedTemporalContextAdapter, GatedTemporalRecurrentAdapter, RobustResidualFeatureAdapter
from .corruptions import NUM_EXPERT_GROUPS
from .attention import install_attention

# Defaults identify our released parent; set both env vars when using a parent you retrained.
PARENT_SHA256 = os.environ.get("SPEAR_PARENT_SHA256", "aa6956626e5f13288a6f3047d6eb7796ba03c303875d5c55bd5537faf8bd9530")
DEFAULT_PARENT = os.environ.get("SPEAR_PARENT", "checkpoints/clean-step4-spring/waft-a2/dav2/42/20000.pth")


def sha256(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class TokenRouter(nn.Module):
    """Gate logits per expert group from the frozen DAv2 tokens of the three frames (no extra backbone)."""
    def __init__(self, embed_dim=1024, layers=4, layer_dim=96, num_groups=1):
        super().__init__()
        self.num_groups = num_groups
        self.project = nn.Sequential(nn.LayerNorm(4 * embed_dim), nn.Linear(4 * embed_dim, layer_dim), nn.GELU())
        width = 5 * layers * layer_dim
        self.classifier = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 256), nn.GELU(), nn.Linear(256, 128), nn.GELU(), nn.Linear(128, num_groups))
        nn.init.zeros_(self.classifier[-1].weight)
        nn.init.constant_(self.classifier[-1].bias, -1.945910149)  # log(1/7): every gate starts at 0.125

    @staticmethod
    def summarize(levels):
        """Per encoder level: [class token, mean, std, max] of the patch tokens."""
        return torch.stack([torch.cat((cls.float(), p.float().mean(1), p.float().std(1, unbiased=False), p.float().amax(1)), dim=1) for p, cls in levels], dim=1)

    def forward(self, previous, center, following):
        with torch.autocast(device_type=center.device.type, enabled=False):
            p, c, n = [self.project(t.float()).flatten(1) for t in (previous, center, following)]
            return self.classifier(torch.cat((p, c, n, (p-c).abs(), (n-c).abs()), dim=1))


class ThreeFrameRobustWAFT(nn.Module):
    def __init__(self, base, checkpointing=True, temporal_dim=128, weather_dim=128, num_expert_groups=NUM_EXPERT_GROUPS):
        super().__init__()
        self.base = base.requires_grad_(False).eval()
        self.num_expert_groups = num_expert_groups
        self.temporal_context = GatedTemporalContextAdapter(base.iter_dim, temporal_dim, -2.0)
        self.temporal_recurrent = GatedTemporalRecurrentAdapter(base.iter_dim, temporal_dim, -2.0)
        self.experts = nn.ModuleList(
            RobustResidualFeatureAdapter(base.iter_dim, weather_dim, blocks=3)
            for _ in range(num_expert_groups)
        )
        self.router = TokenRouter(base.encoder.embed_dim, num_groups=num_expert_groups)
        self.checkpointing = checkpointing
        self.require_native = True
        self.stage = "adapters"
        self.set_stage(self.stage)

    def set_stage(self, stage):
        """router: train only the router.  adapters: train temporal adapters + experts, router frozen."""
        if stage not in ("router", "adapters"):
            raise ValueError(stage)
        self.stage = stage
        self.base.requires_grad_(False)
        for name in ("temporal_context", "temporal_recurrent", "experts"):
            getattr(self, name).requires_grad_(stage == "adapters")
        self.router.requires_grad_(stage == "router")
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()  # frozen BN buffers/dropout must not change
        return self

    def _ckpt(self, fn, *inputs):
        if self.checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(fn, *inputs, use_reentrant=False)
        return fn(*inputs)

    @torch.no_grad()
    def _encode(self, image, tokens_only=False):
        enc = self.base.encoder
        levels = enc.encoder.get_intermediate_layers(image, enc.intermediate_layer_idx[enc.model_name], return_class_token=True)
        summary = TokenRouter.summarize(levels)
        if tokens_only:
            return None, summary
        h, w = image.shape[-2:]
        p = enc.encoder.patch_size
        pretrained = enc.dpt_head(levels, h // p, w // p)[0]
        pretrained = F.interpolate(pretrained, (h//2, w//2), mode="bilinear", align_corners=True)
        features = self.base.fmap_conv(torch.cat((pretrained, self.base.fnet(image)[0]), 1))
        return features, summary

    def _propose(self, center, target, state, flow):
        b, _, h, w = center.shape
        coords = (coords_grid(b, h, w, device=center.device) + flow).detach()
        # grid_sample coordinates/flow stay FP32 under FP16 autocast.
        warped = bilinear_sampler(target, coords.permute(0, 2, 3, 1))
        inp = self.base.warp_linear(torch.cat((center, warped, state, flow), 1))
        out = self.base.refine_net(inp)["out"]
        return self.base.refine_transform(torch.cat((out, state), 1))

    def _apply_experts(self, feature, gate):
        # Experts are applied in sequence, each scaled by its own gate column.
        for k, expert in enumerate(self.experts):
            feature = expert(feature, gate[:, k:k + 1])
        return feature

    def _iteration(self, p, c, n, sb, sf, fb, ff):
        pb = self._propose(c, p, sb, fb)
        pf = self._propose(c, n, sf, ff)
        db, _ = self.temporal_recurrent(pb, pf, fb, ff)
        df, _ = self.temporal_recurrent(pf, pb, ff, fb)
        sb, sf = pb + db, pf + df
        ub, uf = self.base.flow_head(sb), self.base.flow_head(sf)
        fb, ff = fb + ub[:, :2], ff + uf[:, :2]
        # Only target-direction full-resolution output is supervised/stored.
        flow, info = self.base.upsample_data(ff, uf[:, 2:], .25 * self.base.upsample_weight(sf))
        return sb, sf, fb, ff, flow, info

    def _baseline(self, center, following, iters):
        """The frozen 2-frame parent (center -> following), used as clean teacher and for comparison."""
        c, _ = self._encode(center)
        n, _ = self._encode(following)
        state = self.base.hidden_conv(torch.cat((c, n), 1))
        flow = torch.zeros(c.shape[0], 2, *c.shape[-2:], device=c.device)
        for _ in range(iters):
            flow = flow.detach()
            state = self._propose(c, n, state, flow)
            update = self.base.flow_head(state)
            flow = flow + update[:, :2]
        return self.base.upsample_data(flow, update[:, 2:], .25 * self.base.upsample_weight(state))[0]

    def forward(self, previous, center, following, *, baseline=False, router_only=False, final_only=False):
        if previous.shape != center.shape or following.shape != center.shape:
            raise ValueError("three equally-sized RGB frames are required")
        if center.ndim != 4 or center.shape[1] != 3:
            raise ValueError("expected BCHW RGB tensors in [0,255]")
        if self.require_native and center.shape[-2:] != (1080, 1920):
            raise ValueError("production model requires native 1080x1920; no resize/crop fallback")
        padder = Padder(center.shape, factor=self.base.factor)
        images = [padder.pad(self.base.normalize_image(x)) for x in (previous, center, following)]
        iters = self.base.args.iters
        if baseline:
            return {"flow": [padder.unpad(self._baseline(images[1], images[2], iters))]}
        pairs = [self._encode(x, tokens_only=router_only) for x in images]
        logits = self.router(*(s for _, s in pairs))
        gate = logits.sigmoid()
        if router_only:
            return {"logits": logits, "gate": gate}
        p, c, n = [self._ckpt(self._apply_experts, feature, gate) for feature, _ in pairs]
        sb = self.base.hidden_conv(torch.cat((c, p), 1))
        sf = self.base.hidden_conv(torch.cat((c, n), 1))
        db, _ = self._ckpt(self.temporal_context, c, p, n)
        df, _ = self._ckpt(self.temporal_context, c, n, p)
        sb, sf = sb + db, sf + df
        fb = torch.zeros(c.shape[0], 2, *c.shape[-2:], device=c.device)
        ff = torch.zeros_like(fb)
        flows, infos = [], []
        for itr in range(iters):
            sb, sf, fb, ff, flow, info = self._ckpt(self._iteration, p, c, n, sb, sf, fb.detach(), ff.detach())
            if not final_only or itr == iters - 1:
                flows.append(padder.unpad(flow))
                infos.append(padder.unpad(info))
        return {"flow": flows, "info": infos, "logits": logits, "gate": gate}

    def delta_state(self):
        return {k: v.detach().cpu().clone() for k, v in self.state_dict().items() if not k.startswith("base.")}

    def load_delta(self, state):
        expected = {k: v for k, v in self.state_dict().items() if not k.startswith("base.")}
        if set(state) != set(expected):
            raise ValueError("adapter state keys differ from this architecture")
        for k, tensor in state.items():
            if tensor.shape != expected[k].shape or tensor.dtype != expected[k].dtype or not torch.isfinite(tensor).all():
                raise ValueError(f"invalid adapter tensor: {k}")
        with torch.no_grad():
            for k, tensor in state.items():
                expected[k].copy_(tensor)


def build_model(parent=DEFAULT_PARENT, attention_mode="auto", checkpointing=True):
    actual = sha256(parent)
    if actual != PARENT_SHA256:
        raise ValueError(f"wrong bundled parent SHA256: {actual}")
    cfg = json.loads((Path(__file__).resolve().parents[1] / "config/clean-step4-spring.json").read_text())
    cfg["pretrained_backbones"] = False  # strictly load ALL tensors below, no downloads
    base = WAFTv2(SimpleNamespace(**cfg))
    state = torch.load(parent, map_location="cpu", weights_only=True)
    if "state_dict" in state:
        state = state["state_dict"]
    state = {k.removeprefix("module."): v for k, v in state.items()}
    base.load_state_dict(state, strict=True)
    del state
    install_attention(base, attention_mode)
    return ThreeFrameRobustWAFT(base, checkpointing)
