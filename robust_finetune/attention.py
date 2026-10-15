"""Bound attention memory without dropping tokens or resizing images."""
from types import MethodType

import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.checkpoint import checkpoint


def attention(q, k, v, mode="auto", chunk=256):
    if mode == "auto" and q.is_cuda:
        try:
            with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
                return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        except RuntimeError as exc:
            # Never turn an actual OOM into a silent retry or a resolution change.
            if not any(s in str(exc).lower() for s in ("no available kernel", "no viable backend")):
                raise
    def part(query, key, value):
        with sdpa_kernel(SDPBackend.MATH):
            return F.scaled_dot_product_attention(query, key, value, dropout_p=0.0)
    outputs = []
    for query in q.split(chunk, dim=-2):
        if torch.is_grad_enabled() and any(t.requires_grad for t in (query, k, v)):
            outputs.append(checkpoint(part, query, k, v, use_reentrant=False))
        else:
            outputs.append(part(query, k, v))
    return torch.cat(outputs, dim=-2)


def install_attention(base, mode="auto", chunk=256):
    if mode not in ("auto", "chunked") or chunk < 1:
        raise ValueError("invalid attention settings")
    def forward(module, x, attn_bias=None, attn_mask=None, is_causal=False):
        if attn_bias is not None or attn_mask is not None or is_causal:
            raise ValueError("this full-image attention does not accept masks")
        b, n, c = x.shape
        q, k, v = module.qkv(x).reshape(b, n, 3, module.num_heads, c // module.num_heads).permute(2, 0, 3, 1, 4).unbind(0)
        q = module.q_norm(q) if hasattr(module, "q_norm") else q
        k = module.k_norm(k) if hasattr(module, "k_norm") else k
        y = attention(q, k, v, mode, chunk).transpose(1, 2).reshape(b, n, c)
        y = module.norm(y) if hasattr(module, "norm") else y
        return module.proj_drop(module.proj(y))
    for block in list(base.encoder.encoder.blocks) + list(base.refine_net.blks):
        module = block.attn
        if getattr(module, "gate", None) is not None:
            raise ValueError("gated attention is not part of the bundled architecture")
        if getattr(module, "attn_drop").p != 0:
            raise ValueError("only dropout-free frozen attention is supported")
        module.forward = MethodType(forward, module)
