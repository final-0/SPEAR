"""SPEAR adapters: gated temporal adapters (three-frame) and the zero-initialized expert adapter.

Every adapter ends in a zero-initialized projection, so at initialization the model equals the frozen parent.
"""
import torch
from torch import nn
from torch.nn import functional as F

class GatedTemporalContextAdapter(nn.Module):
    """Jointly initialize one direction from the complete frame triplet."""

    def __init__(
        self,
        hidden_dim: int,
        bottleneck_dim: int,
        gate_bias: float,
    ) -> None:
        super().__init__()
        self.in_proj = nn.Conv2d(5 * hidden_dim, bottleneck_dim, 1, bias=True)
        self.mix = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(
                bottleneck_dim,
                bottleneck_dim,
                3,
                padding=1,
                bias=True,
            ),
            nn.GELU(),
        )
        self.out_proj = nn.Conv2d(bottleneck_dim, hidden_dim, 1, bias=True)
        self.gate_proj = nn.Conv2d(bottleneck_dim, 1, 1, bias=True)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, gate_bias)

    def forward(
        self,
        center: torch.Tensor,
        target: torch.Tensor,
        opposite: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.mix(
            self.in_proj(
                torch.cat(
                    (
                        center,
                        target,
                        opposite,
                        target - center,
                        opposite - center,
                    ),
                    dim=1,
                )
            )
        )
        gate = torch.sigmoid(self.gate_proj(hidden))
        return gate * self.out_proj(hidden), gate


class GatedTemporalRecurrentAdapter(nn.Module):
    """Fuse the opposite direction only where a learned gate trusts it."""

    def __init__(
        self,
        hidden_dim: int,
        bottleneck_dim: int,
        gate_bias: float,
    ) -> None:
        super().__init__()
        # Three states plus primary/opposite/consistency flows and two magnitudes.
        input_dim = 3 * hidden_dim + 8
        self.in_proj = nn.Conv2d(input_dim, bottleneck_dim, 1, bias=True)
        self.mix = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(
                bottleneck_dim,
                bottleneck_dim,
                3,
                padding=1,
                bias=True,
            ),
            nn.GELU(),
        )
        self.out_proj = nn.Conv2d(bottleneck_dim, hidden_dim, 1, bias=True)
        self.gate_proj = nn.Conv2d(bottleneck_dim, 1, 1, bias=True)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, gate_bias)

    def forward(
        self,
        primary_state: torch.Tensor,
        opposite_state: torch.Tensor,
        primary_flow: torch.Tensor,
        opposite_flow: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        opposite_as_primary = -opposite_flow
        consistency = primary_flow - opposite_as_primary
        primary_magnitude = torch.linalg.vector_norm(
            primary_flow, dim=1, keepdim=True
        )
        opposite_magnitude = torch.linalg.vector_norm(
            opposite_flow, dim=1, keepdim=True
        )
        hidden = self.mix(
            self.in_proj(
                torch.cat(
                    (
                        primary_state,
                        opposite_state,
                        primary_state - opposite_state,
                        primary_flow,
                        opposite_as_primary,
                        consistency,
                        primary_magnitude,
                        opposite_magnitude,
                    ),
                    dim=1,
                )
            )
        )
        gate = torch.sigmoid(self.gate_proj(hidden))
        return gate * self.out_proj(hidden), gate


class RobustResidualFeatureAdapter(nn.Module):
    """One expert: features + gate * residual(features), with a zero-initialized output projection.

    The gate is the router's sigmoid output for this expert's corruption group, shape (batch, 1).
    """

    def __init__(
        self,
        channels: int,
        hidden_dim: int = 128,
        blocks: int = 3,
    ) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(1, channels)
        self.in_proj = nn.Conv2d(channels, hidden_dim, 1, bias=True)
        self.blocks = nn.ModuleList(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=True)
            for _ in range(blocks)
        )
        self.out_proj = nn.Conv2d(hidden_dim, channels, 1, bias=True)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        features: torch.Tensor,
        gate: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden = F.gelu(self.in_proj(self.norm(features)))
        for block in self.blocks:
            hidden = hidden + F.gelu(block(hidden))
        delta = self.out_proj(hidden)
        if gate is not None:
            if gate.ndim == 1:
                gate = gate[:, None]
            if gate.ndim != 2 or gate.shape != (features.shape[0], 1):
                raise ValueError("adapter gate must have shape (batch, 1)")
            delta = delta * gate[:, :, None, None]
        return features + delta
