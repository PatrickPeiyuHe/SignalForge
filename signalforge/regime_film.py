"""Regime-conditioned FiLM block used in the public architecture sketch."""

from __future__ import annotations

import torch
from torch import nn


class RegimeFiLM(nn.Module):
    """Apply bounded feature-wise affine modulation from a regime vector.

    The module maps a market-regime embedding to gamma/beta parameters and
    applies `x * gamma + beta`. Zero initialization keeps the layer close to
    identity at startup, which is useful when adding regime conditioning to an
    otherwise stable trunk.
    """

    def __init__(
        self,
        feature_dim: int,
        regime_dim: int,
        hidden_dim: int = 64,
        bound: float = 0.20,
        zero_init: bool = True,
    ) -> None:
        super().__init__()
        if feature_dim <= 0 or regime_dim <= 0:
            raise ValueError("feature_dim and regime_dim must be positive")
        self.feature_dim = int(feature_dim)
        self.bound = float(bound)
        self.net = nn.Sequential(
            nn.LayerNorm(regime_dim),
            nn.Linear(regime_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, feature_dim * 2),
        )
        if zero_init:
            last = self.net[-1]
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def forward(self, x: torch.Tensor, regime: torch.Tensor) -> torch.Tensor:
        """Modulate `x`.

        Shapes:
            x: `[batch, feature_dim]` or `[batch, time, feature_dim]`
            regime: `[batch, regime_dim]`
        """
        if x.shape[-1] != self.feature_dim:
            raise ValueError(f"expected last dim {self.feature_dim}, got {x.shape[-1]}")
        params = self.net(regime)
        gamma_raw, beta_raw = params.chunk(2, dim=-1)
        gamma = 1.0 + self.bound * torch.tanh(gamma_raw)
        beta = self.bound * torch.tanh(beta_raw)
        if x.ndim == 3:
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)
        return x * gamma + beta
