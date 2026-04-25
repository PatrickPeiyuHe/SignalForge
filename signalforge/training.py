"""Sanitized multi-head training loop for public examples.

This module shows the engineering shape of the training process without
including production features, private hyperparameters, selectors, checkpoints,
or vendor-specific paths.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class LossCoefficients:
    """External coefficients for composing public multi-head losses."""

    q_return: float = 1.0
    rank: float = 1.0
    buy_block: float = 1.0
    downlock: float = 1.0
    fragility: float = 1.0
    intraday_aux: float = 1.0


@dataclass
class TrainingBatch:
    """Tensor contract expected by the public training loop."""

    features: Mapping[str, torch.Tensor]
    targets: Mapping[str, torch.Tensor]
    meta: Mapping[str, object] = field(default_factory=dict)

    def to(self, device: torch.device | str) -> "TrainingBatch":
        return TrainingBatch(
            features={k: v.to(device) for k, v in self.features.items()},
            targets={k: v.to(device) for k, v in self.targets.items()},
            meta=self.meta,
        )


def quantile_loss(pred: torch.Tensor, target: torch.Tensor, quantile: float) -> torch.Tensor:
    """Mean pinball loss used for the public upper-tail head."""
    error = target - pred
    return torch.maximum(quantile * error, (quantile - 1.0) * error).mean()


def pairwise_rank_loss(
    scores: torch.Tensor,
    target_rank: torch.Tensor,
    *,
    max_pairs: int | None = None,
    margin: float = 0.0,
) -> torch.Tensor:
    """Simple within-batch pairwise rank loss.

    Higher `target_rank` is treated as better. The implementation is compact
    for readability and intended for public examples, not large-scale training.
    """
    valid = torch.isfinite(scores) & torch.isfinite(target_rank)
    scores = scores[valid]
    target_rank = target_rank[valid]
    if scores.numel() < 2:
        return scores.new_tensor(0.0)

    diff_target = target_rank[:, None] - target_rank[None, :]
    pos_i, pos_j = torch.where(diff_target > 0)
    if pos_i.numel() == 0:
        return scores.new_tensor(0.0)
    if max_pairs is not None and pos_i.numel() > max_pairs:
        idx = torch.randperm(pos_i.numel(), device=scores.device)[:max_pairs]
        pos_i = pos_i[idx]
        pos_j = pos_j[idx]
    return F.softplus(margin - (scores[pos_i] - scores[pos_j])).mean()


def compute_multi_head_loss(
    outputs: Mapping[str, torch.Tensor],
    batch: TrainingBatch,
    coefficients: LossCoefficients = LossCoefficients(),
    *,
    tail_quantile: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compose the public multi-head objective.

    Required output keys:
        `q_return`, `rank`, `buy_block`, `downlock`, `fragility`

    Optional output key:
        `intraday_aux`
    """
    targets = batch.targets
    losses: dict[str, torch.Tensor] = {}
    losses["q_return"] = quantile_loss(outputs["q_return"], targets["return"], tail_quantile)
    losses["rank"] = pairwise_rank_loss(outputs["rank"], targets["rank"])
    losses["buy_block"] = F.binary_cross_entropy_with_logits(outputs["buy_block"], targets["buy_block"])
    losses["downlock"] = F.binary_cross_entropy_with_logits(outputs["downlock"], targets["downlock"])
    losses["fragility"] = F.binary_cross_entropy_with_logits(outputs["fragility"], targets["fragility"])

    if "intraday_aux" in outputs and "intraday_rank" in targets:
        losses["intraday_aux"] = pairwise_rank_loss(outputs["intraday_aux"], targets["intraday_rank"])
    else:
        losses["intraday_aux"] = outputs["rank"].new_tensor(0.0)

    total = (
        coefficients.q_return * losses["q_return"]
        + coefficients.rank * losses["rank"]
        + coefficients.buy_block * losses["buy_block"]
        + coefficients.downlock * losses["downlock"]
        + coefficients.fragility * losses["fragility"]
        + coefficients.intraday_aux * losses["intraday_aux"]
    )
    metrics = {f"loss_{name}": float(value.detach().cpu()) for name, value in losses.items()}
    metrics["loss_total"] = float(total.detach().cpu())
    return total, metrics


def train_one_epoch(
    model: nn.Module,
    batches,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device | str,
    tail_quantile: float,
    coefficients: LossCoefficients = LossCoefficients(),
    grad_clip_norm: float | None = None,
) -> dict[str, float]:
    """Run one public training epoch over an iterable of `TrainingBatch`."""
    model.train()
    totals: dict[str, float] = {}
    count = 0
    for batch in batches:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(**batch.features)
        loss, metrics = compute_multi_head_loss(outputs, batch, coefficients, tail_quantile=tail_quantile)
        loss.backward()
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()

        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + value
        count += 1
    return {key: value / max(count, 1) for key, value in totals.items()}


@torch.no_grad()
def evaluate_one_epoch(
    model: nn.Module,
    batches,
    *,
    device: torch.device | str,
    tail_quantile: float,
    coefficients: LossCoefficients = LossCoefficients(),
) -> dict[str, float]:
    """Evaluate the public objective without updating model state."""
    model.eval()
    totals: dict[str, float] = {}
    count = 0
    for batch in batches:
        batch = batch.to(device)
        outputs = model(**batch.features)
        _, metrics = compute_multi_head_loss(outputs, batch, coefficients, tail_quantile=tail_quantile)
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + value
        count += 1
    return {key: value / max(count, 1) for key, value in totals.items()}


def select_epoch(history: list[Mapping[str, float]], *, metric: str, higher_is_better: bool) -> Mapping[str, float]:
    """Select one epoch from a logged metric history."""
    if not history:
        raise ValueError("history is empty")
    key = (lambda row: float(row[metric]))
    return max(history, key=key) if higher_is_better else min(history, key=key)
