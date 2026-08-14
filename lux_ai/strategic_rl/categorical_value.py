"""HL-Gauss categorical value targets."""

from __future__ import annotations

import math

import torch
from torch.nn import functional as F


def support_values(num_bins: int, value_min: float, value_max: float, *, device=None, dtype=None) -> torch.Tensor:
    if num_bins < 2 or value_max <= value_min:
        raise ValueError("Categorical support requires num_bins >= 2 and value_max > value_min")
    return torch.linspace(value_min, value_max, num_bins, device=device, dtype=dtype)


def hl_gauss_encode(
    targets: torch.Tensor,
    *,
    num_bins: int = 101,
    value_min: float = -2.0,
    value_max: float = 2.0,
    sigma_ratio: float = 0.75,
) -> torch.Tensor:
    support = support_values(num_bins, value_min, value_max, device=targets.device, dtype=targets.dtype)
    width = (value_max - value_min) / (num_bins - 1)
    sigma = sigma_ratio * width
    edges = torch.cat((support[:1] - width / 2, support + width / 2))
    z = (edges.view(*([1] * targets.ndim), -1) - targets.unsqueeze(-1)) / (sigma * math.sqrt(2.0))
    cdf = 0.5 * (1.0 + torch.erf(z))
    probs = (cdf[..., 1:] - cdf[..., :-1]).clamp_min(0.0)
    return probs / probs.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(probs.dtype).tiny)


def categorical_value(logits: torch.Tensor, value_min: float = -2.0, value_max: float = 2.0) -> torch.Tensor:
    support = support_values(logits.shape[-1], value_min, value_max, device=logits.device, dtype=logits.dtype)
    return (F.softmax(logits, dim=-1) * support).sum(dim=-1)


def hl_gauss_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    value_min: float = -2.0,
    value_max: float = 2.0,
    sigma_ratio: float = 0.75,
) -> torch.Tensor:
    labels = hl_gauss_encode(
        targets.detach(), num_bins=logits.shape[-1], value_min=value_min,
        value_max=value_max, sigma_ratio=sigma_ratio,
    )
    return -(labels * F.log_softmax(logits, dim=-1)).sum(dim=-1)


def support_outside_fraction(targets: torch.Tensor, value_min: float, value_max: float) -> torch.Tensor:
    return ((targets < value_min) | (targets > value_max)).float().mean()
