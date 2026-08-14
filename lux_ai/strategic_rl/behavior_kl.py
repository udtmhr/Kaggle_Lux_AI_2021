"""Behavior-policy KL diagnostics and adaptive regularization for IMPALA."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np
import torch
from torch.nn import functional as F


def masked_policy_kl(
    learner_logits: torch.Tensor,
    behavior_logits: torch.Tensor,
    active_mask: torch.Tensor,
    *,
    reverse: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return KL summed by trajectory/player and the number of active entities.

    Logits use ``[..., action]`` and ``active_mask`` omits that final action
    dimension. Invalid actions may be represented by ``-inf``.
    """
    learner_logp = F.log_softmax(learner_logits, dim=-1)
    behavior_logp = F.log_softmax(behavior_logits, dim=-1)
    source_logp, other_logp = (
        (learner_logp, behavior_logp) if reverse else (behavior_logp, learner_logp)
    )
    source_p = source_logp.exp()
    # ``where`` alone still lets 0 * inf poison gradients in its unselected
    # branch. Both policies share the legal-action mask, so replacing masked
    # log-probabilities by a finite floor is exact on every non-zero term.
    safe_source_logp = torch.nan_to_num(source_logp, neginf=-30.0, posinf=30.0)
    safe_other_logp = torch.nan_to_num(other_logp, neginf=-30.0, posinf=30.0)
    terms = source_p * (safe_source_logp - safe_other_logp)
    entity_kl = torch.where(active_mask, terms.sum(dim=-1), torch.zeros_like(active_mask, dtype=terms.dtype))
    # [time,batch,plane,player,x,y] -> [time,batch,player]
    trajectory_kl = entity_kl.sum(dim=(2, 4, 5))
    counts = active_mask.sum(dim=(2, 4, 5)).to(trajectory_kl.dtype)
    return trajectory_kl, counts


def masked_normalized_entropy(logits: torch.Tensor, active_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    logp = F.log_softmax(logits, dim=-1)
    probs = logp.exp()
    entropy = -torch.where(probs > 0, probs * logp, torch.zeros_like(probs)).sum(dim=-1)
    valid_actions = torch.isfinite(logits).sum(dim=-1)
    normalizer = valid_actions.to(entropy.dtype).clamp_min(2).log()
    normalized = torch.where(active_mask, entropy / normalizer, torch.zeros_like(entropy))
    return normalized.sum(), active_mask.sum().to(entropy.dtype)


@dataclass
class BehaviorKLController:
    """Adaptive dual controller with an observation-only calibration period."""

    auto_steps: int = 20_000
    dual_lr: float = 0.05
    abort_multiplier: float = 4.0
    target: Optional[float] = None
    beta: float = 1e-4
    ema_decay: float = 0.95
    entropy_abort_fraction: float = 0.70
    min_beta: float = 1e-4
    max_beta: float = 1.0
    calibration_kls: list[float] = field(default_factory=list)
    calibration_entropies: list[float] = field(default_factory=list)
    ema_kl: Optional[float] = None
    high_kl_updates: int = 0
    low_entropy_updates: int = 0
    calibrated: bool = False

    @classmethod
    def from_flags(cls, flags, state: Optional[Mapping] = None) -> Optional["BehaviorKLController"]:
        raw_target = getattr(flags, "behavior_kl_target", None)
        if raw_target is None:
            return None
        target = None if str(raw_target).lower() == "auto" else float(raw_target)
        controller = cls(
            auto_steps=int(getattr(flags, "behavior_kl_auto_steps", 20_000)),
            dual_lr=float(getattr(flags, "behavior_kl_dual_lr", 0.05)),
            abort_multiplier=float(getattr(flags, "behavior_kl_abort_multiplier", 4.0)),
            target=target,
            calibrated=target is not None,
        )
        if state:
            controller.load_state_dict(state)
        return controller

    def observe(self, step: int, kl: float, entropy: float) -> None:
        if not math.isfinite(kl) or not math.isfinite(entropy):
            raise RuntimeError(f"Non-finite behavior KL diagnostics: kl={kl}, entropy={entropy}")
        if self.ema_kl is None:
            self.ema_kl = kl
        else:
            self.ema_kl = self.ema_decay * self.ema_kl + (1.0 - self.ema_decay) * kl

        if not self.calibrated:
            self.calibration_kls.append(kl)
            self.calibration_entropies.append(entropy)
            if step < self.auto_steps:
                return
            self.target = float(np.clip(np.percentile(self.calibration_kls, 75), 0.005, 0.05))
            self.calibrated = True

        assert self.target is not None
        ratio = self.ema_kl / max(self.target, 1e-12)
        self.beta = float(np.clip(self.beta * math.exp(self.dual_lr * (ratio - 1.0)), self.min_beta, self.max_beta))
        self.high_kl_updates = self.high_kl_updates + 1 if ratio > self.abort_multiplier else 0
        entropy_median = float(np.median(self.calibration_entropies)) if self.calibration_entropies else entropy
        self.low_entropy_updates = (
            self.low_entropy_updates + 1
            if entropy < self.entropy_abort_fraction * entropy_median
            else 0
        )
        if self.high_kl_updates >= 5:
            raise RuntimeError(
                f"Behavior KL abort: EMA {self.ema_kl:.6g} exceeded {self.abort_multiplier}x target "
                f"{self.target:.6g} for 5 updates"
            )
        if self.low_entropy_updates >= 100:
            raise RuntimeError("Behavior KL abort: normalized entropy stayed below 70% of calibration median")

    def state_dict(self) -> dict:
        return asdict(self)

    def load_state_dict(self, state: Mapping) -> None:
        for name in self.__dataclass_fields__:
            if name in state:
                setattr(self, name, state[name])
