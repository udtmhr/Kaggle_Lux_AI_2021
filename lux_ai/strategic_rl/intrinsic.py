from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import gym
import torch
import torch.nn.functional as F
from torch import nn

from ..lux_gym.act_spaces import ACTION_MEANINGS
from ..nns.in_blocks import ConvEmbeddingInputLayer


class ControllableEpisodicCuriosity(nn.Module):
    """Small, policy-independent encoder for controllable episodic novelty."""

    def __init__(self, obs_space: gym.spaces.Dict, embedding_dim: int = 32):
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.input_layer = ConvEmbeddingInputLayer(
            obs_space,
            embedding_dim=max(8, self.embedding_dim),
            out_dim=self.embedding_dim,
            n_merge_layers=1,
            sum_player_embeddings=False,
            use_index_select=False,
            activation=nn.SiLU,
        )
        self.encoder = nn.ModuleList(nn.Conv2d(self.embedding_dim, self.embedding_dim, 3, padding=1) for _ in range(3))
        self.inverse_heads = nn.ModuleDict(
            {entity: nn.Conv2d(2 * self.embedding_dim, len(actions), 1) for entity, actions in ACTION_MEANINGS.items()}
        )
        self.intrinsic_value = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.SiLU(),
            nn.Linear(self.embedding_dim, 1),
        )

    def encode(self, model_input: Mapping) -> tuple[torch.Tensor, torch.Tensor]:
        x, mask = self.input_layer((model_input["obs"], model_input["info"]["input_mask"]))
        original = x
        for layer in self.encoder:
            x = F.silu(layer(x)) * mask
        rotated = torch.rot90(original, 2, dims=(-2, -1))
        rotated_mask = torch.rot90(mask, 2, dims=(-2, -1))
        for layer in self.encoder:
            rotated = F.silu(layer(rotated)) * rotated_mask
        x = 0.5 * (x + torch.rot90(rotated, 2, dims=(-2, -1))) * mask
        denominator = mask.flatten(2).sum(dim=-1).clamp_min(1.0)
        pooled = (x * mask).flatten(2).sum(dim=-1) / denominator
        pooled = F.normalize(pooled.float(), dim=-1, eps=1e-6)
        batch = pooled.shape[0] // 2
        spatial = x.view(batch, 2, self.embedding_dim, *x.shape[-2:])
        return pooled.view(batch, 2, self.embedding_dim), spatial

    def forward(self, model_input: Mapping) -> dict[str, torch.Tensor]:
        embedding, spatial = self.encode(model_input)
        return {
            "embedding": embedding,
            "spatial": spatial,
            "intrinsic_value": self.intrinsic_value(embedding).squeeze(-1),
        }

    def inverse_logits(self, current: torch.Tensor, following: torch.Tensor) -> dict[str, torch.Tensor]:
        batch, players, channels, height, width = current.shape
        features = torch.cat((current, following), dim=2).reshape(batch * players, 2 * channels, height, width)
        outputs = {}
        for entity, head in self.inverse_heads.items():
            logits = head(features).view(batch, players, -1, height, width)
            outputs[entity] = logits.permute(0, 1, 3, 4, 2).unsqueeze(1).contiguous()
        return outputs

    def inverse_dynamics_loss(
        self,
        current: torch.Tensor,
        following: torch.Tensor,
        actions: Mapping[str, torch.Tensor],
        actions_taken: Mapping[str, torch.Tensor],
        player_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        logits_by_entity = self.inverse_logits(current, following)
        total_loss = current.new_zeros(())
        total_count = current.new_zeros(())
        stats: dict[str, float] = {}
        for entity, logits in logits_by_entity.items():
            targets = actions[entity]
            taken = actions_taken[entity].bool()
            active = taken.any(dim=-1) & player_mask[:, None, :, None, None]
            expanded_logits = logits.unsqueeze(-2).expand(*logits.shape[:-1], targets.shape[-1], logits.shape[-1])
            slot_count = targets.shape[-1]
            repeated = targets.unsqueeze(-1) == targets.unsqueeze(-2)
            previous_slots = torch.tril(
                torch.ones((slot_count, slot_count), dtype=torch.bool, device=targets.device), diagonal=-1
            )
            duplicate = (repeated & previous_slots).any(dim=-1)
            valid_target = torch.gather(taken, -1, targets).bool() & active.unsqueeze(-1) & ~duplicate
            losses = F.cross_entropy(
                expanded_logits.reshape(-1, expanded_logits.shape[-1]),
                targets.reshape(-1),
                reduction="none",
            ).view_as(targets)
            count = valid_target.sum()
            total_loss = total_loss + torch.where(valid_target, losses, torch.zeros_like(losses)).sum()
            total_count = total_count + count

            correct = (expanded_logits.argmax(dim=-1) == targets) & valid_target
            accuracy = correct.sum().float() / count.clamp_min(1)
            target_counts = torch.bincount(targets[valid_target], minlength=logits.shape[-1])
            majority = target_counts.max().float() / count.clamp_min(1)
            stats[f"{entity}_accuracy"] = float(accuracy.detach().cpu().item())
            stats[f"{entity}_majority"] = float(majority.detach().cpu().item())
            stats[f"{entity}_count"] = int(count.detach().cpu().item())
        return total_loss / total_count.clamp_min(1), stats


class EllipticalEpisodicMemory:
    """Per-environment/player inverse covariance used by the E3B bonus."""

    def __init__(self, n_envs: int, embedding_dim: int, device: torch.device, ridge: float = 1.0):
        eye = torch.eye(embedding_dim, dtype=torch.float32, device=device) / float(ridge)
        self.inverse_covariance = eye.view(1, 1, embedding_dim, embedding_dim).repeat(n_envs, 2, 1, 1)

    @torch.no_grad()
    def reset(self, env_mask: torch.Tensor | None = None) -> None:
        dim = self.inverse_covariance.shape[-1]
        eye = torch.eye(dim, dtype=self.inverse_covariance.dtype, device=self.inverse_covariance.device)
        if env_mask is None:
            self.inverse_covariance.copy_(eye)
        else:
            self.inverse_covariance[env_mask.bool()] = eye

    @torch.no_grad()
    def bonus(self, embedding: torch.Tensor, done: torch.Tensor, clip: float = 5.0) -> torch.Tensor:
        z = F.normalize(embedding.float(), dim=-1, eps=1e-6).unsqueeze(-1)
        projected = self.inverse_covariance @ z
        quadratic = (z.transpose(-2, -1) @ projected).squeeze(-1).squeeze(-1).clamp_min(0.0)
        bonus = quadratic.sqrt().clamp(max=float(clip))
        denominator = (1.0 + quadratic).unsqueeze(-1).unsqueeze(-1)
        self.inverse_covariance.sub_((projected @ projected.transpose(-2, -1)) / denominator)
        bonus = torch.where(done.unsqueeze(-1), torch.zeros_like(bonus), bonus)
        if bool(done.any()):
            self.reset(done)
        return bonus

    @property
    def condition_proxy(self) -> torch.Tensor:
        diagonal = self.inverse_covariance.diagonal(dim1=-2, dim2=-1).abs()
        return diagonal.max(dim=-1).values / diagonal.min(dim=-1).values.clamp_min(1e-8)


@dataclass
class RunningMoments:
    count: float = 0.0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, values: torch.Tensor) -> None:
        finite = values.detach().float()[torch.isfinite(values)]
        if finite.numel() == 0:
            return
        batch_count = float(finite.numel())
        batch_mean = float(finite.mean().cpu().item())
        batch_m2 = float(((finite - batch_mean) ** 2).sum().cpu().item())
        if self.count == 0:
            self.count, self.mean, self.m2 = batch_count, batch_mean, batch_m2
            return
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean += delta * batch_count / total
        self.m2 += batch_m2 + delta * delta * self.count * batch_count / total
        self.count = total

    @property
    def std(self) -> float:
        return math.sqrt(max(self.m2 / max(self.count, 1.0), 1e-8))

    def normalize(self, values: torch.Tensor, clip: float) -> torch.Tensor:
        self.update(values)
        return (values / self.std).clamp(0.0, float(clip))

    def state_dict(self) -> dict[str, float]:
        return {"count": self.count, "mean": self.mean, "m2": self.m2}

    @classmethod
    def from_state_dict(cls, state: Mapping[str, float] | None) -> RunningMoments:
        return cls(**dict(state)) if state else cls()


def intrinsic_beta(flags, step: int) -> float:
    pretrain = int(getattr(flags, "intrinsic_pretrain_steps", 16000))
    ramp_end = int(getattr(flags, "intrinsic_ramp_end_step", 25000))
    decay_start = int(getattr(flags, "intrinsic_decay_start_step", 40000))
    decay_end = int(getattr(flags, "intrinsic_decay_end_step", 50000))
    beta_max = float(getattr(flags, "intrinsic_beta_max", 0.10))
    if step < pretrain or step >= decay_end:
        return 0.0
    if step < ramp_end:
        return beta_max * (step - pretrain) / max(ramp_end - pretrain, 1)
    if step < decay_start:
        return beta_max
    return beta_max * (decay_end - step) / max(decay_end - decay_start, 1)


def controllable_change_masks(observations: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Classify transitions by whether own or opponent controllable state changed more."""
    keys = (
        "worker",
        "cart",
        "city_tile",
        "worker_cargo_wood",
        "worker_cargo_coal",
        "worker_cargo_uranium",
        "cart_cargo_wood",
        "cart_cargo_coal",
        "cart_cargo_uranium",
        "research_points",
        "researched_coal",
        "researched_uranium",
    )
    change = None
    for key in keys:
        if key not in observations:
            continue
        value = observations[key].float()
        if value.shape[3] != 2:
            continue
        delta = (value[1:] - value[:-1]).abs()
        reduce_dims = tuple(index for index in range(2, delta.ndim) if index != 3)
        per_player = delta.sum(dim=reduce_dims)
        change = per_player if change is None else change + per_player
    if change is None:
        example = next(iter(observations.values()))
        shape = (example.shape[0] - 1, example.shape[1], 2)
        return torch.zeros(shape, dtype=torch.bool, device=example.device), torch.zeros(
            shape, dtype=torch.bool, device=example.device
        )
    opponent_change = change.flip(dims=(-1,))
    own_dominant = (change > 0) & (change > 1.5 * opponent_change)
    enemy_dominant = (opponent_change > 0) & (opponent_change > 1.5 * change)
    return own_dominant, enemy_dominant
