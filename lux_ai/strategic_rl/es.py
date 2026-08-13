from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

EVOLVED_PREFIXES = ("base_model.", "actor_base.", "actor.")


def evolved_parameter_names(model: nn.Module) -> list[str]:
    """Return policy parameters that can affect deployed actions."""
    return [name for name, _ in model.named_parameters() if name.startswith(EVOLVED_PREFIXES)]


def _average_ranks(values: Sequence[float]) -> torch.Tensor:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) < 2:
        raise ValueError("centered ranks require at least two scalar fitness values")
    if not np.isfinite(array).all():
        raise ValueError("fitness values must be finite")
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=np.float64)
    start = 0
    while start < len(array):
        stop = start + 1
        while stop < len(array) and array[order[stop]] == array[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2.0
        start = stop
    return torch.from_numpy(ranks)


def centered_rank_utilities(values: Sequence[float]) -> torch.Tensor:
    ranks = _average_ranks(values)
    return ranks / (len(values) - 1) - 0.5


def policy_fitness(records: Sequence[Mapping], tie_break_weight: float = 0.01) -> dict[str, float]:
    if not records:
        raise ValueError("fitness requires at least one match record")
    if not 0.0 <= tie_break_weight < 0.5 / len(records):
        raise ValueError("tie_break_weight must be smaller than one game-outcome increment")
    scores = []
    survival = []
    stranded = []
    extinctions = []
    for record in records:
        winner = int(record["winner"])
        player = int(record["candidate_player"])
        scores.append(0.5 if winner < 0 else float(winner == player))
        survival.append(float(record.get("candidate_city_survival", 1.0)))
        stranded.append(float(record.get("candidate_stranded_fuel", 0.0)))
        extinctions.append(float(int(record.get("candidate_final_city_tiles", 0)) == 0))
    score_rate = float(np.mean(scores))
    city_survival = float(np.mean(survival))
    stranded_fuel = float(np.mean(stranded))
    city_extinction_rate = float(np.mean(extinctions))
    tie_break = tie_break_weight * float(np.clip(0.5 * city_survival + 0.5 * (1.0 - stranded_fuel), 0, 1))
    return {
        "fitness": score_rate + tie_break,
        "score_rate": score_rate,
        "tie_break": tie_break,
        "candidate_city_survival": city_survival,
        "candidate_stranded_fuel": stranded_fuel,
        "candidate_city_extinction_rate": city_extinction_rate,
        "games": len(records),
    }


class ParameterSpace:
    """Flat, layer-scaled coordinates for action-affecting model parameters."""

    def __init__(self, model: nn.Module, scale_floor: float = 1e-3):
        if scale_floor <= 0:
            raise ValueError("scale_floor must be positive")
        selected_names = set(evolved_parameter_names(model))
        selected = [(name, parameter) for name, parameter in model.named_parameters() if name in selected_names]
        if not selected:
            raise ValueError("model has no action-affecting parameters")
        self.names = [name for name, _ in selected]
        self.shapes = [tuple(parameter.shape) for _, parameter in selected]
        self.numels = [parameter.numel() for _, parameter in selected]
        self.slices = []
        offset = 0
        scales = []
        for (_, parameter), numel in zip(selected, self.numels):
            self.slices.append(slice(offset, offset + numel))
            offset += numel
            rms = float(parameter.detach().float().square().mean().sqrt())
            scales.append(max(rms, scale_floor))
        self.dimension = offset
        self.layer_scales = torch.tensor(scales, dtype=torch.float32)
        self.scales = torch.cat(
            [torch.full((numel,), scale, dtype=torch.float32) for numel, scale in zip(self.numels, scales)]
        )

    def flatten_model(self, model: nn.Module) -> torch.Tensor:
        parameters = dict(model.named_parameters())
        return torch.cat([parameters[name].detach().cpu().float().reshape(-1) for name in self.names])

    def flatten_state_dict(self, state_dict: Mapping[str, torch.Tensor]) -> torch.Tensor:
        missing = [name for name in self.names if name not in state_dict]
        if missing:
            raise ValueError(f"checkpoint is missing evolved parameters: {missing[:3]}")
        return torch.cat([state_dict[name].detach().cpu().float().reshape(-1) for name in self.names])

    @torch.no_grad()
    def assign(self, model: nn.Module, vector: torch.Tensor) -> None:
        if vector.numel() != self.dimension:
            raise ValueError(f"parameter vector has {vector.numel()} values; expected {self.dimension}")
        parameters = dict(model.named_parameters())
        for name, shape, section in zip(self.names, self.shapes, self.slices):
            target = parameters[name]
            target.copy_(vector[section].view(shape).to(device=target.device, dtype=target.dtype))

    def perturb(self, center: torch.Tensor, direction: torch.Tensor, sigma: float, sign: int) -> torch.Tensor:
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        if sign not in (-1, 1):
            raise ValueError("sign must be -1 or 1")
        if center.numel() != self.dimension or direction.numel() != self.dimension:
            raise ValueError("center and direction must match the parameter-space dimension")
        return center + sign * sigma * self.scales * direction

    def normalized_delta(self, state_dict: Mapping[str, torch.Tensor], center: torch.Tensor) -> torch.Tensor:
        return (self.flatten_state_dict(state_dict) - center) / self.scales


def normalize_direction(direction: torch.Tensor) -> torch.Tensor:
    direction = direction.detach().cpu().float().reshape(-1)
    norm = torch.linalg.vector_norm(direction)
    if not torch.isfinite(norm) or norm <= 1e-12:
        raise ValueError("cannot normalize a zero or non-finite direction")
    return direction * (math.sqrt(direction.numel()) / norm)


def orthonormalize(vectors: Sequence[torch.Tensor], max_rank: int = 8) -> torch.Tensor:
    if max_rank <= 0:
        raise ValueError("max_rank must be positive")
    basis = []
    for vector in vectors:
        candidate = vector.detach().cpu().float().reshape(-1).clone()
        for existing in basis:
            candidate -= torch.dot(candidate, existing) * existing
        norm = torch.linalg.vector_norm(candidate)
        if torch.isfinite(norm) and norm > 1e-8:
            basis.append(candidate / norm)
        if len(basis) == max_rank:
            break
    if not basis:
        dimension = vectors[0].numel() if vectors else 0
        return torch.empty((0, dimension), dtype=torch.float32)
    return torch.stack(basis)


def sample_direction(
    dimension: int,
    seed: int,
    basis: torch.Tensor | None = None,
    active_probability: float = 0.0,
) -> tuple[torch.Tensor, str]:
    if dimension <= 0:
        raise ValueError("dimension must be positive")
    if not 0.0 <= active_probability <= 1.0:
        raise ValueError("active_probability must be in [0, 1]")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    has_basis = basis is not None and basis.numel() > 0 and basis.shape[1] == dimension
    choose_active = has_basis and bool(torch.rand((), generator=generator) < active_probability)
    if choose_active:
        coefficients = torch.randn(basis.shape[0], generator=generator)
        return normalize_direction(coefficients @ basis), "active"
    direction = torch.randn(dimension, generator=generator)
    return normalize_direction(direction), "isotropic"


def antithetic_gradient(directions: Sequence[torch.Tensor], plus: Sequence[float], minus: Sequence[float]) -> torch.Tensor:
    if not directions or len(directions) != len(plus) or len(directions) != len(minus):
        raise ValueError("directions, plus, and minus must have the same non-zero length")
    utilities = centered_rank_utilities([value for pair in zip(plus, minus) for value in pair])
    gradient = torch.zeros_like(directions[0], dtype=torch.float32)
    for index, direction in enumerate(directions):
        gradient += (utilities[2 * index] - utilities[2 * index + 1]).float() * direction
    return gradient / len(directions)


@dataclass
class ClipUp:
    step_size: float
    max_speed: float
    momentum: float = 0.9
    velocity: torch.Tensor | None = None

    @classmethod
    def from_sigma(cls, sigma: float, dimension: int, momentum: float = 0.9) -> ClipUp:
        if sigma <= 0 or dimension <= 0:
            raise ValueError("sigma and dimension must be positive")
        radius = sigma * math.sqrt(dimension)
        max_speed = radius / 15.0
        return cls(step_size=max_speed / 2.0, max_speed=max_speed, momentum=momentum)

    def update(self, gradient: torch.Tensor) -> torch.Tensor:
        gradient = gradient.detach().cpu().float()
        norm = torch.linalg.vector_norm(gradient)
        if not torch.isfinite(norm) or norm <= 1e-12:
            return torch.zeros_like(gradient)
        step = self.step_size * gradient / norm
        if self.velocity is None:
            self.velocity = torch.zeros_like(step)
        self.velocity = self.momentum * self.velocity + step
        speed = torch.linalg.vector_norm(self.velocity)
        if speed > self.max_speed:
            self.velocity *= self.max_speed / speed
        return self.velocity.clone()

    def reject_and_shrink(self, factor: float = 0.5) -> None:
        if not 0.0 < factor < 1.0:
            raise ValueError("shrink factor must be in (0, 1)")
        self.step_size *= factor
        self.max_speed *= factor
        if self.velocity is not None:
            self.velocity.zero_()

    def state_dict(self) -> dict:
        return {
            "step_size": self.step_size,
            "max_speed": self.max_speed,
            "momentum": self.momentum,
            "velocity": self.velocity,
        }

    @classmethod
    def load_state_dict(cls, state: Mapping) -> ClipUp:
        return cls(
            step_size=float(state["step_size"]),
            max_speed=float(state["max_speed"]),
            momentum=float(state["momentum"]),
            velocity=None if state.get("velocity") is None else state["velocity"].detach().cpu().float(),
        )


@dataclass(frozen=True)
class MatchSpec:
    match_id: str
    opponent: str
    seed: int
    map_size: int
    candidate_player: int


def make_match_schedule(
    *,
    generation: int,
    games: int,
    opponents: Sequence[str],
    seed_start: int,
    map_sizes: Sequence[int] = (12, 16, 24, 32),
    namespace: str = "train",
) -> tuple[MatchSpec, ...]:
    if games <= 0 or not opponents or not map_sizes:
        raise ValueError("games, opponents, and map_sizes must be non-empty")
    schedule = []
    opponent_order = tuple(opponents[generation % len(opponents) :]) + tuple(
        opponents[: generation % len(opponents)]
    )
    base_games, extra_games = divmod(games, len(opponent_order))
    index = 0
    for opponent_index, opponent in enumerate(opponent_order):
        opponent_games = base_games + int(opponent_index < extra_games)
        for local_index in range(opponent_games):
            map_size = int(map_sizes[(2 * generation + local_index) % len(map_sizes)])
            candidate_player = (generation + local_index) % 2
            seed = int(seed_start + generation * games + index)
            schedule.append(
                MatchSpec(
                    match_id=f"{namespace}-g{generation:04d}-m{index:03d}",
                    opponent=opponent,
                    seed=seed,
                    map_size=map_size,
                    candidate_player=candidate_player,
                )
            )
            index += 1
    return tuple(schedule)


def load_policy_state(path: Path) -> Mapping[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping) or "model_state_dict" not in checkpoint:
        raise ValueError(f"checkpoint must contain model_state_dict: {path}")
    return checkpoint["model_state_dict"]
