from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from ..lux.constants import Constants
from ..lux_gym.act_spaces import ACTION_MEANINGS_TO_IDX


@dataclass(frozen=True)
class Opponent:
    name: str
    checkpoint: Optional[str] = None
    kind: str = "snapshot"
    config: Optional[str] = None
    weight: float = 1.0


class PFSPSampler:
    """Prioritized fictitious self-play with an explicit teacher sampling floor."""

    def __init__(self, opponents: Sequence[Opponent], power: float = 2.0, teacher_floor: float = 0.10):
        if not opponents:
            raise ValueError("PFSP requires at least one opponent")
        if not 0.0 <= teacher_floor < 1.0:
            raise ValueError("teacher_floor must be in [0, 1)")
        self.opponents = tuple(opponents)
        self.power = float(power)
        self.teacher_floor = float(teacher_floor)

    def probabilities(self, win_rates: Mapping[str, float]) -> np.ndarray:
        scores = np.asarray(
            [(1.0 - np.clip(win_rates.get(o.name, 0.5), 0.0, 1.0)) ** self.power for o in self.opponents],
            dtype=np.float64,
        )
        probabilities = scores / max(scores.sum(), 1e-12)
        teacher_indices = [i for i, opponent in enumerate(self.opponents) if opponent.kind == "teacher"]
        if teacher_indices and probabilities[teacher_indices].sum() < self.teacher_floor:
            probabilities *= 1.0 - self.teacher_floor
            teacher_share = self.teacher_floor / len(teacher_indices)
            probabilities[teacher_indices] += teacher_share
            probabilities /= probabilities.sum()
        return probabilities

    def sample(self, win_rates: Mapping[str, float], rng: np.random.Generator) -> Opponent:
        return self.opponents[int(rng.choice(len(self.opponents), p=self.probabilities(win_rates)))]


class LeagueSampler:
    """Fixed-weight episode sampler used by IMPALA rollout actors."""

    def __init__(self, opponents: Sequence[Opponent]):
        if not opponents:
            raise ValueError("league_opponents must contain at least one entry")
        weights = np.asarray([opponent.weight for opponent in opponents], dtype=np.float64)
        if not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
            raise ValueError("league opponent weights must be finite, non-negative, and have a positive sum")
        self.opponents = tuple(opponents)
        self.probabilities = weights / weights.sum()

    def sample_index(self, rng: np.random.Generator) -> int:
        return int(rng.choice(len(self.opponents), p=self.probabilities))


def opponents_from_config(entries: Sequence[Mapping]) -> tuple[Opponent, ...]:
    opponents = tuple(
        Opponent(
            name=str(entry["name"]),
            checkpoint=str(entry["checkpoint"]) if entry.get("checkpoint") else None,
            kind=str(entry.get("kind", "snapshot")),
            config=str(entry["config"]) if entry.get("config") else None,
            weight=float(entry.get("weight", 1.0)),
        )
        for entry in entries
    )
    names = [opponent.name for opponent in opponents]
    if len(names) != len(set(names)):
        raise ValueError("league opponent names must be unique")
    for opponent in opponents:
        if opponent.kind not in {"selfplay", "teacher", "snapshot", "rule_based"}:
            raise ValueError(f"Unsupported league opponent kind: {opponent.kind}")
        if opponent.kind in {"teacher", "snapshot"} and opponent.checkpoint is None:
            raise ValueError(f"{opponent.name} requires a checkpoint")
        if opponent.kind == "teacher" and opponent.config is None:
            raise ValueError(f"{opponent.name} requires a config")
    return opponents


def merge_player_actions(
    learner_actions: Mapping[str, torch.Tensor],
    opponent_actions: Mapping[str, torch.Tensor],
    env_indices: Sequence[int],
    opponent_players: Sequence[int],
) -> dict[str, torch.Tensor]:
    """Replace only the external opponent side, preserving learner-side actions."""
    merged = {key: value.clone() for key, value in learner_actions.items()}
    for env_index, player in zip(env_indices, opponent_players):
        for entity in merged:
            merged[entity][env_index, :, player] = opponent_actions[entity][env_index, :, player]
    return merged


def merge_player_actions_inplace(
    merged: Mapping[str, torch.Tensor],
    opponent_actions: Mapping[str, torch.Tensor],
    env_indices: Sequence[int],
    opponent_players: Sequence[int],
) -> None:
    """Replace opponent actions without cloning the full action tensors again."""
    for env_index, player in zip(env_indices, opponent_players):
        for entity in merged:
            merged[entity][env_index, :, player] = opponent_actions[entity][env_index, :, player]


def learner_player_mask(
    opponents: Sequence[Opponent], selected: Sequence[int], learner_players: Sequence[int], device: torch.device
) -> torch.Tensor:
    mask = torch.ones((len(selected), 2), dtype=torch.bool, device=device)
    for env_index, opponent_index in enumerate(selected):
        if opponents[opponent_index].kind != "selfplay":
            mask[env_index] = False
            mask[env_index, learner_players[env_index]] = True
    return mask


def _legal_action(
    available_actions_mask: Mapping[str, torch.Tensor],
    entity: str,
    env_index: int,
    player: int,
    x: int,
    y: int,
    meaning: str,
) -> int:
    action = ACTION_MEANINGS_TO_IDX[entity][meaning]
    if bool(available_actions_mask[entity][env_index, 0, player, x, y, action]):
        return action
    return 0


def rule_based_actions(
    games: Sequence,
    available_actions_mask: Mapping[str, torch.Tensor],
    action_template: Mapping[str, torch.Tensor],
    env_indices: Sequence[int],
    opponent_players: Sequence[int],
) -> dict[str, torch.Tensor]:
    """A small deterministic economy bot expressed in the native tensor action space."""
    actions = {key: torch.zeros_like(value) for key, value in action_template.items()}
    for env_index, player_id in zip(env_indices, opponent_players):
        game = games[env_index].game_state
        player = game.players[player_id]
        resource_positions = [
            cell.pos
            for row in game.map.map
            for cell in row
            if cell.has_resource()
            and (
                cell.resource.type == Constants.RESOURCE_TYPES.WOOD
                or (cell.resource.type == Constants.RESOURCE_TYPES.COAL and player.researched_coal())
                or (cell.resource.type == Constants.RESOURCE_TYPES.URANIUM and player.researched_uranium())
            )
        ]
        city_positions = [tile.pos for city in player.cities.values() for tile in city.citytiles]
        unit_count = len(player.units)
        city_tile_count = player.city_tile_count

        for city in player.cities.values():
            for tile in city.citytiles:
                if not tile.can_act():
                    continue
                meaning = "BUILD_WORKER" if unit_count < city_tile_count else "RESEARCH"
                actions["city_tile"][env_index, 0, player_id, tile.pos.x, tile.pos.y, 0] = _legal_action(
                    available_actions_mask, "city_tile", env_index, player_id, tile.pos.x, tile.pos.y, meaning
                )
                if meaning == "BUILD_WORKER":
                    unit_count += 1

        for unit in player.units:
            if not unit.can_act():
                continue
            entity = "worker" if unit.is_worker() else "cart"
            x, y = unit.pos.x, unit.pos.y
            cell = game.map.get_cell(x, y)
            if unit.is_worker() and unit.get_cargo_space_left() == 0 and cell.citytile is None and not cell.has_resource():
                meaning = "BUILD_CITY"
            else:
                targets = city_positions if unit.get_cargo_space_left() == 0 and city_positions else resource_positions
                if not targets:
                    continue
                target = min(targets, key=lambda pos: abs(pos.x - x) + abs(pos.y - y))
                if abs(target.x - x) >= abs(target.y - y) and target.x != x:
                    direction = "e" if target.x > x else "w"
                elif target.y != y:
                    direction = "s" if target.y > y else "n"
                else:
                    continue
                meaning = f"MOVE_{direction}"
            actions[entity][env_index, 0, player_id, x, y, 0] = _legal_action(
                available_actions_mask, entity, env_index, player_id, x, y, meaning
            )
    return actions
