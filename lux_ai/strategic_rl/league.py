from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from ..lux.constants import Constants
from ..lux.game_constants import GAME_CONSTANTS
from ..lux_gym.act_spaces import ACTION_MEANINGS_TO_IDX
from .obs import night_turns_between


@dataclass(frozen=True)
class Opponent:
    name: str
    checkpoint: Optional[str] = None
    kind: str = "snapshot"
    config: Optional[str] = None
    weight: float = 1.0
    strategy: str = "economy"
    slot: int = 0


class PFSPSampler:
    """Prioritized fictitious self-play with an explicit teacher sampling floor."""

    def __init__(
        self,
        opponents: Sequence[Opponent],
        power: float = 2.0,
        teacher_floor: float = 0.10,
        exploration: float = 0.02,
    ):
        if not opponents:
            raise ValueError("PFSP requires at least one opponent")
        if not 0.0 <= teacher_floor < 1.0:
            raise ValueError("teacher_floor must be in [0, 1)")
        self.opponents = tuple(opponents)
        self.power = float(power)
        self.teacher_floor = float(teacher_floor)
        self.exploration = float(exploration)

    def probabilities(self, win_rates: Mapping[str, float]) -> np.ndarray:
        # Prefer opponents near a 50% learner win rate. Extremely weak or
        # currently impossible opponents retain a small exploration share.
        scores = np.asarray([], dtype=np.float64)
        priorities = []
        for opponent in self.opponents:
            win_rate = float(np.clip(win_rates.get(opponent.name, 0.5), 0.0, 1.0))
            learnability = max(win_rate * (1.0 - win_rate), self.exploration)
            priorities.append(max(opponent.weight, 0.0) * learnability**self.power)
        scores = np.asarray(priorities, dtype=np.float64)
        if scores.sum() <= 0.0:
            scores = np.ones(len(self.opponents), dtype=np.float64)
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

    def sample_index(self, win_rates: Mapping[str, float], rng: np.random.Generator) -> int:
        return int(rng.choice(len(self.opponents), p=self.probabilities(win_rates)))


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
            strategy=str(entry.get("strategy", "economy")),
            slot=int(entry.get("slot", 0)),
        )
        for entry in entries
    )
    names = [opponent.name for opponent in opponents]
    if len(names) != len(set(names)):
        raise ValueError("league opponent names must be unique")
    for opponent in opponents:
        if opponent.kind not in {"selfplay", "teacher", "snapshot", "learner_snapshot", "rule_based"}:
            raise ValueError(f"Unsupported league opponent kind: {opponent.kind}")
        if opponent.kind in {"teacher", "snapshot"} and opponent.checkpoint is None:
            raise ValueError(f"{opponent.name} requires a checkpoint")
        if opponent.kind == "teacher" and opponent.config is None:
            raise ValueError(f"{opponent.name} requires a config")
        if opponent.kind == "learner_snapshot" and opponent.slot < 0:
            raise ValueError(f"{opponent.name} requires a non-negative slot")
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
    """Compatibility wrapper for the survival/logistics-aware rule policy."""
    actions, _, _, _ = rule_based_guidance(
        games,
        available_actions_mask,
        action_template,
        env_indices,
        opponent_players,
        strategy="economy",
    )
    return actions


def _fuel(unit) -> float:
    return float(unit.cargo.wood + 10 * unit.cargo.coal + 40 * unit.cargo.uranium)


def _move_towards(
    unit,
    target,
    game,
    available_actions_mask,
    entity: str,
    env_index: int,
    player_id: int,
    reserved: set[tuple[int, int]],
) -> tuple[str, tuple[int, int]]:
    if unit.pos.distance_to(target) == 0:
        return "NO-OP", (unit.pos.x, unit.pos.y)
    candidates = []
    for direction in Constants.DIRECTIONS.astuple(include_center=False):
        destination = unit.pos.translate(direction, 1)
        if not (0 <= destination.x < game.map_width and 0 <= destination.y < game.map_height):
            continue
        meaning = f"MOVE_{direction}"
        action = ACTION_MEANINGS_TO_IDX[entity][meaning]
        legal = bool(
            available_actions_mask[entity][env_index, 0, player_id, unit.pos.x, unit.pos.y, action]
        )
        if legal and (destination.x, destination.y) not in reserved:
            candidates.append((destination.distance_to(target), direction, destination))
    if not candidates:
        return "NO-OP", (unit.pos.x, unit.pos.y)
    _, direction, destination = min(candidates, key=lambda item: (item[0], item[1]))
    return f"MOVE_{direction}", (destination.x, destination.y)


def rule_based_guidance(
    games: Sequence,
    available_actions_mask: Mapping[str, torch.Tensor],
    action_template: Mapping[str, torch.Tensor],
    env_indices: Sequence[int],
    players: Sequence[int],
    strategy: str = "economy",
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """Return rule actions, certified masks, worker intents, and intent masks.

    Intent labels are ``mine=0``, ``deliver=1``, ``build=2``, ``return=3``.
    Certified masks intentionally cover only phase-safe logistics decisions;
    the policy remains unconstrained in ambiguous tactical positions.
    """
    actions = {key: torch.zeros_like(value) for key, value in action_template.items()}
    confidence = {
        key: torch.zeros_like(value[..., 0], dtype=torch.bool) for key, value in action_template.items()
    }
    worker_intents = torch.zeros_like(action_template["worker"][..., 0], dtype=torch.long)
    intent_mask = torch.zeros_like(worker_intents, dtype=torch.bool)
    day_length = int(GAME_CONSTANTS["PARAMETERS"]["DAY_LENGTH"])
    cycle_length = day_length + int(GAME_CONSTANTS["PARAMETERS"]["NIGHT_LENGTH"])
    max_turns = int(GAME_CONSTANTS["PARAMETERS"]["MAX_DAYS"])
    for env_index, player_id in zip(env_indices, players):
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
        next_cycle = min(((game.turn // cycle_length) + 1) * cycle_length, max_turns)
        next_nights = night_turns_between(game.turn, next_cycle)
        turns_until_night = 0 if game.is_night else max(0, day_length - game.turn % cycle_length)
        city_deficits = {}
        deficit_positions = []
        for city in player.cities.values():
            required = float(city.get_light_upkeep()) * next_nights
            deficit = max(required - float(city.fuel), 0.0)
            city_deficits[city.cityid] = deficit
            if deficit > 0:
                deficit_positions.extend(tile.pos for tile in city.citytiles)
        unit_count = len(player.units)
        city_tile_count = player.city_tile_count

        for city in player.cities.values():
            for tile in city.citytiles:
                if not tile.can_act():
                    continue
                city_safe = city_deficits[city.cityid] <= 0.0
                if unit_count < city_tile_count and (city_safe or strategy == "expansion"):
                    meaning = "BUILD_WORKER"
                elif player.research_points < 200:
                    meaning = "RESEARCH"
                else:
                    meaning = "BUILD_WORKER"
                actions["city_tile"][env_index, 0, player_id, tile.pos.x, tile.pos.y, 0] = _legal_action(
                    available_actions_mask, "city_tile", env_index, player_id, tile.pos.x, tile.pos.y, meaning
                )
                if meaning == "BUILD_WORKER":
                    unit_count += 1
                if meaning == "RESEARCH" or city_safe:
                    confidence["city_tile"][env_index, 0, player_id, tile.pos.x, tile.pos.y] = True

        reserved = set()
        for unit in player.units:
            if not unit.can_act():
                continue
            entity = "worker" if unit.is_worker() else "cart"
            x, y = unit.pos.x, unit.pos.y
            fuel = _fuel(unit)
            nearest_city_distance = min((unit.pos.distance_to(pos) for pos in city_positions), default=game.map_width + game.map_height)
            must_return = bool(game.is_night or turns_until_night <= nearest_city_distance + 2)
            can_safely_build = (
                unit.is_worker()
                and unit.can_build(game.map)
                and not game.is_night
                and (
                    turns_until_night >= (7 if strategy == "survival" else 4)
                    or fuel >= 23.0 * max(next_nights, 1)
                )
            )
            certified = False
            if can_safely_build and unit.get_cargo_space_left() == 0 and strategy != "survival":
                meaning = "BUILD_CITY"
                intent = 2
                certified = True
                destination = (x, y)
            else:
                if must_return and city_positions:
                    targets = city_positions
                    intent = 3
                    certified = True
                elif unit.get_cargo_space_left() == 0 and (deficit_positions or city_positions):
                    targets = deficit_positions or city_positions
                    intent = 1
                    certified = True
                else:
                    targets = resource_positions
                    intent = 0
                if not targets:
                    continue
                target = min(targets, key=lambda pos: abs(pos.x - x) + abs(pos.y - y))
                meaning, destination = _move_towards(
                    unit,
                    target,
                    game,
                    available_actions_mask,
                    entity,
                    env_index,
                    player_id,
                    reserved,
                )
            reserved.add(destination)
            actions[entity][env_index, 0, player_id, x, y, 0] = _legal_action(
                available_actions_mask, entity, env_index, player_id, x, y, meaning
            )
            if certified:
                confidence[entity][env_index, 0, player_id, x, y] = True
            if unit.is_worker():
                worker_intents[env_index, 0, player_id, x, y] = intent
                intent_mask[env_index, 0, player_id, x, y] = True
    return actions, confidence, worker_intents, intent_mask
