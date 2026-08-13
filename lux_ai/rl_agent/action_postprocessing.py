from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch
import torch.nn.functional as F

from ..lux.game_constants import GAME_CONSTANTS
from ..lux_gym.act_spaces import ACTION_MEANINGS
from ..utility_constants import DN_CYCLE_LEN, MAX_BOARD_SIZE, MAX_RESEARCH


def pos_to_loc(position: tuple[int, int], board_dims: tuple[int, int] = MAX_BOARD_SIZE) -> int:
    return position[0] * board_dims[1] + position[1]


def resolve_collision_rankings(
    game_state,
    player_id: int,
    policy_logits: Mapping[str, torch.Tensor],
    *,
    must_research: bool,
    can_build_carts: bool,
    force_last_turn_cart: bool = True,
) -> dict[str, torch.Tensor]:
    """Match the deployed agent's preference-based legality and friendly-collision resolver."""
    flat_log_probs = {
        key: torch.flatten(
            F.log_softmax(value.squeeze(0).squeeze(0), dim=-1),
            start_dim=-3,
            end_dim=-2,
        )
        for key, value in policy_logits.items()
    }
    my_flat_log_probs = {key: value[player_id] for key, value in flat_log_probs.items()}
    my_flat_actions = {
        key: value.argsort(dim=-1, descending=True) for key, value in my_flat_log_probs.items()
    }

    player = game_state.players[player_id]
    city_tile_matrix = np.zeros(MAX_BOARD_SIZE, dtype=bool)
    actionable_city_tiles = {}
    actionable_workers = {}
    actionable_carts = {}
    for unit in player.units:
        if not unit.can_act():
            continue
        if unit.is_worker():
            actionable = actionable_workers
        elif unit.is_cart():
            actionable = actionable_carts
        else:
            continue
        actionable.setdefault(pos_to_loc(unit.pos.astuple()), []).append(unit)
    for city_tile in player.city_tiles:
        city_tile_matrix[city_tile.pos.x, city_tile.pos.y] = True
        if city_tile.can_act():
            actionable_city_tiles[pos_to_loc(city_tile.pos.astuple())] = city_tile

    city_priorities = torch.argsort(
        my_flat_log_probs["city_tile"].max(dim=-1)[0], dim=-1, descending=True
    )
    units_to_build = max(player.city_tile_count - len(player.units), 0)
    research_remaining = max(MAX_RESEARCH - player.research_points, 0)
    for location_tensor in city_priorities:
        location = location_tensor.item()
        actions = my_flat_actions["city_tile"][location]
        if location not in actionable_city_tiles:
            continue
        for action in actions:
            illegal = False
            action_meaning = ACTION_MEANINGS["city_tile"][action]
            if action_meaning == "BUILD_CART" and not can_build_carts:
                illegal = True
            elif action_meaning.startswith("BUILD_"):
                if units_to_build > 0:
                    units_to_build -= 1
                else:
                    illegal = True
            elif action_meaning == "RESEARCH":
                if research_remaining > 0:
                    research_remaining -= 1
                else:
                    illegal = True
            elif (
                action_meaning == "NO-OP"
                and game_state.turn >= DN_CYCLE_LEN
                and research_remaining > 0
                and must_research
            ):
                illegal = True
            if (
                force_last_turn_cart
                and game_state.turn >= GAME_CONSTANTS["PARAMETERS"]["MAX_DAYS"] - 1
            ):
                illegal = action_meaning != "BUILD_CART"
            if illegal:
                my_flat_log_probs["city_tile"][location, action] = float("-inf")
            else:
                break

    occupied_squares = np.zeros(MAX_BOARD_SIZE, dtype=bool)
    max_location = MAX_BOARD_SIZE[0] * MAX_BOARD_SIZE[1]
    combined_unit_log_probs = torch.cat(
        [
            my_flat_log_probs["worker"].max(dim=-1)[0],
            my_flat_log_probs["cart"].max(dim=-1)[0],
        ],
        dim=-1,
    )
    unit_priorities = torch.argsort(combined_unit_log_probs, dim=-1, descending=True)
    for location_tensor in unit_priorities:
        location = location_tensor.item()
        if location >= max_location:
            unit_type = "cart"
            actionable = actionable_carts
        else:
            unit_type = "worker"
            actionable = actionable_workers
        location %= max_location
        actions = my_flat_actions[unit_type][location]
        actionable_units = actionable.get(location)
        if actionable_units is None:
            continue
        acted_count = 0
        for action in actions:
            action_meaning = ACTION_MEANINGS[unit_type][action]
            if action_meaning.startswith("MOVE_"):
                direction = action_meaning.split("_")[1]
                new_position = actionable_units[acted_count].pos.translate(direction, 1)
            else:
                new_position = actionable_units[acted_count].pos
            illegal = (
                new_position.x < 0
                or new_position.x >= game_state.map_width
                or new_position.y < 0
                or new_position.y >= game_state.map_height
                or (
                    occupied_squares[new_position.x, new_position.y]
                    and not city_tile_matrix[new_position.x, new_position.y]
                )
            )
            if illegal:
                my_flat_log_probs[unit_type][location, action] = float("-inf")
            else:
                occupied_squares[new_position.x, new_position.y] = True
                acted_count += 1
            if acted_count >= len(actionable_units):
                break

    return {
        key: value.view(1, *value.shape[:-2], *MAX_BOARD_SIZE, -1).argsort(
            dim=-1, descending=True
        )
        for key, value in flat_log_probs.items()
    }
