from __future__ import annotations

import gym
import numpy as np

from ..lux.game import Game
from ..lux.game_constants import GAME_CONSTANTS
from ..lux_gym.obs_spaces import (
    FixedShapeContinuousObsV2,
    FixedShapeObs,
    _FixedShapeContinuousObsWrapperV2,
)
from ..utility_constants import MAX_BOARD_SIZE

_DAY = int(GAME_CONSTANTS["PARAMETERS"]["DAY_LENGTH"])
_NIGHT = int(GAME_CONSTANTS["PARAMETERS"]["NIGHT_LENGTH"])
_CYCLE = _DAY + _NIGHT
_MAX_TURNS = int(GAME_CONSTANTS["PARAMETERS"]["MAX_DAYS"])


def night_turns_between(start: int, stop: int) -> int:
    """Count night turns in [start, stop), clipped to the game horizon."""
    start = max(0, min(int(start), _MAX_TURNS))
    stop = max(start, min(int(stop), _MAX_TURNS))
    return sum(1 for turn in range(start, stop) if turn % _CYCLE >= _DAY)


class SurvivalStrategicObs(FixedShapeObs):
    """First-place observation plus explicit survival and logistics features."""

    PLAYER_SPATIAL = (
        "unit_next_night_survival",
        "unit_return_margin",
        "city_next_night_survival",
        "city_game_survival",
        "city_fuel_deficit",
        "city_expansion_capacity",
        "city_stranded_fuel",
        "fuel_delivery_proximity",
        "resource_access",
        "resource_control",
    )
    NEUTRAL_SPATIAL = ("buildable", "map_edge")
    TEMPORAL = ("delta_units", "delta_city_tiles", "delta_research")

    def get_obs_spec(self, board_dims: tuple[int, int] = MAX_BOARD_SIZE) -> gym.spaces.Dict:
        x, y = board_dims
        spaces = dict(FixedShapeContinuousObsV2().get_obs_spec(board_dims).spaces)
        spaces.update(
            {key: gym.spaces.Box(-1.0, 1.0, shape=(1, 2, x, y), dtype=np.float32) for key in self.PLAYER_SPATIAL}
        )
        spaces.update(
            {key: gym.spaces.Box(0.0, 1.0, shape=(1, 1, x, y), dtype=np.float32) for key in self.NEUTRAL_SPATIAL}
        )
        spaces.update({key: gym.spaces.Box(-1.0, 1.0, shape=(1, 2), dtype=np.float32) for key in self.TEMPORAL})
        return gym.spaces.Dict(spaces)

    def wrap_env(self, env) -> gym.Wrapper:
        return _SurvivalStrategicObsWrapper(env)


class _SurvivalStrategicObsWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.base = _FixedShapeContinuousObsWrapperV2(env)
        self.previous: dict[int, tuple[int, int, int]] = {}

    def reset(self, **kwargs):
        observation, reward, done, info = self.env.reset(**kwargs)
        self.previous.clear()
        return self.observation(observation), reward, done, info

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return self.observation(observation), reward, done, info

    @staticmethod
    def _fuel(unit) -> float:
        return float(unit.cargo.wood + 10 * unit.cargo.coal + 40 * unit.cargo.uranium)

    def observation(self, game: Game) -> dict[str, np.ndarray]:
        if game.turn == 0:
            self.previous.clear()
        obs = self.base.observation(game)
        width, height = game.map_width, game.map_height
        shape = (1, 2, width, height)
        for key in SurvivalStrategicObs.PLAYER_SPATIAL:
            obs[key] = np.zeros(shape, dtype=np.float32)
        for key in SurvivalStrategicObs.NEUTRAL_SPATIAL:
            obs[key] = np.zeros((1, 1, width, height), dtype=np.float32)
        for key in SurvivalStrategicObs.TEMPORAL:
            obs[key] = np.zeros((1, 2), dtype=np.float32)

        next_cycle = min(((game.turn // _CYCLE) + 1) * _CYCLE, _MAX_TURNS)
        next_night_turns = night_turns_between(game.turn, next_cycle)
        remaining_nights = night_turns_between(game.turn, _MAX_TURNS)
        turns_until_night = 0 if game.is_night else max(0, _DAY - game.turn % _CYCLE)

        all_resource_positions = []
        for row in game.map.map:
            for cell in row:
                x, y = cell.pos.x, cell.pos.y
                obs["map_edge"][0, 0, x, y] = float(x in (0, width - 1) or y in (0, height - 1))
                obs["buildable"][0, 0, x, y] = float(not cell.has_resource() and cell.citytile is None)
                if cell.has_resource():
                    all_resource_positions.append(cell.pos)

        for player in game.players:
            team = player.team
            city_positions = [tile.pos for city in player.cities.values() for tile in city.citytiles]
            total_deficit = total_surplus = total_required = 0.0
            city_values = []
            for city in player.cities.values():
                upkeep = max(float(city.get_light_upkeep()), 1e-6)
                next_required = upkeep * next_night_turns
                game_required = upkeep * remaining_nights
                deficit = max(0.0, game_required - float(city.fuel))
                surplus = max(0.0, float(city.fuel) - game_required)
                total_deficit += deficit
                total_surplus += surplus
                total_required += game_required
                city_values.append((city, upkeep, next_required, game_required, deficit))
            stranded = min(total_deficit, total_surplus) / max(total_required, 1.0)

            for city, upkeep, next_required, game_required, deficit in city_values:
                next_ratio = np.clip(float(city.fuel) / max(next_required, 1.0), 0.0, 1.0)
                game_ratio = np.clip(float(city.fuel) / max(game_required, 1.0), 0.0, 1.0)
                deficit_ratio = np.clip(deficit / max(game_required, 1.0), 0.0, 1.0)
                expansion = np.clip((float(city.fuel) - next_required) / max(upkeep * _NIGHT, 1.0), -1.0, 1.0)
                for tile in city.citytiles:
                    x, y = tile.pos.x, tile.pos.y
                    obs["city_next_night_survival"][0, team, x, y] = next_ratio
                    obs["city_game_survival"][0, team, x, y] = game_ratio
                    obs["city_fuel_deficit"][0, team, x, y] = deficit_ratio
                    obs["city_expansion_capacity"][0, team, x, y] = expansion
                    obs["city_stranded_fuel"][0, team, x, y] = np.clip(stranded, 0.0, 1.0)

            for unit in player.units:
                x, y = unit.pos.x, unit.pos.y
                fuel = self._fuel(unit)
                upkeep = float(GAME_CONSTANTS["PARAMETERS"]["LIGHT_UPKEEP"]["WORKER" if unit.is_worker() else "CART"])
                required = upkeep * max(next_night_turns, 1)
                obs["unit_next_night_survival"][0, team, x, y] = np.clip(fuel / max(required, 1.0), 0.0, 1.0)
                distance = min((unit.pos.distance_to(pos) for pos in city_positions), default=width + height)
                margin = (turns_until_night + fuel / max(upkeep, 1.0) - distance) / max(width + height, 1)
                obs["unit_return_margin"][0, team, x, y] = np.clip(margin, -1.0, 1.0)
                obs["fuel_delivery_proximity"][0, team, x, y] = 1.0 - np.clip(
                    distance / max(width + height, 1), 0.0, 1.0
                )

            own_sources = [u.pos for u in player.units] + city_positions
            enemy = game.players[1 - team]
            enemy_sources = [u.pos for u in enemy.units] + [t.pos for c in enemy.cities.values() for t in c.citytiles]
            for resource_pos in all_resource_positions:
                x, y = resource_pos.x, resource_pos.y
                own_dist = min((resource_pos.distance_to(pos) for pos in own_sources), default=width + height)
                enemy_dist = min((resource_pos.distance_to(pos) for pos in enemy_sources), default=width + height)
                obs["resource_access"][0, team, x, y] = 1.0 - np.clip(own_dist / max(width + height, 1), 0.0, 1.0)
                obs["resource_control"][0, team, x, y] = np.clip(
                    (enemy_dist - own_dist) / max(width + height, 1), -1.0, 1.0
                )

            current = (len(player.units), player.city_tile_count, player.research_points)
            previous = self.previous.get(team, current)
            obs["delta_units"][0, team] = np.clip((current[0] - previous[0]) / 10.0, -1.0, 1.0)
            obs["delta_city_tiles"][0, team] = np.clip((current[1] - previous[1]) / 10.0, -1.0, 1.0)
            obs["delta_research"][0, team] = np.clip((current[2] - previous[2]) / 20.0, -1.0, 1.0)
            self.previous[team] = current
        return obs
