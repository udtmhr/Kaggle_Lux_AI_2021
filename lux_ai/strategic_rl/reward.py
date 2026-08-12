from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

from ..lux.game_constants import GAME_CONSTANTS
from ..lux_gym.reward_spaces import BaseRewardSpace, GameResultReward, RewardSpec
from .obs import night_turns_between


class SurvivalPotentialReward(BaseRewardSpace):
    """Zero-sum potential shaping that decays by local actor episode count."""

    def __init__(self, shaping_weight: float = 0.05, decay_games: int = 2000, **kwargs):
        super().__init__(**kwargs)
        self.shaping_weight = float(shaping_weight)
        self.decay_games = max(int(decay_games), 1)
        self.games = 0
        self.global_game_counter = None
        self.current_alpha = self.shaping_weight
        self.previous = np.zeros(2, dtype=np.float64)

    def set_global_game_counter(self, counter) -> None:
        """Use a process-shared, checkpoint-restored game count for decay."""
        self.global_game_counter = counter

    def _game_count(self) -> int:
        return self.games if self.global_game_counter is None else int(self.global_game_counter.value)

    def _increment_game_count(self) -> None:
        if self.global_game_counter is None:
            self.games += 1
            return
        with self.global_game_counter.get_lock():
            self.global_game_counter.value += 1

    def shaping_alpha(self) -> float:
        return self.shaping_weight * max(0.0, 1.0 - self._game_count() / self.decay_games)

    def get_info(self):
        return {
            "LOGGING_shaping_alpha": np.asarray([self.current_alpha], dtype=np.float32),
            "LOGGING_shaping_games": np.asarray([self._game_count()], dtype=np.float32),
        }

    @staticmethod
    def get_reward_spec() -> RewardSpec:
        return RewardSpec(-1.0, 1.0, True, False)

    @staticmethod
    def _potential(game_state) -> np.ndarray:
        remaining_nights = night_turns_between(game_state.turn, GAME_CONSTANTS["PARAMETERS"]["MAX_DAYS"])
        values = []
        for player in game_state.players:
            city_survival = 0.0
            for city in player.cities.values():
                required = max(city.get_light_upkeep() * remaining_nights, 1.0)
                city_survival += min(city.fuel / required, 1.0) * len(city.citytiles)
            unit_fuel = sum(u.cargo.wood + 10 * u.cargo.coal + 40 * u.cargo.uranium for u in player.units)
            values.append(
                2.0 * city_survival + 0.5 * len(player.units) + 0.01 * unit_fuel + 0.02 * player.research_points
            )
        return np.asarray(values, dtype=np.float64)

    def compute_rewards_and_done(self, game_state, done):
        potential = self._potential(game_state)
        delta = potential - self.previous
        delta -= delta.mean()
        self.current_alpha = self.shaping_alpha()
        rewards = np.clip(self.current_alpha * delta / 10.0, -0.25, 0.25)
        if done:
            terminal = [int(GameResultReward.compute_player_reward(p)) for p in game_state.players]
            rewards += (rankdata(terminal) - 1.0) * 2.0 - 1.0
            self._increment_game_count()
            self.previous[:] = 0.0
        else:
            self.previous = potential
        return tuple(np.clip(rewards, -1.0, 1.0)), done
