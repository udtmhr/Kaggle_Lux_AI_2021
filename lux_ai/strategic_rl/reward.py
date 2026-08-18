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


class RelativeCountPotentialReward(BaseRewardSpace):
    """Reward temporal improvements in city-tile and unit count advantage.

    The shaping signal for player ``i`` is the change in
    ``potential_i - potential_opponent``.  This rewards both creating/keeping
    friendly assets and destroying/preventing enemy assets without repeatedly
    rewarding an advantage that has not changed.
    """

    def __init__(
        self,
        city_tile_weight: float = 2.0,
        unit_weight: float = 0.5,
        shaping_weight: float = 0.05,
        count_scale: float = 10.0,
        max_step_shaping: float = 0.05,
        decay_games: int = 14000,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if city_tile_weight < 0.0 or unit_weight < 0.0:
            raise ValueError("count weights must be non-negative")
        if shaping_weight < 0.0:
            raise ValueError("shaping_weight must be non-negative")
        if count_scale <= 0.0:
            raise ValueError("count_scale must be positive")
        if max_step_shaping < 0.0:
            raise ValueError("max_step_shaping must be non-negative")
        self.city_tile_weight = float(city_tile_weight)
        self.unit_weight = float(unit_weight)
        self.shaping_weight = float(shaping_weight)
        self.count_scale = float(count_scale)
        self.max_step_shaping = float(max_step_shaping)
        self.decay_games = max(int(decay_games), 1)
        self.games = 0
        self.global_game_counter = None
        self.current_alpha = self.shaping_weight
        self.previous_relative = np.zeros(2, dtype=np.float64)
        self.initialized = False

    @staticmethod
    def get_reward_spec() -> RewardSpec:
        return RewardSpec(-1.0, 1.0, True, False)

    def set_global_game_counter(self, counter) -> None:
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

    def _relative_potential(self, game_state) -> np.ndarray:
        potentials = np.asarray(
            [
                self.city_tile_weight * player.city_tile_count
                + self.unit_weight * len(player.units)
                for player in game_state.players
            ],
            dtype=np.float64,
        )
        return potentials - potentials[::-1]

    def compute_rewards_and_done(self, game_state, done):
        relative = self._relative_potential(game_state)
        self.current_alpha = self.shaping_alpha()
        if not self.initialized or game_state.turn == 0:
            delta = np.zeros(2, dtype=np.float64)
            self.initialized = True
        else:
            delta = relative - self.previous_relative
        shaping = np.clip(
            self.current_alpha * delta / self.count_scale,
            -self.max_step_shaping,
            self.max_step_shaping,
        )
        rewards = shaping
        if done:
            terminal = [int(GameResultReward.compute_player_reward(p)) for p in game_state.players]
            rewards += (rankdata(terminal) - 1.0) * 2.0 - 1.0
            self._increment_game_count()
            self.previous_relative[:] = 0.0
            self.initialized = False
        else:
            self.previous_relative = relative
        return tuple(np.clip(rewards, -1.0, 1.0)), done


class StrategicPotentialRewardV2(BaseRewardSpace):
    """Phase-aware zero-sum shaping for survival, delivery, and safe expansion.

    Unlike :class:`SurvivalPotentialReward`, raw unit cargo is not valuable by
    itself. Cargo contributes only when it can be delivered to an under-fuelled
    city, while unreachable cargo is penalised. All features are normalised so
    that large maps and large armies do not change the reward scale.
    """

    def __init__(
        self,
        shaping_weight: float = 0.05,
        shaping_floor: float = 0.01,
        decay_games: int = 14000,
        discounting: float = 0.999,
        max_step_shaping: float = 0.05,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.shaping_weight = float(shaping_weight)
        self.shaping_floor = float(shaping_floor)
        if not 0.0 <= self.shaping_floor <= self.shaping_weight:
            raise ValueError("shaping_floor must be between zero and shaping_weight")
        self.decay_games = max(int(decay_games), 1)
        self.discounting = float(discounting)
        self.max_step_shaping = float(max_step_shaping)
        self.games = 0
        self.global_game_counter = None
        self.current_alpha = self.shaping_weight
        self.previous = np.zeros(2, dtype=np.float64)
        self.initialized = False

    def set_global_game_counter(self, counter) -> None:
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
        progress = min(self._game_count() / self.decay_games, 1.0)
        return self.shaping_weight + progress * (self.shaping_floor - self.shaping_weight)

    def get_info(self):
        return {
            "LOGGING_shaping_alpha": np.asarray([self.current_alpha], dtype=np.float32),
            "LOGGING_shaping_games": np.asarray([self._game_count()], dtype=np.float32),
        }

    @staticmethod
    def get_reward_spec() -> RewardSpec:
        return RewardSpec(-1.0, 1.0, True, False)

    @staticmethod
    def _unit_fuel(unit) -> float:
        return float(unit.cargo.wood + 10 * unit.cargo.coal + 40 * unit.cargo.uranium)

    @classmethod
    def _potential(cls, game_state) -> np.ndarray:
        max_turns = int(GAME_CONSTANTS["PARAMETERS"]["MAX_DAYS"])
        day_length = int(GAME_CONSTANTS["PARAMETERS"]["DAY_LENGTH"])
        night_length = int(GAME_CONSTANTS["PARAMETERS"]["NIGHT_LENGTH"])
        cycle = day_length + night_length
        next_cycle = min(((game_state.turn // cycle) + 1) * cycle, max_turns)
        next_nights = night_turns_between(game_state.turn, next_cycle)
        remaining_nights = night_turns_between(game_state.turn, max_turns)
        turns_until_night = 0 if game_state.is_night else max(0, day_length - game_state.turn % cycle)
        board_area = max(int(game_state.map_width) * int(game_state.map_height), 1)

        values = []
        for player in game_state.players:
            city_tiles = max(int(player.city_tile_count), 1)
            next_safe = game_safe = next_deficit = 0.0
            deficit_positions = []
            city_positions = []
            for city in player.cities.values():
                upkeep = max(float(city.get_light_upkeep()), 1.0)
                n_tiles = len(city.citytiles)
                next_required = max(upkeep * next_nights, 1.0)
                game_required = max(upkeep * remaining_nights, 1.0)
                next_safe += min(float(city.fuel) / next_required, 1.0) * n_tiles
                game_safe += min(float(city.fuel) / game_required, 1.0) * n_tiles
                deficit = max(next_required - float(city.fuel), 0.0)
                next_deficit += deficit
                for tile in city.citytiles:
                    city_positions.append(tile.pos)
                    if deficit > 0.0:
                        deficit_positions.append(tile.pos)

            deliverable = stranded = return_margin = expansion_ready = total_cargo = 0.0
            for unit in player.units:
                fuel = cls._unit_fuel(unit)
                total_cargo += fuel
                upkeep = float(
                    GAME_CONSTANTS["PARAMETERS"]["LIGHT_UPKEEP"]["WORKER" if unit.is_worker() else "CART"]
                )
                nearest_city = min((unit.pos.distance_to(pos) for pos in city_positions), default=board_area)
                margin = turns_until_night + fuel / max(upkeep, 1.0) - nearest_city
                return_margin += np.clip(margin / max(game_state.map_width + game_state.map_height, 1), -1.0, 1.0)

                if deficit_positions and fuel > 0.0:
                    distance = min(unit.pos.distance_to(pos) for pos in deficit_positions)
                    reach = turns_until_night + fuel / max(upkeep, 1.0)
                    fraction = np.clip((reach - distance + 1.0) / max(reach + 1.0, 1.0), 0.0, 1.0)
                    deliverable += fuel * fraction
                    stranded += fuel * (1.0 - fraction)

                if unit.is_worker() and unit.can_build(game_state.map) and not game_state.is_night:
                    if turns_until_night >= 4 or fuel >= 23.0 * max(next_nights, 1):
                        expansion_ready += 1.0

            n_units = max(len(player.units), 1)
            fuel_scale = max(next_deficit + total_cargo, 100.0)
            coal_threshold = float(GAME_CONSTANTS["PARAMETERS"]["RESEARCH_REQUIREMENTS"]["COAL"])
            uranium_threshold = float(GAME_CONSTANTS["PARAMETERS"]["RESEARCH_REQUIREMENTS"]["URANIUM"])
            research = float(player.research_points)
            research_milestones = 0.5 * min(research / coal_threshold, 1.0)
            research_milestones += 0.5 * np.clip(
                (research - coal_threshold) / max(uranium_threshold - coal_threshold, 1.0), 0.0, 1.0
            )
            city_growth = np.log1p(player.city_tile_count) / np.log1p(board_area)

            values.append(
                1.25 * (next_safe / city_tiles)
                + 0.35 * (game_safe / city_tiles)
                - 0.75 * (next_deficit / max(next_deficit + sum(c.fuel for c in player.cities.values()), 1.0))
                + 0.45 * (deliverable / fuel_scale)
                - 0.35 * (stranded / fuel_scale)
                + 0.65 * city_growth
                + 0.30 * (expansion_ready / n_units)
                + 0.20 * (return_margin / n_units)
                + 0.25 * research_milestones
            )
        return np.asarray(values, dtype=np.float64)

    def compute_rewards_and_done(self, game_state, done):
        potential = self._potential(game_state)
        self.current_alpha = self.shaping_alpha()
        if not self.initialized or game_state.turn == 0:
            delta = np.zeros_like(potential)
            self.initialized = True
        elif done:
            # A zero terminal potential preserves the potential-shaping contract.
            delta = -self.previous
        else:
            delta = self.discounting * potential - self.previous
        delta -= delta.mean()
        rewards = np.clip(
            self.current_alpha * delta,
            -self.max_step_shaping,
            self.max_step_shaping,
        )
        if done:
            terminal = [int(GameResultReward.compute_player_reward(p)) for p in game_state.players]
            rewards += (rankdata(terminal) - 1.0) * 2.0 - 1.0
            self._increment_game_count()
            self.previous[:] = 0.0
            self.initialized = False
        else:
            self.previous = potential
        return tuple(np.clip(rewards, -1.0, 1.0)), done


class StrategicPotentialRewardV3(StrategicPotentialRewardV2):
    """Phase-aware zero-sum shaping for survival, delivery, and safe expansion.
    Includes final tile count difference scaling for finer-grained terminal signals.
    """

    def __init__(
        self,
        tile_diff_scale: float = 8.0,
        tile_diff_weight: float = 0.3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.tile_diff_scale = float(tile_diff_scale)
        self.tile_diff_weight = float(tile_diff_weight)

    def compute_rewards_and_done(self, game_state, done):
        rewards, done = super().compute_rewards_and_done(game_state, done)
        if done:
            player_tiles = [p.city_tile_count for p in game_state.players]
            tile_diff = player_tiles[0] - player_tiles[1]
            tile_bonus = np.tanh(tile_diff / self.tile_diff_scale) * self.tile_diff_weight
            
            rewards_list = list(rewards)
            rewards_list[0] += tile_bonus
            rewards_list[1] -= tile_bonus
            rewards = tuple(np.clip(rewards_list, -1.0, 1.0))
            
        return rewards, done

import numpy as np
from scipy.stats import rankdata


class RelativeDifferencePotentialReward(BaseRewardSpace):
    """
    相対状態に基づく Potential-Based Reward Shaping (PBRS)

    Φ(s) =
        w_city     * D_city
      + w_unit     * D_unit
      + w_research * D_research
      + w_fuel     * D_fuel_safety

    shaping:
        F_t = gamma * Φ(s_{t+1}) - Φ(s_t)

    final reward:
        r_t = r_original + shaping_weight * F_t

    ※ gamma は IMPALA の discount と必ず一致させる。
    """

    DAY_LENGTH = 30
    NIGHT_LENGTH = 10
    CYCLE_LENGTH = 40
    MAX_TURNS = 360

    def __init__(
        self,
        city_weight: float = 1.0,
        unit_weight: float = 0.2,
        research_weight: float = 0.05,
        fuel_weight: float = 0.1,
        shaping_weight: float = 0.05,
        discounting: float = 0.999,

        # 各特徴量を概ね同程度のスケールにする
        city_scale: float = 10.0,
        unit_scale: float = 10.0,
        research_scale: float = 200.0,

        **kwargs,
    ):
        super().__init__(**kwargs)

        self.city_weight = float(city_weight)
        self.unit_weight = float(unit_weight)
        self.research_weight = float(research_weight)
        self.fuel_weight = float(fuel_weight)

        self.shaping_weight = float(shaping_weight)
        self.discounting = float(discounting)

        self.city_scale = float(city_scale)
        self.unit_scale = float(unit_scale)
        self.research_scale = float(research_scale)

        self.previous_potential = np.zeros(2, dtype=np.float64)
        self.initialized = False

    @staticmethod
    def get_reward_spec() -> RewardSpec:
        # clipしないので ±1 より少し広くする
        return RewardSpec(-2.0, 2.0, True, False)

    # ---------------------------------------------------------
    # Fuel safety
    # ---------------------------------------------------------

    def _night_turns_to_survive(self, turn: int) -> int:
        """
        現在の状態から、次に考慮すべき夜の残りターン数。

        昼:
            次の夜10ターンを生き残れるか評価

        夜:
            現在の夜の残りターンを評価
        """

        cycle_turn = turn % self.CYCLE_LENGTH

        if cycle_turn < self.DAY_LENGTH:
            # 昼 → 次の夜全体を考える
            night_turns = self.NIGHT_LENGTH
        else:
            # 夜 → 現在の夜の残り
            night_turns = self.CYCLE_LENGTH - cycle_turn

        # ゲーム終了を超えない
        remaining_game_turns = max(0, self.MAX_TURNS - turn)

        return min(night_turns, remaining_game_turns)

    def _fuel_safety(self, player, turn: int) -> float:
        """
        Player全体のCity Fuel Safetyを 0〜1 で返す。

        各Cityについて

            safety = min(
                fuel / (upkeep * night_turns),
                1.0
            )

        とし、City Tile数で重み付き平均する。

        0.0:
            次の夜をほぼ生き残れない

        1.0:
            全Cityが次の夜を生存可能

        fuelを必要以上に貯めても1.0以上にはしない。
        """

        if len(player.cities) == 0:
            return 0.0

        night_turns = self._night_turns_to_survive(turn)

        if night_turns <= 0:
            return 1.0

        weighted_safety = 0.0
        total_city_tiles = 0

        for city in player.cities.values():

            # LuxのCity adjacencyを考慮した実際のupkeep
            upkeep = float(city.get_light_upkeep())

            # Frameworkにより citytiles / city_tiles の可能性があるので
            # 実装に合わせてここは調整
            if hasattr(city, "citytiles"):
                tile_count = len(city.citytiles)
            elif hasattr(city, "city_tiles"):
                tile_count = len(city.city_tiles)
            else:
                tile_count = 1

            required_fuel = upkeep * night_turns

            if required_fuel <= 0:
                safety = 1.0
            else:
                safety = np.clip(
                    float(city.fuel) / required_fuel,
                    0.0,
                    1.0,
                )

            weighted_safety += safety * tile_count
            total_city_tiles += tile_count

        if total_city_tiles == 0:
            return 0.0

        return weighted_safety / total_city_tiles

    # ---------------------------------------------------------
    # Potential
    # ---------------------------------------------------------

    def _potential(self, game_state) -> np.ndarray:

        players = game_state.players

        cities = np.array(
            [p.city_tile_count for p in players],
            dtype=np.float64,
        )

        units = np.array(
            [len(p.units) for p in players],
            dtype=np.float64,
        )

        research = np.array(
            [p.research_points for p in players],
            dtype=np.float64,
        )

        fuel_safety = np.array(
            [
                self._fuel_safety(p, game_state.turn)
                for p in players
            ],
            dtype=np.float64,
        )

        # -----------------------------------------------------
        # 正規化
        # -----------------------------------------------------

        cities /= self.city_scale
        units /= self.unit_scale
        research /= self.research_scale

        # fuel_safety は既に 0〜1
        # -----------------------------------------------------

        # 2人対戦なので
        # player0: self0 - enemy1
        # player1: self1 - enemy0

        d_city = cities - cities[::-1]
        d_unit = units - units[::-1]
        d_research = research - research[::-1]
        d_fuel = fuel_safety - fuel_safety[::-1]

        phi = (
            self.city_weight * d_city
            + self.unit_weight * d_unit
            + self.research_weight * d_research
            + self.fuel_weight * d_fuel
        )

        return phi

    # ---------------------------------------------------------
    # Episode reset
    # ---------------------------------------------------------

    def reset(self, initial_game_state=None):
        """
        可能なら環境reset直後、最初のactionを実行する前に

            reward_space.reset(game_state)

        と呼ぶ。

        これにより s0 -> s1 のPBRSも失わない。
        """

        self.previous_potential[:] = 0.0
        self.initialized = False

        if initial_game_state is not None:
            self.previous_potential = self._potential(initial_game_state)
            self.initialized = True

    # ---------------------------------------------------------
    # Reward
    # ---------------------------------------------------------

    def compute_rewards_and_done(self, game_state, done):

        potential = self._potential(game_state)

        # -----------------------------------------------------
        # 初期状態
        # -----------------------------------------------------

        if not self.initialized:

            # 本来は reset(initial_state) で初期化するのが望ましい。
            # それができない場合のfallback。
            self.previous_potential = potential.copy()
            self.initialized = True

            shaping = np.zeros(2, dtype=np.float64)

        # -----------------------------------------------------
        # Terminal
        # -----------------------------------------------------

        elif done:

            # Φ(terminal) = 0 と定義
            #
            # F_t =
            # gamma * Φ(terminal) - Φ(previous)
            # = -Φ(previous)

            shaping = -self.previous_potential

        # -----------------------------------------------------
        # Normal transition
        # -----------------------------------------------------

        else:

            shaping = (
                self.discounting * potential
                - self.previous_potential
            )

        rewards = self.shaping_weight * shaping

        # -----------------------------------------------------
        # 勝敗報酬
        # -----------------------------------------------------

        if done:

            terminal = np.array(
                [
                    int(GameResultReward.compute_player_reward(p))
                    for p in game_state.players
                ],
                dtype=np.float64,
            )

            # win  -> +1
            # lose -> -1
            # draw -> 0

            terminal_reward = (
                (rankdata(terminal) - 1.0) * 2.0 - 1.0
            )

            rewards += terminal_reward

            # 次episode用
            self.previous_potential[:] = 0.0
            self.initialized = False

        else:
            self.previous_potential = potential.copy()

        # PBRSの理論的性質を維持するためclipしない
        return tuple(rewards), done
