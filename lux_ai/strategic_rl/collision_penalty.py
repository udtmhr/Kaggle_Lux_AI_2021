"""ユニット消滅を衝突リスクの proxy とするリワードラッパー。

ステップ前後のユニットID集合を比較し、ステップ後に消滅したユニットを
味方衝突によるものと推定してペナルティをリワードに加算する。

原因を完全には復元できないため、夜間燃料切れなども対象になる。一方、正常な
都市建設とゲーム終了時の消滅は除外する。
"""

from __future__ import annotations

import numpy as np

from ..lux.game import Game
from ..lux_gym.reward_spaces import BaseRewardSpace


class CollisionPenaltyWrapper(BaseRewardSpace):
    """元のreward spaceに味方衝突ペナルティを加算するラッパー。

    味方同士の衝突は、都市タイル上以外で同じマスに移動した場合に発生し、
    衝突した全ユニットが消滅する。ステップ前後のユニットID差分で検出。

    衝突ペナルティは段階的に増加するランプスケジュールを使用する。
    """

    def __init__(
        self,
        inner: BaseRewardSpace,
        penalty_start: float = 0.005,
        penalty_end: float = 0.02,
        ramp_steps: int = 50_000,
    ):
        self._inner = inner
        self._penalty_start = float(penalty_start)
        self._penalty_end = float(penalty_end)
        self._ramp_steps = max(int(ramp_steps), 1)
        self._step_count = 0
        # Unit id -> (x, y, is_worker), grouped by player.
        self._previous_units: tuple[dict[str, tuple[int, int, bool]], dict[str, tuple[int, int, bool]]] = ({}, {})

    def get_reward_spec(self):
        return self._inner.get_reward_spec()

    def _current_penalty(self) -> float:
        """ランプスケジュールに基づく現在のペナルティ係数。"""
        progress = min(self._step_count / self._ramp_steps, 1.0)
        return self._penalty_start + progress * (self._penalty_end - self._penalty_start)

    @staticmethod
    def _snapshot_units(game_state: Game) -> tuple[dict[str, tuple[int, int, bool]], dict[str, tuple[int, int, bool]]]:
        return tuple(
            {
                unit.id: (unit.pos.x, unit.pos.y, bool(unit.is_worker()))
                for unit in player.units
            }
            for player in game_state.players
        )

    def compute_rewards_and_done(self, game_state: Game, done: bool):
        """内部reward spaceのリワードに衝突ペナルティを加算。"""
        rewards, done = self._inner.compute_rewards_and_done(game_state, done)

        self._step_count += 1
        current_units = self._snapshot_units(game_state)

        # ゲーム開始時やリセット直後はペナルティ計算不要
        if game_state.turn <= 0 or not any(self._previous_units):
            self._previous_units = current_units
            return rewards, done

        penalty = self._current_penalty()
        rewards_array = np.array(rewards, dtype=np.float64)

        for player_idx in range(2):
            vanished_ids = self._previous_units[player_idx].keys() - current_units[player_idx].keys()
            friendly_city_positions = {
                (tile.pos.x, tile.pos.y)
                for tile in game_state.players[player_idx].city_tiles
            }
            # A worker disappears at its old square when BUILD_CITY succeeds;
            # that is desired expansion, not a collision or survival failure.
            city_build_ids = {
                unit_id
                for unit_id in vanished_ids
                if self._previous_units[player_idx][unit_id][2]
                and self._previous_units[player_idx][unit_id][:2] in friendly_city_positions
            }
            vanished_count = len(vanished_ids - city_build_ids)
            if vanished_count > 0 and not done:
                # 消滅原因（衝突 vs 夜間燃料切れ）の完全分離は困難だが、
                # 衝突回避のインセンティブとしてはユニット消滅全般への
                # ペナルティで十分に機能する。夜間燃料切れも回避すべき事象。
                rewards_array[player_idx] -= penalty * vanished_count

        reward_spec = self._inner.get_reward_spec()
        rewards_array = np.clip(rewards_array, reward_spec.reward_min, reward_spec.reward_max)

        # 次ステップのためにユニットIDを記録
        self._previous_units = current_units

        if done:
            self._previous_units = ({}, {})

        return tuple(rewards_array), done

    def get_info(self):
        """内部reward spaceのinfoに衝突ペナルティ情報を追加。"""
        info = self._inner.get_info()
        info["LOGGING_collision_penalty"] = np.asarray(
            [self._current_penalty()], dtype=np.float32
        )
        info["LOGGING_unit_loss_penalty"] = info["LOGGING_collision_penalty"].copy()
        return info

    def set_global_game_counter(self, counter) -> None:
        """内部reward spaceのゲームカウンタを透過的に設定。"""
        if hasattr(self._inner, "set_global_game_counter"):
            self._inner.set_global_game_counter(counter)
