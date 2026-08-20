"""衝突ペナルティラッパーのユニットテスト。"""

import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock

from lux_ai.strategic_rl.collision_penalty import CollisionPenaltyWrapper
from lux_ai.lux_gym.reward_spaces import RewardSpec


class FakeRewardSpace:
    """テスト用のダミーreward space。常に固定のリワードを返す。"""

    def __init__(self, reward=(0.0, 0.0)):
        self._reward = reward
        self._game_counter = None

    @staticmethod
    def get_reward_spec():
        return RewardSpec(-1.0, 1.0, True, False)

    def compute_rewards_and_done(self, game_state, done):
        return self._reward, done

    def get_info(self):
        return {}

    def set_global_game_counter(self, counter):
        self._game_counter = counter


def _make_game_state(turn: int, player_units: tuple[list[str], list[str]]):
    """テスト用のダミーgame_stateを作成。

    player_units: 各プレイヤーのユニットIDリスト。
    """
    game_state = MagicMock()
    game_state.turn = turn
    players = []
    for unit_ids in player_units:
        player = MagicMock()
        units = []
        for uid in unit_ids:
            unit = MagicMock()
            unit.id = uid
            units.append(unit)
        player.units = units
        players.append(player)
    game_state.players = players
    return game_state


class TestCollisionPenaltyWrapper:
    """CollisionPenaltyWrapperの動作を検証するテスト群。"""

    def test_no_penalty_on_first_step(self):
        """初回ステップではペナルティが発生しないこと。"""
        inner = FakeRewardSpace(reward=(0.5, -0.5))
        wrapper = CollisionPenaltyWrapper(
            inner=inner, penalty_start=0.1, penalty_end=0.1, ramp_steps=100
        )
        gs = _make_game_state(turn=0, player_units=(["u1", "u2"], ["u3"]))
        rewards, done = wrapper.compute_rewards_and_done(gs, False)
        assert rewards == (0.5, -0.5), "初回ステップではリワード変更なし"

    def test_no_penalty_when_no_units_vanished(self):
        """ユニットが消滅していなければペナルティなし。"""
        inner = FakeRewardSpace(reward=(0.0, 0.0))
        wrapper = CollisionPenaltyWrapper(
            inner=inner, penalty_start=0.1, penalty_end=0.1, ramp_steps=100
        )
        # ステップ0: ユニット登録
        gs0 = _make_game_state(turn=0, player_units=(["u1", "u2"], ["u3"]))
        wrapper.compute_rewards_and_done(gs0, False)
        # ステップ1: ユニットそのまま
        gs1 = _make_game_state(turn=1, player_units=(["u1", "u2"], ["u3"]))
        rewards, done = wrapper.compute_rewards_and_done(gs1, False)
        assert rewards == (0.0, 0.0), "消滅なしならペナルティなし"

    def test_penalty_applied_when_unit_vanishes(self):
        """ユニットが消滅した場合にペナルティが加算されること。"""
        inner = FakeRewardSpace(reward=(0.0, 0.0))
        wrapper = CollisionPenaltyWrapper(
            inner=inner, penalty_start=0.1, penalty_end=0.1, ramp_steps=100
        )
        gs0 = _make_game_state(turn=0, player_units=(["u1", "u2"], ["u3"]))
        wrapper.compute_rewards_and_done(gs0, False)
        # u2が消滅
        gs1 = _make_game_state(turn=1, player_units=(["u1"], ["u3"]))
        rewards, done = wrapper.compute_rewards_and_done(gs1, False)
        # player0に-0.1のペナルティ、player1は変更なし
        assert rewards[0] == pytest.approx(-0.1, abs=1e-6)
        assert rewards[1] == pytest.approx(0.0, abs=1e-6)

    def test_penalty_proportional_to_vanished_count(self):
        """ペナルティが消滅ユニット数に比例すること。"""
        inner = FakeRewardSpace(reward=(0.0, 0.0))
        wrapper = CollisionPenaltyWrapper(
            inner=inner, penalty_start=0.05, penalty_end=0.05, ramp_steps=100
        )
        gs0 = _make_game_state(turn=0, player_units=(["u1", "u2", "u3"], ["u4"]))
        wrapper.compute_rewards_and_done(gs0, False)
        # u1, u2, u3 のうち u1, u2 が消滅（2ユニット）
        gs1 = _make_game_state(turn=1, player_units=(["u3"], ["u4"]))
        rewards, done = wrapper.compute_rewards_and_done(gs1, False)
        assert rewards[0] == pytest.approx(-0.1, abs=1e-6)

    def test_successful_city_build_is_not_penalized(self):
        inner = FakeRewardSpace(reward=(0.0, 0.0))
        wrapper = CollisionPenaltyWrapper(
            inner=inner, penalty_start=0.1, penalty_end=0.1, ramp_steps=100
        )
        worker = SimpleNamespace(
            id="builder",
            pos=SimpleNamespace(x=2, y=3),
            is_worker=lambda: True,
        )
        player0 = SimpleNamespace(units=[worker], city_tiles=[])
        player1 = SimpleNamespace(units=[], city_tiles=[])
        wrapper.compute_rewards_and_done(
            SimpleNamespace(turn=0, players=[player0, player1]), False
        )

        new_tile = SimpleNamespace(pos=SimpleNamespace(x=2, y=3))
        player0_after = SimpleNamespace(units=[], city_tiles=[new_tile])
        rewards, _ = wrapper.compute_rewards_and_done(
            SimpleNamespace(turn=1, players=[player0_after, player1]), False
        )

        assert rewards == (0.0, 0.0)

    def test_penalty_clipped_to_reward_bounds(self):
        """ペナルティ加算後のリワードが[-1, 1]にクリップされること。"""
        inner = FakeRewardSpace(reward=(-0.95, 0.0))
        wrapper = CollisionPenaltyWrapper(
            inner=inner, penalty_start=0.5, penalty_end=0.5, ramp_steps=100
        )
        gs0 = _make_game_state(turn=0, player_units=(["u1", "u2"], ["u3"]))
        wrapper.compute_rewards_and_done(gs0, False)
        gs1 = _make_game_state(turn=1, player_units=([], ["u3"]))
        rewards, done = wrapper.compute_rewards_and_done(gs1, False)
        # -0.95 - 0.5*2 = -1.95 → クリップで -1.0
        assert rewards[0] == pytest.approx(-1.0, abs=1e-6)

    def test_ramp_schedule(self):
        """ランプスケジュールでペナルティが段階的に増加すること。"""
        inner = FakeRewardSpace(reward=(0.0, 0.0))
        wrapper = CollisionPenaltyWrapper(
            inner=inner, penalty_start=0.01, penalty_end=0.05, ramp_steps=4
        )
        # 初期ペナルティは0.01
        assert wrapper._current_penalty() == pytest.approx(0.01, abs=1e-6)

        # 2ステップ進める（50%進行 → 0.01 + 0.5*(0.05-0.01) = 0.03）
        for _ in range(2):
            gs = _make_game_state(turn=1, player_units=(["u1"], ["u2"]))
            wrapper.compute_rewards_and_done(gs, False)
        assert wrapper._current_penalty() == pytest.approx(0.03, abs=1e-6)

        # さらに2ステップ（100%進行 → 0.05）
        for _ in range(2):
            gs = _make_game_state(turn=1, player_units=(["u1"], ["u2"]))
            wrapper.compute_rewards_and_done(gs, False)
        assert wrapper._current_penalty() == pytest.approx(0.05, abs=1e-6)

    def test_no_penalty_on_done(self):
        """ゲーム終了時はペナルティを適用しないこと。"""
        inner = FakeRewardSpace(reward=(0.0, 0.0))
        wrapper = CollisionPenaltyWrapper(
            inner=inner, penalty_start=0.5, penalty_end=0.5, ramp_steps=100
        )
        gs0 = _make_game_state(turn=0, player_units=(["u1", "u2"], ["u3"]))
        wrapper.compute_rewards_and_done(gs0, False)
        # ゲーム終了で全ユニット消滅
        gs1 = _make_game_state(turn=360, player_units=([], []))
        rewards, done = wrapper.compute_rewards_and_done(gs1, True)
        assert rewards == (0.0, 0.0), "done=Trueではペナルティ適用しない"

    def test_state_reset_on_done(self):
        """done後にprevious_unit_idsがリセットされること。"""
        inner = FakeRewardSpace(reward=(0.0, 0.0))
        wrapper = CollisionPenaltyWrapper(
            inner=inner, penalty_start=0.1, penalty_end=0.1, ramp_steps=100
        )
        gs0 = _make_game_state(turn=0, player_units=(["u1"], ["u2"]))
        wrapper.compute_rewards_and_done(gs0, False)
        gs_done = _make_game_state(turn=360, player_units=([], []))
        wrapper.compute_rewards_and_done(gs_done, True)
        # 新エピソード開始: previous_unit_idsが空なのでペナルティなし
        gs_new = _make_game_state(turn=0, player_units=(["u10"], ["u20"]))
        rewards, done = wrapper.compute_rewards_and_done(gs_new, False)
        assert rewards == (0.0, 0.0)

    def test_get_info_includes_penalty(self):
        """get_infoにcollision_penaltyが含まれること。"""
        inner = FakeRewardSpace(reward=(0.0, 0.0))
        wrapper = CollisionPenaltyWrapper(
            inner=inner, penalty_start=0.01, penalty_end=0.05, ramp_steps=100
        )
        info = wrapper.get_info()
        assert "LOGGING_collision_penalty" in info
        assert "LOGGING_unit_loss_penalty" in info
        assert info["LOGGING_collision_penalty"][0] == pytest.approx(0.01, abs=1e-6)

    def test_global_game_counter_passthrough(self):
        """set_global_game_counterが内部reward spaceに透過されること。"""
        inner = FakeRewardSpace()
        wrapper = CollisionPenaltyWrapper(
            inner=inner, penalty_start=0.01, penalty_end=0.05, ramp_steps=100
        )
        counter = MagicMock()
        wrapper.set_global_game_counter(counter)
        assert inner._game_counter is counter


class TestCreateRewardSpaceWithCollisionPenalty:
    """create_reward_spaceの衝突ペナルティ統合を検証。"""

    def test_collision_penalty_disabled_by_default(self):
        """collision_penalty_cost=0でラッパーが適用されないこと。"""
        from lux_ai.lux_gym import create_reward_space
        from lux_ai.strategic_rl.collision_penalty import CollisionPenaltyWrapper

        flags = _minimal_flags(collision_penalty_cost=0.0)
        reward_space = create_reward_space(flags)
        assert not isinstance(reward_space, CollisionPenaltyWrapper)

    def test_collision_penalty_enabled(self):
        """collision_penalty_cost>0でラッパーが適用されること。"""
        from lux_ai.lux_gym import create_reward_space
        from lux_ai.strategic_rl.collision_penalty import CollisionPenaltyWrapper

        flags = _minimal_flags(collision_penalty_cost=0.01)
        reward_space = create_reward_space(flags)
        assert isinstance(reward_space, CollisionPenaltyWrapper)


def _minimal_flags(**overrides):
    """テスト用の最小限のflagsを作成。"""
    from lux_ai.strategic_rl.reward import StrategicPotentialRewardV2

    defaults = {
        "reward_space": StrategicPotentialRewardV2,
        "reward_space_kwargs": {},
        "collision_penalty_cost": 0.0,
        "collision_penalty_ramp_start": 0.005,
        "collision_penalty_ramp_end": 0.02,
        "collision_penalty_ramp_steps": 50000,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)
