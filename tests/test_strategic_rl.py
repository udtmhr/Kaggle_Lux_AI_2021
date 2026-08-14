import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from lux_ai.lux_gym import create_env
from lux_ai.lux_gym.act_spaces import ACTION_MEANINGS
from lux_ai.lux_gym.wrappers import VecEnv
from lux_ai.nns import create_model
from lux_ai.rl_agent.rl_agent import RLAgent, checkpoint_path, model_directory
from lux_ai.strategic_rl.artifacts import atomic_torch_save
from lux_ai.strategic_rl.evaluate import summarize
from lux_ai.strategic_rl.evaluate_checkpoint import _opponent_model_files, _parity_passed, evaluate_checkpoint
from lux_ai.strategic_rl.league import (
    LeagueSampler,
    Opponent,
    PFSPSampler,
    learner_player_mask,
    merge_player_actions,
    merge_player_actions_inplace,
    opponents_from_config,
    rule_based_guidance,
)
from lux_ai.strategic_rl.models import SurvivalStrategicBackbone
from lux_ai.strategic_rl.behavior_kl import BehaviorKLController, masked_normalized_entropy, masked_policy_kl
from lux_ai.strategic_rl.categorical_value import categorical_value, hl_gauss_encode, support_outside_fraction
from lux_ai.strategic_rl.curriculum import SnapshotPool, turn_band
from lux_ai.strategic_rl.obs import night_turns_between
from lux_ai.strategic_rl.prepare_data import _discover_replays
from lux_ai.strategic_rl.prepare_eval_agent import checkpoint_label, prepare_eval_agent, sha256_file
from lux_ai.strategic_rl.resume import merge_resume_config
from lux_ai.strategic_rl.reward import StrategicPotentialRewardV2, SurvivalPotentialReward
from lux_ai.strategic_rl.run_matches import candidate_last_response_turn, replay_metrics, run_matched_matches
from lux_ai.strategic_rl.schedules import LinearSchedule, teacher_kl_coefficient
from lux_ai.strategic_rl.train_distill import ShardDataset, _compact_collate
from lux_ai.strategic_rl.train_eval import (
    checkpoint_model_max_abs_diff,
    full_checkpoint,
    load_reused_baseline_evaluation,
    promotion_decision,
    run_training_segment,
)
from lux_ai.strategic_rl.tta import (
    ROT180_ACTION_INDICES,
    rot180_ensemble_outputs,
    rotate_compact_distillation_batch_180,
    rotate_model_input_180,
    rotate_observations_180,
    rotate_policy_180,
)
from lux_ai.torchbeast.monobeast import (
    compute_baseline_loss,
    compute_teacher_kl_loss,
    configure_trainable_parameters,
    model_state_dict_cpu,
    state_dict_max_abs_diff,
    sync_actor_model,
    trajectory_weighted_mean,
)
from lux_ai.utils import flags_to_namespace


def test_behavior_policy_kl_is_finite_with_masked_actions():
    behavior = torch.tensor([[[[[[[0.0, float("-inf"), 1.0]]]]]]])
    learner = torch.tensor([[[[[[[0.5, float("-inf"), 0.0]]]]]]])
    active = torch.ones(behavior.shape[:-1], dtype=torch.bool)
    forward, counts = masked_policy_kl(learner, behavior, active, reverse=False)
    reverse, _ = masked_policy_kl(learner, behavior, active, reverse=True)
    entropy, entropy_count = masked_normalized_entropy(learner, active)
    assert torch.isfinite(forward).all() and torch.isfinite(reverse).all()
    assert counts.item() == entropy_count.item() == 1
    assert 0 <= (entropy / entropy_count).item() <= 1
    learner.requires_grad_(True)
    backward_kl, _ = masked_policy_kl(learner, behavior, active, reverse=True)
    backward_kl.sum().backward()
    assert torch.isfinite(learner.grad).all()


def test_behavior_kl_controller_calibration_bounds_abort_and_resume():
    controller = BehaviorKLController(auto_steps=2)
    controller.observe(0, 0.001, 0.8)
    controller.observe(2, 0.2, 0.8)
    assert controller.target == pytest.approx(0.05)
    assert 1e-4 <= controller.beta <= 1.0
    restored = BehaviorKLController()
    restored.load_state_dict(controller.state_dict())
    assert restored.state_dict() == controller.state_dict()
    restored.ema_kl = 1.0
    with pytest.raises(RuntimeError, match="Behavior KL abort"):
        for _ in range(5):
            restored.observe(3, 1.0, 0.8)


def test_hl_gauss_encode_decode_zero_sum_and_support_detection():
    targets = torch.tensor([-2.0, -0.5, 0.0, 1.25, 2.0])
    labels = hl_gauss_encode(targets)
    assert torch.allclose(labels.sum(dim=-1), torch.ones_like(targets), atol=1e-6)
    decoded = categorical_value(labels.clamp_min(1e-12).log())
    assert torch.allclose(decoded, targets, atol=0.08)
    assert support_outside_fraction(torch.tensor([-2.1, 0.0, 2.1]), -2.0, 2.0).item() == pytest.approx(2 / 3)


def test_snapshot_pool_strata_ratio_and_td_error_priority():
    pool = SnapshotPool(capacity=20, td_error_ema_decay=0.0)
    low_id = pool.add({"turn": 20, "map_size": 12, "opponent": "a"}, priority=1.0)
    high_id = pool.add({"turn": 30, "map_size": 12, "opponent": "a"}, priority=1.0)
    pool.update_priority(high_id, 100.0)
    rng = np.random.default_rng(7)
    samples = [pool.sample(rng, prioritized_probability=1.0).snapshot_id for _ in range(200)]
    assert samples.count(high_id) > 180
    assert turn_band(0) == 0 and turn_band(359) == 4
    rng = np.random.default_rng(9)
    snapshot_starts = sum(pool.choose_episode_start(rng, 0.3) is not None for _ in range(5000))
    assert snapshot_starts / 5000 == pytest.approx(0.3, abs=0.03)
    assert low_id != high_id


def test_night_turn_count_boundaries():
    assert night_turns_between(0, 30) == 0
    assert night_turns_between(0, 40) == 10
    assert night_turns_between(35, 45) == 5
    assert night_turns_between(350, 360) == 10


def test_cli_observation_player_attribute_survives_second_turn():
    class CliObservation(dict):
        player = 1

    player = SimpleNamespace(units=[], city_tiles=[])
    agent = object.__new__(RLAgent)
    game_state = SimpleNamespace(players=[player, player], turn=0, id=None)
    agent.env = SimpleNamespace(
        unwrapped=[SimpleNamespace(manual_step=lambda updates: None, game_state=game_state)]
    )
    agent.my_city_tile_mat = np.zeros((32, 32), dtype=np.bool_)
    agent.data_augmentations = []
    observation = CliObservation(step=1, updates=[], remainingOverageTime=60.0)

    agent.preprocess(observation, None)

    assert agent.game_state.turn == 1
    assert agent.game_state.id == 1


def test_named_inference_model_directory(monkeypatch, tmp_path: Path):
    model_dir = tmp_path / "bundle" / "models" / "candidate"
    model_dir.mkdir(parents=True)
    checkpoint = model_dir / "model.pt"
    checkpoint.touch()
    monkeypatch.setenv("LUX_AGENT_MODEL_DIR", str(model_dir))

    assert model_directory() == model_dir.resolve()
    assert checkpoint_path() == checkpoint


def test_prepare_eval_agent_builds_isolated_single_checkpoint(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    checkpoint = run_dir / "001024_weights.pt"
    torch.save({"model_state_dict": {}}, checkpoint)
    config = run_dir / "config.yaml"
    config.write_text("model_arch: survival_strategic\n", encoding="utf-8")
    output_dir = tmp_path / "lux_candidate_001024"

    result = prepare_eval_agent(checkpoint, output_dir, validate_model=False)

    bundled = output_dir / "lux_ai" / "rl_agent" / checkpoint.name
    assert checkpoint_label(checkpoint) == "001024"
    assert Path(result["agent"]) == output_dir / "main.py"
    assert bundled.is_file()
    assert sha256_file(bundled) == sha256_file(checkpoint)
    assert list((output_dir / "lux_ai" / "rl_agent").glob("*.pt")) == [bundled]
    assert (output_dir / "lux_ai" / "rl_agent" / "config.yaml").read_text(encoding="utf-8") == config.read_text(
        encoding="utf-8"
    )


def test_candidate_response_turn_detects_turn_zero_agent_exit():
    crashed = {"allCommands": [[{"agentID": 0, "command": "m u_1 e"}], [{"agentID": 1, "command": "m u_2 w"}]]}
    healthy = {
        "allCommands": [
            [{"agentID": 0, "command": "m u_1 e"}],
            [{"agentID": 0, "command": "dst alive"}],
        ]
    }
    assert candidate_last_response_turn(crashed, 0) == 0
    assert candidate_last_response_turn(healthy, 0) == 1


def test_evaluate_checkpoint_builds_runs_and_writes_report(monkeypatch, tmp_path: Path):
    checkpoint = tmp_path / "001024_weights.pt"
    checkpoint.touch()
    opponent = tmp_path / "opponent.py"
    opponent.touch()
    candidate = tmp_path / "agent" / "main.py"
    candidate.parent.mkdir()
    candidate.touch()
    output_dir = tmp_path / "evaluation"

    monkeypatch.setattr(
        "lux_ai.strategic_rl.evaluate_checkpoint.prepare_eval_agent",
        lambda *args, **kwargs: {"agent": str(candidate)},
    )

    def fake_run_matches(*args, **kwargs):
        output_dir.mkdir()
        games = output_dir / "games.jsonl"
        records = [
            {"opponent": "teacher", "seed": 2021, "map_size": 12, "candidate_player": 0, "winner": 0},
            {"opponent": "teacher", "seed": 2021, "map_size": 12, "candidate_player": 1, "winner": 1},
        ]
        games.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
        return games

    monkeypatch.setattr("lux_ai.strategic_rl.evaluate_checkpoint.run_matched_matches", fake_run_matches)

    result = evaluate_checkpoint(checkpoint, opponent, output_dir, opponent_name="teacher")

    assert result["summary"]["opponents"]["teacher"]["score_rate"] == 1.0
    assert result["backend"]["selected"] == "official"
    assert (output_dir / "report.json").is_file()


def test_evaluate_checkpoint_auto_selects_batched_backend_after_parity(monkeypatch, tmp_path: Path):
    checkpoint = tmp_path / "candidate" / "001024_weights.pt"
    checkpoint.parent.mkdir()
    checkpoint.touch()
    config = checkpoint.parent / "config.yaml"
    config.touch()
    opponent = tmp_path / "opponent" / "main.py"
    opponent_model_dir = opponent.parent / "lux_ai" / "rl_agent"
    opponent_model_dir.mkdir(parents=True)
    opponent.touch()
    (opponent_model_dir / "model.pt").touch()
    (opponent_model_dir / "config.yaml").touch()
    candidate = tmp_path / "agent" / "main.py"
    candidate.parent.mkdir()
    candidate.touch()

    monkeypatch.setattr(
        "lux_ai.strategic_rl.evaluate_checkpoint.prepare_eval_agent",
        lambda *args, **kwargs: {"agent": str(candidate)},
    )
    records = [
        {"opponent": "teacher", "seed": 2021, "map_size": 12, "candidate_player": player, "winner": player}
        for player in (0, 1)
    ]

    def write_records(output):
        output.mkdir(parents=True, exist_ok=True)
        games = output / "games.jsonl"
        games.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
        return games

    monkeypatch.setattr(
        "lux_ai.strategic_rl.evaluate_checkpoint.run_matched_matches",
        lambda *args, **kwargs: write_records(args[2]),
    )
    monkeypatch.setattr(
        "lux_ai.strategic_rl.evaluate_checkpoint._run_batched_matches",
        lambda *args, **kwargs: write_records(args[4]),
    )
    result = evaluate_checkpoint(
        checkpoint,
        opponent,
        tmp_path / "evaluation",
        config=config,
        opponent_name="teacher",
    )
    assert result["backend"]["parity"] == "passed"
    assert result["backend"]["selected"] == "internal"


def test_run_matched_matches_runs_games_in_parallel(monkeypatch, tmp_path: Path):
    lock = threading.Lock()
    active = 0
    max_active = 0

    def fake_run_match(candidate, opponent, candidate_player, seed, map_size, replay_path, python, timeout, opponent_name):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.02)
        with lock:
            active -= 1
        return {
            "opponent": opponent_name,
            "seed": seed,
            "map_size": map_size,
            "candidate_player": candidate_player,
            "winner": candidate_player,
        }

    monkeypatch.setattr("lux_ai.strategic_rl.run_matches.run_match", fake_run_match)
    result_path = run_matched_matches(
        tmp_path / "candidate.py",
        tmp_path / "opponent.py",
        tmp_path / "evaluation",
        seeds=2,
        map_sizes=(12,),
        opponent_name="teacher",
        workers=2,
    )

    records = [json.loads(line) for line in result_path.read_text(encoding="utf-8").splitlines()]
    assert max_active == 2
    assert len(records) == 4
    assert {(record["seed"], record["candidate_player"]) for record in records} == {
        (2021, 0),
        (2021, 1),
        (2022, 0),
        (2022, 1),
    }


def test_batched_evaluation_discovers_bundle_and_checks_parity(tmp_path: Path):
    opponent = tmp_path / "opponent" / "main.py"
    model_dir = opponent.parent / "lux_ai" / "rl_agent"
    model_dir.mkdir(parents=True)
    opponent.touch()
    checkpoint = model_dir / "model.pt"
    config = model_dir / "config.yaml"
    checkpoint.touch()
    config.touch()
    assert _opponent_model_files(opponent) == (checkpoint, config)

    official = tmp_path / "official.jsonl"
    internal = tmp_path / "internal.jsonl"
    records = [
        {"opponent": "teacher", "seed": 7, "map_size": 12, "candidate_player": player, "winner": player}
        for player in (0, 1)
    ]
    payload = "".join(json.dumps(record) + "\n" for record in records)
    official.write_text(payload, encoding="utf-8")
    internal.write_text(payload, encoding="utf-8")
    assert _parity_passed(official, internal)
    internal.write_text(payload.replace('"winner": 1', '"winner": 0'), encoding="utf-8")
    assert not _parity_passed(official, internal)


def test_backbone_masks_padding_and_is_finite():
    model = SurvivalStrategicBackbone(channels=24, attention_blocks=1, heads=6)
    inputs = torch.randn(2, 24, 32, 32)
    mask = torch.zeros(2, 1, 32, 32)
    mask[:, :, :12, :12] = 1
    output, output_mask = model((inputs, mask))
    assert output.shape == inputs.shape
    assert torch.isfinite(output).all()
    assert torch.equal(output_mask, mask)
    assert torch.count_nonzero(output[:, :, 12:, :]) == 0
    assert torch.count_nonzero(output[:, :, :, 12:]) == 0


def test_linear_schedule_and_teacher_floor():
    schedule = LinearSchedule(1.0, 0.0, duration=10, delay=5)
    assert schedule(0) == 1.0
    assert schedule(10) == 0.5
    assert schedule(20) == 0.0
    opponents = [Opponent("teacher", "teacher.pt", "teacher"), Opponent("latest", "latest.pt")]
    probabilities = PFSPSampler(opponents, teacher_floor=0.2).probabilities({"teacher": 0.99, "latest": 0.1})
    assert np.isclose(probabilities.sum(), 1.0)
    assert probabilities[0] >= 0.2
    disabled = type("Flags", (), {"use_teacher": False, "teacher_kl_cost": 1.0})()
    assert teacher_kl_coefficient(disabled, 0) == 0.0
    anchored = type(
        "Flags",
        (),
        {
            "use_teacher": True,
            "teacher_kl_cost": 0.01,
            "teacher_kl_cost_start": 0.01,
            "teacher_kl_cost_end": 0.001,
            "teacher_kl_cost_floor": 0.005,
            "teacher_kl_decay_steps": 100,
            "teacher_kl_delay_steps": 0,
        },
    )()
    assert teacher_kl_coefficient(anchored, 100) == 0.005


def test_league_config_sampling_and_player_action_merge():
    opponents = opponents_from_config(
        [
            {"name": "self", "kind": "selfplay", "weight": 1},
            {"name": "rule", "kind": "rule_based", "weight": 3},
        ]
    )
    sampler = LeagueSampler(opponents)
    assert np.allclose(sampler.probabilities, [0.25, 0.75])

    learner = {"worker": torch.zeros((2, 1, 2, 1, 1, 1), dtype=torch.long)}
    external = {"worker": torch.full_like(learner["worker"], 7)}
    merged = merge_player_actions(learner, external, env_indices=[1], opponent_players=[0])
    assert merged["worker"][1, 0, 0, 0, 0, 0] == 7
    assert merged["worker"][1, 0, 1, 0, 0, 0] == 0
    assert torch.count_nonzero(learner["worker"]) == 0

    merge_player_actions_inplace(merged, external, env_indices=[0], opponent_players=[1])
    assert merged["worker"][0, 0, 1, 0, 0, 0] == 7

    mask = learner_player_mask(opponents, selected=[0, 1], learner_players=[0, 1], device=torch.device("cpu"))
    assert mask.tolist() == [[True, True], [False, True]]


def test_vec_env_steps_independent_engines_concurrently():
    barrier = threading.Barrier(2)

    class FakeEnv:
        def reset(self):
            return np.zeros(1), (0.0, 0.0), False, {"value": np.zeros(1)}

        def step(self, action):
            barrier.wait(timeout=2)
            value = np.asarray(action["value"])
            return value, (0.0, 0.0), False, {"value": value}

        def close(self):
            return None

    env = VecEnv([FakeEnv(), FakeEnv()])
    try:
        env.reset(force=True)
        observation, _, _, _ = env.step({"value": np.asarray([[1.0], [2.0]])})
    finally:
        env.close()
    assert observation.tolist() == [[1.0], [2.0]]


def test_rot180_ensemble_batches_both_views_in_one_forward():
    class DummyModel:
        def __init__(self):
            self.batch_sizes = []

        def __call__(self, model_input, sample, actions_per_square):
            obs = model_input["obs"]["board"]
            batch_size = obs.shape[0]
            self.batch_sizes.append(batch_size)
            base = obs[:, :1, :1].permute(0, 2, 3, 4, 1)
            logits = torch.cat([base + offset for offset in range(len(ACTION_MEANINGS["worker"]))], dim=-1).unsqueeze(1)
            return {
                "policy_logits": {"worker": logits},
                "baseline": obs.mean(dim=(1, 2, 3, 4)).unsqueeze(-1).expand(batch_size, 2),
            }

    model_input = {
        "obs": {"board": torch.arange(2 * 1 * 1 * 3 * 3, dtype=torch.float32).view(2, 1, 1, 3, 3)},
        "info": {
            "input_mask": torch.ones(2, 1, 3, 3, dtype=torch.bool),
            "available_actions_mask": {
                "worker": torch.ones(2, 1, 2, 3, 3, len(ACTION_MEANINGS["worker"]), dtype=torch.bool)
            },
        },
    }
    reference_model = DummyModel()
    original = reference_model(model_input, sample=False, actions_per_square=1)
    rotated = reference_model(rotate_model_input_180(model_input), sample=False, actions_per_square=1)
    expected_policy = (original["policy_logits"]["worker"] + rotate_policy_180(rotated["policy_logits"])["worker"]) / 2
    expected_baseline = (original["baseline"] + rotated["baseline"]) / 2

    fused_model = DummyModel()
    actual = rot180_ensemble_outputs(fused_model, model_input)

    assert fused_model.batch_sizes == [4]
    assert torch.equal(actual["policy_logits"]["worker"], expected_policy)
    assert torch.equal(actual["baseline"], expected_baseline)


def test_player_perspective_stack_matches_original_indexing():
    inputs = torch.arange(2 * 3 * 2 * 2 * 2).view(2, 3, 2, 2, 2)
    original = inputs[:, :, torch.tensor([[0, 1], [1, 0]]), ...]
    optimized = torch.stack((inputs, inputs.flip(dims=(2,))), dim=2)

    assert torch.equal(optimized, original)


def test_external_opponent_value_loss_masks_opponent_side():
    values = torch.tensor([[[0.0, 10.0]]])
    targets = torch.zeros_like(values)
    player_mask = torch.tensor([[[1.0, 0.0]]])
    loss = compute_baseline_loss(values, targets, reduction="sum", player_mask=player_mask)
    assert loss.item() == 0.0


def test_legacy_resume_inherits_only_missing_league_config():
    selected = OmegaConf.create(
        {
            "use_teacher": False,
            "league_enabled": True,
            "league_config_version": 2,
            "league_opponents": [{"name": "first", "kind": "teacher", "checkpoint": "first.pt"}],
        }
    )
    saved = OmegaConf.create({"use_teacher": True, "step_setting": 7})
    merged = merge_resume_config(saved, selected, OmegaConf.create({"step_setting": 9}))
    assert merged.use_teacher is True
    assert merged.step_setting == 9
    assert merged.league_enabled is True
    assert merged.league_config_version == 2
    assert merged.league_opponents[0].name == "first"

    saved_with_league = OmegaConf.create(
        {"league_enabled": False, "league_config_version": 2, "league_opponents": []}
    )
    merged = merge_resume_config(saved_with_league, selected, OmegaConf.create({}))
    assert merged.league_enabled is False
    assert merged.league_opponents == []


def test_resume_upgrades_an_older_versioned_league_but_cli_wins():
    selected = OmegaConf.create(
        {
            "league_enabled": True,
            "league_config_version": 2,
            "league_opponents": [{"name": "v2", "kind": "selfplay", "weight": 1.0}],
        }
    )
    saved = OmegaConf.create(
        {
            "league_enabled": True,
            "league_config_version": 1,
            "league_opponents": [{"name": "v1", "kind": "selfplay", "weight": 1.0}],
        }
    )
    merged = merge_resume_config(saved, selected, OmegaConf.create({"league_enabled": False}))
    assert merged.league_config_version == 2
    assert merged.league_opponents[0].name == "v2"
    assert merged.league_enabled is False


def test_resume_upgrades_global_reward_decay_config():
    selected = OmegaConf.create(
        {
            "reward_space": "SurvivalPotentialReward",
            "reward_space_kwargs": {"shaping_weight": 0.05, "decay_games": 14000},
            "reward_config_version": 2,
        }
    )
    saved = OmegaConf.create(
        {
            "reward_space": "SurvivalPotentialReward",
            "reward_space_kwargs": {"shaping_weight": 0.05, "decay_games": 2000},
        }
    )
    merged = merge_resume_config(saved, selected, OmegaConf.create({}))
    assert merged.reward_config_version == 2
    assert merged.reward_space_kwargs.decay_games == 14000


def test_survival_shaping_uses_shared_global_game_count():
    reward = SurvivalPotentialReward(shaping_weight=0.05, decay_games=14000)
    counter = torch.multiprocessing.Value("q", 9538)
    reward.set_global_game_counter(counter)
    assert np.isclose(reward.shaping_alpha(), 0.05 * (1.0 - 9538 / 14000))
    reward._increment_game_count()
    assert counter.value == 9539
    assert reward.games == 0


def test_teacher_kl_ignores_fully_masked_invalid_entities():
    learner_logits = torch.tensor([[[[[[float("-inf"), float("-inf")]]]]]])
    teacher_logits = learner_logits.clone()
    actions_taken_mask = torch.zeros(1, 1, 1, 1, 1, dtype=torch.bool)

    loss = compute_teacher_kl_loss(learner_logits, teacher_logits, actions_taken_mask)

    assert torch.equal(loss, torch.zeros_like(loss))
    assert torch.isfinite(loss).all()


def test_teacher_kl_is_finite_with_masked_actions():
    learner_logits = torch.tensor([[[[[[0.2, float("-inf"), -0.1]]]]]])
    teacher_logits = torch.tensor([[[[[[0.5, float("-inf"), 0.1]]]]]])
    actions_taken_mask = torch.ones(1, 1, 1, 1, 1, dtype=torch.bool)

    loss = compute_teacher_kl_loss(learner_logits, teacher_logits, actions_taken_mask)

    assert torch.isfinite(loss).all()
    assert (loss >= 0).all()


def test_atomic_checkpoint_replaces_complete_file(tmp_path: Path):
    path = tmp_path / "checkpoint.pt"
    atomic_torch_save({"model_state_dict": {"x": torch.tensor([1])}}, path)
    atomic_torch_save({"model_state_dict": {"x": torch.tensor([2])}}, path)
    assert torch.load(path, weights_only=False)["model_state_dict"]["x"].item() == 2
    assert not list(tmp_path.glob("*.tmp"))


def test_strategic_reward_v2_keeps_nonzero_floor_after_decay():
    reward = StrategicPotentialRewardV2(shaping_weight=0.05, shaping_floor=0.01, decay_games=100)
    counter = torch.multiprocessing.Value("q", 250)
    reward.set_global_game_counter(counter)
    assert np.isclose(reward.shaping_alpha(), 0.01)


def test_trajectory_weighted_mean_balances_trajectory_sizes():
    losses = torch.tensor([[[10.0, 1.0]], [[0.0, 1.0]]])
    weights = torch.tensor([[[10.0, 1.0]], [[0.0, 1.0]]])
    # Player 0: 10 / 10, player 1: 2 / 2.
    assert torch.isclose(trajectory_weighted_mean(losses, weights), torch.tensor(1.0))


def test_pfsp_prefers_learnable_opponent_and_honours_prior_weight():
    opponents = [
        Opponent("too_hard", weight=1.0),
        Opponent("learnable", weight=1.0),
        Opponent("weighted", weight=2.0),
    ]
    probabilities = PFSPSampler(opponents, power=1.0).probabilities(
        {"too_hard": 0.02, "learnable": 0.5, "weighted": 0.5}
    )
    assert probabilities[1] > probabilities[0]
    assert probabilities[2] > probabilities[1]


def test_pfsp_reserves_fixed_first_place_probability():
    opponents = [
        Opponent("selfplay", kind="selfplay", weight=1.0),
        Opponent("first_place", kind="teacher", weight=1.0, fixed_probability=0.25),
        Opponent("snapshot", kind="learner_snapshot", weight=1.0),
    ]
    probabilities = PFSPSampler(opponents, teacher_floor=0.25).probabilities(
        {"selfplay": 0.5, "first_place": 0.0, "snapshot": 0.2}
    )
    assert np.isclose(probabilities.sum(), 1.0)
    assert np.isclose(probabilities[1], 0.25)


def test_quality_gate_and_full_checkpoint_selection(tmp_path: Path):
    baseline = {
        "opponents": {
            "first_place": {"score_rate": 0.20, "candidate_city_extinction_rate": 0.10}
        }
    }
    candidate = {
        "opponents": {
            "first_place": {"score_rate": 0.14, "candidate_city_extinction_rate": 0.18}
        }
    }
    decision = promotion_decision(
        baseline,
        candidate,
        opponent_name="first_place",
        min_score_delta=-0.05,
        max_city_extinction_delta=0.05,
    )
    assert decision["passed"] is False
    assert len(decision["reasons"]) == 2

    atomic_torch_save({"model_state_dict": {"x": torch.tensor(1)}}, tmp_path / "200_weights.pt")
    atomic_torch_save(
        {"model_state_dict": {}, "optimizer_state_dict": {}, "step": 100}, tmp_path / "100.pt"
    )
    atomic_torch_save(
        {"model_state_dict": {}, "optimizer_state_dict": {}, "step": 200}, tmp_path / "200.pt"
    )
    assert full_checkpoint(tmp_path) == tmp_path / "200.pt"


def test_checkpoint_model_difference_detects_identical_and_updated_policy(tmp_path: Path):
    base = tmp_path / "base.pt"
    identical = tmp_path / "identical.pt"
    updated = tmp_path / "updated.pt"
    torch.save({"model_state_dict": {"weight": torch.tensor([1.0, 2.0])}}, base)
    torch.save({"model_state_dict": {"weight": torch.tensor([1.0, 2.0])}}, identical)
    torch.save({"model_state_dict": {"weight": torch.tensor([1.0, 2.25])}}, updated)

    assert checkpoint_model_max_abs_diff(base, identical) == 0.0
    assert checkpoint_model_max_abs_diff(base, updated) == 0.25


def test_training_segment_uses_hydra_append_for_resume_paths(monkeypatch, tmp_path: Path):
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs

    monkeypatch.setattr("lux_ai.strategic_rl.train_eval.subprocess.run", fake_run)
    checkpoint = tmp_path / "base" / "500.pt"
    run_training_segment(
        config_name="survival_strategic_strength_v3",
        run_dir=tmp_path / "stage",
        stop_after_step=250,
        total_steps=1000,
        load_checkpoint=checkpoint,
        weights_only=False,
        python="python",
    )
    assert f"+load_dir={checkpoint.parent}" in captured["command"]
    assert "+checkpoint_file=500.pt" in captured["command"]
    assert "weights_only=false" in captured["command"]
    assert "total_steps=1000" in captured["command"]
    assert "stop_after_step=250" in captured["command"]


def test_reused_baseline_requires_exact_schedule_and_backend(tmp_path: Path):
    evaluation_dir = tmp_path / "baseline_evaluation"
    evaluation_dir.mkdir()
    records = [
        {
            "backend": "internal_batched",
            "opponent": "first_place",
            "seed": 2021,
            "map_size": 12,
            "candidate_player": player,
            "winner": player,
            "candidate_city_survival": 0.75,
            "candidate_final_city_tiles": 1,
            "candidate_final_units": 1,
        }
        for player in (0, 1)
    ]
    (evaluation_dir / "games.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    report = summarize(records, bootstrap_samples=10)
    (evaluation_dir / "report.json").write_text(json.dumps(report), encoding="utf-8")

    reused = load_reused_baseline_evaluation(
        evaluation_dir,
        opponent_name="first_place",
        seed_start=2021,
        seeds=1,
        map_sizes=(12,),
        eval_backend="internal",
        bootstrap_samples=10,
    )
    assert reused["backend"] == "internal"
    assert reused["summary"]["opponents"]["first_place"]["matched_pairs"] == 1

    with pytest.raises(ValueError, match="schedule does not match"):
        load_reused_baseline_evaluation(
            evaluation_dir,
            opponent_name="first_place",
            seed_start=2021,
            seeds=1,
            map_sizes=(12, 16),
            eval_backend="internal",
            bootstrap_samples=10,
        )
    with pytest.raises(ValueError, match="backend is internal"):
        load_reused_baseline_evaluation(
            evaluation_dir,
            opponent_name="first_place",
            seed_start=2021,
            seeds=1,
            map_sizes=(12,),
            eval_backend="official",
            bootstrap_samples=10,
        )


def _cpu_strength_flags(intent_aux_enabled: bool):
    config = OmegaConf.load(Path(__file__).parents[1] / "conf" / "survival_strategic_strength_v2.yaml")
    base = OmegaConf.load(Path(__file__).parents[1] / "conf" / "survival_strategic.yaml")
    shaping = OmegaConf.load(Path(__file__).parents[1] / "conf" / "survival_strategic_shaping.yaml")
    flags = OmegaConf.merge(base, shaping, config)
    flags.intent_aux_enabled = intent_aux_enabled
    flags.actor_device = "cpu"
    flags.learner_device = "cpu"
    flags.n_actor_envs = 1
    flags.num_buffers = 1
    flags.unroll_length = 1
    return flags_to_namespace(OmegaConf.to_container(flags, resolve=False))


def test_intent_head_can_be_enabled_or_disabled():
    enabled_flags = _cpu_strength_flags(True)
    env = create_env(enabled_flags, torch.device("cpu"))
    try:
        model_input = env.reset(force=True)
        enabled = create_model(enabled_flags, torch.device("cpu"))
        enabled_output = enabled(model_input, sample=False)
        assert enabled_output["intent_logits"].shape == (1, 2, 32, 32, 4)

        disabled_flags = _cpu_strength_flags(False)
        disabled = create_model(disabled_flags, torch.device("cpu"))
        assert "intent_logits" not in disabled(model_input, sample=False)
    finally:
        env.close()


def test_categorical_critic_forward_is_zero_sum_and_keeps_policy_schema():
    flags = _cpu_strength_flags(False)
    flags.value_critic = "categorical_hl_gauss"
    flags.value_num_bins = 101
    flags.value_support_min = -2.0
    flags.value_support_max = 2.0
    env = create_env(flags, torch.device("cpu"))
    try:
        model_input = env.reset(force=True)
        output = create_model(flags, torch.device("cpu"))(model_input, sample=False)
        assert output["baseline_logits"].shape == (1, 2, 101)
        assert torch.allclose(output["baseline"].sum(dim=-1), torch.zeros(1), atol=1e-6)
        assert output["policy_logits"].keys() == model_input["info"]["available_actions_mask"].keys()
    finally:
        env.close()


def test_engine_snapshot_replay_round_trip_transition_matches():
    flags = _cpu_strength_flags(False)
    env = create_env(flags, torch.device("cpu"))
    try:
        output = env.reset(force=True)
        actions = {
            entity: torch.zeros((*mask.shape[:-1], 4), dtype=torch.long)
            for entity, mask in output["info"]["available_actions_mask"].items()
        }
        env.step(actions)
        snapshot = env.capture_snapshots([0], ["test"])[0]
        env.step(actions)

        def signature():
            game = env.unwrapped[0].game_state
            roads = tuple(
                round(float(cell.road), 6) for row in game.map.map for cell in row
            )
            players = tuple(
                (
                    player.research_points,
                    tuple(sorted((unit.id, unit.pos.x, unit.pos.y, unit.cooldown, unit.cargo.wood,
                                  unit.cargo.coal, unit.cargo.uranium) for unit in player.units)),
                    tuple(sorted((city.cityid, city.fuel, city.light_upkeep,
                                  tuple(sorted((tile.pos.x, tile.pos.y, tile.cooldown) for tile in city.citytiles)))
                                 for city in player.cities.values())),
                )
                for player in game.players
            )
            return game.turn, roads, players

        expected = signature()
        env.restore_snapshots({0: snapshot})
        env.step(actions)
        assert signature() == expected
    finally:
        env.close()


def test_intent_head_only_mode_freezes_every_other_parameter():
    flags = _cpu_strength_flags(True)
    model = create_model(flags, torch.device("cpu"))
    parameters, names = configure_trainable_parameters(model, intent_head_only=True)

    assert names == ["intent_head.weight", "intent_head.bias"]
    assert parameters == [model.intent_head.weight, model.intent_head.bias]
    assert all(
        parameter.requires_grad == name.startswith("intent_head.")
        for name, parameter in model.named_parameters()
    )


def test_intent_head_only_requires_enabled_head():
    flags = _cpu_strength_flags(False)
    model = create_model(flags, torch.device("cpu"))
    with pytest.raises(ValueError, match="intent_aux_enabled=true"):
        configure_trainable_parameters(model, intent_head_only=True)


def test_model_state_validation_detects_update_and_syncs_actor():
    class RecordingLinear(torch.nn.Linear):
        def load_state_dict(self, state_dict, *args, **kwargs):
            self.loaded_state_devices = {tensor.device.type for tensor in state_dict.values()}
            self.loaded_state_pointers = {name: tensor.data_ptr() for name, tensor in state_dict.items()}
            return super().load_state_dict(state_dict, *args, **kwargs)

    actor = RecordingLinear(3, 2)
    learner = torch.nn.Linear(3, 2)
    initial = model_state_dict_cpu(actor)
    with torch.no_grad():
        learner.weight.fill_(0.5)
        learner.bias.fill_(-0.25)

    assert state_dict_max_abs_diff(model_state_dict_cpu(learner), initial) > 0.0

    sync_actor_model(actor, learner, verify=True)

    assert actor.loaded_state_devices == {"cpu"}
    assert all(
        actor.loaded_state_pointers[name] != tensor.data_ptr()
        for name, tensor in learner.state_dict().items()
    )
    assert state_dict_max_abs_diff(model_state_dict_cpu(actor), model_state_dict_cpu(learner)) == 0.0


def test_model_state_validation_rejects_incompatible_states():
    with pytest.raises(RuntimeError, match="Incompatible model states"):
        state_dict_max_abs_diff({"weight": torch.ones(1)}, {"bias": torch.ones(1)})


def test_rule_guidance_emits_masked_intent_targets():
    flags = _cpu_strength_flags(True)
    env = create_env(flags, torch.device("cpu"))
    try:
        env_output = env.reset(force=True)
        action_template = {
            entity: torch.zeros((*mask.shape[:-1], 4), dtype=torch.long)
            for entity, mask in env_output["info"]["available_actions_mask"].items()
        }
        actions, confidence, intents, intent_mask = rule_based_guidance(
            env.unwrapped,
            env_output["info"]["available_actions_mask"],
            action_template,
            [0, 0],
            [0, 1],
            strategy="economy",
        )
        assert actions.keys() == action_template.keys()
        assert confidence["worker"].shape == intents.shape
        assert intent_mask.sum().item() == 2
        assert set(intents[intent_mask].tolist()).issubset({0, 1, 2, 3})
    finally:
        env.close()


def test_resume_migrates_strength_objective_but_cli_can_disable_intent():
    selected = OmegaConf.create(
        {
            "objective_config_version": 1,
            "loss_normalization": "trajectory",
            "normalize_advantages": True,
            "intent_aux_enabled": True,
            "intent_aux_cost": 0.01,
        }
    )
    merged = merge_resume_config(
        OmegaConf.create({"objective_config_version": 0}),
        selected,
        OmegaConf.create({"intent_aux_enabled": False}),
    )
    assert merged.loss_normalization == "trajectory"
    assert merged.intent_aux_enabled is False


def test_replay_metrics_tracks_night_survival_and_orientation():
    empty = {"observation": {"updates": []}}
    steps = [[empty, empty] for _ in range(41)]
    steps[30] = [
        {"observation": {"updates": ["c 1 c0 50 10", "ct 1 c0 0 0 0", "ct 1 c0 1 0 0"]}},
        empty,
    ]
    steps[40] = [{"observation": {"updates": ["c 1 c0 0 10", "ct 1 c0 0 0 0"]}}, empty]
    metrics = replay_metrics({"rewards": [0, 1], "steps": steps}, candidate_player=1)
    assert metrics["winner"] == 1
    assert metrics["candidate_city_survival"] == 0.5


def test_replay_metrics_reads_official_stateful_cli_schema():
    def state(turn: int, city_tiles: int, units: int, fuel: float) -> dict:
        cities = {}
        if city_tiles:
            cities["c0"] = {
                "team": 1,
                "fuel": fuel,
                "lightupkeep": 10,
                "cityCells": [{"x": index, "y": 0} for index in range(city_tiles)],
            }
        return {
            "turn": turn,
            "cities": cities,
            "teamStates": {"1": {"units": {f"u{index}": {} for index in range(units)}}},
        }

    replay = {
        "results": {"ranks": [{"rank": 1, "agentID": 1}, {"rank": 2, "agentID": 0}]},
        "stateful": [state(30, 2, 3, 50), state(40, 1, 2, 20)],
    }
    metrics = replay_metrics(replay, candidate_player=1)
    assert metrics == {
        "winner": 1,
        "candidate_final_city_tiles": 1,
        "candidate_final_units": 2,
        "candidate_city_survival": 0.5,
        "candidate_stranded_fuel": 0.0,
    }


def test_evaluation_requires_and_counts_matched_map_orientations():
    records = [
        {"opponent": "teacher", "seed": 7, "map_size": 12, "candidate_player": 0, "winner": 0},
        {"opponent": "teacher", "seed": 7, "map_size": 12, "candidate_player": 1, "winner": 0},
        {"opponent": "teacher", "seed": 7, "map_size": 16, "candidate_player": 0, "winner": 0},
        {"opponent": "teacher", "seed": 7, "map_size": 16, "candidate_player": 1, "winner": 1},
    ]
    report = summarize(records, bootstrap_samples=10)
    teacher = report["opponents"]["teacher"]
    assert teacher["matched_pairs"] == 2
    assert teacher["score_rate"] == 0.75


def test_evaluation_reports_extinction_rates():
    records = [
        {
            "opponent": "teacher",
            "seed": 7,
            "map_size": 12,
            "candidate_player": player,
            "winner": player,
            "candidate_final_city_tiles": 0 if player == 0 else 2,
            "candidate_final_units": 0 if player == 0 else 1,
        }
        for player in (0, 1)
    ]
    teacher = summarize(records, bootstrap_samples=10)["opponents"]["teacher"]
    assert teacher["candidate_city_extinction_rate"] == 0.5
    assert teacher["candidate_unit_extinction_rate"] == 0.5


def test_compact_shard_loads_ragged_entities_and_rebuilds_dense_mask(tmp_path: Path):
    shard_name = "compact.npz"
    arrays = {
        "schema_version": np.asarray(3),
        "teacher_tta_rot180": np.asarray(True),
        "turn_count": np.asarray(1),
        "input_mask": np.ones((1, 2, 2), dtype=np.bool_),
        "obs__continuous": np.ones((1, 1, 1, 2, 2), dtype=np.float16),
        "obs__categorical": np.zeros((1, 1, 1, 2, 2), dtype=np.uint8),
    }
    action_counts = {"worker": 19, "cart": 17, "city_tile": 4}
    for entity, action_count in action_counts.items():
        count = 1 if entity == "worker" else 0
        arrays[f"{entity}_offsets"] = np.asarray([0, count], dtype=np.int64)
        arrays[f"{entity}_positions"] = np.asarray([[1, 0, 1]], dtype=np.int16) if count else np.empty((0, 3), np.int16)
        arrays[f"{entity}_legal_mask"] = (
            np.ones((1, action_count), dtype=np.bool_) if count else np.empty((0, action_count), np.bool_)
        )
        arrays[f"{entity}_teacher_logits"] = np.zeros((count, action_count), dtype=np.float16)
    np.savez_compressed(tmp_path / shard_name, **arrays)
    manifest = {
        "schema_version": 3,
        "teacher_tta_rot180": True,
        "shards": [{"path": shard_name, "split": "train", "turn_count": 1}],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    dataset = ShardDataset(tmp_path, "train")
    batch = _compact_collate([dataset[0]])
    assert batch["obs"]["continuous"].dtype == torch.float32
    assert batch["obs"]["categorical"].dtype == torch.int64
    assert batch["positions"]["worker"].tolist() == [[[1, 0, 1]]]
    assert batch["available_actions_mask"]["worker"][0, 0, 1, 0, 1].all()
    assert batch["positions"]["cart"].shape == (1, 0, 3)


def test_rot180_policy_and_observation_round_trip():
    observation = {"spatial": torch.arange(16).reshape(1, 1, 1, 4, 4), "global": torch.ones(1, 1, 2)}
    policy = {
        entity: torch.arange(len(actions)).reshape(1, 1, 1, 1, 1, -1).expand(1, 1, 2, 4, 4, -1)
        for entity, actions in ACTION_MEANINGS.items()
    }
    assert torch.equal(rotate_observations_180(rotate_observations_180(observation))["spatial"], observation["spatial"])
    for entity, values in policy.items():
        assert torch.equal(rotate_policy_180(rotate_policy_180(policy))[entity], values)
        assert sorted(ROT180_ACTION_INDICES[entity]) == list(range(len(ACTION_MEANINGS[entity])))


def test_compact_rot180_moves_positions_and_remaps_targets():
    action_count = len(ACTION_MEANINGS["worker"])
    batch = {
        "obs": {"spatial": torch.arange(16).reshape(1, 1, 1, 4, 4)},
        "input_mask": torch.ones(1, 1, 4, 4, dtype=torch.bool),
        "positions": {"worker": torch.tensor([[[0, 0, 1]]])},
        "legal_mask": {"worker": torch.eye(action_count, dtype=torch.bool)[0].reshape(1, 1, -1)},
        "teacher_logits": {"worker": torch.arange(action_count).reshape(1, 1, -1).float()},
        "available_actions_mask": {
            "worker": torch.ones(1, 1, 2, 4, 4, action_count, dtype=torch.bool),
        },
    }
    for entity, actions in ACTION_MEANINGS.items():
        if entity == "worker":
            continue
        count = len(actions)
        batch["positions"][entity] = torch.empty(1, 0, 3, dtype=torch.long)
        batch["legal_mask"][entity] = torch.empty(1, 0, count, dtype=torch.bool)
        batch["teacher_logits"][entity] = torch.empty(1, 0, count)
        batch["available_actions_mask"][entity] = torch.ones(1, 1, 2, 4, 4, count, dtype=torch.bool)
    rotated = rotate_compact_distillation_batch_180(batch)
    assert rotated["positions"]["worker"].tolist() == [[[0, 3, 2]]]
    assert rotated["teacher_logits"]["worker"][0, 0].tolist() == [
        float(index) for index in ROT180_ACTION_INDICES["worker"]
    ]


def test_replay_discovery_accepts_file_and_ignores_metadata(tmp_path: Path):
    replay = tmp_path / "123.json"
    replay.write_text("{}", encoding="utf-8")
    (tmp_path / "agent_info.json").write_text("{}", encoding="utf-8")
    (tmp_path / "123_info.json").write_text("{}", encoding="utf-8")
    assert _discover_replays(replay) == [replay]
    assert _discover_replays(tmp_path) == [replay]
