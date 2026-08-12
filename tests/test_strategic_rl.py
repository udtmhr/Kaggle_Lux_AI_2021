import json
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from omegaconf import OmegaConf

from lux_ai.lux_gym.act_spaces import ACTION_MEANINGS
from lux_ai.lux_gym.wrappers import VecEnv
from lux_ai.rl_agent.rl_agent import RLAgent, checkpoint_path, model_directory
from lux_ai.strategic_rl.artifacts import atomic_torch_save
from lux_ai.strategic_rl.evaluate import summarize
from lux_ai.strategic_rl.evaluate_checkpoint import evaluate_checkpoint
from lux_ai.strategic_rl.league import (
    LeagueSampler,
    Opponent,
    PFSPSampler,
    learner_player_mask,
    merge_player_actions,
    merge_player_actions_inplace,
    opponents_from_config,
)
from lux_ai.strategic_rl.models import SurvivalStrategicBackbone
from lux_ai.strategic_rl.obs import night_turns_between
from lux_ai.strategic_rl.prepare_data import _discover_replays
from lux_ai.strategic_rl.prepare_eval_agent import checkpoint_label, prepare_eval_agent, sha256_file
from lux_ai.strategic_rl.resume import merge_resume_config
from lux_ai.strategic_rl.reward import SurvivalPotentialReward
from lux_ai.strategic_rl.run_matches import candidate_last_response_turn, replay_metrics
from lux_ai.strategic_rl.schedules import LinearSchedule, teacher_kl_coefficient
from lux_ai.strategic_rl.train_distill import ShardDataset, _compact_collate
from lux_ai.strategic_rl.tta import (
    ROT180_ACTION_INDICES,
    rot180_ensemble_outputs,
    rotate_compact_distillation_batch_180,
    rotate_model_input_180,
    rotate_observations_180,
    rotate_policy_180,
)
from lux_ai.torchbeast.monobeast import compute_baseline_loss, compute_teacher_kl_loss


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
    assert (output_dir / "report.json").is_file()


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
