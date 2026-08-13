from pathlib import Path

import pytest
import torch
import yaml
from torch import nn

from lux_ai.lux.constants import Constants
from lux_ai.lux.game_objects import Player, Unit
from lux_ai.lux_gym.act_spaces import ACTION_MEANINGS, ACTION_MEANINGS_TO_IDX
from lux_ai.rl_agent.action_postprocessing import resolve_collision_rankings
from lux_ai.strategic_rl.es import (
    ClipUp,
    ParameterSpace,
    antithetic_gradient,
    centered_rank_utilities,
    make_match_schedule,
    orthonormalize,
    policy_fitness,
    sample_direction,
)
from lux_ai.strategic_rl.train_es import (
    CandidateRequest,
    OfficialMatchEvaluator,
    _candidate_results,
    _save_es_state,
    load_deployment_action_config,
    paired_schedule,
)


class TinyPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = nn.Linear(3, 4)
        self.actor_base = nn.utils.spectral_norm(nn.Linear(4, 4))
        self.actor = nn.Linear(4, 2)
        self.baseline_base = nn.Linear(4, 4)
        self.baseline = nn.Linear(4, 1)
        self.intent_head = nn.Linear(4, 3)
        self.register_buffer("policy_counter", torch.tensor(0, dtype=torch.long))


def test_parameter_space_evolves_policy_and_excludes_value_and_intent():
    model = TinyPolicy()
    space = ParameterSpace(model)
    assert all(name.startswith(("base_model.", "actor_base.", "actor.")) for name in space.names)
    assert not any(name.startswith(("baseline", "intent_head")) for name in space.names)
    baseline = model.baseline.weight.detach().clone()
    spectral_buffer = model.actor_base.weight_u.detach().clone()
    integer_buffer = model.policy_counter.detach().clone()
    center = space.flatten_model(model)
    direction, _ = sample_direction(space.dimension, 7)
    plus = space.perturb(center, direction, sigma=0.01, sign=1)
    minus = space.perturb(center, direction, sigma=0.01, sign=-1)
    assert torch.allclose((plus + minus) / 2, center)
    space.assign(model, plus)
    assert torch.equal(model.baseline.weight, baseline)
    assert torch.equal(model.actor_base.weight_u, spectral_buffer)
    assert torch.equal(model.policy_counter, integer_buffer)


def test_rank_utilities_average_ties_and_antithetic_gradient_direction():
    utilities = centered_rank_utilities([1.0, 1.0, 3.0, 4.0])
    assert utilities[0] == utilities[1]
    assert utilities.sum().item() == pytest.approx(0.0)
    directions = [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])]
    gradient = antithetic_gradient(directions, plus=[2.0, 0.0], minus=[0.0, 1.0])
    assert gradient[0] > 0
    assert gradient[1] < 0


def test_clipup_clips_velocity_and_shrinks_after_rejection():
    optimizer = ClipUp(step_size=2.0, max_speed=1.0, momentum=0.9)
    update = optimizer.update(torch.tensor([3.0, 4.0]))
    assert torch.linalg.vector_norm(update).item() == pytest.approx(1.0)
    optimizer.reject_and_shrink()
    assert optimizer.step_size == 1.0
    assert optimizer.max_speed == 0.5
    assert torch.count_nonzero(optimizer.velocity) == 0
    restored = ClipUp.load_state_dict(optimizer.state_dict())
    assert restored.max_speed == optimizer.max_speed


def test_active_subspace_is_orthonormal_and_noise_is_reconstructible():
    basis = orthonormalize(
        [torch.tensor([1.0, 0.0, 0.0]), torch.tensor([1.0, 1.0, 0.0])], max_rank=2
    )
    assert torch.allclose(basis @ basis.T, torch.eye(2), atol=1e-6)
    first, kind = sample_direction(3, seed=11, basis=basis, active_probability=1.0)
    second, _ = sample_direction(3, seed=11, basis=basis, active_probability=1.0)
    assert kind == "active"
    assert torch.equal(first, second)
    assert torch.linalg.vector_norm(first).item() == pytest.approx(3**0.5)


def test_non_active_sample_retains_full_space_support():
    basis = torch.tensor([[1.0, 0.0, 0.0]])
    direction, kind = sample_direction(3, seed=19, basis=basis, active_probability=0.0)
    assert kind == "isotropic"
    assert abs(torch.dot(direction, basis[0]).item()) > 1e-6


def test_fitness_uses_win_loss_as_primary_and_bounded_survival_tie_break():
    records = [
        {
            "winner": 0,
            "candidate_player": 0,
            "candidate_city_survival": 0.0,
            "candidate_stranded_fuel": 1.0,
            "candidate_final_city_tiles": 1,
        },
        {
            "winner": 1,
            "candidate_player": 0,
            "candidate_city_survival": 1.0,
            "candidate_stranded_fuel": 0.0,
            "candidate_final_city_tiles": 0,
        },
    ]
    metrics = policy_fitness(records)
    assert metrics["score_rate"] == 0.5
    assert 0.5 <= metrics["fitness"] <= 0.51
    assert metrics["candidate_city_extinction_rate"] == 0.5


def test_training_and_gate_schedules_are_deterministic_and_paired():
    training = make_match_schedule(
        generation=1,
        games=4,
        opponents=("a", "b"),
        seed_start=100,
        map_sizes=(12, 16),
    )
    assert training == make_match_schedule(
        generation=1,
        games=4,
        opponents=("a", "b"),
        seed_start=100,
        map_sizes=(12, 16),
    )
    next_generation = make_match_schedule(
        generation=2,
        games=4,
        opponents=("a", "b"),
        seed_start=100,
        map_sizes=(12, 16),
    )
    assert training[0].seed != next_generation[0].seed
    assert training[0].candidate_player != next_generation[0].candidate_player
    grouped = make_match_schedule(
        generation=0,
        games=6,
        opponents=("a", "b"),
        seed_start=100,
        map_sizes=(12, 16, 24, 32),
    )
    assert [spec.opponent for spec in grouped] == ["a"] * 3 + ["b"] * 3
    assert [spec.candidate_player for spec in grouped] == [0, 1, 0] * 2
    assert [spec.map_size for spec in grouped] == [12, 16, 24] * 2
    gate = paired_schedule(
        generation=0,
        pairs=2,
        opponents=("a", "b"),
        seed_start=200,
        map_sizes=(12, 16),
        namespace="gate",
    )
    assert [(spec.seed, spec.map_size, spec.candidate_player) for spec in gate] == [
        (200, 12, 0),
        (200, 12, 1),
        (201, 16, 0),
        (201, 16, 1),
    ]


def test_deployment_collision_resolver_prevents_friendly_duplicate_destination():
    players = [Player(0), Player(1)]
    players[0].units = [
        Unit(0, Constants.UNIT_TYPES.WORKER, "u0", 0, 0, 0, 0, 0, 0),
        Unit(0, Constants.UNIT_TYPES.WORKER, "u1", 2, 0, 0, 0, 0, 0),
    ]
    game_state = type(
        "GameState",
        (),
        {"players": players, "map_width": 12, "map_height": 12, "turn": 0},
    )()
    logits = {
        entity: torch.full((1, 1, 2, 32, 32, len(actions)), -100.0)
        for entity, actions in ACTION_MEANINGS.items()
    }
    east = ACTION_MEANINGS_TO_IDX["worker"]["MOVE_e"]
    west = ACTION_MEANINGS_TO_IDX["worker"]["MOVE_w"]
    no_op = ACTION_MEANINGS_TO_IDX["worker"]["NO-OP"]
    logits["worker"][0, 0, 0, 0, 0, east] = 10.0
    logits["worker"][0, 0, 0, 0, 0, no_op] = 0.0
    logits["worker"][0, 0, 0, 2, 0, west] = 9.0
    logits["worker"][0, 0, 0, 2, 0, no_op] = 0.0
    rankings = resolve_collision_rankings(
        game_state,
        0,
        logits,
        must_research=True,
        can_build_carts=False,
    )
    assert rankings["worker"][0, 0, 0, 0, 0].item() == east
    assert rankings["worker"][0, 0, 2, 0, 0].item() == no_op


def test_deployment_settings_follow_each_agent_bundle():
    root = Path(__file__).parents[1]
    current = load_deployment_action_config()
    old_agent = (
        root
        / "internal_testing/hall_of_fame/10-10_11-18-12_28576448/main.py"
    )
    old = load_deployment_action_config(old_agent)
    assert current.use_collision_detection and current.use_rot180
    assert current.must_research and not current.can_build_carts and current.force_last_turn_cart
    assert old.use_collision_detection and old.use_rot180
    assert not old.must_research and old.can_build_carts and not old.force_last_turn_cart


def test_es_config_paths_exist():
    root = Path(__file__).parents[1]
    config_path = root / "conf" / "survival_strategic_es.yaml"
    assert config_path.is_file()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["search"]["pilot_games_per_candidate"] == 4
    assert config["search"]["games_per_candidate"] == 6
    assert [opponent["name"] for opponent in config["opponents"]] == [
        "first_place",
        "initial_model",
    ]


def test_resume_state_preserves_next_clipup_update(tmp_path):
    first_gradient = torch.tensor([1.0, -2.0, 0.5])
    next_gradient = torch.tensor([-0.5, 1.5, 2.0])
    continuous = ClipUp(step_size=0.2, max_speed=0.3)
    continuous.update(first_gradient)
    _save_es_state(
        tmp_path / "latest_es.pt",
        center=torch.arange(3, dtype=torch.float32),
        best_center=torch.arange(3, dtype=torch.float32),
        generation=4,
        sigma=0.005,
        optimizer=continuous,
        basis=torch.eye(3)[:2],
        recent_updates=[torch.ones(3)],
        result_cache={
            "g0003-gate-new": {
                "matches": [{"match_id": "gate-g0003-pair000-p0"}],
            }
        },
        init_sha256="init",
        config_sha256="config",
        search_seed=2021,
        status="running",
    )
    saved = torch.load(tmp_path / "latest_es.pt", weights_only=False)
    resumed = ClipUp.load_state_dict(saved["clipup"])
    assert saved["generation"] == 4
    assert saved["completed_candidate_ids"] == ["g0003-gate-new"]
    assert saved["completed_match_ids"] == ["g0003-gate-new:gate-g0003-pair000-p0"]
    assert torch.equal(continuous.update(next_gradient), resumed.update(next_gradient))


def test_official_backend_retries_only_one_turn_zero_cold_start(monkeypatch, tmp_path):
    calls = []

    def fake_run_match(spec, replay):
        calls.append((spec, replay))
        if len(calls) == 1:
            raise RuntimeError("Candidate stopped responding after turn 0")
        return {"winner": 0}

    evaluator = OfficialMatchEvaluator.__new__(OfficialMatchEvaluator)
    evaluator.candidate_agent = tmp_path / "candidate.py"
    evaluator.opponent_agents = {"opponent": tmp_path / "opponent.py"}
    evaluator.python = "python"
    evaluator.timeout = 10
    monkeypatch.setattr(evaluator, "_run_official_worker", fake_run_match)
    spec = make_match_schedule(
        generation=0,
        games=1,
        opponents=("opponent",),
        seed_start=123,
        map_sizes=(12,),
    )[0]
    record = evaluator._run_match_with_cold_start_retry(spec, tmp_path / "replay.json")
    assert record == {"winner": 0, "official_turn0_retries": 1}
    assert len(calls) == 2
    assert calls[0][1].name == "replay-attempt0.json"
    assert calls[1][1].name == "replay-attempt1.json"


def test_candidate_results_uses_parallel_evaluator_and_preserves_order(tmp_path):
    class FakeParallelEvaluator:
        def __init__(self):
            self.calls = []

        def evaluate_many(self, candidate_states, schedule, max_workers):
            self.calls.append((len(candidate_states), len(schedule), max_workers))
            return [
                (
                    [
                        {
                            "winner": 0,
                            "candidate_player": 0,
                            "candidate_final_city_tiles": index + 1,
                        }
                    ],
                    {"evaluation_seconds": float(index + 1)},
                )
                for index, _state in enumerate(candidate_states)
            ]

    model = TinyPolicy()
    parameter_space = ParameterSpace(model)
    center = parameter_space.flatten_model(model)
    evaluator = FakeParallelEvaluator()
    schedule = make_match_schedule(
        generation=0,
        games=1,
        opponents=("opponent",),
        seed_start=123,
        map_sizes=(12,),
    )
    cache = {}
    results = _candidate_results(
        requests=[
            CandidateRequest("plus", center + 0.1, {"sign": 1}),
            CandidateRequest("minus", center - 0.1, {"sign": -1}),
        ],
        evaluator=evaluator,
        model=model,
        parameter_space=parameter_space,
        schedule=schedule,
        tie_break_weight=0.01,
        cache=cache,
        output=tmp_path / "fitness.jsonl",
        candidate_workers=2,
    )
    assert evaluator.calls == [(2, 1, 2)]
    assert [result["candidate_id"] for result in results] == ["plus", "minus"]
    assert all(result["candidate_parallel"] for result in results)
    assert all(result["candidate_group_size"] == 2 for result in results)
    assert [result["backend_profile"]["evaluation_seconds"] for result in results] == [1.0, 2.0]
    assert list(cache) == ["plus", "minus"]
    assert len((tmp_path / "fitness.jsonl").read_text(encoding="utf-8").splitlines()) == 2
