import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from lux_ai.strategic_rl.evaluate import summarize
from lux_ai.strategic_rl.train_ab import (
    final_promotion_decision,
    map_score_rate,
    paired_score_difference_lcb,
    run_ab,
    select_winning_arm,
    stage_gate_decision,
)


def _summary(name: str, score: float, survival: float = 0.75, extinction: float = 0.08) -> dict:
    return {
        "opponents": {
            name: {
                "score_rate": score,
                "candidate_city_survival": survival,
                "candidate_city_extinction_rate": extinction,
            }
        }
    }


def _write_games(path: Path, scores: dict[tuple[int, int], tuple[int, int]], opponent: str = "first_place"):
    records = []
    for (seed, map_size), winners in scores.items():
        for candidate_player, winner in enumerate(winners):
            records.append(
                {
                    "opponent": opponent,
                    "seed": seed,
                    "map_size": map_size,
                    "candidate_player": candidate_player,
                    "winner": winner,
                }
            )
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def test_stage_gate_checks_score_extinction_and_map32():
    decision = stage_gate_decision(
        _summary("first_place", 0.30, extinction=0.05),
        _summary("first_place", 0.27, extinction=0.08),
        opponent_name="first_place",
        baseline_map32_score=0.20,
        candidate_map32_score=0.15,
    )
    assert decision["passed"] is False
    assert len(decision["reasons"]) == 3


def test_arm_selection_uses_score_then_map32_then_extinction():
    arms = {
        "arm_a": {
            "decision": {
                "passed": True,
                "score_delta": 0.01,
                "map32_score_delta": 0.02,
                "city_extinction_delta": 0.01,
            }
        },
        "arm_b": {
            "decision": {
                "passed": True,
                "score_delta": 0.02,
                "map32_score_delta": -0.01,
                "city_extinction_delta": 0.00,
            }
        },
    }
    assert select_winning_arm(arms) == "arm_b"
    arms["arm_b"]["decision"]["passed"] = False
    assert select_winning_arm(arms) == "arm_a"


def test_map_score_and_paired_difference_lcb(tmp_path: Path):
    baseline = tmp_path / "baseline.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    _write_games(baseline, {(1, 12): (1, 0), (1, 32): (1, 0)})
    _write_games(candidate, {(1, 12): (0, 1), (1, 32): (0, 1)})
    assert map_score_rate(candidate, 32) == 1.0
    assert paired_score_difference_lcb(baseline, candidate, bootstrap_samples=50) == 1.0

    mismatched = tmp_path / "mismatched.jsonl"
    _write_games(mismatched, {(2, 12): (0, 1)})
    with pytest.raises(ValueError, match="schedules differ"):
        paired_score_difference_lcb(baseline, mismatched, bootstrap_samples=10)


def test_final_promotion_requires_all_absolute_and_nonregression_gates():
    decision = final_promotion_decision(
        _summary("first_place", 0.25),
        _summary("first_place", 0.30, survival=0.74, extinction=0.09),
        _summary("base", 0.60),
        _summary("base", 0.59),
        teacher_name="first_place",
        base_name="base",
        paired_difference_lcb95=-0.01,
        map32_score_delta=0.05,
    )
    assert decision["passed"] is True

    failed = final_promotion_decision(
        _summary("first_place", 0.25),
        _summary("first_place", 0.27, survival=0.72, extinction=0.11),
        _summary("base", 0.60),
        _summary("base", 0.55),
        teacher_name="first_place",
        base_name="base",
        paired_difference_lcb95=-0.03,
        map32_score_delta=-0.05,
    )
    assert failed["passed"] is False
    assert len(failed["reasons"]) == 6


def test_run_ab_reuses_validated_gate_baseline(monkeypatch, tmp_path: Path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    initial_checkpoint = inputs / "initial_weights.pt"
    initial_config = inputs / "initial.yaml"
    base_checkpoint = inputs / "base_weights.pt"
    base_config = inputs / "base.yaml"
    teacher = inputs / "teacher.py"
    for path in (initial_checkpoint, initial_config, base_checkpoint, base_config, teacher):
        path.touch()

    reused = tmp_path / "reused"
    reused.mkdir()
    records = [
        {
            "backend": "internal_batched",
            "opponent": "first_place",
            "seed": 2021,
            "map_size": 32,
            "candidate_player": player,
            "winner": player,
            "candidate_city_survival": 0.75,
            "candidate_final_city_tiles": 1,
            "candidate_final_units": 1,
        }
        for player in (0, 1)
    ]
    (reused / "games.jsonl").write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    (reused / "report.json").write_text(json.dumps(summarize(records, bootstrap_samples=10)), encoding="utf-8")

    monkeypatch.setattr(
        "lux_ai.strategic_rl.train_ab._evaluate",
        lambda *args, **kwargs: pytest.fail("validated baseline should be reused"),
    )
    monkeypatch.setattr(
        "lux_ai.strategic_rl.train_ab._train_and_evaluate_stage",
        lambda **kwargs: {"decision": {"passed": False, "reasons": ["test stop"]}},
    )
    args = SimpleNamespace(
        initial_checkpoint=initial_checkpoint,
        initial_config=initial_config,
        base_checkpoint=base_checkpoint,
        base_config=base_config,
        teacher_opponent=teacher,
        run_root=tmp_path / "run",
        reuse_baseline_evaluation=reused,
        pilot_steps=25_000,
        extension_steps=(50_000, 100_000),
        map_sizes=(32,),
        gate_seed_start=2021,
        gate_seeds=1,
        final_seed_start=12021,
        final_seeds=1,
        teacher_name="first_place",
        eval_backend="internal",
        bootstrap_samples=10,
        arm_a_config_name="arm_a",
        arm_b_config_name="arm_b",
    )
    result = run_ab(args)

    assert result["status"] == "stopped_no_pilot_passed"
    assert result["gate_baseline_evaluation"] == {
        "reused": True,
        "source": str(reused.resolve()),
        "backend": "internal",
    }
