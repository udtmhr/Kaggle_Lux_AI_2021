import json
from pathlib import Path

import pytest

from lux_ai.strategic_rl.train_ab import (
    final_promotion_decision,
    map_score_rate,
    paired_score_difference_lcb,
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
