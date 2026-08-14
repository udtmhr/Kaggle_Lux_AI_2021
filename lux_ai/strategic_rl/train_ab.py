from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from .evaluate import candidate_score, load_records
from .evaluate_checkpoint import FIRST_PLACE_AGENT, evaluate_checkpoint
from .prepare_eval_agent import prepare_eval_agent
from .train_eval import (
    checkpoint_model_max_abs_diff,
    full_checkpoint,
    load_reused_baseline_evaluation,
    run_training_segment,
    write_progress,
)


def _opponent_metrics(summary: dict, opponent_name: str) -> dict:
    try:
        return summary["opponents"][opponent_name]
    except KeyError as error:
        raise ValueError(f"Evaluation lacks opponent metrics for {opponent_name}") from error


def map_score_rate(games_path: Path, map_size: int) -> float:
    records = [record for record in load_records(games_path) if int(record.get("map_size", -1)) == map_size]
    if not records:
        raise ValueError(f"Evaluation has no games for map size {map_size}: {games_path}")
    return float(np.mean([candidate_score(record) for record in records]))


def paired_score_difference_lcb(
    baseline_games: Path,
    candidate_games: Path,
    *,
    bootstrap_samples: int = 2000,
    seed: int = 2021,
) -> float:
    def paired_scores(path: Path) -> dict[tuple[int, int], float]:
        grouped: dict[tuple[int, int], list[dict]] = {}
        for record in load_records(path):
            key = (int(record["seed"]), int(record.get("map_size", -1)))
            grouped.setdefault(key, []).append(record)
        result = {}
        for key, records in grouped.items():
            if {int(record["candidate_player"]) for record in records} != {0, 1}:
                raise ValueError(f"Incomplete orientation pair for {key} in {path}")
            if len(records) != 2:
                raise ValueError(f"Duplicate orientation games for {key} in {path}")
            result[key] = float(np.mean([candidate_score(record) for record in records]))
        return result

    baseline = paired_scores(baseline_games)
    candidate = paired_scores(candidate_games)
    if baseline.keys() != candidate.keys():
        raise ValueError("Baseline and candidate evaluation schedules differ")
    differences = np.asarray([candidate[key] - baseline[key] for key in sorted(baseline)])
    rng = np.random.default_rng(seed)
    boot = np.asarray(
        [rng.choice(differences, len(differences), replace=True).mean() for _ in range(bootstrap_samples)]
    )
    return float(np.quantile(boot, 0.025))


def stage_gate_decision(
    baseline: dict,
    candidate: dict,
    *,
    opponent_name: str,
    baseline_map32_score: float,
    candidate_map32_score: float,
    min_score_delta: float = -0.025,
    max_city_extinction_delta: float = 0.025,
    min_map32_score_delta: float = 0.0,
) -> dict:
    baseline_metrics = _opponent_metrics(baseline, opponent_name)
    candidate_metrics = _opponent_metrics(candidate, opponent_name)
    score_delta = float(candidate_metrics["score_rate"] - baseline_metrics["score_rate"])
    extinction_delta = float(
        candidate_metrics["candidate_city_extinction_rate"] - baseline_metrics["candidate_city_extinction_rate"]
    )
    map32_score_delta = float(candidate_map32_score - baseline_map32_score)
    reasons = []
    if score_delta < min_score_delta:
        reasons.append(f"score_delta={score_delta:.6f} < {min_score_delta:.6f}")
    if extinction_delta > max_city_extinction_delta:
        reasons.append(f"city_extinction_delta={extinction_delta:.6f} > {max_city_extinction_delta:.6f}")
    if map32_score_delta < min_map32_score_delta:
        reasons.append(f"map32_score_delta={map32_score_delta:.6f} < {min_map32_score_delta:.6f}")
    return {
        "passed": not reasons,
        "score_delta": score_delta,
        "city_extinction_delta": extinction_delta,
        "map32_score_delta": map32_score_delta,
        "reasons": reasons,
    }


def select_winning_arm(arms: dict[str, dict]) -> str | None:
    # Prefer Arm A (the weaker KL anchor) when every measured criterion ties.
    passing = sorted(
        ((name, arm) for name, arm in arms.items() if arm["decision"]["passed"]),
        key=lambda item: item[0],
    )
    if not passing:
        return None
    return max(
        passing,
        key=lambda item: (
            item[1]["decision"]["score_delta"],
            item[1]["decision"]["map32_score_delta"],
            -item[1]["decision"]["city_extinction_delta"],
        ),
    )[0]


def final_promotion_decision(
    initial_teacher: dict,
    candidate_teacher: dict,
    initial_base: dict,
    candidate_base: dict,
    *,
    teacher_name: str,
    base_name: str,
    paired_difference_lcb95: float,
    map32_score_delta: float,
    min_teacher_score_delta: float = 0.025,
    min_paired_difference_lcb95: float = -0.025,
    min_city_survival: float = 0.73,
    max_city_extinction: float = 0.10,
    min_nonregression_delta: float = -0.025,
    min_map32_score_delta: float = 0.0,
) -> dict:
    initial_teacher_metrics = _opponent_metrics(initial_teacher, teacher_name)
    candidate_teacher_metrics = _opponent_metrics(candidate_teacher, teacher_name)
    initial_base_metrics = _opponent_metrics(initial_base, base_name)
    candidate_base_metrics = _opponent_metrics(candidate_base, base_name)
    teacher_delta = float(candidate_teacher_metrics["score_rate"] - initial_teacher_metrics["score_rate"])
    base_delta = float(candidate_base_metrics["score_rate"] - initial_base_metrics["score_rate"])
    city_survival = float(candidate_teacher_metrics["candidate_city_survival"])
    city_extinction = float(candidate_teacher_metrics["candidate_city_extinction_rate"])
    reasons = []
    if teacher_delta < min_teacher_score_delta:
        reasons.append(f"teacher_score_delta={teacher_delta:.6f} < {min_teacher_score_delta:.6f}")
    if paired_difference_lcb95 < min_paired_difference_lcb95:
        reasons.append(f"paired_difference_lcb95={paired_difference_lcb95:.6f} < {min_paired_difference_lcb95:.6f}")
    if city_survival < min_city_survival:
        reasons.append(f"city_survival={city_survival:.6f} < {min_city_survival:.6f}")
    if city_extinction > max_city_extinction:
        reasons.append(f"city_extinction={city_extinction:.6f} > {max_city_extinction:.6f}")
    if teacher_delta < min_nonregression_delta:
        reasons.append(f"teacher_nonregression_delta={teacher_delta:.6f} < {min_nonregression_delta:.6f}")
    if base_delta < min_nonregression_delta:
        reasons.append(f"base_score_delta={base_delta:.6f} < {min_nonregression_delta:.6f}")
    if map32_score_delta < min_map32_score_delta:
        reasons.append(f"map32_score_delta={map32_score_delta:.6f} < {min_map32_score_delta:.6f}")
    return {
        "passed": not reasons,
        "teacher_score_delta": teacher_delta,
        "base_score_delta": base_delta,
        "paired_difference_lcb95": paired_difference_lcb95,
        "city_survival": city_survival,
        "city_extinction": city_extinction,
        "map32_score_delta": map32_score_delta,
        "reasons": reasons,
    }


def _evaluate(
    checkpoint: Path,
    opponent: Path,
    output_dir: Path,
    *,
    config: Path,
    opponent_name: str,
    seed_start: int,
    seeds: int,
    args: argparse.Namespace,
    backend: str,
) -> dict:
    return evaluate_checkpoint(
        checkpoint,
        opponent,
        output_dir,
        config=config,
        opponent_name=opponent_name,
        seed_start=seed_start,
        seeds=seeds,
        map_sizes=tuple(args.map_sizes),
        python=args.engine_python,
        timeout=args.timeout,
        bootstrap_samples=args.bootstrap_samples,
        workers=args.workers,
        backend=backend,
        device=args.eval_device,
        batch_games=args.eval_batch_games,
        parity_games=args.parity_games,
    )


def _train_and_evaluate_stage(
    *,
    config_name: str,
    stage_dir: Path,
    evaluation_dir: Path,
    target_step: int,
    load_checkpoint: Path,
    weights_only: bool,
    baseline: dict,
    baseline_map32_score: float,
    backend: str,
    args: argparse.Namespace,
) -> dict:
    run_training_segment(
        config_name=config_name,
        run_dir=stage_dir,
        stop_after_step=target_step,
        total_steps=max(args.extension_steps),
        load_checkpoint=load_checkpoint,
        weights_only=weights_only,
        python=args.python,
    )
    checkpoint = full_checkpoint(stage_dir)
    model_update_max_abs = checkpoint_model_max_abs_diff(load_checkpoint, checkpoint)
    if model_update_max_abs == 0.0:
        return {
            "target_step": target_step,
            "checkpoint": str(checkpoint),
            "model_update_max_abs": model_update_max_abs,
            "decision": {"passed": False, "reasons": ["checkpoint policy is identical to segment input"]},
        }
    evaluation = _evaluate(
        checkpoint,
        args.teacher_opponent,
        evaluation_dir,
        config=stage_dir / "config.yaml",
        opponent_name=args.teacher_name,
        seed_start=args.gate_seed_start,
        seeds=args.gate_seeds,
        args=args,
        backend=backend,
    )
    candidate_map32_score = map_score_rate(Path(evaluation["games"]), 32)
    decision = stage_gate_decision(
        baseline,
        evaluation["summary"],
        opponent_name=args.teacher_name,
        baseline_map32_score=baseline_map32_score,
        candidate_map32_score=candidate_map32_score,
        min_score_delta=args.min_score_delta,
        max_city_extinction_delta=args.max_city_extinction_delta,
        min_map32_score_delta=args.min_map32_score_delta,
    )
    return {
        "target_step": target_step,
        "checkpoint": str(checkpoint),
        "config": str(stage_dir / "config.yaml"),
        "model_update_max_abs": model_update_max_abs,
        "evaluation": evaluation["summary"],
        "evaluation_dir": str(evaluation_dir),
        "map32_score": candidate_map32_score,
        "decision": decision,
    }


def run_ab(args: argparse.Namespace) -> dict:
    initial_checkpoint = args.initial_checkpoint.expanduser().resolve()
    initial_config = (
        initial_checkpoint.parent / "config.yaml"
        if args.initial_config is None
        else args.initial_config.expanduser().resolve()
    )
    base_checkpoint = args.base_checkpoint.expanduser().resolve()
    base_config = (
        base_checkpoint.parent / "config.yaml" if args.base_config is None else args.base_config.expanduser().resolve()
    )
    teacher_opponent = args.teacher_opponent.expanduser().resolve()
    run_root = args.run_root.expanduser().resolve()
    for path in (initial_checkpoint, initial_config, base_checkpoint, base_config, teacher_opponent):
        if not path.is_file():
            raise FileNotFoundError(f"Missing required input: {path}")
    if args.pilot_steps <= 0 or tuple(args.extension_steps) != tuple(sorted(set(args.extension_steps))):
        raise ValueError("extension steps must be unique and increasing")
    if not args.extension_steps or args.extension_steps[0] <= args.pilot_steps:
        raise ValueError("extension steps must start after the pilot step")
    if 32 not in args.map_sizes:
        raise ValueError("map sizes must include 32 for the 32x32 quality gate")
    if args.final_seed_start < args.gate_seed_start + args.gate_seeds:
        raise ValueError("final seeds must not overlap the gate seeds")

    reused_baseline = None
    if args.reuse_baseline_evaluation is not None:
        reused_baseline = load_reused_baseline_evaluation(
            args.reuse_baseline_evaluation,
            opponent_name=args.teacher_name,
            seed_start=args.gate_seed_start,
            seeds=args.gate_seeds,
            map_sizes=tuple(args.map_sizes),
            eval_backend=args.eval_backend,
            bootstrap_samples=args.bootstrap_samples,
        )

    run_root.mkdir(parents=True, exist_ok=False)
    progress_path = run_root / "ab_progress.json"
    progress = {
        "schema_version": 1,
        "status": "evaluating_initial_gate_baseline",
        "initial_checkpoint": str(initial_checkpoint),
        "initial_config": str(initial_config),
        "arms": {},
    }
    write_progress(progress_path, progress)

    if reused_baseline is None:
        baseline_result = _evaluate(
            initial_checkpoint,
            teacher_opponent,
            run_root / "gate_initial_vs_teacher",
            config=initial_config,
            opponent_name=args.teacher_name,
            seed_start=args.gate_seed_start,
            seeds=args.gate_seeds,
            args=args,
            backend=args.eval_backend,
        )
        gate_backend = baseline_result["backend"]["selected"]
        baseline_source = str(run_root / "gate_initial_vs_teacher")
    else:
        baseline_result = reused_baseline
        gate_backend = reused_baseline["backend"]
        baseline_source = reused_baseline["source"]
    baseline_map32_score = map_score_rate(Path(baseline_result["games"]), 32)
    progress["gate_baseline"] = baseline_result["summary"]
    progress["gate_backend"] = gate_backend
    progress["gate_baseline_map32_score"] = baseline_map32_score
    progress["gate_baseline_evaluation"] = {
        "reused": reused_baseline is not None,
        "source": baseline_source,
        "backend": gate_backend,
    }
    progress["status"] = "training_pilots"
    write_progress(progress_path, progress)

    arm_configs = {"arm_a": args.arm_a_config_name, "arm_b": args.arm_b_config_name}
    for arm_name, config_name in arm_configs.items():
        stage_dir = run_root / arm_name / f"step_{args.pilot_steps:07d}"
        result = _train_and_evaluate_stage(
            config_name=config_name,
            stage_dir=stage_dir,
            evaluation_dir=run_root / arm_name / f"evaluation_step_{args.pilot_steps:07d}",
            target_step=args.pilot_steps,
            load_checkpoint=initial_checkpoint,
            weights_only=True,
            baseline=baseline_result["summary"],
            baseline_map32_score=baseline_map32_score,
            backend=gate_backend,
            args=args,
        )
        result["config_name"] = config_name
        progress["arms"][arm_name] = result
        write_progress(progress_path, progress)

    selected_arm = select_winning_arm(progress["arms"])
    progress["selected_arm"] = selected_arm
    if selected_arm is None:
        progress["status"] = "stopped_no_pilot_passed"
        write_progress(progress_path, progress)
        return progress

    selected = progress["arms"][selected_arm]
    selected["extensions"] = []
    load_checkpoint = Path(selected["checkpoint"])
    config_name = selected["config_name"]
    progress["status"] = "extending_winner"
    write_progress(progress_path, progress)
    for target_step in args.extension_steps:
        result = _train_and_evaluate_stage(
            config_name=config_name,
            stage_dir=run_root / selected_arm / f"step_{target_step:07d}",
            evaluation_dir=run_root / selected_arm / f"evaluation_step_{target_step:07d}",
            target_step=target_step,
            load_checkpoint=load_checkpoint,
            weights_only=False,
            baseline=baseline_result["summary"],
            baseline_map32_score=baseline_map32_score,
            backend=gate_backend,
            args=args,
        )
        selected["extensions"].append(result)
        write_progress(progress_path, progress)
        if not result["decision"]["passed"]:
            progress["status"] = "stopped_winner_failed_extension_gate"
            write_progress(progress_path, progress)
            return progress
        load_checkpoint = Path(result["checkpoint"])

    progress["status"] = "running_final_unused_seed_gate"
    write_progress(progress_path, progress)
    candidate_checkpoint = load_checkpoint
    candidate_config = Path(selected["extensions"][-1]["config"])
    base_bundle = prepare_eval_agent(
        base_checkpoint,
        run_root / "base_opponent_agent",
        base_config,
    )
    base_opponent = Path(base_bundle["agent"])

    initial_teacher = _evaluate(
        initial_checkpoint,
        teacher_opponent,
        run_root / "final_initial_vs_teacher",
        config=initial_config,
        opponent_name=args.teacher_name,
        seed_start=args.final_seed_start,
        seeds=args.final_seeds,
        args=args,
        backend=args.eval_backend,
    )
    final_teacher_backend = initial_teacher["backend"]["selected"]
    candidate_teacher = _evaluate(
        candidate_checkpoint,
        teacher_opponent,
        run_root / "final_candidate_vs_teacher",
        config=candidate_config,
        opponent_name=args.teacher_name,
        seed_start=args.final_seed_start,
        seeds=args.final_seeds,
        args=args,
        backend=final_teacher_backend,
    )
    initial_base = _evaluate(
        initial_checkpoint,
        base_opponent,
        run_root / "final_initial_vs_base",
        config=initial_config,
        opponent_name=args.base_name,
        seed_start=args.final_seed_start,
        seeds=args.final_seeds,
        args=args,
        backend=args.eval_backend,
    )
    final_base_backend = initial_base["backend"]["selected"]
    candidate_base = _evaluate(
        candidate_checkpoint,
        base_opponent,
        run_root / "final_candidate_vs_base",
        config=candidate_config,
        opponent_name=args.base_name,
        seed_start=args.final_seed_start,
        seeds=args.final_seeds,
        args=args,
        backend=final_base_backend,
    )
    paired_lcb = paired_score_difference_lcb(
        Path(initial_teacher["games"]),
        Path(candidate_teacher["games"]),
        bootstrap_samples=args.bootstrap_samples,
    )
    final_map32_delta = map_score_rate(Path(candidate_teacher["games"]), 32) - map_score_rate(
        Path(initial_teacher["games"]), 32
    )
    decision = final_promotion_decision(
        initial_teacher["summary"],
        candidate_teacher["summary"],
        initial_base["summary"],
        candidate_base["summary"],
        teacher_name=args.teacher_name,
        base_name=args.base_name,
        paired_difference_lcb95=paired_lcb,
        map32_score_delta=final_map32_delta,
        min_teacher_score_delta=args.final_min_teacher_score_delta,
        min_paired_difference_lcb95=args.final_min_paired_lcb,
        min_city_survival=args.final_min_city_survival,
        max_city_extinction=args.final_max_city_extinction,
        min_nonregression_delta=args.final_min_nonregression_delta,
        min_map32_score_delta=args.min_map32_score_delta,
    )
    progress["final"] = {
        "candidate_checkpoint": str(candidate_checkpoint),
        "candidate_config": str(candidate_config),
        "initial_vs_teacher": initial_teacher["summary"],
        "candidate_vs_teacher": candidate_teacher["summary"],
        "initial_vs_base": initial_base["summary"],
        "candidate_vs_base": candidate_base["summary"],
        "decision": decision,
    }
    progress["status"] = "promoted" if decision["passed"] else "completed_not_promoted"
    write_progress(progress_path, progress)
    return progress


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run two conservative 25k pilots, extend only the winner, and apply unused-seed gates."
    )
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--initial-config", type=Path)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--base-config", type=Path)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--reuse-baseline-evaluation",
        type=Path,
        help=(
            "Reuse an initial-vs-teacher evaluation after validating games.jsonl/report.json, "
            "opponent, gate seeds, map sizes, orientations, and backend."
        ),
    )
    parser.add_argument("--arm-a-config-name", default="survival_strategic_strength_v5_lr5e7_kl001_2gpu")
    parser.add_argument("--arm-b-config-name", default="survival_strategic_strength_v5_lr5e7_kl002_2gpu")
    parser.add_argument("--pilot-steps", type=int, default=25_000)
    parser.add_argument("--extension-steps", type=int, nargs="+", default=(50_000, 100_000))
    parser.add_argument("--teacher-opponent", type=Path, default=FIRST_PLACE_AGENT)
    parser.add_argument("--teacher-name", default="first_place")
    parser.add_argument("--base-name", default="base_0100096")
    parser.add_argument("--gate-seed-start", type=int, default=2021)
    parser.add_argument("--gate-seeds", type=int, default=5)
    parser.add_argument("--final-seed-start", type=int, default=12021)
    parser.add_argument("--final-seeds", type=int, default=10)
    parser.add_argument("--map-sizes", type=int, nargs="+", default=(12, 16, 24, 32))
    parser.add_argument("--min-score-delta", type=float, default=-0.025)
    parser.add_argument("--max-city-extinction-delta", type=float, default=0.025)
    parser.add_argument("--min-map32-score-delta", type=float, default=0.0)
    parser.add_argument("--final-min-teacher-score-delta", type=float, default=0.025)
    parser.add_argument("--final-min-paired-lcb", type=float, default=-0.025)
    parser.add_argument("--final-min-city-survival", type=float, default=0.73)
    parser.add_argument("--final-max-city-extinction", type=float, default=0.10)
    parser.add_argument("--final-min-nonregression-delta", type=float, default=-0.025)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--engine-python", default=sys.executable)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--eval-backend", choices=("auto", "official", "internal"), default="auto")
    parser.add_argument("--eval-device", default="auto")
    parser.add_argument("--eval-batch-games", type=int, default=8)
    parser.add_argument("--parity-games", type=int, default=4)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    return parser.parse_args()


def main() -> None:
    try:
        result = run_ab(parse_args())
    except (FileExistsError, FileNotFoundError, RuntimeError, subprocess.CalledProcessError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
