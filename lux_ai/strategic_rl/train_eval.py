from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import torch

from .evaluate import load_records, summarize
from .evaluate_checkpoint import FIRST_PLACE_AGENT, evaluate_checkpoint
from .prepare_eval_agent import ROOT


def promotion_decision(
    baseline: dict,
    candidate: dict,
    *,
    opponent_name: str,
    min_score_delta: float,
    max_city_extinction_delta: float,
) -> dict:
    baseline_metrics = baseline["opponents"][opponent_name]
    candidate_metrics = candidate["opponents"][opponent_name]
    score_delta = float(candidate_metrics["score_rate"] - baseline_metrics["score_rate"])
    extinction_delta = float(
        candidate_metrics["candidate_city_extinction_rate"]
        - baseline_metrics["candidate_city_extinction_rate"]
    )
    reasons = []
    if score_delta < min_score_delta:
        reasons.append(f"score_delta={score_delta:.6f} < {min_score_delta:.6f}")
    if extinction_delta > max_city_extinction_delta:
        reasons.append(
            f"city_extinction_delta={extinction_delta:.6f} > {max_city_extinction_delta:.6f}"
        )
    return {
        "passed": not reasons,
        "score_delta": score_delta,
        "city_extinction_delta": extinction_delta,
        "reasons": reasons,
    }


def full_checkpoint(run_dir: Path) -> Path:
    checkpoints = []
    for path in run_dir.glob("*.pt"):
        if path.name.endswith("_weights.pt"):
            continue
        try:
            state = torch.load(path, map_location="cpu", weights_only=False)
        except (OSError, RuntimeError, ValueError):
            continue
        if isinstance(state, dict) and "optimizer_state_dict" in state and "step" in state:
            checkpoints.append((int(state["step"]), path))
    if not checkpoints:
        raise FileNotFoundError(f"No full training checkpoint found in {run_dir}")
    return max(checkpoints)[1]


def checkpoint_model_max_abs_diff(left_path: Path, right_path: Path) -> float:
    """Compare the policy tensors stored in two checkpoints."""
    left = torch.load(left_path, map_location="cpu", weights_only=False)["model_state_dict"]
    right = torch.load(right_path, map_location="cpu", weights_only=False)["model_state_dict"]
    if left.keys() != right.keys():
        raise ValueError("Cannot compare checkpoints with different model-state keys")
    maximum = 0.0
    for name, left_tensor in left.items():
        right_tensor = right[name]
        if left_tensor.shape != right_tensor.shape or left_tensor.dtype != right_tensor.dtype:
            raise ValueError(f"Cannot compare incompatible checkpoint tensor: {name}")
        if torch.is_floating_point(left_tensor) or torch.is_complex(left_tensor):
            difference = (left_tensor - right_tensor).abs()
            if difference.numel():
                maximum = max(maximum, float(difference.max().item()))
        elif not torch.equal(left_tensor, right_tensor):
            return float("inf")
    return maximum


def write_progress(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_reused_baseline_evaluation(
    evaluation_dir: Path,
    *,
    opponent_name: str,
    seed_start: int,
    seeds: int,
    map_sizes: tuple[int, ...],
    eval_backend: str,
    bootstrap_samples: int,
) -> dict:
    """Validate and load an existing matched baseline without rerunning games."""
    evaluation_dir = evaluation_dir.expanduser().resolve()
    games_path = evaluation_dir / "games.jsonl"
    report_path = evaluation_dir / "report.json"
    if not games_path.is_file() or not report_path.is_file():
        raise FileNotFoundError(
            f"Reused baseline requires games.jsonl and report.json: {evaluation_dir}"
        )

    records = load_records(games_path)
    expected = {
        (seed, map_size, candidate_player)
        for seed in range(seed_start, seed_start + seeds)
        for map_size in map_sizes
        for candidate_player in (0, 1)
    }
    actual = []
    for record in records:
        if record["opponent"] != opponent_name:
            raise ValueError(
                f"Reused baseline opponent mismatch: expected {opponent_name}, "
                f"found {record['opponent']}"
            )
        if "map_size" not in record or "candidate_final_city_tiles" not in record:
            raise ValueError("Reused baseline lacks map_size or city-extinction diagnostics")
        actual.append(
            (int(record["seed"]), int(record["map_size"]), int(record["candidate_player"]))
        )
    if len(actual) != len(set(actual)):
        raise ValueError("Reused baseline contains duplicate seed/map/orientation games")
    actual_set = set(actual)
    if actual_set != expected:
        missing = sorted(expected - actual_set)
        unexpected = sorted(actual_set - expected)
        raise ValueError(
            "Reused baseline schedule does not match the requested evaluation: "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}"
        )

    record_backends = {
        "internal" if record.get("backend") == "internal_batched" else "official"
        for record in records
    }
    if len(record_backends) != 1:
        raise ValueError(f"Reused baseline mixes evaluation backends: {sorted(record_backends)}")
    selected_backend = next(iter(record_backends))
    if eval_backend != "auto" and selected_backend != eval_backend:
        raise ValueError(
            f"Reused baseline backend is {selected_backend}, but --eval-backend is {eval_backend}"
        )

    summary = summarize(records, bootstrap_samples=bootstrap_samples)
    saved_report = json.loads(report_path.read_text(encoding="utf-8"))
    for key in ("matched_pairs", "score_rate", "candidate_city_extinction_rate"):
        saved = saved_report.get("opponents", {}).get(opponent_name, {}).get(key)
        recomputed = summary["opponents"][opponent_name][key]
        if saved != recomputed:
            raise ValueError(
                f"Reused baseline report does not match games.jsonl for {key}: "
                f"report={saved}, recomputed={recomputed}"
            )
    return {
        "summary": summary,
        "source": str(evaluation_dir),
        "backend": selected_backend,
        "games": str(games_path),
        "report": str(report_path),
    }


def run_training_segment(
    *,
    config_name: str,
    run_dir: Path,
    stop_after_step: int,
    total_steps: int,
    load_checkpoint: Path,
    weights_only: bool,
    python: str,
) -> None:
    command = [
        python,
        str(ROOT / "run_monobeast.py"),
        f"--config-name={config_name}",
        f"hydra.run.dir={run_dir}",
        f"+load_dir={load_checkpoint.parent}",
        f"+checkpoint_file={load_checkpoint.name}",
        f"weights_only={'true' if weights_only else 'false'}",
        f"total_steps={total_steps}",
        f"stop_after_step={stop_after_step}",
    ]
    subprocess.run(command, cwd=ROOT, check=True)


def run_train_eval(args: argparse.Namespace) -> dict:
    run_root = args.run_root.expanduser().resolve()
    base_checkpoint = args.base_checkpoint.expanduser().resolve()
    if not base_checkpoint.is_file():
        raise FileNotFoundError(f"Missing base checkpoint: {base_checkpoint}")
    milestones = sorted(set(args.milestones))
    if not milestones or milestones[-1] > args.total_steps or milestones[0] <= 0:
        raise ValueError("milestones must be positive and no larger than total_steps")

    if args.reuse_baseline_evaluation is not None:
        baseline_result = load_reused_baseline_evaluation(
            args.reuse_baseline_evaluation,
            opponent_name=args.opponent_name,
            seed_start=args.seed_start,
            seeds=args.seeds,
            map_sizes=tuple(args.map_sizes),
            eval_backend=args.eval_backend,
            bootstrap_samples=args.bootstrap_samples,
        )
    else:
        baseline_result = None
    milestone_eval_backend = (
        baseline_result["backend"]
        if baseline_result is not None and args.eval_backend == "auto"
        else args.eval_backend
    )

    run_root.mkdir(parents=True, exist_ok=False)
    progress_path = run_root / "evaluation_progress.json"
    if baseline_result is None:
        baseline_result = evaluate_checkpoint(
            base_checkpoint,
            args.opponent,
            run_root / "baseline_evaluation",
            opponent_name=args.opponent_name,
            seed_start=args.seed_start,
            seeds=args.seeds,
            map_sizes=tuple(args.map_sizes),
            python=args.engine_python,
            timeout=args.timeout,
            bootstrap_samples=args.bootstrap_samples,
            workers=args.workers,
            backend=args.eval_backend,
            device=args.eval_device,
            batch_games=args.eval_batch_games,
            parity_games=args.parity_games,
        )
    progress = {
        "schema_version": 2,
        "status": "running",
        "config_name": args.config_name,
        "base_checkpoint": str(base_checkpoint),
        "baseline": baseline_result["summary"],
        "baseline_evaluation": {
            "reused": args.reuse_baseline_evaluation is not None,
            "source": baseline_result.get("source", str(run_root / "baseline_evaluation")),
            "backend": baseline_result.get("backend"),
        },
        "milestones": [],
    }
    write_progress(progress_path, progress)

    load_checkpoint = base_checkpoint
    weights_only = True
    for target in milestones:
        stage_dir = run_root / f"step_{target:07d}"
        run_training_segment(
            config_name=args.config_name,
            run_dir=stage_dir,
            stop_after_step=target,
            total_steps=args.total_steps,
            load_checkpoint=load_checkpoint,
            weights_only=weights_only,
            python=args.python,
        )
        checkpoint = full_checkpoint(stage_dir)
        model_update_max_abs = checkpoint_model_max_abs_diff(load_checkpoint, checkpoint)
        if model_update_max_abs == 0.0:
            progress["milestones"].append(
                {
                    "target_step": target,
                    "checkpoint": str(checkpoint),
                    "model_update_max_abs": model_update_max_abs,
                    "decision": {
                        "passed": False,
                        "reasons": ["checkpoint policy is identical to the segment input"],
                    },
                }
            )
            progress["status"] = "stopped_by_model_integrity_gate"
            write_progress(progress_path, progress)
            return progress
        evaluation = evaluate_checkpoint(
            checkpoint,
            args.opponent,
            run_root / f"evaluation_step_{target:07d}",
            config=stage_dir / "config.yaml",
            opponent_name=args.opponent_name,
            seed_start=args.seed_start,
            seeds=args.seeds,
            map_sizes=tuple(args.map_sizes),
            python=args.engine_python,
            timeout=args.timeout,
            bootstrap_samples=args.bootstrap_samples,
            workers=args.workers,
            backend=milestone_eval_backend,
            device=args.eval_device,
            batch_games=args.eval_batch_games,
            parity_games=args.parity_games,
        )
        decision = promotion_decision(
            baseline_result["summary"],
            evaluation["summary"],
            opponent_name=args.opponent_name,
            min_score_delta=args.min_score_delta,
            max_city_extinction_delta=args.max_city_extinction_delta,
        )
        progress["milestones"].append(
            {
                "target_step": target,
                "checkpoint": str(checkpoint),
                "model_update_max_abs": model_update_max_abs,
                "evaluation": evaluation["summary"],
                "decision": decision,
            }
        )
        write_progress(progress_path, progress)
        if not decision["passed"] and not args.continue_on_fail:
            progress["status"] = "stopped_by_quality_gate"
            write_progress(progress_path, progress)
            return progress
        load_checkpoint = checkpoint
        weights_only = False

    progress["status"] = "completed"
    write_progress(progress_path, progress)
    return progress


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train in full-resume segments and run matched first-place gates between segments."
    )
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--reuse-baseline-evaluation",
        type=Path,
        help=(
            "Reuse an existing baseline_evaluation directory after validating its "
            "opponent, seeds, map sizes, orientations, backend, and report."
        ),
    )
    parser.add_argument("--config-name", default="survival_strategic_strength_v3")
    parser.add_argument("--total-steps", type=int, default=1_000_000)
    parser.add_argument("--milestones", type=int, nargs="+", default=(250_000, 500_000, 750_000, 1_000_000))
    parser.add_argument("--opponent", type=Path, default=FIRST_PLACE_AGENT)
    parser.add_argument("--opponent-name", default="first_place")
    parser.add_argument("--seed-start", type=int, default=2021)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--map-sizes", type=int, nargs="+", default=(12, 16, 24, 32))
    parser.add_argument("--min-score-delta", type=float, default=-0.05)
    parser.add_argument("--max-city-extinction-delta", type=float, default=0.05)
    parser.add_argument("--python", default=sys.executable, help="Python used for run_monobeast.py.")
    parser.add_argument("--engine-python", default=sys.executable, help="Python passed to lux-ai-2021 agents.")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--workers", type=int, default=2, help="Concurrent evaluation matches.")
    parser.add_argument("--eval-backend", choices=("auto", "official", "internal"), default="auto")
    parser.add_argument("--eval-device", default="auto")
    parser.add_argument("--eval-batch-games", type=int, default=8)
    parser.add_argument("--parity-games", type=int, default=4)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--continue-on-fail", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        result = run_train_eval(args)
    except (FileExistsError, FileNotFoundError, RuntimeError, subprocess.CalledProcessError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
