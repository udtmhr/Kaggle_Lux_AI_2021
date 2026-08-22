from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from .evaluate import candidate_score, load_records, summarize
from .evaluate_checkpoint import FIRST_PLACE_AGENT, evaluate_checkpoint
from .prepare_eval_agent import ROOT


def promotion_decision(
    baseline: dict,
    candidate: dict,
    *,
    opponent_name: str,
    min_score_delta: float,
    max_city_extinction_delta: float,
    slice_deltas: dict[str, dict] | None = None,
    min_slice_score_delta: float | None = None,
    max_slice_city_extinction_delta: float | None = None,
) -> dict:
    baseline_metrics = baseline["opponents"][opponent_name]
    candidate_metrics = candidate["opponents"][opponent_name]
    score_delta = float(candidate_metrics["score_rate"] - baseline_metrics["score_rate"])
    extinction_delta = float(
        candidate_metrics["candidate_city_extinction_rate"] - baseline_metrics["candidate_city_extinction_rate"]
    )
    reasons = []
    if score_delta < min_score_delta:
        reasons.append(f"score_delta={score_delta:.6f} < {min_score_delta:.6f}")
    if extinction_delta > max_city_extinction_delta:
        reasons.append(f"city_extinction_delta={extinction_delta:.6f} > {max_city_extinction_delta:.6f}")
    slice_deltas = slice_deltas or {}
    for label, metrics in sorted(slice_deltas.items()):
        if min_slice_score_delta is not None and metrics["score_delta"] < min_slice_score_delta:
            reasons.append(f"slice_{label}_score_delta={metrics['score_delta']:.6f} < {min_slice_score_delta:.6f}")
        if (
            max_slice_city_extinction_delta is not None
            and metrics["city_extinction_delta"] > max_slice_city_extinction_delta
        ):
            reasons.append(
                f"slice_{label}_city_extinction_delta={metrics['city_extinction_delta']:.6f} "
                f"> {max_slice_city_extinction_delta:.6f}"
            )
    return {
        "passed": not reasons,
        "score_delta": score_delta,
        "city_extinction_delta": extinction_delta,
        "slice_deltas": slice_deltas,
        "reasons": reasons,
    }


def paired_score_difference_lcb(
    baseline_games: Path,
    candidate_games: Path,
    *,
    bootstrap_samples: int = 2000,
    seed: int = 2021,
) -> float:
    """Bootstrap the lower 95% bound of matched candidate-minus-baseline scores."""
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")

    def paired_scores(path: Path) -> dict[tuple[str, int, int], float]:
        grouped: dict[tuple[str, int, int], list[dict]] = {}
        for record in load_records(path):
            key = (
                str(record["opponent"]),
                int(record["seed"]),
                int(record.get("map_size", -1)),
            )
            grouped.setdefault(key, []).append(record)
        result = {}
        for key, records in grouped.items():
            if len(records) != 2 or {int(record["candidate_player"]) for record in records} != {
                0,
                1,
            }:
                raise ValueError(f"Incomplete or duplicate orientation pair for {key} in {path}")
            result[key] = float(np.mean([candidate_score(record) for record in records]))
        return result

    baseline = paired_scores(baseline_games)
    candidate = paired_scores(candidate_games)
    if not baseline or baseline.keys() != candidate.keys():
        raise ValueError("Baseline and candidate evaluation schedules differ")
    differences = np.asarray([candidate[key] - baseline[key] for key in sorted(baseline)])
    rng = np.random.default_rng(seed)
    boot = np.asarray(
        [rng.choice(differences, len(differences), replace=True).mean() for _ in range(bootstrap_samples)]
    )
    return float(np.quantile(boot, 0.025))


def evaluation_metrics_by_map(games: Path, *, opponent_name: str) -> dict[int, dict[str, float]]:
    """Aggregate promotion diagnostics per map size from matched game records."""
    grouped: dict[int, list[dict]] = {}
    for record in load_records(games):
        if str(record.get("opponent")) != opponent_name:
            continue
        if "candidate_final_city_tiles" not in record:
            raise ValueError(f"Map promotion gates require city diagnostics in {games}")
        grouped.setdefault(int(record.get("map_size", -1)), []).append(record)
    if not grouped:
        raise ValueError(f"No games for opponent {opponent_name!r} in {games}")
    return {
        map_size: {
            "score_rate": float(np.mean([candidate_score(record) for record in records])),
            "city_extinction_rate": float(
                np.mean([int(record["candidate_final_city_tiles"]) == 0 for record in records])
            ),
        }
        for map_size, records in sorted(grouped.items())
    }


def map_metric_deltas(
    baseline_games: Path,
    candidate_games: Path,
    *,
    opponent_name: str,
) -> dict[int, dict[str, float]]:
    """Return candidate-minus-baseline score and city-extinction deltas per map."""
    baseline = evaluation_metrics_by_map(baseline_games, opponent_name=opponent_name)
    candidate = evaluation_metrics_by_map(candidate_games, opponent_name=opponent_name)
    if baseline.keys() != candidate.keys():
        raise ValueError("Baseline and candidate map schedules differ")
    return {
        map_size: {
            "score_delta": float(candidate[map_size]["score_rate"] - baseline[map_size]["score_rate"]),
            "city_extinction_delta": float(
                candidate[map_size]["city_extinction_rate"] - baseline[map_size]["city_extinction_rate"]
            ),
        }
        for map_size in baseline
    }


def parse_required_slice(value: str) -> dict[str, int]:
    """Parse map=24, player=1, or their comma-separated intersection."""
    aliases = {
        "map": "map_size",
        "map_size": "map_size",
        "player": "candidate_player",
        "candidate_player": "candidate_player",
    }
    result: dict[str, int] = {}
    for item in value.split(","):
        key, separator, raw = item.strip().partition("=")
        if not separator or key not in aliases:
            raise argparse.ArgumentTypeError("required slices use map=SIZE and/or player=0|1")
        canonical = aliases[key]
        try:
            parsed = int(raw)
        except ValueError as error:
            raise argparse.ArgumentTypeError(f"invalid required slice value: {item!r}") from error
        if canonical == "map_size" and parsed not in {12, 16, 24, 32}:
            raise argparse.ArgumentTypeError("slice map must be one of 12, 16, 24, 32")
        if canonical == "candidate_player" and parsed not in {0, 1}:
            raise argparse.ArgumentTypeError("slice player must be 0 or 1")
        result[canonical] = parsed
    if not result:
        raise argparse.ArgumentTypeError("required slice cannot be empty")
    return result


def _slice_label(filters: dict[str, int]) -> str:
    parts = []
    if "map_size" in filters:
        parts.append(f"map_{filters['map_size']}")
    if "candidate_player" in filters:
        parts.append(f"player_{filters['candidate_player']}")
    return "_".join(parts)


def _slice_metrics(
    games: Path,
    *,
    opponent_name: str,
    filters: dict[str, int],
) -> tuple[set[tuple[int, int, int]], dict[str, float]]:
    selected = []
    keys = set()
    for record in load_records(games):
        if str(record.get("opponent")) != opponent_name:
            continue
        if any(int(record[field]) != expected for field, expected in filters.items()):
            continue
        if "candidate_final_city_tiles" not in record:
            raise ValueError(f"Slice gates require city diagnostics in {games}")
        key = (int(record["seed"]), int(record["map_size"]), int(record["candidate_player"]))
        if key in keys:
            raise ValueError(f"Duplicate slice game {key} in {games}")
        keys.add(key)
        selected.append(record)
    if not selected:
        raise ValueError(f"No games for required slice {_slice_label(filters)} in {games}")
    return keys, {
        "score_rate": float(np.mean([candidate_score(record) for record in selected])),
        "city_survival": float(np.mean([record["candidate_city_survival"] for record in selected])),
        "city_extinction_rate": float(np.mean([int(record["candidate_final_city_tiles"]) == 0 for record in selected])),
        "games": len(selected),
    }


def slice_metric_deltas(
    baseline_games: Path,
    candidate_games: Path,
    *,
    opponent_name: str,
    required_slices: list[dict[str, int]],
) -> dict[str, dict]:
    """Return matched candidate-minus-baseline metrics for mandatory slices."""
    result = {}
    for filters in required_slices:
        label = _slice_label(filters)
        baseline_keys, baseline = _slice_metrics(baseline_games, opponent_name=opponent_name, filters=filters)
        candidate_keys, candidate = _slice_metrics(candidate_games, opponent_name=opponent_name, filters=filters)
        if baseline_keys != candidate_keys:
            raise ValueError(f"Baseline and candidate schedules differ for required slice {label}")
        result[label] = {
            "filters": filters,
            "games": candidate["games"],
            "score_delta": float(candidate["score_rate"] - baseline["score_rate"]),
            "city_survival_delta": float(candidate["city_survival"] - baseline["city_survival"]),
            "city_extinction_delta": float(candidate["city_extinction_rate"] - baseline["city_extinction_rate"]),
        }
    return result


def final_promotion_decision(
    baseline: dict,
    candidate: dict,
    *,
    opponent_name: str,
    paired_score_delta_lcb95: float,
    min_paired_score_delta_lcb95: float,
    min_city_survival_delta: float,
    max_city_extinction_delta: float,
    map_deltas: dict[int, dict[str, float]] | None = None,
    min_map_score_delta: float | None = None,
    max_map_city_extinction_delta: float | None = None,
    slice_deltas: dict[str, dict] | None = None,
    min_slice_score_delta: float | None = None,
    max_slice_city_extinction_delta: float | None = None,
) -> dict:
    """Require confident score improvement and non-regressing city safety."""
    baseline_metrics = baseline["opponents"][opponent_name]
    candidate_metrics = candidate["opponents"][opponent_name]
    baseline_survival = baseline_metrics.get("candidate_city_survival")
    candidate_survival = candidate_metrics.get("candidate_city_survival")
    if baseline_survival is None or candidate_survival is None:
        raise ValueError("Final promotion requires candidate_city_survival diagnostics")
    city_survival_delta = float(candidate_survival - baseline_survival)
    city_extinction_delta = float(
        candidate_metrics["candidate_city_extinction_rate"] - baseline_metrics["candidate_city_extinction_rate"]
    )
    reasons = []
    # The contract is strictly greater than the LCB threshold, not equal to it.
    if paired_score_delta_lcb95 <= min_paired_score_delta_lcb95:
        reasons.append(f"paired_score_delta_lcb95={paired_score_delta_lcb95:.6f} <= {min_paired_score_delta_lcb95:.6f}")
    if city_survival_delta < min_city_survival_delta:
        reasons.append(f"city_survival_delta={city_survival_delta:.6f} < {min_city_survival_delta:.6f}")
    if city_extinction_delta > max_city_extinction_delta:
        reasons.append(f"city_extinction_delta={city_extinction_delta:.6f} > {max_city_extinction_delta:.6f}")
    map_deltas = map_deltas or {}
    for map_size, metrics in sorted(map_deltas.items()):
        if min_map_score_delta is not None and metrics["score_delta"] < min_map_score_delta:
            reasons.append(f"map_{map_size}_score_delta={metrics['score_delta']:.6f} < {min_map_score_delta:.6f}")
        if (
            max_map_city_extinction_delta is not None
            and metrics["city_extinction_delta"] > max_map_city_extinction_delta
        ):
            reasons.append(
                f"map_{map_size}_city_extinction_delta={metrics['city_extinction_delta']:.6f} "
                f"> {max_map_city_extinction_delta:.6f}"
            )
    slice_deltas = slice_deltas or {}
    for label, metrics in sorted(slice_deltas.items()):
        if min_slice_score_delta is not None and metrics["score_delta"] < min_slice_score_delta:
            reasons.append(f"slice_{label}_score_delta={metrics['score_delta']:.6f} < {min_slice_score_delta:.6f}")
        if (
            max_slice_city_extinction_delta is not None
            and metrics["city_extinction_delta"] > max_slice_city_extinction_delta
        ):
            reasons.append(
                f"slice_{label}_city_extinction_delta={metrics['city_extinction_delta']:.6f} "
                f"> {max_slice_city_extinction_delta:.6f}"
            )
    return {
        "passed": not reasons,
        "paired_score_delta_lcb95": float(paired_score_delta_lcb95),
        "city_survival_delta": city_survival_delta,
        "city_extinction_delta": city_extinction_delta,
        "map_deltas": map_deltas,
        "slice_deltas": slice_deltas,
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
        raise FileNotFoundError(f"Reused baseline requires games.jsonl and report.json: {evaluation_dir}")

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
            raise ValueError(f"Reused baseline opponent mismatch: expected {opponent_name}, found {record['opponent']}")
        if "map_size" not in record or "candidate_final_city_tiles" not in record:
            raise ValueError("Reused baseline lacks map_size or city-extinction diagnostics")
        actual.append((int(record["seed"]), int(record["map_size"]), int(record["candidate_player"])))
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

    record_backends = {"internal" if record.get("backend") == "internal_batched" else "official" for record in records}
    if len(record_backends) != 1:
        raise ValueError(f"Reused baseline mixes evaluation backends: {sorted(record_backends)}")
    selected_backend = next(iter(record_backends))
    if eval_backend != "auto" and selected_backend != eval_backend:
        raise ValueError(f"Reused baseline backend is {selected_backend}, but --eval-backend is {eval_backend}")

    summary = summarize(records, bootstrap_samples=bootstrap_samples)
    saved_report = json.loads(report_path.read_text(encoding="utf-8"))
    for key in ("matched_pairs", "score_rate", "candidate_city_extinction_rate"):
        saved = saved_report.get("opponents", {}).get(opponent_name, {}).get(key)
        recomputed = summary["opponents"][opponent_name][key]
        if saved != recomputed:
            raise ValueError(
                f"Reused baseline report does not match games.jsonl for {key}: report={saved}, recomputed={recomputed}"
            )
    return {
        "summary": summary,
        "source": str(evaluation_dir),
        "backend": selected_backend,
        "games": str(games_path),
        "report": str(report_path),
    }


def subset_baseline_evaluation(
    baseline_result: dict,
    output_path: Path,
    *,
    opponent_name: str,
    seed_start: int,
    seeds: int,
    map_sizes: tuple[int, ...],
    bootstrap_samples: int,
) -> dict:
    """Materialize an exact screen schedule from a validated larger baseline."""
    required = {
        (seed, map_size, candidate_player)
        for seed in range(seed_start, seed_start + seeds)
        for map_size in map_sizes
        for candidate_player in (0, 1)
    }
    selected: dict[tuple[int, int, int], dict] = {}
    for record in load_records(Path(baseline_result["games"])):
        if str(record.get("opponent")) != opponent_name:
            continue
        key = (int(record["seed"]), int(record["map_size"]), int(record["candidate_player"]))
        if key not in required:
            continue
        if key in selected:
            raise ValueError(f"Duplicate baseline game for {key}")
        selected[key] = record
    if selected.keys() != required:
        raise ValueError(
            f"Validated baseline does not cover screen schedule: missing={sorted(required - selected.keys())[:5]}"
        )
    records = [selected[key] for key in sorted(required)]
    output_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    return {
        "summary": summarize(records, bootstrap_samples=bootstrap_samples),
        "source": baseline_result.get("source"),
        "backend": baseline_result.get("backend"),
        "games": str(output_path),
        "report": None,
        "subset_of": baseline_result.get("games"),
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
        f"++load_dir={load_checkpoint.parent}",
        f"++checkpoint_file={load_checkpoint.name}",
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
    final_seeds = args.final_seeds if args.final_seeds is not None else args.seeds
    if args.seeds <= 0 or final_seeds < args.seeds:
        raise ValueError("final_seeds must be at least seeds, and both must be positive")
    required_slices = list(args.required_slice or [])

    if args.reuse_baseline_evaluation is not None:
        final_baseline_result = load_reused_baseline_evaluation(
            args.reuse_baseline_evaluation,
            opponent_name=args.opponent_name,
            seed_start=args.seed_start,
            seeds=final_seeds,
            map_sizes=tuple(args.map_sizes),
            eval_backend=args.eval_backend,
            bootstrap_samples=args.bootstrap_samples,
        )
    else:
        final_baseline_result = None
    milestone_eval_backend = (
        final_baseline_result["backend"]
        if final_baseline_result is not None and args.eval_backend == "auto"
        else args.eval_backend
    )

    run_root.mkdir(parents=True, exist_ok=False)
    progress_path = run_root / "evaluation_progress.json"
    if final_baseline_result is None:
        final_baseline_result = evaluate_checkpoint(
            base_checkpoint,
            args.opponent,
            run_root / "baseline_evaluation",
            opponent_name=args.opponent_name,
            seed_start=args.seed_start,
            seeds=final_seeds,
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
        if args.eval_backend == "auto":
            milestone_eval_backend = final_baseline_result["backend"]["selected"]
    if final_seeds == args.seeds:
        baseline_result = final_baseline_result
    else:
        baseline_result = subset_baseline_evaluation(
            final_baseline_result,
            run_root / "screen_baseline_games.jsonl",
            opponent_name=args.opponent_name,
            seed_start=args.seed_start,
            seeds=args.seeds,
            map_sizes=tuple(args.map_sizes),
            bootstrap_samples=args.bootstrap_samples,
        )
    progress = {
        "schema_version": 4,
        "status": "running",
        "config_name": args.config_name,
        "base_checkpoint": str(base_checkpoint),
        "baseline": baseline_result["summary"],
        "baseline_evaluation": {
            "reused": args.reuse_baseline_evaluation is not None,
            "source": baseline_result.get("source", str(run_root / "baseline_evaluation")),
            "backend": baseline_result.get("backend"),
        },
        "final_baseline": final_baseline_result["summary"],
        "final_baseline_evaluation": {
            "reused": args.reuse_baseline_evaluation is not None,
            "source": final_baseline_result.get("source", str(run_root / "baseline_evaluation")),
            "backend": final_baseline_result.get("backend"),
        },
        "evaluation_schedule": {
            "screen_seeds": args.seeds,
            "final_seeds": final_seeds,
            "map_sizes": list(args.map_sizes),
            "required_slices": required_slices,
        },
        "milestones": [],
        "promotion": None,
    }
    write_progress(progress_path, progress)

    load_checkpoint = base_checkpoint
    weights_only = True
    for target in milestones:
        is_final = target == milestones[-1]
        active_baseline = final_baseline_result if is_final else baseline_result
        evaluation_seeds = final_seeds if is_final else args.seeds
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
            seeds=evaluation_seeds,
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
        candidate_games = Path(evaluation["games"])
        per_slice_deltas = slice_metric_deltas(
            Path(active_baseline["games"]),
            candidate_games,
            opponent_name=args.opponent_name,
            required_slices=required_slices,
        )
        decision = promotion_decision(
            active_baseline["summary"],
            evaluation["summary"],
            opponent_name=args.opponent_name,
            min_score_delta=args.min_score_delta,
            max_city_extinction_delta=args.max_city_extinction_delta,
            slice_deltas=per_slice_deltas,
            min_slice_score_delta=args.min_slice_score_delta,
            max_slice_city_extinction_delta=args.max_slice_city_extinction_delta,
        )
        progress["milestones"].append(
            {
                "target_step": target,
                "checkpoint": str(checkpoint),
                "model_update_max_abs": model_update_max_abs,
                "evaluation": evaluation["summary"],
                "evaluation_seeds": evaluation_seeds,
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

    final_target = milestones[-1]
    final_evaluation_dir = run_root / f"evaluation_step_{final_target:07d}"
    paired_lcb95 = paired_score_difference_lcb(
        Path(final_baseline_result["games"]),
        final_evaluation_dir / "games.jsonl",
        bootstrap_samples=args.bootstrap_samples,
    )
    per_map_deltas = map_metric_deltas(
        Path(final_baseline_result["games"]),
        final_evaluation_dir / "games.jsonl",
        opponent_name=args.opponent_name,
    )
    final_slice_deltas = slice_metric_deltas(
        Path(final_baseline_result["games"]),
        final_evaluation_dir / "games.jsonl",
        opponent_name=args.opponent_name,
        required_slices=required_slices,
    )
    progress["promotion"] = final_promotion_decision(
        final_baseline_result["summary"],
        progress["milestones"][-1]["evaluation"],
        opponent_name=args.opponent_name,
        paired_score_delta_lcb95=paired_lcb95,
        min_paired_score_delta_lcb95=args.min_paired_score_delta_lcb95,
        min_city_survival_delta=args.min_city_survival_delta,
        max_city_extinction_delta=args.max_promotion_city_extinction_delta,
        map_deltas=per_map_deltas,
        min_map_score_delta=args.min_map_score_delta,
        max_map_city_extinction_delta=args.max_map_city_extinction_delta,
        slice_deltas=final_slice_deltas,
        min_slice_score_delta=args.min_slice_score_delta,
        max_slice_city_extinction_delta=args.max_slice_city_extinction_delta,
    )
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
    parser.add_argument(
        "--final-seeds",
        type=int,
        help="Use this many seeds only at the final milestone; must be at least --seeds.",
    )
    parser.add_argument("--map-sizes", type=int, nargs="+", default=(12, 16, 24, 32))
    parser.add_argument("--min-score-delta", type=float, default=-0.05)
    parser.add_argument("--max-city-extinction-delta", type=float, default=0.05)
    parser.add_argument(
        "--min-paired-score-delta-lcb95",
        type=float,
        default=0.0,
        help="Final promotion requires the paired score-delta LCB95 to be strictly above this value.",
    )
    parser.add_argument(
        "--min-city-survival-delta",
        type=float,
        default=0.0,
        help="Minimum final candidate-minus-baseline city-survival delta.",
    )
    parser.add_argument(
        "--max-promotion-city-extinction-delta",
        type=float,
        default=0.0,
        help="Maximum final candidate-minus-baseline city-extinction delta.",
    )
    parser.add_argument(
        "--min-map-score-delta",
        type=float,
        help="Optional minimum final candidate-minus-baseline score delta required on every map size.",
    )
    parser.add_argument(
        "--max-map-city-extinction-delta",
        type=float,
        help="Optional maximum final city-extinction delta allowed on every map size.",
    )
    parser.add_argument(
        "--required-slice",
        type=parse_required_slice,
        action="append",
        default=[],
        help=("Mandatory evaluation slice, repeatable: map=24, player=1, or map=24,player=1."),
    )
    parser.add_argument(
        "--min-slice-score-delta",
        type=float,
        help="Minimum candidate-minus-baseline score delta on every required slice.",
    )
    parser.add_argument(
        "--max-slice-city-extinction-delta",
        type=float,
        help="Maximum city-extinction delta on every required slice.",
    )
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
