from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .evaluate import load_records, summarize
from .evaluate_checkpoint import FIRST_PLACE_AGENT, evaluate_checkpoint
from .train_eval import paired_score_difference_lcb


def _schedule_keys(seed_start: int, seeds: int, map_sizes: tuple[int, ...]) -> set[tuple[int, int, int]]:
    return {
        (seed, map_size, candidate_player)
        for seed in range(seed_start, seed_start + seeds)
        for map_size in map_sizes
        for candidate_player in (0, 1)
    }


def extract_baseline_schedule(
    baseline_games: Path,
    *,
    seed_start: int,
    seeds: int,
    map_sizes: tuple[int, ...],
    opponent_name: str,
) -> list[dict]:
    """Return exactly the baseline games corresponding to one screen schedule."""
    required = _schedule_keys(seed_start, seeds, map_sizes)
    selected: dict[tuple[int, int, int], dict] = {}
    for record in load_records(baseline_games):
        if str(record.get("opponent")) != opponent_name:
            continue
        key = (int(record["seed"]), int(record["map_size"]), int(record["candidate_player"]))
        if key not in required:
            continue
        if key in selected:
            raise ValueError(f"Duplicate baseline game for {key}: {baseline_games}")
        selected[key] = record
    missing = required - selected.keys()
    if missing:
        raise ValueError(f"Baseline evaluation lacks {len(missing)} required screen games: {sorted(missing)[:3]}")
    return [selected[key] for key in sorted(selected)]


def screen_decision(
    baseline_summary: dict,
    candidate_summary: dict,
    *,
    opponent_name: str,
    min_score_delta: float,
    min_city_survival_delta: float,
    max_city_extinction_delta: float,
) -> dict:
    baseline = baseline_summary["opponents"][opponent_name]
    candidate = candidate_summary["opponents"][opponent_name]
    score_delta = float(candidate["score_rate"] - baseline["score_rate"])
    survival_delta = float(candidate["candidate_city_survival"] - baseline["candidate_city_survival"])
    extinction_delta = float(
        candidate["candidate_city_extinction_rate"] - baseline["candidate_city_extinction_rate"]
    )
    reasons = []
    if score_delta < min_score_delta:
        reasons.append(f"score_delta={score_delta:.6f} < {min_score_delta:.6f}")
    if survival_delta < min_city_survival_delta:
        reasons.append(f"city_survival_delta={survival_delta:.6f} < {min_city_survival_delta:.6f}")
    if extinction_delta > max_city_extinction_delta:
        reasons.append(f"city_extinction_delta={extinction_delta:.6f} > {max_city_extinction_delta:.6f}")
    return {
        "passed": not reasons,
        "score_delta": score_delta,
        "city_survival_delta": survival_delta,
        "city_extinction_delta": extinction_delta,
        "reasons": reasons,
    }


def rank_screen_results(results: list[dict], top_k: int) -> list[dict]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    passing = [result for result in results if result["screen_gate"]["passed"]]
    return sorted(
        passing,
        key=lambda result: (
            result["screen_gate"]["score_delta"],
            result["paired_score_delta_lcb95"],
            result["screen_gate"]["city_survival_delta"],
            -result["screen_gate"]["city_extinction_delta"],
        ),
        reverse=True,
    )[:top_k]


def _write_records(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Screen multiple checkpoints on one matched schedule before running a full promotion evaluation."
    )
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--baseline-games", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, help="Defaults to config.yaml beside each checkpoint.")
    parser.add_argument("--opponent", type=Path, default=FIRST_PLACE_AGENT)
    parser.add_argument("--opponent-name", default="first_place")
    parser.add_argument("--seed-start", type=int, default=2021)
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--map-sizes", type=int, nargs="+", default=(12, 16, 24, 32))
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--min-score-delta", type=float, default=-0.025)
    parser.add_argument("--min-city-survival-delta", type=float, default=-0.025)
    parser.add_argument("--max-city-extinction-delta", type=float, default=0.05)
    parser.add_argument("--python", default=sys.executable, help="Lux engine Python; defaults to the current interpreter.")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--backend", choices=("auto", "official", "internal"), default="auto")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-games", type=int, default=8)
    parser.add_argument("--parity-games", type=int, default=4)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seeds <= 0:
        raise SystemExit("--seeds must be positive")
    if args.top_k <= 0:
        raise SystemExit("--top-k must be positive")
    map_sizes = tuple(args.map_sizes)
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and not args.resume:
        raise SystemExit(f"Refusing to reuse screen output directory without --resume: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_records = extract_baseline_schedule(
        args.baseline_games,
        seed_start=args.seed_start,
        seeds=args.seeds,
        map_sizes=map_sizes,
        opponent_name=args.opponent_name,
    )
    baseline_path = output_dir / "baseline_games.jsonl"
    _write_records(baseline_path, baseline_records)
    baseline_summary = summarize(baseline_records, bootstrap_samples=args.bootstrap_samples)
    (output_dir / "baseline_report.json").write_text(
        json.dumps(baseline_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    results = []
    for checkpoint in args.checkpoints:
        checkpoint = checkpoint.expanduser().resolve()
        candidate_dir = output_dir / checkpoint.stem
        result = evaluate_checkpoint(
            checkpoint,
            args.opponent,
            candidate_dir,
            config=args.config,
            opponent_name=args.opponent_name,
            seed_start=args.seed_start,
            seeds=args.seeds,
            map_sizes=map_sizes,
            python=args.python,
            timeout=args.timeout,
            resume=args.resume,
            bootstrap_samples=args.bootstrap_samples,
            workers=args.workers,
            backend=args.backend,
            device=args.device,
            batch_games=args.batch_games,
            parity_games=args.parity_games,
        )
        candidate_games = Path(result["games"])
        decision = screen_decision(
            baseline_summary,
            result["summary"],
            opponent_name=args.opponent_name,
            min_score_delta=args.min_score_delta,
            min_city_survival_delta=args.min_city_survival_delta,
            max_city_extinction_delta=args.max_city_extinction_delta,
        )
        results.append(
            {
                "checkpoint": str(checkpoint),
                "evaluation_dir": str(candidate_dir),
                "games": str(candidate_games),
                "backend": result["backend"],
                "summary": result["summary"],
                "screen_gate": decision,
                "paired_score_delta_lcb95": paired_score_difference_lcb(
                    baseline_path, candidate_games, bootstrap_samples=args.bootstrap_samples
                ),
            }
        )
    finalists = rank_screen_results(results, args.top_k)
    payload = {
        "kind": "checkpoint_screen",
        "screen_is_not_promotion": True,
        "schedule": {
            "seed_start": args.seed_start,
            "seeds": args.seeds,
            "map_sizes": list(map_sizes),
            "games": len(baseline_records),
            "opponent": args.opponent_name,
        },
        "baseline_games": str(baseline_path),
        "baseline_summary": baseline_summary,
        "thresholds": {
            "min_score_delta": args.min_score_delta,
            "min_city_survival_delta": args.min_city_survival_delta,
            "max_city_extinction_delta": args.max_city_extinction_delta,
        },
        "results": results,
        "finalists": finalists,
        "full_promotion_requirement": "Run a fresh 10-seed/80-game matched evaluation; require score LCB95 > 0, non-regressing city survival, and non-increasing city extinction.",
    }
    summary_path = output_dir / "screen_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"screen_summary": str(summary_path), "finalists": finalists}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
