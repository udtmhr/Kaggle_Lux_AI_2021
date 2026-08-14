from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import torch

from .evaluate import load_records, summarize
from .prepare_eval_agent import ROOT, checkpoint_label, prepare_eval_agent
from .run_matches import run_matched_matches

FIRST_PLACE_AGENT = (
    ROOT / "internal_testing" / "hall_of_fame" / "11-24_12-56-23_062179520_must_research" / "main.py"
)


def _opponent_model_files(opponent: Path) -> tuple[Path, Path]:
    model_dir = opponent.parent / "lux_ai" / "rl_agent"
    checkpoints = sorted(model_dir.glob("*.pt"))
    config = model_dir / "config.yaml"
    if len(checkpoints) != 1 or not config.is_file():
        raise FileNotFoundError(
            "Batched evaluation requires an opponent bundle with exactly one "
            f"lux_ai/rl_agent/*.pt and config.yaml beside its main.py: {opponent}"
        )
    return checkpoints[0], config


def _run_batched_matches(
    checkpoint: Path,
    config: Path,
    opponent: Path,
    opponent_name: str,
    output_dir: Path,
    *,
    seed_start: int,
    seeds: int,
    map_sizes: tuple[int, ...],
    device: str,
    batch_games: int,
    resume: bool,
) -> Path:
    # Imported lazily because train_es also uses evaluate_checkpoint for its final official gate.
    from .es import MatchSpec, load_policy_state
    from .train_es import InternalMatchEvaluator, OpponentSpec

    if batch_games <= 0:
        raise ValueError("batch_games must be positive")
    opponent_checkpoint, opponent_config = _opponent_model_files(opponent)
    resolved_device = torch.device(
        "cuda:0" if device == "auto" and torch.cuda.is_available() else "cpu" if device == "auto" else device
    )
    opponent_spec = OpponentSpec(
        name=opponent_name,
        checkpoint=opponent_checkpoint,
        config=opponent_config,
        agent=opponent,
    )
    schedule = [
        MatchSpec(
            match_id=f"seed-{seed}-size-{map_size}-p{candidate_player}",
            opponent=opponent_name,
            seed=seed,
            map_size=map_size,
            candidate_player=candidate_player,
        )
        for seed in range(seed_start, seed_start + seeds)
        for map_size in map_sizes
        for candidate_player in (0, 1)
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "games.jsonl"
    if result_path.exists() and not resume:
        raise FileExistsError(f"Refusing to mix evaluation runs; use a new directory or --resume: {result_path}")
    completed = set()
    if result_path.exists():
        for record in load_records(result_path):
            completed.add((int(record["seed"]), int(record["map_size"]), int(record["candidate_player"])))
    pending = [
        spec
        for spec in schedule
        if (spec.seed, spec.map_size, spec.candidate_player) not in completed
    ]
    evaluator = InternalMatchEvaluator(config, [opponent_spec], resolved_device)
    candidate_state = load_policy_state(checkpoint)
    started = time.monotonic()
    profile_totals: dict[str, float] = {}
    with result_path.open("a" if resume else "x", encoding="utf-8") as result_file:
        for start in range(0, len(pending), batch_games):
            records = evaluator.evaluate(candidate_state, pending[start : start + batch_games])
            for key, value in evaluator.last_profile.items():
                if isinstance(value, (int, float)):
                    profile_totals[key] = profile_totals.get(key, 0.0) + float(value)
            for record in records:
                record["backend"] = "internal_batched"
                line = json.dumps(record, sort_keys=True)
                result_file.write(line + "\n")
                print(line)
            result_file.flush()
    elapsed = time.monotonic() - started
    forward_seconds = profile_totals.get("candidate_forward_seconds", 0.0) + profile_totals.get(
        "opponent_forward_seconds", 0.0
    )
    profile = {
        **profile_totals,
        "wall_seconds": elapsed,
        "games": len(pending),
        "games_per_second": len(pending) / max(elapsed, 1e-12),
        "forward_fraction": forward_seconds / max(elapsed, 1e-12),
        "device": str(resolved_device),
        "batch_games": batch_games,
        "candidate_digest": getattr(evaluator, "candidate_digest", None),
        "rng_scheme": getattr(evaluator, "rng_scheme", None),
    }
    for key, changed in list(profile.items()):
        if key.endswith(".changed"):
            active_key = key[:-len(".changed")] + ".active"
            profile[key[:-len(".changed")] + ".change_rate"] = changed / max(profile.get(active_key, 0.0), 1.0)
        if key.endswith(".friendly_collision_candidates"):
            active_key = key[:-len(".friendly_collision_candidates")] + ".actionable_units"
            profile[key[:-len(".friendly_collision_candidates")] + ".friendly_collision_rate"] = (
                changed / max(profile.get(active_key, 0.0), 1.0)
            )
    (output_dir / "backend_profile.json").write_text(
        json.dumps(profile, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result_path


def _parity_passed(official_path: Path, internal_path: Path) -> bool:
    def outcomes(path: Path) -> dict[tuple[int, int, int], int]:
        return {
            (int(record["seed"]), int(record["map_size"]), int(record["candidate_player"])): int(record["winner"])
            for record in load_records(path)
        }

    return outcomes(official_path) == outcomes(internal_path)


def evaluate_checkpoint(
    checkpoint: Path,
    opponent: Path,
    output_dir: Path,
    *,
    config: Path | None = None,
    agent_dir: Path | None = None,
    opponent_name: str = "first_place",
    seed_start: int = 2021,
    seeds: int = 1,
    map_sizes: tuple[int, ...] = (12,),
    python: str = sys.executable,
    timeout: int = 600,
    resume: bool = False,
    bootstrap_samples: int = 2000,
    workers: int = 2,
    backend: str = "auto",
    device: str = "auto",
    batch_games: int = 8,
    parity_games: int = 4,
) -> dict:
    checkpoint = checkpoint.expanduser().resolve()
    opponent = opponent.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not opponent.is_file():
        raise FileNotFoundError(f"Missing opponent agent: {opponent}")
    if seeds <= 0:
        raise ValueError("seeds must be positive")
    if any(size not in (12, 16, 24, 32) for size in map_sizes):
        raise ValueError(f"Unsupported map size: {map_sizes}")
    if backend not in {"auto", "official", "internal"}:
        raise ValueError(f"Unsupported evaluation backend: {backend}")
    try:
        checkpoint_metadata = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except (EOFError, RuntimeError):
        # Some tests and legacy bundles use an external/empty checkpoint stub.
        checkpoint_metadata = {}
    if checkpoint_metadata.get("evaluation_eligible") is False:
        fraction = checkpoint_metadata.get("max_support_outside_fraction")
        raise ValueError(
            f"Categorical critic support gate failed ({fraction:.3%} outside support); "
            "redesign support before evaluation"
        )
    label = checkpoint_label(checkpoint)
    agent_dir = (
        Path(tempfile.gettempdir()) / f"lux_candidate_{label}" if agent_dir is None else agent_dir.expanduser().resolve()
    )
    bundle = prepare_eval_agent(checkpoint, agent_dir, config, force=True)
    selected_backend = backend
    parity = "not_requested"
    if backend == "auto":
        try:
            _opponent_model_files(opponent)
        except FileNotFoundError:
            selected_backend = "official"
            parity = "opponent_not_batchable"
        else:
            requested_games = seeds * len(map_sizes) * 2
            parity_seeds = max(1, min(seeds, (min(parity_games, requested_games) + 1) // 2))
            parity_map_sizes = map_sizes[: max(1, min(len(map_sizes), parity_games // (2 * parity_seeds)))]
            parity_root = output_dir / ".backend_parity"
            official_parity = run_matched_matches(
                Path(bundle["agent"]),
                opponent,
                parity_root / "official",
                seed_start=seed_start,
                seeds=parity_seeds,
                map_sizes=parity_map_sizes,
                python=python,
                timeout=timeout,
                opponent_name=opponent_name,
                workers=workers,
                resume=resume,
            )
            internal_parity = _run_batched_matches(
                checkpoint,
                config or checkpoint.parent / "config.yaml",
                opponent,
                opponent_name,
                parity_root / "internal",
                seed_start=seed_start,
                seeds=parity_seeds,
                map_sizes=parity_map_sizes,
                device=device,
                batch_games=batch_games,
                resume=resume,
            )
            parity = "passed" if _parity_passed(official_parity, internal_parity) else "failed"
            selected_backend = "internal" if parity == "passed" else "official"
    if selected_backend == "internal":
        results_path = _run_batched_matches(
            checkpoint,
            config or checkpoint.parent / "config.yaml",
            opponent,
            opponent_name,
            output_dir,
            seed_start=seed_start,
            seeds=seeds,
            map_sizes=map_sizes,
            device=device,
            batch_games=batch_games,
            resume=resume,
        )
    else:
        results_path = run_matched_matches(
            Path(bundle["agent"]),
            opponent,
            output_dir,
            seed_start=seed_start,
            seeds=seeds,
            map_sizes=map_sizes,
            python=python,
            timeout=timeout,
            resume=resume,
            opponent_name=opponent_name,
            workers=workers,
        )
    report = summarize(load_records(results_path), bootstrap_samples=bootstrap_samples)
    backend_report = {
        "requested": backend,
        "selected": selected_backend,
        "parity": parity,
        "device": device,
        "batch_games": batch_games if selected_backend == "internal" else None,
    }
    profile_path = output_dir / "backend_profile.json"
    if profile_path.is_file():
        backend_report["profile"] = json.loads(profile_path.read_text(encoding="utf-8"))
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    backend_path = output_dir / "backend.json"
    backend_path.write_text(json.dumps(backend_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "agent": bundle["agent"],
        "games": str(results_path),
        "report": str(report_path),
        "backend": backend_report,
        "summary": report,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and evaluate one Lux checkpoint in matched orientations.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--opponent", type=Path, default=FIRST_PLACE_AGENT)
    parser.add_argument("--opponent-name", default="first_place")
    parser.add_argument("--config", type=Path, help="Defaults to config.yaml beside the checkpoint.")
    parser.add_argument("--agent-dir", type=Path, help="Defaults to /tmp/lux_candidate_<checkpoint step>.")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed-start", type=int, default=2021)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--map-sizes", type=int, nargs="+", default=(12,))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--workers", type=int, default=2, help="Concurrent matches; use 1 to disable parallelism.")
    parser.add_argument(
        "--backend",
        choices=("auto", "official", "internal"),
        default="auto",
        help="auto parity-checks the batched internal engine before using it.",
    )
    parser.add_argument("--device", default="auto", help="Device for internal batched inference.")
    parser.add_argument("--batch-games", type=int, default=8, help="Games per internal GPU inference batch.")
    parser.add_argument("--parity-games", type=int, default=4, help="Maximum auto-backend parity games.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args()
    if args.output_dir is None:
        label = checkpoint_label(args.checkpoint)
        args.output_dir = ROOT / "outputs" / "evaluation" / f"step_{label}_vs_{args.opponent_name}"
    return args


def main() -> None:
    args = parse_args()
    try:
        result = evaluate_checkpoint(
            args.checkpoint,
            args.opponent,
            args.output_dir,
            config=args.config,
            agent_dir=args.agent_dir,
            opponent_name=args.opponent_name,
            seed_start=args.seed_start,
            seeds=args.seeds,
            map_sizes=tuple(args.map_sizes),
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
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
