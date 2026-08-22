from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch

from .artifacts import sha256_file
from .evaluate_checkpoint import FIRST_PLACE_AGENT, _opponent_model_files
from .prepare_eval_agent import prepare_eval_agent
from .run_matches import run_matched_matches


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def annotate_dagger_replays(
    output_dir: Path,
    games_path: Path,
    *,
    source_name: str,
    candidate_checkpoint: Path,
) -> dict:
    records = [json.loads(line) for line in games_path.read_text(encoding="utf-8").splitlines() if line]
    candidate_sha = sha256_file(candidate_checkpoint)
    replay_records = []
    for record in records:
        replay_path = output_dir / record["replay"]
        replay = json.loads(replay_path.read_text(encoding="utf-8"))
        metadata = {
            "source": source_name,
            "candidate_player": int(record["candidate_player"]),
            "seed": int(record["seed"]),
            "map_size": int(record["map_size"]),
            "candidate_checkpoint_sha256": candidate_sha,
        }
        replay["distillation"] = metadata
        _atomic_json(replay_path, replay)
        replay_records.append({"replay": replay_path.name, **metadata})
    manifest = {
        "schema_version": 1,
        "kind": "student_teacher_dagger_replays",
        "source": source_name,
        "candidate_checkpoint": str(candidate_checkpoint.resolve()),
        "candidate_checkpoint_sha256": candidate_sha,
        "games": len(replay_records),
        "records": sorted(
            replay_records,
            key=lambda item: (item["seed"], item["map_size"], item["candidate_player"]),
        ),
    }
    _atomic_json(output_dir / "dagger_info.json", manifest)
    return manifest


def validate_internal_parity(checkpoint: Path, parity_report: Path) -> dict:
    from .es import load_policy_state
    from .train_es import policy_state_digest

    report = json.loads(parity_report.read_text(encoding="utf-8"))
    if report.get("parity") != "passed":
        raise ValueError(f"Internal backend parity has not passed: {parity_report}")
    expected_digest = report.get("profile", {}).get("candidate_digest")
    actual_digest = policy_state_digest(load_policy_state(checkpoint))
    if not expected_digest or expected_digest != actual_digest:
        raise ValueError(
            "Parity report candidate differs from the requested checkpoint: "
            f"expected={expected_digest} actual={actual_digest}"
        )
    return {
        "parity_report": str(parity_report.resolve()),
        "candidate_digest": actual_digest,
    }


def run_internal_dagger(
    checkpoint: Path,
    config: Path,
    opponent: Path,
    output_dir: Path,
    *,
    seed_start: int,
    seeds: int,
    map_sizes: tuple[int, ...],
    device: str,
    batch_games: int,
    resume: bool,
    parity: dict,
) -> Path:
    from .es import MatchSpec, load_policy_state
    from .evaluate import load_records
    from .train_es import InternalMatchEvaluator, OpponentSpec

    if batch_games <= 0:
        raise ValueError("--batch-games must be positive")
    opponent_checkpoint, opponent_config = _opponent_model_files(opponent)
    resolved_device = torch.device(
        "cuda:0" if device == "auto" and torch.cuda.is_available() else "cpu" if device == "auto" else device
    )
    schedule = [
        MatchSpec(
            match_id=f"seed-{seed}-size-{map_size}-p{candidate_player}",
            opponent="first_place",
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
        raise FileExistsError(f"Refusing to mix DAgger runs; use --resume or a new directory: {result_path}")
    completed = set()
    if result_path.exists():
        completed = {
            (int(record["seed"]), int(record["map_size"]), int(record["candidate_player"]))
            for record in load_records(result_path)
        }
    pending = [
        spec
        for spec in schedule
        if (spec.seed, spec.map_size, spec.candidate_player) not in completed
    ]
    evaluator = InternalMatchEvaluator(
        config,
        [
            OpponentSpec(
                name="first_place",
                checkpoint=opponent_checkpoint,
                config=opponent_config,
                agent=opponent,
            )
        ],
        resolved_device,
    )
    candidate_state = load_policy_state(checkpoint)
    started = time.monotonic()
    profile_totals: dict[str, float] = {}
    print(
        f"Internal batched DAgger: pending={len(pending)} batch_games={batch_games} device={resolved_device}",
        flush=True,
    )
    with result_path.open("a" if resume else "x", encoding="utf-8") as result_file:
        for start in range(0, len(pending), batch_games):
            batch = pending[start : start + batch_games]
            batch_started = time.monotonic()
            print(
                f"Starting games {start + 1}-{start + len(batch)}/{len(pending)}: "
                f"{batch[0].match_id} ... {batch[-1].match_id}",
                flush=True,
            )
            records = evaluator.evaluate(candidate_state, batch, replay_dir=output_dir)
            for key, value in evaluator.last_profile.items():
                if isinstance(value, (int, float)):
                    profile_totals[key] = profile_totals.get(key, 0.0) + float(value)
            for record in records:
                record["backend"] = "internal_batched"
                result_file.write(json.dumps(record, sort_keys=True) + "\n")
            result_file.flush()
            elapsed = time.monotonic() - batch_started
            print(
                f"Completed {start + len(batch)}/{len(pending)} games in {elapsed:.1f}s "
                f"({len(batch) / max(elapsed, 1e-12):.3f} games/s)",
                flush=True,
            )
    elapsed = time.monotonic() - started
    _atomic_json(
        output_dir / "dagger_backend.json",
        {
            "backend": "internal_batched",
            "device": str(resolved_device),
            "batch_games": batch_games,
            "games": len(pending),
            "wall_seconds": elapsed,
            "games_per_second": len(pending) / max(elapsed, 1e-12),
            "profile": profile_totals,
            **parity,
        },
    )
    return result_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate matched student-vs-first-place stateful replays for DAgger distillation."
    )
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-config", type=Path)
    parser.add_argument("--opponent", type=Path, default=FIRST_PLACE_AGENT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-name", default="dagger")
    parser.add_argument("--seed-start", type=int, default=20000)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--map-sizes", type=int, nargs="+", default=(12, 16, 24, 32))
    parser.add_argument("--python", default="python")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--backend", choices=("official", "internal"), default="official")
    parser.add_argument("--device", default="auto", help="Device for internal batched inference.")
    parser.add_argument("--batch-games", type=int, default=8)
    parser.add_argument(
        "--parity-report",
        type=Path,
        help="backend.json from a passed official/internal parity evaluation of this exact checkpoint.",
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    candidate_checkpoint = args.candidate_checkpoint.expanduser().resolve()
    candidate_config = (
        candidate_checkpoint.parent / "config.yaml"
        if args.candidate_config is None
        else args.candidate_config.expanduser().resolve()
    )
    if not args.source_name or "=" in args.source_name:
        raise SystemExit("--source-name must be non-empty and cannot contain '='")
    opponent = args.opponent.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.backend == "internal":
        if args.parity_report is None:
            raise SystemExit("--backend internal requires --parity-report")
        parity = validate_internal_parity(candidate_checkpoint, args.parity_report.expanduser().resolve())
        games_path = run_internal_dagger(
            candidate_checkpoint,
            candidate_config,
            opponent,
            output_dir,
            seed_start=args.seed_start,
            seeds=args.seeds,
            map_sizes=tuple(args.map_sizes),
            device=args.device,
            batch_games=args.batch_games,
            resume=args.resume,
            parity=parity,
        )
    else:
        bundle = prepare_eval_agent(
            candidate_checkpoint,
            output_dir / ".candidate_agent",
            candidate_config,
            force=True,
        )
        games_path = run_matched_matches(
            Path(bundle["agent"]),
            opponent,
            output_dir,
            seed_start=args.seed_start,
            seeds=args.seeds,
            map_sizes=tuple(args.map_sizes),
            python=args.python,
            timeout=args.timeout,
            resume=args.resume,
            opponent_name="first_place",
            workers=args.workers,
        )
    manifest = annotate_dagger_replays(
        output_dir,
        games_path,
        source_name=args.source_name,
        candidate_checkpoint=candidate_checkpoint,
    )
    print(json.dumps({"games": manifest["games"], "source": manifest["source"]}, sort_keys=True))


if __name__ == "__main__":
    main()
