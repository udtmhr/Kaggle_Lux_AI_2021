from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import torch

from ..nns import create_model
from .artifacts import sha256_file
from .es import MatchSpec, load_policy_state
from .evaluate_checkpoint import FIRST_PLACE_AGENT
from .generate_dagger import _atomic_json, validate_internal_parity
from .prepare_data import (
    DATASET_SCHEMA_VERSION,
    FIRST_PLACE_TEACHER_SHA256,
    _load_flags,
    prepare_replay,
)
from .train_es import InternalMatchEvaluator, OpponentSpec, load_deployment_action_config


def _write_manifest(
    output_dir: Path,
    records: list[dict],
    *,
    teacher_sha: str,
    student_config: Path,
) -> dict:
    sources = sorted({str(record["data_source"]) for record in records})
    manifest = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "teacher_sha256": teacher_sha,
        "teacher_tta_rot180": True,
        "student_config": str(student_config.resolve()),
        "legacy_prepared_cache_dir": None,
        "streamed_raw_replays": True,
        "replay_count": len(records),
        "sample_count": sum(int(record["turn_count"]) for record in records),
        "size_bytes": sum(int(record["size_bytes"]) for record in records),
        "shards": sorted(
            records,
            key=lambda item: (
                str(item["data_source"]),
                int(item.get("seed", -1)),
                int(item.get("map_size", -1)),
                int(item.get("candidate_player", -1)),
            ),
        ),
        "sources": {
            source: {
                "replays": sum(record["data_source"] == source for record in records),
                "samples": sum(int(record["turn_count"]) for record in records if record["data_source"] == source),
            }
            for source in sources
        },
        "candidate_checkpoint_sha256": sorted(
            {
                str(record["candidate_checkpoint_sha256"])
                for record in records
                if record.get("candidate_checkpoint_sha256")
            }
        ),
    }
    temporary = output_dir / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output_dir / "manifest.json")
    return manifest


def _load_existing_records(output_dir: Path, teacher_sha: str, student_config: Path) -> list[dict]:
    manifest_path = output_dir / "manifest.json"
    shards = list(output_dir.glob("*.npz"))
    if not manifest_path.exists():
        if shards:
            raise ValueError(f"Dataset has shards but no manifest: {output_dir}")
        return []
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != DATASET_SCHEMA_VERSION:
        raise ValueError("Streaming append requires a schema-v4 dataset")
    if manifest.get("teacher_sha256") != teacher_sha:
        raise ValueError("Cannot mix teacher checkpoints in one dataset")
    configured = Path(manifest["student_config"]).expanduser().resolve()
    if configured != student_config.resolve():
        raise ValueError(f"Cannot mix student observation configs: {configured} != {student_config.resolve()}")
    return list(manifest["shards"])


def _schedule(
    mode: str,
    source: str,
    seed_start: int,
    seeds: int,
    map_sizes: tuple[int, ...],
) -> list[MatchSpec]:
    players = (0,) if mode == "teacher_selfplay" else (0, 1)
    return [
        MatchSpec(
            match_id=f"{source}-seed-{seed}-size-{map_size}-p{player}",
            opponent="first_place",
            seed=seed,
            map_size=map_size,
            candidate_player=player,
        )
        for seed in range(seed_start, seed_start + seeds)
        for map_size in map_sizes
        for player in players
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate games in temporary batches and keep only compact distillation shards."
    )
    parser.add_argument("--mode", choices=("teacher_selfplay", "dagger"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-name")
    parser.add_argument("--student-config", type=Path, required=True)
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--teacher-config", type=Path, required=True)
    parser.add_argument("--teacher-agent", type=Path, default=FIRST_PLACE_AGENT)
    parser.add_argument("--candidate-checkpoint", type=Path)
    parser.add_argument("--candidate-config", type=Path)
    parser.add_argument("--parity-report", type=Path, required=True)
    parser.add_argument("--expected-teacher-sha256", default=FIRST_PLACE_TEACHER_SHA256)
    parser.add_argument("--seed-start", type=int, default=30000)
    parser.add_argument("--seeds", type=int, required=True)
    parser.add_argument("--map-sizes", type=int, nargs="+", default=(12, 16, 24, 32))
    parser.add_argument("--batch-games", type=int, default=8)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-turns", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seeds <= 0 or args.batch_games <= 0:
        raise SystemExit("--seeds and --batch-games must be positive")
    teacher_checkpoint = args.teacher_checkpoint.expanduser().resolve()
    teacher_config = args.teacher_config.expanduser().resolve()
    teacher_agent = args.teacher_agent.expanduser().resolve()
    student_config = args.student_config.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    source = args.source_name or args.mode
    teacher_sha = sha256_file(teacher_checkpoint)
    if args.expected_teacher_sha256 and teacher_sha != args.expected_teacher_sha256:
        raise ValueError(f"Teacher SHA mismatch: expected={args.expected_teacher_sha256} actual={teacher_sha}")
    parity_report = args.parity_report.expanduser().resolve()
    parity = json.loads(parity_report.read_text(encoding="utf-8"))
    if parity.get("parity") != "passed":
        raise ValueError(f"Internal backend parity has not passed: {parity_report}")

    if args.mode == "dagger":
        if args.candidate_checkpoint is None:
            raise ValueError("dagger mode requires --candidate-checkpoint")
        candidate_checkpoint = args.candidate_checkpoint.expanduser().resolve()
        candidate_config = (
            candidate_checkpoint.parent / "config.yaml"
            if args.candidate_config is None
            else args.candidate_config.expanduser().resolve()
        )
        validate_internal_parity(candidate_checkpoint, parity_report)
    else:
        if args.candidate_checkpoint is not None or args.candidate_config is not None:
            raise ValueError("teacher_selfplay mode always uses the teacher as both players")
        candidate_checkpoint = teacher_checkpoint
        candidate_config = teacher_config

    device = torch.device(args.device)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = _load_existing_records(output_dir, teacher_sha, student_config)
    completed = {
        (
            str(record["data_source"]),
            int(record.get("seed", -1)),
            int(record.get("map_size", -1)),
            int(record.get("candidate_player", -1)),
        )
        for record in records
    }
    schedule = _schedule(args.mode, source, args.seed_start, args.seeds, tuple(args.map_sizes))
    pending = [
        spec
        for spec in schedule
        if (
            source,
            spec.seed,
            spec.map_size,
            -1 if args.mode == "teacher_selfplay" else spec.candidate_player,
        )
        not in completed
    ]

    student_flags = _load_flags(student_config, args.device)
    teacher_flags = _load_flags(teacher_config, args.device)
    teacher_model = create_model(student_flags, device, teacher_model_flags=teacher_flags, is_teacher_model=True)
    teacher_state = torch.load(teacher_checkpoint, map_location=device, weights_only=False)
    teacher_model.load_state_dict(teacher_state["model_state_dict"], strict=True)
    teacher_model.eval()
    evaluator = InternalMatchEvaluator(
        candidate_config,
        [
            OpponentSpec(
                name="first_place",
                checkpoint=teacher_checkpoint,
                config=teacher_config,
                agent=teacher_agent,
            )
        ],
        device,
    )
    if args.mode == "teacher_selfplay":
        evaluator.candidate_action_config = load_deployment_action_config(teacher_agent)
    candidate_state = load_policy_state(candidate_checkpoint)
    candidate_sha = sha256_file(candidate_checkpoint)

    print(
        f"Streaming {args.mode}: pending={len(pending)} existing={len(records)} "
        f"batch_games={args.batch_games} device={device}",
        flush=True,
    )
    for start in range(0, len(pending), args.batch_games):
        batch = pending[start : start + args.batch_games]
        with tempfile.TemporaryDirectory(prefix="lux-distill-raw-") as temporary_text:
            temporary = Path(temporary_text)
            game_records = evaluator.evaluate(candidate_state, batch, replay_dir=temporary)
            for game_record in game_records:
                replay_path = temporary / game_record["replay"]
                replay = json.loads(replay_path.read_text(encoding="utf-8"))
                replay["distillation"] = {
                    "source": source,
                    "candidate_player": (
                        -1 if args.mode == "teacher_selfplay" else int(game_record["candidate_player"])
                    ),
                    "seed": int(game_record["seed"]),
                    "map_size": int(game_record["map_size"]),
                    "candidate_checkpoint_sha256": candidate_sha,
                }
                _atomic_json(replay_path, replay)
                record = prepare_replay(
                    replay_path,
                    output_dir,
                    teacher_model,
                    teacher_sha,
                    student_flags,
                    teacher_flags,
                    device,
                    args.max_turns,
                    source=source,
                )
                records.append(record)
                manifest = _write_manifest(
                    output_dir,
                    records,
                    teacher_sha=teacher_sha,
                    student_config=student_config,
                )
        print(
            f"Prepared {min(start + len(batch), len(pending))}/{len(pending)} pending games; "
            f"dataset_replays={manifest['replay_count']} turns={manifest['sample_count']}",
            flush=True,
        )
    if not pending:
        manifest = _write_manifest(
            output_dir,
            records,
            teacher_sha=teacher_sha,
            student_config=student_config,
        )
    print(json.dumps({key: manifest[key] for key in ("replay_count", "sample_count", "sources")}, indent=2))


if __name__ == "__main__":
    main()
