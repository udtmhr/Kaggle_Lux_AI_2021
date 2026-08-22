from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

from .artifacts import sha256_file
from .generate_dagger import _atomic_json, validate_internal_parity
from .prepare_data import FIRST_PLACE_TEACHER_SHA256


def validate_candidate_config(config_path: Path) -> None:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("learner_policy_tta_rot180") is not False:
        raise ValueError("Online distillation requires learner_policy_tta_rot180=false")
    prior_keys = ("rule_prior_alpha", "rule_prior_alpha_worker", "rule_prior_alpha_cart", "rule_prior_alpha_city_tile")
    active = {key: config.get(key) for key in prior_keys if float(config.get(key, 0.0) or 0.0) != 0.0}
    if active:
        raise ValueError(f"Online distillation must remain rule-prior-free: {active}")


def collection_command(args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        "-m",
        "lux_ai.strategic_rl.build_streaming_dataset",
        "--mode",
        "dagger",
        "--source-name",
        "dagger",
        "--output-dir",
        str(args.dataset_dir),
        "--student-config",
        str(args.dataset_config),
        "--teacher-checkpoint",
        str(args.teacher_checkpoint),
        "--teacher-config",
        str(args.teacher_config),
        "--teacher-agent",
        str(args.teacher_agent),
        "--candidate-checkpoint",
        str(args.candidate_checkpoint),
        "--candidate-config",
        str(args.candidate_config),
        "--parity-report",
        str(args.parity_report),
        "--expected-teacher-sha256",
        args.expected_teacher_sha256,
        "--seed-start",
        str(args.seed_start),
        "--seeds",
        str(args.seeds),
        "--map-sizes",
        *(str(size) for size in args.map_sizes),
        "--batch-games",
        str(args.batch_games),
        "--device",
        args.device,
    ]


def training_command(args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        "-m",
        "lux_ai.strategic_rl.train_distill",
        "--dataset-dir",
        str(args.dataset_dir),
        "--output-dir",
        str(args.output_dir),
        "--config",
        str(args.candidate_config),
        "--load-weights",
        str(args.candidate_checkpoint),
        "--epochs",
        "1",
        "--batch-size",
        str(args.train_batch_size),
        "--lr",
        str(args.lr),
        "--weight-decay",
        str(args.weight_decay),
        "--temperature",
        "2.0",
        "--hard-label-weight",
        "0.25",
        "--teacher-margin-threshold",
        "0.5",
        "--rare-action-weight",
        "2.0",
        "--rot180-consistency-weight",
        "0.1",
        "--rot180-augmentation-prob",
        "0",
        "--outcome-weight",
        "0",
        "--selection-metric",
        "loss",
        "--source-weight",
        "teacher_selfplay=0.4",
        "--source-weight",
        "dagger=0.6",
        "--samples-per-epoch",
        str(args.samples_per_round),
        "--checkpoint-every-samples",
        str(args.checkpoint_every_samples),
        "--num-workers",
        str(args.num_workers),
        "--device",
        args.device,
        "--seed",
        str(args.train_seed),
    ]


def _load_dataset_manifest(dataset_dir: Path) -> dict:
    path = dataset_dir / "manifest.json"
    if not path.is_file():
        raise ValueError(f"Dataset manifest does not exist: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_sources(manifest: dict, *, require_dagger: bool) -> None:
    sources = set(manifest.get("sources", {}))
    required = {"teacher_selfplay"} | ({"dagger"} if require_dagger else set())
    missing = required - sources
    unexpected = sources - {"teacher_selfplay", "dagger"}
    if missing or unexpected:
        raise ValueError(f"Dataset sources are incompatible: missing={sorted(missing)} unexpected={sorted(unexpected)}")


def scheduled_dagger_count(manifest: dict, seed_start: int, seeds: int, map_sizes: tuple[int, ...]) -> int:
    seed_stop = seed_start + seeds
    keys = {
        (int(shard.get("seed", -1)), int(shard.get("map_size", -1)), int(shard.get("candidate_player", -1)))
        for shard in manifest.get("shards", [])
        if shard.get("data_source") == "dagger"
        and seed_start <= int(shard.get("seed", -1)) < seed_stop
        and int(shard.get("map_size", -1)) in map_sizes
        and int(shard.get("candidate_player", -1)) in (0, 1)
    }
    return len(keys)


def scheduled_dagger_digests(manifest: dict, seed_start: int, seeds: int, map_sizes: tuple[int, ...]) -> set[str]:
    seed_stop = seed_start + seeds
    return {
        str(shard.get("candidate_checkpoint_sha256", ""))
        for shard in manifest.get("shards", [])
        if shard.get("data_source") == "dagger"
        and seed_start <= int(shard.get("seed", -1)) < seed_stop
        and int(shard.get("map_size", -1)) in map_sizes
        and int(shard.get("candidate_player", -1)) in (0, 1)
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect one on-policy DAgger batch and distill one conservative round.")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--dataset-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-config", type=Path, required=True)
    parser.add_argument("--parity-report", type=Path, required=True)
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--teacher-config", type=Path, required=True)
    parser.add_argument("--teacher-agent", type=Path, required=True)
    parser.add_argument("--expected-teacher-sha256", default=FIRST_PLACE_TEACHER_SHA256)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--seeds", type=int, default=32)
    parser.add_argument("--map-sizes", type=int, nargs="+", default=(24, 32))
    parser.add_argument("--batch-games", type=int, default=8)
    parser.add_argument("--samples-per-round", type=int, default=100000)
    parser.add_argument("--checkpoint-every-samples", type=int, default=50000)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--train-seed", type=int, default=2021)
    return parser.parse_args()


def _resolve_paths(args: argparse.Namespace) -> None:
    for name in (
        "dataset_dir",
        "dataset_config",
        "output_dir",
        "candidate_checkpoint",
        "candidate_config",
        "parity_report",
        "teacher_checkpoint",
        "teacher_config",
        "teacher_agent",
    ):
        setattr(args, name, getattr(args, name).expanduser().resolve())


def main() -> None:
    args = parse_args()
    _resolve_paths(args)
    if args.output_dir.exists():
        raise SystemExit(f"Refusing to reuse output directory: {args.output_dir}")
    if args.seeds <= 0 or args.batch_games <= 0 or args.samples_per_round <= 0:
        raise ValueError("Seeds, batch games, and samples per round must be positive")
    if args.checkpoint_every_samples <= 0 or args.checkpoint_every_samples > args.samples_per_round:
        raise ValueError("Checkpoint interval must be in [1, samples-per-round]")
    if not args.map_sizes or any(size not in (12, 16, 24, 32) for size in args.map_sizes):
        raise ValueError("Map sizes must be selected from 12, 16, 24, 32")
    validate_candidate_config(args.candidate_config)
    validate_internal_parity(args.candidate_checkpoint, args.parity_report)
    if sha256_file(args.teacher_checkpoint) != args.expected_teacher_sha256:
        raise ValueError("Teacher checkpoint SHA does not match the approved black-box teacher")
    before = _load_dataset_manifest(args.dataset_dir)
    _validate_sources(before, require_dagger=False)
    scheduled_before = scheduled_dagger_count(before, args.seed_start, args.seeds, tuple(args.map_sizes))
    collect = collection_command(args)
    train = training_command(args)
    print("COLLECT " + " ".join(collect), flush=True)
    subprocess.run(collect, check=True)
    after = _load_dataset_manifest(args.dataset_dir)
    _validate_sources(after, require_dagger=True)
    expected_games = args.seeds * len(args.map_sizes) * 2
    added_games = int(after["replay_count"]) - int(before["replay_count"])
    scheduled_after = scheduled_dagger_count(after, args.seed_start, args.seeds, tuple(args.map_sizes))
    if scheduled_after != expected_games:
        raise RuntimeError(f"Partial DAgger collection: expected {expected_games}, completed {scheduled_after}")
    candidate_sha = sha256_file(args.candidate_checkpoint)
    scheduled_digests = scheduled_dagger_digests(after, args.seed_start, args.seeds, tuple(args.map_sizes))
    if scheduled_digests != {candidate_sha}:
        raise RuntimeError(f"DAgger schedule contains unexpected candidate checkpoint digests: {scheduled_digests}")
    print("TRAIN " + " ".join(train), flush=True)
    subprocess.run(train, check=True)
    payload = {
        "kind": "online_distillation_round",
        "candidate_checkpoint": str(args.candidate_checkpoint),
        "candidate_checkpoint_sha256": candidate_sha,
        "candidate_parity_report": str(args.parity_report),
        "teacher_checkpoint_sha256": sha256_file(args.teacher_checkpoint),
        "teacher_used_as_labeler_only": True,
        "learner_policy_tta_rot180": False,
        "map_sizes": list(args.map_sizes),
        "seed_start": args.seed_start,
        "seeds": args.seeds,
        "expected_dagger_games": expected_games,
        "added_dagger_games": added_games,
        "scheduled_dagger_games_before": scheduled_before,
        "scheduled_dagger_games_after": scheduled_after,
        "dataset_replays_before": int(before["replay_count"]),
        "dataset_replays_after": int(after["replay_count"]),
        "source_weights": {"teacher_selfplay": 0.4, "dagger": 0.6},
        "outcome_weight": 0.0,
        "commands": {"collect": collect, "train": train},
        "result_checkpoint": str((args.output_dir / "best.pt").resolve()),
        "result_checkpoint_sha256": sha256_file(args.output_dir / "best.pt"),
    }
    _atomic_json(args.output_dir / "online_round.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
