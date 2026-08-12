from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm.auto import tqdm

from ..lux_gym import obs_spaces
from ..lux_gym.act_spaces import ACTION_MEANINGS, BasicActionSpace
from ..lux_gym.lux_env import LuxEnv
from ..lux_gym.wrappers import PadFixedShapeEnv
from ..nns import create_model
from ..utils import flags_to_namespace
from .artifacts import sha256_file
from .obs import SurvivalStrategicObs
from .tta import rot180_ensemble_outputs

DATASET_SCHEMA_VERSION = 3
LEGACY_PREPARED_CACHE_VERSION = 1
FIRST_PLACE_TEACHER_SHA256 = "40248f0fbc9b8e1e1b1f7cc6fc674c041d8dac43b964ae45bd976d927cdffd22"


def _load_flags(path: Path, device: str):
    values = yaml.safe_load(path.read_text(encoding="utf-8"))
    values["actor_device"] = device
    values["learner_device"] = device
    return flags_to_namespace(values)


def _discover_replays(path: Path) -> list[Path]:
    replays = (
        [path]
        if path.is_file() and path.suffix == ".json"
        else sorted(
            p
            for p in path.rglob("*.json")
            if p.is_file() and p.name != "agent_info.json" and not p.name.endswith("_info.json")
        )
    )
    if not replays:
        raise FileNotFoundError(f"No replay JSON files found below {path}")
    return replays


def _actionable_masks(game, width: int, height: int) -> dict[str, np.ndarray]:
    result = {name: np.zeros((1, 2, width, height), dtype=np.bool_) for name in ACTION_MEANINGS}
    for player in game.players:
        for unit in player.units:
            if unit.can_act():
                name = "worker" if unit.is_worker() else "cart"
                result[name][0, player.team, unit.pos.x, unit.pos.y] = True
        for city in player.cities.values():
            for tile in city.citytiles:
                if tile.can_act():
                    result["city_tile"][0, player.team, tile.pos.x, tile.pos.y] = True
    return result


def _strip_prefix(values: dict[str, np.ndarray], prefix: str) -> dict[str, np.ndarray]:
    return {key[len(prefix) :]: value for key, value in values.items() if key.startswith(prefix)}


def _split_for_digest(digest: str) -> str:
    bucket = int(digest[:8], 16) % 100
    return "train" if bucket < 80 else "validation" if bucket < 90 else "test"


def _compact_observation(value: np.ndarray) -> np.ndarray:
    if np.issubdtype(value.dtype, np.floating):
        return value.astype(np.float16, copy=False)
    return value.astype(np.uint8, copy=False)


def _atomic_save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as output:
            np.savez_compressed(output, **arrays)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _legacy_prepared_cache_path(replay_path: Path, cache_dir: Path) -> Path:
    source = replay_path.resolve()
    stat = source.stat()
    fingerprint = (
        f"{LEGACY_PREPARED_CACHE_VERSION}\0{source}\0{stat.st_size}\0{stat.st_mtime_ns}"
        f"\0{FIRST_PLACE_TEACHER_SHA256}\0float16"
    )
    return cache_dir / f"{hashlib.sha256(fingerprint.encode()).hexdigest()}.npz"


def _load_legacy_prepared_targets(path: Path, turn_count: int) -> dict[str, dict[str, np.ndarray]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing legacy prepared cache: {path}")
    result = {}
    with np.load(path, allow_pickle=False) as archive:
        if int(archive["cache_version"]) != LEGACY_PREPARED_CACHE_VERSION:
            raise ValueError(f"Unsupported legacy prepared cache version: {path}")
        cached_turns = int(archive["turn_count"])
        if cached_turns < turn_count:
            raise ValueError(f"Legacy cache has {cached_turns} turns, but {turn_count} are required: {path}")
        for entity, actions in ACTION_MEANINGS.items():
            offsets = archive[f"{entity}_offsets"]
            if len(offsets) < turn_count * 2 + 1:
                raise ValueError(f"Legacy {entity} offsets are incomplete: {path}")
            legal_mask = archive[f"{entity}_legal_mask"]
            teacher_logits = archive[f"{entity}_teacher_logits"]
            if legal_mask.shape[-1] != len(actions) or teacher_logits.shape[-1] != len(actions):
                raise ValueError(f"Legacy {entity} action schema is incompatible: {path}")
            result[entity] = {
                "offsets": offsets[: turn_count * 2 + 1].copy(),
                "positions": archive[f"{entity}_positions"].copy(),
                "legal_mask": legal_mask.copy(),
                "teacher_logits": teacher_logits.copy(),
            }
    return result


def _shard_metadata(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as shard:
        return {
            "schema_version": int(shard["schema_version"]),
            "replay": str(shard["replay"]),
            "replay_sha256": str(shard["replay_sha256"]),
            "teacher_sha256": str(shard["teacher_sha256"]),
            "teacher_tta_rot180": bool(shard["teacher_tta_rot180"]),
            "observation_space": str(shard["observation_space"]),
            "turn_count": int(shard["turn_count"]),
            "split": str(shard["split"]),
        }


def prepare_replay(
    replay_path: Path,
    output_dir: Path,
    teacher_model,
    teacher_sha: str,
    student_flags,
    teacher_flags,
    device: torch.device,
    max_turns: int | None,
    legacy_prepared_cache_dir: Path | None = None,
) -> dict:
    replay_sha = sha256_file(replay_path)
    cache_key = hashlib.sha256(
        (
            f"{DATASET_SCHEMA_VERSION}:{replay_sha}:{teacher_sha}:{student_flags.obs_space.__name__}:"
            f"{max_turns if max_turns is not None else 'all'}:teacher_rot180_tta_v1"
        ).encode()
    ).hexdigest()
    output_path = output_dir / f"{cache_key}.npz"
    if output_path.is_file():
        metadata = _shard_metadata(output_path)
        if metadata["schema_version"] != DATASET_SCHEMA_VERSION:
            raise ValueError(f"Incompatible compact shard: {output_path}")
        if metadata["teacher_tta_rot180"] is not True:
            raise ValueError(f"Compact shard does not contain Rot180-TTA teacher targets: {output_path}")
        return metadata | {"path": output_path.name, "created": False, "size_bytes": output_path.stat().st_size}

    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    steps = replay.get("steps") or []
    if len(steps) < 2:
        raise ValueError(f"Replay has no playable turns: {replay_path}")
    multi_obs = obs_spaces.MultiObs(
        {
            "teacher_": teacher_flags.obs_space(**teacher_flags.obs_space_kwargs),
            "student_": student_flags.obs_space(**student_flags.obs_space_kwargs),
        }
    )
    raw_env = LuxEnv(BasicActionSpace(), multi_obs, run_game_automatically=False)
    obs_wrapper = multi_obs.wrap_env(raw_env)

    initial_observation = steps[0][0]["observation"]
    initial_updates = list(initial_observation["updates"])
    try:
        int(initial_updates[0])
        has_initialization_header = True
    except (IndexError, ValueError):
        has_initialization_header = False
    if not has_initialization_header:
        player = int(initial_observation.get("player", 0))
        width = int(initial_observation["width"])
        height = int(initial_observation["height"])
        initial_updates = [str(player), f"{width} {height}", *initial_updates]
    raw_env.reset(observation_updates=initial_updates)
    pad_wrapper = PadFixedShapeEnv(obs_wrapper)
    observations: dict[str, list[np.ndarray]] = {}
    entity_values = {entity: {"positions": [], "legal_mask": [], "teacher_logits": []} for entity in ACTION_MEANINGS}
    entity_offsets = {entity: [0] for entity in ACTION_MEANINGS}
    input_mask = None
    turn_count = min(len(steps) - 1, max_turns if max_turns is not None else len(steps) - 1)
    legacy_targets = None
    if legacy_prepared_cache_dir is not None:
        legacy_path = _legacy_prepared_cache_path(replay_path, legacy_prepared_cache_dir)
        legacy_targets = _load_legacy_prepared_targets(legacy_path, turn_count)
    with torch.inference_mode():
        for turn in range(turn_count):
            if turn:
                updates = steps[turn][0]["observation"]["updates"]
                raw_env.manual_step(updates)
                raw_env.game_state.turn = turn
                raw_env._update_internal_state()
            combined_obs = pad_wrapper.observation(obs_wrapper.observation(raw_env.game_state))
            padded_info = pad_wrapper.info(raw_env.info)
            student_obs = _strip_prefix(combined_obs, "student_")
            for key, value in student_obs.items():
                observations.setdefault(key, []).append(_compact_observation(value))
            if input_mask is None:
                input_mask = padded_info["input_mask"].astype(np.bool_, copy=True)
            elif not np.array_equal(input_mask, padded_info["input_mask"]):
                raise ValueError(f"Board mask changed within replay: {replay_path}")
            if legacy_targets is not None:
                for entity, values in legacy_targets.items():
                    position_parts = []
                    legal_parts = []
                    logits_parts = []
                    for player in (0, 1):
                        sample_index = turn * 2 + player
                        start = int(values["offsets"][sample_index])
                        stop = int(values["offsets"][sample_index + 1])
                        # LuxPythonEnvGym stores positions as (row=y, column=x), while the
                        # first-place environment indexes spatial tensors as (x, y).
                        xy = values["positions"][start:stop, ::-1].astype(np.int16, copy=False)
                        player_column = np.full((len(xy), 1), player, dtype=np.int16)
                        position_parts.append(np.concatenate((player_column, xy), axis=1))
                        legal_parts.append(values["legal_mask"][start:stop])
                        logits_parts.append(values["teacher_logits"][start:stop])
                    positions = np.concatenate(position_parts, axis=0)
                    legal = np.concatenate(legal_parts, axis=0)
                    logits = np.concatenate(logits_parts, axis=0)
                    entity_values[entity]["positions"].append(positions)
                    entity_values[entity]["legal_mask"].append(legal.astype(np.bool_, copy=False))
                    entity_values[entity]["teacher_logits"].append(logits.astype(np.float16, copy=False))
                    entity_offsets[entity].append(entity_offsets[entity][-1] + len(positions))
                continue

            teacher_input = {
                "obs": {
                    key: torch.from_numpy(value).unsqueeze(0).to(device)
                    for key, value in combined_obs.items()
                    if key.startswith("teacher_")
                },
                "info": {
                    "input_mask": torch.from_numpy(padded_info["input_mask"]).unsqueeze(0).to(device),
                    "available_actions_mask": {
                        key: torch.from_numpy(value).unsqueeze(0).to(device)
                        for key, value in padded_info["available_actions_mask"].items()
                    },
                },
            }
            teacher_output = rot180_ensemble_outputs(teacher_model, teacher_input)
            masks = pad_wrapper._pad(
                _actionable_masks(raw_env.game_state, raw_env.game_state.map_width, raw_env.game_state.map_height)
            )
            for entity, dense_logits in teacher_output["policy_logits"].items():
                positions = np.argwhere(masks[entity][0]).astype(np.int16, copy=False)
                legal_dense = padded_info["available_actions_mask"][entity][0]
                logits_dense = dense_logits[0, 0].detach().to(dtype=torch.float16).cpu().numpy()
                if len(positions):
                    player, x, y = positions.T
                    legal = legal_dense[player, x, y]
                    logits = logits_dense[player, x, y]
                else:
                    action_count = logits_dense.shape[-1]
                    legal = np.empty((0, action_count), dtype=np.bool_)
                    logits = np.empty((0, action_count), dtype=np.float16)
                entity_values[entity]["positions"].append(positions)
                entity_values[entity]["legal_mask"].append(legal.astype(np.bool_, copy=False))
                entity_values[entity]["teacher_logits"].append(logits.astype(np.float16, copy=False))
                entity_offsets[entity].append(entity_offsets[entity][-1] + len(positions))
    metadata = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "replay": replay_path.name,
        "replay_sha256": replay_sha,
        "teacher_sha256": teacher_sha,
        "observation_space": student_flags.obs_space.__name__,
        "teacher_tta_rot180": True,
        "turn_count": turn_count,
        "split": _split_for_digest(replay_sha),
    }
    arrays = {key: np.asarray(value) for key, value in metadata.items()}
    arrays["input_mask"] = input_mask
    for key, values in observations.items():
        arrays[f"obs__{key}"] = np.stack(values)
    for entity in ACTION_MEANINGS:
        arrays[f"{entity}_offsets"] = np.asarray(entity_offsets[entity], dtype=np.int64)
        arrays[f"{entity}_positions"] = np.concatenate(entity_values[entity]["positions"], axis=0)
        arrays[f"{entity}_legal_mask"] = np.concatenate(entity_values[entity]["legal_mask"], axis=0)
        arrays[f"{entity}_teacher_logits"] = np.concatenate(entity_values[entity]["teacher_logits"], axis=0)
    _atomic_save_npz(output_path, arrays)
    return metadata | {"path": output_path.name, "created": True, "size_bytes": output_path.stat().st_size}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create content-addressed scratch-distillation shards.")
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--teacher-config", type=Path, required=True)
    parser.add_argument("--student-config", type=Path, default=Path("conf/survival_strategic.yaml"))
    parser.add_argument("--expected-teacher-sha256")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-replays", type=int)
    parser.add_argument("--max-turns", type=int)
    parser.add_argument(
        "--legacy-prepared-cache-dir",
        type=Path,
        help="Reuse LuxPythonEnvGym prepared teacher logits and recompute only strategic observations.",
    )
    parser.add_argument("--seed", type=int, default=2021)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    device = torch.device(args.device)
    if args.teacher_checkpoint.stat().st_size < 1024 and args.teacher_checkpoint.read_bytes().startswith(
        b"version https://git-lfs"
    ):
        raise ValueError(f"Teacher checkpoint is a Git LFS pointer; run `git lfs pull`: {args.teacher_checkpoint}")
    teacher_sha = sha256_file(args.teacher_checkpoint)
    if args.expected_teacher_sha256 and teacher_sha != args.expected_teacher_sha256:
        raise ValueError(f"Teacher SHA mismatch: expected={args.expected_teacher_sha256} actual={teacher_sha}")
    if args.legacy_prepared_cache_dir is not None and teacher_sha != FIRST_PLACE_TEACHER_SHA256:
        raise ValueError("Legacy prepared caches are only compatible with the official first-place teacher")
    student_flags = _load_flags(args.student_config, args.device)
    teacher_flags = _load_flags(args.teacher_config, args.device)
    if student_flags.obs_space is not SurvivalStrategicObs:
        raise ValueError("student config must use SurvivalStrategicObs")
    teacher_model = None
    if args.legacy_prepared_cache_dir is None:
        teacher_model = create_model(student_flags, device, teacher_model_flags=teacher_flags, is_teacher_model=True)
        checkpoint = torch.load(args.teacher_checkpoint, map_location=device, weights_only=False)
        teacher_model.load_state_dict(checkpoint["model_state_dict"])
        teacher_model.eval()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    replays = _discover_replays(args.replay_dir)
    if args.max_replays is not None:
        replays = replays[: args.max_replays]
    records = []
    created_count = 0
    replay_progress = tqdm(replays, desc="Preparing replays", unit="replay", dynamic_ncols=True)
    for path in replay_progress:
        record = prepare_replay(
            path,
            output_dir,
            teacher_model,
            teacher_sha,
            student_flags,
            teacher_flags,
            device,
            args.max_turns,
            args.legacy_prepared_cache_dir,
        )
        records.append(record)
        created_count += int(record["created"])
        replay_progress.set_postfix(
            created=created_count,
            cached=len(records) - created_count,
            turns=sum(item["turn_count"] for item in records),
        )
    manifest = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "teacher_sha256": teacher_sha,
        "teacher_tta_rot180": True,
        "student_config": str(args.student_config),
        "legacy_prepared_cache_dir": (
            str(args.legacy_prepared_cache_dir.resolve()) if args.legacy_prepared_cache_dir is not None else None
        ),
        "replay_count": len(records),
        "sample_count": sum(record["turn_count"] for record in records),
        "size_bytes": sum(record["size_bytes"] for record in records),
        "shards": records,
    }
    temporary = output_dir / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output_dir / "manifest.json")
    print(json.dumps({key: manifest[key] for key in ("replay_count", "sample_count", "teacher_sha256")}, indent=2))


if __name__ == "__main__":
    main()
