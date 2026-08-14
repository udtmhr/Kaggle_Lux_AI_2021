"""Replay-based endgame state curriculum utilities."""

from __future__ import annotations

import copy
import zlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

TURN_BANDS = ((0, 79), (80, 159), (160, 239), (240, 319), (320, 359))


def encode_action_dict(action: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    encoded = {}
    for key, value in action.items():
        array = np.asarray(value, dtype=np.int16)
        encoded[key] = {
            "shape": tuple(array.shape),
            "data": zlib.compress(array.tobytes(), level=1),
        }
    return encoded


def decode_action_dict(action: Mapping[str, Mapping[str, Any]]) -> dict[str, np.ndarray]:
    return {
        key: np.frombuffer(zlib.decompress(value["data"]), dtype=np.int16)
        .reshape(value["shape"])
        .astype(np.int64)
        for key, value in action.items()
    }


def turn_band(turn: int) -> int:
    return min(max(int(turn), 0) // 80, len(TURN_BANDS) - 1)


def capture_env_snapshot(env, opponent: str = "unknown") -> dict[str, Any]:
    base = env.unwrapped
    return {
        "schema_version": 1,
        "configuration": copy.deepcopy(base.episode_configuration),
        "actions": copy.deepcopy(base.raw_action_history),
        "turn": int(base.game_state.turn),
        "map_size": int(base.game_state.map_width),
        "opponent": str(opponent),
    }


def restore_env_snapshot(env, snapshot: Mapping[str, Any]):
    """Restore by deterministic engine replay through the complete wrapper stack."""
    if int(snapshot.get("schema_version", 0)) != 1:
        raise ValueError("Unsupported curriculum snapshot schema")
    base = env.unwrapped
    configuration = copy.deepcopy(snapshot["configuration"])
    # LuxEnv.reset increments the seed before starting Dimensions.
    configuration["seed"] = int(configuration["seed"]) - 1
    base.configuration = configuration
    output = env.reset()
    for action in snapshot["actions"]:
        output = env.step(decode_action_dict(action))
    if int(base.game_state.turn) != int(snapshot["turn"]):
        raise RuntimeError("Snapshot replay reached a different turn")
    return output


@dataclass
class SnapshotEntry:
    snapshot_id: int
    payload: dict[str, Any]
    priority: float = 1.0

    @property
    def stratum(self) -> tuple[int, int, str]:
        return turn_band(self.payload["turn"]), int(self.payload["map_size"]), str(self.payload["opponent"])


@dataclass
class SnapshotPool:
    capacity: int = 2048
    td_error_ema_decay: float = 0.9
    entries: list[SnapshotEntry] = field(default_factory=list)
    next_id: int = 0

    def add(self, payload: Mapping[str, Any], priority: float = 1.0) -> int:
        entry = SnapshotEntry(self.next_id, copy.deepcopy(dict(payload)), max(abs(float(priority)), 1e-6))
        self.next_id += 1
        if len(self.entries) >= self.capacity:
            # Evict from the most populated stratum to preserve rare states.
            counts: dict[tuple[int, int, str], int] = {}
            for current in self.entries:
                counts[current.stratum] = counts.get(current.stratum, 0) + 1
            largest = max(counts, key=counts.get)
            self.entries.pop(next(i for i, current in enumerate(self.entries) if current.stratum == largest))
        self.entries.append(entry)
        return entry.snapshot_id

    def update_priority(self, snapshot_id: int, absolute_td_error: float) -> None:
        for entry in self.entries:
            if entry.snapshot_id == snapshot_id:
                value = max(abs(float(absolute_td_error)), 1e-6)
                entry.priority = self.td_error_ema_decay * entry.priority + (1 - self.td_error_ema_decay) * value
                return
        raise KeyError(snapshot_id)

    def sample(self, rng: np.random.Generator, prioritized_probability: float = 0.8) -> Optional[SnapshotEntry]:
        if not self.entries:
            return None
        strata = sorted({entry.stratum for entry in self.entries})
        stratum = strata[int(rng.integers(len(strata)))]
        candidates = [entry for entry in self.entries if entry.stratum == stratum]
        if rng.random() < prioritized_probability:
            weights = np.asarray([entry.priority for entry in candidates], dtype=np.float64)
            index = int(rng.choice(len(candidates), p=weights / weights.sum()))
        else:
            index = int(rng.integers(len(candidates)))
        return candidates[index]

    def choose_episode_start(self, rng: np.random.Generator, snapshot_probability: float = 0.3):
        if rng.random() >= snapshot_probability:
            return None
        return self.sample(rng, prioritized_probability=0.8)
