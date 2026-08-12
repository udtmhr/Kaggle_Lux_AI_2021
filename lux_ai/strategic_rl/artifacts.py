from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

RUN_SCHEMA_VERSION = 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_torch_save(payload: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_run_manifest(run_dir: Path, values: Mapping[str, Any]) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"schema_version": RUN_SCHEMA_VERSION, **values}
    path = run_dir / "run_manifest.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return path


def validate_run(run_dir: Path) -> dict[str, Any]:
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing run manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != RUN_SCHEMA_VERSION:
        raise ValueError(f"Unsupported run schema: {manifest.get('schema_version')}")
    checkpoint_name = manifest.get("checkpoint", "best.pt")
    checkpoint = run_dir / checkpoint_name
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
    actual_sha = sha256_file(checkpoint)
    expected_sha = manifest.get("checkpoint_sha256")
    if expected_sha and expected_sha != actual_sha:
        raise ValueError(f"Checkpoint SHA mismatch: expected={expected_sha} actual={actual_sha}")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(state, dict) or "model_state_dict" not in state:
        raise ValueError("Checkpoint must contain model_state_dict")
    return {"checkpoint": str(checkpoint), "checkpoint_sha256": actual_sha, "step": state.get("step")}


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a strategic RL run and its checkpoint.")
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate_run(args.run_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
