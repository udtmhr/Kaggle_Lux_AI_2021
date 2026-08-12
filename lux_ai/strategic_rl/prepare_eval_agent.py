from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import torch
import yaml

from ..nns import create_model
from ..utils import flags_to_namespace

ROOT = Path(__file__).resolve().parents[2]


def checkpoint_label(checkpoint: Path) -> str:
    label = checkpoint.stem
    label = label.removesuffix("_weights")
    if not label or any(
        character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for character in label
    ):
        raise ValueError(f"Checkpoint name is not safe for an evaluation directory: {checkpoint.name}")
    return label


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_checkpoint(checkpoint: Path, config: Path) -> int:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(state, dict) or "model_state_dict" not in state:
        raise ValueError(f"Checkpoint must contain model_state_dict: {checkpoint}")
    flags = flags_to_namespace(yaml.safe_load(config.read_text(encoding="utf-8")))
    model = create_model(flags, torch.device("cpu"))
    model.load_state_dict(state["model_state_dict"], strict=True)
    return sum(parameter.numel() for parameter in model.parameters())


def prepare_eval_agent(
    checkpoint: Path,
    output_dir: Path,
    config: Path | None = None,
    *,
    force: bool = False,
    validate_model: bool = True,
) -> dict[str, str | int]:
    checkpoint = checkpoint.expanduser().resolve()
    config = checkpoint.parent / "config.yaml" if config is None else config.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
    if not config.is_file():
        raise FileNotFoundError(f"Missing model config: {config}")
    if output_dir == ROOT or output_dir == output_dir.parent:
        raise ValueError(f"Unsafe output directory: {output_dir}")
    if output_dir.exists() and not force:
        raise FileExistsError(f"Output already exists; choose another directory or pass --force: {output_dir}")

    parameter_count = validate_checkpoint(checkpoint, config) if validate_model else 0
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        shutil.copy2(ROOT / "main.py", temporary / "main.py")
        shutil.copy2(ROOT / "__init__.py", temporary / "__init__.py")
        shutil.copytree(
            ROOT / "lux_ai",
            temporary / "lux_ai",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pt"),
        )
        model_dir = temporary / "lux_ai" / "rl_agent"
        shutil.copy2(config, model_dir / "config.yaml")
        bundled_checkpoint = model_dir / checkpoint.name
        shutil.copy2(checkpoint, bundled_checkpoint)
        source_sha = sha256_file(checkpoint)
        if sha256_file(bundled_checkpoint) != source_sha:
            raise OSError("Bundled checkpoint SHA256 does not match its source")
        if len(list(model_dir.glob("*.pt"))) != 1:
            raise RuntimeError(f"Evaluation agent must contain exactly one checkpoint: {model_dir}")

        if output_dir.exists():
            shutil.rmtree(output_dir)
        os.replace(temporary, output_dir)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)

    return {
        "agent": str(output_dir / "main.py"),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": source_sha,
        "config": str(config),
        "output_dir": str(output_dir),
        "parameter_count": parameter_count,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create an isolated Lux evaluation agent from one checkpoint.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--config", type=Path, help="Defaults to config.yaml beside the checkpoint.")
    parser.add_argument("--output-dir", type=Path, help="Defaults to /tmp/lux_candidate_<checkpoint step>.")
    parser.add_argument("--force", action="store_true", help="Replace the exact output directory if it exists.")
    args = parser.parse_args()
    if args.output_dir is None:
        try:
            label = checkpoint_label(args.checkpoint)
        except ValueError as error:
            parser.error(str(error))
        args.output_dir = Path(tempfile.gettempdir()) / f"lux_candidate_{label}"
    return args


def main() -> None:
    args = parse_args()
    try:
        result = prepare_eval_agent(
            args.checkpoint,
            args.output_dir,
            args.config,
            force=args.force,
        )
    except (FileNotFoundError, FileExistsError, OSError, RuntimeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
