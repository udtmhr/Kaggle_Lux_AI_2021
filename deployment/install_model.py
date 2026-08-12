"""Install or replace one model without rebuilding the agent runtime."""

import argparse
import os
import shutil
import tempfile
from pathlib import Path


def valid_model_name(value: str) -> str:
    if not value or Path(value).name != value or value in {".", ".."}:
        raise argparse.ArgumentTypeError("model name must be one filename-safe component")
    return value


def copy_atomic(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Register a checkpoint/config pair in this agent bundle.")
    parser.add_argument("name", type=valid_model_name)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()

    model_dir = Path(__file__).resolve().parent / "models" / args.name
    model_dir.mkdir(parents=True, exist_ok=True)
    copy_atomic(args.checkpoint.resolve(), model_dir / "model.pt")
    copy_atomic(args.config.resolve(), model_dir / "config.yaml")
    print(f"Installed model {args.name!r} in {model_dir}")


if __name__ == "__main__":
    main()
