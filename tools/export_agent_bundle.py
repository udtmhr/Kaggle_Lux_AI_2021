"""Export a reusable Lux S1 inference runtime with an initial selectable model."""

import argparse
import shutil
import tarfile
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if Path(args.model_name).name != args.model_name or args.model_name in {".", ".."}:
        parser.error("--model-name must be one filename-safe component")
    for source in (args.checkpoint, args.config):
        if not source.is_file():
            parser.error(f"file does not exist: {source}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="lux-agent-export-") as temporary:
        bundle = Path(temporary) / "survival_strategic_agent"
        bundle.mkdir()
        shutil.copy2(ROOT / "main.py", bundle / "main.py")
        shutil.copy2(ROOT / "__init__.py", bundle / "__init__.py")
        shutil.copytree(ROOT / "lux_ai", bundle / "lux_ai", ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pt"))
        for name in ("launcher.py", "install_model.py", "run_match.py", "setup_venv.sh"):
            shutil.copy2(ROOT / "deployment" / name, bundle / name)
        model_dir = bundle / "models" / args.model_name
        model_dir.mkdir(parents=True)
        shutil.copy2(args.checkpoint, model_dir / "model.pt")
        shutil.copy2(args.config, model_dir / "config.yaml")
        with tarfile.open(args.output, "w:gz") as archive:
            archive.add(bundle, arcname=bundle.name)
    print(f"Exported reusable agent bundle: {args.output.resolve()}")


if __name__ == "__main__":
    main()
