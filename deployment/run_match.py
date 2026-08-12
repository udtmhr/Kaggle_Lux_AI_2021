"""Run an installed model through the official Lux S1 CLI."""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one installed model against an official-CLI agent.")
    parser.add_argument("model", help="Directory name below bundle/models")
    parser.add_argument("opponent", type=Path)
    parser.add_argument("--candidate-player", type=int, choices=(0, 1), default=0)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--out", type=Path, default=Path("replays/model-match.json"))
    parser.add_argument("--maxtime", type=int, default=20000)
    parser.add_argument("--engine-python", default=sys.executable)
    args = parser.parse_args()

    agent_dir = Path(__file__).resolve().parent
    model_dir = agent_dir / "models" / args.model
    required = (model_dir / "config.yaml", model_dir / "model.pt")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Model {args.model!r} is incomplete; missing: {missing}")
    if shutil.which("lux-ai-2021") is None:
        raise FileNotFoundError("lux-ai-2021 was not found on PATH")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    agents = [str(args.opponent.resolve()), str(agent_dir / "launcher.py")]
    agents[args.candidate_player] = str(agent_dir / "launcher.py")
    agents[1 - args.candidate_player] = str(args.opponent.resolve())
    command = [
        "lux-ai-2021",
        *agents,
        "--python",
        # Keep a venv interpreter path intact. Resolving the symlink can bypass
        # that environment and make the opponent lose its installed packages.
        str(Path(args.engine_python).absolute()),
        "--seed",
        str(args.seed),
        "--maxtime",
        str(args.maxtime),
        "--statefulReplay",
        "true",
        "--out",
        str(args.out.resolve()),
    ]
    env = os.environ.copy()
    env["LUX_AGENT_MODEL"] = args.model
    print("Running:", " ".join(command))
    subprocess.run(command, check=True, env=env)


if __name__ == "__main__":
    main()
