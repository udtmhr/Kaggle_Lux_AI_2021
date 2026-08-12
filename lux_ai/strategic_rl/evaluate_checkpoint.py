from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from .evaluate import load_records, summarize
from .prepare_eval_agent import ROOT, checkpoint_label, prepare_eval_agent
from .run_matches import run_matched_matches

FIRST_PLACE_AGENT = (
    ROOT / "internal_testing" / "hall_of_fame" / "11-24_12-56-23_062179520_must_research" / "main.py"
)


def evaluate_checkpoint(
    checkpoint: Path,
    opponent: Path,
    output_dir: Path,
    *,
    config: Path | None = None,
    agent_dir: Path | None = None,
    opponent_name: str = "first_place",
    seed_start: int = 2021,
    seeds: int = 1,
    map_sizes: tuple[int, ...] = (12,),
    python: str = sys.executable,
    timeout: int = 600,
    resume: bool = False,
    bootstrap_samples: int = 2000,
) -> dict:
    checkpoint = checkpoint.expanduser().resolve()
    opponent = opponent.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not opponent.is_file():
        raise FileNotFoundError(f"Missing opponent agent: {opponent}")
    if seeds <= 0:
        raise ValueError("seeds must be positive")
    if any(size not in (12, 16, 24, 32) for size in map_sizes):
        raise ValueError(f"Unsupported map size: {map_sizes}")
    label = checkpoint_label(checkpoint)
    agent_dir = (
        Path(tempfile.gettempdir()) / f"lux_candidate_{label}" if agent_dir is None else agent_dir.expanduser().resolve()
    )
    bundle = prepare_eval_agent(checkpoint, agent_dir, config, force=True)
    results_path = run_matched_matches(
        Path(bundle["agent"]),
        opponent,
        output_dir,
        seed_start=seed_start,
        seeds=seeds,
        map_sizes=map_sizes,
        python=python,
        timeout=timeout,
        resume=resume,
        opponent_name=opponent_name,
    )
    report = summarize(load_records(results_path), bootstrap_samples=bootstrap_samples)
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "agent": bundle["agent"],
        "games": str(results_path),
        "report": str(report_path),
        "summary": report,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and evaluate one Lux checkpoint in matched orientations.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--opponent", type=Path, default=FIRST_PLACE_AGENT)
    parser.add_argument("--opponent-name", default="first_place")
    parser.add_argument("--config", type=Path, help="Defaults to config.yaml beside the checkpoint.")
    parser.add_argument("--agent-dir", type=Path, help="Defaults to /tmp/lux_candidate_<checkpoint step>.")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed-start", type=int, default=2021)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--map-sizes", type=int, nargs="+", default=(12,))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args()
    if args.output_dir is None:
        label = checkpoint_label(args.checkpoint)
        args.output_dir = ROOT / "outputs" / "evaluation" / f"step_{label}_vs_{args.opponent_name}"
    return args


def main() -> None:
    args = parse_args()
    try:
        result = evaluate_checkpoint(
            args.checkpoint,
            args.opponent,
            args.output_dir,
            config=args.config,
            agent_dir=args.agent_dir,
            opponent_name=args.opponent_name,
            seed_start=args.seed_start,
            seeds=args.seeds,
            map_sizes=tuple(args.map_sizes),
            python=args.python,
            timeout=args.timeout,
            resume=args.resume,
            bootstrap_samples=args.bootstrap_samples,
        )
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
