from __future__ import annotations

import argparse
import json
from pathlib import Path

from .run_matches import run_match


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one isolated official Lux match for ES.")
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--opponent", type=Path, required=True)
    parser.add_argument("--candidate-player", type=int, choices=(0, 1), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--map-size", type=int, choices=(12, 16, 24, 32), required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--engine-python", required=True)
    parser.add_argument("--timeout", type=int, required=True)
    parser.add_argument("--opponent-name", required=True)
    parser.add_argument("--max-time-ms", type=int, default=60_000)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    try:
        record = run_match(
            args.candidate,
            args.opponent,
            args.candidate_player,
            args.seed,
            args.map_size,
            args.replay,
            args.engine_python,
            args.timeout,
            args.opponent_name,
            max_time_ms=args.max_time_ms,
        )
        payload = {"ok": True, "record": record}
    except (RuntimeError, json.JSONDecodeError) as error:
        payload = {
            "ok": False,
            "error_type": type(error).__name__,
            "error": str(error),
        }
    _write_json(args.result, payload)


if __name__ == "__main__":
    main()
