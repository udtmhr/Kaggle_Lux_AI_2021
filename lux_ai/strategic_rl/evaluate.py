from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_records(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                record = json.loads(line)
                for key in ("opponent", "seed", "candidate_player", "winner"):
                    if key not in record:
                        raise ValueError(f"Missing {key} at {path}:{line_number}")
                records.append(record)
    return records


def candidate_score(record: dict) -> float:
    winner = int(record["winner"])
    return 0.5 if winner < 0 else float(winner == int(record["candidate_player"]))


def summarize(records: list[dict], bootstrap_samples: int = 2000, seed: int = 2021) -> dict:
    grouped = defaultdict(list)
    for record in records:
        grouped[(record["opponent"], int(record["seed"]), int(record.get("map_size", -1)))].append(record)
    rng = np.random.default_rng(seed)
    report = {"schema_version": 1, "opponents": {}}
    for opponent in sorted({record["opponent"] for record in records}):
        pairs = []
        survival = []
        city_extinctions = []
        unit_extinctions = []
        for (pair_opponent, _, _), pair_records in grouped.items():
            if pair_opponent != opponent:
                continue
            orientations = {int(record["candidate_player"]) for record in pair_records}
            if orientations != {0, 1}:
                continue
            pairs.append(np.mean([candidate_score(record) for record in pair_records]))
            survival.extend(
                float(record["candidate_city_survival"])
                for record in pair_records
                if "candidate_city_survival" in record
            )
            city_extinctions.extend(
                float(int(record["candidate_final_city_tiles"]) == 0)
                for record in pair_records
                if "candidate_final_city_tiles" in record
            )
            unit_extinctions.extend(
                float(int(record["candidate_final_units"]) == 0)
                for record in pair_records
                if "candidate_final_units" in record
            )
        if not pairs:
            raise ValueError(f"No complete matched seed/orientation pairs for {opponent}")
        values = np.asarray(pairs)
        boot = np.asarray([rng.choice(values, len(values), replace=True).mean() for _ in range(bootstrap_samples)])
        report["opponents"][opponent] = {
            "matched_pairs": len(values),
            "score_rate": float(values.mean()),
            "paired_delta_from_even": float(values.mean() - 0.5),
            "bootstrap_lcb95": float(np.quantile(boot, 0.025)),
            "candidate_city_survival": float(np.mean(survival)) if survival else None,
            "candidate_city_extinction_rate": float(np.mean(city_extinctions)) if city_extinctions else None,
            "candidate_unit_extinction_rate": float(np.mean(unit_extinctions)) if unit_extinctions else None,
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate matched seed/orientation Lux evaluation results.")
    parser.add_argument("--results-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args()
    report = summarize(load_records(args.results_jsonl), args.bootstrap_samples)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
