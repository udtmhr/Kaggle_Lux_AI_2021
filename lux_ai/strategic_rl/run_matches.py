from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np


def _turn_stats(updates: list[str], team: int) -> tuple[int, int, float]:
    cities = {}
    city_tiles = 0
    units = 0
    for update in updates:
        fields = update.split()
        if not fields:
            continue
        if fields[0] == "c" and int(fields[1]) == team:
            cities[fields[2]] = (float(fields[3]), float(fields[4]))
        elif fields[0] == "ct" and int(fields[1]) == team:
            city_tiles += 1
        elif fields[0] == "u" and int(fields[2]) == team:
            units += 1
    deficit = sum(max(0.0, upkeep * 10.0 - fuel) for fuel, upkeep in cities.values())
    surplus = sum(max(0.0, fuel - upkeep * 10.0) for fuel, upkeep in cities.values())
    required = sum(upkeep * 10.0 for _, upkeep in cities.values())
    stranded = min(deficit, surplus) / max(required, 1.0)
    return city_tiles, units, stranded


def replay_metrics(replay: dict, candidate_player: int) -> dict[str, float | int]:
    if "stateful" in replay:
        ranks = replay.get("results", {}).get("ranks", ())
        first_place = [int(result["agentID"]) for result in ranks if int(result["rank"]) == 1]
        winner = first_place[0] if len(first_place) == 1 else -1

        city_by_turn: dict[int, int] = {}
        unit_by_turn: dict[int, int] = {}
        stranded_at_night = []
        for state in replay["stateful"]:
            turn = int(state["turn"])
            cities = [city for city in state.get("cities", {}).values() if int(city["team"]) == candidate_player]
            city_by_turn[turn] = sum(len(city.get("cityCells", ())) for city in cities)
            team_state = state.get("teamStates", {}).get(str(candidate_player), {})
            unit_by_turn[turn] = len(team_state.get("units", {}))
            if turn % 40 == 30:
                deficit = sum(
                    max(0.0, float(city["lightupkeep"]) * 10.0 - float(city["fuel"])) for city in cities
                )
                surplus = sum(
                    max(0.0, float(city["fuel"]) - float(city["lightupkeep"]) * 10.0) for city in cities
                )
                required = sum(float(city["lightupkeep"]) * 10.0 for city in cities)
                stranded_at_night.append(min(deficit, surplus) / max(required, 1.0))

        night_survival = []
        for night_start in range(30, min(max(city_by_turn, default=0), 360), 40):
            before = city_by_turn.get(night_start, 0)
            after = city_by_turn.get(night_start + 10, city_by_turn.get(max(city_by_turn), 0))
            if before:
                night_survival.append(after / before)
        final_turn = max(city_by_turn, default=0)
        return {
            "winner": winner,
            "candidate_final_city_tiles": city_by_turn.get(final_turn, 0),
            "candidate_final_units": unit_by_turn.get(final_turn, 0),
            "candidate_city_survival": float(np.mean(night_survival)) if night_survival else 1.0,
            "candidate_stranded_fuel": float(np.mean(stranded_at_night)) if stranded_at_night else 0.0,
        }

    rewards = np.asarray(replay.get("rewards", ()), dtype=np.float64)
    winner = -1 if len(rewards) != 2 or rewards[0] == rewards[1] else int(rewards.argmax())
    city_by_turn = []
    unit_by_turn = []
    stranded_at_night = []
    for turn, agents in enumerate(replay.get("steps", ())):
        observation = agents[0].get("observation", {})
        city_tiles, units, stranded = _turn_stats(observation.get("updates", []), candidate_player)
        city_by_turn.append(city_tiles)
        unit_by_turn.append(units)
        if turn % 40 == 30:
            stranded_at_night.append(stranded)
    night_survival = []
    for night_start in range(30, min(len(city_by_turn), 360), 40):
        after_night = min(night_start + 10, len(city_by_turn) - 1)
        before = city_by_turn[night_start]
        if before:
            night_survival.append(city_by_turn[after_night] / before)
    return {
        "winner": winner,
        "candidate_final_city_tiles": city_by_turn[-1] if city_by_turn else 0,
        "candidate_final_units": unit_by_turn[-1] if unit_by_turn else 0,
        "candidate_city_survival": float(np.mean(night_survival)) if night_survival else 1.0,
        "candidate_stranded_fuel": float(np.mean(stranded_at_night)) if stranded_at_night else 0.0,
    }


def run_match(
    candidate: Path,
    opponent: Path,
    candidate_player: int,
    seed: int,
    map_size: int,
    replay_path: Path,
    python: str,
    timeout: int,
) -> dict:
    agents = [str(opponent), str(candidate)]
    agents[candidate_player] = str(candidate)
    agents[1 - candidate_player] = str(opponent)
    command = [
        "lux-ai-2021",
        *agents,
        "--seed",
        str(seed),
        "--loglevel",
        "0",
        "--memory",
        "8000",
        "--maxtime",
        "20000",
        "--storeLogs",
        "false",
        "--statefulReplay",
        "true",
        "--width",
        str(map_size),
        "--height",
        str(map_size),
        "--out",
        str(replay_path),
        "--python",
        python,
    ]
    subprocess.run(command, check=True, timeout=timeout)
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    return {
        "opponent": opponent.stem,
        "seed": seed,
        "map_size": map_size,
        "candidate_player": candidate_player,
        "replay": replay_path.name,
        **replay_metrics(replay, candidate_player),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run exact matched seeds in both player orientations.")
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--opponent", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=2021)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--map-sizes", type=int, nargs="+", default=(12, 16, 24, 32))
    parser.add_argument("--python", default="python")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--resume", action="store_true", help="Skip completed seed/map/orientation records.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "games.jsonl"
    if result_path.exists() and not args.resume:
        raise FileExistsError(f"Refusing to mix evaluation runs; use a new directory or --resume: {result_path}")
    completed = set()
    if result_path.exists():
        with result_path.open(encoding="utf-8") as existing_file:
            for line in existing_file:
                record = json.loads(line)
                completed.add((int(record["seed"]), int(record["map_size"]), int(record["candidate_player"])))
    with result_path.open("a" if args.resume else "x", encoding="utf-8") as result_file:
        for seed in range(args.seed_start, args.seed_start + args.seeds):
            for map_size in args.map_sizes:
                for candidate_player in (0, 1):
                    if (seed, map_size, candidate_player) in completed:
                        continue
                    name = f"seed-{seed}-size-{map_size}-p{candidate_player}.json"
                    record = run_match(
                        args.candidate,
                        args.opponent,
                        candidate_player,
                        seed,
                        map_size,
                        args.output_dir / name,
                        args.python,
                        args.timeout,
                    )
                    result_file.write(json.dumps(record, sort_keys=True) + "\n")
                    result_file.flush()
                    print(json.dumps(record, sort_keys=True))


if __name__ == "__main__":
    main()
