from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Lock

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from ..lux_gym import create_env
from ..lux_gym.act_spaces import ACTION_MEANINGS, MAX_OVERLAPPING_ACTIONS
from ..lux_gym.reward_spaces import GameResultReward
from ..nns import create_model
from ..nns.models import DictActor
from ..rl_agent.action_postprocessing import resolve_collision_rankings
from ..utils import flags_to_namespace
from .artifacts import atomic_torch_save
from .es import (
    ClipUp,
    MatchSpec,
    ParameterSpace,
    antithetic_gradient,
    load_policy_state,
    make_match_schedule,
    normalize_direction,
    orthonormalize,
    policy_fitness,
    sample_direction,
)
from .evaluate_checkpoint import FIRST_PLACE_AGENT, evaluate_checkpoint
from .prepare_eval_agent import ROOT, prepare_eval_agent, sha256_file
from .train_distill import ShardDataset, _compact_collate, _select_entity_logits, move_to
from .tta import rot180_ensemble_outputs

SCHEMA_VERSION = 1
_ENV_CREATION_LOCK = Lock()
_MODEL_CREATION_LOCK = Lock()
INTERNAL_RNG_SCHEME = "lux-internal-v2:seed-map-orientation"


def policy_state_digest(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def internal_match_rng_id(spec: MatchSpec) -> str:
    payload = f"{INTERNAL_RNG_SCHEME}:{spec.seed}:{spec.map_size}:{spec.candidate_player}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def friendly_collision_candidates(game_state, player: int, rankings: Mapping[str, torch.Tensor]) -> tuple[int, int]:
    """Count raw top-ranked unit destinations that conflict off friendly cities."""
    city_positions = {
        city_tile.pos.astuple()
        for city in game_state.players[player].cities.values()
        for city_tile in city.citytiles
    }
    destinations = set()
    stack_indices = defaultdict(int)
    candidates = 0
    active = 0
    for unit in game_state.players[player].units:
        if not unit.can_act():
            continue
        entity = "worker" if unit.is_worker() else "cart"
        position = unit.pos.astuple()
        stack_key = entity, position
        plane = stack_indices[stack_key]
        stack_indices[stack_key] += 1
        if plane >= rankings[entity].shape[0]:
            continue
        action = int(rankings[entity][plane, player, unit.pos.x, unit.pos.y, 0].item())
        meaning = ACTION_MEANINGS[entity][action]
        destination = position
        if meaning.startswith("MOVE_"):
            destination = unit.pos.translate(meaning.split("_")[1], 1).astuple()
        active += 1
        if destination in destinations and destination not in city_positions:
            candidates += 1
        destinations.add(destination)
    return candidates, active


@dataclass(frozen=True)
class ESConfig:
    parameter_scope: str
    pilot_sigmas: tuple[float, ...]
    pilot_directions: int
    pilot_games_per_candidate: int
    generations: int
    directions: int
    games_per_candidate: int
    gate_pairs: int
    gate_confirm_pairs: int
    gate_confirm_seed_start: int
    gate_require_score_improvement: bool
    active_probability: float
    active_warmup_generations: int
    active_rank: int
    tie_break_weight: float
    scale_floor: float
    seed: int
    pilot_seed_start: int
    train_seed_start: int
    gate_seed_start: int
    map_sizes: tuple[int, ...]
    action_probe_states: int
    action_disagreement_min: float
    action_disagreement_max: float
    signal_fraction_min: float
    max_city_extinction_delta: float
    formal_seeds: int
    formal_bootstrap_samples: int
    formal_min_head_to_head_score: float
    formal_min_head_to_head_lcb95: float
    formal_min_first_place_delta: float


@dataclass(frozen=True)
class OpponentSpec:
    name: str
    checkpoint: Path
    config: Path
    agent: Path | None
    source: str = "checkpoint"


@dataclass(frozen=True)
class CandidateRequest:
    candidate_id: str
    vector: torch.Tensor
    metadata: Mapping


@dataclass(frozen=True)
class DeploymentActionConfig:
    use_collision_detection: bool
    must_research: bool
    can_build_carts: bool
    force_last_turn_cart: bool
    use_rot180: bool


def _resolve(path: str | Path, base: Path = ROOT) -> Path:
    path = Path(path).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def load_es_config(path: Path) -> tuple[ESConfig, list[dict], tuple[Path, ...]]:
    values = yaml.safe_load(path.read_text(encoding="utf-8"))
    if int(values.get("schema_version", 0)) != SCHEMA_VERSION:
        raise ValueError(f"unsupported ES config schema: {values.get('schema_version')}")
    search = values["search"]
    evaluation = values["evaluation"]
    config = ESConfig(
        parameter_scope=str(search.get("parameter_scope", "all_policy")),
        pilot_sigmas=tuple(float(value) for value in search["pilot_sigmas"]),
        pilot_directions=int(search["pilot_directions"]),
        pilot_games_per_candidate=int(search["pilot_games_per_candidate"]),
        generations=int(search["generations"]),
        directions=int(search["directions"]),
        games_per_candidate=int(search["games_per_candidate"]),
        gate_pairs=int(evaluation["gate_pairs"]),
        gate_confirm_pairs=int(evaluation.get("gate_confirm_pairs", 0)),
        gate_confirm_seed_start=int(
            evaluation.get("gate_confirm_seed_start", int(evaluation["gate_seed_start"]) + 1_000_000)
        ),
        gate_require_score_improvement=bool(
            evaluation.get("gate_require_score_improvement", False)
        ),
        active_probability=float(search["active_probability"]),
        active_warmup_generations=int(search["active_warmup_generations"]),
        active_rank=int(search["active_rank"]),
        tie_break_weight=float(search["tie_break_weight"]),
        scale_floor=float(search["scale_floor"]),
        seed=int(search["seed"]),
        pilot_seed_start=int(evaluation["pilot_seed_start"]),
        train_seed_start=int(evaluation["train_seed_start"]),
        gate_seed_start=int(evaluation["gate_seed_start"]),
        map_sizes=tuple(int(value) for value in evaluation["map_sizes"]),
        action_probe_states=int(evaluation["action_probe_states"]),
        action_disagreement_min=float(evaluation["action_disagreement_min"]),
        action_disagreement_max=float(evaluation["action_disagreement_max"]),
        signal_fraction_min=float(evaluation["signal_fraction_min"]),
        max_city_extinction_delta=float(evaluation["max_city_extinction_delta"]),
        formal_seeds=int(evaluation["formal_seeds"]),
        formal_bootstrap_samples=int(evaluation["formal_bootstrap_samples"]),
        formal_min_head_to_head_score=float(evaluation["formal_min_head_to_head_score"]),
        formal_min_head_to_head_lcb95=float(evaluation["formal_min_head_to_head_lcb95"]),
        formal_min_first_place_delta=float(evaluation["formal_min_first_place_delta"]),
    )
    if not config.pilot_sigmas or any(value <= 0 for value in config.pilot_sigmas):
        raise ValueError("pilot_sigmas must contain positive values")
    for name in (
        "pilot_directions",
        "pilot_games_per_candidate",
        "generations",
        "directions",
        "games_per_candidate",
        "gate_pairs",
        "active_rank",
        "formal_seeds",
        "formal_bootstrap_samples",
    ):
        if getattr(config, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if config.action_probe_states < 0 or config.active_warmup_generations < 0:
        raise ValueError("action_probe_states and active_warmup_generations must be non-negative")
    if config.parameter_scope not in {"all_policy", "actor_head"}:
        raise ValueError(f"unsupported ES parameter scope: {config.parameter_scope}")
    if config.gate_confirm_pairs < 0:
        raise ValueError("gate_confirm_pairs must be non-negative")
    if 0 < config.gate_confirm_pairs <= config.gate_pairs:
        raise ValueError("gate_confirm_pairs must be zero or greater than gate_pairs")
    if not 0.0 <= config.active_probability <= 1.0:
        raise ValueError("active_probability must be in [0, 1]")
    if not config.map_sizes or any(size not in (12, 16, 24, 32) for size in config.map_sizes):
        raise ValueError(f"unsupported ES map sizes: {config.map_sizes}")
    history = tuple(_resolve(checkpoint) for checkpoint in values.get("history_checkpoints", ()))
    return config, list(values["opponents"]), history


def load_flags(path: Path, device: torch.device):
    values = yaml.safe_load(path.read_text(encoding="utf-8"))
    values["actor_device"] = str(device)
    values["learner_device"] = str(device)
    return flags_to_namespace(values), values


def resolve_opponents(
    entries: Sequence[Mapping], init_checkpoint: Path, model_config: Path, run_dir: Path
) -> tuple[OpponentSpec, ...]:
    opponents = []
    for entry in entries:
        source = str(entry.get("source", "checkpoint"))
        if source == "initial":
            opponents.append(
                OpponentSpec(
                    name=str(entry["name"]),
                    checkpoint=init_checkpoint,
                    config=model_config,
                    agent=None,
                    source=source,
                )
            )
            continue
        opponents.append(
            OpponentSpec(
                name=str(entry["name"]),
                checkpoint=_resolve(entry["checkpoint"]),
                config=_resolve(entry["config"]),
                agent=_resolve(entry["agent"]) if entry.get("agent") else None,
                source=source,
            )
        )
    names = [opponent.name for opponent in opponents]
    if len(names) != len(set(names)):
        raise ValueError("ES opponent names must be unique")
    for opponent in opponents:
        for path in (opponent.checkpoint, opponent.config):
            if not path.is_file():
                raise FileNotFoundError(path)
        if opponent.source != "initial" and (opponent.agent is None or not opponent.agent.is_file()):
            raise FileNotFoundError(f"missing official agent for {opponent.name}: {opponent.agent}")
    return tuple(opponents)


def paired_schedule(
    *,
    generation: int,
    pairs: int,
    opponents: Sequence[str],
    seed_start: int,
    map_sizes: Sequence[int],
    namespace: str,
) -> tuple[MatchSpec, ...]:
    schedule = []
    opponent_order = tuple(opponents[generation % len(opponents) :]) + tuple(
        opponents[: generation % len(opponents)]
    )
    base_pairs, extra_pairs = divmod(pairs, len(opponent_order))
    pair = 0
    for opponent_index, opponent in enumerate(opponent_order):
        opponent_pairs = base_pairs + int(opponent_index < extra_pairs)
        for _local_pair in range(opponent_pairs):
            map_size = int(map_sizes[(2 * generation + pair) % len(map_sizes)])
            seed = int(seed_start + generation * pairs + pair)
            for player in (0, 1):
                schedule.append(
                    MatchSpec(
                        match_id=f"{namespace}-g{generation:04d}-pair{pair:03d}-p{player}",
                        opponent=opponent,
                        seed=seed,
                        map_size=map_size,
                        candidate_player=player,
                    )
                )
            pair += 1
    return tuple(schedule)


def quality_gate_result(
    old_metrics: Mapping,
    new_metrics: Mapping,
    *,
    max_city_extinction_delta: float,
    require_score_improvement: bool,
) -> dict:
    score_delta = float(new_metrics["score_rate"] - old_metrics["score_rate"])
    extinction_delta = float(
        new_metrics["candidate_city_extinction_rate"]
        - old_metrics["candidate_city_extinction_rate"]
    )
    score_passed = score_delta > 0.0 if require_score_improvement else score_delta >= 0.0
    return {
        "passed": score_passed and extinction_delta <= max_city_extinction_delta,
        "score_delta": score_delta,
        "city_extinction_delta": extinction_delta,
    }


def load_deployment_action_config(agent: Path | None = None) -> DeploymentActionConfig:
    agent_root = ROOT if agent is None else agent.parent
    config_path = agent_root / "lux_ai" / "rl_agent" / "rl_agent_config.yaml"
    values = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    augmentations = tuple(values.get("data_augmentations", ()))
    unsupported = set(augmentations) - {"Rot180"}
    if unsupported:
        raise ValueError(
            f"internal ES backend does not implement deployment augmentations {sorted(unsupported)} "
            f"from {config_path}"
        )
    # Older bundled agents predate the cart ban and final-turn override. The
    # absence of can_build_carts is therefore behavioral, not merely a default.
    current_resolver = "can_build_carts" in values
    return DeploymentActionConfig(
        use_collision_detection=bool(values.get("use_collision_detection", False)),
        must_research=bool(values.get("must_research", False)),
        can_build_carts=bool(values.get("can_build_carts", True)),
        force_last_turn_cart=current_resolver,
        use_rot180="Rot180" in augmentations,
    )


def _policy_output(model, env_output: Mapping, use_tta: bool) -> Mapping:
    return rot180_ensemble_outputs(model, env_output) if use_tta else model(env_output, sample=False)


def _ranked_actions(output: Mapping) -> dict[str, torch.Tensor]:
    return {
        entity: DictActor.logits_to_actions(
            logits.flatten(0, -2), sample=False, actions_per_square=MAX_OVERLAPPING_ACTIONS
        ).view(*logits.shape[:-1], -1)
        for entity, logits in output["policy_logits"].items()
    }


class GameTracker:
    def __init__(self, player: int):
        self.player = player
        self.city_by_turn: dict[int, int] = {}
        self.unit_by_turn: dict[int, int] = {}
        self.stranded_at_night: list[float] = []

    def observe(self, game_state) -> None:
        turn = int(game_state.turn)
        player = game_state.players[self.player]
        self.city_by_turn[turn] = int(player.city_tile_count)
        self.unit_by_turn[turn] = len(player.units)
        if turn % 40 != 30:
            return
        cities = list(player.cities.values())
        deficit = sum(max(0.0, city.get_light_upkeep() * 10.0 - city.fuel) for city in cities)
        surplus = sum(max(0.0, city.fuel - city.get_light_upkeep() * 10.0) for city in cities)
        required = sum(city.get_light_upkeep() * 10.0 for city in cities)
        self.stranded_at_night.append(min(deficit, surplus) / max(required, 1.0))

    def summary(self) -> dict[str, float | int]:
        night_survival = []
        final_turn = max(self.city_by_turn, default=0)
        for night_start in range(30, min(final_turn, 360), 40):
            before = self.city_by_turn.get(night_start, 0)
            after = self.city_by_turn.get(night_start + 10, self.city_by_turn.get(final_turn, 0))
            if before:
                night_survival.append(after / before)
        return {
            "candidate_final_city_tiles": self.city_by_turn.get(final_turn, 0),
            "candidate_final_units": self.unit_by_turn.get(final_turn, 0),
            "candidate_city_survival": float(np.mean(night_survival)) if night_survival else 1.0,
            "candidate_stranded_fuel": (
                float(np.mean(self.stranded_at_night)) if self.stranded_at_night else 0.0
            ),
        }


class InternalMatchEvaluator:
    """Evaluate deterministic policies in the repository's official Dimensions-backed LuxEnv."""

    def __init__(self, model_config: Path, opponents: Sequence[OpponentSpec], device: torch.device):
        self.device = device
        self.candidate_flags, _ = load_flags(model_config, device)
        self.opponents = {opponent.name: opponent for opponent in opponents}
        self.opponent_flags = {
            opponent.name: load_flags(opponent.config, device)[0] for opponent in opponents
        }
        self.opponent_states = {
            opponent.name: load_policy_state(opponent.checkpoint) for opponent in opponents
        }
        self.candidate_action_config = load_deployment_action_config()
        self.opponent_action_configs = {
            opponent.name: load_deployment_action_config(opponent.agent)
            for opponent in opponents
        }

    def fork(self) -> InternalMatchEvaluator:
        worker = copy.copy(self)
        worker.candidate_flags = copy.copy(self.candidate_flags)
        worker.opponent_flags = {
            name: copy.copy(flags) for name, flags in self.opponent_flags.items()
        }
        return worker

    def evaluate_many(
        self,
        candidate_states: Sequence[Mapping[str, torch.Tensor]],
        schedule: Sequence[MatchSpec],
        max_workers: int,
    ) -> list[tuple[list[dict], dict[str, float]]]:
        def evaluate_one(candidate_state):
            worker = self.fork()
            records = worker.evaluate(candidate_state, schedule)
            return records, worker.last_profile

        workers = min(max_workers, len(candidate_states))
        if workers <= 1:
            return [evaluate_one(state) for state in candidate_states]
        with ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="lux-es-candidate",
        ) as executor:
            return list(executor.map(evaluate_one, candidate_states))

    @torch.inference_mode()
    def evaluate(self, candidate_state: Mapping[str, torch.Tensor], schedule: Sequence[MatchSpec]) -> list[dict]:
        evaluation_started = time.monotonic()
        self.last_profile = defaultdict(float)
        records = []
        self.candidate_digest = policy_state_digest(candidate_state)
        self.rng_scheme = INTERNAL_RNG_SCHEME
        grouped = defaultdict(list)
        for spec in schedule:
            grouped[spec.opponent].append(spec)
        for opponent_name, specs in grouped.items():
            records.extend(self._evaluate_group(candidate_state, opponent_name, specs))
        total = time.monotonic() - evaluation_started
        self.last_profile["evaluation_seconds"] = total
        forward = self.last_profile["candidate_forward_seconds"] + self.last_profile[
            "opponent_forward_seconds"
        ]
        self.last_profile["forward_fraction"] = forward / max(total, 1e-12)
        self.last_profile = dict(self.last_profile)
        return sorted(records, key=lambda record: record["match_id"])

    def _evaluate_group(
        self, candidate_state: Mapping[str, torch.Tensor], opponent_name: str, specs: Sequence[MatchSpec]
    ) -> list[dict]:
        opponent_flags = self.opponent_flags[opponent_name]
        env_flags = copy.copy(self.candidate_flags)
        env_flags.n_actor_envs = len(specs)
        env_flags.reward_space = GameResultReward
        env_flags.reward_space_kwargs = {}
        # kaggle_environments' Lux module owns a process in module-global state;
        # concurrent make/reset during LuxEnv construction can wait forever on
        # the wrong process queue. Only construction needs serialization: each
        # LuxEnv uses its own Dimensions subprocess afterwards.
        with _ENV_CREATION_LOCK:
            env = create_env(
                env_flags,
                self.device,
                teacher_flags=opponent_flags,
                seed=specs[0].seed,
            )
        try:
            for game, spec in zip(env.unwrapped, specs):
                game.configuration["seed"] = spec.seed - 1
                game.configuration["width"] = spec.map_size
                game.configuration["height"] = spec.map_size
            # Model constructors consume the global torch RNG before the full
            # state dict is loaded. Serialize that short section and restore
            # RNG state so thread scheduling cannot affect ES resume state.
            with _MODEL_CREATION_LOCK:
                rng_state = torch.get_rng_state()
                try:
                    candidate = create_model(
                        self.candidate_flags,
                        self.device,
                        teacher_model_flags=opponent_flags,
                        is_teacher_model=False,
                    )
                    opponent = create_model(
                        self.candidate_flags,
                        self.device,
                        teacher_model_flags=opponent_flags,
                        is_teacher_model=True,
                    )
                finally:
                    torch.set_rng_state(rng_state)
            candidate.load_state_dict(candidate_state, strict=True)
            candidate.eval()
            opponent.load_state_dict(self.opponent_states[opponent_name], strict=True)
            opponent.eval()
            output = env.reset(force=True)
            trackers = [GameTracker(spec.candidate_player) for spec in specs]
            completed = [False] * len(specs)
            records: list[dict] = []
            for index, tracker in enumerate(trackers):
                tracker.observe(env.unwrapped[index].game_state)
            while not all(completed):
                forward_started = time.monotonic()
                candidate_output = _policy_output(
                    candidate, output, self.candidate_action_config.use_rot180
                )
                self.last_profile["candidate_forward_seconds"] += time.monotonic() - forward_started
                forward_started = time.monotonic()
                opponent_output = _policy_output(
                    opponent, output, self.opponent_action_configs[opponent_name].use_rot180
                )
                self.last_profile["opponent_forward_seconds"] += time.monotonic() - forward_started
                action_postprocess_started = time.monotonic()
                merged = _ranked_actions(candidate_output)
                for index, spec in enumerate(specs):
                    players = (
                        (spec.candidate_player, candidate_output, self.candidate_action_config),
                        (
                            1 - spec.candidate_player,
                            opponent_output,
                            self.opponent_action_configs[opponent_name],
                        ),
                    )
                    for player, policy_output, action_config in players:
                        if action_config.use_collision_detection:
                            unresolved = _ranked_actions(policy_output)
                            collision_candidates, collision_active = friendly_collision_candidates(
                                env.unwrapped[index].game_state,
                                player,
                                {entity: actions[index] for entity, actions in unresolved.items()},
                            )
                            resolved = resolve_collision_rankings(
                                env.unwrapped[index].game_state,
                                player,
                                {
                                    entity: logits[index : index + 1].detach().cpu()
                                    for entity, logits in policy_output["policy_logits"].items()
                                },
                                must_research=action_config.must_research,
                                can_build_carts=action_config.can_build_carts,
                                force_last_turn_cart=action_config.force_last_turn_cart,
                            )
                            rankings = {
                                entity: actions.unsqueeze(1) for entity, actions in resolved.items()
                            }
                            for entity in ("worker", "cart"):
                                raw_top = unresolved[entity][index, :, player, ..., 0]
                                resolved_top = rankings[entity][0, :, player, ..., 0].to(raw_top.device)
                                active = output["info"]["available_actions_mask"][entity][
                                    index, :, player
                                ].any(dim=-1)
                                active_count = int(active.sum().item())
                                changed_count = int(((raw_top != resolved_top) & active).sum().item())
                                turn = int(env.unwrapped[index].game_state.turn)
                                turn_band = min(turn // 80, 4)
                                prefix = f"resolver.map_{spec.map_size}.turn_band_{turn_band}.{entity}"
                                self.last_profile[f"{prefix}.active"] += active_count
                                self.last_profile[f"{prefix}.changed"] += changed_count
                            collision_prefix = f"resolver.map_{spec.map_size}.turn_band_{turn_band}"
                            self.last_profile[f"{collision_prefix}.friendly_collision_candidates"] += (
                                collision_candidates
                            )
                            self.last_profile[f"{collision_prefix}.actionable_units"] += collision_active
                        else:
                            rankings = {
                                entity: actions[index : index + 1]
                                for entity, actions in _ranked_actions(policy_output).items()
                            }
                        for entity, actions in merged.items():
                            actions[index, :, player] = (
                                rankings[entity][0, :, player, ..., :MAX_OVERLAPPING_ACTIONS]
                                .to(actions.device)
                            )
                self.last_profile["action_postprocess_seconds"] += (
                    time.monotonic() - action_postprocess_started
                )
                environment_started = time.monotonic()
                output = env.step(merged)
                self.last_profile["environment_step_seconds"] += time.monotonic() - environment_started
                for index, (spec, tracker) in enumerate(zip(specs, trackers)):
                    if completed[index]:
                        continue
                    tracker.observe(env.unwrapped[index].game_state)
                    if not bool(output["done"][index]):
                        continue
                    rewards = output["reward"][index].detach().cpu().numpy()
                    winner = -1 if rewards[0] == rewards[1] else int(rewards.argmax())
                    records.append(
                        {
                            "match_id": spec.match_id,
                            "opponent": spec.opponent,
                            "seed": spec.seed,
                            "map_size": spec.map_size,
                            "candidate_player": spec.candidate_player,
                            "winner": winner,
                            "candidate_digest": self.candidate_digest,
                            "rng_scheme": self.rng_scheme,
                            "rng_id": internal_match_rng_id(spec),
                            **tracker.summary(),
                        }
                    )
                    completed[index] = True
                if any(bool(value) for value in output["done"]):
                    output = env.reset()
            return records
        finally:
            env.close()


class OfficialMatchEvaluator:
    def __init__(
        self,
        model_config: Path,
        init_checkpoint: Path,
        opponents: Sequence[OpponentSpec],
        work_dir: Path,
        python: str,
        timeout: int,
        workers: int,
    ):
        self.model_config = model_config
        self.init_checkpoint = init_checkpoint
        self.opponents = {opponent.name: opponent for opponent in opponents}
        self.work_dir = work_dir
        self.python = python
        self.timeout = timeout
        self.workers = workers
        self.candidate_dir = work_dir / "candidate_agent"
        bundle = prepare_eval_agent(
            init_checkpoint,
            self.candidate_dir,
            model_config,
            force=True,
            validate_model=False,
        )
        self.candidate_checkpoint = self.candidate_dir / "lux_ai" / "rl_agent" / init_checkpoint.name
        self.candidate_agent = Path(bundle["agent"])
        self.opponent_agents = {}
        for opponent in opponents:
            if opponent.source == "initial":
                target = work_dir / "initial_agent"
                self.opponent_agents[opponent.name] = Path(
                    prepare_eval_agent(
                        init_checkpoint,
                        target,
                        model_config,
                        force=True,
                        validate_model=False,
                    )["agent"]
                )
            else:
                self.opponent_agents[opponent.name] = opponent.agent

    def evaluate(self, candidate_state: Mapping[str, torch.Tensor], schedule: Sequence[MatchSpec]) -> list[dict]:
        atomic_torch_save({"model_state_dict": candidate_state}, self.candidate_checkpoint)
        records = []
        with tempfile.TemporaryDirectory(prefix="lux-es-matches-", dir=self.work_dir) as directory:
            temporary = Path(directory)
            if self.workers == 1:
                for spec in schedule:
                    replay = temporary / f"{spec.match_id}.json"
                    records.append(
                        {
                            "match_id": spec.match_id,
                            **self._run_match_with_cold_start_retry(spec, replay),
                        }
                    )
                return sorted(records, key=lambda record: record["match_id"])
            with ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix="lux-es") as executor:
                futures = {}
                for spec in schedule:
                    replay = temporary / f"{spec.match_id}.json"
                    future = executor.submit(self._run_match_with_cold_start_retry, spec, replay)
                    futures[future] = spec
                for future in as_completed(futures):
                    spec = futures[future]
                    records.append({"match_id": spec.match_id, **future.result()})
        return sorted(records, key=lambda record: record["match_id"])

    def _run_match_with_cold_start_retry(self, spec: MatchSpec, replay: Path) -> dict:
        max_attempts = 3
        for attempt in range(max_attempts):
            attempt_replay = replay.with_name(f"{replay.stem}-attempt{attempt}{replay.suffix}")
            try:
                record = self._run_official_worker(spec, attempt_replay)
                record["official_turn0_retries"] = attempt
                return record
            except (RuntimeError, json.JSONDecodeError) as error:
                cold_start_failure = isinstance(error, json.JSONDecodeError) or (
                    "stopped responding after turn 0" in str(error)
                    or "JSONDecodeError:" in str(error)
                )
                if not cold_start_failure:
                    raise
                if attempt == max_attempts - 1:
                    raise RuntimeError(
                        f"official match remained invalid after cold-start retries: {spec.match_id}"
                    ) from error
                time.sleep(2.0)
        raise AssertionError("unreachable")

    def _run_official_worker(self, spec: MatchSpec, replay: Path) -> dict:
        result_path = replay.with_suffix(".result.json")
        command = [
            self.python,
            "-m",
            "lux_ai.strategic_rl.official_match_worker",
            "--candidate",
            str(self.candidate_agent),
            "--opponent",
            str(self.opponent_agents[spec.opponent]),
            "--candidate-player",
            str(spec.candidate_player),
            "--seed",
            str(spec.seed),
            "--map-size",
            str(spec.map_size),
            "--replay",
            str(replay),
            "--engine-python",
            self.python,
            "--timeout",
            str(self.timeout),
            "--opponent-name",
            spec.opponent,
            "--max-time-ms",
            "60000",
            "--result",
            str(result_path),
        ]
        subprocess.run(command, check=True, timeout=self.timeout + 30)
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        if not payload["ok"]:
            raise RuntimeError(f"{payload['error_type']}: {payload['error']}")
        return payload["record"]


def materialize_state(
    model, parameter_space: ParameterSpace, vector: torch.Tensor
) -> dict[str, torch.Tensor]:
    parameter_space.assign(model, vector)
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def action_disagreement(
    model,
    parameter_space: ParameterSpace,
    center: torch.Tensor,
    candidates: Sequence[torch.Tensor],
    dataset_dir: Path,
    states: int,
    device: torch.device,
    use_tta: bool = False,
) -> float:
    dataset = ShardDataset(dataset_dir, "train")
    count = min(states, len(dataset))
    if count <= 0:
        raise ValueError("action-probe dataset has no train states")
    indices = np.linspace(0, len(dataset) - 1, count, dtype=np.int64).tolist()
    loader = DataLoader(Subset(dataset, indices), batch_size=32, collate_fn=_compact_collate)

    def predictions(vector: torch.Tensor) -> list[torch.Tensor]:
        parameter_space.assign(model, vector)
        model.eval()
        selected = []
        with torch.inference_mode():
            for batch in loader:
                batch = move_to(batch, device)
                model_input = {
                    "obs": batch["obs"],
                    "info": {
                        "input_mask": batch["input_mask"],
                        "available_actions_mask": batch["available_actions_mask"],
                    },
                }
                output = (
                    rot180_ensemble_outputs(model, model_input)
                    if use_tta
                    else model(model_input, sample=False)
                )
                pieces = []
                for entity, logits in output["policy_logits"].items():
                    positions = batch["positions"][entity]
                    active = positions[..., 0] >= 0
                    chosen = _select_entity_logits(logits, positions).argmax(-1)
                    pieces.append(chosen[active].detach().cpu())
                selected.append(torch.cat(pieces) if pieces else torch.empty(0, dtype=torch.long))
        return selected

    reference = predictions(center)
    disagreements = []
    for candidate in candidates:
        proposal = predictions(candidate)
        changed = sum(int((left != right).sum()) for left, right in zip(reference, proposal))
        total = sum(left.numel() for left in reference)
        disagreements.append(changed / max(total, 1))
    parameter_space.assign(model, center)
    return float(np.mean(disagreements))


def _write_json(path: Path, payload: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _append_jsonl(path: Path, payload: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()


def _load_jsonl(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    records = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                records[record["candidate_id"]] = record
    return records


def _result(
    evaluator,
    state: Mapping[str, torch.Tensor],
    schedule: Sequence[MatchSpec],
    tie_break_weight: float,
) -> dict:
    records = evaluator.evaluate(state, schedule)
    result = {"metrics": policy_fitness(records, tie_break_weight), "matches": records}
    if hasattr(evaluator, "last_profile"):
        result["backend_profile"] = evaluator.last_profile
    return result


def _candidate_results(
    *,
    requests: Sequence[CandidateRequest],
    evaluator,
    model,
    parameter_space: ParameterSpace,
    schedule: Sequence[MatchSpec],
    tie_break_weight: float,
    cache: dict[str, dict],
    output: Path,
    candidate_workers: int,
) -> list[dict]:
    if not requests:
        return []
    results: list[dict | None] = [None] * len(requests)
    missing_indices = []
    candidate_states = []
    for index, request in enumerate(requests):
        if request.candidate_id in cache:
            results[index] = cache[request.candidate_id]
            continue
        missing_indices.append(index)
        candidate_states.append(materialize_state(model, parameter_space, request.vector))
    if not missing_indices:
        return [result for result in results if result is not None]

    model_device = next(model.parameters()).device
    if model_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(model_device)
    started = time.monotonic()
    cpu_started = time.process_time()
    parallel = (
        candidate_workers > 1
        and len(candidate_states) > 1
        and hasattr(evaluator, "evaluate_many")
    )
    if parallel:
        evaluated = evaluator.evaluate_many(candidate_states, schedule, candidate_workers)
        evaluations = [
            {
                "metrics": policy_fitness(records, tie_break_weight),
                "matches": records,
                "backend_profile": profile,
            }
            for records, profile in evaluated
        ]
    else:
        evaluations = [
            _result(evaluator, state, schedule, tie_break_weight) for state in candidate_states
        ]
    elapsed = time.monotonic() - started
    cpu_seconds = time.process_time() - cpu_started
    peak_memory = (
        torch.cuda.max_memory_allocated(model_device) if model_device.type == "cuda" else None
    )
    total_games = sum(len(evaluation["matches"]) for evaluation in evaluations)
    for missing_index, evaluation in zip(missing_indices, evaluations):
        request = requests[missing_index]
        result = {
            "candidate_id": request.candidate_id,
            **request.metadata,
            **evaluation,
            "elapsed_seconds": elapsed,
            "games_per_second": len(evaluation["matches"]) / max(elapsed, 1e-12),
            "process_cpu_seconds": cpu_seconds,
            "process_cpu_to_wall_ratio": cpu_seconds / max(elapsed, 1e-12),
            "candidate_group_size": len(evaluations),
            "candidate_group_games_per_second": total_games / max(elapsed, 1e-12),
            "candidate_parallel": parallel,
        }
        if peak_memory is not None:
            result["cuda_peak_memory_bytes"] = peak_memory
        _append_jsonl(output, result)
        cache[request.candidate_id] = result
        results[missing_index] = result
    return [result for result in results if result is not None]


def _candidate_result(
    *,
    candidate_id: str,
    evaluator,
    model,
    parameter_space: ParameterSpace,
    vector: torch.Tensor,
    schedule: Sequence[MatchSpec],
    tie_break_weight: float,
    cache: dict[str, dict],
    output: Path,
    metadata: Mapping,
) -> dict:
    return _candidate_results(
        requests=[CandidateRequest(candidate_id, vector, metadata)],
        evaluator=evaluator,
        model=model,
        parameter_space=parameter_space,
        schedule=schedule,
        tie_break_weight=tie_break_weight,
        cache=cache,
        output=output,
        candidate_workers=1,
    )[0]


def _history_basis(
    parameter_space: ParameterSpace,
    center: torch.Tensor,
    checkpoints: Sequence[Path],
    max_rank: int,
) -> torch.Tensor:
    vectors = []
    for checkpoint in reversed(checkpoints):
        if checkpoint.is_file():
            vectors.append(parameter_space.normalized_delta(load_policy_state(checkpoint), center))
    return orthonormalize(vectors, max_rank=max_rank)


def _save_es_state(
    path: Path,
    *,
    center: torch.Tensor,
    best_center: torch.Tensor,
    generation: int,
    sigma: float,
    optimizer: ClipUp,
    basis: torch.Tensor,
    recent_updates: Sequence[torch.Tensor],
    result_cache: Mapping[str, Mapping],
    init_sha256: str,
    config_sha256: str,
    search_seed: int,
    status: str,
) -> None:
    atomic_torch_save(
        {
            "schema_version": SCHEMA_VERSION,
            "center": center,
            "best_center": best_center,
            "generation": generation,
            "sigma": sigma,
            "clipup": optimizer.state_dict(),
            "active_basis": basis,
            "recent_updates": list(recent_updates),
            "completed_candidate_ids": sorted(result_cache),
            "completed_match_ids": sorted(
                f"{candidate_id}:{match['match_id']}"
                for candidate_id, result in result_cache.items()
                for match in result.get("matches", ())
            ),
            "init_checkpoint_sha256": init_sha256,
            "model_config_sha256": config_sha256,
            "search_seed": search_seed,
            "torch_rng_state": torch.get_rng_state(),
            "numpy_rng_state": np.random.get_state(),
            "status": status,
        },
        path,
    )


def _select_backend(args, model_config, init_checkpoint, opponents, run_dir, initial_state):
    internal = InternalMatchEvaluator(model_config, opponents, torch.device(args.device))
    if args.backend == "internal":
        return internal, {"selected": "internal", "parity": "skipped_by_user"}
    official = OfficialMatchEvaluator(
        model_config,
        init_checkpoint,
        opponents,
        run_dir / ".official_backend",
        args.engine_python,
        args.timeout,
        args.workers,
    )
    if args.backend == "official":
        return official, {"selected": "official", "parity": "skipped_by_user"}
    if args.skip_parity:
        return internal, {"selected": "internal", "parity": "skipped_by_user"}
    schedule = paired_schedule(
        generation=0,
        pairs=2,
        opponents=[opponent.name for opponent in opponents],
        seed_start=args.parity_seed_start,
        map_sizes=(12, 16),
        namespace="parity",
    )
    official_records = official.evaluate(initial_state, schedule)
    try:
        internal_records = internal.evaluate(initial_state, schedule)
    except Exception as error:  # noqa: BLE001 - any internal failure requires the official fallback
        return official, {
            "selected": "official",
            "parity": "internal_error",
            "internal_error_type": type(error).__name__,
            "internal_error": str(error),
            "official_winners": [record["winner"] for record in official_records],
            "schedule": [asdict(spec) for spec in schedule],
        }
    internal_winners = [record["winner"] for record in internal_records]
    official_winners = [record["winner"] for record in official_records]
    matched = internal_winners == official_winners
    report = {
        "selected": "internal" if matched else "official",
        "parity": "passed" if matched else "failed",
        "internal_winners": internal_winners,
        "official_winners": official_winners,
        "schedule": [asdict(spec) for spec in schedule],
    }
    return (internal if matched else official), report


def _apply_overrides(config: ESConfig, args) -> ESConfig:
    values = asdict(config)
    for field, argument in (
        ("parameter_scope", args.parameter_scope),
        ("pilot_directions", args.pilot_directions),
        ("pilot_games_per_candidate", args.pilot_games_per_candidate),
        ("generations", args.generations),
        ("directions", args.directions),
        ("games_per_candidate", args.games_per_candidate),
        ("gate_pairs", args.gate_pairs),
        ("gate_confirm_pairs", args.gate_confirm_pairs),
        ("gate_require_score_improvement", args.gate_require_score_improvement),
        ("action_probe_states", args.action_probe_states),
        ("formal_seeds", args.formal_seeds),
    ):
        if argument is not None:
            values[field] = argument
    return ESConfig(**values)


def run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    candidate_workers = args.candidate_workers
    if candidate_workers is None:
        candidate_workers = 2 if device.type == "cuda" else 1
    if device.type == "cpu":
        torch.set_num_threads(args.cpu_threads)
        os.environ.setdefault("OMP_NUM_THREADS", str(args.cpu_threads))
        os.environ.setdefault("MKL_NUM_THREADS", str(args.cpu_threads))
    init_checkpoint = args.init_checkpoint.expanduser().resolve()
    model_config = args.config.expanduser().resolve()
    es_config_path = args.es_config.expanduser().resolve()
    dataset_dir = args.dataset_dir.expanduser().resolve()
    run_dir = args.run_dir.expanduser().resolve()
    for path in (init_checkpoint, model_config, es_config_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    config, opponent_entries, configured_history = load_es_config(es_config_path)
    config = _apply_overrides(config, args)
    opponents = resolve_opponents(opponent_entries, init_checkpoint, model_config, run_dir)
    flags, _ = load_flags(model_config, device)
    model = create_model(flags, device)
    initial_state = load_policy_state(init_checkpoint)
    model.load_state_dict(initial_state, strict=True)
    model.eval()
    parameter_space = ParameterSpace(
        model,
        scale_floor=config.scale_floor,
        parameter_scope=config.parameter_scope,
    )
    center = parameter_space.flatten_model(model)
    init_sha = sha256_file(init_checkpoint)
    config_sha = sha256_file(model_config)
    history = configured_history + tuple(_resolve(path) for path in args.history_checkpoint)
    missing_history = [path for path in history if not path.is_file()]
    if missing_history:
        raise FileNotFoundError(f"missing active-subspace history checkpoint: {missing_history[0]}")
    if config.action_probe_states > 0 and not dataset_dir.is_dir():
        raise FileNotFoundError(f"missing action-probe dataset: {dataset_dir}")
    validation = {
        "init_checkpoint": str(init_checkpoint),
        "init_checkpoint_sha256": init_sha,
        "model_config": str(model_config),
        "model_config_sha256": config_sha,
        "evolved_parameters": parameter_space.dimension,
        "evolved_parameter_names": parameter_space.names,
        "parameter_scope": config.parameter_scope,
        "excluded_prefixes": (
            ["baseline_base.", "baseline.", "intent_head."]
            if config.parameter_scope == "all_policy"
            else ["base_model.", "baseline_base.", "baseline.", "intent_head."]
        ),
        "opponents": [opponent.name for opponent in opponents],
        "history_checkpoints": [str(path) for path in history],
        "runtime": {
            "device": str(device),
            "cpu_threads": args.cpu_threads if device.type == "cpu" else None,
            "candidate_workers": candidate_workers,
        },
    }
    if args.dry_run:
        return {"status": "dry_run", **validation, "search": asdict(config)}

    if args.resume:
        if not run_dir.is_dir():
            raise FileNotFoundError(f"resume run directory does not exist: {run_dir}")
    else:
        run_dir.mkdir(parents=True, exist_ok=False)
        shutil.copy2(model_config, run_dir / "config.yaml")
        shutil.copy2(es_config_path, run_dir / "es_config.yaml")
    manifest_path = run_dir / "manifest.json"
    if args.resume:
        if not manifest_path.is_file():
            raise FileNotFoundError(f"missing ES manifest for resume: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        requested_search = json.loads(json.dumps(asdict(config)))
        saved_search = dict(manifest.get("search", {}))
        saved_search.setdefault("parameter_scope", "all_policy")
        saved_search.setdefault("gate_confirm_pairs", 0)
        saved_search.setdefault("gate_require_score_improvement", False)
        saved_search.setdefault(
            "gate_confirm_seed_start",
            int(saved_search.get("gate_seed_start", config.gate_seed_start)) + 1_000_000,
        )
        if saved_search != requested_search:
            raise ValueError("ES resume search configuration changed; use the original CLI/config")
        if manifest.get("history_checkpoints") != validation["history_checkpoints"]:
            raise ValueError("ES resume active-subspace history checkpoints changed")
        if manifest.get("runtime") not in (None, validation["runtime"]):
            raise ValueError(
                f"ES resume runtime changed: {manifest.get('runtime')} != {validation['runtime']}"
            )
        manifest.update(
            {
                "status": "resuming",
                "backend_requested": args.backend,
                **validation,
                "search": requested_search,
            }
        )
    else:
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "kind": "pure_parameter_es",
            "status": "initializing",
            "backend_requested": args.backend,
            **validation,
            "search": asdict(config),
        }
    _write_json(manifest_path, manifest)
    backend_args = args
    if args.resume and manifest.get("backend", {}).get("selected") in {"internal", "official"}:
        backend_args = copy.copy(args)
        backend_args.backend = manifest["backend"]["selected"]
    previous_backend_report = copy.deepcopy(manifest.get("backend"))
    try:
        evaluator, backend_report = _select_backend(
            backend_args, model_config, init_checkpoint, opponents, run_dir, initial_state
        )
    except Exception as error:
        manifest["status"] = "backend_parity_error"
        manifest["backend"] = {
            "selected": None,
            "parity": "error",
            "error_type": type(error).__name__,
            "error": str(error),
        }
        _write_json(manifest_path, manifest)
        raise RuntimeError(f"backend parity check failed: {type(error).__name__}: {error}") from error
    if args.resume:
        backend_report = previous_backend_report or backend_report
        backend_report["resume_reused_backend"] = True
    effective_candidate_workers = (
        candidate_workers if isinstance(evaluator, InternalMatchEvaluator) else 1
    )
    backend_report["candidate_workers"] = effective_candidate_workers
    manifest["backend"] = backend_report
    _write_json(manifest_path, manifest)

    state_path = run_dir / "latest_es.pt"
    result_cache = _load_jsonl(run_dir / "fitness.jsonl")
    recent_updates: list[torch.Tensor] = []
    if args.resume:
        if not state_path.is_file():
            raise FileNotFoundError(f"missing ES resume state: {state_path}")
        saved = torch.load(state_path, map_location="cpu", weights_only=False)
        if saved["init_checkpoint_sha256"] != init_sha or saved["model_config_sha256"] != config_sha:
            raise ValueError("ES resume checkpoint does not match the initial checkpoint/config")
        center = saved["center"].float()
        best_center = saved["best_center"].float()
        generation_start = int(saved["generation"])
        sigma = float(saved["sigma"])
        optimizer = ClipUp.load_state_dict(saved["clipup"])
        basis = saved["active_basis"].float()
        recent_updates = [value.float() for value in saved.get("recent_updates", [])]
        torch.set_rng_state(saved["torch_rng_state"])
        np.random.set_state(saved["numpy_rng_state"])
    else:
        opponent_names = [opponent.name for opponent in opponents]
        if args.force_sigma is not None:
            if args.force_sigma <= 0:
                raise ValueError("--force-sigma must be positive")
            pilot = []
            sigma = float(args.force_sigma)
            manifest["pilot"] = {"status": "skipped_by_force_sigma"}
        else:
            pilot = []
            for sigma_index, sigma_value in enumerate(config.pilot_sigmas):
                schedule = make_match_schedule(
                    generation=sigma_index,
                    games=config.pilot_games_per_candidate,
                    opponents=opponent_names,
                    seed_start=config.pilot_seed_start,
                    map_sizes=config.map_sizes,
                    namespace=f"pilot-s{sigma_index}",
                )
                plus_fitness, minus_fitness = [], []
                plus_scores, minus_scores = [], []
                probes = []
                probe_vectors = []
                for direction_index in range(config.pilot_directions):
                    seed = config.seed + sigma_index * 100_000 + direction_index
                    direction, kind = sample_direction(parameter_space.dimension, seed)
                    requests = []
                    for sign, label in ((1, "plus"), (-1, "minus")):
                        vector = parameter_space.perturb(center, direction, sigma_value, sign)
                        candidate_id = f"pilot-s{sigma_index}-d{direction_index:03d}-{label}"
                        requests.append(
                            CandidateRequest(
                                candidate_id,
                                vector,
                                {
                                    "phase": "pilot",
                                    "sigma": sigma_value,
                                    "noise_seed": seed,
                                    "kind": kind,
                                    "sign": sign,
                                },
                            )
                        )
                        probe_vectors.append(vector)
                    pair_records = _candidate_results(
                        requests=requests,
                        evaluator=evaluator,
                        model=model,
                        parameter_space=parameter_space,
                        schedule=schedule,
                        tie_break_weight=config.tie_break_weight,
                        cache=result_cache,
                        output=run_dir / "fitness.jsonl",
                        candidate_workers=effective_candidate_workers,
                    )
                    pair_fitness = [record["metrics"]["fitness"] for record in pair_records]
                    pair_scores = [record["metrics"]["score_rate"] for record in pair_records]
                    plus_fitness.append(pair_fitness[0])
                    minus_fitness.append(pair_fitness[1])
                    plus_scores.append(pair_scores[0])
                    minus_scores.append(pair_scores[1])
                if config.action_probe_states > 0:
                    probes.append(
                        action_disagreement(
                            model,
                            parameter_space,
                            center,
                            probe_vectors,
                            dataset_dir,
                            config.action_probe_states,
                            device,
                            use_tta=bool(getattr(flags, "actor_policy_tta_rot180", False)),
                        )
                    )
                disagreement = float(np.mean(probes)) if probes else None
                score_differences = np.abs(np.asarray(plus_scores) - np.asarray(minus_scores))
                fitness_differences = np.abs(np.asarray(plus_fitness) - np.asarray(minus_fitness))
                signal_fraction = float(np.mean(score_differences > 1e-12))
                signal = float(fitness_differences.mean())
                action_gate = disagreement is None or (
                    config.action_disagreement_min <= disagreement <= config.action_disagreement_max
                )
                passed = action_gate and signal_fraction >= config.signal_fraction_min
                pilot.append(
                    {
                        "sigma": sigma_value,
                        "action_disagreement": disagreement,
                        "signal_fraction": signal_fraction,
                        "mean_antithetic_fitness_difference": signal,
                        "passed": passed,
                    }
                )
                _write_json(run_dir / "pilot.json", {"results": pilot})
            eligible = [record for record in pilot if record["passed"]]
            if not eligible:
                manifest["status"] = "pilot_rejected"
                manifest["pilot"] = pilot
                _write_json(manifest_path, manifest)
                return manifest
            selected = max(
                eligible,
                key=lambda record: (
                    record["mean_antithetic_fitness_difference"],
                    -record["sigma"],
                ),
            )
            sigma = float(selected["sigma"])
            manifest["pilot"] = pilot
        optimizer = ClipUp.from_sigma(sigma, parameter_space.dimension)
        basis = _history_basis(parameter_space, center, history, config.active_rank)
        best_center = center.clone()
        generation_start = 0
        _save_es_state(
            state_path,
            center=center,
            best_center=best_center,
            generation=0,
            sigma=sigma,
            optimizer=optimizer,
            basis=basis,
            recent_updates=recent_updates,
            result_cache=result_cache,
            init_sha256=init_sha,
            config_sha256=config_sha,
            search_seed=config.seed,
            status="running",
        )
        manifest["selected_sigma"] = sigma
        manifest["status"] = "running"
        _write_json(manifest_path, manifest)

    opponent_names = [opponent.name for opponent in opponents]
    for generation in range(generation_start, config.generations):
        schedule = make_match_schedule(
            generation=generation,
            games=config.games_per_candidate,
            opponents=opponent_names,
            seed_start=config.train_seed_start,
            map_sizes=config.map_sizes,
            namespace="train",
        )
        directions = []
        direction_kinds = []
        plus_fitness, minus_fitness = [], []
        active = generation >= config.active_warmup_generations
        active_probability = config.active_probability if active else 0.0
        sampling_basis = basis if active else None
        for direction_index in range(config.directions):
            seed = config.seed + 1_000_000 + generation * 10_000 + direction_index
            direction, kind = sample_direction(
                parameter_space.dimension,
                seed,
                basis=sampling_basis,
                active_probability=active_probability,
            )
            directions.append(direction)
            direction_kinds.append(kind)
            requests = []
            for sign, label in ((1, "plus"), (-1, "minus")):
                vector = parameter_space.perturb(center, direction, sigma, sign)
                requests.append(
                    CandidateRequest(
                        f"g{generation:04d}-d{direction_index:03d}-{label}",
                        vector,
                        {
                            "phase": "search",
                            "generation": generation,
                            "direction": direction_index,
                            "sign": sign,
                            "sigma": sigma,
                            "noise_seed": seed,
                            "kind": kind,
                        },
                    )
                )
            pair_records = _candidate_results(
                requests=requests,
                evaluator=evaluator,
                model=model,
                parameter_space=parameter_space,
                schedule=schedule,
                tie_break_weight=config.tie_break_weight,
                cache=result_cache,
                output=run_dir / "fitness.jsonl",
                candidate_workers=effective_candidate_workers,
            )
            pair = [record["metrics"]["fitness"] for record in pair_records]
            plus_fitness.append(pair[0])
            minus_fitness.append(pair[1])

        gradient = antithetic_gradient(directions, plus_fitness, minus_fitness)
        update = optimizer.update(gradient)
        has_update = bool(torch.linalg.vector_norm(update) > 1e-12)
        proposal = center + parameter_space.scales * update
        two_stage_gate = config.gate_confirm_pairs > 0
        screen_namespace = "gate-screen" if two_stage_gate else "gate"
        gate_schedule = paired_schedule(
            generation=generation,
            pairs=config.gate_pairs,
            opponents=opponent_names,
            seed_start=config.gate_seed_start,
            map_sizes=config.map_sizes,
            namespace=screen_namespace,
        )
        screen_old, screen_new = _candidate_results(
            requests=[
                CandidateRequest(
                    f"g{generation:04d}-{screen_namespace}-old",
                    center,
                    {
                        "phase": "gate",
                        "gate_stage": "screen",
                        "generation": generation,
                        "role": "old",
                    },
                ),
                CandidateRequest(
                    f"g{generation:04d}-{screen_namespace}-new",
                    proposal,
                    {
                        "phase": "gate",
                        "gate_stage": "screen",
                        "generation": generation,
                        "role": "new",
                    },
                ),
            ],
            evaluator=evaluator,
            model=model,
            parameter_space=parameter_space,
            schedule=gate_schedule,
            tie_break_weight=config.tie_break_weight,
            cache=result_cache,
            output=run_dir / "fitness.jsonl",
            candidate_workers=effective_candidate_workers,
        )
        screen_result = quality_gate_result(
            screen_old["metrics"],
            screen_new["metrics"],
            max_city_extinction_delta=config.max_city_extinction_delta,
            require_score_improvement=(
                config.gate_require_score_improvement if not two_stage_gate else False
            ),
        )
        confirm_gate = None
        old_gate, new_gate = screen_old, screen_new
        final_result = screen_result
        if has_update and two_stage_gate and screen_result["passed"]:
            confirm_schedule = paired_schedule(
                generation=generation,
                pairs=config.gate_confirm_pairs,
                opponents=opponent_names,
                seed_start=config.gate_confirm_seed_start,
                map_sizes=config.map_sizes,
                namespace="gate-confirm",
            )
            confirm_old, confirm_new = _candidate_results(
                requests=[
                    CandidateRequest(
                        f"g{generation:04d}-gate-confirm-old",
                        center,
                        {
                            "phase": "gate",
                            "gate_stage": "confirm",
                            "generation": generation,
                            "role": "old",
                        },
                    ),
                    CandidateRequest(
                        f"g{generation:04d}-gate-confirm-new",
                        proposal,
                        {
                            "phase": "gate",
                            "gate_stage": "confirm",
                            "generation": generation,
                            "role": "new",
                        },
                    ),
                ],
                evaluator=evaluator,
                model=model,
                parameter_space=parameter_space,
                schedule=confirm_schedule,
                tie_break_weight=config.tie_break_weight,
                cache=result_cache,
                output=run_dir / "fitness.jsonl",
                candidate_workers=effective_candidate_workers,
            )
            final_result = quality_gate_result(
                confirm_old["metrics"],
                confirm_new["metrics"],
                max_city_extinction_delta=config.max_city_extinction_delta,
                require_score_improvement=config.gate_require_score_improvement,
            )
            confirm_gate = {
                "old": confirm_old["metrics"],
                "new": confirm_new["metrics"],
                **final_result,
            }
            old_gate, new_gate = confirm_old, confirm_new
        accepted = bool(
            has_update
            and final_result["passed"]
            and (not two_stage_gate or confirm_gate is not None)
        )
        if accepted:
            center = proposal
            best_center = center.clone()
            recent_updates.append(normalize_direction(update))
            recent_updates = recent_updates[-config.active_rank :]
            history_vectors = [value for value in recent_updates]
            history_vectors.extend(
                parameter_space.normalized_delta(load_policy_state(checkpoint), center)
                for checkpoint in reversed(history)
            )
            basis = orthonormalize(history_vectors, max_rank=config.active_rank)
            atomic_torch_save(
                {"model_state_dict": materialize_state(model, parameter_space, best_center)},
                run_dir / "best_weights.pt",
            )
        else:
            sigma *= 0.5
            optimizer.reject_and_shrink()
        generation_record = {
            "generation": generation,
            "accepted": accepted,
            "rejection_reason": None if accepted else ("quality_gate" if has_update else "zero_gradient"),
            "sigma": sigma,
            "score_delta": final_result["score_delta"],
            "city_extinction_delta": final_result["city_extinction_delta"],
            "old_gate": old_gate["metrics"],
            "new_gate": new_gate["metrics"],
            "gate_stage": "confirm" if confirm_gate is not None else "screen",
            "screen_gate": {
                "old": screen_old["metrics"],
                "new": screen_new["metrics"],
                **screen_result,
            },
            "confirm_gate": confirm_gate,
            "direction_kinds": {kind: direction_kinds.count(kind) for kind in sorted(set(direction_kinds))},
            "clipup_speed": float(torch.linalg.vector_norm(optimizer.velocity)) if optimizer.velocity is not None else 0,
        }
        _append_jsonl(run_dir / "generations.jsonl", generation_record)
        _save_es_state(
            state_path,
            center=center,
            best_center=best_center,
            generation=generation + 1,
            sigma=sigma,
            optimizer=optimizer,
            basis=basis,
            recent_updates=recent_updates,
            result_cache=result_cache,
            init_sha256=init_sha,
            config_sha256=config_sha,
            search_seed=config.seed,
            status="running",
        )
        print(json.dumps(generation_record, sort_keys=True))

    if not (run_dir / "best_weights.pt").is_file():
        atomic_torch_save(
            {"model_state_dict": materialize_state(model, parameter_space, best_center)},
            run_dir / "best_weights.pt",
        )
    manifest["status"] = "search_completed"
    manifest["generations_completed"] = config.generations
    manifest["best_checkpoint"] = str(run_dir / "best_weights.pt")
    _write_json(manifest_path, manifest)
    _save_es_state(
        state_path,
        center=center,
        best_center=best_center,
        generation=config.generations,
        sigma=sigma,
        optimizer=optimizer,
        basis=basis,
        recent_updates=recent_updates,
        result_cache=result_cache,
        init_sha256=init_sha,
        config_sha256=config_sha,
        search_seed=config.seed,
        status="search_completed",
    )
    if not args.skip_formal_eval:
        formal = formal_selection(
            best_checkpoint=run_dir / "best_weights.pt",
            init_checkpoint=init_checkpoint,
            model_config=run_dir / "config.yaml",
            run_dir=run_dir,
            config=config,
            python=args.engine_python,
            timeout=args.timeout,
            workers=args.workers,
        )
        manifest["formal_selection"] = formal
        manifest["status"] = "promoted" if formal["promoted"] else "not_promoted"
        _write_json(manifest_path, manifest)
    return manifest


def formal_selection(
    *,
    best_checkpoint: Path,
    init_checkpoint: Path,
    model_config: Path,
    run_dir: Path,
    config: ESConfig,
    python: str,
    timeout: int,
    workers: int,
) -> dict:
    output = run_dir / "formal_evaluation"
    initial_agent = Path(
        prepare_eval_agent(init_checkpoint, output / "initial_agent", model_config, force=True)["agent"]
    )
    initial_first = evaluate_checkpoint(
        init_checkpoint,
        FIRST_PLACE_AGENT,
        output / "initial_vs_first_place",
        config=model_config,
        seed_start=50_000,
        seeds=config.formal_seeds,
        map_sizes=config.map_sizes,
        python=python,
        timeout=timeout,
        bootstrap_samples=config.formal_bootstrap_samples,
        workers=workers,
    )["summary"]["opponents"]["first_place"]
    best_first = evaluate_checkpoint(
        best_checkpoint,
        FIRST_PLACE_AGENT,
        output / "best_vs_first_place",
        config=model_config,
        seed_start=50_000,
        seeds=config.formal_seeds,
        map_sizes=config.map_sizes,
        python=python,
        timeout=timeout,
        bootstrap_samples=config.formal_bootstrap_samples,
        workers=workers,
    )["summary"]["opponents"]["first_place"]
    head_to_head = evaluate_checkpoint(
        best_checkpoint,
        initial_agent,
        output / "best_vs_initial",
        config=model_config,
        opponent_name="initial_model",
        seed_start=60_000,
        seeds=config.formal_seeds,
        map_sizes=config.map_sizes,
        python=python,
        timeout=timeout,
        bootstrap_samples=config.formal_bootstrap_samples,
        workers=workers,
    )["summary"]["opponents"]["initial_model"]
    first_place_delta = best_first["score_rate"] - initial_first["score_rate"]
    extinction_delta = (
        best_first["candidate_city_extinction_rate"] - initial_first["candidate_city_extinction_rate"]
    )
    reasons = []
    if head_to_head["score_rate"] <= config.formal_min_head_to_head_score:
        reasons.append("head_to_head_score")
    if head_to_head["bootstrap_lcb95"] < config.formal_min_head_to_head_lcb95:
        reasons.append("head_to_head_lcb95")
    if first_place_delta < config.formal_min_first_place_delta:
        reasons.append("first_place_non_regression")
    if extinction_delta > config.max_city_extinction_delta:
        reasons.append("city_extinction")
    result = {
        "promoted": not reasons,
        "reasons": reasons,
        "initial_vs_first_place": initial_first,
        "best_vs_first_place": best_first,
        "best_vs_initial": head_to_head,
        "first_place_score_delta": first_place_delta,
        "first_place_city_extinction_delta": extinction_delta,
    }
    _write_json(output / "promotion.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a Lux policy with pure parameter-space ES.")
    parser.add_argument("--init-checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True, help="Model config matching the initial checkpoint.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--es-config", type=Path, default=ROOT / "conf" / "survival_strategic_es.yaml")
    parser.add_argument(
        "--dataset-dir", type=Path, default=ROOT / "outputs" / "datasets" / "distill_firstplace_selfplay_v1"
    )
    parser.add_argument("--backend", choices=("auto", "internal", "official"), default="auto")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=4,
        help="PyTorch intra-op threads for the CPU backend; keep this modest for small forwards.",
    )
    parser.add_argument(
        "--candidate-workers",
        type=int,
        choices=(1, 2),
        help="Concurrent candidate evaluations; defaults to 2 on CUDA and 1 on CPU.",
    )
    parser.add_argument("--engine-python", default=sys.executable)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--parity-seed-start", type=int, default=10_000)
    parser.add_argument("--history-checkpoint", type=Path, action="append", default=[])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-parity", action="store_true")
    parser.add_argument("--skip-formal-eval", action="store_true")
    parser.add_argument(
        "--force-sigma",
        type=float,
        help="Skip the usefulness pilot and use this sigma (intended for smoke/debug runs).",
    )
    parser.add_argument("--parameter-scope", choices=("all_policy", "actor_head"))
    parser.add_argument("--pilot-directions", type=int)
    parser.add_argument("--pilot-games-per-candidate", type=int)
    parser.add_argument("--generations", type=int)
    parser.add_argument("--directions", type=int)
    parser.add_argument("--games-per-candidate", type=int)
    parser.add_argument("--gate-pairs", type=int)
    parser.add_argument("--gate-confirm-pairs", type=int)
    parser.add_argument(
        "--gate-require-score-improvement",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Require a strictly positive score delta at the final quality-gate stage.",
    )
    parser.add_argument("--action-probe-states", type=int)
    parser.add_argument("--formal-seeds", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers <= 0 or args.cpu_threads <= 0:
        raise SystemExit("--workers and --cpu-threads must be positive")
    try:
        result = run(args)
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
