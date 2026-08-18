# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import math
import numpy as np
from omegaconf import OmegaConf
import os
from pathlib import Path
import pprint
import threading
import time
import timeit
import traceback
from types import SimpleNamespace
from typing import Dict, Mapping, Optional, Tuple, Union
import wandb
import warnings

import torch
from torch import amp
from torch import multiprocessing as mp
from torch import nn
from torch.nn import functional as F

from .core import prof, td_lambda, upgo, vtrace
from .core.buffer_utils import (
    Buffers,
    create_buffers,
    fill_buffers_inplace,
    stack_buffers,
    split_buffers,
    buffers_apply,
)
from ..lux_gym import create_env
from ..lux_gym.act_spaces import ACTION_MEANINGS, MAX_OVERLAPPING_ACTIONS
from ..nns import create_model
from ..nns.models import DictActor
from ..utils import flags_to_namespace
from ..strategic_rl.artifacts import atomic_torch_save
from ..strategic_rl.league import (
    LeagueSampler,
    PFSPSampler,
    learner_player_mask,
    merge_player_actions_inplace,
    opponents_from_config,
    rule_based_guidance,
)
from ..strategic_rl.schedules import teacher_kl_coefficient
from ..strategic_rl.behavior_kl import BehaviorKLController, masked_normalized_entropy, masked_policy_kl
from ..strategic_rl.categorical_value import hl_gauss_loss, support_outside_fraction
from ..strategic_rl.curriculum import SnapshotPool
from ..strategic_rl.tta import rot180_ensemble_outputs


KL_DIV_LOSS = nn.KLDivLoss(reduction="none")
logging.basicConfig(
    format=("[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] %(message)s"),
    level=0,
)


def combine_policy_logits_to_log_probs(
    behavior_policy_logits: torch.Tensor, actions: torch.Tensor, actions_taken_mask: torch.Tensor
) -> torch.Tensor:
    """
    Combines all policy_logits at a given step to get a single action_log_probs value for that step

    Initial shape: time, batch, 1, players, x, y, n_actions
    Returned shape: time, batch, players
    """
    # Get the action probabilities
    probs = F.softmax(behavior_policy_logits, dim=-1)
    # Ignore probabilities for actions that were not used
    probs = actions_taken_mask * probs
    # Select the probabilities for actions that were taken by stacked agents and sum these
    selected_probs = torch.gather(probs, -1, actions)
    # Convert the probs to conditional probs, since we sample without replacement
    remaining_probability_density = 1.0 - torch.cat(
        [
            torch.zeros((*selected_probs.shape[:-1], 1), device=selected_probs.device, dtype=selected_probs.dtype),
            selected_probs[..., :-1].cumsum(dim=-1),
        ],
        dim=-1,
    )
    # Avoid division by zero
    remaining_probability_density = remaining_probability_density + torch.where(
        remaining_probability_density == 0,
        torch.ones_like(remaining_probability_density),
        torch.zeros_like(remaining_probability_density),
    )
    conditional_selected_probs = selected_probs / remaining_probability_density
    # Remove 0-valued conditional_selected_probs in order to eliminate neg-inf valued log_probs
    conditional_selected_probs = conditional_selected_probs + torch.where(
        conditional_selected_probs == 0,
        torch.ones_like(conditional_selected_probs),
        torch.zeros_like(conditional_selected_probs),
    )
    log_probs = torch.log(conditional_selected_probs)
    # Sum over actions, y and x dimensions to combine log_probs from different actions
    # Squeeze out action_planes dimension as well
    return torch.flatten(log_probs, start_dim=-3, end_dim=-1).sum(dim=-1).squeeze(dim=-2)


def combine_policy_entropy(policy_logits: torch.Tensor, actions_taken_mask: torch.Tensor) -> torch.Tensor:
    """
    Computes and combines policy entropy for a given step.
    NB: We are just computing the sum of individual entropies, not the joint entropy, because I don't think there is
    an efficient way to compute the joint entropy?

    Initial shape: time, batch, action_planes, players, x, y, n_actions
    Returned shape: time, batch, players
    """
    policy = F.softmax(policy_logits, dim=-1)
    log_policy = F.log_softmax(policy_logits, dim=-1)
    log_policy_masked_zeroed = torch.where(log_policy.isneginf(), torch.zeros_like(log_policy), log_policy)
    entropies = (policy * log_policy_masked_zeroed).sum(dim=-1)
    assert actions_taken_mask.shape == entropies.shape
    entropies_masked = entropies * actions_taken_mask.float()
    # Sum over y, x, and action_planes dimensions to combine entropies from different actions
    return entropies_masked.sum(dim=-1).sum(dim=-1).squeeze(dim=-2)


def compute_teacher_kl_loss(
    learner_policy_logits: torch.Tensor, teacher_policy_logits: torch.Tensor, actions_taken_mask: torch.Tensor
) -> torch.Tensor:
    learner_policy_log_probs = F.log_softmax(learner_policy_logits, dim=-1)
    teacher_policy_log_probs = F.log_softmax(teacher_policy_logits, dim=-1)
    teacher_policy = F.softmax(teacher_policy_logits, dim=-1)
    # F.kl_div produces NaN for masked actions through 0 * (log(0) - -inf).
    # Define those zero-probability terms as zero explicitly.
    kl_terms = torch.where(
        teacher_policy > 0,
        teacher_policy.detach() * (teacher_policy_log_probs.detach() - learner_policy_log_probs),
        torch.zeros_like(teacher_policy),
    )
    kl_div = kl_terms.sum(dim=-1)
    assert actions_taken_mask.shape == kl_div.shape
    # Invalid entities may have every action masked to -inf, which makes their
    # softmax/KL NaN. Multiplication cannot mask NaN because 0 * NaN is NaN.
    kl_div_masked = torch.where(actions_taken_mask, kl_div, torch.zeros_like(kl_div))
    # Sum over y, x, and action_planes dimensions to combine kl divergences from different actions
    return kl_div_masked.sum(dim=-1).sum(dim=-1).squeeze(dim=-2)


def reduce(losses: torch.Tensor, reduction: str) -> torch.Tensor:
    if reduction == "mean":
        return losses.mean()
    elif reduction == "sum":
        return losses.sum()
    else:
        raise ValueError(f"Reduction must be one of 'sum' or 'mean', was: {reduction}")


def trajectory_weighted_mean(losses: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Average each batch/player trajectory before averaging trajectories.

    ``losses`` may already contain the sum of several entity losses at a time
    step. ``weights`` is the corresponding active entity (or state) count.
    """
    if losses.shape != weights.shape or losses.ndim != 3:
        raise ValueError(f"Expected matching [time,batch,player] tensors, got {losses.shape} and {weights.shape}")
    weights = weights.to(losses.dtype)
    denominators = weights.sum(dim=0)
    active = denominators > 0
    if not bool(active.any()):
        return losses.sum() * 0.0
    per_trajectory = losses.sum(dim=0) / denominators.clamp_min(1.0)
    return per_trajectory[active].mean()


def compute_baseline_loss(
    values: torch.Tensor,
    value_targets: torch.Tensor,
    reduction: str,
    player_mask: Optional[torch.Tensor] = None,
    trajectory_normalize: bool = False,
) -> torch.Tensor:
    baseline_loss = F.smooth_l1_loss(values, value_targets.detach(), reduction="none")
    if player_mask is not None:
        baseline_loss = baseline_loss * player_mask
    if trajectory_normalize:
        weights = torch.ones_like(baseline_loss) if player_mask is None else player_mask
        return trajectory_weighted_mean(baseline_loss, weights)
    return reduce(baseline_loss, reduction=reduction)


def compute_categorical_baseline_loss(
    logits: torch.Tensor,
    value_targets: torch.Tensor,
    reduction: str,
    player_mask: Optional[torch.Tensor],
    trajectory_normalize: bool,
    value_min: float,
    value_max: float,
    sigma_ratio: float,
) -> torch.Tensor:
    loss = hl_gauss_loss(
        logits, value_targets, value_min=value_min, value_max=value_max, sigma_ratio=sigma_ratio
    )
    if player_mask is not None:
        loss = loss * player_mask
    if trajectory_normalize:
        weights = torch.ones_like(loss) if player_mask is None else player_mask
        return trajectory_weighted_mean(loss, weights)
    return reduce(loss, reduction=reduction)


def explained_variance(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    targets = targets.detach().float()
    predictions = predictions.detach().float()
    variance = targets.var(unbiased=False)
    if variance <= 1e-12:
        return float("nan")
    return float((1.0 - (targets - predictions).var(unbiased=False) / variance).item())


def compute_policy_gradient_loss(
    action_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    reduction: str,
    trajectory_weights: Optional[torch.Tensor] = None,
    action_counts: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    cross_entropy = -action_log_probs.view_as(advantages)
    # if action_counts is not None:
    #     cross_entropy = cross_entropy / action_counts.clamp_min(1.0)
    losses = cross_entropy * advantages.detach()
    if trajectory_weights is not None:
        return trajectory_weighted_mean(losses, trajectory_weights)
    return reduce(losses, reduction)


def _load_model_state(model: nn.Module, state_dict: Mapping, allow_new_intent_head: bool) -> None:
    allow_value_migration = getattr(model, "value_critic", "scalar") == "categorical_hl_gauss"
    if not allow_new_intent_head and not allow_value_migration:
        model.load_state_dict(state_dict)
        return
    if allow_value_migration:
        current = model.state_dict()
        invalid = [
            key for key, value in state_dict.items()
            if (key not in current or current[key].shape != value.shape)
            and not key.startswith("baseline.")
        ]
        if invalid:
            raise RuntimeError(f"Incompatible checkpoint tensors outside value head: {invalid}")
        state_dict = {
            key: value for key, value in state_dict.items()
            if key in current and current[key].shape == value.shape
        }
    incompatible = model.load_state_dict(state_dict, strict=False)
    unexpected = list(incompatible.unexpected_keys)
    missing = [
        key for key in incompatible.missing_keys
        if not (allow_new_intent_head and key.startswith("intent_head."))
        and not (allow_value_migration and key.startswith("baseline."))
    ]
    if unexpected or missing:
        raise RuntimeError(f"Incompatible checkpoint: missing={missing}, unexpected={unexpected}")
    if incompatible.missing_keys:
        logging.info("Initialized new intent head while loading legacy policy weights")


def configure_trainable_parameters(model: nn.Module, intent_head_only: bool) -> tuple[list[nn.Parameter], list[str]]:
    """Freeze all non-intent parameters for the dedicated head fitting stage."""
    if intent_head_only and getattr(model, "intent_head", None) is None:
        raise ValueError("intent_head_only_finetune requires intent_aux_enabled=true")
    trainable_parameters = []
    trainable_names = []
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(not intent_head_only or name.startswith("intent_head."))
        if parameter.requires_grad:
            trainable_parameters.append(parameter)
            trainable_names.append(name)
    if not trainable_parameters:
        raise ValueError("No trainable parameters were selected")
    return trainable_parameters, trainable_names


def model_state_dict_cpu(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Return an owned CPU snapshot suitable for validation and checkpointing."""
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def state_dict_max_abs_diff(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]) -> float:
    """Return the largest absolute tensor difference, rejecting incompatible states."""
    if left.keys() != right.keys():
        missing = sorted(set(right) - set(left))
        unexpected = sorted(set(left) - set(right))
        raise RuntimeError(f"Incompatible model states: missing={missing}, unexpected={unexpected}")
    max_abs_diff = 0.0
    for name, left_tensor in left.items():
        right_tensor = right[name]
        if left_tensor.shape != right_tensor.shape or left_tensor.dtype != right_tensor.dtype:
            raise RuntimeError(
                f"Incompatible tensor {name}: {left_tensor.shape}/{left_tensor.dtype} != "
                f"{right_tensor.shape}/{right_tensor.dtype}"
            )
        if torch.is_floating_point(left_tensor) or torch.is_complex(left_tensor):
            difference = (left_tensor.detach().cpu() - right_tensor.detach().cpu()).abs()
            if difference.numel():
                max_abs_diff = max(max_abs_diff, float(difference.max().item()))
        elif not torch.equal(left_tensor.detach().cpu(), right_tensor.detach().cpu()):
            return float("inf")
    return max_abs_diff


def sync_actor_model(actor_model: nn.Module, learner_model: nn.Module, verify: bool = False) -> None:
    """Synchronize the rollout model through owned CPU tensors.

    The actor model can live in CUDA IPC memory shared with spawned rollout
    processes.  A direct cross-GPU ``load_state_dict`` into that storage can
    return without updating it, so use CPU as the synchronization boundary.
    """
    learner_state = model_state_dict_cpu(learner_model)
    actor_model.load_state_dict(learner_state)
    actor_cuda_devices = {
        tensor.device
        for tensor in actor_model.state_dict().values()
        if tensor.device.type == "cuda"
    }
    for device in actor_cuda_devices:
        torch.cuda.synchronize(device)
    if verify:
        difference = state_dict_max_abs_diff(
            model_state_dict_cpu(actor_model),
            learner_state,
        )
        if difference != 0.0:
            raise RuntimeError(f"Learner-to-actor model synchronization failed: max_abs_diff={difference}")
        logging.info("Verified learner-to-actor model synchronization: max_abs_diff=0")


@torch.inference_mode()
def actor_model_output(flags: SimpleNamespace, actor_model: nn.Module, env_output: Dict) -> Dict:
    mixed_precision = getattr(flags, "actor_mixed_precision", flags.use_mixed_precision)
    with amp.autocast("cuda", enabled=mixed_precision and flags.actor_device.type == "cuda"):
        if not getattr(flags, "actor_policy_tta_rot180", False):
            output = actor_model(env_output)
            output.pop("baseline_logits", None)
            if "policy_logits" in output:
                for k in list(output["policy_logits"].keys()):
                    if k.startswith("pre_prior_"):
                        del output["policy_logits"][k]
            return output
        output = rot180_ensemble_outputs(actor_model, env_output)
    output["actions"] = {
        entity: DictActor.logits_to_actions(
            torch.flatten(logits, start_dim=0, end_dim=-2),
            sample=True,
            actions_per_square=MAX_OVERLAPPING_ACTIONS,
        ).view(*logits.shape[:-1], -1)
        for entity, logits in output["policy_logits"].items()
        if not entity.startswith("pre_prior_")
    }
    output.pop("baseline_logits", None)
    if "policy_logits" in output:
        for k in list(output["policy_logits"].keys()):
            if k.startswith("pre_prior_"):
                del output["policy_logits"][k]
    return output


def _attach_rule_guidance(
    flags,
    env,
    env_output,
    agent_output,
    opponents,
    selected_opponents,
    learner_players,
) -> None:
    # Intent logits are learner-only outputs and do not need to occupy rollout
    # shared memory. Supervision labels are generated from the live game state.
    agent_output.pop("intent_logits", None)
    if not (getattr(flags, "rule_aux_enabled", False) or getattr(flags, "intent_aux_enabled", False)):
        return
    env_indices = []
    players = []
    for env_index, opponent_index in enumerate(selected_opponents):
        if opponents[opponent_index].kind == "selfplay":
            env_indices.extend((env_index, env_index))
            players.extend((0, 1))
        else:
            env_indices.append(env_index)
            players.append(learner_players[env_index])
    rule_actions, confidence, worker_intents, intent_mask = rule_based_guidance(
        env.unwrapped,
        env_output["info"]["available_actions_mask"],
        agent_output["actions"],
        env_indices,
        players,
        strategy=getattr(flags, "rule_aux_strategy", "economy"),
    )
    if getattr(flags, "rule_aux_enabled", False):
        agent_output["rule_actions"] = {key: value[..., 0] for key, value in rule_actions.items()}
        agent_output["rule_confidence"] = confidence
    if getattr(flags, "intent_aux_enabled", False):
        agent_output["worker_intent"] = worker_intents
        agent_output["worker_intent_mask"] = intent_mask


def _league_actions(
    flags,
    env,
    env_output: Dict,
    learner_actions: Dict[str, torch.Tensor],
    opponents,
    selected_opponents,
    learner_players,
    opponent_models,
) -> Dict[str, torch.Tensor]:
    merged = {key: value.clone() for key, value in learner_actions.items()}
    for opponent_index, opponent in enumerate(opponents):
        if opponent.kind == "selfplay":
            continue
        env_indices = [i for i, selected in enumerate(selected_opponents) if selected == opponent_index]
        if not env_indices:
            continue
        opponent_players = [1 - learner_players[i] for i in env_indices]
        if opponent.kind == "rule_based":
            opponent_actions, _, _, _ = rule_based_guidance(
                env.unwrapped,
                env_output["info"]["available_actions_mask"],
                learner_actions,
                env_indices,
                opponent_players,
                strategy=opponent.strategy,
            )
        else:
            model_input = buffers_apply(env_output, lambda value, indices=env_indices: value[indices])
            mixed_precision = getattr(flags, "actor_mixed_precision", flags.use_mixed_precision)
            with amp.autocast("cuda", enabled=mixed_precision and flags.actor_device.type == "cuda"):
                if opponent.kind == "teacher" and getattr(flags, "teacher_policy_tta_rot180", False):
                    model_output = rot180_ensemble_outputs(opponent_models[opponent_index], model_input)
                    model_actions = {
                        entity: DictActor.logits_to_actions(
                            logits.flatten(start_dim=0, end_dim=-2),
                            sample=False,
                            actions_per_square=MAX_OVERLAPPING_ACTIONS,
                        ).view(*logits.shape[:-1], -1)
                        for entity, logits in model_output["policy_logits"].items()
                    }
                else:
                    model_actions = opponent_models[opponent_index](model_input, sample=False)["actions"]
            opponent_actions = {key: torch.zeros_like(value) for key, value in learner_actions.items()}
            for local_index, env_index in enumerate(env_indices):
                for entity in opponent_actions:
                    opponent_actions[entity][env_index] = model_actions[entity][local_index]
        merge_player_actions_inplace(merged, opponent_actions, env_indices, opponent_players)
    return merged


def _set_learner_player_info(env_output, opponents, selected_opponents, learner_players, device):
    mask = learner_player_mask(opponents, selected_opponents, learner_players, device)
    env_output["info"]["learner_player_mask"] = mask
    env_output["info"]["league_opponent_id"] = torch.as_tensor(
        selected_opponents, dtype=torch.int64, device=device
    )
    action_mask = mask[:, None, :, None, None, None]
    for entity in env_output["info"]["actions_taken"]:
        env_output["info"]["actions_taken"][entity] &= action_mask


def _load_league_opponent_models(flags, league_obs_flags, league_opponents, snapshot_paths=None):
    models = {}
    for opponent_index, opponent in enumerate(league_opponents):
        if opponent.kind in {"selfplay", "rule_based"}:
            continue
        if opponent.kind == "learner_snapshot":
            if snapshot_paths is None or opponent.slot not in snapshot_paths:
                raise ValueError(f"Missing learner snapshot slot {opponent.slot} for {opponent.name}")
            model = create_model(
                flags, flags.actor_device, teacher_model_flags=league_obs_flags, is_teacher_model=False
            )
            state = torch.load(snapshot_paths[opponent.slot], map_location=torch.device("cpu"), weights_only=False)
            _load_model_state(
                model,
                state["model_state_dict"],
                allow_new_intent_head=getattr(flags, "intent_aux_enabled", False),
            )
            model.eval()
            models[opponent_index] = model
            continue
        if opponent.kind == "teacher":
            model_flags = flags_to_namespace(OmegaConf.to_container(OmegaConf.load(opponent.config)))
            model = create_model(flags, flags.actor_device, teacher_model_flags=model_flags, is_teacher_model=True)
        else:
            model = create_model(
                flags, flags.actor_device, teacher_model_flags=league_obs_flags, is_teacher_model=False
            )
        state = torch.load(Path(opponent.checkpoint), map_location=torch.device("cpu"), weights_only=False)
        _load_model_state(
            model,
            state["model_state_dict"],
            allow_new_intent_head=getattr(flags, "intent_aux_enabled", False),
        )
        model.eval()
        models[opponent_index] = model
        logging.info("Actor league loaded %s from %s", opponent.name, opponent.checkpoint)
    return models


def _refresh_learner_snapshot_models(
    flags,
    opponents,
    opponent_models,
    snapshot_paths,
    snapshot_versions,
    local_versions,
    snapshot_lock,
) -> None:
    for opponent_index, opponent in enumerate(opponents):
        if opponent.kind != "learner_snapshot":
            continue
        version = int(snapshot_versions[opponent.slot])
        if local_versions.get(opponent.slot) == version:
            continue
        with snapshot_lock:
            state = torch.load(
                snapshot_paths[opponent.slot], map_location=torch.device("cpu"), weights_only=False
            )
        _load_model_state(
            opponent_models[opponent_index],
            state["model_state_dict"],
            allow_new_intent_head=getattr(flags, "intent_aux_enabled", False),
        )
        opponent_models[opponent_index].eval()
        local_versions[opponent.slot] = version
        logging.info("Actor refreshed learner snapshot slot %d version %d", opponent.slot, version)


def _league_win_rates(opponents, league_outcomes, prior_games: float) -> dict[str, float]:
    wins, games, lock = league_outcomes
    with lock:
        return {
            opponent.name: (float(wins[index]) + 0.5 * prior_games) / (float(games[index]) + prior_games)
            for index, opponent in enumerate(opponents)
        }


def _sample_league_index(flags, opponents, fixed_sampler, pfsp_sampler, league_outcomes, rng) -> int:
    if getattr(flags, "league_sampling", "fixed") != "pfsp":
        return fixed_sampler.sample_index(rng)
    win_rates = _league_win_rates(opponents, league_outcomes, float(getattr(flags, "pfsp_prior_games", 20)))
    return pfsp_sampler.sample_index(win_rates, rng)


def _record_league_outcome(league_outcomes, opponent_index: int, score: float) -> None:
    wins, games, lock = league_outcomes
    with lock:
        wins[opponent_index] += float(score)
        games[opponent_index] += 1


@torch.no_grad()
def act(
    flags: SimpleNamespace,
    teacher_flags: Optional[SimpleNamespace],
    actor_index: int,
    free_queue: mp.SimpleQueue,
    full_queue: mp.SimpleQueue,
    actor_model: torch.nn.Module,
    league_opponents,
    league_outcomes,
    snapshot_paths,
    snapshot_versions,
    snapshot_lock,
    reward_game_counter,
    actor_policy_version,
    buffers: Buffers,
):
    if flags.debug:
        catch_me = AssertionError
    else:
        catch_me = Exception
    try:
        logging.info("Actor %i started.", actor_index)
        timings = prof.Timings()

        env = create_env(
            flags,
            device=flags.actor_device,
            teacher_flags=teacher_flags,
            reward_game_counter=reward_game_counter,
        )
        opponent_models = _load_league_opponent_models(
            flags, teacher_flags, league_opponents, snapshot_paths
        )
        local_snapshot_versions = {
            opponent.slot: int(snapshot_versions[opponent.slot])
            for opponent in league_opponents
            if opponent.kind == "learner_snapshot"
        }
        if flags.seed is not None:
            env.seed(flags.seed + actor_index * flags.n_actor_envs)
        else:
            env.seed()
        env_output = env.reset(force=True)
        league_sampler = LeagueSampler(league_opponents)
        pfsp_sampler = PFSPSampler(
            league_opponents,
            power=float(getattr(flags, "pfsp_power", 2.0)),
            teacher_floor=float(getattr(flags, "pfsp_teacher_floor", 0.15)),
            exploration=float(getattr(flags, "pfsp_exploration", 0.02)),
        )
        league_rng = np.random.default_rng(None if flags.seed is None else flags.seed + 100000 + actor_index)
        curriculum_enabled = bool(getattr(flags, "state_curriculum_enabled", False))
        snapshot_pool = SnapshotPool(capacity=int(getattr(flags, "snapshot_pool_capacity", 2048)))
        pending_snapshot_priorities = {}
        selected_opponents = [
            _sample_league_index(
                flags, league_opponents, league_sampler, pfsp_sampler, league_outcomes, league_rng
            )
            for _ in range(flags.n_actor_envs)
        ]
        learner_players = league_rng.integers(0, 2, size=flags.n_actor_envs).tolist()
        _set_learner_player_info(
            env_output, league_opponents, selected_opponents, learner_players, flags.actor_device
        )
        agent_output = actor_model_output(flags, actor_model, env_output)
        agent_output["policy_version"] = torch.full(
            (flags.n_actor_envs,), int(actor_policy_version.value), dtype=torch.int64,
            device=flags.actor_device,
        )
        _attach_rule_guidance(
            flags, env, env_output, agent_output, league_opponents, selected_opponents, learner_players
        )
        while True:
            index = free_queue.get()
            if index is None:
                break

            # Write old rollout end.
            fill_buffers_inplace(buffers[index], dict(**env_output, **agent_output), 0)

            # Do new rollout.
            for t in range(flags.unroll_length):
                timings.reset()

                agent_output = actor_model_output(flags, actor_model, env_output)
                agent_output["policy_version"] = torch.full(
                    (flags.n_actor_envs,), int(actor_policy_version.value), dtype=torch.int64,
                    device=flags.actor_device,
                )
                if curriculum_enabled and pending_snapshot_priorities:
                    for env_index, (snapshot_id, reward, previous_value, player) in list(
                        pending_snapshot_priorities.items()
                    ):
                        next_value = float(agent_output["baseline"][env_index, player].item())
                        td_error = reward + float(flags.discounting) * next_value - previous_value
                        snapshot_pool.update_priority(snapshot_id, td_error)
                    pending_snapshot_priorities.clear()
                _attach_rule_guidance(
                    flags, env, env_output, agent_output, league_opponents, selected_opponents, learner_players
                )
                timings.time("model")

                _refresh_learner_snapshot_models(
                    flags,
                    league_opponents,
                    opponent_models,
                    snapshot_paths,
                    snapshot_versions,
                    local_snapshot_versions,
                    snapshot_lock,
                )

                actions = _league_actions(
                    flags,
                    env,
                    env_output,
                    agent_output["actions"],
                    league_opponents,
                    selected_opponents,
                    learner_players,
                    opponent_models,
                )
                agent_output["actions"] = actions
                env_output = env.step(actions)
                if curriculum_enabled:
                    snapshot_interval = max(int(getattr(flags, "snapshot_capture_interval", 16)), 1)
                    capture_indices = [
                        index for index, game in enumerate(env.unwrapped)
                        if not bool(env_output["done"][index]) and game.game_state.turn % snapshot_interval == 0
                    ]
                    if capture_indices:
                        opponent_names = [league_opponents[selected_opponents[index]].name for index in capture_indices]
                        for env_index, payload in zip(
                            capture_indices, env.capture_snapshots(capture_indices, opponent_names)
                        ):
                            payload["learner_player"] = int(learner_players[env_index])
                            snapshot_id = snapshot_pool.add(payload)
                            player = learner_players[env_index]
                            pending_snapshot_priorities[env_index] = (
                                snapshot_id,
                                float(env_output["reward"][env_index, player].item()),
                                float(agent_output["baseline"][env_index, player].item()),
                                player,
                            )
                _set_learner_player_info(
                    env_output, league_opponents, selected_opponents, learner_players, flags.actor_device
                )
                if env_output["done"].any():
                    # Cache reward, done, and info["actions_taken"] from the terminal step
                    cached_reward = env_output["reward"]
                    cached_done = env_output["done"]
                    cached_info_actions_taken = env_output["info"]["actions_taken"]
                    cached_info_logging = {
                        key: val
                        for key, val in env_output["info"].items()
                        if key.startswith("LOGGING_") or key in {"learner_player_mask", "league_opponent_id"}
                    }

                    env_output = env.reset()
                    restored_metadata = {}
                    if curriculum_enabled:
                        restored = {}
                        for env_index in cached_done.nonzero(as_tuple=False).flatten().tolist():
                            entry = snapshot_pool.choose_episode_start(
                                league_rng,
                                snapshot_probability=float(getattr(flags, "snapshot_start_probability", 0.30)),
                            )
                            if entry is not None:
                                restored[env_index] = entry.payload
                                restored_metadata[env_index] = (
                                    entry.payload["opponent"], int(entry.payload.get("learner_player", 0))
                                )
                        if restored:
                            env_output = env.restore_snapshots(restored)
                    env_output["reward"] = cached_reward
                    env_output["done"] = cached_done
                    env_output["info"]["actions_taken"] = cached_info_actions_taken
                    env_output["info"].update(cached_info_logging)
                    for env_index in env_output["done"].nonzero(as_tuple=False).flatten().tolist():
                        learner_player = learner_players[env_index]
                        learner_reward = float(cached_reward[env_index, learner_player])
                        opponent_reward = float(cached_reward[env_index, 1 - learner_player])
                        score = 1.0 if learner_reward > opponent_reward else 0.0 if learner_reward < opponent_reward else 0.5
                        _record_league_outcome(league_outcomes, selected_opponents[env_index], score)
                        if env_index in restored_metadata:
                            opponent_name, learner_player = restored_metadata[env_index]
                            selected_opponents[env_index] = next(
                                index for index, opponent in enumerate(league_opponents)
                                if opponent.name == opponent_name
                            )
                            learner_players[env_index] = learner_player
                        else:
                            selected_opponents[env_index] = _sample_league_index(
                                flags,
                                league_opponents,
                                league_sampler,
                                pfsp_sampler,
                                league_outcomes,
                                league_rng,
                            )
                            learner_players[env_index] = int(league_rng.integers(0, 2))
                timings.time("step")

                fill_buffers_inplace(buffers[index], dict(**env_output, **agent_output), t + 1)
                timings.time("write")
            full_queue.put(index)

        if actor_index == 0:
            logging.info("Actor %i: %s", actor_index, timings.summary())

    except KeyboardInterrupt:
        pass  # Return silently.
    except catch_me as e:
        logging.error("Exception in worker process %i", actor_index)
        traceback.print_exc()
        print()
        raise e


def get_batch(
    flags: SimpleNamespace,
    free_queue: mp.SimpleQueue,
    full_queue: mp.SimpleQueue,
    buffers: Buffers,
    timings: prof.Timings,
    lock=threading.Lock(),
):
    with lock:
        timings.time("lock")
        indices = [full_queue.get() for _ in range(max(flags.batch_size // flags.n_actor_envs, 1))]
        timings.time("dequeue")
    batch = stack_buffers([buffers[m] for m in indices], dim=1)
    timings.time("batch")
    batch = buffers_apply(batch, lambda x: x.to(device=flags.learner_device, non_blocking=True))
    timings.time("device")
    for m in indices:
        free_queue.put(m)
    timings.time("enqueue")
    return batch


def learn(
    flags: SimpleNamespace,
    actor_model: nn.Module,
    learner_model: nn.Module,
    teacher_model: Optional[nn.Module],
    batch: Dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    grad_scaler: amp.grad_scaler,
    lr_scheduler: torch.optim.lr_scheduler,
    total_games_played: int,
    learner_step: int = 0,
    baseline_only: bool = False,
    behavior_kl_controller: Optional[BehaviorKLController] = None,
    model_update_reference: Optional[Mapping[str, torch.Tensor]] = None,
    verify_actor_sync: bool = False,
    lock=threading.Lock(),
) -> Tuple[Dict, int]:
    """Performs a learning (optimization) step."""
    with lock:
        with amp.autocast("cuda", enabled=flags.use_mixed_precision and flags.learner_device.type == "cuda"):
            flattened_batch = buffers_apply(batch, lambda x: torch.flatten(x, start_dim=0, end_dim=1))
            learner_outputs = learner_model(flattened_batch)
            learner_outputs = buffers_apply(
                learner_outputs, lambda x: x.view(flags.unroll_length + 1, flags.batch_size, *x.shape[1:])
            )
            if flags.use_teacher:
                with torch.no_grad():
                    if getattr(flags, "teacher_policy_tta_rot180", False):
                        teacher_outputs = rot180_ensemble_outputs(teacher_model, flattened_batch)
                    else:
                        teacher_outputs = teacher_model(flattened_batch)
                    teacher_outputs = buffers_apply(
                        teacher_outputs, lambda x: x.view(flags.unroll_length + 1, flags.batch_size, *x.shape[1:])
                    )
            else:
                teacher_outputs = None

            # Take final value function slice for bootstrapping.
            bootstrap_value = learner_outputs["baseline"][-1]

            # Move from obs[t] -> action[t] to action[t] -> obs[t].
            batch = buffers_apply(batch, lambda x: x[1:])
            learner_outputs = buffers_apply(learner_outputs, lambda x: x[:-1])
            if flags.use_teacher:
                teacher_outputs = buffers_apply(teacher_outputs, lambda x: x[:-1])

            combined_behavior_action_log_probs = torch.zeros(
                (flags.unroll_length, flags.batch_size, 2), device=flags.learner_device
            )
            combined_learner_action_log_probs = torch.zeros_like(combined_behavior_action_log_probs)
            combined_teacher_kl_loss = torch.zeros_like(combined_behavior_action_log_probs)
            teacher_kl_losses = {}
            combined_learner_entropy = torch.zeros_like(combined_behavior_action_log_probs)
            action_counts = torch.zeros_like(combined_behavior_action_log_probs)
            entropies = {}
            normalized_entropies = {}
            active_entity_counts = {}
            combined_forward_behavior_kl = torch.zeros_like(combined_behavior_action_log_probs)
            combined_reverse_behavior_kl = torch.zeros_like(combined_behavior_action_log_probs)
            behavior_kl_counts = torch.zeros_like(combined_behavior_action_log_probs)
            stats = {}
            for act_space in batch["actions"].keys():
                actions = batch["actions"][act_space]
                actions_taken_mask = batch["info"]["actions_taken"][act_space]

                behavior_policy_logits = batch["policy_logits"][act_space]
                behavior_action_log_probs = combine_policy_logits_to_log_probs(
                    behavior_policy_logits, actions, actions_taken_mask
                )
                combined_behavior_action_log_probs = combined_behavior_action_log_probs + behavior_action_log_probs

                learner_policy_logits = learner_outputs["policy_logits"][act_space]
                pre_prior_logits = learner_outputs["policy_logits"].get(f"pre_prior_{act_space}", learner_policy_logits)
                
                learner_action_log_probs = combine_policy_logits_to_log_probs(
                    learner_policy_logits, actions, actions_taken_mask
                )
                combined_learner_action_log_probs = combined_learner_action_log_probs + learner_action_log_probs

                # Only take entropy and KL loss for tiles where at least one action was taken
                any_actions_taken = actions_taken_mask.any(dim=-1)
                action_counts = action_counts + any_actions_taken.sum(dim=(2, 4, 5))
                if flags.use_teacher:
                    teacher_kl_loss = compute_teacher_kl_loss(
                        learner_policy_logits, teacher_outputs["policy_logits"][act_space], any_actions_taken
                    )
                else:
                    teacher_kl_loss = torch.zeros_like(combined_teacher_kl_loss)
                combined_teacher_kl_loss = combined_teacher_kl_loss + teacher_kl_loss
                n_actions_taken = any_actions_taken.sum().clamp_min(1)
                teacher_kl_losses[act_space] = (
                    (
                        reduce(
                            teacher_kl_loss,
                            reduction="sum",
                        )
                        / n_actions_taken
                    )
                    .detach()
                    .cpu()
                    .item()
                )

                learner_policy_entropy = combine_policy_entropy(learner_policy_logits, any_actions_taken)
                combined_learner_entropy = combined_learner_entropy + learner_policy_entropy
                entropies[act_space] = (
                    -(reduce(learner_policy_entropy, reduction="sum") / n_actions_taken).detach().cpu().item()
                )
                normalized_entropy_sum, active_count = masked_normalized_entropy(
                    learner_policy_logits, any_actions_taken
                )
                normalized_entropies[act_space] = float(
                    (normalized_entropy_sum / active_count.clamp_min(1)).detach().cpu().item()
                )
                active_entity_counts[act_space] = int(active_count.detach().cpu().item())
                
                # pre_prior entropy
                pre_prior_entropy_sum, _ = masked_normalized_entropy(
                    pre_prior_logits, any_actions_taken
                )
                stats.setdefault("Entropy_PrePrior", {})[act_space] = float(
                    (pre_prior_entropy_sum / active_count.clamp_min(1)).detach().cpu().item()
                )

                forward_kl, kl_counts = masked_policy_kl(
                    learner_policy_logits, behavior_policy_logits, any_actions_taken, reverse=False
                )
                reverse_kl, _ = masked_policy_kl(
                    learner_policy_logits, behavior_policy_logits, any_actions_taken, reverse=True
                )
                combined_forward_behavior_kl += forward_kl
                combined_reverse_behavior_kl += reverse_kl
                behavior_kl_counts += kl_counts
                
                # Top1 Prob and Margins
                if int(active_count) > 0:
                    probs = F.softmax(learner_policy_logits, dim=-1)
                    top2_probs, top2_indices = torch.topk(probs, 2, dim=-1)
                    top1_prob = top2_probs[..., 0]
                    top2_prob = top2_probs[..., 1]
                    top1_idx = top2_indices[..., 0]
                    margin = top1_prob - top2_prob
                    
                    active_top1 = top1_prob[any_actions_taken]
                    active_margin = margin[any_actions_taken]
                    active_top1_idx = top1_idx[any_actions_taken]
                    
                    total_active = active_top1_idx.numel()
                    for action_idx, action_name in enumerate(ACTION_MEANINGS[act_space]):
                        count = (active_top1_idx == action_idx).sum().float().item()
                        if count > 0:
                            stats.setdefault("Policy", {})[f"{act_space}_top1_action_{action_name}_fraction"] = count / total_active
                    
                    stats.setdefault("Policy", {})[f"{act_space}_top1_prob_mean"] = float(active_top1.mean().cpu().item())
                    stats.setdefault("Policy", {})[f"{act_space}_top1_prob_p95"] = float(torch.quantile(active_top1, 0.95).cpu().item())
                    stats.setdefault("Policy", {})[f"{act_space}_top1_top2_margin"] = float(active_margin.mean().cpu().item())
                    stats.setdefault("Policy", {})[f"{act_space}_logits_abs_max"] = float(learner_policy_logits[any_actions_taken].abs().max().cpu().item())
                    
                    # Rule Prior explicit stats
                    if "rule_prior" in batch and act_space in batch["rule_prior"]:
                        prior = batch["rule_prior"][act_space]
                        stats.setdefault("RulePrior", {})[f"{act_space}_abs_mean"] = float(prior[any_actions_taken].abs().mean().cpu().item())
                        stats.setdefault("RulePrior", {})[f"{act_space}_abs_max"] = float(prior[any_actions_taken].abs().max().cpu().item())
                
                # Delta logp per entity
                if int(active_count) > 0:
                    delta_logp = learner_action_log_probs - behavior_action_log_probs
                    active_delta = delta_logp[delta_logp != 0]
                    if active_delta.numel() > 0:
                        stats.setdefault("Behavior_Policy", {})[f"{act_space}_delta_logp_p05"] = float(torch.quantile(active_delta, 0.05).cpu().item())

            discounts = (~batch["done"]).float() * flags.discounting
            discounts = discounts.unsqueeze(-1).expand_as(combined_behavior_action_log_probs)
            values = learner_outputs["baseline"]
            vtrace_returns = vtrace.from_action_log_probs(
                behavior_action_log_probs=combined_behavior_action_log_probs,
                target_action_log_probs=combined_learner_action_log_probs,
                discounts=discounts,
                rewards=batch["reward"],
                values=values,
                bootstrap_value=bootstrap_value,
            )
            td_lambda_returns = td_lambda.td_lambda(
                rewards=batch["reward"],
                values=values,
                bootstrap_value=bootstrap_value,
                discounts=discounts,
                lmb=flags.lmb,
            )
            upgo_returns = upgo.upgo(
                rewards=batch["reward"],
                values=values,
                bootstrap_value=bootstrap_value,
                discounts=discounts,
                lmb=flags.lmb,
            )

            learner_player_mask_batch = batch["info"]["learner_player_mask"].to(values.dtype)
            trajectory_normalize = getattr(flags, "loss_normalization", "legacy") == "trajectory"
            vtrace_advantages = vtrace_returns.pg_advantages
            upgo_advantages = upgo_returns.advantages
            if getattr(flags, "normalize_advantages", False):
                active = (learner_player_mask_batch > 0) & (action_counts > 0)

                def normalize_advantage(advantage):
                    selected = advantage[active]
                    if selected.numel() < 2:
                        return advantage
                    normalized = (advantage - selected.mean()) / selected.std(unbiased=False).clamp_min(1e-6)
                    clip = float(getattr(flags, "advantage_clip", 5.0))
                    return normalized.clamp(-clip, clip)

                vtrace_advantages = normalize_advantage(vtrace_advantages)
                upgo_advantages = normalize_advantage(upgo_advantages)

            vtrace_pg_loss = compute_policy_gradient_loss(
                combined_learner_action_log_probs,
                vtrace_advantages,
                reduction=flags.reduction,
                trajectory_weights=action_counts if trajectory_normalize else None,
                action_counts=action_counts,
            )
            upgo_clipped_importance = torch.minimum(
                vtrace_returns.log_rhos.exp(), torch.ones_like(vtrace_returns.log_rhos)
            ).detach()
            upgo_pg_loss = compute_policy_gradient_loss(
                combined_learner_action_log_probs,
                upgo_clipped_importance * upgo_advantages,
                reduction=flags.reduction,
                trajectory_weights=action_counts if trajectory_normalize else None,
                action_counts=action_counts,
            )
            if getattr(flags, "value_critic", "scalar") == "categorical_hl_gauss":
                baseline_loss = compute_categorical_baseline_loss(
                    learner_outputs["baseline_logits"], td_lambda_returns.vs,
                    reduction=flags.reduction, player_mask=learner_player_mask_batch,
                    trajectory_normalize=trajectory_normalize,
                    value_min=float(getattr(flags, "value_support_min", -2.0)),
                    value_max=float(getattr(flags, "value_support_max", 2.0)),
                    sigma_ratio=float(getattr(flags, "value_hl_gauss_sigma_ratio", 0.75)),
                )
            else:
                baseline_loss = compute_baseline_loss(
                    values, td_lambda_returns.vs, reduction=flags.reduction,
                    player_mask=learner_player_mask_batch, trajectory_normalize=trajectory_normalize,
                )
            teacher_kl_cost = teacher_kl_coefficient(flags, learner_step)
            reduced_teacher_kl = (
                trajectory_weighted_mean(combined_teacher_kl_loss, action_counts)
                if trajectory_normalize
                else reduce(combined_teacher_kl_loss, reduction=flags.reduction)
            )
            teacher_kl_loss = teacher_kl_cost * reduced_teacher_kl
            if flags.use_teacher:
                teacher_baseline_loss = flags.teacher_baseline_cost * compute_baseline_loss(
                    values,
                    teacher_outputs["baseline"],
                    reduction=flags.reduction,
                    player_mask=learner_player_mask_batch,
                    trajectory_normalize=trajectory_normalize,
                )
            else:
                teacher_baseline_loss = torch.zeros_like(baseline_loss)
            reduced_entropy = (
                trajectory_weighted_mean(combined_learner_entropy, action_counts)
                if trajectory_normalize
                else reduce(combined_learner_entropy, reduction=flags.reduction)
            )
            entropy_loss = flags.entropy_cost * reduced_entropy
            reduced_reverse_behavior_kl = trajectory_weighted_mean(
                combined_reverse_behavior_kl, behavior_kl_counts
            )
            reduced_forward_behavior_kl = trajectory_weighted_mean(
                combined_forward_behavior_kl, behavior_kl_counts
            )
            behavior_kl_loss = torch.zeros_like(baseline_loss)
            overall_normalized_entropy = (
                sum(normalized_entropies[key] * active_entity_counts[key] for key in normalized_entropies)
                / max(sum(active_entity_counts.values()), 1)
            )
            if behavior_kl_controller is not None:
                behavior_kl_controller.observe(
                    learner_step,
                    float(reduced_reverse_behavior_kl.detach().cpu().item()),
                    overall_normalized_entropy,
                )
                behavior_kl_loss = behavior_kl_controller.beta * reduced_reverse_behavior_kl

            rule_aux_loss = torch.zeros_like(baseline_loss)
            if getattr(flags, "rule_aux_enabled", False):
                rule_losses = torch.zeros_like(combined_learner_action_log_probs)
                rule_counts = torch.zeros_like(action_counts)
                for act_space, logits in learner_outputs["policy_logits"].items():
                    targets = batch["rule_actions"][act_space]
                    mask = batch["rule_confidence"][act_space]
                    ce = F.cross_entropy(
                        logits.flatten(0, -2), targets.flatten(), reduction="none"
                    ).view_as(targets)
                    masked_ce = torch.where(mask, ce, torch.zeros_like(ce))
                    rule_losses += masked_ce.sum(dim=(2, 4, 5))
                    rule_counts += mask.sum(dim=(2, 4, 5))
                rule_aux_loss = float(flags.rule_aux_cost) * trajectory_weighted_mean(rule_losses, rule_counts)

            intent_aux_loss = torch.zeros_like(baseline_loss)
            intent_accuracy = float("nan")
            intent_target_counts = torch.zeros(4, dtype=torch.long, device=values.device)
            if getattr(flags, "intent_aux_enabled", False):
                targets = batch["worker_intent"]
                mask = batch["worker_intent_mask"]
                logits = learner_outputs["intent_logits"]
                ce = F.cross_entropy(logits.flatten(0, -2), targets.flatten(), reduction="none").view_as(targets)
                intent_losses = torch.where(mask, ce, torch.zeros_like(ce)).sum(dim=(2, 4, 5))
                intent_counts = mask.sum(dim=(2, 4, 5))
                intent_aux_loss = float(flags.intent_aux_cost) * trajectory_weighted_mean(
                    intent_losses, intent_counts
                )
                if bool(mask.any()):
                    predictions = logits.argmax(dim=-1)
                    targets_for_logits = targets.view_as(predictions)
                    mask_for_logits = mask.view_as(predictions)
                    intent_accuracy = (
                        (predictions[mask_for_logits] == targets_for_logits[mask_for_logits]).float().mean().item()
                    )
                    intent_target_counts = torch.bincount(targets_for_logits[mask_for_logits], minlength=4)
            if baseline_only:
                total_loss = baseline_loss + teacher_baseline_loss
                vtrace_pg_loss, upgo_pg_loss, teacher_kl_loss, entropy_loss = torch.zeros(4) + float("nan")
                rule_aux_loss = intent_aux_loss = torch.zeros_like(baseline_loss)
            else:
                total_loss = (
                    vtrace_pg_loss
                    + upgo_pg_loss
                    + baseline_loss
                    + teacher_kl_loss
                    + teacher_baseline_loss
                    + entropy_loss
                    + behavior_kl_loss
                    + rule_aux_loss
                    + intent_aux_loss
                )
            if getattr(flags, "intent_head_only_finetune", False):
                total_loss = intent_aux_loss

            last_lr = lr_scheduler.get_last_lr()
            assert len(last_lr) == 1, "Logging per-parameter LR still needs support"
            last_lr = last_lr[0]
            action_distributions_flat = {
                key[16:]: val[batch["done"]][~val[batch["done"]].isnan()].sum().item()
                for key, val in batch["info"].items()
                if key.startswith("LOGGING_") and "ACTIONS_" in key
            }
            action_distributions = {space: {} for space in ACTION_MEANINGS.keys()}
            for flat_name, n in action_distributions_flat.items():
                space, meaning = flat_name.split(".")
                action_distributions[space][meaning] = n
            action_distributions_aggregated = {}
            for space, dist in action_distributions.items():
                if space == "city_tile":
                    action_distributions_aggregated[space] = dist
                elif space in ("cart", "worker"):
                    aggregated = {a: n for a, n in dist.items() if "TRANSFER" not in a and "MOVE" not in a}
                    aggregated["TRANSFER"] = sum({a: n for a, n in dist.items() if "TRANSFER" in a}.values())
                    aggregated["MOVE"] = sum({a: n for a, n in dist.items() if "MOVE" in a}.values())
                    action_distributions_aggregated[space] = aggregated
                else:
                    raise RuntimeError(f"Unrecognized action_space: {space}")
                n_actions = sum(action_distributions_aggregated[space].values())
                if n_actions == 0:
                    action_distributions_aggregated[space] = {
                        key: float("nan") for key in action_distributions_aggregated[space].keys()
                    }
                else:
                    action_distributions_aggregated[space] = {
                        key: val / n_actions for key, val in action_distributions_aggregated[space].items()
                    }

            total_games_played += batch["done"].sum().item()
            
            _log_rhos = vtrace_returns.log_rhos.detach().float()
            _N_t = action_counts.detach().float()
            _N_t_clamp = _N_t.clamp_min(1)
            _geomean_log_rhos = _log_rhos / _N_t_clamp
            
            _N_t_flat = _N_t.flatten()
            _abs_log_rhos_flat = _log_rhos.abs().flatten()
            _N_t_mean = _N_t_flat.mean()
            _abs_log_rhos_mean = _abs_log_rhos_flat.mean()
            _corr_num = torch.sum((_N_t_flat - _N_t_mean) * (_abs_log_rhos_flat - _abs_log_rhos_mean))
            _corr_den = torch.sqrt(torch.sum((_N_t_flat - _N_t_mean)**2) * torch.sum((_abs_log_rhos_flat - _abs_log_rhos_mean)**2)) + 1e-8
            _corr_N_abs_log_rho = float((_corr_num / _corr_den).cpu().item())
            
            # --- Reward Analysis ---
            _done_mask = batch["done"].unsqueeze(-1).expand_as(batch["reward"])
            _terminal_rewards = torch.where(
                _done_mask,
                torch.round(batch["reward"]),
                torch.zeros_like(batch["reward"])
            )
            _shaping_rewards = batch["reward"] - _terminal_rewards
            
            _num_dones = max(1, int(batch["done"].sum().item()))
            _shaping_sum_all_abs = float(_shaping_rewards.abs().sum().item())
            _shaping_episode_sum = _shaping_sum_all_abs / (_num_dones * 2)
            _terminal_abs_mean = float(_terminal_rewards[_done_mask].abs().mean().item()) if batch["done"].any() else 0.0
            _shaping_to_terminal_ratio = abs(_shaping_episode_sum) / 1.0

            _shaping_active = _shaping_rewards[_shaping_rewards != 0]
            _potential_mean = float(_shaping_active.mean().item() / 0.05) if _shaping_active.numel() > 0 else 0.0
            _potential_std = float(_shaping_active.std().item() / 0.05) if _shaping_active.numel() > 1 else 0.0
            _potential_delta_p95 = float(torch.quantile(_shaping_active.abs(), 0.95).item() / 0.05) if _shaping_active.numel() > 0 else 0.0
            
            # --- Advantage Analysis ---
            _active_adv = vtrace_advantages[action_counts > 0]
            _adv_std = float(_active_adv.std().item()) if _active_adv.numel() > 1 else 0.0
            _adv_p01 = float(torch.quantile(_active_adv, 0.01).item()) if _active_adv.numel() > 0 else 0.0
            _adv_p05 = float(torch.quantile(_active_adv, 0.05).item()) if _active_adv.numel() > 0 else 0.0
            _adv_p50 = float(torch.quantile(_active_adv, 0.50).item()) if _active_adv.numel() > 0 else 0.0
            _adv_p95 = float(torch.quantile(_active_adv.abs(), 0.95).item()) if _active_adv.numel() > 0 else 0.0
            _adv_p99 = float(torch.quantile(_active_adv.abs(), 0.99).item()) if _active_adv.numel() > 0 else 0.0
            _adv_max_abs = float(_active_adv.abs().max().item()) if _active_adv.numel() > 0 else 0.0
            _adv_clipped_fraction = float((_active_adv.abs() >= 4.9).float().mean().item()) if _active_adv.numel() > 0 else 0.0

            # --- Policy Gradient ---
            _N_t_active = action_counts[action_counts > 0]
            _effective_action_count = float(_N_t_active.mean().item()) if _N_t_active.numel() > 0 else 0.0
            _abs_adv_times_N = float((_active_adv.abs() * _N_t_active).mean().item()) if _N_t_active.numel() > 0 else 0.0

            # --- Advantage Analysis Given Action ---
            T, B, _ = vtrace_advantages.shape
            _adv_expanded = vtrace_advantages.view(T, B, 1, 2, 1, 1)
            for act_space in batch["actions"].keys():
                _actions = batch["actions"][act_space][..., 0]
                _mask = batch["info"]["actions_taken"][act_space].any(dim=-1)
                _legal_mask = batch["info"]["available_actions_mask"][act_space]
                
                _can_act_mask = _mask & _legal_mask[..., 1:].any(dim=-1)
                total_taken = _mask.sum().float().item()
                total_can_act = _can_act_mask.sum().float().item()

                if total_taken > 0:
                    for action_idx, action_name in enumerate(ACTION_MEANINGS[act_space]):
                        _this_action_mask = _mask & (_actions == action_idx)
                        _count = _this_action_mask.sum().float().item()
                        _rate = _count / total_taken
                        
                        _this_legal_mask = _mask & _legal_mask[..., action_idx]
                        _legal_count = _this_legal_mask.sum().float().item()
                        _rate_legal = _count / _legal_count if _legal_count > 0 else 0.0
                        
                        _this_action_can_act_mask = _can_act_mask & (_actions == action_idx)
                        _count_can_act = _this_action_can_act_mask.sum().float().item()
                        _rate_can_act = _count_can_act / total_can_act if total_can_act > 0 else 0.0
                        
                        stats.setdefault("ActionRate", {})[f"{act_space}_{action_name}"] = stats.get("ActionRate", {}).get(f"{act_space}_{action_name}", 0.0) + _rate
                        
                        if _legal_count > 0:
                            stats.setdefault("ActionRateLegal", {})[f"{act_space}_{action_name}"] = stats.get("ActionRateLegal", {}).get(f"{act_space}_{action_name}", 0.0) + _rate_legal
                            
                        if total_can_act > 0:
                            stats.setdefault("ActionRateCanAct", {})[f"{act_space}_{action_name}"] = stats.get("ActionRateCanAct", {}).get(f"{act_space}_{action_name}", 0.0) + _rate_can_act
                        
                        if _count > 0:
                            _adv_for_action = _adv_expanded.expand_as(_this_action_mask)[_this_action_mask]
                            _mean_adv = float(_adv_for_action.mean().item())
                            
                            _current_mean = stats.setdefault("AdvantageGivenAction", {}).get(f"{act_space}_{action_name}", 0.0)
                            _current_weight = stats.setdefault("_AdvantageGivenActionWeight", {}).get(f"{act_space}_{action_name}", 0.0)
                            
                            _new_weight = _current_weight + _count
                            _new_mean = (_current_mean * _current_weight + _mean_adv * _count) / _new_weight
                            
                            stats["AdvantageGivenAction"][f"{act_space}_{action_name}"] = _new_mean
                            stats["_AdvantageGivenActionWeight"][f"{act_space}_{action_name}"] = _new_weight
            
            if "_AdvantageGivenActionWeight" in stats:
                del stats["_AdvantageGivenActionWeight"]

            stats.update({
                "Reward": {
                    "shaping_episode_sum": _shaping_episode_sum,
                    "terminal": _terminal_abs_mean,
                    "shaping_to_terminal_ratio": _shaping_to_terminal_ratio,
                    "potential_mean": _potential_mean,
                    "potential_std": _potential_std,
                    "potential_delta_p95": _potential_delta_p95,
                },
                "Advantage": {
                    "std": _adv_std,
                    "p01": _adv_p01,
                    "p05": _adv_p05,
                    "p50": _adv_p50,
                    "p95": _adv_p95,
                    "p99": _adv_p99,
                    "max_abs": _adv_max_abs,
                    "clipped_fraction": _adv_clipped_fraction,
                },
                "PolicyGradient": {
                    "effective_action_count": _effective_action_count,
                    "abs_advantage_times_decision_count": _abs_adv_times_N,
                },
                "Env": {
                    key[8:]: val[batch["done"]][~val[batch["done"]].isnan()].mean().item()
                    for key, val in batch["info"].items()
                    if key.startswith("LOGGING_") and "ACTIONS_" not in key
                },
                "Actions": action_distributions_aggregated,
                "Loss": {
                    "vtrace_pg_loss": vtrace_pg_loss.detach().item(),
                    "upgo_pg_loss": upgo_pg_loss.detach().item(),
                    "baseline_loss": baseline_loss.detach().item(),
                    "teacher_kl_loss": teacher_kl_loss.detach().item(),
                    "teacher_baseline_loss": teacher_baseline_loss.detach().item(),
                    "entropy_loss": entropy_loss.detach().item(),
                    "behavior_kl_loss": behavior_kl_loss.detach().item(),
                    "rule_aux_loss": rule_aux_loss.detach().item(),
                    "intent_aux_loss": intent_aux_loss.detach().item(),
                    "total_loss": total_loss.detach().item(),
                },
                "Entropy": {"overall": sum(e for e in entropies.values() if not math.isnan(e)), **entropies},
                "Normalized_Entropy": {"overall": overall_normalized_entropy, **normalized_entropies},
                "Active_Entities": active_entity_counts,
                "Behavior_Policy": {
                    "forward_kl": float(reduced_forward_behavior_kl.detach().cpu().item()),
                    "reverse_kl": float(reduced_reverse_behavior_kl.detach().cpu().item()),
                    "beta": behavior_kl_controller.beta if behavior_kl_controller is not None else 0.0,
                    "target": behavior_kl_controller.target if behavior_kl_controller is not None else None,
                    "ema_kl": behavior_kl_controller.ema_kl if behavior_kl_controller is not None else None,
                    "log_rho_mean": float(_log_rhos.mean().cpu().item()),
                    "log_rho_p95": float(torch.quantile(_log_rhos, 0.95).cpu().item()),
                    "log_rho_max": float(_log_rhos.max().cpu().item()),
                    "vtrace_rho_clipped_fraction": float((_log_rhos > 0).float().mean().cpu().item()),
                    "decision_count_mean": float(_N_t.mean().cpu().item()),
                    "decision_count_p95": float(torch.quantile(_N_t, 0.95).cpu().item()),
                    "abs_log_rho_per_action": float(_geomean_log_rhos.abs().mean().cpu().item()),
                    "geomean_log_rho_mean": float(_geomean_log_rhos.mean().cpu().item()),
                    "geomean_log_rho_p05": float(torch.quantile(_geomean_log_rhos, 0.05).cpu().item()),
                    "geomean_log_rho_p50": float(torch.quantile(_geomean_log_rhos, 0.50).cpu().item()),
                    "geomean_log_rho_p95": float(torch.quantile(_geomean_log_rhos, 0.95).cpu().item()),
                    "delta_logp_p01": float(torch.quantile(_log_rhos, 0.01).cpu().item()),
                    "delta_logp_p05": float(torch.quantile(_log_rhos, 0.05).cpu().item()),
                    "delta_logp_p50": float(torch.quantile(_log_rhos, 0.50).cpu().item()),
                    "delta_logp_p95": float(torch.quantile(_log_rhos, 0.95).cpu().item()),
                    "corr_N_abs_log_rho": _corr_N_abs_log_rho,
                    "learner_version": int(learner_step),
                    "actor_version_min": int(batch["policy_version"].min().cpu().item()),
                    "actor_version_max": int(batch["policy_version"].max().cpu().item()),
                    "buffer_lag_max": int(learner_step - batch["policy_version"].min().cpu().item()),
                },
                "Value": {
                    "explained_variance": explained_variance(values, td_lambda_returns.vs),
                    "target_std": float(td_lambda_returns.vs.detach().float().std(unbiased=False).cpu().item()),
                    "bias": float((values.detach() - td_lambda_returns.vs.detach()).float().mean().cpu().item()),
                    "support_outside_fraction": float(support_outside_fraction(
                        td_lambda_returns.vs.detach(),
                        float(getattr(flags, "value_support_min", -2.0)),
                        float(getattr(flags, "value_support_max", 2.0)),
                    ).cpu().item()) if getattr(flags, "value_critic", "scalar") == "categorical_hl_gauss" else 0.0,
                },
                "Teacher_KL_Divergence": {
                    "overall": sum(tkld for tkld in teacher_kl_losses.values() if not math.isnan(tkld)),
                    **teacher_kl_losses,
                },
                "Intent": {
                    "accuracy": intent_accuracy,
                    "target_mine": int(intent_target_counts[0]),
                    "target_deliver": int(intent_target_counts[1]),
                    "target_build": int(intent_target_counts[2]),
                    "target_return": int(intent_target_counts[3]),
                },
                "Misc": {
                    "learning_rate": last_lr,
                    "teacher_kl_cost": teacher_kl_cost,
                    "actor_policy_tta_rot180": float(getattr(flags, "actor_policy_tta_rot180", False)),
                    "teacher_policy_tta_rot180": float(getattr(flags, "teacher_policy_tta_rot180", False)),
                    "total_games_played": total_games_played,
                    "league_opponents": {
                        opponent["name"]: (batch["info"]["league_opponent_id"] == opponent_index)
                        .float()
                        .mean()
                        .item()
                        for opponent_index, opponent in enumerate(flags.league_opponents)
                    },
                },
            })

            optimizer.zero_grad()
            if flags.use_mixed_precision:
                grad_scaler.scale(total_loss).backward()
                grad_scaler.unscale_(optimizer)
                if flags.clip_grads is not None:
                    gradient_norm = torch.nn.utils.clip_grad_norm_(learner_model.parameters(), flags.clip_grads)
                else:
                    gradient_norm = torch.sqrt(sum(
                        parameter.grad.detach().float().square().sum()
                        for parameter in learner_model.parameters() if parameter.grad is not None
                    ))
                grad_scaler.step(optimizer)
                grad_scaler.update()
            else:
                total_loss.backward()
                if flags.clip_grads is not None:
                    gradient_norm = torch.nn.utils.clip_grad_norm_(learner_model.parameters(), flags.clip_grads)
                else:
                    gradient_norm = torch.sqrt(sum(
                        parameter.grad.detach().float().square().sum()
                        for parameter in learner_model.parameters() if parameter.grad is not None
                    ))
                optimizer.step()
            stats["Gradient"] = {
                "norm_before_clip": float(gradient_norm.detach().cpu().item()),
                "clipped": float(gradient_norm.detach().cpu().item() > float(flags.clip_grads))
                if flags.clip_grads is not None else 0.0,
            }
            if lr_scheduler is not None:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)
                    lr_scheduler.step()

            if model_update_reference is not None:
                model_update_max_abs = state_dict_max_abs_diff(
                    model_state_dict_cpu(learner_model), model_update_reference
                )
                if model_update_max_abs == 0.0:
                    raise RuntimeError(
                        "Optimizer step did not change the learner model; refusing to continue a no-op run"
                    )
                stats["Misc"]["model_update_max_abs"] = model_update_max_abs

        # noinspection PyTypeChecker
        sync_actor_model(actor_model, learner_model, verify=verify_actor_sync)
        return stats, total_games_played


def train(flags):
    # Necessary for multithreading and multiprocessing
    os.environ["OMP_NUM_THREADS"] = "1"

    if flags.num_buffers < flags.num_actors:
        raise ValueError("num_buffers should >= num_actors")
    if flags.num_buffers < flags.batch_size // flags.n_actor_envs:
        raise ValueError("num_buffers should be larger than batch_size // n_actor_envs")

    t = flags.unroll_length
    b = flags.batch_size
    training_stop_step = min(
        int(flags.total_steps),
        int(flags.stop_after_step) if getattr(flags, "stop_after_step", None) is not None else int(flags.total_steps),
    )
    if training_stop_step <= 0:
        raise ValueError("stop_after_step must be positive")

    league_opponents = opponents_from_config(flags.league_opponents) if flags.league_enabled else (
        opponents_from_config([{"name": "selfplay", "kind": "selfplay", "weight": 1.0}])
    )
    legacy_opponents = [opponent for opponent in league_opponents if opponent.kind == "teacher"]

    if flags.use_teacher:
        teacher_flags = OmegaConf.load(Path(flags.teacher_load_dir) / "config.yaml")
        teacher_flags = flags_to_namespace(OmegaConf.to_container(teacher_flags))
    else:
        teacher_flags = None

    league_obs_flags = teacher_flags
    if legacy_opponents:
        first_legacy_config = OmegaConf.load(legacy_opponents[0].config)
        league_obs_flags = flags_to_namespace(OmegaConf.to_container(first_legacy_config))
        for opponent in legacy_opponents[1:]:
            other_config = OmegaConf.load(opponent.config)
            if (
                other_config.obs_space != first_legacy_config.obs_space
                or other_config.get("obs_space_kwargs", {}) != first_legacy_config.get("obs_space_kwargs", {})
            ):
                raise ValueError("All teacher-kind league opponents must use the same observation space")
        if teacher_flags is not None and teacher_flags.obs_space != league_obs_flags.obs_space:
            raise ValueError("Online KL teacher and league teacher must use the same observation space")

    if flags.load_dir:
        checkpoint_state = torch.load(Path(flags.load_dir) / flags.checkpoint_file, map_location=torch.device("cpu"))
    else:
        checkpoint_state = None
    restored_outcomes = checkpoint_state.get("league_outcomes", {}) if checkpoint_state is not None else {}
    restored_wins = [float(restored_outcomes.get(opponent.name, {}).get("wins", 0.0)) for opponent in league_opponents]
    restored_games_by_opponent = [
        int(restored_outcomes.get(opponent.name, {}).get("games", 0)) for opponent in league_opponents
    ]
    league_outcomes = (
        mp.Array("d", restored_wins, lock=False),
        mp.Array("q", restored_games_by_opponent, lock=False),
        mp.Lock(),
    )
    restored_games = (
        int(checkpoint_state.get("reward_games_completed", checkpoint_state.get("total_games_played", 0)))
        if checkpoint_state is not None
        else 0
    )
    reward_game_counter = mp.Value("q", restored_games)
    actor_policy_version = mp.Value("q", 0)

    example_env = create_env(flags, torch.device("cpu"), teacher_flags=league_obs_flags)
    example_output = example_env.reset(force=True)
    example_output["info"]["learner_player_mask"] = torch.ones((flags.n_actor_envs, 2), dtype=torch.bool)
    example_output["info"]["league_opponent_id"] = torch.zeros(flags.n_actor_envs, dtype=torch.int64)
    buffers = create_buffers(flags, example_env.unwrapped[0].obs_space, example_output["info"])
    example_env.close()
    del example_env

    actor_model = create_model(flags, flags.actor_device, teacher_model_flags=league_obs_flags, is_teacher_model=False)
    if checkpoint_state is not None:
        _load_model_state(
            actor_model,
            checkpoint_state["model_state_dict"],
            allow_new_intent_head=getattr(flags, "intent_aux_enabled", False),
        )
    configure_trainable_parameters(
        actor_model, intent_head_only=getattr(flags, "intent_head_only_finetune", False)
    )
    actor_model.eval()
    actor_model.share_memory()
    snapshot_lock = mp.Lock()
    snapshot_state_dicts = checkpoint_state.get("learner_snapshot_state_dicts", {}) if checkpoint_state else {}
    snapshot_slots = sorted({opponent.slot for opponent in league_opponents if opponent.kind == "learner_snapshot"})
    snapshot_dir = Path.cwd() / ".learner_snapshots"
    snapshot_paths = {slot: snapshot_dir / f"slot_{slot}.pt" for slot in snapshot_slots}
    snapshot_versions = mp.Array("q", [1 for _ in range(max(snapshot_slots, default=-1) + 1)], lock=False)
    if snapshot_slots:
        snapshot_dir.mkdir(parents=True, exist_ok=True)
    for slot in snapshot_slots:
        state_dict = snapshot_state_dicts.get(str(slot), actor_model.state_dict())
        atomic_torch_save({"model_state_dict": state_dict}, snapshot_paths[slot])
    n_trainable_params = sum(p.numel() for p in actor_model.parameters() if p.requires_grad)
    logging.info(f"Training model with {n_trainable_params:,d} parameters.")

    actor_processes = []
    free_queue = mp.SimpleQueue()
    full_queue = mp.SimpleQueue()

    for i in range(flags.num_actors):
        actor_start = threading.Thread if flags.debug else mp.Process
        actor = actor_start(
            target=act,
            args=(
                flags,
                league_obs_flags,
                i,
                free_queue,
                full_queue,
                actor_model,
                league_opponents,
                league_outcomes,
                snapshot_paths,
                snapshot_versions,
                snapshot_lock,
                reward_game_counter,
                actor_policy_version,
                buffers,
            ),
        )
        actor.start()
        actor_processes.append(actor)
        time.sleep(0.5)

    learner_model = create_model(
        flags, flags.learner_device, teacher_model_flags=league_obs_flags, is_teacher_model=False
    )
    if checkpoint_state is not None:
        _load_model_state(
            learner_model,
            checkpoint_state["model_state_dict"],
            allow_new_intent_head=getattr(flags, "intent_aux_enabled", False),
        )
    learner_model.train()
    intent_head_only = getattr(flags, "intent_head_only_finetune", False)
    trainable_parameters, trainable_parameter_names = configure_trainable_parameters(
        learner_model, intent_head_only=intent_head_only
    )
    if intent_head_only:
        # Keep spectral-normalisation buffers and every frozen module fixed.
        learner_model.eval()
        learner_model.intent_head.train()
    learner_model = learner_model.share_memory()
    if not flags.disable_wandb:
        wandb.watch(learner_model, flags.model_log_freq, log="all", log_graph=True)

    optimizer = flags.optimizer_class(trainable_parameters, **flags.optimizer_kwargs)
    if checkpoint_state is not None and not flags.weights_only:
        saved_trainable_names = checkpoint_state.get("trainable_parameter_names")
        if saved_trainable_names is None and intent_head_only:
            raise ValueError(
                "A legacy/full-model optimizer cannot resume into intent-head-only fine-tuning; use weights_only=true"
            )
        if saved_trainable_names is not None and list(saved_trainable_names) != trainable_parameter_names:
            raise ValueError("Checkpoint optimizer trainable parameters do not match the selected fine-tuning mode")
        optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])

    # Load teacher model for KL loss
    if flags.use_teacher:
        if flags.teacher_kl_cost <= 0.0 and flags.teacher_baseline_cost <= 0.0:
            raise ValueError(
                "It does not make sense to use teacher when teacher_kl_cost <= 0 and teacher_baseline_cost <= 0"
            )
        teacher_model = create_model(
            flags, flags.learner_device, teacher_model_flags=teacher_flags, is_teacher_model=True
        )
        teacher_model.load_state_dict(
            torch.load(Path(flags.teacher_load_dir) / flags.teacher_checkpoint_file, map_location=torch.device("cpu"))[
                "model_state_dict"
            ]
        )
        teacher_model.eval()
    else:
        teacher_model = None
        if flags.teacher_kl_cost > 0.0:
            logging.warning(
                f"flags.teacher_kl_cost is {flags.teacher_kl_cost}, but use_teacher is False. "
                f"Setting flags.teacher_kl_cost to 0."
            )
        if flags.teacher_baseline_cost > 0.0:
            logging.warning(
                f"flags.teacher_baseline_cost is {flags.teacher_baseline_cost}, but use_teacher is False. "
                f"Setting flags.teacher_baseline_cost to 0."
            )
        flags.teacher_kl_cost = 0.0
        flags.teacher_baseline_cost = 0.0

    def lr_lambda(epoch):
        min_pct = flags.min_lr_mod
        pct_complete = min(epoch * t * b, flags.total_steps) / flags.total_steps
        if getattr(flags, "lr_schedule", "linear") == "cosine":
            # cosine annealing: 1.0 -> min_pct
            return min_pct + (1.0 - min_pct) * 0.5 * (1.0 + math.cos(math.pi * pct_complete))
        # Default: linear decay (existing behavior)
        scaled_pct_complete = pct_complete * (1.0 - min_pct)
        return 1.0 - scaled_pct_complete

    grad_scaler = amp.GradScaler("cuda", enabled=flags.use_mixed_precision and flags.learner_device.type == "cuda")
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    if checkpoint_state is not None and not flags.weights_only:
        scheduler.load_state_dict(checkpoint_state["scheduler_state_dict"])

    step, total_games_played, stats = 0, 0, {}
    if checkpoint_state is not None and not flags.weights_only:
        if "step" in checkpoint_state.keys():
            step = checkpoint_state["step"]
        # Backwards compatibility
        else:
            logging.warning("Loading old checkpoint_state without 'step' saved. Starting at step 0.")
        if "total_games_played" in checkpoint_state.keys():
            total_games_played = checkpoint_state["total_games_played"]
        # Backwards compatibility
        else:
            logging.warning("Loading old checkpoint_state without 'total_games_played' saved. Starting at step 0.")
    training_start_step = step
    actor_policy_version.value = step
    behavior_kl_controller = BehaviorKLController.from_flags(
        flags,
        checkpoint_state.get("behavior_kl_controller")
        if checkpoint_state is not None and not flags.weights_only else None,
    )
    initial_learner_state = model_state_dict_cpu(learner_model)
    verify_model_updates = bool(getattr(flags, "verify_model_updates", True))
    learner_lock = threading.Lock()
    snapshot_last_step = int(checkpoint_state.get("learner_snapshot_last_step", step)) if checkpoint_state else step
    snapshot_next_slot = int(checkpoint_state.get("learner_snapshot_next_slot", 0)) if checkpoint_state else 0
    max_support_outside_fraction = float(
        checkpoint_state.get("max_support_outside_fraction", 0.0) if checkpoint_state else 0.0
    )

    def maybe_update_learner_snapshot(current_step: int) -> None:
        nonlocal snapshot_last_step, snapshot_next_slot
        if not snapshot_slots:
            return
        interval = max(int(getattr(flags, "learner_snapshot_interval_steps", 250000)), 1)
        if current_step - snapshot_last_step < interval:
            return
        slot = snapshot_slots[snapshot_next_slot % len(snapshot_slots)]
        with learner_lock, snapshot_lock:
            sync_actor_model(actor_model, learner_model, verify=verify_model_updates)
            atomic_torch_save({"model_state_dict": model_state_dict_cpu(learner_model)}, snapshot_paths[slot])
            snapshot_versions[slot] += 1
        snapshot_next_slot = (snapshot_next_slot + 1) % len(snapshot_slots)
        snapshot_last_step = current_step
        logging.info("Updated learner snapshot slot %d at step %d", slot, current_step)

    def batch_and_learn(learner_idx, lock=threading.Lock()):
        """Thread target for the learning process."""
        nonlocal step, total_games_played, stats, max_support_outside_fraction
        timings = prof.Timings()
        while step < training_stop_step:
            timings.reset()
            full_batch = get_batch(
                flags,
                free_queue,
                full_queue,
                buffers,
                timings,
            )
            if flags.batch_size < flags.n_actor_envs:
                batches = split_buffers(full_batch, flags.batch_size, dim=1, contiguous=True)
            else:
                batches = [full_batch]
            for batch in batches:
                stats, total_games_played = learn(
                    flags=flags,
                    actor_model=actor_model,
                    learner_model=learner_model,
                    teacher_model=teacher_model,
                    batch=batch,
                    optimizer=optimizer,
                    grad_scaler=grad_scaler,
                    lr_scheduler=scheduler,
                    total_games_played=total_games_played,
                    learner_step=step,
                    baseline_only=step / (t * b) < flags.n_value_warmup_batches,
                    behavior_kl_controller=behavior_kl_controller,
                    model_update_reference=(
                        initial_learner_state
                        if verify_model_updates and step == training_start_step
                        else None
                    ),
                    verify_actor_sync=verify_model_updates and step == training_start_step,
                    lock=learner_lock,
                )
                with lock:
                    max_support_outside_fraction = max(
                        max_support_outside_fraction,
                        float(stats.get("Value", {}).get("support_outside_fraction", 0.0)),
                    )
                    step += t * b
                    actor_policy_version.value = step
                    maybe_update_learner_snapshot(step)
                    if not flags.disable_wandb:
                        wandb.log(stats, step=step)
            timings.time("learn")
        if learner_idx == 0:
            logging.info(f"Batch and learn timing statistics: {timings.summary()}")

    for m in range(flags.num_buffers):
        free_queue.put(m)

    learner_threads = []
    for i in range(flags.num_learner_threads):
        thread = threading.Thread(target=batch_and_learn, name=f"batch-and-learn-{i}", args=(i,))
        thread.start()
        learner_threads.append(thread)

    def checkpoint(checkpoint_path: Union[str, Path]):
        logging.info(f"Saving checkpoint to {checkpoint_path}")
        with learner_lock:
            sync_actor_model(actor_model, learner_model, verify=verify_model_updates)
            learner_state = model_state_dict_cpu(learner_model)
            model_update_max_abs = state_dict_max_abs_diff(learner_state, initial_learner_state)
            if verify_model_updates and step > training_start_step and model_update_max_abs == 0.0:
                raise RuntimeError(
                    "Training steps advanced but the learner model is unchanged; refusing to save a no-op checkpoint"
                )
            logging.info("Checkpoint learner delta from run start: max_abs_diff=%.9g", model_update_max_abs)
            atomic_torch_save(
                {
                    "model_state_dict": learner_state,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "step": step,
                    "total_games_played": total_games_played,
                    "reward_games_completed": reward_game_counter.value,
                    "trainable_parameter_names": trainable_parameter_names,
                    "league_outcomes": {
                        opponent.name: {
                            "wins": float(league_outcomes[0][index]),
                            "games": int(league_outcomes[1][index]),
                        }
                        for index, opponent in enumerate(league_opponents)
                    },
                    "learner_snapshot_state_dicts": {
                        str(slot): torch.load(
                            snapshot_paths[slot], map_location=torch.device("cpu"), weights_only=False
                        )["model_state_dict"]
                        for slot in snapshot_slots
                    },
                    "learner_snapshot_last_step": snapshot_last_step,
                    "learner_snapshot_next_slot": snapshot_next_slot,
                    "model_update_max_abs": model_update_max_abs,
                    "max_support_outside_fraction": max_support_outside_fraction,
                    "evaluation_eligible": max_support_outside_fraction <= 0.001,
                    "behavior_kl_controller": (
                        behavior_kl_controller.state_dict() if behavior_kl_controller is not None else None
                    ),
                    "value_head_metadata": {
                        "type": getattr(flags, "value_critic", "scalar"),
                        "num_bins": int(getattr(flags, "value_num_bins", 101)),
                        "support": [
                            float(getattr(flags, "value_support_min", -2.0)),
                            float(getattr(flags, "value_support_max", 2.0)),
                        ],
                        "sigma_ratio": float(getattr(flags, "value_hl_gauss_sigma_ratio", 0.75)),
                    },
                },
                checkpoint_path + ".pt",
            )
            atomic_torch_save(
                {
                    "model_state_dict": learner_state,
                    "max_support_outside_fraction": max_support_outside_fraction,
                    "evaluation_eligible": max_support_outside_fraction <= 0.001,
                    "value_head_metadata": {
                        "type": getattr(flags, "value_critic", "scalar"),
                        "num_bins": int(getattr(flags, "value_num_bins", 101)),
                        "support": [
                            float(getattr(flags, "value_support_min", -2.0)),
                            float(getattr(flags, "value_support_max", 2.0)),
                        ],
                    },
                },
                checkpoint_path + "_weights.pt",
            )

    timer = timeit.default_timer
    training_failed = False
    try:
        last_checkpoint_time = timer()
        while step < training_stop_step:
            dead_actors = [index for index, actor in enumerate(actor_processes) if not actor.is_alive()]
            if dead_actors:
                raise RuntimeError(f"Rollout actors terminated before training completed: {dead_actors}")
            dead_learners = [index for index, thread in enumerate(learner_threads) if not thread.is_alive()]
            if dead_learners:
                raise RuntimeError(f"Learner threads terminated before training completed: {dead_learners}")
            start_step = step
            start_time = timer()
            time.sleep(5)

            # Save every checkpoint_freq minutes
            if timer() - last_checkpoint_time > flags.checkpoint_freq * 60:
                cp_path = str(step).zfill(int(math.log10(flags.total_steps)) + 1)
                checkpoint(cp_path)
                last_checkpoint_time = timer()

            sps = (step - start_step) / (timer() - start_time)
            bps = (step - start_step) / (t * b) / (timer() - start_time)
            logging.info(f"Steps {step:d} @ {sps:.1f} SPS / {bps:.1f} BPS. Stats:\n{pprint.pformat(stats)}")
    except KeyboardInterrupt:
        # Try checkpointing and joining actors then quit.
        return
    except BaseException:
        # Preserve the original training exception. A final actor sync would
        # otherwise be able to replace it with a secondary checkpoint error.
        training_failed = True
        raise
    else:
        for thread in learner_threads:
            thread.join()
        logging.info(f"Learning finished after {step:d} steps.")
    finally:
        for _ in range(flags.num_actors):
            free_queue.put(None)
        for actor in actor_processes:
            actor.join(timeout=10)
            if actor.is_alive():
                logging.warning("Terminating rollout actor that did not stop after the shutdown signal")
                actor.terminate()
                actor.join(timeout=10)
        if not training_failed:
            cp_path = str(step).zfill(int(math.log10(flags.total_steps)) + 1)
            checkpoint(cp_path)
