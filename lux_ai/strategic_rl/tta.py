from __future__ import annotations

from collections.abc import Mapping

import torch

from ..lux_gym.act_spaces import ACTION_MEANINGS, ACTION_MEANINGS_TO_IDX

_ROT180_DIRECTIONS = {"n": "s", "e": "w", "s": "n", "w": "e"}


def _rot180_action_indices(entity: str) -> list[int]:
    indices = []
    meanings_to_idx = ACTION_MEANINGS_TO_IDX[entity]
    for action in ACTION_MEANINGS[entity]:
        direction = action.rsplit("_", 1)[-1]
        if direction in _ROT180_DIRECTIONS:
            rotated_action = f"{action[:-1]}{_ROT180_DIRECTIONS[direction]}"
            indices.append(meanings_to_idx[rotated_action])
        else:
            indices.append(meanings_to_idx[action])
    return indices


ROT180_ACTION_INDICES = {entity: _rot180_action_indices(entity) for entity in ACTION_MEANINGS}


def rotate_observations_180(observations: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Rotate only spatial observation planes; global features are unchanged."""
    return {
        key: torch.rot90(value, 2, dims=(-2, -1)) if value.ndim == 5 else value for key, value in observations.items()
    }


def rotate_policy_180(policy: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Rotate spatial axes and remap directional actions into the rotated frame."""
    return {
        entity: torch.rot90(values, 2, dims=(-3, -2))[..., ROT180_ACTION_INDICES[entity]]
        for entity, values in policy.items()
    }


def rotate_model_input_180(model_input: Mapping) -> dict:
    rotated = dict(model_input)
    rotated["obs"] = rotate_observations_180(model_input["obs"])
    info = dict(model_input["info"])
    info["input_mask"] = torch.rot90(model_input["info"]["input_mask"], 2, dims=(-2, -1))
    info["available_actions_mask"] = rotate_policy_180(model_input["info"]["available_actions_mask"])
    rotated["info"] = info
    return rotated


def rot180_ensemble_outputs(model, model_input: Mapping) -> dict[str, object]:
    """Average both views using one larger forward instead of two small forwards."""
    rotated_input = rotate_model_input_180(model_input)
    batched_input = {
        "obs": {
            key: torch.cat((value, rotated_input["obs"][key]), dim=0) for key, value in model_input["obs"].items()
        },
        "info": {
            "input_mask": torch.cat(
                (model_input["info"]["input_mask"], rotated_input["info"]["input_mask"]), dim=0
            ),
            "available_actions_mask": {
                key: torch.cat((value, rotated_input["info"]["available_actions_mask"][key]), dim=0)
                for key, value in model_input["info"]["available_actions_mask"].items()
            },
        },
    }
    if "subtask_embeddings" in model_input["info"]:
        batched_input["info"]["subtask_embeddings"] = torch.cat(
            (model_input["info"]["subtask_embeddings"], rotated_input["info"]["subtask_embeddings"]), dim=0
        )
    batch_size = model_input["info"]["input_mask"].shape[0]
    combined = model(batched_input, sample=False, actions_per_square=1)
    original = {
        "policy_logits": {key: value[:batch_size] for key, value in combined["policy_logits"].items()},
        "baseline": combined["baseline"][:batch_size],
    }
    rotated = {
        "policy_logits": {key: value[batch_size:] for key, value in combined["policy_logits"].items()},
        "baseline": combined["baseline"][batch_size:],
    }
    if "intent_logits" in combined:
        original["intent_logits"] = combined["intent_logits"][:batch_size]
        rotated["intent_logits"] = combined["intent_logits"][batch_size:]
    rotated_policy = rotate_policy_180(rotated["policy_logits"])
    output = {
        "policy_logits": {
            entity: (original["policy_logits"][entity] + rotated_policy[entity]) / 2
            for entity in original["policy_logits"]
        },
        "baseline": (original["baseline"] + rotated["baseline"]) / 2,
    }
    if "intent_logits" in original:
        rotated_intent = torch.rot90(rotated["intent_logits"], 2, dims=(-3, -2))
        output["intent_logits"] = (original["intent_logits"] + rotated_intent) / 2
    return output


def rotate_compact_distillation_batch_180(batch: Mapping) -> dict:
    """Rotate a compact distillation batch and its ragged policy targets."""
    rotated = dict(batch)
    rotated["obs"] = rotate_observations_180(batch["obs"])
    rotated["input_mask"] = torch.rot90(batch["input_mask"], 2, dims=(-2, -1))
    height, width = batch["input_mask"].shape[-2:]
    rotated["positions"] = {}
    rotated["legal_mask"] = {}
    rotated["teacher_logits"] = {}
    for entity in ACTION_MEANINGS:
        positions = batch["positions"][entity].clone()
        valid = positions[..., 0] >= 0
        positions[..., 1] = torch.where(valid, height - 1 - positions[..., 1], positions[..., 1])
        positions[..., 2] = torch.where(valid, width - 1 - positions[..., 2], positions[..., 2])
        indices = ROT180_ACTION_INDICES[entity]
        rotated["positions"][entity] = positions
        rotated["legal_mask"][entity] = batch["legal_mask"][entity][..., indices]
        rotated["teacher_logits"][entity] = batch["teacher_logits"][entity][..., indices]
    rotated["available_actions_mask"] = rotate_policy_180(batch["available_actions_mask"])
    return rotated
