from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from tqdm.auto import tqdm

from ..lux_gym.act_spaces import ACTION_MEANINGS
from ..nns import create_model
from ..utils import flags_to_namespace
from .artifacts import atomic_torch_save, sha256_file, write_run_manifest
from .tta import rotate_policy_180, rot180_ensemble_outputs, rotate_compact_distillation_batch_180


class ShardDataset(Dataset):
    def __init__(self, dataset_dir: Path, split: str):
        manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("schema_version") not in (3, 4) or manifest.get("teacher_tta_rot180") is not True:
            raise ValueError(
                f"Dataset schema {manifest.get('schema_version')} is incompatible; regenerate Rot180-TTA schema v4"
            )
        self.dataset_dir = dataset_dir
        self.samples = []
        self._shard_cache = {}
        for shard in manifest["shards"]:
            if shard["split"] == split:
                source = str(shard.get("data_source", "legacy"))
                map_size = int(shard.get("map_size", -1))
                if map_size < 0:
                    with np.load(self.dataset_dir / shard["path"], allow_pickle=False) as archive:
                        mask = archive["input_mask"]
                        active = np.argwhere(mask.reshape(-1, *mask.shape[-2:]).any(axis=0))
                        map_size = int(max(active[:, 0].max() + 1, active[:, 1].max() + 1))
                self.samples.extend(
                    {
                        "path": shard["path"],
                        "turn": index,
                        "source": source,
                        "map_size": map_size,
                        "turn_band": min(index // 72, 4),
                        "is_night": index % 40 >= 30,
                    }
                    for index in range(shard["turn_count"])
                )

    def _load(self, name: str):
        if name not in self._shard_cache:
            if len(self._shard_cache) >= 2:
                self._shard_cache.pop(next(iter(self._shard_cache)))
            with np.load(self.dataset_dir / name, allow_pickle=False) as shard:
                if int(shard["schema_version"]) not in (3, 4) or not bool(shard["teacher_tta_rot180"]):
                    raise ValueError(f"Incompatible compact shard: {self.dataset_dir / name}")
                self._shard_cache[name] = {key: shard[key] for key in shard.files}
        return self._shard_cache[name]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        turn = sample["turn"]
        shard = self._load(sample["path"])
        result = {
            "obs": {},
            "input_mask": torch.from_numpy(shard["input_mask"]),
            "turn": turn,
            "sample_source": sample["source"],
            "map_size": sample["map_size"],
            "turn_band": sample["turn_band"],
            "is_night": sample["is_night"],
            "outcome": torch.from_numpy(shard["outcome"]).float() if "outcome" in shard else torch.zeros(2),
            "outcome_valid": bool(shard["outcome_valid"]) if "outcome_valid" in shard else False,
        }
        for key, value in shard.items():
            if not key.startswith("obs__"):
                continue
            tensor = torch.from_numpy(value[turn])
            result["obs"][key[5:]] = tensor.float() if tensor.is_floating_point() else tensor.long()
        for entity in ACTION_MEANINGS:
            offsets = shard[f"{entity}_offsets"]
            start, stop = int(offsets[turn]), int(offsets[turn + 1])
            result[f"{entity}_positions"] = torch.from_numpy(shard[f"{entity}_positions"][start:stop]).long()
            result[f"{entity}_legal_mask"] = torch.from_numpy(shard[f"{entity}_legal_mask"][start:stop])
            result[f"{entity}_teacher_logits"] = torch.from_numpy(shard[f"{entity}_teacher_logits"][start:stop])
        return result


def _compact_collate(samples):
    if not samples:
        raise ValueError("Cannot collate an empty compact distillation batch")
    result = {
        "obs": {key: torch.stack([sample["obs"][key] for sample in samples]) for key in samples[0]["obs"]},
        "input_mask": torch.stack([sample["input_mask"] for sample in samples]),
        "turn": torch.as_tensor([sample["turn"] for sample in samples]),
        "sample_source": [sample["sample_source"] for sample in samples],
        "map_size": torch.as_tensor([sample["map_size"] for sample in samples]),
        "turn_band": torch.as_tensor([sample["turn_band"] for sample in samples]),
        "is_night": torch.as_tensor([sample["is_night"] for sample in samples]),
        "outcome": torch.stack([sample["outcome"] for sample in samples]),
        "outcome_valid": torch.as_tensor([sample["outcome_valid"] for sample in samples]),
        "positions": {},
        "legal_mask": {},
        "teacher_logits": {},
        "available_actions_mask": {},
    }
    batch_size = len(samples)
    height, width = result["input_mask"].shape[-2:]
    for entity, action_names in ACTION_MEANINGS.items():
        positions = pad_sequence(
            [sample[f"{entity}_positions"] for sample in samples], batch_first=True, padding_value=-1
        )
        legal = pad_sequence(
            [sample[f"{entity}_legal_mask"] for sample in samples], batch_first=True, padding_value=False
        )
        teacher = pad_sequence(
            [sample[f"{entity}_teacher_logits"] for sample in samples], batch_first=True, padding_value=0.0
        )
        dense_legal = torch.ones((batch_size, 1, 2, height, width, len(action_names)), dtype=torch.bool)
        for batch_index in range(batch_size):
            valid = positions[batch_index, :, 0] >= 0
            if valid.any():
                player, x, y = positions[batch_index, valid].unbind(dim=-1)
                dense_legal[batch_index, 0, player, x, y] = legal[batch_index, valid]
        result["positions"][entity] = positions
        result["legal_mask"][entity] = legal
        result["teacher_logits"][entity] = teacher
        result["available_actions_mask"][entity] = dense_legal
    return result


def _select_entity_logits(dense_logits: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    batch_size, _, _, height, width, action_count = dense_logits.shape
    safe = positions.clamp_min(0)
    flat_positions = safe[..., 0] * height * width + safe[..., 1] * width + safe[..., 2]
    flattened = dense_logits[:, 0].reshape(batch_size, 2 * height * width, action_count)
    return torch.gather(flattened, 1, flat_positions.unsqueeze(-1).expand(-1, -1, action_count))


def distillation_loss(
    output,
    batch,
    temperature: float = 2.0,
    illegal_weight: float = 0.05,
    hard_label_weight: float = 0.0,
    teacher_margin_threshold: float = 0.5,
    outcome_weight: float = 0.0,
    rare_action_weight: float = 1.0,
):
    total_kl = torch.zeros((), device=batch["input_mask"].device)
    total_illegal = torch.zeros_like(total_kl)
    total_hard = torch.zeros_like(total_kl)
    confident_count = torch.zeros_like(total_kl)
    active_total = torch.zeros_like(total_kl)
    rare_count = torch.zeros_like(total_kl)
    agreements = []
    entity_metrics = {}
    for entity, student in output["policy_logits"].items():
        positions = batch["positions"][entity]
        student = _select_entity_logits(student, positions)
        teacher = batch["teacher_logits"][entity].to(student.dtype)
        legal = batch["legal_mask"][entity].bool()
        active = positions[..., 0] >= 0
        student_masked = student.masked_fill(~legal, -1e4)
        teacher_masked = teacher.masked_fill(~legal, -1e4)
        kl = (
            F.kl_div(
                F.log_softmax(student_masked / temperature, dim=-1),
                F.softmax(teacher_masked / temperature, dim=-1),
                reduction="none",
            ).sum(dim=-1)
            * temperature**2
        )
        active_count = active.sum()
        active_denominator = active_count.clamp_min(1)
        active_total = active_total + active_count
        top_values, top_indices = teacher_masked.topk(k=min(2, teacher_masked.shape[-1]), dim=-1)
        rare_indices = [
            index
            for index, action in enumerate(ACTION_MEANINGS[entity])
            if "NO-OP" not in action and "MOVE_" not in action
        ]
        rare = active & torch.isin(
            top_indices[..., 0],
            torch.as_tensor(rare_indices, device=top_indices.device),
        )
        decision_weights = torch.where(
            rare,
            torch.as_tensor(rare_action_weight, device=kl.device, dtype=kl.dtype),
            torch.ones((), device=kl.device, dtype=kl.dtype),
        )
        weighted_active = decision_weights * active
        denominator = weighted_active.sum().clamp_min(1)
        rare_count = rare_count + rare.sum()
        total_kl = total_kl + (kl * weighted_active).sum() / denominator
        illegal_mass = (F.softmax(student, dim=-1) * ~legal).sum(dim=-1)
        total_illegal = total_illegal + (illegal_mass * weighted_active).sum() / denominator
        if top_values.shape[-1] == 1:
            margin = torch.full_like(top_values[..., 0], float("inf"))
        else:
            margin = top_values[..., 0] - top_values[..., 1]
        confident = active & (margin >= teacher_margin_threshold)
        hard = F.cross_entropy(
            student_masked.transpose(1, 2),
            top_indices[..., 0],
            reduction="none",
        )
        confident_weights = decision_weights * confident
        total_hard = total_hard + (hard * confident_weights).sum() / confident_weights.sum().clamp_min(1)
        confident_count = confident_count + confident.sum()
        agreement = (
            ((student_masked.argmax(-1) == teacher_masked.argmax(-1)) & active).sum() / active_denominator
        ).float()
        entity_metrics[f"{entity}_kl"] = ((kl * active).sum() / active_denominator).detach()
        entity_metrics[f"{entity}_illegal_mass"] = ((illegal_mass * active).sum() / active_denominator).detach()
        entity_metrics[f"{entity}_agreement"] = agreement.detach()
        entity_metrics[f"{entity}_decisions"] = active_count.detach()
        if int(active_count.item()) > 0:
            agreements.append(agreement)
    outcome_loss = torch.zeros_like(total_kl)
    if outcome_weight > 0.0 and "outcome" in batch:
        valid = batch["outcome_valid"].bool()
        if valid.any():
            outcome_loss = F.mse_loss(output["baseline"][valid], batch["outcome"][valid].to(output["baseline"].dtype))
    return (
        total_kl + illegal_weight * total_illegal + hard_label_weight * total_hard + outcome_weight * outcome_loss
    ), {
        "kl": total_kl.detach(),
        "illegal_mass": total_illegal.detach(),
        "hard_label_loss": total_hard.detach(),
        "confident_fraction": (confident_count / active_total.clamp_min(1)).detach(),
        "rare_action_fraction": (rare_count / active_total.clamp_min(1)).detach(),
        "outcome_loss": outcome_loss.detach(),
        "teacher_agreement": (torch.stack(agreements).mean().detach() if agreements else torch.zeros_like(total_kl)),
        **entity_metrics,
    }


def rot180_consistency_loss(output, rotated_output, batch, temperature: float = 1.0) -> torch.Tensor:
    """Match both student views after mapping the rotated policy back to the original frame."""
    rotated_back = rotate_policy_180(rotated_output["policy_logits"])
    total = torch.zeros((), device=batch["input_mask"].device)
    for entity, original_dense in output["policy_logits"].items():
        positions = batch["positions"][entity]
        active = positions[..., 0] >= 0
        if not active.any():
            continue
        legal = batch["legal_mask"][entity].bool()
        original = _select_entity_logits(original_dense, positions).masked_fill(~legal, -1e4)
        rotated = _select_entity_logits(rotated_back[entity], positions).masked_fill(~legal, -1e4)
        original_log = F.log_softmax(original / temperature, dim=-1)
        rotated_log = F.log_softmax(rotated / temperature, dim=-1)
        original_prob = original_log.exp()
        rotated_prob = rotated_log.exp()
        symmetric = 0.5 * (
            (original_prob * (original_log - rotated_log)).sum(-1)
            + (rotated_prob * (rotated_log - original_log)).sum(-1)
        )
        total = total + (symmetric * active).sum() / active.sum().clamp_min(1)
    return total * temperature**2


def _load_flags(path: Path, device: str):
    values = yaml.safe_load(path.read_text(encoding="utf-8"))
    values["actor_device"] = device
    values["learner_device"] = device
    return flags_to_namespace(values), values


def parse_source_weights(values: list[str]) -> dict[str, float]:
    result = {}
    for value in values:
        name, separator, weight_text = value.partition("=")
        if not separator or not name:
            raise ValueError(f"--source-weight must be NAME=WEIGHT, got {value!r}")
        weight = float(weight_text)
        if weight <= 0.0:
            raise ValueError(f"source weight must be positive: {value!r}")
        result[name] = weight
    total = sum(result.values())
    return {name: weight / total for name, weight in result.items()}


def configure_distillation_trainable_parameters(model, scope: str) -> list[str]:
    if scope not in {"full", "worker_head"}:
        raise ValueError(f"Unknown distillation trainable scope: {scope}")
    trainable_names = []
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(scope == "full" or name.startswith("actor.actors.worker."))
        if parameter.requires_grad:
            trainable_names.append(name)
    if not trainable_names:
        raise ValueError(f"Distillation scope {scope!r} selected no parameters")
    return trainable_names


def set_distillation_train_mode(model, training: bool, scope: str) -> None:
    if not training or scope == "worker_head":
        # Keeping the frozen spectral-normalized actor base in eval mode prevents
        # weight_u/weight_v buffers from changing during worker-head-only updates.
        model.eval()
        if training and scope == "worker_head":
            model.actor.actors["worker"].train()
        return
    model.train()


def _accumulate_behavior_diagnostics(
    buckets,
    output,
    batch,
    temperature: float,
    reference_output=None,
) -> None:
    sources = batch["sample_source"]
    map_sizes = batch["map_size"].detach().cpu().tolist()
    for entity, dense_student in output["policy_logits"].items():
        positions = batch["positions"][entity]
        student = _select_entity_logits(dense_student, positions)
        teacher = batch["teacher_logits"][entity].to(student.dtype)
        legal = batch["legal_mask"][entity].bool()
        active = positions[..., 0] >= 0
        student_masked = student.masked_fill(~legal, -1e4)
        teacher_masked = teacher.masked_fill(~legal, -1e4)
        kl = (
            F.kl_div(
                F.log_softmax(student_masked / temperature, dim=-1),
                F.softmax(teacher_masked / temperature, dim=-1),
                reduction="none",
            ).sum(dim=-1)
            * temperature**2
        )
        student_actions = student_masked.argmax(-1)
        teacher_actions = teacher_masked.argmax(-1)
        reference_actions = None
        if reference_output is not None:
            reference = _select_entity_logits(reference_output["policy_logits"][entity], positions)
            reference_actions = reference.masked_fill(~legal, -1e4).argmax(-1)
        for batch_index, (source, map_size) in enumerate(zip(sources, map_sizes)):
            selected = active[batch_index]
            decisions = int(selected.sum().item())
            for map_key in (str(map_size), "all"):
                values = buckets[(source, map_key, entity)]
                values["decisions"] += decisions
                values["kl_sum"] += float(kl[batch_index][selected].sum().item())
                values["teacher_matches"] += int(
                    (student_actions[batch_index][selected] == teacher_actions[batch_index][selected]).sum().item()
                )
                if reference_actions is not None:
                    values["reference_disagreements"] += int(
                        (student_actions[batch_index][selected] != reference_actions[batch_index][selected])
                        .sum()
                        .item()
                    )


def finalize_behavior_diagnostics(buckets) -> dict:
    result = {}
    for (source, map_size, entity), values in sorted(buckets.items()):
        decisions = int(values["decisions"])
        entity_result = {
            "decisions": decisions,
            "kl": values["kl_sum"] / max(decisions, 1),
            "teacher_agreement": values["teacher_matches"] / max(decisions, 1),
        }
        if "reference_disagreements" in values:
            entity_result["reference_disagreement"] = values["reference_disagreements"] / max(decisions, 1)
        result.setdefault(source, {}).setdefault(map_size, {})[entity] = entity_result
    return result


def evaluate_behavior_gate(
    diagnostics: dict,
    source: str,
    worker_min: float | None,
    worker_max: float | None,
    require_city_exact: bool,
) -> dict:
    enabled = worker_min is not None or worker_max is not None or require_city_exact
    result = {"enabled": enabled, "source": source, "passed": True, "checks": {}}
    if not enabled:
        return result
    source_metrics = diagnostics.get(source, {}).get("all", {})
    worker = source_metrics.get("worker", {}).get("reference_disagreement")
    if worker_min is not None:
        passed = worker is not None and worker >= worker_min
        result["checks"]["worker_disagreement_min"] = {
            "value": worker,
            "threshold": worker_min,
            "passed": passed,
        }
        result["passed"] &= passed
    if worker_max is not None:
        passed = worker is not None and worker <= worker_max
        result["checks"]["worker_disagreement_max"] = {
            "value": worker,
            "threshold": worker_max,
            "passed": passed,
        }
        result["passed"] &= passed
    if require_city_exact:
        city = source_metrics.get("city_tile", {}).get("reference_disagreement")
        passed = city == 0.0
        result["checks"]["city_exact"] = {"value": city, "threshold": 0.0, "passed": passed}
        result["passed"] &= passed
    return result


def stratified_behavior_probe_indices(dataset: ShardDataset, samples_per_cell: int, seed: int) -> list[int]:
    if samples_per_cell <= 0:
        raise ValueError("behavior probe samples per cell must be positive")
    groups = defaultdict(list)
    for index, sample in enumerate(dataset.samples):
        groups[(sample["source"], sample["map_size"])].append(index)
    generator = random.Random(seed)
    selected = []
    for key in sorted(groups):
        candidates = groups[key]
        selected.extend(generator.sample(candidates, min(samples_per_cell, len(candidates))))
    return selected


def collect_behavior_diagnostics(
    model,
    reference_model,
    loader,
    device: torch.device,
    temperature: float,
    tta_rot180: bool,
) -> dict:
    model.eval()
    reference_model.eval()
    buckets = defaultdict(Counter)
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
            if tta_rot180:
                output = rot180_ensemble_outputs(model, model_input)
                reference_output = rot180_ensemble_outputs(reference_model, model_input)
            else:
                output = model(model_input, sample=False, actions_per_square=1)
                reference_output = reference_model(model_input, sample=False, actions_per_square=1)
            _accumulate_behavior_diagnostics(
                buckets,
                output,
                batch,
                temperature,
                reference_output,
            )
    return finalize_behavior_diagnostics(buckets)


def balanced_sample_weights(
    dataset: ShardDataset,
    source_weights: dict[str, float] | None = None,
    night_weight: float = 1.0,
) -> torch.Tensor:
    if night_weight <= 0.0:
        raise ValueError("night_weight must be positive")
    sources = sorted({sample["source"] for sample in dataset.samples})
    if source_weights:
        unknown = set(source_weights) - set(sources)
        missing = set(sources) - set(source_weights)
        if unknown or missing:
            raise ValueError(f"source weights mismatch: unknown={sorted(unknown)}, missing={sorted(missing)}")
        targets = source_weights
    else:
        targets = {source: 1.0 / len(sources) for source in sources}
    cells = Counter((sample["source"], sample["map_size"], sample["turn_band"]) for sample in dataset.samples)
    source_maps = {
        source: sorted({sample["map_size"] for sample in dataset.samples if sample["source"] == source})
        for source in sources
    }
    source_map_bands = {
        (source, map_size): sorted(
            {
                sample["turn_band"]
                for sample in dataset.samples
                if sample["source"] == source and sample["map_size"] == map_size
            }
        )
        for source in sources
        for map_size in source_maps[source]
    }
    weights = []
    for sample in dataset.samples:
        source = sample["source"]
        map_size = sample["map_size"]
        band = sample["turn_band"]
        cell_weight = (
            targets[source]
            / len(source_maps[source])
            / len(source_map_bands[(source, map_size)])
            / cells[(source, map_size, band)]
        )
        if sample["is_night"]:
            cell_weight *= night_weight
        weights.append(cell_weight)
    source_totals = Counter()
    for sample, weight in zip(dataset.samples, weights):
        source_totals[sample["source"]] += weight
    weights = [
        weight * targets[sample["source"]] / source_totals[sample["source"]]
        for sample, weight in zip(dataset.samples, weights)
    ]
    return torch.as_tensor(weights, dtype=torch.double)


def parse_args():
    parser = argparse.ArgumentParser(description="Train or fine-tune a student from offline teacher logits.")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("conf/survival_strategic.yaml"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--illegal-weight", type=float, default=0.05)
    parser.add_argument("--hard-label-weight", type=float, default=0.0)
    parser.add_argument("--teacher-margin-threshold", type=float, default=0.5)
    parser.add_argument("--rot180-consistency-weight", type=float, default=0.0)
    parser.add_argument("--outcome-weight", type=float, default=0.0)
    parser.add_argument("--rare-action-weight", type=float, default=1.0)
    parser.add_argument("--selection-metric", choices=("kl", "loss"), default="kl")
    parser.add_argument("--rot180-augmentation-prob", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--source-weight",
        action="append",
        default=[],
        metavar="NAME=WEIGHT",
        help="Target source mixture for balanced sampling; specify every dataset source.",
    )
    parser.add_argument("--night-weight", type=float, default=1.0)
    parser.add_argument("--samples-per-epoch", type=int)
    parser.add_argument(
        "--trainable-scope",
        choices=("full", "worker_head"),
        default="full",
        help="Restrict fine-tuning to the worker policy head while freezing the backbone and other heads.",
    )
    parser.add_argument(
        "--reference-weights",
        type=Path,
        help="Frozen reference checkpoint for validation action-disagreement metrics; defaults to --load-weights for worker_head.",
    )
    parser.add_argument("--checkpoint-every-samples", type=int, default=0)
    parser.add_argument("--behavior-gate-source", default="teacher_selfplay")
    parser.add_argument("--worker-disagreement-min", type=float)
    parser.add_argument("--worker-disagreement-max", type=float)
    parser.add_argument("--require-city-exact", action="store_true")
    parser.add_argument(
        "--behavior-probe-tta-rot180",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply the deployment Rot180 ensemble when computing reference-disagreement diagnostics.",
    )
    parser.add_argument(
        "--behavior-probe-samples-per-cell",
        type=int,
        default=24,
        help="Deterministic validation samples per source/map cell used for reference diagnostics.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--load-weights", type=Path, default=None, help="Path to checkpoint weights to start from")
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0.0 <= args.rot180_augmentation_prob <= 1.0:
        raise ValueError("--rot180-augmentation-prob must be in [0, 1]")
    for name in ("hard_label_weight", "teacher_margin_threshold", "rot180_consistency_weight", "outcome_weight"):
        if getattr(args, name) < 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative")
    if args.rare_action_weight < 1.0:
        raise ValueError("--rare-action-weight must be at least 1")
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.checkpoint_every_samples < 0:
        raise ValueError("--checkpoint-every-samples must be non-negative")
    if args.behavior_probe_samples_per_cell <= 0:
        raise ValueError("--behavior-probe-samples-per-cell must be positive")
    if args.worker_disagreement_min is not None and not 0.0 <= args.worker_disagreement_min <= 1.0:
        raise ValueError("--worker-disagreement-min must be in [0, 1]")
    if args.worker_disagreement_max is not None and not 0.0 <= args.worker_disagreement_max <= 1.0:
        raise ValueError("--worker-disagreement-max must be in [0, 1]")
    if (
        args.worker_disagreement_min is not None
        and args.worker_disagreement_max is not None
        and args.worker_disagreement_min > args.worker_disagreement_max
    ):
        raise ValueError("--worker-disagreement-min cannot exceed --worker-disagreement-max")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    flags, config_values = _load_flags(args.config, args.device)
    model = create_model(flags, device)
    if args.load_weights:
        print(f"Loading weights from {args.load_weights}")
        checkpoint = torch.load(args.load_weights, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
        initial_digest = hashlib_state_dict(model.state_dict())
        initial_checkpoint_sha256 = sha256_file(args.load_weights)
    else:
        # Deliberately no checkpoint load: the student always starts from this seeded random initialization.
        initial_digest = hashlib_state_dict(model.state_dict())
        initial_checkpoint_sha256 = None
    trainable_names = configure_distillation_trainable_parameters(model, args.trainable_scope)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable_parameters, lr=args.lr, weight_decay=args.weight_decay)
    reference_path = args.reference_weights
    if reference_path is None and args.trainable_scope == "worker_head":
        reference_path = args.load_weights
    gates_enabled = (
        args.worker_disagreement_min is not None or args.worker_disagreement_max is not None or args.require_city_exact
    )
    if gates_enabled and reference_path is None:
        raise ValueError("Behavior gates require --reference-weights or worker_head with --load-weights")
    reference_model = None
    reference_checkpoint_sha256 = None
    if reference_path is not None:
        reference_model = create_model(flags, device)
        reference_checkpoint = torch.load(reference_path, map_location=device, weights_only=True)
        reference_model.load_state_dict(reference_checkpoint.get("model_state_dict", reference_checkpoint))
        reference_model.eval()
        reference_model.requires_grad_(False)
        reference_checkpoint_sha256 = sha256_file(reference_path)
    train_data = ShardDataset(args.dataset_dir, "train")
    validation_data = ShardDataset(args.dataset_dir, "validation")
    if not train_data:
        raise ValueError("Dataset has no train samples")
    source_weights = parse_source_weights(args.source_weight)
    sample_weights = balanced_sample_weights(train_data, source_weights or None, args.night_weight)
    samples_per_epoch = args.samples_per_epoch or len(train_data)
    if samples_per_epoch <= 0:
        raise ValueError("--samples-per-epoch must be positive")
    sampler_generator = torch.Generator().manual_seed(args.seed)
    train_sampler = WeightedRandomSampler(
        sample_weights,
        num_samples=samples_per_epoch,
        replacement=True,
        generator=sampler_generator,
    )
    loaders = {
        "train": DataLoader(
            train_data,
            args.batch_size,
            sampler=train_sampler,
            num_workers=args.num_workers,
            collate_fn=_compact_collate,
        ),
        "validation": DataLoader(
            validation_data,
            args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=_compact_collate,
        )
        if validation_data
        else None,
    }
    behavior_probe_loader = None
    behavior_probe_sample_count = 0
    if reference_model is not None and validation_data:
        behavior_probe_indices = stratified_behavior_probe_indices(
            validation_data,
            args.behavior_probe_samples_per_cell,
            args.seed,
        )
        behavior_probe_sample_count = len(behavior_probe_indices)
        behavior_probe_loader = DataLoader(
            Subset(validation_data, behavior_probe_indices),
            args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=_compact_collate,
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.yaml").write_text(yaml.safe_dump(config_values, sort_keys=False), encoding="utf-8")
    training_args = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "temperature": args.temperature,
        "illegal_weight": args.illegal_weight,
        "hard_label_weight": args.hard_label_weight,
        "teacher_margin_threshold": args.teacher_margin_threshold,
        "rot180_consistency_weight": args.rot180_consistency_weight,
        "outcome_weight": args.outcome_weight,
        "rare_action_weight": args.rare_action_weight,
        "selection_metric": args.selection_metric,
        "source_weights": source_weights,
        "night_weight": args.night_weight,
        "samples_per_epoch": samples_per_epoch,
        "rot180_augmentation_prob": args.rot180_augmentation_prob,
        "trainable_scope": args.trainable_scope,
        "trainable_parameter_names": trainable_names,
        "trainable_parameter_count": sum(parameter.numel() for parameter in trainable_parameters),
        "checkpoint_every_samples": args.checkpoint_every_samples,
        "behavior_gate_source": args.behavior_gate_source,
        "worker_disagreement_min": args.worker_disagreement_min,
        "worker_disagreement_max": args.worker_disagreement_max,
        "require_city_exact": args.require_city_exact,
        "behavior_probe_tta_rot180": args.behavior_probe_tta_rot180,
        "behavior_probe_samples_per_cell": args.behavior_probe_samples_per_cell,
        "behavior_probe_sample_count": behavior_probe_sample_count,
        "reference_checkpoint": str(reference_path.resolve()) if reference_path is not None else None,
        "reference_checkpoint_sha256": reference_checkpoint_sha256,
    }
    best_metric = float("inf")
    best_gated_metric = float("inf")
    best_gated_written = False
    history = []
    checkpoint_records = []
    samples_seen = 0
    next_sample_checkpoint = args.checkpoint_every_samples or None
    for epoch in range(args.epochs):
        epoch_metrics = {}
        for split, loader in loaders.items():
            if loader is None:
                continue
            set_distillation_train_mode(model, split == "train", args.trainable_scope)
            sums = {
                "loss": 0.0,
                "kl": 0.0,
                "illegal_mass": 0.0,
                "teacher_agreement": 0.0,
                "hard_label_loss": 0.0,
                "confident_fraction": 0.0,
                "rare_action_fraction": 0.0,
                "outcome_loss": 0.0,
                "rot180_consistency": 0.0,
            }
            entity_sums = {
                entity: {"decisions": 0.0, "kl": 0.0, "illegal_mass": 0.0, "agreement": 0.0}
                for entity in ACTION_MEANINGS
            }
            source_samples = Counter()
            map_samples = Counter()
            night_samples = 0
            count = 0
            progress = tqdm(
                loader,
                desc=f"Epoch {epoch + 1}/{args.epochs} {split}",
                unit="batch",
                dynamic_ncols=True,
            )
            for batch in progress:
                source_samples.update(batch["sample_source"])
                map_samples.update(int(value) for value in batch["map_size"])
                night_samples += int(batch["is_night"].sum().item())
                batch = move_to(batch, device)
                if split == "train" and random.random() < args.rot180_augmentation_prob:
                    batch = rotate_compact_distillation_batch_180(batch)
                model_input = {
                    "obs": batch["obs"],
                    "info": {
                        "input_mask": batch["input_mask"],
                        "available_actions_mask": batch["available_actions_mask"],
                    },
                }
                output = model(
                    model_input,
                    sample=False,
                    actions_per_square=1,
                )
                loss, metrics = distillation_loss(
                    output,
                    batch,
                    args.temperature,
                    args.illegal_weight,
                    args.hard_label_weight,
                    args.teacher_margin_threshold,
                    args.outcome_weight,
                    args.rare_action_weight,
                )
                consistency = torch.zeros((), device=device)
                if args.rot180_consistency_weight > 0.0:
                    rotated_batch = rotate_compact_distillation_batch_180(batch)
                    rotated_output = model(
                        {
                            "obs": rotated_batch["obs"],
                            "info": {
                                "input_mask": rotated_batch["input_mask"],
                                "available_actions_mask": rotated_batch["available_actions_mask"],
                            },
                        },
                        sample=False,
                        actions_per_square=1,
                    )
                    consistency = rot180_consistency_loss(output, rotated_output, batch)
                    loss = loss + args.rot180_consistency_weight * consistency
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite distillation loss at epoch={epoch + 1} split={split}")
                if split == "train":
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(trainable_parameters, 5.0)
                    optimizer.step()
                    samples_seen += int(batch["turn"].shape[0])
                    if next_sample_checkpoint is not None and samples_seen >= next_sample_checkpoint:
                        checkpoint_name = f"sample_{samples_seen:08d}.pt"
                        atomic_torch_save(
                            {
                                "model_state_dict": model.state_dict(),
                                "epoch": epoch + 1,
                                "samples_seen": samples_seen,
                                "scratch_initialization": args.load_weights is None,
                                "initial_checkpoint_sha256": initial_checkpoint_sha256,
                                "initial_state_sha256": initial_digest,
                                "teacher_tta_rot180": True,
                                "rot180_augmentation_prob": args.rot180_augmentation_prob,
                                "config": config_values,
                                "training_args": training_args,
                            },
                            args.output_dir / checkpoint_name,
                        )
                        checkpoint_records.append(
                            {"checkpoint": checkpoint_name, "epoch": epoch + 1, "samples_seen": samples_seen}
                        )
                        while next_sample_checkpoint <= samples_seen:
                            next_sample_checkpoint += args.checkpoint_every_samples
                count += 1
                sums["loss"] += loss.detach().item()
                for name in (
                    "kl",
                    "illegal_mass",
                    "teacher_agreement",
                    "hard_label_loss",
                    "confident_fraction",
                    "rare_action_fraction",
                    "outcome_loss",
                ):
                    sums[name] += metrics[name].item()
                sums["rot180_consistency"] += consistency.detach().item()
                for entity in ACTION_MEANINGS:
                    decisions = metrics[f"{entity}_decisions"].item()
                    entity_sums[entity]["decisions"] += decisions
                    for name in ("kl", "illegal_mass", "agreement"):
                        entity_sums[entity][name] += metrics[f"{entity}_{name}"].item() * decisions
                progress.set_postfix(
                    loss=f"{sums['loss'] / count:.4f}",
                    kl=f"{sums['kl'] / count:.4f}",
                    agreement=f"{sums['teacher_agreement'] / count:.3f}",
                )
            result = {key: value / max(count, 1) for key, value in sums.items()}
            result["entities"] = {
                entity: {
                    "decisions": int(values["decisions"]),
                    **{
                        name: values[name] / max(values["decisions"], 1.0)
                        for name in ("kl", "illegal_mass", "agreement")
                    },
                }
                for entity, values in entity_sums.items()
            }
            result["sample_distribution"] = {
                "sources": dict(sorted(source_samples.items())),
                "map_sizes": {str(key): value for key, value in sorted(map_samples.items())},
                "night": night_samples,
            }
            epoch_metrics[split] = result
        if behavior_probe_loader is not None:
            validation_metrics = epoch_metrics["validation"]
            validation_metrics["behavior_diagnostics"] = collect_behavior_diagnostics(
                model,
                reference_model,
                behavior_probe_loader,
                device,
                args.temperature,
                args.behavior_probe_tta_rot180,
            )
            validation_metrics["behavior_probe_sample_count"] = behavior_probe_sample_count
            validation_metrics["behavior_gate"] = evaluate_behavior_gate(
                validation_metrics["behavior_diagnostics"],
                args.behavior_gate_source,
                args.worker_disagreement_min,
                args.worker_disagreement_max,
                args.require_city_exact,
            )
        history.append({"epoch": epoch + 1, **epoch_metrics})
        score = epoch_metrics.get("validation", epoch_metrics["train"])[args.selection_metric]
        policy_state = {
            "model_state_dict": model.state_dict(),
            "epoch": epoch + 1,
            "scratch_initialization": args.load_weights is None,
            "initial_checkpoint_sha256": initial_checkpoint_sha256,
            "initial_state_sha256": initial_digest,
            "teacher_tta_rot180": True,
            "rot180_augmentation_prob": args.rot180_augmentation_prob,
            "config": config_values,
            "training_args": training_args,
            "samples_seen": samples_seen,
            "metrics": epoch_metrics,
        }
        atomic_torch_save(
            {**policy_state, "optimizer_state_dict": optimizer.state_dict()}, args.output_dir / "latest.pt"
        )
        if score < best_metric:
            best_metric = score
            atomic_torch_save(policy_state, args.output_dir / "best.pt")
        gate = epoch_metrics.get("validation", {}).get("behavior_gate", {"passed": not gates_enabled})
        if gate["passed"] and score < best_gated_metric:
            best_gated_metric = score
            atomic_torch_save(policy_state, args.output_dir / "best_gated.pt")
            best_gated_written = True
        atomic_torch_save(policy_state, args.output_dir / f"epoch_{epoch + 1:03d}.pt")
        print(json.dumps(history[-1], sort_keys=True))
    dataset_manifest = json.loads((args.dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    best_sha = sha256_file(args.output_dir / "best.pt")
    best_gated_path = args.output_dir / "best_gated.pt"
    write_run_manifest(
        args.output_dir,
        {
            "kind": "scratch_distillation" if args.load_weights is None else "dagger_finetune_distillation",
            "checkpoint": "best.pt",
            "checkpoint_sha256": best_sha,
            "promotion_checkpoint": "best_gated.pt" if best_gated_written else None,
            "promotion_checkpoint_sha256": sha256_file(best_gated_path) if best_gated_written else None,
            "teacher_sha256": dataset_manifest["teacher_sha256"],
            "teacher_tta_rot180": True,
            "rot180_augmentation_prob": args.rot180_augmentation_prob,
            "scratch_initialization": args.load_weights is None,
            "initial_checkpoint_sha256": initial_checkpoint_sha256,
            "source_weights": source_weights,
            "night_weight": args.night_weight,
            "samples_per_epoch": samples_per_epoch,
            "training_args": training_args,
            "checkpoint_records": checkpoint_records,
            "samples_seen": samples_seen,
            "offline_gate_passed": best_gated_written if gates_enabled else None,
            "initial_state_sha256": initial_digest,
            "dataset_manifest": str((args.dataset_dir / "manifest.json").resolve()),
            "history": history,
        },
    )


def move_to(value, device):
    if isinstance(value, dict):
        return {key: move_to(item, device) for key, item in value.items()}
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    return value


def hashlib_state_dict(state_dict) -> str:
    import hashlib

    digest = hashlib.sha256()
    for name, tensor in sorted(state_dict.items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


if __name__ == "__main__":
    main()
