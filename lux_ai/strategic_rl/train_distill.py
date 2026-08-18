from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from ..lux_gym.act_spaces import ACTION_MEANINGS
from ..nns import create_model
from ..utils import flags_to_namespace
from .artifacts import atomic_torch_save, sha256_file, write_run_manifest
from .tta import rotate_compact_distillation_batch_180


class ShardDataset(Dataset):
    def __init__(self, dataset_dir: Path, split: str):
        manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("schema_version") != 3 or manifest.get("teacher_tta_rot180") is not True:
            raise ValueError(
                f"Dataset schema {manifest.get('schema_version')} is incompatible; regenerate Rot180-TTA schema v3"
            )
        self.dataset_dir = dataset_dir
        self.samples = []
        self._shard_cache = {}
        for shard in manifest["shards"]:
            if shard["split"] == split:
                self.samples.extend((shard["path"], index) for index in range(shard["turn_count"]))

    def _load(self, name: str):
        if name not in self._shard_cache:
            if len(self._shard_cache) >= 2:
                self._shard_cache.pop(next(iter(self._shard_cache)))
            with np.load(self.dataset_dir / name, allow_pickle=False) as shard:
                if int(shard["schema_version"]) != 3 or not bool(shard["teacher_tta_rot180"]):
                    raise ValueError(f"Incompatible compact shard: {self.dataset_dir / name}")
                self._shard_cache[name] = {key: shard[key] for key in shard.files}
        return self._shard_cache[name]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        name, turn = self.samples[index]
        shard = self._load(name)
        result = {
            "obs": {},
            "input_mask": torch.from_numpy(shard["input_mask"]),
            "turn": turn,
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


def distillation_loss(output, batch, temperature: float = 2.0, illegal_weight: float = 0.05):
    total_kl = torch.zeros((), device=batch["input_mask"].device)
    total_illegal = torch.zeros_like(total_kl)
    agreements = []
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
        denominator = active.sum().clamp_min(1)
        total_kl = total_kl + (kl * active).sum() / denominator
        illegal_mass = (F.softmax(student, dim=-1) * ~legal).sum(dim=-1)
        total_illegal = total_illegal + (illegal_mass * active).sum() / denominator
        agreements.append(
            (((student_masked.argmax(-1) == teacher_masked.argmax(-1)) & active).sum() / denominator).float()
        )
    return total_kl + illegal_weight * total_illegal, {
        "kl": total_kl.detach(),
        "illegal_mass": total_illegal.detach(),
        "teacher_agreement": torch.stack(agreements).mean().detach(),
    }


def _load_flags(path: Path, device: str):
    values = yaml.safe_load(path.read_text(encoding="utf-8"))
    values["actor_device"] = device
    values["learner_device"] = device
    return flags_to_namespace(values), values


def parse_args():
    parser = argparse.ArgumentParser(description="Train a new student from teacher logits without copying weights.")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("conf/survival_strategic.yaml"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--illegal-weight", type=float, default=0.05)
    parser.add_argument("--rot180-augmentation-prob", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--load-weights", type=Path, default=None, help="Path to checkpoint weights to start from")
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0.0 <= args.rot180_augmentation_prob <= 1.0:
        raise ValueError("--rot180-augmentation-prob must be in [0, 1]")
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
    else:
        # Deliberately no checkpoint load: the student always starts from this seeded random initialization.
        initial_digest = hashlib_state_dict(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_data = ShardDataset(args.dataset_dir, "train")
    validation_data = ShardDataset(args.dataset_dir, "validation")
    if not train_data:
        raise ValueError("Dataset has no train samples")
    loaders = {
        "train": DataLoader(
            train_data, args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=_compact_collate
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
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.yaml").write_text(yaml.safe_dump(config_values, sort_keys=False), encoding="utf-8")
    best_metric = float("inf")
    history = []
    for epoch in range(args.epochs):
        epoch_metrics = {}
        for split, loader in loaders.items():
            if loader is None:
                continue
            model.train(split == "train")
            sums = {"loss": 0.0, "kl": 0.0, "illegal_mass": 0.0, "teacher_agreement": 0.0}
            count = 0
            progress = tqdm(
                loader,
                desc=f"Epoch {epoch + 1}/{args.epochs} {split}",
                unit="batch",
                dynamic_ncols=True,
            )
            for batch in progress:
                batch = move_to(batch, device)
                if split == "train" and random.random() < args.rot180_augmentation_prob:
                    batch = rotate_compact_distillation_batch_180(batch)
                output = model(
                    {
                        "obs": batch["obs"],
                        "info": {
                            "input_mask": batch["input_mask"],
                            "available_actions_mask": batch["available_actions_mask"],
                        },
                    }
                )
                loss, metrics = distillation_loss(output, batch, args.temperature, args.illegal_weight)
                if split == "train":
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
                count += 1
                sums["loss"] += loss.detach().item()
                for name, value in metrics.items():
                    sums[name] += value.item()
                progress.set_postfix(
                    loss=f"{sums['loss'] / count:.4f}",
                    kl=f"{sums['kl'] / count:.4f}",
                    agreement=f"{sums['teacher_agreement'] / count:.3f}",
                )
            epoch_metrics[split] = {key: value / max(count, 1) for key, value in sums.items()}
        history.append({"epoch": epoch + 1, **epoch_metrics})
        score = epoch_metrics.get("validation", epoch_metrics["train"])["kl"]
        policy_state = {
            "model_state_dict": model.state_dict(),
            "epoch": epoch + 1,
            "scratch_initialization": True,
            "initial_state_sha256": initial_digest,
            "teacher_tta_rot180": True,
            "rot180_augmentation_prob": args.rot180_augmentation_prob,
            "config": config_values,
            "metrics": epoch_metrics,
        }
        atomic_torch_save(
            {**policy_state, "optimizer_state_dict": optimizer.state_dict()}, args.output_dir / "latest.pt"
        )
        if score < best_metric:
            best_metric = score
            atomic_torch_save(policy_state, args.output_dir / "best.pt")
        print(json.dumps(history[-1], sort_keys=True))
    dataset_manifest = json.loads((args.dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    best_sha = sha256_file(args.output_dir / "best.pt")
    write_run_manifest(
        args.output_dir,
        {
            "kind": "scratch_distillation",
            "checkpoint": "best.pt",
            "checkpoint_sha256": best_sha,
            "teacher_sha256": dataset_manifest["teacher_sha256"],
            "teacher_tta_rot180": True,
            "rot180_augmentation_prob": args.rot180_augmentation_prob,
            "scratch_initialization": True,
            "initial_state_sha256": initial_digest,
            "dataset_manifest": str((args.dataset_dir / "manifest.json").resolve()),
            "history": history,
        },
    )


def move_to(value, device):
    if isinstance(value, dict):
        return {key: move_to(item, device) for key, item in value.items()}
    return value.to(device, non_blocking=True)


def hashlib_state_dict(state_dict) -> str:
    import hashlib

    digest = hashlib.sha256()
    for name, tensor in sorted(state_dict.items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


if __name__ == "__main__":
    main()
