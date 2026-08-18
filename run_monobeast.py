from contextlib import redirect_stdout
import io

# Silence "Loading environment football failed: No module named 'gfootball'" message
with redirect_stdout(io.StringIO()):
    import kaggle_environments  # noqa: F401

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd
import logging
import os
from omegaconf import OmegaConf, DictConfig
from pathlib import Path
from torch import multiprocessing as mp
import wandb

from lux_ai.utils import flags_to_namespace
from lux_ai.torchbeast.monobeast import train
from lux_ai.strategic_rl.resume import merge_resume_config


os.environ["OMP_NUM_THREADS"] = "1"

logging.basicConfig(
    format=("[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] %(message)s"),
    level=0,
)


def get_default_flags(flags: DictConfig) -> DictConfig:
    flags = OmegaConf.to_container(flags)
    # Env params
    flags.setdefault("seed", None)
    flags.setdefault("num_buffers", max(2 * flags["num_actors"], flags["batch_size"] // flags["n_actor_envs"]))
    flags.setdefault("obs_space_kwargs", {})
    flags.setdefault("reward_space_kwargs", {})

    # Training params
    flags.setdefault("use_mixed_precision", True)
    # Keep rollout policy logits in FP32 unless actor AMP is explicitly opted in.
    # Small FP16 logit changes can alter sampled actions and therefore trajectories.
    flags.setdefault("actor_mixed_precision", False)
    flags.setdefault("discounting", 0.999)
    flags.setdefault("reduction", "mean")
    flags.setdefault("clip_grads", 10.0)
    flags.setdefault("checkpoint_freq", 10.0)
    flags.setdefault("num_learner_threads", 1)
    flags.setdefault("use_teacher", False)
    flags.setdefault("teacher_baseline_cost", flags.get("teacher_kl_cost", 0.0) / 2.0)
    flags.setdefault("teacher_kl_cost_floor", 0.0)
    flags.setdefault("actor_policy_tta_rot180", False)
    flags.setdefault("teacher_policy_tta_rot180", False)
    flags.setdefault("league_enabled", False)
    flags.setdefault("league_opponents", [])
    flags.setdefault("league_config_version", 0)
    flags.setdefault("reward_config_version", 0)
    flags.setdefault("objective_config_version", 0)
    flags.setdefault("league_sampling", "fixed")
    flags.setdefault("pfsp_power", 2.0)
    flags.setdefault("pfsp_teacher_floor", 0.15)
    flags.setdefault("pfsp_exploration", 0.02)
    flags.setdefault("pfsp_prior_games", 20)
    flags.setdefault("learner_snapshot_interval_steps", 250000)
    flags.setdefault("loss_normalization", "legacy")
    flags.setdefault("normalize_advantages", False)
    flags.setdefault("advantage_clip", 5.0)
    flags.setdefault("rule_aux_enabled", False)
    flags.setdefault("rule_aux_cost", 0.02)
    flags.setdefault("rule_aux_strategy", "economy")
    flags.setdefault("intent_aux_enabled", False)
    flags.setdefault("intent_aux_cost", 0.01)
    flags.setdefault("intent_head_only_finetune", False)
    flags.setdefault("stop_after_step", None)
    flags.setdefault("verify_model_updates", True)

    # Model params
    flags.setdefault("use_index_select", True)
    if flags.get("use_index_select"):
        logging.info("index_select disables padding_index and is equivalent to using a learnable pad embedding.")

    # Reloading previous run params
    flags.setdefault("load_dir", None)
    flags.setdefault("checkpoint_file", None)
    flags.setdefault("weights_only", False)
    flags.setdefault("n_value_warmup_batches", 0)

    # Miscellaneous params
    flags.setdefault("disable_wandb", False)
    flags.setdefault("debug", False)

    return OmegaConf.create(flags)


@hydra.main(config_path="conf", config_name="resume_config")
def main(flags: DictConfig):
    selected_flags = flags
    # Read only Hydra task overrides. OmegaConf.from_cli() also sees Hydra's
    # own arguments (for example ``--config-name``) and treats them as config
    # keys, which breaks structured-config merges when resuming a full run.
    cli_overrides = [override.lstrip("+") for override in HydraConfig.get().overrides.task]
    cli_conf = OmegaConf.from_dotlist(cli_overrides)
    if Path("config.yaml").exists():
        new_flags = OmegaConf.load("config.yaml")
        flags = merge_resume_config(new_flags, selected_flags, cli_conf)

    if flags.get("load_dir", None) and not flags.get("weights_only", False):
        # this ignores the local config.yaml and replaces it completely with saved one
        # however, you can override parameters from the cli still
        # this is useful e.g. if you did total_steps=N before and want to increase it
        logging.info("Loading existing configuration, we're continuing a previous run")
        new_flags = OmegaConf.load(Path(flags.load_dir) / "config.yaml")
        # Preserve every saved setting and apply only explicitly supplied task
        # overrides. Merging the selected base config here can silently reset
        flags = merge_resume_config(new_flags, selected_flags, cli_conf)
        logging.info(f"After merge_resume_config: total_steps={flags.get('total_steps')}, weights_only={flags.get('weights_only')}")

    flags = get_default_flags(flags)
    original_cwd = Path(get_original_cwd())
    if flags.get("teacher_load_dir"):
        teacher_path = Path(flags.teacher_load_dir)
        if not teacher_path.is_absolute():
            flags.teacher_load_dir = str(original_cwd / teacher_path)
    for opponent in flags.league_opponents:
        for key in ("config", "checkpoint"):
            if opponent.get(key):
                path = Path(opponent[key])
                if not path.is_absolute():
                    opponent[key] = str(original_cwd / path)
    logging.info(OmegaConf.to_yaml(flags, resolve=True))
    OmegaConf.save(flags, "config.yaml")
    if not flags.disable_wandb:
        wandb.init(
            config=OmegaConf.to_container(flags, resolve=True),
            project=flags.project,
            entity=flags.entity,
            group=flags.group,
            name=flags.name,
        )

    flags = flags_to_namespace(OmegaConf.to_container(flags))
    mp.set_sharing_strategy(flags.sharing_strategy)
    train(flags)


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
