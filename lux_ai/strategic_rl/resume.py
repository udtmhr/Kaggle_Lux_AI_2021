from omegaconf import DictConfig, OmegaConf


def merge_resume_config(saved_flags: DictConfig, selected_flags: DictConfig, cli_conf: DictConfig) -> DictConfig:
    """Migrate newly introduced league fields without resetting saved run settings."""
    migrations = {}
    for key in ("league_enabled", "league_opponents", "league_config_version", "reward_config_version"):
        if key not in saved_flags and key in selected_flags:
            migrations[key] = selected_flags[key]
    saved_version = int(saved_flags.get("league_config_version", 0))
    selected_version = int(selected_flags.get("league_config_version", 0))
    if selected_version > saved_version:
        league_keys = (
            "league_enabled",
            "league_opponents",
            "league_config_version",
            "league_sampling",
            "pfsp_power",
            "pfsp_teacher_floor",
            "pfsp_exploration",
            "pfsp_prior_games",
            "learner_snapshot_interval_steps",
        )
        migrations.update({key: selected_flags[key] for key in league_keys if key in selected_flags})
    saved_reward_version = int(saved_flags.get("reward_config_version", 0))
    selected_reward_version = int(selected_flags.get("reward_config_version", 0))
    if selected_reward_version > saved_reward_version:
        migrations.update(
            {
                "reward_space": selected_flags.reward_space,
                "reward_space_kwargs": selected_flags.reward_space_kwargs,
                "reward_config_version": selected_reward_version,
            }
        )
    saved_objective_version = int(saved_flags.get("objective_config_version", 0))
    selected_objective_version = int(selected_flags.get("objective_config_version", 0))
    if selected_objective_version > saved_objective_version:
        objective_keys = (
            "objective_config_version",
            "loss_normalization",
            "normalize_advantages",
            "advantage_clip",
            "rule_aux_enabled",
            "rule_aux_cost",
            "rule_aux_strategy",
            "intent_aux_enabled",
            "intent_aux_cost",
            "intent_head_only_finetune",
        )
        migrations.update({key: selected_flags[key] for key in objective_keys if key in selected_flags})
    return OmegaConf.merge(saved_flags, OmegaConf.create(migrations), cli_conf)
