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
        migrations.update(
            {
                "league_enabled": selected_flags.league_enabled,
                "league_opponents": selected_flags.league_opponents,
                "league_config_version": selected_version,
            }
        )
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
    return OmegaConf.merge(saved_flags, OmegaConf.create(migrations), cli_conf)
