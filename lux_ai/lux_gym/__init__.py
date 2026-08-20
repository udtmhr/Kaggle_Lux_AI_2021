import torch
from typing import Optional

from . import act_spaces, obs_spaces, reward_spaces, multi_subtask
from .lux_env import LuxEnv
from .wrappers import DictEnv, LoggingEnv, PadFixedShapeEnv, PytorchEnv, RewardSpaceWrapper, RulePriorWrapper, VecEnv

ACT_SPACES_DICT = {
    key: val for key, val in act_spaces.__dict__.items()
    if isinstance(val, type) and issubclass(val, act_spaces.BaseActSpace)
}
OBS_SPACES_DICT = {
    key: val for key, val in obs_spaces.__dict__.items()
    if isinstance(val, type) and issubclass(val, obs_spaces.BaseObsSpace)
}
REWARD_SPACES_DICT = {
    key: val for key, val in reward_spaces.__dict__.items()
    if isinstance(val, type) and issubclass(val, reward_spaces.BaseRewardSpace)
}
REWARD_SPACES_DICT.update({
    key: val for key, val in multi_subtask.__dict__.items()
    if isinstance(val, type) and issubclass(val, reward_spaces.BaseRewardSpace)
})
SUBTASKS_DICT = {
    key: val for key, val in reward_spaces.__dict__.items()
    if isinstance(val, type) and issubclass(val, reward_spaces.Subtask)
}
SUBTASK_SAMPLERS_DICT = {
    key: val for key, val in multi_subtask.__dict__.items()
    if isinstance(val, type) and issubclass(val, multi_subtask.SubtaskSampler)
}


def create_flexible_obs_space(flags, teacher_flags: Optional) -> obs_spaces.BaseObsSpace:
    if teacher_flags is not None and teacher_flags.obs_space != flags.obs_space:
        # Train a student using a different observation space than the teacher
        return obs_spaces.MultiObs({
            "teacher_": teacher_flags.obs_space(**teacher_flags.obs_space_kwargs),
            "student_": flags.obs_space(**flags.obs_space_kwargs)
        })
    else:
        return flags.obs_space(**flags.obs_space_kwargs)


def rule_prior_enabled(flags) -> bool:
    """Return whether any action head has a non-zero configured rule prior."""
    default_alpha = float(getattr(flags, "rule_prior_alpha", 0.0))
    return any(
        float(getattr(flags, f"rule_prior_alpha_{entity}", default_alpha)) != 0.0
        for entity in ("worker", "cart", "city_tile")
    )


def create_env(
    flags,
    device: torch.device,
    teacher_flags: Optional = None,
    seed: Optional[int] = None,
    reward_game_counter=None,
) -> DictEnv:
    if seed is None:
        seed = flags.seed
    envs = []
    for i in range(flags.n_actor_envs):
        env = LuxEnv(
            act_space=flags.act_space(),
            obs_space=create_flexible_obs_space(flags, teacher_flags),
            seed=seed
        )
        reward_space = create_reward_space(flags, reward_game_counter=reward_game_counter)
        env = RewardSpaceWrapper(env, reward_space)
        env = env.obs_space.wrap_env(env)
        if rule_prior_enabled(flags):
            env = RulePriorWrapper(env)
        env = PadFixedShapeEnv(env)
        env = LoggingEnv(env, reward_space)
        envs.append(env)
    env = VecEnv(envs)
    env = PytorchEnv(env, device)
    env = DictEnv(env)
    return env


def create_reward_space(flags, reward_game_counter=None) -> reward_spaces.BaseRewardSpace:
    if flags.reward_space is multi_subtask.MultiSubtask:
        assert "subtasks" in flags.reward_space_kwargs and "subtask_sampler" in flags.reward_space_kwargs
        subtasks = [SUBTASKS_DICT[s] for s in flags.reward_space_kwargs["subtasks"]]
        subtask_sampler = SUBTASK_SAMPLERS_DICT[flags.reward_space_kwargs["subtask_sampler"]]
        reward_space = flags.reward_space(subtasks, subtask_sampler)
    else:
        reward_space = flags.reward_space(**flags.reward_space_kwargs)

    if reward_game_counter is not None and hasattr(reward_space, "set_global_game_counter"):
        reward_space.set_global_game_counter(reward_game_counter)

    # 衝突ペナルティが有効な場合、reward spaceをラッパーで包む
    collision_cost = float(getattr(flags, "collision_penalty_cost", 0.0))
    if collision_cost > 0.0:
        from ..strategic_rl.collision_penalty import CollisionPenaltyWrapper
        reward_space = CollisionPenaltyWrapper(
            inner=reward_space,
            penalty_start=float(getattr(flags, "collision_penalty_ramp_start", collision_cost)),
            penalty_end=float(getattr(flags, "collision_penalty_ramp_end", collision_cost)),
            ramp_steps=int(getattr(flags, "collision_penalty_ramp_steps", 50_000)),
        )
        if reward_game_counter is not None:
            reward_space.set_global_game_counter(reward_game_counter)

    return reward_space
