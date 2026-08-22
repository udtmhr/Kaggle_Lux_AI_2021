from types import SimpleNamespace

import gym
import pytest
import torch

from lux_ai.lux_gym.act_spaces import ACTION_MEANINGS, MAX_OVERLAPPING_ACTIONS
from lux_ai.strategic_rl.intrinsic import (
    ControllableEpisodicCuriosity,
    EllipticalEpisodicMemory,
    RunningMoments,
    controllable_change_masks,
    intrinsic_beta,
)
from lux_ai.strategic_rl.obs import SurvivalStrategicObs


def _model_input(batch_size: int = 2):
    obs_space = SurvivalStrategicObs().get_obs_spec()
    obs = {}
    for key, spec in obs_space.spaces.items():
        dtype = torch.float32 if isinstance(spec, gym.spaces.Box) else torch.int64
        obs[key] = torch.zeros((batch_size, *spec.shape), dtype=dtype)
    return obs_space, {
        "obs": obs,
        "info": {"input_mask": torch.ones((batch_size, 1, 32, 32), dtype=torch.bool)},
    }


def test_elliptical_bonus_decays_resets_and_zeroes_terminal():
    memory = EllipticalEpisodicMemory(2, 4, torch.device("cpu"))
    embedding = torch.tensor([[[1.0, 0, 0, 0], [0, 1.0, 0, 0]]] * 2)
    first = memory.bonus(embedding, torch.zeros(2, dtype=torch.bool))
    second = memory.bonus(embedding, torch.zeros(2, dtype=torch.bool))
    assert torch.all(second < first)
    assert torch.isfinite(memory.condition_proxy).all()
    terminal = memory.bonus(embedding, torch.tensor([True, False]))
    assert terminal[0].eq(0).all()
    after_reset = memory.bonus(embedding, torch.zeros(2, dtype=torch.bool))
    assert torch.allclose(after_reset[0], first[0])


def test_curiosity_embedding_is_player_relative_rot180_invariant_and_finite():
    obs_space, model_input = _model_input()
    model_input["obs"]["worker"][0, 0, 0, 3, 5] = 1
    model = ControllableEpisodicCuriosity(obs_space, embedding_dim=8).eval()
    embedding, _ = model.encode(model_input)

    swapped = {
        "obs": {
            key: value.flip(2) if value.shape[2] == 2 else value.clone() for key, value in model_input["obs"].items()
        },
        "info": model_input["info"],
    }
    swapped_embedding, _ = model.encode(swapped)
    assert torch.isfinite(embedding).all()
    assert torch.allclose(swapped_embedding[:, 0], embedding[:, 1], atol=1e-6)
    assert torch.allclose(swapped_embedding[:, 1], embedding[:, 0], atol=1e-6)

    rotated = {
        "obs": {
            key: torch.rot90(value, 2, dims=(-2, -1)) if value.ndim == 5 else value.clone()
            for key, value in model_input["obs"].items()
        },
        "info": {"input_mask": torch.rot90(model_input["info"]["input_mask"], 2, dims=(-2, -1))},
    }
    rotated_embedding, _ = model.encode(rotated)
    assert torch.allclose(rotated_embedding, embedding, atol=1e-6)


def test_inverse_dynamics_handles_empty_and_active_action_masks():
    obs_space, model_input = _model_input(batch_size=1)
    model = ControllableEpisodicCuriosity(obs_space, embedding_dim=8)
    _, spatial = model.encode(model_input)
    actions = {}
    taken = {}
    for entity, meanings in ACTION_MEANINGS.items():
        actions[entity] = torch.zeros((1, 1, 2, 32, 32, MAX_OVERLAPPING_ACTIONS), dtype=torch.long)
        taken[entity] = torch.zeros((1, 1, 2, 32, 32, len(meanings)), dtype=torch.bool)
    loss, stats = model.inverse_dynamics_loss(
        spatial,
        spatial,
        actions,
        taken,
        torch.ones((1, 2), dtype=torch.bool),
    )
    assert torch.isfinite(loss)
    assert all(stats[f"{entity}_count"] == 0 for entity in ACTION_MEANINGS)

    taken["worker"][0, 0, 0, 2, 3, 0] = True
    loss, stats = model.inverse_dynamics_loss(
        spatial,
        spatial,
        actions,
        taken,
        torch.ones((1, 2), dtype=torch.bool),
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert stats["worker_count"] == 1
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_intrinsic_schedule_and_running_moments_resume():
    flags = SimpleNamespace(
        intrinsic_pretrain_steps=16000,
        intrinsic_ramp_end_step=25000,
        intrinsic_decay_start_step=40000,
        intrinsic_decay_end_step=50000,
        intrinsic_beta_max=0.1,
    )
    assert intrinsic_beta(flags, 15999) == 0
    assert intrinsic_beta(flags, 25000) == pytest.approx(0.1)
    assert intrinsic_beta(flags, 45000) == pytest.approx(0.05)
    assert intrinsic_beta(flags, 50000) == 0

    moments = RunningMoments()
    normalized = moments.normalize(torch.tensor([0.0, 1.0, 2.0]), clip=5.0)
    restored = RunningMoments.from_state_dict(moments.state_dict())
    assert torch.isfinite(normalized).all()
    assert restored.state_dict() == moments.state_dict()


def test_controllable_change_masks_are_player_relative():
    worker = torch.zeros((2, 1, 1, 2, 3, 3))
    worker[1, 0, 0, 0, 1, 1] = 1
    own, enemy = controllable_change_masks({"worker": worker})
    assert own.tolist() == [[[True, False]]]
    assert enemy.tolist() == [[[False, True]]]
