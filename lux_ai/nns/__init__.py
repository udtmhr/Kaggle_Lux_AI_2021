import logging
import torch
from torch import nn
from typing import Optional

from .models import BasicActorCriticNetwork
from .in_blocks import ConvEmbeddingInputLayer
from .attn_blocks import ViTBlock, RPSA, GPSA
from .conv_blocks import ResidualBlock, ParallelDilationResidualBlock
from .unet import UNET
from ..lux_gym import create_flexible_obs_space, obs_spaces
from ..utility_constants import MAX_BOARD_SIZE
from ..strategic_rl.models import SurvivalStrategicBackbone, compatible_attention_heads


def create_model(
        flags,
        device: torch.device,
        teacher_model_flags: Optional = None,
        is_teacher_model: bool = False
) -> nn.Module:
    obs_space = create_flexible_obs_space(flags, teacher_model_flags)
    if isinstance(obs_space, obs_spaces.MultiObs):
        if is_teacher_model:
            obs_space_prefix = "teacher_"
        else:
            obs_space_prefix = "student_"
        assert obs_space_prefix in obs_space.named_obs_spaces, f"{obs_space_prefix} not in {obs_space.named_obs_spaces}"
    else:
        obs_space_prefix = ""

    return _create_model(
        teacher_model_flags if is_teacher_model else flags,
        device,
        obs_space,
        obs_space_prefix
    )


def _create_model(
        flags,
        device: torch.device,
        obs_space: obs_spaces.BaseObsSpace,
        obs_space_prefix: str
):
    act_space = flags.act_space()
    conv_embedding_input_layer = ConvEmbeddingInputLayer(
        obs_space=obs_space.get_obs_spec(),
        embedding_dim=flags.embedding_dim,
        out_dim=flags.hidden_dim,
        n_merge_layers=flags.n_merge_layers,
        sum_player_embeddings=flags.sum_player_embeddings,
        use_index_select=flags.use_index_select,
        obs_space_prefix=obs_space_prefix
    )
    if flags.model_arch == "survival_strategic":
        global_channels = flags.hidden_dim * 2
        base_model = nn.Sequential(
            conv_embedding_input_layer,
            SurvivalStrategicBackbone(
                channels=flags.hidden_dim,
                attention_blocks=getattr(flags, "attention_blocks", 4),
                heads=compatible_attention_heads(global_channels, getattr(flags, "n_heads", 6)),
            ),
        )
    elif flags.model_arch == "conv16_matched":
        base_model = nn.Sequential(
            conv_embedding_input_layer,
            *[ResidualBlock(
                in_channels=flags.hidden_dim,
                out_channels=flags.hidden_dim,
                height=MAX_BOARD_SIZE[0],
                width=MAX_BOARD_SIZE[1],
                kernel_size=flags.kernel_size,
                normalize=flags.normalize,
                activation=nn.LeakyReLU,
                rescale_se_input=flags.rescale_se_input,
            ) for _ in range(16)]
        )
    elif flags.model_arch == "conv_model":
        base_model = nn.Sequential(
            conv_embedding_input_layer,
            *[ResidualBlock(
                in_channels=flags.hidden_dim,
                out_channels=flags.hidden_dim,
                height=MAX_BOARD_SIZE[0],
                width=MAX_BOARD_SIZE[1],
                kernel_size=flags.kernel_size,
                normalize=flags.normalize,
                activation=nn.LeakyReLU,
                rescale_se_input=flags.rescale_se_input,
            ) for _ in range(flags.n_blocks)]
        )
    elif flags.model_arch == "pd_conv_model":
        logging.warning("Dilation is slow for some Pytorch/CUDNN versions.")
        base_model = nn.Sequential(
            conv_embedding_input_layer,
            *[ParallelDilationResidualBlock(
                in_channels=flags.hidden_dim,
                out_channels=flags.hidden_dim,
                height=MAX_BOARD_SIZE[0],
                width=MAX_BOARD_SIZE[1],
                kernel_size=flags.kernel_size,
                normalize=flags.normalize,
                activation=nn.LeakyReLU,
                rescale_se_input=flags.rescale_se_input,
            ) for _ in range(flags.n_blocks)]
        )
    elif flags.model_arch == "unet_model":
        base_model = nn.Sequential(
            conv_embedding_input_layer,
            UNET(
                n_blocks_per_reduction=flags.n_blocks_per_reduction,
                in_out_channels=flags.hidden_dim,
                height=MAX_BOARD_SIZE[0],
                width=MAX_BOARD_SIZE[1],
                # Residual block kwargs
                kernel_size=flags.kernel_size,
                normalize=flags.normalize,
                activation=nn.LeakyReLU,
                rescale_se_input=flags.rescale_se_input,
            )
        )
    elif flags.model_arch == "RPSA_model":
        base_model = nn.Sequential(
            conv_embedding_input_layer,
            *[ViTBlock(
                in_channels=flags.hidden_dim,
                out_channels=flags.hidden_dim,
                height=MAX_BOARD_SIZE[0],
                width=MAX_BOARD_SIZE[1],
                mhsa_layer=RPSA(
                    in_channels=flags.hidden_dim,
                    heads=flags.n_heads,
                    height=MAX_BOARD_SIZE[0],
                    width=MAX_BOARD_SIZE[1]
                ),
                normalize=flags.normalize,
            ) for _ in range(flags.n_blocks)]
        )
    elif flags.model_arch == "GPSA_model":
        base_model = nn.Sequential(
            conv_embedding_input_layer,
            *[ViTBlock(
                in_channels=flags.hidden_dim,
                out_channels=flags.hidden_dim,
                height=MAX_BOARD_SIZE[0],
                width=MAX_BOARD_SIZE[1],
                mhsa_layer=GPSA(
                    dim=flags.hidden_dim,
                    n_heads=flags.n_heads,
                    height=MAX_BOARD_SIZE[0],
                    width=MAX_BOARD_SIZE[1]
                ),
                normalize=flags.normalize,
            ) for _ in range(flags.n_blocks)]
        )
    else:
        raise NotImplementedError(f"Model_arch: {flags.model_arch}")

    model = BasicActorCriticNetwork(
        base_model=base_model,
        base_out_channels=flags.hidden_dim,
        action_space=act_space.get_action_space(),
        reward_space=flags.reward_space.get_reward_spec(),
        n_value_heads=1,
        rescale_value_input=flags.rescale_value_input,
        intent_classes=(4 if getattr(flags, "intent_aux_enabled", False) else 0),
        value_critic=getattr(flags, "value_critic", "scalar"),
        value_num_bins=int(getattr(flags, "value_num_bins", 101)),
        value_support_min=float(getattr(flags, "value_support_min", -2.0)),
        value_support_max=float(getattr(flags, "value_support_max", 2.0)),
    )
    rule_prior_alpha = float(getattr(flags, "rule_prior_alpha", 0.0))
    model.actor.rule_prior_alpha.copy_(torch.tensor(rule_prior_alpha))
    return model.to(device=device)
