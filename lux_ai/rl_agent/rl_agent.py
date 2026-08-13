import os
from pathlib import Path
from types import SimpleNamespace
from typing import *

import numpy as np
import torch
import yaml

from ..lux import annotate
from ..lux.game import Game
from ..lux.game_objects import CityTile, Unit
from ..lux_gym import LuxEnv, create_reward_space, wrappers
from ..nns import create_model, models
from ..utility_constants import MAX_BOARD_SIZE
from ..utils import DEBUG_MESSAGE, LOCAL_EVAL, Stopwatch, flags_to_namespace
from . import data_augmentation
from .action_postprocessing import resolve_collision_rankings

RL_AGENT_CONFIG_PATH = Path(__file__).parent / "rl_agent_config.yaml"
AGENT = None

os.environ["OMP_NUM_THREADS"] = "1"


def model_directory() -> Path:
    explicit = os.environ.get("LUX_AGENT_MODEL_DIR")
    if explicit:
        return Path(explicit).expanduser().resolve()

    model_name = os.environ.get("LUX_AGENT_MODEL")
    if model_name:
        if Path(model_name).name != model_name or model_name in {".", ".."}:
            raise ValueError(f"LUX_AGENT_MODEL must be a model name, not a path: {model_name!r}")
        return Path(__file__).resolve().parents[2] / "models" / model_name

    # Backward-compatible Kaggle layout: config and one checkpoint live beside rl_agent.py.
    return Path(__file__).parent


def checkpoint_path(directory: Optional[Path] = None) -> Path:
    directory = model_directory() if directory is None else directory
    checkpoints = sorted(directory.glob("*.pt"))
    if len(checkpoints) != 1:
        raise ValueError(f"Expected exactly one agent checkpoint in {directory}, found {len(checkpoints)}: {checkpoints}")
    return checkpoints[0]


def pos_to_loc(pos: Tuple[int, int], board_dims: Tuple[int, int] = MAX_BOARD_SIZE) -> int:
    return pos[0] * board_dims[1] + pos[1]


class RLAgent:
    def __init__(self, obs, conf):
        model_dir = model_directory()
        model_config_path = model_dir / "config.yaml"
        if not model_config_path.is_file():
            raise FileNotFoundError(f"Missing model config: {model_config_path}")
        with open(model_config_path, 'r') as f:
            self.model_flags = flags_to_namespace(yaml.safe_load(f))
        with open(RL_AGENT_CONFIG_PATH, 'r') as f:
            self.agent_flags = SimpleNamespace(**yaml.safe_load(f))
        if torch.cuda.is_available():
            if self.agent_flags.device == "player_id":
                device_id = f"cuda:{min(obs.player, torch.cuda.device_count() - 1)}"
            else:
                device_id = self.agent_flags.device
        else:
            device_id = "cpu"
        self.device = torch.device(device_id)

        # Build the env used to convert observations for the model
        env = LuxEnv(
            act_space=self.model_flags.act_space(),
            obs_space=self.model_flags.obs_space(),
            configuration=conf,
            run_game_automatically=False
        )
        reward_space = create_reward_space(self.model_flags)
        env = wrappers.RewardSpaceWrapper(env, reward_space)
        env = env.obs_space.wrap_env(env)
        env = wrappers.PadFixedShapeEnv(env)
        env = wrappers.VecEnv([env])
        # We'll move the data onto the target device if necessary after preprocessing
        env = wrappers.PytorchEnv(env, torch.device("cpu"))
        env = wrappers.DictEnv(env)
        self.env = env
        self.env.reset(observation_updates=obs["updates"], force=True)
        self.action_placeholder = {
            key: torch.zeros(space.shape)
            for key, space in self.unwrapped_env.action_space.get_action_space().spaces.items()
        }

        # Load the model
        self.model = create_model(self.model_flags, self.device)
        checkpoint_states = torch.load(checkpoint_path(model_dir), map_location=self.device)
        self.model.load_state_dict(checkpoint_states["model_state_dict"])
        self.model.eval()

        # Load the data augmenters
        self.data_augmentations = []
        for da_factory in self.agent_flags.data_augmentations:
            da = data_augmentation.__dict__[da_factory](game_state=self.game_state)
            if not isinstance(da, data_augmentation.DataAugmenter):
                raise ValueError(f"Unrecognized data augmentation '{da}' created by: {da_factory}")
            self.data_augmentations.append(da)

        # Various utility properties
        self.me = self.game_state.players[obs.player]
        self.opp = self.game_state.players[(obs.player + 1) % 2]
        self.my_city_tile_mat = np.zeros(MAX_BOARD_SIZE, dtype=bool)
        # NB: loc = pos[0] * n_cols + pos[1]
        self.loc_to_actionable_city_tiles = {}
        self.loc_to_actionable_workers = {}
        self.loc_to_actionable_carts = {}

        # Logging
        self.stopwatch = Stopwatch()

    def __call__(self, obs, conf, raw_model_output: bool = False):
        self.stopwatch.reset()

        self.stopwatch.start("Observation processing")
        self.preprocess(obs, conf)
        env_output = self.get_env_output()
        relevant_env_output_augmented = {
            "obs": self.augment_data(env_output["obs"], is_policy=False),
            "info": {
                "input_mask": self.augment_data(env_output["info"]["input_mask"].unsqueeze(1),
                                                is_policy=False).squeeze(1),
                "available_actions_mask": self.augment_data(env_output["info"]["available_actions_mask"],
                                                            is_policy=True),
            },
        }

        self.stopwatch.stop().start("Model inference")
        with torch.no_grad():
            agent_output_augmented = self.model.select_best_actions(relevant_env_output_augmented)
            agent_output = {
                "policy_logits": self.aggregate_augmented_predictions(agent_output_augmented["policy_logits"]),
                "baseline": agent_output_augmented["baseline"].mean(dim=0, keepdim=True).cpu()
            }
            agent_output["actions"] = {
                key: models.DictActor.logits_to_actions(
                    torch.flatten(val, start_dim=0, end_dim=-2),
                    sample=False,
                    actions_per_square=None
                ).view(*val.shape[:-1], -1)
                for key, val in agent_output["policy_logits"].items()
            }
        # Used for debugging and visualization
        if raw_model_output:
            return agent_output

        self.stopwatch.stop().start("Collision detection")
        if self.agent_flags.use_collision_detection:
            actions = self.resolve_collision_detection(obs, agent_output)
        else:
            actions, _ = self.unwrapped_env.process_actions({
                key: value.squeeze(0).numpy() for key, value in agent_output["actions"].items()
            })
            actions = actions[obs.player]
        self.stopwatch.stop()

        if LOCAL_EVAL:
            # Add transfer annotations locally
            actions.extend(self.get_transfer_annotations(actions))

        value = agent_output["baseline"].squeeze().numpy()[obs.player]
        value_msg = f"Turn: {self.game_state.turn} - Predicted value: {value:.2f}"
        timing_msg = f"{str(self.stopwatch)}"
        overage_time_msg = f"Remaining overage time: {obs['remainingOverageTime']:.2f}"

        actions.append(annotate.sidetext(value_msg))
        DEBUG_MESSAGE(" - ".join([value_msg, timing_msg, overage_time_msg]))
        return actions

    def preprocess(self, obs, conf) -> NoReturn:
        # Do not call manual_step on the first turn, or you will be off-by-1 turn the entire game
        if obs["step"] > 0:
            self.unwrapped_env.manual_step(obs["updates"])
            # need to update turn with obs, otherwise things get messed up if
            # you give the agent obs out of strict order
            self.game_state.turn = obs["step"]
            # I use this in the visualisation code, so need it to be set correctly
            # The official CLI exposes the player id as an attribute on its
            # dict-like Observation object; it is not present as a mapping key.
            self.game_state.id = obs.player

        self.me = self.game_state.players[obs.player]
        self.opp = self.game_state.players[(obs.player + 1) % 2]

        self.my_city_tile_mat[:] = False
        self.loc_to_actionable_city_tiles: Dict[int, CityTile] = {}
        self.loc_to_actionable_workers: Dict[int, List[Unit]] = {}
        self.loc_to_actionable_carts: Dict[int, List[Unit]] = {}
        for unit in self.me.units:
            if unit.can_act():
                if unit.is_worker():
                    dictionary = self.loc_to_actionable_workers
                elif unit.is_cart():
                    dictionary = self.loc_to_actionable_carts
                else:
                    DEBUG_MESSAGE(f"Unrecognized unit type: {unit}")
                    continue
                dictionary.setdefault(pos_to_loc(unit.pos.astuple()), []).append(unit)
        for city_tile in self.me.city_tiles:
            self.my_city_tile_mat[city_tile.pos.x, city_tile.pos.y] = True
            if city_tile.can_act():
                self.loc_to_actionable_city_tiles[pos_to_loc(city_tile.pos.astuple())] = city_tile

        # Remove data augmentations if there are fewer overage seconds than 2x the number of data augmentations
        while max(obs["remainingOverageTime"], 0.) < len(self.data_augmentations) * 2:
            DEBUG_MESSAGE(f"Removing data augmentation: {self.data_augmentations[-1]}")
            del self.data_augmentations[-1]

    def get_env_output(self) -> Dict:
        return self.env.step(self.action_placeholder)

    def augment_data(
            self,
            data: Union[torch.Tensor, Dict[str, torch.Tensor]],
            is_policy: bool
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Applies and concatenates all augmented observations into a single tensor/dict of tensors and moves the tensor
        to the correct device for inference.
        """
        if isinstance(data, dict):
            augmented_data = [data] + [augmentation.apply(data, inverse=False, is_policy=is_policy)
                                       for augmentation in self.data_augmentations]
            return {
                key: torch.cat([d[key] for d in augmented_data], dim=0).to(device=self.device)
                for key in data.keys()
            }
        else:
            augmented_data = [data] + [augmentation.op(data, inverse=False, is_policy=is_policy)
                                       for augmentation in self.data_augmentations]
            return torch.cat(augmented_data, dim=0).to(device=self.device)

    def aggregate_augmented_predictions(self, policy: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Moves the predictions to the cpu, applies the inverse of all augmentations,
        and then returns the mean prediction for each available action.
        """
        policy = {key: val.cpu() for key, val in policy.items()}
        if len(self.data_augmentations) == 0:
            return policy

        policy_reoriented = [{key: val[0].unsqueeze(0) for key, val in policy.items()}]
        for i, augmentation in enumerate(self.data_augmentations):
            augmented_policy = {key: val[i + 1].unsqueeze(0) for key, val in policy.items()}
            policy_reoriented.append(augmentation.apply(augmented_policy, inverse=True, is_policy=True))
        return {
            key: torch.cat([d[key] for d in policy_reoriented], dim=0).mean(dim=0, keepdim=True)
            for key in policy.keys()
        }

    def resolve_collision_detection(self, obs, agent_output) -> List[str]:
        actions_tensors = resolve_collision_rankings(
            self.game_state,
            obs.player,
            agent_output["policy_logits"],
            must_research=self.agent_flags.must_research,
            can_build_carts=self.agent_flags.can_build_carts,
        )
        actions, _ = self.unwrapped_env.process_actions({
            key: value.numpy() for key, value in actions_tensors.items()
        })
        actions = actions[obs.player]
        return actions

    def get_transfer_annotations(self, actions: List[str]) -> List[str]:
        annotations = []
        for act in actions:
            act_split = act.split(" ")
            if act_split[0] == "t":
                unit_from = self.me.get_unit_by_id(act_split[1])
                unit_to = self.me.get_unit_by_id(act_split[2])
                if unit_from is None or unit_to is None:
                    DEBUG_MESSAGE(f"Unrecognized transfer: {act}")
                    continue
                annotations.append(annotate.line(unit_from.pos.x, unit_from.pos.y, unit_to.pos.x, unit_to.pos.y))
                annotations.append(annotate.x(unit_to.pos.x, unit_to.pos.y))
        return annotations

    @property
    def unwrapped_env(self) -> LuxEnv:
        return self.env.unwrapped[0]

    @property
    def game_state(self) -> Game:
        return self.unwrapped_env.game_state

    # Helper functions for debugging
    def set_to_turn_and_call(self, turn: int, *args, **kwargs):
        self.game_state.turn = max(turn - 1, 0)
        return self(*args, **kwargs)


def agent(obs, conf) -> List[str]:
    global AGENT
    if AGENT is None:
        AGENT = RLAgent(obs, conf)
    return AGENT(obs, conf)
