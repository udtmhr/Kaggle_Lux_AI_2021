"""Survival-aware scratch distillation and RL extensions."""

from .obs import SurvivalStrategicObs
from .reward import StrategicPotentialRewardV3

__all__ = ["SurvivalStrategicObs", "StrategicPotentialRewardV3"]
