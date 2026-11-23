"""
Grandmaster Ensembling Primitives
"""

from src.qaggle.ensembling.hill_climbing import HillClimbingOptimizer
from src.qaggle.ensembling.model_soup import (
    SwitchModelEMA,
    average_state_dicts,
    dare_ties_merge,
    greedy_model_soup,
)
from src.qaggle.ensembling.seed_averaging import (
    SoftContinuousPseudoLabeler,
    ThreeSeedAveraging,
)

__all__ = [
    "HillClimbingOptimizer",
    "SoftContinuousPseudoLabeler",
    "SwitchModelEMA",
    "ThreeSeedAveraging",
    "average_state_dicts",
    "dare_ties_merge",
    "greedy_model_soup",
]
