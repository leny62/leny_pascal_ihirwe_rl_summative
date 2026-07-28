"""Non-learned policies to compare against.

Beating random proves nothing. Beating the scripted agronomist is the actual
claim, for the report and for the venture case.
"""

from __future__ import annotations

import numpy as np

from environment.custom_env import Action

# Scripted agronomist thresholds, the rule of thumb a field officer uses
IRRIGATE_DEPLETION_THRESHOLD = 0.50
SPRAY_PEST_THRESHOLD = 0.40
WEEDING_DAY = 25
N_SPLIT_DAYS = (12, 34, 56)

# Indexed by zone, so the scripted policy can act on the driest bench
IRRIGATE_LIGHT = (
    Action.IRRIGATE_Z0_LIGHT,
    Action.IRRIGATE_Z1_LIGHT,
    Action.IRRIGATE_Z2_LIGHT,
    Action.IRRIGATE_Z3_LIGHT,
)
IRRIGATE_HEAVY = (
    Action.IRRIGATE_Z0_HEAVY,
    Action.IRRIGATE_Z1_HEAVY,
    Action.IRRIGATE_Z2_HEAVY,
    Action.IRRIGATE_Z3_HEAVY,
)


class RandomPolicy:
    def __init__(self, action_space, seed: int = 0) -> None:
        self.action_space = action_space
        self.action_space.seed(seed)

    def predict(self, obs: np.ndarray, deterministic: bool = True):
        return self.action_space.sample(), None


class ScriptedAgronomist:
    """Threshold rules, in the priority order a field officer would apply them."""

    def predict(self, obs: np.ndarray, deterministic: bool = True):
        # TODO: order is harvest at maturity, spray on threshold outside the PHI,
        # nitrogen on schedule, weed once, then irrigate the driest zone over
        # threshold, else idle.
        raise NotImplementedError
