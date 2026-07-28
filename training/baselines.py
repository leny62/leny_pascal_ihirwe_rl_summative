"""Non-learned policies to compare against.

Beating random proves nothing. Beating the scripted agronomist is the actual
claim, for the report and for the venture case.
"""

from __future__ import annotations

import numpy as np

from environment.custom_env import (
    B_DAY,
    B_GDD,
    B_SPRAY_CLOCK,
    HORIZON,
    N_ZONES,
    PRE_HARVEST_INTERVAL_DAYS,
    Z_DEPLETION,
    Z_PEST,
    ZONE_STRIDE,
    Action,
)

# Scripted agronomist thresholds, the rule of thumb a field officer uses
IRRIGATE_DEPLETION_THRESHOLD = 0.40
IRRIGATE_HEAVY_THRESHOLD = 0.60
SPRAY_PEST_THRESHOLD = 0.40
WEEDING_DAYS = (22, 45)
N_SPLIT_DAYS = (12, 34, 56)
K_DAY = 30
HARVEST_GDD_FRACTION = 0.88

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
    """Threshold rules, in the priority order a field officer would apply them.

    Reads only the observation vector, so it runs against the same interface as
    a trained policy. Nitrogen scheduling keys off the normalised day, which is
    all the agent gets too, so this is a fair rule-of-thumb comparison.
    """

    def predict(self, obs: np.ndarray, deterministic: bool = True):
        day = int(round(obs[B_DAY] * HORIZON))
        gdd_frac = obs[B_GDD]
        days_since_spray = obs[B_SPRAY_CLOCK] * 21.0

        depletion = np.array([obs[z * ZONE_STRIDE + Z_DEPLETION] for z in range(N_ZONES)])
        pest = np.array([obs[z * ZONE_STRIDE + Z_PEST] for z in range(N_ZONES)])

        # 1. Harvest once the crop has reached ripening.
        if gdd_frac >= HARVEST_GDD_FRACTION:
            return int(Action.HARVEST_ALL), None

        # 2. Spray on pest pressure, but never inside the pre-harvest interval.
        if pest.max() > SPRAY_PEST_THRESHOLD and days_since_spray >= PRE_HARVEST_INTERVAL_DAYS:
            return int(Action.SPRAY_BIOPESTICIDE), None

        # 3. Nitrogen splits and one potash dressing on the calendar.
        if day in N_SPLIT_DAYS:
            return int(Action.APPLY_N_SPLIT), None
        if day == K_DAY:
            return int(Action.APPLY_K), None

        # 4. Weed before the canopy closes, then once more mid-season.
        if day in WEEDING_DAYS:
            return int(Action.HIRE_WEEDING_CREW), None

        # 5. Irrigate. If several benches are dry, water the whole block; otherwise
        #    the driest one, heavy when it is very dry.
        dry = depletion > IRRIGATE_DEPLETION_THRESHOLD
        if dry.sum() >= 2:
            return int(Action.IRRIGATE_ALL_LIGHT), None
        driest = int(depletion.argmax())
        if depletion[driest] > IRRIGATE_DEPLETION_THRESHOLD:
            heavy = depletion[driest] > IRRIGATE_HEAVY_THRESHOLD
            return int(IRRIGATE_HEAVY[driest] if heavy else IRRIGATE_LIGHT[driest]), None

        return int(Action.IDLE), None
