"""Season-long irrigation and input scheduling on a terraced horticulture block.

Four terrace zones down a hillside, one action per simulated day, up to 120 days.

Do not change the spaces or the reward weights once a sweep has started. Every
run completed before the change becomes incomparable.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

N_ZONES = 4
HORIZON = 120
OBS_DIM = 43

# Observation layout. analysis/plots.py labels figures from these, so keep the
# names and the vector in sync.
ZONE_STRIDE = 5
Z_DEPLETION, Z_CANOPY, Z_NITROGEN, Z_PEST, Z_WEED = range(5)

B_DAY = 20
B_GDD = 21
B_STAGE = 22
B_WATER_STRESS = 23
B_N_STRESS = 24
B_RESERVOIR = 25
B_CASH = 26
B_CREW = 27
B_SPRAY_CLOCK = 28
B_N_CLOCK = 29
B_LABOUR_DAYS = 30
B_HARVESTED = 31

X_RAIN_F1, X_RAIN_F2, X_RAIN_F3 = 32, 33, 34
X_ET0 = 35
X_TMAX, X_TMIN = 36, 37
X_PRICE = 38
X_SEASON = 39
X_RAIN_TODAY = 40
X_HUMIDITY = 41
X_MARKET_WINDOW = 42


class Action(IntEnum):
    IDLE = 0
    IRRIGATE_Z0_LIGHT = 1
    IRRIGATE_Z1_LIGHT = 2
    IRRIGATE_Z2_LIGHT = 3
    IRRIGATE_Z3_LIGHT = 4
    IRRIGATE_Z0_HEAVY = 5
    IRRIGATE_Z1_HEAVY = 6
    IRRIGATE_Z2_HEAVY = 7
    IRRIGATE_Z3_HEAVY = 8
    IRRIGATE_ALL_LIGHT = 9
    APPLY_N_SPLIT = 10
    APPLY_K = 11
    SPRAY_BIOPESTICIDE = 12
    HIRE_WEEDING_CREW = 13
    HIRE_WEEDING_CREW_LARGE = 14
    SCOUT = 15
    HARVEST_PARTIAL = 16
    HARVEST_ALL = 17


IRRIGATION_LIGHT_MM = 10.0
IRRIGATION_HEAVY_MM = 25.0
N_SPLIT_KG = 30.0
K_APPLICATION_KG = 40.0
PRE_HARVEST_INTERVAL_DAYS = 7

# Reward weights. Calibrate once so the scripted baseline returns roughly
# -20 to +40 and random is clearly negative, then freeze before the sweep.
W_LABOUR = 0.15
C_WATER = 0.02
C_INPUT = 0.01
C_LEACH = 0.35
C_STRESS = 1.20
C_DEBT = 0.05
C_TIME = 0.05
FLOWERING_STRESS_WEIGHT = 2.5
PENALTY_CROP_FAILURE = -200.0
PENALTY_INSOLVENCY = -150.0

# Observation noise on pest, weed and nitrogen grows until the agent scouts.
SCOUT_NOISE_PER_DAY = 0.015
SCOUT_NOISE_CAP = 0.25


class UmurimaEnv(gym.Env):
    """Block operations controller, one action per simulated day."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 12}

    def __init__(
        self,
        render_mode: str | None = None,
        horizon: int = HORIZON,
        block_id: str = "KG-NYA-004",
    ) -> None:
        super().__init__()
        self.render_mode = render_mode
        self.horizon = horizon
        self.block_id = block_id

        self.action_space = spaces.Discrete(len(Action))
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32
        )

        self._renderer = None  # built lazily, training must never import OpenGL

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        # TODO: draw the randomised start state, generate the full season of
        # weather from self.np_random, reset the ledger.
        raise NotImplementedError

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        # TODO: apply action, advance agronomy one day, accumulate reward,
        # evaluate termination. Keep terminated and truncated distinct.
        raise NotImplementedError

    def render(self) -> np.ndarray | None:
        if self.render_mode is None:
            return None
        if self._renderer is None:
            from environment.rendering import BlockRenderer  # lazy on purpose

            self._renderer = BlockRenderer(self.render_mode)
        return self._renderer.draw(self._render_state())

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _observe(self) -> np.ndarray:
        """Build the 43 vector, applying scout noise to pest, weed and nitrogen."""
        raise NotImplementedError

    def _reward(self, action: int) -> float:
        raise NotImplementedError

    def _terminated(self) -> tuple[bool, str | None]:
        """Returns the flag and the cause, for logging and the info dict."""
        raise NotImplementedError

    def _render_state(self) -> dict[str, Any]:
        """Flat dict the renderer and the JSON API both consume."""
        raise NotImplementedError
