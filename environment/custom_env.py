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

from environment import agronomy as ag
from environment import stochastics as st
from environment.traceability import BlockLedger, FieldEvent

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
PARTIAL_HARVEST_FRACTION = 0.40
# Pods do not exist before this much of the thermal season has passed, so a
# harvest called earlier picks nothing and does not end the season. Without this
# the agent can end any episode on day one at almost no cost, which scores better
# than exploring and becomes the policy every run converges to.
HARVEST_MIN_GDD_FRAC = 0.60
CONTRACT_WINDOW_DAY = 100

# Physical scales
ZONE_AREA_M2 = ag.ZONE_AREA_M2
MM_TO_M3_PER_ZONE = ZONE_AREA_M2 / 1000.0  # 1 mm over one terrace, in m3
RESERVOIR_CAPACITY_M3 = 250.0
RESERVOIR_CATCHMENT_M3_PER_MM = 4.0

# Real cash costs and revenue, thousands of RWF (kRWF)
COST_PUMP_PER_MM = 0.05
COST_N_SPLIT = 15.0
COST_K = 12.0
COST_SPRAY = 8.0
COST_CREW_DAY = 3.0
COST_SCOUT = 1.0
COST_PICKING = 5.0
K_HARVEST_INDEX_BONUS = 0.02
K_HARVEST_INDEX_BONUS_CAP = 0.04

# Reward weights. Calibrate once so the scripted baseline returns roughly
# -20 to +40 and random is clearly negative, then freeze before the sweep.
# Revenue leads; the shaping terms nudge without swamping the realised harvest.
# Debt is only lightly penalised because pre-harvest working capital is normal.
REVENUE_SCALE = 0.08
W_LABOUR = 0.15
C_WATER = 0.02
C_INPUT = 0.01
C_LEACH = 0.05
C_STRESS = 0.80
C_DEBT = 0.01
C_TIME = 0.03
FLOWERING_STRESS_WEIGHT = 2.5
PENALTY_CROP_FAILURE = -50.0
PENALTY_INSOLVENCY = -50.0

# Termination thresholds. The stress ceiling is about 0.28 (you cannot irrigate
# less than nothing, and rain sustains the rest), so the failure line sits below
# that: a dry season left largely unirrigated fails, competent play (~0.03) does
# not. The canopy check is gated to the growth phase, so it fires on a genuine
# collapse but not on establishment or on deliberate senescence at ripening.
FAIL_WATER_STRESS = 0.22
FAIL_CANOPY = 0.08
FAIL_CANOPY_MIN_GDD_FRAC = 0.35
FAIL_CANOPY_MAX_GDD_FRAC = 0.85
FAIL_MIN_DAY = 25
INSOLVENCY_KRWF = -100.0

# Observation noise on pest, weed and nitrogen grows until the agent scouts.
SCOUT_NOISE_PER_DAY = 0.015
SCOUT_NOISE_CAP = 0.25

# Zone soil depth means, ridge (shallow) to valley bottom (deep)
ZONE_DEPTH_MEAN_M = np.array([0.60, 0.80, 1.00, 1.20])


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
        rng = self.np_random

        self._season = "A" if rng.random() < 0.5 else "B"
        self._weather = st.generate_season(rng, self._season, self.horizon)

        self._zones = []
        for z in range(N_ZONES):
            depth = float(np.clip(rng.normal(ZONE_DEPTH_MEAN_M[z], 0.08), 0.30, 1.5))
            taw0 = ag.total_available_water(depth, ag.ROOT_DEPTH_MIN_M)
            self._zones.append(
                ag.ZoneState(
                    index=z,
                    soil_depth_m=depth,
                    depletion_mm=float(rng.uniform(0.10, 0.50) * taw0),
                    canopy_cover=ag.CC_0,
                    biomass_g_m2=0.0,
                    nitrogen_kg_ha=float(rng.uniform(20.0, 70.0)),
                    pest_pressure=0.0,
                    weed_pressure=float(rng.uniform(0.02, 0.08)),
                    pest_damage=0.0,
                )
            )

        self._day = 0
        self._gdd_cum = 0.0
        self._cash = float(rng.uniform(150.0, 400.0))
        self._reservoir_m3 = float(rng.uniform(0.40, 1.00) * RESERVOIR_CAPACITY_M3)
        self._crew_schedule = rng.integers(1, 4, size=self.horizon)
        self._pest_arrival_day = int(rng.poisson(35))

        self._harvested_fraction = 0.0
        self._days_since_spray = PRE_HARVEST_INTERVAL_DAYS + 1
        self._days_since_n = 30
        self._scout_staleness = 0
        self._labour_days_cum = 0.0
        self._k_applications = 0

        self._stress_accum = 0.0
        self._n_stress_accum = 0.0
        self._flower_stress_accum = 0.0
        self._flower_days = 0

        self._last_action = Action.IDLE
        self._last_reward = 0.0
        self._reward_terms: dict[str, float] = {}
        self._return = 0.0
        self._terminated = False
        self._truncated = False
        self._cause: str | None = None

        self._ledger = BlockLedger(block_id=self.block_id, season=f"2026{self._season}")

        return self._observe(), self._info()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = Action(int(action))
        self._last_action = action

        irrigation_mm = np.zeros(N_ZONES)
        input_cost = 0.0
        labour_days = 0.0
        revenue = 0.0
        harvested_before = self._harvested_fraction

        # Seed the first pest incursion once its arrival day passes.
        if self._day >= self._pest_arrival_day:
            for zone in self._zones:
                if zone.pest_pressure == 0.0:
                    zone.pest_pressure = 0.05

        # --- apply the action -------------------------------------------------
        if action in _IRRIGATE_LIGHT_ZONE:
            irrigation_mm[_IRRIGATE_LIGHT_ZONE[action]] = IRRIGATION_LIGHT_MM
        elif action in _IRRIGATE_HEAVY_ZONE:
            irrigation_mm[_IRRIGATE_HEAVY_ZONE[action]] = IRRIGATION_HEAVY_MM
        elif action == Action.IRRIGATE_ALL_LIGHT:
            irrigation_mm[:] = IRRIGATION_LIGHT_MM
        elif action == Action.APPLY_N_SPLIT:
            for zone in self._zones:
                zone.nitrogen_kg_ha += N_SPLIT_KG
            input_cost += COST_N_SPLIT
            self._days_since_n = 0
        elif action == Action.APPLY_K:
            input_cost += COST_K
            self._k_applications += 1
        elif action == Action.SPRAY_BIOPESTICIDE:
            for zone in self._zones:
                zone.pest_pressure *= ag.SPRAY_KNOCKDOWN
            input_cost += COST_SPRAY
            self._days_since_spray = 0
        elif action == Action.HIRE_WEEDING_CREW:
            for zone in self._zones:
                zone.weed_pressure *= 0.60
            labour_days = 1.0
        elif action == Action.HIRE_WEEDING_CREW_LARGE:
            for zone in self._zones:
                zone.weed_pressure *= 0.25
            labour_days = 3.0
        elif action == Action.SCOUT:
            self._scout_staleness = 0
            self._cash -= COST_SCOUT
        elif action in (Action.HARVEST_PARTIAL, Action.HARVEST_ALL):
            revenue = self._do_harvest(action)
            labour_days = 1.0

        irrigation_mm = self._draw_from_reservoir(irrigation_mm)
        irrigation_total = float(irrigation_mm.sum())
        self._cash -= COST_PUMP_PER_MM * irrigation_total
        self._cash -= input_cost
        self._cash -= COST_CREW_DAY * labour_days if action in _WEEDING else 0.0
        if action in (Action.HARVEST_PARTIAL, Action.HARVEST_ALL):
            self._cash -= COST_PICKING
        self._cash += revenue
        self._labour_days_cum += labour_days
        picked = self._harvested_fraction - harvested_before

        # --- advance the agronomy one day ------------------------------------
        n_leached = 0.0
        ks_values = []
        w = self._weather
        day = self._day
        self._gdd_cum += ag.growing_degree_days(w.t_max[day], w.t_min[day])

        run_on = 0.0
        for z in range(N_ZONES):
            new_zone, fluxes = ag.step_zone(
                self._zones[z],
                rain_mm=float(w.rain_mm[day]),
                run_on_mm=run_on,
                irrigation_mm=float(irrigation_mm[z]),
                et0_mm=float(w.et0_mm[day]),
                t_max=float(w.t_max[day]),
                t_min=float(w.t_min[day]),
                humidity=float(w.humidity[day]),
                gdd_cum=self._gdd_cum,
                day=day,
            )
            self._zones[z] = new_zone
            run_on = ag.RUNOFF_ROUTED_DOWNSLOPE * fluxes["runoff_mm"]
            n_leached += fluxes["nitrogen_leached_kg"]
            ks_values.append(fluxes["ks"])

        self._reservoir_m3 = min(
            RESERVOIR_CAPACITY_M3,
            self._reservoir_m3 + RESERVOIR_CATCHMENT_M3_PER_MM * float(w.rain_mm[day]),
        )

        mean_ks = float(np.mean(ks_values))
        self._stress_accum += 1.0 - mean_ks
        self._n_stress_accum += 1.0 - float(np.mean([ag.nitrogen_stress(z.nitrogen_kg_ha) for z in self._zones]))
        if self._is_flowering():
            self._flower_stress_accum += 1.0 - mean_ks
            self._flower_days += 1

        self._log_action(action, irrigation_mm, revenue)

        # --- reward -----------------------------------------------------------
        stage_weight = FLOWERING_STRESS_WEIGHT if self._is_flowering() else 1.0
        terms = {
            "revenue": REVENUE_SCALE * revenue,
            # Employment pays out with the crop, not per day worked. Paid daily
            # it is worth ~18 points over a season with no harvest required, so a
            # policy can hire crews for 120 days, sell nothing, and still score
            # well. Settling it against the fraction actually picked keeps the
            # job-creation objective while making it conditional on a real farm.
            "labour": W_LABOUR * self._labour_days_cum * picked,
            "water": -C_WATER * irrigation_total,
            "input": -C_INPUT * input_cost,
            "leach": -C_LEACH * n_leached,
            "stress": -C_STRESS * stage_weight * (1.0 - mean_ks),
            "debt": -C_DEBT * max(0.0, -self._cash),
            "time": -C_TIME,
        }
        reward = sum(terms.values())
        self._reward_terms = terms

        # --- clocks and day advance ------------------------------------------
        self._days_since_spray += 1
        self._days_since_n += 1
        self._scout_staleness += 1
        self._day += 1

        # --- termination ------------------------------------------------------
        terminated, cause = self._check_terminated()
        truncated = (not terminated) and self._day >= self.horizon
        if cause == "crop_failure":
            reward += PENALTY_CROP_FAILURE
        elif cause == "insolvency":
            reward += PENALTY_INSOLVENCY

        self._terminated, self._truncated, self._cause = terminated, truncated, cause
        self._last_reward = reward
        self._return += reward

        return self._observe(), float(reward), terminated, truncated, self._info()

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
    # Action helpers
    # ------------------------------------------------------------------

    def _draw_from_reservoir(self, irrigation_mm: np.ndarray) -> np.ndarray:
        """Clamp irrigation to available water, drawing the tank down."""
        needed_m3 = float(irrigation_mm.sum()) * MM_TO_M3_PER_ZONE
        if needed_m3 <= 0.0:
            return irrigation_mm
        deliverable = min(1.0, self._reservoir_m3 / needed_m3)
        irrigation_mm = irrigation_mm * deliverable
        self._reservoir_m3 -= float(irrigation_mm.sum()) * MM_TO_M3_PER_ZONE
        return irrigation_mm

    def _do_harvest(self, action: Action) -> float:
        standing = 1.0 - self._harvested_fraction
        if standing <= 0.0:
            return 0.0
        if self._gdd_frac() < HARVEST_MIN_GDD_FRAC:
            # Nothing has podded yet. The crew still went out, so the day and the
            # labour are spent, but the crop stays in the ground.
            return 0.0
        picked = min(PARTIAL_HARVEST_FRACTION, standing) if action == Action.HARVEST_PARTIAL else standing
        revenue = picked * self._standing_yield_kg() * self._price_today() * self._quality() / 1000.0
        self._harvested_fraction += picked
        return revenue

    def _standing_yield_kg(self) -> float:
        flower_stress = self._flower_stress_accum / max(self._flower_days, 1)
        hi = ag.harvest_index(flower_stress) + min(
            self._k_applications * K_HARVEST_INDEX_BONUS, K_HARVEST_INDEX_BONUS_CAP
        )
        return sum(hi * z.biomass_g_m2 * ZONE_AREA_M2 / 1000.0 for z in self._zones)

    def _quality(self) -> float:
        mean_damage = float(np.mean([z.pest_damage for z in self._zones]))
        q = 1.0 - 0.4 * mean_damage
        if self._days_since_spray < PRE_HARVEST_INTERVAL_DAYS:
            q *= 0.35
        if self._gdd_frac() < 0.85:
            q *= 0.80
        return q

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    def _gdd_frac(self) -> float:
        return min(self._gdd_cum / ag.GDD_MATURITY, 1.0)

    def _is_flowering(self) -> bool:
        return 0.45 <= self._gdd_frac() < 0.75

    def _stage_value(self) -> float:
        f = self._gdd_frac()
        if f < 0.15:
            return 0.0
        if f < 0.45:
            return 0.33
        if f < 0.75:
            return 0.66
        return 1.0

    def _price_today(self) -> float:
        return float(self._weather.price_rwf[min(self._day, self.horizon - 1)])

    def _zone_depletion_frac(self, zone) -> float:
        """Root zone depletion as a fraction of that zone's total available water."""
        root_depth = ag.root_depth_m(self._gdd_frac())
        taw = ag.total_available_water(zone.soil_depth_m, root_depth)
        return float(np.clip(zone.depletion_mm / taw if taw > 0 else 0.0, 0.0, 1.0))

    def _mean_canopy(self) -> float:
        return float(np.mean([z.canopy_cover for z in self._zones]))

    def _check_terminated(self) -> tuple[bool, str | None]:
        if self._harvested_fraction >= 1.0:
            return True, "harvested"
        if self._cash < INSOLVENCY_KRWF:
            return True, "insolvency"
        if self._day >= FAIL_MIN_DAY and self._stress_accum / max(self._day, 1) > FAIL_WATER_STRESS:
            return True, "crop_failure"
        if FAIL_CANOPY_MIN_GDD_FRAC < self._gdd_frac() < FAIL_CANOPY_MAX_GDD_FRAC and self._mean_canopy() < FAIL_CANOPY:
            return True, "crop_failure"
        return False, None

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _observe(self) -> np.ndarray:
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        noise_std = min(SCOUT_NOISE_PER_DAY * self._scout_staleness, SCOUT_NOISE_CAP)
        root_depth = ag.root_depth_m(self._gdd_frac())

        for z, zone in enumerate(self._zones):
            base = z * ZONE_STRIDE
            taw = ag.total_available_water(zone.soil_depth_m, root_depth)
            obs[base + Z_DEPLETION] = np.clip(zone.depletion_mm / taw if taw > 0 else 0.0, 0.0, 1.0)
            obs[base + Z_CANOPY] = zone.canopy_cover
            n_noise = self.np_random.normal(0.0, noise_std)
            p_noise = self.np_random.normal(0.0, noise_std)
            w_noise = self.np_random.normal(0.0, noise_std)
            obs[base + Z_NITROGEN] = np.clip(zone.nitrogen_kg_ha / 120.0 + n_noise, 0.0, 1.0)
            obs[base + Z_PEST] = np.clip(zone.pest_pressure + p_noise, 0.0, 1.0)
            obs[base + Z_WEED] = np.clip(zone.weed_pressure + w_noise, 0.0, 1.0)

        d = min(self._day, self.horizon - 1)
        w = self._weather
        obs[B_DAY] = self._day / self.horizon
        obs[B_GDD] = self._gdd_frac()
        obs[B_STAGE] = self._stage_value()
        obs[B_WATER_STRESS] = np.clip(self._stress_accum / max(self._day, 1), 0.0, 1.0)
        obs[B_N_STRESS] = np.clip(self._n_stress_accum / max(self._day, 1), 0.0, 1.0)
        obs[B_RESERVOIR] = self._reservoir_m3 / RESERVOIR_CAPACITY_M3
        obs[B_CASH] = np.clip(self._cash / 600.0, -1.0, 1.0)
        obs[B_CREW] = self._crew_schedule[d] / 4.0
        obs[B_SPRAY_CLOCK] = min(self._days_since_spray / 21.0, 1.0)
        obs[B_N_CLOCK] = min(self._days_since_n / 30.0, 1.0)
        obs[B_LABOUR_DAYS] = min(self._labour_days_cum / 60.0, 1.0)
        obs[B_HARVESTED] = self._harvested_fraction

        forecast = w.forecast(d)
        obs[X_RAIN_F1] = np.clip(forecast[0] / 50.0, 0.0, 1.0)
        obs[X_RAIN_F2] = np.clip(forecast[1] / 50.0, 0.0, 1.0)
        obs[X_RAIN_F3] = np.clip(forecast[2] / 50.0, 0.0, 1.0)
        obs[X_ET0] = np.clip(w.et0_mm[d] / 8.0, 0.0, 1.0)
        obs[X_TMAX] = np.clip((w.t_max[d] - 21.5) / 13.5, -1.0, 1.0)
        obs[X_TMIN] = np.clip((w.t_min[d] - 21.5) / 13.5, -1.0, 1.0)
        obs[X_PRICE] = np.clip((w.price_rwf[d] - st.PRICE_MEAN_RWF) / st.PRICE_MEAN_RWF, -1.0, 1.0)
        obs[X_SEASON] = 0.0 if self._season == "A" else 1.0
        obs[X_RAIN_TODAY] = np.clip(w.rain_mm[d] / 50.0, 0.0, 1.0)
        obs[X_HUMIDITY] = w.humidity[d]
        obs[X_MARKET_WINDOW] = max(0.0, (CONTRACT_WINDOW_DAY - self._day) / self.horizon)

        return np.clip(obs, -1.0, 1.0).astype(np.float32)

    def _info(self) -> dict[str, Any]:
        info: dict[str, Any] = {
            "day": self._day,
            "season": self._season,
            "return": self._return,
            "cash_krwf": self._cash,
            "harvested_fraction": self._harvested_fraction,
            "labour_days": self._labour_days_cum,
        }
        if self._terminated or self._truncated:
            info["termination_cause"] = self._cause if self._terminated else "truncated"
        return info

    # ------------------------------------------------------------------
    # Ledger and rendering
    # ------------------------------------------------------------------

    def _log_action(self, action: Action, irrigation_mm: np.ndarray, revenue: float) -> None:
        if action == Action.IDLE or action == Action.SCOUT:
            return
        if irrigation_mm.sum() > 0:
            for z, mm in enumerate(irrigation_mm):
                if mm > 0:
                    self._ledger.append(FieldEvent(day=self._day, action="irrigate", zone=z, quantity=float(mm), unit="mm"))
        elif action == Action.APPLY_N_SPLIT:
            self._ledger.append(FieldEvent(day=self._day, action="apply_n", quantity=N_SPLIT_KG, unit="kg_ha"))
        elif action == Action.APPLY_K:
            self._ledger.append(FieldEvent(day=self._day, action="apply_k", quantity=K_APPLICATION_KG, unit="kg_ha"))
        elif action == Action.SPRAY_BIOPESTICIDE:
            self._ledger.append(FieldEvent(day=self._day, action="spray_biopesticide"))
        elif action in _WEEDING:
            self._ledger.append(FieldEvent(day=self._day, action="hire_weeding_crew"))
        elif action in (Action.HARVEST_PARTIAL, Action.HARVEST_ALL):
            self._ledger.append(FieldEvent(day=self._day, action="harvest", quantity=round(revenue, 2), unit="krwf"))

    @property
    def render_closed(self) -> bool:
        """True once the viewer window has been dismissed by the user."""
        return self._renderer is not None and self._renderer.closed

    def _render_state(self) -> dict[str, Any]:
        return {
            "block_id": self.block_id,
            "day": self._day,
            "horizon": self.horizon,
            "season": self._season,
            "stage": self._stage_value(),
            "gdd_fraction": self._gdd_frac(),
            "flowering": self._is_flowering(),
            "action": self._last_action.name,
            "reward": self._last_reward,
            "return": self._return,
            "cash_krwf": self._cash,
            "reservoir_fraction": self._reservoir_m3 / RESERVOIR_CAPACITY_M3,
            "labour_days": self._labour_days_cum,
            "days_since_spray": self._days_since_spray,
            "within_phi": self._days_since_spray < PRE_HARVEST_INTERVAL_DAYS,
            "harvested_fraction": self._harvested_fraction,
            "yield_forecast_kg": self._standing_yield_kg(),
            "water_stress": min(self._stress_accum / max(self._day, 1), 1.0),
            "price_rwf": self._price_today(),
            "rain_today_mm": float(self._weather.rain_mm[min(self._day, self.horizon - 1)]),
            "rain_forecast_mm": self._weather.forecast(min(self._day, self.horizon - 1)).tolist(),
            "zones": [
                {
                    "index": z.index,
                    "depletion_mm": z.depletion_mm,
                    # Depletion against this zone's own capacity. Absolute mm are
                    # not comparable between benches because a deep valley soil
                    # holds far more water than a thin ridge, so anything drawing
                    # a moisture bar or shading soil needs the fraction.
                    "depletion_frac": self._zone_depletion_frac(z),
                    "canopy_cover": z.canopy_cover,
                    "nitrogen_kg_ha": z.nitrogen_kg_ha,
                    "pest_pressure": z.pest_pressure,
                    "pest_damage": z.pest_damage,
                    "weed_pressure": z.weed_pressure,
                    "soil_depth_m": z.soil_depth_m,
                }
                for z in self._zones
            ],
        }

    def ledger_json(self, indent: int | None = 2) -> str:
        """Expose the hash-chained field record for the JSON API."""
        return self._ledger.to_json(indent=indent)


_IRRIGATE_LIGHT_ZONE = {
    Action.IRRIGATE_Z0_LIGHT: 0,
    Action.IRRIGATE_Z1_LIGHT: 1,
    Action.IRRIGATE_Z2_LIGHT: 2,
    Action.IRRIGATE_Z3_LIGHT: 3,
}
_IRRIGATE_HEAVY_ZONE = {
    Action.IRRIGATE_Z0_HEAVY: 0,
    Action.IRRIGATE_Z1_HEAVY: 1,
    Action.IRRIGATE_Z2_HEAVY: 2,
    Action.IRRIGATE_Z3_HEAVY: 3,
}
_WEEDING = (Action.HIRE_WEEDING_CREW, Action.HIRE_WEEDING_CREW_LARGE)
