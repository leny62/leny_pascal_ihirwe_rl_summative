"""Soil water, canopy, nitrogen, pest and weed dynamics.

FAO-56 water balance, SCS runoff, AquaCrop canopy and biomass. Parameters are
literature values, not calibrated against Rwandan field trials, so absolute
yields are indicative. Sources are listed in the README.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# FAO-56 soil water
THETA_FC = 0.32
THETA_WP = 0.18
P_TABLE = 0.45
ROOT_DEPTH_MIN_M = 0.15
ROOT_DEPTH_MAX_M = 0.60

# Crop coefficient curve, French bean
KC_INI, KC_MID, KC_END = 0.50, 1.05, 0.90
STAGE_DAYS = (20, 30, 30, 15)

# Thermal time
T_BASE_C = 10.0
GDD_MATURITY = 1050.0

# Canopy, AquaCrop style
CC_0, CC_MAX, CGC = 0.02, 0.92, 0.0072
WP_STAR = 15.0  # g/m2
HARVEST_INDEX_0 = 0.45

# Nitrogen
N_MINERALISATION_MAX = 0.5  # kg/ha/day
N_FIXATION_MAX = 2.2  # kg/ha/day
N_FIXATION_SUPPRESS = 60.0  # kg/ha
N_CRITICAL = 45.0  # kg/ha
N_UPTAKE_FRACTION = 0.035
LEACH_EFFICIENCY = 0.7

# Pest and weed
PEST_GROWTH_RATE = 0.18
WEED_GROWTH_RATE = 0.11
SPRAY_KNOCKDOWN = 0.30
PEST_DAMAGE_RATE = 0.012

# Runoff, per zone from ridge (0) to valley bottom (3)
CURVE_NUMBER_BARE = np.array([86.0, 83.0, 80.0, 76.0])
CURVE_NUMBER_COVERED = np.array([72.0, 70.0, 68.0, 64.0])
RUNOFF_ROUTED_DOWNSLOPE = 0.60


@dataclass
class ZoneState:
    """One terrace bench."""

    soil_depth_m: float
    depletion_mm: float
    canopy_cover: float
    biomass_g_m2: float
    nitrogen_kg_ha: float
    pest_pressure: float
    weed_pressure: float
    pest_damage: float
    waterlogged_days: int = 0


def total_available_water(soil_depth_m: float, root_depth_m: float) -> float:
    """TAW in mm over the current root zone."""
    return 1000.0 * (THETA_FC - THETA_WP) * min(root_depth_m, soil_depth_m)


def depletion_fraction_p(et_c: float) -> float:
    """FAO-56 p adjusted for evaporative demand."""
    return float(np.clip(P_TABLE + 0.04 * (5.0 - et_c), 0.1, 0.8))


def water_stress_coefficient(depletion_mm: float, taw: float, raw: float) -> float:
    """Ks. 1.0 above RAW, falling linearly to 0 at TAW."""
    if depletion_mm <= raw:
        return 1.0
    return float(np.clip((taw - depletion_mm) / max(taw - raw, 1e-9), 0.0, 1.0))


def runoff_scs(rain_mm: float, curve_number: float) -> float:
    """SCS curve number runoff."""
    s = 25400.0 / curve_number - 254.0
    ia = 0.2 * s
    if rain_mm <= ia:
        return 0.0
    return float((rain_mm - ia) ** 2 / (rain_mm - ia + s))


def crop_coefficient(day: int) -> float:
    """Kc interpolated across the four FAO-56 stages."""
    # TODO: piecewise linear over STAGE_DAYS
    raise NotImplementedError


def growing_degree_days(t_max: float, t_min: float) -> float:
    return max(0.0, 0.5 * (t_max + t_min) - T_BASE_C)


def potential_canopy_cover(gdd_cum: float) -> float:
    """Logistic canopy expansion in thermal time."""
    # TODO: two-branch AquaCrop form, switching at CC_MAX / 2
    raise NotImplementedError


def nitrogen_fixation(gdd_fraction: float, nitrogen_kg_ha: float, ks: float) -> float:
    """Biological fixation, suppressed as mineral N rises."""
    # TODO: F_max * nodulation(gdd_fraction) * (1 - N/N_supp)+ * ks
    raise NotImplementedError


def step_zone(
    zone: ZoneState,
    *,
    rain_mm: float,
    run_on_mm: float,
    irrigation_mm: float,
    et0_mm: float,
    t_max: float,
    t_min: float,
    humidity: float,
    gdd_cum: float,
    day: int,
) -> tuple[ZoneState, dict[str, float]]:
    """Advance one zone by one day.

    Returns the new state and the fluxes the reward function needs: runoff,
    deep percolation, nitrogen leached, and Ks.
    """
    # TODO: order matters. Runoff, then water balance, then Ks, then canopy,
    # then biomass, then nitrogen, then pest and weed.
    raise NotImplementedError


def harvest_index(flowering_stress: float) -> float:
    """HI penalised by mean water stress during flowering only."""
    return HARVEST_INDEX_0 * (1.0 - 0.6 * float(np.clip(flowering_stress, 0.0, 1.0)))
