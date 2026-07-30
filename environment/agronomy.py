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


AERATION_PENALTY = 0.15  # per-day canopy loss once waterlogged
WATERLOG_DAYS_TRIGGER = 3
SENESCENCE_START_FRAC = 0.85
ZONE_AREA_M2 = 2500.0  # one of four terraces on a 1 ha block


@dataclass
class ZoneState:
    """One terrace bench. `index` runs 0 (ridge) to 3 (valley bottom)."""

    index: int
    soil_depth_m: float
    depletion_mm: float
    canopy_cover: float
    biomass_g_m2: float
    nitrogen_kg_ha: float
    pest_pressure: float
    weed_pressure: float
    pest_damage: float
    waterlogged_days: int = 0


def root_depth_m(gdd_frac: float) -> float:
    """Rooting depth, growing linearly to its maximum by mid season."""
    return ROOT_DEPTH_MIN_M + (ROOT_DEPTH_MAX_M - ROOT_DEPTH_MIN_M) * min(gdd_frac / 0.5, 1.0)


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
    """Kc interpolated across the four FAO-56 stages by calendar day."""
    d_ini, d_dev, d_mid, d_late = STAGE_DAYS
    if day < d_ini:
        return KC_INI
    if day < d_ini + d_dev:
        return KC_INI + (KC_MID - KC_INI) * (day - d_ini) / d_dev
    if day < d_ini + d_dev + d_mid:
        return KC_MID
    if day < d_ini + d_dev + d_mid + d_late:
        return KC_MID + (KC_END - KC_MID) * (day - d_ini - d_dev - d_mid) / d_late
    return KC_END


def growing_degree_days(t_max: float, t_min: float) -> float:
    return max(0.0, 0.5 * (t_max + t_min) - T_BASE_C)


def potential_canopy_cover(gdd_cum: float) -> float:
    """Logistic canopy expansion in thermal time, AquaCrop two-branch form."""
    cc_half = CC_MAX / 2.0
    cc = CC_0 * np.exp(gdd_cum * CGC)
    if cc > cc_half:
        cc = CC_MAX - 0.25 * (CC_MAX**2 / CC_0) * np.exp(-gdd_cum * CGC)
    return float(np.clip(cc, 0.0, CC_MAX))


def nitrogen_stress(nitrogen_kg_ha: float) -> float:
    """Kn, the nitrogen analogue of Ks. Floors at 0.4 so N never fully halts growth."""
    return float(np.clip(nitrogen_kg_ha / N_CRITICAL, 0.4, 1.0))


def nodulation(gdd_fraction: float) -> float:
    """Ramps from 0 at GDD fraction 0.12 to 1 at 0.30, then holds."""
    return float(np.clip((gdd_fraction - 0.12) / (0.30 - 0.12), 0.0, 1.0))


def nitrogen_fixation(gdd_fraction: float, nitrogen_kg_ha: float, ks: float) -> float:
    """Biological fixation, suppressed as mineral N approaches N_FIXATION_SUPPRESS.

    The suppression term is why a heavy urea dose is doubly wasteful: it leaches
    and it switches off the free nitrogen the crop would have fixed itself.
    """
    suppression = max(0.0, 1.0 - nitrogen_kg_ha / N_FIXATION_SUPPRESS)
    return N_FIXATION_MAX * nodulation(gdd_fraction) * suppression * ks


def cascade_runoff(runoffs: np.ndarray) -> tuple[np.ndarray, float]:
    """Route per-zone runoff downslope. A fraction reaches the next bench as
    run-on, the rest leaves the block; the valley bottom's runoff all leaves.

    Returns the run-on delivered to each zone and the total leaving the block.
    Conserves water: sum(runoffs) == sum(run_on) + left_block.
    """
    n = len(runoffs)
    run_on = np.zeros(n)
    left = 0.0
    for z in range(n):
        routed = RUNOFF_ROUTED_DOWNSLOPE * runoffs[z]
        if z + 1 < n:
            run_on[z + 1] = routed
            left += runoffs[z] - routed
        else:
            left += runoffs[z]
    return run_on, float(left)


def effective_curve_number(zone: ZoneState) -> float:
    """CN falls as the canopy closes and intercepts rainfall."""
    bare = CURVE_NUMBER_BARE[zone.index]
    covered = CURVE_NUMBER_COVERED[zone.index]
    return float(bare + (covered - bare) * zone.canopy_cover)


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

    Order matters: runoff, water balance, Ks, canopy, biomass, nitrogen, pests.
    Returns the new state and the fluxes the reward needs.
    """
    gdd_frac = min(gdd_cum / GDD_MATURITY, 1.0)
    root_depth = root_depth_m(gdd_frac)
    taw = total_available_water(zone.soil_depth_m, root_depth)

    # 1. Runoff from rainfall only. Run-on infiltrates, it does not re-run off.
    runoff = runoff_scs(rain_mm, effective_curve_number(zone))
    infiltration = (rain_mm - runoff) + run_on_mm + irrigation_mm

    # 2. Ks from depletion at the start of the day, then crop ET.
    kc = crop_coefficient(day)
    raw = depletion_fraction_p(kc * et0_mm) * taw
    ks = water_stress_coefficient(zone.depletion_mm, taw, raw)
    etc = ks * kc * et0_mm

    # 3. Water balance. Deep percolation is whatever pushes past field capacity.
    dr_provisional = zone.depletion_mm - infiltration + etc
    deep_percolation = max(0.0, -dr_provisional)
    depletion = min(max(dr_provisional + deep_percolation, 0.0), taw)

    waterlogged = zone.waterlogged_days + 1 if depletion < 0.02 * taw else 0
    aeration = AERATION_PENALTY if waterlogged >= WATERLOG_DAYS_TRIGGER else 0.0

    # 4. Canopy: potential from thermal time, cut by every stress, then senescence.
    kn = nitrogen_stress(zone.nitrogen_kg_ha)
    cc = (
        potential_canopy_cover(gdd_cum)
        * ks**0.5
        * kn
        * (1.0 - 0.5 * zone.weed_pressure)
        * (1.0 - zone.pest_damage)
        * (1.0 - aeration)
    )
    if gdd_frac > SENESCENCE_START_FRAC:
        senescence = (gdd_frac - SENESCENCE_START_FRAC) / (1.0 - SENESCENCE_START_FRAC)
        cc *= 1.0 - 0.6 * senescence
    canopy = float(np.clip(cc, 0.0, CC_MAX))

    # 5. Biomass via normalised water productivity.
    transpiration = ks * kc * et0_mm * canopy
    biomass_gain = WP_STAR * (transpiration / et0_mm) if et0_mm > 1e-6 else 0.0
    biomass = zone.biomass_g_m2 + biomass_gain

    # 6. Nitrogen pool. Weeds steal some uptake, high mineral N suppresses fixation.
    f_temp = float(np.clip((0.5 * (t_max + t_min) - 5.0) / 20.0, 0.0, 1.0))
    f_moist = 1.0 - depletion / taw if taw > 0 else 0.0
    mineralisation = N_MINERALISATION_MAX * f_temp * max(f_moist, 0.0)
    fixation = nitrogen_fixation(gdd_frac, zone.nitrogen_kg_ha, ks)
    uptake = N_UPTAKE_FRACTION * biomass_gain * (1.0 + 0.4 * zone.weed_pressure)
    leached = zone.nitrogen_kg_ha * (deep_percolation / (deep_percolation + taw)) * LEACH_EFFICIENCY
    nitrogen = max(zone.nitrogen_kg_ha + mineralisation + fixation - uptake - leached, 0.0)

    # 7. Pest and weed pressure, both logistic and weather or canopy gated.
    f_hum = float(np.clip((humidity - 0.4) / 0.5, 0.0, 1.0))
    f_temp_pest = float(np.clip((0.5 * (t_max + t_min) - 12.0) / 15.0, 0.0, 1.0))
    pest = zone.pest_pressure + PEST_GROWTH_RATE * zone.pest_pressure * (1.0 - zone.pest_pressure) * f_hum * f_temp_pest
    pest = float(np.clip(pest, 0.0, 1.0))
    pest_damage = float(np.clip(zone.pest_damage + PEST_DAMAGE_RATE * zone.pest_pressure, 0.0, 1.0))
    weed = zone.weed_pressure + WEED_GROWTH_RATE * zone.weed_pressure * (1.0 - zone.weed_pressure) * (1.0 - canopy)
    weed = float(np.clip(weed, 0.0, 1.0))

    new_zone = ZoneState(
        index=zone.index,
        soil_depth_m=zone.soil_depth_m,
        depletion_mm=depletion,
        canopy_cover=canopy,
        biomass_g_m2=biomass,
        nitrogen_kg_ha=nitrogen,
        pest_pressure=pest,
        weed_pressure=weed,
        pest_damage=pest_damage,
        waterlogged_days=waterlogged,
    )
    fluxes = {
        "runoff_mm": runoff,
        "infiltration_mm": infiltration,
        "deep_percolation_mm": deep_percolation,
        "etc_mm": etc,
        "ks": ks,
        "nitrogen_leached_kg": leached,
        "biomass_gain_g_m2": biomass_gain,
    }
    return new_zone, fluxes


def harvest_index(flowering_stress: float) -> float:
    """HI penalised by mean water stress during flowering only."""
    return HARVEST_INDEX_0 * (1.0 - 0.6 * float(np.clip(flowering_stress, 0.0, 1.0)))
