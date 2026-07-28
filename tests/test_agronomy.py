"""Agronomy checks.

The water balance test is the important one. A leaking balance makes every
downstream result meaningless and the failure is invisible in reward curves.
"""

import numpy as np
import pytest

from environment import agronomy as ag


def test_taw_scales_with_root_depth():
    shallow = ag.total_available_water(soil_depth_m=1.0, root_depth_m=0.20)
    deep = ag.total_available_water(soil_depth_m=1.0, root_depth_m=0.60)
    assert deep > shallow
    assert deep == pytest.approx(1000 * (ag.THETA_FC - ag.THETA_WP) * 0.60)


def test_taw_is_capped_by_soil_depth():
    """Roots cannot draw water from below a shallow terrace."""
    assert ag.total_available_water(0.30, 0.60) == pytest.approx(
        ag.total_available_water(0.30, 0.30)
    )


def test_ks_is_one_above_raw():
    taw, raw = 100.0, 45.0
    assert ag.water_stress_coefficient(0.0, taw, raw) == 1.0
    assert ag.water_stress_coefficient(raw, taw, raw) == 1.0


def test_ks_falls_monotonically_below_raw():
    taw, raw = 100.0, 45.0
    depletions = np.linspace(raw, taw, 20)
    values = [ag.water_stress_coefficient(d, taw, raw) for d in depletions]
    assert all(later <= earlier for earlier, later in zip(values, values[1:], strict=False))
    assert values[-1] == pytest.approx(0.0)


def test_no_runoff_below_initial_abstraction():
    assert ag.runoff_scs(1.0, curve_number=82.0) == 0.0


def test_runoff_rises_with_rainfall_and_curve_number():
    assert ag.runoff_scs(40.0, 82.0) > ag.runoff_scs(20.0, 82.0)
    assert ag.runoff_scs(40.0, 86.0) > ag.runoff_scs(40.0, 76.0)


def test_growing_degree_days_floor_at_zero():
    assert ag.growing_degree_days(t_max=12.0, t_min=4.0) == 0.0
    assert ag.growing_degree_days(t_max=28.0, t_min=16.0) == pytest.approx(12.0)


def test_harvest_index_penalised_by_flowering_stress():
    assert ag.harvest_index(0.0) == pytest.approx(ag.HARVEST_INDEX_0)
    assert ag.harvest_index(1.0) < ag.harvest_index(0.5) < ag.harvest_index(0.0)


def a_zone(**overrides) -> ag.ZoneState:
    base = dict(
        index=1,
        soil_depth_m=1.0,
        depletion_mm=30.0,
        canopy_cover=0.4,
        biomass_g_m2=100.0,
        nitrogen_kg_ha=40.0,
        pest_pressure=0.1,
        weed_pressure=0.1,
        pest_damage=0.0,
    )
    base.update(overrides)
    return ag.ZoneState(**base)


def test_water_balance_closes():
    """Storage change equals inputs minus outputs, to under 1e-6 mm."""
    zone = a_zone(depletion_mm=25.0)
    new, f = ag.step_zone(
        zone, rain_mm=8.0, run_on_mm=3.0, irrigation_mm=10.0, et0_mm=4.5,
        t_max=27.0, t_min=16.0, humidity=0.6, gdd_cum=300.0, day=40,
    )
    storage_change = zone.depletion_mm - new.depletion_mm
    balance = f["infiltration_mm"] - f["etc_mm"] - f["deep_percolation_mm"]
    assert abs(storage_change - balance) < 1e-6


def test_runoff_cascade_conserves_water():
    runoffs = np.array([12.0, 8.0, 5.0, 3.0])
    run_on, left = ag.cascade_runoff(runoffs)
    assert run_on[0] == 0.0
    assert abs(runoffs.sum() - (run_on.sum() + left)) < 1e-9
    # Each upper zone routes exactly the downslope fraction to the bench below.
    for z in range(len(runoffs) - 1):
        assert run_on[z + 1] == pytest.approx(ag.RUNOFF_ROUTED_DOWNSLOPE * runoffs[z])


def test_fixation_shuts_off_as_mineral_nitrogen_rises():
    """The reason a heavy urea dose is doubly wasteful."""
    low = ag.nitrogen_fixation(0.5, nitrogen_kg_ha=5.0, ks=1.0)
    high = ag.nitrogen_fixation(0.5, nitrogen_kg_ha=55.0, ks=1.0)
    assert low > high
    assert ag.nitrogen_fixation(0.5, ag.N_FIXATION_SUPPRESS, ks=1.0) == pytest.approx(0.0)
    # No nodulation before the crop establishes root nodules.
    assert ag.nitrogen_fixation(0.05, nitrogen_kg_ha=5.0, ks=1.0) == 0.0


def test_canopy_cover_stays_within_bounds():
    zone = a_zone(canopy_cover=0.02, biomass_g_m2=0.0)
    gdd = 0.0
    for day in range(120):
        gdd += ag.growing_degree_days(27.0, 16.0)
        zone, _ = ag.step_zone(
            zone, rain_mm=5.0, run_on_mm=0.0, irrigation_mm=6.0, et0_mm=4.0,
            t_max=27.0, t_min=16.0, humidity=0.6, gdd_cum=gdd, day=day,
        )
        assert 0.0 <= zone.canopy_cover <= ag.CC_MAX


def test_deep_percolation_only_when_water_exceeds_demand():
    """A dry soil taking less water than it evaporates should not percolate."""
    zone = a_zone(depletion_mm=60.0)
    _, f = ag.step_zone(
        zone, rain_mm=1.0, run_on_mm=0.0, irrigation_mm=0.0, et0_mm=5.0,
        t_max=28.0, t_min=15.0, humidity=0.5, gdd_cum=200.0, day=30,
    )
    assert f["deep_percolation_mm"] == 0.0
