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


@pytest.mark.skip(reason="needs step_zone")
def test_water_balance_closes():
    """Inputs minus outputs minus storage change under 1e-6 mm per step."""
    raise NotImplementedError


@pytest.mark.skip(reason="needs step_zone")
def test_runoff_cascade_conserves_water():
    """What leaves zone z arrives at z+1 or leaves the block, never both."""
    raise NotImplementedError


@pytest.mark.skip(reason="needs nitrogen_fixation")
def test_fixation_shuts_off_as_mineral_nitrogen_rises():
    """The reason a heavy urea dose is doubly wasteful."""
    raise NotImplementedError


@pytest.mark.skip(reason="needs potential_canopy_cover")
def test_canopy_cover_stays_within_bounds():
    raise NotImplementedError
