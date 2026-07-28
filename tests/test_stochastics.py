import numpy as np

from environment import stochastics as st


def make(season="A", seed=0, n=120):
    rng = np.random.default_rng(seed)
    return st.generate_season(rng, season, n)


def test_same_seed_reproduces_the_season():
    a = make(seed=7)
    b = make(seed=7)
    np.testing.assert_array_equal(a.rain_mm, b.rain_mm)
    np.testing.assert_array_equal(a.price_rwf, b.price_rwf)
    np.testing.assert_array_equal(a.rain_forecast, b.rain_forecast)


def test_different_seeds_diverge():
    assert not np.allclose(make(seed=1).rain_mm, make(seed=2).rain_mm)


def test_shapes_and_horizon():
    w = make(n=120)
    assert w.rain_mm.shape == (120,)
    assert w.rain_forecast.shape == (120, st.FORECAST_HORIZON)


def test_physical_ranges():
    w = make(n=120)
    assert (w.rain_mm >= 0).all()
    assert (w.t_max > w.t_min).all()
    assert (w.et0_mm > 0).all() and (w.et0_mm < 12).all()
    assert (w.humidity >= 0.2).all() and (w.humidity <= 1.0).all()
    assert (w.price_rwf >= st.PRICE_FLOOR_RWF).all()


def test_wetter_season_b_rains_more_on_average():
    # Averaged over seeds so a single draw does not decide it.
    a = np.mean([make("A", s).rain_mm.sum() for s in range(20)])
    b = np.mean([make("B", s).rain_mm.sum() for s in range(20)])
    assert b > a


def test_extraterrestrial_radiation_is_near_equatorial():
    vals = [st.extraterrestrial_radiation(d) for d in range(1, 366, 30)]
    assert all(30.0 < v < 42.0 for v in vals)


def test_forecast_matches_precomputed_row():
    w = make(seed=3)
    np.testing.assert_array_equal(w.forecast(10), w.rain_forecast[10])
