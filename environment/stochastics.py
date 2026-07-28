"""Weather and market generation.

Everything derives from the episode seed and is generated for the whole season
at reset, so the forecast can be built by adding noise to known future values.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

LATITUDE_DEG = -1.94  # Kigali

# Rainfall: two-state Markov chain, gamma wet-day depths, per season
SEASON_PARAMS = {
    "A": {"p_ww": 0.62, "p_dw": 0.28, "gamma_shape": 0.90, "gamma_scale": 9.0},
    "B": {"p_ww": 0.68, "p_dw": 0.33, "gamma_shape": 0.95, "gamma_scale": 10.5},
}

TMAX_MEAN_C, TMAX_AMPLITUDE_C = 27.0, 2.0
TMIN_MEAN_C, TMIN_AMPLITUDE_C = 16.0, 1.5
TEMP_AR1 = 0.7

PRICE_THETA = 0.15
PRICE_SIGMA = 40.0  # RWF/kg
PRICE_MEAN_RWF = 620.0

FORECAST_NOISE_PER_DAY = 0.12
FORECAST_STATE_FLIP_PROB = 0.15


@dataclass
class SeasonWeather:
    """One full generated season."""

    season: str
    rain_mm: np.ndarray
    t_max: np.ndarray
    t_min: np.ndarray
    et0_mm: np.ndarray
    humidity: np.ndarray
    price_rwf: np.ndarray

    def forecast(self, day: int, rng: np.random.Generator) -> np.ndarray:
        """Noisy 3 day rainfall outlook. Day+3 is materially less reliable."""
        # TODO: scale noise by horizon, flip wet/dry state with
        # FORECAST_STATE_FLIP_PROB, clip at zero
        raise NotImplementedError


def extraterrestrial_radiation(day_of_year: int, latitude_deg: float = LATITUDE_DEG) -> float:
    """Ra in MJ/m2/day for the Hargreaves ET0 estimate."""
    # TODO: FAO-56 equations 21 to 25
    raise NotImplementedError


def hargreaves_et0(t_max: float, t_min: float, ra: float) -> float:
    """Temperature-only ET0. Avoids inventing wind and radiation series."""
    t_mean = 0.5 * (t_max + t_min)
    return float(0.0023 * ra * (t_mean + 17.8) * np.sqrt(max(t_max - t_min, 0.0)))


def generate_season(
    rng: np.random.Generator, season: str, n_days: int, start_doy: int
) -> SeasonWeather:
    """Generate rainfall, temperature, ET0, humidity and price for a season."""
    # TODO: Markov rainfall, AR(1) temperature, Hargreaves ET0, OU price
    raise NotImplementedError
