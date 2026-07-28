"""Weather and market generation.

Everything derives from the episode seed and is generated for the whole season
at reset, so the forecast can be built by adding noise to known future values.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

LATITUDE_DEG = -1.94  # Kigali
SOLAR_CONSTANT = 0.0820  # MJ m-2 min-1
RA_MJ_TO_MM = 0.408  # MJ m-2 day-1 to mm/day equivalent evaporation (FAO-56 eq 20)

# Season start as day of year. A runs Sep to Jan, B runs Feb to Jun.
SEASON_START_DOY = {"A": 244, "B": 32}

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
PRICE_FLOOR_RWF = 150.0
HARVEST_GLUT_DAY = 95
HARVEST_GLUT_WIDTH = 15.0
HARVEST_GLUT_DEPTH = 0.18

FORECAST_NOISE_PER_DAY = 0.12
FORECAST_STATE_FLIP_PROB = 0.15
FORECAST_HORIZON = 3


@dataclass
class SeasonWeather:
    """One full generated season. Arrays are indexed by day, 0 to n_days-1."""

    season: str
    rain_mm: np.ndarray
    t_max: np.ndarray
    t_min: np.ndarray
    et0_mm: np.ndarray
    humidity: np.ndarray
    price_rwf: np.ndarray
    rain_forecast: np.ndarray  # (n_days, FORECAST_HORIZON), the outlook seen each day

    def forecast(self, day: int) -> np.ndarray:
        """The 3 day rainfall outlook seen on `day`. Baked in at generation, so
        it is stable across calls and reproducible under a fixed seed."""
        return self.rain_forecast[day]


def extraterrestrial_radiation(day_of_year: int, latitude_deg: float = LATITUDE_DEG) -> float:
    """Ra in MJ/m2/day, FAO-56 equations 21 to 25. Near the equator this barely
    moves across the year, which is correct for Rwanda."""
    phi = np.radians(latitude_deg)
    dr = 1.0 + 0.033 * np.cos(2.0 * np.pi / 365.0 * day_of_year)
    delta = 0.409 * np.sin(2.0 * np.pi / 365.0 * day_of_year - 1.39)
    ws = np.arccos(np.clip(-np.tan(phi) * np.tan(delta), -1.0, 1.0))
    ra = (
        (24.0 * 60.0 / np.pi)
        * SOLAR_CONSTANT
        * dr
        * (ws * np.sin(phi) * np.sin(delta) + np.cos(phi) * np.cos(delta) * np.sin(ws))
    )
    return float(ra)


def hargreaves_et0(t_max: float, t_min: float, ra: float) -> float:
    """Temperature-only ET0 in mm/day. `ra` is in MJ/m2/day and is converted to
    mm/day here, per FAO-56. Avoids inventing wind and radiation series."""
    t_mean = 0.5 * (t_max + t_min)
    ra_mm = RA_MJ_TO_MM * ra
    return float(0.0023 * ra_mm * (t_mean + 17.8) * np.sqrt(max(t_max - t_min, 0.0)))


def _generate_rainfall(rng: np.random.Generator, params: dict, n_days: int) -> np.ndarray:
    rain = np.zeros(n_days)
    wet = rng.random() < 0.4
    for t in range(n_days):
        p = params["p_ww"] if wet else params["p_dw"]
        wet = rng.random() < p
        if wet:
            rain[t] = rng.gamma(params["gamma_shape"], params["gamma_scale"])
    return rain


def _generate_temperature(
    rng: np.random.Generator, doy: np.ndarray, rain: np.ndarray, n_days: int
) -> tuple[np.ndarray, np.ndarray]:
    # Seasonal sinusoid, phase so the warm peak sits near the dry months.
    phase = np.sin(2.0 * np.pi * (doy - 80) / 365.0)
    tmax_mean = TMAX_MEAN_C + TMAX_AMPLITUDE_C * phase
    tmin_mean = TMIN_MEAN_C + TMIN_AMPLITUDE_C * phase

    t_max = np.zeros(n_days)
    t_min = np.zeros(n_days)
    dev_max = dev_min = 0.0
    for t in range(n_days):
        dev_max = TEMP_AR1 * dev_max + rng.normal(0.0, 1.5)
        dev_min = TEMP_AR1 * dev_min + rng.normal(0.0, 1.0)
        wet = rain[t] > 0.0
        # Cloud cover caps the daytime high and lifts the overnight low.
        t_max[t] = tmax_mean[t] + dev_max - (2.0 if wet else 0.0)
        t_min[t] = tmin_mean[t] + dev_min + (1.0 if wet else 0.0)
    t_min = np.minimum(t_min, t_max - 1.0)
    return t_max, t_min


def _generate_humidity(rng: np.random.Generator, rain: np.ndarray, n_days: int) -> np.ndarray:
    humidity = np.zeros(n_days)
    h = 0.6
    for t in range(n_days):
        target = 0.85 if rain[t] > 0.0 else 0.55
        h = 0.6 * h + 0.4 * target + rng.normal(0.0, 0.03)
        humidity[t] = np.clip(h, 0.2, 1.0)
    return humidity


def _generate_price(rng: np.random.Generator, n_days: int) -> np.ndarray:
    days = np.arange(n_days)
    glut = np.exp(-((days - HARVEST_GLUT_DAY) ** 2) / (2.0 * HARVEST_GLUT_WIDTH**2))
    mu = PRICE_MEAN_RWF * (1.0 - HARVEST_GLUT_DEPTH * glut)
    price = np.zeros(n_days)
    p = PRICE_MEAN_RWF * np.exp(rng.normal(0.0, 0.1))
    for t in range(n_days):
        p = p + PRICE_THETA * (mu[t] - p) + rng.normal(0.0, PRICE_SIGMA)
        price[t] = max(p, PRICE_FLOOR_RWF)
    return price


def _generate_forecast(
    rng: np.random.Generator, rain: np.ndarray, params: dict, n_days: int
) -> np.ndarray:
    wet_depth = params["gamma_shape"] * params["gamma_scale"]
    forecast = np.zeros((n_days, FORECAST_HORIZON))
    for t in range(n_days):
        for i, h in enumerate(range(1, FORECAST_HORIZON + 1)):
            fut = t + h
            true_val = rain[fut] if fut < n_days else 0.0
            reported_wet = true_val > 0.0
            if rng.random() < FORECAST_STATE_FLIP_PROB:
                reported_wet = not reported_wet
            if reported_wet:
                base = true_val if true_val > 0.0 else wet_depth
                forecast[t, i] = max(base * (1.0 + rng.normal(0.0, FORECAST_NOISE_PER_DAY * h)), 0.0)
    return forecast


def generate_season(
    rng: np.random.Generator, season: str, n_days: int, start_doy: int | None = None
) -> SeasonWeather:
    """Generate rainfall, temperature, ET0, humidity, price and forecast."""
    params = SEASON_PARAMS[season]
    if start_doy is None:
        start_doy = SEASON_START_DOY[season]
    doy = (start_doy + np.arange(n_days)) % 365

    rain = _generate_rainfall(rng, params, n_days)
    t_max, t_min = _generate_temperature(rng, doy, rain, n_days)
    et0 = np.array([hargreaves_et0(t_max[t], t_min[t], extraterrestrial_radiation(int(doy[t]))) for t in range(n_days)])
    humidity = _generate_humidity(rng, rain, n_days)
    price = _generate_price(rng, n_days)
    forecast = _generate_forecast(rng, rain, params, n_days)

    return SeasonWeather(season, rain, t_max, t_min, et0, humidity, price, forecast)
