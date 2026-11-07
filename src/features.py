"""Feature engineering for predicting tomorrow's weather.

All features use only information available at the end of day t (the current day and the past),
so they can also be computed in streaming mode, day by day, without leaking information from the future.

Targets:
- rain_tomorrow          — binary: precipitation ≥ 1 mm on day t+1;
- weather_type_tomorrow  — 3 classes: 0 favourable, 1 unfavourable, 2 critical (on day t+1).
"""
import numpy as np
import pandas as pd

RAIN_MM = 1.0
TARGET = "rain_tomorrow"
TARGET3 = "weather_type_tomorrow"
# Display names of the weather types (used in figures and tables)
WEATHER_TYPES = ["Благоприятная", "Неблагоприятная", "Критическая"]

# Criteria for the weather type of a day (adapted from common weather-warning thresholds)
CRITICAL = {"precip_mm": 10, "gusts_kmh": 60, "tmax_c": 35, "tmin_c": -15, "snow_cm": 5}
UNFAVORABLE = {"precip_mm": RAIN_MM, "gusts_kmh": 45, "tmax_c": 30, "tmin_c": -10}

# Russian display names of the features (figures, tables, app)
FEATURE_NAMES_RU = {
    "temperature_2m_mean": "Средняя температура, °C", "temp_range": "Суточная амплитуда, °C",
    "dew_point_2m_mean": "Точка росы, °C", "dew_point_depression": "Дефицит точки росы, °C",
    "relative_humidity_2m_mean": "Средняя влажность, %", "relative_humidity_2m_max": "Максимальная влажность, %",
    "pressure_msl_mean": "Давление, гПа", "pressure_intraday_range": "Колебание давления за день, гПа",
    "cloud_cover_mean": "Облачность, %", "sunshine_ratio": "Доля солнечного сияния",
    "precipitation_sum": "Осадки сегодня, мм", "precipitation_hours": "Часы с осадками",
    "snowfall_sum": "Снегопад, см", "wind_speed_10m_max": "Скорость ветра, км/ч",
    "wind_dir_sin": "Направление ветра (sin)", "wind_dir_cos": "Направление ветра (cos)",
    "soil_moisture_0_to_7cm_mean": "Влажность почвы", "vapour_pressure_deficit_max": "Дефицит упругости пара, кПа",
    "pressure_change_1d": "Изменение давления за сутки, гПа", "pressure_change_3d": "Изменение давления за 3 дня, гПа",
    "cloud_cover_lag1": "Облачность вчера, %", "humidity_change_1d": "Изменение влажности за сутки",
    "precip_sum_3d": "Осадки за 3 дня, мм", "precip_days_7d": "Дождливых дней за 7",
    "doy_sin": "День года (sin)", "doy_cos": "День года (cos)",
}

FEATURES = [
    # today's state
    "temperature_2m_mean", "temp_range", "dew_point_2m_mean", "dew_point_depression",
    "relative_humidity_2m_mean", "relative_humidity_2m_max",
    "pressure_msl_mean", "pressure_intraday_range",
    "cloud_cover_mean", "sunshine_ratio",
    "precipitation_sum", "precipitation_hours", "snowfall_sum",
    "wind_speed_10m_max", "wind_dir_sin", "wind_dir_cos",
    "soil_moisture_0_to_7cm_mean", "vapour_pressure_deficit_max",
    # dynamics (trends and memory)
    "pressure_change_1d", "pressure_change_3d",
    "cloud_cover_lag1", "humidity_change_1d",
    "precip_sum_3d", "precip_days_7d",
    # season (cyclic encoding)
    "doy_sin", "doy_cos",
]


def weather_type(raw: pd.DataFrame) -> pd.Series:
    """Weather type of each day: 0 favourable, 1 unfavourable, 2 critical."""
    critical = ((raw.precipitation_sum >= CRITICAL["precip_mm"]) | (raw.wind_gusts_10m_max >= CRITICAL["gusts_kmh"])
                | (raw.temperature_2m_max >= CRITICAL["tmax_c"]) | (raw.temperature_2m_min <= CRITICAL["tmin_c"])
                | (raw.snowfall_sum >= CRITICAL["snow_cm"]))
    unfavorable = ((raw.precipitation_sum >= UNFAVORABLE["precip_mm"]) | (raw.wind_gusts_10m_max >= UNFAVORABLE["gusts_kmh"])
                   | (raw.temperature_2m_max >= UNFAVORABLE["tmax_c"]) | (raw.temperature_2m_min <= UNFAVORABLE["tmin_c"]))
    return pd.Series(np.select([critical, unfavorable], [2, 1], default=0), index=raw.index)


def compute_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Features for every day (no targets). The first days contain NaN because of the lags."""
    df = raw.sort_values("date").reset_index(drop=True)
    out = pd.DataFrame({"date": df["date"]})

    for col in ["temperature_2m_mean", "dew_point_2m_mean", "relative_humidity_2m_mean", "relative_humidity_2m_max",
                "pressure_msl_mean", "cloud_cover_mean", "precipitation_sum", "precipitation_hours", "snowfall_sum",
                "wind_speed_10m_max", "soil_moisture_0_to_7cm_mean", "vapour_pressure_deficit_max"]:
        out[col] = df[col]

    out["temp_range"] = df.temperature_2m_max - df.temperature_2m_min
    out["dew_point_depression"] = df.temperature_2m_mean - df.dew_point_2m_mean
    out["pressure_intraday_range"] = df.pressure_msl_max - df.pressure_msl_min
    out["sunshine_ratio"] = df.sunshine_duration / df.daylight_duration

    theta = np.deg2rad(df.wind_direction_10m_dominant)
    out["wind_dir_sin"], out["wind_dir_cos"] = np.sin(theta), np.cos(theta)

    out["pressure_change_1d"] = df.pressure_msl_mean.diff(1)
    out["pressure_change_3d"] = df.pressure_msl_mean.diff(3)
    out["cloud_cover_lag1"] = df.cloud_cover_mean.shift(1)
    out["humidity_change_1d"] = df.relative_humidity_2m_mean.diff(1)
    out["precip_sum_3d"] = df.precipitation_sum.rolling(3).sum()
    out["precip_days_7d"] = (df.precipitation_sum >= RAIN_MM).rolling(7).sum()

    doy = df.date.dt.dayofyear
    out["doy_sin"], out["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    return out


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Features + targets (for training/evaluation). Drops days without full history or without a "tomorrow"."""
    df = raw.sort_values("date").reset_index(drop=True)
    out = compute_features(df)
    tomorrow_missing = df.precipitation_sum.shift(-1).isna()
    out[TARGET] = (df.precipitation_sum.shift(-1) >= RAIN_MM).astype(float).mask(tomorrow_missing)
    out[TARGET3] = weather_type(df).shift(-1).astype(float).mask(tomorrow_missing)
    return out.dropna().astype({TARGET: int, TARGET3: int}).reset_index(drop=True)
