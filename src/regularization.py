"""Regularization in streaming (lecture 4): L1 / L2 / Elastic Net logistic regression on the weather features.

Shared by notebook 09 and the Streamlit app:
- make_model           — StandardScaler + LogisticRegression; L1/L2/ENet are set with l1_ratio (1 / 0 / in between),
                         the `penalty` argument is deprecated since scikit-learn 1.8
- fit_timed, predict_latency_ms — latency measurements (training of a batch, forecast of one record)
- add_sensor_noise     — Gaussian noise on selected raw sensors in a time window (a faulty sensor)
- noise_index          — "how jumpy are the sensors now" relative to the same calendar days of the training years
- stream_retrain       — the stream 2024-2025: periodic retraining (micro-batches) with a policy that picks C and alpha
"""
from __future__ import annotations

import time
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from features import FEATURES, TARGET, build_features

METHODS = {"L1 (Lasso)": 1.0, "L2 (Ridge)": 0.0, "Elastic Net": 0.5}
METHOD_COLORS = {"L1 (Lasso)": "#C0392B", "L2 (Ridge)": "#2B6CB0", "Elastic Net": "#8E44AD"}
NOISY_SENSORS = ["relative_humidity_2m_mean", "cloud_cover_mean", "pressure_msl_mean", "dew_point_2m_mean"]
SENSOR_NAMES_RU = {"relative_humidity_2m_mean": "влажность", "cloud_cover_mean": "облачность",
                   "pressure_msl_mean": "давление", "dew_point_2m_mean": "точка росы"}
TRAIN_END = "2023-01-01"


def make_model(C: float = 1.0, l1_ratio: float = 0.5, max_iter: int = 5000) -> Pipeline:
    """Scaling is required: the penalty treats all coefficients alike, so features must be on one scale."""
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, l1_ratio=l1_ratio, solver="saga", max_iter=max_iter,
                                                              class_weight="balanced", random_state=0))


def coefficients(model: Pipeline) -> np.ndarray:
    return model[-1].coef_[0] if model[-1].coef_.shape[0] == 1 else model[-1].coef_


def fit_timed(model, X, y, repeats: int = 1) -> tuple[object, float]:
    """Fits the model and returns it with the median training time, ms."""
    times = []
    for _ in range(repeats):
        t = time.perf_counter()
        model.fit(X, y)
        times.append((time.perf_counter() - t) * 1000)
    return model, float(np.median(times))


def predict_latency_ms(model, X: pd.DataFrame, n: int = 300) -> float:
    """Median time to forecast ONE record (what a streaming system pays for every incoming observation)."""
    rows = [X.iloc[[i % len(X)]] for i in range(n)]
    times = []
    for r in rows:
        t = time.perf_counter()
        model.predict_proba(r)
        times.append((time.perf_counter() - t) * 1000)
    return float(np.median(times))


def add_sensor_noise(raw: pd.DataFrame, sensors=NOISY_SENSORS, sigma: float = 0.5, start: str = "2024-01-01",
                     end: str = "2099-12-31", seed: int = 0) -> pd.DataFrame:
    """Gaussian noise with std = sigma × (std of the sensor in the training years), only inside [start, end]."""
    out = raw.copy()
    mask = (out.date >= start) & (out.date <= end)
    rng = np.random.default_rng(seed)
    for c in sensors:
        out.loc[mask, c] = out.loc[mask, c] + rng.normal(0, sigma * raw.loc[raw.date < TRAIN_END, c].std(), mask.sum())
    if "relative_humidity_2m_mean" in sensors:
        out["relative_humidity_2m_mean"] = out["relative_humidity_2m_mean"].clip(0, 100)
    if "cloud_cover_mean" in sensors:
        out["cloud_cover_mean"] = out["cloud_cover_mean"].clip(0, 100)
    return out


def noise_index(raw: pd.DataFrame, start, days: int = 14, sensors=NOISY_SENSORS) -> float:
    """Median |day-to-day jump| of the sensors over the last `days` days divided by the typical jump on the same
    calendar days of the training years (seasonal reference: in winter the weather itself is jumpier).
    ≈ 1 — usual behaviour, > 1.5 — the sensors are noisier than usual."""
    start = pd.Timestamp(start)
    win = raw[(raw.date < start) & (raw.date >= start - pd.Timedelta(days=days + 1))]
    hist = raw[raw.date < TRAIN_END]
    same = hist[hist.date.dt.dayofyear.isin(win.date.dt.dayofyear)]
    ratios = []
    for c in sensors:
        ref = same.groupby(same.date.dt.year)[c].apply(lambda s: s.diff().abs().median()).median()
        ratios.append(win[c].diff().abs().median() / ref)
    return float(np.median(ratios))


def stream_retrain(raw: pd.DataFrame, policy: Callable[[float], tuple[float, float]], freq: str = "M",
                   start: str = "2024-01", end: str = "2025-12", data: pd.DataFrame | None = None):
    """Streaming with micro-batch retraining: at the start of every period the policy looks at the noise index and
    returns (C, l1_ratio); the model is retrained on the whole history and forecasts the period day by day.
    Returns (daily forecasts, log of periods with coefficients and training time)."""
    data = build_features(raw) if data is None else data
    preds, log = [], []
    for per in pd.period_range(start, end, freq=freq):
        t0 = per.start_time
        hist = data[data.date < t0]
        ni = noise_index(raw, t0)
        C, a = policy(ni)
        model, fit_ms = fit_timed(make_model(C, a), hist[FEATURES], hist[TARGET])
        cur = data[(data.date >= t0) & (data.date <= per.end_time)]
        if cur.empty:
            continue
        preds.append(pd.DataFrame({"date": cur.date.to_numpy(), "y": cur[TARGET].to_numpy(),
                                   "p": model.predict_proba(cur[FEATURES])[:, 1]}))
        log.append({"period": per.start_time, "C": C, "l1_ratio": a, "noise_index": ni, "fit_ms": fit_ms,
                    "nonzero": int((np.abs(coefficients(model)) > 1e-10).sum()),
                    **dict(zip(FEATURES, coefficients(model)))})
    return pd.concat(preds, ignore_index=True), pd.DataFrame(log)
