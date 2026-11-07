"""Simulation of a real-time forecasting system (Streaming + RTAP).

Components (production equivalent in parentheses):
- sensor_stream           — data source: one daily record from the weather station (MQTT / Kafka topic)
- TimeSeriesStore         — time-series database (InfluxDB), implemented on SQLite
- SensorHealthMonitor     — data quality checks: stuck sensors, physically impossible values
- SpatialConsistencyCheck — compares the barometer with neighbouring stations: detects miscalibration, auto-corrects
- AlertEngine             — warning levels (green / yellow / orange / red) and their text
- Notifier                — sends notifications (SMS / Telegram / e-mail / webhook); here only a local log file
- StreamProcessor         — processes each record: features → forecast → alert → evaluation → adaptation
                            (a Spark Structured Streaming job)
- adaptation policies (RTAP): StaticPolicy, PeriodicPolicy, DriftPolicy

User-facing texts (alerts, technical notifications, labels shown in tables and figures) are in Russian.
"""
from __future__ import annotations

import json
import sqlite3
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone

from features import FEATURES, RAIN_MM, TARGET, build_features, compute_features

HISTORY_DAYS = 8  # days of history needed for lags and rolling windows (up to 7 days + the current day)


# --------------------------------------------------------------------------- data source
def sensor_stream(raw: pd.DataFrame, start: str, end: str):
    """Producer: yields one daily record (dict) in chronological order, like a weather station."""
    for rec in raw[(raw.date >= start) & (raw.date <= end)].sort_values("date").to_dict("records"):
        yield rec


# --------------------------------------------------------------------------- time-series storage
class TimeSeriesStore:
    """InfluxDB-style time-series database: measurement + timestamp + tags + fields."""

    def __init__(self, path: str | Path = ":memory:"):
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).unlink(missing_ok=True)
        # autocommit: every point is immediately visible to other readers (e.g. a dashboard)
        self.conn = sqlite3.connect(str(path), isolation_level=None)
        self.conn.execute("PRAGMA synchronous = OFF")
        self.conn.execute("CREATE TABLE points (measurement TEXT, ts TEXT, tags TEXT, fields TEXT)")
        self.conn.execute("CREATE INDEX idx ON points (measurement, ts)")

    def write(self, measurement: str, ts, fields: dict, tags: dict | None = None) -> None:
        clean = {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in fields.items()}
        self.conn.execute("INSERT INTO points VALUES (?, ?, ?, ?)",
                          (measurement, pd.Timestamp(ts).isoformat(), json.dumps(tags or {}), json.dumps(clean, default=float)))

    def query(self, measurement: str, start=None, end=None) -> pd.DataFrame:
        sql, args = "SELECT ts, tags, fields FROM points WHERE measurement = ?", [measurement]
        if start is not None:
            sql += " AND ts >= ?"; args.append(pd.Timestamp(start).isoformat())
        if end is not None:
            sql += " AND ts <= ?"; args.append(pd.Timestamp(end).isoformat())
        rows = self.conn.execute(sql + " ORDER BY ts", args).fetchall()
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame([{"ts": pd.Timestamp(ts), **json.loads(tg), **json.loads(fl)} for ts, tg, fl in rows])

    def count(self) -> pd.Series:
        return pd.Series(dict(self.conn.execute("SELECT measurement, COUNT(*) FROM points GROUP BY measurement").fetchall()))


# --------------------------------------------------------------------------- data quality
class SensorHealthMonitor:
    """Detects stuck sensors (the same value several days in a row) and physically impossible values."""

    # sensor name (used in notifications) -> monitored column
    STUCK_COLUMNS = {"Барометр": "pressure_msl_mean", "Термометр": "temperature_2m_mean",
                     "Гигрометр": "relative_humidity_2m_mean", "Анемометр": "wind_speed_10m_max"}
    RANGES = {"pressure_msl_mean": (950, 1060), "temperature_2m_mean": (-40, 45),
              "relative_humidity_2m_mean": (0, 100), "precipitation_sum": (0, 300), "wind_speed_10m_max": (0, 200)}

    def __init__(self, stuck_days: int = 3):
        self.stuck_days = stuck_days

    def check(self, buffer: pd.DataFrame) -> list[str]:
        issues = []
        last = buffer.iloc[-1]
        for sensor, col in self.STUCK_COLUMNS.items():
            recent = buffer[col].tail(self.stuck_days)
            if len(recent) == self.stuck_days and recent.nunique() == 1:
                issues.append(f"{sensor} завис: значение {recent.iloc[-1]} не менялось {self.stuck_days} дня подряд")
        for col, (lo, hi) in self.RANGES.items():
            if not lo <= last[col] <= hi:
                issues.append(f"Физически невозможное значение: {col} = {last[col]}")
        return issues


class SpatialConsistencyCheck:
    """Compares the station pressure with the median of neighbouring stations. The difference between nearby stations
    is very stable (historically its 7-day mean never deviated by more than ~1.3 hPa), so a large deviation means
    a miscalibrated barometer."""

    COLUMNS = ["pressure_msl_mean", "pressure_msl_max", "pressure_msl_min"]

    def __init__(self, neighbor_pressure: pd.Series, normal_diff: float, threshold: float = 3.0, window: int = 7):
        self.neighbors, self.normal_diff = neighbor_pressure, normal_diff
        self.threshold, self.diffs = threshold, deque(maxlen=window)
        self.offset = None                                  # estimated offset (hPa), or None if everything is fine

    def update(self, day, own_pressure: float):
        self.diffs.append(own_pressure - self.neighbors.loc[pd.Timestamp(day)] - self.normal_diff)
        mean = float(np.mean(self.diffs))
        self.offset = mean if len(self.diffs) == self.diffs.maxlen and abs(mean) > self.threshold else None
        return self.offset

    def correct(self, rec: dict) -> dict:
        if self.offset is None:
            return rec
        return {**rec, **{c: rec[c] - self.offset for c in self.COLUMNS}}


# --------------------------------------------------------------------------- warnings
# Display names of the warning levels: green (no warning) / yellow / orange / red
LEVELS = ["Зелёный", "Жёлтый", "Оранжевый", "Красный"]
# Recommended actions per level (part of the notification text)
ACTIONS = {
    1: "Планируйте мероприятия на открытом воздухе с запасным вариантом.",
    2: "Отложите полевые работы и мероприятия на открытом воздухе; подготовьте аварийные службы.",
    3: "Введите план действий в чрезвычайной ситуации; оповестите население; избегайте лишних поездок.",
}


@dataclass
class AlertEngine:
    thr_rain: float     # P(rain) from which a yellow warning is issued
    thr_orange: float   # P(critical weather) from which an orange warning is issued
    thr_red: float      # P(critical weather) from which a red warning is issued

    def level(self, p_rain: float, p_crit: float) -> int:
        if p_crit >= self.thr_red:
            return 3
        if p_crit >= self.thr_orange:
            return 2
        return 1 if p_rain >= self.thr_rain else 0

    @staticmethod
    def causes(f: pd.Series) -> list[str]:
        """Plain-language reasons based on known physical signals (see EDA and feature importance)."""
        c = []
        if f.pressure_change_3d <= -4:
            c.append(f"давление упало на {-f.pressure_change_3d:.1f} гПа за 3 дня")
        if f.pressure_msl_mean <= 1010:
            c.append(f"низкое давление ({f.pressure_msl_mean:.0f} гПа)")
        if f.cloud_cover_mean >= 80:
            c.append(f"сплошная облачность ({f.cloud_cover_mean:.0f} %)")
        if f.dew_point_depression <= 3:
            c.append("воздух близок к насыщению")
        if f.temperature_2m_mean >= 25:
            c.append(f"жара ({f.temperature_2m_mean:.1f} °C в среднем)")
        if f.wind_speed_10m_max >= 30:
            c.append(f"сильный ветер сегодня ({f.wind_speed_10m_max:.0f} км/ч)")
        if f.wind_dir_sin > 0.5 or f.wind_dir_cos < -0.5:
            c.append("ветер восточного–южного направления (влажный воздух)")
        return c

    def message(self, day, level: int, p_rain: float, p_crit: float, f: pd.Series) -> str:
        tomorrow = (pd.Timestamp(day) + pd.Timedelta(days=1)).strftime("%d.%m.%Y")
        cause = "; ".join(self.causes(f)) or "сочетание факторов"
        return (f"{LEVELS[level].upper()} КОД на {tomorrow}: P(осадки) = {p_rain:.0%}, "
                f"P(критическая погода) = {p_crit:.0%}. Причины: {cause}. {ACTIONS[level]}")


class Notifier:
    """Sends notifications. Here they are written to a log file; in production — SMS / Telegram / e-mail / webhook."""

    def __init__(self, log_path: str | Path | None = None, min_level: int = 2):
        self.min_level, self.sent = min_level, []
        self.log_path = Path(log_path) if log_path else None
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self.log_path.write_text("")

    def notify(self, day, level: int, text: str, kind: str = "alert") -> None:
        """kind: "alert" (weather warning, sent only from `min_level` up) or "technical" (sensor problem)."""
        if kind == "alert" and level < self.min_level:
            return
        entry = {"date": pd.Timestamp(day), "kind": kind, "level": LEVELS[level] if kind == "alert" else "—", "message": text}
        self.sent.append(entry)
        if self.log_path:
            with self.log_path.open("a") as fh:
                fh.write(f"{entry['date'].date()} [{kind.upper()}] {text}\n")


# --------------------------------------------------------------------------- adaptation policies (RTAP)
# `name` is a display label (shown in tables and figures)
class StaticPolicy:
    name = "Статичная (без переобучения)"

    def should_retrain(self, day, proc) -> bool:
        return False

    def window(self, feats: pd.DataFrame, day) -> pd.DataFrame:
        return feats


class PeriodicPolicy:
    """Retrain at the start of every month; expanding window (all history) or sliding window (last N years)."""

    def __init__(self, window_years: int | None = None):
        self.window_years = window_years
        self.name = "Ежемесячно, " + (f"скользящее окно {window_years} года" if window_years else "расширяющееся окно")

    def should_retrain(self, day, proc) -> bool:
        return pd.Timestamp(day).day == 1

    def window(self, feats, day):
        if self.window_years is None:
            return feats
        return feats[feats.date > pd.Timestamp(day) - pd.DateOffset(years=self.window_years)]


class DriftPolicy:
    """Retrain only when drift is detected: the Brier skill score (vs. climatology) over the last `window` days
    drops below `min_bss`. After a retrain, wait at least `min_gap` days."""

    def __init__(self, window: int = 60, min_bss: float = 0.0, min_gap: int = 30, window_years: int | None = None):
        self.bss_window, self.min_bss, self.min_gap, self.window_years = window, min_bss, min_gap, window_years
        self.name = f"При дрейфе (BSS за {window} дн. < {min_bss})" + (f", окно {window_years} года" if window_years else "")

    def should_retrain(self, day, proc) -> bool:
        if proc.last_retrain is not None and (pd.Timestamp(day) - proc.last_retrain).days < self.min_gap:
            return False
        bss = proc.rolling_bss(self.bss_window)
        return bss is not None and bss < self.min_bss

    def window(self, feats, day):
        if self.window_years is None:
            return feats
        return feats[feats.date > pd.Timestamp(day) - pd.DateOffset(years=self.window_years)]


# --------------------------------------------------------------------------- stream processor
@dataclass
class StreamProcessor:
    rain_model: object
    thr_rain: float
    climatology: dict                       # month -> historical frequency of rainy days (reference for the BSS)
    history_raw: pd.DataFrame               # raw data known before the stream starts (used for retraining)
    crit_model: object | None = None
    alerts: AlertEngine | None = None
    store: TimeSeriesStore | None = None
    notifier: Notifier | None = None
    health: SensorHealthMonitor | None = None
    spatial: SpatialConsistencyCheck | None = None
    auto_correct: bool = False              # automatic intervention: correct the barometer offset
    policy: object = field(default_factory=StaticPolicy)

    def __post_init__(self):
        self.buffer = deque(maxlen=HISTORY_DAYS)
        self.seen = []                      # records received through the stream (used for retraining)
        self.pending = None                 # yesterday's forecast, waiting for today's outcome
        self.outcomes = []                  # (forecast date, p, y, p_climatology)
        self.retrains, self.last_retrain = [], None
        self.latencies = []
        for rec in self.history_raw.sort_values("date").tail(HISTORY_DAYS).to_dict("records"):
            self.buffer.append(rec)

    # --- monitoring
    def rolling_bss(self, window: int):
        if len(self.outcomes) < window:
            return None
        o = np.array([(p, y, pc) for _, p, y, pc in self.outcomes[-window:]])
        brier, brier_clim = np.mean((o[:, 0] - o[:, 1]) ** 2), np.mean((o[:, 2] - o[:, 1]) ** 2)
        return 1 - brier / brier_clim if brier_clim > 0 else None

    # --- adaptation
    def retrain(self, day) -> None:
        t0 = time.perf_counter()
        raw_all = pd.concat([self.history_raw, pd.DataFrame(self.seen)], ignore_index=True)
        feats = self.policy.window(build_features(raw_all), day)    # the last day has no "tomorrow" yet → dropped
        model = clone(self.rain_model)
        if "n_jobs" in model.get_params():
            model.set_params(n_jobs=-1)                     # train on all cores...
        self.rain_model = model.fit(feats[FEATURES], feats[TARGET])
        if "n_jobs" in model.get_params():
            self.rain_model.set_params(n_jobs=1)            # ...predict single rows without parallelisation overhead
        self.retrains.append({"date": pd.Timestamp(day), "train_days": len(feats), "duration_s": time.perf_counter() - t0})
        self.last_retrain = pd.Timestamp(day)

    # --- main step
    def process(self, rec: dict) -> dict:
        t0 = time.perf_counter()
        day = pd.Timestamp(rec["date"])
        raw_pressure = rec["pressure_msl_mean"]

        # 0) spatial check of the barometer (vs. neighbouring stations) and, optionally, automatic correction
        spatial_issue = None
        if self.spatial is not None and self.spatial.update(day, raw_pressure) is not None:
            spatial_issue = (f"Барометр раскалиброван: отклонение {self.spatial.offset:+.1f} гПа от соседних станций"
                             + (" — применена автоматическая коррекция" if self.auto_correct else ""))
            if self.auto_correct:
                rec = self.spatial.correct(rec)
        self.buffer.append(rec)
        self.seen.append(rec)

        # 1) evaluate yesterday's forecast against what happened today
        if self.pending is not None:
            y = int(rec["precipitation_sum"] >= RAIN_MM)
            self.outcomes.append((self.pending["date"], self.pending["p_rain"], y, self.pending["p_clim"]))
            if self.store:
                self.store.write("outcomes", self.pending["date"], {"p_rain": self.pending["p_rain"], "y": y,
                                                                   "pred": int(self.pending["p_rain"] >= self.thr_rain)})

        # 2) adaptation (RTAP): retrain if the policy asks for it
        if self.policy.should_retrain(day, self):
            self.retrain(day)

        # 3) data quality
        buf = pd.DataFrame(list(self.buffer))
        issues = self.health.check(buf) if self.health else []
        if spatial_issue:
            issues.append(spatial_issue)

        # 4) features + forecast for tomorrow
        f = compute_features(buf).iloc[-1]
        x = f[FEATURES].to_frame().T.astype(float)
        p_rain = float(self.rain_model.predict_proba(x)[0, 1])
        p_crit = float(self.crit_model.predict_proba(x)[0, 2]) if self.crit_model is not None else 0.0
        level = self.alerts.level(p_rain, p_crit) if self.alerts else int(p_rain >= self.thr_rain)

        # 5) warnings and notifications
        if self.notifier:
            for issue in issues:
                self.notifier.notify(day, 0, issue + " — проверьте станцию; доверие к прогнозу снижено.", kind="technical")
            if self.alerts and level > 0:
                self.notifier.notify(day, level, self.alerts.message(day, level, p_rain, p_crit, f))

        out = {"date": day, "p_rain": p_rain, "p_crit": p_crit, "level": level, "issues": len(issues),
               "pressure_raw": raw_pressure, "pressure_used": rec["pressure_msl_mean"],
               "offset": self.spatial.offset if self.spatial is not None else None}
        if self.store:
            self.store.write("observations", day, {k: rec[k] for k in ("temperature_2m_mean", "pressure_msl_mean",
                                                                      "precipitation_sum", "cloud_cover_mean")},
                             tags={"station": "Chisinau"})
            self.store.write("forecasts", day, {"p_rain": p_rain, "p_crit": p_crit, "level": level, "sensor_issues": len(issues)},
                             tags={"model_version": str(len(self.retrains))})

        self.pending = {"date": day, "p_rain": p_rain, "p_clim": self.climatology[(day + pd.Timedelta(days=1)).month]}
        self.latencies.append(time.perf_counter() - t0)
        return out
