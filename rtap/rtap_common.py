"""Shared settings of the RTAP pipeline (lecture 5): Kafka topics, paths, station data, Spark session, models.

Pipeline (one process per "window" of the lecture):
  1. rtap/producer_stations.py — 6 weather stations = distributed sources → topic weather_events (JSON, key = station)
  2. rtap/spark_aggregator.py  — Spark Structured Streaming: event-time windows over all stations (regional picture,
                                 sensor consistency) + processing-time windows (throughput) → topic rtap_agg + CSV
  3. rtap/alert_service.py     — per-station features → RF / Elastic Net forecast + SHAP explanation → alert level,
                                 regional consensus, sensor checks → topic rtap_alerts + SQLite (dashboard)
  4. verdict (notebook 10 / app page) — tuning C, l1_ratio, window with time-series cross-validation, verdict
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
TOPIC_EVENTS, TOPIC_AGG, TOPIC_ALERTS = "weather_events", "rtap_agg", "rtap_alerts"
OUT = ROOT / "data" / "rtap"
JAVA_HOME_21 = "/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home"
SPARK_KAFKA_PACKAGE = "org.apache.spark:spark-sql-kafka-0-10_2.13:4.2.0"

STATIONS = {"chisinau": "Кишинёв", "soroca": "Сороки", "balti": "Бельцы", "ungheni": "Унгены",
            "comrat": "Комрат", "cahul": "Кагул"}
# raw daily measurements sent by a station (everything needed for the 26 features and the weather type)
MEASUREMENTS = ["temperature_2m_max", "temperature_2m_min", "temperature_2m_mean", "dew_point_2m_mean",
                "precipitation_sum", "snowfall_sum", "precipitation_hours", "relative_humidity_2m_mean",
                "relative_humidity_2m_max", "pressure_msl_mean", "pressure_msl_max", "pressure_msl_min",
                "cloud_cover_mean", "wind_speed_10m_max", "wind_gusts_10m_max", "wind_direction_10m_dominant",
                "sunshine_duration", "daylight_duration", "vapour_pressure_deficit_max", "soil_moisture_0_to_7cm_mean"]


def load_station_daily() -> dict[str, pd.DataFrame]:
    """Daily measurements of the 6 stations: from the PostgreSQL warehouse, or from the CSV copies."""
    try:
        from db import load_daily, query
        query("SELECT 1")
        return {s: load_daily(s) for s in STATIONS}
    except Exception:
        out = {}
        for s in STATIONS:
            path = ROOT / "data" / "raw" / ("chisinau_daily_2015_2025.csv" if s == "chisinau" else f"cities/{s}_daily_2015_2025.csv")
            out[s] = pd.read_csv(path, parse_dates=["date"])
        return out


def to_message(station: str, rec: dict, seq: int, sent_at_ms: int) -> dict:
    msg = {"station": station, "date": pd.Timestamp(rec["date"]).strftime("%Y-%m-%d"), "seq": seq, "sent_at_ms": sent_at_ms}
    for c in MEASUREMENTS:
        v = rec.get(c)
        msg[c] = None if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)
    return msg


def json_bytes(obj) -> bytes:
    return json.dumps(obj, ensure_ascii=False, default=str).encode("utf-8")


def spark_session(app: str, cores: int = 2):
    """Local Spark with the Kafka connector. Spark 4.2 needs Java 17/21 — the keg-only Homebrew openjdk@21 is used."""
    if Path(JAVA_HOME_21).exists():
        os.environ["JAVA_HOME"] = JAVA_HOME_21
    from pyspark.sql import SparkSession
    spark = (SparkSession.builder.appName(app).master(f"local[{cores}]")
             .config("spark.jars.packages", SPARK_KAFKA_PACKAGE)
             .config("spark.sql.shuffle.partitions", "4")
             .config("spark.sql.session.timeZone", "UTC")
             .config("spark.sql.mapKeyDedupPolicy", "LAST_WIN")
             .config("spark.ui.enabled", "false")
             .getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def build_models():
    """Rain model (Random Forest of notebook 02), critical-weather model and alert thresholds (the rules of notebook 04),
    and the fast Elastic Net of notebook 09 as a second opinion."""
    import joblib
    from sklearn.base import clone
    from sklearn.metrics import f1_score, precision_score

    from db import load_dataset
    from features import FEATURES, TARGET, TARGET3
    from regularization import make_model
    from streaming import AlertEngine

    data = load_dataset("chisinau")
    train, valid = data[data.date.dt.year <= 2022], data[data.date.dt.year == 2023]
    bundle = joblib.load(ROOT / "models" / "baseline_model.joblib")
    rf, thr = bundle["model"], bundle["threshold"]
    crit = clone(rf).set_params(class_weight="balanced").fit(train[FEATURES], train[TARGET3])
    pc = crit.predict_proba(valid[FEATURES])[:, 2]
    is_crit = (valid[TARGET3] == 2).astype(int)
    grid = np.round(np.arange(0.05, 0.95, 0.01), 2)
    thr_orange = grid[int(np.argmax([f1_score(is_crit, pc >= t) for t in grid]))]
    ok = [t for t in grid if (pc >= t).sum() >= 3 and precision_score(is_crit, pc >= t) >= 0.5]
    thr_red = max(min(ok) if ok else float(np.round(np.quantile(pc, 0.98), 2)), thr_orange + 0.05)
    best = joblib.load(ROOT / "models" / "regularization_best.joblib")["Elastic Net"]
    enet = make_model(best["C"], best["l1_ratio"]).fit(train[FEATURES], train[TARGET])
    return {"rf": rf, "thr": thr, "crit": crit, "enet": enet, "enet_thr": best["threshold"],
            "alerts": AlertEngine(thr_rain=thr, thr_orange=float(thr_orange), thr_red=float(thr_red)), "train": train}
