"""Window 4 of RTAP — verdict and tuning (used by notebook 10 and the app page "RTAP").

* load_run(name)          — everything a pipeline run produced (forecasts, regional decisions, sensor checks, Spark output)
* truth()                 — what really happened: rain tomorrow (≥ 1 mm) for every station and day
* station_metrics(fc)     — quality of the streamed forecasts against the truth
* regional_features(W)    — features of the 5 OTHER stations (the distributed context) with a window of W days
* tuning_experiment()     — the lecture's "simple vs tuned" model: C, l1_ratio and the window W chosen by time-series
                            cross-validation on 2015–2022, compared with the defaults and with a single validation split
"""
from __future__ import annotations

import glob
import json
import sqlite3
import time

import numpy as np
import pandas as pd

from rtap_common import OUT, STATIONS, load_station_daily

RAIN_MM = 1.0


def load_run(name: str) -> dict:
    out = OUT / name
    con = sqlite3.connect(out / "alerts.db")
    res = {t: pd.read_sql(f"SELECT * FROM {t}", con) for t in ("forecasts", "regional", "sensor")}
    con.close()
    reg = [pd.read_csv(f) for f in glob.glob(str(out / "regional" / "*.csv"))]
    res["spark_regional"] = pd.concat(reg).sort_values("window_start").reset_index(drop=True) if reg else pd.DataFrame()
    tp = [pd.read_csv(f) for f in glob.glob(str(out / "throughput" / "*.csv"))]
    if tp:
        tp = pd.concat(tp).sort_values("batch_id").groupby(["window_start", "station"]).tail(1)
        res["throughput"] = tp.sort_values(["window_start", "station"]).reset_index(drop=True)
    else:
        res["throughput"] = pd.DataFrame()
    res["progress"] = pd.DataFrame([json.loads(line) for line in open(out / "progress.jsonl")])
    res["info"] = json.loads((out / "run_info.json").read_text()) if (out / "run_info.json").exists() else {}
    return res


def truth() -> pd.DataFrame:
    rows = []
    for s, d in load_station_daily().items():
        d = d.sort_values("date")
        rows.append(pd.DataFrame({"station": s, "date": d.date.dt.strftime("%Y-%m-%d"),
                                  "rain_tomorrow": (d.precipitation_sum.shift(-1) >= RAIN_MM).astype(float).where(d.precipitation_sum.shift(-1).notna())}))
    return pd.concat(rows, ignore_index=True)


def station_metrics(fc: pd.DataFrame, thr: float, start: str = "2024-01-01", col: str = "p_rain") -> pd.DataFrame:
    from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
    f = fc[(fc.date >= start) & fc[col].notna()].merge(truth(), on=["station", "date"]).dropna(subset=["rain_tomorrow"])
    rows = []
    for s, g in list(f.groupby("station")) + [("все станции", f)]:
        y, p = g.rain_tomorrow.astype(int), g[col]
        rows.append({"Станция": STATIONS.get(s, s), "Прогнозов": len(g), "ROC-AUC": roc_auc_score(y, p), "F1": f1_score(y, p >= thr),
                     "Recall": recall_score(y, p >= thr), "Precision": precision_score(y, p >= thr)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- distributed context as features
def regional_features(window: int, target_station: str = "chisinau") -> pd.DataFrame:
    """Context from the 5 other stations: the same aggregates Spark computes (mean pressure, rain share, humidity, cloud),
    plus their change over `window` days. Only past and present days are used — no leakage."""
    daily = load_station_daily()
    others = [s for s in STATIONS if s != target_station]
    p = pd.DataFrame({s: daily[s].set_index("date").pressure_msl_mean for s in others})
    rain = pd.DataFrame({s: (daily[s].set_index("date").precipitation_sum >= RAIN_MM).astype(float) for s in others})
    hum = pd.DataFrame({s: daily[s].set_index("date").relative_humidity_2m_mean for s in others})
    cloud = pd.DataFrame({s: daily[s].set_index("date").cloud_cover_mean for s in others})
    reg = pd.DataFrame({"reg_pressure": p.mean(axis=1), "reg_rain_share": rain.mean(axis=1),
                        "reg_humidity": hum.mean(axis=1), "reg_cloud": cloud.mean(axis=1)})
    reg[f"reg_pressure_change"] = reg.reg_pressure.diff(window)
    reg[f"reg_rain_share_mean"] = reg.reg_rain_share.rolling(window).mean()
    reg[f"reg_cloud_change"] = reg.reg_cloud.diff(window)
    return reg.rename_axis("date").reset_index()


REGIONAL_NAMES_RU = {"reg_pressure": "Давление соседей, гПа", "reg_rain_share": "Доля соседей с дождём сегодня",
                     "reg_humidity": "Влажность соседей, %", "reg_cloud": "Облачность соседей, %",
                     "reg_pressure_change": "Изменение давления соседей за окно, гПа",
                     "reg_rain_share_mean": "Доля дождливых у соседей за окно", "reg_cloud_change": "Изменение облачности соседей за окно"}


def tuning_experiment(windows=(1, 3, 7), n_splits: int = 5, seed: int = 0) -> dict:
    """Simple vs tuned Elastic Net for Chișinău, with and without the distributed context.
    Tuned = GridSearchCV over C × l1_ratio with TimeSeriesSplit on 2015–2022 (no shuffling: the future never trains the
    past); the window W of the regional features is a hyper-parameter too. Test = 2024–2025, used once."""
    from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
    from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

    from db import load_dataset
    from features import FEATURES, TARGET
    from regularization import make_model

    data = load_dataset("chisinau")
    grid = {"logisticregression__C": [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
            "logisticregression__l1_ratio": [0.0, 0.25, 0.5, 0.75, 1.0]}
    thr_grid = np.round(np.arange(0.05, 0.95, 0.01), 2)
    rows, cv_tables, coefs = [], {}, {}

    def evaluate(name, model, feats, tr, va, te, cv_score=np.nan, params="", fit_s=np.nan):
        pv = model.predict_proba(va[feats])[:, 1]
        thr = float(thr_grid[int(np.argmax([f1_score(va[TARGET], pv >= t) for t in thr_grid]))])
        pt = model.predict_proba(te[feats])[:, 1]
        rows.append({"Модель": name, "Параметры": params, "Признаков": len(feats), "PR-AUC CV (2015–2022)": cv_score,
                     "PR-AUC валидация 2023": average_precision_score(va[TARGET], pv), "Порог": thr,
                     "ROC-AUC тест": roc_auc_score(te[TARGET], pt), "PR-AUC тест": average_precision_score(te[TARGET], pt),
                     "F1 тест": f1_score(te[TARGET], pt >= thr), "Время подбора, с": fit_s})

    for W in (None,) + tuple(windows):
        if W is None:
            d, feats, tag = data, FEATURES, "26 признаков станции"
        else:
            reg = regional_features(W)
            d = data.merge(reg, on="date", how="left").dropna()
            feats = FEATURES + [c for c in reg.columns if c != "date"]
            tag = f"+ соседи, окно {W} сут."
        tr, va, te = d[d.date.dt.year <= 2022], d[d.date.dt.year == 2023], d[d.date.dt.year >= 2024]
        if W is None or W == windows[0]:
            simple = make_model(1.0, 0.5).fit(tr[feats], tr[TARGET])
            evaluate(f"Простая ENet (C = 1, α = 0.5), {tag}", simple, feats, tr, va, te, params="C = 1, α = 0.5")
        t = time.time()
        gs = GridSearchCV(make_model(), grid, cv=TimeSeriesSplit(n_splits=n_splits), scoring="average_precision", n_jobs=-1)
        gs.fit(tr[feats], tr[TARGET])
        bp = gs.best_params_
        params = f"C = {bp['logisticregression__C']}, α = {bp['logisticregression__l1_ratio']}"
        evaluate(f"Настроенная ENet (CV), {tag}", gs.best_estimator_, feats, tr, va, te, cv_score=gs.best_score_,
                 params=params, fit_s=time.time() - t)
        cv = pd.DataFrame(gs.cv_results_)[["param_logisticregression__C", "param_logisticregression__l1_ratio",
                                            "mean_test_score", "std_test_score"]]
        cv.columns = ["C", "α", "PR-AUC CV", "Разброс по фолдам"]
        cv_tables[tag] = cv
        coefs[tag] = pd.Series(gs.best_estimator_[-1].coef_[0], index=feats)
    return {"results": pd.DataFrame(rows), "cv": cv_tables, "coefs": coefs}
