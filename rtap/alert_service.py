"""Window 3 of RTAP — algorithmic decision with an explanation, for every station and for the region.

Consumes two topics:
  weather_events — the station records: per-station history buffer → 26 features → P(rain) of the Random Forest
                   (+ the fast Elastic Net of notebook 09 as a second opinion), P(critical weather), alert level,
                   TreeSHAP explanation (the 3 strongest contributions) and the plain-language causes;
  rtap_agg       — the regional windows of Spark: completeness (did all stations report?) and the barometer
                   cross-check (station pressure vs the median of the other stations, compared with the normal offset).
Decisions:
  * a station-level yellow code is a RECOMMENDATION; a REGIONAL ALERT needs ≥ 4 of 6 stations agreeing;
  * a station whose barometer deviates by > 3 hPa from its normal offset for 5 windows in a row, with a STABLE
    deviation (std < 1.5 hPa — random noise is not stable) gets a technical alert,
    and the estimated bias is subtracted from its later records (automatic intervention, as in notebook 04);
  * the noise index of every station (day-to-day jumps of 4 sensors vs the same season of 2015–2022) is checked weekly;
    a noisy station gets a technical alert and its forecasts are marked "trust reduced";
  * a measurement that a station stopped sending is taken from the neighbours of the same day;
  * a missing day in the 8-day history (a late or lost message) is rebuilt from the neighbours (median of ≥ 3 stations,
    at most 2 days); a late record replaces the rebuilt one. Without enough neighbours the record is only flagged.
Results: topic rtap_alerts (JSON) + SQLite <out>/alerts.db (tables forecasts, regional, sensor).
Example:  .venv/bin/python rtap/alert_service.py --out run
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import time
from collections import defaultdict

import numpy as np
import pandas as pd
from kafka import KafkaConsumer, KafkaProducer

from rtap_common import (BOOTSTRAP, MEASUREMENTS, OUT, STATIONS, TOPIC_AGG, TOPIC_ALERTS, TOPIC_EVENTS, build_models,
                         json_bytes, load_station_daily)

HISTORY = 8          # days needed for lags and rolling windows of the features
BIAS_HPA, BIAS_RUN, BIAS_MAX_STD = 3.0, 5, 1.5   # a shifted barometer = a STABLE offset (noise is not stable)
WAIT_S = 2.0                                     # how long a record may wait for its neighbours (seconds of real time)
REGIONAL_MIN = 4


def normal_offsets(daily: dict[str, pd.DataFrame]) -> dict[str, float]:
    """Typical difference 'station − median of the other stations' of the sea-level pressure in 2015–2022."""
    p = pd.DataFrame({s: d.set_index("date").pressure_msl_mean for s, d in daily.items()})
    p = p[p.index < "2023-01-01"]
    return {s: float((p[s] - p.drop(columns=s).median(axis=1)).median()) for s in p}


NOISE_SENSORS = ["pressure_msl_mean", "relative_humidity_2m_mean", "cloud_cover_mean", "dew_point_2m_mean"]
NOISE_ON, NOISE_OFF, NOISE_DAYS = 1.8, 1.3, 14       # calibrated on clean 2015–2022: max ≈ 1.83, q95 = 1.35 (notebook 10)


def noise_reference(daily: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Typical |day-to-day jump| of the noise-prone sensors for every day of the year (median over 2015–2022,
    smoothed over ±7 days): in winter the weather itself jumps more, so the reference is seasonal."""
    ref = {}
    for st, d in daily.items():
        d = d[d.date < "2023-01-01"].sort_values("date")
        jumps = d[NOISE_SENSORS].diff().abs()
        jumps["doy"] = d.date.dt.dayofyear.clip(upper=365)
        med = jumps.groupby("doy").median()
        ref[st] = pd.concat([med.iloc[-7:], med, med.iloc[:7]]).rolling(15, center=True).mean().iloc[7:-7]
    return ref


class AlertService:
    def __init__(self, out, models, offsets, use_shap=True, auto_correct=True, noise_ref=None):
        from features import FEATURES, compute_features
        import shap
        self.FEATURES, self.compute_features = FEATURES, compute_features
        self.m = models
        self.m["rf"].set_params(n_jobs=1)
        self.m["crit"].set_params(n_jobs=1)
        self.explainer = shap.TreeExplainer(self.m["rf"]) if use_shap else None
        self.offsets = offsets
        self.auto_correct = auto_correct
        self.noise_ref = noise_ref or {}
        self.noisy = defaultdict(bool)            # station → its sensors are currently noisier than usual
        self.pending: list[dict] = []
        self.buffers = defaultdict(dict)          # station → {date: record}
        self.deviation = defaultdict(dict)        # station → {day: deviation from the neighbours, hPa}
        self.correction = defaultdict(float)      # station → hPa subtracted from later records
        self.regional = defaultdict(dict)         # date → {station: p_rain}
        self.db = sqlite3.connect(out / "alerts.db")
        self.db.executescript("""
            DROP TABLE IF EXISTS forecasts; DROP TABLE IF EXISTS regional; DROP TABLE IF EXISTS sensor;
            CREATE TABLE forecasts (station TEXT, date TEXT, status TEXT, p_rain REAL, p_rain_enet REAL, p_crit REAL,
                level INTEGER, kind TEXT, shap TEXT, causes TEXT, message TEXT, correction REAL, sent_at_ms INTEGER,
                processed_at_ms INTEGER, latency_ms REAL);
            CREATE TABLE regional (date TEXT, n_stations INTEGER, n_rain INTEGER, share REAL, decision TEXT, stations TEXT);
            CREATE TABLE sensor (date TEXT, station TEXT, check_name TEXT, value REAL, message TEXT);""")
        self.producer = KafkaProducer(bootstrap_servers=BOOTSTRAP, linger_ms=5)

    # ------------------------------------------------------------------ station records
    def on_events(self, msgs: list[dict], final: bool = False):
        """Records that cannot be completed yet (a missing day or field and fewer than 3 neighbours have reported
        that day) wait up to WAIT_S seconds for the neighbours; then they are processed as they are."""
        now = time.time()
        msgs = sorted(self.pending + [dict(m, _arrived=now) for m in msgs], key=lambda v: (v["date"], v["station"]))
        self.pending = []
        rows, meta = [], []
        for m in msgs:
            st = m["station"]
            if st == "_heartbeat":
                continue
            rec = {c: m.get(c) for c in MEASUREMENTS}
            rec["date"] = pd.Timestamp(m["date"])
            if self.correction[st]:
                for c in ("pressure_msl_mean", "pressure_msl_max", "pressure_msl_min"):
                    rec[c] -= self.correction[st]
            buf = self.buffers[st]
            late = rec["date"] in buf and buf[rec["date"]].get("imputed")
            buf[rec["date"]] = rec
            for d in sorted(buf)[:-40]:            # keep a short history only
                del buf[d]
            need = pd.date_range(rec["date"] - pd.Timedelta(days=HISTORY - 1), rec["date"])
            missing = [d for d in need if d not in buf]
            imputed = []
            for d in missing:   # a late or lost record: rebuild it from the neighbours (median of ≥ 3 other stations)
                nb = [self.buffers[o][d] for o in STATIONS if o != st and d in self.buffers[o]]
                if len(nb) >= 3 and len(missing) <= 2:
                    fill = {c: float(np.nanmedian([n[c] for n in nb])) for c in MEASUREMENTS}
                    fill["date"] = d
                    buf[d] = {**fill, "imputed": True}
                    imputed.append(d.strftime("%d.%m"))
            if any(d not in buf for d in need):
                if not final and now - m["_arrived"] < WAIT_S:
                    self.pending.append(m)
                    continue
                self.store(st, m, status="нет полной истории (опоздавшее или потерянное сообщение)")
                continue
            hist = pd.DataFrame([buf[d] for d in need])
            gaps = [c for c in MEASUREMENTS if pd.isna(rec.get(c))]
            for c in gaps:   # a station stopped sending one measurement: take it from the neighbours of the same day
                nb = [self.buffers[o][rec["date"]][c] for o in STATIONS
                      if o != st and rec["date"] in self.buffers[o] and not pd.isna(self.buffers[o][rec["date"]].get(c))]
                if len(nb) >= 3:
                    rec[c] = float(np.median(nb))
                    hist.loc[hist.date == rec["date"], c] = rec[c]
                    imputed.append(c)
            if any(pd.isna(rec.get(c)) for c in gaps) and not final and now - m["_arrived"] < WAIT_S:
                self.pending.append(m)                     # the neighbours of this day have not reported yet
                continue
            self.check_noise(st, buf, rec["date"])
            f = self.compute_features(hist).iloc[-1]
            if f[self.FEATURES].isna().any():
                self.store(st, m, status="пропуски в признаках")
                continue
            rows.append(f[self.FEATURES].astype(float).to_numpy())
            meta.append((st, m, f, "прогноз" + (f" (дополнено по соседям: {', '.join(imputed)})" if imputed else "")
                         + (" (опоздавшая запись)" if late else "") + (" (шум датчиков: доверие снижено)" if self.noisy[st] else "")))
        if not rows:
            return
        X = pd.DataFrame(np.vstack(rows), columns=self.FEATURES)       # micro-batch: one model call for all records
        p_rain = self.m["rf"].predict_proba(X)[:, 1]
        p_enet = self.m["enet"].predict_proba(X)[:, 1]
        p_crit = self.m["crit"].predict_proba(X)[:, 2]
        sv = None
        if self.explainer is not None:
            sv = np.asarray(self.explainer.shap_values(X))
            sv = sv[..., 1] if sv.ndim == 3 else sv
        from features import FEATURE_NAMES_RU
        eng = self.m["alerts"]
        for k, (st, m, f, status) in enumerate(meta):
            level = eng.level(p_rain[k], p_crit[k])
            top = []
            if sv is not None:
                for j in np.argsort(-np.abs(sv[k]))[:3]:
                    top.append({"признак": FEATURE_NAMES_RU[self.FEATURES[j]], "вклад": round(float(sv[k, j]), 3)})
            kind = "рекомендация" if level == 1 else ("тревога" if level >= 2 else "")
            msg = eng.message(m["date"], level, p_rain[k], p_crit[k], f) if level >= 1 else ""
            self.store(st, m, status=status, p_rain=p_rain[k], p_enet=p_enet[k], p_crit=p_crit[k], level=level,
                       kind=kind, shap=top, causes=eng.causes(f), message=msg)
            self.regional[m["date"]][st] = float(p_rain[k])
            self.check_regional(m["date"])

    def check_noise(self, st, buf, day):
        """Noise index of a station over the last NOISE_DAYS days; a change of state is reported as a technical alert."""
        if st not in self.noise_ref or day.dayofweek != 0:          # once a week is enough
            return
        h = pd.DataFrame([buf[d] for d in sorted(buf) if day - pd.Timedelta(days=NOISE_DAYS) <= d <= day])
        if len(h) < NOISE_DAYS:
            return
        ref = self.noise_ref[st].loc[h.date.dt.dayofyear.clip(upper=365)].mean()
        idx = float(np.median([h[c].diff().abs().median() / ref[c] for c in NOISE_SENSORS]))
        noisy = idx > NOISE_ON if not self.noisy[st] else idx > NOISE_OFF
        if noisy != self.noisy[st]:
            self.noisy[st] = noisy
            text = (f"датчики станции {STATIONS[st]} шумят: индекс шума {idx:.1f} (обычно ≈ 1) — доверие к прогнозу снижено"
                    if noisy else f"датчики станции {STATIONS[st]} снова в норме: индекс шума {idx:.1f}")
            self.sensor(day.strftime("%Y-%m-%d"), st, "шум", idx, text)
            self.producer.send(TOPIC_ALERTS, value=json_bytes({"type": "техническое", "station": st,
                                                                "date": day.strftime("%Y-%m-%d"), "noise_index": round(idx, 2)}))

    def store(self, st, m, status, p_rain=None, p_enet=None, p_crit=None, level=None, kind="", shap=None, causes=None, message=""):
        now = int(time.time() * 1000)
        row = (st, m["date"], status, p_rain, p_enet, p_crit, level, kind, json.dumps(shap or [], ensure_ascii=False),
               "; ".join(causes or []), message, self.correction[st], m.get("sent_at_ms"), now,
               now - m["sent_at_ms"] if m.get("sent_at_ms") else None)
        self.db.execute("INSERT INTO forecasts VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        tuple(float(v) if isinstance(v, np.floating) else v for v in row))
        if level and level >= 1:
            self.producer.send(TOPIC_ALERTS, value=json_bytes({"type": kind, "station": st, "date": m["date"], "level": int(level),
                                                               "p_rain": round(float(p_rain), 3), "shap": shap, "message": message}))

    def check_regional(self, date, final=False):
        votes = self.regional[date]
        if len(votes) < len(STATIONS) and not final:
            return
        n_rain = sum(p >= self.m["thr"] for p in votes.values())
        if n_rain >= REGIONAL_MIN:
            decision = "РЕГИОНАЛЬНАЯ ТРЕВОГА"
        elif n_rain >= 1:
            decision = "локальные рекомендации"
        else:
            decision = "спокойно"
        if len(votes) < len(STATIONS):
            decision += f" (неполные данные: {len(votes)} из {len(STATIONS)})"
        self.db.execute("INSERT INTO regional VALUES (?,?,?,?,?,?)",
                        (date, len(votes), n_rain, n_rain / len(votes), decision,
                         ", ".join(STATIONS[s] for s, p in votes.items() if p >= self.m["thr"])))
        if decision.startswith("РЕГИОНАЛЬНАЯ ТРЕВОГА"):
            self.producer.send(TOPIC_ALERTS, value=json_bytes({"type": "региональная тревога", "date": date, "n_rain": n_rain}))
        del self.regional[date]

    # ------------------------------------------------------------------ regional windows from Spark
    def on_agg(self, msgs: list[dict]):
        for a in msgs:
            day = str(a["window_start"])[:10]
            if a.get("n_stations", 6) < len(STATIONS):
                self.sensor(day, "—", "полнота", a.get("n_stations"), f"в окне {day} данные только от {a.get('n_stations')} из 6 станций")
            pressures = {k.split("|")[0]: v for k, v in json.loads(a.get("station_pressure") or "{}").items() if k.split("|")[1] == day}
            if len(pressures) < 3:
                continue
            for st, p in pressures.items():
                others = np.median([v for s, v in pressures.items() if s != st])
                self.deviation[st][pd.Timestamp(day)] = p - others - self.offsets.get(st, 0.0)   # records arrive uncorrected
        # one micro-batch of Spark may carry many windows, in any order → scan the sorted dates for the FIRST run of
        # BIAS_RUN consecutive deviating days (not only the last days of the batch)
        for st, devs in self.deviation.items():
            if self.correction[st] or self.noisy[st]:       # a noisy station is not checked for a stable shift
                continue
            days = sorted(devs)
            for i in range(len(days) - BIAS_RUN + 1):
                run = days[i:i + BIAS_RUN]
                vals = [devs[d] for d in run]
                if ((run[-1] - run[0]).days == BIAS_RUN - 1 and all(abs(v) > BIAS_HPA for v in vals)
                        and len({np.sign(v) for v in vals}) == 1 and np.std(vals) < BIAS_MAX_STD):
                    est = float(np.median([devs[d] for d in run]))
                    self.correction[st] = est if self.auto_correct else 1e-9   # 1e-9: detected, but not corrected
                    day = run[-1].strftime("%Y-%m-%d")
                    self.sensor(day, st, "барометр", est,
                                f"барометр станции {STATIONS[st]} отклоняется от соседей на {est:+.1f} гПа {BIAS_RUN} дней "
                                f"подряд (с {run[0]:%d.%m}) — " + ("включена автоматическая коррекция" if self.auto_correct
                                                                         else "коррекция отключена"))
                    self.producer.send(TOPIC_ALERTS, value=json_bytes({"type": "техническое", "station": st, "date": day,
                                                                        "bias_hpa": round(est, 2)}))
                    break

    def sensor(self, day, st, check, value, text):
        self.db.execute("INSERT INTO sensor VALUES (?,?,?,?,?)", (day, st, check, value, text))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="run")
    ap.add_argument("--idle-seconds", type=float, default=25)
    ap.add_argument("--no-shap", action="store_true")
    ap.add_argument("--no-correct", action="store_true", help="detect a shifted barometer, but do not correct it")
    ap.add_argument("--group", default=None, help="Kafka consumer group (default: a new one each run)")
    args = ap.parse_args()
    out = OUT / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "ALERTS_READY").unlink(missing_ok=True)

    t = time.time()
    daily = load_station_daily()
    service = AlertService(out, build_models(), normal_offsets(daily), use_shap=not args.no_shap,
                           auto_correct=not args.no_correct, noise_ref=noise_reference(daily))
    consumer = KafkaConsumer(TOPIC_EVENTS, TOPIC_AGG, bootstrap_servers=BOOTSTRAP, auto_offset_reset="earliest",
                             group_id=args.group or f"rtap-alerts-{int(time.time())}", enable_auto_commit=True,
                             value_deserializer=lambda b: json.loads(b.decode("utf-8")))
    # ready only when the consumer group has joined and the partitions are assigned: otherwise the first seconds of the
    # stream wait in Kafka and look like "slow processing" (a cold-start artefact)
    early = []
    while not consumer.assignment() and time.time() - t < 120:
        for tp, recs in consumer.poll(timeout_ms=200).items():
            early += [(tp.topic, r.value) for r in recs]
    if early:
        service.on_events(sorted((v for tpc, v in early if tpc == TOPIC_EVENTS), key=lambda v: (v["date"], v["station"])))
    print(f"Сервис тревог готов за {time.time() - t:.1f} с (разделов назначено: {len(consumer.assignment())})", flush=True)
    (out / "ALERTS_READY").write_text("ok")
    last, got = time.time(), False
    n_events = n_agg = 0
    while True:
        polled = consumer.poll(timeout_ms=500, max_records=500)
        if polled:
            last, got = time.time(), True
            # records of one poll come partition by partition → order them by event time (date, then station), so that
            # day d of all stations is processed before day d+1 (needed for the neighbour-based rebuilding)
            events = sorted((r.value for tp, recs in polled.items() if tp.topic == TOPIC_EVENTS for r in recs),
                            key=lambda v: (v["date"], v["station"]))
            aggs = sorted((r.value for tp, recs in polled.items() if tp.topic == TOPIC_AGG for r in recs),
                          key=lambda v: str(v["window_start"]))
            if events or service.pending:
                service.on_events(events)
                n_events += len(events)
            if aggs:
                service.on_agg(aggs)
                n_agg += len(aggs)
            service.db.commit()
        elif service.pending:
            service.on_events([])
        elif got and time.time() - last > args.idle_seconds:
            break
        elif not got and time.time() - t > 900:
            break
    if service.pending:
        service.on_events([], final=True)
    for date in list(service.regional):          # dates where a station never reported: decide on the votes we have
        service.check_regional(date, final=True)
    service.db.commit()
    service.producer.flush()
    consumer.close()
    print(f"Обработано: {n_events} записей станций, {n_agg} региональных окон", flush=True)


if __name__ == "__main__":
    main()
