"""Window 1 of RTAP — distributed sources: 6 weather stations publish their daily records to Kafka.

Each station is an independent thread (an independent source) with its own Kafka key; the replay is accelerated
(--speed seconds of real time per simulated day). Faults can be injected to test the next windows:
  --bias  chisinau:6:2024-07-01            barometer of a station shifted by +6 hPa from a date
  --noise balti:0.5:2024-09-01:2024-10-31  Gaussian noise (σ × sensor std) on pressure, humidity, cloud cover, dew point
  --late  cahul:0.1:3                      10 % of messages of a station arrive 3 simulated days late
  --drop  comrat:0.05                      5 % of messages of a station are lost
  --drop-field ungheni:wind_direction_10m_dominant:2024-03-01   a station stops sending one measurement
Example:  .venv/bin/python rtap/producer_stations.py --start 2024-01-01 --end 2025-12-31 --speed 0.05
"""
from __future__ import annotations

import argparse
import threading
import time

import numpy as np
import pandas as pd
from kafka import KafkaProducer

from rtap_common import BOOTSTRAP, STATIONS, TOPIC_EVENTS, json_bytes, load_station_daily, to_message

NOISY = ["pressure_msl_mean", "pressure_msl_max", "pressure_msl_min", "relative_humidity_2m_mean", "cloud_cover_mean",
         "dew_point_2m_mean"]


def parse_faults(args):
    faults = {s: {} for s in STATIONS}
    for spec in args.bias or []:
        s, hpa, start = spec.split(":")
        faults[s]["bias"] = (float(hpa), pd.Timestamp(start))
    for spec in args.noise or []:
        s, sigma, start, end = spec.split(":")
        faults[s]["noise"] = (float(sigma), pd.Timestamp(start), pd.Timestamp(end))
    for spec in args.late or []:
        s, frac, days = spec.split(":")
        faults[s]["late"] = (float(frac), int(days))
    for spec in args.drop or []:
        s, frac = spec.split(":")
        faults[s]["drop"] = float(frac)
    for spec in args.drop_field or []:
        s, field, start = spec.split(":")
        faults[s]["drop_field"] = (field, pd.Timestamp(start))
    return faults


def apply_faults(rec: dict, f: dict, std: dict, rng) -> dict:
    day = pd.Timestamp(rec["date"])
    if "bias" in f and day >= f["bias"][1]:
        for c in ("pressure_msl_mean", "pressure_msl_max", "pressure_msl_min"):
            rec[c] += f["bias"][0]
    if "noise" in f and f["noise"][1] <= day <= f["noise"][2]:
        for c in NOISY:
            rec[c] += rng.normal(0, f["noise"][0] * std[c])
        rec["relative_humidity_2m_mean"] = float(np.clip(rec["relative_humidity_2m_mean"], 0, 100))
        rec["cloud_cover_mean"] = float(np.clip(rec["cloud_cover_mean"], 0, 100))
    if "drop_field" in f and day >= f["drop_field"][1]:
        rec[f["drop_field"][0]] = None
    return rec


def run_station(station, records, producer, t0, speed, f, std, seed, stats):
    rng = np.random.default_rng(seed)
    pending = []  # (send at day index, message) — late messages
    for i, rec in enumerate(records):
        wait = t0 + i * speed + rng.uniform(0, speed * 0.3) - time.time()  # small jitter: stations are not in lockstep
        if wait > 0:
            time.sleep(wait)
        for item in [p for p in pending if p[0] <= i]:
            producer.send(TOPIC_EVENTS, key=station.encode(), value=json_bytes({**item[1], "sent_at_ms": int(time.time() * 1000)}))
            pending.remove(item)
            stats[station]["late"] += 1
        if rng.random() < f.get("drop", 0):
            stats[station]["dropped"] += 1
            continue
        msg = to_message(station, apply_faults(dict(rec), f, std, rng), i, int(time.time() * 1000))
        if "late" in f and rng.random() < f["late"][0]:
            pending.append((i + f["late"][1], msg))
            continue
        producer.send(TOPIC_EVENTS, key=station.encode(), value=json_bytes(msg))
        stats[station]["sent"] += 1
    for _, msg in pending:  # flush messages still waiting at the end of the replay
        producer.send(TOPIC_EVENTS, key=station.encode(), value=json_bytes({**msg, "sent_at_ms": int(time.time() * 1000)}))
        stats[station]["late"] += 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--speed", type=float, default=0.05, help="real seconds per simulated day")
    ap.add_argument("--warmup", type=int, default=8, help="days before --start sent first (history for lags)")
    ap.add_argument("--stations", default=",".join(STATIONS))
    ap.add_argument("--bias", action="append")
    ap.add_argument("--noise", action="append")
    ap.add_argument("--late", action="append")
    ap.add_argument("--drop", action="append")
    ap.add_argument("--drop-field", action="append")
    ap.add_argument("--no-close", action="store_true", help="do not send the final heartbeat that closes the windows")
    args = ap.parse_args()

    daily = load_station_daily()
    faults = parse_faults(args)
    start = pd.Timestamp(args.start) - pd.Timedelta(days=args.warmup)
    producer = KafkaProducer(bootstrap_servers=BOOTSTRAP, linger_ms=5, acks=1)
    stats = {s: {"sent": 0, "late": 0, "dropped": 0} for s in STATIONS}
    t0 = time.time() + 0.5
    threads = []
    for k, s in enumerate(args.stations.split(",")):
        d = daily[s]
        recs = d[(d.date >= start) & (d.date <= args.end)].sort_values("date").to_dict("records")
        std = {c: float(d.loc[d.date < "2023-01-01", c].std()) for c in NOISY}
        th = threading.Thread(target=run_station, args=(s, recs, producer, t0, args.speed, faults[s], std, 100 + k, stats),
                              name=f"station-{s}", daemon=True)
        threads.append(th)
        th.start()
    for th in threads:
        th.join()
    if not args.no_close:
        # heartbeat with a date far after the end: advances Spark's watermark so that the last windows are emitted
        hb = {"station": "_heartbeat", "date": (pd.Timestamp(args.end) + pd.Timedelta(days=30)).strftime("%Y-%m-%d"),
              "seq": -1, "sent_at_ms": int(time.time() * 1000)}
        producer.send(TOPIC_EVENTS, key=b"_heartbeat", value=json_bytes(hb))
    producer.flush()
    producer.close()
    total = sum(v["sent"] + v["late"] for v in stats.values())
    print(f"Отправлено сообщений: {total} за {time.time() - t0:.1f} с")
    for s, v in stats.items():
        if s in args.stations.split(","):
            print(f"  {STATIONS[s]:8s}: вовремя {v['sent']}, с опозданием {v['late']}, потеряно {v['dropped']}")


if __name__ == "__main__":
    main()
