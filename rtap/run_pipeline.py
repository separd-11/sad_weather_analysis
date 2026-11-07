"""Runs the whole RTAP pipeline in the order of the lecture, each window as its own process:
  BAT1 — Kafka broker (KRaft)            rtap/kafka.sh start
  BAT2 — topics                          rtap/kafka.sh clean   (a clean replay: topics re-created)
  window 2 — Spark aggregator            rtap/spark_aggregator.py   (background, logs in <out>/spark.log)
  window 3 — alert service               rtap/alert_service.py      (background, logs in <out>/alerts.log)
  window 1 — the 6 stations              rtap/producer_stations.py  (foreground; starts when windows 2–3 are ready)
Window 4 (verdict and tuning) works on the results: notebook 10 and the app page "RTAP".
All unknown options are passed to the producer, e.g.:
  .venv/bin/python rtap/run_pipeline.py --out run --start 2024-01-01 --end 2025-12-31 --speed 0.05 \\
      --bias chisinau:6:2024-07-01 --late cahul:0.05:3 --drop comrat:0.02 --window-days 1
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PY = sys.executable


def wait_file(path: Path, timeout: float, proc) -> None:
    t = time.time()
    while not path.exists():
        if proc.poll() is not None:
            raise RuntimeError(f"процесс завершился раньше времени: {proc.args}")
        if time.time() - t > timeout:
            raise TimeoutError(path)
        time.sleep(0.5)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="run")
    ap.add_argument("--window-days", type=int, default=1)
    ap.add_argument("--slide-days", type=int, default=1)
    ap.add_argument("--watermark-days", type=int, default=2)
    ap.add_argument("--tp-window-sec", type=int, default=10)
    ap.add_argument("--trigger-sec", type=float, default=1.0)
    ap.add_argument("--no-shap", action="store_true")
    ap.add_argument("--no-correct", action="store_true")
    args, producer_args = ap.parse_known_args()
    out = ROOT / "data" / "rtap" / args.out
    out.mkdir(parents=True, exist_ok=True)

    for flag in ("READY", "ALERTS_READY"):          # stale flags of a previous run would start the stations too early
        (out / flag).unlink(missing_ok=True)
    subprocess.run([str(HERE / "kafka.sh"), "start"], check=True, capture_output=True)
    subprocess.run([str(HERE / "kafka.sh"), "clean"], check=True, capture_output=True)
    t0 = time.time()
    spark = subprocess.Popen([PY, str(HERE / "spark_aggregator.py"), "--out", args.out, "--window-days", str(args.window_days),
                              "--slide-days", str(args.slide_days), "--watermark-days", str(args.watermark_days),
                              "--tp-window-sec", str(args.tp_window_sec), "--trigger-sec", str(args.trigger_sec),
                              "--idle-seconds", "15"],
                             stdout=open(out / "spark.log", "w"), stderr=subprocess.STDOUT)
    alerts = subprocess.Popen([PY, str(HERE / "alert_service.py"), "--out", args.out] + (["--no-shap"] if args.no_shap else [])
                              + (["--no-correct"] if args.no_correct else []),
                              stdout=open(out / "alerts.log", "w"), stderr=subprocess.STDOUT)
    wait_file(out / "READY", 300, spark)
    wait_file(out / "ALERTS_READY", 300, alerts)
    print(f"Окна 2 и 3 готовы за {time.time() - t0:.0f} с; запускаю 6 станций", flush=True)
    t1 = time.time()
    prod = subprocess.run([PY, str(HERE / "producer_stations.py")] + producer_args, capture_output=True, text=True)
    print(prod.stdout.strip(), flush=True)
    if prod.returncode:
        print(prod.stderr[-2000:])
    spark.wait(timeout=1800)
    alerts.wait(timeout=1800)
    info = {"out": str(out), "producer_args": producer_args, "window_days": args.window_days, "slide_days": args.slide_days,
            "watermark_days": args.watermark_days, "tp_window_sec": args.tp_window_sec, "trigger_sec": args.trigger_sec, "no_correct": args.no_correct,
            "replay_seconds": round(time.time() - t1, 1), "spark_rc": spark.returncode, "alerts_rc": alerts.returncode}
    (out / "run_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=1))
    print(f"Готово за {time.time() - t0:.0f} с (Spark: код {spark.returncode}, тревоги: код {alerts.returncode}). "
          f"Последние строки журнала тревог:", flush=True)
    print("\n".join((out / "alerts.log").read_text().strip().splitlines()[-2:]))


if __name__ == "__main__":
    main()
