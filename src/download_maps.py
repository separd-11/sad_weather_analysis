"""Download daily weather MAPS over Moldova (ERA5, 0.25° grid) — the "images" of lab 4 (noise in images).

Grid: latitude 45.5 … 48.5, longitude 26.75 … 30.25 → 13 × 15 = 195 points; 2024 (366 days); 6 variables.
The free API counts a request as (locations) × (two-week periods) calls, so the grid goes in batches of 20 points
with pauses and retries on HTTP 429.
Run:     .venv/bin/python src/download_maps.py
Output:  data/raw/maps/era5_maps_2024.npz  (arrays [variable][day, lat, lon] + lat, lon, dates)
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "maps" / "era5_maps_2024.npz"
URL = "https://archive-api.open-meteo.com/v1/archive"
LATS = np.round(np.arange(48.5, 45.49, -0.25), 2)      # north at the top of the image
LONS = np.round(np.arange(26.75, 30.26, 0.25), 2)
VARS = ["temperature_2m_mean", "pressure_msl_mean", "cloud_cover_mean", "precipitation_sum", "relative_humidity_2m_mean",
        "wind_speed_10m_max"]
START, END, BATCH = "2024-01-01", "2024-12-31", 20


def fetch(points):
    params = {"latitude": ",".join(str(p[0]) for p in points), "longitude": ",".join(str(p[1]) for p in points),
              "start_date": START, "end_date": END, "daily": ",".join(VARS), "timezone": "Europe/Chisinau", "models": "era5"}
    for attempt in range(10):
        r = requests.get(URL, params=params, timeout=120)
        if r.status_code != 429:
            r.raise_for_status()
            js = r.json()
            return js if isinstance(js, list) else [js]
        print(f"  лимит API, жду 65 с (попытка {attempt + 1})", flush=True)
        time.sleep(65)
    raise RuntimeError("API limit")


def main():
    if OUT.exists():
        print(f"Уже есть: {OUT.relative_to(ROOT)}")
        return
    points = [(la, lo) for la in LATS for lo in LONS]
    dates = pd.date_range(START, END)
    cube = {v: np.full((len(dates), len(LATS), len(LONS)), np.nan) for v in VARS}
    for b in range(0, len(points), BATCH):
        chunk = points[b:b + BATCH]
        for (la, lo), js in zip(chunk, fetch(chunk)):
            i, j = int(np.where(LATS == la)[0][0]), int(np.where(LONS == lo)[0][0])
            for v in VARS:
                cube[v][:, i, j] = np.array(js["daily"][v], dtype=float)
        print(f"  точки {b + 1}–{b + len(chunk)} из {len(points)}", flush=True)
        time.sleep(8)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, lat=LATS, lon=LONS, dates=dates.strftime("%Y-%m-%d").to_numpy(), **cube)
    print(f"Сохранено: {OUT.relative_to(ROOT)} — {len(dates)} дней × {len(LATS)}×{len(LONS)} × {len(VARS)} переменных; "
          f"пропусков: {sum(int(np.isnan(c).sum()) for c in cube.values())}")


if __name__ == "__main__":
    main()
