"""Download daily weather data from the Open-Meteo archive (ERA5 reanalysis).

Run:     .venv/bin/python src/download_data.py
Output:  data/raw/chisinau_daily_2015_2025.csv (main dataset)
         data/raw/cities/<city>_daily_2015_2025.csv (other Moldovan towns, for the geographic fairness test)

New data (never seen by the model, for the drift / "new data" lab):
         .venv/bin/python src/download_data.py --new
         → data/raw/new/chisinau_daily_2026.csv (from 2026-01-01 to the last day available in the archive)
"""
import argparse
from pathlib import Path

import pandas as pd
import requests
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "chisinau_daily_2015_2025.csv"
CITIES_DIR = ROOT / "data" / "raw" / "cities"

# (latitude, longitude) — Chisinau (centre) + one town from each region of the country
LOCATIONS = {
    "chisinau": (47.01, 28.86),
    "soroca": (48.16, 28.30),   # north-east, on the Dniester
    "balti": (47.76, 27.93),    # north
    "ungheni": (47.21, 27.80),  # west, on the Prut
    "comrat": (46.30, 28.66),   # south (Gagauzia)
    "cahul": (45.90, 28.19),    # south-west
}

URL = "https://archive-api.open-meteo.com/v1/archive"
PARAMS = {
    "start_date": "2015-01-01",
    "end_date": "2025-12-31",
    "timezone": "Europe/Chisinau",
    "daily": ",".join([
        "weather_code",
        "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
        "apparent_temperature_mean", "dew_point_2m_mean",
        "precipitation_sum", "rain_sum", "snowfall_sum", "precipitation_hours",
        "relative_humidity_2m_mean", "relative_humidity_2m_max", "relative_humidity_2m_min",
        "pressure_msl_mean", "pressure_msl_max", "pressure_msl_min",
        "cloud_cover_mean",
        "wind_speed_10m_max", "wind_gusts_10m_max", "wind_direction_10m_dominant",
        "shortwave_radiation_sum", "sunshine_duration", "daylight_duration",
        "et0_fao_evapotranspiration", "vapour_pressure_deficit_max",
        "soil_moisture_0_to_7cm_mean",
    ]),
}


def download(lat: float, lon: float, out: Path, retries: int = 6, start: str | None = None, end: str | None = None) -> None:
    # one request for 11 years x 26 variables "costs" hundreds of calls of the free API quota (600/minute),
    # so on HTTP 429 (Too Many Requests) we wait a minute and retry
    for attempt in range(retries):
        period = {k: v for k, v in (("start_date", start), ("end_date", end)) if v}
        resp = requests.get(URL, params={**PARAMS, **period, "latitude": lat, "longitude": lon}, timeout=60)
        if resp.status_code != 429:
            break
        print(f"  API rate limit hit, waiting 65 s (attempt {attempt + 1}/{retries})")
        time.sleep(65)
    resp.raise_for_status()
    payload = resp.json()

    df = pd.DataFrame(payload["daily"]).rename(columns={"time": "date"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(how="any")   # the most recent days of the archive may not be published yet

    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} days x {df.shape[1]} columns -> {out.relative_to(ROOT)}  "
          f"(grid {payload['latitude']:.2f}, {payload['longitude']:.2f}, elevation {payload['elevation']:.0f} m)")


def main() -> None:
    for name, (lat, lon) in LOCATIONS.items():
        out = OUT if name == "chisinau" else CITIES_DIR / f"{name}_daily_2015_2025.csv"
        if out.exists():
            print(f"Already exists: {out.relative_to(ROOT)}")
            continue
        download(lat, lon, out)


def main_new() -> None:
    end = (pd.Timestamp.today().normalize() - pd.Timedelta(days=1)).date().isoformat()
    download(*LOCATIONS["chisinau"], ROOT / "data" / "raw" / "new" / "chisinau_daily_2026.csv", start="2026-01-01", end=end)


if __name__ == "__main__":
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument("--new", action="store_true", help="download the 2026 data for Chisinau (new data for the drift lab)")
    main_new() if args.parse_args().new else main()
