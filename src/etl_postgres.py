"""ETL: Open-Meteo landing files (data/raw) → PostgreSQL staging (stg) → star-schema warehouse (dwh).

Extract   — src/download_data.py has already pulled the data from the Open-Meteo API into data/raw/*.csv
Transform — sql/02_transform.sql (dimensions + fact table)
Load      — this script (schema creation, COPY into staging, station dimension, quality checks)

Run:  .venv/bin/python src/etl_postgres.py
"""
import time
from pathlib import Path

import pandas as pd

from db import execute_script, get_engine, query, sql_file

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

# station metadata; grid coordinates and elevation are those returned by the Open-Meteo API for each request
STATIONS = [
    # code,      name_ru,     region_ru,            lat,   lon,   grid_lat, grid_lon, elevation, is_reference
    ("chisinau", "Кишинёв",  "Центр",              47.01, 28.86, 47.07,    28.83,    48,  True),
    ("soroca",   "Сороки",   "Север (Днестр)",     48.16, 28.30, 48.19,    28.25,    72,  False),
    ("balti",    "Бельцы",   "Север",              47.76, 27.93, 47.77,    27.97,    118, False),
    ("ungheni",  "Унгены",   "Запад (Прут)",       47.21, 27.80, 47.21,    27.75,    67,  False),
    ("comrat",   "Комрат",   "Юг (Гагаузия)",      46.30, 28.66, 46.29,    28.61,    61,  False),
    ("cahul",    "Кагул",    "Юго-запад",          45.90, 28.19, 45.87,    28.20,    40,  False),
]


def landing_file(code: str) -> Path:
    return RAW / "chisinau_daily_2015_2025.csv" if code == "chisinau" else RAW / "cities" / f"{code}_daily_2015_2025.csv"


def run_etl(verbose: bool = True) -> pd.DataFrame:
    t0 = time.perf_counter()
    log = lambda msg: print(msg) if verbose else None

    execute_script(sql_file("01_schema.sql"))
    log("1. Schema created: stg, dwh")

    # load the landing files into staging with COPY (the fastest bulk-load path in PostgreSQL)
    raw_conn = get_engine().raw_connection()
    try:
        cur = raw_conn.cursor()
        for code, *_ in STATIONS:
            path = landing_file(code)
            df = pd.read_csv(path)
            cols = list(df.columns)
            with cur.copy(f"COPY stg.open_meteo_daily (station_code, source_file, {', '.join(cols)}) FROM STDIN") as copy:
                for row in df.itertuples(index=False):
                    copy.write_row([code, path.name, *[None if pd.isna(v) else v for v in row]])
            log(f"2. Staging: {code:9s} ← {path.relative_to(ROOT)} ({len(df)} rows)")
        cur.executemany(
            "INSERT INTO dwh.dim_station (station_code, name_ru, region_ru, latitude, longitude, grid_latitude, "
            "grid_longitude, elevation_m, is_reference) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)", STATIONS)
        raw_conn.commit()
    finally:
        raw_conn.close()
    log("3. dim_station filled")

    execute_script(sql_file("02_transform.sql"))
    counts = query("""
        SELECT 'stg.open_meteo_daily' AS "Таблица", COUNT(*) AS "Строк" FROM stg.open_meteo_daily
        UNION ALL SELECT 'dwh.dim_station',       COUNT(*) FROM dwh.dim_station
        UNION ALL SELECT 'dwh.dim_date',          COUNT(*) FROM dwh.dim_date
        UNION ALL SELECT 'dwh.dim_weather_code',  COUNT(*) FROM dwh.dim_weather_code
        UNION ALL SELECT 'dwh.fact_weather_daily', COUNT(*) FROM dwh.fact_weather_daily""")
    log(f"4. Transform done in {time.perf_counter() - t0:.1f} s")
    return counts


if __name__ == "__main__":
    print(run_etl().to_string(index=False))
