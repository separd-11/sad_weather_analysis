"""Access to the PostgreSQL data warehouse.

Connection string: env var SAD_DATABASE_URL, default — local PostgreSQL via the Unix socket, database `sad_weather`.
"""
import os
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine

ROOT = Path(__file__).resolve().parents[1]
SQL_DIR = ROOT / "sql"
DATABASE_URL = os.environ.get("SAD_DATABASE_URL", "postgresql+psycopg:///sad_weather")

_engine = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(DATABASE_URL)
    return _engine


def sql_file(name: str) -> str:
    return (SQL_DIR / name).read_text(encoding="utf-8")


def query(sql: str, params: dict | None = None) -> pd.DataFrame:
    """Run a SELECT and return a DataFrame. Parameters use the psycopg style: %(name)s."""
    with get_engine().connect() as conn:
        # a plain string is executed by pandas via exec_driver_sql, so the psycopg parameter style works
        return pd.read_sql_query(sql, conn, params=params)


def load_dataset(station_code: str = "chisinau") -> pd.DataFrame:
    """The modelling dataset: features + targets, built by the SQL query sql/03_dataset.sql."""
    df = query(sql_file("03_dataset.sql"), {"station_code": station_code})
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_daily(station_code: str = "chisinau") -> pd.DataFrame:
    """Daily measurements of one station in the source layout (sql/04_daily.sql)."""
    df = query(sql_file("04_daily.sql"), {"station_code": station_code})
    df["date"] = pd.to_datetime(df["date"])
    return df


def execute_script(sql: str) -> None:
    """Run a multi-statement SQL script (DDL / DML)."""
    with get_engine().begin() as conn:
        conn.exec_driver_sql(sql)
