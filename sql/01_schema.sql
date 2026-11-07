-- =====================================================================================
-- 01_schema.sql — data warehouse schema (star schema) for weather data
--   stg  — staging area: raw data exactly as delivered by the source (Open-Meteo, ERA5)
--   dwh  — data warehouse: dimensions + fact table
-- The script is idempotent: it drops and recreates everything.
-- =====================================================================================

DROP SCHEMA IF EXISTS dwh CASCADE;
DROP SCHEMA IF EXISTS stg CASCADE;
CREATE SCHEMA stg;
CREATE SCHEMA dwh;

-- ------------------------------------------------------------------ staging
CREATE TABLE stg.open_meteo_daily (
    station_code                 TEXT    NOT NULL,
    date                         DATE    NOT NULL,
    weather_code                 SMALLINT,
    temperature_2m_max           DOUBLE PRECISION,
    temperature_2m_min           DOUBLE PRECISION,
    temperature_2m_mean          DOUBLE PRECISION,
    apparent_temperature_mean    DOUBLE PRECISION,
    dew_point_2m_mean            DOUBLE PRECISION,
    precipitation_sum            DOUBLE PRECISION,
    rain_sum                     DOUBLE PRECISION,
    snowfall_sum                 DOUBLE PRECISION,
    precipitation_hours          DOUBLE PRECISION,
    relative_humidity_2m_mean    DOUBLE PRECISION,
    relative_humidity_2m_max     DOUBLE PRECISION,
    relative_humidity_2m_min     DOUBLE PRECISION,
    pressure_msl_mean            DOUBLE PRECISION,
    pressure_msl_max             DOUBLE PRECISION,
    pressure_msl_min             DOUBLE PRECISION,
    cloud_cover_mean             DOUBLE PRECISION,
    wind_speed_10m_max           DOUBLE PRECISION,
    wind_gusts_10m_max           DOUBLE PRECISION,
    wind_direction_10m_dominant  DOUBLE PRECISION,
    shortwave_radiation_sum      DOUBLE PRECISION,
    sunshine_duration            DOUBLE PRECISION,
    daylight_duration            DOUBLE PRECISION,
    et0_fao_evapotranspiration   DOUBLE PRECISION,
    vapour_pressure_deficit_max  DOUBLE PRECISION,
    soil_moisture_0_to_7cm_mean  DOUBLE PRECISION,
    source_file                  TEXT    NOT NULL,
    loaded_at                    TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ------------------------------------------------------------------ dimensions
CREATE TABLE dwh.dim_station (
    station_id    SMALLSERIAL PRIMARY KEY,
    station_code  TEXT UNIQUE NOT NULL,      -- technical key used by the source files
    name_ru       TEXT NOT NULL,
    region_ru     TEXT NOT NULL,
    latitude      DOUBLE PRECISION NOT NULL,  -- requested coordinates
    longitude     DOUBLE PRECISION NOT NULL,
    grid_latitude DOUBLE PRECISION,           -- coordinates of the ERA5 grid cell actually returned
    grid_longitude DOUBLE PRECISION,
    elevation_m   DOUBLE PRECISION,
    is_reference  BOOLEAN NOT NULL DEFAULT FALSE   -- the station the model is built for (Chisinau)
);

CREATE TABLE dwh.dim_date (
    date_id       INTEGER PRIMARY KEY,        -- YYYYMMDD
    full_date     DATE UNIQUE NOT NULL,
    year          SMALLINT NOT NULL,
    quarter       SMALLINT NOT NULL,
    month         SMALLINT NOT NULL,
    month_name_ru TEXT NOT NULL,
    day_of_month  SMALLINT NOT NULL,
    day_of_year   SMALLINT NOT NULL,
    iso_week      SMALLINT NOT NULL,
    season_ru     TEXT NOT NULL,
    is_weekend    BOOLEAN NOT NULL
);

CREATE TABLE dwh.dim_weather_code (
    weather_code   SMALLINT PRIMARY KEY,      -- WMO code
    description_ru TEXT NOT NULL,
    group_ru       TEXT NOT NULL
);

-- ------------------------------------------------------------------ fact table
-- grain: one row = one station x one day
CREATE TABLE dwh.fact_weather_daily (
    station_id                   SMALLINT NOT NULL REFERENCES dwh.dim_station (station_id),
    date_id                      INTEGER  NOT NULL REFERENCES dwh.dim_date (date_id),
    weather_code                 SMALLINT REFERENCES dwh.dim_weather_code (weather_code),
    temperature_2m_max           DOUBLE PRECISION,
    temperature_2m_min           DOUBLE PRECISION,
    temperature_2m_mean          DOUBLE PRECISION,
    apparent_temperature_mean    DOUBLE PRECISION,
    dew_point_2m_mean            DOUBLE PRECISION,
    precipitation_sum            DOUBLE PRECISION CHECK (precipitation_sum >= 0),
    rain_sum                     DOUBLE PRECISION CHECK (rain_sum >= 0),
    snowfall_sum                 DOUBLE PRECISION CHECK (snowfall_sum >= 0),
    precipitation_hours          DOUBLE PRECISION CHECK (precipitation_hours BETWEEN 0 AND 24),
    relative_humidity_2m_mean    DOUBLE PRECISION CHECK (relative_humidity_2m_mean BETWEEN 0 AND 100),
    relative_humidity_2m_max     DOUBLE PRECISION CHECK (relative_humidity_2m_max BETWEEN 0 AND 100),
    relative_humidity_2m_min     DOUBLE PRECISION CHECK (relative_humidity_2m_min BETWEEN 0 AND 100),
    pressure_msl_mean            DOUBLE PRECISION CHECK (pressure_msl_mean BETWEEN 900 AND 1100),
    pressure_msl_max             DOUBLE PRECISION,
    pressure_msl_min             DOUBLE PRECISION,
    cloud_cover_mean             DOUBLE PRECISION CHECK (cloud_cover_mean BETWEEN 0 AND 100),
    wind_speed_10m_max           DOUBLE PRECISION CHECK (wind_speed_10m_max >= 0),
    wind_gusts_10m_max           DOUBLE PRECISION CHECK (wind_gusts_10m_max >= 0),
    wind_direction_10m_dominant  DOUBLE PRECISION CHECK (wind_direction_10m_dominant BETWEEN 0 AND 360),
    shortwave_radiation_sum      DOUBLE PRECISION,
    sunshine_duration            DOUBLE PRECISION,
    daylight_duration            DOUBLE PRECISION,
    et0_fao_evapotranspiration   DOUBLE PRECISION,
    vapour_pressure_deficit_max  DOUBLE PRECISION,
    soil_moisture_0_to_7cm_mean  DOUBLE PRECISION,
    PRIMARY KEY (station_id, date_id),
    CHECK (temperature_2m_min <= temperature_2m_mean AND temperature_2m_mean <= temperature_2m_max),
    CHECK (pressure_msl_min <= pressure_msl_mean AND pressure_msl_mean <= pressure_msl_max)
);

CREATE INDEX ix_fact_date ON dwh.fact_weather_daily (date_id);
