-- =====================================================================================
-- 02_transform.sql — transform the staging data into the star schema (T and L of ETL)
-- Expects: stg.open_meteo_daily filled, dwh.dim_station filled (by the ETL script).
-- =====================================================================================

-- ------------------------------------------------------------------ dim_date
INSERT INTO dwh.dim_date
SELECT  TO_CHAR(d, 'YYYYMMDD')::INT                         AS date_id,
        d::DATE                                             AS full_date,
        EXTRACT(YEAR FROM d)::SMALLINT                      AS year,
        EXTRACT(QUARTER FROM d)::SMALLINT                   AS quarter,
        EXTRACT(MONTH FROM d)::SMALLINT                     AS month,
        (ARRAY['Январь','Февраль','Март','Апрель','Май','Июнь','Июль','Август',
               'Сентябрь','Октябрь','Ноябрь','Декабрь'])[EXTRACT(MONTH FROM d)::INT] AS month_name_ru,
        EXTRACT(DAY FROM d)::SMALLINT                       AS day_of_month,
        EXTRACT(DOY FROM d)::SMALLINT                       AS day_of_year,
        EXTRACT(WEEK FROM d)::SMALLINT                      AS iso_week,
        CASE WHEN EXTRACT(MONTH FROM d) IN (12, 1, 2) THEN 'Зима'
             WHEN EXTRACT(MONTH FROM d) IN (3, 4, 5)  THEN 'Весна'
             WHEN EXTRACT(MONTH FROM d) IN (6, 7, 8)  THEN 'Лето'
             ELSE 'Осень' END                               AS season_ru,
        EXTRACT(ISODOW FROM d) IN (6, 7)                    AS is_weekend
FROM generate_series((SELECT MIN(date) FROM stg.open_meteo_daily),
                     (SELECT MAX(date) FROM stg.open_meteo_daily),
                     INTERVAL '1 day') AS d;

-- ------------------------------------------------------------------ dim_weather_code (WMO 4677, as used by Open-Meteo)
INSERT INTO dwh.dim_weather_code VALUES
    (0,  'Ясно',                              'Без осадков'),
    (1,  'Преимущественно ясно',              'Без осадков'),
    (2,  'Переменная облачность',             'Без осадков'),
    (3,  'Пасмурно',                          'Без осадков'),
    (45, 'Туман',                             'Туман'),
    (48, 'Туман с изморозью',                 'Туман'),
    (51, 'Слабая морось',                     'Морось'),
    (53, 'Умеренная морось',                  'Морось'),
    (55, 'Сильная морось',                    'Морось'),
    (56, 'Слабая ледяная морось',             'Морось'),
    (57, 'Сильная ледяная морось',            'Морось'),
    (61, 'Слабый дождь',                      'Дождь'),
    (63, 'Умеренный дождь',                   'Дождь'),
    (65, 'Сильный дождь',                     'Дождь'),
    (66, 'Слабый ледяной дождь',              'Дождь'),
    (67, 'Сильный ледяной дождь',             'Дождь'),
    (71, 'Слабый снегопад',                   'Снег'),
    (73, 'Умеренный снегопад',                'Снег'),
    (75, 'Сильный снегопад',                  'Снег'),
    (77, 'Снежные зёрна',                     'Снег'),
    (80, 'Слабый ливень',                     'Ливень'),
    (81, 'Умеренный ливень',                  'Ливень'),
    (82, 'Сильный ливень',                    'Ливень'),
    (85, 'Слабый снежный ливень',             'Снег'),
    (86, 'Сильный снежный ливень',            'Снег'),
    (95, 'Гроза',                             'Гроза'),
    (96, 'Гроза со слабым градом',            'Гроза'),
    (99, 'Гроза с сильным градом',            'Гроза');

-- ------------------------------------------------------------------ fact_weather_daily
INSERT INTO dwh.fact_weather_daily
SELECT  s.station_id,
        TO_CHAR(g.date, 'YYYYMMDD')::INT,
        g.weather_code,
        g.temperature_2m_max, g.temperature_2m_min, g.temperature_2m_mean,
        g.apparent_temperature_mean, g.dew_point_2m_mean,
        g.precipitation_sum, g.rain_sum, g.snowfall_sum, g.precipitation_hours,
        g.relative_humidity_2m_mean, g.relative_humidity_2m_max, g.relative_humidity_2m_min,
        g.pressure_msl_mean, g.pressure_msl_max, g.pressure_msl_min,
        g.cloud_cover_mean,
        g.wind_speed_10m_max, g.wind_gusts_10m_max, g.wind_direction_10m_dominant,
        g.shortwave_radiation_sum, g.sunshine_duration, g.daylight_duration,
        g.et0_fao_evapotranspiration, g.vapour_pressure_deficit_max, g.soil_moisture_0_to_7cm_mean
FROM stg.open_meteo_daily g
JOIN dwh.dim_station s ON s.station_code = g.station_code;

ANALYZE dwh.dim_date;
ANALYZE dwh.dim_station;
ANALYZE dwh.fact_weather_daily;
