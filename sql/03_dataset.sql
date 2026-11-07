-- =====================================================================================
-- 03_dataset.sql — THE query that produces the modelling dataset from the data warehouse
--
-- For one station it returns, per day t:
--   * 26 features computed from day t and the preceding days only (window functions LAG / SUM OVER),
--   * the targets for day t+1 (window function LEAD):
--       rain_tomorrow          — precipitation on day t+1 >= 1 mm (0/1),
--       weather_type_tomorrow  — 0 favourable / 1 unfavourable / 2 critical weather on day t+1.
-- Days without 6 days of history (first rows) and the last day (no "tomorrow") are excluded.
-- Parameter: %(station_code)s, e.g. 'chisinau'.
-- =====================================================================================
WITH daily AS (
    SELECT  d.full_date                                  AS date,
            d.day_of_year,
            f.*,
            -- weather type of the day (thresholds: see src/features.py)
            CASE WHEN f.precipitation_sum >= 10 OR f.wind_gusts_10m_max >= 60 OR f.temperature_2m_max >= 35
                      OR f.temperature_2m_min <= -15 OR f.snowfall_sum >= 5            THEN 2
                 WHEN f.precipitation_sum >= 1  OR f.wind_gusts_10m_max >= 45 OR f.temperature_2m_max >= 30
                      OR f.temperature_2m_min <= -10                                    THEN 1
                 ELSE 0 END                              AS weather_type
    FROM    dwh.fact_weather_daily f
    JOIN    dwh.dim_date    d ON d.date_id    = f.date_id
    JOIN    dwh.dim_station s ON s.station_id = f.station_id
    WHERE   s.station_code = %(station_code)s
),
features AS (
    SELECT  date,
            -- today's state
            temperature_2m_mean,
            temperature_2m_max - temperature_2m_min                         AS temp_range,
            dew_point_2m_mean,
            temperature_2m_mean - dew_point_2m_mean                         AS dew_point_depression,
            relative_humidity_2m_mean,
            relative_humidity_2m_max,
            pressure_msl_mean,
            pressure_msl_max - pressure_msl_min                             AS pressure_intraday_range,
            cloud_cover_mean,
            sunshine_duration / daylight_duration                           AS sunshine_ratio,
            precipitation_sum,
            precipitation_hours,
            snowfall_sum,
            wind_speed_10m_max,
            SIN(RADIANS(wind_direction_10m_dominant))                       AS wind_dir_sin,
            COS(RADIANS(wind_direction_10m_dominant))                       AS wind_dir_cos,
            soil_moisture_0_to_7cm_mean,
            vapour_pressure_deficit_max,
            -- dynamics (trends and memory)
            pressure_msl_mean - LAG(pressure_msl_mean, 1) OVER w            AS pressure_change_1d,
            pressure_msl_mean - LAG(pressure_msl_mean, 3) OVER w            AS pressure_change_3d,
            LAG(cloud_cover_mean, 1) OVER w                                 AS cloud_cover_lag1,
            relative_humidity_2m_mean - LAG(relative_humidity_2m_mean, 1) OVER w AS humidity_change_1d,
            SUM(precipitation_sum) OVER (w ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)                AS precip_sum_3d,
            SUM((precipitation_sum >= 1)::INT) OVER (w ROWS BETWEEN 6 PRECEDING AND CURRENT ROW)::DOUBLE PRECISION AS precip_days_7d,
            -- season (cyclic encoding of the day of year)
            SIN(2 * PI() * day_of_year / 365.25)                            AS doy_sin,
            COS(2 * PI() * day_of_year / 365.25)                            AS doy_cos,
            -- targets: what happens tomorrow
            LEAD(precipitation_sum, 1) OVER w                               AS precipitation_next,
            LEAD(weather_type, 1) OVER w                                    AS weather_type_next,
            ROW_NUMBER() OVER w                                             AS day_number
    FROM    daily
    WINDOW  w AS (ORDER BY date)
)
SELECT  date,
        temperature_2m_mean, temp_range, dew_point_2m_mean, dew_point_depression,
        relative_humidity_2m_mean, relative_humidity_2m_max,
        pressure_msl_mean, pressure_intraday_range,
        cloud_cover_mean, sunshine_ratio,
        precipitation_sum, precipitation_hours, snowfall_sum,
        wind_speed_10m_max, wind_dir_sin, wind_dir_cos,
        soil_moisture_0_to_7cm_mean, vapour_pressure_deficit_max,
        pressure_change_1d, pressure_change_3d,
        cloud_cover_lag1, humidity_change_1d,
        precip_sum_3d, precip_days_7d,
        doy_sin, doy_cos,
        (precipitation_next >= 1)::INT    AS rain_tomorrow,
        weather_type_next                 AS weather_type_tomorrow
FROM    features
WHERE   day_number >= 7                   -- full 7-day history available
  AND   precipitation_next IS NOT NULL    -- "tomorrow" is known
ORDER BY date;
