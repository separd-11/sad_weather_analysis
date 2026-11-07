-- =====================================================================================
-- 04_daily.sql — daily measurements of one station, in the original (source) layout.
-- Used where raw sensor values are needed (EDA, noise simulation, streaming).
-- Parameter: %(station_code)s
-- =====================================================================================
SELECT  d.full_date AS date,
        f.weather_code,
        f.temperature_2m_max, f.temperature_2m_min, f.temperature_2m_mean,
        f.apparent_temperature_mean, f.dew_point_2m_mean,
        f.precipitation_sum, f.rain_sum, f.snowfall_sum, f.precipitation_hours,
        f.relative_humidity_2m_mean, f.relative_humidity_2m_max, f.relative_humidity_2m_min,
        f.pressure_msl_mean, f.pressure_msl_max, f.pressure_msl_min,
        f.cloud_cover_mean,
        f.wind_speed_10m_max, f.wind_gusts_10m_max, f.wind_direction_10m_dominant,
        f.shortwave_radiation_sum, f.sunshine_duration, f.daylight_duration,
        f.et0_fao_evapotranspiration, f.vapour_pressure_deficit_max, f.soil_moisture_0_to_7cm_mean
FROM    dwh.fact_weather_daily f
JOIN    dwh.dim_date    d ON d.date_id    = f.date_id
JOIN    dwh.dim_station s ON s.station_id = f.station_id
WHERE   s.station_code = %(station_code)s
ORDER BY d.full_date;
