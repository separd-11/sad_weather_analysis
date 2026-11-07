import pandas as pd
import plotly.express as px
import streamlit as st

from common import FIG, ROOT, SEASONS, STATIONS, db_available, header, load_all, source_badge

header("🗄️ Хранилище данных — этап 1", "PostgreSQL · схема «звезда» · обучающий набор получается SQL-запросом")
source_badge()
datasets, daily, _ = load_all()
online = db_available()

c1, c2 = st.columns([1.2, 1])
with c1:
    st.image(str(FIG / "fig_1_2_star_schema.png"), width="stretch")
with c2:
    st.subheader("Содержимое хранилища")
    if online:
        from db import query
        counts = query("""
            SELECT 'dwh.fact_weather_daily' AS "Таблица", COUNT(*) AS "Строк" FROM dwh.fact_weather_daily
            UNION ALL SELECT 'dwh.dim_station', COUNT(*) FROM dwh.dim_station
            UNION ALL SELECT 'dwh.dim_date', COUNT(*) FROM dwh.dim_date
            UNION ALL SELECT 'dwh.dim_weather_code', COUNT(*) FROM dwh.dim_weather_code
            UNION ALL SELECT 'stg.open_meteo_daily', COUNT(*) FROM stg.open_meteo_daily""")
        st.dataframe(counts, hide_index=True, width="stretch")
        st.dataframe(query("SELECT station_code AS \"Код\", name_ru AS \"Станция\", region_ru AS \"Регион\", "
                           "latitude AS \"Широта\", longitude AS \"Долгота\", elevation_m AS \"Высота, м\" "
                           "FROM dwh.dim_station ORDER BY station_id"), hide_index=True, width="stretch")
    else:
        st.warning("База данных сейчас недоступна — показаны данные из резервных CSV-файлов.")

st.subheader("Главный запрос этапа 1: `sql/03_dataset.sql`")
st.caption("Оконные функции LAG / LEAD / SUM OVER строят 26 признаков сегодняшнего дня и целевые переменные «завтра».")
sql_text = (ROOT / "sql" / "03_dataset.sql").read_text(encoding="utf-8")
with st.expander("Показать текст запроса", expanded=False):
    st.code(sql_text, language="sql")
cs, cb = st.columns([1, 3])
station = cs.selectbox("Станция", list(STATIONS), format_func=STATIONS.get)
if cs.button("▶ Выполнить запрос", type="primary"):
    if online:
        from db import load_dataset
        res = load_dataset(station)
        cb.success(f"Запрос выполнен в PostgreSQL: {len(res)} строк × {res.shape[1]} столбцов")
    else:
        res = datasets[station]
        cb.info(f"Результат из резервной копии: {len(res)} строк × {res.shape[1]} столбцов")
    cb.dataframe(res.head(15), width="stretch")

st.subheader("Аналитические срезы (OLAP)")
rows = []
for s, d in daily.items():
    dd = d.assign(season=d.date.dt.month.map(SEASONS), rain=(d.precipitation_sum >= 1).astype(int))
    for season, g in dd.groupby("season"):
        rows.append({"Станция": STATIONS[s], "Сезон": season, "% дней с осадками": g.rain.mean() * 100})
freq = pd.DataFrame(rows).pivot(index="Станция", columns="Сезон", values="% дней с осадками")
freq = freq.loc[[STATIONS[s] for s in STATIONS], ["Зима", "Весна", "Лето", "Осень"]]
c1, c2 = st.columns(2)
fig = px.imshow(freq.round(1), text_auto=True, color_continuous_scale="Blues", aspect="auto",
                title="Частота дней с осадками ≥ 1 мм, % (станция × сезон)")
c1.plotly_chart(fig, width="stretch")
crit = []
for s, d in daily.items():
    crit.append({"Станция": STATIONS[s], "Осадки ≥ 10 мм": (d.precipitation_sum >= 10).sum(),
                 "Порывы ≥ 60 км/ч": (d.wind_gusts_10m_max >= 60).sum(), "Жара ≥ 35 °C": (d.temperature_2m_max >= 35).sum(),
                 "Мороз ≤ −15 °C": (d.temperature_2m_min <= -15).sum(), "Снег ≥ 5 см": (d.snowfall_sum >= 5).sum()})
crit = pd.DataFrame(crit).melt(id_vars="Станция", var_name="Причина", value_name="Дней")
fig = px.bar(crit, x="Станция", y="Дней", color="Причина", title="Дни с критическими явлениями по станциям, 2015–2025")
c2.plotly_chart(fig, width="stretch")
