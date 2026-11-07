import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import (C_LEVEL, FEATURES, LEVELS, TARGET, TARGET3, WEATHER_TYPES, header, load_all, load_models,
                    source_badge, splits, weather_type)

header("🌦️ Прогноз на день", "Выберите вечер в тестовом периоде — система выдаёт прогноз и предупреждение на следующий день")
source_badge()
datasets, daily, _ = load_all()
rain_model, thr, crit_model, alerts = load_models()
_, _, test = splits(datasets["chisinau"])
raw = daily["chisinau"].set_index("date")

day = st.date_input("Дата наблюдения (вечер, когда выдаётся прогноз)", value=pd.Timestamp("2024-06-14"),
                    min_value=test.date.min(), max_value=test.date.max(), format="DD.MM.YYYY")
row = test[test.date == pd.Timestamp(day)]
if row.empty:
    st.stop()
f = row.iloc[0]
x = row[FEATURES]
p_rain = float(rain_model.predict_proba(x)[0, 1]); p_crit = float(crit_model.predict_proba(x)[0, 2])
level = alerts.level(p_rain, p_crit)

st.subheader(f"Погода {pd.Timestamp(day):%d.%m.%Y} (сегодня)")
c = st.columns(6)
c[0].metric("Давление", f"{f.pressure_msl_mean:.0f} гПа", f"{f.pressure_change_1d:+.1f} за сутки")
c[1].metric("Облачность", f"{f.cloud_cover_mean:.0f} %")
c[2].metric("Влажность", f"{f.relative_humidity_2m_mean:.0f} %")
c[3].metric("Температура", f"{f.temperature_2m_mean:.1f} °C")
c[4].metric("Осадки сегодня", f"{f.precipitation_sum:.1f} мм")
wd = raw.loc[pd.Timestamp(day), "wind_direction_10m_dominant"]
c[5].metric("Ветер", f"{f.wind_speed_10m_max:.0f} км/ч", f"направление {wd:.0f}°")

def gauge(value, title, thr_line):
    fig = go.Figure(go.Indicator(mode="gauge+number", value=value * 100, number={"suffix": " %"}, title={"text": title},
                                 gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#2B6CB0"},
                                        "threshold": {"line": {"color": "black", "width": 3}, "value": thr_line * 100}}))
    fig.update_layout(height=260, margin=dict(t=70, b=10, l=30, r=30))
    return fig

st.subheader(f"Прогноз на {pd.Timestamp(day) + pd.Timedelta(days=1):%d.%m.%Y}")
g1, g2, g3 = st.columns([1, 1, 1.4])
g1.plotly_chart(gauge(p_rain, "P(осадки ≥ 1 мм)", thr), width="stretch")
g2.plotly_chart(gauge(p_crit, "P(критическая погода)", alerts.thr_orange), width="stretch")
with g3:
    st.markdown(f"<div style='background:{C_LEVEL[level]};padding:18px;border-radius:10px;color:{'white' if level >= 2 else 'black'};"
                f"font-size:22px;font-weight:bold;text-align:center'>{LEVELS[level].upper()} КОД</div>", unsafe_allow_html=True)
    if level > 0:
        st.write(alerts.message(day, level, p_rain, p_crit, f))
    else:
        st.write("Предупреждение не выдаётся: признаки осадков слабые.")
    nxt = pd.Timestamp(day) + pd.Timedelta(days=1)
    if nxt in raw.index:
        real_type = int(weather_type(raw.loc[[nxt]]).iloc[0])
        st.markdown(f"**Что случилось на самом деле:** осадки {raw.loc[nxt, 'precipitation_sum']:.1f} мм, "
                    f"погода — *{WEATHER_TYPES[real_type].lower()}*.")

st.divider()
st.subheader("Что если? Измените сегодняшнюю погоду")
st.caption("Модель пересчитывает вероятность для изменённых признаков — так видно, на что она реагирует.")
w = st.columns(4)
dp = w[0].slider("Изменить давление, гПа", -15.0, 15.0, 0.0, 0.5)
dp3 = w[1].slider("Изменение давления за 3 дня, гПа", -20.0, 20.0, float(np.clip(f.pressure_change_3d, -20, 20)), 0.5)
cloud = w[2].slider("Облачность, %", 0, 100, int(f.cloud_cover_mean))
wind = w[3].select_slider("Направление ветра", options=[0, 45, 90, 135, 180, 225, 270, 315],
                          value=int(round(wd / 45) * 45) % 360,
                          format_func=lambda d: {0: "С", 45: "СВ", 90: "В", 135: "ЮВ", 180: "Ю", 225: "ЮЗ", 270: "З", 315: "СЗ"}[d])
x2 = x.copy()
x2["pressure_msl_mean"] += dp; x2["pressure_change_1d"] += dp; x2["pressure_change_3d"] = dp3
x2["cloud_cover_mean"] = cloud
x2["wind_dir_sin"], x2["wind_dir_cos"] = np.sin(np.deg2rad(wind)), np.cos(np.deg2rad(wind))
p2 = float(rain_model.predict_proba(x2)[0, 1])
r1, r2 = st.columns(2)
r1.metric("Вероятность осадков после изменений", f"{p2:.0%}", f"{(p2 - p_rain) * 100:+.0f} п. п.")
r2.metric("Решение", "дождь" if p2 >= thr else "без дождя", "изменилось" if (p2 >= thr) != (p_rain >= thr) else "не изменилось",
          delta_color="off")
