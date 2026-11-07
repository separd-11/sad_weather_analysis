import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from common import C_DRY, C_WET, FEATURES, MONTHS, TARGET, header, load_all, name, source_badge

header("📊 Данные: что предвещает дождь", "Кишинёв, 2015–2025 · по сегодняшним признакам и осадкам завтра")
source_badge()
datasets, daily, _ = load_all()
data = datasets["chisinau"]
lab = data[TARGET].map({0: "завтра без дождя", 1: "завтра дождь"})

c1, c2, c3 = st.columns(3)
c1.metric("Дней в наборе", len(data))
c2.metric("Доля дней «дождь завтра»", f"{data[TARGET].mean():.1%}")
pers = pd.crosstab(data.precipitation_sum >= 1, data[TARGET], normalize="index")
c3.metric("P(дождь завтра | дождь сегодня)", f"{pers.loc[True, 1]:.0%}", f"против {pers.loc[False, 1]:.0%} после сухого дня")

st.subheader("Признак сегодня ↔ осадки завтра")
default = FEATURES.index("pressure_msl_mean")
feat = st.selectbox("Признак", FEATURES, index=default, format_func=name)
c1, c2 = st.columns([1.4, 1])
fig = px.histogram(data.assign(Класс=lab), x=feat, color="Класс", barmode="overlay", histnorm="probability density",
                   nbins=50, opacity=0.55, color_discrete_map={"завтра без дождя": C_DRY, "завтра дождь": C_WET},
                   labels={feat: name(feat)})
fig.update_layout(title=f"Распределение: {name(feat)}", legend_title="")
c1.plotly_chart(fig, width="stretch")
fig = px.box(data.assign(Класс=lab), x="Класс", y=feat, color="Класс", points=False,
             color_discrete_map={"завтра без дождя": C_DRY, "завтра дождь": C_WET}, labels={feat: name(feat), "Класс": ""})
fig.update_layout(showlegend=False, title="Медианы и разброс")
c2.plotly_chart(fig, width="stretch")
med = data.groupby(TARGET)[feat].median()
st.info(f"Медиана перед сухим днём: **{med[0]:.2f}**, перед дождливым: **{med[1]:.2f}**.")

st.subheader("Сезонность")
c1, c2 = st.columns(2)
m = data.groupby(data.date.dt.month)[TARGET].mean() * 100
fig = go.Figure(go.Bar(x=MONTHS, y=m.values, marker_color=C_WET))
fig.add_hline(y=data[TARGET].mean() * 100, line_dash="dash", annotation_text="среднее за год")
fig.update_layout(title="Частота дней с осадками по месяцам, %", yaxis_title="%")
c1.plotly_chart(fig, width="stretch")
d = daily["chisinau"]
clim = d.groupby(d.date.dt.month).agg(t=("temperature_2m_mean", "mean"))
pr = d.groupby([d.date.dt.year, d.date.dt.month]).precipitation_sum.sum().groupby(level=1).mean()
fig = go.Figure()
fig.add_bar(x=MONTHS, y=pr.values, name="Осадки, мм", marker_color=C_WET, opacity=0.6)
fig.add_scatter(x=MONTHS, y=clim.t.values, name="Температура, °C", yaxis="y2", line=dict(color="#C0392B", width=3))
fig.update_layout(title="Климатограмма Кишинёва", yaxis=dict(title="мм"),
                  yaxis2=dict(title="°C", overlaying="y", side="right"), legend=dict(orientation="h"))
c2.plotly_chart(fig, width="stretch")

st.subheader("Роза ветров: откуда приходит дождь")
dirs = daily["chisinau"].set_index("date").wind_direction_10m_dominant.reindex(data.date).to_numpy()
bins = np.arange(0, 361, 22.5)
fig = go.Figure()
for cls, col, nm in [(0, C_DRY, "завтра без дождя"), (1, C_WET, "завтра дождь")]:
    h, _ = np.histogram(dirs[data[TARGET].to_numpy() == cls], bins=bins)
    fig.add_barpolar(r=h / h.sum() * 100, theta=bins[:-1] + 11.25, width=22.5, name=nm, marker_color=col, opacity=0.7)
fig.update_layout(polar=dict(angularaxis=dict(direction="clockwise", rotation=90,
                                              tickvals=[0, 90, 180, 270], ticktext=["С", "В", "Ю", "З"])),
                  title="Преобладающее направление ветра сегодня, % дней класса", height=500)
st.plotly_chart(fig, width="stretch")
st.caption("Ветер восточного–южного направления (влажный воздух с Чёрного моря) чаще предшествует осадкам; северо-западный — сухой погоде.")
