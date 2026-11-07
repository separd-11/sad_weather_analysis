import json
import sqlite3
import time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import C_LEVEL, FIG, LEVELS, ROOT, WEATHER_TYPES, header, load_all, source_badge, weather_type

header("📡 Поток и тревоги (Streaming, RTAP)",
       "Записи станции обрабатываются по одной: признаки → прогноз → предупреждение → проверка на следующий день")
source_badge()
DB = ROOT / "data" / "stream" / "weather_stream.db"
LOG = ROOT / "data" / "stream" / "notifications.log"
if not DB.exists():
    st.error("Нет результатов потока: запустите ноутбук 04_Streaming_RTAP.ipynb.")
    st.stop()

_, daily, _ = load_all()
raw = daily["chisinau"].set_index("date")


@st.cache_data
def load_stream():
    con = sqlite3.connect(DB)
    q = lambda m: pd.read_sql_query("SELECT ts, fields FROM points WHERE measurement = ? ORDER BY ts", con, params=[m])
    def expand(df):
        out = pd.json_normalize(df.fields.map(json.loads))
        out.insert(0, "date", pd.to_datetime(df.ts.str[:10]))
        return out
    fc, oc = expand(q("forecasts")), expand(q("outcomes"))
    con.close()
    return fc, oc

fc, oc = load_stream()
fc["next_day"] = fc.date + pd.Timedelta(days=1)
fc["precip_next"] = fc.next_day.map(raw.precipitation_sum)
types = weather_type(raw.reset_index()).set_axis(raw.index)
fc["type_next"] = fc.next_day.map(types)

c = st.columns(5)
c[0].metric("Обработано записей", len(fc))
c[1].metric("Оценено прогнозов", len(oc))
for i, lvl in enumerate([1, 2, 3]):
    c[2 + i].metric(f"{LEVELS[lvl]} код", int((fc.level == lvl).sum()))

lo, hi = fc.date.min().date(), fc.date.max().date()
rng = st.slider("Период", min_value=lo, max_value=hi, value=(pd.Timestamp("2024-04-01").date(), pd.Timestamp("2024-07-31").date()),
                format="DD.MM.YYYY")
w = fc[(fc.date >= pd.Timestamp(rng[0])) & (fc.date <= pd.Timestamp(rng[1]))]
fig = go.Figure()
fig.add_bar(x=w.next_day, y=w.precip_next, name="Фактические осадки, мм", marker_color="#2B6CB0", opacity=0.35, yaxis="y2")
fig.add_scatter(x=w.next_day, y=w.p_rain, name="P(осадки), выдана накануне", line=dict(color="black"))
for lvl in (1, 2, 3):
    m = w.level == lvl
    fig.add_scatter(x=w.next_day[m], y=w.p_rain[m], mode="markers", name=f"{LEVELS[lvl]} код",
                    marker=dict(color=C_LEVEL[lvl], size=11, line=dict(color="black", width=1)))
fig.update_layout(height=480, yaxis=dict(title="Вероятность", range=[0, 1]),
                  yaxis2=dict(title="мм", overlaying="y", side="right", showgrid=False), legend=dict(orientation="h"),
                  title="Прогнозы потока и фактические осадки")
st.plotly_chart(fig, width="stretch")

st.subheader("▶ Проигрывание потока")
c1, c2, c3 = st.columns([1, 1, 2])
start = c1.date_input("С какого дня", value=pd.Timestamp("2024-06-08"), min_value=lo, max_value=hi, format="DD.MM.YYYY")
n_days = c2.slider("Сколько дней", 5, 30, 10)
speed = c3.slider("Пауза между записями, с", 0.1, 2.0, 0.6, 0.1)
if st.button("▶ Запустить поток", type="primary"):
    box = st.empty()
    seq = fc[fc.date >= pd.Timestamp(start)].head(n_days)
    log = []
    for _, r in seq.iterrows():
        obs = raw.loc[r.date]
        actual = "—" if pd.isna(r.precip_next) else f"{r.precip_next:.1f} мм, {WEATHER_TYPES[int(r.type_next)].lower()}"
        log.append({"Дата записи": r.date.strftime("%d.%m.%Y"), "Давление, гПа": round(obs.pressure_msl_mean, 1),
                    "Облачность, %": int(obs.cloud_cover_mean), "Осадки сегодня, мм": obs.precipitation_sum,
                    "P(осадки завтра)": f"{r.p_rain:.0%}", "P(критич.)": f"{r.p_crit:.0%}", "Код": LEVELS[int(r.level)],
                    "Что было завтра": actual})
        with box.container():
            st.markdown(f"<div style='background:{C_LEVEL[int(r.level)]};padding:10px;border-radius:8px;color:"
                        f"{'white' if r.level >= 2 else 'black'};font-size:18px'>📥 {r.date:%d.%m.%Y}: запись получена → "
                        f"P(осадки) = {r.p_rain:.0%} → <b>{LEVELS[int(r.level)].upper()} КОД</b></div>", unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(log), hide_index=True, width="stretch")
        time.sleep(speed)

st.subheader("Журнал уведомлений")
if LOG.exists():
    lines = [l.rstrip() for l in LOG.read_text(encoding="utf-8").splitlines() if l.strip()]
    df = pd.DataFrame({"Запись": lines})
    df["Тип"] = df.Запись.str.extract(r"\[(\w+)\]")[0].map({"ALERT": "тревога", "TECHNICAL": "техническое"})
    kind = st.radio("Показать", ["тревога", "техническое"], horizontal=True)
    st.dataframe(df[df.Тип == kind].Запись.str.replace(r"\[\w+\] ", "", regex=True).to_frame("Уведомление").head(60),
                 hide_index=True, width="stretch")

with st.expander("Контроль датчиков и адаптация модели (из ноутбука 04)"):
    st.image(str(FIG / "fig_5_5_sensor_faults.png"), width="stretch")
    st.image(str(FIG / "fig_5_6_adaptation.png"), width="stretch")
