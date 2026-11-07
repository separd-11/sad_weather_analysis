import json
import sqlite3
import subprocess
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import C_LEVEL, COORDS, FIG, LEVELS, ROOT, STATIONS, header

sys.path.insert(0, str(ROOT / "rtap"))
RTAP = ROOT / "data" / "rtap"
PY = str(ROOT / ".venv" / "bin" / "python")

header("🛰️ RTAP: Kafka + Spark (лекция 5)",
       "6 станций → Kafka → Spark (окна) → решение с SHAP → вердикт. Каждое окно — отдельный процесс, как в лекции")


def kafka_up() -> bool:
    r = subprocess.run([str(ROOT / "rtap" / "kafka.sh"), "status"], capture_output=True, text=True)
    return "running" in r.stdout


def read_db(run: str):
    db = RTAP / run / "alerts.db"
    if not db.exists():
        return None, None, None
    con = sqlite3.connect(db)
    try:
        out = tuple(pd.read_sql(f"SELECT * FROM {t}", con) for t in ("forecasts", "regional", "sensor"))
    except Exception:
        out = (None, None, None)
    con.close()
    return out


tab_live, tab_runs, tab_tune = st.tabs(["▶ Живой поток", "Результаты прогонов (ноутбук 10)", "Окно 4: настройка и вердикт"])

# ---------------------------------------------------------------- live
with tab_live:
    st.markdown("Запуск всего конвейера из приложения: `rtap/run_pipeline.py` поднимает брокер Kafka (KRaft), пересоздаёт "
                "топики, запускает Spark и сервис тревог, затем 6 станций. Дашборд ниже обновляется каждые 2 секунды.")
    c1, c2, c3 = st.columns(3)
    period = c1.date_input("Период воспроизведения", value=(pd.Timestamp("2024-06-01").date(), pd.Timestamp("2024-08-31").date()),
                           min_value=pd.Timestamp("2015-02-01").date(), max_value=pd.Timestamp("2025-12-31").date(), format="DD.MM.YYYY")
    speed = c2.select_slider("Секунд реального времени на сутки", [0.05, 0.1, 0.2, 0.5, 1.0], value=0.2)
    c3.metric("Брокер Kafka", "работает" if kafka_up() else "остановлен")
    st.markdown("**Неисправности (система о них не знает):**")
    f1, f2, f3, f4, f5 = st.columns(5)
    fb = f1.checkbox("Барометр Кишинёва +6 гПа", value=True)
    fn = f2.checkbox("Шум датчиков в Бельцах", value=False)
    fl = f3.checkbox("Опоздания Кагула (5 %)", value=True)
    fd = f4.checkbox("Потери Комрата (3 %)", value=False)
    ff = f5.checkbox("Унгены без направления ветра", value=False)
    if st.button("▶ Запустить конвейер", type="primary", disabled=len(period) != 2):
        start, end = [pd.Timestamp(d) for d in period]
        mid = (start + (end - start) / 3).strftime("%Y-%m-%d")
        args = [PY, str(ROOT / "rtap" / "run_pipeline.py"), "--out", "live", "--start", start.strftime("%Y-%m-%d"),
                "--end", end.strftime("%Y-%m-%d"), "--speed", str(speed)]
        if fb:
            args += ["--bias", f"chisinau:6:{mid}"]
        if fn:
            args += ["--noise", f"balti:0.75:{mid}:{end.strftime('%Y-%m-%d')}"]
        if fl:
            args += ["--late", "cahul:0.05:3"]
        if fd:
            args += ["--drop", "comrat:0.03"]
        if ff:
            args += ["--drop-field", f"ungheni:wind_direction_10m_dominant:{mid}"]
        (RTAP / "live").mkdir(parents=True, exist_ok=True)
        for f in ("alerts.db", "READY", "ALERTS_READY", "run_info.json"):
            (RTAP / "live" / f).unlink(missing_ok=True)
        st.session_state["rtap_proc"] = subprocess.Popen(args, stdout=open(RTAP / "live" / "pipeline.log", "w"), stderr=subprocess.STDOUT)
        st.session_state["rtap_args"] = " ".join(args[2:])
        st.success("Конвейер запущен: Spark стартует ~10–20 с, затем станции начинают передачу.")

    @st.fragment(run_every=2)
    def live_board():
        proc = st.session_state.get("rtap_proc")
        running = proc is not None and proc.poll() is None
        fc, reg, sens = read_db("live")
        if fc is None or fc.empty:
            st.info("Данных пока нет." + (" Конвейер запускается…" if running else " Нажмите «Запустить конвейер»."))
            return
        fc = fc[fc.p_rain.notna()]
        last = fc.sort_values("date").groupby("station").tail(1).set_index("station")
        day = fc.date.max()
        k = st.columns(5)
        k[0].metric("Статус", "идёт поток" if running else "завершён")
        k[1].metric("Обработано записей", len(fc))
        k[2].metric("Сутки потока", pd.Timestamp(day).strftime("%d.%m.%Y"))
        k[3].metric("Задержка (медиана)", f"{fc.latency_ms.tail(60).median():.0f} мс")
        rlast = reg.sort_values("date").tail(1) if reg is not None and len(reg) else None
        short = {"РЕГИОНАЛЬНАЯ ТРЕВОГА": "тревога", "локальные рекомендации": "рекомендации", "спокойно": "спокойно"}
        dec = rlast.decision.iloc[0].split(" (")[0] if rlast is not None else "—"
        k[4].metric("Регион", short.get(dec, dec), help=rlast.decision.iloc[0] if rlast is not None else None)
        c1, c2 = st.columns([1, 1.4])
        fig = go.Figure()
        for s, (lat, lon) in COORDS.items():
            if s not in last.index:
                continue
            r = last.loc[s]
            lvl = int(r.level) if pd.notna(r.level) else 0
            fig.add_scatter(x=[lon], y=[lat], mode="markers+text", text=[f"{STATIONS[s]}<br>P = {r.p_rain:.0%}"],
                            textposition="top center", marker=dict(size=36, color=C_LEVEL[lvl], line=dict(color="black", width=1)),
                            showlegend=False, hovertext=r.message or LEVELS[lvl])
        fig.update_layout(height=430, xaxis=dict(range=[27.3, 29.3], title="долгота"), yaxis=dict(range=[45.6, 48.5], title="широта"),
                          title=f"Последние решения по станциям ({pd.Timestamp(day):%d.%m.%Y})", margin=dict(t=50))
        c1.plotly_chart(fig, width="stretch", key="rtap_map")
        al = fc[fc.level >= 1].sort_values(["date", "station"], ascending=False).head(8)
        rows = []
        for _, r in al.iterrows():
            sh = json.loads(r.shap) if r.shap else []
            rows.append({"Дата": r.date, "Станция": STATIONS[r.station], "Код": LEVELS[int(r.level)], "Тип": r.kind,
                         "P(дождь)": f"{r.p_rain:.0%}", "ENet": f"{r.p_rain_enet:.0%}",
                         "Главный вклад SHAP": f"{sh[0]['признак']} ({sh[0]['вклад']:+.3f})" if sh else "", "Статус": r.status})
        c2.markdown("**Последние рекомендации и тревоги**")
        c2.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", height=320)
        if sens is not None and len(sens):
            c2.markdown("**Технические тревоги (контроль датчиков)**")
            c2.dataframe(sens[["date", "station", "message"]].rename(columns={"date": "Дата", "station": "Станция", "message": "Сообщение"}),
                         hide_index=True, width="stretch", height=150)
        ch = fc[fc.station == "chisinau"].tail(120)
        fig = go.Figure()
        for s in STATIONS:
            g = fc[fc.station == s].tail(120)
            fig.add_scatter(x=pd.to_datetime(g.date), y=g.p_rain, mode="lines", name=STATIONS[s], line=dict(width=2.5 if s == "chisinau" else 1))
        fig.add_hline(y=0.23, line_dash="dash", annotation_text="порог жёлтого кода")
        fig.update_layout(height=320, yaxis_title="P(дождь завтра)", legend=dict(orientation="h"), margin=dict(t=20))
        st.plotly_chart(fig, width="stretch", key="rtap_lines")

    live_board()

# ---------------------------------------------------------------- runs of notebook 10
with tab_runs:
    runs = [p.name for p in sorted(RTAP.iterdir()) if (p / "alerts.db").exists()] if RTAP.exists() else []
    if not runs:
        st.info("Прогонов нет — выполните ноутбук 10.")
    else:
        run = st.selectbox("Прогон", runs, index=runs.index("faults") if "faults" in runs else 0)
        info = json.loads((RTAP / run / "run_info.json").read_text()) if (RTAP / run / "run_info.json").exists() else {}
        st.caption("Параметры: " + " ".join(info.get("producer_args", [])) + f" · окно {info.get('window_days', '?')} сут.")
        fc, reg, sens = read_db(run)
        from verdict import station_metrics
        st.dataframe(station_metrics(fc, 0.23).round(3), hide_index=True, width="stretch")
        c1, c2 = st.columns(2)
        c1.markdown("**Региональные решения**")
        c1.dataframe(reg.decision.value_counts().rename("Дней"), width="stretch")
        c2.markdown("**Контроль датчиков**")
        c2.dataframe(sens[sens.check_name != "полнота"][["date", "station", "message"]], hide_index=True, width="stretch")
        import re
        for f in sorted(FIG.glob("fig_11_*.png"), key=lambda p: (int(re.match(r"\d+", p.stem.split("_")[2]).group()), p.stem)):
            with st.expander(f.stem):
                st.image(str(f), width="stretch")

# ---------------------------------------------------------------- tuning
with tab_tune:
    path = ROOT / "data" / "processed" / "rtap_tuning_results.csv"
    if path.exists():
        st.markdown("**Результаты ноутбука 10:** настройка по кросс-валидации `TimeSeriesSplit` на 2015–2022, тест 2024–2025.")
        st.dataframe(pd.read_csv(path).round(3), hide_index=True, width="stretch")
    st.markdown("**Попробуйте сами (упражнения E–F):** модель Elastic Net для Кишинёва, с признаками соседних станций или без.")
    c1, c2, c3 = st.columns(3)
    C = c1.select_slider("C", [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0], value=0.03)
    a = c2.slider("α (l1_ratio)", 0.0, 1.0, 0.5, 0.25)
    W = c3.selectbox("Соседи: окно, суток", ["без соседей", 1, 3, 7], index=0)

    @st.cache_data(show_spinner="Обучение…")
    def fit_try(C, a, W):
        sys.path.insert(0, str(ROOT / "src"))
        from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

        from db import load_dataset
        from features import FEATURES, TARGET
        from regularization import coefficients, make_model
        from verdict import REGIONAL_NAMES_RU, regional_features
        d = load_dataset("chisinau")
        feats = list(FEATURES)
        if W != "без соседей":
            reg = regional_features(int(W))
            d = d.merge(reg, on="date", how="left").dropna()
            feats += [c for c in reg.columns if c != "date"]
        tr, te = d[d.date.dt.year <= 2022], d[d.date.dt.year >= 2024]
        m = make_model(C, a).fit(tr[feats], tr[TARGET])
        p = m.predict_proba(te[feats])[:, 1]
        from features import FEATURE_NAMES_RU
        names = [FEATURE_NAMES_RU.get(f, REGIONAL_NAMES_RU.get(f, f)) for f in feats]
        return (roc_auc_score(te[TARGET], p), average_precision_score(te[TARGET], p), f1_score(te[TARGET], p >= 0.5),
                pd.Series(coefficients(m), index=names))
    auc, ap, f1, w = fit_try(C, a, W)
    k = st.columns(4)
    k[0].metric("ROC-AUC тест", f"{auc:.3f}")
    k[1].metric("PR-AUC тест", f"{ap:.3f}")
    k[2].metric("F1 (порог 0.5)", f"{f1:.3f}")
    k[3].metric("Ненулевых весов", f"{int((w.abs() > 1e-10).sum())} из {len(w)}")
    w = w[w.abs().sort_values().index[-15:]]
    fig = go.Figure(go.Bar(y=w.index, x=w.values, orientation="h",
                           marker_color=["#8E44AD" if "соседей" in i else ("#C0392B" if v > 0 else "#2E8B57") for i, v in w.items()]))
    fig.update_layout(height=480, title="15 самых больших весов (фиолетовые — признаки соседних станций)", xaxis_title="Вес")
    st.plotly_chart(fig, width="stretch")
    st.info("Вердикт ноутбука 10: настройка по кросс-валидации даёт небольшой, но устойчивый выигрыш, а контекст соседних станций — "
            "больше, чем любая настройка. «Простая модель + хорошие данные» важнее «сложной настройки на бедных данных».")
