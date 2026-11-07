import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, precision_score, recall_score, roc_auc_score

from common import FEATURES, FIG, TARGET, header, load_all, load_models, source_badge, splits, SEASONS

header("🌦️ Погода как знак и алгоритм",
       "Тема №23 «Анализ и прогноз погоды» · курс «Наука и анализ данных» · УТМ, группа SD-251M")
source_badge()

datasets, daily, _ = load_all()
rain_model, thr, _, _ = load_models()
train, valid, test = splits(datasets["chisinau"])
p = rain_model.predict_proba(test[FEATURES])[:, 1]
y = test[TARGET].to_numpy()
pred = (p >= thr).astype(int)
pers = (test.precipitation_sum >= 1).astype(int).to_numpy()
clim = train.groupby(train.date.dt.month)[TARGET].mean()
p_clim = test.date.dt.month.map(clim).to_numpy()

c1, c2, c3, c4 = st.columns(4)
c1.metric("ROC-AUC (тест 2024–2025)", f"{roc_auc_score(y, p):.3f}", f"{roc_auc_score(y, p) - roc_auc_score(y, p_clim):+.3f} к климатологии")
c2.metric("Найдено дождливых дней", f"{recall_score(y, pred):.0%}", f"{recall_score(y, pred) - recall_score(y, pers):+.0%} к «завтра как сегодня»")
c3.metric("F1 (дождь)", f"{f1_score(y, pred):.3f}", f"{f1_score(y, pred) - f1_score(y, pers):+.3f} к «завтра как сегодня»")
c4.metric("Дней в хранилище", f"{sum(len(d) for d in daily.values()):,}".replace(",", " "), "6 станций × 11 лет")

left, right = st.columns([1.1, 1])
with left:
    st.subheader("Задача")
    st.markdown("""
**Проблема.** Неблагоприятная погода мешает сельскому хозяйству, транспорту, мероприятиям и работе экстренных служб; решения принимаются **накануне**.

**Цель.** По сегодняшним метеоданным станции Кишинёв предсказать:
1. будут ли **завтра осадки ≥ 1 мм**;
2. **тип погоды** завтра — благоприятная / неблагоприятная / критическая;

и выдавать **предупреждения в реальном времени** с понятным объяснением.

**Данные.** Реанализ ERA5 (Open-Meteo), 2015–2025, 6 станций Молдовы → хранилище **PostgreSQL** (схема «звезда») → обучающий набор **SQL-запросом**.

**Разбиение по времени:** обучение 2015–2022, выбор порога 2023, проверка 2024–2025.
""")
with right:
    st.subheader("Ключевые показатели (KPI)")
    r_pers = f1_score(y, pers)
    bss = 1 - brier_score_loss(y, p) / brier_score_loss(y, p_clim)
    season = test.date.dt.month.map(SEASONS).to_numpy()
    min_auc = min(roc_auc_score(y[season == s], p[season == s]) for s in ["Зима", "Весна", "Лето", "Осень"])
    kpi = pd.DataFrame([
        ("KPI-1", "Различающая способность (ROC-AUC)", "≥ 0.75", roc_auc_score(y, p), roc_auc_score(y, p) >= 0.75),
        ("KPI-2", "Обнаружение дождливых дней (Recall)", "≥ 0.70", recall_score(y, pred), recall_score(y, pred) >= 0.70),
        ("KPI-3", "PR-AUC для редкого класса", f"≥ {2 * y.mean():.2f}", average_precision_score(y, p), average_precision_score(y, p) >= 2 * y.mean()),
        ("KPI-4", "Прирост F1 к инерционности", "≥ +0.05", f1_score(y, pred) - r_pers, f1_score(y, pred) - r_pers >= 0.05),
        ("KPI-5", "Brier skill score к климатологии", "> 0", bss, bss > 0),
        ("KPI-6", "Доля ложных тревог", "≤ 0.60", 1 - precision_score(y, pred), 1 - precision_score(y, pred) <= 0.60),
        ("KPI-7", "Минимальный ROC-AUC по сезонам", "≥ 0.70", min_auc, min_auc >= 0.70),
    ], columns=["Код", "Показатель", "Цель", "Значение", "Достигнут"])
    kpi["Значение"] = kpi["Значение"].round(3)
    kpi["Достигнут"] = kpi["Достигнут"].map({True: "✅", False: "⚠️ на грани"})
    st.dataframe(kpi, hide_index=True, width="stretch")

st.subheader("Конвейер данных")
st.image(str(FIG / "fig_1_1_etl_architecture.png"), width="stretch")
st.subheader("Архитектура системы реального времени")
st.image(str(FIG / "fig_5_1_architecture.png"), width="stretch")
