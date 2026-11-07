import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (confusion_matrix, f1_score, precision_recall_curve, precision_score, recall_score,
                             roc_auc_score, roc_curve)

from common import FEATURES, TARGET, header, load_all, load_models, name, source_badge, splits

header("🎯 Модель и порог решения", "Random Forest · тест 2024–2025 · порог — это решение, а не техническая деталь")
source_badge()
datasets, _, _ = load_all()
rain_model, thr_default, _, _ = load_models()
_, _, test = splits(datasets["chisinau"])
y = test[TARGET].to_numpy()
p = rain_model.predict_proba(test[FEATURES])[:, 1]

thr = st.slider("Порог вероятности, начиная с которого объявляем «дождь завтра»", 0.05, 0.80,
                float(thr_default), 0.01, help=f"Выбран на валидации 2023: {thr_default:.2f}")
pred = (p >= thr).astype(int)
cm = confusion_matrix(y, pred)
tn, fp, fn, tp = cm.ravel()
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Найдено дождливых дней", f"{tp} из {tp + fn}", f"Recall {recall_score(y, pred):.0%}")
c2.metric("Пропущено дождей", fn)
c3.metric("Ложных тревог", fp, f"{fp / max(tp + fp, 1):.0%} от тревог", delta_color="inverse")
c4.metric("Precision", f"{precision_score(y, pred, zero_division=0):.2f}")
c5.metric("F1", f"{f1_score(y, pred):.3f}")

c1, c2, c3 = st.columns([1, 1.2, 1.2])
fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                x=["Прогноз: нет", "Прогноз: дождь"], y=["Факт: нет", "Факт: дождь"])
fig.update_layout(title="Матрица ошибок", coloraxis_showscale=False)
c1.plotly_chart(fig, width="stretch")
fpr, tpr, roc_t = roc_curve(y, p)
fig = go.Figure(go.Scatter(x=fpr, y=tpr, name="Random Forest", line=dict(color="#2E8B57", width=3)))
fig.add_scatter(x=[0, 1], y=[0, 1], line=dict(dash="dot", color="gray"), name="случайно")
fig.add_scatter(x=[fp / (fp + tn)], y=[tp / (tp + fn)], mode="markers", marker=dict(size=14, color="black", symbol="x"),
                name=f"порог {thr:.2f}")
fig.update_layout(title=f"ROC-кривая (AUC = {roc_auc_score(y, p):.3f})", xaxis_title="Доля ложных тревог среди сухих дней",
                  yaxis_title="Доля найденных дождей", legend=dict(x=0.45, y=0.08, bgcolor="rgba(255,255,255,0.7)"))
c2.plotly_chart(fig, width="stretch")
ts = np.round(np.arange(0.05, 0.81, 0.01), 2)
curve = pd.DataFrame({"Порог": ts, "Precision": [precision_score(y, p >= t, zero_division=0) for t in ts],
                      "Recall": [recall_score(y, p >= t) for t in ts], "F1": [f1_score(y, p >= t) for t in ts]})
fig = px.line(curve, x="Порог", y=["Precision", "Recall", "F1"], title="Компромисс при разных порогах",
              labels={"value": "Значение метрики", "variable": "Метрика"})
fig.add_vline(x=thr, line_dash="dash", line_color="black")
c3.plotly_chart(fig, width="stretch")

if thr < 0.2:
    st.warning("Низкий порог: почти все дожди найдены, но тревог так много, что им перестанут доверять.")
elif thr > 0.45:
    st.warning("Высокий порог: тревоги почти всегда верны, но большинство дождей пропускается.")
else:
    st.success("Сбалансированный порог: большая часть дождей найдена при умеренном числе ложных тревог.")

st.subheader("Сравнение с простыми правилами")
pers = (test.precipitation_sum >= 1).astype(int).to_numpy()
cmp = pd.DataFrame({
    "Метод": ["Random Forest (текущий порог)", "«Завтра как сегодня»", "«Дождя никогда не будет»"],
    "F1": [f1_score(y, pred), f1_score(y, pers), 0.0],
    "Recall": [recall_score(y, pred), recall_score(y, pers), 0.0],
    "Accuracy": [(pred == y).mean(), (pers == y).mean(), (y == 0).mean()],
})
st.dataframe(cmp.round(3), hide_index=True, width="stretch")
st.caption("Accuracy у правила «дождя никогда не будет» самая высокая — поэтому при несбалансированных классах accuracy не используется.")

st.subheader("Какие признаки важнее всего для модели")
imp = pd.Series(rain_model.feature_importances_, index=[name(f) for f in FEATURES]).sort_values().tail(12)
fig = px.bar(imp, orientation="h", labels={"value": "Важность (уменьшение неоднородности)", "index": ""})
fig.update_layout(showlegend=False, height=450)
st.plotly_chart(fig, width="stretch")

st.divider()
st.subheader("💰 Цена ошибок: какой порог выгоден конкретному пользователю")
st.caption("Лекция 2: максимальный F1 ≠ минимальная стоимость. Порог выбирается на валидации 2023, результат проверяется на тесте 2024–2025.")
SCEN = {
    "Фермер": dict(c_fp=2_000, c_fn=12_000, g_tp=10_000, g_tn=1_000),
    "Организатор мероприятия": dict(c_fp=10_000, c_fn=30_000, g_tp=20_000, g_tn=5_000),
    "Дорожная служба": dict(c_fp=5_000, c_fn=5_000, g_tp=3_000, g_tn=3_000),
}
scen = st.radio("Пользователь прогноза", list(SCEN), horizontal=True)
s0 = SCEN[scen]
c1, c2, c3, c4 = st.columns(4)
c_fp = c1.number_input("Ложная тревога, лей", 0, 100_000, s0["c_fp"], 500, key=f"cfp_{scen}")
c_fn = c2.number_input("Пропущенный дождь, лей", 0, 100_000, s0["c_fn"], 500, key=f"cfn_{scen}")
g_tp = c3.number_input("Выгода верной тревоги, лей", 0, 100_000, s0["g_tp"], 500, key=f"gtp_{scen}")
g_tn = c4.number_input("Выгода верного «сухо», лей", 0, 100_000, s0["g_tn"], 500, key=f"gtn_{scen}")


def net_result(y_true, prob, t):
    tn_, fp_, fn_, tp_ = confusion_matrix(y_true, prob >= t, labels=[0, 1]).ravel()
    return tp_ * g_tp + tn_ * g_tn - fp_ * c_fp - fn_ * c_fn


_, valid, _ = splits(datasets["chisinau"])
p_va = rain_model.predict_proba(valid[FEATURES])[:, 1]
grid = np.round(np.arange(0.05, 0.81, 0.01), 2)
t_best = float(grid[int(np.argmax([net_result(valid[TARGET].to_numpy(), p_va, t) for t in grid]))])
denom = c_fp + g_tn + c_fn + g_tp
t_star = (c_fp + g_tn) / denom if denom else 0.5
net_test = np.array([net_result(y, p, t) for t in grid]) / 1000

c1, c2, c3, c4 = st.columns(4)
c1.metric("Выгодный порог (валидация 2023)", f"{t_best:.2f}")
c2.metric("Теория t* = (c_FP+g_TN)/(сумма)", f"{t_star:.2f}")
c3.metric(f"Результат при пороге {t_best:.2f}", f"{net_result(y, p, t_best) / 1000:+,.0f} тыс. лей")
c4.metric(f"Результат при пороге по F1 ({thr_default:.2f})", f"{net_result(y, p, thr_default) / 1000:+,.0f} тыс. лей",
          f"{(net_result(y, p, thr_default) - net_result(y, p, t_best)) / 1000:+,.0f} тыс. к выгодному")

c1, c2 = st.columns([1.6, 1])
fig = go.Figure(go.Scatter(x=grid, y=net_test, line=dict(color="#2B6CB0", width=3), name="чистый результат"))
fig.add_vline(x=t_best, line_dash="dash", line_color="#2E8B57", annotation_text=f"выгодный {t_best:.2f}",
              annotation_position="top left")
fig.add_vline(x=thr_default, line_dash="dot", line_color="black", annotation_text=f"по F1 {thr_default:.2f}",
              annotation_position="top right")
fig.add_vline(x=thr, line_color="#D32F2F", annotation_text=f"ползунок {thr:.2f}", annotation_position="bottom right")
fig.add_hline(y=0, line_color="gray")
fig.update_layout(title=f"Чистый результат за тест 2024–2025: {scen.lower()}", xaxis_title="Порог",
                  yaxis_title="Тыс. лей (выгода − потери)", showlegend=False)
c1.plotly_chart(fig, width="stretch")
cost_m = np.array([[0, fp * c_fp], [fn * c_fn, 0]]) / 1000
prof_m = np.array([[tn * g_tn, 0], [0, tp * g_tp]]) / 1000
labels = dict(x=["Прогноз: нет", "Прогноз: дождь"], y=["Факт: нет", "Факт: дождь"])
fig = px.imshow(cost_m, text_auto=".0f", color_continuous_scale="Reds", **labels)
fig.update_layout(title=f"Потери при пороге {thr:.2f}, тыс. лей", coloraxis_showscale=False, height=260,
                  margin=dict(t=40, b=10))
c2.plotly_chart(fig, width="stretch")
fig = px.imshow(prof_m, text_auto=".0f", color_continuous_scale="Greens", **labels)
fig.update_layout(title=f"Выгода при пороге {thr:.2f}, тыс. лей", coloraxis_showscale=False, height=260,
                  margin=dict(t=40, b=10))
c2.plotly_chart(fig, width="stretch")
st.info("Одна модель — разные пороги: чем дороже пропуск дождя относительно ложной тревоги, тем ниже выгодный порог. "
        "Формула t* работает потому, что вероятности модели откалиброваны.")
