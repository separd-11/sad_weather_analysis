import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import f1_score, recall_score, roc_auc_score

from common import FEATURES, TARGET, WEATHER_TYPES, header, load_all, load_models, name, source_badge, splits
from features import TARGET3, build_features
from regularization import (METHOD_COLORS, METHODS, NOISY_SENSORS, SENSOR_NAMES_RU, add_sensor_noise, coefficients, fit_timed,
                            make_model, predict_latency_ms, stream_retrain)

header("🎭 Регуляризация в потоке (лекция 4)",
       "L1 — «скульптор» (обнуляет), L2 — «художник» (сжимает всё), Elastic Net — «режиссёр» (баланс). "
       "Логистическая регрессия на 26 признаках, цель — осадки завтра")
source_badge()
datasets, daily, _ = load_all()
data = datasets["chisinau"]
raw = daily["chisinau"]
train, valid, test = splits(data)
X_tr, y_tr, X_te, y_te = train[FEATURES], train[TARGET], test[FEATURES], test[TARGET].to_numpy()
rf, rf_thr, *_ = load_models()
C_OPTIONS = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]

sb = st.sidebar
sb.header("Параметры «актёров»")
C = sb.select_slider("C — обратная сила штрафа (меньше = сильнее)", options=C_OPTIONS, value=0.01)
alpha = sb.slider("α Elastic Net (0 = L2, 1 = L1)", 0.0, 1.0, 0.25, 0.05)
sb.caption("По умолчанию — Elastic Net, выбранный по валидации 2023 в ноутбуке 09 (C = 0.01, α = 0.25).")
sb.caption("В scikit-learn 1.9 метод задаётся `l1_ratio` (1 — L1, 0 — L2); параметр `penalty` из скриптов лекции устарел.")
ACTORS = {"L1 (Lasso)": 1.0, "L2 (Ridge)": 0.0, "Elastic Net": alpha}


@st.cache_resource(show_spinner="Обучение трёх моделей…")
def trio(C: float, alpha: float):
    out = {}
    for m_name, r in {"L1 (Lasso)": 1.0, "L2 (Ridge)": 0.0, "Elastic Net": alpha}.items():
        m, fit_ms = fit_timed(make_model(C, r), X_tr, y_tr, repeats=3)
        out[m_name] = (m, fit_ms, predict_latency_ms(m, X_te, n=200))
    return out


@st.cache_data(show_spinner="Перебор C…")
def c_sweep(alpha: float):
    rows, paths = [], {}
    for c in np.logspace(-3, 1, 13):
        for m_name, r in {"L1 (Lasso)": 1.0, "L2 (Ridge)": 0.0, "Elastic Net": alpha}.items():
            m, ms = fit_timed(make_model(c, r), X_tr, y_tr)
            p = m.predict_proba(X_te)[:, 1]
            w = coefficients(m)
            paths.setdefault(m_name, []).append(w)
            rows.append({"C": c, "Метод": m_name, "ROC-AUC": roc_auc_score(y_te, p), "F1 (0.5)": f1_score(y_te, p >= 0.5),
                         "Ненулевых": int((np.abs(w) > 1e-10).sum()), "Обучение, мс": ms})
    return pd.DataFrame(rows), {k: np.array(v) for k, v in paths.items()}


@st.cache_data(show_spinner="Перебор α…")
def a_sweep(C: float):
    rows, paths = [], []
    for a in np.round(np.linspace(0, 1, 11), 2):
        m, ms = fit_timed(make_model(C, a), X_tr, y_tr)
        p = m.predict_proba(X_te)[:, 1]
        w = coefficients(m)
        paths.append(w)
        rows.append({"α": a, "ROC-AUC": roc_auc_score(y_te, p), "F1 (0.5)": f1_score(y_te, p >= 0.5),
                     "Ненулевых": int((np.abs(w) > 1e-10).sum()), "Обучение, мс": ms})
    return pd.DataFrame(rows), np.array(paths)


@st.cache_data(show_spinner="Поток 2024–2025: переобучение по микропакетам…")
def run_stream(C: float, a: float, freq: str):
    return stream_retrain(raw, lambda ni: (C, a), freq=freq, data=data)


@st.cache_data(show_spinner="Модели на зашумлённых датчиках…")
def robustness(sensors: tuple, sigma: float):
    d = build_features(add_sensor_noise(raw, sensors=list(sensors), sigma=sigma, start="2024-01-01"))
    t = d[d.date.dt.year >= 2024]
    rows = []
    for c in [1.0, 0.1, 0.01]:
        for m_name, r in METHODS.items():
            m = make_model(c, r).fit(X_tr, y_tr)
            rows.append({"C": c, "Метод": m_name, "ROC-AUC": roc_auc_score(t[TARGET], m.predict_proba(t[FEATURES])[:, 1])})
    rows.append({"C": "—", "Метод": "Random Forest", "ROC-AUC": roc_auc_score(t[TARGET], rf.predict_proba(t[FEATURES])[:, 1])})
    return pd.DataFrame(rows)


@st.cache_data(show_spinner="Поток с неисправными датчиками: четыре политики регуляризации…")
def director(sigma: float, C_base: float, a_base: float, thr_noise: float, freq: str):
    start, end = "2024-07-01", "2024-12-31"
    rw = add_sensor_noise(raw, sigma=sigma, start=start, end=end)
    dd = build_features(rw)
    pols = {"Фиксированная L2, C = 1": lambda ni: (1.0, 0.0),
            "Фиксированный Elastic Net": lambda ni: (C_base, a_base),
            "Правило лекции (шум → L2)": lambda ni: (C_base, 0.0) if ni > thr_noise else (C_base, a_base),
            "Зеркальное правило (шум → L1)": lambda ni: (C_base, 1.0) if ni > thr_noise else (C_base, a_base),
            "Перегиб (шум → L1, C / 6)": lambda ni: (C_base / 6, 1.0) if ni > thr_noise else (C_base, a_base)}
    rows, logs = [], {}
    for p_name, pol in pols.items():
        P, LOG = stream_retrain(rw, pol, freq=freq, data=dd)
        nz = (P.date >= start) & (P.date <= end)
        logs[p_name] = (P, LOG)
        noisy = LOG.noise_index > thr_noise
        rows.append({"Политика": p_name, "ROC-AUC 2024–2025": roc_auc_score(P.y, P.p),
                     "ROC-AUC в период шума": roc_auc_score(P.y[nz], P.p[nz]),
                     "Срабатываний индекса": int(noisy.sum()), "Переобучений": len(LOG),
                     "Признаков при шуме (медиана)": float(LOG.loc[noisy, "nonzero"].median()) if noisy.any() else np.nan,
                     "Время переобучений, с": LOG.fit_ms.sum() / 1000})
    return pd.DataFrame(rows), logs


@st.cache_data(show_spinner="SHAP для линейной модели…")
def linear_shap(C: float, a: float):
    import shap
    m = make_model(C, a).fit(X_tr, y_tr)
    sc, lr = m[0], m[-1]
    Z_tr, Z_te = sc.transform(X_tr), sc.transform(X_te)
    sv = shap.LinearExplainer(lr, shap.maskers.Independent(Z_tr, max_samples=len(Z_tr))).shap_values(Z_te)
    return np.asarray(sv), lr.coef_[0], float(lr.intercept_[0] + lr.coef_[0] @ Z_tr.mean(0)), m.predict_proba(X_te)[:, 1]


@st.cache_data(show_spinner="Мультиномиальная регрессия на 3 класса…")
def multiclass(C: float, alpha: float):
    import shap
    from sklearn.base import clone
    y3_tr, y3_te = train[TARGET3], test[TARGET3].to_numpy()
    rows, models = [], {}
    for m_name, r in {"L1 (Lasso)": 1.0, "L2 (Ridge)": 0.0, "Elastic Net": alpha}.items():
        m, ms = fit_timed(make_model(C, r), X_tr, y3_tr)
        pr = m.predict(X_te); rc = recall_score(y3_te, pr, average=None)
        models[m_name] = m
        rows.append({"Модель": m_name, "F1 macro": f1_score(y3_te, pr, average="macro"), "Recall: благоприятная": rc[0],
                     "Recall: неблагоприятная": rc[1], "Recall: критическая": rc[2],
                     "Ненулевых по классам": " / ".join(str(int((np.abs(w) > 1e-10).sum())) for w in m[-1].coef_), "Обучение, мс": ms})
    rf3 = clone(rf).set_params(class_weight="balanced").fit(X_tr, y3_tr)
    pr = rf3.predict(X_te); rc = recall_score(y3_te, pr, average=None)
    rows.append({"Модель": "Random Forest (balanced)", "F1 macro": f1_score(y3_te, pr, average="macro"), "Recall: благоприятная": rc[0],
                 "Recall: неблагоприятная": rc[1], "Recall: критическая": rc[2], "Ненулевых по классам": "—", "Обучение, мс": np.nan})
    shap_cls = {}
    for m_name, m in models.items():
        Z_tr, Z_te = m[0].transform(X_tr), m[0].transform(X_te)
        sv = np.asarray(shap.LinearExplainer(m[-1], shap.maskers.Independent(Z_tr, max_samples=len(Z_tr))).shap_values(Z_te))
        sv = sv if sv.shape[-1] == 3 else np.moveaxis(sv, 0, -1)
        shap_cls[m_name] = (np.abs(sv).mean(0), m[-1].coef_)
    return pd.DataFrame(rows), shap_cls


def coef_figure(ws: dict, title: str, height: int = 620):
    fig = go.Figure()
    order = np.argsort(-np.max(np.abs(np.array(list(ws.values()))), axis=0))[::-1]
    for m_name, w in ws.items():
        fig.add_bar(y=[name(FEATURES[j]) for j in order], x=w[order], orientation="h", name=m_name,
                    marker_color=[METHOD_COLORS[m_name] if abs(w[j]) > 1e-10 else "#CCCCCC" for j in order])
    fig.add_vline(x=0, line_color="black")
    fig.update_layout(barmode="group", height=height, title=title, legend=dict(orientation="h", y=1.04),
                      xaxis_title="Коэффициент (на стандартизованный признак; > 0 — к дождю)")
    return fig


tabs = st.tabs(["Трио: L1, L2, ENet", "Перебор C и α", "Поток: коэффициенты во времени", "Шум и «адаптивный режиссёр»",
                "SHAP против коэффициентов", "Тип погоды (3 класса)"])

# ---------------------------------------------------------------- trio
with tabs[0]:
    T = trio(C, alpha)
    p_rf = rf.predict_proba(X_te)[:, 1]
    k = st.columns(3)
    rows = []
    for col, (m_name, (m, fit_ms, pred_ms)) in zip(k, T.items()):
        p = m.predict_proba(X_te)[:, 1]
        nz = int((np.abs(coefficients(m)) > 1e-10).sum())
        col.markdown(f"#### {m_name}")
        col.metric("ROC-AUC (тест 2024–2025)", f"{roc_auc_score(y_te, p):.3f}", f"{roc_auc_score(y_te, p) - roc_auc_score(y_te, p_rf):+.3f} к Random Forest")
        col.metric("F1 при пороге 0.5", f"{f1_score(y_te, p >= 0.5):.3f}")
        col.metric("Признаков в модели", f"{nz} из 26")
        col.metric("Обучение на 8 годах", f"{fit_ms:.0f} мс")
        col.metric("Прогноз 1 записи", f"{pred_ms:.2f} мс")
    st.caption(f"Для сравнения Random Forest (400 деревьев): ROC-AUC {roc_auc_score(y_te, p_rf):.3f}, F1 {f1_score(y_te, p_rf >= rf_thr):.3f} "
               f"при пороге {rf_thr:.2f}; прогноз одной записи ≈ 10–15 мс (в 40–50 раз медленнее линейной модели).")
    st.plotly_chart(coef_figure({m: coefficients(v[0]) for m, v in T.items()}, f"Коэффициенты при C = {C}, α ENet = {alpha} (серые — обнулены)"),
                    width="stretch")
    st.info("**Скульптор (L1)** отсекает лишнее: при малом C остаются давление, облачность, часы с осадками и ветер. "
            "**Художник (L2)** сохраняет все 26 признаков, но «приглушает» их. **Режиссёр (ENet)** — между ними, ползунок α двигает его к одному из полюсов. "
            "При C ≥ 1 штраф почти не действует (2 916 дней обучения), и три актёра играют одинаково.")

# ---------------------------------------------------------------- sweeps
with tabs[1]:
    S, paths = c_sweep(alpha)
    c1, c2, c3 = st.columns(3)
    for col, metric, title in [(c1, "ROC-AUC", "Качество на тесте"), (c2, "Ненулевых", "Сколько признаков осталось"), (c3, "Обучение, мс", "Цена обучения")]:
        fig = px.line(S, x="C", y=metric, color="Метод", markers=True, log_x=True, color_discrete_map=METHOD_COLORS, title=title)
        if metric == "ROC-AUC":
            fig.add_hline(y=roc_auc_score(y_te, rf.predict_proba(X_te)[:, 1]), line_dash="dash", annotation_text="Random Forest")
        fig.update_layout(height=380, legend=dict(orientation="h", y=-0.25))
        col.plotly_chart(fig, width="stretch")
    m_path = st.radio("Путь коэффициентов для", list(METHODS), horizontal=True)
    M = paths[m_path].T
    fig = go.Figure(go.Heatmap(z=np.clip(M, -1, 1), x=np.round(np.log10(np.logspace(-3, 1, 13)), 2), y=[name(f) for f in FEATURES],
                               colorscale="RdBu_r", zmid=0, colorbar=dict(title="коэф.")))
    fig.update_layout(height=650, xaxis_title="log₁₀ C (слева — сильный штраф)", title=f"{m_path}: коэффициенты в зависимости от C")
    st.plotly_chart(fig, width="stretch")
    A, apaths = a_sweep(C)
    c1, c2 = st.columns([1, 1.3])
    fig = go.Figure()
    fig.add_scatter(x=A["α"], y=A["ROC-AUC"], name="ROC-AUC", line=dict(color="#8E44AD"))
    fig.add_scatter(x=A["α"], y=A["Ненулевых"], name="ненулевых", yaxis="y2", line=dict(color="#C0392B", dash="dot"))
    fig.update_layout(height=420, title=f"Перебор α при C = {C}", xaxis_title="α (0 = L2, 1 = L1)",
                      yaxis=dict(title="ROC-AUC"), yaxis2=dict(title="ненулевых", overlaying="y", side="right", showgrid=False),
                      legend=dict(orientation="h", y=-0.2))
    c1.plotly_chart(fig, width="stretch")
    fig = go.Figure(go.Heatmap(z=np.clip(apaths.T, -0.6, 0.6), x=A["α"], y=[name(f) for f in FEATURES], colorscale="RdBu_r", zmid=0))
    fig.update_layout(height=420, title="Коэффициенты в зависимости от α", xaxis_title="α", yaxis=dict(tickfont=dict(size=9)))
    c2.plotly_chart(fig, width="stretch")
    st.markdown("""
| Метод | C мал (сильный штраф, C ≈ 0.005–0.05) | C велик (слабый штраф, C ≥ 1) | α мал | α велик |
|---|---|---|---|---|
| **L1** | 3–15 признаков (давление, облачность, часы с осадками, ветер); ROC-AUC до 0.797; обучение 7–13 мс; при C ≤ 0.002 — все веса 0 | 25 признаков; ROC-AUC 0.76; обучение 220–520 мс | — | — |
| **L2** | все 26, веса малы; ROC-AUC до 0.791 (при C = 0.001); 6–30 мс | все 26, большие веса разных знаков у связанных признаков; ROC-AUC 0.76; 120–140 мс | — | — |
| **ENet** | 8–19 признаков, группы связанных признаков; ROC-AUC до 0.796 | как L2 и L1 (25–26 признаков, 0.76) | ≈ L2: все 26 признаков | ≈ L1: 13 признаков, быстрее |
""")
    st.caption("Скорость решателя saga определяется прежде всего силой штрафа: при C = 1 L1 обучается дольше L2, при C = 0.01 — быстрее всех.")


# ---------------------------------------------------------------- stream
with tabs[2]:
    c1, c2, c3 = st.columns(3)
    s_method = c1.selectbox("Актёр", list(ACTORS))
    freq = c2.radio("Переобучение (микропакет)", ["M", "W"], format_func=lambda f: {"M": "ежемесячно", "W": "еженедельно"}[f], horizontal=True)
    n_top = c3.slider("Сколько признаков выделить цветом", 3, 10, 7)
    P, LOG = run_stream(C, ACTORS[s_method], freq)
    W = LOG[FEATURES]
    top = W.abs().mean().sort_values(ascending=False).index[:n_top]
    k = st.columns(4)
    thr = 0.5
    k[0].metric("ROC-AUC в потоке", f"{roc_auc_score(P.y, P.p):.3f}")
    k[1].metric("F1 (порог 0.5)", f"{f1_score(P.y, P.p >= thr):.3f}")
    k[2].metric("Переобучений / время", f"{len(LOG)} / {LOG.fit_ms.sum() / 1000:.1f} с")
    on = np.abs(W.to_numpy()) > 1e-10
    k[3].metric("Включений/выключений признаков", int(np.abs(np.diff(on.astype(int), axis=0)).sum()))
    fig = go.Figure()
    for f in FEATURES:
        hl = f in top
        fig.add_scatter(x=LOG.period, y=LOG[f], mode="lines", name=name(f), showlegend=bool(hl),
                        line=dict(width=2.5 if hl else 0.8, color=None if hl else "#BBBBBB"), hovertemplate=f"{name(f)}: %{{y:.3f}}<extra></extra>")
    fig.add_hline(y=0, line_color="black")
    fig.update_layout(height=520, title=f"{s_method}: коэффициенты при каждом переобучении 2024–2025 (C = {C})",
                      xaxis_title="Начало периода", yaxis_title="Коэффициент", legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, width="stretch")
    st.markdown("**▶ «Анимация» как в `elastic_net_streaming.py`:** коэффициенты меняются от переобучения к переобучению.")
    speed = st.slider("Пауза между кадрами, с", 0.05, 1.0, 0.25, 0.05)
    if st.button("▶ Проиграть поток", type="primary"):
        box = st.empty()
        lim = float(np.abs(W.to_numpy()).max()) * 1.05
        order = W.abs().mean().sort_values().index
        for i in range(len(LOG)):
            w = W.iloc[i][order]
            fig = go.Figure(go.Bar(y=[name(f) for f in order], x=w, orientation="h",
                                   marker_color=[METHOD_COLORS.get(s_method, "#8E44AD") if abs(v) > 1e-10 else "#CCCCCC" for v in w]))
            fig.update_layout(height=560, xaxis_range=[-lim, lim], title=f"{LOG.period.iloc[i]:%d.%m.%Y}: признаков в модели {int((np.abs(w) > 1e-10).sum())}")
            box.plotly_chart(fig, width="stretch", key=f"anim_{i}")
            time.sleep(speed)

# ---------------------------------------------------------------- noise and adaptive director
with tabs[3]:
    st.markdown("**1. Статичная модель на зашумлённых датчиках** (шум вносится в исходные измерения 2024–2025, признаки пересчитываются).")
    c1, c2 = st.columns([1, 2])
    sensors = c1.multiselect("Датчики с шумом", NOISY_SENSORS, default=NOISY_SENSORS, format_func=lambda s: SENSOR_NAMES_RU[s])
    sigma = c1.slider("Сила шума σ (доля ст. отклонения датчика)", 0.0, 1.5, 0.5, 0.05)
    if sensors:
        R = robustness(tuple(sensors), sigma)
        R["Модель"] = R["Метод"] + np.where(R["C"].astype(str) == "—", "", ", C = " + R["C"].astype(str))
        fig = px.bar(R, x="ROC-AUC", y="Модель", orientation="h", color="Метод",
                     color_discrete_map={**METHOD_COLORS, "Random Forest": "#555555"}, range_x=[0.5, 0.82])
        fig.update_layout(height=430, showlegend=False, title=f"ROC-AUC на тесте при шуме σ = {sigma} (без шума: ≈ 0.76–0.80)")
        c2.plotly_chart(fig, width="stretch")
    st.caption("Вывод ноутбука 09: устойчивость определяется прежде всего силой штрафа C — при слабом штрафе связанные признаки "
               "(температура, точка росы, дефицит точки росы = T − Td) получают большие коэффициенты разных знаков, и шум усиливается.")
    st.markdown("**2. Поток с неисправностью (07–12.2024) и «адаптивный режиссёр».** Перед каждым переобучением считается индекс шума "
                "(скачки датчиков день к дню против тех же календарных дней 2015–2022).")
    c1, c2, c3 = st.columns(3)
    sigma_s = c1.slider("Шум в период неисправности, σ", 0.1, 1.5, 0.5, 0.1, key="sig_s")
    thr_noise = c2.slider("Порог индекса шума", 1.1, 3.0, 1.5, 0.1)
    freq_s = c3.radio("Переобучение", ["M", "W"], format_func=lambda f: {"M": "ежемесячно", "W": "еженедельно"}[f], horizontal=True, key="fq_s")
    D, logs = director(sigma_s, C, alpha, thr_noise, freq_s)
    st.dataframe(D.round(3), hide_index=True, width="stretch")
    fig = go.Figure()
    LOG0 = logs["Фиксированный Elastic Net"][1]
    fig.add_scatter(x=LOG0.period, y=LOG0.noise_index, name="индекс шума", line=dict(color="black"))
    fig.add_hline(y=thr_noise, line_dash="dash", annotation_text="порог")
    fig.add_vrect(x0="2024-07-01", x1="2024-12-31", fillcolor="#D32F2F", opacity=0.08, line_width=0, annotation_text="неисправность")
    fig.update_layout(height=360, title="Обнаружение: индекс шума перед каждым переобучением", yaxis_title="индекс (1 = обычно)")
    st.plotly_chart(fig, width="stretch")
    st.info("Лекция советует при шуме сдвигать α к L2 («сохранить детали»). На наших данных лучше обратное — **α к L1 при том же C**: "
            "модель на время неисправности сокращается до ~8 признаков и меньше «собирает» шум (ноутбук 09: ROC-AUC 0.831 против 0.808). "
            "«Перегиб» (ещё и C / 6) обнуляет все признаки — сильнее не значит лучше. Базовые C и α Elastic Net — из боковой панели; "
            "в ноутбуке использованы выбранные по валидации C = 0.01, α = 0.25.")

# ---------------------------------------------------------------- SHAP
with tabs[4]:
    s_method = st.radio("Модель", list(ACTORS), horizontal=True, key="shap_m")
    sv, w, base, p = linear_shap(C, ACTORS[s_method])
    imp = np.abs(sv).mean(0)
    rho = pd.Series(np.abs(w)).corr(pd.Series(imp), method="spearman")
    c1, c2 = st.columns(2)
    order = np.argsort(imp)[-15:]
    fig = go.Figure()
    fig.add_bar(y=[name(FEATURES[j]) for j in order], x=imp[order], orientation="h", name="средний |SHAP|", marker_color=METHOD_COLORS.get(s_method, "#8E44AD"))
    fig.add_bar(y=[name(FEATURES[j]) for j in order], x=np.abs(w[order]), orientation="h", name="|коэффициент|", marker_color="#BBBBBB")
    fig.update_layout(barmode="group", height=560, title=f"Глобально: SHAP и коэффициенты (ρ Спирмена = {rho:.2f})", legend=dict(orientation="h", y=1.05))
    c1.plotly_chart(fig, width="stretch")
    dates = test.date.dt.date.tolist()
    day = c2.date_input("День теста", value=pd.Timestamp("2024-05-30").date(), min_value=dates[0], max_value=dates[-1], format="DD.MM.YYYY")
    i = dates.index(day)
    s = pd.Series(sv[i], index=[name(f) for f in FEATURES])
    s = s[s.abs().sort_values().index[-12:]]
    fig = go.Figure(go.Bar(y=s.index, x=s.values, orientation="h", marker_color=np.where(s > 0, "#C0392B", "#2E8B57")))
    fig.add_vline(x=0, line_color="black")
    fig.update_layout(height=500, title=f"{day:%d.%m.%Y}: логит = база {base:.2f} + сумма вкладов {sv[i].sum():+.2f} → P = {p[i]:.2f}; "
                                        f"факт: {'дождь' if y_te[i] else 'сухо'}", xaxis_title="Вклад SHAP в логит (> 0 — к дождю)")
    c2.plotly_chart(fig, width="stretch")
    st.caption("Для линейной модели SHAP точно равен w·(z − z̄): то же, что коэффициент, умноженный на отклонение признака дня от среднего. "
               "Поэтому глобальная важность SHAP повторяет |коэффициенты|, а отличается лишь для признаков с особым распределением "
               "(например, осадки: в большинство дней 0).")

# ---------------------------------------------------------------- multiclass
with tabs[5]:
    Mt, shp = multiclass(C, alpha)
    st.dataframe(Mt.round(3), hide_index=True, width="stretch")
    s_method = st.radio("Модель для SHAP по классам", list(shp), horizontal=True, key="mc_m")
    imp3, W3 = shp[s_method]
    cols = st.columns(3)
    for k_, col in enumerate(cols):
        s = pd.Series(imp3[:, k_], index=[name(f) for f in FEATURES]).sort_values()[-10:]
        sign = pd.Series(W3[k_], index=[name(f) for f in FEATURES])[s.index]
        fig = go.Figure(go.Bar(y=s.index, x=s.values, orientation="h", marker_color=["#C0392B" if v > 0 else "#2E8B57" for v in sign],
                               text=["+" if v > 0 else "−" for v in sign], textposition="outside"))
        fig.update_layout(height=430, title=f"«{WEATHER_TYPES[k_]}»", xaxis_title="средний |SHAP|", margin=dict(l=10, r=10))
        col.plotly_chart(fig, width="stretch")
    st.caption("Красный «+» — признак повышает шанс класса, зелёный «−» — понижает. Лекция: «в мультиклассе каждая метка — оттенок; "
               "SHAP показывает, какой голос задал тон для каждого».")
