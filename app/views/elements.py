import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from common import (COORDS, FEATURES, STATIONS, TARGET, TARGET3, compute_features, header, load_all, load_models,
                    source_badge, splits)

header("⚖️ Фундаментальные элементы: Bias, Noise, Fitting, SMOTE",
       "Интерактивные версии экспериментов из ноутбуков 03, 05 и 06")
source_badge()
datasets, daily, _ = load_all()
rain_model, THR, _, _ = load_models()
train, valid, test = splits(datasets["chisinau"])
RF = dict(n_estimators=150, max_depth=12, min_samples_leaf=20, max_features=0.5)

tab_bias, tab_noise, tab_fit, tab_smote, tab_sum = st.tabs(
    ["Bias: география", "Noise и denoising", "Underfitting / Overfitting", "SMOTE", "Итог: принятые меры"])

# ------------------------------------------------------------------ bias
with tab_bias:
    st.markdown("Модель обучена **только на Кишинёве**. Справедлива ли она для остальной страны? Сравниваем с локальной моделью каждого пункта.")

    @st.cache_data(show_spinner="Проверка на всех станциях…")
    def geo():
        rows = []
        for s in STATIONS:
            tr, va, te = splits(datasets[s])
            p = rain_model.predict_proba(te[FEATURES])[:, 1]
            loc = RandomForestClassifier(random_state=0, n_jobs=-1, **RF).fit(tr[FEATURES], tr[TARGET])
            pl = loc.predict_proba(te[FEATURES])[:, 1]
            rows.append({"Станция": STATIONS[s], "lat": COORDS[s][0], "lon": COORDS[s][1],
                         "% дней с дождём": te[TARGET].mean() * 100, "Средняя вероятность, %": p.mean() * 100,
                         "ROC-AUC (модель Кишинёва)": roc_auc_score(te[TARGET], p), "F1 (модель Кишинёва)": f1_score(te[TARGET], p >= THR),
                         "ROC-AUC (локальная модель)": roc_auc_score(te[TARGET], pl)})
        return pd.DataFrame(rows)
    g = geo()
    c1, c2 = st.columns([1, 1.4])
    fig = px.scatter(g, x="lon", y="lat", size="% дней с дождём", color="F1 (модель Кишинёва)", text="Станция",
                     color_continuous_scale="RdYlGn", range_color=(0.4, 0.6), size_max=40,
                     labels={"lon": "Долгота", "lat": "Широта"}, title="F1 модели Кишинёва по станциям")
    fig.update_traces(textposition="bottom center"); fig.update_layout(height=520)
    fig.update_yaxes(scaleanchor="x", scaleratio=1.4)
    c1.plotly_chart(fig, width="stretch")
    c2.dataframe(g.drop(columns=["lat", "lon"]).round(3), hide_index=True, width="stretch")
    gg = g.melt(id_vars="Станция", value_vars=["ROC-AUC (модель Кишинёва)", "ROC-AUC (локальная модель)"], var_name="Модель", value_name="ROC-AUC")
    c2.plotly_chart(px.bar(gg, x="Станция", y="ROC-AUC", color="Модель", barmode="group", range_y=(0.7, 0.82)), width="stretch")
    st.info("Перенос модели честный: локальные модели почти не лучше. На юге (Комрат, Кагул) модель слегка завышает вероятность дождя — "
            "мера: свой порог для каждого региона.")

# ------------------------------------------------------------------ noise
with tab_noise:
    st.markdown("Искажаем **сырые показания датчиков** за 2024–2025, пересчитываем признаки и смотрим, как меняется прогноз. "
                "Фильтры — только **каузальные** (по прошлым дням), как в реальном времени.")
    SENS = ["temperature_2m_max", "temperature_2m_min", "temperature_2m_mean", "dew_point_2m_mean", "relative_humidity_2m_mean",
            "relative_humidity_2m_max", "pressure_msl_mean", "pressure_msl_max", "pressure_msl_min", "cloud_cover_mean",
            "wind_speed_10m_max", "soil_moisture_0_to_7cm_mean", "vapour_pressure_deficit_max", "sunshine_duration"]
    raw = daily["chisinau"].copy()
    c1, c2, c3 = st.columns(3)
    kind = c1.radio("Тип шума", ["Гауссов", "Импульсный (выбросы ±6σ)"])
    level = c2.slider("Сила: σ для гауссова / доля выбросов, %", 0.0, 1.0 if kind == "Гауссов" else 20.0,
                      0.3 if kind == "Гауссов" else 5.0, 0.05 if kind == "Гауссов" else 1.0)
    flt = c3.radio("Denoising", ["Без фильтра", "Медиана 3 дня", "EWMA α=0.5", "Хампель"])

    @st.cache_data(show_spinner="Пересчёт…")
    def noisy_eval(kind, level, flt):
        r = raw.copy()
        period = (r.date >= test.date.min() - pd.Timedelta(days=14)).values
        std = r.loc[r.date.dt.year <= 2022, SENS].std().values
        rng = np.random.default_rng(42)
        block = r.loc[period, SENS].values.astype(float)
        if kind == "Гауссов":
            block = block + rng.normal(0, 1, block.shape) * level * std
        else:
            mask = rng.random(block.shape) < level / 100
            block = np.where(mask, block + rng.choice([-6, 6], block.shape) * std, block)
        r[SENS] = r[SENS].astype(float)
        r.loc[period, SENS] = block
        for c in ["relative_humidity_2m_mean", "relative_humidity_2m_max", "cloud_cover_mean"]:
            r[c] = r[c].clip(0, 100)
        for c in ["wind_speed_10m_max", "soil_moisture_0_to_7cm_mean", "vapour_pressure_deficit_max"]:
            r[c] = r[c].clip(lower=0)
        r["sunshine_duration"] = r["sunshine_duration"].clip(0, r["daylight_duration"])
        noisy_p = r.pressure_msl_mean.copy()
        for c in SENS:
            s = r[c]
            if flt == "Медиана 3 дня":
                r[c] = s.rolling(3, min_periods=1).median()
            elif flt == "EWMA α=0.5":
                r[c] = s.ewm(alpha=0.5, adjust=False).mean()
            elif flt == "Хампель":
                med = s.rolling(5, min_periods=1).median()
                mad = (s - med).abs().rolling(5, min_periods=1).median() * 1.4826
                r[c] = s.where((s - med).abs() <= 3 * mad.replace(0, np.inf), med)
        t = test[["date", TARGET]].merge(compute_features(r), on="date")
        p = rain_model.predict_proba(t[FEATURES])[:, 1]
        return roc_auc_score(t[TARGET], p), f1_score(t[TARGET], p >= THR), r.date, noisy_p, r.pressure_msl_mean

    auc, f1, dates, noisy_p, used_p = noisy_eval(kind, level, flt)
    auc0, f10, *_ = noisy_eval("Гауссов", 0.0, "Без фильтра")
    m1, m2 = st.columns(2)
    m1.metric("ROC-AUC прогноза", f"{auc:.3f}", f"{auc - auc0:+.3f} к чистым данным")
    m2.metric("F1 прогноза", f"{f1:.3f}", f"{f1 - f10:+.3f} к чистым данным")
    w = (dates >= "2024-10-01") & (dates <= "2024-12-15")
    fig = go.Figure()
    fig.add_scatter(x=dates[w], y=raw.pressure_msl_mean[w], name="истинное давление", line=dict(color="black", width=3))
    fig.add_scatter(x=dates[w], y=noisy_p[w], name="с шумом", line=dict(color="#C0392B", width=1))
    if flt != "Без фильтра":
        fig.add_scatter(x=dates[w], y=used_p[w], name=f"после фильтра: {flt}", line=dict(color="#2E8B57", dash="dash", width=2))
    fig.update_layout(title="Пример: давление, октябрь – декабрь 2024", yaxis_title="гПа", height=420, legend=dict(orientation="h"))
    st.plotly_chart(fig, width="stretch")
    st.caption("Вывод из ноутбука 05: EWMA помогает против гауссова шума, фильтр Хампеля — против выбросов; на чистых данных любой фильтр немного вредит.")

# ------------------------------------------------------------------ fitting
with tab_fit:
    st.markdown("Сложность Random Forest: **глубина деревьев** и **минимальное число дней в листе**. Сравниваем качество на обучении и на новых данных.")
    c1, c2 = st.columns(2)
    depth = c1.slider("Максимальная глубина", 1, 30, 12)
    leaf = c2.slider("Минимум дней в листе", 1, 200, 20)

    @st.cache_data(show_spinner="Обучение…")
    def fit_eval(depth, leaf):
        m = RandomForestClassifier(n_estimators=150, max_depth=depth, min_samples_leaf=leaf, max_features=0.5,
                                   random_state=0, n_jobs=-1).fit(train[FEATURES], train[TARGET])
        return (roc_auc_score(train[TARGET], m.predict_proba(train[FEATURES])[:, 1]),
                roc_auc_score(valid[TARGET], m.predict_proba(valid[FEATURES])[:, 1]),
                roc_auc_score(test[TARGET], m.predict_proba(test[FEATURES])[:, 1]))

    @st.cache_data(show_spinner="Строим кривую сложности…")
    def curve(leaf):
        return pd.DataFrame([(d, *fit_eval(d, leaf)) for d in [1, 2, 3, 4, 6, 8, 12, 16, 24, 30]],
                            columns=["Глубина", "Обучение", "Валидация 2023", "Тест 2024–2025"])
    a_tr, a_va, a_te = fit_eval(depth, leaf)
    m = st.columns(4)
    m[0].metric("ROC-AUC обучение", f"{a_tr:.3f}"); m[1].metric("ROC-AUC валидация", f"{a_va:.3f}")
    m[2].metric("ROC-AUC тест", f"{a_te:.3f}"); m[3].metric("Разрыв обучение − тест", f"{a_tr - a_te:+.3f}", delta_color="off")
    if a_tr - a_te > 0.15:
        st.error("Переобучение: модель запоминает обучающие дни.")
    elif a_tr < 0.76:
        st.warning("Недообучение: модель слишком проста.")
    else:
        st.success("Разумный баланс сложности.")
    cv = curve(leaf)
    fig = px.line(cv, x="Глубина", y=["Обучение", "Валидация 2023", "Тест 2024–2025"], markers=True,
                  title=f"Кривая сложности при минимуме {leaf} дней в листе", labels={"value": "ROC-AUC", "variable": ""})
    fig.add_vline(x=depth, line_dash="dash")
    st.plotly_chart(fig, width="stretch")

# ------------------------------------------------------------------ smote
with tab_smote:
    st.markdown("Редкий класс — **критическая погода завтра** (≈ 7 % дней). Без балансировки модель её не предсказывает.")
    c1, c2 = st.columns(2)
    strat = c1.radio("Стратегия", ["Без балансировки", "class_weight", "SMOTE"])
    ratio = c2.slider("Доля критического класса после SMOTE (от числа остальных дней)", 0.1, 1.0, 1.0, 0.05,
                      disabled=strat != "SMOTE")

    @st.cache_data(show_spinner="Обучение…")
    def smote_eval(strat, ratio):
        X, y = train[FEATURES].to_numpy(), (train[TARGET3] == 2).astype(int).to_numpy()
        kw = dict(RF, random_state=0, n_jobs=-1)
        if strat == "class_weight":
            kw["class_weight"] = "balanced"
        if strat == "SMOTE":
            X, y = SMOTE(sampling_strategy=ratio, random_state=0).fit_resample(X, y)
        m = RandomForestClassifier(**kw).fit(X, y)
        yt = (test[TARGET3] == 2).astype(int).to_numpy()
        p = m.predict_proba(test[FEATURES].to_numpy())[:, 1]; pred = p >= 0.5
        return {"Recall (крит.)": recall_score(yt, pred), "Precision (крит.)": precision_score(yt, pred, zero_division=0),
                "F1 (крит.)": f1_score(yt, pred), "ROC-AUC": roc_auc_score(yt, p), "Предсказано критических дней": int(pred.sum()),
                "Реальных критических дней": int(yt.sum())}
    r = smote_eval(strat, ratio)
    m = st.columns(5)
    m[0].metric("Найдено критических дней", f"{r['Recall (крит.)']:.0%}", f"из {r['Реальных критических дней']}", delta_color="off")
    m[1].metric("Precision", f"{r['Precision (крит.)']:.2f}"); m[2].metric("F1", f"{r['F1 (крит.)']:.2f}")
    m[3].metric("ROC-AUC", f"{r['ROC-AUC']:.3f}"); m[4].metric("Тревог выдано", r["Предсказано критических дней"])
    st.caption("Из ноутбука 05: SMOTE даёт голос редкому классу, но синтетические дни отличимы от реальных (adversarial AUC 0.74), "
               "поэтому в итоговой системе используется class_weight.")

# ------------------------------------------------------------------ summary
with tab_sum:
    st.markdown("""
| Фактор | Как обнаружен | Принятая мера и почему | Эффект |
|---|---|---|---|
| **Bias классов** | Recall дождя 0.30 против 0.97 у сухих дней при пороге 0.5 | порог по валидации + перевзвешивание классов — восстанавливают редкий класс без синтетики | Recall 0.30 → 0.76 |
| **Bias на признаке** | ROC-AUC 0.43 при искажённой облачности | исключение или перекалибровка признака — нормализация связь с классом не исправляет | ROC-AUC 0.43 → 0.79 |
| **Bias регион / сезон / прибор** | метрики по группам, сверка с соседями | контроль по сезонам и регионам, сверка барометра с соседними станциями | неисправность найдена за 2 дня |
| **Noise гауссов** | деградация с ростом σ | устойчивая модель (RF) + EWMA при сигнале контроля качества | ROC-AUC +0.01–0.015 |
| **Noise импульсный** | выбросы ±6σ | фильтр Хампеля — меняет только выбросы | ошибка сигнала ÷5 |
| **Шум меток** | 20 % инверсий | очистка меток по согласию с моделью | ROC-AUC 0.767 → 0.778 |
| **Систематическая ошибка меток** | «замёрзший осадкомер» | метки соседних станций — изнутри данных ошибку не видно | ROC-AUC 0.777 → 0.796 |
| **Overfitting** | AUC обучения 1.0 против 0.66–0.77 на тесте | регуляризация RF по кросс-валидации (глубина 12, лист ≥ 20) | разрыв 0.23 → 0.10 |
| **Дисбаланс** | 0 критических прогнозов | SMOTE на обучении / class_weight в системе | Recall 0 → 0.35–0.56 |
""")
