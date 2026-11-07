import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

from common import (FEATURES, ROOT, TARGET, drift_null, header, lime_values, load_all, load_models, load_reference,
                    mean_distance, name, normalize, pca_probability_map, prepare_new_data, psi, shap_values, source_badge,
                    splits, test_shap, to_pca)

header("🆕 Новые данные и дрейф",
       "Лаб. №3: модель уже обучена и сохранена — можно ли доверять её прогнозам на данных, которых она не видела?")
source_badge()
datasets, _, _ = load_all()
rain_model, thr, _, _ = load_models()
train, _, test = splits(datasets["chisinau"])
X_ref, ref_names, *_ = load_reference()

# ---------------------------------------------------------------- 1–2. upload and column check
st.subheader("1. Загрузка нового файла и проверка столбцов")
sample = ROOT / "data" / "raw" / "new" / "chisinau_daily_2026.csv"
c1, c2 = st.columns([1.3, 1])
up = c1.file_uploader("CSV: суточные измерения (как из Open-Meteo) или готовые 26 признаков в порядке обучения", type="csv")
if sample.exists():
    c2.download_button("⬇️ Образец: Кишинёв 2026", sample.read_bytes(), sample.name, "text/csv")
    c2.download_button("⬇️ Шаблон признаков (порядок столбцов)", ",".join(ref_names).encode(), "feature_names.csv", "text/csv")
if up is not None:
    raw, src = pd.read_csv(up), up.name
elif sample.exists():
    raw, src = pd.read_csv(sample), f"{sample.name} (образец, 2026 год)"
else:
    st.info("Загрузите CSV-файл.")
    st.stop()
shift = st.sidebar.slider("Сценарий: сдвиг барометра, гПа", -10.0, 10.0, 0.0, 0.5,
                          help="Имитация неисправного датчика давления — заметит ли его детектор?")

feats, err, kind = prepare_new_data(raw)
if err:
    st.error(f"Анализ остановлен: {err}")
    st.stop()
if shift:
    feats["pressure_msl_mean"] = feats["pressure_msl_mean"] + shift
X_new = feats[ref_names]
has_dates = "date" in feats
st.success(f"Файл «{src}»: {len(feats)} наблюдений; "
           f"{'признаки рассчитаны из суточных измерений (лаги — из хранилища)' if kind == 'daily' else 'готовые признаки'}; "
           f"все 26 признаков на месте, порядок совпадает с обучением." + (f" Сдвиг барометра {shift:+.1f} гПа." if shift else ""))

# ---------------------------------------------------------------- 3. predictions
st.subheader("2. Прогноз сохранённой модели")
p_new = rain_model.predict_proba(X_new)[:, 1]
pred = (p_new >= thr).astype(int)
out = pd.DataFrame({"Дата": feats.date.dt.date if has_dates else np.arange(len(feats)),
                    "P(дождь завтра)": p_new.round(3), "Прогноз": np.where(pred == 1, "дождь", "сухо"),
                    "Неуверенный (|P − порог| < 0.05)": np.abs(p_new - thr) < 0.05})
k = st.columns(4)
k[0].metric("Наблюдений", len(feats))
k[1].metric("Объявлено «дождь завтра»", f"{pred.sum()}", f"{pred.mean():.0%}", delta_color="off")
k[2].metric("Неуверенных прогнозов", int(out.iloc[:, 3].sum()))
k[3].metric("Средняя вероятность", f"{p_new.mean():.3f}", f"на тесте {rain_model.predict_proba(test[FEATURES])[:, 1].mean():.3f}",
            delta_color="off")
c1, c2 = st.columns([1.6, 1])
if has_dates:
    fig = go.Figure(go.Scatter(x=feats.date, y=p_new, mode="lines", line=dict(color="#2B6CB0"), name="P(дождь завтра)"))
    fig.add_hline(y=thr, line_dash="dash", line_color="black", annotation_text=f"порог {thr:.2f}")
    fig.update_layout(height=330, title="Вероятность дождя на следующий день", yaxis_title="P", margin=dict(t=40))
    c1.plotly_chart(fig, width="stretch")
c2.dataframe(out, hide_index=True, width="stretch", height=330)
st.caption("Вероятность 0.82 означает оценку модели, а не гарантию 82 %: проверить её можно только по реальным исходам (раздел 6).")

# ---------------------------------------------------------------- 4. PCA map
st.subheader("3. Где лежат новые точки: карта вероятностей в плоскости PCA")
gx, gy, P = pca_probability_map()
T2, N2 = to_pca(X_ref), to_pca(X_new)
fig = go.Figure(go.Contour(x=gx, y=gy, z=P, colorscale="Blues", opacity=0.8, contours=dict(coloring="heatmap"),
                           line=dict(width=0), colorbar=dict(title="P(дождь)")))
fig.add_contour(x=gx, y=gy, z=P, contours=dict(start=thr, end=thr, coloring="none"), line=dict(color="black", width=2), showscale=False,
                showlegend=False, hoverinfo="skip")
sub = np.random.default_rng(1).choice(len(T2), min(800, len(T2)), replace=False)
fig.add_scatter(x=T2[sub, 0], y=T2[sub, 1], mode="markers", marker=dict(color="lightgray", size=4), name="обучение (800 дней)")
for cls, col, sym, nm in [(0, "#E0A030", "circle", "новые: прогноз «сухо»"), (1, "#C0392B", "triangle-up", "новые: прогноз «дождь»")]:
    m = pred == cls
    fig.add_scatter(x=N2[m, 0], y=N2[m, 1], mode="markers", name=f"{nm} ({m.sum()})",
                    marker=dict(color=col, symbol=sym, size=8, line=dict(color="black", width=0.6)),
                    text=[f"{d}: P = {v:.2f}" for d, v in zip(out["Дата"][m], p_new[m])], hovertemplate="%{text}<extra></extra>")
fig.update_layout(height=560, xaxis_range=[-8, 8], yaxis_range=[-6, 8], legend=dict(orientation="h", y=1.05),
                  xaxis_title="PC1", yaxis_title="PC2",
                  title="Чёрная линия — порог в плоскости PCA (приближение: две компоненты сохраняют ≈ 43 % дисперсии)")
st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------- 5. drift
st.subheader("4. Детектор дрейфа")
null = drift_null(len(X_new))
d_new = mean_distance(X_new)
q99 = float(np.quantile(null, 0.99))
c1, c2 = st.columns([1, 1.3])
drift_thr = c1.slider("Порог дрейфа (средняя дистанция до центра обучения в PCA)", float(np.round(null.min(), 2)),
                      float(np.round(max(null.max(), d_new) + 0.3, 2)), float(np.round(q99, 3)), 0.001,
                      help="По умолчанию — 99-й процентиль для случайных лотов того же размера из обучения")
is_drift = d_new > drift_thr
share_false = float((null > drift_thr).mean())
c1.metric("Средняя дистанция нового лота", f"{d_new:.3f}", f"порог {drift_thr:.3f}", delta_color="off")
(c1.error if is_drift else c1.success)("⚠️ Дрейф: новые данные дальше от обучения, чем обычный лот" if is_drift
                                       else "Дрейфа не обнаружено")
c1.caption(f"При этом пороге детектор ошибочно срабатывает на {share_false:.1%} случайных лотов из самих данных обучения. "
           "Ниже порог — больше ложных тревог, выше — пропуск реального дрейфа.")
fig = go.Figure(go.Histogram(x=null, nbinsx=50, marker_color="#BBBBBB", name="случайные лоты обучения"))
fig.add_vline(x=drift_thr, line_dash="dash", line_color="black", annotation_text="порог")
fig.add_vline(x=d_new, line_color="#D32F2F", line_width=3, annotation_text="новый лот", annotation_position="top left")
fig.update_layout(height=320, xaxis_title="Средняя дистанция до центра", showlegend=False, margin=dict(t=30),
                  xaxis_range=[min(null.min(), d_new) - 0.05, max(null.max(), d_new, drift_thr) + 0.08])
c2.plotly_chart(fig, width="stretch")

months = feats.date.dt.month.unique() if has_dates else np.arange(1, 13)
ref_m = train[train.date.dt.month.isin(months)][FEATURES]
per = pd.DataFrame({"Признак": [name(f) for f in FEATURES],
                    "PSI": [psi(ref_m[f].to_numpy(), X_new[f].to_numpy()) for f in FEATURES],
                    "Среднее (обучение, те же месяцы)": ref_m.mean().to_numpy(), "Среднее (новые)": X_new[FEATURES].mean().to_numpy()}
                   ).sort_values("PSI", ascending=False)
c1, c2 = st.columns([1.2, 1])
top8 = per.head(8).iloc[::-1]
fig = px.bar(top8, x="PSI", y="Признак", orientation="h", color=top8.PSI > 0.25,
             color_discrete_map={True: "#C0392B", False: "#6C8EBF"}, title="Проверка по каждому признаку (PSI)")
fig.add_vline(x=0.25, line_dash="dash", annotation_text="0.25 — значимый сдвиг", annotation_position="bottom right")
fig.update_layout(height=360, showlegend=False, yaxis_title="")
c1.plotly_chart(fig, width="stretch")
c2.dataframe(per.round(3), hide_index=True, width="stretch", height=360)
st.caption("Общий детектор отвечает «что-то изменилось», PSI — «что именно». Сдвиг одного датчика (попробуйте ползунок барометра "
           "слева) общий детектор почти не замечает, а PSI сразу находит.")

# ---------------------------------------------------------------- 6. SHAP + LIME
st.subheader("5. Объяснения на новых данных: SHAP и LIME")
sv_new, base = shap_values(X_new)
sv_te, _ = test_shap()
imp = pd.DataFrame({"Тест 2024–2025": np.abs(sv_te).mean(0), "Новые данные": np.abs(sv_new).mean(0)},
                   index=[name(f) for f in FEATURES])
rho = imp.rank().corr().iloc[0, 1]
top = imp.sum(axis=1).sort_values().index[-12:]
c1, c2 = st.columns(2)
fig = go.Figure()
for col, color in [("Тест 2024–2025", "#6C8EBF"), ("Новые данные", "#2ECC71")]:
    fig.add_bar(y=top, x=imp.loc[top, col], orientation="h", name=col, marker_color=color)
fig.update_layout(barmode="group", height=520, title=f"Важность признаков: тест vs новые (ρ Спирмена = {rho:.2f})",
                  xaxis_title="Средний |вклад SHAP|", legend=dict(orientation="h", y=1.08, x=0))
c1.plotly_chart(fig, width="stretch")
opts = list(range(len(feats)))
default = int(np.argmin(np.abs(p_new - thr)))
i = c2.selectbox("Наблюдение для локального объяснения", opts, index=default,
                 format_func=lambda r: f"{out['Дата'].iloc[r]} — P = {p_new[r]:.2f}")
lv, _ = lime_values(X_new.iloc[i].to_numpy())
Sn, Ln = normalize(sv_new[i]), normalize(lv)
corr = float(np.corrcoef(Sn, Ln)[0, 1])
t10 = np.argsort(np.abs(Sn) + np.abs(Ln))[-10:]
fig = go.Figure()
fig.add_bar(y=[name(FEATURES[t]) for t in t10], x=Sn[t10], orientation="h", name="SHAP", marker_color="#C0392B")
fig.add_bar(y=[name(FEATURES[t]) for t in t10], x=Ln[t10], orientation="h", name="LIME", marker_color="#6C8EBF")
fig.add_vline(x=0, line_color="black")
fig.update_layout(barmode="group", height=450, title=f"Корреляция SHAP–LIME = {corr:.2f}", legend=dict(orientation="h", y=1.1, x=0.6))
c2.plotly_chart(fig, width="stretch")
if rho < 0.8:
    st.warning("Порядок важности признаков заметно изменился — модель «рассуждает» на новых данных иначе, чем на тесте.")

# ---------------------------------------------------------------- 7. real validation
st.subheader("6. Проверка по реальным исходам")
known = feats[TARGET].notna().to_numpy() if TARGET in feats else np.zeros(len(feats), bool)
if known.sum() > 10 and feats.loc[known, TARGET].nunique() == 2:
    yk, pk = feats.loc[known, TARGET].astype(int).to_numpy(), p_new[known]
    y_t, p_t = test[TARGET].to_numpy(), rain_model.predict_proba(test[FEATURES])[:, 1]

    def scores(y, p):
        return {"ROC-AUC": roc_auc_score(y, p), "PR-AUC": average_precision_score(y, p), "F1": f1_score(y, p >= thr),
                "Recall": recall_score(y, p >= thr), "Precision": precision_score(y, p >= thr, zero_division=0),
                "Доля дождливых дней": y.mean()}

    val = pd.DataFrame({"Тест 2024–2025": scores(y_t, p_t), "Новые данные": scores(yk, pk)}).T
    if has_dates and "precipitation_sum" in feats:
        pers = (feats.loc[known, "precipitation_sum"] >= 1).astype(int).to_numpy()
        val["F1 инерционного прогноза"] = [f1_score(y_t, test.precipitation_sum >= 1), f1_score(yk, pers)]
    st.dataframe(val.round(3), width="stretch")
    st.caption(f"Исход известен для {known.sum()} из {len(feats)} наблюдений. Без реальных меток ни детектор дрейфа, "
               "ни SHAP/LIME не заменяют проверку качества.")
    out["Факт"] = np.where(known, np.where(feats[TARGET] == 1, "дождь", "сухо"), "неизвестно")
else:
    st.info("В файле нет реальных исходов (осадков следующего дня) — качество на новых данных пока не измерить.")

# ---------------------------------------------------------------- export
st.subheader("Экспорт результатов")
export = pd.concat([out.reset_index(drop=True), X_new.reset_index(drop=True)], axis=1)
c1, c2 = st.columns(2)
c1.download_button("⬇️ Прогнозы и вероятности (CSV)", export.to_csv(index=False).encode("utf-8-sig"), "new_predictions.csv", "text/csv")
c2.download_button("⬇️ Проверка дрейфа по признакам (CSV)", per.to_csv(index=False).encode("utf-8-sig"), "drift_psi.csv", "text/csv")
