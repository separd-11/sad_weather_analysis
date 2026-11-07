import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from common import C_DRY, C_WET, FEATURES, TARGET, header, load_all, name, source_badge, splits

header("🧭 Исследователь границ решений",
       "Аналог приложения из материалов курса — на данных нашей задачи: два признака, модель, шум и bias на X и y")
source_badge()
datasets, daily, _ = load_all()
train, _, test = splits(datasets["chisinau"])

sb = st.sidebar
sb.header("Настройки")
x1 = sb.selectbox("Признак X1", FEATURES, index=FEATURES.index("pressure_msl_mean"), format_func=name)
x2 = sb.selectbox("Признак X2", FEATURES, index=FEATURES.index("cloud_cover_mean"), format_func=name)
model_name = sb.selectbox("Модель", ["Логистическая регрессия", "SVM (RBF-ядро)", "KNN", "Random Forest", "Нейросеть (MLP)"])
params = {}
if model_name == "KNN":
    params["k"] = sb.slider("Число соседей k (меньше = сложнее)", 1, 201, 25, 2)
elif model_name == "Random Forest":
    params["leaf"] = sb.slider("Минимум дней в листе (меньше = сложнее)", 1, 200, 20)
    params["depth"] = sb.slider("Максимальная глубина (0 = без ограничения)", 0, 30, 0)
elif model_name == "SVM (RBF-ядро)":
    params["C"] = sb.select_slider("C (больше = сложнее)", options=[0.01, 0.1, 1, 10, 100], value=1)
elif model_name == "Нейросеть (MLP)":
    params["alpha"] = sb.select_slider("Регуляризация alpha", options=[1e-4, 1e-3, 1e-2, 1e-1, 1.0], value=1e-2)
sb.subheader("Искажения (в единицах станд. отклонения)")
sigma = sb.slider("Гауссов шум на X, σ", 0.0, 1.5, 0.0, 0.05)
bias = sb.slider("Bias на X (сдвиг)", 0.0, 1.0, 0.0, 0.05)
where = sb.radio("Где искажён X", ["только обучение", "обучение и тест"])
flip = sb.slider("Случайно перевёрнутые метки y, %", 0, 40, 0)
frozen = sb.checkbox("Bias y: «замёрзший осадкомер» (T завтра < +2 °C → дождь записан как сухой день)")
seed = sb.number_input("Случайное зерно", 0, 9999, 42)
band = sb.slider("Зона чувствительных точек: |P − порог| <", 0.0, 0.2, 0.05, 0.01,
                 help="Дни, где вероятность близка к порогу: малый шум может перевести их в другой класс")


class SVMScore:
    """SVM with a sigmoid of its decision function as the score, so that P = 0.5 is the SVM's own class-weighted boundary."""

    def __init__(self, C, seed):
        self.m = SVC(C=C, class_weight="balanced", random_state=seed)

    def fit(self, X, y):
        self.m.fit(X, y)
        return self

    def predict_proba(self, X):
        p = 1 / (1 + np.exp(-self.m.decision_function(X)))
        return np.c_[1 - p, p]


@st.cache_data(show_spinner="Обучение модели…")
def run(x1, x2, model_name, params, sigma, bias, where, flip, frozen, seed):
    rng = np.random.default_rng(seed)
    sc = StandardScaler().fit(train[[x1, x2]])
    A_tr, A_te = sc.transform(train[[x1, x2]]), sc.transform(test[[x1, x2]])
    y_tr, y_te = train[TARGET].to_numpy().copy(), test[TARGET].to_numpy()
    clean_tr = A_tr.copy()
    A_tr = A_tr + rng.normal(0, sigma, A_tr.shape) + bias if (sigma or bias) else A_tr
    if where == "обучение и тест" and (sigma or bias):
        A_te = A_te + rng.normal(0, sigma, A_te.shape) + bias
    y_noisy = y_tr.copy()
    if flip:
        m = rng.random(len(y_noisy)) < flip / 100
        y_noisy[m] = 1 - y_noisy[m]
    if frozen:
        t_next = train.date.map(daily["chisinau"].set_index("date").temperature_2m_mean.shift(-1)).to_numpy()
        y_noisy[(t_next < 2) & (y_tr == 1)] = 0

    def make():
        if model_name == "Логистическая регрессия":
            return LogisticRegression(class_weight="balanced")
        if model_name == "SVM (RBF-ядро)":
            return SVMScore(params["C"], seed)
        if model_name == "KNN":
            return KNeighborsClassifier(n_neighbors=params["k"])
        if model_name == "Random Forest":
            return RandomForestClassifier(n_estimators=150, min_samples_leaf=params["leaf"], max_depth=params["depth"] or None,
                                          class_weight="balanced", random_state=seed, n_jobs=-1)
        return make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), alpha=params["alpha"],
                                                             max_iter=400, random_state=seed))
        # no early stopping: on imbalanced data its validation accuracy stalls at the "always dry" level after ~10 epochs

    m = make().fit(A_tr, y_noisy)
    ref = LogisticRegression(class_weight="balanced").fit(clean_tr, y_tr)
    lo = np.percentile(np.r_[A_tr, A_te], 0.5, axis=0) - 0.5
    hi = np.percentile(np.r_[A_tr, A_te], 99.5, axis=0) + 0.5
    gx, gy = np.meshgrid(np.linspace(lo[0], hi[0], 200), np.linspace(lo[1], hi[1], 200))
    grid = np.c_[gx.ravel(), gy.ravel()]
    prob = m.predict_proba(grid)[:, 1].reshape(gx.shape)
    prob_ref = ref.predict_proba(grid)[:, 1].reshape(gx.shape)
    p_tr, p_te = m.predict_proba(A_tr)[:, 1], m.predict_proba(A_te)[:, 1]
    thr = 0.5 if model_name not in ("KNN", "Нейросеть (MLP)") else y_tr.mean()
    metrics = {"ROC-AUC обучение": roc_auc_score(y_noisy, p_tr), "ROC-AUC тест": roc_auc_score(y_te, p_te),
               "Balanced acc. тест": balanced_accuracy_score(y_te, p_te >= thr), "F1 тест": f1_score(y_te, p_te >= thr)}
    return gx, gy, prob, prob_ref, A_te, y_te, p_te, thr, metrics


gx, gy, prob, prob_ref, A_te, y_te, p_te, thr, met = run(x1, x2, model_name, params, sigma, bias, where, flip, frozen, seed)

c = st.columns(5)
c[0].metric("ROC-AUC (обучение)", f"{met['ROC-AUC обучение']:.3f}")
c[1].metric("ROC-AUC (тест)", f"{met['ROC-AUC тест']:.3f}")
gap = met["ROC-AUC обучение"] - met["ROC-AUC тест"]
c[2].metric("Разрыв обучение − тест", f"{gap:+.3f}", delta_color="off")
c[3].metric("Balanced accuracy (тест)", f"{met['Balanced acc. тест']:.3f}")
c[4].metric("F1 (тест)", f"{met['F1 тест']:.3f}")
if gap > 0.12:
    st.error("Переобучение: модель запоминает обучающие дни (и шум), на новых данных заметно хуже.")
elif met["ROC-AUC обучение"] < 0.72 and met["ROC-AUC тест"] < 0.72:
    st.warning("Похоже на недообучение или сильные искажения: модель плохо работает и на обучении, и на тесте.")
else:
    st.success("Баланс: качество на обучении и на тесте близко.")

fig = go.Figure()
fig.add_contour(x=gx[0], y=gy[:, 0], z=prob, colorscale=[[0, "#F6E3B9"], [0.5, "#FFFFFF"], [1, "#BBD3EE"]], opacity=0.85,
                contours=dict(start=0, end=1, size=0.05, coloring="heatmap"), line=dict(width=0),
                colorbar=dict(title="P(дождь)"), name="P(дождь)")
fig.add_contour(x=gx[0], y=gy[:, 0], z=prob, contours=dict(start=thr, end=thr, coloring="none"), line=dict(color="black", width=3),
                showscale=False, name=f"граница P = {thr:.2f}")
fig.add_contour(x=gx[0], y=gy[:, 0], z=prob_ref, contours=dict(start=0.5, end=0.5, coloring="none"),
                line=dict(color="gray", width=2, dash="dash"), showscale=False, name="граница LR на чистых данных")
for cls, col, nm, sym in [(0, C_DRY, "завтра без дождя", "circle"), (1, C_WET, "завтра дождь", "triangle-up")]:
    m = y_te == cls
    fig.add_scatter(x=A_te[m, 0], y=A_te[m, 1], mode="markers", marker=dict(color=col, size=6, symbol=sym, line=dict(width=0.3)),
                    name=nm)
near = np.abs(p_te - thr) < band
fig.add_scatter(x=A_te[near, 0], y=A_te[near, 1], mode="markers", name=f"чувствительные точки ({near.sum()})",
                marker=dict(size=11, color="rgba(0,0,0,0)", line=dict(color="#D32F2F", width=1.6)))
fig.update_layout(height=650, xaxis_title=f"X1: {name(x1)} (станд.)", yaxis_title=f"X2: {name(x2)} (станд.)",
                  title=f"Граница решений (чёрная линия: P = {thr:.2f}; красные кольца — чувствительные точки; пунктир — логистическая регрессия на чистых данных; точки — реальные тестовые дни)",
                  legend=dict(orientation="h", y=-0.12))
st.plotly_chart(fig, width="stretch")
st.subheader("Чувствительные точки у границы")
err_near = ((p_te[near] >= thr).astype(int) != y_te[near]).mean() if near.any() else 0.0
err_far = ((p_te[~near] >= thr).astype(int) != y_te[~near]).mean() if (~near).any() else 0.0
c = st.columns(3)
c[0].metric("Дней в зоне у границы", f"{near.sum()} из {len(near)}", f"{near.mean():.0%} теста", delta_color="off")
c[1].metric("Доля ошибок у границы", f"{err_near:.0%}")
c[2].metric("Доля ошибок вдали от границы", f"{err_far:.0%}")
pts = pd.DataFrame({"Дата": test.date.dt.date.to_numpy(), name(x1): test[x1].to_numpy(), name(x2): test[x2].to_numpy(),
                    "P(дождь)": p_te.round(3), "Прогноз": (p_te >= thr).astype(int), "Факт": y_te,
                    "Удалённость от порога": np.abs(p_te - thr).round(3)})[near].sort_values("Удалённость от порога")
st.dataframe(pts, hide_index=True, width="stretch", height=240)
st.download_button("⬇️ Скачать чувствительные точки (CSV)", pts.to_csv(index=False).encode("utf-8-sig"),
                   "boundary_points.csv", "text/csv")
st.caption("Как в приложении преподавателя: эти дни первыми меняют класс при шуме и сдвигах — для них разумно выдавать «неуверенный прогноз».")

with st.expander("Как читать этот график"):
    st.markdown("""
- Фон — вероятность дождя, которую модель выдаёт в каждой точке плоскости; **чёрная линия** — порог решения (0.5, для KNN и MLP — средняя частота дождя).
- **Простая модель** (логистическая регрессия, KNN с большим k, Random Forest с крупными листьями) даёт гладкую границу; **сложная** (KNN с k = 1, лес без ограничений) — «островки» вокруг отдельных дней: это переобучение.
- **Шум на X** «размывает» точки обучения; **bias на X** сдвигает их — опасно, когда сдвинуто только обучение, а реальные данные нет.
- **Перевёрнутые метки** сбивают модель случайно; **«замёрзший осадкомер»** — систематически, и это сложнее заметить.
- Граница на плоскости двух признаков — упрощение: настоящая модель использует 26 признаков.
""")
