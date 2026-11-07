"""Shared data, models and helpers for the Streamlit presentation app."""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.base import clone
from sklearn.metrics import f1_score, precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from features import (FEATURE_NAMES_RU, FEATURES, RAIN_MM, TARGET, TARGET3, WEATHER_TYPES, build_features,  # noqa: E402
                      compute_features, weather_type)
from streaming import LEVELS, AlertEngine  # noqa: E402

FIG = ROOT / "figures"
STATIONS = {"chisinau": "Кишинёв", "soroca": "Сороки", "balti": "Бельцы", "ungheni": "Унгены", "comrat": "Комрат", "cahul": "Кагул"}
COORDS = {"chisinau": (47.01, 28.86), "soroca": (48.16, 28.30), "balti": (47.76, 27.93), "ungheni": (47.21, 27.80),
          "comrat": (46.30, 28.66), "cahul": (45.90, 28.19)}
C_DRY, C_WET = "#E0A030", "#2B6CB0"
C3 = ["#4CAF50", "#F2A93B", "#D32F2F"]
C_LEVEL = ["#9E9E9E", "#F5D33B", "#F08A24", "#D32F2F"]
MONTHS = ["Янв", "Фев", "Мар", "Апр", "Май", "Июн", "Июл", "Авг", "Сен", "Окт", "Ноя", "Дек"]
SEASONS = {12: "Зима", 1: "Зима", 2: "Зима", 3: "Весна", 4: "Весна", 5: "Весна",
           6: "Лето", 7: "Лето", 8: "Лето", 9: "Осень", 10: "Осень", 11: "Осень"}

FEATURE_NAMES = FEATURE_NAMES_RU


def name(feature: str) -> str:
    return FEATURE_NAMES.get(feature, feature)


@st.cache_data(show_spinner="Загрузка данных из хранилища…")
def load_all():
    """Dataset + daily measurements for all stations: from PostgreSQL, or from the CSV copies if the DB is unavailable."""
    try:
        from db import load_daily, load_dataset, query
        query("SELECT 1")
        daily = {s: load_daily(s) for s in STATIONS}
        datasets = {s: load_dataset(s) for s in STATIONS}
        source = "PostgreSQL (хранилище sad_weather)"
    except Exception as exc:  # the presentation must work without the database
        daily = {}
        for s in STATIONS:
            path = ROOT / "data" / "raw" / ("chisinau_daily_2015_2025.csv" if s == "chisinau" else f"cities/{s}_daily_2015_2025.csv")
            daily[s] = pd.read_csv(path, parse_dates=["date"])
        datasets = {s: build_features(d) for s, d in daily.items()}
        source = f"резервная копия CSV (база недоступна: {type(exc).__name__})"
    return datasets, daily, source


def db_available() -> bool:
    try:
        from db import query
        query("SELECT 1")
        return True
    except Exception:
        return False


def splits(data: pd.DataFrame):
    return (data[data.date.dt.year <= 2022], data[data.date.dt.year == 2023], data[data.date.dt.year >= 2024])


@st.cache_resource(show_spinner="Загрузка моделей…")
def load_models():
    datasets, _, _ = load_all()
    train, valid, _ = splits(datasets["chisinau"])
    bundle = joblib.load(ROOT / "models" / "baseline_model.joblib")
    rain_model = bundle["model"]
    crit_model = clone(rain_model).set_params(class_weight="balanced").fit(train[FEATURES], train[TARGET3])
    # alert thresholds — the same rules as in notebook 04
    pc_va = crit_model.predict_proba(valid[FEATURES])[:, 2]
    is_crit = (valid[TARGET3] == 2).astype(int)
    grid = np.round(np.arange(0.05, 0.95, 0.01), 2)
    thr_orange = grid[int(np.argmax([f1_score(is_crit, pc_va >= t) for t in grid]))]
    ok = [t for t in grid if (pc_va >= t).sum() >= 3 and precision_score(is_crit, pc_va >= t) >= 0.5]
    thr_red = max(min(ok) if ok else float(np.round(np.quantile(pc_va, 0.98), 2)), thr_orange + 0.05)
    alerts = AlertEngine(thr_rain=bundle["threshold"], thr_orange=thr_orange, thr_red=thr_red)
    return rain_model, bundle["threshold"], crit_model, alerts


def header(title: str, subtitle: str = ""):
    st.title(title)
    if subtitle:
        st.caption(subtitle)


def source_badge():
    _, _, source = load_all()
    if source.startswith("PostgreSQL"):
        st.sidebar.success(f"Источник данных: {source}")
    else:
        st.sidebar.warning(f"Источник данных: {source}")


# ---------------------------------------------------------------- explainability and new data (lab 3)
@st.cache_resource(show_spinner="Подготовка эталона обучения (PCA)…")
def load_reference():
    """Saved reference training data + the PCA plane used for the probability map and the drift detector."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    ref = joblib.load(ROOT / "models" / "reference_train.joblib")
    X_ref = ref["X_train"][ref["feature_names"]]
    scaler = StandardScaler().fit(X_ref)
    pca = PCA(2, random_state=42).fit(scaler.transform(X_ref))
    T2 = pca.transform(scaler.transform(X_ref))
    return X_ref, ref["feature_names"], scaler, pca, T2


def to_pca(X: pd.DataFrame) -> np.ndarray:
    _, names, scaler, pca, _ = load_reference()
    return pca.transform(scaler.transform(X[names]))


@st.cache_data(show_spinner="Карта вероятностей в плоскости PCA…")
def pca_probability_map(n: int = 120):
    """Probability background: PCA-plane points mapped back to 26 features (an approximation of the real boundary)."""
    rain_model, *_ = load_models()
    _, names, scaler, pca, _ = load_reference()
    gx, gy = np.meshgrid(np.linspace(-8, 8, n), np.linspace(-6, 8, n))
    back = scaler.inverse_transform(pca.inverse_transform(np.c_[gx.ravel(), gy.ravel()]))
    P = rain_model.predict_proba(pd.DataFrame(back, columns=names))[:, 1].reshape(gx.shape)
    return gx[0], gy[:, 0], P


@st.cache_data(show_spinner="Распределение дистанций для случайных лотов обучения…")
def drift_null(lot_size: int, n_lots: int = 2000):
    """Mean distance to the training centroid for random training lots of the same size — what is 'normal'."""
    *_, T2 = load_reference()
    centroid = T2.mean(0)
    rng = np.random.default_rng(42)
    size = min(lot_size, len(T2))
    return np.array([np.linalg.norm(T2[rng.choice(len(T2), size, replace=False)] - centroid, axis=1).mean()
                     for _ in range(n_lots)])


def mean_distance(X: pd.DataFrame) -> float:
    *_, T2 = load_reference()
    return float(np.linalg.norm(to_pca(X) - T2.mean(0), axis=1).mean())


def psi(ref_v, new_v, bins: int = 10) -> float:
    edges = np.unique(np.quantile(ref_v, np.linspace(0, 1, bins + 1)))
    r = np.histogram(np.clip(ref_v, edges[0], edges[-1]), edges)[0] / len(ref_v) + 1e-6
    n = np.histogram(np.clip(new_v, edges[0], edges[-1]), edges)[0] / len(new_v) + 1e-6
    return float(np.sum((n - r) * np.log(n / r)))


@st.cache_resource(show_spinner="Подготовка SHAP…")
def shap_explainer():
    import shap
    rain_model, *_ = load_models()
    return shap.TreeExplainer(rain_model)


def shap_values(X: pd.DataFrame):
    """SHAP contributions to P(rain) and the base value."""
    ex = shap_explainer()
    sv = np.asarray(ex.shap_values(X[FEATURES]))
    sv = sv[..., 1] if sv.ndim == 3 else sv
    ev = np.atleast_1d(ex.expected_value)
    return sv, float(ev[1] if len(ev) > 1 else ev[0])


@st.cache_resource(show_spinner="Подготовка LIME…")
def lime_explainer():
    from lime.lime_tabular import LimeTabularExplainer
    X_ref, names, *_ = load_reference()
    return LimeTabularExplainer(X_ref.to_numpy(), feature_names=names, class_names=["сухо", "дождь"],
                                discretize_continuous=True, mode="classification", random_state=42)


def lime_values(x, seed: int = 42, num_samples: int = 5000):
    """LIME weights for one day as a 26-vector, plus the human-readable rules."""
    rain_model, *_ = load_models()
    ex = lime_explainer()
    ex.random_state = np.random.RandomState(seed)
    e = ex.explain_instance(np.asarray(x, dtype=float),
                            lambda A: rain_model.predict_proba(pd.DataFrame(A, columns=FEATURES)),
                            num_features=len(FEATURES), num_samples=num_samples)
    v = np.zeros(len(FEATURES))
    for j, w in e.as_map()[1]:
        v[j] = w
    rules = {FEATURES[j]: r.replace(FEATURES[j], name(FEATURES[j])) for (j, _), (r, _) in zip(e.as_map()[1], e.as_list())}
    return v, rules


NORMALIZATIONS = {"sum": "на сумму модулей (наш вариант, порог 0.15)", "max": "на максимум модуля (DEC_XAI преподавателя, порог 0.5)"}
DEFAULT_DIFF_THR = {"sum": 0.15, "max": 0.5}
LEVELS4 = ["сильное согласие", "умеренная разница (ограничения LIME)", "большая разница (взаимодействия)",
           "очень большая разница (нестабильность)"]


def normalize(v, method: str = "sum") -> np.ndarray:
    """Scales each explanation vector: by the sum of |values| (shares of the total contribution) or, as in the corrected
    course app (docs/DEC_XAI), by the max |value| (the main feature becomes ±1)."""
    v = np.asarray(v, dtype=float)
    s = np.abs(v).sum(-1, keepdims=True) if method == "sum" else np.abs(v).max(-1, keepdims=True)
    return np.divide(v, s, out=np.zeros_like(v), where=s > 0)


def safe_corr(a, b) -> float:
    """Pearson correlation; undefined (NaN) when one of the vectors is constant, as in the corrected course app."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return np.nan if (a.std() == 0 or b.std() == 0) else float(np.corrcoef(a, b)[0, 1])


def level4(d) -> np.ndarray:
    """Interpretation of a SHAP–LIME difference on the max-normalized scale (4 levels of the corrected course app)."""
    d = np.asarray(d, dtype=float)
    return np.select([d < 0.2, d < 0.5, d < 0.8], LEVELS4[:3], LEVELS4[3])


def prepare_new_data(raw: pd.DataFrame):
    """Uploaded CSV → features.  Accepts daily measurements (features are computed, lags from the warehouse history)
    or ready-made 26 features in the training order.  Returns (features, error message, kind)."""
    _, daily, _ = load_all()
    if all(f in raw.columns for f in FEATURES):
        if [c for c in raw.columns if c in FEATURES] != FEATURES:
            return None, "Признаки есть, но их порядок отличается от порядка обучения — анализ остановлен.", "features"
        out = raw.copy()
        if "date" in out:
            out["date"] = pd.to_datetime(out["date"])
        return out, None, "features"
    expected = [c for c in daily["chisinau"].columns]
    if "date" not in raw.columns:
        return None, "В файле нет ни 26 признаков модели, ни столбца date с суточными измерениями.", "unknown"
    missing = [c for c in expected if c not in raw.columns]
    if missing:
        return None, f"Не хватает столбцов: {', '.join(missing)}", "daily"
    raw = raw[expected].copy()
    raw["date"] = pd.to_datetime(raw["date"])
    hist = daily["chisinau"][daily["chisinau"].date < raw.date.min()].tail(7)
    feats = compute_features(pd.concat([hist, raw], ignore_index=True))
    feats = feats[feats.date >= raw.date.min()].dropna(subset=FEATURES).reset_index(drop=True)
    nxt = raw.set_index("date").precipitation_sum.shift(-1)
    outcome = feats.date.map(nxt)
    feats[TARGET] = (outcome >= RAIN_MM).astype(float).where(outcome.notna())
    return feats, None, "daily"


@st.cache_data(show_spinner="SHAP по 730 тестовым дням…")
def test_shap():
    datasets, _, _ = load_all()
    _, _, test = splits(datasets["chisinau"])
    return shap_values(test[FEATURES])
