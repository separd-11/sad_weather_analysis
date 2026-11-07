import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from common import FEATURES, ROOT, TARGET, header, lime_values, load_all, load_models, name, source_badge, splits, test_shap
from xai_verdict import VERDICT_COLORS, VERDICTS, make_poster, suggest_verdict

header("🖼️ Вердикт и постер (к экзамену)",
       "Материалы XAI: выбрать случай → SHAP и LIME → вердикт «устойчива / смещена / искажена» → рефлексия → постер")
source_badge()
datasets, _, _ = load_all()
train, valid, test = splits(datasets["chisinau"])
test = test.reset_index(drop=True)
X_te, y_te = test[FEATURES], test[TARGET].to_numpy()
rf, thr, *_ = load_models()
p_te = rf.predict_proba(X_te)[:, 1]
sv, base = test_shap()
detail_path = ROOT / "data" / "processed" / "xai_shap_lime_detail.csv"


@st.cache_resource(show_spinner="Elastic Net (вторая модель)…")
def enet_model():
    import joblib
    from regularization import make_model
    b = joblib.load(ROOT / "models" / "regularization_best.joblib")["Elastic Net"]
    return make_model(b["C"], b["l1_ratio"]).fit(train[FEATURES], train[TARGET]), b["threshold"]


@st.cache_data(show_spinner="Вердикты по всем 730 дням…")
def all_verdicts():
    d = pd.read_csv(detail_path, parse_dates=["Дата"])
    L = d.pivot(index="Дата", columns="Признак", values="LIME (норм.)")
    order = [name(f) for f in FEATURES]
    L = L[order].to_numpy()
    rows = []
    for i in range(len(test)):
        v, reasons, nums = suggest_verdict(p_te[i], thr, sv[i], L[i])
        rows.append({"Дата": test.date.iloc[i].date(), "Вердикт": v, "P(дождь)": p_te[i], "Факт": int(y_te[i]),
                     "Ошибка": int((p_te[i] >= thr) != y_te[i]), "Доля главного признака": nums["dominance"],
                     "Главный признак": name(FEATURES[int(np.argmax(np.abs(sv[i])))]), "Корреляция SHAP–LIME": nums["corr"]})
    return pd.DataFrame(rows)


tab_p, tab_two, tab_all = st.tabs(["Постер для одного дня", "Две модели: общие ошибки", "Вердикты по всем дням"])

with tab_p:
    dates = test.date.dt.date.tolist()
    presets = {"Верно предсказанный дождь (30.05.2024)": "2024-05-30", "Ложная тревога (12.09.2024)": "2024-09-12",
               "Пропущенный дождь (10.06.2025)": "2025-06-10", "Свой выбор": None}
    c1, c2 = st.columns(2)
    pick = c1.radio("Случай", list(presets))
    day = pd.Timestamp(presets[pick]).date() if presets[pick] else c2.date_input("День теста", value=dates[100], min_value=dates[0],
                                                                                   max_value=dates[-1], format="DD.MM.YYYY")
    i = dates.index(day)
    lv, rules = lime_values(X_te.iloc[i].to_numpy())
    verdict0, reasons, nums = suggest_verdict(p_te[i], thr, sv[i], lv)
    nxt = test.date.iloc[i] + pd.Timedelta(days=1)
    k = st.columns(4)
    k[0].metric(f"P(дождь {nxt:%d.%m})", f"{p_te[i]:.2f}", "дождь" if p_te[i] >= thr else "сухо", delta_color="off")
    k[1].metric("Факт", "дождь" if y_te[i] else "сухо", "верно" if (p_te[i] >= thr) == y_te[i] else "ошибка", delta_color="off")
    k[2].metric("Корреляция SHAP–LIME", f"{nums['corr']:.2f}")
    k[3].metric("Доля главного признака", f"{nums['dominance']:.0%}")
    st.markdown(f"**Предложенный вердикт: <span style='color:{VERDICT_COLORS[verdict0]}'>{verdict0}</span>** — "
                + "; ".join(reasons), unsafe_allow_html=True)
    verdict = st.radio("Ваш вердикт", VERDICTS, index=VERDICTS.index(verdict0), horizontal=True)
    top = name(FEATURES[int(np.argmax(np.abs(sv[i])))])
    default_refl = {
        "Устойчива": f"Модель выслушала несколько «голосов» — {top.lower()} и другие признаки — и оба метода объяснения согласны. "
                     f"Такому объяснению можно доверять, даже если сам прогноз ошибся.",
        "Смещена": f"Модель смотрит почти только на один признак — {top.lower()}. Как в притче: «смотрит только на солнце и не видит неба».",
        "Искажена": "Объяснения расходятся или прогноз стоит на самой границе решения: малое изменение данных поменяло бы вывод. "
                    "Этот день нужно проверять вручную.",
    }[verdict]
    reflection = st.text_area("Рефлексия (своими словами)", value=default_refl, height=90)
    shap_s = pd.Series(sv[i], index=[name(f) for f in FEATURES])
    lime_s = pd.Series(lv, index=[name(f) for f in FEATURES])
    png = make_poster(f"Прогноз дождя на {nxt:%d.%m.%Y} — Кишинёв",
                      f"P(дождь) = {p_te[i]:.2f} при пороге {thr:.2f} → {'дождь' if p_te[i] >= thr else 'сухо'}; "
                      f"факт: {'дождь' if y_te[i] else 'сухо'} · Random Forest, 26 признаков · база SHAP {base:.2f}",
                      shap_s, lime_s, verdict, reasons, reflection)
    st.image(png, width="stretch")
    st.download_button("⬇️ Скачать постер (PNG)", png, f"poster_{day:%Y%m%d}.png", "image/png", type="primary")

with tab_two:
    enet, enet_thr = enet_model()
    p_en = enet.predict_proba(X_te)[:, 1]
    z = enet[0].transform(X_te)
    contrib = enet[-1].coef_[0] * (z - enet[0].transform(train[FEATURES]).mean(0))   # = SHAP of the linear model
    wrong_rf, wrong_en = (p_te >= thr) != y_te, (p_en >= enet_thr) != y_te
    both = np.where(wrong_rf & wrong_en)[0]
    k = st.columns(4)
    k[0].metric("Ошибок Random Forest", int(wrong_rf.sum()))
    k[1].metric("Ошибок Elastic Net", int(wrong_en.sum()))
    k[2].metric("Общих ошибок", len(both), f"{len(both) / max(wrong_rf.sum(), 1):.0%} ошибок RF", delta_color="off")
    agree = sum(np.argmax(np.abs(sv[j])) == np.argmax(np.abs(contrib[j])) for j in both)
    k[3].metric("Один и тот же «ответственный»", f"{agree} из {len(both)}")
    rows = [{"Дата": test.date.iloc[j].date(), "Факт": "дождь" if y_te[j] else "сухо", "P RF": round(p_te[j], 3),
             "P ENet": round(p_en[j], 3), "Ответственный (RF)": name(FEATURES[int(np.argmax(np.abs(sv[j])))]),
             "Ответственный (ENet)": name(FEATURES[int(np.argmax(np.abs(contrib[j])))]),
             "Тип ошибки": "пропуск дождя" if y_te[j] else "ложная тревога"} for j in both]
    tbl = pd.DataFrame(rows)
    st.dataframe(tbl, hide_index=True, width="stretch", height=380)
    c1, c2 = st.columns(2)
    c1.plotly_chart(px.bar(tbl["Ответственный (RF)"].value_counts().head(8)[::-1], orientation="h",
                           labels={"value": "Общих ошибок", "index": ""}, title="«Ответственный» признак у Random Forest"),
                    width="stretch")
    c2.plotly_chart(px.bar(tbl["Тип ошибки"].value_counts(), labels={"value": "Дней", "index": ""}, title="Тип общих ошибок"),
                    width="stretch")
    st.caption("Ошибки, которые делают обе модели — и нелинейный лес, и линейная модель — чаще всего не «ошибка модели», а "
               "нехватка информации в суточных данных (например, конвективная гроза при высоком давлении, ноутбук 08).")

with tab_all:
    if not detail_path.exists():
        st.error("Нет data/processed/xai_shap_lime_detail.csv — запустите ноутбук 08.")
    else:
        V = all_verdicts()
        s = V.groupby("Вердикт").agg(Дней=("Дата", "size"), Ошибок=("Ошибка", "mean"), **{"Средняя доля главного признака": ("Доля главного признака", "mean")})
        k = st.columns(3)
        for col, v in zip(k, VERDICTS):
            if v in s.index:
                col.metric(v, f"{int(s.loc[v, 'Дней'])} дней", f"ошибок {s.loc[v, 'Ошибок']:.0%}", delta_color="off")
        st.plotly_chart(px.histogram(V, x="P(дождь)", color="Вердикт", color_discrete_map=VERDICT_COLORS, nbins=40, barmode="stack",
                                     title="Вердикты по вероятности прогноза (вертикаль — порог)").add_vline(x=thr, line_dash="dash"),
                        width="stretch")
        st.dataframe(V.round(3), hide_index=True, width="stretch", height=300)
