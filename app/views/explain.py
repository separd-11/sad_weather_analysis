import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from common import (C_DRY, C_WET, DEFAULT_DIFF_THR, FEATURES, LEVELS4, NORMALIZATIONS, ROOT, TARGET, header, level4, lime_values,
                    load_all, load_models, name, normalize, pca_probability_map, safe_corr, source_badge, splits, test_shap, to_pca)

header("🔍 Объяснения прогноза: SHAP и LIME",
       "Лаб. №3: почему модель приняла решение и можно ли доверять объяснению · тест 2024–2025")
source_badge()
datasets, _, _ = load_all()
rain_model, thr, _, _ = load_models()
_, _, test = splits(datasets["chisinau"])
test = test.reset_index(drop=True)
X_te, y_te = test[FEATURES], test[TARGET].to_numpy()
p_te = rain_model.predict_proba(X_te)[:, 1]


sv, base = test_shap()

# precomputed LIME for all test days (notebook 08) — recomputing 730 LIME runs live is too slow
detail_path = ROOT / "data" / "processed" / "xai_shap_lime_detail.csv"
detail = pd.read_csv(detail_path, parse_dates=["Дата"]) if detail_path.exists() else None

sb = st.sidebar
sb.header("Сравнение SHAP и LIME")
norm = sb.radio("Нормировка вкладов дня", list(NORMALIZATIONS), format_func=NORMALIZATIONS.get)
corr_thr = sb.slider("Корреляция SHAP–LIME ниже", 0.0, 1.0, 0.5, 0.05)
diff_thr = sb.slider("Разница по признаку больше", 0.05, 1.0, DEFAULT_DIFF_THR[norm], 0.01, key=f"diff_thr_{norm}")
st.sidebar.caption("Пороги — параметры анализа, а не универсальные границы. Пороги разных нормировок несопоставимы: "
                   "«0.5 от максимума» ≠ «0.15 от суммы». В ноутбуке 08 (раздел C.2): 253 FLAGGED-дня по методу DEC_XAI против 74 по нашему.")

tab_g, tab_l, tab_f = st.tabs(["Глобально: что важно для модели", "Один день: SHAP против LIME", "FLAGGED: где объяснения расходятся"])

with tab_g:
    imp = pd.Series(np.abs(sv).mean(0), index=[name(f) for f in FEATURES]).sort_values().tail(15)
    c1, c2 = st.columns(2)
    fig = px.bar(imp, orientation="h", labels={"value": "Средний |вклад SHAP| в P(дождь)", "index": ""},
                 title="Глобальная важность (15 главных признаков)")
    fig.update_layout(showlegend=False, height=520)
    c1.plotly_chart(fig, width="stretch")
    f = c2.selectbox("Зависимость вклада от значения признака", FEATURES, index=FEATURES.index("pressure_msl_mean"),
                     format_func=name)
    j = FEATURES.index(f)
    fig = px.scatter(x=X_te[f], y=sv[:, j], color=np.where(y_te == 1, "завтра дождь", "завтра сухо"),
                     color_discrete_map={"завтра дождь": C_WET, "завтра сухо": C_DRY}, opacity=0.7,
                     labels={"x": name(f), "y": "Вклад SHAP в P(дождь)", "color": ""})
    fig.add_hline(y=0, line_color="black")
    fig.update_layout(height=470, title=f"Как {name(f).split(',')[0].lower()} сдвигает вероятность дождя")
    c2.plotly_chart(fig, width="stretch")
    st.info(f"SHAP раскладывает каждую вероятность точно: база {base:.3f} (средний прогноз) + сумма вкладов 26 признаков "
            f"= вероятность модели. Положительный вклад тянет прогноз к дождю, отрицательный — к сухой погоде.")

with tab_l:
    dates = test.date.dt.date.tolist()
    cases = {"Верно предсказанный дождь (30.05.2024)": pd.Timestamp("2024-05-30").date(),
             "Ложная тревога (12.09.2024)": pd.Timestamp("2024-09-12").date(),
             "Пропущенный дождь (10.06.2025)": pd.Timestamp("2025-06-10").date(), "Свой выбор": None}
    c1, c2 = st.columns([1, 1])
    case = c1.radio("Пример", list(cases), horizontal=False)
    day = cases[case] or c2.date_input("День теста", value=dates[-1], min_value=dates[0], max_value=dates[-1])
    i = dates.index(day)
    seed = c2.number_input("Случайное зерно LIME (проверка стабильности)", 0, 999, 42)
    lv, rules = lime_values(X_te.iloc[i].to_numpy(), seed=int(seed))
    Sn, Ln = normalize(sv[i], norm), normalize(lv, norm)
    corr = safe_corr(Sn, Ln)
    diff = np.abs(Sn - Ln)
    lvl = level4(np.abs(normalize(sv[i], "max") - normalize(lv, "max")))
    flag_c, flag_d = corr < corr_thr, diff.max() > diff_thr
    nxt = test.date.iloc[i] + pd.Timedelta(days=1)

    k = st.columns(5)
    k[0].metric(f"P(дождь {nxt:%d.%m})", f"{p_te[i]:.2f}", "дождь" if p_te[i] >= thr else "сухо", delta_color="off")
    k[1].metric("Факт", "дождь" if y_te[i] == 1 else "сухо")
    k[2].metric("Корреляция SHAP–LIME", f"{corr:.2f}")
    k[3].metric("Макс. разница", f"{diff.max():.3f}", name(FEATURES[int(diff.argmax())]), delta_color="off")
    k[4].metric("Статус", "FLAGGED" if (flag_c or flag_d) else "OK")
    if flag_c or flag_d:
        reason = " и ".join(r for r, fl in [(f"корреляция < {corr_thr} (НИЗКОЕ СОГЛАСИЕ)", flag_c), (f"разница > {diff_thr}", flag_d)] if fl)
        flagged_f = ", ".join(name(FEATURES[j]) for j in np.where(diff > diff_thr)[0]) or "нет"
        st.warning(f"День помечен FLAGGED: {reason}. Признаки с большим расхождением: {flagged_f}. "
                   "Это сигнал проверить вручную, а не доказательство ошибки модели.")

    top = np.argsort(np.abs(Sn) + np.abs(Ln))[-12:]
    fig = go.Figure()
    fig.add_bar(y=[name(FEATURES[t]) for t in top], x=Sn[top], orientation="h", name="SHAP (норм.)", marker_color="#C0392B")
    fig.add_bar(y=[name(FEATURES[t]) for t in top], x=Ln[top], orientation="h", name="LIME (норм.)", marker_color="#6C8EBF")
    fig.add_vline(x=0, line_color="black")
    fig.update_layout(barmode="group", height=560, legend=dict(orientation="h", y=1.06),
                      title=f"{day:%d.%m.%Y}: база SHAP {base:.2f} + сумма вкладов {sv[i].sum():+.2f} = {p_te[i]:.2f}",
                      xaxis_title=("Доля вклада" if norm == "sum" else "Вклад относительно главного признака") + " (> 0 — к дождю, < 0 — к сухой погоде)")
    c1, c2 = st.columns([1.5, 1])
    c1.plotly_chart(fig, width="stretch")
    tbl = pd.DataFrame({"Признак": [name(FEATURES[t]) for t in top[::-1]],
                        "Значение": [X_te.iloc[i][FEATURES[t]] for t in top[::-1]],
                        "Правило LIME": [rules.get(FEATURES[t], "") for t in top[::-1]],
                        "SHAP": Sn[top[::-1]].round(3), "LIME": Ln[top[::-1]].round(3),
                        "Разница": diff[top[::-1]].round(3),
                        "Интерпретация (DEC_XAI)": lvl[top[::-1]],
                        "Флаг признака": np.where(diff[top[::-1]] > diff_thr, "БОЛЬШОЕ РАСХОЖДЕНИЕ", "OK")})
    c2.dataframe(tbl, hide_index=True, width="stretch", height=560)
    st.caption("SHAP точно раскладывает прогноз дерева; LIME строит локальную линейную модель по случайным соседям "
               "и дискретизирует признаки — отсюда расхождения, особенно для «осадков сегодня» (много нулей). "
               "Интерпретация DEC_XAI всегда считается по шкале «от максимума»: < 0.2 сильное согласие, < 0.5 умеренная разница, "
               "< 0.8 большая, ≥ 0.8 очень большая.")

with tab_f:
    if detail is None:
        st.error("Нет файла data/processed/xai_shap_lime_detail.csv — запустите ноутбук 08.")
    else:
        S = detail.pivot(index="Дата", columns="Признак", values="SHAP (норм.)")
        L = detail.pivot(index="Дата", columns="Признак", values="LIME (норм.)")
        S_max, L_max = S.div(S.abs().max(axis=1), axis=0), L.div(L.abs().max(axis=1), axis=0)
        if norm == "max":
            S, L = S_max, L_max
        corr_all = np.array([safe_corr(S.iloc[r], L.iloc[r]) for r in range(len(S))])
        diff_all = (S - L).abs()
        diff_max_scale = (S_max - L_max).abs()
        fc, fd = corr_all < corr_thr, diff_all.max(axis=1).to_numpy() > diff_thr
        flagged = fc | fd
        summary = pd.DataFrame({"Дата": S.index.date, "Корреляция SHAP–LIME": corr_all.round(3),
                                "Макс. разница": diff_all.max(axis=1).round(3).to_numpy(),
                                "Статус": np.where(flagged, "FLAGGED", "OK"),
                                "Флаг дня": np.where(fc, "НИЗКОЕ СОГЛАСИЕ", "OK"),
                                "Причина": np.select([fc & fd, fc, fd], ["корреляция и разница", "низкая корреляция", "большая разница"], ""),
                                "Признаки с большой разницей": [", ".join(diff_all.columns[diff_all.iloc[r].to_numpy() > diff_thr]) or "нет"
                                                                for r in range(len(diff_all))]})
        summary = summary.merge(pd.DataFrame({"Дата": test.date.dt.date, "P(дождь)": p_te.round(3), "Факт": y_te}), on="Дата")
        err = ((summary["P(дождь)"] >= thr).astype(int) != summary["Факт"]).to_numpy()
        k = st.columns(5)
        k[0].metric("FLAGGED", f"{flagged.sum()} из {len(flagged)}", f"{flagged.mean():.0%}", delta_color="off")
        k[1].metric("Только по корреляции", int((fc & ~fd).sum()))
        k[2].metric("Только по разнице", int((fd & ~fc).sum()))
        k[3].metric("Медианная корреляция", f"{np.nanmedian(corr_all):.2f}")
        k[4].metric("Ошибок среди FLAGGED", f"{err[flagged].mean():.0%}" if flagged.any() else "—",
                    f"у остальных {err[~flagged].mean():.0%}", delta_color="off")

        gx, gy, P = pca_probability_map()
        Z = to_pca(X_te)
        fig = go.Figure(go.Contour(x=gx, y=gy, z=P, colorscale="Blues", opacity=0.8, contours=dict(coloring="heatmap"),
                                   colorbar=dict(title="P(дождь)"), line=dict(width=0), hoverinfo="skip"))
        fig.add_contour(x=gx, y=gy, z=P, contours=dict(start=thr, end=thr, coloring="none"), line=dict(color="black", width=2),
                        showscale=False, showlegend=False, hoverinfo="skip")
        for st_, col in [("OK", "#4CAF50"), ("FLAGGED", "#D32F2F")]:
            m = summary["Статус"].to_numpy() == st_
            fig.add_scatter(x=Z[m, 0], y=Z[m, 1], mode="markers", marker=dict(color=col, size=6 if st_ == "OK" else 9),
                            name=f"{st_} ({m.sum()})", text=summary["Дата"][m].astype(str), customdata=np.where(m)[0],
                            hovertemplate="%{text}<extra></extra>")
        fig.update_layout(height=560, xaxis_title="PC1", yaxis_title="PC2", xaxis_range=[-8, 8], yaxis_range=[-6, 8], dragmode="select",
                          title="Тестовые дни на карте вероятностей (PCA) — выделите дни лассо или рамкой",
                          legend=dict(orientation="h", y=1.08, x=0))
        c1, c2 = st.columns([1.4, 1])
        event = c1.plotly_chart(fig, width="stretch", on_select="rerun", selection_mode=("box", "lasso"), key="flag_map")
        dist = np.abs(summary["P(дождь)"] - thr)
        fig = px.scatter(x=dist, y=summary["Корреляция SHAP–LIME"], color=summary["Статус"],
                         color_discrete_map={"OK": "#4CAF50", "FLAGGED": "#D32F2F"},
                         labels={"x": f"Удалённость от порога |P − {thr:.2f}|", "y": "Корреляция SHAP–LIME", "color": ""},
                         title="Расхождения — в основном у порога")
        fig.update_layout(height=560)
        c2.plotly_chart(fig, width="stretch")

        lv = pd.Series(level4(diff_max_scale.to_numpy().ravel())).value_counts().reindex(LEVELS4).fillna(0).astype(int)
        c1, c2 = st.columns(2)
        fig = go.Figure(go.Bar(y=LEVELS4, x=lv.values, orientation="h", marker_color=["#4CAF50", "#F5D33B", "#F08A24", "#D32F2F"],
                               text=[f"{v} ({v / lv.sum():.1%})" for v in lv.values], textposition="auto"))
        fig.update_layout(height=330, yaxis=dict(autorange="reversed"), xaxis_title="Пар «день × признак»",
                          title="4 уровня интерпретации разницы (DEC_XAI, шкала «от максимума»)")
        c1.plotly_chart(fig, width="stretch")
        per_f = (diff_all > diff_thr).sum().sort_values().tail(8)
        fig = px.bar(per_f, orientation="h", labels={"value": "Дней с флагом признака", "index": ""},
                     title=f"Какие признаки чаще расходятся (разница > {diff_thr})")
        fig.update_layout(showlegend=False, height=330)
        c2.plotly_chart(fig, width="stretch")

        # batch analysis of the selected days (as in the corrected course app)
        sel = sorted({int(p["customdata"]) for p in event.selection.points if p.get("customdata") is not None}) if event and event.selection else []
        st.subheader("Пакетный анализ выбранных дней")
        if not sel:
            st.info("Выделите дни на карте (лассо или рамка) — появятся таблица по признакам, сводка и графики SHAP против LIME, "
                    "как в исправленной версии приложения преподавателя. Ниже — все FLAGGED-дни.")
            st.dataframe(summary[summary["Статус"] == "FLAGGED"], hide_index=True, width="stretch", height=300)
        else:
            days = summary["Дата"].iloc[sel].tolist()
            Sb, Lb = S.loc[pd.to_datetime(days)], L.loc[pd.to_datetime(days)]
            batch = pd.DataFrame({"Дата": np.repeat([d for d in days], Sb.shape[1]), "Признак": np.tile(Sb.columns, len(days)),
                                  "SHAP (норм.)": Sb.to_numpy().ravel().round(4), "LIME (норм.)": Lb.to_numpy().ravel().round(4),
                                  "Разница": (Sb - Lb).abs().to_numpy().ravel().round(4),
                                  "Интерпретация (DEC_XAI)": level4((S_max.loc[pd.to_datetime(days)] - L_max.loc[pd.to_datetime(days)]).abs().to_numpy().ravel())})
            batch["Флаг признака"] = np.where(batch["Разница"] > diff_thr, "БОЛЬШОЕ РАСХОЖДЕНИЕ", "OK")
            vals = detail.pivot(index="Дата", columns="Признак", values="Значение").loc[pd.to_datetime(days)]
            batch.insert(2, "Значение", vals.to_numpy().ravel())
            ssum = summary.iloc[sel]
            k = st.columns(3)
            k[0].metric("Выбрано дней", len(sel))
            k[1].metric("Из них FLAGGED", int((ssum["Статус"] == "FLAGGED").sum()))
            k[2].metric("Пар с «большим расхождением»", int((batch["Флаг признака"] != "OK").sum()))
            first = st.selectbox("День для графиков SHAP и LIME", days, format_func=lambda d: f"{d:%d.%m.%Y}")
            r = days.index(first)
            c1, c2 = st.columns(2)
            for col, M, title, colr in [(c1, Sb, "SHAP", "#C0392B"), (c2, Lb, "LIME", "#6C8EBF")]:
                v = M.iloc[r]
                v = v[v.abs().sort_values().index[-12:]]
                fig = go.Figure(go.Bar(y=v.index, x=v.values, orientation="h",
                                       marker_color=np.where(v > 0, "#D62728", "#1F77B4")))
                fig.add_vline(x=0, line_color="black")
                fig.update_layout(height=430, title=f"{title}: {first:%d.%m.%Y}", xaxis_title="Нормированный вклад (> 0 — к дождю)")
                col.plotly_chart(fig, width="stretch")
            st.markdown("**Сводка по выбранным дням**")
            st.dataframe(ssum, hide_index=True, width="stretch")
            st.markdown("**Подробно: день × признак**")
            st.dataframe(batch, hide_index=True, width="stretch", height=320)
            c1, c2 = st.columns(2)
            c1.download_button("⬇️ Пакетный анализ (CSV)", batch.to_csv(index=False).encode("utf-8-sig"), "xai_batch.csv", "text/csv")
            c2.download_button("⬇️ Сводка выбранных дней (CSV)", ssum.to_csv(index=False).encode("utf-8-sig"), "xai_flag_summary.csv", "text/csv")

        c1, c2 = st.columns(2)
        c1.download_button("⬇️ Сводка по всем дням (CSV)", summary.to_csv(index=False).encode("utf-8-sig"), "xai_summary.csv", "text/csv")
        c2.download_button("⬇️ Подробно по признакам, все дни (CSV)", detail.to_csv(index=False).encode("utf-8-sig"), "xai_detail.csv", "text/csv")
        st.caption("CSV открывается в Excel для фильтрации и сводных таблиц; пересчитать SHAP/LIME там нельзя — для этого нужен код. "
                   "Экспорт всех дней содержит оба варианта нормировки и столбцы DEC_XAI (интерпретация, флаг признака, флаг дня).")
