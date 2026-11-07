import re

import streamlit as st

from common import FIG, header

header("🖼️ Рисунки отчёта", "Все рисунки из ноутбуков 00–12, сгруппированные по разделам")
GROUPS = {"fig_1": "Этап 1: хранилище данных", "fig_2": "Разведочный анализ", "fig_3": "Модели и оценка",
          "fig_4": "Decision Boundaries, SMOTE, Noise, Bias", "fig_5": "Streaming и RTAP",
          "fig_6": "Лабораторная №1", "fig_7": "Лабораторная №2", "fig_8": "Лекция 2: порог и цена ошибок",
          "fig_9": "Лабораторная №3: SHAP, LIME, дрейф", "fig_10": "Лекция 4: регуляризация в потоке",
          "fig_11": "Лекция 5: Kafka, Spark, RTAP", "fig_12": "12 методов XAI и вердикты", "fig_13": "Лаб. 4: шум в изображениях"}
group = st.radio("Раздел", list(GROUPS), format_func=GROUPS.get, horizontal=True)


def order(p):
    num = re.match(r"(\d+)(\w?)", p.stem.split("_")[2])
    return (int(num.group(1)), num.group(2)) if num else (0, "")


files = sorted(FIG.glob(f"{group}_*.png"), key=order)
for p in files:
    st.markdown(f"**{p.stem}**")
    st.image(str(p), width="stretch")
