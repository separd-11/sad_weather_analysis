"""Presentation app for topic #23 "Weather forecasting".  Run:  .venv/bin/streamlit run app/app.py"""
import streamlit as st

st.set_page_config(page_title="Погода как знак и алгоритм", page_icon="🌦️", layout="wide")

pages = {
    "Проект": [
        st.Page("views/home.py", title="О проекте", icon="🏠", default=True),
        st.Page("views/dwh.py", title="Хранилище данных (этап 1)", icon="🗄️"),
        st.Page("views/eda.py", title="Данные", icon="📊"),
    ],
    "Модель": [
        st.Page("views/model.py", title="Модель и порог решения", icon="🎯"),
        st.Page("views/forecast.py", title="Прогноз на день", icon="🌦️"),
        st.Page("views/boundaries.py", title="Границы решений", icon="🧭"),
        st.Page("views/explain.py", title="Объяснения (SHAP и LIME)", icon="🔍"),
        st.Page("views/poster.py", title="Вердикт и постер (к экзамену)", icon="🖼️"),
        st.Page("views/newdata.py", title="Новые данные и дрейф", icon="🆕"),
    ],
    "6 элементов": [
        st.Page("views/elements.py", title="Bias, Noise, SMOTE, Fitting", icon="⚖️"),
        st.Page("views/stream.py", title="Поток и тревоги (Streaming, RTAP)", icon="📡"),
        st.Page("views/regularization.py", title="Регуляризация в потоке (L1, L2, ENet)", icon="🎭"),
        st.Page("views/rtap.py", title="RTAP: Kafka + Spark", icon="🛰️"),
        st.Page("views/gallery.py", title="Рисунки отчёта", icon="🖼️"),
    ],
}
st.navigation(pages).run()
