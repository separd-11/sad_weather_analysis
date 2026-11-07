# Анализ и прогноз погоды — «Погода как знак и алгоритм»

Итоговая работа по курсу «Наука и анализ данных» (ТУМ, группа SD-251M), тема №23. Автор — Шепеленко Александр.

Система прогноза погоды на следующий день для Кишинёва и Молдовы:

- прогнозирует **осадки ≥ 1 мм завтра** и **тип погоды** (благоприятная / неблагоприятная / критическая);
- работает в **распределённом потоке** Apache Kafka (KRaft) → Spark Structured Streaming → сервис тревог;
- выдаёт объяснённые (SHAP) тревоги и сама контролирует качество датчиков.

На её примере исследованы шесть элементов курса: Bias, Noise, SMOTE, Decision Boundaries, Streaming, RTAP, а также объяснимость (12 методов XAI) и дрейф на новых данных 2026 года.

## Результаты (тест 2024–2025)

| Модель | ROC-AUC | PR-AUC | F1 | Recall |
|---|---|---|---|---|
| Random Forest (выбрана, порог 0.23) | 0.793 | 0.551 | 0.519 | 0.758 |
| Инерционность «завтра как сегодня» | — | — | 0.441 | 0.441 |
| Климатология | 0.528 | 0.243 | — | — |

Выполнено 6 из 7 KPI. Признаки соседних станций (агрегаты Spark) поднимают PR-AUC до 0.571.

## Структура

```
sql/        01_schema.sql (схема «звезда»), 02_transform.sql (ETL), 03_dataset.sql (главный запрос этапа 1), 04_daily.sql
src/        download_data.py (Extract), etl_postgres.py (Load + Transform), db.py (доступ к хранилищу), features.py (признаки),
            streaming.py (поток на Python), regularization.py (L1/L2/Elastic Net), xai_verdict.py (вердикт и постер),
            download_maps.py (карты ERA5 для лаб. 4)
notebooks/  13 ноутбуков с результатами (таблица ниже)
rtap/       распределённый поток: kafka.sh, producer_stations.py, spark_aggregator.py, alert_service.py, verdict.py,
            run_pipeline.py (см. rtap/README.md)
app/        приложение Streamlit для защиты (14 страниц)
models/     сохранённые модели (Random Forest, эталон обучения, Elastic Net)
data/       raw/ — исходные данные Open-Meteo (ERA5), processed/ — результаты ноутбуков, stream/ и rtap/ — результаты прогонов потока
figures/    рисунки ноутбуков (fig_<ноутбук>_<номер>) и постеры
```

| Ноутбук | Содержание |
|---|---|
| 00_DWH_Etap1 | этап 1: KPI, признаки, ETL в PostgreSQL, схема «звезда», контроль качества, OLAP, главный SQL-запрос |
| 01_EDA | разведочный анализ |
| 02_Modeling | признаки, базовые правила, 3 модели, порог, оценка, KPI |
| 03_Elements_DB_SMOTE_Noise_Bias | типы погоды, границы решений, SMOTE, шум датчиков, bias |
| 04_Streaming_RTAP | поток 2024–2025, коды тревог, контроль датчиков, переобучение |
| 05_Lab1_Bias_Noise_Fitting_SMOTE | лаб. №1: bias, noise, denoising, недо- и переобучение, SMOTE |
| 06_Lab2_Noise_Bias_X_y_Boundaries | лаб. №2: шум и bias на X, y, X + y (100 повторений), MLP + PCA |
| 07_Lecture2_Threshold_Cost | порог решения, цена ошибок, чувствительные точки |
| 08_Lab3_XAI_SHAP_LIME_Drift | лаб. №3: SHAP, LIME, новые данные 2026, дрейф |
| 09_Lecture4_Streaming_Regularization | лекция 4: L1 / L2 / Elastic Net в потоке |
| 10_Lecture5_Kafka_Spark_RTAP | лекция 5: Kafka + Spark, RTAP, неисправности, настройка |
| 11_XAI_12_Methods_Verdicts | 12 методов XAI, вердикты, постеры |
| 12_Lab4_Image_Noise_Augmentation | лаб. №4: шум в картах-изображениях, автоэнкодер, аугментация |

Ноутбуки сохранены вместе с результатами: их можно просмотреть без запуска.

## Запуск

Нужны Python 3.12, PostgreSQL 17; для RTAP — Kafka и Java 21 (`brew install kafka openjdk@21`).

```bash
brew services start postgresql@17 && createdb sad_weather
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt

.venv/bin/python src/download_data.py          # Extract: Open-Meteo (ERA5), 6 станций, 2015–2025
.venv/bin/python src/download_data.py --new    # новые данные Кишинёва за 2026 год
.venv/bin/python src/etl_postgres.py           # Load + Transform → схема «звезда» dwh

# ноутбуки запускаются по порядку: 00 строит хранилище, 02 сохраняет модель для 03–12
.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/00_DWH_Etap1.ipynb

.venv/bin/python rtap/run_pipeline.py --out run   # поток Kafka → Spark → тревоги
.venv/bin/streamlit run app/app.py                # приложение: http://localhost:8501
```

Подключение к базе задаётся переменной `SAD_DATABASE_URL` (по умолчанию `postgresql+psycopg:///sad_weather`). Если база недоступна, приложение берёт резервные CSV из `data/processed/`.
