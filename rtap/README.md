# RTAP: распределённый поток на Kafka и Spark (лекция 5)

6 метеостанций Молдовы — распределённые источники. Конвейер повторяет 4 окна RTAP из лекции; каждое окно — отдельный процесс.

| Окно лекции | Скрипт лекции | Здесь | Роль |
|---|---|---|---|
| BAT1, BAT2 | `start_kafka_kraft.bat`, `create_topic_video_events.bat` | `kafka.sh start`, `kafka.sh topics` / `clean` | брокер Kafka 4.3 в режиме KRaft (без ZooKeeper), топики `weather_events` (6 разделов), `rtap_agg`, `rtap_alerts` |
| 1 | `kafka_trio_RTAP.py` | `producer_stations.py` | 6 станций → JSON в `weather_events`; ускоренное воспроизведение; неисправности: `--bias`, `--noise`, `--late`, `--drop`, `--drop-field` |
| 2 | `spark_trio_RTAP.py` | `spark_aggregator.py` | Spark 4.2 Structured Streaming: окна по времени события (сутки, watermark) и по времени обработки (10 с) → `rtap_agg` + CSV |
| 3 | `alert_trio_RTAP.py` | `alert_service.py` | признаки → Random Forest + Elastic Net + SHAP → уровень тревоги; региональный консенсус (≥ 4 из 6); контроль барометра и шума по соседям; восстановление пропусков → `rtap_alerts` + SQLite |
| 4 | `verdict_trio_RTAP.py` | `verdict.py`, ноутбук 10, страница «RTAP» приложения | настройка C, l1_ratio, окна с кросс-валидацией по времени; вердикт |

## Запуск (macOS)

```bash
rtap/kafka.sh start          # один раз форматирует хранилище (UUID кластера) и запускает брокер
rtap/kafka.sh topics         # один раз создаёт топики
.venv/bin/python rtap/run_pipeline.py --out run --start 2024-01-01 --end 2025-12-31 --speed 0.1 \
    --bias chisinau:6:2024-07-01 --late cahul:0.05:3
rtap/kafka.sh stop
```

`run_pipeline.py` запускает окна в порядке лекции: брокер → пересоздание топиков → Spark → сервис тревог → станции; результаты — в `data/rtap/<out>/` (`alerts.db`, `regional/`, `throughput/`, `progress.jsonl`, журналы).

Требования: `brew install kafka openjdk@21` (Spark 4.2 работает на Java 17/21; JDK 21 «keg-only» и системную Java не трогает), `pip install pyspark==4.2.0 kafka-python==3.0.11`. При первом запуске Spark скачивает коннектор `spark-sql-kafka-0-10_2.13:4.2.0` из Maven.

## Что важно знать (найдено при отладке)

- **Фильтр до watermark.** Spark опускает фильтр ниже оператора watermark, поэтому служебное «закрывающее» сообщение нельзя отфильтровывать до агрегации — иначе последние окна не закроются.
- **Окна по времени обработки** в режиме `append` закрываются, только когда приходят более новые сообщения; после последней пачки их нет, поэтому используется режим `update` через `foreachBatch`.
- **Порядок внутри микропачки.** Kafka отдаёт записи по разделам, Spark — окна в произвольном порядке. Сервис тревог сортирует записи по времени события, а сдвиг барометра ищет как первую серию из 5 дней подряд.
- **Сдвиг или шум.** Сдвиг барометра — это стабильное смещение (разброс < 1.5 гПа); шум тоже даёт большие отклонения, но случайные. Без этого условия шумный барометр «корректировался» бы ошибочно.
