"""Window 2 of RTAP — Spark Structured Streaming aggregates the distributed stream.

Two streaming queries read the Kafka topic weather_events:
  regional   — event-time windows (--window-days long, sliding by --slide-days) over all stations: regional pressure,
               rain share, completeness (how many stations reported), the pressure of every station (for the barometer
               cross-check in window 3). Late records are accepted up to --watermark-days; later ones are dropped.
               Output: topic rtap_agg (JSON) + CSV files in <out>/regional/.
  throughput — processing-time windows of --tp-window-sec seconds (as in the lecture's 10-second windows): messages
               per station and the producer → broker delay. Output: CSV files in <out>/throughput/ (update mode:
               a window may appear several times — the last row with the largest batch_id is final).
Progress metrics of every micro-batch (rows/s, rows dropped by the watermark, batch duration) go to <out>/progress.jsonl.
Example:  .venv/bin/python rtap/spark_aggregator.py --run-seconds 120
"""
from __future__ import annotations

import argparse
import json
import shutil
import time

from rtap_common import BOOTSTRAP, OUT, TOPIC_AGG, TOPIC_EVENTS, spark_session


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--window-days", type=int, default=1)
    ap.add_argument("--slide-days", type=int, default=1)
    ap.add_argument("--watermark-days", type=int, default=2)
    ap.add_argument("--tp-window-sec", type=int, default=10)
    ap.add_argument("--trigger-sec", type=float, default=2.0)
    ap.add_argument("--run-seconds", type=float, default=0, help="stop after N seconds (0 = run until idle)")
    ap.add_argument("--idle-seconds", type=float, default=20, help="stop when no new input for N seconds")
    ap.add_argument("--out", default="run")
    ap.add_argument("--starting-offsets", default="earliest")
    ap.add_argument("--no-kafka-sink", action="store_true")
    args = ap.parse_args()

    out = OUT / args.out
    shutil.rmtree(out / "regional", ignore_errors=True)
    shutil.rmtree(out / "throughput", ignore_errors=True)
    shutil.rmtree(out / "checkpoints", ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)
    (out / "progress.jsonl").unlink(missing_ok=True)
    (out / "READY").unlink(missing_ok=True)

    spark = spark_session("RTAP-Spark-Aggregator")
    from pyspark.sql import functions as F
    from pyspark.sql.types import DoubleType, LongType, StringType, StructField, StructType

    from rtap_common import MEASUREMENTS
    schema = StructType([StructField("station", StringType()), StructField("date", StringType()),
                         StructField("seq", LongType()), StructField("sent_at_ms", LongType())]
                        + [StructField(c, DoubleType()) for c in MEASUREMENTS])

    raw = (spark.readStream.format("kafka").option("kafka.bootstrap.servers", BOOTSTRAP)
           .option("subscribe", TOPIC_EVENTS).option("startingOffsets", args.starting_offsets).load())
    events = (raw.select(F.from_json(F.col("value").cast("string"), schema).alias("d"), F.col("timestamp").alias("kafka_ts"))
              .select("d.*", "kafka_ts")
              .withColumn("event_time", F.to_timestamp("date"))
              .withColumn("lag_ms", F.col("kafka_ts").cast("double") * 1000 - F.col("sent_at_ms")))

    # ---------------- regional picture: event-time windows with a watermark
    # NB: the heartbeat must NOT be filtered here — Spark pushes filters below the watermark operator, and the heartbeat
    # would then not advance the watermark; it lands in its own far-away window, which is never emitted.
    regional = (events.withWatermark("event_time", f"{args.watermark_days} days")
                .groupBy(F.window("event_time", f"{args.window_days} days", f"{args.slide_days} days"))
                .agg(F.count("*").alias("n_messages"),
                     F.approx_count_distinct("station").alias("n_stations"),
                     F.avg("pressure_msl_mean").alias("pressure_mean"),
                     F.percentile_approx("pressure_msl_mean", 0.5).alias("pressure_median"),
                     F.stddev("pressure_msl_mean").alias("pressure_std"),
                     F.avg("temperature_2m_mean").alias("temperature_mean"),
                     F.avg("relative_humidity_2m_mean").alias("humidity_mean"),
                     F.avg("cloud_cover_mean").alias("cloud_mean"),
                     F.avg((F.col("precipitation_sum") >= 1).cast("double")).alias("rain_share"),
                     F.max("precipitation_sum").alias("precip_max"),
                     F.max("wind_gusts_10m_max").alias("gusts_max"),
                     F.avg("lag_ms").alias("producer_lag_ms"),
                     F.to_json(F.map_from_entries(F.collect_list(F.struct(
                         F.concat_ws("|", "station", "date"), "pressure_msl_mean")))).alias("station_pressure"))
                .select(F.col("window.start").alias("window_start"), F.col("window.end").alias("window_end"), "*")
                .drop("window"))

    def write_regional(batch, batch_id):
        if batch.isEmpty():
            return
        batch = batch.withColumn("emitted_at", F.current_timestamp())
        batch.write.mode("append").option("header", True).option("escape", "\"").csv(str(out / "regional"))
        if not args.no_kafka_sink:
            (batch.select(F.to_json(F.struct("*")).alias("value"))
             .write.format("kafka").option("kafka.bootstrap.servers", BOOTSTRAP).option("topic", TOPIC_AGG).save())

    q1 = (regional.writeStream.outputMode("append").foreachBatch(write_regional)
          .option("checkpointLocation", str(out / "checkpoints" / "regional"))
          .trigger(processingTime=f"{args.trigger_sec} seconds").queryName("regional").start())

    # ---------------- throughput: processing-time windows (lecture: 10 seconds)
    throughput = (events.filter(F.col("station") != "_heartbeat")
                  .withWatermark("kafka_ts", f"{args.tp_window_sec} seconds")
                  .groupBy(F.window("kafka_ts", f"{args.tp_window_sec} seconds"), "station")
                  .agg(F.count("*").alias("n_messages"), F.avg("lag_ms").alias("producer_lag_ms"),
                       F.min("date").alias("first_date"), F.max("date").alias("last_date"))
                  .select(F.col("window.start").alias("window_start"), F.col("window.end").alias("window_end"),
                          "station", "n_messages", "producer_lag_ms", "first_date", "last_date"))
    # update mode: a processing-time window is reported as soon as it changes (in append mode it would wait for newer
    # messages, which never come after the last burst); the analysis keeps the last row of every window
    def write_throughput(batch, batch_id):
        if not batch.isEmpty():
            (batch.withColumn("batch_id", F.lit(batch_id)).write.mode("append").option("header", True)
             .csv(str(out / "throughput")))

    q2 = (throughput.writeStream.outputMode("update").foreachBatch(write_throughput)
          .option("checkpointLocation", str(out / "checkpoints" / "throughput"))
          .trigger(processingTime=f"{args.trigger_sec} seconds").queryName("throughput").start())

    print(f"Spark: окна {args.window_days} сут. (шаг {args.slide_days}), watermark {args.watermark_days} сут.; "
          f"пропускная способность — окна {args.tp_window_sec} с. Вывод: {out}", flush=True)
    (out / "READY").write_text("ok")  # the orchestrator starts the producer after this file appears
    t0, last_input, got_input = time.time(), time.time(), False
    seen = set()
    with open(out / "progress.jsonl", "a") as log:
        while True:
            time.sleep(1)
            for q in (q1, q2):
                for p in q.recentProgress:
                    key = (p["name"], p["batchId"])
                    if key in seen:
                        continue
                    seen.add(key)
                    if p.get("numInputRows", 0) > 0:
                        last_input, got_input = time.time(), True
                    state = p.get("stateOperators") or [{}]
                    log.write(json.dumps({"query": p["name"], "batch": p["batchId"], "timestamp": p["timestamp"],
                                          "input_rows": p.get("numInputRows", 0),
                                          "input_rows_per_s": p.get("inputRowsPerSecond", 0),
                                          "processed_rows_per_s": p.get("processedRowsPerSecond", 0),
                                          "batch_ms": p.get("durationMs", {}).get("triggerExecution", 0),
                                          "dropped_by_watermark": sum(s.get("numRowsDroppedByWatermark", 0) for s in state),
                                          "state_rows": sum(s.get("numRowsTotal", 0) for s in state),
                                          "watermark": p.get("eventTime", {}).get("watermark")}) + "\n")
                    log.flush()
            if args.run_seconds and time.time() - t0 > args.run_seconds:
                break
            if not args.run_seconds and got_input and time.time() - last_input > args.idle_seconds:
                break
            if not args.run_seconds and not got_input and time.time() - t0 > 600:
                break
    for q in (q1, q2):
        q.stop()
    spark.stop()
    print("Spark остановлен", flush=True)


if __name__ == "__main__":
    main()
