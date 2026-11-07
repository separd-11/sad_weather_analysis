#!/usr/bin/env bash
# Local Kafka (KRaft mode, no ZooKeeper) for the RTAP pipeline — the macOS counterpart of the lecture's .bat files.
#   rtap/kafka.sh start    — format the storage once (cluster UUID), then start the broker in the background
#   rtap/kafka.sh topics   — create the topics (once): weather_events (6 partitions = 6 stations), rtap_agg, rtap_alerts
#   rtap/kafka.sh clean    — delete and re-create the topics (clean replay)
#   rtap/kafka.sh status   — broker and topics
#   rtap/kafka.sh stop     — stop the broker
#   rtap/kafka.sh reset    — stop and delete all Kafka data (the UUID is generated again on the next start)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
KAFKA_BIN="${KAFKA_BIN:-/opt/homebrew/opt/kafka/bin}"
DATA="$ROOT/data/kafka"
CONF="$DATA/server.properties"
BOOT="localhost:9092"
mkdir -p "$DATA"

write_config() {
  sed "s#^log.dirs=.*#log.dirs=$DATA/kraft-logs#; s#^num.partitions=.*#num.partitions=6#" \
    "$ROOT/rtap/server.properties" > "$CONF"
}

is_up() { "$KAFKA_BIN/kafka-broker-api-versions" --bootstrap-server "$BOOT" >/dev/null 2>&1; }

case "${1:-status}" in
  start)
    write_config
    if [ ! -f "$DATA/kraft-logs/meta.properties" ]; then
      CLUSTER_ID="$("$KAFKA_BIN/kafka-storage" random-uuid)"
      echo "Formatting storage, cluster UUID = $CLUSTER_ID (done once)"
      echo "$CLUSTER_ID" > "$DATA/cluster_id"
      "$KAFKA_BIN/kafka-storage" format --standalone -t "$CLUSTER_ID" -c "$CONF" >/dev/null
    fi
    if is_up; then echo "Broker is already running on $BOOT"; exit 0; fi
    nohup "$KAFKA_BIN/kafka-server-start" "$CONF" > "$DATA/server.log" 2>&1 &
    echo $! > "$DATA/broker.pid"
    for _ in $(seq 1 60); do is_up && { echo "Broker started on $BOOT (pid $(cat "$DATA/broker.pid"))"; exit 0; }; sleep 1; done
    echo "Broker did not start — see $DATA/server.log"; exit 1 ;;
  topics)
    "$KAFKA_BIN/kafka-topics" --bootstrap-server "$BOOT" --create --if-not-exists --topic weather_events --partitions 6 --replication-factor 1
    "$KAFKA_BIN/kafka-topics" --bootstrap-server "$BOOT" --create --if-not-exists --topic rtap_agg --partitions 1 --replication-factor 1
    "$KAFKA_BIN/kafka-topics" --bootstrap-server "$BOOT" --create --if-not-exists --topic rtap_alerts --partitions 1 --replication-factor 1
    "$KAFKA_BIN/kafka-topics" --bootstrap-server "$BOOT" --list ;;
  clean)
    # delete and re-create the topics — a clean replay (the cluster UUID is kept)
    for t in weather_events rtap_agg rtap_alerts; do
      "$KAFKA_BIN/kafka-topics" --bootstrap-server "$BOOT" --delete --if-exists --topic "$t" >/dev/null 2>&1 || true
    done
    sleep 3
    "$0" topics >/dev/null && echo "Topics re-created" ;;
  status)
    if is_up; then echo "Broker: running on $BOOT"; "$KAFKA_BIN/kafka-topics" --bootstrap-server "$BOOT" --describe 2>/dev/null | grep -E "^Topic:" || true
    else echo "Broker: stopped"; fi ;;
  stop)
    "$KAFKA_BIN/kafka-server-stop" >/dev/null 2>&1 || true
    for _ in $(seq 1 30); do is_up || { echo "Broker stopped"; exit 0; }; sleep 1; done
    [ -f "$DATA/broker.pid" ] && kill "$(cat "$DATA/broker.pid")" 2>/dev/null || true ;;
  reset)
    "$0" stop || true
    rm -rf "$DATA/kraft-logs" "$DATA/cluster_id" "$DATA/broker.pid"
    echo "Kafka data deleted" ;;
  *) echo "usage: $0 {start|topics|clean|status|stop|reset}"; exit 2 ;;
esac
