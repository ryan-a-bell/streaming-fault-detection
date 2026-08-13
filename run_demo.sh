#!/usr/bin/env bash
#
# One-command demo.
#
# Brings up Kafka, trains the model if needed, starts the three sensor
# producers + the ML consumer + the Streamlit dashboard, then plays a scripted
# fault scenario while you watch the dashboard react. Press Ctrl+C to tear
# everything down.
#
#   ./run_demo.sh
#
# Env knobs:
#   KEEP_KAFKA=1   leave the Kafka container running on exit (faster re-runs)
#   HOLD_SCALE=2   stretch/compress every step in the scenario
#
set -euo pipefail
cd "$(dirname "$0")"

PY="$(command -v python3 || command -v python || true)"
[ -n "$PY" ] || { echo "python3 not found on PATH"; exit 1; }
command -v docker >/dev/null || { echo "docker is required for the Kafka broker"; exit 1; }

export KAFKA_BOOTSTRAP_SERVERS="${KAFKA_BOOTSTRAP_SERVERS:-localhost:9092}"
KEEP_KAFKA="${KEEP_KAFKA:-0}"
PIDS=()

log() { printf '\n\033[1;36m[demo]\033[0m %s\n' "$*"; }

cleanup() {
  log "shutting down..."
  for pid in "${PIDS[@]:-}"; do kill "$pid" 2>/dev/null || true; done
  # Backstop in case a child re-parented.
  pkill -f 'producers.sensor_' 2>/dev/null || true
  pkill -f 'streaming.ml_consumer' 2>/dev/null || true
  pkill -f 'scripts.scenario' 2>/dev/null || true
  if [ "$KEEP_KAFKA" != "1" ]; then
    log "stopping Kafka (set KEEP_KAFKA=1 to keep it running)"
    docker compose down >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

log "starting Kafka..."
docker compose up -d

log "waiting for Kafka to report healthy..."
until [ "$(docker inspect -f '{{.State.Health.Status}}' sfd-kafka 2>/dev/null)" = "healthy" ]; do
  sleep 2
done

if [ ! -f model/classifier.pkl ]; then
  log "no model found — training once (~10s)..."
  "$PY" -m model.train
else
  log "using existing model/classifier.pkl"
fi

log "starting producers + ML consumer..."
"$PY" -m producers.sensor_a & PIDS+=($!)
"$PY" -m producers.sensor_b & PIDS+=($!)
"$PY" -m producers.sensor_c & PIDS+=($!)
"$PY" -m streaming.ml_consumer & PIDS+=($!)

log "launching dashboard — a browser tab should open at http://localhost:8501"
"$PY" -m streamlit run dashboard/app.py & PIDS+=($!)

log "letting the stack warm up..."
sleep 12

log "playing the scripted fault scenario on a loop — watch the dashboard!"
"$PY" -m scripts.scenario --loop & PIDS+=($!)

log "demo is live. Press Ctrl+C to stop everything."
wait
