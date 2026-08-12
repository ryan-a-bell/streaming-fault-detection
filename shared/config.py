"""Central configuration, sourced from environment variables with sane defaults.

Every process (producers, injector, streaming consumer, dashboard) imports from
here so that a single ``KAFKA_BOOTSTRAP_SERVERS`` override reconfigures the whole
system.
"""
from __future__ import annotations

import os


def _get(name: str, default: str) -> str:
    return os.environ.get(name, default)


# --- Kafka -----------------------------------------------------------------
KAFKA_BOOTSTRAP_SERVERS: str = _get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")

# Consumer group ids. Each logical consumer role gets its own group so they all
# receive every message independently.
GROUP_ML_CONSUMER: str = _get("GROUP_ML_CONSUMER", "ml-consumer")
GROUP_DASHBOARD: str = _get("GROUP_DASHBOARD", "dashboard")

# --- Simulation ------------------------------------------------------------
# Nominal sampling period (seconds) for each sensor. They are deliberately
# different so telemetry arrives asynchronously.
SENSOR_PERIODS = {
    "sensor_a": float(_get("SENSOR_A_PERIOD", "1.0")),   # temperature
    "sensor_b": float(_get("SENSOR_B_PERIOD", "1.5")),   # pressure
    "sensor_c": float(_get("SENSOR_C_PERIOD", "0.5")),   # vibration
}

# Jitter applied to each period so producers are not phase-locked.
SENSOR_JITTER: float = float(_get("SENSOR_JITTER", "0.15"))

# --- ML inference ----------------------------------------------------------
# How often the streaming consumer builds a feature vector and classifies.
INFERENCE_PERIOD: float = float(_get("INFERENCE_PERIOD", "1.0"))

# Sliding window (seconds) used to build features from telemetry history.
FEATURE_WINDOW: float = float(_get("FEATURE_WINDOW", "10.0"))

# Path to the trained classifier.
MODEL_PATH: str = _get("MODEL_PATH", os.path.join(os.path.dirname(__file__), "..", "model", "classifier.pkl"))
