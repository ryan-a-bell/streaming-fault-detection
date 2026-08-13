"""Scripted fault-injection scenario for the live demo.

Publishes a timed sequence of ``fault-control`` commands so the dashboard shows
the model detecting and recovering from each fault in turn. Each step prints a
banner in the terminal telling you what to watch for as it happens.

Usually launched by ``./run_demo.sh`` (which brings the whole stack up first),
but you can also run it on its own against an already-running system:

    python -m scripts.scenario           # one pass through every fault
    python -m scripts.scenario --loop    # repeat until Ctrl+C

Set ``HOLD_SCALE`` to speed up / slow down every hold (e.g. ``HOLD_SCALE=0.5``).
"""
from __future__ import annotations

import argparse
import os
import time
import warnings

# kafka-python warns that our plain-callable serializer isn't a Serializer
# subclass; harmless, and we keep the demo output clean.
warnings.filterwarnings("ignore", category=DeprecationWarning)

from kafka import KafkaProducer

from shared import config
from shared.schemas import FaultControl, encode
from shared.topics import FAULT_CONTROL

HOLD_SCALE = float(os.environ.get("HOLD_SCALE", "1.0"))

# Each step: (fault, severity, duration, hold_seconds, what-to-watch).
# Sentinels "__baseline__" and "__clear__" don't inject / clear respectively.
STEPS = [
    ("__baseline__", 0.0, 0.0, 8, "Healthy system — predictions should read NORMAL at high confidence."),
    ("PRESSURE_LEAK", 0.9, 15.0, 22, "Pressure ramps downward; model should latch PRESSURE_LEAK within a few seconds."),
    ("__clear__", 0.0, 0.0, 10, "Recovering — note the short window-flush lag before it reads NORMAL again."),
    ("TEMP_SENSOR_BIAS", 0.85, 30.0, 16, "Temperature jumps ~+20 as a step — this one is caught almost instantly."),
    ("__clear__", 0.0, 0.0, 8, "Back to NORMAL."),
    ("VIBRATION_SPIKE", 0.8, 30.0, 16, "Vibration becomes high and noisy."),
    ("__clear__", 0.0, 0.0, 8, "Back to NORMAL."),
    ("SENSOR_DROPOUT", 0.7, 30.0, 16, "Pressure sensor stops transmitting — its readings age out of the window."),
    ("__clear__", 0.0, 0.0, 8, "Back to NORMAL."),
    ("SENSOR_STUCK", 0.7, 30.0, 16, "Temperature freezes flat (std -> 0) while the others keep moving."),
    ("__clear__", 0.0, 0.0, 8, "Back to NORMAL."),
    ("COMBINED_FAULT", 0.9, 20.0, 24, "Pressure falls AND vibration rises at the same time."),
    ("__clear__", 0.0, 0.0, 10, "Back to NORMAL — end of scenario."),
]


def _send(producer: KafkaProducer, fault: str, action: str, severity: float, duration: float) -> None:
    fc = FaultControl(fault=fault, action=action, severity=severity, duration=duration)
    producer.send(FAULT_CONTROL, fc.to_dict())
    producer.flush()


def _banner(title: str, note: str) -> None:
    print(f"\n\033[1;33m=== {title} ===\033[0m\n    watch: {note}", flush=True)


def run_once(producer: KafkaProducer) -> None:
    for fault, sev, dur, hold, note in STEPS:
        if fault == "__baseline__":
            _banner("BASELINE (no fault)", note)
        elif fault == "__clear__":
            _send(producer, "NORMAL", "clear", 0.0, 0.0)
            _banner("CLEAR -> NORMAL", note)
        else:
            _send(producer, fault, "start", sev, dur)
            _banner(f"INJECT {fault}", note)
        time.sleep(max(1.0, hold * HOLD_SCALE))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Play a scripted fault scenario.")
    ap.add_argument("--loop", action="store_true", help="repeat the sequence until interrupted")
    args = ap.parse_args(argv)

    producer = KafkaProducer(bootstrap_servers=config.KAFKA_BOOTSTRAP_SERVERS, value_serializer=encode)
    print(f"Scenario connected to {config.KAFKA_BOOTSTRAP_SERVERS}. Driving the fault sequence...")
    try:
        while True:
            run_once(producer)
            if not args.loop:
                break
            print("\n\033[1;36m--- looping scenario ---\033[0m")
    except KeyboardInterrupt:
        print("\nscenario interrupted")
    finally:
        _send(producer, "NORMAL", "clear", 0.0, 0.0)  # never leave a fault stuck on
        producer.flush()
        producer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
