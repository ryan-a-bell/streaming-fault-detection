"""Generate a labelled feature dataset by simulating faults offline.

Crucially this reuses the *live* building blocks — ``SystemSimulator``,
``apply_fault`` and ``build_features`` — so the training distribution matches
what the streaming consumer sees. Each episode plays out a single fault
(possibly ``NORMAL``): telemetry is fed through the same sliding-window state
store and feature builder used at inference time, and each feature row is
labelled with the fault active at that instant (``NORMAL`` before onset).

Run standalone to dump a CSV:  ``python -m model.generate_training_data``
"""
from __future__ import annotations

import os
from typing import List

import numpy as np
import pandas as pd

from injector.fault_definitions import CLASSES, FAULTS
from producers.fault_state import ActiveFault
from producers.simulator import SystemSimulator
from shared import config
from shared.schemas import SENSOR_CHANNEL, Telemetry
from streaming.feature_builder import FEATURE_NAMES, build_features
from streaming.state_store import StateStore

EPISODE_SECONDS = 45.0
ONSET_MIN, ONSET_MAX = 8.0, 16.0
SIM_STEP = 0.25            # virtual-time granularity
WARMUP = 3.0              # start emitting feature rows once some history exists


def _simulate_episode(fault_name: str, severity: float, duration: float, seed: int) -> List[dict]:
    rng = np.random.default_rng(seed)
    sim = SystemSimulator(rng)
    store = StateStore(window=config.FEATURE_WINDOW)

    onset = float("inf") if fault_name == "NORMAL" else rng.uniform(ONSET_MIN, ONSET_MAX)

    # Independent, jittered sampling schedule per sensor -> asynchronous arrival.
    next_sample = {sid: rng.uniform(0.0, config.SENSOR_PERIODS[sid]) for sid in SENSOR_CHANNEL}

    rows: List[dict] = []
    t = 0.0
    next_infer = WARMUP
    while t <= EPISODE_SECONDS:
        active = [ActiveFault(fault_name, onset, severity, duration)] if t >= onset else []

        for sid, channel in SENSOR_CHANNEL.items():
            if t >= next_sample[sid]:
                value = sim.sample(channel, t, active)
                if value is not None:  # None => dropout, emit nothing
                    store.update(Telemetry(sensor_id=sid, channel=channel, value=value, ts=t))
                jitter = 1.0 + rng.uniform(-config.SENSOR_JITTER, config.SENSOR_JITTER)
                next_sample[sid] = t + config.SENSOR_PERIODS[sid] * jitter

        if t >= next_infer:
            feats = build_features(store.snapshot(t), t, config.FEATURE_WINDOW)
            feats["label"] = fault_name if t >= onset else "NORMAL"
            rows.append(feats)
            next_infer += config.INFERENCE_PERIOD

        t += SIM_STEP

    return rows


def generate_dataset(n_per_fault: int = 30, n_normal: int = 18, seed: int = 7) -> pd.DataFrame:
    """Build the full labelled dataset as a DataFrame (features + ``label``)."""
    rng = np.random.default_rng(seed)
    all_rows: List[dict] = []

    faults = [f for f in CLASSES if f != "NORMAL"]
    for fi, fault_name in enumerate(faults):
        spec = FAULTS[fault_name]
        for e in range(n_per_fault):
            severity = float(rng.uniform(0.5, 1.0))
            duration = float(rng.uniform(0.6, 1.4) * spec.default_duration)
            all_rows.extend(_simulate_episode(fault_name, severity, duration, seed=1000 * fi + e))

    for e in range(n_normal):
        all_rows.extend(_simulate_episode("NORMAL", 0.0, 0.0, seed=90000 + e))

    df = pd.DataFrame(all_rows)
    # Stable column order: features first, label last.
    return df[FEATURE_NAMES + ["label"]]


def main() -> None:
    df = generate_dataset()
    out = os.path.join(os.path.dirname(__file__), "training_data.csv")
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} rows to {out}")
    print(df["label"].value_counts())


if __name__ == "__main__":
    main()
