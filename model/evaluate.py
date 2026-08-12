"""Offline evaluation of the trained classifier.

Generates a fresh (differently-seeded) batch of episodes, classifies every
feature row, and reports the metrics that matter for streaming fault detection:

* per-class precision / recall / F1
* confusion matrix
* false-alarm rate (fraction of true-NORMAL instants flagged as a fault)
* mean time-to-detect after fault onset

Run: ``python -m model.evaluate``
"""
from __future__ import annotations

import os
from collections import defaultdict
from typing import List

import joblib
import numpy as np
import pandas as pd

from injector.fault_definitions import CLASSES, FAULTS
from model.generate_training_data import (
    EPISODE_SECONDS,
    ONSET_MIN,
    WARMUP,
    _simulate_episode,
)
from producers.fault_state import ActiveFault  # noqa: F401  (kept for parity/imports)
from shared import config
from streaming.feature_builder import FEATURE_NAMES

MODEL_PATH = os.path.join(os.path.dirname(__file__), "classifier.pkl")


def _load():
    bundle = joblib.load(MODEL_PATH)
    return bundle["model"], list(bundle["classes"])


def _confusion(y_true: List[str], y_pred: List[str], labels: List[str]) -> pd.DataFrame:
    idx = {l: i for i, l in enumerate(labels)}
    m = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        if t in idx and p in idx:
            m[idx[t], idx[p]] += 1
    return pd.DataFrame(m, index=[f"true_{l}" for l in labels], columns=[f"pred_{l}" for l in labels])


def evaluate(n_per_fault: int = 12) -> None:
    model, classes = _load()

    y_true: List[str] = []
    y_pred: List[str] = []
    detect_latencies = defaultdict(list)   # fault -> list of seconds-to-detect
    false_alarms = 0
    normal_instants = 0

    faults = [f for f in CLASSES if f != "NORMAL"]
    for fi, fault_name in enumerate(faults):
        spec = FAULTS[fault_name]
        for e in range(n_per_fault):
            # Re-simulate one episode and re-derive its per-row timestamps so we
            # can measure detection latency against onset.
            seed = 500000 + 1000 * fi + e
            rows = _simulate_episode(fault_name, severity=0.8, duration=spec.default_duration, seed=seed)
            df = pd.DataFrame(rows)
            preds = model.predict(df[FEATURE_NAMES].values)

            detected = False
            # Reconstruct the inference timeline (WARMUP, +INFERENCE_PERIOD each row).
            for i, (label, pred) in enumerate(zip(df["label"].values, preds)):
                y_true.append(label)
                y_pred.append(pred)
                t = WARMUP + i * config.INFERENCE_PERIOD
                if label == "NORMAL":
                    normal_instants += 1
                    if pred != "NORMAL":
                        false_alarms += 1
                elif not detected and pred == fault_name:
                    # First correct detection after onset.
                    onset_est = _onset_from_rows(df["label"].values)
                    if onset_est is not None:
                        detect_latencies[fault_name].append(max(0.0, t - onset_est))
                        detected = True

    _report(y_true, y_pred, classes, detect_latencies, false_alarms, normal_instants)


def _onset_from_rows(labels: np.ndarray) -> float:
    """Virtual onset time = first row whose label is not NORMAL."""
    for i, l in enumerate(labels):
        if l != "NORMAL":
            return WARMUP + i * config.INFERENCE_PERIOD
    return None


def _report(y_true, y_pred, classes, detect_latencies, false_alarms, normal_instants) -> None:
    from sklearn.metrics import classification_report

    labels = [c for c in classes]
    print("=== Classification report ===")
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0))

    print("=== Confusion matrix ===")
    print(_confusion(y_true, y_pred, labels).to_string())

    far = (false_alarms / normal_instants) if normal_instants else 0.0
    print(f"\n=== False-alarm rate === {far:.3f} "
          f"({false_alarms}/{normal_instants} NORMAL instants misclassified)")

    print("\n=== Mean time-to-detect (s) ===")
    for fault in [c for c in CLASSES if c != "NORMAL"]:
        lats = detect_latencies.get(fault, [])
        if lats:
            print(f"  {fault:<18} {np.mean(lats):5.1f}s  (detected {len(lats)} episodes)")
        else:
            print(f"  {fault:<18}   n/a")


if __name__ == "__main__":
    evaluate()
