"""Turn windowed telemetry history into a fixed feature vector.

This is the contract between training and inference: ``model/generate_training_data``
and ``streaming/ml_consumer`` both call :func:`build_features`, so the columns
always line up. Keep ``FEATURE_NAMES`` and this function in lock-step.

Feature intuition per channel:
* ``mean/std/min/max/range`` describe level and spread (bias, leak, spike).
* ``slope`` captures gradual drift (a pressure leak trends down).
* ``last`` is the freshest reading.
* ``n`` is how many readings landed in the window.
* ``age`` is seconds since the last reading — the tell-tale for a dropout, and
  large-and-flat combined with ``std≈0`` for a stuck sensor.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from shared.schemas import CHANNELS

_PER_CHANNEL = ["mean", "std", "min", "max", "last", "slope", "range", "n", "age"]

FEATURE_NAMES: List[str] = [f"{ch}_{stat}" for ch in CHANNELS for stat in _PER_CHANNEL]


def _channel_features(points: List[Tuple[float, float]], now: float, window: float) -> Dict[str, float]:
    if not points:
        # No readings in the window: signal "very stale, nothing to see".
        return {
            "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "last": 0.0,
            "slope": 0.0, "range": 0.0, "n": 0.0, "age": float(window),
        }

    ts = np.array([p[0] for p in points], dtype=float)
    vals = np.array([p[1] for p in points], dtype=float)

    slope = 0.0
    if len(vals) >= 2 and np.ptp(ts) > 1e-6:
        # Least-squares slope of value vs. time (units per second).
        slope = float(np.polyfit(ts - ts[0], vals, 1)[0])

    age = float(min(window, now - ts[-1]))

    return {
        "mean": float(vals.mean()),
        "std": float(vals.std()),
        "min": float(vals.min()),
        "max": float(vals.max()),
        "last": float(vals[-1]),
        "slope": slope,
        "range": float(vals.max() - vals.min()),
        "n": float(len(vals)),
        "age": age,
    }


def build_features(history: Dict[str, List[Tuple[float, float]]], now: float, window: float) -> Dict[str, float]:
    """Build the flat, ordered feature dict for one classification instant."""
    features: Dict[str, float] = {}
    for channel in CHANNELS:
        ch_feats = _channel_features(history.get(channel, []), now, window)
        for stat, val in ch_feats.items():
            features[f"{channel}_{stat}"] = val
    return features


def to_vector(features: Dict[str, float]) -> List[float]:
    """Order a feature dict into the canonical model input vector."""
    return [features[name] for name in FEATURE_NAMES]
