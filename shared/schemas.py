"""Message schemas and (de)serialization helpers.

Messages are plain JSON dicts on the wire. These dataclasses document the shape
of each message and provide ``to_dict`` / ``from_dict`` helpers so producers and
consumers agree on field names.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


# --- Channels --------------------------------------------------------------
# Logical measurement channels and which sensor emits each.
CHANNELS = ("temperature", "pressure", "vibration")

SENSOR_CHANNEL = {
    "sensor_a": "temperature",
    "sensor_b": "pressure",
    "sensor_c": "vibration",
}
CHANNEL_SENSOR = {v: k for k, v in SENSOR_CHANNEL.items()}


def now_ts() -> float:
    """Wall-clock seconds. One clock for the whole demo."""
    return time.time()


# --- telemetry topic -------------------------------------------------------
@dataclass
class Telemetry:
    sensor_id: str          # sensor_a / sensor_b / sensor_c
    channel: str            # temperature / pressure / vibration
    value: float
    ts: float = field(default_factory=now_ts)
    seq: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Telemetry":
        return Telemetry(
            sensor_id=d["sensor_id"],
            channel=d["channel"],
            value=float(d["value"]),
            ts=float(d["ts"]),
            seq=int(d.get("seq", 0)),
        )


# --- fault-control topic ---------------------------------------------------
@dataclass
class FaultControl:
    """A command that changes the behaviour of the simulated process.

    ``action`` is ``"start"`` to begin a fault or ``"clear"`` to return the
    system to normal. ``fault`` is a key in ``injector.fault_definitions.FAULTS``
    (``"NORMAL"`` for a clear).
    """
    fault: str
    action: str = "start"           # start | clear
    severity: float = 0.7           # 0..1
    duration: float = 30.0          # seconds the fault takes to fully develop
    ts: float = field(default_factory=now_ts)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "FaultControl":
        return FaultControl(
            fault=d["fault"],
            action=d.get("action", "start"),
            severity=float(d.get("severity", 0.7)),
            duration=float(d.get("duration", 30.0)),
            ts=float(d["ts"]),
        )


# --- predictions topic -----------------------------------------------------
@dataclass
class Prediction:
    predicted: str
    confidence: float
    ts: float = field(default_factory=now_ts)
    probabilities: Dict[str, float] = field(default_factory=dict)
    features: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Prediction":
        return Prediction(
            predicted=d["predicted"],
            confidence=float(d["confidence"]),
            ts=float(d["ts"]),
            probabilities={k: float(v) for k, v in d.get("probabilities", {}).items()},
            features=d.get("features"),
        )


# --- wire helpers ----------------------------------------------------------
def encode(obj: Dict[str, Any]) -> bytes:
    return json.dumps(obj).encode("utf-8")


def decode(raw: bytes) -> Dict[str, Any]:
    return json.loads(raw.decode("utf-8"))
