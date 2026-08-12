"""The catalogue of faults and the *physics* of how each one distorts telemetry.

This module is the single source of truth for fault behaviour. Both the live
simulator (``producers/simulator.py``) and the offline training-data generator
(``model/generate_training_data.py``) apply faults through :func:`apply_fault`
so that what the model trains on matches what it sees in production.

Design rule (important): a fault never tells the model "a fault happened". It
only changes the *underlying process*, which changes telemetry. The classifier
has to notice the change from the data alone.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional


# The classes the model predicts. Order is stable so we can rely on it.
CLASSES: List[str] = [
    "NORMAL",
    "TEMP_SENSOR_BIAS",
    "PRESSURE_LEAK",
    "VIBRATION_SPIKE",
    "SENSOR_STUCK",
    "SENSOR_DROPOUT",
    "COMBINED_FAULT",
]


@dataclass
class EffectContext:
    """Everything a fault effect needs to distort one reading.

    ``base_value`` is the healthy baseline reading for ``channel`` at the current
    instant. ``base_at_onset`` is the healthy baseline for ``channel`` at the
    moment the fault started (used by "stuck" to freeze a value). ``elapsed`` is
    seconds since the fault started. ``rng`` is a ``numpy`` Generator.
    """
    channel: str
    base_value: float
    base_at_onset: float
    elapsed: float
    severity: float
    duration: float
    rng: "object"  # numpy.random.Generator


# An effect returns the distorted reading, or ``None`` to suppress the reading
# entirely (sensor dropout).
Effect = Callable[[EffectContext], Optional[float]]


def _temp_bias(ctx: EffectContext) -> float:
    # Step offset: temperature suddenly reads high and stays there.
    return ctx.base_value + 20.0 * ctx.severity


def _pressure_leak(ctx: EffectContext) -> float:
    # Gradual monotonic decay toward a lower pressure as the leak develops.
    frac = min(1.0, ctx.elapsed / max(ctx.duration, 1e-6))
    return ctx.base_value - 35.0 * ctx.severity * frac


def _vibration_spike(ctx: EffectContext) -> float:
    # Elevated mean plus much noisier signal.
    extra_noise = ctx.rng.normal(0.0, 2.5 * ctx.severity)
    return ctx.base_value + 4.0 * ctx.severity + extra_noise


def _sensor_stuck(ctx: EffectContext) -> float:
    # Reading freezes at whatever it was when the fault started.
    return ctx.base_at_onset


def _sensor_dropout(ctx: EffectContext) -> Optional[float]:
    # Producer stops transmitting: no reading at all.
    return None


def _combined(ctx: EffectContext) -> float:
    # Pressure decays while vibration climbs. Applied per-channel.
    if ctx.channel == "pressure":
        return _pressure_leak(ctx)
    if ctx.channel == "vibration":
        return _vibration_spike(ctx)
    return ctx.base_value


@dataclass
class FaultSpec:
    name: str
    label: str
    description: str
    channels: List[str]          # channels whose readings this fault distorts
    effect: Effect
    default_severity: float = 0.7
    default_duration: float = 30.0


FAULTS: Dict[str, FaultSpec] = {
    "NORMAL": FaultSpec(
        name="NORMAL",
        label="Normal",
        description="Healthy operation, no fault active.",
        channels=[],
        effect=lambda ctx: ctx.base_value,
    ),
    "TEMP_SENSOR_BIAS": FaultSpec(
        name="TEMP_SENSOR_BIAS",
        label="Temp Bias",
        description="Temperature sensor suddenly reads ~+20 and stays there.",
        channels=["temperature"],
        effect=_temp_bias,
    ),
    "PRESSURE_LEAK": FaultSpec(
        name="PRESSURE_LEAK",
        label="Pressure Leak",
        description="Pressure decays gradually as the system leaks.",
        channels=["pressure"],
        effect=_pressure_leak,
    ),
    "VIBRATION_SPIKE": FaultSpec(
        name="VIBRATION_SPIKE",
        label="Vibration Spike",
        description="Vibration becomes high and noisy.",
        channels=["vibration"],
        effect=_vibration_spike,
    ),
    "SENSOR_STUCK": FaultSpec(
        name="SENSOR_STUCK",
        label="Sensor Stuck",
        description="Temperature sensor freezes at its last value.",
        channels=["temperature"],
        effect=_sensor_stuck,
    ),
    "SENSOR_DROPOUT": FaultSpec(
        name="SENSOR_DROPOUT",
        label="Sensor Dropout",
        description="Pressure producer stops transmitting.",
        channels=["pressure"],
        effect=_sensor_dropout,
    ),
    "COMBINED_FAULT": FaultSpec(
        name="COMBINED_FAULT",
        label="Combined Fault",
        description="Pressure decreases while vibration rises.",
        channels=["pressure", "vibration"],
        effect=_combined,
    ),
}


def apply_fault(fault_name: str, ctx: EffectContext) -> Optional[float]:
    """Apply the named fault's effect to one channel reading.

    If the fault does not touch ``ctx.channel`` the baseline reading is returned
    unchanged. Returns ``None`` when the reading should be suppressed.
    """
    spec = FAULTS.get(fault_name)
    if spec is None or fault_name == "NORMAL":
        return ctx.base_value
    if ctx.channel not in spec.channels:
        return ctx.base_value
    return spec.effect(ctx)
