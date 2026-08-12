"""The simulated physical process and the generic producer loop.

``SystemSimulator`` knows the healthy baseline behaviour of each channel and how
to distort a reading given the currently active fault(s). The same simulator is
used live (here) and offline (``model/generate_training_data.py``), so training
and inference see the same process.
"""
from __future__ import annotations

import math
import random
import time
from typing import List, Optional

import numpy as np

from injector.fault_definitions import EffectContext, apply_fault
from producers.fault_state import ActiveFault, FaultState, start_listener
from shared import config
from shared.schemas import SENSOR_CHANNEL, Telemetry, encode
from shared.topics import TELEMETRY

# Healthy baseline: (mean, seasonal amplitude, seasonal period sec, noise std).
_BASELINE = {
    "temperature": (70.0, 1.5, 60.0, 0.4),
    "pressure": (100.0, 1.0, 90.0, 0.6),
    "vibration": (2.0, 0.3, 40.0, 0.25),
}


class SystemSimulator:
    """Generates readings for one channel as a function of wall-clock time."""

    def __init__(self, rng: Optional[np.random.Generator] = None) -> None:
        self.rng = rng if rng is not None else np.random.default_rng()

    def deterministic_baseline(self, channel: str, t: float) -> float:
        """Noise-free healthy value. Deterministic in ``t`` so a frozen ("stuck")
        value at fault onset is well defined."""
        mean, amp, period, _ = _BASELINE[channel]
        return mean + amp * math.sin(2.0 * math.pi * t / period)

    def sample(self, channel: str, t: float, faults: List[ActiveFault]) -> Optional[float]:
        """Return the observed reading for ``channel`` at time ``t``.

        Applies every active fault in turn. Returns ``None`` if a fault
        suppresses the reading (sensor dropout).
        """
        _, _, _, noise = _BASELINE[channel]
        value: Optional[float] = self.deterministic_baseline(channel, t) + self.rng.normal(0.0, noise)

        for fault in faults:
            base_at_onset = self.deterministic_baseline(channel, fault.onset_ts)
            ctx = EffectContext(
                channel=channel,
                base_value=value,
                base_at_onset=base_at_onset,
                elapsed=max(0.0, t - fault.onset_ts),
                severity=fault.severity,
                duration=fault.duration,
                rng=self.rng,
            )
            value = apply_fault(fault.name, ctx)
            if value is None:
                return None

        return value


def run_producer(sensor_id: str) -> None:
    """Run one sensor: sample its channel on a jittered period and publish.

    Each producer independently tails ``fault-control`` so it knows the current
    process state, then emits to ``telemetry``.
    """
    from kafka import KafkaProducer  # lazy: importing SystemSimulator needs no broker

    channel = SENSOR_CHANNEL[sensor_id]
    period = config.SENSOR_PERIODS[sensor_id]

    state = FaultState()
    start_listener(state, client_id=sensor_id)

    producer = KafkaProducer(
        bootstrap_servers=config.KAFKA_BOOTSTRAP_SERVERS,
        value_serializer=encode,
        linger_ms=10,
    )
    sim = SystemSimulator()

    print(f"[{sensor_id}] producing '{channel}' every ~{period}s -> topic '{TELEMETRY}'")
    seq = 0
    try:
        while True:
            now = time.time()
            value = sim.sample(channel, now, state.active_faults())

            if value is not None:  # None => sensor dropout, emit nothing
                msg = Telemetry(sensor_id=sensor_id, channel=channel, value=round(value, 4), ts=now, seq=seq)
                producer.send(TELEMETRY, msg.to_dict())
                seq += 1

            jitter = 1.0 + random.uniform(-config.SENSOR_JITTER, config.SENSOR_JITTER)
            time.sleep(max(0.05, period * jitter))
    except KeyboardInterrupt:
        print(f"[{sensor_id}] stopping")
    finally:
        producer.flush()
        producer.close()
