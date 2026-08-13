"""Tracks which fault is currently active, driven by the ``fault-control`` topic.

Each producer runs its own :class:`FaultState`. A background thread tails
``fault-control`` and updates the state, so every producer independently learns
what the "true" process is doing without any shared memory. This is what makes
the telemetry genuinely asynchronous and decentralised.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import List, Optional

from shared import config
from shared.schemas import FaultControl, decode
from shared.topics import FAULT_CONTROL


@dataclass
class ActiveFault:
    name: str
    onset_ts: float
    severity: float
    duration: float


class FaultState:
    """Thread-safe holder of the single currently-active fault (or none).

    We model one fault at a time; ``COMBINED_FAULT`` is itself a single fault
    that touches multiple channels. This keeps ground-truth labelling
    unambiguous.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._current: Optional[ActiveFault] = None

    def apply_control(self, fc: FaultControl) -> None:
        with self._lock:
            if fc.action == "clear" or fc.fault == "NORMAL":
                self._current = None
            else:
                self._current = ActiveFault(
                    name=fc.fault,
                    onset_ts=fc.ts,
                    severity=fc.severity,
                    duration=fc.duration,
                )

    def active_faults(self) -> List[ActiveFault]:
        with self._lock:
            return [self._current] if self._current is not None else []

    def label(self) -> str:
        with self._lock:
            return self._current.name if self._current is not None else "NORMAL"


def start_listener(state: FaultState, client_id: str) -> threading.Thread:
    """Spawn a daemon thread that keeps ``state`` in sync with ``fault-control``.

    Uses ``group_id=None`` so this consumer is a pure broadcast reader: it gets
    every fault-control message and never competes for partitions with other
    producers.
    """

    def _run() -> None:
        from kafka import KafkaConsumer  # lazy: offline training needs no broker

        consumer = KafkaConsumer(
            FAULT_CONTROL,
            bootstrap_servers=config.KAFKA_BOOTSTRAP_SERVERS,
            value_deserializer=decode,
            auto_offset_reset="latest",
            enable_auto_commit=False,
            group_id=None,
            client_id=client_id,
        )
        for msg in consumer:
            try:
                state.apply_control(FaultControl.from_dict(msg.value))
            except Exception as exc:  # noqa: BLE001 - keep the listener alive
                print(f"[{client_id}] bad fault-control message: {exc}")

    thread = threading.Thread(target=_run, name=f"fault-listener-{client_id}", daemon=True)
    thread.start()
    return thread
