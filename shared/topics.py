"""Kafka topic names, in one place.

Three topics carry the whole system:

* ``telemetry``      raw asynchronous sensor measurements (producers -> everyone)
* ``fault-control``  commands telling the simulator what fault to create
                     (injector -> producers). Also used as ground truth for
                     evaluation, but the ML consumer never reads it at inference.
* ``predictions``    ML classifier outputs (streaming -> dashboard)
"""
from __future__ import annotations

TELEMETRY = "telemetry"
FAULT_CONTROL = "fault-control"
PREDICTIONS = "predictions"

ALL_TOPICS = (TELEMETRY, FAULT_CONTROL, PREDICTIONS)
