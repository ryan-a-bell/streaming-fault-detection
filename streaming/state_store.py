"""In-memory sliding-window state for the stream processor.

Telemetry arrives asynchronously and out of step across channels. The state
store keeps a bounded, time-windowed history per channel that the feature
builder turns into a single feature vector.
"""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Deque, Dict, List, Tuple

from shared.schemas import CHANNELS, Telemetry


class StateStore:
    """Keeps the last ``window`` seconds of readings for each channel."""

    def __init__(self, window: float) -> None:
        self.window = window
        self._hist: Dict[str, Deque[Tuple[float, float]]] = defaultdict(deque)

    def update(self, t: Telemetry) -> None:
        self._hist[t.channel].append((t.ts, t.value))
        self._trim(t.channel, t.ts)

    def _trim(self, channel: str, now: float) -> None:
        cutoff = now - self.window
        hist = self._hist[channel]
        while hist and hist[0][0] < cutoff:
            hist.popleft()

    def snapshot(self, now: float) -> Dict[str, List[Tuple[float, float]]]:
        """Return the current in-window history for every channel.

        Trims against ``now`` first so a channel that has gone silent (dropout)
        correctly shows an empty or shrinking window.
        """
        out: Dict[str, List[Tuple[float, float]]] = {}
        for channel in CHANNELS:
            self._trim(channel, now)
            out[channel] = list(self._hist[channel])
        return out
