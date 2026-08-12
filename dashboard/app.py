"""Live dashboard: telemetry, predictions, and ML quality metrics.

Run:  ``streamlit run dashboard/app.py``

Consumes all three topics:
* ``telemetry``     to plot the signals as faults develop
* ``predictions``   the model's current call + confidence
* ``fault-control`` as *ground truth* (what was actually injected) so we can
                    score the model live: confusion matrix, precision/recall/F1,
                    false-alarm rate, and time-to-detect.
"""
from __future__ import annotations

import threading
import time
from bisect import bisect_right
from collections import defaultdict, deque
from typing import Deque, Dict, List, Tuple

import pandas as pd
import streamlit as st
from kafka import KafkaConsumer

from injector.fault_definitions import CLASSES
from shared import config
from shared.schemas import CHANNELS, FaultControl, Prediction, Telemetry, decode
from shared.topics import FAULT_CONTROL, PREDICTIONS, TELEMETRY

MAXLEN = 2000


class Collector:
    """Background Kafka reader feeding rolling in-memory buffers."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.telemetry: Dict[str, Deque[Tuple[float, float]]] = {c: deque(maxlen=MAXLEN) for c in CHANNELS}
        self.predictions: Deque[dict] = deque(maxlen=MAXLEN)
        # Ground-truth timeline: sorted (ts, fault_name) transitions.
        self.truth_changes: List[Tuple[float, str]] = [(0.0, "NORMAL")]
        self._start()

    def _start(self) -> None:
        for topic, handler in (
            (TELEMETRY, self._on_telemetry),
            (PREDICTIONS, self._on_prediction),
            (FAULT_CONTROL, self._on_control),
        ):
            t = threading.Thread(target=self._consume, args=(topic, handler), daemon=True)
            t.start()

    def _consume(self, topic: str, handler) -> None:
        consumer = KafkaConsumer(
            topic,
            bootstrap_servers=config.KAFKA_BOOTSTRAP_SERVERS,
            value_deserializer=decode,
            auto_offset_reset="latest",
            enable_auto_commit=False,
            group_id=None,
            client_id=f"dashboard-{topic}",
        )
        for msg in consumer:
            try:
                handler(msg.value)
            except Exception:  # noqa: BLE001 - never kill the reader
                pass

    def _on_telemetry(self, value: dict) -> None:
        t = Telemetry.from_dict(value)
        with self.lock:
            self.telemetry[t.channel].append((t.ts, t.value))

    def _on_prediction(self, value: dict) -> None:
        p = Prediction.from_dict(value)
        with self.lock:
            self.predictions.append({"ts": p.ts, "predicted": p.predicted, "confidence": p.confidence})

    def _on_control(self, value: dict) -> None:
        fc = FaultControl.from_dict(value)
        name = "NORMAL" if fc.action == "clear" else fc.fault
        with self.lock:
            self.truth_changes.append((fc.ts, name))
            self.truth_changes.sort(key=lambda x: x[0])

    # --- read helpers ------------------------------------------------------
    def truth_at(self, ts: float) -> str:
        times = [c[0] for c in self.truth_changes]
        i = bisect_right(times, ts) - 1
        return self.truth_changes[max(0, i)][1]

    def snapshot(self):
        with self.lock:
            telem = {c: list(v) for c, v in self.telemetry.items()}
            preds = list(self.predictions)
            changes = list(self.truth_changes)
        return telem, preds, changes


@st.cache_resource
def get_collector() -> Collector:
    return Collector()


# --- metrics ---------------------------------------------------------------
def paired(preds: List[dict], collector: Collector) -> pd.DataFrame:
    rows = [{"ts": p["ts"], "truth": collector.truth_at(p["ts"]),
             "predicted": p["predicted"], "confidence": p["confidence"]} for p in preds]
    return pd.DataFrame(rows)


def time_to_detect(changes: List[Tuple[float, str]], preds: List[dict]) -> List[dict]:
    out = []
    pred_sorted = sorted(preds, key=lambda p: p["ts"])
    for ts, fault in changes:
        if fault == "NORMAL":
            continue
        hit = next((p for p in pred_sorted if p["ts"] >= ts and p["predicted"] == fault), None)
        out.append({"fault": fault, "injected_at": ts,
                    "latency_s": round(hit["ts"] - ts, 1) if hit else None})
    return out[-8:]


# --- UI --------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="Streaming Fault Detection", layout="wide")
    st.title("🛰️ Streaming Fault Detection & Isolation")
    st.caption("Kafka telemetry → sliding-window features → ML classifier. "
               "Faults change the simulated process; the model is never told when.")

    collector = get_collector()
    live = st.sidebar.toggle("Live refresh", value=True)
    refresh_s = st.sidebar.slider("Refresh interval (s)", 1, 10, 2)
    st.sidebar.markdown("**Inject faults from a terminal:**")
    st.sidebar.code("python -m injector.inject --fault pressure_leak\n"
                    "python -m injector.inject --clear", language="bash")

    telem, preds, changes = collector.snapshot()

    now = time.time()
    current_truth = collector.truth_at(now)
    latest = preds[-1] if preds else None

    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted state", latest["predicted"] if latest else "—",
              f"conf {latest['confidence']:.0%}" if latest else "")
    c2.metric("Ground truth (injected)", current_truth)
    ok = latest and latest["predicted"] == current_truth
    c3.metric("Agreement", "✅ match" if ok else "⚠️ differs" if latest else "—")

    # Telemetry plots
    st.subheader("Telemetry")
    tcols = st.columns(3)
    for col, channel in zip(tcols, CHANNELS):
        pts = telem[channel]
        if pts:
            df = pd.DataFrame(pts, columns=["ts", channel]).tail(300)
            df["t"] = df["ts"] - df["ts"].iloc[0]
            col.line_chart(df.set_index("t")[[channel]], height=200)
        else:
            col.info(f"no {channel} yet")

    if not preds:
        st.info("Waiting for predictions… start the ML consumer: `python -m streaming.ml_consumer`")
        _maybe_refresh(live, refresh_s)
        return

    df = paired(preds, collector)

    # Metrics
    st.subheader("Model quality (live, scored against injected ground truth)")
    labels = [c for c in CLASSES if c in set(df["truth"]) | set(df["predicted"])]
    try:
        from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
        prec, rec, f1, sup = precision_recall_fscore_support(
            df["truth"], df["predicted"], labels=labels, zero_division=0)
        mcol, ccol = st.columns([1, 1])
        with mcol:
            st.markdown("**Precision / Recall / F1 by class**")
            st.dataframe(pd.DataFrame(
                {"precision": prec.round(2), "recall": rec.round(2),
                 "f1": f1.round(2), "support": sup}, index=labels),
                use_container_width=True)
            normal_mask = df["truth"] == "NORMAL"
            far = float((df.loc[normal_mask, "predicted"] != "NORMAL").mean()) if normal_mask.any() else 0.0
            st.metric("False-alarm rate", f"{far:.1%}")
        with ccol:
            st.markdown("**Confusion matrix**")
            cm = confusion_matrix(df["truth"], df["predicted"], labels=labels)
            st.dataframe(pd.DataFrame(cm, index=[f"t:{l}" for l in labels],
                                      columns=[f"p:{l}" for l in labels]),
                         use_container_width=True)
    except Exception as exc:  # noqa: BLE001
        st.warning(f"metrics unavailable: {exc}")

    # Time-to-detect
    st.subheader("Time-to-detect (per injection)")
    ttd = time_to_detect(changes, preds)
    if ttd:
        st.dataframe(pd.DataFrame(ttd), use_container_width=True)
    else:
        st.caption("No faults injected yet.")

    # Recent predictions
    with st.expander("Recent predictions"):
        st.dataframe(df.tail(25).iloc[::-1], use_container_width=True)

    _maybe_refresh(live, refresh_s)


def _maybe_refresh(live: bool, refresh_s: int) -> None:
    if live:
        time.sleep(refresh_s)
        st.rerun()


if __name__ == "__main__":
    main()
