"""The streaming ML classifier.

Consumes ``telemetry`` only (never ``fault-control`` — that would give away the
answer), maintains sliding-window state, builds a feature vector on a fixed
cadence, classifies the current system condition, and publishes the result to
``predictions``.

Run: ``python -m streaming.ml_consumer``
"""
from __future__ import annotations

import os
import time

import joblib
from kafka import KafkaConsumer, KafkaProducer

from shared import config
from shared.schemas import Prediction, Telemetry, decode, encode
from shared.topics import PREDICTIONS, TELEMETRY
from streaming.feature_builder import build_features, to_vector
from streaming.state_store import StateStore


def _load_model():
    path = os.path.abspath(config.MODEL_PATH)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model not found at {path}. Train it first: python -m model.train"
        )
    bundle = joblib.load(path)
    print(f"Loaded model from {path} (classes: {list(bundle['classes'])})")
    return bundle


def run() -> None:
    bundle = _load_model()
    model = bundle["model"]
    classes = list(bundle["classes"])

    store = StateStore(window=config.FEATURE_WINDOW)

    consumer = KafkaConsumer(
        TELEMETRY,
        bootstrap_servers=config.KAFKA_BOOTSTRAP_SERVERS,
        value_deserializer=decode,
        auto_offset_reset="latest",
        enable_auto_commit=True,
        group_id=config.GROUP_ML_CONSUMER,
    )
    producer = KafkaProducer(bootstrap_servers=config.KAFKA_BOOTSTRAP_SERVERS, value_serializer=encode)

    print(f"ML consumer running: '{TELEMETRY}' -> '{PREDICTIONS}', inferring every {config.INFERENCE_PERIOD}s")
    last_infer = 0.0
    try:
        while True:
            batch = consumer.poll(timeout_ms=200)
            for _, records in batch.items():
                for record in records:
                    try:
                        store.update(Telemetry.from_dict(record.value))
                    except Exception as exc:  # noqa: BLE001
                        print(f"bad telemetry: {exc}")

            now = time.time()
            if now - last_infer >= config.INFERENCE_PERIOD:
                last_infer = now
                _infer_and_publish(model, classes, store, producer, now)
    except KeyboardInterrupt:
        print("ML consumer stopping")
    finally:
        producer.flush()
        producer.close()
        consumer.close()


def _infer_and_publish(model, classes, store: StateStore, producer: KafkaProducer, now: float) -> None:
    features = build_features(store.snapshot(now), now, config.FEATURE_WINDOW)
    vector = [to_vector(features)]

    probs = model.predict_proba(vector)[0]
    idx = int(probs.argmax())
    predicted = classes[idx]
    confidence = float(probs[idx])

    prediction = Prediction(
        predicted=predicted,
        confidence=round(confidence, 4),
        ts=now,
        probabilities={cls: round(float(p), 4) for cls, p in zip(classes, probs)},
        features={k: round(v, 4) for k, v in features.items()},
    )
    producer.send(PREDICTIONS, prediction.to_dict())
    print(f"[{time.strftime('%H:%M:%S')}] {predicted:<16} conf={confidence:.2f}")


if __name__ == "__main__":
    run()
