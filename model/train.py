"""Train the fault classifier and save it to ``model/classifier.pkl``.

A RandomForest is a good fit here: the features are heterogeneous per-channel
statistics, it needs no scaling, it gives calibrated-enough probabilities for a
confidence read-out, and it trains in seconds on this dataset.

Run: ``python -m model.train``
"""
from __future__ import annotations

import os

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from injector.fault_definitions import CLASSES
from model.generate_training_data import generate_dataset
from streaming.feature_builder import FEATURE_NAMES

MODEL_OUT = os.path.join(os.path.dirname(__file__), "classifier.pkl")


def train() -> None:
    print("Generating training data (simulating faults)...")
    df = generate_dataset()
    print(f"  {len(df)} rows")
    print(df["label"].value_counts().to_string())

    X = df[FEATURE_NAMES].values
    y = df["label"].to_numpy(dtype=object)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    print("Training RandomForest...")
    clf.fit(X_train, y_train)

    print("\nHeld-out performance:")
    y_pred = clf.predict(X_test)
    labels = [c for c in CLASSES if c in set(y)]
    print(classification_report(y_test, y_pred, labels=labels, zero_division=0))

    bundle = {
        "model": clf,
        "classes": list(clf.classes_),
        "feature_names": FEATURE_NAMES,
    }
    joblib.dump(bundle, MODEL_OUT)
    print(f"Saved model -> {MODEL_OUT}")


if __name__ == "__main__":
    train()
