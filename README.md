# Streaming Fault Detection & Isolation

A miniature **condition-based monitoring** system: three synthetic sensors stream
asynchronous telemetry through Kafka, faults are injected manually to change the
*behaviour of the simulated process*, and a streaming ML classifier has to detect
and isolate which fault is happening — **without ever being told when it occurred.**

> The central question: *Can a model identify faults from asynchronous streaming
> telemetry without being told when the fault happened?* Kafka is just the event
> backbone; the interesting part is the fault detection & isolation on top of it.

```
 Synthetic Producers                       Manual Fault Injection
 ┌───────────────┐                         ┌──────────────────┐
 │ Sensor A temp │──┐                      │  inject.py CLI   │
 │ Sensor B pres │──┼──────┐               └────────┬─────────┘
 │ Sensor C vibr │──┘      │                        │
 └───────────────┘         ▼                        ▼
                      ┌──────────────── KAFKA ───────────────┐
                      │  telemetry      (raw measurements)    │
                      │  fault-control  (what fault to make)  │
                      │  predictions    (model output)        │
                      └───────┬───────────────────┬───────────┘
              ┌───────────────┘                   │ (ground truth only,
              ▼                                    │  never seen at inference)
   ┌─────────────────────┐                        │
   │ Stream Processor /  │                        │
   │ ML Classifier       │  maintain window state │
   │                     │  build features        │
   │  telemetry ─▶ 27 f  │  classify condition    │
   └──────────┬──────────┘                        │
              │ predictions                        │
              ▼                                    ▼
   NORMAL / TEMP_BIAS / PRESSURE_LEAK / ...   Dashboard (scores model live)
```

### The key design decision

The fault injector **does not tell the model a fault happened.** It publishes to
`fault-control`, which changes what the *simulated system* does. That flows into
`telemetry`, and the classifier has to notice the change from the data alone.
`fault-control` is retained separately as **ground truth** so the dashboard can
score the model — but the ML consumer only ever reads `telemetry` at inference.

## The system being monitored

| Stream | Sensor | Measurement | Normal | Period |
|--------|--------|-------------|--------|--------|
| A | `sensor_a` | Temperature | ~70 | 1.0 s |
| B | `sensor_b` | Pressure | ~100 | 1.5 s |
| C | `sensor_c` | Vibration | ~2 | 0.5 s |

Producers sample at different, jittered rates, so telemetry is genuinely
asynchronous and out of step.

## The faults

| Fault | Effect on the process |
|-------|-----------------------|
| `TEMP_SENSOR_BIAS` | Temperature suddenly reads +20 and stays there (step) |
| `PRESSURE_LEAK` | Pressure decays gradually over the fault's duration |
| `VIBRATION_SPIKE` | Vibration becomes high and noisy |
| `SENSOR_STUCK` | Temperature freezes at its last value (std → 0) |
| `SENSOR_DROPOUT` | Pressure producer stops transmitting (readings age out) |
| `COMBINED_FAULT` | Pressure decreases while vibration rises |

The fault physics live in one place — [`injector/fault_definitions.py`](injector/fault_definitions.py) —
and are reused by both the live simulator and the offline training-data
generator, so **the model trains on the same process it later sees in production.**

## Quickstart

Requires Docker (for Kafka) and Python 3.10+.

```bash
pip install -r requirements.txt

# 1. Start Kafka (single-node KRaft, topics auto-create)
docker compose up -d

# 2. Train the classifier (simulates faults offline, ~10s -> model/classifier.pkl)
python -m model.train

# 3. Start the three sensor producers (each in its own terminal, or backgrounded)
python -m producers.sensor_a &
python -m producers.sensor_b &
python -m producers.sensor_c &

# 4. Start the streaming ML classifier
python -m streaming.ml_consumer

# 5. In another terminal, open the dashboard
streamlit run dashboard/app.py

# 6. Inject faults and watch the model react
python -m injector.inject --list
python -m injector.inject --fault pressure_leak --severity 0.9 --duration 20
python -m injector.inject --fault vibration_spike
python -m injector.inject --clear
```

## How the ML works

1. **State** — `streaming/state_store.py` keeps a sliding `FEATURE_WINDOW` (10 s)
   of readings per channel.
2. **Features** — `streaming/feature_builder.py` turns each window into a fixed
   **27-dim vector**: for every channel, `mean / std / min / max / last / slope /
   range / count / age`. `slope` catches gradual drift (leaks), `age` and `count`
   catch dropouts, `std≈0` catches a stuck sensor.
3. **Model** — a `RandomForestClassifier` (`model/train.py`). No scaling needed,
   fast, and its class probabilities give a natural confidence read-out.
4. **Inference** — `streaming/ml_consumer.py` classifies once per second and
   publishes to `predictions`.

Because features come from a *time window*, detection has realistic latency: a
step fault (temp bias) is caught in well under a second, while a gradual
pressure leak takes a few seconds to become visible — exactly the streaming
trade-off worth measuring.

## Evaluation

`python -m model.evaluate` replays fresh episodes and reports the metrics that
matter for a streaming detector — not just accuracy:

- per-class **precision / recall / F1**
- **confusion matrix**
- **false-alarm rate** (how often a healthy system is flagged)
- **mean time-to-detect** after each injection

Representative run:

```
false-alarm rate: 0.035

mean time-to-detect (s)
  TEMP_SENSOR_BIAS   0.4      VIBRATION_SPIKE  0.2
  SENSOR_DROPOUT     0.7      COMBINED_FAULT   1.8
  PRESSURE_LEAK      2.4      SENSOR_STUCK     3.9
```

A model that is eventually right but takes 45 s to notice a fault is far less
useful than one that is a hair less accurate but reacts in 2 s. Time-to-detect
makes that trade-off explicit.

The dashboard computes the same metrics **live**, scoring predictions against the
injected ground truth as you drive the system.

## Layout

```
docker-compose.yml        Single-node Kafka (KRaft)
shared/                   config, topic names, message schemas (one source of truth)
producers/                sensor_a/b/c, the process simulator, fault-state listener
injector/                 inject.py CLI + fault_definitions.py (fault physics)
streaming/                state_store, feature_builder, ml_consumer
model/                    generate_training_data, train, evaluate
dashboard/                Streamlit live dashboard
notebooks/                fault_classification.ipynb (offline exploration)
```

## Notes

- `model/classifier.pkl` and `model/training_data.csv` are git-ignored — they
  rebuild in seconds with `python -m model.train`.
- Everything talks to `KAFKA_BOOTSTRAP_SERVERS` (default `localhost:9092`);
  override it and the whole system reconfigures. See `shared/config.py`.
