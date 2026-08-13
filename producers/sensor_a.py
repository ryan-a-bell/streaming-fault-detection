"""Sensor A -> temperature (~70). Run: ``python -m producers.sensor_a``."""
from producers.simulator import run_producer

if __name__ == "__main__":
    run_producer("sensor_a")
