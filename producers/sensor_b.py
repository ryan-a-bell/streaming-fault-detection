"""Sensor B -> pressure (~100). Run: ``python -m producers.sensor_b``."""
from producers.simulator import run_producer

if __name__ == "__main__":
    run_producer("sensor_b")
