"""Manual fault-injection CLI.

Publishes a command to the ``fault-control`` topic. The command does *not* touch
the ML consumer directly; it changes the simulated process, and the classifier
has to figure the rest out from telemetry.

Examples
--------
    python -m injector.inject --fault pressure_leak
    python -m injector.inject --fault vibration_spike --severity 0.9 --duration 20
    python -m injector.inject --clear
    python -m injector.inject --list
"""
from __future__ import annotations

import argparse
import sys

from kafka import KafkaProducer

from injector.fault_definitions import FAULTS
from shared import config
from shared.schemas import FaultControl, encode
from shared.topics import FAULT_CONTROL


def _print_faults() -> None:
    print("Available faults:")
    for name, spec in FAULTS.items():
        if name == "NORMAL":
            continue
        print(f"  {name:<18} {spec.description}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Inject a fault into the simulated system.")
    parser.add_argument("--fault", "-f", help="Fault name (case-insensitive), e.g. pressure_leak")
    parser.add_argument("--severity", "-s", type=float, default=None, help="0..1 (default: fault's own default)")
    parser.add_argument("--duration", "-d", type=float, default=None, help="Seconds to fully develop")
    parser.add_argument("--clear", action="store_true", help="Clear all faults (return to NORMAL)")
    parser.add_argument("--list", action="store_true", help="List available faults and exit")
    args = parser.parse_args(argv)

    if args.list:
        _print_faults()
        return 0

    if args.clear:
        control = FaultControl(fault="NORMAL", action="clear")
    else:
        if not args.fault:
            parser.error("provide --fault NAME, or --clear, or --list")
        name = args.fault.upper()
        spec = FAULTS.get(name)
        if spec is None or name == "NORMAL":
            print(f"Unknown fault: {args.fault!r}", file=sys.stderr)
            _print_faults()
            return 2
        control = FaultControl(
            fault=name,
            action="start",
            severity=args.severity if args.severity is not None else spec.default_severity,
            duration=args.duration if args.duration is not None else spec.default_duration,
        )

    producer = KafkaProducer(bootstrap_servers=config.KAFKA_BOOTSTRAP_SERVERS, value_serializer=encode)
    producer.send(FAULT_CONTROL, control.to_dict())
    producer.flush()
    producer.close()

    if control.action == "clear":
        print("Cleared faults -> NORMAL")
    else:
        print(f"Injected {control.fault} (severity={control.severity}, duration={control.duration}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
