from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

from .canary import evaluate_canary
from .fingerprint import drift_fingerprint
from .report import build_health_report


def _load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def monitor_command(path: str) -> dict:
    payload = _load(path)
    reference = payload["reference"]
    current = payload["current"]
    latencies = payload["latencies_ms"]
    accuracy = payload.get("accuracy")
    fingerprint = drift_fingerprint(reference, current)
    health = build_health_report(reference, current, latencies, accuracy=accuracy)
    return {"fingerprint": fingerprint.to_dict(), "health": health.to_dict()}


def canary_command(path: str) -> dict:
    payload = _load(path)
    result = evaluate_canary(
        payload["incumbent_correct"],
        payload["challenger_correct"],
        payload["incumbent_latency_ms"],
        payload["challenger_latency_ms"],
        min_accuracy_gain=float(payload.get("min_accuracy_gain", 0.01)),
        max_p95_latency_regression_ms=float(payload.get("max_p95_latency_regression_ms", 20.0)),
    )
    return result.to_dict()


def demo(seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    reference = {
        "age": rng.normal(42, 10, 1500).tolist(),
        "annual_spend": rng.lognormal(8.2, 0.35, 1500).tolist(),
        "sessions_30d": rng.poisson(8, 1500).astype(float).tolist(),
    }
    current = {
        "age": rng.normal(42.3, 10, 1500).tolist(),
        "annual_spend": rng.lognormal(8.65, 0.35, 1500).tolist(),
        "sessions_30d": rng.poisson(8.2, 1500).astype(float).tolist(),
    }
    latencies = rng.lognormal(4.7, 0.22, 1000).tolist()
    fingerprint = drift_fingerprint(reference, current)
    health = build_health_report(reference, current, latencies, accuracy=0.91)
    return {"fingerprint": fingerprint.to_dict(), "health": health.to_dict()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Explain model drift and gate shadow-model promotions")
    sub = parser.add_subparsers(dest="command", required=True)
    monitor = sub.add_parser("monitor", help="analyze a JSON telemetry window")
    monitor.add_argument("input")
    canary = sub.add_parser("canary", help="evaluate a shadow challenger")
    canary.add_argument("input")
    demo_parser = sub.add_parser("demo", help="run a deterministic synthetic drift scenario")
    demo_parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.command == "monitor":
        result = monitor_command(args.input)
    elif args.command == "canary":
        result = canary_command(args.input)
    else:
        result = demo(args.seed)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
