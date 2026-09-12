from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Mapping, Sequence

from .fingerprint import drift_fingerprint
from .metrics import latency_percentile


@dataclass(frozen=True)
class HealthReport:
    status: str
    drift_total: float
    dominant_feature: str | None
    p95_latency_ms: float
    accuracy: float | None
    actions: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)


def build_health_report(
    reference: Mapping[str, Sequence[float]],
    current: Mapping[str, Sequence[float]],
    latencies_ms: Sequence[float],
    *,
    accuracy: float | None = None,
    drift_watch: float = 0.25,
    drift_critical: float = 0.50,
    latency_slo_ms: float = 250.0,
    accuracy_floor: float = 0.80,
) -> HealthReport:
    fingerprint = drift_fingerprint(reference, current)
    p95 = latency_percentile(latencies_ms, 95)
    actions: list[str] = []
    status = "healthy"

    if fingerprint.total_drift >= drift_critical:
        status = "critical"
        actions.append(f"inspect {fingerprint.dominant_feature}; it dominates the observed population shift")
    elif fingerprint.total_drift >= drift_watch:
        status = "watch"
        actions.append(f"watch {fingerprint.dominant_feature} and compare against the next labeled window")

    if p95 > latency_slo_ms:
        status = "critical" if p95 > latency_slo_ms * 1.5 else max(status, "watch", key={"healthy": 0, "watch": 1, "critical": 2}.get)
        actions.append(f"p95 latency {p95:.1f}ms exceeds {latency_slo_ms:.1f}ms SLO")

    if accuracy is not None and accuracy < accuracy_floor:
        status = "critical"
        actions.append(f"accuracy {accuracy:.3f} is below floor {accuracy_floor:.3f}; freeze promotion")

    if not actions:
        actions.append("no intervention required; continue collecting telemetry")

    return HealthReport(
        status=status,
        drift_total=round(fingerprint.total_drift, 6),
        dominant_feature=fingerprint.dominant_feature,
        p95_latency_ms=round(p95, 3),
        accuracy=None if accuracy is None else round(float(accuracy), 6),
        actions=tuple(actions),
    )
