from __future__ import annotations
from dataclasses import dataclass
from .metrics import population_stability_index, rolling_accuracy, latency_percentile

@dataclass(frozen=True)
class HealthSnapshot:
    accuracy: float
    psi: float
    p95_latency_ms: float
    alerts: tuple[str, ...]


def evaluate_health(reference_feature, current_feature, y_true, y_pred, latencies_ms,
                    min_accuracy: float = .80, max_psi: float = .20,
                    max_p95_latency_ms: float = 250.0, window: int = 100) -> HealthSnapshot:
    accuracy = float(rolling_accuracy(y_true, y_pred, window)[-1])
    psi = population_stability_index(reference_feature, current_feature)
    p95 = latency_percentile(latencies_ms)
    alerts = []
    if accuracy < min_accuracy:
        alerts.append(f"accuracy below threshold: {accuracy:.3f}")
    if psi > max_psi:
        alerts.append(f"population drift detected: PSI={psi:.3f}")
    if p95 > max_p95_latency_ms:
        alerts.append(f"latency above threshold: p95={p95:.1f}ms")
    return HealthSnapshot(accuracy, psi, p95, tuple(alerts))
