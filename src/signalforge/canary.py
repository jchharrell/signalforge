from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Sequence
import numpy as np


@dataclass(frozen=True)
class CanaryDecision:
    promote: bool
    incumbent_accuracy: float
    challenger_accuracy: float
    accuracy_delta: float
    incumbent_p95_ms: float
    challenger_p95_ms: float
    latency_delta_ms: float
    reasons: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_canary(
    incumbent_correct: Sequence[bool],
    challenger_correct: Sequence[bool],
    incumbent_latency_ms: Sequence[float],
    challenger_latency_ms: Sequence[float],
    *,
    min_accuracy_gain: float = 0.01,
    max_p95_latency_regression_ms: float = 20.0,
) -> CanaryDecision:
    """Decide whether a shadow challenger is safe to promote.

    The challenger must improve quality enough to justify itself and cannot buy
    that improvement with an unacceptable tail-latency regression.  This makes
    the decision multi-objective instead of reducing model deployment to one
    metric.
    """
    inc = np.asarray(incumbent_correct, dtype=bool)
    chal = np.asarray(challenger_correct, dtype=bool)
    inc_lat = np.asarray(incumbent_latency_ms, dtype=float)
    chal_lat = np.asarray(challenger_latency_ms, dtype=float)
    if inc.size == 0 or chal.size == 0 or inc.size != chal.size:
        raise ValueError("accuracy samples must be non-empty and aligned")
    if inc_lat.size == 0 or chal_lat.size == 0:
        raise ValueError("latency samples cannot be empty")

    incumbent_accuracy = float(inc.mean())
    challenger_accuracy = float(chal.mean())
    accuracy_delta = challenger_accuracy - incumbent_accuracy
    incumbent_p95 = float(np.percentile(inc_lat, 95))
    challenger_p95 = float(np.percentile(chal_lat, 95))
    latency_delta = challenger_p95 - incumbent_p95

    reasons: list[str] = []
    if accuracy_delta < min_accuracy_gain:
        reasons.append(f"accuracy gain {accuracy_delta:.3f} is below required {min_accuracy_gain:.3f}")
    if latency_delta > max_p95_latency_regression_ms:
        reasons.append(
            f"p95 latency regression {latency_delta:.1f}ms exceeds allowed {max_p95_latency_regression_ms:.1f}ms"
        )
    if not reasons:
        reasons.append("challenger clears both quality and tail-latency gates")

    return CanaryDecision(
        promote=len(reasons) == 1 and reasons[0].startswith("challenger clears"),
        incumbent_accuracy=round(incumbent_accuracy, 6),
        challenger_accuracy=round(challenger_accuracy, 6),
        accuracy_delta=round(accuracy_delta, 6),
        incumbent_p95_ms=round(incumbent_p95, 3),
        challenger_p95_ms=round(challenger_p95, 3),
        latency_delta_ms=round(latency_delta, 3),
        reasons=tuple(reasons),
    )


@dataclass
class AlertBudget:
    """Simple alert-fatigue guardrail for a monitoring window."""

    max_alerts: int = 3
    emitted: int = 0

    def allow(self, severity: str) -> bool:
        if severity == "critical":
            self.emitted += 1
            return True
        if self.emitted >= self.max_alerts:
            return False
        self.emitted += 1
        return True
