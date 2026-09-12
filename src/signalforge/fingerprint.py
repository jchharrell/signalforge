from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Mapping, Sequence
import numpy as np

from .metrics import population_stability_index


@dataclass(frozen=True)
class FeatureDrift:
    feature: str
    psi: float
    severity: str
    share_of_total: float


@dataclass(frozen=True)
class DriftFingerprint:
    total_drift: float
    dominant_feature: str | None
    features: tuple[FeatureDrift, ...]

    def to_dict(self) -> dict:
        return {
            "total_drift": self.total_drift,
            "dominant_feature": self.dominant_feature,
            "features": [asdict(item) for item in self.features],
        }


def _severity(psi: float) -> str:
    if psi < 0.10:
        return "stable"
    if psi < 0.25:
        return "watch"
    return "shifted"


def drift_fingerprint(
    reference: Mapping[str, Sequence[float]],
    current: Mapping[str, Sequence[float]],
    bins: int = 10,
) -> DriftFingerprint:
    """Attribute distribution drift across features.

    Instead of emitting a single alarm, SignalForge answers a more useful
    operational question: *which features are responsible for the alarm?*
    PSI is computed per feature, then normalized into a drift contribution.
    """
    shared = sorted(set(reference).intersection(current))
    if not shared:
        raise ValueError("reference and current must share at least one feature")

    raw = [(name, population_stability_index(reference[name], current[name], bins)) for name in shared]
    total = float(sum(value for _, value in raw))
    denominator = total if total > 0 else 1.0
    ranked = sorted(raw, key=lambda item: item[1], reverse=True)
    features = tuple(
        FeatureDrift(name, round(value, 6), _severity(value), round(value / denominator, 6))
        for name, value in ranked
    )
    return DriftFingerprint(round(total, 6), features[0].feature if features else None, features)


def distribution_distance(reference: Sequence[float], current: Sequence[float], bins: int = 20) -> float:
    """Symmetric Jensen-Shannon style distance without scipy.

    This complements PSI in interviews: PSI is conventional and easy to
    threshold; this metric is symmetric and bounded after square-rooting.
    """
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    if ref.size == 0 or cur.size == 0:
        raise ValueError("inputs cannot be empty")
    low = min(float(ref.min()), float(cur.min()))
    high = max(float(ref.max()), float(cur.max()))
    if low == high:
        return 0.0
    ref_hist, edges = np.histogram(ref, bins=bins, range=(low, high), density=False)
    cur_hist, _ = np.histogram(cur, bins=edges, density=False)
    eps = 1e-12
    p = np.clip(ref_hist / ref_hist.sum(), eps, None)
    q = np.clip(cur_hist / cur_hist.sum(), eps, None)
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))
    return float(np.sqrt(0.5 * (kl_pm + kl_qm)))
