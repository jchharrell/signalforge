from __future__ import annotations
import numpy as np


def population_stability_index(reference, current, bins: int = 10) -> float:
    """Measure distribution shift using reference-derived quantile bins."""
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    if ref.size < bins or cur.size == 0:
        raise ValueError("not enough observations")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if edges.size < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)
    eps = 1e-6
    ref_pct = np.clip(ref_counts / ref_counts.sum(), eps, None)
    cur_pct = np.clip(cur_counts / cur_counts.sum(), eps, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def rolling_accuracy(y_true, y_pred, window: int = 100) -> np.ndarray:
    true = np.asarray(y_true)
    pred = np.asarray(y_pred)
    if true.shape != pred.shape or window < 1 or len(true) < window:
        raise ValueError("invalid shapes or window")
    correct = (true == pred).astype(float)
    return np.convolve(correct, np.ones(window) / window, mode="valid")


def latency_percentile(latencies_ms, percentile: float = 95) -> float:
    values = np.asarray(latencies_ms, dtype=float)
    if values.size == 0:
        raise ValueError("latency series is empty")
    return float(np.percentile(values, percentile))
