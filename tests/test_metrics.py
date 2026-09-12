import numpy as np
import pytest
from signalforge.metrics import population_stability_index, rolling_accuracy, latency_percentile
from signalforge.monitor import evaluate_health


def test_psi_is_small_for_same_distribution():
    rng = np.random.default_rng(7)
    ref = rng.normal(0, 1, 4000)
    cur = rng.normal(0, 1, 4000)
    assert population_stability_index(ref, cur) < 0.1


def test_psi_detects_shift():
    rng = np.random.default_rng(7)
    ref = rng.normal(0, 1, 4000)
    shifted = rng.normal(2, 1, 4000)
    assert population_stability_index(ref, shifted) > 0.2


def test_rolling_accuracy_uses_latest_window():
    truth = [1, 1, 1, 1]
    pred = [1, 0, 1, 1]
    assert rolling_accuracy(truth, pred, 2)[-1] == 1.0


def test_health_combines_independent_signals():
    rng = np.random.default_rng(3)
    snap = evaluate_health(
        rng.normal(0, 1, 1000), rng.normal(2, 1, 1000),
        [1] * 100, [0] * 100, [400] * 100,
        window=50,
    )
    assert len(snap.alerts) == 3


def test_empty_latency_rejected():
    with pytest.raises(ValueError):
        latency_percentile([])
