import numpy as np

from signalforge.canary import AlertBudget, evaluate_canary
from signalforge.fingerprint import distribution_distance, drift_fingerprint
from signalforge.report import build_health_report


def test_drift_fingerprint_identifies_dominant_feature():
    rng = np.random.default_rng(7)
    reference = {
        "income": rng.normal(60, 8, 1000),
        "age": rng.normal(40, 5, 1000),
    }
    current = {
        "income": rng.normal(88, 8, 1000),
        "age": rng.normal(40.2, 5, 1000),
    }
    result = drift_fingerprint(reference, current)
    assert result.dominant_feature == "income"
    assert result.features[0].share_of_total > 0.7
    assert result.features[0].severity == "shifted"


def test_distribution_distance_is_zero_for_identical_constant_data():
    assert distribution_distance([3, 3, 3], [3, 3, 3]) == 0.0


def test_canary_promotes_quality_gain_without_large_latency_regression():
    incumbent = [True] * 80 + [False] * 20
    challenger = [True] * 90 + [False] * 10
    decision = evaluate_canary(
        incumbent,
        challenger,
        [90, 100, 110, 120] * 25,
        [95, 105, 115, 125] * 25,
        min_accuracy_gain=0.05,
        max_p95_latency_regression_ms=15,
    )
    assert decision.promote is True
    assert decision.accuracy_delta == 0.1


def test_canary_blocks_fast_accuracy_gain_if_tail_latency_is_too_expensive():
    decision = evaluate_canary(
        [True] * 80 + [False] * 20,
        [True] * 95 + [False] * 5,
        [100] * 100,
        [180] * 100,
        min_accuracy_gain=0.05,
        max_p95_latency_regression_ms=20,
    )
    assert decision.promote is False
    assert any("latency regression" in reason for reason in decision.reasons)


def test_alert_budget_suppresses_noise_but_never_critical():
    budget = AlertBudget(max_alerts=2)
    assert budget.allow("watch") is True
    assert budget.allow("watch") is True
    assert budget.allow("watch") is False
    assert budget.allow("critical") is True


def test_health_report_gives_action_not_just_number():
    rng = np.random.default_rng(2)
    report = build_health_report(
        {"score": rng.normal(0, 1, 600)},
        {"score": rng.normal(3, 1, 600)},
        [100, 120, 140, 160],
        accuracy=0.92,
    )
    assert report.status in {"watch", "critical"}
    assert report.dominant_feature == "score"
    assert "inspect" in report.actions[0] or "watch" in report.actions[0]
