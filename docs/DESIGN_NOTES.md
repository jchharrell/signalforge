# SignalForge Design Notes

SignalForge is a lightweight Python/NumPy toolkit for reasoning about model health after deployment. It focuses on drift attribution, operational tradeoffs, and explainable rollout decisions rather than model training.

## Design goals

- separate raw monitoring metrics from operational decisions
- attribute drift to individual features rather than only reporting one aggregate alarm
- combine quality and latency when evaluating a challenger model
- model alert fatigue explicitly instead of assuming unlimited operator attention
- keep every example deterministic and runnable without cloud services

## Monitoring primitives

`metrics.py` implements population stability index, rolling accuracy, and latency percentiles. These functions are intentionally framework-neutral so they can be used with models trained in scikit-learn, PyTorch, TensorFlow, or another system.

`fingerprint.py` calculates feature-level drift and normalizes contribution so an operator can identify which inputs are responsible for most of the distribution change.

`report.py` combines drift, optional labeled performance, and latency into a single health report while preserving the underlying measurements for inspection.

## Challenger evaluation

`canary.py` compares an incumbent model with a shadow challenger. Promotion requires meaningful quality improvement without an unacceptable p95-latency regression. This makes rollout logic a multi-objective decision instead of a simple 'higher accuracy wins' rule.

The thresholds in the repository are demonstration policy, not universal production defaults.

## Alert budget

The alert-budget component models a practical monitoring problem: weak alerts become useless when operators receive too many of them. Ordinary alerts can be suppressed after the budget is exhausted, while critical conditions still surface.

## Why PSI is not enough

Feature drift does not prove model degradation. A model can remain accurate while inputs shift, and performance can degrade because relationships between variables changed even when univariate drift is modest. SignalForge therefore treats drift as one signal alongside labeled accuracy and service telemetry.

## Verification

The repository includes pytest coverage for metric behavior, drift attribution, challenger decisions, alert budgeting, and CLI execution. GitHub Actions installs the package, runs the tests, and executes a deterministic CLI smoke scenario.

## Scope

SignalForge uses synthetic examples and intentionally small implementations so the monitoring logic stays inspectable. It is not a replacement for a distributed telemetry platform or a complete statistical rollout framework.