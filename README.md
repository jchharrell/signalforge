# SignalForge

A small ML-observability toolkit plus interactive browser demo. It focuses on a practical question: **how do you know a model is becoming unreliable after deployment?**

## Signals implemented

- rolling classification accuracy
- Population Stability Index (PSI) for feature-distribution shift
- p95 inference latency
- configurable alert policy combining independent signals
- reproducible synthetic drift simulation

## Why this project

Model quality is not just training accuracy. Input populations change, labels may arrive late, and infrastructure can slow down even when predictions remain correct. SignalForge keeps those concerns separate so each alert has a reason an engineer can investigate.

## Python package

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest -q
python examples/simulate_drift.py
```

Core modules:
- `metrics.py`: PSI, rolling accuracy, latency percentiles
- `monitor.py`: threshold policy and immutable health snapshot
- `tests/`: normal, shifted, invalid-input, and multi-alert cases

The browser prototype (`index.html`) remains available for a quick visual demonstration.

## Technical decisions I can explain

PSI bins are derived from reference quantiles so the baseline population defines expected buckets. Small epsilon clipping prevents division-by-zero when a bucket is empty. I do not treat PSI as a universal truth: thresholds are policy decisions and should be calibrated using historical incidents and false-alert cost.

Rolling accuracy requires labels, so a production version would also monitor label-free signals such as feature drift and prediction distributions while labels are delayed.

## Production direction

Persist time-series metrics, segment monitoring by model/version/cohort, add Prometheus/OpenTelemetry export, scheduled baselines, alert deduplication, and a service endpoint that evaluates batches.

## Scope

All telemetry is synthetic. This is an engineering/monitoring project, not a generative-AI wrapper and not a production monitoring service.
