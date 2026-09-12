# SignalForge

**Explainable ML monitoring that tells you *what drifted*, whether a shadow model is actually worth promoting, and when alerting should stop before humans tune it out.**

SignalForge is a Python monitoring toolkit built around three practical ML-engineering problems:

1. A single drift score is not enough — teams need to know **which feature caused it**.
2. A challenger model should not ship just because accuracy is higher — it also has to respect **tail-latency constraints**.
3. Monitoring can fail socially even when it works technically — too many low-value alerts create **alert fatigue**.

The browser UI remains a lightweight visual demo; the `src/signalforge/` package contains the substantive implementation.

> All telemetry in the included examples is synthetic.

## What makes this project different

### Drift Fingerprints

`drift_fingerprint()` calculates PSI for every shared feature, ranks the shifted features, and normalizes each feature's contribution to total observed drift. Instead of returning only `PSI = 0.63`, SignalForge can answer: **annual_spend created 82% of the shift**.

It also includes a symmetric Jensen-Shannon-style distribution distance implemented with NumPy only, which gives a useful second way to reason about distribution change.

### Shadow-model promotion gate

`evaluate_canary()` compares an incumbent and challenger on two dimensions:

- accuracy gain
- p95 latency regression

A challenger is promoted only if it clears both gates. This creates a realistic engineering conversation: a model that is 2% more accurate but adds 200 ms of tail latency may be a worse product.

### Alert budget

`AlertBudget` caps ordinary alerts during a monitoring window while still allowing critical alerts through. It is intentionally small, but it demonstrates an operational idea that matters: **a monitoring system that humans ignore is not a monitoring system.**

## Architecture

```text
telemetry window / synthetic demo
           |
           v
   +-------------------+
   | metrics.py        |  PSI, rolling accuracy, latency percentiles
   +-------------------+
           |
           +----------> fingerprint.py  -> feature attribution
           |
           +----------> report.py       -> health + recommended actions
           |
shadow samples --------> canary.py       -> promote / hold decision
                                      
CLI (cli.py) exposes demo, monitor, and canary commands
```

The package has no external service dependency. NumPy is the only runtime dependency.

## Quick start

Requires Python 3.10+.

```bash
git clone https://github.com/jchharrell/signalforge.git
cd signalforge
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Run the deterministic built-in scenario:

```bash
signalforge demo
```

or:

```bash
python -m signalforge.cli demo
```

The output includes a drift fingerprint, dominant feature, health status, p95 latency, and recommended action.

## Use your own telemetry JSON

```bash
signalforge monitor telemetry.json
```

Expected shape:

```json
{
  "reference": {
    "age": [31, 44, 27],
    "annual_spend": [2200, 4800, 3100]
  },
  "current": {
    "age": [32, 45, 29],
    "annual_spend": [5100, 6200, 5900]
  },
  "latencies_ms": [91, 104, 137],
  "accuracy": 0.91
}
```

For a shadow-model decision:

```bash
signalforge canary canary.json
```

The canary command reports incumbent/challenger accuracy, p95 latency for each, deltas, a promotion decision, and the reasons behind that decision.

## Tests

```bash
pytest
```

The test suite covers:

- PSI and rolling accuracy edge cases
- dominant-feature drift attribution
- symmetric distribution distance
- successful and blocked canary promotions
- alert-budget behavior
- health-report recommendations
- deterministic CLI demo output

GitHub Actions installs the package and runs the full suite on every push and pull request.

## Design decisions I can explain

**Why PSI?** It is common in risk/model monitoring and easy to explain to non-specialists. I did not treat it as magical: SignalForge adds feature attribution and a second symmetric distance measure because a single threshold should not be the whole monitoring strategy.

**Why no full ML framework?** The interesting part is monitoring, not training. Keeping the package NumPy-only makes the calculations easy to inspect and lets it monitor predictions from any framework.

**Why gate on latency and accuracy together?** Deployment is a systems decision. A better offline metric can still produce a worse user experience if inference becomes too slow.

**Why an alert budget?** Repeated weak alerts train operators to ignore the system. Production monitoring should consider human attention as a scarce resource.

## Production extensions

A production version would ingest telemetry from Kafka/Kinesis or a feature store, persist time-windowed metrics, track model/feature versions, support categorical drift, include delayed-label joins, send alerts to incident tooling, expose Prometheus/OpenTelemetry metrics, and support policy configuration per model.

## Interview walkthrough

See [`docs/INTERVIEW_GUIDE.md`](docs/INTERVIEW_GUIDE.md) for a five-minute project story and technical questions I should be ready to answer.
