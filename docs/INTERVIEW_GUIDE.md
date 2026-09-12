# SignalForge interview guide

## 30-second version

SignalForge is a NumPy-based ML monitoring toolkit. It attributes drift to individual features, creates an operational health report, and evaluates whether a shadow challenger should replace the incumbent based on both quality and p95 latency. I also added a small alert budget to model the human side of monitoring: too many weak alerts destroy trust.

## Five-minute walkthrough

1. **Problem:** model quality can degrade after deployment even if the code never changes.
2. **`metrics.py`:** PSI, rolling accuracy, and latency percentiles are the raw signals.
3. **`fingerprint.py`:** instead of one drift alarm, calculate drift per feature and normalize contribution. The result answers *what moved?*
4. **`report.py`:** combine drift, latency, and optional accuracy into an actionable health report.
5. **`canary.py`:** compare an incumbent and shadow challenger. Promotion requires enough accuracy gain without unacceptable p95 latency regression.
6. **`AlertBudget`:** suppress ordinary alerts once the monitoring window has spent its attention budget; critical alerts are never suppressed.
7. **Tests/CLI:** deterministic synthetic scenarios make every concept runnable without external data or cloud services.

## Questions I should be able to answer

### What is PSI?
Population Stability Index compares the proportion of observations in reference-derived bins with the current proportions. Larger changes produce larger contributions. SignalForge uses reference quantiles so bins represent the baseline distribution.

### What PSI threshold is “bad”?
There is no universal threshold. Values such as 0.1/0.25 are common heuristics, but the correct threshold depends on feature behavior, sample size, model sensitivity, and business cost. The project deliberately reports the raw value and attribution instead of pretending one threshold is universally correct.

### Why calculate a Jensen-Shannon-style distance too?
PSI is directional because the bins come from the reference population. Jensen-Shannon distance is symmetric and gives another view of distribution change. I would not alert on both independently; I would use the second metric to investigate or validate an unusual PSI result.

### Why is feature drift not the same as model degradation?
A feature can drift while predictions remain correct, and a model can degrade without dramatic univariate feature drift because relationships between variables changed. That is why the health report can also include labeled accuracy when labels become available.

### What are delayed labels?
In many production systems you know the prediction immediately but only learn the correct outcome later. Monitoring has to join historical predictions to outcomes once labels arrive. Until then, distribution and service telemetry act as leading indicators.

### Why p95 latency instead of average latency?
Average latency can hide a bad tail. Users experiencing the slowest 5% of requests can have a poor experience even when the mean looks fine.

### Why shadow traffic?
A shadow challenger receives copies of real requests without controlling production decisions. It lets the team compare behavior under real traffic before taking deployment risk.

### Why no scikit-learn or PyTorch?
The toolkit monitors models rather than trains them. Keeping the runtime NumPy-only makes the implementation easy to inspect and framework-neutral.

## Memorable angle

My favorite part is that SignalForge asks two questions that ordinary dashboards often skip:

- **Which feature owns the drift?**
- **Is the new model better enough to justify what it costs at runtime?**

That makes the project about production decisions, not just plotting metrics.

## What I would not claim

- PSI alone proves a model is broken.
- The synthetic thresholds are appropriate for every model.
- The canary logic replaces statistical significance testing or a real rollout policy.
- The toolkit currently handles every data type or distributed telemetry source.
