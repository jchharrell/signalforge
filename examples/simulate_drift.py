import numpy as np
from signalforge import evaluate_health

rng = np.random.default_rng(42)
reference = rng.normal(0, 1, 2000)
current = rng.normal(0.8, 1.15, 2000)
y_true = rng.integers(0, 2, 300)
y_pred = np.where(rng.random(300) < 0.82, y_true, 1 - y_true)
latency = rng.lognormal(mean=4.6, sigma=.35, size=300)

snapshot = evaluate_health(reference, current, y_true, y_pred, latency, window=100)
print(snapshot)
for alert in snapshot.alerts:
    print("ALERT:", alert)
