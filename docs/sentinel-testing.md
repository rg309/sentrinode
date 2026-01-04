# AI Sentinel Testing ("Chaos Button")

Use `test_ai.sh` to trigger repeatable workloads and verify the end-to-end pipeline.

## 1. Healthy Baseline

```bash
./test_ai.sh --healthy
```

- Emits standard telemetrygen traces (default span duration ~75 ms) with four workers.
- The Sentinel (`python sentinel.py`) should log `Healthy` for each span pair because actual latency stays close to the prediction.

## 2. Slow Database Scenario

```bash
./test_ai.sh --slow-db
```

- Emits traces with `--span-duration 500ms`.
- The Sentinel should flip to `ANOMALY` as actual child span latency exceeds the model’s prediction by more than 50%.

## 3. Measuring the “Delta”

- Run `--healthy` while the system is tagged with a `Low` load (e.g., set `system_load="Low"` in parent spans). A 100 ms service should raise an alert because the model has learned that low-load requests should remain fast.
- Switch to a `High` load scenario (run `--slow-db` and ensure the parent spans carry `system_load="High"`). The AI allows ~100 ms spans without alerting because historical data shows High load tolerates larger latencies.
- This demonstrates the adaptive “intelligence”: identical latency values are interpreted differently depending on the learned load profile.
