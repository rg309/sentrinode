# Mock Microservice Suite

This suite produces high-fidelity traces and metrics for the latency predictor pipeline. It spins up six FastAPI services arranged in a non-linear DAG plus a Locust load generator and chaos scripts.

## Topology
```
Gateway -> Orchestrator -> [Payments -> (Recommendation -> DB, DB write)
                             Inventory -> DB read]
```
All services emit OpenTelemetry traces/metrics to the `ingestion` OTLP receiver and expose Prometheus metrics on `/metrics`.

## Running
```
docker compose up --build gateway orchestrator payments inventory recommendation db locust ingestion
```
(Or simply `docker compose up --build` to include Kafka/Neo4j/Prometheus already defined in the repo.) Once running:
- Entry point: `http://localhost:8080/api/checkout`
- Locust UI: `http://localhost:8089`

## Chaos Injection
Scripts under `chaos/` help reproduce latency incidents:
- `./chaos/inject_latency.sh svc-payments eth0 100ms 30ms` adds netem delay between Payments and DB.
- `./chaos/clear_latency.sh svc-payments` removes the rule.
- `./chaos/cpu_saturation.sh svc-inventory 180 2` runs `stress-ng` inside the Inventory container.

Every injection should be logged along with timestamps so you can correlate to P99 spikes downstream. After each chaos window, run `python scripts/validate_pipeline.py --trace-id <id> ...` (details in `docs/pipeline-validation.md`) to confirm the pipeline stored the complete trace and matching p99 latency samples.

## Load Generation
Locust targets the Gateway with checkout flows; scale workers by setting `LOCUST_WORKERS` and `LOCUST_USERS` environment variables in `docker-compose.yml`.

## Extending
- Add additional branches (e.g., Shipping, Fraud) by duplicating service templates in `services/`.
- Update Prometheus scrape config to include each service (ports 8000-8005) so metrics land in TSDB.
- Adjust environment variables (`*_URL`, `CPU_WORK_MS`, `DB_BASE_LATENCY_MS`) to craft scenarios.
