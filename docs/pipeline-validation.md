# Pipeline Integrity Test

Use `scripts/validate_pipeline.py` to confirm the ingest path (OpenTelemetry -> Kafka -> Neo4j/Prometheus) preserves ordering and values for a specific trace.

## Synthetic Trace Generator
Run the lightweight telemetry generator that now lives in this repo whenever you need ad-hoc spans hitting the OTLP endpoint:

```bash
# Build/install once so `telemetrygen` is placed on your $PATH.
go install ./cmd/telemetrygen

# Emit synthetic traces into the ingestion port.
telemetrygen traces --traces 50 --otlp-endpoint localhost:4317

# Alternatively, run it without installing a binary:
go run ./cmd/telemetrygen traces --service-name loadtest
```

The generator speaks OTLP gRPC, so it works against the bundled ingestion container or any Collector running on `localhost:4317`.

## Steps
1. Run the mock suite and chaos scenarios via `docker compose up --build` (includes Kafka, Neo4j, Prometheus, OTLP ingestion, mock services, Locust).
2. Capture interesting trace IDs from the ingestion logs or the `svc-*` service logs (search for `TraceID=` entries).
3. For each trace, record the expected causal path (e.g., `gateway,orchestrator,payments,recommendation,db`) and the timestamp of the span when a breach occurred.
4. Execute:
   ```bash
   python scripts/validate_pipeline.py \
     --trace-id <trace> \
     --expected-path gateway,orchestrator,payments,recommendation,db \
     --service payments \
     --event-ts 1715020000 \
     --latency-metric service_latency_ms
   ```
   - Kafka check: Confirms the keyed trace exists without reordering.
   - Neo4j check: Traverses the stored graph to ensure every service/dependency is represented.
   - Prometheus check: Calculates p99 latency around the event timestamp and verifies it breaches your threshold.

If any stage fails, inspect:
- Kafka: ensure the OTLP ingestion service uses `trace_id` as the message key.
- Neo4j: verify the ingestion job writes nodes/relationships with the right labels/properties.
- Prometheus: confirm the metrics pipeline emits `service_latency_ms{service="..."}` and that scrape jobs include the target service.
