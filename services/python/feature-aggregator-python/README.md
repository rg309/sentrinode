# Feature Aggregator Service

This Python service computes training features by blending graph-derived context from Neo4j with time-series metrics from Prometheus. It can run on a schedule or react to new transaction windows pushed through Kafka.

## Capabilities
- Graph feature extraction: Neo4j Cypher queries walk from an entry service to critical downstream systems (e.g., databases) and report structural metrics (longest transactional path, path depth, fan-out for high-risk services).
- Time-series feature extraction: PromQL queries fetch aligned CPU rolling averages and future (t + 5m) latency targets for the same critical services.
- Feature store persistence: Every aggregation produces a Parquet file that can be used as model training data (columns for `cpu_rolling_avg`, `path_depth`, `fan_out`, `target_latency_p99`, etc.). A sidecar JSON file stores run metadata.

## Layout
```
feature-aggregator-python/
├── app.py                      # CLI entrypoint
├── requirements.txt            # Python dependencies
└── feature_aggregator/
    ├── config.py               # Environment-driven configuration
    ├── jobs.py                 # Job payload parsing/scheduling helpers
    ├── neo4j_extractor.py      # Cypher helpers for path depth/fan-out
    ├── prometheus_extractor.py # PromQL-based CPU + latency fetchers
    ├── feature_store.py        # Parquet persistence utilities
    ├── service.py              # Aggregation orchestrator
    └── triggers.py             # Kafka trigger implementation
```

## Prerequisites
1. Python 3.11+
2. Install dependencies:
   ```bash
   cd feature-aggregator-python
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. Ensure Neo4j, Prometheus, and Kafka endpoints from `docker-compose.yml` are running (or set the environment variables below to point at existing clusters).

## Configuration
Environment variables drive all connections and scheduling:

| Variable | Default | Description |
| --- | --- | --- |
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j Bolt endpoint |
| `NEO4J_USER` / `NEO4J_PASSWORD` | `neo4j` / `testpassword` | Neo4j credentials |
| `PROMETHEUS_URL` | `http://localhost:9090` | Prometheus base URL |
| `CPU_METRIC` | `service_cpu_utilization` | PromQL metric for CPU utilization |
| `LATENCY_METRIC` | `service_latency_ms` | PromQL histogram/counter for latency |
| `ENTRY_SERVICE` | `api-gateway` | Root node for longest path computations |
| `HIGH_RISK_SERVICES` | `payments-service,checkout-service,db-writer` | CSV list of services to score |
| `FEATURE_WINDOW_MINUTES` | `5` | Rolling window for CPU metrics |
| `TARGET_HORIZON_MINUTES` | `5` | Future latency target offset |
| `FEATURE_STORE_PATH` | `./feature_store` | Output path for Parquet files |
| `FEATURE_FILE_PREFIX` | `features` | Filename prefix |
| `FEATURE_AGGREGATOR_MODE` | `once` | `once`, `schedule`, or `kafka` |
| `SCHEDULE_INTERVAL_SECONDS` | `300` | Interval for schedule mode |
| `KAFKA_TRIGGER_ENABLED` | `false` | Enable Kafka consumer |
| `KAFKA_BOOTSTRAP` | `localhost:9093` | Kafka bootstrap brokers |
| `KAFKA_FEATURE_TOPIC` | `feature_aggregation_requests` | Trigger topic |
| `KAFKA_GROUP_ID` | `feature-aggregator` | Kafka consumer group |

## Usage
### Run Once
```
python app.py --mode once
```

### Interval Scheduler
```
FEATURE_AGGREGATOR_MODE=schedule SCHEDULE_INTERVAL_SECONDS=600 python app.py
```

### Kafka Trigger
```
export FEATURE_AGGREGATOR_MODE=kafka
export KAFKA_TRIGGER_ENABLED=true
python app.py
```
Each Kafka message should be JSON like:
```json
{
  "entry_service": "api-gateway",
  "high_risk_services": ["payments", "inventory", "db-writer"],
  "window_minutes": 5,
  "target_horizon_minutes": 5,
  "timestamp": "2024-05-07T12:00:00Z"
}
```

## Docker Compose Quickstart
The repository `docker-compose.yml` now includes Prometheus and the Feature Aggregator container. To exercise the full path end-to-end:
1. From the repo root run `docker compose up --build`.
2. Once services are healthy, exec into the aggregator container to inspect generated Parquet files: `docker compose exec feature-aggregator ls /data/feature_store`.
3. To validate Prometheus connectivity, query the API directly:
   ```bash
   curl 'http://localhost:9090/api/v1/query?query=up'
   ```
4. When real service metrics become available, update `prometheus/prometheus.yml` targets and restart the stack.

The aggregator runs in schedule mode by default (10 minute interval). Override by setting environment variables in `docker-compose.yml` or using `docker compose run feature-aggregator python app.py --mode once`.

## Feature Store Output
- Location: `${FEATURE_STORE_PATH}` (default `feature_store/`).
- File naming: `features_<window_end>.parquet` + `features_<window_end>.json` metadata.
- Sample columns:
  - `timestamp`
  - `entry_service`
  - `service`
  - `cpu_rolling_avg`
  - `path_depth`
  - `fan_out`
  - `critical_path`
  - `target_latency_p99`

These rows map directly to the training structure `(X1=CPU, X2=Path Depth, X3=Fan-Out, ..., Y=Future Latency)` required by the latency predictor pipeline.
