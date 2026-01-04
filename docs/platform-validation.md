# Platform Validation (Trust & Reliability)

This checklist exercises the production-like mock environment to ensure the latency predictor delivers the alerting guarantees SREs expect. Each scenario ties back to the metrics defined in the spec.

---

## 1. Alerting Integrity
**Goal:** An alert fires within < 1 minute of the relevant data hitting the pipeline and includes the correct service + remediation hint.

**Setup**
1. `docker compose up --build` (brings up Kafka, Neo4j, Prometheus, OTLP ingestion, mock services, feature aggregator).
2. Drive steady traffic with Locust:
   ```bash
   docker compose up locust
   # Optional: override load
   LOCUST_USERS=200 LOCUST_SPAWN_RATE=20 docker compose up locust
   ```
3. Start the real-time prediction/alerting process (Step 5 service once available). Ensure it consumes feature rows or OTLP spans in real time and emits alerts (Kafka topic, webhook, etc.).

**Trigger**
1. Inject latency between Payments (Service B) and DB D:
   ```bash
   ./chaos/inject_latency.sh svc-payments eth0 150ms 40ms
   ```
   Note the exact timestamp printed by the script (this becomes the “fault start” marker).
2. Optionally spike CPU on Inventory to simulate concurrent saturation:
   ```bash
   ./chaos/cpu_saturation.sh svc-inventory 180 2
   ```

**Measurement**
1. The alerting service should log an incident (or post to your alert sink) referencing `service=payments` (or whichever tier actually breached).
2. Capture two timestamps:
   - `t_ingest`: when the feature window ending at the breach entered Kafka/feature store. You can read `feature_store/features_<ts>.json` and use `window_end`.
   - `t_alert`: when the alert message was emitted.
3. Compute Time-to-Alert `TTA = t_alert - t_ingest` (must be < 60s). Record in your runbook.
4. Validate the alert payload contains the causal hint from SHAP (e.g., “High CPU on Inventory”). If it doesn’t, run `python model-training/analyze_causality.py ...` for that trace window and compare the top feature.

**Pass Criteria**
- TTA < 1 minute for every injected incident.
- Alert body references the correct service + recommended mitigation.

---

## 2. Model Drift
**Goal:** Accuracy remains ≥90% after topology or workload changes.

**Scenarios**
1. **Deployment change:** Edit one of the mock services (e.g., make Payments call an additional Fraud service) or adjust env vars to change latency distributions (`DB_BASE_LATENCY_MS`, `CPU_WORK_MS`).
2. **Network profile change:** Use `./chaos/inject_latency.sh` with a new latency magnitude/pattern.

**Process**
1. Collect several hours of feature data after the change (Feature Aggregator already persists Parquet files).
2. Run backtesting on the pre-change window to establish baseline metrics:
   ```bash
   cd model-training
   python backtest.py --feature-store-path ../feature_store --output artifacts/backtest_pre.json
   ```
3. Run the same backtest on the post-change range (set `--start-ts` / `--end-ts` by filtering the feature store manually or by temporarily moving older Parquet files to another folder).
4. Compare `metrics[].metrics.breach_accuracy` and `false_positive_rate` between pre/post results. Accuracy delta should remain above the agreed threshold (e.g., ≥90% accuracy and ≤1% FPR). If not:
   - Retrain the LSTM with the new samples (`python train.py ...`).
   - Validate the updated checkpoint via `metrics.json`.

**Pass Criteria**
- Accuracy drop ≤ 10 percentage points.
- FPR still meets the <1% requirement after retraining.

---

## 3. Scaling Test
**Goal:** 10× traffic increases do not degrade ingestion or prediction latency.

**Steps**
1. Scale Locust users/spawn rate by ~10×:
   ```bash
   LOCUST_USERS=500 LOCUST_SPAWN_RATE=50 docker compose up locust
   ```
2. Monitor pipeline resource usage:
   - `docker stats ingestion kafka feature-aggregator`
   - Or capture OTEL metrics (if enabled) for CPU/memory.
3. Measure end-to-end prediction latency:
   - Record the timestamp when a trace enters the OTLP ingestion service (`ingestion` logs).
   - Record when the real-time prediction is produced (prediction logs or alert timestamp).
   - `prediction_latency = t_prediction - t_ingest` must stay < 500 ms at P99.
   - Automate via log scraping or instrument the prediction service to emit latency metrics (`prediction_latency_ms` histogram).

4. Watch Kafka lag (`kafka-consumer-groups --bootstrap-server localhost:9093 --describe --group <prediction_group>`) to ensure no backlog builds under peak load.

**Pass Criteria**
- Ingestion CPU/memory scale roughly linearly (no runaway memory or CPU).
- `prediction_latency_p99 < 500 ms`.
- Kafka lag remains near zero; no dropped spans/features.

---

## Evidence Collection
For each scenario, capture:
- Commands executed (chaos scripts, Locust settings).
- Metrics files (`artifacts/backtest*.json`, `metrics.json`, `causality_report.json`).
- Screenshots/logs of alerts and `docker stats` / Prometheus dashboards.

Store these artifacts alongside your runbooks to build an audit trail proving the platform meets its trust & reliability requirements.

## Multi-Tenancy Guardrails
To keep customer data isolated, every span persisted to Neo4j now carries a `tenant_id`. Set `DEFAULT_TENANT_ID` on the ingestion service (or embed `tenant_id`/`org_id` in OTLP resource attributes) so new spans are tagged automatically. Downstream apps and retraining jobs read `TENANT_ID` (falling back to `DEFAULT_TENANT_ID`) to ensure queries like `MATCH (p:Span)-[:CALLS]->(c:Span)` only see spans for the active tenant. Configure this per environment so Company A never sees Company B’s traces.

## Data Hygiene & Reporting
- **History pruning:** Run `python scripts/prune_history.py --ttl-days 30` on a schedule to `DETACH DELETE` spans older than your retention window. This keeps Neo4j lean and avoids runaway disk usage.
- **User roles:** The Streamlit dashboard now requires role selection (Admin vs Viewer). Provide `ADMIN_PASSWORD` and `VIEWER_PASSWORD` env vars so only admins can retrain or change settings; viewers get read-only access.
- **Exportable reports:** Install `fpdf` and use the “Download Weekly Health Report (PDF)” button to hand executives a snapshot containing health %, anomalies, prediction error, and top incidents.
