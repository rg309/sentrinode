# SentriNode Ingest Service

Lightweight FastAPI service that accepts OTLP HTTP payloads from the collector, validates multi-tenant headers against Supabase, and stores only the latest per-node state in memory.

## Endpoints
- `GET /healthz` – liveness probe.
- `POST /v1/otlp/traces` – OTLP JSON traces (with `X-Tenant-Id` and `X-SentriNode-Key` headers).
- `POST /v1/otlp/metrics` – OTLP JSON metrics (with auth headers).
- `POST /v1/heartbeat` – `{ node_name, ts_unix, attributes }` heartbeat (with auth headers).
- `GET /v1/tenants/{tenant_slug}/nodes` – latest nodes for the tenant (with auth headers).
- `GET /v1/tenants/{tenant_slug}/nodes/{node_name}` – latest state for a node (with auth headers).

## Local testing
1) Start stack:
   ```bash
   docker compose up -d --build
   ```
2) Health check:
   ```bash
   curl http://localhost:8000/healthz
   ```
3) Heartbeat injection (replace values):
   ```bash
   curl -X POST http://localhost:8000/v1/heartbeat \
     -H "Content-Type: application/json" \
     -H "X-Tenant-Id: <tenant_slug>" \
     -H "X-SentriNode-Key: <raw_api_key>" \
     -d '{"node_name":"edge-1","ts_unix":'$(date +%s)'}'
   ```
4) OTLP trace JSON through local collector:
   ```bash
   curl -X POST http://localhost:4318/v1/traces \
     -H "Content-Type: application/json" \
     -H "X-Tenant-Id: <tenant_slug>" \
     -H "X-SentriNode-Key: <raw_api_key>" \
     -d '{"resourceSpans":[{"resource":{"attributes":[{"key":"service.name","value":{"stringValue":"edge-1"}}]},"scopeSpans":[]}]}'
   ```
   Confirm the collector forwards to ingest (view via `GET /v1/tenants/<tenant>/nodes`).
5) Dashboard checks:
   - Login once; it should route immediately.
   - Node Manager should refresh every 2 seconds and pull from ingest.
