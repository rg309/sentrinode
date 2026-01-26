# sentrinode

## Getting Started
1. Clone this repo.
2. Install dependencies (`pip install -r requirements.txt` for Python tooling, `go mod tidy` under `services/go` for Go services).
3. Use `docker-compose up` to start the stack locally when needed.

### Fly deploy

```bash
fly deploy -a sentrinode -c fly.sentrinode.toml
fly deploy -a sentrinode-api -c fly.sentrinode-api.toml
```


### Hosted Console

UI: `https://sentrinode.fly.dev`

API: `https://sentrinode-api.fly.dev`

### Metrics source (optional)

Set `PIPELINE_METRICS_URL` to a Prometheus-compatible `/metrics` endpoint to enable live pipeline charts. If unset, the UI skips polling and shows that the metrics source is not configured.

Example:

```bash
PIPELINE_METRICS_URL=http://localhost:9464/metrics
```

### Initial Dashboard Login

The Streamlit UI now boots in a "setup" mode. Provide one of the bootstrap keys listed in the `AUTHORIZED_KEYS` environment variable (or `[bootstrap] authorized_keys` inside `.streamlit/secrets.toml`) to unlock the credential wizard. If `SENTRINODE_ADMIN_*` / `SENTRINODE_VIEWER_*` environment variables are defined (see `.env`), the dashboard immediately exposes the username/password login form so you can retain centralized credential management. Otherwise, use the bootstrap wizard once to create credentials; they will be persisted to `.streamlit/secrets.toml`. Keep both `.env` and `.streamlit/secrets.toml` out of source control.

### Build Go services

Binaries are not checked in. Build them locally before running any Go-based services:

```bash
cd services/go
go build -o bin/graph-aggregator ./cmd/graph-aggregator
go build -o bin/server ./cmd/server
```

### Neo4j data requirement

The `/neo4j` and `/neo4j_data` directories are intentionally ignored so their large database files stay out of Git. After cloning, you must initialize Neo4j locally before running any graph services:

1. Run `docker-compose up neo4j`. The compose file will create the `neo4j_data/` directory structure automatically.
2. (Optional) If you have seed data, copy it from `docs/neo4j-seed/` into `neo4j_data/` before starting the rest of the stack.
3. Keep these directories on your machine; they should not be committed.
