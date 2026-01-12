# sentrinode

## Getting Started

1. Clone this repo.
2. Install dependencies (`pip install -r requirements.txt` for Python tooling, `go mod tidy` under `services/go` for Go services).
3. Use `docker-compose up` to start the stack locally when needed.

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
