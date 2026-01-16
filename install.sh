#!/bin/bash
set -eo pipefail

if ! command -v docker >/dev/null 2>&1; then
  echo "Error: Docker is not installed or not on PATH." >&2
  exit 1
fi

if ! command -v docker compose >/dev/null 2>&1 && ! command -v docker-compose >/dev/null 2>&1; then
  echo "Error: Docker Compose is not installed." >&2
  exit 1
fi

ROOT_DIR="sentrinode"
ENV_FILE="$ROOT_DIR/.env"
mkdir -p "$ROOT_DIR"

API_KEY="${1:-}"

if [ -f "$ENV_FILE" ]; then
  echo "Existing provisioning detected. Using credentials from $ENV_FILE"
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
  if [ -z "$NEO4J_URI" ] || [ -z "$NEO4J_PASSWORD" ]; then
    echo "Error: $ENV_FILE is missing NEO4J_URI or NEO4J_PASSWORD values." >&2
    exit 1
  fi
else
  if [ -z "$API_KEY" ]; then
    echo "Usage: bash install.sh <NEO4J_ADMIN_PASSWORD>" >&2
    exit 1
  fi
  read -r -p "Enter Railway Neo4j Bolt URI (e.g. bolt://...): " INPUT_URI
  if [ -z "$INPUT_URI" ]; then
    echo "Error: Neo4j URI is required." >&2
    exit 1
  fi
  read -r -p "Enter Neo4j HTTP endpoint (e.g. https://...): " INPUT_HTTP
  if [ -z "$INPUT_HTTP" ]; then
    echo "Error: Neo4j HTTP endpoint is required." >&2
    exit 1
  fi

  SERIAL_NUMBER="$(system_profiler SPHardwareDataType 2>/dev/null | awk -F': ' '/Serial Number/{print $2; exit}')"
  if [ -z "$SERIAL_NUMBER" ]; then
    SERIAL_NUMBER="$(uuidgen)"
  fi

  NEO4J_USER=${NEO4J_USER:-neo4j}
  NEO4J_HTTP="${INPUT_HTTP%/}"

  echo "Registering device serial: $SERIAL_NUMBER"
  status_payload=$(cat <<JSON
{"statements":[{"statement":"MATCH (l:License {serial:$serial}) RETURN l.status AS status","parameters":{"serial":"$SERIAL_NUMBER"}}]}
JSON
)

  status_response=$(curl -sS -u "$NEO4J_USER:$API_KEY" -H "Content-Type: application/json" -d "$status_payload" "$NEO4J_HTTP/db/neo4j/tx/commit") || {
    echo "Error: Unable to contact Neo4j HTTP endpoint." >&2
    exit 1
  }
  current_status=$(python3 - <<'PY'
import json,sys
try:
    data=json.load(sys.stdin)
    rows=data["results"][0]["data"]
    if rows:
        print(rows[0]["row"][0] or "")
except Exception:
    pass
PY
<<<"$status_response")

  if [ -z "$current_status" ]; then
    create_payload=$(cat <<JSON
{"statements":[{"statement":"MERGE (l:License {serial:$serial}) ON CREATE SET l.status='active', l.type='trial', l.created_at=timestamp() SET l.updated=timestamp(), l.last_seen=timestamp() RETURN l.status","parameters":{"serial":"$SERIAL_NUMBER"}}]}
JSON
)
    create_resp=$(curl -sS -u "$NEO4J_USER:$API_KEY" -H "Content-Type: application/json" -d "$create_payload" "$NEO4J_HTTP/db/neo4j/tx/commit") || {
      echo "Error: Unable to create license in Neo4j." >&2
      exit 1
    }
    if ! echo "$create_resp" | grep -q '"row"'; then
      echo "Error: Failed to register license in Neo4j." >&2
      exit 1
    fi
    echo "Created trial license node."
  else
    echo "Existing license detected with status: $current_status"
  fi

  cat > "$ENV_FILE" <<EOF
NEO4J_URI=$INPUT_URI
NEO4J_HTTP=$NEO4J_HTTP
NEO4J_USER=$NEO4J_USER
NEO4J_PASSWORD=$SERIAL_NUMBER
LICENSE_SERIAL=$SERIAL_NUMBER
EOF
  chmod 600 "$ENV_FILE"
  export NEO4J_URI="$INPUT_URI"
  export NEO4J_HTTP="$NEO4J_HTTP"
  export NEO4J_PASSWORD="$SERIAL_NUMBER"
fi

export NEO4J_URI
export NEO4J_PASSWORD
export NEO4J_HTTP
export NEO4J_USER

mkdir -p "$ROOT_DIR/dashboard"
cd "$ROOT_DIR"

cat <<'EOF' > dashboard/app.py
#!/usr/bin/env python3
from __future__ import annotations

import os
import pandas as pd
import streamlit as st
from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError, ServiceUnavailable

st.set_page_config(page_title="SentriNode Console", layout="wide")

NEO4J_URI = (os.getenv("NEO4J_URI", "bolt://localhost:7687").strip().rstrip("/"))
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

def connect_neo4j():
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        driver.verify_connectivity()
        driver.close()
        return True, "Connected to Neo4j"
    except ServiceUnavailable:
        return False, "Neo4j host unreachable"
    except Neo4jError as exc:
        return False, f"Neo4j error: {exc}"
    except Exception as exc:
        return False, f"Unexpected error: {exc}"

connected, status = connect_neo4j()

st.title("SentriNode Console")
st.sidebar.header("Connection Status")
icon = "🟢" if connected else "🔴"
st.sidebar.write(f"{icon} {status}")
st.sidebar.write(f"URI: `{NEO4J_URI}`")

st.write("### Topology (Placeholder)")
placeholder = pd.DataFrame(
    [
        {"service": "gateway", "dependency": "payments", "latency_ms": 180},
        {"service": "payments", "dependency": "db-writer", "latency_ms": 240},
    ]
)
st.dataframe(placeholder, width="stretch")
EOF

cat <<'EOF' > dashboard/Dockerfile
FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir streamlit neo4j pandas
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
EOF

cat <<'EOF' > docker-compose.yaml
version: "3.9"
services:
  sentrinode-console:
    build: ./dashboard
    ports:
      - "8501:8501"
    environment:
      NEO4J_URI: ${NEO4J_URI}
      NEO4J_PASSWORD: ${NEO4J_PASSWORD}
EOF

if command -v docker compose >/dev/null 2>&1; then
  docker compose up --build -d
else
  docker-compose up --build -d
fi

echo "SentriNode Console is running."
echo "Access it at http://localhost:8501"
