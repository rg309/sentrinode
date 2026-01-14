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

API_KEY="${1:-}"
if [ -z "$API_KEY" ]; then
  echo "Usage: bash install.sh <NEO4J_API_KEY>" >&2
  exit 1
fi

mkdir -p sentrinode/dashboard
cd sentrinode

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

read -r -p "Enter Railway Neo4j URI: " INPUT_URI
NEO4J_URI="$INPUT_URI"
NEO4J_PASSWORD="$API_KEY"
export NEO4J_URI
export NEO4J_PASSWORD

if command -v docker compose >/dev/null 2>&1; then
  docker compose up --build -d
else
  docker-compose up --build -d
fi

echo "SentriNode Console is running."
echo "Access it at http://localhost:8501"
