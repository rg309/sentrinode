#!/bin/sh
set -e

STREAMLIT_PORT="${PORT:-8080}"
COLLECTOR_PORT="${COLLECTOR_PORT:-9000}"

uvicorn collector_app:app --host 0.0.0.0 --port "$COLLECTOR_PORT" &

exec streamlit run /app/app.py --server.address=0.0.0.0 --server.port="$STREAMLIT_PORT"
