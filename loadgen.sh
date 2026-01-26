#!/usr/bin/env bash
set -u

INGEST_URL="${INGEST_URL:-https://sentrinode-api.fly.dev/ingest}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-1}"

while true; do
  ts="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  duration_ms=$((RANDOM % 901 + 50))
  payload=$(printf '{"source":"loadgen","event":"ping","ts":"%s","duration_ms":%d}' "$ts" "$duration_ms")
  status=$(curl -s -o /dev/null -w "%{http_code}" \
    -H "Content-Type: application/json" \
    -d "$payload" \
    "$INGEST_URL" || echo "000")
  printf "%s duration_ms=%d status=%s\n" "$ts" "$duration_ms" "$status"
  sleep "$INTERVAL_SECONDS"
done
