#!/usr/bin/env bash
set -euo pipefail

mode="${1:-}" || true

if [[ -z "$mode" ]]; then
  echo "Usage: $0 --healthy|--slow-db"
  exit 1
fi

case "$mode" in
  --healthy)
    echo "Running healthy workload..."
    /usr/local/bin/telemetrygen traces --traces 200 --concurrency 4 --otlp-endpoint "127.0.0.1:4317"
    ;;
  --slow-db)
    echo "Running slow DB workload (500ms spans)..."
    /usr/local/bin/telemetrygen traces --traces 200 --concurrency 2 --otlp-endpoint "127.0.0.1:4317" --span-duration 500ms
    ;;
  *)
    echo "Unknown mode: $mode"
    exit 1
    ;;
 esac
