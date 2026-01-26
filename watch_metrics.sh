#!/usr/bin/env bash
set -u

METRICS_URL="${METRICS_URL:-https://sentrinode-api.fly.dev/metrics}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-2}"

while true; do
  ts="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  body=$(curl -s "$METRICS_URL" || true)
  calls=$(printf "%s\n" "$body" | awk '$1=="spanmetrics_calls_total"{print $2; exit}')
  count=$(printf "%s\n" "$body" | awk '$1=="spanmetrics_duration_count"{print $2; exit}')
  sum=$(printf "%s\n" "$body" | awk '$1=="spanmetrics_duration_sum"{print $2; exit}')
  calls=${calls:-0}
  count=${count:-0}
  sum=${sum:-0}
  printf "%s calls=%s count=%s sum=%s\n" "$ts" "$calls" "$count" "$sum"
  sleep "$INTERVAL_SECONDS"
done
