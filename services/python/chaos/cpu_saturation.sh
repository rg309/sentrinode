#!/usr/bin/env bash
set -euo pipefail
CONTAINER=${1:-svc-inventory}
DURATION=${2:-120}
CPUS=${3:-2}

echo "Running stress-ng inside $CONTAINER for $DURATION seconds"
docker exec "$CONTAINER" stress-ng --cpu "$CPUS" --timeout "${DURATION}s" --metrics-brief
