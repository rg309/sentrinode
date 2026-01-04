#!/usr/bin/env bash
set -euo pipefail
CONTAINER=${1:-svc-payments}
IFACE=${2:-eth0}

echo "Clearing latency on $CONTAINER $IFACE"
docker exec "$CONTAINER" tc qdisc del dev "$IFACE" root || true
