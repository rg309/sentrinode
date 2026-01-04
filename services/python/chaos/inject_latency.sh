#!/usr/bin/env bash
set -euo pipefail
CONTAINER=${1:-svc-payments}
IFACE=${2:-eth0}
DELAY=${3:-100ms}
JITTER=${4:-20ms}

echo "Injecting latency on $CONTAINER $IFACE: delay=$DELAY jitter=$JITTER"
docker exec "$CONTAINER" tc qdisc add dev "$IFACE" root netem delay "$DELAY" "$JITTER" distribution normal
