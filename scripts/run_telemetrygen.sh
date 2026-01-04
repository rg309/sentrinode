#!/usr/bin/env bash
set -euo pipefail

BIN="${HOME}/go/bin/telemetrygen"

if [[ ! -x "${BIN}" ]]; then
  echo "telemetrygen binary not found at ${BIN}. Run 'go install ./cmd/telemetrygen' first." >&2
  exit 1
fi

exec "${BIN}" "$@"
