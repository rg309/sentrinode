#!/usr/bin/env bash
set -euo pipefail

URL="${1:-https://sentrinode.fly.dev/}"

output="$(curl -s -D- "$URL")"

# Print the first 20 lines for quick inspection.
echo "$output" | head -n 20

if echo "$output" | grep -qi '^content-type:.*application/json'; then
  echo "FAIL: ${URL} returned JSON content-type." >&2
  exit 1
fi

if echo "$output" | grep -q '{"detail":"Not Found"}'; then
  echo "FAIL: ${URL} returned FastAPI JSON 404 body." >&2
  exit 1
fi

echo "OK: ${URL} did not return JSON 404."
