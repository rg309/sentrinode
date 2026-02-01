#!/usr/bin/env bash
set -euo pipefail

URL="${1:-https://sentrinode.fly.dev/}"

tmp_dir="$(mktemp -d)"
cleanup() {
  rm -rf "$tmp_dir"
}
trap cleanup EXIT

headers="${tmp_dir}/headers.txt"
body="${tmp_dir}/body.txt"

curl -s -D "$headers" -o "$body" "$URL"

status_line="$(head -n 1 "$headers" | tr -d '\r')"
status_code="$(echo "$status_line" | awk '{print $2}')"
content_type="$(grep -i '^content-type:' "$headers" | tail -n 1 | tr -d '\r')"

if echo "$content_type" | grep -qi 'application/json'; then
  echo "FAIL: ${URL} returned JSON content-type (${content_type})." >&2
  exit 1
fi

if grep -q '{"detail":"Not Found"}' "$body"; then
  echo "FAIL: ${URL} returned FastAPI JSON 404 body." >&2
  exit 1
fi

if [[ "$status_code" =~ ^3 ]]; then
  echo "OK: ${URL} returned redirect (${status_line})."
  exit 0
fi

if echo "$content_type" | grep -qi 'text/html'; then
  echo "OK: ${URL} returned HTML (${status_line})."
  exit 0
fi

echo "FAIL: ${URL} returned unexpected content-type (${content_type})." >&2
exit 1
