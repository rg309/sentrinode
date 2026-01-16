#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

COMPOSE_FILE="docker-compose.yml"
ENV_FILE=".env"

if [[ ! -f "$COMPOSE_FILE" ]]; then
  echo "Error: docker-compose.yml was not found in $SCRIPT_DIR" >&2
  exit 1
fi

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Error: $1 is not installed or not on PATH." >&2
    exit 1
  fi
}

require_command docker

if ! docker info >/dev/null 2>&1; then
  echo "Error: Docker daemon is not running. Start Docker Desktop or dockerd and re-run install.sh." >&2
  exit 1
fi

if docker compose version >/dev/null 2>&1; then
  COMPOSE_BIN="docker compose"
elif command -v docker-compose >/dev/null 2>&1; then
  COMPOSE_BIN="docker-compose"
else
  echo "Error: Docker Compose v1 or v2 is required but was not found." >&2
  exit 1
fi

if [[ -f "$ENV_FILE" ]]; then
  echo "Detected existing $ENV_FILE. Loading current values..."
  # shellcheck disable=SC1090
  set -a && source "$ENV_FILE" && set +a
fi

detect_serial() {
  local serial=""
  if command -v system_profiler >/dev/null 2>&1; then
    serial="$(system_profiler SPHardwareDataType 2>/dev/null | awk -F': ' '/Serial Number/{print $2; exit}')"
  fi
  if [[ -z "$serial" ]] && command -v ioreg >/dev/null 2>&1; then
    serial="$(ioreg -l | awk -F'"' '/IOPlatformSerialNumber/{print $4; exit}' 2>/dev/null)"
  fi
  if [[ -z "$serial" ]]; then
    if command -v uuidgen >/dev/null 2>&1; then
      serial="$(uuidgen)"
    else
      serial="$(python3 - <<'PY'
import uuid
print(uuid.uuid4())
PY
)"
    fi
    echo "Warning: Could not determine hardware serial. Generated identifier: $serial" >&2
  else
    echo "Detected hardware serial: $serial"
  fi
  printf "%s" "$serial"
}

prompt_with_default() {
  local prompt="$1"
  local default="$2"
  local response
  read -r -p "$prompt [$default]: " response
  if [[ -z "$response" ]]; then
    response="$default"
  fi
  printf "%s" "$response"
}

NEO4J_URI="${NEO4J_URI:-}"
if [[ -z "$NEO4J_URI" ]]; then
  NEO4J_URI="$(prompt_with_default "Enter Neo4j Bolt URI" "bolt://neo4j:7687")"
fi
NEO4J_URI="${NEO4J_URI%/}"
NEO4J_USER="${NEO4J_USER:-neo4j}"
SERIAL_NUMBER="$(detect_serial)"
NEO4J_PASSWORD="${NEO4J_PASSWORD:-$SERIAL_NUMBER}"
LICENSE_SERIAL="${LICENSE_SERIAL:-$NEO4J_PASSWORD}"

python3 - "$ENV_FILE" \
  "NEO4J_URI=$NEO4J_URI" \
  "NEO4J_USER=$NEO4J_USER" \
  "NEO4J_PASSWORD=$NEO4J_PASSWORD" \
  "LICENSE_SERIAL=$LICENSE_SERIAL" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
pairs = [arg.split("=", 1) for arg in sys.argv[2:]]
updates = {key: value for key, value in pairs}

if path.exists():
    lines = path.read_text().splitlines()
else:
    lines = []

seen = set()
for idx, line in enumerate(lines):
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or "=" not in line:
        continue
    key = line.split("=", 1)[0]
    if key in updates:
        lines[idx] = f"{key}={updates[key]}"
        seen.add(key)

for key, value in updates.items():
    if key not in seen:
        lines.append(f"{key}={value}")

content = "\n".join(lines).strip()
if content:
    path.write_text(content + "\n")
else:
    path.write_text("")
PY

chmod 600 "$ENV_FILE"
set -a && source "$ENV_FILE" && set +a

echo "Launching SentriNode stack with $COMPOSE_BIN ..."
$COMPOSE_BIN up -d --build
$COMPOSE_BIN ps

echo
echo "Provisioning complete. Streamlit UI: http://localhost:8501"
