#!/usr/bin/env bash
set -euo pipefail

REPO_OWNER="rg309"
REPO_NAME="sentrinode"
BRANCH="main"

TARGET_DIR="${1:-sentrinode}"
ARCHIVE_URL="https://github.com/${REPO_OWNER}/${REPO_NAME}/archive/refs/heads/${BRANCH}.tar.gz"

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Error: $1 is not installed or not on PATH." >&2
    exit 1
  fi
}

require_command curl
require_command tar
require_command docker

if ! docker info >/dev/null 2>&1; then
  echo "Error: Docker daemon is not running. Start Docker Desktop and re-run." >&2
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

mkdir -p "$TARGET_DIR"
cd "$TARGET_DIR"

TMP_DIR=".sentrinode_tmp_$$"
rm -rf "$TMP_DIR"
mkdir -p "$TMP_DIR"

echo "Downloading ${REPO_OWNER}/${REPO_NAME}@${BRANCH} ..."
curl -fsSL "$ARCHIVE_URL" | tar -xz -C "$TMP_DIR"

ROOT_DIR="$(find "$TMP_DIR" -maxdepth 1 -type d -name "${REPO_NAME}-*" | head -n 1)"
if [[ -z "${ROOT_DIR:-}" ]]; then
  echo "Error: Could not find extracted repo folder." >&2
  exit 1
fi

# Copy repo contents into target folder
cp -R "$ROOT_DIR"/. .

rm -rf "$TMP_DIR"

COMPOSE_FILE="docker-compose.yml"
ENV_FILE=".env"

if [[ ! -f "$COMPOSE_FILE" ]]; then
  echo "Error: docker-compose.yml was not found after download." >&2
  echo "Make sure docker-compose.yml exists in the repo root." >&2
  exit 1
fi

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

# Defaults for local docker compose networking
NEO4J_URI="${NEO4J_URI:-}"
if [[ -z "$NEO4J_URI" ]]; then
  NEO4J_URI="$(prompt_with_default "Enter Neo4j Bolt URI" "bolt://neo4j:7687")"
fi
NEO4J_URI="${NEO4J_URI%/}"
NEO4J_USER="${NEO4J_USER:-neo4j}"

# Generate a default password if not supplied
if [[ -z "${NEO4J_PASSWORD:-}" ]]; then
  if command -v uuidgen >/dev/null 2>&1; then
    NEO4J_PASSWORD="$(uuidgen)"
  else
    NEO4J_PASSWORD="$(python3 - <<'PY'
import uuid
print(uuid.uuid4())
PY
)"
  fi
  echo "Generated NEO4J_PASSWORD (saved to .env)."
fi

LICENSE_SERIAL="${LICENSE_SERIAL:-$NEO4J_PASSWORD}"

cat > "$ENV_FILE" <<EOF
NEO4J_URI=$NEO4J_URI
NEO4J_USER=$NEO4J_USER
NEO4J_PASSWORD=$NEO4J_PASSWORD
LICENSE_SERIAL=$LICENSE_SERIAL
EOF

chmod 600 "$ENV_FILE"

echo "Launching SentriNode stack with $COMPOSE_BIN ..."
$COMPOSE_BIN up -d --build
$COMPOSE_BIN ps

echo
echo "Provisioning complete."
echo "Streamlit UI: http://localhost:8501"
