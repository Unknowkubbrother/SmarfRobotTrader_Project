#!/usr/bin/env bash
set -euo pipefail

cd /app

SCHEMA_PATH="${PRISMA_SCHEMA_PATH:-src/database/schema.prisma}"
WAIT_TIMEOUT_SECONDS="${WAIT_TIMEOUT_SECONDS:-90}"
PORT="${PORT:-8000}"
UVICORN_WORKERS="${UVICORN_WORKERS:-1}"

is_truthy() {
  local raw
  raw="$(printf '%s' "${1:-}" | tr '[:upper:]' '[:lower:]')"
  case "$raw" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

wait_for_tcp() {
  local host="$1"
  local port="$2"
  local name="$3"
  local timeout="$4"

  python3 - "$host" "$port" "$name" "$timeout" <<'PY'
import socket
import sys
import time

host = sys.argv[1]
port = int(sys.argv[2])
name = sys.argv[3]
timeout = int(sys.argv[4])

deadline = time.time() + timeout
while time.time() < deadline:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(1.5)
    try:
        if s.connect_ex((host, port)) == 0:
            print(f"{name} is reachable at {host}:{port}")
            sys.exit(0)
    finally:
        s.close()
    time.sleep(1)

print(f"Timeout waiting for {name} at {host}:{port}", file=sys.stderr)
sys.exit(1)
PY
}

if [[ -z "${DATABASE_URL:-}" ]]; then
  echo "Error: DATABASE_URL is required."
  exit 1
fi

db_host="$(
  python3 - <<'PY'
from urllib.parse import urlparse
import os

url = os.getenv("DATABASE_URL", "")
parsed = urlparse(url)
print(parsed.hostname or "postgres")
PY
)"
db_port="$(
  python3 - <<'PY'
from urllib.parse import urlparse
import os

url = os.getenv("DATABASE_URL", "")
parsed = urlparse(url)
print(parsed.port or 5432)
PY
)"
redis_host="${REDIS_HOST:-redis}"
redis_port="${REDIS_PORT:-6379}"

wait_for_tcp "$db_host" "$db_port" "postgres" "$WAIT_TIMEOUT_SECONDS"
wait_for_tcp "$redis_host" "$redis_port" "redis" "$WAIT_TIMEOUT_SECONDS"

if is_truthy "${PRISMA_GENERATE_ON_START:-1}"; then
  echo "[startup] prisma generate"
  python -m prisma generate --schema="$SCHEMA_PATH"
fi

if is_truthy "${PRISMA_DB_PUSH_ON_START:-0}"; then
  echo "[startup] ensure postgres extension uuid-ossp"
  echo 'CREATE EXTENSION IF NOT EXISTS "uuid-ossp";' \
    | python -m prisma db execute --stdin --schema="$SCHEMA_PATH"

  echo "[startup] prisma db push"
  db_push_args=(--schema="$SCHEMA_PATH" --skip-generate)
  if is_truthy "${PRISMA_DB_PUSH_ACCEPT_DATA_LOSS:-0}"; then
    echo "[startup] prisma db push will accept data-loss warnings"
    db_push_args+=(--accept-data-loss)
  fi
  python -m prisma db push "${db_push_args[@]}"
fi

echo "[startup] starting uvicorn on 0.0.0.0:${PORT} workers=${UVICORN_WORKERS}"
exec uvicorn src.main:app --host 0.0.0.0 --port "$PORT" --workers "$UVICORN_WORKERS"
