#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

SERVICE_NAME="${SERVICE_NAME:-smarfrobot-mt5}"
OUTPUT_PATH="${1:-snapshots/mt5-config-snapshot.tgz}"
OUTPUT_DIR="$(dirname "$OUTPUT_PATH")"
OUTPUT_FILE="$(basename "$OUTPUT_PATH")"
OUTPUT_ABS_DIR="$(cd "$OUTPUT_DIR" 2>/dev/null || { mkdir -p "$OUTPUT_DIR" && cd "$OUTPUT_DIR"; } && pwd)"
OUTPUT_ABS_PATH="$OUTPUT_ABS_DIR/$OUTPUT_FILE"
TMP_OUTPUT_PATH="${OUTPUT_ABS_PATH}.tmp"

compose() {
  if docker compose version >/dev/null 2>&1; then
    docker compose "$@"
    return
  fi

  if command -v docker-compose >/dev/null 2>&1; then
    docker-compose "$@"
    return
  fi

  echo "Error: ไม่พบ docker compose หรือ docker-compose"
  exit 1
}

container_id="$(compose ps -a -q "$SERVICE_NAME" | head -n1)"
if [[ -z "$container_id" ]]; then
  echo "Error: container for service '$SERVICE_NAME' not found"
  echo "Run once first: docker compose up -d"
  exit 1
fi

was_running="false"
if [[ -n "$(compose ps --status running -q "$SERVICE_NAME" | head -n1)" ]]; then
  was_running="true"
fi

restore_service() {
  if [[ "$was_running" == "true" ]]; then
    echo "Starting service again..."
    compose up -d "$SERVICE_NAME" >/dev/null || true
  fi
}
trap restore_service EXIT

if [[ "$was_running" == "true" ]]; then
  echo "Stopping service temporarily for a consistent snapshot..."
  compose stop "$SERVICE_NAME" >/dev/null
fi

echo "Creating snapshot from config volume..."
rm -f "$TMP_OUTPUT_PATH"
if ! compose run --rm --no-deps --entrypoint bash "$SERVICE_NAME" -lc "
set -euo pipefail
if [ ! -d /config/.wine ]; then
  echo 'Error: /config/.wine not found (MT5 may not be installed yet)' >&2
  exit 1
fi
cd /config
tar --warning=no-file-changed -czf - .wine
" > "$TMP_OUTPUT_PATH"; then
  rm -f "$TMP_OUTPUT_PATH"
  exit 1
fi

mv "$TMP_OUTPUT_PATH" "$OUTPUT_ABS_PATH"
echo "Done. Snapshot saved to: $OUTPUT_ABS_PATH"
