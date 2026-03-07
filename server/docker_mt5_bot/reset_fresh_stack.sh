#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

INSTANCE_NAME="${1:-user_a}"
PROFILE="${2:-eurusd_h1}"
SKIP_START="${SKIP_START:-0}"
BOT_API_PORT="${BOT_API_PORT:-8080}"

compose() {
  if docker compose version >/dev/null 2>&1; then
    docker compose "$@"
    return
  fi

  if command -v docker-compose >/dev/null 2>&1; then
    docker-compose "$@"
    return
  fi

  echo "Error: docker compose not found"
  exit 1
}

echo "[1/7] Stopping API on port ${BOT_API_PORT} (if running)..."
api_pid="$(lsof -nP -iTCP:${BOT_API_PORT} -sTCP:LISTEN -Fp 2>/dev/null | sed 's/^p//' | head -n1 || true)"
if [[ -n "$api_pid" ]]; then
  kill "$api_pid" || true
  sleep 1
fi

echo "[2/7] Stopping and removing all mt5_* projects..."
project_names="$(
  docker ps -a --format '{{.Names}}' \
    | sed -n 's/-smarfrobot-mt5-1$//p' \
    | rg '^mt5_' \
    | sort -u || true
)"
if [[ -n "$project_names" ]]; then
  while IFS= read -r project; do
    [[ -z "$project" ]] && continue
    echo "  - docker compose -p ${project} down -v --remove-orphans"
    docker compose -p "$project" down -v --remove-orphans || true
  done <<< "$project_names"
fi

echo "[3/7] Removing old mt5 volumes..."
mt5_volumes="$(docker volume ls --format '{{.Name}}' | rg '^mt5_.*_config$|^mt5_pydeps_shared$' || true)"
if [[ -n "$mt5_volumes" ]]; then
  while IFS= read -r vol; do
    [[ -z "$vol" ]] && continue
    docker volume rm -f "$vol" || true
  done <<< "$mt5_volumes"
else
  echo "  - no mt5 volumes found"
fi

echo "[4/7] Removing old MT5 images..."
mt5_images="$(docker images --format '{{.Repository}}:{{.Tag}}' | rg '^smarfrobot_mt5:|^unknowkubbrother/smarfrobot_mt5:' || true)"
if [[ -n "$mt5_images" ]]; then
  while IFS= read -r img; do
    [[ -z "$img" ]] && continue
    docker rmi -f "$img" || true
  done <<< "$mt5_images"
else
  echo "  - no MT5 images found"
fi

echo "[5/7] Cleaning local runtime state..."
rm -rf "$ROOT_DIR/.instances" "$ROOT_DIR/logs"
mkdir -p "$ROOT_DIR/.instances" "$ROOT_DIR/logs"

echo "[6/7] Building new MT5 image..."
compose build smarfrobot-mt5

if [[ "$SKIP_START" =~ ^(1|true|yes|on)$ ]]; then
  echo "[7/7] SKIP_START enabled, skipping instance start."
  echo "Done. To start manually:"
  echo "  AUTO_BUILD=0 ./run_instance.sh ${INSTANCE_NAME} ${PROFILE}"
  exit 0
fi

echo "[7/7] Starting fresh instance..."
echo "  instance=${INSTANCE_NAME} profile=${PROFILE}"
AUTO_BUILD=0 ./run_instance.sh "$INSTANCE_NAME" "$PROFILE"
