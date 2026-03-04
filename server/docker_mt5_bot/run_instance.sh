#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

STATE_DIR="$ROOT_DIR/.instances"
mkdir -p "$STATE_DIR"

CMD_RAW="${1:-}"
ARG1="${2:-}"
ARG2="${3:-}"
COMMAND="start"
INSTANCE_RAW=""
PROFILE=""

migrate_legacy_instance_env_files() {
  shopt -s nullglob
  local legacy_file legacy_name target_dir target_file
  for legacy_file in "$STATE_DIR"/*.env; do
    [[ -f "$legacy_file" ]] || continue
    legacy_name="$(basename "$legacy_file" .env)"
    [[ -n "$legacy_name" ]] || continue
    target_dir="$STATE_DIR/$legacy_name"
    target_file="$target_dir/.env"
    mkdir -p "$target_dir"
    if [[ ! -f "$target_file" ]]; then
      mv "$legacy_file" "$target_file"
    fi
  done
}

usage() {
  cat <<'EOF'
Usage:
  ./run_instance.sh start
  ./run_instance.sh start <profile>
  ./run_instance.sh start <instance_name> <profile>
  ./run_instance.sh restart
  ./run_instance.sh restart <profile>
  ./run_instance.sh restart <instance_name> <profile>
  ./run_instance.sh stop
  ./run_instance.sh stop <instance_name>
  ./run_instance.sh
  ./run_instance.sh <profile>
  ./run_instance.sh <instance_name> <profile>
  ./run_instance.sh list

Example:
  ./run_instance.sh start
  ./run_instance.sh restart
  ./run_instance.sh stop
  ./run_instance.sh
  ./run_instance.sh eurusd_h1
  ./run_instance.sh 103064429 eurusd_h1

Auto mode:
  - instance_name = MT5_LOGIN (from env)
  - profile = <LIVE_SYMBOL>_<LIVE_TIMEFRAME> (lowercase), e.g. EURUSD + H1 => eurusd_h1
  - required env keys: MT5_LOGIN, MT5_PASSWORD, MT5_SERVER, LIVE_SYMBOL, LIVE_TIMEFRAME, BOT_CONFIG_ID, BOT_WS_URL, VISION_LLM_API_URL
EOF
}

read_dotenv_value() {
  local key="$1"
  local value=""

  if [[ -f ".env" ]]; then
    local line
    line="$(grep -E "^${key}=" .env | tail -n 1 || true)"
    if [[ -n "$line" ]]; then
      value="${line#*=}"
      value="${value%\"}"
      value="${value#\"}"
      value="${value%\'}"
      value="${value#\'}"
    fi
  fi

  printf '%s' "$value"
}

trim_outer_whitespace() {
  local raw="$1"
  printf '%s' "$raw" | sed -E 's/^[[:space:]]+//; s/[[:space:]]+$//'
}

sanitize_instance_name() {
  local raw="$1"
  local cleaned
  cleaned="$(printf '%s' "$raw" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/_/g; s/^_+//; s/_+$//')"
  printf '%s' "$cleaned"
}

sanitize_profile_token() {
  local raw="$1"
  local cleaned
  cleaned="$(printf '%s' "$raw" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+//g')"
  printf '%s' "$cleaned"
}

resolve_mt5_login() {
  local login
  login="${MT5_LOGIN:-$(read_dotenv_value MT5_LOGIN)}"
  login="$(printf '%s' "$login" | tr -d '[:space:]')"
  if [[ -z "$login" ]]; then
    echo "Error: MT5_LOGIN is required (set in env or .env)"
    exit 1
  fi
  printf '%s' "$login"
}

resolve_required_env() {
  local key="$1"
  local value
  value="${!key:-$(read_dotenv_value "$key")}"
  value="$(printf '%s' "$value" | tr -d '\r')"
  if [[ -z "$value" ]]; then
    echo "Error: $key is required (set in env or .env)"
    exit 1
  fi
  printf '%s' "$value"
}

validate_link_env() {
  local ws_url="$1"
  local vision_url="$2"

  if [[ ! "$ws_url" =~ ^wss?:// ]]; then
    echo "Error: BOT_WS_URL must start with ws:// or wss://"
    exit 1
  fi

  if [[ ! "$vision_url" =~ ^https?:// ]]; then
    echo "Error: VISION_LLM_API_URL must start with http:// or https://"
    exit 1
  fi
}

derive_profile_from_env() {
  local symbol_raw timeframe_raw symbol timeframe
  symbol_raw="${LIVE_SYMBOL:-$(read_dotenv_value LIVE_SYMBOL)}"
  timeframe_raw="${LIVE_TIMEFRAME:-$(read_dotenv_value LIVE_TIMEFRAME)}"

  symbol="$(sanitize_profile_token "$symbol_raw")"
  timeframe="$(sanitize_profile_token "$timeframe_raw")"

  if [[ -z "$symbol" || -z "$timeframe" ]]; then
    echo "Error: LIVE_SYMBOL and LIVE_TIMEFRAME are required (set in env or .env)"
    exit 1
  fi

  printf '%s_%s' "$symbol" "$timeframe"
}

list_available_profiles() {
  shopt -s nullglob
  local dir
  for dir in "$ROOT_DIR"/bots/*; do
    [[ -d "$dir" ]] || continue
    [[ -f "$dir/run_live.py" ]] || continue
    basename "$dir"
  done | LC_ALL=C sort
}

profile_exists() {
  local profile="$1"
  [[ -f "$ROOT_DIR/bots/$profile/run_live.py" ]]
}

resolve_bot_defaults_from_profile() {
  local profile="$1"
  local profile_dir="$ROOT_DIR/bots/$profile"
  local req_live req_base
  req_live="/bots/$profile/requirements-live.txt"
  req_base="/bots/$profile/requirements.txt"

  if [[ ! -f "$profile_dir/run_live.py" ]]; then
    echo "Error: bot profile '$profile' not found at $profile_dir/run_live.py"
    echo "Available profiles:"
    list_available_profiles | sed 's/^/  /'
    exit 1
  fi

  BOT_SCRIPT_DEFAULT="$profile/run_live.py"
  if [[ -f "$profile_dir/requirements-live.txt" ]]; then
    BOT_REQUIREMENTS_DEFAULT="$req_live"
  elif [[ -f "$profile_dir/requirements.txt" ]]; then
    BOT_REQUIREMENTS_DEFAULT="$req_base"
  else
    echo "Error: requirements file not found for profile '$profile'"
    echo "  expected one of:"
    echo "    $profile_dir/requirements-live.txt"
    echo "    $profile_dir/requirements.txt"
    exit 1
  fi
}

port_is_free() {
  local port="$1"
  python3 - "$port" <<'PY'
import socket, sys
port = int(sys.argv[1])
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    s.bind(("127.0.0.1", port))
except OSError:
    sys.exit(1)
finally:
    s.close()
sys.exit(0)
PY
}

pick_free_port() {
  local preferred="$1"
  local max_tries="${2:-200}"
  local p
  for ((i=0; i<=max_tries; i++)); do
    p=$((preferred + i))
    if port_is_free "$p"; then
      printf '%s' "$p"
      return 0
    fi
  done
  return 1
}

show_list() {
  echo "Profiles:"
  local any_profiles=0
  local profiles=()
  local profile_name
  while IFS= read -r profile_name; do
    [[ -n "$profile_name" ]] || continue
    profiles+=("$profile_name")
    any_profiles=1
  done < <(list_available_profiles)
  if [[ "$any_profiles" -eq 0 ]]; then
    echo "  (none)"
  else
    printf '%s\n' "${profiles[@]}" | LC_ALL=C sort -u | sed 's/^/  /'
  fi
  echo ""
  echo "Instances:"
  shopt -s nullglob
  local any_instances=0
  local instances=()
  local f
  for f in "$STATE_DIR"/*/.env "$STATE_DIR"/*.env; do
    [[ -f "$f" ]] || continue
    if [[ "$(basename "$(dirname "$f")")" == ".instances" ]]; then
      instances+=("$(basename "$f" .env)")
    else
      instances+=("$(basename "$(dirname "$f")")")
    fi
    any_instances=1
  done
  if [[ "$any_instances" -eq 0 ]]; then
    echo "  (none)"
  else
    printf '%s\n' "${instances[@]}" | LC_ALL=C sort -u | sed 's/^/  /'
  fi
}

resolve_instance_and_profile() {
  # Mode 1: no args -> derive both from env.
  if [[ -z "$INSTANCE_RAW" ]]; then
    INSTANCE_RAW="$(resolve_mt5_login)"
    PROFILE="$(derive_profile_from_env)"
  elif [[ -z "$PROFILE" ]]; then
    # Mode 2: one arg.
    # 2a) if arg matches an existing profile, derive instance from MT5_LOGIN.
    # 2b) otherwise treat arg as instance name and derive profile from env.
    if profile_exists "$INSTANCE_RAW"; then
      PROFILE="$INSTANCE_RAW"
      INSTANCE_RAW="$(resolve_mt5_login)"
    else
      PROFILE="$(derive_profile_from_env)"
    fi
  fi

  INSTANCE_NAME="$(sanitize_instance_name "$INSTANCE_RAW")"
  if [[ -z "$INSTANCE_NAME" ]]; then
    echo "Error: invalid instance name '$INSTANCE_RAW'"
    exit 1
  fi

  PROFILE="$(sanitize_instance_name "$PROFILE")"
  if [[ -z "$PROFILE" ]]; then
    echo "Error: invalid profile"
    exit 1
  fi
}

if [[ "$CMD_RAW" == "list" ]]; then
  migrate_legacy_instance_env_files
  show_list
  exit 0
fi

if [[ "$CMD_RAW" =~ ^(start|stop|restart)$ ]]; then
  COMMAND="$CMD_RAW"
  INSTANCE_RAW="$ARG1"
  PROFILE="$ARG2"
else
  COMMAND="start"
  INSTANCE_RAW="$CMD_RAW"
  PROFILE="$ARG1"
fi

if [[ "$COMMAND" == "stop" ]]; then
  local_instance="$INSTANCE_RAW"
  if [[ -z "$local_instance" ]]; then
    local_instance="$(resolve_mt5_login)"
  fi
  instance_name="$(sanitize_instance_name "$local_instance")"
  if [[ -z "$instance_name" ]]; then
    echo "Error: invalid instance name '$local_instance'"
    exit 1
  fi
  export COMPOSE_PROJECT_NAME="mt5_${instance_name}"
  docker compose down || true
  exit 0
fi

resolve_instance_and_profile
migrate_legacy_instance_env_files

if [[ "$COMMAND" == "restart" ]]; then
  export COMPOSE_PROJECT_NAME="mt5_${INSTANCE_NAME}"
  docker compose down || true
fi

BOT_SCRIPT_DEFAULT=""
BOT_REQUIREMENTS_DEFAULT=""
resolve_bot_defaults_from_profile "$PROFILE"

MT5_LOGIN_VAL="$(resolve_required_env MT5_LOGIN)"
MT5_PASSWORD_VAL="$(resolve_required_env MT5_PASSWORD)"
MT5_SERVER_VAL="$(resolve_required_env MT5_SERVER)"
MT5_SERVER_FALLBACKS_VAL="${MT5_SERVER_FALLBACKS:-$(read_dotenv_value MT5_SERVER_FALLBACKS)}"
MT5_LOGIN_VAL="$(printf '%s' "$MT5_LOGIN_VAL" | tr -d '[:space:]')"
MT5_SERVER_VAL="$(trim_outer_whitespace "$MT5_SERVER_VAL")"
if [[ -z "$MT5_LOGIN_VAL" ]]; then
  echo "Error: MT5_LOGIN is required (set in env or .env)"
  exit 1
fi
if [[ -z "$MT5_SERVER_VAL" ]]; then
  echo "Error: MT5_SERVER is required and cannot be blank/whitespace."
  exit 1
fi

BOT_CONFIG_ID_VAL="$(resolve_required_env BOT_CONFIG_ID)"
BOT_WS_URL_VAL="$(resolve_required_env BOT_WS_URL)"
VISION_LLM_API_URL_VAL="$(resolve_required_env VISION_LLM_API_URL)"
LIVE_MAGIC_NUMBER_VAL="${LIVE_MAGIC_NUMBER:-$(read_dotenv_value LIVE_MAGIC_NUMBER)}"
LIVE_MANAGE_MANUAL_POSITIONS_VAL="${LIVE_MANAGE_MANUAL_POSITIONS:-$(read_dotenv_value LIVE_MANAGE_MANUAL_POSITIONS)}"
validate_link_env "$BOT_WS_URL_VAL" "$VISION_LLM_API_URL_VAL"

INSTANCE_DIR="$STATE_DIR/${INSTANCE_NAME}"
INSTANCE_ENV_FILE="$INSTANCE_DIR/.env"
LEGACY_INSTANCE_ENV_FILE="$STATE_DIR/${INSTANCE_NAME}.env"
mkdir -p "$INSTANCE_DIR"
if [[ -f "$LEGACY_INSTANCE_ENV_FILE" && ! -f "$INSTANCE_ENV_FILE" ]]; then
  mv "$LEGACY_INSTANCE_ENV_FILE" "$INSTANCE_ENV_FILE"
fi

FORCED_WEB_PORT="${MT5_WEB_PORT:-}"
if [[ -f "$INSTANCE_ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$INSTANCE_ENV_FILE"
  if [[ -n "$FORCED_WEB_PORT" ]]; then
    MT5_WEB_PORT="$FORCED_WEB_PORT"
  fi
else
  seed="$(printf '%s' "$INSTANCE_NAME" | cksum | awk '{print $1}')"
  web_preferred=$((3100 + (seed % 500)))

  if [[ -n "$FORCED_WEB_PORT" ]]; then
    web_port="$FORCED_WEB_PORT"
  else
    web_port="$(pick_free_port "$web_preferred")"
  fi

  if [[ -z "$web_port" ]]; then
    echo "Error: cannot find free web port for instance '$INSTANCE_NAME'"
    exit 1
  fi

  cat > "$INSTANCE_ENV_FILE" <<EOF
COMPOSE_PROJECT_NAME=mt5_${INSTANCE_NAME}
MT5_WEB_PORT=$web_port
EOF
  MT5_WEB_PORT="$web_port"
fi

cat > "$INSTANCE_ENV_FILE" <<EOF
COMPOSE_PROJECT_NAME=mt5_${INSTANCE_NAME}
MT5_WEB_PORT=$MT5_WEB_PORT
EOF

COMPOSE_PROJECT_NAME="mt5_${INSTANCE_NAME}"

SNAPSHOT_HOST_DIR_RAW="${MT5_SNAPSHOT_HOST_DIR:-$(read_dotenv_value MT5_SNAPSHOT_HOST_DIR)}"
if [[ -z "$SNAPSHOT_HOST_DIR_RAW" ]]; then
  SNAPSHOT_HOST_DIR_RAW="./snapshots"
fi
if [[ "$SNAPSHOT_HOST_DIR_RAW" == /* ]]; then
  SNAPSHOT_HOST_DIR_ABS="$SNAPSHOT_HOST_DIR_RAW"
else
  SNAPSHOT_HOST_DIR_ABS="$ROOT_DIR/${SNAPSHOT_HOST_DIR_RAW#./}"
fi
mkdir -p "$SNAPSHOT_HOST_DIR_ABS"

export COMPOSE_PROJECT_NAME
export MT5_WEB_PORT
export MT5_SNAPSHOT_HOST_DIR="$SNAPSHOT_HOST_DIR_ABS"
export MT5_INSTANCE_HOST_DIR="$STATE_DIR"
export MT5_LOGIN="$MT5_LOGIN_VAL"
export MT5_PASSWORD="$MT5_PASSWORD_VAL"
export MT5_SERVER="$MT5_SERVER_VAL"
if [[ -n "$MT5_SERVER_FALLBACKS_VAL" ]]; then
  export MT5_SERVER_FALLBACKS="$MT5_SERVER_FALLBACKS_VAL"
fi
export BOT_CONFIG_ID="$BOT_CONFIG_ID_VAL"
export BOT_WS_URL="$BOT_WS_URL_VAL"
export VISION_LLM_API_URL="$VISION_LLM_API_URL_VAL"
if [[ -n "$LIVE_MAGIC_NUMBER_VAL" ]]; then
  export LIVE_MAGIC_NUMBER="$(printf '%s' "$LIVE_MAGIC_NUMBER_VAL" | tr -d '[:space:]')"
fi
if [[ -z "$LIVE_MANAGE_MANUAL_POSITIONS_VAL" ]]; then
  LIVE_MANAGE_MANUAL_POSITIONS_VAL="0"
fi
export LIVE_MANAGE_MANUAL_POSITIONS="$LIVE_MANAGE_MANUAL_POSITIONS_VAL"

if [[ -z "${MT5_SNAPSHOT_PATH:-}" ]]; then
  export MT5_SNAPSHOT_PATH="/snapshots/mt5-config-snapshot.tgz"
fi

if [[ -z "${LIVE_MODELS_DIR:-}" ]]; then
  export LIVE_MODELS_DIR="/bots/${PROFILE}/models"
fi
if [[ -z "${LIVE_LLM_SEMANTIC_CACHE_FILE:-}" ]]; then
  export LIVE_LLM_SEMANTIC_CACHE_FILE="${LIVE_MODELS_DIR}/time_to_embedding_llm_cls.joblib"
fi
if [[ -z "${LIVE_LLM_TEXT_LOG_FILE:-}" ]]; then
  export LIVE_LLM_TEXT_LOG_FILE="${LIVE_MODELS_DIR}/time_to_llm_text.jsonl"
fi
if [[ -z "${LIVE_PREWARM_SEMANTIC_ON_START:-}" ]]; then
  export LIVE_PREWARM_SEMANTIC_ON_START="1"
fi
if [[ -z "${LIVE_PREWARM_SEMANTIC_MAX_SECONDS:-}" ]]; then
  export LIVE_PREWARM_SEMANTIC_MAX_SECONDS="45"
fi
if [[ -z "${LIVE_PREWARM_SEMANTIC_MAX_MISSING:-}" ]]; then
  export LIVE_PREWARM_SEMANTIC_MAX_MISSING="24"
fi
if [[ -z "${LIVE_PREWARM_REQUEST_TIMEOUT_SEC:-}" ]]; then
  export LIVE_PREWARM_REQUEST_TIMEOUT_SEC="20"
fi
if [[ -z "${LIVE_PERFORMANCE_SYNC_INTERVAL_SEC:-}" ]]; then
  export LIVE_PERFORMANCE_SYNC_INTERVAL_SEC="120"
fi
if [[ -z "${LIVE_PERFORMANCE_BOOT_LOOKBACK_DAYS:-}" ]]; then
  export LIVE_PERFORMANCE_BOOT_LOOKBACK_DAYS="3650"
fi
if [[ -z "${LIVE_MT5_HISTORY_END_AHEAD_HOURS:-}" ]]; then
  export LIVE_MT5_HISTORY_END_AHEAD_HOURS="2"
fi
if [[ -z "${LIVE_PERFORMANCE_SCOPE:-}" ]]; then
  export LIVE_PERFORMANCE_SCOPE="symbol"
fi
if [[ -z "${LIVE_MANAGED_MAGIC_SET:-}" ]]; then
  BASE_MANAGED_MAGIC="${LIVE_MAGIC_NUMBER:-123456}"
  BASE_MANAGED_MAGIC="$(printf '%s' "$BASE_MANAGED_MAGIC" | tr -d '[:space:]')"
  if [[ -z "$BASE_MANAGED_MAGIC" ]]; then
    BASE_MANAGED_MAGIC="123456"
  fi
  export LIVE_MANAGED_MAGIC_SET="${BASE_MANAGED_MAGIC},123456,12345,0"
fi
if [[ -z "${LIVE_PERFORMANCE_MAGIC_SET:-}" ]]; then
  export LIVE_PERFORMANCE_MAGIC_SET="${LIVE_MANAGED_MAGIC_SET}"
fi

if [[ -z "${BOT_SCRIPT:-}" ]]; then
  export BOT_SCRIPT="$BOT_SCRIPT_DEFAULT"
fi
if [[ -z "${BOT_REQUIREMENTS:-}" ]]; then
  export BOT_REQUIREMENTS="$BOT_REQUIREMENTS_DEFAULT"
fi
if [[ -z "${BOT_LOG:-}" ]]; then
  export BOT_LOG="/config/${INSTANCE_NAME}_${PROFILE}.log"
fi
if [[ -z "${LIVE_STATE_FILE:-}" ]]; then
  export LIVE_STATE_FILE="/instances/${INSTANCE_NAME}/run_live_state.json"
fi

INSTANCE_STATE_HOST_FILE="$INSTANCE_DIR/run_live_state.json"

echo "Instance: $INSTANCE_NAME"
echo "Project: $COMPOSE_PROJECT_NAME"
echo "Profile: $PROFILE"
echo "Web: http://localhost:$MT5_WEB_PORT"
echo "MT5_LOGIN: $MT5_LOGIN"
echo "MT5_SERVER: $MT5_SERVER"
echo "Bot: $BOT_SCRIPT"
echo "BOT_CONFIG_ID: $BOT_CONFIG_ID"
echo "BOT_WS_URL: $BOT_WS_URL"
echo "VISION_LLM_API_URL: $VISION_LLM_API_URL"
echo "LIVE_MAGIC_NUMBER: ${LIVE_MAGIC_NUMBER:-auto}"
echo "LIVE_MANAGE_MANUAL_POSITIONS: ${LIVE_MANAGE_MANUAL_POSITIONS:-0}"
echo "LIVE_MODELS_DIR: ${LIVE_MODELS_DIR:-auto}"
echo "LIVE_LLM_SEMANTIC_CACHE_FILE: ${LIVE_LLM_SEMANTIC_CACHE_FILE:-auto}"
echo "LIVE_LLM_TEXT_LOG_FILE: ${LIVE_LLM_TEXT_LOG_FILE:-auto}"
echo "LIVE_PREWARM_SEMANTIC_ON_START: ${LIVE_PREWARM_SEMANTIC_ON_START:-auto}"
echo "LIVE_PREWARM_SEMANTIC_MAX_SECONDS: ${LIVE_PREWARM_SEMANTIC_MAX_SECONDS:-auto}"
echo "LIVE_PREWARM_SEMANTIC_MAX_MISSING: ${LIVE_PREWARM_SEMANTIC_MAX_MISSING:-auto}"
echo "LIVE_PREWARM_REQUEST_TIMEOUT_SEC: ${LIVE_PREWARM_REQUEST_TIMEOUT_SEC:-auto}"
echo "LIVE_PERFORMANCE_SYNC_INTERVAL_SEC: ${LIVE_PERFORMANCE_SYNC_INTERVAL_SEC:-auto}"
echo "LIVE_PERFORMANCE_BOOT_LOOKBACK_DAYS: ${LIVE_PERFORMANCE_BOOT_LOOKBACK_DAYS:-auto}"
echo "LIVE_MT5_HISTORY_END_AHEAD_HOURS: ${LIVE_MT5_HISTORY_END_AHEAD_HOURS:-auto}"
echo "LIVE_PERFORMANCE_SCOPE: ${LIVE_PERFORMANCE_SCOPE:-auto}"
echo "LIVE_MANAGED_MAGIC_SET: ${LIVE_MANAGED_MAGIC_SET:-auto}"
echo "LIVE_PERFORMANCE_MAGIC_SET: ${LIVE_PERFORMANCE_MAGIC_SET:-auto}"
echo "LIVE_STATE_FILE: ${LIVE_STATE_FILE:-auto}"
echo "HOST_STATE_FILE: $INSTANCE_STATE_HOST_FILE"
echo "Snapshot host dir: $MT5_SNAPSHOT_HOST_DIR"
echo "Snapshot path in container: $MT5_SNAPSHOT_PATH"

exec ./start_bot.sh
