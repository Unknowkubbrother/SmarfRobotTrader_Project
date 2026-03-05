#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

SERVICE_NAME="${SERVICE_NAME:-metatrader5-macos}"
BOT_SCRIPT="${BOT_SCRIPT:-run_live.py}"
BOT_LOG="${BOT_LOG:-/config/bot.log}"
BOT_REQUIREMENTS="${BOT_REQUIREMENTS:-/bots/requirements-live.txt}"
BOT_STREAM_LOG_TO_CONTAINER_STDOUT="${BOT_STREAM_LOG_TO_CONTAINER_STDOUT:-1}"
RUNNER_PROGRESS_TO_CONTAINER_LOGS="${RUNNER_PROGRESS_TO_CONTAINER_LOGS:-1}"
WAIT_TIMEOUT_SECONDS="${WAIT_TIMEOUT_SECONDS:-900}"
WAIT_INTERVAL_SECONDS="${WAIT_INTERVAL_SECONDS:-5}"
TORCH_VERSION="${TORCH_VERSION:-2.5.1}"
AUTO_BUILD="${AUTO_BUILD:-1}"
FORCE_REBUILD="${FORCE_REBUILD:-0}"
PULL_LATEST_IMAGE="${PULL_LATEST_IMAGE:-1}"
METATRADER_IMAGE="${METATRADER_IMAGE:-metatrader5_macos}"
USE_SHARED_PYDEPS="${USE_SHARED_PYDEPS:-1}"
SHARED_PYDEPS_DIR="${SHARED_PYDEPS_DIR:-/shared-pydeps}"
MT5_LOGIN_CHECK_TIMEOUT_SECONDS="${MT5_LOGIN_CHECK_TIMEOUT_SECONDS:-180}"
MT5_TRADE_CHECK_TIMEOUT_SECONDS="${MT5_TRADE_CHECK_TIMEOUT_SECONDS:-45}"
MT5_DIALOG_SEARCH_WAIT_SECONDS="${MT5_DIALOG_SEARCH_WAIT_SECONDS:-6}"
MT5_COMPANY_DISCOVERY_BEFORE_LOGIN="${MT5_COMPANY_DISCOVERY_BEFORE_LOGIN:-1}"
MT5_COMPANY_DIALOG_CLEANUP_AFTER_LOGIN="${MT5_COMPANY_DIALOG_CLEANUP_AFTER_LOGIN:-1}"
MT5_REFRESH_COMPANY_CACHE_AFTER_START="${MT5_REFRESH_COMPANY_CACHE_AFTER_START:-0}"
MT5_COMPANY_SEARCH_QUERY="${MT5_COMPANY_SEARCH_QUERY:-}"
MT5_SKIP_PRECHECKS="${MT5_SKIP_PRECHECKS:-0}"
MT5_ALLOW_PARTIAL_START="${MT5_ALLOW_PARTIAL_START:-0}"
BOT_WAIT_FOR_WS_REGISTER="${BOT_WAIT_FOR_WS_REGISTER:-1}"
BOT_WS_READY_TIMEOUT_SECONDS="${BOT_WS_READY_TIMEOUT_SECONDS:-120}"
MT5_ALGO_STABILIZE_SECONDS="${MT5_ALGO_STABILIZE_SECONDS:-20}"

trim_outer_whitespace() {
  local raw="$1"
  printf '%s' "$raw" | sed -E 's/^[[:space:]]+//; s/[[:space:]]+$//'
}

is_truthy() {
  local raw="${1:-}"
  raw="$(printf '%s' "$raw" | tr '[:upper:]' '[:lower:]' | tr -d '[:space:]')"
  [[ "$raw" =~ ^(1|true|yes|on)$ ]]
}

read_dotenv_value() {
  local key="$1"
  local value=""

  if [[ -f ".env" ]]; then
    local line
    line="$(grep -E "^${key}=" .env | tail -n 1 || true)"
    if [[ -n "$line" ]]; then
      value="${line#*=}"
      # Strip optional single/double quotes around value.
      value="${value%\"}"
      value="${value#\"}"
      value="${value%\'}"
      value="${value#\'}"
    fi
  fi

  printf '%s' "$value"
}

MT5_LOGIN_VAL="${MT5_LOGIN:-$(read_dotenv_value MT5_LOGIN)}"
MT5_PASSWORD_VAL="${MT5_PASSWORD:-$(read_dotenv_value MT5_PASSWORD)}"
MT5_SERVER_VAL="${MT5_SERVER:-$(read_dotenv_value MT5_SERVER)}"
MT5_INIT_TIMEOUT_VAL="${MT5_INIT_TIMEOUT:-$(read_dotenv_value MT5_INIT_TIMEOUT)}"
MT5_LOGIN_RETRIES_VAL="${MT5_LOGIN_RETRIES:-$(read_dotenv_value MT5_LOGIN_RETRIES)}"
MT5_RETRY_SECONDS_VAL="${MT5_RETRY_SECONDS:-$(read_dotenv_value MT5_RETRY_SECONDS)}"
MT5_RPC_TIMEOUT_MS_VAL="${MT5_RPC_TIMEOUT_MS:-$(read_dotenv_value MT5_RPC_TIMEOUT_MS)}"
MT5_STRICT_SERVER_MATCH_VAL="${MT5_STRICT_SERVER_MATCH:-$(read_dotenv_value MT5_STRICT_SERVER_MATCH)}"
CUSTOM_USER_VAL="${CUSTOM_USER:-$(read_dotenv_value CUSTOM_USER)}"
VNC_PASSWORD_VAL="${PASSWORD:-$(read_dotenv_value PASSWORD)}"

MT5_LOGIN_VAL="$(printf '%s' "$MT5_LOGIN_VAL" | tr -d '[:space:]')"
MT5_SERVER_VAL="$(trim_outer_whitespace "$MT5_SERVER_VAL")"
MT5_STRICT_SERVER_MATCH_VAL="$(trim_outer_whitespace "$MT5_STRICT_SERVER_MATCH_VAL")"

if [[ -z "$MT5_SERVER_VAL" && -n "${MT5_SERVER_FALLBACKS:-}" ]]; then
  MT5_SERVER_VAL="$(
    printf '%s' "${MT5_SERVER_FALLBACKS}" \
      | tr ';' ',' \
      | awk -F, '{for (i=1;i<=NF;i++){gsub(/^[ \t\r\n]+|[ \t\r\n]+$/, "", $i); if(length($i)>0){print $i; exit}}}'
  )"
  if [[ -n "$MT5_SERVER_VAL" ]]; then
    echo "warning: MT5_SERVER is blank, using first MT5_SERVER_FALLBACKS value: '$MT5_SERVER_VAL'"
  fi
fi

# If VNC credentials are not explicitly configured, reuse MT5 credentials.
if [[ -z "$CUSTOM_USER_VAL" && -n "$MT5_LOGIN_VAL" ]]; then
  CUSTOM_USER_VAL="$MT5_LOGIN_VAL"
fi
if [[ -z "$VNC_PASSWORD_VAL" && -n "$MT5_PASSWORD_VAL" ]]; then
  VNC_PASSWORD_VAL="$MT5_PASSWORD_VAL"
fi
if [[ -z "$MT5_STRICT_SERVER_MATCH_VAL" ]]; then
  MT5_STRICT_SERVER_MATCH_VAL="1"
fi

# Final fallback for first-time setup without MT5 credentials.
if [[ -z "$CUSTOM_USER_VAL" ]]; then
  CUSTOM_USER_VAL="admin"
fi
if [[ -z "$VNC_PASSWORD_VAL" ]]; then
  VNC_PASSWORD_VAL="12345"
fi

export CUSTOM_USER="$CUSTOM_USER_VAL"
export PASSWORD="$VNC_PASSWORD_VAL"

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

run_with_timeout() {
  local timeout_seconds="${1:-0}"
  shift

  if (( timeout_seconds <= 0 )); then
    "$@"
    return $?
  fi

  "$@" &
  local cmd_pid=$!
  local elapsed=0

  while kill -0 "$cmd_pid" >/dev/null 2>&1; do
    if (( elapsed >= timeout_seconds )); then
      kill "$cmd_pid" >/dev/null 2>&1 || true
      sleep 1
      kill -9 "$cmd_pid" >/dev/null 2>&1 || true
      wait "$cmd_pid" >/dev/null 2>&1 || true
      return 124
    fi
    sleep 1
    elapsed=$((elapsed + 1))
  done

  wait "$cmd_pid"
}

cleanup_stale_mt5_exec_processes() {
  compose exec -T "$SERVICE_NAME" sh -lc \
    "pkill -f '/usr/bin/python3 python3 -' >/dev/null 2>&1 || true"
}

ensure_image_ready() {
  local image_name="${METATRADER_IMAGE}"
  local should_build=0
  local should_pull=0
  local pull_succeeded=0
  local auto_build_lc
  local force_rebuild_lc
  local pull_latest_lc

  auto_build_lc="$(printf '%s' "$AUTO_BUILD" | tr '[:upper:]' '[:lower:]')"
  force_rebuild_lc="$(printf '%s' "$FORCE_REBUILD" | tr '[:upper:]' '[:lower:]')"
  pull_latest_lc="$(printf '%s' "$PULL_LATEST_IMAGE" | tr '[:upper:]' '[:lower:]')"

  case "$auto_build_lc" in
    1|true|yes|on|always)
      should_build=1
      ;;
    0|false|no|off|never)
      should_build=0
      ;;
    *)
      should_build=1
      ;;
  esac

  case "$pull_latest_lc" in
    1|true|yes|on|always)
      should_pull=1
      ;;
    0|false|no|off|never)
      should_pull=0
      ;;
    *)
      should_pull=1
      ;;
  esac

  if [[ "$force_rebuild_lc" =~ ^(1|true|yes|on)$ ]]; then
    echo "[0/6] Building MT5 image (forced rebuild)..."
    compose build --no-cache "$SERVICE_NAME"
    return 0
  fi

  if (( should_pull == 1 )); then
    echo "[0/6] Pulling MT5 image (${image_name})..."
    if compose pull "$SERVICE_NAME"; then
      pull_succeeded=1
    else
      echo "  warning: docker pull failed, fallback to local image/build."
    fi
  fi

  if (( should_build == 1 )); then
    echo "[0/6] Building/refreshing MT5 image..."
    compose build "$SERVICE_NAME"
    return 0
  fi

  if (( pull_succeeded == 1 )); then
    echo "[0/6] AUTO_BUILD disabled. Using pulled image."
    return 0
  fi

  if ! docker image inspect "$image_name" >/dev/null 2>&1; then
    echo "[0/6] MT5 image not found locally. Trying pull once..."
    if compose pull "$SERVICE_NAME"; then
      return 0
    fi
    echo "  warning: pull failed. Building once..."
    compose build "$SERVICE_NAME"
  else
    echo "[0/6] AUTO_BUILD disabled and image exists. Skip build."
  fi
}

check_trade_allowed() {
  local output
  local rc=0

  output="$(
    run_with_timeout "$MT5_TRADE_CHECK_TIMEOUT_SECONDS" \
      compose exec -T \
        -e MT5_LOGIN="$MT5_LOGIN_VAL" \
        -e MT5_PASSWORD="$MT5_PASSWORD_VAL" \
        -e MT5_SERVER="$MT5_SERVER_VAL" \
        -e MT5_RPC_TIMEOUT_MS="${MT5_RPC_TIMEOUT_MS_VAL:-180000}" \
        "$SERVICE_NAME" python3 - <<'PY'
from mt5linux import MetaTrader5
import rpyc
import os
import sys

try:
    timeout_ms = int(os.getenv("MT5_RPC_TIMEOUT_MS", "180000"))
    rpyc.core.protocol.DEFAULT_CONFIG["sync_request_timeout"] = max(60.0, float(timeout_ms) / 1000.0)
    mt5 = MetaTrader5(host="localhost", port=8001)

    # Read-only probe: avoid credentialed re-login side effects that can
    # flip MT5 AutoTrading state while probing.
    if not mt5.initialize(timeout=timeout_ms):
        print("trade_allowed=0 tradeapi_disabled=unknown (initialize failed)")
        sys.exit(2)

    info = mt5.terminal_info()
    if info is None:
        print("trade_allowed=0 tradeapi_disabled=unknown (terminal_info is None)")
        sys.exit(2)
    trade_allowed = bool(getattr(info, "trade_allowed", False))
    tradeapi_disabled = bool(getattr(info, "tradeapi_disabled", False))
    print(f"trade_allowed={int(trade_allowed)} tradeapi_disabled={int(tradeapi_disabled)}")
    if trade_allowed and not tradeapi_disabled:
        sys.exit(0)
    if tradeapi_disabled:
        sys.exit(5)
    sys.exit(1)
except Exception as exc:
    print(f"trade_allowed=0 check_exception={exc}")
    sys.exit(3)
PY
  )" || rc=$?

  if [[ -n "$output" ]]; then
    printf '%s\n' "$output"
  fi

  if [[ "$rc" -eq 124 ]]; then
    echo "trade_allowed=0 check_exception=timeout"
    cleanup_stale_mt5_exec_processes || true
    return 3
  fi

  return "$rc"
}

ensure_mt5_login_if_configured() {
  if [[ -z "$MT5_LOGIN_VAL" || -z "$MT5_PASSWORD_VAL" ]]; then
    echo "[2.4/6] MT5 credentials not set, skip forced API login."
    append_runner_progress "[2.4/6] MT5 credentials not set, skip forced API login."
    return 0
  fi

  echo "[2.4/6] Ensuring MT5 account login via API..."
  append_runner_progress "[2.4/6] Ensuring MT5 account login via API..."
  cleanup_stale_mt5_exec_processes || true

  if is_truthy "$MT5_COMPANY_DISCOVERY_BEFORE_LOGIN" && [[ -n "$MT5_SERVER_VAL" ]]; then
    echo "  discovering MT5 company list for server: $MT5_SERVER_VAL"
    append_runner_progress "discovering MT5 company list for server: $MT5_SERVER_VAL"
    search_company_dialog_by_server "$MT5_SERVER_VAL" "1" || true
    dismiss_mt5_dialogs_retry 3 1 || true
  fi

  local login_output=""
  local rc=0
  login_output="$(
    run_with_timeout "$MT5_LOGIN_CHECK_TIMEOUT_SECONDS" \
      compose exec -T \
        -e MT5_LOGIN="$MT5_LOGIN_VAL" \
        -e MT5_PASSWORD="$MT5_PASSWORD_VAL" \
        -e MT5_SERVER="$MT5_SERVER_VAL" \
        -e MT5_LOGIN_RETRIES="${MT5_LOGIN_RETRIES_VAL:-20}" \
        -e MT5_RETRY_SECONDS="${MT5_RETRY_SECONDS_VAL:-5}" \
        -e MT5_RPC_TIMEOUT_MS="${MT5_RPC_TIMEOUT_MS_VAL:-180000}" \
        -e MT5_LOGIN_ATTEMPT_TIMEOUT_SEC="${MT5_LOGIN_ATTEMPT_TIMEOUT_SEC:-25}" \
        -e MT5_SERVER_FALLBACKS="${MT5_SERVER_FALLBACKS:-}" \
        -e MT5_STRICT_SERVER_MATCH="${MT5_STRICT_SERVER_MATCH_VAL}" \
        "$SERVICE_NAME" python3 - <<'PY'
from mt5linux import MetaTrader5
import rpyc
import multiprocessing as mp
import os
import sys
import time

login_text = os.getenv("MT5_LOGIN", "").strip()
password = os.getenv("MT5_PASSWORD", "").strip()
server = os.getenv("MT5_SERVER", "").strip()
retries = int(os.getenv("MT5_LOGIN_RETRIES", "20"))
retry_seconds = int(os.getenv("MT5_RETRY_SECONDS", "5"))
timeout_ms = int(os.getenv("MT5_RPC_TIMEOUT_MS", "180000"))
rpyc.core.protocol.DEFAULT_CONFIG["sync_request_timeout"] = max(60.0, float(timeout_ms) / 1000.0)
attempt_timeout_sec = int(os.getenv("MT5_LOGIN_ATTEMPT_TIMEOUT_SEC", "25"))
fallbacks_raw = str(os.getenv("MT5_SERVER_FALLBACKS", "") or "").strip()
strict_server = str(os.getenv("MT5_STRICT_SERVER_MATCH", "0") or "").strip().lower() in {"1", "true", "yes", "on"}

if not login_text or not password:
    print("skip_login=1")
    sys.exit(0)

try:
    login_id = int(login_text)
except ValueError:
    print(f"invalid_login={login_text}")
    sys.exit(1)

def normalize_server_name(name: str) -> str:
    return " ".join(str(name or "").split()).strip()


def add_candidate(candidates: list[str], value: str) -> None:
    normalized = normalize_server_name(value)
    if normalized and normalized not in candidates:
        candidates.append(normalized)


def build_server_candidates(primary: str, fallbacks: str) -> list[str]:
    candidates: list[str] = []
    primary_name = normalize_server_name(primary)
    add_candidate(candidates, primary_name)

    if fallbacks:
        for token in fallbacks.split(","):
            add_candidate(candidates, token)

    if strict_server:
        return candidates

    if primary_name:
        if primary_name.lower().startswith("mt5 "):
            add_candidate(candidates, primary_name[4:])
        else:
            add_candidate(candidates, f"MT5 {primary_name}")

    if not candidates:
        # Allow MT5 to resolve the server automatically when env is blank.
        return [""]
    return candidates


def probe_login(server_name: str, login_id: int, pwd: str, rpc_timeout: int, result_queue) -> None:
    payload = {
        "server": server_name,
        "ok": False,
        "login": 0,
        "account_server": "",
        "error": "",
    }
    mt5 = None
    try:
        # Keep each login attempt bounded by the per-attempt watchdog, even if
        # global MT5 RPC timeout is configured higher.
        attempt_timeout_ms = max(5000, int(attempt_timeout_sec) * 1000)
        effective_timeout_ms = max(5000, min(int(rpc_timeout), attempt_timeout_ms))
        try:
            rpyc.core.protocol.DEFAULT_CONFIG["sync_request_timeout"] = max(10.0, float(effective_timeout_ms) / 1000.0)
        except Exception:
            pass
        mt5 = MetaTrader5(host="localhost", port=8001)
        init_kwargs = {"timeout": effective_timeout_ms, "login": login_id, "password": pwd}
        if server_name:
            init_kwargs["server"] = server_name
        init_ok = bool(mt5.initialize(**init_kwargs))
        payload["ok"] = init_ok
        try:
            payload["error"] = str(mt5.last_error())
        except Exception:
            payload["error"] = ""

        account = mt5.account_info()
        if account is not None:
            payload["login"] = int(getattr(account, "login", 0) or 0)
            payload["account_server"] = str(getattr(account, "server", "") or "")

        if not (init_ok and payload["login"] == int(login_id)):
            payload["ok"] = False
    except Exception as exc:
        payload["error"] = f"exception:{exc}"
        payload["ok"] = False
    finally:
        try:
            if mt5 is not None:
                mt5.shutdown()
        except Exception:
            pass
        try:
            result_queue.put(payload)
        except Exception:
            pass


server_candidates = build_server_candidates(server, fallbacks_raw)
if not server_candidates:
    print("login_failed=1")
    sys.exit(1)

for attempt in range(1, retries + 1):
    for server_name in server_candidates:
        server_display = server_name if server_name else "<auto>"
        queue = mp.Queue()
        proc = mp.Process(
            target=probe_login,
            args=(server_name, login_id, password, timeout_ms, queue),
            daemon=True,
        )
        proc.start()
        proc.join(attempt_timeout_sec)

        if proc.is_alive():
            proc.terminate()
            proc.join(3)
            print(f"login_attempt_{attempt}_server={server_display} result=timeout")
            continue

        result = None
        try:
            result = queue.get_nowait()
        except Exception:
            result = {
                "server": server_name,
                "ok": False,
                "login": 0,
                "account_server": "",
                "error": "no_result",
            }

        print(
            f"login_attempt_{attempt}_server={result.get('server','') or '<auto>'}"
            f" ok={int(bool(result.get('ok')))}"
            f" login={int(result.get('login') or 0)}"
            f" account_server={result.get('account_server','')}"
            f" error={result.get('error','')}"
        )

        if bool(result.get("ok")):
            resolved_server = normalize_server_name(result.get("account_server") or server_name)
            if resolved_server:
                print(f"resolved_server={resolved_server}")
            print(f"login_ok={int(result.get('login') or login_id)}")
            sys.exit(0)

    if attempt < retries:
        print(f"waiting_login_retry={attempt}/{retries}")
    time.sleep(retry_seconds)

print("login_failed=1")
sys.exit(1)
PY
  )" || rc=$?

  if [[ -n "$login_output" ]]; then
    printf '%s\n' "$login_output"
    while IFS= read -r line; do
      [[ -n "$line" ]] && append_runner_progress "$line"
    done <<< "$login_output"
  fi

  if [[ "$rc" -eq 124 ]]; then
    echo "  warning: MT5 login probe timed out (${MT5_LOGIN_CHECK_TIMEOUT_SECONDS}s)."
    append_runner_progress "warning: MT5 login probe timed out (${MT5_LOGIN_CHECK_TIMEOUT_SECONDS}s)."
    cleanup_stale_mt5_exec_processes || true
    return 1
  fi

  local resolved_server
  resolved_server="$(printf '%s\n' "$login_output" | sed -n 's/^resolved_server=//p' | tail -n 1 | tr -d '\r')"
  if [[ -n "$resolved_server" && "$resolved_server" != "$MT5_SERVER_VAL" ]]; then
    echo "  resolved MT5 server alias: '$MT5_SERVER_VAL' -> '$resolved_server'"
    append_runner_progress "resolved MT5 server alias: '$MT5_SERVER_VAL' -> '$resolved_server'"
    MT5_SERVER_VAL="$resolved_server"
    export MT5_SERVER="$MT5_SERVER_VAL"
  fi

  if [[ "$rc" -ne 0 ]]; then
    search_company_dialog_by_server "$MT5_SERVER_VAL" || true
    dismiss_mt5_dialogs || true
  elif is_truthy "$MT5_COMPANY_DIALOG_CLEANUP_AFTER_LOGIN"; then
    dismiss_mt5_dialogs_retry 6 1 || true
  fi

  cleanup_stale_mt5_exec_processes || true
  return "$rc"
}

ensure_xdotool() {
  if compose exec -T "$SERVICE_NAME" sh -lc "command -v xdotool >/dev/null 2>&1"; then
    return 0
  fi

  # Fallback for old images that do not have xdotool baked in yet.
  compose exec -T "$SERVICE_NAME" sh -lc \
    "apt-get update >/dev/null 2>&1 && apt-get install -y --no-install-recommends xdotool >/dev/null 2>&1"
}

dismiss_mt5_dialogs() {
  compose exec -T -u abc "$SERVICE_NAME" sh -lc '
display=":1"

# Press ESC a few times to close modal dialogs (open account/company wizard).
for _ in 1 2 3; do
  DISPLAY="$display" xdotool key --clearmodifiers Escape >/dev/null 2>&1 || true
  sleep 0.4
done

# If known dialog windows exist, activate and close/cancel them.
for pattern in "Select a company" "Open an account" "Find your company"; do
  for wid in $(DISPLAY="$display" xdotool search --onlyvisible --name "$pattern" 2>/dev/null | head -n 5); do
    DISPLAY="$display" xdotool windowactivate --sync "$wid" >/dev/null 2>&1 || true
    DISPLAY="$display" xdotool key --window "$wid" --clearmodifiers Escape >/dev/null 2>&1 || true
    DISPLAY="$display" xdotool key --window "$wid" --clearmodifiers Alt+c >/dev/null 2>&1 || true
    DISPLAY="$display" xdotool key --window "$wid" --clearmodifiers Alt+F4 >/dev/null 2>&1 || true
  done
done

# Handle first-run updater popup that can block AutoTrading toggles.
for pattern in "LiveUpdate" "Welcome to LiveUpdate"; do
  for wid in $(DISPLAY="$display" xdotool search --onlyvisible --name "$pattern" 2>/dev/null | head -n 5); do
    DISPLAY="$display" xdotool windowactivate --sync "$wid" >/dev/null 2>&1 || true
    sleep 0.2

    WIDTH=""
    HEIGHT=""
    eval "$(DISPLAY="$display" xdotool getwindowgeometry --shell "$wid" 2>/dev/null | grep -E "^(WIDTH|HEIGHT)=")" || true
    WIDTH="${WIDTH:-640}"
    HEIGHT="${HEIGHT:-380}"

    # Prefer clicking "Later" (right button on this popup layout).
    later_x=$(( WIDTH - 95 ))
    later_y=$(( HEIGHT - 36 ))
    [ "$later_x" -lt 200 ] && later_x=200
    [ "$later_y" -lt 120 ] && later_y=120

    DISPLAY="$display" xdotool mousemove --window "$wid" "$later_x" "$later_y" click 1 >/dev/null 2>&1 || true
    DISPLAY="$display" xdotool key --window "$wid" --clearmodifiers Alt+l >/dev/null 2>&1 || true
    DISPLAY="$display" xdotool key --window "$wid" --clearmodifiers Escape >/dev/null 2>&1 || true
    DISPLAY="$display" xdotool key --window "$wid" --clearmodifiers Alt+F4 >/dev/null 2>&1 || true
  done
done
'
}

dismiss_mt5_dialogs_retry() {
  local retries="${1:-5}"
  local interval_seconds="${2:-1}"
  local i=0
  if [[ -z "$retries" || ! "$retries" =~ ^[0-9]+$ ]]; then
    retries=5
  fi
  if [[ -z "$interval_seconds" || ! "$interval_seconds" =~ ^[0-9]+$ ]]; then
    interval_seconds=1
  fi
  for (( i=1; i<=retries; i++ )); do
    dismiss_mt5_dialogs || true
    sleep "$interval_seconds"
  done
}

ensure_company_dialog_open() {
  if ! ensure_xdotool; then
    return 1
  fi

  compose exec -T -u abc "$SERVICE_NAME" sh -lc '
display=":1"

wid="$(DISPLAY="$display" xdotool search --onlyvisible --name "Select a company" 2>/dev/null | head -n1 || true)"
if [ -n "$wid" ]; then
  exit 0
fi

main_wid="$(DISPLAY="$display" xdotool search --onlyvisible --name "MetaTrader|Netting|MetaQuotes" 2>/dev/null | head -n1 || true)"
if [ -z "$main_wid" ]; then
  exit 1
fi

DISPLAY="$display" xdotool windowactivate --sync "$main_wid" >/dev/null 2>&1 || true
sleep 0.2

# Try File -> Open an Account (common hotkey "O" on English UI)
DISPLAY="$display" xdotool key --window "$main_wid" --clearmodifiers alt+f >/dev/null 2>&1 || true
sleep 0.25
DISPLAY="$display" xdotool key --window "$main_wid" --clearmodifiers o >/dev/null 2>&1 || true
sleep 1.0

wid="$(DISPLAY="$display" xdotool search --onlyvisible --name "Select a company" 2>/dev/null | head -n1 || true)"
if [ -n "$wid" ]; then
  exit 0
fi

# Fallback alternate accelerator.
DISPLAY="$display" xdotool key --window "$main_wid" --clearmodifiers alt+f >/dev/null 2>&1 || true
sleep 0.25
DISPLAY="$display" xdotool key --window "$main_wid" --clearmodifiers a >/dev/null 2>&1 || true
sleep 1.0

wid="$(DISPLAY="$display" xdotool search --onlyvisible --name "Select a company" 2>/dev/null | head -n1 || true)"
[ -n "$wid" ] && exit 0 || exit 1
'
}

search_company_dialog_by_server() {
  local server_name="${1:-}"
  local force_open="${2:-0}"
  local -a company_queries=()
  local server_no_suffix=""
  local server_spaced=""
  local company_query_hint_raw=""
  local company_query_hint_item=""
  local query_debug=""
  server_name="$(printf '%s' "$server_name" | sed -E 's/[[:space:]]+/ /g; s/^ //; s/ $//')"
  if [[ -z "$server_name" ]]; then
    return 0
  fi
  company_query_hint_raw="$(printf '%s' "$MT5_COMPANY_SEARCH_QUERY" | sed -E 's/[[:space:]]+/ /g; s/^ //; s/ $//')"

  add_company_query() {
    local raw_query="${1:-}"
    local normalized_query=""
    normalized_query="$(printf '%s' "$raw_query" | sed -E 's/[[:space:]]+/ /g; s/^ //; s/ $//')"
    if [[ -z "$normalized_query" ]]; then
      return 0
    fi
    local existing=""
    for existing in "${company_queries[@]:-}"; do
      if [[ "$existing" == "$normalized_query" ]]; then
        return 0
      fi
    done
    company_queries+=("$normalized_query")
  }

  add_company_query "$server_name"
  server_no_suffix="$(printf '%s' "$server_name" | sed -E 's/[[:space:]]*[-_](demo|live|real|ecn|pro)$//I')"
  add_company_query "$server_no_suffix"
  add_company_query "$(printf '%s' "$server_name" | sed -E 's/[[:space:]]*[-_].*$//')"
  add_company_query "$(printf '%s' "$server_name" | sed -E 's/[-_]+/ /g; s/[[:space:]]+/ /g; s/^ //; s/ $//')"
  server_spaced="$(printf '%s' "$server_no_suffix" | sed -E 's/([[:lower:]])([[:upper:]])/\1 \2/g; s/[[:space:]]+/ /g; s/^ //; s/ $//')"
  add_company_query "$server_spaced"
  add_company_query "$(printf '%s' "$server_spaced" | awk '{print $1}')"
  add_company_query "$(printf '%s' "$server_spaced" | awk '{print $1, $2}' | sed -E 's/[[:space:]]+/ /g; s/^ //; s/ $//')"
  add_company_query "$(printf '%s' "$server_spaced" | awk '{print $1$2}')"
  add_company_query "$(printf '%s' "$server_spaced" | awk '{print tolower($1$2)}')"
  add_company_query "$(printf '%s' "$server_name" | sed -E 's/[[:space:]]+.*$//')"
  if [[ -n "$company_query_hint_raw" ]]; then
    while IFS= read -r company_query_hint_item; do
      add_company_query "$company_query_hint_item"
    done < <(
      printf '%s' "$company_query_hint_raw" \
        | tr ';|' '\n' \
        | tr ',' '\n'
    )
  fi
  if [[ "${#company_queries[@]}" -eq 0 ]]; then
    company_queries+=("$server_name")
  fi
  query_debug="$(printf '%s | ' "${company_queries[@]}")"
  query_debug="${query_debug% | }"
  if [[ -n "$query_debug" ]]; then
    echo "  company discovery queries: $query_debug"
    append_runner_progress "company discovery queries: $query_debug"
  fi

  if ! ensure_xdotool; then
    return 1
  fi

  if [[ "$force_open" =~ ^(1|true|yes|on)$ ]]; then
    ensure_company_dialog_open || true
  fi

  compose exec -T -u abc \
    -e MT5_SERVER="$server_name" \
    -e MT5_COMPANY_QUERIES="$(printf '%s\n' "${company_queries[@]}")" \
    -e MT5_DIALOG_SEARCH_WAIT_SECONDS="$MT5_DIALOG_SEARCH_WAIT_SECONDS" \
    "$SERVICE_NAME" sh -lc '
display=":1"
server="${MT5_SERVER:-}"
query_list="${MT5_COMPANY_QUERIES:-}"
wait_seconds="${MT5_DIALOG_SEARCH_WAIT_SECONDS:-6}"

wid="$(DISPLAY="$display" xdotool search --onlyvisible --name "Select a company" 2>/dev/null | head -n1 || true)"
if [ -z "$wid" ]; then
  exit 0
fi

if [ -z "$query_list" ]; then
  query_list="$server"
fi

while IFS= read -r query; do
  [ -n "$query" ] || continue
  wid="$(DISPLAY="$display" xdotool search --onlyvisible --name "Select a company" 2>/dev/null | head -n1 || true)"
  [ -n "$wid" ] || exit 0

  DISPLAY="$display" xdotool windowactivate --sync "$wid" >/dev/null 2>&1 || true
  sleep 0.2

  WIDTH=""
  HEIGHT=""
  eval "$(DISPLAY="$display" xdotool getwindowgeometry --shell "$wid" 2>/dev/null | grep -E "^(WIDTH|HEIGHT)=")" || true
  WIDTH="${WIDTH:-700}"
  HEIGHT="${HEIGHT:-520}"

  search_x=$(( WIDTH / 3 ))
  search_y=$(( HEIGHT / 5 ))
  find_x=$(( WIDTH - 75 ))
  find_y="$search_y"
  result_x=$(( WIDTH / 3 ))
  result_y=$(( HEIGHT / 3 ))
  result_link_x=$(( WIDTH - 70 ))
  next_x=$(( WIDTH - 130 ))
  next_y=$(( HEIGHT - 28 ))

  [ "$search_x" -lt 120 ] && search_x=120
  [ "$search_y" -lt 80 ] && search_y=80
  [ "$find_x" -lt 220 ] && find_x=220
  [ "$result_x" -lt 150 ] && result_x=150
  [ "$result_y" -lt 140 ] && result_y=140
  [ "$result_link_x" -lt 260 ] && result_link_x=260
  [ "$next_x" -lt 220 ] && next_x=220
  [ "$next_y" -lt 220 ] && next_y=220

  # Focus company search input.
  DISPLAY="$display" xdotool mousemove --window "$wid" "$search_x" "$search_y" click 1 >/dev/null 2>&1 || true
  sleep 0.2
  DISPLAY="$display" xdotool key --window "$wid" --clearmodifiers ctrl+a BackSpace >/dev/null 2>&1 || true
  sleep 0.1
  DISPLAY="$display" xdotool type --window "$wid" --delay 1 "$query" >/dev/null 2>&1 || true
  sleep 0.2

  # Trigger "Find your company" using both click and keyboard fallback.
  DISPLAY="$display" xdotool mousemove --window "$wid" "$find_x" "$find_y" click 1 >/dev/null 2>&1 || true
  DISPLAY="$display" xdotool key --window "$wid" --clearmodifiers Return >/dev/null 2>&1 || true
  DISPLAY="$display" xdotool key --window "$wid" --clearmodifiers Tab Return >/dev/null 2>&1 || true
  sleep "$wait_seconds"

  # Try selecting first result row to seed broker/company into terminal cache.
  DISPLAY="$display" xdotool mousemove --window "$wid" "$result_x" "$result_y" click 1 >/dev/null 2>&1 || true
  sleep 0.2
  DISPLAY="$display" xdotool mousemove --window "$wid" "$result_link_x" "$result_y" click 1 >/dev/null 2>&1 || true
  sleep 0.2
  DISPLAY="$display" xdotool key --window "$wid" --clearmodifiers Down >/dev/null 2>&1 || true
  sleep 0.1
  DISPLAY="$display" xdotool key --window "$wid" --clearmodifiers Return >/dev/null 2>&1 || true
  DISPLAY="$display" xdotool key --window "$wid" --clearmodifiers Alt+n >/dev/null 2>&1 || true
  DISPLAY="$display" xdotool mousemove --window "$wid" "$next_x" "$next_y" click 1 >/dev/null 2>&1 || true
  sleep 0.8

  # Stop early if the dialog disappeared after selection.
  still_open="$(DISPLAY="$display" xdotool search --onlyvisible --name "Select a company" 2>/dev/null | head -n1 || true)"
  if [ -z "$still_open" ]; then
    exit 0
  fi
done <<EOF
$query_list
EOF

wid="$(DISPLAY="$display" xdotool search --onlyvisible --name "Select a company" 2>/dev/null | head -n1 || true)"
if [ -n "$wid" ]; then
  # Close dialog if it is still blocking.
  DISPLAY="$display" xdotool key --window "$wid" --clearmodifiers Escape >/dev/null 2>&1 || true
  DISPLAY="$display" xdotool key --window "$wid" --clearmodifiers Alt+F4 >/dev/null 2>&1 || true
fi
'
}

login_mt5_account_via_ui() {
  local login_text="${1:-}"
  local password_text="${2:-}"
  local server_name="${3:-}"

  if [[ -z "$login_text" || -z "$password_text" ]]; then
    return 0
  fi
  if [[ -z "$server_name" ]]; then
    echo "  warning: skip MT5 UI login fallback because server is blank."
    return 1
  fi
  if ! ensure_xdotool; then
    return 1
  fi

  compose exec -T -u abc \
    -e MT5_LOGIN="$login_text" \
    -e MT5_PASSWORD="$password_text" \
    -e MT5_SERVER="$server_name" \
    "$SERVICE_NAME" sh -lc '
display=":1"
login="${MT5_LOGIN:-}"
password="${MT5_PASSWORD:-}"
server="${MT5_SERVER:-}"

main_wid="$(DISPLAY="$display" xdotool search --onlyvisible --name "MetaTrader|Netting" 2>/dev/null | head -n1 || true)"
if [ -z "$main_wid" ]; then
  exit 1
fi

DISPLAY="$display" xdotool windowactivate --sync "$main_wid" >/dev/null 2>&1 || true
DISPLAY="$display" xdotool key --window "$main_wid" --clearmodifiers alt+f >/dev/null 2>&1 || true
sleep 0.25
DISPLAY="$display" xdotool key --window "$main_wid" --clearmodifiers l >/dev/null 2>&1 || true
sleep 1

login_wid="$(DISPLAY="$display" xdotool search --onlyvisible --name "Login|Authorization|Trade Account" 2>/dev/null | head -n1 || true)"
if [ -z "$login_wid" ]; then
  exit 1
fi

DISPLAY="$display" xdotool windowactivate --sync "$login_wid" >/dev/null 2>&1 || true
sleep 0.2
DISPLAY="$display" xdotool key --window "$login_wid" --clearmodifiers ctrl+a BackSpace >/dev/null 2>&1 || true
DISPLAY="$display" xdotool type --window "$login_wid" --delay 1 "$login" >/dev/null 2>&1 || true
DISPLAY="$display" xdotool key --window "$login_wid" --clearmodifiers Tab >/dev/null 2>&1 || true
sleep 0.1
DISPLAY="$display" xdotool key --window "$login_wid" --clearmodifiers ctrl+a BackSpace >/dev/null 2>&1 || true
DISPLAY="$display" xdotool type --window "$login_wid" --delay 1 "$password" >/dev/null 2>&1 || true
DISPLAY="$display" xdotool key --window "$login_wid" --clearmodifiers Tab >/dev/null 2>&1 || true
sleep 0.1
DISPLAY="$display" xdotool key --window "$login_wid" --clearmodifiers ctrl+a BackSpace >/dev/null 2>&1 || true
if [ -n "$server" ]; then
  DISPLAY="$display" xdotool type --window "$login_wid" --delay 1 "$server" >/dev/null 2>&1 || true
fi

# Save password checkbox + Login button.
DISPLAY="$display" xdotool key --window "$login_wid" --clearmodifiers Tab >/dev/null 2>&1 || true
sleep 0.1
DISPLAY="$display" xdotool key --window "$login_wid" --clearmodifiers space >/dev/null 2>&1 || true
DISPLAY="$display" xdotool key --window "$login_wid" --clearmodifiers Tab >/dev/null 2>&1 || true
sleep 0.1
DISPLAY="$display" xdotool key --window "$login_wid" --clearmodifiers Return >/dev/null 2>&1 || true
'
}

wait_for_mt5_main_window() {
  if ! ensure_xdotool; then
    return 1
  fi

  for attempt in $(seq 1 30); do
    if compose exec -T -u abc "$SERVICE_NAME" sh -lc \
      'DISPLAY=":1" xdotool search --onlyvisible --name "MetaTrader|MetaQuotes" >/dev/null 2>&1'; then
      return 0
    fi
    echo "  waiting MT5 window... ${attempt}/30"
    sleep 2
  done

  return 1
}

restart_mt5_terminal_for_experts_config() {
  compose exec -T -u abc "$SERVICE_NAME" sh -lc '
display=":1"
mt5exe="/config/.wine/drive_c/Program Files/MetaTrader 5/terminal64.exe"

pkill -f "terminal64.exe" >/dev/null 2>&1 || true
sleep 2

if [ ! -f "$mt5exe" ]; then
  exit 1
fi

nohup env DISPLAY="$display" WINEPREFIX="/config/.wine" wine "$mt5exe" >/tmp/mt5-restart.log 2>&1 &
' >/dev/null 2>&1 || return 1

  sleep 3
  wait_for_mt5_main_window
}

force_enable_algo_via_options_dialog() {
  if ! ensure_xdotool; then
    return 1
  fi

  compose exec -T -u abc "$SERVICE_NAME" sh -lc '
display=":1"

# Focus MT5 main window first.
main_wid="$(DISPLAY="$display" xdotool search --onlyvisible --name "MetaTrader|MetaQuotes" | head -n1 || true)"
if [ -z "$main_wid" ]; then
  exit 1
fi
DISPLAY="$display" xdotool windowactivate --sync "$main_wid" >/dev/null 2>&1 || true
sleep 0.3

# Open Options (Ctrl+O).
DISPLAY="$display" xdotool key --window "$main_wid" --clearmodifiers ctrl+o >/dev/null 2>&1 || true
sleep 1.2

# Get Options dialog if available.
opt_wid="$(DISPLAY="$display" xdotool search --onlyvisible --name "Options|MetaTrader|MetaQuotes" | tail -n1 || true)"
if [ -z "$opt_wid" ]; then
  opt_wid="$main_wid"
fi

DISPLAY="$display" xdotool windowactivate --sync "$opt_wid" >/dev/null 2>&1 || true
sleep 0.2

# Try selecting Expert Advisors tab (coordinates relative to options window).
DISPLAY="$display" xdotool mousemove --window "$opt_wid" 760 86 click 1 >/dev/null 2>&1 || true
sleep 0.25

# Fallback second tap for slightly different layout scaling.
DISPLAY="$display" xdotool mousemove --window "$opt_wid" 700 86 click 1 >/dev/null 2>&1 || true
sleep 0.25

# Toggle "Allow algorithmic trading" checkbox near top-left in Expert Advisors tab.
DISPLAY="$display" xdotool mousemove --window "$opt_wid" 74 132 click 1 >/dev/null 2>&1 || true
sleep 0.25

# Confirm dialog.
DISPLAY="$display" xdotool key --window "$opt_wid" --clearmodifiers Return >/dev/null 2>&1 || true
'
}

enable_algo_trading_if_needed() {
  echo "[2.5/6] Ensuring MT5 Algo Trading is enabled..."
  append_runner_progress "[2.5/6] Ensuring MT5 Algo Trading is enabled..."
  local probe_rc=0
  local actionable_rc=0
  for probe_attempt in $(seq 1 15); do
    if check_trade_allowed; then
      echo "  trade_allowed already enabled."
      append_runner_progress "trade_allowed already enabled."
      return 0
    fi
    probe_rc=$?
    if [[ "$probe_rc" -eq 5 ]]; then
      actionable_rc=5
      echo "  detected tradeapi_disabled=1 (external Python API blocked)."
      append_runner_progress "detected tradeapi_disabled=1 (external Python API blocked)."
      break
    fi
    if [[ "$probe_rc" -eq 1 ]]; then
      actionable_rc=1
      break
    fi
    echo "  MT5 trade API not ready yet (rc=${probe_rc}), retry probe ${probe_attempt}/15..."
    append_runner_progress "MT5 trade API not ready yet (rc=${probe_rc}), retry probe ${probe_attempt}/15..."
    sleep 2
  done

  if [[ "$actionable_rc" -eq 5 ]]; then
    echo "  restarting MT5 terminal once to apply Experts config..."
    append_runner_progress "restarting MT5 terminal once to apply Experts config..."
    if restart_mt5_terminal_for_experts_config; then
      dismiss_mt5_dialogs_retry 3 1 || true
      for retry_attempt in $(seq 1 10); do
        if check_trade_allowed; then
          echo "  Algo Trading enabled after MT5 restart."
          append_runner_progress "Algo Trading enabled after MT5 restart."
          return 0
        fi
        probe_rc=$?
        if [[ "$probe_rc" -ne 2 && "$probe_rc" -ne 3 ]]; then
          break
        fi
        sleep 2
      done
    else
      echo "  warning: MT5 restart attempt failed, fallback to UI toggle."
      append_runner_progress "warning: MT5 restart attempt failed, fallback to UI toggle."
    fi
  fi

  echo "  MT5 trading gate still blocked. Trying automatic Ctrl+E toggle..."
  append_runner_progress "MT5 trading gate still blocked. Trying automatic Ctrl+E toggle..."
  if ! ensure_xdotool; then
    echo "  warning: cannot install/find xdotool, auto-toggle skipped."
    append_runner_progress "warning: cannot install/find xdotool, auto-toggle skipped."
    return 1
  fi
  if ! wait_for_mt5_main_window; then
    echo "  warning: MT5 UI window not ready, auto-toggle skipped for now."
    append_runner_progress "warning: MT5 UI window not ready, auto-toggle skipped for now."
    return 1
  fi

  for attempt in 1 2 3 4 5; do
    dismiss_mt5_dialogs || true

    compose exec -T -u abc "$SERVICE_NAME" sh -lc '
display=":1"
wid="$(DISPLAY="$display" xdotool search --onlyvisible --name "MetaTrader|MetaQuotes" | head -n1 || true)"
if [ -z "$wid" ]; then
  exit 0
fi
DISPLAY="$display" xdotool windowactivate --sync "$wid" >/dev/null 2>&1 || true
DISPLAY="$display" xdotool key --window "$wid" --clearmodifiers Escape >/dev/null 2>&1 || true
DISPLAY="$display" xdotool key --window "$wid" --clearmodifiers ctrl+e >/dev/null 2>&1 || true
'
    sleep 2
    if check_trade_allowed; then
      echo "  Algo Trading enabled."
      append_runner_progress "Algo Trading enabled."
      return 0
    fi
    echo "  attempt ${attempt}/5 did not enable yet, retrying..."
    append_runner_progress "attempt ${attempt}/5 did not enable yet, retrying..."
  done

  echo "  Ctrl+E toggle failed. Trying Options -> Expert Advisors fallback..."
  append_runner_progress "Ctrl+E toggle failed. Trying Options -> Expert Advisors fallback..."
  for fallback_attempt in 1 2 3; do
    dismiss_mt5_dialogs || true
    force_enable_algo_via_options_dialog || true
    sleep 2
    if check_trade_allowed; then
      echo "  Algo Trading enabled via Options dialog."
      append_runner_progress "Algo Trading enabled via Options dialog."
      return 0
    fi
    echo "  options fallback attempt ${fallback_attempt}/3 did not enable yet."
    append_runner_progress "options fallback attempt ${fallback_attempt}/3 did not enable yet."
  done

  echo "  warning: auto-toggle failed. MT5 may have a modal dialog open in VNC."
  append_runner_progress "warning: auto-toggle failed. MT5 may have a modal dialog open in VNC."
  return 1
}

stabilize_algo_trading_after_start() {
  local stabilize_seconds_raw="${MT5_ALGO_STABILIZE_SECONDS:-20}"
  local stabilize_seconds
  local interval_seconds=2
  local elapsed=0
  local rc=0

  stabilize_seconds="$(printf '%s' "$stabilize_seconds_raw" | tr -dc '0-9')"
  if [[ -z "$stabilize_seconds" ]]; then
    stabilize_seconds=20
  fi
  if (( stabilize_seconds <= 0 )); then
    return 0
  fi

  echo "[6.65/6] Stabilizing MT5 Algo Trading for ${stabilize_seconds}s..."
  append_runner_progress "[6.65/6] Stabilizing MT5 Algo Trading for ${stabilize_seconds}s..."

  while (( elapsed < stabilize_seconds )); do
    if check_trade_allowed; then
      :
    else
      rc=$?
      if [[ "$rc" -eq 1 || "$rc" -eq 5 ]]; then
        echo "  trade gate dropped during stabilization (rc=${rc}), re-enabling..."
        append_runner_progress "trade gate dropped during stabilization (rc=${rc}), re-enabling..."
        if ! enable_algo_trading_if_needed; then
          return 1
        fi
      fi
    fi
    sleep "$interval_seconds"
    elapsed=$((elapsed + interval_seconds))
  done

  return 0
}

trade_allowed_confirmed() {
  local confirm_round
  for confirm_round in 1 2; do
    if ! check_trade_allowed; then
      return 1
    fi
    sleep 1
  done
  return 0
}

trade_gate_definitively_off() {
  local attempt
  local rc=0
  local disabled_hits=0
  for attempt in 1 2 3; do
    if check_trade_allowed; then
      return 1
    fi
    rc=$?
    if [[ "$rc" -eq 1 || "$rc" -eq 5 ]]; then
      disabled_hits=$((disabled_hits + 1))
      if (( disabled_hits >= 2 )); then
        return 0
      fi
    fi
    sleep 1
  done
  return 1
}

force_set_algo_trading_on() {
  local cycle=0

  # Fast path: avoid UI toggle if trade gate is already enabled.
  if trade_allowed_confirmed; then
    return 0
  fi

  if ! ensure_xdotool; then
    if trade_allowed_confirmed; then
      return 0
    fi
    return 1
  fi
  if ! wait_for_mt5_main_window; then
    if trade_allowed_confirmed; then
      return 0
    fi
    return 1
  fi

  for cycle in 1 2 3; do
    if trade_allowed_confirmed; then
      return 0
    fi

    compose exec -T -u abc "$SERVICE_NAME" sh -lc '
display=":1"
wid="$(DISPLAY="$display" xdotool search --onlyvisible --name "MetaTrader|MetaQuotes" | head -n1 || true)"
if [ -z "$wid" ]; then
  exit 1
fi
DISPLAY="$display" xdotool windowactivate --sync "$wid" >/dev/null 2>&1 || true
DISPLAY="$display" xdotool key --window "$wid" --clearmodifiers ctrl+e >/dev/null 2>&1 || true
' || true
    sleep 1

    if trade_allowed_confirmed; then
      return 0
    fi

    compose exec -T -u abc "$SERVICE_NAME" sh -lc '
display=":1"
wid="$(DISPLAY="$display" xdotool search --onlyvisible --name "MetaTrader|MetaQuotes" | head -n1 || true)"
if [ -z "$wid" ]; then
  exit 1
fi
DISPLAY="$display" xdotool windowactivate --sync "$wid" >/dev/null 2>&1 || true
DISPLAY="$display" xdotool key --window "$wid" --clearmodifiers ctrl+e >/dev/null 2>&1 || true
' || true
    sleep 1
  done

  return 1
}

ensure_experts_common_ini() {
  echo "[2.45/6] Ensuring MT5 Experts config in common.ini..."
  append_runner_progress "[2.45/6] Ensuring MT5 Experts config in common.ini..."
  local patch_output=""
  patch_output="$(
    compose exec -T \
    -e MT5_COMMON_INI_WAIT_SECONDS="${MT5_COMMON_INI_WAIT_SECONDS:-120}" \
    -e MT5_FORCE_ALGO_TRADING="${MT5_FORCE_ALGO_TRADING:-1}" \
    -e MT5_EXPERTS_DISABLE_ON_ACCOUNT_CHANGE="${MT5_EXPERTS_DISABLE_ON_ACCOUNT_CHANGE:-0}" \
    -e MT5_EXPERTS_DISABLE_ON_PROFILE_CHANGE="${MT5_EXPERTS_DISABLE_ON_PROFILE_CHANGE:-0}" \
    -e MT5_EXPERTS_DISABLE_ON_CHART_CHANGE="${MT5_EXPERTS_DISABLE_ON_CHART_CHANGE:-0}" \
    -e MT5_EXPERTS_DISABLE_VIA_PYTHON_API="${MT5_EXPERTS_DISABLE_VIA_PYTHON_API:-0}" \
    -e MT5_ALLOW_DLL_IMPORTS="${MT5_ALLOW_DLL_IMPORTS:-1}" \
    -e MT5_ALLOW_WEBREQUEST="${MT5_ALLOW_WEBREQUEST:-1}" \
    "$SERVICE_NAME" python3 - <<'PY'
import configparser
import io
import os
import sys
import time
from pathlib import Path

COMMON_INI = Path("/config/.wine/drive_c/Program Files/MetaTrader 5/Config/common.ini")

wait_seconds = int(os.getenv("MT5_COMMON_INI_WAIT_SECONDS", "120") or "120")
deadline = time.time() + max(wait_seconds, 0)
while not COMMON_INI.exists() and time.time() < deadline:
    time.sleep(2)

if not COMMON_INI.exists():
    print("warning: common.ini not found, skip Experts config patch")
    sys.exit(1)


def as_flag(name: str, default: str = "0") -> str:
    raw = str(os.getenv(name, default) or "").strip().lower()
    return "1" if raw in {"1", "true", "yes", "on"} else "0"


try:
    text = COMMON_INI.read_text(encoding="utf-16")
except Exception as exc:
    print(f"warning: failed to read common.ini ({exc})")
    sys.exit(1)

cfg = configparser.ConfigParser(interpolation=None, strict=False)
cfg.optionxform = str
cfg.read_string(text.lstrip("\ufeff"))

if not cfg.has_section("Experts"):
    cfg.add_section("Experts")

cfg.set("Experts", "Enabled", as_flag("MT5_FORCE_ALGO_TRADING", "1"))
cfg.set("Experts", "Account", as_flag("MT5_EXPERTS_DISABLE_ON_ACCOUNT_CHANGE", "0"))
cfg.set("Experts", "Profile", as_flag("MT5_EXPERTS_DISABLE_ON_PROFILE_CHANGE", "0"))
cfg.set("Experts", "Chart", as_flag("MT5_EXPERTS_DISABLE_ON_CHART_CHANGE", "0"))
cfg.set("Experts", "Api", as_flag("MT5_EXPERTS_DISABLE_VIA_PYTHON_API", "0"))

allow_dll_raw = str(os.getenv("MT5_ALLOW_DLL_IMPORTS", "") or "").strip()
if allow_dll_raw:
    cfg.set("Experts", "AllowDllImport", "1" if allow_dll_raw.lower() in {"1", "true", "yes", "on"} else "0")

allow_webrequest_raw = str(os.getenv("MT5_ALLOW_WEBREQUEST", "") or "").strip()
if allow_webrequest_raw:
    cfg.set("Experts", "WebRequest", "1" if allow_webrequest_raw.lower() in {"1", "true", "yes", "on"} else "0")

buf = io.StringIO()
cfg.write(buf, space_around_delimiters=False)
COMMON_INI.write_text(buf.getvalue(), encoding="utf-16")

print(
    "experts_config:"
    f" Enabled={cfg.get('Experts', 'Enabled', fallback='')}"
    f" Account={cfg.get('Experts', 'Account', fallback='')}"
    f" Profile={cfg.get('Experts', 'Profile', fallback='')}"
    f" Chart={cfg.get('Experts', 'Chart', fallback='')}"
    f" Api={cfg.get('Experts', 'Api', fallback='')}"
    f" AllowDllImport={cfg.get('Experts', 'AllowDllImport', fallback='')}"
    f" WebRequest={cfg.get('Experts', 'WebRequest', fallback='')}"
)
PY
  )"
  if [[ -n "$patch_output" ]]; then
    printf '%s\n' "$patch_output"
    while IFS= read -r line; do
      [[ -n "$line" ]] && append_runner_progress "$line"
    done <<< "$patch_output"
  fi
}

bot_process_is_running() {
  compose exec -T "$SERVICE_NAME" bash -lc "
set -euo pipefail
script='/bots/${BOT_SCRIPT}'
for proc in /proc/[0-9]*; do
  [ -r \"\$proc/cmdline\" ] || continue
  if tr '\0' '\n' < \"\$proc/cmdline\" | grep -Fxq \"\$script\"; then
    exit 0
  fi
done
exit 1
"
}

wait_for_bot_ws_registration() {
  local timeout_seconds="${BOT_WS_READY_TIMEOUT_SECONDS:-120}"
  timeout_seconds="$(printf '%s' "$timeout_seconds" | tr -dc '0-9')"
  if [[ -z "$timeout_seconds" ]]; then
    timeout_seconds=120
  fi
  if (( timeout_seconds <= 0 )); then
    return 0
  fi

  local elapsed=0
  while (( elapsed < timeout_seconds )); do
    local ws_line=""
    ws_line="$(
      compose exec -T "$SERVICE_NAME" sh -lc \
        "grep -E '\\[WS\\] registered|WS registered with BotHub' '${BOT_LOG}' 2>/dev/null | tail -n 1"
    )" || true
    if [[ -n "$ws_line" ]]; then
      echo "  bot websocket registered."
      append_runner_progress "bot websocket registered."
      return 0
    fi

    if ! bot_process_is_running >/dev/null 2>&1; then
      echo "  bot process exited while waiting websocket registration."
      append_runner_progress "bot process exited while waiting websocket registration."
      return 1
    fi

    sleep 5
    elapsed=$((elapsed + 5))
    echo "  waiting bot websocket register... ${elapsed}s/${timeout_seconds}s"
    append_runner_progress "waiting bot websocket register... ${elapsed}s/${timeout_seconds}s"
  done

  return 1
}

ensure_bot_log_relay() {
  if ! is_truthy "$BOT_STREAM_LOG_TO_CONTAINER_STDOUT"; then
    return 0
  fi

  local relay_tag relay_name
  relay_tag="$(printf '%s' "$BOT_LOG" | tr '[:space:]/:.' '_' | tr -cd '[:alnum:]_-')"
  if [[ -z "$relay_tag" ]]; then
    relay_tag="botlog"
  fi
  relay_name="mt5-bot-log-relay-${relay_tag}"

  compose exec -T \
    -e BOT_LOG_PATH="$BOT_LOG" \
    -e BOT_LOG_RELAY_NAME="$relay_name" \
    "$SERVICE_NAME" bash -lc '
set -euo pipefail
log_path="${BOT_LOG_PATH:-/config/bot.log}"
relay_name="${BOT_LOG_RELAY_NAME:-mt5-bot-log-relay}"

# Avoid duplicate relays for the same bot log.
pkill -f "$relay_name" >/dev/null 2>&1 || true

touch "$log_path"
nohup bash -lc "exec -a \"$relay_name\" tail -n 0 -F \"$log_path\" >> /proc/1/fd/1 2>> /proc/1/fd/2" >/dev/null 2>&1 &
'
}

append_bot_log_marker() {
  local marker="${1:-}"
  if [[ -z "$marker" ]]; then
    return 0
  fi
  compose exec -T "$SERVICE_NAME" sh -lc \
    "printf '\n%s\n' \"${marker}\" >> '${BOT_LOG}'" >/dev/null 2>&1 || true
}

BOT_LOG_RELAY_READY=0

enable_bot_log_stream_if_needed() {
  if [[ "$BOT_LOG_RELAY_READY" -eq 1 ]]; then
    return 0
  fi
  if ! is_truthy "$BOT_STREAM_LOG_TO_CONTAINER_STDOUT"; then
    return 0
  fi

  if ensure_bot_log_relay; then
    BOT_LOG_RELAY_READY=1
    append_bot_log_marker "[RUNNER] bot_log_stream_enabled"
    return 0
  fi
  return 1
}

append_runner_progress() {
  local message="$*"
  if [[ -z "$message" ]]; then
    return 0
  fi
  if ! is_truthy "$RUNNER_PROGRESS_TO_CONTAINER_LOGS"; then
    return 0
  fi
  append_bot_log_marker "[RUNNER] ${message}"
}

record_bot_runtime_version_state() {
  local current_tag current_script
  local state_file state_dir
  local prev_tag="" prev_script=""

  current_tag="$(trim_outer_whitespace "${BOT_VERSION_TAG:-}")"
  if [[ -z "$current_tag" ]]; then
    current_tag="base"
  fi
  current_script="/bots/${BOT_SCRIPT}"

  state_file="${BOT_RUNTIME_VERSION_STATE_FILE:-}"
  if [[ -z "$state_file" ]]; then
    state_file="${LIVE_STATE_FILE:-/instances/runtime_state.json}"
    state_dir="$(dirname "$state_file")"
    state_file="${state_dir}/runner_version_state.env"
  else
    state_dir="$(dirname "$state_file")"
  fi

  if [[ -f "$state_file" ]]; then
    # shellcheck disable=SC1090
    source "$state_file" || true
    prev_tag="${RUNNER_BOT_VERSION_TAG:-}"
    prev_script="${RUNNER_BOT_SCRIPT_PATH:-}"
  fi

  append_runner_progress "bot_runtime_version current_tag=${current_tag} current_script=${current_script} image=${METATRADER_IMAGE}"
  if [[ -z "$prev_tag" && -z "$prev_script" ]]; then
    append_bot_log_marker "[RUNNER] bot_version_initialized tag=${current_tag} script=${current_script}"
  elif [[ "$prev_tag" != "$current_tag" || "$prev_script" != "$current_script" ]]; then
    append_bot_log_marker "[RUNNER] bot_version_changed from_tag=${prev_tag:-unknown} to_tag=${current_tag} from_script=${prev_script:-unknown} to_script=${current_script}"
  else
    append_bot_log_marker "[RUNNER] bot_version_unchanged tag=${current_tag} script=${current_script}"
  fi

  mkdir -p "$state_dir" >/dev/null 2>&1 || true
  {
    printf "RUNNER_BOT_VERSION_TAG=%q\n" "$current_tag"
    printf "RUNNER_BOT_SCRIPT_PATH=%q\n" "$current_script"
  } > "$state_file" 2>/dev/null || true
}

ensure_image_ready

echo "[1/6] Starting MT5 container..."
compose up -d --no-build "$SERVICE_NAME"
if ! enable_bot_log_stream_if_needed; then
  echo "  warning: failed to enable runner progress stream to Docker logs."
fi
append_runner_progress "[1/6] Starting MT5 container..."
append_runner_progress "runtime_image=${METATRADER_IMAGE} bot_version_tag=${BOT_VERSION_TAG:-base}"

echo "[2/6] Waiting for mt5linux server (port 8001)..."
echo "  note: first startup can take 10-30 minutes while Wine Python packages are installed."
append_runner_progress "[2/6] Waiting for mt5linux server (port 8001)..."
append_runner_progress "note: first startup can take 10-30 minutes while Wine Python packages are installed."
elapsed=0
last_status=""

until compose exec -T "$SERVICE_NAME" python3 -c "import socket,sys;s=socket.socket();s.settimeout(1);rc=s.connect_ex(('127.0.0.1',8001));s.close();sys.exit(0 if rc==0 else 1)" >/dev/null 2>&1; do
  status_line="$(compose exec -T "$SERVICE_NAME" sh -lc "grep -E 'Starting MT5 startup script|\\[[1-7]/7\\]|The mt5linux server is running' /config/startup.log 2>/dev/null | tail -n 1" 2>/dev/null || true)"

  if [[ -n "$status_line" && "$status_line" != "$last_status" ]]; then
    echo "  startup: $status_line"
    append_runner_progress "startup: $status_line"
    last_status="$status_line"

    if [[ "$status_line" == *"[6/7] Installing Python libraries"* ]]; then
      echo "  info: step [6/7] may take several minutes on Apple Silicon emulation."
      append_runner_progress "info: step [6/7] may take several minutes on Apple Silicon emulation."
    fi
  fi

  if (( elapsed >= WAIT_TIMEOUT_SECONDS )); then
    echo "Error: timeout waiting for mt5linux server."
    append_runner_progress "Error: timeout waiting for mt5linux server."
    echo "Recent startup log:"
    compose exec -T "$SERVICE_NAME" sh -lc "tail -n 80 /config/startup.log" || true
    exit 1
  fi

  sleep "$WAIT_INTERVAL_SECONDS"
  elapsed=$((elapsed + WAIT_INTERVAL_SECONDS))
  echo "  waiting... ${elapsed}s/${WAIT_TIMEOUT_SECONDS}s"
  append_runner_progress "waiting... ${elapsed}s/${WAIT_TIMEOUT_SECONDS}s"
done
echo "  mt5linux server is ready."
append_runner_progress "mt5linux server is ready."
dismiss_mt5_dialogs || true
if [[ ! "$MT5_SKIP_PRECHECKS" =~ ^(1|true|yes|on)$ ]]; then
  mt5_login_ready=0
  if ! ensure_mt5_login_if_configured; then
    echo "  warning: MT5 API login precheck failed, trying UI login fallback."
    append_runner_progress "warning: MT5 API login precheck failed, trying UI login fallback."
    search_company_dialog_by_server "$MT5_SERVER_VAL" || true
    dismiss_mt5_dialogs || true
    login_mt5_account_via_ui "$MT5_LOGIN_VAL" "$MT5_PASSWORD_VAL" "$MT5_SERVER_VAL" || true
    dismiss_mt5_dialogs || true
    if ensure_mt5_login_if_configured; then
      mt5_login_ready=1
    fi
  else
    mt5_login_ready=1
  fi

  if [[ "$mt5_login_ready" -ne 1 ]]; then
    if is_truthy "$MT5_ALLOW_PARTIAL_START"; then
      echo "  warning: MT5 login precheck still failed, continuing because MT5_ALLOW_PARTIAL_START=$MT5_ALLOW_PARTIAL_START"
      append_runner_progress "warning: MT5 login precheck still failed, continuing because MT5_ALLOW_PARTIAL_START=$MT5_ALLOW_PARTIAL_START"
    else
      echo "Error: MT5 login precheck failed and fallback login did not succeed."
      append_runner_progress "Error: MT5 login precheck failed and fallback login did not succeed."
      echo "       Set MT5_ALLOW_PARTIAL_START=1 only if you intentionally want best-effort start."
      exit 1
    fi
  fi
else
  echo "[2.4/6] Skipping MT5 API prechecks (MT5_SKIP_PRECHECKS=$MT5_SKIP_PRECHECKS)."
  append_runner_progress "[2.4/6] Skipping MT5 API prechecks (MT5_SKIP_PRECHECKS=$MT5_SKIP_PRECHECKS)."
fi
ensure_experts_common_ini || true
if [[ ! "$MT5_SKIP_PRECHECKS" =~ ^(1|true|yes|on)$ ]]; then
  if ! enable_algo_trading_if_needed; then
    if is_truthy "$MT5_ALLOW_PARTIAL_START"; then
      echo "  warning: MT5 Algo Trading precheck failed, continuing because MT5_ALLOW_PARTIAL_START=$MT5_ALLOW_PARTIAL_START"
      append_runner_progress "warning: MT5 Algo Trading precheck failed, continuing because MT5_ALLOW_PARTIAL_START=$MT5_ALLOW_PARTIAL_START"
    else
      echo "Error: MT5 Algo Trading is still disabled after auto-toggle attempts."
      append_runner_progress "Error: MT5 Algo Trading is still disabled after auto-toggle attempts."
      echo "       Set MT5_ALLOW_PARTIAL_START=1 only if you intentionally want best-effort start."
      exit 1
    fi
  fi
else
  echo "[2.5/6] Skipping trade_allowed probe/toggle precheck."
  append_runner_progress "[2.5/6] Skipping trade_allowed probe/toggle precheck."
fi

echo "[3/6] Installing bot dependencies (only when requirements changed)..."
append_runner_progress "[3/6] Installing bot dependencies (only when requirements changed)..."
step3_output="$(
compose exec -T \
  -e USE_SHARED_PYDEPS="${USE_SHARED_PYDEPS}" \
  -e SHARED_PYDEPS_DIR="${SHARED_PYDEPS_DIR}" \
  "$SERVICE_NAME" bash -lc '
set -euo pipefail

req_file="'"${BOT_REQUIREMENTS}"'"
if [ ! -f "$req_file" ]; then
  if [ -f /bots/requirements.txt ]; then
    req_file="/bots/requirements.txt"
    echo "  warning: /bots/requirements-live.txt not found, fallback to /bots/requirements.txt"
  else
    echo "Error: requirements file not found: $req_file"
    exit 1
  fi
fi

echo "  using requirements: $req_file"

shared_pydeps_dir="${SHARED_PYDEPS_DIR:-/shared-pydeps}"
use_shared_pydeps_lc="$(printf "%s" "${USE_SHARED_PYDEPS:-1}" | tr "[:upper:]" "[:lower:]")"
python_cmd="python3"
pip_mode="system"
pip_install_base=(python3 -m pip install --break-system-packages --no-cache-dir)
stamp_dir="/config/.requirements-cache"
venv_recreated=0

case "$use_shared_pydeps_lc" in
  1|true|yes|on)
    mkdir -p "$shared_pydeps_dir"
    shared_venv_dir="$shared_pydeps_dir/venv-py311"
    if [ ! -x "$shared_venv_dir/bin/python" ]; then
      echo "  creating shared python venv: $shared_venv_dir"
      rm -rf "$shared_venv_dir"
      python3 -m venv "$shared_venv_dir"
      venv_recreated=1
    fi
    python_cmd="$shared_venv_dir/bin/python"
    # Ensure pip exists in the venv even if previous creation was interrupted.
    if ! "$python_cmd" -m pip --version >/dev/null 2>&1; then
      "$python_cmd" -m ensurepip --upgrade >/dev/null 2>&1 || true
      "$python_cmd" -m pip install --upgrade pip >/dev/null 2>&1 || true
    fi
    pip_mode="venv"
    pip_install_base=("$python_cmd" -m pip install --no-cache-dir)
    stamp_dir="$shared_pydeps_dir/.requirements-cache"
    ;;
  *)
    ;;
esac

if [ "$venv_recreated" = "1" ]; then
  echo "  shared venv recreated, clearing requirements cache"
  rm -rf "$stamp_dir"
fi

if [ -n "${PIP_CACHE_DIR:-}" ]; then
  export PIP_CACHE_DIR
else
  if [ "$pip_mode" = "system" ]; then
    export PIP_CACHE_DIR="/tmp/pip-cache"
  else
    export PIP_CACHE_DIR="$shared_pydeps_dir/.pip-cache"
  fi
fi
mkdir -p "$PIP_CACHE_DIR"
mkdir -p "$stamp_dir"

if [ "$pip_mode" = "venv" ]; then
  if ! "$python_cmd" -c "import mt5linux" >/dev/null 2>&1; then
    echo "  installing mt5linux runtime into shared venv ..."
    "$python_cmd" -m pip install --no-cache-dir --no-deps mt5linux==0.2.4
    "$python_cmd" -m pip install --no-cache-dir rpyc==5.2.3 plumbum==1.7.0 "pyparsing>=3.1.0,<4" numpy pyxdg pyzmq
  fi
fi

torch_target="'"${TORCH_VERSION}"'"
installed_torch="$("$python_cmd" -c "import importlib.util; spec=importlib.util.find_spec('torch'); print('' if spec is None else __import__('torch').__version__)" 2>/dev/null || true)"
if [[ "$installed_torch" != "$torch_target" && "$installed_torch" != "$torch_target+cpu" ]]; then
  echo "  installing torch CPU wheel $torch_target ..."
  "${pip_install_base[@]}" --index-url https://download.pytorch.org/whl/cpu "torch==$torch_target"
else
  echo "  torch $installed_torch already installed"
fi

if [ ! -f "$req_file" ]; then
  echo "Error: requirements file not found after fallback: $req_file"
  exit 1
fi

req_hash="$(sha256sum "$req_file" | awk "{print \$1}")"
stamp_file="${stamp_dir}/bot_requirements_${req_hash}"
if [ ! -f "$stamp_file" ]; then
  if ! "${pip_install_base[@]}" -r "$req_file"; then
    echo "  full requirements install failed, retry without git+ lines..."
    grep -Ev "^[[:space:]]*git\\+" "$req_file" > /tmp/bot_requirements_nogit.txt
    "${pip_install_base[@]}" -r /tmp/bot_requirements_nogit.txt
  fi
  touch "$stamp_file"
else
  echo "  requirements already installed (hash: $req_hash), skipping install"
fi
' 2>&1
)"
if [[ -n "${step3_output:-}" ]]; then
  printf '%s\n' "$step3_output"
  while IFS= read -r line; do
    [[ -n "$line" ]] && append_runner_progress "$line"
  done <<< "$step3_output"
fi

echo "[4/6] Stopping old bot process (if any)..."
append_runner_progress "[4/6] Stopping old bot process (if any)..."
step4_output="$(
compose exec -T "$SERVICE_NAME" bash -lc "
set -euo pipefail
script='/bots/${BOT_SCRIPT}'
pids=''
for proc in /proc/[0-9]*; do
  [ -r \"\$proc/cmdline\" ] || continue
  if tr '\0' '\n' < \"\$proc/cmdline\" | grep -Fxq \"\$script\"; then
    pids=\"\$pids \${proc##*/}\"
  fi
done
pids=\${pids# }
if [ -n \"\$pids\" ]; then
  echo \"  stopping existing bot pid(s): \$pids\"
  kill \$pids
fi
" 2>&1
)"
if [[ -n "${step4_output:-}" ]]; then
  printf '%s\n' "$step4_output"
  while IFS= read -r line; do
    [[ -n "$line" ]] && append_runner_progress "$line"
  done <<< "$step4_output"
fi

echo "[4.5/6] Re-checking MT5 Algo Trading before bot launch..."
append_runner_progress "[4.5/6] Re-checking MT5 Algo Trading before bot launch..."
ensure_experts_common_ini || true
if [[ ! "$MT5_SKIP_PRECHECKS" =~ ^(1|true|yes|on)$ ]]; then
  if ! enable_algo_trading_if_needed; then
    if is_truthy "$MT5_ALLOW_PARTIAL_START"; then
      echo "  warning: second MT5 Algo precheck failed, continuing because MT5_ALLOW_PARTIAL_START=$MT5_ALLOW_PARTIAL_START"
      append_runner_progress "warning: second MT5 Algo precheck failed, continuing because MT5_ALLOW_PARTIAL_START=$MT5_ALLOW_PARTIAL_START"
    else
      echo "Error: MT5 Algo Trading precheck failed before bot launch."
      append_runner_progress "Error: MT5 Algo Trading precheck failed before bot launch."
      echo "       Set MT5_ALLOW_PARTIAL_START=1 only if you intentionally want best-effort start."
      exit 1
    fi
  fi
else
  echo "[4.5/6] Skipping second MT5 API precheck."
  append_runner_progress "[4.5/6] Skipping second MT5 API precheck."
fi

echo "[5/6] Starting bot..."
append_runner_progress "[5/6] Starting bot..."
record_bot_runtime_version_state
bot_python_cmd="python3"
require_shared_python="0"
use_shared_pydeps_lc_host="$(printf '%s' "$USE_SHARED_PYDEPS" | tr '[:upper:]' '[:lower:]')"
case "$use_shared_pydeps_lc_host" in
  1|true|yes|on)
    bot_python_cmd="${SHARED_PYDEPS_DIR}/venv-py311/bin/python"
    require_shared_python="1"
    ;;
  *)
    ;;
esac

step5_output="$(
compose exec -T \
  -e MT5_LOGIN="$MT5_LOGIN_VAL" \
  -e MT5_PASSWORD="$MT5_PASSWORD_VAL" \
  -e MT5_SERVER="$MT5_SERVER_VAL" \
  -e LIVE_MAGIC_NUMBER="${LIVE_MAGIC_NUMBER:-}" \
  -e LIVE_MANAGE_MANUAL_POSITIONS="${LIVE_MANAGE_MANUAL_POSITIONS:-0}" \
  -e LIVE_MODELS_DIR="${LIVE_MODELS_DIR:-}" \
  -e VISION_LLM_EMBED_TEXT_API_URL="${VISION_LLM_EMBED_TEXT_API_URL:-}" \
  -e LIVE_LLM_SEMANTIC_CACHE_FILE="${LIVE_LLM_SEMANTIC_CACHE_FILE:-}" \
  -e LIVE_LLM_TEXT_LOG_FILE="${LIVE_LLM_TEXT_LOG_FILE:-}" \
  -e LIVE_PREWARM_SEMANTIC_ON_START="${LIVE_PREWARM_SEMANTIC_ON_START:-}" \
  -e LIVE_PREWARM_SEMANTIC_MAX_SECONDS="${LIVE_PREWARM_SEMANTIC_MAX_SECONDS:-}" \
  -e LIVE_PREWARM_SEMANTIC_MAX_MISSING="${LIVE_PREWARM_SEMANTIC_MAX_MISSING:-}" \
  -e LIVE_PREWARM_REQUEST_TIMEOUT_SEC="${LIVE_PREWARM_REQUEST_TIMEOUT_SEC:-}" \
  -e LIVE_CATCHUP_MAX_BARS="${LIVE_CATCHUP_MAX_BARS:-}" \
  -e LIVE_SEMANTIC_ALIAS_HOURS="${LIVE_SEMANTIC_ALIAS_HOURS:-}" \
  -e LIVE_SEMANTIC_NO_DATA_RETRY_SECONDS="${LIVE_SEMANTIC_NO_DATA_RETRY_SECONDS:-}" \
  -e LIVE_PERFORMANCE_SYNC_INTERVAL_SEC="${LIVE_PERFORMANCE_SYNC_INTERVAL_SEC:-}" \
  -e LIVE_PERFORMANCE_BOOT_LOOKBACK_DAYS="${LIVE_PERFORMANCE_BOOT_LOOKBACK_DAYS:-}" \
  -e LIVE_MT5_HISTORY_END_AHEAD_HOURS="${LIVE_MT5_HISTORY_END_AHEAD_HOURS:-}" \
  -e LIVE_PERFORMANCE_SCOPE="${LIVE_PERFORMANCE_SCOPE:-}" \
  -e LIVE_MANAGED_MAGIC_SET="${LIVE_MANAGED_MAGIC_SET:-}" \
  -e LIVE_PERFORMANCE_MAGIC_SET="${LIVE_PERFORMANCE_MAGIC_SET:-}" \
  -e LIVE_STATE_FILE="${LIVE_STATE_FILE:-}" \
  -e MT5_INIT_TIMEOUT="${MT5_INIT_TIMEOUT_VAL:-900}" \
  -e MT5_LOGIN_RETRIES="${MT5_LOGIN_RETRIES_VAL:-20}" \
  -e MT5_RETRY_SECONDS="${MT5_RETRY_SECONDS_VAL:-5}" \
  -e MT5_RPC_TIMEOUT_MS="${MT5_RPC_TIMEOUT_MS_VAL:-180000}" \
  "$SERVICE_NAME" \
  bash -lc "set -euo pipefail; py_cmd='${bot_python_cmd}'; require_shared='${require_shared_python}'; if [ ! -x \"\$py_cmd\" ]; then if [ \"\$require_shared\" = '1' ]; then echo \"Error: shared python not found at \$py_cmd\"; exit 1; fi; py_cmd='python3'; fi; echo \"  bot python: \$py_cmd\"; cd /bots && nohup env PYTHONUNBUFFERED=1 \"\$py_cmd\" -u /bots/${BOT_SCRIPT} > '${BOT_LOG}' 2>&1 &" 2>&1
)"
if [[ -n "${step5_output:-}" ]]; then
  printf '%s\n' "$step5_output"
  while IFS= read -r line; do
    [[ -n "$line" ]] && append_runner_progress "$line"
  done <<< "$step5_output"
fi

sleep 5
if is_truthy "$BOT_STREAM_LOG_TO_CONTAINER_STDOUT"; then
  echo "[5.5/6] Streaming bot log to Docker container logs..."
  append_runner_progress "[5.5/6] Streaming bot log to Docker container logs..."
  if enable_bot_log_stream_if_needed; then
    :
  else
    echo "  warning: failed to enable bot log stream."
    append_runner_progress "warning: failed to enable bot log stream."
  fi
fi
append_bot_log_marker "[RUNNER] bot_process_started"

echo "[6/6] Verifying bot process..."
append_runner_progress "[6/6] Verifying bot process..."
if bot_process_is_running; then
  if is_truthy "$BOT_WAIT_FOR_WS_REGISTER"; then
    echo "[6.5/6] Waiting for bot websocket registration..."
    append_runner_progress "[6.5/6] Waiting for bot websocket registration..."
    if ! wait_for_bot_ws_registration; then
      if is_truthy "$MT5_ALLOW_PARTIAL_START"; then
        echo "  warning: websocket registration not confirmed yet, continuing because MT5_ALLOW_PARTIAL_START=$MT5_ALLOW_PARTIAL_START"
        append_runner_progress "warning: websocket registration not confirmed yet, continuing because MT5_ALLOW_PARTIAL_START=$MT5_ALLOW_PARTIAL_START"
        append_bot_log_marker "[RUNNER] start_warn reason=ws_register_timeout"
      else
        echo "Error: bot websocket registration timed out (${BOT_WS_READY_TIMEOUT_SECONDS}s)."
        append_runner_progress "Error: bot websocket registration timed out (${BOT_WS_READY_TIMEOUT_SECONDS}s)."
        append_bot_log_marker "[RUNNER] start_failed reason=ws_register_timeout"
        echo "Recent bot log:"
        compose exec -T "$SERVICE_NAME" bash -lc "tail -n 120 '${BOT_LOG}'" || true
        exit 1
      fi
    fi
  fi
	  if [[ ! "$MT5_SKIP_PRECHECKS" =~ ^(1|true|yes|on)$ ]]; then
	    echo "[6.6/6] Final MT5 Algo Trading check after bot startup..."
	    append_runner_progress "[6.6/6] Final MT5 Algo Trading check after bot startup..."
    if ! check_trade_allowed; then
      echo "  trade_allowed dropped after bot startup. Retrying auto-enable..."
      append_runner_progress "trade_allowed dropped after bot startup. Retrying auto-enable..."
      if ! enable_algo_trading_if_needed; then
        if is_truthy "$MT5_ALLOW_PARTIAL_START"; then
          echo "  warning: final MT5 Algo check failed, continuing because MT5_ALLOW_PARTIAL_START=$MT5_ALLOW_PARTIAL_START"
          append_runner_progress "warning: final MT5 Algo check failed, continuing because MT5_ALLOW_PARTIAL_START=$MT5_ALLOW_PARTIAL_START"
          append_bot_log_marker "[RUNNER] start_warn reason=final_algo_check_failed"
        else
          echo "Error: final MT5 Algo Trading check failed after bot startup."
          append_runner_progress "Error: final MT5 Algo Trading check failed after bot startup."
          append_bot_log_marker "[RUNNER] start_failed reason=final_algo_check_failed"
          echo "Recent bot log:"
          compose exec -T "$SERVICE_NAME" bash -lc "tail -n 120 '${BOT_LOG}'" || true
          exit 1
        fi
	      fi
	    fi
      if ! stabilize_algo_trading_after_start; then
        if is_truthy "$MT5_ALLOW_PARTIAL_START"; then
          echo "  warning: MT5 Algo stabilization failed, continuing because MT5_ALLOW_PARTIAL_START=$MT5_ALLOW_PARTIAL_START"
          append_runner_progress "warning: MT5 Algo stabilization failed, continuing because MT5_ALLOW_PARTIAL_START=$MT5_ALLOW_PARTIAL_START"
          append_bot_log_marker "[RUNNER] start_warn reason=algo_stabilization_failed"
        else
          echo "Error: MT5 Algo Trading became unstable after startup."
          append_runner_progress "Error: MT5 Algo Trading became unstable after startup."
          append_bot_log_marker "[RUNNER] start_failed reason=algo_stabilization_failed"
          exit 1
        fi
      fi
      echo "[6.66/6] Forcing deterministic MT5 Algo ON state..."
      append_runner_progress "[6.66/6] Forcing deterministic MT5 Algo ON state..."
      if ! force_set_algo_trading_on; then
        if trade_gate_definitively_off; then
          if is_truthy "$MT5_ALLOW_PARTIAL_START"; then
            echo "  warning: deterministic MT5 Algo ON enforcement failed, continuing because MT5_ALLOW_PARTIAL_START=$MT5_ALLOW_PARTIAL_START"
            append_runner_progress "warning: deterministic MT5 Algo ON enforcement failed, continuing because MT5_ALLOW_PARTIAL_START=$MT5_ALLOW_PARTIAL_START"
            append_bot_log_marker "[RUNNER] start_warn reason=algo_force_on_failed"
          else
            echo "Error: unable to force MT5 Algo Trading ON state."
            append_runner_progress "Error: unable to force MT5 Algo Trading ON state."
            append_bot_log_marker "[RUNNER] start_failed reason=algo_force_on_failed"
            exit 1
          fi
        else
          echo "  warning: deterministic MT5 Algo ON enforcement was inconclusive, continuing because trade gate is not definitively OFF."
          append_runner_progress "warning: deterministic MT5 Algo ON enforcement was inconclusive, continuing because trade gate is not definitively OFF."
          append_bot_log_marker "[RUNNER] start_warn reason=algo_force_on_inconclusive"
        fi
      fi
	  fi
	  if is_truthy "$MT5_REFRESH_COMPANY_CACHE_AFTER_START" && [[ -n "$MT5_SERVER_VAL" ]]; then
	    echo "[6.7/6] Refreshing MT5 company cache after startup..."
	    append_runner_progress "[6.7/6] Refreshing MT5 company cache after startup..."
	    search_company_dialog_by_server "$MT5_SERVER_VAL" "1" || true
      dismiss_mt5_dialogs_retry 3 1 || true
      if [[ ! "$MT5_SKIP_PRECHECKS" =~ ^(1|true|yes|on)$ ]]; then
        echo "[6.8/6] Re-checking MT5 Algo Trading after company refresh..."
        append_runner_progress "[6.8/6] Re-checking MT5 Algo Trading after company refresh..."
        if ! enable_algo_trading_if_needed; then
          if is_truthy "$MT5_ALLOW_PARTIAL_START"; then
            echo "  warning: post-refresh MT5 Algo check failed, continuing because MT5_ALLOW_PARTIAL_START=$MT5_ALLOW_PARTIAL_START"
            append_runner_progress "warning: post-refresh MT5 Algo check failed, continuing because MT5_ALLOW_PARTIAL_START=$MT5_ALLOW_PARTIAL_START"
            append_bot_log_marker "[RUNNER] start_warn reason=post_refresh_algo_check_failed"
          else
            echo "Error: MT5 Algo Trading check failed after company refresh."
            append_runner_progress "Error: MT5 Algo Trading check failed after company refresh."
            append_bot_log_marker "[RUNNER] start_failed reason=post_refresh_algo_check_failed"
            exit 1
          fi
        fi
      fi
	  fi
	  dismiss_mt5_dialogs_retry 3 1 || true
	  append_bot_log_marker "[RUNNER] bot_version_active tag=${BOT_VERSION_TAG:-base} script=/bots/${BOT_SCRIPT} image=${METATRADER_IMAGE}"
	  append_bot_log_marker "[RUNNER] start_ok"
	  append_runner_progress "Bot started successfully."
	  echo "Bot started successfully."
  echo "Log file: ${BOT_LOG}"
  echo "Follow logs with:"
  echo "  docker compose exec -it ${SERVICE_NAME} tail -f ${BOT_LOG}"
else
  echo "Error: bot process not found, showing last log lines:"
  append_runner_progress "Error: bot process not found, showing last log lines."
  append_bot_log_marker "[RUNNER] start_failed reason=bot_process_missing"
  compose exec -T "$SERVICE_NAME" bash -lc "tail -n 50 '${BOT_LOG}'" || true
  exit 1
fi
