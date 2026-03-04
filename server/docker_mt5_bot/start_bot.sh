#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

SERVICE_NAME="${SERVICE_NAME:-metatrader5-macos}"
BOT_SCRIPT="${BOT_SCRIPT:-run_live.py}"
BOT_LOG="${BOT_LOG:-/config/bot.log}"
BOT_REQUIREMENTS="${BOT_REQUIREMENTS:-/bots/requirements-live.txt}"
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
MT5_SKIP_PRECHECKS="${MT5_SKIP_PRECHECKS:-1}"

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

# If VNC credentials are not explicitly configured, reuse MT5 credentials.
if [[ -z "$CUSTOM_USER_VAL" && -n "$MT5_LOGIN_VAL" ]]; then
  CUSTOM_USER_VAL="$MT5_LOGIN_VAL"
fi
if [[ -z "$VNC_PASSWORD_VAL" && -n "$MT5_PASSWORD_VAL" ]]; then
  VNC_PASSWORD_VAL="$MT5_PASSWORD_VAL"
fi
if [[ -z "$MT5_STRICT_SERVER_MATCH_VAL" ]]; then
  MT5_STRICT_SERVER_MATCH_VAL="0"
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
import os
import sys

try:
    mt5 = MetaTrader5(host="localhost", port=8001)
    login_text = os.getenv("MT5_LOGIN", "").strip()
    password = os.getenv("MT5_PASSWORD", "").strip()
    server = os.getenv("MT5_SERVER", "").strip()
    timeout_ms = int(os.getenv("MT5_RPC_TIMEOUT_MS", "180000"))

    init_kwargs = {"timeout": timeout_ms}
    if login_text and password:
        init_kwargs["login"] = int(login_text)
        init_kwargs["password"] = password
        if server:
            init_kwargs["server"] = server

    if not mt5.initialize(**init_kwargs):
        print("trade_allowed=0 tradeapi_disabled=unknown (initialize failed)")
        sys.exit(2)

    info = mt5.terminal_info()
    if info is None:
        print("trade_allowed=0 tradeapi_disabled=unknown (terminal_info is None)")
        sys.exit(2)
    trade_allowed = bool(getattr(info, "trade_allowed", False))
    tradeapi_disabled = bool(getattr(info, "tradeapi_disabled", False))
    print(f"trade_allowed={int(trade_allowed)} tradeapi_disabled={int(tradeapi_disabled)}")
    sys.exit(0 if trade_allowed else 1)
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
    return 0
  fi

  echo "[2.4/6] Ensuring MT5 account login via API..."
  dismiss_mt5_dialogs || true
  search_company_dialog_by_server "$MT5_SERVER_VAL" || true
  dismiss_mt5_dialogs || true
  cleanup_stale_mt5_exec_processes || true

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
        mt5 = MetaTrader5(host="localhost", port=8001)
        init_kwargs = {"timeout": rpc_timeout, "login": login_id, "password": pwd}
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
            print(f"login_attempt_{attempt}_server={server_name} result=timeout")
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
            f"login_attempt_{attempt}_server={result.get('server','')}"
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
  fi

  if [[ "$rc" -eq 124 ]]; then
    echo "  warning: MT5 login probe timed out (${MT5_LOGIN_CHECK_TIMEOUT_SECONDS}s)."
    cleanup_stale_mt5_exec_processes || true
    return 1
  fi

  local resolved_server
  resolved_server="$(printf '%s\n' "$login_output" | sed -n 's/^resolved_server=//p' | tail -n 1 | tr -d '\r')"
  if [[ -n "$resolved_server" && "$resolved_server" != "$MT5_SERVER_VAL" ]]; then
    echo "  resolved MT5 server alias: '$MT5_SERVER_VAL' -> '$resolved_server'"
    MT5_SERVER_VAL="$resolved_server"
    export MT5_SERVER="$MT5_SERVER_VAL"
  fi

  if [[ "$rc" -ne 0 ]]; then
    search_company_dialog_by_server "$MT5_SERVER_VAL" || true
    dismiss_mt5_dialogs || true
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
    DISPLAY="$display" xdotool key --window "$wid" --clearmodifiers Alt+F4 >/dev/null 2>&1 || true
  done
done
'
}

search_company_dialog_by_server() {
  local server_name="${1:-}"
  server_name="$(printf '%s' "$server_name" | sed -E 's/[[:space:]]+/ /g; s/^ //; s/ $//')"
  if [[ -z "$server_name" ]]; then
    return 0
  fi

  if ! ensure_xdotool; then
    return 1
  fi

  compose exec -T -u abc -e MT5_SERVER="$server_name" -e MT5_DIALOG_SEARCH_WAIT_SECONDS="$MT5_DIALOG_SEARCH_WAIT_SECONDS" "$SERVICE_NAME" sh -lc '
display=":1"
server="${MT5_SERVER:-}"
wait_seconds="${MT5_DIALOG_SEARCH_WAIT_SECONDS:-6}"

wid="$(DISPLAY="$display" xdotool search --onlyvisible --name "Select a company" 2>/dev/null | head -n1 || true)"
if [ -z "$wid" ]; then
  exit 0
fi

DISPLAY="$display" xdotool windowactivate --sync "$wid" >/dev/null 2>&1 || true
sleep 0.2

# Focus company search input.
DISPLAY="$display" xdotool mousemove --window "$wid" 280 132 click 1 >/dev/null 2>&1 || true
sleep 0.2
DISPLAY="$display" xdotool key --window "$wid" --clearmodifiers ctrl+a BackSpace >/dev/null 2>&1 || true
sleep 0.1
DISPLAY="$display" xdotool type --window "$wid" --delay 1 "$server" >/dev/null 2>&1 || true
sleep 0.2

# Click "Find your company".
DISPLAY="$display" xdotool mousemove --window "$wid" 820 132 click 1 >/dev/null 2>&1 || true
sleep "$wait_seconds"

# Close dialog if it is still blocking.
DISPLAY="$display" xdotool key --window "$wid" --clearmodifiers Escape >/dev/null 2>&1 || true
'
}

login_mt5_account_via_ui() {
  local login_text="${1:-}"
  local password_text="${2:-}"
  local server_name="${3:-}"

  if [[ -z "$login_text" || -z "$password_text" ]]; then
    return 0
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
  for probe_attempt in $(seq 1 15); do
    if check_trade_allowed; then
      echo "  trade_allowed already enabled."
      return 0
    fi
    probe_rc=$?
    if [[ "$probe_rc" -eq 1 ]]; then
      break
    fi
    echo "  MT5 trade API not ready yet (rc=${probe_rc}), retry probe ${probe_attempt}/15..."
    sleep 2
  done

  echo "  trade_allowed is OFF. Trying automatic Ctrl+E toggle..."
  if ! ensure_xdotool; then
    echo "  warning: cannot install/find xdotool, auto-toggle skipped."
    return 1
  fi
  if ! wait_for_mt5_main_window; then
    echo "  warning: MT5 UI window not ready, auto-toggle skipped for now."
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
      return 0
    fi
    echo "  attempt ${attempt}/5 did not enable yet, retrying..."
  done

  echo "  Ctrl+E toggle failed. Trying Options -> Expert Advisors fallback..."
  for fallback_attempt in 1 2 3; do
    dismiss_mt5_dialogs || true
    force_enable_algo_via_options_dialog || true
    sleep 2
    if check_trade_allowed; then
      echo "  Algo Trading enabled via Options dialog."
      return 0
    fi
    echo "  options fallback attempt ${fallback_attempt}/3 did not enable yet."
  done

  echo "  warning: auto-toggle failed. MT5 may have a modal dialog open in VNC."
  return 1
}

ensure_experts_common_ini() {
  echo "[2.45/6] Ensuring MT5 Experts config in common.ini..."
  compose exec -T \
    -e MT5_COMMON_INI_WAIT_SECONDS="${MT5_COMMON_INI_WAIT_SECONDS:-120}" \
    -e MT5_FORCE_ALGO_TRADING="${MT5_FORCE_ALGO_TRADING:-1}" \
    -e MT5_EXPERTS_DISABLE_ON_ACCOUNT_CHANGE="${MT5_EXPERTS_DISABLE_ON_ACCOUNT_CHANGE:-0}" \
    -e MT5_EXPERTS_DISABLE_ON_PROFILE_CHANGE="${MT5_EXPERTS_DISABLE_ON_PROFILE_CHANGE:-0}" \
    -e MT5_EXPERTS_DISABLE_ON_CHART_CHANGE="${MT5_EXPERTS_DISABLE_ON_CHART_CHANGE:-0}" \
    -e MT5_EXPERTS_DISABLE_VIA_PYTHON_API="${MT5_EXPERTS_DISABLE_VIA_PYTHON_API:-0}" \
    -e MT5_ALLOW_DLL_IMPORTS="${MT5_ALLOW_DLL_IMPORTS:-}" \
    -e MT5_ALLOW_WEBREQUEST="${MT5_ALLOW_WEBREQUEST:-}" \
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
}

ensure_image_ready

echo "[1/6] Starting MT5 container..."
compose up -d --no-build "$SERVICE_NAME"

echo "[2/6] Waiting for mt5linux server (port 8001)..."
echo "  note: first startup can take 10-30 minutes while Wine Python packages are installed."
elapsed=0
last_status=""

until compose exec -T "$SERVICE_NAME" python3 -c "import socket,sys;s=socket.socket();s.settimeout(1);rc=s.connect_ex(('127.0.0.1',8001));s.close();sys.exit(0 if rc==0 else 1)" >/dev/null 2>&1; do
  status_line="$(compose exec -T "$SERVICE_NAME" sh -lc "grep -E 'Starting MT5 startup script|\\[[1-7]/7\\]|The mt5linux server is running' /config/startup.log 2>/dev/null | tail -n 1" 2>/dev/null || true)"

  if [[ -n "$status_line" && "$status_line" != "$last_status" ]]; then
    echo "  startup: $status_line"
    last_status="$status_line"

    if [[ "$status_line" == *"[6/7] Installing Python libraries"* ]]; then
      echo "  info: step [6/7] may take several minutes on Apple Silicon emulation."
    fi
  fi

  if (( elapsed >= WAIT_TIMEOUT_SECONDS )); then
    echo "Error: timeout waiting for mt5linux server."
    echo "Recent startup log:"
    compose exec -T "$SERVICE_NAME" sh -lc "tail -n 80 /config/startup.log" || true
    exit 1
  fi

  sleep "$WAIT_INTERVAL_SECONDS"
  elapsed=$((elapsed + WAIT_INTERVAL_SECONDS))
  echo "  waiting... ${elapsed}s/${WAIT_TIMEOUT_SECONDS}s"
done
echo "  mt5linux server is ready."
dismiss_mt5_dialogs || true
search_company_dialog_by_server "$MT5_SERVER_VAL" || true
dismiss_mt5_dialogs || true
login_mt5_account_via_ui "$MT5_LOGIN_VAL" "$MT5_PASSWORD_VAL" "$MT5_SERVER_VAL" || true
if [[ ! "$MT5_SKIP_PRECHECKS" =~ ^(1|true|yes|on)$ ]]; then
  ensure_mt5_login_if_configured || true
else
  echo "[2.4/6] Skipping MT5 API prechecks (MT5_SKIP_PRECHECKS=$MT5_SKIP_PRECHECKS)."
fi
ensure_experts_common_ini || true
if [[ ! "$MT5_SKIP_PRECHECKS" =~ ^(1|true|yes|on)$ ]]; then
  enable_algo_trading_if_needed || true
else
  echo "[2.5/6] Skipping trade_allowed probe/toggle precheck."
fi

echo "[3/6] Installing bot dependencies (only when requirements changed)..."
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
    "$python_cmd" -m pip install --no-cache-dir --no-deps mt5linux
    "$python_cmd" -m pip install --no-cache-dir rpyc==5.0.1 plumbum numpy pyxdg pyzmq
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
'

echo "[4/6] Stopping old bot process (if any)..."
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
"

echo "[4.5/6] Re-checking MT5 Algo Trading before bot launch..."
ensure_experts_common_ini || true
if [[ ! "$MT5_SKIP_PRECHECKS" =~ ^(1|true|yes|on)$ ]]; then
  enable_algo_trading_if_needed || true
else
  echo "[4.5/6] Skipping second MT5 API precheck."
fi

echo "[5/6] Starting bot..."
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

compose exec -T \
  -e MT5_LOGIN="$MT5_LOGIN_VAL" \
  -e MT5_PASSWORD="$MT5_PASSWORD_VAL" \
  -e MT5_SERVER="$MT5_SERVER_VAL" \
  -e LIVE_MAGIC_NUMBER="${LIVE_MAGIC_NUMBER:-}" \
  -e LIVE_MANAGE_MANUAL_POSITIONS="${LIVE_MANAGE_MANUAL_POSITIONS:-0}" \
  -e LIVE_MODELS_DIR="${LIVE_MODELS_DIR:-}" \
  -e LIVE_LLM_SEMANTIC_CACHE_FILE="${LIVE_LLM_SEMANTIC_CACHE_FILE:-}" \
  -e LIVE_LLM_TEXT_LOG_FILE="${LIVE_LLM_TEXT_LOG_FILE:-}" \
  -e LIVE_PREWARM_SEMANTIC_ON_START="${LIVE_PREWARM_SEMANTIC_ON_START:-}" \
  -e LIVE_PREWARM_SEMANTIC_MAX_SECONDS="${LIVE_PREWARM_SEMANTIC_MAX_SECONDS:-}" \
  -e LIVE_PREWARM_SEMANTIC_MAX_MISSING="${LIVE_PREWARM_SEMANTIC_MAX_MISSING:-}" \
  -e LIVE_PREWARM_REQUEST_TIMEOUT_SEC="${LIVE_PREWARM_REQUEST_TIMEOUT_SEC:-}" \
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
  bash -lc "set -euo pipefail; py_cmd='${bot_python_cmd}'; require_shared='${require_shared_python}'; if [ ! -x \"\$py_cmd\" ]; then if [ \"\$require_shared\" = '1' ]; then echo \"Error: shared python not found at \$py_cmd\"; exit 1; fi; py_cmd='python3'; fi; echo \"  bot python: \$py_cmd\"; cd /bots && nohup env PYTHONUNBUFFERED=1 \"\$py_cmd\" -u /bots/${BOT_SCRIPT} > '${BOT_LOG}' 2>&1 &"

sleep 5

echo "[6/6] Verifying bot process..."
if compose exec -T "$SERVICE_NAME" bash -lc "
set -euo pipefail
script='/bots/${BOT_SCRIPT}'
for proc in /proc/[0-9]*; do
  [ -r \"\$proc/cmdline\" ] || continue
  if tr '\0' '\n' < \"\$proc/cmdline\" | grep -Fxq \"\$script\"; then
    exit 0
  fi
done
exit 1
"; then
  echo "Bot started successfully."
  echo "Log file: ${BOT_LOG}"
  echo "Follow logs with:"
  echo "  docker compose exec -it ${SERVICE_NAME} tail -f ${BOT_LOG}"
else
  echo "Error: bot process not found, showing last log lines:"
  compose exec -T "$SERVICE_NAME" bash -lc "tail -n 50 '${BOT_LOG}'" || true
  exit 1
fi
