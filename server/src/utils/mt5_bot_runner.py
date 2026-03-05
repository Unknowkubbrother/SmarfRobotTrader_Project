import base64
import hashlib
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken


@dataclass
class RunnerCommandResult:
    stdout: str
    stderr: str
    project_name: str | None = None
    container_id: str | None = None


@dataclass
class BotRuntimeHealthResult:
    project_name: str
    container_id: str | None
    trade_allowed: bool | None
    tradeapi_disabled: bool | None
    detail: str
    stdout: str = ""
    stderr: str = ""


class BotRunnerError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        returncode: int | None = None,
        stdout: str = "",
        stderr: str = "",
    ) -> None:
        super().__init__(message)
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def decrypt_mt5_password(encrypted_password: str) -> str:
    key = base64.urlsafe_b64encode(
        hashlib.sha256(os.getenv("SECRET_KEY", "UknownmeInLove").encode()).digest()
    )
    fernet = Fernet(key)
    try:
        return fernet.decrypt(encrypted_password.encode()).decode()
    except InvalidToken as exc:
        raise BotRunnerError("Unable to decrypt MT5 password. Check SECRET_KEY.") from exc


def build_profile_name(symbol: str, timeframe: str) -> str:
    cleaned_symbol = _sanitize_profile_token(symbol)
    cleaned_timeframe = _sanitize_profile_token(timeframe)
    if not cleaned_symbol or not cleaned_timeframe:
        raise BotRunnerError("LIVE_SYMBOL and LIVE_TIMEFRAME are required.")
    return f"{cleaned_symbol}_{cleaned_timeframe}"


def build_bot_runtime_env(
    *,
    bot_config_id: str,
    mt5_login: str,
    mt5_password: str,
    mt5_server: str,
    broker_name: str | None = None,
    live_symbol: str,
    live_timeframe: str,
    docker_image_id: str | None = None,
    magic_number: int | None = None,
) -> dict[str, str]:
    ws_url = (
        os.getenv("BOT_RUNNER_WS_URL")
        or os.getenv("BOT_WS_URL")
        or "ws://host.docker.internal:8000/bot/ws"
    )
    vision_url = (
        os.getenv("BOT_RUNNER_VISION_LLM_API_URL")
        or os.getenv("VISION_LLM_API_URL")
        or "http://host.docker.internal:8000/vision_llm/"
    )

    rpc_timeout_ms_raw = str(os.getenv("BOT_RUNNER_MT5_RPC_TIMEOUT_MS", "180000") or "").strip()
    try:
        rpc_timeout_ms = max(30000, int(rpc_timeout_ms_raw))
    except ValueError:
        rpc_timeout_ms = 180000

    sync_timeout_raw = str(os.getenv("BOT_RUNNER_MT5_SYNC_TIMEOUT_SEC", "") or "").strip()
    try:
        sync_timeout_sec = max(60, int(sync_timeout_raw)) if sync_timeout_raw else max(60, rpc_timeout_ms // 1000)
    except ValueError:
        sync_timeout_sec = max(60, rpc_timeout_ms // 1000)

    env = {
        "MT5_LOGIN": str(mt5_login).strip(),
        "MT5_PASSWORD": str(mt5_password).strip(),
        "MT5_SERVER": str(mt5_server).strip(),
        "LIVE_SYMBOL": str(live_symbol).strip().upper(),
        "LIVE_TIMEFRAME": str(live_timeframe).strip().upper(),
        "BOT_CONFIG_ID": str(bot_config_id).strip(),
        "BOT_WS_URL": ws_url.strip(),
        "VISION_LLM_API_URL": vision_url.strip(),
        "MT5_RPC_TIMEOUT_MS": str(rpc_timeout_ms),
        "LIVE_MT5_SYNC_TIMEOUT_SEC": str(sync_timeout_sec),
        "MT5_FORCE_ALGO_TRADING": os.getenv("BOT_RUNNER_FORCE_ALGO_TRADING", "1").strip() or "1",
        "MT5_EXPERTS_DISABLE_ON_ACCOUNT_CHANGE": os.getenv(
            "BOT_RUNNER_DISABLE_EXPERTS_ON_ACCOUNT_CHANGE", "0"
        ).strip()
        or "0",
        "MT5_EXPERTS_DISABLE_ON_PROFILE_CHANGE": os.getenv(
            "BOT_RUNNER_DISABLE_EXPERTS_ON_PROFILE_CHANGE", "0"
        ).strip()
        or "0",
        "MT5_EXPERTS_DISABLE_ON_CHART_CHANGE": os.getenv(
            "BOT_RUNNER_DISABLE_EXPERTS_ON_CHART_CHANGE", "0"
        ).strip()
        or "0",
        "MT5_EXPERTS_DISABLE_VIA_PYTHON_API": os.getenv(
            "BOT_RUNNER_DISABLE_EXPERTS_VIA_API", "0"
        ).strip()
        or "0",
        "MT5_ALLOW_DLL_IMPORTS": os.getenv("BOT_RUNNER_ALLOW_DLL_IMPORTS", "1").strip() or "1",
        "MT5_ALLOW_WEBREQUEST": os.getenv("BOT_RUNNER_ALLOW_WEBREQUEST", "1").strip() or "1",
        "MT5_ALLOW_PARTIAL_START": os.getenv("BOT_RUNNER_ALLOW_PARTIAL_START", "0").strip() or "0",
        "MT5_CLEAN_ACCOUNT_CACHE_ON_START": os.getenv(
            "BOT_RUNNER_MT5_CLEAN_ACCOUNT_CACHE_ON_START", "1"
        ).strip()
        or "1",
        "MT5_COMPANY_DISCOVERY_BEFORE_LOGIN": os.getenv(
            "BOT_RUNNER_COMPANY_DISCOVERY_BEFORE_LOGIN", "1"
        ).strip()
        or "1",
        "MT5_COMPANY_DIALOG_CLEANUP_AFTER_LOGIN": os.getenv(
            "BOT_RUNNER_COMPANY_DIALOG_CLEANUP_AFTER_LOGIN", "1"
        ).strip()
        or "1",
        "BOT_WAIT_FOR_WS_REGISTER": os.getenv("BOT_RUNNER_WAIT_FOR_WS_REGISTER", "1").strip() or "1",
        "BOT_WS_READY_TIMEOUT_SECONDS": os.getenv("BOT_RUNNER_WS_READY_TIMEOUT_SECONDS", "120").strip() or "120",
        "BOT_STREAM_LOG_TO_CONTAINER_STDOUT": os.getenv("BOT_RUNNER_STREAM_CONTAINER_LOGS", "1").strip() or "1",
        "RUNNER_PROGRESS_TO_CONTAINER_LOGS": os.getenv("BOT_RUNNER_PROGRESS_TO_CONTAINER_LOGS", "1").strip() or "1",
        "LIVE_MANAGE_MANUAL_POSITIONS": os.getenv("BOT_RUNNER_MANAGE_MANUAL_POSITIONS", "0").strip() or "0",
        "AUTO_BUILD": os.getenv("BOT_RUNNER_AUTO_BUILD", "0").strip() or "0",
        "PULL_LATEST_IMAGE": os.getenv("BOT_RUNNER_PULL_LATEST", "1").strip() or "1",
        "FORCE_REBUILD": os.getenv("BOT_RUNNER_FORCE_REBUILD", "0").strip() or "0",
    }

    broker_query = str(broker_name or "").strip()
    if broker_query:
        env["MT5_COMPANY_SEARCH_QUERY"] = broker_query

    optional_passthrough = {
        "BOT_RUNNER_MODELS_DIR": "LIVE_MODELS_DIR",
        "BOT_RUNNER_LLM_SEMANTIC_CACHE_FILE": "LIVE_LLM_SEMANTIC_CACHE_FILE",
        "BOT_RUNNER_LLM_TEXT_LOG_FILE": "LIVE_LLM_TEXT_LOG_FILE",
        "BOT_RUNNER_STATE_FILE": "LIVE_STATE_FILE",
        "BOT_RUNNER_MANAGED_MAGIC_SET": "LIVE_MANAGED_MAGIC_SET",
        "BOT_RUNNER_PERFORMANCE_MAGIC_SET": "LIVE_PERFORMANCE_MAGIC_SET",
        "BOT_RUNNER_PERFORMANCE_SCOPE": "LIVE_PERFORMANCE_SCOPE",
        "BOT_RUNNER_PERFORMANCE_SYNC_INTERVAL_SEC": "LIVE_PERFORMANCE_SYNC_INTERVAL_SEC",
        "BOT_RUNNER_PERFORMANCE_BOOT_LOOKBACK_DAYS": "LIVE_PERFORMANCE_BOOT_LOOKBACK_DAYS",
        "BOT_RUNNER_MT5_HISTORY_END_AHEAD_HOURS": "LIVE_MT5_HISTORY_END_AHEAD_HOURS",
        "BOT_RUNNER_PREWARM_SEMANTIC_ON_START": "LIVE_PREWARM_SEMANTIC_ON_START",
        "BOT_RUNNER_PREWARM_SEMANTIC_MAX_SECONDS": "LIVE_PREWARM_SEMANTIC_MAX_SECONDS",
        "BOT_RUNNER_PREWARM_SEMANTIC_MAX_MISSING": "LIVE_PREWARM_SEMANTIC_MAX_MISSING",
        "BOT_RUNNER_PREWARM_REQUEST_TIMEOUT_SEC": "LIVE_PREWARM_REQUEST_TIMEOUT_SEC",
        "BOT_RUNNER_CATCHUP_MAX_BARS": "LIVE_CATCHUP_MAX_BARS",
        "BOT_RUNNER_SEMANTIC_ALIAS_HOURS": "LIVE_SEMANTIC_ALIAS_HOURS",
        "BOT_RUNNER_SEMANTIC_NO_DATA_RETRY_SECONDS": "LIVE_SEMANTIC_NO_DATA_RETRY_SECONDS",
        "BOT_RUNNER_VISION_LLM_EMBED_TEXT_API_URL": "VISION_LLM_EMBED_TEXT_API_URL",
        "BOT_RUNNER_STRICT_SERVER_MATCH": "MT5_STRICT_SERVER_MATCH",
        "BOT_RUNNER_SERVER_FALLBACKS": "MT5_SERVER_FALLBACKS",
        "BOT_RUNNER_COMPANY_SEARCH_QUERY": "MT5_COMPANY_SEARCH_QUERY",
        "BOT_RUNNER_MT5_SNAPSHOT_PATH": "MT5_SNAPSHOT_PATH",
        "BOT_RUNNER_MT5_SEED_CONFIG_DIR": "MT5_SEED_CONFIG_DIR",
        "BOT_RUNNER_MT5_BASELINE_INSTANCE_ID": "MT5_BASELINE_INSTANCE_ID",
        "BOT_RUNNER_MT5_AUTO_EXPORT_BASELINE_SNAPSHOT": "MT5_AUTO_EXPORT_BASELINE_SNAPSHOT",
        "BOT_RUNNER_MT5_AUTO_EXPORT_BASELINE_SEED": "MT5_AUTO_EXPORT_BASELINE_SEED",
        "BOT_RUNNER_MT5_REFRESH_BASELINE_SNAPSHOT": "MT5_REFRESH_BASELINE_SNAPSHOT",
        "BOT_RUNNER_MT5_REFRESH_BASELINE_SEED": "MT5_REFRESH_BASELINE_SEED",
    }
    for source_name, target_name in optional_passthrough.items():
        raw_value = os.getenv(source_name)
        if raw_value is None:
            continue
        value = str(raw_value).strip()
        if value:
            env[target_name] = value

    if magic_number is not None:
        env["LIVE_MAGIC_NUMBER"] = str(int(magic_number))

    if docker_image_id and str(docker_image_id).strip():
        env["METATRADER_IMAGE"] = str(docker_image_id).strip()

    return env


def run_bot_instance_action(
    *,
    action: str,
    instance_name: str,
    profile_name: str | None = None,
    env_overrides: dict[str, str] | None = None,
    timeout_sec: int = 7200,
) -> RunnerCommandResult:
    normalized_action = str(action or "").strip().lower()
    if normalized_action not in {"start", "stop", "restart"}:
        raise BotRunnerError("Invalid action. Use start, stop, or restart.")

    if not str(instance_name or "").strip():
        raise BotRunnerError("instance_name is required.")

    if normalized_action in {"start", "restart"} and not str(profile_name or "").strip():
        raise BotRunnerError("profile_name is required for start/restart.")

    runner_dir = _resolve_runner_dir()
    _ensure_runner_script(runner_dir)
    _ensure_docker_access()

    args = ["./run_instance.sh", normalized_action, str(instance_name).strip()]
    if normalized_action in {"start", "restart"}:
        args.append(str(profile_name).strip())

    env = os.environ.copy()
    if env_overrides:
        env.update({k: str(v) for k, v in env_overrides.items() if v is not None})

    try:
        proc = subprocess.run(
            args,
            cwd=str(runner_dir),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired as exc:
        raise BotRunnerError(
            f"Runner command timed out after {timeout_sec}s.",
            stdout=_shorten_output(exc.stdout),
            stderr=_shorten_output(exc.stderr),
        ) from exc

    if proc.returncode != 0:
        raise BotRunnerError(
            f"Runner command failed with exit code {proc.returncode}.",
            returncode=proc.returncode,
            stdout=_shorten_output(proc.stdout),
            stderr=_shorten_output(proc.stderr),
        )

    project_name = f"mt5_{_sanitize_instance_name(str(instance_name))}"
    container_id = None
    if normalized_action in {"start", "restart"}:
        container_id = _resolve_container_id(project_name)

    return RunnerCommandResult(
        stdout=_shorten_output(proc.stdout),
        stderr=_shorten_output(proc.stderr),
        project_name=project_name,
        container_id=container_id,
    )


def purge_bot_instance_state(
    *,
    instance_name: str,
    timeout_sec: int = 300,
) -> RunnerCommandResult:
    normalized_instance = _sanitize_instance_name(str(instance_name or ""))
    if not normalized_instance:
        raise BotRunnerError("instance_name is required.")

    runner_dir = _resolve_runner_dir()
    _ensure_runner_script(runner_dir)
    _ensure_docker_access()

    project_name = f"mt5_{normalized_instance}"
    env = os.environ.copy()
    env["COMPOSE_PROJECT_NAME"] = project_name

    down_cmd = ["docker", "compose", "down", "--remove-orphans"]
    try:
        down_proc = subprocess.run(
            down_cmd,
            cwd=str(runner_dir),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired as exc:
        raise BotRunnerError(
            f"Runner purge timed out after {timeout_sec}s (compose down).",
            stdout=_shorten_output(exc.stdout),
            stderr=_shorten_output(exc.stderr),
        ) from exc

    if down_proc.returncode != 0:
        raise BotRunnerError(
            "Runner purge failed while stopping compose stack.",
            returncode=down_proc.returncode,
            stdout=_shorten_output(down_proc.stdout),
            stderr=_shorten_output(down_proc.stderr),
        )

    config_volume_name = f"{project_name}_config"
    volume_proc = subprocess.run(
        ["docker", "volume", "rm", "-f", config_volume_name],
        capture_output=True,
        text=True,
    )
    if volume_proc.returncode != 0:
        stderr_text = str(volume_proc.stderr or "")
        if "No such volume" not in stderr_text:
            raise BotRunnerError(
                f"Runner purge failed while removing volume '{config_volume_name}'.",
                returncode=volume_proc.returncode,
                stdout=_shorten_output(volume_proc.stdout),
                stderr=_shorten_output(volume_proc.stderr),
            )

    state_dir = runner_dir / ".instances"
    instance_dir = state_dir / normalized_instance
    legacy_env_file = state_dir / f"{normalized_instance}.env"
    if instance_dir.exists():
        shutil.rmtree(instance_dir, ignore_errors=True)
    if legacy_env_file.exists():
        try:
            legacy_env_file.unlink()
        except Exception as exc:
            raise BotRunnerError(
                f"Runner purge failed while removing legacy state file '{legacy_env_file}'."
            ) from exc

    combined_stdout = "\n".join(
        part
        for part in [
            _shorten_output(down_proc.stdout),
            _shorten_output(volume_proc.stdout),
            f"purged_instance={normalized_instance}",
        ]
        if part
    )
    combined_stderr = "\n".join(
        part
        for part in [
            _shorten_output(down_proc.stderr),
            _shorten_output(volume_proc.stderr),
        ]
        if part
    )

    return RunnerCommandResult(
        stdout=combined_stdout,
        stderr=combined_stderr,
        project_name=project_name,
        container_id=None,
    )


def pull_docker_image(image_ref: str, timeout_sec: int = 1200) -> RunnerCommandResult:
    image = str(image_ref or "").strip()
    if not image:
        raise BotRunnerError("docker image reference is required.")

    _ensure_docker_access()
    try:
        proc = subprocess.run(
            ["docker", "pull", image],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired as exc:
        raise BotRunnerError(
            f"docker pull timed out after {timeout_sec}s.",
            stdout=_shorten_output(exc.stdout),
            stderr=_shorten_output(exc.stderr),
        ) from exc

    if proc.returncode != 0:
        raise BotRunnerError(
            f"docker pull failed for image '{image}'.",
            returncode=proc.returncode,
            stdout=_shorten_output(proc.stdout),
            stderr=_shorten_output(proc.stderr),
        )

    return RunnerCommandResult(
        stdout=_shorten_output(proc.stdout),
        stderr=_shorten_output(proc.stderr),
    )


def check_bot_runtime_health(
    *,
    instance_name: str,
    timeout_sec: int = 90,
    rpc_timeout_ms: int | str = 180000,
) -> BotRuntimeHealthResult:
    normalized_instance = str(instance_name or "").strip()
    if not normalized_instance:
        raise BotRunnerError("instance_name is required.")

    _ensure_docker_access()
    project_name = f"mt5_{_sanitize_instance_name(normalized_instance)}"
    container_id = _resolve_container_id(project_name)
    if not container_id:
        return BotRuntimeHealthResult(
            project_name=project_name,
            container_id=None,
            trade_allowed=None,
            tradeapi_disabled=None,
            detail="container_not_found",
        )

    probe_script = """
import json
import os
import sys
import rpyc
from mt5linux import MetaTrader5

payload = {
    "trade_allowed": None,
    "tradeapi_disabled": None,
    "detail": "",
}

try:
    timeout_ms = int(os.getenv("MT5_RPC_TIMEOUT_MS", "180000"))
except Exception:
    timeout_ms = 180000

try:
    rpyc.core.protocol.DEFAULT_CONFIG["sync_request_timeout"] = max(60.0, float(timeout_ms) / 1000.0)
except Exception:
    pass

try:
    mt5 = MetaTrader5(host="localhost", port=8001)
    # Runtime health check must be read-only.
    # Do not call login() or initialize() with credentials here, because
    # the endpoint is polled frequently by UI and can create repeated
    # re-login side-effects in MT5.
    if not mt5.initialize(timeout=timeout_ms):
        payload["detail"] = "initialize_failed"
        print(json.dumps(payload))
        sys.exit(2)

    info = mt5.terminal_info()
    if info is None:
        payload["detail"] = "terminal_info_none"
        print(json.dumps(payload))
        sys.exit(3)

    payload["trade_allowed"] = bool(getattr(info, "trade_allowed", False))
    payload["tradeapi_disabled"] = bool(getattr(info, "tradeapi_disabled", False))
    payload["detail"] = "ok"
    print(json.dumps(payload))
    sys.exit(0 if payload["trade_allowed"] else 1)
except Exception as exc:
    payload["detail"] = f"exception:{exc}"
    print(json.dumps(payload))
    sys.exit(4)
""".strip()

    env = os.environ.copy()
    env["MT5_RPC_TIMEOUT_MS"] = str(rpc_timeout_ms)
    # Avoid health-check process pile-up inside container (can starve MT5 RPC slots).
    cleanup_cmd = (
        'pkill -f "python3 -c import json import os import sys.*mt5linux import MetaTrader5" '
        '>/dev/null 2>&1 || true'
    )
    try:
        subprocess.run(
            ["docker", "exec", container_id, "sh", "-lc", cleanup_cmd],
            capture_output=True,
            text=True,
            timeout=20,
            env=env,
        )
    except Exception:
        pass

    try:
        rpc_timeout_ms_int = max(30000, int(rpc_timeout_ms))
    except Exception:
        rpc_timeout_ms_int = 180000
    inner_timeout_sec = max(30, min(max(30, rpc_timeout_ms_int // 1000 + 30), max(30, int(timeout_sec))))
    outer_timeout_sec = max(int(timeout_sec), inner_timeout_sec + 15)
    try:
        proc = subprocess.run(
            [
                "docker",
                "exec",
                "-i",
                "-e",
                f"MT5_RPC_TIMEOUT_MS={rpc_timeout_ms_int}",
                container_id,
                "timeout",
                str(inner_timeout_sec),
                "python3",
                "-",
            ],
            input=probe_script,
            capture_output=True,
            text=True,
            timeout=outer_timeout_sec,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        raise BotRunnerError(
            f"Runtime health check timed out after {outer_timeout_sec}s.",
            stdout=_shorten_output(exc.stdout),
            stderr=_shorten_output(exc.stderr),
        ) from exc

    stdout = _shorten_output(proc.stdout)
    stderr = _shorten_output(proc.stderr)
    payload = _extract_json_payload(stdout)
    detail = str(payload.get("detail", "") or "").strip() or (
        "trade_allowed_on" if proc.returncode == 0 else "trade_allowed_off" if proc.returncode == 1 else f"exit_{proc.returncode}"
    )

    trade_allowed_raw = payload.get("trade_allowed", None)
    tradeapi_disabled_raw = payload.get("tradeapi_disabled", None)
    trade_allowed = bool(trade_allowed_raw) if isinstance(trade_allowed_raw, bool) else None
    tradeapi_disabled = bool(tradeapi_disabled_raw) if isinstance(tradeapi_disabled_raw, bool) else None

    return BotRuntimeHealthResult(
        project_name=project_name,
        container_id=container_id,
        trade_allowed=trade_allowed,
        tradeapi_disabled=tradeapi_disabled,
        detail=detail,
        stdout=stdout,
        stderr=stderr,
    )


def _resolve_runner_dir() -> Path:
    candidates: list[Path] = []
    configured = str(os.getenv("RUNNER_DIR", "") or "").strip()
    if configured:
        candidates.append(Path(configured))

    candidates.append(Path("/opt/mt5-runner"))
    candidates.append(Path(__file__).resolve().parents[2] / "docker_mt5_bot")

    for path in candidates:
        if path.is_dir() and (path / "run_instance.sh").is_file():
            return path

    raise BotRunnerError(
        "Runner directory not found. Set RUNNER_DIR to a directory containing run_instance.sh."
    )


def _ensure_runner_script(runner_dir: Path) -> None:
    script = runner_dir / "run_instance.sh"
    if not script.is_file():
        raise BotRunnerError(f"Runner script not found: {script}")


def _ensure_docker_access() -> None:
    if shutil.which("docker") is None:
        raise BotRunnerError(
            "docker CLI not found. Install Docker CLI in server runtime."
        )

    if os.path.exists("/.dockerenv") and not os.path.exists("/var/run/docker.sock"):
        raise BotRunnerError(
            "Server is running in Docker but /var/run/docker.sock is not mounted."
        )

    probe = subprocess.run(
        ["docker", "version", "--format", "{{.Server.Version}}"],
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0:
        raise BotRunnerError(
            "Cannot access Docker daemon. Mount /var/run/docker.sock and verify permissions.",
            returncode=probe.returncode,
            stdout=_shorten_output(probe.stdout),
            stderr=_shorten_output(probe.stderr),
        )


def _resolve_container_id(project_name: str) -> str | None:
    service_name = str(os.getenv("MT5_SERVICE_NAME", "metatrader5-macos")).strip() or "metatrader5-macos"
    proc = subprocess.run(
        [
            "docker",
            "ps",
            "-aq",
            "--filter",
            f"label=com.docker.compose.project={project_name}",
            "--filter",
            f"label=com.docker.compose.service={service_name}",
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return None
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    return lines[0] if lines else None


def _shorten_output(output: str | bytes | None, limit: int = 4000) -> str:
    if output is None:
        return ""
    if isinstance(output, bytes):
        text = output.decode(errors="ignore")
    else:
        text = str(output)
    if len(text) <= limit:
        return text
    return text[-limit:]


def _extract_json_payload(raw_text: str | None) -> dict:
    text = str(raw_text or "").strip()
    if not text:
        return {}
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in reversed(lines):
        if not line.startswith("{"):
            continue
        try:
            parsed = json.loads(line)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    return {}


def _sanitize_instance_name(raw: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(raw or "").strip().lower()).strip("_")


def _sanitize_profile_token(raw: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(raw or "").strip().lower())
