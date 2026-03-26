import asyncio
import logging
from datetime import datetime

from fastapi import APIRouter, HTTPException, Request
from prisma import Json
from ..models.bot_model import (
    Create_Bot_Configuration,
    Update_Bot_Status,
    Update_Bot_Risk,
    Update_Bot_Schedule,
    Change_Bot_Model,
    Delete_Bot,
    Apply_Bot_Update,
    Emergency_Bot_Stop,
)
from ..database.client import db
from ..utils.bot_runtime_config import (
    CUSTOM_LOT_RISK_MODE,
    merge_bot_runtime_settings,
    normalize_custom_lot,
    normalize_risk_level,
    normalize_risk_mode,
    parse_bot_runtime_settings,
    serialize_bot_runtime_settings,
)
from ..utils.trading_schedule import normalize_trading_schedule
from ..utils.mt5_bot_runner import (
    BotRunnerError,
    build_bot_runtime_env,
    build_profile_name,
    check_bot_runtime_health,
    decrypt_mt5_password,
    purge_bot_instance_state,
    pull_docker_image,
    run_bot_instance_action,
)
from ..utils.bot_operation_events import emit_and_store_bot_operation_event
from ..utils.bot_magic import derive_magic_number, normalize_magic_number
from ..utils.notification_delivery import dispatch_notification_to_user_id
from ..utils.subscription_access import assert_user_subscription_allows_bot_usage
from ..utils.ws_manager import bot_hub

bot_router = APIRouter()
logger = logging.getLogger(__name__)
ACTIVE_RECORD_STATUS = "active"
MAGIC_RESEED_LIMIT = 128


def _enum_value(value):
    return value.value if hasattr(value, "value") else value


def _normalize_pair_value(value: str | None) -> str:
    return str(value or "").strip().upper()


async def _find_account_symbol_timeframe_conflict(
    *,
    account_id: str,
    symbol: str | None,
    timeframe: str | None,
    exclude_config_id: str | None = None,
):
    normalized_symbol = _normalize_pair_value(symbol)
    normalized_timeframe = _normalize_pair_value(timeframe)
    if not normalized_symbol or not normalized_timeframe:
        return None

    where: dict = {
        "accountId": str(account_id),
        "recordStatus": ACTIVE_RECORD_STATUS,
    }
    if exclude_config_id:
        where["id"] = {"not": str(exclude_config_id)}

    configs = await db.botconfiguration.find_many(
        where=where,
        include={"botVersion": True},
    )

    for config in configs:
        bot_version = getattr(config, "botVersion", None)
        existing_symbol = _normalize_pair_value(getattr(bot_version, "symbol", None))
        existing_timeframe = _normalize_pair_value(getattr(bot_version, "timeframe", None))
        if existing_symbol == normalized_symbol and existing_timeframe == normalized_timeframe:
            return config

    return None


async def _allocate_magic_number_for_account(
    *,
    account_id: str,
    bot_instance_id: int,
    exclude_config_id: str | None = None,
) -> int:
    account_text = str(account_id or "").strip()
    instance_int = int(bot_instance_id or 0)
    if not account_text or instance_int <= 0:
        raise HTTPException(status_code=400, detail="Cannot resolve bot magic number")

    for salt in range(MAGIC_RESEED_LIMIT):
        candidate = derive_magic_number(account_text, instance_int, salt=salt)
        where: dict = {
            "accountId": account_text,
            "magicNumber": int(candidate),
        }
        if exclude_config_id:
            where["id"] = {"not": str(exclude_config_id)}
        existing = await db.botconfiguration.find_first(where=where)
        if not existing:
            return int(candidate)

    raise HTTPException(status_code=500, detail="Unable to allocate unique bot magic number")


async def _ensure_bot_magic_number(config) -> int:
    current = normalize_magic_number(getattr(config, "magicNumber", None))
    if current is not None:
        return int(current)

    account_id = str(getattr(config, "accountId", "") or "").strip()
    bot_instance_id = int(getattr(config, "botInstanceId", 0) or 0)
    resolved = await _allocate_magic_number_for_account(
        account_id=account_id,
        bot_instance_id=bot_instance_id,
        exclude_config_id=str(getattr(config, "id", "") or "").strip() or None,
    )
    await db.botconfiguration.update(
        where={"id": str(config.id)},
        data={"magicNumber": int(resolved)},
    )
    try:
        setattr(config, "magicNumber", int(resolved))
    except Exception:
        pass
    return int(resolved)


async def _extract_runtime_context(
    config,
    image_override: str | None = None,
    bot_version_override=None,
    version_tag_override: str | None = None,
) -> dict[str, object | None]:
    account = getattr(config, "account", None)
    if not account:
        raise HTTPException(status_code=400, detail="Trading account is missing for this bot.")

    mt5_login = str(getattr(account, "mt5LoginId", "") or "").strip()
    mt5_server = str(getattr(account, "serverName", "") or "").strip()
    encrypted_password = str(getattr(account, "mt5Password", "") or "").strip()
    if not mt5_login or not mt5_server or not encrypted_password:
        raise HTTPException(
            status_code=400,
            detail="Trading account credentials are incomplete. Please update account login/password/server first.",
        )

    bot_version = bot_version_override or getattr(config, "botVersion", None)
    if not bot_version:
        raise HTTPException(status_code=400, detail="Bot version is missing for this bot.")

    live_symbol = str(getattr(bot_version, "symbol", "") or "").strip().upper()
    live_timeframe = str(getattr(bot_version, "timeframe", "") or "").strip().upper()
    if not live_symbol or not live_timeframe:
        raise HTTPException(
            status_code=400,
            detail="Bot version must include symbol and timeframe to run docker profile.",
        )

    try:
        mt5_password = decrypt_mt5_password(encrypted_password)
    except BotRunnerError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    installed_image_ref = str(getattr(config, "installedDockerImageId", "") or "").strip() or None
    installed_version_tag = str(getattr(config, "installedVersionTag", "") or "").strip() or None
    latest_image_ref = str(getattr(bot_version, "dockerImageId", "") or "").strip() or None
    latest_version_tag = str(getattr(bot_version, "versionTag", "") or "").strip() or None

    image_ref = None
    if image_override and str(image_override).strip():
        image_ref = str(image_override).strip()
    elif bot_version_override is not None:
        image_ref = latest_image_ref
    else:
        # Keep runtime pinned to installed image until apply_update/change_model modifies it.
        image_ref = installed_image_ref or latest_image_ref

    effective_version_tag = None
    if version_tag_override and str(version_tag_override).strip():
        effective_version_tag = str(version_tag_override).strip()
    elif bot_version_override is not None:
        effective_version_tag = latest_version_tag
    else:
        # Keep runtime pinned to installed version until apply_update/change_model modifies it.
        effective_version_tag = installed_version_tag or latest_version_tag

    try:
        profile_name = build_profile_name(live_symbol, live_timeframe)
    except BotRunnerError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    magic_number = await _ensure_bot_magic_number(config)
    runtime_settings = parse_bot_runtime_settings(
        getattr(config, "tradingSchedule", None),
        risk_level=_enum_value(getattr(config, "riskLevel", None)),
    )

    runtime_env = build_bot_runtime_env(
        bot_config_id=str(config.id),
        mt5_login=mt5_login,
        mt5_password=mt5_password,
        mt5_server=mt5_server,
        broker_name=str(getattr(account, "brokerName", "") or "").strip(),
        live_symbol=live_symbol,
        live_timeframe=live_timeframe,
        docker_image_id=image_ref,
        bot_version_tag=effective_version_tag,
        magic_number=magic_number,
        risk_level=str(runtime_settings["risk_level"] or "").strip() or None,
        risk_mode=str(runtime_settings["risk_mode"] or "").strip() or None,
        custom_lot=runtime_settings["custom_lot"],
        trading_schedule=runtime_settings["schedule"],
    )

    return {
        "profile_name": profile_name,
        "docker_image_id": image_ref,
        "magic_number": magic_number,
        "runtime_env": runtime_env,
    }


def _runner_error_message(prefix: str, exc: BotRunnerError) -> str:
    summarized = _extract_runner_failure_detail(exc)
    if summarized:
        return f"{prefix}: {summarized}"
    stderr = str(getattr(exc, "stderr", "") or "").strip()
    stdout = str(getattr(exc, "stdout", "") or "").strip()
    if stderr:
        return f"{prefix}: {exc}. stderr={stderr}"
    if stdout:
        return f"{prefix}: {exc}. stdout={stdout}"
    return f"{prefix}: {exc}"


def _extract_runner_failure_detail(exc: BotRunnerError) -> str | None:
    stderr = str(getattr(exc, "stderr", "") or "")
    stdout = str(getattr(exc, "stdout", "") or "")
    combined = "\n".join(part for part in (stderr, stdout) if part)
    if not combined.strip():
        return None

    markers: dict[str, str] = {}
    lines = [str(line or "").strip() for line in combined.splitlines() if str(line or "").strip()]
    for line in lines:
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        normalized_key = str(key or "").strip()
        if normalized_key in {
            "fatal_login_error",
            "login_failure_reason",
            "invalid_login",
        }:
            markers[normalized_key] = str(value or "").strip()

    invalid_login = markers.get("invalid_login")
    if invalid_login:
        return f"MT5 login ID is invalid: {invalid_login}"

    reason = (
        str(markers.get("fatal_login_error") or "").strip()
        or str(markers.get("login_failure_reason") or "").strip()
    )
    if reason == "invalid_login_id":
        return "MT5 login ID is invalid. Check the trading account login."
    if reason == "account_disabled":
        return "MT5 account is disabled or blocked. Check login ID, password, and server."
    if reason == "invalid_credentials":
        return "MT5 login failed. Check login ID, password, and server."

    normalized_text = " ".join(combined.lower().split())
    if "error: mt5 login failed. check login id, password, and server." in normalized_text:
        return "MT5 login failed. Check login ID, password, and server."
    if "error: mt5 account is disabled or blocked." in normalized_text:
        return "MT5 account is disabled or blocked. Check login ID, password, and server."
    if "error: mt5 login id is invalid." in normalized_text:
        return "MT5 login ID is invalid. Check the trading account login."
    if "error: mt5 login precheck failed and fallback login did not succeed." in normalized_text:
        return "MT5 login failed. Check login ID, password, and server."
    return None


def _runner_failure_detail(exc: BotRunnerError) -> str:
    return _extract_runner_failure_detail(exc) or str(exc)


async def _emit_lifecycle_event(
    bot_config_id: str,
    action: str,
    phase: str,
    detail: str,
    status: str | None = None,
    source: str = "user",
    metadata: dict | None = None,
    owner_user_id: str | None = None,
) -> None:
    try:
        await emit_and_store_bot_operation_event(
            bot_config_id=bot_config_id,
            action=action,
            phase=phase,
            detail=detail,
            status=status,
            source=source,
            metadata=metadata,
            owner_user_id=owner_user_id,
        )
    except Exception as exc:
        logger.warning(
            "failed to broadcast bot lifecycle event bot=%s action=%s phase=%s: %s",
            bot_config_id,
            action,
            phase,
            exc,
        )


# === Helper: verify bot config ownership ===
async def verify_bot_ownership(bot_config_id: str, user_id: str):
    config = await db.botconfiguration.find_first(
        where={
            "id": bot_config_id,
            "recordStatus": ACTIVE_RECORD_STATUS,
            "account": {
                "userId": user_id,
                "recordStatus": ACTIVE_RECORD_STATUS,
            },
        },
        include={"account": True, "botVersion": True}
    )
    if not config:
        raise HTTPException(status_code=404, detail="Bot configuration not found")
    return config


@bot_router.get("/versions", tags=["bot"])
async def get_bot_versions(request: Request):
    if not request.state.user_id:
        raise HTTPException(status_code=400, detail="User ID is required")
    
    bot_versions = await db.botversion.find_many(
        where={
            "isActive": True
        },
        order={
            "releaseDate": "desc"
        }
    )

    result = []
    for bv in bot_versions:
        runner_profile = None
        try:
            if bv.symbol and bv.timeframe:
                runner_profile = build_profile_name(bv.symbol, bv.timeframe)
        except Exception:
            runner_profile = None

        result.append({
            "id": str(bv.modelId),
            "model_id": str(bv.modelId),
            "label": bv.label,
            "version_tag": bv.versionTag,
            "symbol": bv.symbol,
            "timeframe": bv.timeframe,
            "is_active": bool(getattr(bv, "isActive", True)),
            "runner_profile": runner_profile,
            "release_notes": bv.releaseNotes,
            "release_date": bv.releaseDate.strftime("%Y-%m-%d") if bv.releaseDate else None,
        })

    return {
        "status_code": 200,
        "data": result
    }

# @@comment bot_configuration
@bot_router.post('/create_bot_configuration')
async def create_bot_configuration(request: Request, data: Create_Bot_Configuration):
    if not request.state.user_id:
        raise HTTPException(status_code=400, detail="User ID is required")
    await assert_user_subscription_allows_bot_usage(
        request.state.user_id,
        action_label="create new bots",
    )

    trading_account = await db.tradingaccount.find_first(
        where={
            "id": data.accountId,
            "userId": request.state.user_id,
            "recordStatus": ACTIVE_RECORD_STATUS,
        }
    )
    
    if not trading_account:
        raise HTTPException(status_code=404, detail="Trading account not found")

    bot_version = await db.botversion.find_unique(
        where={
            "modelId": data.modelId
        }
    )

    if not bot_version:
        raise HTTPException(status_code=404, detail="Bot version not found")
    if not getattr(bot_version, "isActive", True):
        raise HTTPException(status_code=400, detail="This bot version is inactive")

    conflict = await _find_account_symbol_timeframe_conflict(
        account_id=str(data.accountId),
        symbol=getattr(bot_version, "symbol", None),
        timeframe=getattr(bot_version, "timeframe", None),
    )
    if conflict is not None:
        symbol = _normalize_pair_value(getattr(bot_version, "symbol", None)) or "UNKNOWN"
        timeframe = _normalize_pair_value(getattr(bot_version, "timeframe", None)) or "UNKNOWN"
        raise HTTPException(
            status_code=400,
            detail=f"Bot for {symbol} {timeframe} already exists in this account",
        )

    bot_configuration_count = await db.botconfiguration.count(
        where={
            "accountId": data.accountId
        }
    )

    bot_configuration_count = bot_configuration_count + 1

    botInstanceId = 1000 + bot_configuration_count
    magic_number = await _allocate_magic_number_for_account(
        account_id=str(data.accountId),
        bot_instance_id=int(botInstanceId),
    )
    
    requested_risk_mode = normalize_risk_mode(_enum_value(getattr(data, "riskMode", None)))
    requested_custom_lot = normalize_custom_lot(getattr(data, "customLot", None))
    if requested_risk_mode == CUSTOM_LOT_RISK_MODE and requested_custom_lot is None:
        raise HTTPException(status_code=400, detail="Custom lot must be at least 0.01")

    trading_settings = serialize_bot_runtime_settings(
        schedule={},
        risk_mode=requested_risk_mode,
        custom_lot=requested_custom_lot,
    )

    bot_configuration = await db.botconfiguration.create(
        data={
            "account": {"connect": {"id": data.accountId}},
            "botVersion": {"connect": {"modelId": data.modelId}},
            "riskLevel": data.riskLevel.value,
            "tradingSchedule": Json(trading_settings),
            "isActive": False,
            "recordStatus": ACTIVE_RECORD_STATUS,
            "deletedAt": None,
            "containerStatus": "stopped",
            "installedDockerImageId": bot_version.dockerImageId,
            "installedVersionTag": bot_version.versionTag,
            "botInstanceId": botInstanceId,
            "magicNumber": int(magic_number),
        }
    )
    
    if not bot_configuration:
        raise HTTPException(status_code=400, detail="Bot configuration creation failed")
    
    return {
        "status_code": 200,
        "message": "Bot configuration created successfully",
        "magic_number": int(magic_number),
    }


# === Bot Control Endpoints ===

@bot_router.patch('/update_status', tags=["bot"])
async def update_bot_status(request: Request, data: Update_Bot_Status):
    if not request.state.user_id:
        raise HTTPException(status_code=400, detail="User ID is required")

    config = await verify_bot_ownership(data.botConfigId, request.state.user_id)

    requested_status = str(data.status or "").strip().lower()
    if requested_status not in ("running", "stopped"):
        raise HTTPException(status_code=400, detail="Status must be 'running' or 'stopped'")
    if requested_status == "running":
        await assert_user_subscription_allows_bot_usage(
            request.state.user_id,
            action_label="start bots",
        )
    if requested_status == "running" and config.botVersion and not getattr(config.botVersion, "isActive", True):
        raise HTTPException(status_code=400, detail="Cannot run bot. Its version is inactive.")

    runner_result = None
    if requested_status == "running":
        runtime = await _extract_runtime_context(config)
        current_status = _enum_value(getattr(config, "containerStatus", None))
        action = "restart" if current_status == "running" or bool(config.isActive) else "start"
        await db.botconfiguration.update(
            where={"id": data.botConfigId},
            data={
                "containerStatus": "starting",
            },
        )
        await _emit_lifecycle_event(
            bot_config_id=str(data.botConfigId),
            action=action,
            phase="requested",
            detail="Bot runtime start requested",
            status="running",
            owner_user_id=request.state.user_id,
        )

        try:
            runner_result = await asyncio.to_thread(
                run_bot_instance_action,
                action=action,
                instance_name=str(data.botConfigId),
                profile_name=str(runtime["profile_name"]),
                env_overrides=dict(runtime["runtime_env"] or {}),
            )
        except BotRunnerError as exc:
            await db.botconfiguration.update(
                where={"id": data.botConfigId},
                data={
                    "containerStatus": "error",
                    "isActive": False,
                }
            )
            await _emit_lifecycle_event(
                bot_config_id=str(data.botConfigId),
                action=action,
                phase="failed",
                detail=_runner_failure_detail(exc),
                status="error",
                owner_user_id=request.state.user_id,
            )
            raise HTTPException(
                status_code=500,
                detail=_runner_error_message("Failed to start bot docker instance", exc),
            ) from exc

        update_payload = {
            "containerStatus": "running",
            "isActive": True,
            "dockerContainerId": runner_result.container_id,
        }
        current_version_tag = str(getattr(config.botVersion, "versionTag", "") or "").strip()
        installed_version_tag = str(getattr(config, "installedVersionTag", "") or "").strip()
        if current_version_tag and not installed_version_tag:
            update_payload["installedVersionTag"] = current_version_tag
        installed_image_id = str(getattr(config, "installedDockerImageId", "") or "").strip()
        if runtime.get("docker_image_id") and not installed_image_id:
            update_payload["installedDockerImageId"] = runtime["docker_image_id"]
        await db.botconfiguration.update(
            where={"id": data.botConfigId},
            data=update_payload,
        )
        await _emit_lifecycle_event(
            bot_config_id=str(data.botConfigId),
            action=action,
            phase="succeeded",
            detail="Bot runtime is running",
            status="running",
            metadata={
                "docker_project_name": getattr(runner_result, "project_name", None),
                "docker_container_id": getattr(runner_result, "container_id", None),
            },
            owner_user_id=request.state.user_id,
        )
    else:
        await _emit_lifecycle_event(
            bot_config_id=str(data.botConfigId),
            action="stop",
            phase="requested",
            detail="Bot runtime stop requested",
            status="stopped",
            owner_user_id=request.state.user_id,
        )
        try:
            runner_result = await asyncio.to_thread(
                run_bot_instance_action,
                action="stop",
                instance_name=str(data.botConfigId),
                timeout_sec=300,
            )
        except BotRunnerError as exc:
            await _emit_lifecycle_event(
                bot_config_id=str(data.botConfigId),
                action="stop",
                phase="failed",
                detail=_runner_failure_detail(exc),
                status="error",
                owner_user_id=request.state.user_id,
            )
            raise HTTPException(
                status_code=500,
                detail=_runner_error_message("Failed to stop bot docker instance", exc),
            ) from exc

        await db.botconfiguration.update(
            where={"id": data.botConfigId},
            data={
                "containerStatus": "stopped",
                "isActive": False,
                "dockerContainerId": None,
            },
        )
        await _emit_lifecycle_event(
            bot_config_id=str(data.botConfigId),
            action="stop",
            phase="succeeded",
            detail="Bot runtime stopped",
            status="stopped",
            metadata={
                "docker_project_name": getattr(runner_result, "project_name", None),
                "docker_container_id": getattr(runner_result, "container_id", None),
            },
            owner_user_id=request.state.user_id,
        )

    # Log Activity
    try:
        user_agent = request.headers.get("user-agent", "Unknown")
        ip_address = request.client.host if request.client else "0.0.0.0"
        await db.activitylog.create(
            data={
                "userId": request.state.user_id,
                "topic": "Bot Control",
                "detail": f"Bot {data.botConfigId} status updated to {requested_status}",
                "ipAddress": ip_address,
                "deviceInfo": user_agent[:255]
            }
        )
    except Exception as e:
        print(f"Failed to log activity: {e}")

    return {
        "status_code": 200,
        "message": f"Bot status updated to {requested_status}",
        "docker_project_name": getattr(runner_result, "project_name", None),
        "docker_container_id": getattr(runner_result, "container_id", None),
    }


@bot_router.get('/runtime_health', tags=["bot"])
async def get_bot_runtime_health(request: Request, botConfigId: str):
    if not request.state.user_id:
        raise HTTPException(status_code=400, detail="User ID is required")

    config = await verify_bot_ownership(botConfigId, request.state.user_id)
    bot_config_id = str(config.id)
    container_status = _enum_value(getattr(config, "containerStatus", None))
    is_active = bool(getattr(config, "isActive", False))
    db_container_id = str(getattr(config, "dockerContainerId", "") or "").strip() or None
    live_hub_connected = bot_hub.get_bot(bot_config_id) is not None

    trade_allowed = None
    tradeapi_disabled = None
    health_detail = "container_not_running"
    docker_project_name = None
    docker_container_id = db_container_id
    probe_stdout = ""
    probe_stderr = ""

    should_probe_runtime = (container_status == "running" or is_active or bool(db_container_id)) and not live_hub_connected
    if live_hub_connected:
        # Live WS heartbeat from bot runtime is already active; avoid expensive MT5 RPC probing
        # on every polling request.
        health_detail = "live_hub_connected"
    if should_probe_runtime:
        try:
            probe = await asyncio.to_thread(
                check_bot_runtime_health,
                instance_name=bot_config_id,
            )
            docker_project_name = probe.project_name
            docker_container_id = probe.container_id or docker_container_id
            trade_allowed = probe.trade_allowed
            tradeapi_disabled = probe.tradeapi_disabled
            health_detail = probe.detail
            probe_stdout = probe.stdout
            probe_stderr = probe.stderr
        except BotRunnerError as exc:
            health_detail = _runner_error_message("Runtime health check failed", exc)
            probe_stdout = str(getattr(exc, "stdout", "") or "")
            probe_stderr = str(getattr(exc, "stderr", "") or "")

    return {
        "status_code": 200,
        "data": {
            "bot_config_id": bot_config_id,
            "magic_number": normalize_magic_number(getattr(config, "magicNumber", None)),
            "container_status": container_status,
            "is_active": is_active,
            "docker_project_name": docker_project_name,
            "docker_container_id": docker_container_id,
            "live_hub_connected": live_hub_connected,
            "trade_allowed": trade_allowed,
            "tradeapi_disabled": tradeapi_disabled,
            "health_detail": health_detail,
            "probe_stdout": probe_stdout,
            "probe_stderr": probe_stderr,
        },
    }


@bot_router.get('/operation_logs', tags=["bot"])
async def get_bot_operation_logs(request: Request, botConfigId: str, limit: int = 50):
    if not request.state.user_id:
        raise HTTPException(status_code=400, detail="User ID is required")

    config = await verify_bot_ownership(botConfigId, request.state.user_id)
    take = max(1, min(int(limit), 200))

    bot_operation_model = getattr(db, "botoperationlog", None)
    if bot_operation_model is None:
        logger.error(
            "Prisma client is missing model botoperationlog; run `prisma generate --schema=src/database/schema.prisma`."
        )
        return {
            "status_code": 200,
            "data": [],
        }

    logs = await bot_operation_model.find_many(
        where={"botConfigId": str(config.id)},
        order={"createdAt": "desc"},
        take=take,
    )

    return {
        "status_code": 200,
        "data": [
            {
                "id": str(log.id),
                "bot_config_id": str(log.botConfigId),
                "user_id": str(log.userId) if getattr(log, "userId", None) else None,
                "source": str(getattr(log, "source", "") or "").strip() or None,
                "action": str(getattr(log, "action", "") or "").strip() or None,
                "phase": str(getattr(log, "phase", "") or "").strip() or None,
                "level": str(getattr(log, "level", "") or "").strip() or "info",
                "message": str(getattr(log, "message", "") or "").strip() or None,
                "status": str(getattr(log, "status", "") or "").strip() or None,
                "meta": getattr(log, "meta", None),
                "created_at": log.createdAt.isoformat() if getattr(log, "createdAt", None) else None,
            }
            for log in logs
        ],
    }


@bot_router.patch('/update_risk', tags=["bot"])
async def update_bot_risk(request: Request, data: Update_Bot_Risk):
    if not request.state.user_id:
        raise HTTPException(status_code=400, detail="User ID is required")

    config = await verify_bot_ownership(data.botConfigId, request.state.user_id)

    requested_risk_mode = normalize_risk_mode(_enum_value(getattr(data, "riskMode", None)))
    requested_custom_lot = normalize_custom_lot(getattr(data, "customLot", None))
    if requested_risk_mode == CUSTOM_LOT_RISK_MODE and requested_custom_lot is None:
        raise HTTPException(status_code=400, detail="Custom lot must be at least 0.01")

    current_risk_level = normalize_risk_level(_enum_value(getattr(config, "riskLevel", None)))
    next_risk_level = (
        normalize_risk_level(data.riskLevel.value)
        if getattr(data, "riskLevel", None) is not None
        else current_risk_level
    )
    next_trading_settings = merge_bot_runtime_settings(
        getattr(config, "tradingSchedule", None),
        risk_mode=requested_risk_mode,
        custom_lot=requested_custom_lot if requested_risk_mode == CUSTOM_LOT_RISK_MODE else None,
    )

    update_payload: dict[str, object] = {
        "tradingSchedule": Json(next_trading_settings),
    }
    if requested_risk_mode == CUSTOM_LOT_RISK_MODE:
        if getattr(data, "riskLevel", None) is not None:
            update_payload["riskLevel"] = next_risk_level
    else:
        update_payload["riskLevel"] = next_risk_level

    await db.botconfiguration.update(
        where={"id": data.botConfigId},
        data=update_payload,
    )
    await bot_hub.send_bot_config(
        data.botConfigId,
        {
            "risk_level": next_risk_level,
            "risk_mode": requested_risk_mode,
            "custom_lot": requested_custom_lot if requested_risk_mode == CUSTOM_LOT_RISK_MODE else None,
        },
    )

    # Log Activity
    try:
        user_agent = request.headers.get("user-agent", "Unknown")
        ip_address = request.client.host if request.client else "0.0.0.0"
        await db.activitylog.create(
            data={
                "userId": request.state.user_id,
                "topic": "Bot Config",
                "detail": (
                    f"Bot {data.botConfigId} risk updated to custom lot "
                    f"{requested_custom_lot:.2f}"
                    if requested_risk_mode == CUSTOM_LOT_RISK_MODE and requested_custom_lot is not None
                    else f"Bot {data.botConfigId} risk level updated to {next_risk_level}"
                ),
                "ipAddress": ip_address,
                "deviceInfo": user_agent[:255]
            }
        )
    except Exception as e:
        print(f"Failed to log activity: {e}")

    return {
        "status_code": 200,
        "message": (
            f"Risk updated to custom lot {requested_custom_lot:.2f}"
            if requested_risk_mode == CUSTOM_LOT_RISK_MODE and requested_custom_lot is not None
            else f"Risk level updated to {next_risk_level}"
        ),
    }


@bot_router.patch('/update_schedule', tags=["bot"])
async def update_bot_schedule(request: Request, data: Update_Bot_Schedule):
    if not request.state.user_id:
        raise HTTPException(status_code=400, detail="User ID is required")

    config = await verify_bot_ownership(data.botConfigId, request.state.user_id)

    normalized_schedule = normalize_trading_schedule(data.tradingSchedule)
    next_trading_settings = merge_bot_runtime_settings(
        getattr(config, "tradingSchedule", None),
        schedule=normalized_schedule,
    )

    await db.botconfiguration.update(
        where={"id": data.botConfigId},
        data={"tradingSchedule": Json(next_trading_settings)}
    )
    await bot_hub.send_bot_config(
        data.botConfigId,
        {"trading_schedule": normalized_schedule},
    )

    # Log Activity
    try:
        user_agent = request.headers.get("user-agent", "Unknown")
        ip_address = request.client.host if request.client else "0.0.0.0"
        await db.activitylog.create(
            data={
                "userId": request.state.user_id,
                "topic": "Bot Config",
                "detail": f"Bot {data.botConfigId} trading schedule updated",
                "ipAddress": ip_address,
                "deviceInfo": user_agent[:255]
            }
        )
    except Exception as e:
        print(f"Failed to log activity: {e}")

    return {"status_code": 200, "message": "Trading schedule updated"}


@bot_router.patch('/emergency_stop', tags=["bot"])
async def emergency_stop_bot(request: Request, data: Emergency_Bot_Stop):
    if not request.state.user_id:
        raise HTTPException(status_code=400, detail="User ID is required")

    config = await verify_bot_ownership(data.botConfigId, request.state.user_id)
    current_status = _enum_value(getattr(config, "containerStatus", None))
    is_running = current_status == "running" or bool(getattr(config, "isActive", False))
    await _emit_lifecycle_event(
        bot_config_id=str(data.botConfigId),
        action="emergency_stop",
        phase="requested",
        detail="Emergency stop requested (close bot-managed positions + stop runtime)",
        status="stopped",
        owner_user_id=request.state.user_id,
    )

    close_result = "skipped_bot_offline"
    command_id = None
    ack_payload = None

    connected_bot = bot_hub.get_bot(data.botConfigId)
    if connected_bot is not None:
        command_id = await bot_hub.send_bot_command(
            data.botConfigId,
            "emergency_close_all",
            payload={"source": "server_emergency_stop"},
        )
        if command_id:
            ack_payload = await bot_hub.wait_for_command_ack(
                data.botConfigId,
                command_id,
                timeout_sec=60.0,
            )
            if ack_payload is None:
                close_result = "timeout_waiting_ack"
            elif bool((ack_payload or {}).get("ok")):
                close_result = "close_all_ok"
            else:
                close_result = "close_all_partial_or_failed"
        else:
            close_result = "command_send_failed"

    runner_result = None
    try:
        runner_result = await asyncio.to_thread(
            run_bot_instance_action,
            action="stop",
            instance_name=str(data.botConfigId),
            timeout_sec=300,
        )
    except BotRunnerError as exc:
        if is_running:
            await db.botconfiguration.update(
                where={"id": data.botConfigId},
                data={
                    "containerStatus": "error",
                    "isActive": False,
                },
            )
            await _emit_lifecycle_event(
                bot_config_id=str(data.botConfigId),
                action="emergency_stop",
                phase="failed",
                detail=_runner_failure_detail(exc),
                status="error",
                metadata={"close_result": close_result},
                owner_user_id=request.state.user_id,
            )
            raise HTTPException(
                status_code=500,
                detail=_runner_error_message("Emergency stop failed while stopping docker instance", exc),
            ) from exc

    await db.botconfiguration.update(
        where={"id": data.botConfigId},
        data={
            "containerStatus": "stopped",
            "isActive": False,
            "dockerContainerId": None,
        },
    )
    await _emit_lifecycle_event(
        bot_config_id=str(data.botConfigId),
        action="emergency_stop",
        phase="succeeded",
        detail="Emergency stop completed",
        status="stopped",
        metadata={
            "close_result": close_result,
            "docker_project_name": getattr(runner_result, "project_name", None),
            "docker_container_id": getattr(runner_result, "container_id", None),
        },
        owner_user_id=request.state.user_id,
    )

    # Log Activity
    try:
        user_agent = request.headers.get("user-agent", "Unknown")
        ip_address = request.client.host if request.client else "0.0.0.0"
        await db.activitylog.create(
            data={
                "userId": request.state.user_id,
                "topic": "Bot Emergency Stop",
                "detail": f"Bot {data.botConfigId} emergency stop (close_result={close_result})",
                "ipAddress": ip_address,
                "deviceInfo": user_agent[:255],
            }
        )
    except Exception as e:
        print(f"Failed to log activity: {e}")

    warning_message = None
    if close_result == "skipped_bot_offline":
        warning_message = "Bot was offline, so close-managed-position command was skipped before container stop."
    elif close_result == "timeout_waiting_ack":
        warning_message = "Close-managed-position command was sent but no acknowledgment was received before timeout."
    elif close_result in {"close_all_partial_or_failed", "command_send_failed"}:
        warning_message = "Close-managed-position command did not finish cleanly. Please verify open positions on MT5."

    return {
        "status_code": 200,
        "message": "Emergency stop completed",
        "close_result": close_result,
        "warning": warning_message,
        "bot_command_id": command_id,
        "bot_command_ack": ack_payload,
        "docker_project_name": getattr(runner_result, "project_name", None),
        "docker_container_id": getattr(runner_result, "container_id", None),
    }


@bot_router.patch('/change_model', tags=["bot"])
async def change_bot_model(request: Request, data: Change_Bot_Model):
    if not request.state.user_id:
        raise HTTPException(status_code=400, detail="User ID is required")

    config = await verify_bot_ownership(data.botConfigId, request.state.user_id)

    new_version = await db.botversion.find_unique(
        where={"modelId": data.newModelId}
    )
    if not new_version:
        raise HTTPException(status_code=404, detail="Bot version not found")
    if not getattr(new_version, "isActive", True):
        raise HTTPException(status_code=400, detail="Selected bot version is inactive")
    if str(getattr(config, "modelId", "") or "") == str(data.newModelId):
        return {"status_code": 200, "message": "Bot already uses this model", "restarted": False}

    conflict = await _find_account_symbol_timeframe_conflict(
        account_id=str(getattr(config, "accountId", "") or ""),
        symbol=getattr(new_version, "symbol", None),
        timeframe=getattr(new_version, "timeframe", None),
        exclude_config_id=str(getattr(config, "id", "") or ""),
    )
    if conflict is not None:
        symbol = _normalize_pair_value(getattr(new_version, "symbol", None)) or "UNKNOWN"
        timeframe = _normalize_pair_value(getattr(new_version, "timeframe", None)) or "UNKNOWN"
        raise HTTPException(
            status_code=400,
            detail=f"Another bot already uses {symbol} {timeframe} in this account",
        )

    current_status = _enum_value(getattr(config, "containerStatus", None))
    was_running = current_status == "running" or bool(getattr(config, "isActive", False))
    runner_result = None
    await _emit_lifecycle_event(
        bot_config_id=str(data.botConfigId),
        action="change_model",
        phase="requested",
        detail=f"Model change requested to {data.newModelId}",
        status="running" if was_running else "stopped",
        metadata={"new_model_id": str(data.newModelId)},
        owner_user_id=request.state.user_id,
    )

    if was_running:
        runtime = await _extract_runtime_context(
            config,
            image_override=str(getattr(new_version, "dockerImageId", "") or "").strip() or None,
            bot_version_override=new_version,
        )
        try:
            runner_result = await asyncio.to_thread(
                run_bot_instance_action,
                action="restart",
                instance_name=str(data.botConfigId),
                profile_name=str(runtime["profile_name"]),
                env_overrides=dict(runtime["runtime_env"] or {}),
            )
        except BotRunnerError as exc:
            await db.botconfiguration.update(
                where={"id": data.botConfigId},
                data={
                    "containerStatus": "error",
                    "isActive": False,
                },
            )
            await _emit_lifecycle_event(
                bot_config_id=str(data.botConfigId),
                action="change_model",
                phase="failed",
                detail=_runner_failure_detail(exc),
                status="error",
                metadata={"new_model_id": str(data.newModelId)},
                owner_user_id=request.state.user_id,
            )
            raise HTTPException(
                status_code=500,
                detail=_runner_error_message("Failed to restart bot with new model", exc),
            ) from exc

    update_payload = {
        "botVersion": {"connect": {"modelId": data.newModelId}},
        "installedDockerImageId": new_version.dockerImageId,
        "installedVersionTag": new_version.versionTag,
    }
    if was_running:
        update_payload["containerStatus"] = "running"
        update_payload["isActive"] = True
        update_payload["dockerContainerId"] = getattr(runner_result, "container_id", None)

    await db.botconfiguration.update(
        where={"id": data.botConfigId},
        data=update_payload,
    )
    await _emit_lifecycle_event(
        bot_config_id=str(data.botConfigId),
        action="change_model",
        phase="succeeded",
        detail="Model changed successfully",
        status="running" if was_running else "stopped",
        metadata={
            "new_model_id": str(data.newModelId),
            "restarted": was_running,
            "docker_project_name": getattr(runner_result, "project_name", None),
            "docker_container_id": getattr(runner_result, "container_id", None),
        },
        owner_user_id=request.state.user_id,
    )

    # Log Activity
    try:
        user_agent = request.headers.get("user-agent", "Unknown")
        ip_address = request.client.host if request.client else "0.0.0.0"
        await db.activitylog.create(
            data={
                "userId": request.state.user_id,
                "topic": "Bot Config",
                "detail": f"Bot {data.botConfigId} model changed to {data.newModelId} (restarted={int(was_running)})",
                "ipAddress": ip_address,
                "deviceInfo": user_agent[:255]
            }
        )
    except Exception as e:
        print(f"Failed to log activity: {e}")

    return {
        "status_code": 200,
        "message": "Bot model changed and restarted successfully" if was_running else "Bot model changed successfully",
        "restarted": was_running,
        "docker_project_name": getattr(runner_result, "project_name", None),
        "docker_container_id": getattr(runner_result, "container_id", None),
    }


@bot_router.patch('/apply_update', tags=["bot"])
async def apply_bot_update(request: Request, data: Apply_Bot_Update):
    if not request.state.user_id:
        raise HTTPException(status_code=400, detail="User ID is required")

    config = await verify_bot_ownership(data.botConfigId, request.state.user_id)

    bot_version = getattr(config, "botVersion", None)
    if not bot_version:
        raise HTTPException(status_code=404, detail="Bot version not found")

    latest_version_tag = str(getattr(bot_version, "versionTag", "") or "").strip()
    if not latest_version_tag:
        raise HTTPException(status_code=400, detail="Bot version has no version_tag configured")

    latest_image_id = str(getattr(bot_version, "dockerImageId", "") or "").strip() or None
    installed_version_tag = str(getattr(config, "installedVersionTag", "") or "").strip() or None
    installed_image_id = str(getattr(config, "installedDockerImageId", "") or "").strip() or None

    if not installed_version_tag:
        update_payload = {"installedVersionTag": latest_version_tag}
        if latest_image_id:
            update_payload["installedDockerImageId"] = latest_image_id
        await db.botconfiguration.update(
            where={"id": data.botConfigId},
            data=update_payload,
        )
        return {
            "status_code": 200,
            "message": "Bot version baseline initialized",
            "status": "up_to_date",
            "installed_version_tag": latest_version_tag,
            "latest_version_tag": latest_version_tag,
        }

    if installed_version_tag == latest_version_tag:
        return {
            "status_code": 200,
            "message": "Bot is already on latest version",
            "status": "up_to_date",
            "installed_version_tag": installed_version_tag,
            "latest_version_tag": latest_version_tag,
        }

    current_status = _enum_value(config.containerStatus)
    was_running = current_status == "running" or bool(config.isActive)
    if was_running:
        await db.botconfiguration.update(
            where={"id": data.botConfigId},
            data={
                "containerStatus": "starting",
                "isActive": True,
            },
        )
    await _emit_lifecycle_event(
        bot_config_id=str(data.botConfigId),
        action="apply_update",
        phase="requested",
        detail=f"Apply update requested ({installed_version_tag} -> {latest_version_tag})",
        status="starting" if was_running else "stopped",
        metadata={
            "installed_version_tag": installed_version_tag,
            "latest_version_tag": latest_version_tag,
        },
        owner_user_id=request.state.user_id,
    )

    runner_result = None
    operation = "metadata_only"
    if was_running:
        runtime = await _extract_runtime_context(
            config,
            image_override=latest_image_id,
            version_tag_override=latest_version_tag,
        )
        operation = "restart"
        try:
            runner_result = await asyncio.to_thread(
                run_bot_instance_action,
                action="restart",
                instance_name=str(data.botConfigId),
                profile_name=str(runtime["profile_name"]),
                env_overrides=dict(runtime["runtime_env"] or {}),
            )
        except BotRunnerError as exc:
            await db.botconfiguration.update(
                where={"id": data.botConfigId},
                data={
                    "containerStatus": "error",
                    "isActive": False,
                },
            )
            await _emit_lifecycle_event(
                bot_config_id=str(data.botConfigId),
                action="apply_update",
                phase="failed",
                detail=_runner_failure_detail(exc),
                status="error",
                owner_user_id=request.state.user_id,
            )
            raise HTTPException(
                status_code=500,
                detail=_runner_error_message("Failed to restart bot to apply version update", exc),
            ) from exc
        final_status = "running"
        is_active = True
        container_id = runner_result.container_id
    else:
        if latest_image_id and latest_image_id != installed_image_id:
            operation = "pull_only"
            try:
                runner_result = await asyncio.to_thread(
                    pull_docker_image,
                    latest_image_id,
                )
            except BotRunnerError as exc:
                await _emit_lifecycle_event(
                    bot_config_id=str(data.botConfigId),
                    action="apply_update",
                    phase="failed",
                    detail=_runner_failure_detail(exc),
                    status="error",
                    owner_user_id=request.state.user_id,
                )
                raise HTTPException(
                    status_code=500,
                    detail=_runner_error_message(f"Failed to pull docker image '{latest_image_id}'", exc),
                ) from exc
        final_status = "stopped"
        is_active = False
        container_id = None

    update_payload = {
        "installedVersionTag": latest_version_tag,
        "containerStatus": final_status,
        "isActive": is_active,
        "dockerContainerId": container_id,
    }
    if latest_image_id:
        update_payload["installedDockerImageId"] = latest_image_id

    await db.botconfiguration.update(
        where={"id": data.botConfigId},
        data=update_payload,
    )
    await _emit_lifecycle_event(
        bot_config_id=str(data.botConfigId),
        action="apply_update",
        phase="succeeded",
        detail=f"Bot update applied ({latest_version_tag})",
        status=final_status,
        metadata={
            "operation": operation,
            "previous_version_tag": installed_version_tag,
            "latest_version_tag": latest_version_tag,
            "docker_project_name": getattr(runner_result, "project_name", None),
            "docker_container_id": getattr(runner_result, "container_id", None),
        },
        owner_user_id=request.state.user_id,
    )

    bot_label = bot_version.label or "Trading Bot"
    version_tag = latest_version_tag
    await dispatch_notification_to_user_id(
        request.state.user_id,
        title=f"Bot updated: {bot_label}",
        message=f"{bot_label} is now on {version_tag}.",
        related_link="/bot-control",
        email_subject=f"Bot updated to {version_tag} - SmarfRobotTrade",
        action_label="Open bot control",
    )

    try:
        user_agent = request.headers.get("user-agent", "Unknown")
        ip_address = request.client.host if request.client else "0.0.0.0"
        await db.activitylog.create(
            data={
                "userId": request.state.user_id,
                "topic": "Bot Update",
                "detail": f"Bot {data.botConfigId} updated version from {installed_version_tag} to {latest_version_tag}",
                "ipAddress": ip_address,
                "deviceInfo": user_agent[:255],
            }
        )
    except Exception as e:
        print(f"Failed to log activity: {e}")

    return {
        "status_code": 200,
        "message": "Bot updated successfully",
        "previous_version_tag": installed_version_tag,
        "latest_version_tag": latest_version_tag,
        "latest_image_id": latest_image_id,
        "container_status": final_status,
        "operation": operation,
        "docker_project_name": getattr(runner_result, "project_name", None),
        "docker_container_id": getattr(runner_result, "container_id", None),
    }


@bot_router.delete('/delete', tags=["bot"])
async def delete_bot(request: Request, data: Delete_Bot):
    if not request.state.user_id:
        raise HTTPException(status_code=400, detail="User ID is required")

    config = await verify_bot_ownership(data.botConfigId, request.state.user_id)

    current_status = _enum_value(getattr(config, "containerStatus", None))
    is_running = current_status == "running" or bool(getattr(config, "isActive", False))
    await _emit_lifecycle_event(
        bot_config_id=str(data.botConfigId),
        action="delete",
        phase="requested",
        detail="Delete bot requested",
        status="deleted",
        owner_user_id=request.state.user_id,
    )
    try:
        await asyncio.to_thread(
            run_bot_instance_action,
            action="stop",
            instance_name=str(data.botConfigId),
            timeout_sec=300,
        )
    except BotRunnerError as exc:
        if is_running:
            await _emit_lifecycle_event(
                bot_config_id=str(data.botConfigId),
                action="delete",
                phase="failed",
                detail=_runner_failure_detail(exc),
                status="error",
                owner_user_id=request.state.user_id,
            )
            raise HTTPException(
                status_code=500,
                detail=_runner_error_message("Failed to stop bot docker instance before delete", exc),
            ) from exc

    try:
        await asyncio.to_thread(
            purge_bot_instance_state,
            instance_name=str(data.botConfigId),
            timeout_sec=300,
        )
    except BotRunnerError as exc:
        await _emit_lifecycle_event(
            bot_config_id=str(data.botConfigId),
            action="delete",
            phase="failed",
            detail=f"purge_failed: {exc}",
            status="error",
            owner_user_id=request.state.user_id,
        )
        raise HTTPException(
            status_code=500,
            detail=_runner_error_message("Failed to purge bot runtime state during delete", exc),
        ) from exc

    await db.botconfiguration.update(
        where={"id": data.botConfigId},
        data={
            "containerStatus": "stopped",
            "isActive": False,
            "dockerContainerId": None,
            "recordStatus": "deleted",
            "deletedAt": datetime.utcnow(),
        },
    )
    await _emit_lifecycle_event(
        bot_config_id=str(data.botConfigId),
        action="delete",
        phase="succeeded",
        detail="Bot configuration archived",
        status="deleted",
        owner_user_id=request.state.user_id,
    )

    return {"status_code": 200, "message": "Bot configuration archived"}
