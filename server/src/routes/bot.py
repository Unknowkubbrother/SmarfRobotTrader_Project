import asyncio

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
from ..utils.trading_schedule import normalize_trading_schedule
from ..utils.mt5_bot_runner import (
    BotRunnerError,
    build_bot_runtime_env,
    build_profile_name,
    decrypt_mt5_password,
    pull_docker_image,
    run_bot_instance_action,
)
from ..utils.ws_manager import bot_hub

bot_router = APIRouter()


def _enum_value(value):
    return value.value if hasattr(value, "value") else value


def _extract_runtime_context(
    config,
    image_override: str | None = None,
    bot_version_override=None,
) -> dict[str, str | None]:
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

    image_ref = (
        str(image_override).strip()
        if image_override and str(image_override).strip()
        else str(getattr(bot_version, "dockerImageId", "") or "").strip() or None
    )

    try:
        profile_name = build_profile_name(live_symbol, live_timeframe)
    except BotRunnerError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    runtime_env = build_bot_runtime_env(
        bot_config_id=str(config.id),
        mt5_login=mt5_login,
        mt5_password=mt5_password,
        mt5_server=mt5_server,
        live_symbol=live_symbol,
        live_timeframe=live_timeframe,
        docker_image_id=image_ref,
    )

    return {
        "profile_name": profile_name,
        "docker_image_id": image_ref,
        "runtime_env": runtime_env,
    }


def _runner_error_message(prefix: str, exc: BotRunnerError) -> str:
    stderr = str(getattr(exc, "stderr", "") or "").strip()
    stdout = str(getattr(exc, "stdout", "") or "").strip()
    if stderr:
        return f"{prefix}: {exc}. stderr={stderr}"
    if stdout:
        return f"{prefix}: {exc}. stdout={stdout}"
    return f"{prefix}: {exc}"


# === Helper: verify bot config ownership ===
async def verify_bot_ownership(bot_config_id: str, user_id: str):
    config = await db.botconfiguration.find_unique(
        where={"id": bot_config_id},
        include={"account": True, "botVersion": True}
    )
    if not config:
        raise HTTPException(status_code=404, detail="Bot configuration not found")
    if config.account.userId != user_id:
        raise HTTPException(status_code=403, detail="Not authorized")
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

    trading_account = await db.tradingaccount.find_unique(
        where={
            "id": data.accountId
        }
    )
    
    if not trading_account:
        raise HTTPException(status_code=404, detail="Trading account not found")

    if trading_account.userId != request.state.user_id:
        raise HTTPException(status_code=403, detail="You are not authorized to create bot configuration for this trading account")

    bot_version = await db.botversion.find_unique(
        where={
            "modelId": data.modelId
        }
    )

    if not bot_version:
        raise HTTPException(status_code=404, detail="Bot version not found")
    if not getattr(bot_version, "isActive", True):
        raise HTTPException(status_code=400, detail="This bot version is inactive")

    bot_configuration_count = await db.botconfiguration.count(
        where={
            "accountId": data.accountId
        }
    )

    bot_configuration_count = bot_configuration_count + 1

    botInstanceId = 1000 + bot_configuration_count
    
    tradingSchedule = normalize_trading_schedule({})

    bot_configuration = await db.botconfiguration.create(
        data={
            "account": {"connect": {"id": data.accountId}},
            "botVersion": {"connect": {"modelId": data.modelId}},
            "riskLevel": data.riskLevel.value,
            "tradingSchedule": Json(tradingSchedule),
            "isActive": False,
            "containerStatus": "stopped",
            "installedDockerImageId": bot_version.dockerImageId,
            "installedVersionTag": bot_version.versionTag,
            "botInstanceId": botInstanceId
        }
    )
    
    if not bot_configuration:
        raise HTTPException(status_code=400, detail="Bot configuration creation failed")
    
    return {
        "status_code": 200,
        "message": "Bot configuration created successfully"
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
    if requested_status == "running" and config.botVersion and not getattr(config.botVersion, "isActive", True):
        raise HTTPException(status_code=400, detail="Cannot run bot. Its version is inactive.")

    runner_result = None
    if requested_status == "running":
        runtime = _extract_runtime_context(config)
        current_status = _enum_value(getattr(config, "containerStatus", None))
        action = "restart" if current_status == "running" or bool(config.isActive) else "start"

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
        if current_version_tag:
            update_payload["installedVersionTag"] = current_version_tag
        if runtime.get("docker_image_id"):
            update_payload["installedDockerImageId"] = runtime["docker_image_id"]
        await db.botconfiguration.update(
            where={"id": data.botConfigId},
            data=update_payload,
        )
    else:
        try:
            runner_result = await asyncio.to_thread(
                run_bot_instance_action,
                action="stop",
                instance_name=str(data.botConfigId),
                timeout_sec=300,
            )
        except BotRunnerError as exc:
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


@bot_router.patch('/update_risk', tags=["bot"])
async def update_bot_risk(request: Request, data: Update_Bot_Risk):
    if not request.state.user_id:
        raise HTTPException(status_code=400, detail="User ID is required")

    await verify_bot_ownership(data.botConfigId, request.state.user_id)

    await db.botconfiguration.update(
        where={"id": data.botConfigId},
        data={"riskLevel": data.riskLevel.value}
    )
    await bot_hub.send_bot_config(
        data.botConfigId,
        {"risk_level": data.riskLevel.value},
    )

    # Log Activity
    try:
        user_agent = request.headers.get("user-agent", "Unknown")
        ip_address = request.client.host if request.client else "0.0.0.0"
        await db.activitylog.create(
            data={
                "userId": request.state.user_id,
                "topic": "Bot Config",
                "detail": f"Bot {data.botConfigId} risk level updated to {data.riskLevel.value}",
                "ipAddress": ip_address,
                "deviceInfo": user_agent[:255]
            }
        )
    except Exception as e:
        print(f"Failed to log activity: {e}")

    return {"status_code": 200, "message": "Risk level updated"}


@bot_router.patch('/update_schedule', tags=["bot"])
async def update_bot_schedule(request: Request, data: Update_Bot_Schedule):
    if not request.state.user_id:
        raise HTTPException(status_code=400, detail="User ID is required")

    await verify_bot_ownership(data.botConfigId, request.state.user_id)

    normalized_schedule = normalize_trading_schedule(data.tradingSchedule)

    await db.botconfiguration.update(
        where={"id": data.botConfigId},
        data={"tradingSchedule": Json(normalized_schedule)}
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
        warning_message = "Bot was offline, so close-position command was skipped before container stop."
    elif close_result == "timeout_waiting_ack":
        warning_message = "Close-position command was sent but no acknowledgment was received before timeout."
    elif close_result in {"close_all_partial_or_failed", "command_send_failed"}:
        warning_message = "Close-position command did not finish cleanly. Please verify open positions on MT5."

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

    current_status = _enum_value(getattr(config, "containerStatus", None))
    was_running = current_status == "running" or bool(getattr(config, "isActive", False))
    runner_result = None

    if was_running:
        runtime = _extract_runtime_context(
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

    runner_result = None
    operation = "metadata_only"
    if was_running:
        runtime = _extract_runtime_context(config, image_override=latest_image_id)
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

    bot_label = bot_version.label or "Trading Bot"
    version_tag = latest_version_tag
    await db.notification.create(
        data={
            "userId": request.state.user_id,
            "title": f"Bot updated: {bot_label}",
            "message": f"{bot_label} is now on {version_tag}.",
            "relatedLink": "/bot-control",
        }
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
    try:
        await asyncio.to_thread(
            run_bot_instance_action,
            action="stop",
            instance_name=str(data.botConfigId),
            timeout_sec=300,
        )
    except BotRunnerError as exc:
        if is_running:
            raise HTTPException(
                status_code=500,
                detail=_runner_error_message("Failed to stop bot docker instance before delete", exc),
            ) from exc

    await db.botconfiguration.delete(
        where={"id": data.botConfigId}
    )

    return {"status_code": 200, "message": "Bot configuration deleted"}
