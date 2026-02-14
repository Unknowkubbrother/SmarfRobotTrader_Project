from fastapi import APIRouter, HTTPException, Depends, status, Response, Request, Form
from pydantic import BaseModel
from prisma import Json
from ..models.bot import (
    Create_Bot_Version, Create_Bot_Configuration, RiskLevelEnum,
    Update_Bot_Status, Update_Bot_Risk, Update_Bot_Schedule,
    Change_Bot_Model, Delete_Bot
)
from ..database.client import db

bot_router = APIRouter()


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
        result.append({
            "id": str(bv.modelId),
            "model_id": str(bv.modelId),
            "label": bv.label,
            "version_tag": bv.versionTag,
            "symbol": bv.symbol,
            "timeframe": bv.timeframe,
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
    
    tradingSchedule = {
        "mon": True,
        "tue": True,
        "wed": True,
        "thu": True,
        "fri": True
    }

    bot_configuration = await db.botconfiguration.create(
        data={
            "account": {"connect": {"id": data.accountId}},
            "botVersion": {"connect": {"modelId": data.modelId}},
            "riskLevel": data.riskLevel.value,
            "tradingSchedule": Json(tradingSchedule),
            "isActive": False,
            "containerStatus": "stopped",
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

    if data.status not in ("running", "stopped"):
        raise HTTPException(status_code=400, detail="Status must be 'running' or 'stopped'")
    if data.status == "running" and config.botVersion and not getattr(config.botVersion, "isActive", True):
        raise HTTPException(status_code=400, detail="Cannot run bot. Its version is inactive.")

    await db.botconfiguration.update(
        where={"id": data.botConfigId},
        data={
            "containerStatus": data.status,
            "isActive": data.status == "running"
        }
    )

    # Log Activity
    try:
        user_agent = request.headers.get("user-agent", "Unknown")
        ip_address = request.client.host if request.client else "0.0.0.0"
        await db.activitylog.create(
            data={
                "userId": request.state.user_id,
                "topic": "Bot Control",
                "detail": f"Bot {data.botConfigId} status updated to {data.status}",
                "ipAddress": ip_address,
                "deviceInfo": user_agent[:255]
            }
        )
    except Exception as e:
        print(f"Failed to log activity: {e}")

    return {"status_code": 200, "message": f"Bot status updated to {data.status}"}


@bot_router.patch('/update_risk', tags=["bot"])
async def update_bot_risk(request: Request, data: Update_Bot_Risk):
    if not request.state.user_id:
        raise HTTPException(status_code=400, detail="User ID is required")

    await verify_bot_ownership(data.botConfigId, request.state.user_id)

    await db.botconfiguration.update(
        where={"id": data.botConfigId},
        data={"riskLevel": data.riskLevel.value}
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

    await db.botconfiguration.update(
        where={"id": data.botConfigId},
        data={"tradingSchedule": Json(data.tradingSchedule)}
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


@bot_router.patch('/change_model', tags=["bot"])
async def change_bot_model(request: Request, data: Change_Bot_Model):
    if not request.state.user_id:
        raise HTTPException(status_code=400, detail="User ID is required")

    await verify_bot_ownership(data.botConfigId, request.state.user_id)

    new_version = await db.botversion.find_unique(
        where={"modelId": data.newModelId}
    )
    if not new_version:
        raise HTTPException(status_code=404, detail="Bot version not found")

    await db.botconfiguration.update(
        where={"id": data.botConfigId},
        data={"botVersion": {"connect": {"modelId": data.newModelId}}}
    )

    # Log Activity
    try:
        user_agent = request.headers.get("user-agent", "Unknown")
        ip_address = request.client.host if request.client else "0.0.0.0"
        await db.activitylog.create(
            data={
                "userId": request.state.user_id,
                "topic": "Bot Config",
                "detail": f"Bot {data.botConfigId} model changed to {data.newModelId}",
                "ipAddress": ip_address,
                "deviceInfo": user_agent[:255]
            }
        )
    except Exception as e:
        print(f"Failed to log activity: {e}")

    return {"status_code": 200, "message": "Bot model changed successfully"}


@bot_router.delete('/delete', tags=["bot"])
async def delete_bot(request: Request, data: Delete_Bot):
    if not request.state.user_id:
        raise HTTPException(status_code=400, detail="User ID is required")

    await verify_bot_ownership(data.botConfigId, request.state.user_id)

    await db.botconfiguration.delete(
        where={"id": data.botConfigId}
    )

    return {"status_code": 200, "message": "Bot configuration deleted"}
