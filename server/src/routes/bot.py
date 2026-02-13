from fastapi import APIRouter, HTTPException, Depends, status, Response, Request, Form
from pydantic import BaseModel
from ..models.bot import Create_Bot_Version,Create_Bot_Configuration, RiskLevelEnum
from ..database.client import db
from json import dumps

bot_router = APIRouter()

## @@comment bot version
@bot_router.post("/create_bot_version", tags=["bot"])
async def create_bot_version(request: Request, data: Create_Bot_Version):
    if not request.state.user_id:
        raise HTTPException(status_code=400, detail="User ID is required")

    user = await db.user.find_unique(
        where={
            "id": request.state.user_id
        }
    )
    
    if not user or user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")

    bot_version = await db.botversion.create(
        data={
            "label": data.label,
            "dockerImageId": data.dockerImageId,
            "versionTag": data.versionTag,
            "symbol": data.symbol,
            "timeframe": data.timeframe,
            "releaseNotes": data.releaseNotes
        }
    )
    
    if not bot_version:
        raise HTTPException(status_code=400, detail="Bot version creation failed")
    
    return {
        "status_code": 200,
        "message": "Bot version created successfully"
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

    bot_configuration_count = await db.botconfiguration.count(
        where={
            "accountId": data.accountId,
            "modelId": data.modelId
        }
    )

    bot_configuration_count = bot_configuration_count + 1

    botInstanceId = 1000 + bot_configuration_count
    
    tradingSchedule = dumps({
        "fri": True,
        "mon": True,
        "sat": False,
        "sun": False,
        "thu": True,
        "tue": True,
        "wed": True
    })

    bot_configuration = await db.botconfiguration.create(
        data={
            "accountId": data.accountId,
            "modelId": data.modelId,
            "riskLevel": data.riskLevel.value,
            "tradingSchedule": tradingSchedule,
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
    