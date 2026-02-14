from fastapi import APIRouter, HTTPException, Depends, status, Response, Request, Form
from pydantic import BaseModel
from cryptography.fernet import Fernet
import base64
import hashlib
import os
from datetime import date, datetime
from ..models.trading_model import Create_Trading_Account
from ..database.client import db

trading_router = APIRouter()

@trading_router.get("/calendar", tags=["trading"])
async def get_trading_calendar(request: Request, year: int, month: int):
    if not request.state.user_id:
        raise HTTPException(status_code=400, detail="User ID is required")
    
    # Calculate start and end date for the month
    try:
        start_date = date(year, month, 1)
        if month == 12:
            end_date = date(year + 1, 1, 1)
        else:
            end_date = date(year, month + 1, 1)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid year or month")
    
    # Fetch aggregates for all user's accounts within the date range
    aggregates = await db.dailyaggregate.find_many(
        where={
            "account": {
                "userId": request.state.user_id
            },
            "date": {
                "gte": datetime.combine(start_date, datetime.min.time()),
                "lt": datetime.combine(end_date, datetime.min.time())
            }
        }
    )
    
    # Group by date and sum profits
    calendar_data = {}
    for agg in aggregates:
        # agg.date might be datetime or date, normalize to day number
        day = agg.date.day
        if day not in calendar_data:
            calendar_data[day] = {
                "date": day,
                "profit": 0.0,
                "trades": 0,
                "winRate": 0.0 # Placeholder, would need detailed trade data for real winrate
            }
        
        # safely add values
        profit = float(agg.dailyNetProfit) if agg.dailyNetProfit else 0.0
        trades = agg.totalTrades if agg.totalTrades else 0
        
        calendar_data[day]["profit"] += profit
        calendar_data[day]["trades"] += trades
        
    # Convert to list
    return {
        "status_code": 200,
        "data": list(calendar_data.values())
    }

@trading_router.get("/accounts_with_bots", tags=["trading"])
async def get_accounts_with_bots(request: Request):
    if not request.state.user_id:
        raise HTTPException(status_code=400, detail="User ID is required")
    
    try:
        trading_accounts = await db.tradingaccount.find_many(
            where={
                "userId": request.state.user_id
            },
            include={
                "botConfigurations": {
                    "include": {
                        "botVersion": True
                    }
                },
                "dailyAggregates": True
            }
        )

        today_str = date.today().isoformat()
        result = []
        for account in trading_accounts:
            bot_configs = []
            for config in account.botConfigurations:
                bot_version = None
                latest_image_id = None
                latest_release_notes = []
                latest_release_date = None
                latest_version_tag = None
                if config.botVersion:
                    bv = config.botVersion
                    latest_image_id = bv.dockerImageId
                    latest_release_notes = bv.releaseNotes or []
                    latest_release_date = str(bv.releaseDate) if bv.releaseDate else None
                    latest_version_tag = bv.versionTag
                    bot_version = {
                        "model_id": str(bv.modelId),
                        "label": bv.label,
                        "docker_image_id": bv.dockerImageId,
                        "version_tag": bv.versionTag,
                        "symbol": bv.symbol,
                        "timeframe": bv.timeframe,
                        "release_notes": bv.releaseNotes,
                    }

                installed_image_id = getattr(config, "installedDockerImageId", None)
                effective_installed_image_id = installed_image_id or latest_image_id
                has_pending_update = bool(
                    latest_image_id
                    and effective_installed_image_id
                    and latest_image_id != effective_installed_image_id
                )

                bot_configs.append({
                    "id": str(config.id),
                    "account_id": str(config.accountId),
                    "model_id": str(config.modelId),
                    "bot_instance_id": config.botInstanceId,
                    "risk_level": config.riskLevel if config.riskLevel else None,
                    "trading_schedule": config.tradingSchedule,
                    "is_active": config.isActive,
                    "docker_container_id": config.dockerContainerId,
                    "installed_docker_image_id": effective_installed_image_id,
                    "container_status": config.containerStatus if config.containerStatus else None,
                    "has_pending_update": has_pending_update,
                    "latest_docker_image_id": latest_image_id,
                    "latest_version_tag": latest_version_tag,
                    "latest_release_notes": latest_release_notes if has_pending_update else [],
                    "latest_release_date": latest_release_date if has_pending_update else None,
                    "bot_version": bot_version,
                })

            today_agg = next(
                (a for a in account.dailyAggregates if str(a.date.date() if hasattr(a.date, 'date') else a.date) == today_str),
                None
            )
            total_today_pnl = float(today_agg.dailyNetProfit) if today_agg and today_agg.dailyNetProfit else 0

            result.append({
                "id": str(account.id),
                "user_id": str(account.userId),
                "broker_name": account.brokerName,
                "server_name": account.serverName,
                "mt5_login_id": account.mt5LoginId,
                "balance": float(account.balance) if account.balance else 0,
                "equity": float(account.equity) if account.equity else 0,
                "leverage": account.leverage,
                "margin": float(account.margin) if account.margin else 0,
                "margin_free": float(account.marginFree) if account.marginFree else 0,
                "margin_level": float(account.marginLevel) if account.marginLevel else 0,
                "created_at": str(account.createdAt) if account.createdAt else None,
                "bot_configurations": bot_configs,
                "total_today_pnl": total_today_pnl,
            })

        return {
            "status_code": 200,
            "data": result
        }
    except Exception as e:
        print(f"[ERROR] get_accounts_with_bots: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@trading_router.post("/create_account", tags=["trading"])
async def create_account(request: Request, data: Create_Trading_Account):
    if not request.state.user_id:
        raise HTTPException(status_code=400, detail="User ID is required")
    
    key = base64.urlsafe_b64encode(hashlib.sha256(os.getenv("SECRET_KEY", "UknownmeInLove").encode()).digest())
    fernet = Fernet(key)
    
    encrypted_password = fernet.encrypt(data.mt5Password.encode()).decode()
    userId = request.state.user_id

    trading_account = await db.tradingaccount.create(
        data={
            "userId": userId,
            "brokerName": data.brokerName,
            "serverName": data.serverName,
            "mt5LoginId": data.mt5LoginId,
            "mt5Password": encrypted_password
        }
    )

    if not trading_account:
        raise HTTPException(status_code=400, detail="Trading account creation failed")

    await db.dailyaggregate.create(
        data={
            "account": {"connect": {"id": trading_account.id}},
            "date": datetime.combine(date.today(), datetime.min.time()),
            "dailyNetProfit": 0,
            "totalTrades": 0
        }
    )

    return {
        "status_code": 200,
        "message": "Trading account created successfully"
    }

@trading_router.get("/", tags=["trading"])
async def trading_by_user(request: Request, accountId: str):
    if not request.state.user_id:
        raise HTTPException(status_code=400, detail="User ID is required")
    
    trading_account = await db.tradingaccount.find_first(
        where={
            "id": accountId,
            "userId": request.state.user_id
        }
    )
    
    if not trading_account:
        raise HTTPException(status_code=400, detail="Trading account not found")
    
    return {
        "status_code": 200,
        "message": trading_account
    }

@trading_router.get("/gets_trading_account", tags=["trading"])
async def trading_by_user(request: Request):
    if not request.state.user_id:
        raise HTTPException(status_code=400, detail="User ID is required")
    
    trading_account = await db.tradingaccount.find_many(
        where={
            "userId": request.state.user_id
        }
    )
    
    if not trading_account:
        raise HTTPException(status_code=400, detail="Trading account not found")
    
    return {
        "status_code": 200,
        "message": trading_account
    }

    
@trading_router.get("/gets_trading_account_admin", tags=["trading"])
async def trading_by_user_admin(request: Request, userId: str):
    requester_id = request.state.user_id
    if not requester_id:
        raise HTTPException(status_code=401, detail="User ID is required")

    admin_user = await db.user.find_unique(
        where={
            "id": requester_id
        }
    )
    
    if not admin_user or admin_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    trading_accounts = await db.tradingaccount.find_many(
        where={
            "userId": userId
        }
    )
    
    if not trading_accounts:
        raise HTTPException(status_code=400, detail="Trading account not found")
    
    return {
        "status_code": 200,
        "message": trading_accounts
    }
