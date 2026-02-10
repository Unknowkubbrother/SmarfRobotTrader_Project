from fastapi import APIRouter, HTTPException, Depends, status, Response, Request, Form
from pydantic import BaseModel
from cryptography.fernet import Fernet
import base64
import hashlib
import os
from ..models.trading_model import Create_Trading_Account
from ..database.client import db

trading_router = APIRouter()

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