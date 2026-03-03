from fastapi import APIRouter, Request, Query
from ..database.client import db
from typing import List, Optional
from pydantic import BaseModel

class SearchResult(BaseModel):
    id: str
    type: str  # "bot" or "account"
    label: str
    subLabel: Optional[str]
    link: str

search_router = APIRouter(tags=["Search"])

@search_router.get("/", response_model=List[SearchResult])
async def search_all(request: Request, q: str = Query(..., min_length=2)):
    user_id = request.state.user_id
    query = q.lower()
    results = []
    
    # 1. Search Trading Accounts
    accounts = await db.tradingaccount.find_many(
        where={
            "userId": user_id,
            "recordStatus": "active",
            "OR": [
                {"mt5LoginId": {"contains": query}},
                {"brokerName": {"contains": query, "mode": "insensitive"}}
            ]
        },
        take=5
    )
    
    for acc in accounts:
        results.append(SearchResult(
            id=acc.id,
            type="account",
            label=f"{acc.brokerName} - {acc.mt5LoginId}",
            subLabel=acc.serverName,
            link=f"/settings?tab=accounts" # Or dedicated account page if exists
        ))
        
    # 2. Search Bot Configurations (joined with Version)
    bots = await db.botconfiguration.find_many(
        where={
            "recordStatus": "active",
            "account": {
                "userId": user_id,
                "recordStatus": "active",
            },
            "botVersion": {
                "OR": [
                    {"label": {"contains": query, "mode": "insensitive"}},
                    {"symbol": {"contains": query, "mode": "insensitive"}}
                ]
            }
        },
        include={
            "botVersion": True,
            "account": True
        },
        take=5
    )
    
    for bot in bots:
        if bot.botVersion:
            results.append(SearchResult(
                id=bot.id,
                type="bot",
                label=f"{bot.botVersion.label} ({bot.botVersion.symbol})",
                subLabel=f"Account: {bot.account.mt5LoginId}",
                link=f"/bot-control?botId={bot.id}"
            ))
            
    return results[:10] 
