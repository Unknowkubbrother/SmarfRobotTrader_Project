from fastapi import APIRouter, HTTPException, Depends, status, Response, Request, Form
from pydantic import BaseModel
from cryptography.fernet import Fernet
from prisma import Json
import base64
import hashlib
import os
from datetime import date, datetime, timedelta
from ..models.trading_model import Create_Trading_Account, UpsertTradingJournalRequest
from ..database.client import db
from ..utils.trading_schedule import normalize_trading_schedule

trading_router = APIRouter()


def _as_date(value) -> date:
    if isinstance(value, datetime):
        return value.date()
    return value


def _to_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _normalize_string_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if not isinstance(value, (list, tuple)):
        return []

    out: list[str] = []
    for item in value:
        txt = str(item or "").strip()
        if txt:
            out.append(txt)
    return out


async def _get_user_account_ids(user_id: str) -> list[str]:
    accounts = await db.tradingaccount.find_many(
        where={"userId": user_id},
    )
    return [str(a.id) for a in accounts]


async def _sync_daily_aggregates_from_orders(
    user_id: str,
    start_date: date,
    end_date: date,
):
    start_dt = datetime.combine(start_date, datetime.min.time())
    end_dt = datetime.combine(end_date, datetime.min.time())

    accounts = await db.tradingaccount.find_many(
        where={"userId": user_id},
    )
    account_ids = [str(a.id) for a in accounts]
    if not account_ids:
        return {}

    orders = await db.orderhistory.find_many(
        where={
            "accountId": {"in": account_ids},
            "closeTime": {"gte": start_dt, "lt": end_dt},
        },
    )

    account_day = {}
    day_rollup = {}
    for order in orders:
        close_time = getattr(order, "closeTime", None)
        if close_time is None:
            continue
        trade_day = _as_date(close_time)
        if trade_day < start_date or trade_day >= end_date:
            continue

        account_id = str(order.accountId)
        pnl = _to_float(getattr(order, "profit", 0.0))

        acc_key = (account_id, trade_day)
        acc_item = account_day.setdefault(
            acc_key,
            {"profit": 0.0, "trades": 0},
        )
        acc_item["profit"] += pnl
        acc_item["trades"] += 1

        day_item = day_rollup.setdefault(
            int(trade_day.day),
            {"wins": 0, "trades": 0},
        )
        day_item["trades"] += 1
        if pnl > 0:
            day_item["wins"] += 1

    existing_rows = await db.dailyaggregate.find_many(
        where={
            "accountId": {"in": account_ids},
            "date": {"gte": start_dt, "lt": end_dt},
        },
    )
    existing_by_key = {
        (str(row.accountId), _as_date(row.date)): row
        for row in existing_rows
    }

    for (account_id, trade_day), stats in account_day.items():
        day_dt = datetime.combine(trade_day, datetime.min.time())
        daily_profit = round(float(stats["profit"]), 2)
        total_trades = int(stats["trades"])
        existing = existing_by_key.get((account_id, trade_day))
        if existing:
            await db.dailyaggregate.update(
                where={"id": existing.id},
                data={
                    "dailyNetProfit": daily_profit,
                    "totalTrades": total_trades,
                },
            )
        else:
            await db.dailyaggregate.create(
                data={
                    "account": {"connect": {"id": account_id}},
                    "date": day_dt,
                    "dailyNetProfit": daily_profit,
                    "totalTrades": total_trades,
                }
            )

    return day_rollup

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
    
    # Pull monthly trade history into daily_aggregates so calendar stays synced
    try:
        day_rollup = await _sync_daily_aggregates_from_orders(
            user_id=request.state.user_id,
            start_date=start_date,
            end_date=end_date,
        )
    except Exception as e:
        print(f"[WARN] trading calendar sync failed: {e}")
        day_rollup = {}

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
        profit = _to_float(agg.dailyNetProfit, 0.0)
        trades = agg.totalTrades if agg.totalTrades else 0
        
        calendar_data[day]["profit"] += profit
        calendar_data[day]["trades"] += trades

    for day, payload in calendar_data.items():
        payload["profit"] = round(float(payload["profit"]), 2)
        wins = int(day_rollup.get(day, {}).get("wins", 0))
        trades = int(payload["trades"] or 0)
        payload["winRate"] = round((wins / trades) * 100, 1) if trades > 0 else 0.0

    data = [calendar_data[d] for d in sorted(calendar_data.keys())]
    total_profit = round(sum(float(d["profit"]) for d in data), 2)
    total_trades = int(sum(int(d["trades"]) for d in data))
    trading_days = int(sum(1 for d in data if int(d["trades"]) > 0))
    profitable_days = int(sum(1 for d in data if int(d["trades"]) > 0 and float(d["profit"]) > 0))
    average_win_rate = round(
        (sum(float(d["winRate"]) for d in data if int(d["trades"]) > 0) / trading_days)
        if trading_days > 0 else 0.0,
        1,
    )

    return {
        "status_code": 200,
        "data": data,
        "summary": {
            "month": month,
            "year": year,
            "totalProfit": total_profit,
            "totalTrades": total_trades,
            "tradingDays": trading_days,
            "profitableDays": profitable_days,
            "averageWinRate": average_win_rate,
        }
    }


@trading_router.get("/history_by_day", tags=["trading"])
async def get_trading_history_by_day(request: Request, year: int, month: int, day: int):
    if not request.state.user_id:
        raise HTTPException(status_code=400, detail="User ID is required")

    try:
        target_date = date(year, month, day)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date")

    start_dt = datetime.combine(target_date, datetime.min.time())
    end_dt = start_dt + timedelta(days=1)

    accounts = await db.tradingaccount.find_many(
        where={"userId": request.state.user_id},
    )
    account_ids = [str(a.id) for a in accounts]
    if not account_ids:
        return {
            "status_code": 200,
            "data": [],
            "summary": {
                "date": target_date.isoformat(),
                "totalTrades": 0,
                "netProfit": 0.0,
                "wins": 0,
                "losses": 0,
            },
        }

    orders = await db.orderhistory.find_many(
        where={
            "accountId": {"in": account_ids},
            "closeTime": {"gte": start_dt, "lt": end_dt},
        },
        order={"closeTime": "desc"},
    )

    rows = []
    total_profit = 0.0
    wins = 0
    losses = 0

    for order in orders:
        profit = _to_float(getattr(order, "profit", 0.0), 0.0)
        total_profit += profit
        if profit > 0:
            wins += 1
        elif profit < 0:
            losses += 1

        open_time = getattr(order, "openTime", None)
        close_time = getattr(order, "closeTime", None)
        rows.append(
            {
                "ticketId": int(getattr(order, "ticketId", 0) or 0),
                "accountId": str(getattr(order, "accountId", "") or ""),
                "symbol": str(getattr(order, "symbol", "") or ""),
                "type": str(getattr(order, "type", "") or "").upper(),
                "status": str(getattr(order, "status", "") or ""),
                "volume": _to_float(getattr(order, "volume", 0.0), 0.0),
                "openPrice": _to_float(getattr(order, "openPrice", 0.0), 0.0),
                "closePrice": _to_float(getattr(order, "closePrice", 0.0), 0.0),
                "commission": _to_float(getattr(order, "commission", 0.0), 0.0),
                "swap": _to_float(getattr(order, "swap", 0.0), 0.0),
                "profit": round(float(profit), 2),
                "openTime": open_time.isoformat() if open_time else None,
                "closeTime": close_time.isoformat() if close_time else None,
            }
        )

    return {
        "status_code": 200,
        "data": rows,
        "summary": {
            "date": target_date.isoformat(),
            "totalTrades": int(len(rows)),
            "netProfit": round(float(total_profit), 2),
            "wins": int(wins),
            "losses": int(losses),
        },
    }


@trading_router.get("/journal_feed", tags=["trading"])
async def get_trading_journal_feed(
    request: Request,
    q: str = "",
    limit: int = 200,
):
    if not request.state.user_id:
        raise HTTPException(status_code=400, detail="User ID is required")

    safe_limit = max(1, min(int(limit), 1000))
    account_ids = await _get_user_account_ids(request.state.user_id)
    if not account_ids:
        return {
            "status_code": 200,
            "data": [],
            "summary": {
                "totalRows": 0,
                "withJournal": 0,
                "withoutJournal": 0,
            },
        }

    fetch_size = max(safe_limit * 3, 400)
    orders = await db.orderhistory.find_many(
        where={
            "accountId": {"in": account_ids},
            "closeTime": {"not": None},
        },
        order={"closeTime": "desc"},
        take=fetch_size,
    )
    if not orders:
        return {
            "status_code": 200,
            "data": [],
            "summary": {
                "totalRows": 0,
                "withJournal": 0,
                "withoutJournal": 0,
            },
        }

    ticket_ids = [int(getattr(o, "ticketId", 0) or 0) for o in orders if int(getattr(o, "ticketId", 0) or 0) > 0]
    journals = await db.tradingjournal.find_many(
        where={"ticketId": {"in": ticket_ids}},
    ) if ticket_ids else []
    journal_by_ticket = {
        int(getattr(j, "ticketId", 0) or 0): j
        for j in journals
        if int(getattr(j, "ticketId", 0) or 0) > 0
    }

    query = str(q or "").strip().lower()
    rows = []
    with_journal = 0

    for order in orders:
        ticket_id = int(getattr(order, "ticketId", 0) or 0)
        journal = journal_by_ticket.get(ticket_id)

        tags = _normalize_string_list(getattr(journal, "tags", [])) if journal else []
        attachment_urls = _normalize_string_list(getattr(journal, "attachmentUrls", [])) if journal else []
        trade_rationale = str(getattr(journal, "tradeRationale", "") or "").strip() if journal else ""
        mistake_lesson = str(getattr(journal, "mistakeLesson", "") or "").strip() if journal else ""

        row = {
            "journalId": str(getattr(journal, "id", "") or "") if journal else None,
            "ticketId": int(ticket_id),
            "accountId": str(getattr(order, "accountId", "") or ""),
            "symbol": str(getattr(order, "symbol", "") or ""),
            "type": str(getattr(order, "type", "") or "").upper(),
            "status": str(getattr(order, "status", "") or ""),
            "volume": _to_float(getattr(order, "volume", 0.0), 0.0),
            "openPrice": _to_float(getattr(order, "openPrice", 0.0), 0.0),
            "closePrice": _to_float(getattr(order, "closePrice", 0.0), 0.0),
            "profit": round(_to_float(getattr(order, "profit", 0.0), 0.0), 2),
            "commission": _to_float(getattr(order, "commission", 0.0), 0.0),
            "swap": _to_float(getattr(order, "swap", 0.0), 0.0),
            "openTime": getattr(order, "openTime", None).isoformat() if getattr(order, "openTime", None) else None,
            "closeTime": getattr(order, "closeTime", None).isoformat() if getattr(order, "closeTime", None) else None,
            "tradeRationale": trade_rationale or None,
            "mistakeLesson": mistake_lesson or None,
            "tags": tags,
            "attachmentUrls": attachment_urls,
            "journalCreatedAt": getattr(journal, "createdAt", None).isoformat() if journal and getattr(journal, "createdAt", None) else None,
            "journalUpdatedAt": getattr(journal, "updatedAt", None).isoformat() if journal and getattr(journal, "updatedAt", None) else None,
        }

        if query:
            haystack = " ".join([
                str(row["ticketId"]),
                str(row["symbol"]),
                str(row["type"]),
                str(row["tradeRationale"] or ""),
                str(row["mistakeLesson"] or ""),
                " ".join(tags),
            ]).lower()
            if query not in haystack:
                continue

        if journal:
            with_journal += 1
        rows.append(row)
        if len(rows) >= safe_limit:
            break

    return {
        "status_code": 200,
        "data": rows,
        "summary": {
            "totalRows": int(len(rows)),
            "withJournal": int(with_journal),
            "withoutJournal": int(len(rows) - with_journal),
        },
    }


@trading_router.post("/journal/upsert", tags=["trading"])
async def upsert_trading_journal(request: Request, data: UpsertTradingJournalRequest):
    if not request.state.user_id:
        raise HTTPException(status_code=400, detail="User ID is required")

    ticket_id = int(data.ticketId or 0)
    if ticket_id <= 0:
        raise HTTPException(status_code=400, detail="ticketId is required")

    account_ids = await _get_user_account_ids(request.state.user_id)
    if not account_ids:
        raise HTTPException(status_code=404, detail="Trading account not found")

    owned_order = await db.orderhistory.find_first(
        where={
            "ticketId": ticket_id,
            "accountId": {"in": account_ids},
        },
    )
    if not owned_order:
        raise HTTPException(status_code=404, detail="Order not found for this user")

    trade_rationale = str(data.tradeRationale or "").strip() or None
    mistake_lesson = str(data.mistakeLesson or "").strip() or None
    tags = _normalize_string_list(data.tags)
    attachment_urls = _normalize_string_list(data.attachmentUrls)

    existing = await db.tradingjournal.find_unique(
        where={"ticketId": ticket_id},
    )
    payload = {
        "tradeRationale": trade_rationale,
        "mistakeLesson": mistake_lesson,
        "tags": Json(tags),
        "attachmentUrls": Json(attachment_urls),
    }
    if existing:
        journal = await db.tradingjournal.update(
            where={"id": existing.id},
            data=payload,
        )
    else:
        payload["ticketId"] = ticket_id
        journal = await db.tradingjournal.create(
            data=payload,
        )

    return {
        "status_code": 200,
        "data": {
            "id": str(journal.id),
            "ticketId": int(getattr(journal, "ticketId", 0) or 0),
            "tradeRationale": str(getattr(journal, "tradeRationale", "") or ""),
            "mistakeLesson": str(getattr(journal, "mistakeLesson", "") or ""),
            "tags": _normalize_string_list(getattr(journal, "tags", [])),
            "attachmentUrls": _normalize_string_list(getattr(journal, "attachmentUrls", [])),
            "createdAt": getattr(journal, "createdAt", None).isoformat() if getattr(journal, "createdAt", None) else None,
            "updatedAt": getattr(journal, "updatedAt", None).isoformat() if getattr(journal, "updatedAt", None) else None,
        },
    }


@trading_router.delete("/journal/{journal_id}", tags=["trading"])
async def delete_trading_journal(request: Request, journal_id: str):
    if not request.state.user_id:
        raise HTTPException(status_code=400, detail="User ID is required")

    journal = await db.tradingjournal.find_unique(
        where={"id": journal_id},
    )
    if not journal:
        raise HTTPException(status_code=404, detail="Journal entry not found")

    ticket_id = int(getattr(journal, "ticketId", 0) or 0)
    if ticket_id <= 0:
        raise HTTPException(status_code=403, detail="Journal entry cannot be verified")

    account_ids = await _get_user_account_ids(request.state.user_id)
    if not account_ids:
        raise HTTPException(status_code=404, detail="Trading account not found")

    owned_order = await db.orderhistory.find_first(
        where={
            "ticketId": ticket_id,
            "accountId": {"in": account_ids},
        },
    )
    if not owned_order:
        raise HTTPException(status_code=403, detail="Not allowed to delete this journal entry")

    await db.tradingjournal.delete(
        where={"id": journal_id},
    )
    return {
        "status_code": 200,
        "message": "Journal entry deleted",
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
                    "trading_schedule": normalize_trading_schedule(config.tradingSchedule),
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
