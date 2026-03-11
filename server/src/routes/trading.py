import asyncio

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel
from cryptography.fernet import Fernet
from prisma import Json
import base64
import hashlib
import os
from datetime import date, datetime, timedelta
from ..models.trading_model import (
    Create_Trading_Account,
    Delete_Trading_Account,
    Update_Trading_Account,
    UpsertTradingJournalRequest,
)
from ..database.client import db
from ..constants.mt5_server_catalog import (
    get_all_mt5_servers,
    get_mt5_broker_server_catalog,
    validate_mt5_broker_server_pair,
)
from ..utils.mt5_bot_runner import BotRunnerError, build_profile_name, run_bot_instance_action
from ..utils.notification_delivery import (
    build_generic_notification_email_html,
    claim_notification_dedupe,
    dispatch_notification_to_user_id,
)
from ..utils.subscription_access import assert_user_subscription_allows_bot_usage
from ..utils.trading_schedule import normalize_trading_schedule

trading_router = APIRouter()
ACTIVE_RECORD_STATUS = "active"
_TRADING_NOTIFICATION_RELATED_LINK = "/dashboard"


def _as_date(value) -> date:
    if isinstance(value, datetime):
        return value.date()
    return value


def _to_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _normalize_order_type_display(value) -> str:
    raw = value.value if hasattr(value, "value") else value
    text = str(raw or "").strip().lower()
    if text.endswith(".buy"):
        text = "buy"
    elif text.endswith(".sell"):
        text = "sell"
    if text in {"buy", "0"}:
        return "BUY"
    if text in {"sell", "1"}:
        return "SELL"
    return ""


def _to_optional_price(value) -> float | None:
    if value is None:
        return None
    parsed = _to_float(value, 0.0)
    if parsed <= 0.0:
        return None
    return float(parsed)


def _order_net_profit(order) -> float:
    profit = _to_float(getattr(order, "profit", 0.0), 0.0)
    commission = _to_float(getattr(order, "commission", 0.0), 0.0)
    swap = _to_float(getattr(order, "swap", 0.0), 0.0)
    return profit + commission + swap


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


def _encrypt_mt5_password(raw_password: str) -> str:
    key = base64.urlsafe_b64encode(
        hashlib.sha256(os.getenv("SECRET_KEY", "UknownmeInLove").encode()).digest()
    )
    fernet = Fernet(key)
    return fernet.encrypt(raw_password.encode()).decode()


def _format_trading_account_label(account) -> str:
    broker_name = str(getattr(account, "brokerName", "") or "").strip()
    server_name = str(getattr(account, "serverName", "") or "").strip()
    login_id = str(getattr(account, "mt5LoginId", "") or "").strip()
    label_parts = [part for part in [broker_name, server_name] if part]
    label = " / ".join(label_parts)
    if login_id:
        label = f"{label} ({login_id})" if label else f"MT5 {login_id}"
    return label or "MT5 account"


def _fallback_trading_account_label(account_id: str | None) -> str:
    safe_id = str(account_id or "").strip()
    return f"Account {safe_id[:8]}" if safe_id else "Unknown account"


def _serialize_trading_account_source(account) -> dict:
    account_id = str(getattr(account, "id", "") or "").strip()
    broker_name = str(getattr(account, "brokerName", "") or "").strip() or None
    server_name = str(getattr(account, "serverName", "") or "").strip() or None
    mt5_login_id = str(getattr(account, "mt5LoginId", "") or "").strip() or None
    return {
        "id": account_id,
        "label": _format_trading_account_label(account),
        "brokerName": broker_name,
        "serverName": server_name,
        "mt5LoginId": mt5_login_id,
    }


async def _notify_trading_account_event(
    *,
    user_id: str,
    event_type: str,
    account,
    linked_bots: int = 0,
    stopped_instances: int = 0,
) -> None:
    try:
        account_id = str(getattr(account, "id", "") or "").strip()
        if not user_id or not account_id:
            return

        account_label = _format_trading_account_label(account)
        title = "Trading account updated"
        subject = "Trading account updated - SmarfRobotTrade"
        message = f"{account_label} was updated."
        email_message = f"{account_label} was updated in your SmarfRobotTrade profile."

        if event_type == "created":
            title = "Trading account connected"
            subject = "Trading account connected - SmarfRobotTrade"
            message = f"{account_label} was added to your profile."
            email_message = (
                f"{account_label} was connected to your SmarfRobotTrade profile. "
                "You can now configure bots for this account."
            )
        elif event_type == "deleted":
            title = "Trading account archived"
            subject = "Trading account archived - SmarfRobotTrade"
            message = f"{account_label} was archived."
            email_message = f"{account_label} was archived from your SmarfRobotTrade profile."

        if linked_bots > 0:
            bot_note = f" {linked_bots} linked bot(s) were affected."
            message = f"{message}{bot_note}"
            email_message = f"{email_message}{bot_note}"
        if stopped_instances > 0:
            stop_note = f" {stopped_instances} running bot instance(s) were stopped."
            message = f"{message}{stop_note}"
            email_message = f"{email_message}{stop_note}"

        dedupe_basis = f"{event_type}|{account_id}|{getattr(account, 'brokerName', '')}|{getattr(account, 'serverName', '')}|{getattr(account, 'mt5LoginId', '')}|{linked_bots}|{stopped_instances}"
        if not claim_notification_dedupe(
            key=f"trading_account_notify:{user_id}:{dedupe_basis}",
            ttl_seconds=90,
        ):
            return

        email_html = build_generic_notification_email_html(
            title=title,
            greeting="Hi Trader,",
            message=email_message,
            related_link=_TRADING_NOTIFICATION_RELATED_LINK,
            action_label="Open dashboard",
        )

        await dispatch_notification_to_user_id(
            user_id,
            title=title,
            message=message,
            related_link=_TRADING_NOTIFICATION_RELATED_LINK,
            email_subject=subject,
            email_html=email_html,
            action_label="Open dashboard",
            send_discord_channel=False,
        )
    except Exception:
        return


async def _stop_account_bot_instances(account_id: str) -> int:
    bot_configs = await db.botconfiguration.find_many(
        where={"accountId": account_id, "recordStatus": ACTIVE_RECORD_STATUS},
    )
    if not bot_configs:
        return 0

    stopped_count = 0
    for config in bot_configs:
        try:
            await asyncio.to_thread(
                run_bot_instance_action,
                action="stop",
                instance_name=str(config.id),
                timeout_sec=300,
            )
            stopped_count += 1
        except BotRunnerError as exc:
            print(f"[WARN] failed to stop bot instance {config.id}: {exc}")

    return stopped_count


async def _get_user_account_ids(user_id: str, *, include_archived: bool = False) -> list[str]:
    where: dict = {"userId": user_id}
    if not include_archived:
        where["recordStatus"] = ACTIVE_RECORD_STATUS
    accounts = await db.tradingaccount.find_many(where=where)
    return [str(a.id) for a in accounts]


async def _build_user_order_scope(user_id: str, *, include_archived: bool) -> tuple[list[str], dict]:
    account_ids = await _get_user_account_ids(user_id, include_archived=include_archived)
    if not account_ids:
        return [], {}

    if include_archived:
        return account_ids, {"accountId": {"in": account_ids}}

    active_configs = await db.botconfiguration.find_many(
        where={
            "recordStatus": ACTIVE_RECORD_STATUS,
            "account": {
                "userId": user_id,
                "recordStatus": ACTIVE_RECORD_STATUS,
            },
        },
    )

    pair_set: set[tuple[str, int]] = set()
    pair_filters: list[dict] = []
    for cfg in active_configs:
        account_id = str(getattr(cfg, "accountId", "") or "").strip()
        bot_instance_id = int(getattr(cfg, "botInstanceId", 0) or 0)
        if not account_id or bot_instance_id <= 0:
            continue
        pair_key = (account_id, bot_instance_id)
        if pair_key in pair_set:
            continue
        pair_set.add(pair_key)
        pair_filters.append(
            {
                "accountId": account_id,
                "botInstanceId": bot_instance_id,
            }
        )

    manual_filter = {
        "accountId": {"in": account_ids},
        "botInstanceId": None,
    }
    if not pair_filters:
        return account_ids, manual_filter

    return account_ids, {"OR": [manual_filter, *pair_filters]}


async def _resolve_requested_account_id(
    user_id: str,
    requested_account_id: str | None,
    *,
    include_archived: bool,
) -> str | None:
    account_id = str(requested_account_id or "").strip()
    if not account_id:
        return None

    account_where: dict = {
        "id": account_id,
        "userId": user_id,
    }
    if not include_archived:
        account_where["recordStatus"] = ACTIVE_RECORD_STATUS

    account = await db.tradingaccount.find_first(where=account_where)
    if not account:
        raise HTTPException(status_code=404, detail="Trading account not found")

    return str(account.id)


async def _resolve_requested_bot_configuration(
    user_id: str,
    requested_bot_config_id: str | None,
    *,
    include_archived: bool,
    requested_account_id: str | None = None,
):
    bot_config_id = str(requested_bot_config_id or "").strip()
    if not bot_config_id:
        return None

    account_where: dict = {"userId": user_id}
    if not include_archived:
        account_where["recordStatus"] = ACTIVE_RECORD_STATUS

    config_where: dict = {
        "id": bot_config_id,
        "account": account_where,
    }
    if not include_archived:
        config_where["recordStatus"] = ACTIVE_RECORD_STATUS
    if requested_account_id:
        config_where["accountId"] = requested_account_id

    config = await db.botconfiguration.find_first(where=config_where)
    if not config:
        raise HTTPException(status_code=404, detail="Bot configuration not found")

    return config


def _combine_where_and(*conditions: dict) -> dict:
    valid_conditions = [c for c in conditions if c]
    if not valid_conditions:
        return {}
    if len(valid_conditions) == 1:
        return valid_conditions[0]
    return {"AND": valid_conditions}


def _build_bot_order_filter(bot_config) -> dict:
    if not bot_config:
        return {}

    account_id = str(getattr(bot_config, "accountId", "") or "").strip()
    bot_instance_id = int(getattr(bot_config, "botInstanceId", 0) or 0)
    magic_number = int(getattr(bot_config, "magicNumber", 0) or 0)

    filters: list[dict] = []
    if account_id and bot_instance_id > 0:
        filters.append(
            {
                "accountId": account_id,
                "botInstanceId": bot_instance_id,
            }
        )
    if account_id and magic_number > 0:
        filters.append(
            {
                "accountId": account_id,
                "magicNumber": magic_number,
            }
        )

    if not filters:
        return {"accountId": account_id, "ticketId": -1} if account_id else {"ticketId": -1}
    if len(filters) == 1:
        return filters[0]
    return {"OR": filters}


async def _sync_daily_aggregates_from_orders(
    user_id: str,
    start_date: date,
    end_date: date,
    include_archived_accounts: bool = True,
):
    start_dt = datetime.combine(start_date, datetime.min.time())
    end_dt = datetime.combine(end_date, datetime.min.time())

    account_where: dict = {"userId": user_id}
    if not include_archived_accounts:
        account_where["recordStatus"] = ACTIVE_RECORD_STATUS
    accounts = await db.tradingaccount.find_many(where=account_where)
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
        pnl = _order_net_profit(order)

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
async def get_trading_calendar(
    request: Request,
    year: int,
    month: int,
    include_archived: bool = Query(True, alias="includeArchived"),
    account_id: str | None = Query(None, alias="accountId"),
    bot_config_id: str | None = Query(None, alias="botConfigId"),
):
    if not request.state.user_id:
        raise HTTPException(status_code=400, detail="User ID is required")
    selected_account_id = await _resolve_requested_account_id(
        request.state.user_id,
        account_id,
        include_archived=include_archived,
    )
    selected_bot_config = await _resolve_requested_bot_configuration(
        request.state.user_id,
        bot_config_id,
        include_archived=include_archived,
        requested_account_id=selected_account_id,
    )
    if selected_bot_config and not selected_account_id:
        selected_account_id = str(getattr(selected_bot_config, "accountId", "") or "").strip() or None
    
    # Calculate start and end date for the month
    try:
        start_date = date(year, month, 1)
        if month == 12:
            end_date = date(year + 1, 1, 1)
        else:
            end_date = date(year, month + 1, 1)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid year or month")
    
    day_rollup = {}
    if not selected_bot_config:
        # Pull monthly trade history into daily_aggregates so calendar stays synced.
        try:
            day_rollup = await _sync_daily_aggregates_from_orders(
                user_id=request.state.user_id,
                start_date=start_date,
                end_date=end_date,
                include_archived_accounts=True,
            )
        except Exception as e:
            print(f"[WARN] trading calendar sync failed: {e}")
            day_rollup = {}

    calendar_data = {}
    if include_archived and not selected_bot_config:
        account_filter: dict = {"userId": request.state.user_id}
        aggregate_where: dict = {
            "account": account_filter,
            "date": {
                "gte": datetime.combine(start_date, datetime.min.time()),
                "lt": datetime.combine(end_date, datetime.min.time())
            },
        }
        if selected_account_id:
            aggregate_where["accountId"] = selected_account_id

        aggregates = await db.dailyaggregate.find_many(
            where=aggregate_where
        )

        for agg in aggregates:
            day = agg.date.day
            if day not in calendar_data:
                calendar_data[day] = {
                    "date": day,
                    "profit": 0.0,
                    "trades": 0,
                    "winRate": 0.0,
                }

            profit = _to_float(agg.dailyNetProfit, 0.0)
            trades = agg.totalTrades if agg.totalTrades else 0
            calendar_data[day]["profit"] += profit
            calendar_data[day]["trades"] += trades

        if selected_account_id:
            day_rollup = {}
            winrate_orders = await db.orderhistory.find_many(
                where={
                    "accountId": selected_account_id,
                    "closeTime": {
                        "gte": datetime.combine(start_date, datetime.min.time()),
                        "lt": datetime.combine(end_date, datetime.min.time()),
                    },
                },
            )
            for order in winrate_orders:
                close_time = getattr(order, "closeTime", None)
                if close_time is None:
                    continue
                trade_day = _as_date(close_time)
                if trade_day < start_date or trade_day >= end_date:
                    continue

                day = int(trade_day.day)
                profit = _order_net_profit(order)
                day_item = day_rollup.setdefault(day, {"wins": 0, "trades": 0})
                day_item["trades"] += 1
                if profit > 0:
                    day_item["wins"] += 1
    else:
        day_rollup = {}
        _, order_scope = await _build_user_order_scope(
            request.state.user_id,
            include_archived=include_archived,
        )
        if order_scope:
            where_scope = _combine_where_and(
                order_scope,
                {"accountId": selected_account_id} if selected_account_id else {},
                {
                    "closeTime": {
                        "gte": datetime.combine(start_date, datetime.min.time()),
                        "lt": datetime.combine(end_date, datetime.min.time()),
                    },
                },
                _build_bot_order_filter(selected_bot_config),
            )
            orders = await db.orderhistory.find_many(
                where=where_scope,
            )

            for order in orders:
                close_time = getattr(order, "closeTime", None)
                if close_time is None:
                    continue
                trade_day = _as_date(close_time)
                if trade_day < start_date or trade_day >= end_date:
                    continue

                day = int(trade_day.day)
                if day not in calendar_data:
                    calendar_data[day] = {
                        "date": day,
                        "profit": 0.0,
                        "trades": 0,
                        "winRate": 0.0,
                    }

                profit = _order_net_profit(order)
                calendar_data[day]["profit"] += profit
                calendar_data[day]["trades"] += 1

                day_item = day_rollup.setdefault(day, {"wins": 0, "trades": 0})
                day_item["trades"] += 1
                if profit > 0:
                    day_item["wins"] += 1

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
async def get_trading_history_by_day(
    request: Request,
    year: int,
    month: int,
    day: int,
    include_archived: bool = Query(True, alias="includeArchived"),
    account_id: str | None = Query(None, alias="accountId"),
):
    if not request.state.user_id:
        raise HTTPException(status_code=400, detail="User ID is required")
    selected_account_id = await _resolve_requested_account_id(
        request.state.user_id,
        account_id,
        include_archived=include_archived,
    )

    try:
        target_date = date(year, month, day)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date")

    start_dt = datetime.combine(target_date, datetime.min.time())
    end_dt = start_dt + timedelta(days=1)

    account_ids, order_scope = await _build_user_order_scope(
        request.state.user_id,
        include_archived=include_archived,
    )
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

    where_scope = _combine_where_and(
        order_scope,
        {"accountId": selected_account_id} if selected_account_id else {},
        {"closeTime": {"gte": start_dt, "lt": end_dt}},
    )
    orders = await db.orderhistory.find_many(
        where=where_scope,
        order={"closeTime": "desc"},
    )

    rows = []
    total_profit = 0.0
    wins = 0
    losses = 0

    for order in orders:
        gross_profit = _to_float(getattr(order, "profit", 0.0), 0.0)
        commission = _to_float(getattr(order, "commission", 0.0), 0.0)
        swap = _to_float(getattr(order, "swap", 0.0), 0.0)
        net_profit = gross_profit + commission + swap
        total_profit += net_profit
        if net_profit > 0:
            wins += 1
        elif net_profit < 0:
            losses += 1

        open_time = getattr(order, "openTime", None)
        close_time = getattr(order, "closeTime", None)
        order_type = _normalize_order_type_display(getattr(order, "type", ""))
        open_price = _to_optional_price(getattr(order, "openPrice", None))
        close_price = _to_optional_price(getattr(order, "closePrice", None))
        rows.append(
            {
                "ticketId": int(getattr(order, "ticketId", 0) or 0),
                "accountId": str(getattr(order, "accountId", "") or ""),
                "magicNumber": int(getattr(order, "magicNumber", 0) or 0) or None,
                "symbol": str(getattr(order, "symbol", "") or ""),
                "type": order_type,
                "status": str(getattr(order, "status", "") or ""),
                "volume": _to_float(getattr(order, "volume", 0.0), 0.0),
                "openPrice": open_price,
                "closePrice": close_price,
                "commission": commission,
                "swap": swap,
                "profit": round(float(gross_profit), 2),
                "netProfit": round(float(net_profit), 2),
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
    include_archived: bool = Query(True, alias="includeArchived"),
):
    if not request.state.user_id:
        raise HTTPException(status_code=400, detail="User ID is required")

    safe_limit = max(1, min(int(limit), 1000))
    account_ids, order_scope = await _build_user_order_scope(
        request.state.user_id,
        include_archived=include_archived,
    )
    if not account_ids:
        return {
            "status_code": 200,
            "data": [],
            "accounts": [],
            "summary": {
                "totalRows": 0,
                "withJournal": 0,
                "withoutJournal": 0,
            },
        }

    trading_accounts = await db.tradingaccount.find_many(
        where={"id": {"in": account_ids}},
    )
    account_by_id = {
        str(getattr(account, "id", "") or "").strip(): account
        for account in trading_accounts
        if str(getattr(account, "id", "") or "").strip()
    }
    account_sources = [
        _serialize_trading_account_source(account)
        for account in trading_accounts
        if str(getattr(account, "id", "") or "").strip()
    ]

    fetch_size = max(safe_limit * 3, 400)
    where_scope = {
        **order_scope,
        "closeTime": {"not": None},
    }
    orders = await db.orderhistory.find_many(
        where=where_scope,
        order={"closeTime": "desc"},
        take=fetch_size,
    )
    if not orders:
        return {
            "status_code": 200,
            "data": [],
            "accounts": account_sources,
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
        account_id = str(getattr(order, "accountId", "") or "").strip()
        account = account_by_id.get(account_id)
        account_label = (
            _format_trading_account_label(account)
            if account
            else _fallback_trading_account_label(account_id)
        )
        account_broker_name = (
            str(getattr(account, "brokerName", "") or "").strip() or None
            if account
            else None
        )
        account_server_name = (
            str(getattr(account, "serverName", "") or "").strip() or None
            if account
            else None
        )
        account_login_id = (
            str(getattr(account, "mt5LoginId", "") or "").strip() or None
            if account
            else None
        )

        tags = _normalize_string_list(getattr(journal, "tags", [])) if journal else []
        attachment_urls = _normalize_string_list(getattr(journal, "attachmentUrls", [])) if journal else []
        trade_rationale = str(getattr(journal, "tradeRationale", "") or "").strip() if journal else ""
        mistake_lesson = str(getattr(journal, "mistakeLesson", "") or "").strip() if journal else ""

        order_type = _normalize_order_type_display(getattr(order, "type", ""))
        open_price = _to_optional_price(getattr(order, "openPrice", None))
        close_price = _to_optional_price(getattr(order, "closePrice", None))
        row = {
            "journalId": str(getattr(journal, "id", "") or "") if journal else None,
            "ticketId": int(ticket_id),
            "accountId": account_id,
            "accountLabel": account_label,
            "accountBrokerName": account_broker_name,
            "accountServerName": account_server_name,
            "accountLoginId": account_login_id,
            "magicNumber": int(getattr(order, "magicNumber", 0) or 0) or None,
            "symbol": str(getattr(order, "symbol", "") or ""),
            "type": order_type,
            "status": str(getattr(order, "status", "") or ""),
            "volume": _to_float(getattr(order, "volume", 0.0), 0.0),
            "openPrice": open_price,
            "closePrice": close_price,
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
                str(row["accountId"]),
                str(row["accountLabel"]),
                str(row["accountBrokerName"] or ""),
                str(row["accountServerName"] or ""),
                str(row["accountLoginId"] or ""),
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
        "accounts": account_sources,
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

    account_ids = await _get_user_account_ids(
        request.state.user_id,
        include_archived=True,
    )
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

    account_ids = await _get_user_account_ids(
        request.state.user_id,
        include_archived=True,
    )
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
async def get_accounts_with_bots(
    request: Request,
    include_archived: bool = Query(False, alias="includeArchived"),
):
    if not request.state.user_id:
        raise HTTPException(status_code=400, detail="User ID is required")
    
    try:
        account_where: dict = {"userId": request.state.user_id}
        if not include_archived:
            account_where["recordStatus"] = ACTIVE_RECORD_STATUS

        bot_include: dict = {"include": {"botVersion": True}}
        if not include_archived:
            bot_include["where"] = {"recordStatus": ACTIVE_RECORD_STATUS}

        trading_accounts = await db.tradingaccount.find_many(
            where=account_where,
            include={
                "botConfigurations": bot_include,
                "dailyAggregates": True
            }
        )

        account_ids = [str(a.id) for a in trading_accounts]
        today_local = date.today()
        today_str = today_local.isoformat()
        total_net_by_account: dict[str, float] = {aid: 0.0 for aid in account_ids}
        today_net_by_account: dict[str, float] = {aid: 0.0 for aid in account_ids}
        if account_ids:
            metric_orders = await db.orderhistory.find_many(
                where={
                    "accountId": {"in": account_ids},
                    "closeTime": {"not": None},
                },
            )
            for order in metric_orders:
                account_id = str(getattr(order, "accountId", "") or "").strip()
                if not account_id:
                    continue
                net_profit = _order_net_profit(order)
                total_net_by_account[account_id] = float(total_net_by_account.get(account_id, 0.0)) + net_profit

                close_time = getattr(order, "closeTime", None)
                if close_time is None:
                    continue
                trade_day = _as_date(close_time)
                if trade_day == today_local:
                    today_net_by_account[account_id] = float(today_net_by_account.get(account_id, 0.0)) + net_profit

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
                    runner_profile = None
                    try:
                        if bv.symbol and bv.timeframe:
                            runner_profile = build_profile_name(bv.symbol, bv.timeframe)
                    except Exception:
                        runner_profile = None
                    bot_version = {
                        "model_id": str(bv.modelId),
                        "label": bv.label,
                        "docker_image_id": bv.dockerImageId,
                        "version_tag": bv.versionTag,
                        "symbol": bv.symbol,
                        "timeframe": bv.timeframe,
                        "is_active": bool(getattr(bv, "isActive", True)),
                        "runner_profile": runner_profile,
                        "release_notes": bv.releaseNotes,
                    }

                installed_image_id = getattr(config, "installedDockerImageId", None)
                installed_version_tag = getattr(config, "installedVersionTag", None)
                effective_installed_image_id = installed_image_id or latest_image_id
                effective_installed_version_tag = installed_version_tag or latest_version_tag
                has_pending_update = bool(
                    latest_version_tag
                    and effective_installed_version_tag
                    and latest_version_tag != effective_installed_version_tag
                )

                bot_configs.append({
                    "id": str(config.id),
                    "account_id": str(config.accountId),
                    "model_id": str(config.modelId),
                    "bot_instance_id": config.botInstanceId,
                    "magic_number": int(getattr(config, "magicNumber", 0) or 0) or None,
                    "risk_level": config.riskLevel if config.riskLevel else None,
                    "trading_schedule": normalize_trading_schedule(config.tradingSchedule),
                    "is_active": config.isActive,
                    "docker_container_id": config.dockerContainerId,
                    "installed_docker_image_id": effective_installed_image_id,
                    "installed_version_tag": effective_installed_version_tag,
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
            account_id = str(account.id)
            total_today_pnl = float(today_net_by_account.get(account_id, 0.0))
            if abs(total_today_pnl) <= 1e-9 and today_agg and today_agg.dailyNetProfit is not None:
                total_today_pnl = float(today_agg.dailyNetProfit)
            total_net_pnl = float(total_net_by_account.get(account_id, 0.0))

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
                "total_net_pnl": total_net_pnl,
            })

        return {
            "status_code": 200,
            "data": result
        }
    except Exception as e:
        print(f"[ERROR] get_accounts_with_bots: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@trading_router.get("/mt5_server_catalog", tags=["trading"])
async def get_mt5_server_catalog(request: Request):
    if not request.state.user_id:
        raise HTTPException(status_code=400, detail="User ID is required")

    return {
        "status_code": 200,
        "data": {
            "brokers": get_mt5_broker_server_catalog(),
            "all_servers": get_all_mt5_servers(),
        },
    }


@trading_router.post("/create_account", tags=["trading"])
async def create_account(request: Request, data: Create_Trading_Account):
    if not request.state.user_id:
        raise HTTPException(status_code=400, detail="User ID is required")
    await assert_user_subscription_allows_bot_usage(
        request.state.user_id,
        action_label="connect trading accounts",
    )

    broker_name = str(data.brokerName or "").strip()
    server_name = str(data.serverName or "").strip()
    mt5_login_id = str(data.mt5LoginId or "").strip()
    mt5_password = str(data.mt5Password or "").strip()

    if not broker_name:
        raise HTTPException(status_code=400, detail="brokerName cannot be empty")
    if not server_name:
        raise HTTPException(status_code=400, detail="serverName cannot be empty")
    if not mt5_login_id:
        raise HTTPException(status_code=400, detail="mt5LoginId cannot be empty")
    if not mt5_password:
        raise HTTPException(status_code=400, detail="mt5Password cannot be empty")

    pair_ok, pair_error = validate_mt5_broker_server_pair(broker_name, server_name)
    if not pair_ok:
        raise HTTPException(status_code=400, detail=pair_error or "Invalid brokerName/serverName")

    encrypted_password = _encrypt_mt5_password(mt5_password)
    userId = request.state.user_id

    trading_account = await db.tradingaccount.create(
        data={
            "userId": userId,
            "brokerName": broker_name,
            "serverName": server_name,
            "mt5LoginId": mt5_login_id,
            "mt5Password": encrypted_password,
            "recordStatus": ACTIVE_RECORD_STATUS,
            "deletedAt": None,
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

    await _notify_trading_account_event(
        user_id=str(userId),
        event_type="created",
        account=trading_account,
    )

    return {
        "status_code": 200,
        "message": "Trading account created successfully"
    }


@trading_router.patch("/update_account", tags=["trading"])
async def update_account(request: Request, data: Update_Trading_Account):
    if not request.state.user_id:
        raise HTTPException(status_code=400, detail="User ID is required")

    account = await db.tradingaccount.find_first(
        where={
            "id": data.accountId,
            "userId": request.state.user_id,
            "recordStatus": ACTIVE_RECORD_STATUS,
        }
    )
    if not account:
        raise HTTPException(status_code=404, detail="Trading account not found")

    update_payload = {}
    next_broker_name = str(getattr(account, "brokerName", "") or "").strip()
    next_server_name = str(getattr(account, "serverName", "") or "").strip()

    if data.brokerName is not None:
        broker_name = str(data.brokerName).strip()
        if not broker_name:
            raise HTTPException(status_code=400, detail="brokerName cannot be empty")
        update_payload["brokerName"] = broker_name
        next_broker_name = broker_name

    if data.serverName is not None:
        server_name = str(data.serverName).strip()
        if not server_name:
            raise HTTPException(status_code=400, detail="serverName cannot be empty")
        update_payload["serverName"] = server_name
        next_server_name = server_name

    if data.mt5LoginId is not None:
        mt5_login_id = str(data.mt5LoginId).strip()
        if not mt5_login_id:
            raise HTTPException(status_code=400, detail="mt5LoginId cannot be empty")
        update_payload["mt5LoginId"] = mt5_login_id

    if data.mt5Password is not None:
        mt5_password = str(data.mt5Password).strip()
        if not mt5_password:
            raise HTTPException(status_code=400, detail="mt5Password cannot be empty")
        update_payload["mt5Password"] = _encrypt_mt5_password(mt5_password)

    if data.brokerName is not None or data.serverName is not None:
        pair_ok, pair_error = validate_mt5_broker_server_pair(next_broker_name, next_server_name)
        if not pair_ok:
            raise HTTPException(status_code=400, detail=pair_error or "Invalid brokerName/serverName")

    if not update_payload:
        raise HTTPException(status_code=400, detail="No update fields provided")

    linked_bots = await db.botconfiguration.count(
        where={"accountId": data.accountId, "recordStatus": ACTIVE_RECORD_STATUS},
    )
    stopped_instances = 0
    if linked_bots > 0:
        stopped_instances = await _stop_account_bot_instances(data.accountId)

    await db.tradingaccount.update(
        where={"id": data.accountId},
        data=update_payload,
    )

    updated_account = await db.tradingaccount.find_unique(where={"id": data.accountId})

    if linked_bots > 0:
        await db.botconfiguration.update_many(
            where={"accountId": data.accountId, "recordStatus": ACTIVE_RECORD_STATUS},
            data={
                "containerStatus": "stopped",
                "isActive": False,
            },
        )

    if updated_account:
        await _notify_trading_account_event(
            user_id=str(request.state.user_id),
            event_type="updated",
            account=updated_account,
            linked_bots=int(linked_bots),
            stopped_instances=int(stopped_instances),
        )

    return {
        "status_code": 200,
        "message": "Trading account updated successfully",
        "affected_bots": int(linked_bots),
        "stopped_instances": int(stopped_instances),
    }


@trading_router.delete("/delete_account", tags=["trading"])
async def delete_account(request: Request, data: Delete_Trading_Account):
    if not request.state.user_id:
        raise HTTPException(status_code=400, detail="User ID is required")

    account = await db.tradingaccount.find_first(
        where={
            "id": data.accountId,
            "userId": request.state.user_id,
            "recordStatus": ACTIVE_RECORD_STATUS,
        }
    )
    if not account:
        raise HTTPException(status_code=404, detail="Trading account not found")

    linked_bots = await db.botconfiguration.count(
        where={"accountId": data.accountId, "recordStatus": ACTIVE_RECORD_STATUS},
    )
    stopped_instances = 0
    if linked_bots > 0:
        stopped_instances = await _stop_account_bot_instances(data.accountId)

    if linked_bots > 0:
        await db.botconfiguration.update_many(
            where={"accountId": data.accountId, "recordStatus": ACTIVE_RECORD_STATUS},
            data={
                "containerStatus": "stopped",
                "isActive": False,
                "dockerContainerId": None,
                "recordStatus": "deleted",
                "deletedAt": datetime.utcnow(),
            },
        )

    await db.tradingaccount.update(
        where={"id": data.accountId},
        data={
            "recordStatus": "deleted",
            "deletedAt": datetime.utcnow(),
        },
    )

    await _notify_trading_account_event(
        user_id=str(request.state.user_id),
        event_type="deleted",
        account=account,
        linked_bots=int(linked_bots),
        stopped_instances=int(stopped_instances),
    )

    return {
        "status_code": 200,
        "message": "Trading account archived successfully",
        "deleted_bots": int(linked_bots),
        "stopped_instances": int(stopped_instances),
    }


@trading_router.get("/", tags=["trading"])
async def trading_by_user(request: Request, accountId: str):
    if not request.state.user_id:
        raise HTTPException(status_code=400, detail="User ID is required")
    
    trading_account = await db.tradingaccount.find_first(
        where={
            "id": accountId,
            "userId": request.state.user_id,
            "recordStatus": ACTIVE_RECORD_STATUS,
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
            "userId": request.state.user_id,
            "recordStatus": ACTIVE_RECORD_STATUS,
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
            "userId": userId,
            "recordStatus": ACTIVE_RECORD_STATUS,
        }
    )
    
    if not trading_accounts:
        raise HTTPException(status_code=400, detail="Trading account not found")
    
    return {
        "status_code": 200,
        "message": trading_accounts
    }
