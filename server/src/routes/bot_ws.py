"""
Bot WebSocket routes — unified hub for bot ↔ server ↔ dashboard.

Endpoints:
- WS  /ws              → Bot connects, registers, pushes state, receives LLM
- WS  /ws/dashboard     → Dashboard connects, receives all bot states
- POST /ws/cron         → Cron: compute vision_llm + broadcast to matching bots
"""

import asyncio
import json
import logging
import math
import time
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

from ..models.vision_llm_model import VisionLLMRequest
from ..utils.vision_llm.chart import MT5ConnectionError, NoMarketDataError
from ..utils.vision_llm.llm_client import (
    VisionLLMConfigError,
    VisionLLMServiceUnavailableError,
)
from ..utils.trading_schedule import normalize_trading_schedule
from ..utils.vision_llm.use_llm import generate_llm_cls_for_bar
from ..utils.ws_manager import bot_hub
from ..database.client import db, r_cache

logger = logging.getLogger(__name__)

bot_ws_router = APIRouter()

_CACHE_TTL = 3900  # 65 minutes
_BOT_CONTEXT: dict[str, dict] = {}
_BOT_OPEN_POSITIONS: dict[str, dict[int, dict]] = {}
_BOT_ACCOUNT_SYNC_CACHE: dict[str, dict] = {}
_BOT_INSUFFICIENT_FUNDS_ALERT_CACHE: dict[str, dict[str, float]] = {}
_ACCOUNT_SYNC_MIN_INTERVAL_SEC = 5.0
_ACCOUNT_SYNC_FORCE_INTERVAL_SEC = 60.0
_INSUFFICIENT_FUNDS_ALERT_TTL_SEC = 60.0 * 60.0 * 6.0
_INSUFFICIENT_FUNDS_ALERT_CACHE_MAX = 300
_INSUFFICIENT_FUNDS_HINTS = (
    "no money",
    "not enough money",
    "insufficient funds",
    "insufficient margin",
    "margin",
    "funds",
)


def _cache_key(symbol: str, timeframe: str, dt_str: str) -> str:
    return f"vision_llm:{symbol}:{timeframe}:{dt_str}"


def _enum_value(value):
    return value.value if hasattr(value, "value") else value


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _trim_text(value) -> str:
    return str(value or "").strip()


def _contains_insufficient_funds_hint(text: str) -> bool:
    raw = _trim_text(text).lower()
    if not raw:
        return False
    return any(hint in raw for hint in _INSUFFICIENT_FUNDS_HINTS)


def _remember_insufficient_funds_alert(bot_config_id: str, dedup_key: str) -> bool:
    if not bot_config_id or not dedup_key:
        return False
    now_epoch = time.time()
    bucket = _BOT_INSUFFICIENT_FUNDS_ALERT_CACHE.setdefault(bot_config_id, {})
    expired = [
        key
        for key, ts in bucket.items()
        if (now_epoch - float(ts)) >= _INSUFFICIENT_FUNDS_ALERT_TTL_SEC
    ]
    for key in expired:
        bucket.pop(key, None)

    if dedup_key in bucket:
        return False

    bucket[dedup_key] = now_epoch
    if len(bucket) > _INSUFFICIENT_FUNDS_ALERT_CACHE_MAX:
        oldest_keys = sorted(bucket.items(), key=lambda item: float(item[1]))[: max(1, _INSUFFICIENT_FUNDS_ALERT_CACHE_MAX // 2)]
        for key, _ in oldest_keys:
            bucket.pop(key, None)
    return True


def _normalize_order_side(value) -> str | None:
    text = str(_enum_value(value) or "").strip().lower()
    if text in {"buy", "sell"}:
        return text
    if text in {"0", "buy market", "deal_type_buy"}:
        return "buy"
    if text in {"1", "sell market", "deal_type_sell"}:
        return "sell"
    return None


def _safe_finite_float(value, default: float | None = None) -> float | None:
    try:
        parsed = float(value)
    except Exception:
        return default
    if not math.isfinite(parsed):
        return default
    return parsed


def _parse_open_time(raw_pos: dict) -> datetime | None:
    ts = raw_pos.get("opened_at_ts", None)
    if ts is not None:
        try:
            ts_int = int(ts)
            if ts_int > 0:
                return datetime.fromtimestamp(ts_int, tz=timezone.utc)
        except Exception:
            pass

    opened_at = str(raw_pos.get("opened_at", "")).strip()
    if opened_at:
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
            try:
                return datetime.strptime(opened_at, fmt).replace(tzinfo=timezone.utc)
            except Exception:
                continue
    return None


async def _get_bot_context(bot_config_id: str) -> dict | None:
    cached = _BOT_CONTEXT.get(bot_config_id)
    if cached:
        return cached

    config = await db.botconfiguration.find_first(
        where={
            "id": bot_config_id,
            "recordStatus": "active",
            "account": {"recordStatus": "active"},
        },
        include={"account": True, "botVersion": True},
    )
    if not config:
        return None

    context = {
        "account_id": str(config.accountId),
        "bot_instance_id": _safe_int(getattr(config, "botInstanceId", 0), 0),
        "magic_number": _safe_int(getattr(config, "magicNumber", 0), 0),
        "owner_user_id": str(getattr(getattr(config, "account", None), "userId", "") or ""),
        "broker_name": str(getattr(getattr(config, "account", None), "brokerName", "") or ""),
        "server_name": str(getattr(getattr(config, "account", None), "serverName", "") or ""),
        "mt5_login_id": str(getattr(getattr(config, "account", None), "mt5LoginId", "") or ""),
        "bot_label": str(getattr(getattr(config, "botVersion", None), "label", "") or ""),
    }
    _BOT_CONTEXT[bot_config_id] = context
    return context


async def _emit_insufficient_funds_notification(bot_config_id: str, state: dict) -> None:
    recent_logs = state.get("recent_logs")
    if not isinstance(recent_logs, list) or len(recent_logs) == 0:
        return

    context = await _get_bot_context(bot_config_id)
    if not isinstance(context, dict):
        return
    user_id = _trim_text(context.get("owner_user_id"))
    if not user_id:
        return

    bot_label = _trim_text(context.get("bot_label")) or "Trading Bot"
    symbol = _trim_text(state.get("symbol")).upper() or "-"
    broker_name = _trim_text(context.get("broker_name"))
    server_name = _trim_text(context.get("server_name"))
    mt5_login_id = _trim_text(context.get("mt5_login_id"))
    account_label_parts = [part for part in [broker_name, server_name] if part]
    account_label = " / ".join(account_label_parts)
    if mt5_login_id:
        account_label = f"{account_label} ({mt5_login_id})" if account_label else f"MT5 {mt5_login_id}"

    for entry in recent_logs:
        if not isinstance(entry, dict):
            continue
        phase = _trim_text(entry.get("phase")).upper()
        event = _trim_text(entry.get("event")).lower()
        message = _trim_text(entry.get("message"))
        meta = entry.get("meta") if isinstance(entry.get("meta"), dict) else {}
        reason = _trim_text(meta.get("reason")) or message
        side = _trim_text(meta.get("side")).upper() or "ORDER"

        is_alert = event in {"open_blocked_insufficient_funds", "open_failed_insufficient_funds"}
        if not is_alert and phase == "ORDER" and event == "open_failed":
            is_alert = _contains_insufficient_funds_hint(f"{message} {reason}")
        if not is_alert:
            continue

        timestamp_key = _trim_text(entry.get("timestamp_utc")) or _trim_text(entry.get("timestamp"))
        dedup_key = f"{timestamp_key}|{event}|{side}|{reason[:200]}"
        if not _remember_insufficient_funds_alert(bot_config_id, dedup_key):
            continue

        reason_text = reason or "Insufficient funds or margin"
        title = "Order blocked: insufficient funds"
        body = f"{bot_label} cannot open {side} on {symbol}: {reason_text}"
        if account_label:
            body = f"{body} | Account: {account_label}"
        await db.notification.create(
            data={
                "userId": user_id,
                "title": title[:100],
                "message": body,
                "relatedLink": "/bot-control",
            }
        )


def _resolve_allowed_magic_set(context: dict | None = None) -> set[int]:
    out = {0, 12345, 123456}
    if isinstance(context, dict):
        magic_number = _safe_int(context.get("magic_number"), 0)
        if magic_number > 0:
            out.add(magic_number)
    return out


def _extract_live_positions(state: dict, allowed_magic_set: set[int] | None = None) -> dict[int, dict]:
    raw_positions = state.get("positions")
    if not isinstance(raw_positions, list):
        return {}

    positions: dict[int, dict] = {}
    for item in raw_positions:
        if not isinstance(item, dict):
            continue
        ticket = _safe_int(item.get("ticket"), 0)
        if ticket <= 0:
            continue
        magic = _safe_int(item.get("magic"), 0)
        if isinstance(allowed_magic_set, set) and len(allowed_magic_set) > 0:
            if magic not in allowed_magic_set:
                continue
        side = _normalize_order_side(item.get("type"))
        pos_payload = {
            "ticket": ticket,
            "symbol": str(item.get("symbol") or state.get("symbol") or "").strip().upper() or None,
            "type": side,
            "magic": magic,
            "volume": _safe_float(item.get("volume", 0.0)),
            "openPrice": _safe_float(item.get("price_open", 0.0)),
            "closePrice": _safe_float(item.get("price_current", 0.0)),
            "profit": _safe_float(item.get("profit", 0.0)),
            "swap": _safe_float(item.get("swap", 0.0)),
            "commission": _safe_float(item.get("commission", 0.0)),
            "openTime": _parse_open_time(item),
        }
        positions[ticket] = pos_payload
    return positions


async def _persist_closed_orders_only(
    bot_config_id: str,
    account_id: str,
    bot_instance_id: int,
    state: dict,
    *,
    allowed_magic_set: set[int] | None = None,
    skip_ticket_ids: set[int] | None = None,
) -> None:
    current_positions = _extract_live_positions(state, allowed_magic_set=allowed_magic_set)
    previous_positions = _BOT_OPEN_POSITIONS.get(bot_config_id, {})
    closed_ticket_ids = [
        int(ticket) for ticket in previous_positions.keys()
        if int(ticket) not in current_positions
    ]

    existing = await db.orderhistory.find_many(
        where={"ticketId": {"in": closed_ticket_ids}},
    )
    existing_by_ticket = {
        _safe_int(row.ticketId, 0): row
        for row in existing
    }
    if skip_ticket_ids:
        filtered_ticket_ids: list[int] = []
        for ticket in closed_ticket_ids:
            if int(ticket) not in skip_ticket_ids:
                filtered_ticket_ids.append(int(ticket))
                continue
            existing_row = existing_by_ticket.get(int(ticket))
            # Keep tickets synced by closed_deals only when row already has complete open-side data.
            if existing_row is None:
                filtered_ticket_ids.append(int(ticket))
                continue
            existing_type = _normalize_order_side(getattr(existing_row, "type", None))
            existing_open_price = _safe_float(getattr(existing_row, "openPrice", 0.0), 0.0)
            existing_open_time = _as_utc_datetime(getattr(existing_row, "openTime", None))
            if existing_type is None or existing_open_price <= 0.0 or existing_open_time is None:
                filtered_ticket_ids.append(int(ticket))
        closed_ticket_ids = filtered_ticket_ids

    if not closed_ticket_ids:
        _BOT_OPEN_POSITIONS[bot_config_id] = current_positions
        return

    close_dt = datetime.now(timezone.utc)
    for ticket in closed_ticket_ids:
        pos = previous_positions.get(ticket) or {}
        data = {
            "botInstanceId": bot_instance_id,
            "magicNumber": _safe_int(pos.get("magic", 0), 0) or None,
            "symbol": pos.get("symbol"),
            "volume": pos.get("volume"),
            "openPrice": _safe_float(pos.get("openPrice", 0.0)) or None,
            "openTime": pos.get("openTime"),
            "closePrice": _safe_float(pos.get("closePrice", 0.0)) or None,
            "closeTime": close_dt,
            "profit": _safe_float(pos.get("profit", 0.0)),
            "swap": _safe_float(pos.get("swap", 0.0)),
            "commission": _safe_float(pos.get("commission", 0.0)),
            "status": "closed",
        }
        if pos.get("type") is not None:
            data["type"] = pos.get("type")

        row = existing_by_ticket.get(int(ticket))
        if row:
            row_account_id = str(getattr(row, "accountId", "") or "").strip()
            if row_account_id and row_account_id != account_id:
                logger.warning(
                    "ticket collision skipped ticket=%s existing_account=%s incoming_account=%s",
                    ticket,
                    row_account_id,
                    account_id,
                )
                continue
            await db.orderhistory.update(
                where={"ticketId": ticket},
                data=data,
            )
        else:
            create_data = {
                "ticketId": ticket,
                "accountId": account_id,
                "botInstanceId": bot_instance_id,
                "magicNumber": _safe_int(pos.get("magic", 0), 0) or None,
                "symbol": pos.get("symbol"),
                "volume": pos.get("volume"),
                "openPrice": _safe_float(pos.get("openPrice", 0.0)) or None,
                "openTime": pos.get("openTime"),
                "closePrice": _safe_float(pos.get("closePrice", 0.0)) or None,
                "closeTime": close_dt,
                "profit": _safe_float(pos.get("profit", 0.0)),
                "swap": _safe_float(pos.get("swap", 0.0)),
                "commission": _safe_float(pos.get("commission", 0.0)),
                "status": "closed",
            }
            if pos.get("type") is not None:
                create_data["type"] = pos.get("type")
            await db.orderhistory.create(
                data=create_data,
            )

    _BOT_OPEN_POSITIONS[bot_config_id] = current_positions


def _parse_closed_time(raw) -> datetime | None:
    if raw is None:
        return None
    if isinstance(raw, datetime):
        return raw if raw.tzinfo else raw.replace(tzinfo=timezone.utc)

    text = str(raw or "").strip()
    if not text:
        return None

    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except Exception:
        pass

    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
        except Exception:
            continue
    return None


def _as_utc_datetime(value) -> datetime | None:
    if not isinstance(value, datetime):
        return None
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def _almost_equal(left: float | int | None, right: float | int | None, tolerance: float = 1e-6) -> bool:
    try:
        l_val = float(left or 0.0)
        r_val = float(right or 0.0)
    except Exception:
        return False
    return abs(l_val - r_val) <= tolerance


def _should_update_closed_order(existing, incoming: dict) -> bool:
    existing_magic = _safe_int(getattr(existing, "magicNumber", None), 0)
    incoming_magic = _safe_int(incoming.get("magicNumber"), 0)
    if existing_magic != incoming_magic:
        return True

    existing_symbol = str(getattr(existing, "symbol", "") or "").strip().upper()
    incoming_symbol = str(incoming.get("symbol", "") or "").strip().upper()
    if existing_symbol != incoming_symbol:
        return True

    existing_type = _normalize_order_side(getattr(existing, "type", None))
    incoming_type = _normalize_order_side(incoming.get("type"))
    if incoming_type is not None and existing_type != incoming_type:
        return True

    if not _almost_equal(getattr(existing, "volume", 0.0), incoming.get("volume", 0.0), tolerance=1e-4):
        return True

    incoming_open_price = incoming.get("openPrice")
    if incoming_open_price is not None and not _almost_equal(getattr(existing, "openPrice", 0.0), incoming_open_price, tolerance=1e-6):
        return True

    incoming_close_price = incoming.get("closePrice")
    if incoming_close_price is not None and not _almost_equal(getattr(existing, "closePrice", 0.0), incoming_close_price, tolerance=1e-6):
        return True

    if not _almost_equal(getattr(existing, "profit", 0.0), incoming.get("profit", 0.0), tolerance=1e-4):
        return True

    if not _almost_equal(getattr(existing, "swap", 0.0), incoming.get("swap", 0.0), tolerance=1e-4):
        return True

    if not _almost_equal(getattr(existing, "commission", 0.0), incoming.get("commission", 0.0), tolerance=1e-4):
        return True

    existing_close_time = _as_utc_datetime(getattr(existing, "closeTime", None))
    incoming_close_time = _as_utc_datetime(incoming.get("closeTime"))
    if existing_close_time is None and incoming_close_time is not None:
        return True
    if existing_close_time is not None and incoming_close_time is None:
        return True
    if existing_close_time is not None and incoming_close_time is not None:
        if abs((existing_close_time - incoming_close_time).total_seconds()) > 0.5:
            return True

    existing_open_time = _as_utc_datetime(getattr(existing, "openTime", None))
    incoming_open_time = _as_utc_datetime(incoming.get("openTime"))
    if existing_open_time is None and incoming_open_time is not None:
        return True
    if existing_open_time is not None and incoming_open_time is None:
        return True
    if existing_open_time is not None and incoming_open_time is not None:
        if abs((existing_open_time - incoming_open_time).total_seconds()) > 0.5:
            return True

    existing_status = str(getattr(existing, "status", "") or "").strip().lower()
    if existing_status != "closed":
        return True

    return False


async def _persist_closed_deals_from_state(
    bot_config_id: str,
    account_id: str,
    bot_instance_id: int,
    state: dict,
    *,
    allowed_magic_set: set[int] | None = None,
) -> set[int]:
    raw_deals = state.get("closed_deals")
    if not isinstance(raw_deals, list) or len(raw_deals) == 0:
        return set()

    previous_positions = _BOT_OPEN_POSITIONS.get(bot_config_id, {}) if bot_config_id else {}
    normalized = []
    for raw in raw_deals:
        if not isinstance(raw, dict):
            continue
        ticket = _safe_int(raw.get("ticket"), 0)
        if ticket <= 0:
            continue
        magic = _safe_int(raw.get("magic"), 0)
        if isinstance(allowed_magic_set, set) and len(allowed_magic_set) > 0:
            if magic not in allowed_magic_set:
                continue

        close_time = _parse_closed_time(raw.get("closeTime"))
        if close_time is None:
            close_time_msc = _safe_int(raw.get("closeTimeMsc"), 0)
            if close_time_msc > 0:
                close_time = datetime.fromtimestamp(close_time_msc / 1000.0, tz=timezone.utc)
        if close_time is None:
            close_time = datetime.now(timezone.utc)

        previous_pos = previous_positions.get(int(ticket)) or {}
        side = _normalize_order_side(raw.get("type")) or _normalize_order_side(previous_pos.get("type"))
        open_price = _safe_float(raw.get("openPrice"), 0.0)
        if open_price <= 0.0:
            open_price = _safe_float(previous_pos.get("openPrice"), 0.0)
        open_time = _parse_closed_time(raw.get("openTime"))
        if open_time is None:
            open_time_msc = _safe_int(raw.get("openTimeMsc"), 0)
            if open_time_msc > 0:
                open_time = datetime.fromtimestamp(open_time_msc / 1000.0, tz=timezone.utc)
        if open_time is None:
            open_time_ts = _safe_int(raw.get("openTimeTs"), 0)
            if open_time_ts > 0:
                open_time = datetime.fromtimestamp(open_time_ts, tz=timezone.utc)
        if open_time is None:
            open_time = _as_utc_datetime(previous_pos.get("openTime"))

        close_price = _safe_float(raw.get("closePrice"), 0.0)
        normalized.append(
            {
                "ticket": int(ticket),
                "magic": magic,
                "symbol": str(raw.get("symbol", "") or "").strip().upper() or None,
                "type": side,
                "volume": _safe_float(raw.get("volume"), 0.0),
                "openPrice": open_price if open_price > 0 else None,
                "openTime": open_time,
                "closePrice": close_price if close_price > 0 else None,
                "closeTime": close_time,
                "profit": _safe_float(raw.get("profit"), 0.0),
                "swap": _safe_float(raw.get("swap"), 0.0),
                "commission": _safe_float(raw.get("commission"), 0.0),
            }
        )

    if len(normalized) == 0:
        return set()

    existing_rows = await db.orderhistory.find_many(
        where={"ticketId": {"in": [row["ticket"] for row in normalized]}},
    )
    existing_by_ticket = {
        _safe_int(row.ticketId, 0): row
        for row in existing_rows
    }

    synced_ticket_ids: set[int] = set()
    for row in normalized:
        ticket = int(row["ticket"])
        data: dict = {
            "botInstanceId": bot_instance_id,
            "magicNumber": _safe_int(row.get("magic"), 0) or None,
            "symbol": row["symbol"],
            "volume": row["volume"],
            "closeTime": row["closeTime"],
            "profit": row["profit"],
            "swap": row["swap"],
            "commission": row["commission"],
            "status": "closed",
        }
        if row["type"] is not None:
            data["type"] = row["type"]
        if row["openPrice"] is not None:
            data["openPrice"] = row["openPrice"]
        if row["openTime"] is not None:
            data["openTime"] = row["openTime"]
        if row["closePrice"] is not None:
            data["closePrice"] = row["closePrice"]

        existing = existing_by_ticket.get(ticket)
        if existing:
            row_account_id = str(getattr(existing, "accountId", "") or "").strip()
            if row_account_id and row_account_id != account_id:
                logger.warning(
                    "closed_deal ticket collision skipped ticket=%s existing_account=%s incoming_account=%s",
                    ticket,
                    row_account_id,
                    account_id,
                )
                continue
            if not _should_update_closed_order(existing, data):
                synced_ticket_ids.add(ticket)
                continue
            await db.orderhistory.update(
                where={"ticketId": ticket},
                data=data,
            )
            synced_ticket_ids.add(ticket)
        else:
            await db.orderhistory.create(
                data={
                    "ticketId": ticket,
                    "accountId": account_id,
                    **data,
                }
            )
            synced_ticket_ids.add(ticket)

    return synced_ticket_ids


def _build_account_update_payload(state: dict) -> dict:
    payload: dict = {}

    balance = _safe_finite_float(state.get("balance"), None)
    if balance is not None:
        payload["balance"] = round(balance, 2)

    equity = _safe_finite_float(state.get("equity"), None)
    if equity is not None:
        payload["equity"] = round(equity, 2)

    margin = _safe_finite_float(state.get("margin"), None)
    if margin is not None:
        payload["margin"] = round(margin, 2)

    margin_free = _safe_finite_float(state.get("free_margin"), None)
    if margin_free is not None:
        payload["marginFree"] = round(margin_free, 2)

    margin_level = _safe_finite_float(state.get("margin_level"), None)
    if margin_level is not None:
        payload["marginLevel"] = round(margin_level, 2)

    leverage = _safe_int(state.get("leverage"), 0)
    if leverage > 0:
        payload["leverage"] = leverage

    login = _safe_int(state.get("login"), 0)
    if login > 0:
        payload["mt5LoginId"] = str(login)

    server_name = str(state.get("server", "") or "").strip()
    if server_name:
        payload["serverName"] = server_name[:100]

    return payload


def _should_sync_account_snapshot(bot_config_id: str, payload: dict) -> bool:
    if not payload:
        return False

    now = time.time()
    previous = _BOT_ACCOUNT_SYNC_CACHE.get(bot_config_id) or {}
    previous_ts = _safe_finite_float(previous.get("ts"), 0.0) or 0.0
    previous_payload = previous.get("payload") if isinstance(previous.get("payload"), dict) else {}
    elapsed = max(0.0, now - previous_ts)

    if elapsed < _ACCOUNT_SYNC_MIN_INTERVAL_SEC:
        return False

    if payload != previous_payload:
        return True

    return elapsed >= _ACCOUNT_SYNC_FORCE_INTERVAL_SEC


async def _persist_account_snapshot(bot_config_id: str, account_id: str, state: dict) -> None:
    payload = _build_account_update_payload(state)
    if not _should_sync_account_snapshot(bot_config_id, payload):
        return

    await db.tradingaccount.update_many(
        where={"id": account_id, "recordStatus": "active"},
        data=payload,
    )
    _BOT_ACCOUNT_SYNC_CACHE[bot_config_id] = {
        "ts": time.time(),
        "payload": payload,
    }


async def _persist_bot_state(bot_config_id: str, state: dict) -> None:
    context = await _get_bot_context(bot_config_id)
    if not context:
        return

    account_id = str(context.get("account_id", "")).strip()
    bot_instance_id = _safe_int(context.get("bot_instance_id", 0), 0)
    if not account_id:
        return
    allowed_magic_set = _resolve_allowed_magic_set(context)

    try:
        await _persist_account_snapshot(bot_config_id, account_id, state)
    except Exception as exc:
        logger.warning("account snapshot sync failed for %s: %s", bot_config_id, exc)

    synced_deal_ticket_ids: set[int] = set()
    try:
        synced_deal_ticket_ids = await _persist_closed_deals_from_state(
            bot_config_id,
            account_id,
            bot_instance_id,
            state,
            allowed_magic_set=allowed_magic_set,
        )
    except Exception as exc:
        logger.warning("closed deals sync failed for %s: %s", bot_config_id, exc)

    try:
        await _persist_closed_orders_only(
            bot_config_id,
            account_id,
            bot_instance_id,
            state,
            allowed_magic_set=allowed_magic_set,
            skip_ticket_ids=synced_deal_ticket_ids,
        )
    except Exception as exc:
        logger.warning("closed orders sync failed for %s: %s", bot_config_id, exc)


def _get_cached(symbol: str, timeframe: str, dt_str: str) -> dict | None:
    try:
        raw = r_cache.get(_cache_key(symbol, timeframe, dt_str))
        if raw:
            return json.loads(raw)
    except Exception:
        pass
    return None


def _set_cached(symbol: str, timeframe: str, dt_str: str, data: dict) -> None:
    try:
        r_cache.setex(
            _cache_key(symbol, timeframe, dt_str),
            _CACHE_TTL,
            json.dumps(data, ensure_ascii=False),
        )
    except Exception as exc:
        logger.warning("Redis cache set failed: %s", exc)


# ── WS /ws — Bot connection ──────────────────────────────────────────

@bot_ws_router.websocket("/ws")
async def bot_websocket(websocket: WebSocket):
    """Bot connects, sends register message, then pushes states.

    Protocol:
    1. Bot sends: ``{"type":"register", "bot_config_id":"...", "symbol":"EURUSD", "timeframe":"H1"}``
    2. Bot sends: ``{"type":"state", ...runtime_state_payload}``
    3. Server sends: ``{"type":"llm_result", ...}`` when vision_llm is ready
    """
    await websocket.accept()
    bot_config_id = None

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except Exception:
                continue

            msg_type = msg.get("type", "")

            if msg_type == "register":
                bot_config_id = str(msg.get("bot_config_id", "")).strip()
                symbol = msg.get("symbol", "")
                timeframe = msg.get("timeframe", "H1")
                if bot_config_id and symbol:
                    config = await db.botconfiguration.find_first(
                        where={
                            "id": bot_config_id,
                            "recordStatus": "active",
                            "account": {"recordStatus": "active"},
                        },
                        include={"account": True, "botVersion": True},
                    )
                    if not config:
                        await websocket.send_text(json.dumps({
                            "type": "register_rejected",
                            "reason": "bot_config_not_active",
                        }))
                        bot_config_id = None
                        continue

                    bot_hub.register_bot(websocket, bot_config_id, symbol, timeframe)
                    await websocket.send_text(json.dumps({
                        "type": "registered",
                        "bot_config_id": bot_config_id,
                    }))
                    try:
                        _BOT_CONTEXT[bot_config_id] = {
                            "account_id": str(config.accountId),
                            "bot_instance_id": _safe_int(getattr(config, "botInstanceId", 0), 0),
                            "magic_number": _safe_int(getattr(config, "magicNumber", 0), 0),
                            "owner_user_id": str(getattr(getattr(config, "account", None), "userId", "") or ""),
                            "broker_name": str(getattr(getattr(config, "account", None), "brokerName", "") or ""),
                            "server_name": str(getattr(getattr(config, "account", None), "serverName", "") or ""),
                            "mt5_login_id": str(getattr(getattr(config, "account", None), "mt5LoginId", "") or ""),
                            "bot_label": str(getattr(getattr(config, "botVersion", None), "label", "") or ""),
                        }
                        raw_schedule = getattr(config, "tradingSchedule", None)
                        schedule = normalize_trading_schedule(raw_schedule)
                        await bot_hub.send_bot_config(
                            bot_config_id,
                            {
                                "risk_level": _enum_value(getattr(config, "riskLevel", None)),
                                "trading_schedule": schedule,
                            },
                        )
                    except Exception as exc:
                        logger.warning("bot register config sync failed for %s: %s", bot_config_id, exc)

            elif msg_type == "state" and bot_config_id:
                state = {k: v for k, v in msg.items() if k != "type"}
                await bot_hub.update_bot_state(bot_config_id, state)
                try:
                    await _persist_bot_state(bot_config_id, state)
                except Exception as exc:
                    logger.warning("bot state db sync failed for %s: %s", bot_config_id, exc)
                try:
                    await _emit_insufficient_funds_notification(bot_config_id, state)
                except Exception as exc:
                    logger.warning("bot insufficient-funds notification failed for %s: %s", bot_config_id, exc)
            elif msg_type == "bot_command_ack" and bot_config_id:
                try:
                    await bot_hub.receive_bot_command_ack(bot_config_id, msg)
                except Exception as exc:
                    logger.warning("bot command ack handling failed for %s: %s", bot_config_id, exc)

            elif msg_type == "ping":
                await websocket.send_text("pong")

    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        if bot_config_id:
            bot_hub.disconnect_bot(bot_config_id)
            _BOT_OPEN_POSITIONS.pop(bot_config_id, None)
            _BOT_ACCOUNT_SYNC_CACHE.pop(bot_config_id, None)
            _BOT_INSUFFICIENT_FUNDS_ALERT_CACHE.pop(bot_config_id, None)


# ── WS /ws/dashboard — Dashboard connection ──────────────────────────

@bot_ws_router.websocket("/ws/dashboard")
async def dashboard_websocket(websocket: WebSocket):
    """Dashboard connects to receive real-time bot state updates.

    On connect, receives current snapshot of all connected bots.
    Then receives incremental ``bot_state`` messages as they arrive.
    """
    await bot_hub.connect_dashboard(websocket)
    try:
        # Send initial snapshot
        snapshot = bot_hub.get_all_bot_states()
        lifecycle_events = bot_hub.get_recent_lifecycle_events()
        await websocket.send_text(json.dumps({
            "type": "snapshot",
            "bots": snapshot,
            "lifecycle_events": lifecycle_events,
        }))

        # Keep alive
        while True:
            msg = await websocket.receive_text()
            if msg == "ping":
                await websocket.send_text("pong")

    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        bot_hub.disconnect_dashboard(websocket)


# ── POST /ws/cron — Cron trigger ──────────────────────────────────────

@bot_ws_router.post("/ws/cron")
async def bot_ws_cron(data: VisionLLMRequest):
    """Cron: compute vision_llm for symbol+timeframe, broadcast to matching bots.

    Request body: ``{"date_time": "2025.12.31 15.00", "symbol": "EURUSD", "timeframe": "H1"}``
    """
    dt_str = data.date_time.strftime("%Y-%m-%d %H:%M:%S")
    timeframe = getattr(data, "timeframe", "H1") or "H1"
    logger.info("cron  ▶  %s/%s  %s", data.symbol, timeframe, dt_str)

    # Check cache
    cached = _get_cached(data.symbol, timeframe, dt_str)
    if cached:
        result_data = cached
        logger.info("cron  ⚡  cache hit %s/%s", data.symbol, timeframe)
    else:
        try:
            start = time.perf_counter()
            result, cls_vec = await asyncio.to_thread(
                generate_llm_cls_for_bar, data.date_time, data.symbol, timeframe,
            )
            elapsed = time.perf_counter() - start
            result_data = {
                "symbol": data.symbol.upper(),
                "timeframe": timeframe.upper(),
                "date_time": dt_str,
                "llm_text": result,
                "cls_vec": cls_vec.tolist(),
                "elapsed_seconds": round(elapsed, 2),
            }
            _set_cached(data.symbol, timeframe, dt_str, result_data)
        except NoMarketDataError as exc:
            logger.warning("cron skip %s/%s %s: %s", data.symbol, timeframe, dt_str, exc)
            return {
                "message": "skipped_no_market_data",
                "symbol": data.symbol,
                "timeframe": timeframe,
                "date_time": dt_str,
                "detail": str(exc),
            }
        except MT5ConnectionError as exc:
            logger.warning("cron MT5 unavailable %s/%s %s: %s", data.symbol, timeframe, dt_str, exc)
            raise HTTPException(status_code=503, detail=str(exc))
        except (VisionLLMConfigError, VisionLLMServiceUnavailableError) as exc:
            logger.warning("cron LLM unavailable %s/%s %s: %s", data.symbol, timeframe, dt_str, exc)
            raise HTTPException(status_code=503, detail=str(exc))
        except Exception as exc:
            logger.exception("cron pipeline failed for %s", data.symbol)
            raise HTTPException(status_code=500, detail=str(exc))

    # Broadcast to matching bots
    await bot_hub.broadcast_llm(data.symbol, timeframe, result_data)

    return {
        "message": "success",
        "symbol": data.symbol,
        "timeframe": timeframe,
        "date_time": dt_str,
        "elapsed_seconds": result_data.get("elapsed_seconds", 0),
    }
