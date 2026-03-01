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
import time
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

from ..models.vision_llm_model import VisionLLMRequest
from ..utils.vision_llm.chart import NoMarketDataError
from ..utils.trading_schedule import normalize_trading_schedule
from ..utils.vision_llm.use_llm import generate_llm_cls_for_bar
from ..utils.ws_manager import bot_hub
from ..database.client import db, r_cache

logger = logging.getLogger(__name__)

bot_ws_router = APIRouter()

_CACHE_TTL = 3900  # 65 minutes
_BOT_CONTEXT: dict[str, dict] = {}
_BOT_OPEN_POSITIONS: dict[str, dict[int, dict]] = {}


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

    config = await db.botconfiguration.find_unique(
        where={"id": bot_config_id},
        include={"account": True},
    )
    if not config:
        return None

    context = {
        "account_id": str(config.accountId),
        "bot_instance_id": _safe_int(getattr(config, "botInstanceId", 0), 0),
    }
    _BOT_CONTEXT[bot_config_id] = context
    return context


def _extract_live_positions(state: dict) -> dict[int, dict]:
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
        side_raw = str(item.get("type", "")).strip().upper()
        side = "buy" if side_raw == "BUY" else "sell" if side_raw == "SELL" else None
        pos_payload = {
            "ticket": ticket,
            "symbol": str(item.get("symbol") or state.get("symbol") or "").strip().upper() or None,
            "type": side,
            "volume": _safe_float(item.get("volume", 0.0)),
            "openPrice": _safe_float(item.get("price_open", 0.0)),
            "closePrice": _safe_float(item.get("price_current", 0.0)),
            "profit": _safe_float(item.get("profit", 0.0)),
            "swap": _safe_float(item.get("swap", 0.0)),
            "openTime": _parse_open_time(item),
        }
        positions[ticket] = pos_payload
    return positions


async def _persist_closed_orders_only(bot_config_id: str, account_id: str, bot_instance_id: int, state: dict) -> None:
    current_positions = _extract_live_positions(state)
    previous_positions = _BOT_OPEN_POSITIONS.get(bot_config_id, {})
    closed_ticket_ids = [
        int(ticket) for ticket in previous_positions.keys()
        if int(ticket) not in current_positions
    ]
    if not closed_ticket_ids:
        _BOT_OPEN_POSITIONS[bot_config_id] = current_positions
        return

    existing = await db.orderhistory.find_many(
        where={"ticketId": {"in": closed_ticket_ids}},
    )
    existing_by_ticket = {
        _safe_int(row.ticketId, 0): row
        for row in existing
    }

    close_dt = datetime.now(timezone.utc)
    for ticket in closed_ticket_ids:
        pos = previous_positions.get(ticket) or {}
        data = {
            "botInstanceId": bot_instance_id,
            "symbol": pos.get("symbol"),
            "volume": pos.get("volume"),
            "openPrice": _safe_float(pos.get("openPrice", 0.0)) or None,
            "openTime": pos.get("openTime"),
            "closePrice": _safe_float(pos.get("closePrice", 0.0)) or None,
            "closeTime": close_dt,
            "profit": _safe_float(pos.get("profit", 0.0)),
            "swap": _safe_float(pos.get("swap", 0.0)),
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
                "account": {"connect": {"id": account_id}},
                "botInstanceId": bot_instance_id,
                "symbol": pos.get("symbol"),
                "volume": pos.get("volume"),
                "openPrice": _safe_float(pos.get("openPrice", 0.0)) or None,
                "openTime": pos.get("openTime"),
                "closePrice": _safe_float(pos.get("closePrice", 0.0)) or None,
                "closeTime": close_dt,
                "profit": _safe_float(pos.get("profit", 0.0)),
                "swap": _safe_float(pos.get("swap", 0.0)),
                "status": "closed",
            }
            if pos.get("type") is not None:
                create_data["type"] = pos.get("type")
            await db.orderhistory.create(
                data=create_data,
            )

    _BOT_OPEN_POSITIONS[bot_config_id] = current_positions


async def _persist_bot_state(bot_config_id: str, state: dict) -> None:
    context = await _get_bot_context(bot_config_id)
    if not context:
        return

    account_id = str(context.get("account_id", "")).strip()
    bot_instance_id = _safe_int(context.get("bot_instance_id", 0), 0)
    if not account_id:
        return

    await _persist_closed_orders_only(bot_config_id, account_id, bot_instance_id, state)


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
                    bot_hub.register_bot(websocket, bot_config_id, symbol, timeframe)
                    await websocket.send_text(json.dumps({
                        "type": "registered",
                        "bot_config_id": bot_config_id,
                    }))
                    try:
                        config = await db.botconfiguration.find_unique(
                            where={"id": bot_config_id},
                            include={"account": True},
                        )
                        if config:
                            _BOT_CONTEXT[bot_config_id] = {
                                "account_id": str(config.accountId),
                                "bot_instance_id": _safe_int(getattr(config, "botInstanceId", 0), 0),
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
        await websocket.send_text(json.dumps({
            "type": "snapshot",
            "bots": snapshot,
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
                generate_llm_cls_for_bar, data.date_time, data.symbol,
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
