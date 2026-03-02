"""
Bot Hub — WebSocket connection manager for bots and dashboard clients.

Two types of clients:
- **Bots** (from Docker containers): identified by ``bot_config_id``, grouped by symbol+timeframe
- **Dashboard** (Next.js): receive all bot state updates in real-time
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, Set

from fastapi import WebSocket

logger = logging.getLogger(__name__)
_WS_SEND_TIMEOUT_SEC = 0.75


@dataclass
class BotConnection:
    """A registered bot WebSocket client."""
    websocket: WebSocket
    bot_config_id: str
    symbol: str
    timeframe: str
    last_state: dict = field(default_factory=dict)


class BotHub:
    """Central hub managing bot ↔ server ↔ dashboard WebSocket connections."""

    def __init__(self) -> None:
        self._bots: Dict[str, BotConnection] = {}       # bot_config_id -> BotConnection
        self._dashboards: Set[WebSocket] = set()

    # ── Bot connections ──────────────────────────────────────────────

    def register_bot(self, websocket: WebSocket, bot_config_id: str,
                     symbol: str, timeframe: str) -> None:
        bot_id = str(bot_config_id).strip()
        if not bot_id:
            return
        conn = BotConnection(
            websocket=websocket,
            bot_config_id=bot_id,
            symbol=symbol.upper(),
            timeframe=timeframe.upper(),
        )
        self._bots[bot_id] = conn
        logger.info(
            "bot register  ▶  %s  %s/%s  (total_bots=%d)",
            bot_id, symbol, timeframe, len(self._bots),
        )

    def disconnect_bot(self, bot_config_id: str) -> None:
        bot_id = str(bot_config_id).strip()
        if not bot_id:
            return
        self._bots.pop(bot_id, None)
        logger.info("bot disconnect  ◀  %s  (total_bots=%d)", bot_id, len(self._bots))

    def get_bot(self, bot_config_id: str) -> BotConnection | None:
        bot_id = str(bot_config_id).strip()
        if not bot_id:
            return None
        return self._bots.get(bot_id)

    def get_all_bot_states(self) -> list[dict]:
        """Return list of last-known states of all connected bots."""
        states = []
        for conn in self._bots.values():
            states.append({
                "bot_config_id": conn.bot_config_id,
                "symbol": conn.symbol,
                "timeframe": conn.timeframe,
                "connected": True,
                **conn.last_state,
            })
        return states

    # ── Dashboard connections ────────────────────────────────────────

    async def connect_dashboard(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self._dashboards.add(websocket)
        logger.info("dashboard connect  ▶  (total=%d)", len(self._dashboards))

    def disconnect_dashboard(self, websocket: WebSocket) -> None:
        self._dashboards.discard(websocket)
        logger.info("dashboard disconnect  ◀  (total=%d)", len(self._dashboards))

    # ── Bot state updates ────────────────────────────────────────────

    async def update_bot_state(self, bot_config_id: str, state: dict) -> None:
        """Store bot state and forward to all dashboard clients."""
        bot_id = str(bot_config_id).strip()
        if not bot_id:
            return

        conn = self._bots.get(bot_id)
        if conn:
            conn.last_state = state

        # Forward to dashboards
        payload = {
            "type": "bot_state",
            "bot_config_id": bot_id,
            **state,
        }
        await self._broadcast_dashboards(payload)

    # ── Vision LLM broadcast ────────────────────────────────────────

    async def broadcast_llm(self, symbol: str, timeframe: str, data: dict) -> None:
        """Push vision_llm result to all bots matching symbol+timeframe."""
        symbol = symbol.upper()
        timeframe = timeframe.upper()
        matching = [
            conn for conn in self._bots.values()
            if conn.symbol == symbol and conn.timeframe == timeframe
        ]
        if not matching:
            logger.info("llm broadcast  📡  %s/%s  no matching bots", symbol, timeframe)
            return

        message = json.dumps({"type": "llm_result", **data}, ensure_ascii=False)
        disconnected = []

        async def _send(conn: BotConnection):
            try:
                await asyncio.wait_for(conn.websocket.send_text(message), timeout=_WS_SEND_TIMEOUT_SEC)
            except Exception:
                disconnected.append(conn.bot_config_id)

        await asyncio.gather(*[_send(c) for c in matching])

        for bid in disconnected:
            self.disconnect_bot(bid)

        logger.info(
            "llm broadcast  📡  %s/%s  sent=%d  dropped=%d",
            symbol, timeframe,
            len(matching) - len(disconnected), len(disconnected),
        )

    async def send_bot_config(self, bot_config_id: str, config_data: dict) -> bool:
        """Push runtime config (risk/schedule/etc.) to one connected bot."""
        bot_id = str(bot_config_id).strip()
        if not bot_id:
            return False

        conn = self._bots.get(bot_id)
        if conn is None:
            return False

        payload = {
            "type": "bot_config",
            "bot_config_id": bot_id,
            **(config_data or {}),
        }
        try:
            await asyncio.wait_for(
                conn.websocket.send_text(json.dumps(payload, ensure_ascii=False)),
                timeout=_WS_SEND_TIMEOUT_SEC,
            )
            return True
        except Exception:
            self.disconnect_bot(bot_id)
            return False

    # ── Internal ─────────────────────────────────────────────────────

    async def _broadcast_dashboards(self, data: dict) -> None:
        if not self._dashboards:
            return
        message = json.dumps(data, ensure_ascii=False)
        disconnected = []

        async def _send(ws: WebSocket):
            try:
                await asyncio.wait_for(ws.send_text(message), timeout=_WS_SEND_TIMEOUT_SEC)
            except Exception:
                disconnected.append(ws)

        await asyncio.gather(*[_send(ws) for ws in list(self._dashboards)])

        for ws in disconnected:
            self.disconnect_dashboard(ws)


# Singleton
bot_hub = BotHub()
