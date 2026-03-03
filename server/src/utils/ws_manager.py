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
from datetime import datetime, timezone
from typing import Dict, Set
from uuid import uuid4

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
        self._command_waiters: Dict[tuple[str, str], asyncio.Future] = {}
        self._command_acks: Dict[tuple[str, str], dict] = {}
        self._recent_lifecycle_events: list[dict] = []

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
        self._clear_bot_command_state(bot_id)
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

    def get_recent_lifecycle_events(self, limit: int = 150) -> list[dict]:
        size = max(1, min(int(limit), 500))
        return list(self._recent_lifecycle_events[-size:])

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

    async def broadcast_lifecycle_event(
        self,
        bot_config_id: str,
        action: str,
        phase: str,
        detail: str | None = None,
        status: str | None = None,
        source: str | None = None,
        metadata: dict | None = None,
    ) -> None:
        """Broadcast bot lifecycle events (requested/succeeded/failed) to dashboards."""
        bot_id = str(bot_config_id).strip()
        action_name = str(action or "").strip().lower()
        phase_name = str(phase or "").strip().lower()
        if not bot_id or not action_name or not phase_name:
            return

        payload = {
            "type": "bot_lifecycle",
            "event_id": str(uuid4()),
            "bot_config_id": bot_id,
            "action": action_name,
            "phase": phase_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if detail:
            payload["detail"] = str(detail)
        if status:
            payload["status"] = str(status).strip().lower()
        if source:
            payload["source"] = str(source).strip().lower()
        if metadata and isinstance(metadata, dict):
            payload["meta"] = dict(metadata)

        self._recent_lifecycle_events.append(payload)
        if len(self._recent_lifecycle_events) > 500:
            self._recent_lifecycle_events = self._recent_lifecycle_events[-500:]

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

        payload = {
            "type": "bot_config",
            "bot_config_id": bot_id,
            **(config_data or {}),
        }
        return await self._send_payload_to_bot(bot_id, payload)

    async def send_bot_command(
        self,
        bot_config_id: str,
        command: str,
        payload: dict | None = None,
        command_id: str | None = None,
    ) -> str | None:
        """Push one control command to bot. Returns command_id on success."""
        bot_id = str(bot_config_id).strip()
        command_name = str(command or "").strip().lower()
        if not bot_id or not command_name:
            return None

        resolved_command_id = str(command_id or uuid4()).strip()
        if not resolved_command_id:
            return None

        packet = {
            "type": "bot_command",
            "bot_config_id": bot_id,
            "command": command_name,
            "command_id": resolved_command_id,
            **(payload or {}),
        }
        sent = await self._send_payload_to_bot(bot_id, packet)
        return resolved_command_id if sent else None

    async def wait_for_command_ack(
        self,
        bot_config_id: str,
        command_id: str,
        timeout_sec: float = 30.0,
    ) -> dict | None:
        """Wait for a command ack from bot; returns None on timeout."""
        bot_id = str(bot_config_id).strip()
        cmd_id = str(command_id or "").strip()
        if not bot_id or not cmd_id:
            return None

        key = (bot_id, cmd_id)
        cached = self._command_acks.pop(key, None)
        if cached is not None:
            return cached

        loop = asyncio.get_running_loop()
        waiter = loop.create_future()
        self._command_waiters[key] = waiter
        try:
            return await asyncio.wait_for(waiter, timeout=timeout_sec)
        except asyncio.TimeoutError:
            return None
        finally:
            self._command_waiters.pop(key, None)

    async def receive_bot_command_ack(self, bot_config_id: str, ack_data: dict) -> None:
        """Register command ack pushed by bot runtime."""
        bot_id = str(bot_config_id).strip()
        if not bot_id:
            return

        command_id = str((ack_data or {}).get("command_id", "")).strip()
        if not command_id:
            return

        key = (bot_id, command_id)
        waiter = self._command_waiters.get(key)
        if waiter is not None and not waiter.done():
            waiter.set_result(dict(ack_data or {}))
        else:
            self._command_acks[key] = dict(ack_data or {})
            # Keep memory bounded if caller never consumes.
            if len(self._command_acks) > 256:
                oldest_key = next(iter(self._command_acks.keys()))
                self._command_acks.pop(oldest_key, None)

    # ── Internal ─────────────────────────────────────────────────────

    async def _send_payload_to_bot(self, bot_id: str, payload: dict) -> bool:
        conn = self._bots.get(bot_id)
        if conn is None:
            return False
        try:
            await asyncio.wait_for(
                conn.websocket.send_text(json.dumps(payload, ensure_ascii=False)),
                timeout=_WS_SEND_TIMEOUT_SEC,
            )
            return True
        except Exception:
            self.disconnect_bot(bot_id)
            return False

    def _clear_bot_command_state(self, bot_id: str) -> None:
        keys = [key for key in self._command_waiters.keys() if key[0] == bot_id]
        for key in keys:
            waiter = self._command_waiters.pop(key, None)
            if waiter is not None and not waiter.done():
                waiter.set_result(
                    {
                        "bot_config_id": bot_id,
                        "command_id": key[1],
                        "ok": False,
                        "detail": "Bot disconnected before command ack",
                    }
                )

        ack_keys = [key for key in self._command_acks.keys() if key[0] == bot_id]
        for key in ack_keys:
            self._command_acks.pop(key, None)

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
