"""
Vision LLM route — one-off HTTP endpoint (with Redis cache).

WebSocket and cron endpoints have been moved to ``bot_ws.py``.
"""

import asyncio
import json
import logging
import time

from fastapi import APIRouter, HTTPException, Request

from ..models.vision_llm_model import VisionLLMRequest
from ..utils.vision_llm.use_llm import generate_llm_cls_for_bar
from ..database.client import r_cache

logger = logging.getLogger(__name__)

vision_llm_router = APIRouter()

_CACHE_TTL = 3900  # 65 minutes
_INFLIGHT_LOCKS: dict[str, asyncio.Lock] = {}
_INFLIGHT_LOCKS_GUARD = asyncio.Lock()


def _normalize_symbol(symbol: str) -> str:
    return str(symbol or "").strip().upper()


def _normalize_timeframe(timeframe: str) -> str:
    tf = str(timeframe or "H1").strip().upper()
    return tf or "H1"


def _cache_key(symbol: str, timeframe: str, dt_str: str) -> str:
    return f"vision_llm:{_normalize_symbol(symbol)}:{_normalize_timeframe(timeframe)}:{dt_str}"


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


async def _acquire_inflight_lock(cache_key: str) -> asyncio.Lock:
    async with _INFLIGHT_LOCKS_GUARD:
        lock = _INFLIGHT_LOCKS.get(cache_key)
        if lock is None:
            lock = asyncio.Lock()
            _INFLIGHT_LOCKS[cache_key] = lock
    await lock.acquire()
    return lock


async def _release_inflight_lock(cache_key: str, lock: asyncio.Lock) -> None:
    try:
        lock.release()
    except Exception:
        pass
    async with _INFLIGHT_LOCKS_GUARD:
        current = _INFLIGHT_LOCKS.get(cache_key)
        if current is lock and not lock.locked():
            _INFLIGHT_LOCKS.pop(cache_key, None)


@vision_llm_router.post("/")
async def vision_llm(request: Request, data: VisionLLMRequest):
    """One-off vision-LLM analysis with Redis cache."""
    dt_str = data.date_time.strftime("%Y-%m-%d %H:%M:%S")
    symbol = _normalize_symbol(data.symbol)
    timeframe = _normalize_timeframe(getattr(data, "timeframe", "H1"))
    logger.info("vision_llm  ▶  %s/%s  %s", symbol, timeframe, dt_str)

    cached = _get_cached(symbol, timeframe, dt_str)
    if cached:
        logger.info("vision_llm  ⚡  cache hit")
        return {"message": "success", "cached": True, **cached}

    cache_key = _cache_key(symbol, timeframe, dt_str)
    lock = await _acquire_inflight_lock(cache_key)
    try:
        # Double-check after waiting lock: another request may have filled cache.
        cached_after_wait = _get_cached(symbol, timeframe, dt_str)
        if cached_after_wait:
            logger.info("vision_llm  ⚡  cache hit after wait")
            return {"message": "success", "cached": True, **cached_after_wait}

        try:
            start = time.perf_counter()
            result, cls_vec = await asyncio.to_thread(
                generate_llm_cls_for_bar, data.date_time, symbol,
            )
            elapsed = time.perf_counter() - start

            result_data = {
                "symbol": symbol,
                "timeframe": timeframe,
                "date_time": dt_str,
                "llm_text": result,
                "cls_vec": cls_vec.tolist(),
                "elapsed_seconds": round(elapsed, 2),
            }
            _set_cached(symbol, timeframe, dt_str, result_data)

        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            logger.exception("vision_llm pipeline failed")
            raise HTTPException(status_code=500, detail=str(exc))
    finally:
        await _release_inflight_lock(cache_key, lock)

    logger.info("vision_llm  ✔  %.1fs", elapsed)
    return {"message": "success", "cached": False, **result_data}
