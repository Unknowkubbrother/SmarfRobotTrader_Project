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


def _cache_key(symbol: str, dt_str: str) -> str:
    return f"vision_llm:{symbol}:{dt_str}"


def _get_cached(symbol: str, dt_str: str) -> dict | None:
    try:
        raw = r_cache.get(_cache_key(symbol, dt_str))
        if raw:
            return json.loads(raw)
    except Exception:
        pass
    return None


def _set_cached(symbol: str, dt_str: str, data: dict) -> None:
    try:
        r_cache.setex(
            _cache_key(symbol, dt_str),
            _CACHE_TTL,
            json.dumps(data, ensure_ascii=False),
        )
    except Exception as exc:
        logger.warning("Redis cache set failed: %s", exc)


@vision_llm_router.post("/")
async def vision_llm(request: Request, data: VisionLLMRequest):
    """One-off vision-LLM analysis with Redis cache."""
    dt_str = data.date_time.strftime("%Y-%m-%d %H:%M:%S")
    logger.info("vision_llm  ▶  %s  %s", data.symbol, dt_str)

    cached = _get_cached(data.symbol, dt_str)
    if cached:
        logger.info("vision_llm  ⚡  cache hit")
        return {"message": "success", "cached": True, **cached}

    try:
        start = time.perf_counter()
        result, cls_vec = await asyncio.to_thread(
            generate_llm_cls_for_bar, data.date_time, data.symbol,
        )
        elapsed = time.perf_counter() - start

        result_data = {
            "symbol": data.symbol,
            "date_time": dt_str,
            "llm_text": result,
            "cls_vec": cls_vec.tolist(),
            "elapsed_seconds": round(elapsed, 2),
        }
        _set_cached(data.symbol, dt_str, result_data)

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("vision_llm pipeline failed")
        raise HTTPException(status_code=500, detail=str(exc))

    logger.info("vision_llm  ✔  %.1fs", elapsed)
    return {"message": "success", "cached": False, **result_data}