import asyncio
import hashlib
import json
import logging
import time

from fastapi import APIRouter, HTTPException, Request

from ..models.vision_llm_model import VisionLLMRequest, VisionLLMTextEmbeddingRequest
from ..utils.vision_llm.chart import MT5ConnectionError
from ..utils.vision_llm.source_context import build_vision_cache_key, resolve_vision_source_context
from ..utils.vision_llm.llm_client import (
    VisionLLMConfigError,
    VisionLLMServiceUnavailableError,
)
from ..utils.vision_llm.embedding import text_to_cls_embedding
from ..utils.vision_llm.use_llm import generate_llm_cls_for_bar
from ..database.client import r_cache
from ..utils.ws_manager import bot_hub

logger = logging.getLogger(__name__)

vision_llm_router = APIRouter()

_CACHE_TTL = 3900  # 65 minutes
_TEXT_EMBED_CACHE_TTL = 86400 * 7
_INFLIGHT_LOCKS: dict[str, asyncio.Lock] = {}
_INFLIGHT_LOCKS_GUARD = asyncio.Lock()


def _normalize_symbol(symbol: str) -> str:
    return str(symbol or "").strip().upper()


def _normalize_timeframe(timeframe: str) -> str:
    tf = str(timeframe or "H1").strip().upper()
    return tf or "H1"


def _cache_key(symbol: str, timeframe: str, dt_str: str, source_context) -> str:
    return build_vision_cache_key(
        _normalize_symbol(symbol),
        _normalize_timeframe(timeframe),
        dt_str,
        source_context,
    )


def _text_embed_cache_key(text: str) -> str:
    digest = hashlib.sha256(str(text or "").encode("utf-8")).hexdigest()
    return f"vision_llm:text_embed:{digest}"


def _get_cached(symbol: str, timeframe: str, dt_str: str, source_context) -> dict | None:
    try:
        raw = r_cache.get(_cache_key(symbol, timeframe, dt_str, source_context))
        if raw:
            return json.loads(raw)
    except Exception:
        pass
    return None


def _set_cached(symbol: str, timeframe: str, dt_str: str, source_context, data: dict) -> None:
    try:
        r_cache.setex(
            _cache_key(symbol, timeframe, dt_str, source_context),
            _CACHE_TTL,
            json.dumps(data, ensure_ascii=False),
        )
    except Exception as exc:
        logger.warning("Redis cache set failed: %s", exc)


def _get_cached_text_embedding(text: str) -> list[float] | None:
    try:
        raw = r_cache.get(_text_embed_cache_key(text))
        if not raw:
            return None
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            return None
        cls_vec = payload.get("cls_vec")
        if isinstance(cls_vec, list) and len(cls_vec) > 0:
            return cls_vec
    except Exception:
        return None
    return None


def _set_cached_text_embedding(text: str, cls_vec: list[float]) -> None:
    try:
        r_cache.setex(
            _text_embed_cache_key(text),
            _TEXT_EMBED_CACHE_TTL,
            json.dumps({"cls_vec": cls_vec}, ensure_ascii=False),
        )
    except Exception as exc:
        logger.warning("Redis text-embed cache set failed: %s", exc)


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
    bot_config_id = str(getattr(data, "bot_config_id", "") or "").strip()
    source_context = resolve_vision_source_context(bot_config_id)
    logger.info(
        "vision_llm  ▶  %s/%s  %s  source=%s",
        symbol,
        timeframe,
        dt_str,
        source_context.display_label,
    )

    cached = _get_cached(symbol, timeframe, dt_str, source_context)
    if cached:
        logger.info("vision_llm  ⚡  cache hit")
        if bot_config_id and cached.get("llm_text"):
            await bot_hub.patch_bot_state(bot_config_id, {"llm_text": str(cached.get("llm_text") or "")})
        return {"message": "success", "cached": True, **cached}

    cache_key = _cache_key(symbol, timeframe, dt_str, source_context)
    lock = await _acquire_inflight_lock(cache_key)
    try:
        # Double-check after waiting lock: another request may have filled cache.
        cached_after_wait = _get_cached(symbol, timeframe, dt_str, source_context)
        if cached_after_wait:
            logger.info("vision_llm  ⚡  cache hit after wait")
            if bot_config_id and cached_after_wait.get("llm_text"):
                await bot_hub.patch_bot_state(bot_config_id, {"llm_text": str(cached_after_wait.get("llm_text") or "")})
            return {"message": "success", "cached": True, **cached_after_wait}

        try:
            start = time.perf_counter()
            chart_rates_payload = [row.model_dump() for row in (data.chart_rates or [])] or None
            result, cls_vec, chart_result = await asyncio.to_thread(
                generate_llm_cls_for_bar,
                date_time=data.date_time,
                symbol=symbol,
                timeframe=timeframe,
                dataset_json=None,
                bot_config_id=(bot_config_id or None) if chart_rates_payload else None,
                chart_rates=chart_rates_payload,
                resolved_bar_time=str(getattr(data, "resolved_bar_time", "") or "").strip() or None,
                source_server=str(getattr(data, "source_server", "") or "").strip() or None,
                source_login=str(getattr(data, "source_login", "") or "").strip() or None,
            )
            elapsed = time.perf_counter() - start

            result_data = {
                "symbol": symbol,
                "timeframe": timeframe,
                "date_time": dt_str,
                "llm_text": result,
                "cls_vec": cls_vec.tolist(),
                "elapsed_seconds": round(elapsed, 2),
                "chart_source": {
                    "cache_scope": source_context.cache_scope,
                    "requested_label": source_context.display_label,
                    "mode": chart_result.source_mode,
                    "label": chart_result.source_label,
                    "resolved_bar_time": chart_result.resolved_bar_time,
                },
            }
            _set_cached(symbol, timeframe, dt_str, source_context, result_data)

        except MT5ConnectionError as exc:
            raise HTTPException(status_code=503, detail=str(exc))
        except (VisionLLMConfigError, VisionLLMServiceUnavailableError) as exc:
            raise HTTPException(status_code=503, detail=str(exc))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            logger.exception("vision_llm pipeline failed")
            raise HTTPException(status_code=500, detail=str(exc))
    finally:
        await _release_inflight_lock(cache_key, lock)

    logger.info("vision_llm  ✔  %.1fs", elapsed)
    if bot_config_id and result_data.get("llm_text"):
        await bot_hub.patch_bot_state(bot_config_id, {"llm_text": str(result_data.get("llm_text") or "")})
    return {"message": "success", "cached": False, **result_data}


@vision_llm_router.post("/embed_text")
async def vision_llm_embed_text(data: VisionLLMTextEmbeddingRequest):
    text = str(getattr(data, "text", "") or "")
    if not text.strip():
        raise HTTPException(status_code=400, detail="text is required")

    cached_vec = _get_cached_text_embedding(text)
    if cached_vec is not None:
        return {
            "message": "success",
            "cached": True,
            "cls_vec": cached_vec,
            "text_len": len(text),
        }

    try:
        cls_vec = await asyncio.to_thread(text_to_cls_embedding, text)
        cls_list = cls_vec.tolist()
        _set_cached_text_embedding(text, cls_list)
        return {
            "message": "success",
            "cached": False,
            "cls_vec": cls_list,
            "text_len": len(text),
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("vision_llm text embedding failed")
        raise HTTPException(status_code=500, detail=str(exc))
