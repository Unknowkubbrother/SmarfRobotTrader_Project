"""
Vision LLM route — async endpoint that runs the heavy LLM pipeline
in a background thread so it never blocks the FastAPI event loop.
"""

import asyncio
import logging
import time

from fastapi import APIRouter, HTTPException, Request

from ..models.vision_llm_model import VisionLLMRequest
from ..utils.vision_llm.use_llm import generate_llm_cls_for_bar

logger = logging.getLogger(__name__)

vision_llm_router = APIRouter()


@vision_llm_router.post("/")
async def vision_llm(request: Request, data: VisionLLMRequest):
    """Analyse a chart bar with the Vision-LLM RAG pipeline.

    The heavy CPU/IO work runs in a separate thread via
    ``asyncio.to_thread`` so other requests are never blocked.
    """
    logger.info("vision_llm  ▶  %s  %s", data.symbol, data.date_time)
    start = time.perf_counter()

    try:
        result, cls_vec = await asyncio.to_thread(
            generate_llm_cls_for_bar,
            data.date_time,
            data.symbol,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("vision_llm pipeline failed")
        raise HTTPException(status_code=500, detail=str(exc))

    elapsed = time.perf_counter() - start
    logger.info("vision_llm  ✔  %.1fs", elapsed)

    return {
        "message": "success",
        "data": result,
        "cls_vec": cls_vec.tolist(),
        "elapsed_seconds": round(elapsed, 2),
    }