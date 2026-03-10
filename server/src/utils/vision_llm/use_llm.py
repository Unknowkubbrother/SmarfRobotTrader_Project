"""
use_llm — Public API facade for the Vision LLM pipeline.

This module re-exports the key components from the sub-modules and
provides the two main entry-point functions used by the route layer.
"""

import os
from typing import Optional, Tuple
from datetime import datetime

import numpy as np

from .chart import ChartImageResult, NoMarketDataError, generate_image_result   # noqa: F401
from .llm_client import (                               # noqa: F401
    get_runtime,
    run_rag_pipeline,
)
from .embedding import text_to_cls_embedding            # noqa: F401

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ── Main entry points ────────────────────────────────────────────────

def generate_llm_text_for_bar(
    date_time: datetime,
    symbol: str = "EURUSD",
    timeframe: str = "H1",
    dataset_json: Optional[str] = None,
    bot_config_id: Optional[str] = None,
) -> Tuple[str, ChartImageResult]:
    """Generate a full RAG-based analysis text for one chart bar."""
    runtime = get_runtime(dataset_json)
    chart_result = generate_image_result(
        date_time,
        symbol=symbol,
        timeframe=timeframe,
        bot_config_id=bot_config_id,
    )
    answer = run_rag_pipeline(
        chart_db=runtime["chart_db"],
        text_db=runtime["text_db"],
        vision_llm=runtime["vision_llm"],
        dataset_json=runtime["dataset_json"],
        base64_image=chart_result.image_base64,
    )
    return str(answer or "").strip(), chart_result


def generate_llm_cls_for_bar(
    date_time: datetime,
    symbol: str = "EURUSD",
    timeframe: str = "H1",
    dataset_json: Optional[str] = None,
    bot_config_id: Optional[str] = None,
) -> Tuple[str, np.ndarray, ChartImageResult]:
    """Generate analysis text *and* its CLS embedding vector."""
    llm_text, chart_result = generate_llm_text_for_bar(
        date_time=date_time,
        symbol=symbol,
        timeframe=timeframe,
        dataset_json=dataset_json,
        bot_config_id=bot_config_id,
    )
    cls_vec = text_to_cls_embedding(llm_text)
    return llm_text, cls_vec, chart_result
