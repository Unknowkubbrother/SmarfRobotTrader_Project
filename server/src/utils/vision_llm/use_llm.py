"""
use_llm — Public API facade for the Vision LLM pipeline.

This module re-exports the key components from the sub-modules and
provides the two main entry-point functions used by the route layer.
"""

import os
from typing import Optional, Tuple
from datetime import datetime

import numpy as np

from .chart import NoMarketDataError, generate_image   # noqa: F401
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
    dataset_json: Optional[str] = None,
) -> str:
    """Generate a full RAG-based analysis text for one chart bar."""
    runtime = get_runtime(dataset_json)
    base64_image = generate_image(date_time, symbol=symbol)
    answer = run_rag_pipeline(
        chart_db=runtime["chart_db"],
        text_db=runtime["text_db"],
        vision_llm=runtime["vision_llm"],
        dataset_json=runtime["dataset_json"],
        base64_image=base64_image,
    )
    return str(answer or "").strip()


def generate_llm_cls_for_bar(
    date_time: datetime,
    symbol: str = "EURUSD",
    dataset_json: Optional[str] = None,
) -> Tuple[str, np.ndarray]:
    """Generate analysis text *and* its CLS embedding vector."""
    llm_text = generate_llm_text_for_bar(
        date_time=date_time,
        symbol=symbol,
        dataset_json=dataset_json,
    )
    cls_vec = text_to_cls_embedding(llm_text)
    return llm_text, cls_vec
