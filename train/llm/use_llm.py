import os
import base64
from io import BytesIO
from typing import Optional, Tuple
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
import numpy as np
from FlagEmbedding import BGEM3FlagModel

from datetime import datetime, timedelta, timezone

import pandas as pd
import mplfinance as mpf
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
from mt5linux import MetaTrader5

from retrieval import (
    upsert_image_dataset,
    upsert_text_dataset,
    hybrid_search_image_query,
)
from retrieval.utils import (
    mask_numbers,
    strip_markdown,
    build_query_text_from_auto,
    build_rag_context,
    print_results
)
from prompts import (
    PROMPT_DRAFT_FROM_IMAGE,
    PROMPT_DOMAIN_REWRITE,
    AUTO_TEXT_COMPRESS_NOTE,
    RAG_TEMPLATE
)

load_dotenv()

os.environ["TOKENIZERS_PARALLELISM"] = "false"

LLM_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATASET_JSON = os.getenv("LLM_DATASET_JSON", os.path.join(LLM_ROOT, "dataset.json"))
DEFAULT_EMBED_MODEL = os.getenv("LLM_EMBED_MODEL", "BAAI/bge-m3")

_runtime_cache = {}
_embedder = None

class VisionLLMClient:
    def __init__(self):
        # self.llm= init_chat_model(
        #     model="ministral-3:14b",
        #     model_provider="ollama",
        #     base_url="http://localhost:11434",
        # )

        self.llm= init_chat_model(
            model="ministral-3:14b",
            model_provider="ollama",
            base_url="http://202.44.40.197:11434",
        )

    def invoke(self, text: str, image_base64: str) -> str:
        messages = [
            HumanMessage(
                content=[
                    {"type": "text", "text": text},
                    {
                        "type": "image_url",
                        "image_url": f"data:image/png;base64,{image_base64}",
                    },
                ]
            )
        ]
        response = self.llm.invoke(messages)
        return strip_markdown(response.content)

def _normalize_dataset_json(dataset_json: Optional[str]) -> str:
    path = (dataset_json or DEFAULT_DATASET_JSON).strip() or DEFAULT_DATASET_JSON
    if not os.path.isabs(path):
        path = os.path.join(LLM_ROOT, path)
    return os.path.abspath(path)


def _get_runtime(dataset_json: Optional[str] = None):
    dataset_path = _normalize_dataset_json(dataset_json)
    runtime = _runtime_cache.get(dataset_path)
    if runtime is not None:
        return runtime

    chart_db = upsert_image_dataset(dataset_path)
    text_db = upsert_text_dataset(dataset_path)
    vision_llm = VisionLLMClient()
    runtime = {
        "dataset_json": dataset_path,
        "chart_db": chart_db,
        "text_db": text_db,
        "vision_llm": vision_llm,
    }
    _runtime_cache[dataset_path] = runtime
    return runtime


def _get_embedder() -> BGEM3FlagModel:
    global _embedder
    if _embedder is None:
        _embedder = BGEM3FlagModel(DEFAULT_EMBED_MODEL, use_fp16=True)
    return _embedder


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-12:
        return arr
    return (arr / norm).astype(np.float32)


def text_to_cls_embedding(text: str) -> np.ndarray:
    embedder = _get_embedder()
    clean = str(text or "").strip()
    out = embedder.encode(
        [clean],
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=True,
    )

    colbert = out.get("colbert_vecs", [])
    if len(colbert) > 0:
        token_vecs = np.asarray(colbert[0], dtype=np.float32)
        if token_vecs.ndim == 2 and token_vecs.shape[0] > 0:
            return _l2_normalize(token_vecs[0])

    dense = out.get("dense_vecs", [])
    if len(dense) > 0:
        return _l2_normalize(np.asarray(dense[0], dtype=np.float32))
    return np.zeros(1024, dtype=np.float32)

def run_rag_pipeline(chart_db, text_db, vision_llm, DATASET_JSON ,base64_image : str) -> str:

    draft_clean = vision_llm.invoke(PROMPT_DRAFT_FROM_IMAGE, base64_image)

    ex_docs = text_db.similarity_search(draft_clean, k=6)
    domain_examples = "\n\n---\n\n".join(
        mask_numbers(d.page_content) for d in ex_docs if getattr(d, "page_content", None)
    )
    
    if not domain_examples:
        domain_examples = "ไม่มีตัวอย่าง (fallback): ให้ใช้สำนวนเทคนิคแบบนักเทรดไทย เน้น PA logic และคำค้นที่ชัดเจน"

    rewrite_prompt = f"""
DRAFT:
{draft_clean}

DOMAIN EXAMPLES (จาก dataset เดิม):
{domain_examples}

{PROMPT_DOMAIN_REWRITE}

{AUTO_TEXT_COMPRESS_NOTE}
""".strip()

    auto_text = vision_llm.invoke(rewrite_prompt, base64_image)

    query_text = build_query_text_from_auto(auto_text)

    results = hybrid_search_image_query(
        chart_db=chart_db,
        text_db=text_db,
        dataset_json=DATASET_JSON,
        base64_image=base64_image,
        auto_text=query_text,
        k_img=10,
        k_t=10,
        final_k=5,
        w_img=0.85,
        w_t=0.15,
        rerank=True,
        rerank_top_m=20,
        w_rerank=0.45,
    )

    # print_results("IMAGE → HYBRID (Chart + Text via auto_text)", results)

    rag_context = build_rag_context(results, max_chars=1500)

    formatted_prompt = RAG_TEMPLATE.format(context=rag_context)
    
    final_answer = vision_llm.invoke(formatted_prompt, base64_image)

    return final_answer    


def generate_image(date_time: datetime, symbol: str = "EURUSD") -> str:
    mt5 = MetaTrader5(host="localhost", port=8001)
    mt5.initialize()

    try:
        start_input = date_time.replace(minute=0, second=0, microsecond=0)
        if start_input.tzinfo is None:
            start_utc = start_input.replace(tzinfo=timezone.utc)
        else:
            start_utc = start_input.astimezone(timezone.utc)
        end_utc = start_utc + timedelta(hours=1)

        rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M1, start_utc, end_utc)

        if rates is None or len(rates) == 0:
            raise ValueError(f"❌ No data found for {symbol} at {date_time}")

        # 5. Prepare DataFrame
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s', utc=True).dt.tz_convert(None)
        df.set_index('time', inplace=True)
        df = df.rename(columns={
            'open': 'Open', 'high': 'High',
            'low': 'Low', 'close': 'Close',
            'tick_volume': 'Volume'
        })

        mc = mpf.make_marketcolors(
            up='#089981', down='#F23645',
            edge='inherit', wick='inherit', ohlc='i'
        )
        s = mpf.make_mpf_style(
            base_mpf_style='nightclouds', marketcolors=mc,
            gridstyle='', facecolor='#131722', y_on_right=True
        )

        def format_date(x, pos=None):
            if x < 0 or x >= len(df):
                return ''
            return df.index[int(x)].strftime('%H:%M')
        fig, axlist = mpf.plot(
            df,
            type='candle',
            style=s,
            volume=False,
            show_nontrading=False,
            tight_layout=True,
            figratio=(16, 9),
            returnfig=True
        )

        axlist[0].xaxis.set_major_formatter(ticker.FuncFormatter(format_date))
        if len(df) > 1:
            axlist[0].xaxis.set_major_locator(ticker.MaxNLocator(nbins=7))
        else:
            axlist[0].xaxis.set_major_locator(ticker.IndexLocator(base=1, offset=0))

        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        plt.close(fig)
        buf.seek(0)

        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        return image_base64

    finally:
        mt5.shutdown()


def generate_llm_text_for_bar(date_time: datetime, symbol: str = "EURUSD", dataset_json: Optional[str] = None) -> str:
    runtime = _get_runtime(dataset_json)
    base64_image = generate_image(date_time, symbol=symbol)
    final_answer = run_rag_pipeline(
        chart_db=runtime["chart_db"],
        text_db=runtime["text_db"],
        vision_llm=runtime["vision_llm"],
        DATASET_JSON=runtime["dataset_json"],
        base64_image=base64_image,
    )
    return str(final_answer or "").strip()


def generate_llm_cls_for_bar(
    date_time: datetime,
    symbol: str = "EURUSD",
    dataset_json: Optional[str] = None,
) -> Tuple[str, np.ndarray]:
    llm_text = generate_llm_text_for_bar(date_time=date_time, symbol=symbol, dataset_json=dataset_json)
    cls_vec = text_to_cls_embedding(llm_text)
    return llm_text, cls_vec


def use_llm() -> str:
    runtime = _get_runtime(DEFAULT_DATASET_JSON)
    base64_image = generate_image(datetime.now(timezone.utc))
    return run_rag_pipeline(
        runtime["chart_db"],
        runtime["text_db"],
        runtime["vision_llm"],
        runtime["dataset_json"],
        base64_image,
    )


if __name__ == "__main__":
    print(use_llm())
    
