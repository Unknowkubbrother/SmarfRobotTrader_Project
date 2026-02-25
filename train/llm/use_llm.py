import os
import base64
from io import BytesIO
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage

from datetime import datetime, timedelta

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

UTC_TO_BROKER = 7
BROKER_TO_TARGET = 7
TOTAL_OFFSET = UTC_TO_BROKER + BROKER_TO_TARGET

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

    print_results("IMAGE → HYBRID (Chart + Text via auto_text)", results)

    rag_context = build_rag_context(results, max_chars=1500)

    formatted_prompt = RAG_TEMPLATE.format(context=rag_context)
    
    final_answer = vision_llm.invoke(formatted_prompt, base64_image)

    return final_answer    


def generate_image(date_time: datetime, symbol: str = "EURUSD") -> str:
    mt5 = MetaTrader5(host="localhost", port=8001)
    mt5.initialize()

    try:
        start_input = date_time.replace(minute=0, second=0, microsecond=0)
        end_input = start_input + timedelta(hours=1)

        broker_start = start_input - timedelta(hours=BROKER_TO_TARGET)
        broker_end = end_input - timedelta(hours=BROKER_TO_TARGET)

        rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M1, broker_start, broker_end)

        if rates is None or len(rates) == 0:
            raise ValueError(f"❌ No data found for {symbol} at {date_time}")

        # 5. Prepare DataFrame
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df['time'] = df['time'] + timedelta(hours=TOTAL_OFFSET)
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


    

def use_llm() -> str:
    DATASET_JSON = "dataset.json"

    chart_db = upsert_image_dataset(DATASET_JSON)
    text_db = upsert_text_dataset(DATASET_JSON)

    vision_llm = VisionLLMClient()

    base64_image = generate_image(datetime.now())

    print(base64_image)

    return run_rag_pipeline(chart_db, text_db, vision_llm, DATASET_JSON, base64_image)


if __name__ == "__main__":
    print(use_llm())
    
