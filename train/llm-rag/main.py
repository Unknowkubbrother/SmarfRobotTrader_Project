import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from typing import List, Dict, Any, Optional

# Supabase Text Embedding
from supabase import create_client
from FlagEmbedding import BGEM3FlagModel
from datetime import datetime

from retrieval import (
    upsert_image_dataset,
    upsert_text_dataset,
    hybrid_search_image_query,
)
from retrieval.utils import (
    mask_numbers,
    strip_markdown,
    build_query_text_from_auto,
    print_results,
    build_rag_context,
    encode_image,
)
from prompts import (
    PROMPT_DRAFT_FROM_IMAGE,
    PROMPT_DOMAIN_REWRITE,
    AUTO_TEXT_COMPRESS_NOTE,
    RAG_TEMPLATE
)

load_dotenv()

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class VisionLLMClient:
    def __init__(self):
        self.llm = init_chat_model(
            model="nvidia/nemotron-nano-12b-v2-vl:free",
            model_provider="openai",
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )

    def invoke(self, text: str, image_base64: str) -> str:
        messages = [
            HumanMessage(
                content=[
                    {"type": "text", "text": text},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                    },
                ]
            )
        ]
        response = self.llm.invoke(messages)
        return strip_markdown(response.content)

class TextEmbeddingClient:
    def __init__(self):
        load_dotenv()
        self.url = os.environ["SUPABASE_URL"]
        self.key = os.environ["SUPABASE_ANON_KEY"]
        self.supabase = create_client(self.url, self.key)
        self.model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

    def embed_dense(self, texts: list[str]) -> list[list[float]]:
        out = self.model.encode(
            texts,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        dense = out["dense_vecs"]
        return [v.tolist() for v in dense]

    def insert_document(self, symbol: str, symbol_datetime: datetime, timeframe: str, content: str):
        emb = self.embed_dense([content])[0]

        payload = {
            "symbol": symbol,
            "symbol_datetime": symbol_datetime.strftime("%Y-%m-%d %H:%M:%S"),
            "content": content,
            "embedding": emb,
            "timeframe": timeframe,
        }

        res = self.supabase.table("documents").insert(payload).execute()
        return res.data

    def search_similar(self, query_text: str, match_threshold: float = 0.78, match_count: int = 1):
        q_emb = self.embed_dense([query_text])[0]

        res = self.supabase.rpc(
            "match_documents",
            {
                "query_embedding": q_emb,
                "match_threshold": match_threshold,
                "match_count": match_count,
            },
        ).execute()

        return res.data

    def get_document(self, document_id: int):
        res = self.supabase.table("documents").select("*").eq("id", document_id).execute()
        return res.data


def run_rag_pipeline(chart_db, text_db, vision_llm, DATASET_JSON ,QUERY_IMAGE : str) -> str:
    base64_image = encode_image(QUERY_IMAGE)

    # 3. Draft from image
    draft_clean = vision_llm.invoke(PROMPT_DRAFT_FROM_IMAGE, base64_image)
    print("\n📝 Draft (from image):\n", draft_clean)

    # 4. Search for examples
    ex_docs = text_db.similarity_search(draft_clean, k=6)
    domain_examples = "\n\n---\n\n".join(
        mask_numbers(d.page_content) for d in ex_docs if getattr(d, "page_content", None)
    )
    
    if not domain_examples:
        domain_examples = "ไม่มีตัวอย่าง (fallback): ให้ใช้สำนวนเทคนิคแบบนักเทรดไทย เน้น PA logic และคำค้นที่ชัดเจน"

    # 5. Domain Rewrite
    rewrite_prompt = f"""
DRAFT:
{draft_clean}

DOMAIN EXAMPLES (จาก dataset เดิม):
{domain_examples}

{PROMPT_DOMAIN_REWRITE}

{AUTO_TEXT_COMPRESS_NOTE}
""".strip()

    print(f"Rewrite Prompt Length: {len(rewrite_prompt)}")
    auto_text = vision_llm.invoke(rewrite_prompt, base64_image)
    print("\n📝 Auto-text (domain rewritten):\n", auto_text)

    # 6. Hybrid Search
    query_text = build_query_text_from_auto(auto_text)
    print("\n🔎 Query text (used for text):\n", query_text)

    results = hybrid_search_image_query(
        chart_db=chart_db,
        text_db=text_db,
        dataset_json=DATASET_JSON,
        query_image=QUERY_IMAGE,
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

    # 7. Final Analysis
    rag_context = build_rag_context(results, max_chars=1500)
    print("\n🔎 RAG Context (preview):")
    print(rag_context)

    formatted_prompt = RAG_TEMPLATE.format(context=rag_context)
    
    final_answer = vision_llm.invoke(formatted_prompt, base64_image)

    return final_answer    

if __name__ == "__main__":

    DATASET_JSON = "dataset.json"

    LIST_QUERY_IMAGE = [
        "datasets1/NVDA.png",
        "datasets1/THBUSD.png",
        "datasets1/XAUUSD.png",
    ]

    chart_db = upsert_image_dataset(DATASET_JSON)
    text_db = upsert_text_dataset(DATASET_JSON)

    vision_llm = VisionLLMClient()

    text_embedding = TextEmbeddingClient()

    for query_image in LIST_QUERY_IMAGE:
        final_answer = run_rag_pipeline(chart_db, text_db, vision_llm, DATASET_JSON, query_image)

        symbol = query_image.split("/")[1].split(".")[0]
        timeframe = "H1"
        

        print("\n🧠 Final Analysis:")
        print(final_answer)

        inserted = text_embedding.insert_document(symbol, datetime(2025, 1, 23, 23, 0), timeframe, final_answer)

        if inserted:
            print(f"\n✅ Inserted document {symbol}")
        else:
            print(f"\n❌ Failed to insert document {symbol}")
        