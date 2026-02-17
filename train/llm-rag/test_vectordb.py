import os
from datetime import date
from dotenv import load_dotenv
from supabase import create_client
from FlagEmbedding import BGEM3FlagModel

load_dotenv()

# SUPABASE_URL = os.getenv("SUPABASE_URL")
# SUPABASE_KEY = os.getenv("SUPABASE_KEY")
url = os.environ["SUPABASE_URL"]
key = os.environ["SUPABASE_ANON_KEY"]
supabase = create_client(url, key)

model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

def embed_dense(texts: list[str]) -> list[list[float]]:
    out = model.encode(
        texts,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )
    dense = out["dense_vecs"]
    return [v.tolist() for v in dense]

def insert_document(symbol: str, symbol_date: date, content: str):
    emb = embed_dense([content])[0]

    payload = {
        "symbol": symbol,
        "symbol_date": symbol_date.isoformat(),
        "content": content,
        "embedding": emb,
    }

    res = supabase.table("documents").insert(payload).execute()
    return res.data

def search_similar(query_text: str, match_threshold: float = 0.78, match_count: int = 1):
    q_emb = embed_dense([query_text])[0]

    res = supabase.rpc(
        "match_documents",
        {
            "query_embedding": q_emb,
            "match_threshold": match_threshold,
            "match_count": match_count,
        },
    ).execute()

    return res.data

if __name__ == "__main__":
    # insert_document("TESLA", date(2026, 2, 17), "example document text about price action and swing behavior tesla")
    hits = search_similar("Tesla", match_threshold=0.0, match_count=1)

    # print(
    #     supabase.table("documents")
    #     .select("id,symbol,symbol_date,content,embedding")
    #     .ilike("symbol", "TESLA")
    #     .execute()
    #     .data
    # )

    # for h in hits:
        # print(h["id"], h.get("symbol"), h.get("symbol_date"), h.get("content"), h.get("similarity"))
