from typing import Any, Dict, List, Tuple, Optional

import numpy as np
from langchain_community.vectorstores import Chroma
from .dataset_utils import norm_path, build_text_lookup

from sentence_transformers import CrossEncoder

# ----------------------------
# RERANK MODEL (lazy singleton)
# ----------------------------
_reranker: Optional[CrossEncoder] = None

def get_reranker(model_name: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1") -> CrossEncoder:
    global _reranker
    if _reranker is None or getattr(_reranker, "model_name", None) != model_name:
        _reranker = CrossEncoder(model_name)
        # attach for cheap identity check
        _reranker.model_name = model_name  # type: ignore[attr-defined]
    return _reranker


# ----------------------------
# CHROMA SEARCH (rank only)
# ----------------------------
def chroma_search_rank_only(db: Chroma, query, k: int):
    """
    Return list[(Document, None)] โดยสนใจแค่อันดับ (rank) เพื่อใช้ RRF
    ลดปัญหา relevance score / score negative
    """
    if hasattr(db, "similarity_search_with_score"):
        hits = db.similarity_search_with_score(query, k=k)
        return [(doc, None) for doc, _ in hits]

    docs = db.similarity_search(query, k=k)
    return [(d, None) for d in docs]


# ----------------------------
# RRF FUSION
# ----------------------------
def rrf_fuse_multi(
    hit_lists: List[List[Tuple[Any, Optional[float]]]],
    weights: List[float],
    k0: int = 60,
    final_k: int = 10
):
    assert len(hit_lists) == len(weights)

    # normalize weights (so image can be heavier safely)
    ws = [max(0.0, float(w)) for w in weights]
    s = sum(ws)
    if s <= 1e-9:
        ws = [1.0] * len(ws)
        s = float(len(ws))
    ws = [w / s for w in ws]

    def get_key(doc):
        return norm_path(doc.metadata.get("image") or doc.page_content)

    fused: Dict[str, Dict[str, Any]] = {}

    for src_i, hits in enumerate(hit_lists):
        w = ws[src_i]
        for r, (doc, _) in enumerate(hits, start=1):
            key = get_key(doc)
            fused.setdefault(key, {"doc": doc, "rrf": 0.0, "ranks": {}})
            fused[key]["rrf"] += w * (1.0 / (k0 + r))
            fused[key]["ranks"][src_i] = r

    items = [(k, v["rrf"], v["ranks"]) for k, v in fused.items()]
    items.sort(key=lambda x: x[1], reverse=True)
    return items[:final_k]


# ----------------------------
# RERANK (CrossEncoder on query_text vs candidate 'data')
# ----------------------------
def rerank_with_cross_encoder(
    results: List[Dict[str, Any]],
    query_text: str,
    model_name: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
    top_m: int = 30,
    w_rerank: float = 0.35,
) -> List[Dict[str, Any]]:
    """
    - rerank top_m ด้วย CrossEncoder(query_text, candidate_data)
    - combine กับ rrf เดิม => final_score แล้ว sort
    """
    if not results or not query_text:
        return results

    reranker = get_reranker(model_name)
    cand = results[: max(1, int(top_m))]

    pairs = []
    for r in cand:
        doc = (r.get("data") or "").strip()
        pairs.append((query_text, doc))

    scores = reranker.predict(pairs)
    scores = np.asarray(scores, dtype=np.float32)

    # normalize rerank scores 0..1
    smin, smax = float(scores.min()), float(scores.max())
    if smax - smin < 1e-9:
        rerank_n = np.zeros_like(scores)
    else:
        rerank_n = (scores - smin) / (smax - smin)

    # normalize rrf 0..1 on the same candidate set (สำคัญ เพราะ rrf เป็นเลขเล็ก)
    rrf_vals = np.asarray([float(r.get("rrf", 0.0)) for r in cand], dtype=np.float32)
    rmin, rmax = float(rrf_vals.min()), float(rrf_vals.max())
    if rmax - rmin < 1e-9:
        rrf_n = np.zeros_like(rrf_vals)
    else:
        rrf_n = (rrf_vals - rmin) / (rmax - rmin)

    w_rerank = float(np.clip(w_rerank, 0.0, 1.0))

    for r, s_raw, s_norm, r_norm in zip(cand, scores.tolist(), rerank_n.tolist(), rrf_n.tolist()):
        r["rerank_text_score"] = float(s_raw)
        r["final_score"] = (1.0 - w_rerank) * float(r_norm) + w_rerank * float(s_norm)

    # items beyond top_m keep final_score from normalized rrf-ish fallback
    for r in results[len(cand):]:
        r["final_score"] = float(r.get("rrf", 0.0))

    results.sort(key=lambda x: float(x.get("final_score", 0.0)), reverse=True)
    return results


# ----------------------------
# HYBRID SEARCH (image + text) + optional rerank
# ----------------------------
def hybrid_search_image_query(
    chart_db: Chroma,
    text_db: Chroma,
    dataset_json: str,
    query_image: str,
    auto_text: Optional[str] = None,
    k_img: int = 30,
    k_t: int = 30,
    final_k: int = 10,
    w_img: float = 0.85,     # ✅ ให้ image หนักกว่า
    w_t: float = 0.15,

    # ✅ rerank options
    rerank: bool = True,
    rerank_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
    rerank_top_m: int = 30,
    w_rerank: float = 0.35,
    rrf_pool_k: Optional[int] = None,
):
    """
    IMAGE -> (chart search) + (text search via auto_text) -> RRF merge
    แล้ว (optional) rerank ด้วย CrossEncoder บน query_text vs dataset.data
    """
    img_hits = chroma_search_rank_only(chart_db, query_image, k=int(k_img))

    hit_lists = [img_hits]
    weights = [float(w_img)]

    if auto_text:
        t_hits = chroma_search_rank_only(text_db, auto_text, k=int(k_t))
        hit_lists.append(t_hits)
        weights.append(float(w_t))

    # pool candidates มากกว่า final_k เพื่อให้ rerank มีพื้นที่ทำงาน
    if rrf_pool_k is None:
        rrf_pool_k = max(int(final_k) * 5, int(rerank_top_m), int(final_k))

    fused = rrf_fuse_multi(hit_lists, weights, final_k=int(rrf_pool_k))
    lookup = build_text_lookup(dataset_json)

    results: List[Dict[str, Any]] = []
    for key, rrf_score, ranks in fused:
        results.append({
            "image": key,
            "rrf": float(rrf_score),
            "img_rank": ranks.get(0),
            "t_rank": ranks.get(1),
            "data": lookup.get(key, ""),
        })

    # rerank only when we actually have text query
    if rerank and auto_text:
        results = rerank_with_cross_encoder(
            results,
            query_text=auto_text,
            model_name=rerank_model,
            top_m=int(rerank_top_m),
            w_rerank=float(w_rerank),
        )

    return results[: int(final_k)]
