from typing import Any, Dict, List, Tuple, Optional

from langchain_community.vectorstores import Chroma
from .dataset_utils import norm_path, build_text_lookup


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


def rrf_fuse_multi(
    hit_lists: List[List[Tuple[Any, Optional[float]]]],
    weights: List[float],
    k0: int = 60,
    final_k: int = 10
):
    assert len(hit_lists) == len(weights)

    def get_key(doc):
        return norm_path(doc.metadata.get("image") or doc.page_content)

    fused: Dict[str, Dict[str, Any]] = {}

    for src_i, hits in enumerate(hit_lists):
        w = float(weights[src_i])
        for r, (doc, _) in enumerate(hits, start=1):
            key = get_key(doc)
            fused.setdefault(key, {"doc": doc, "rrf": 0.0, "ranks": {}})
            fused[key]["rrf"] += w * (1.0 / (k0 + r))
            fused[key]["ranks"][src_i] = r

    items = [(k, v["rrf"], v["ranks"]) for k, v in fused.items()]
    items.sort(key=lambda x: x[1], reverse=True)
    return items[:final_k]


def hybrid_search_image_query(
    chart_db: Chroma,
    text_db: Chroma,
    dataset_json: str,
    query_image: str,
    auto_text: Optional[str] = None,
    xmodal_image_db: Optional[Chroma] = None,
    k_img=30,
    k_t=30,
    k_x=30,
    final_k=10,
    w_img=0.75,
    w_t=0.18,
    w_x=0.07,
):
    """
    IMAGE -> (chart search) + (text search via auto_text) + (xmodal via auto_text) -> RRF merge
    """
    img_hits = chroma_search_rank_only(chart_db, query_image, k=k_img)

    hit_lists = [img_hits]
    weights = [w_img]

    if auto_text:
        t_hits = chroma_search_rank_only(text_db, auto_text, k=k_t)
        hit_lists.append(t_hits)
        weights.append(w_t)

        if xmodal_image_db is not None:
            x_hits = chroma_search_rank_only(xmodal_image_db, auto_text, k=k_x)
            hit_lists.append(x_hits)
            weights.append(w_x)

    fused = rrf_fuse_multi(hit_lists, weights, final_k=final_k)
    lookup = build_text_lookup(dataset_json)

    results = []
    for key, rrf_score, ranks in fused:
        results.append({
            "image": key,
            "rrf": float(rrf_score),
            "img_rank": ranks.get(0),
            "t_rank": ranks.get(1),
            "x_rank": ranks.get(2),
            "data": lookup.get(key, ""),
        })
    return results
