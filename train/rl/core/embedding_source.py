import os

import joblib
import numpy as np

from chroma_client import ChromaDBClient


EMBED_SOURCE_MODE = "cls"


def normalize_source_mode(_source_mode: str) -> str:
    return EMBED_SOURCE_MODE


def _cache_path(models_dir: str) -> str:
    return os.path.join(models_dir, "time_to_embedding_cls.joblib")


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-12:
        return arr
    return (arr / norm).astype(np.float32)


def _load_cached_map(cache_file: str):
    if not os.path.exists(cache_file):
        return None
    try:
        payload = joblib.load(cache_file)
    except Exception:
        return None
    if isinstance(payload, dict):
        if "time_to_vec" in payload and isinstance(payload["time_to_vec"], dict):
            return payload["time_to_vec"]
        return payload
    return None


def _save_cached_map(cache_file: str, time_to_vec: dict):
    payload = {
        "mode": EMBED_SOURCE_MODE,
        "count": int(len(time_to_vec)),
        "time_to_vec": time_to_vec,
    }
    joblib.dump(payload, cache_file)


def _load_chroma_rows(models_dir: str):
    client = ChromaDBClient(persist_path=os.path.join(models_dir, "chroma_db"))
    docs = client.collection.get(include=["metadatas", "embeddings", "documents"])

    metadatas = docs.get("metadatas")
    if metadatas is None:
        metadatas = []
    embeddings = docs.get("embeddings")
    if embeddings is None:
        embeddings = []
    documents = docs.get("documents")
    if documents is None:
        documents = []

    rows = []
    for i, (meta, emb) in enumerate(zip(metadatas, embeddings)):
        if not meta or not meta.get("symbol_datetime"):
            continue
        ts = str(meta["symbol_datetime"])
        dense = np.asarray(emb, dtype=np.float32)
        text = ""
        if i < len(documents) and documents[i]:
            text = str(documents[i])
        if not text and isinstance(meta, dict):
            for key in ("content", "text", "document", "description", "summary"):
                value = meta.get(key)
                if value:
                    text = str(value)
                    break
        rows.append((ts, dense, text))
    return rows


def _build_cls_map(rows):
    from FlagEmbedding import BGEM3FlagModel

    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
    time_to_vec = {}
    total = len(rows)

    for i, (key, dense_vec, text) in enumerate(rows):
        if i > 0 and i % 200 == 0:
            print(f" Building cls embeddings: {i}/{total}")

        out = model.encode(
            [text if text else ""],
            return_dense=False,
            return_sparse=False,
            return_colbert_vecs=True,
        )
        batch_colbert = out.get("colbert_vecs", [])
        dense_vec = np.asarray(dense_vec, dtype=np.float32)
        if len(batch_colbert) > 0:
            col = np.asarray(batch_colbert[0], dtype=np.float32)
            if col.ndim == 2 and col.shape[0] > 0:
                cls_vec = col[0].astype(np.float32)
            else:
                cls_vec = dense_vec
        else:
            cls_vec = dense_vec

        time_to_vec[key] = _l2_normalize(cls_vec).astype(np.float32)
    return time_to_vec


def load_time_to_embedding_map(
    models_dir: str,
    source_mode: str = EMBED_SOURCE_MODE,
    force_rebuild: bool = False,
    batch_size: int = 16,
) -> tuple[dict, str]:
    _ = source_mode
    _ = batch_size
    cache_file = _cache_path(models_dir)
    if not force_rebuild:
        cached = _load_cached_map(cache_file)
        if cached is not None and len(cached) > 0:
            return cached, EMBED_SOURCE_MODE

    rows = _load_chroma_rows(models_dir)
    time_to_vec = _build_cls_map(rows)
    _save_cached_map(cache_file, time_to_vec)
    return time_to_vec, EMBED_SOURCE_MODE
