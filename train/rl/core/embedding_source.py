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
    tmp_file = f"{cache_file}.tmp"
    joblib.dump(payload, tmp_file)
    os.replace(tmp_file, cache_file)


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


def _load_chroma_time_keys(models_dir: str) -> set[str]:
    client = ChromaDBClient(persist_path=os.path.join(models_dir, "chroma_db"))
    docs = client.collection.get(include=["metadatas"])
    metadatas = docs.get("metadatas")
    if metadatas is None:
        metadatas = []

    keys: set[str] = set()
    for meta in metadatas:
        if not meta or not meta.get("symbol_datetime"):
            continue
        keys.add(str(meta["symbol_datetime"]))
    return keys


def _build_cls_map(rows, batch_size: int = 16):
    from FlagEmbedding import BGEM3FlagModel

    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
    time_to_vec = {}
    total = len(rows)
    batch_size = max(1, int(batch_size))

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        chunk = rows[start:end]
        texts = [(text if text else "") for _, _, text in chunk]

        out = model.encode(
            texts,
            return_dense=False,
            return_sparse=False,
            return_colbert_vecs=True,
        )
        batch_colbert = out.get("colbert_vecs", [])

        for idx, (key, dense_vec, _text) in enumerate(chunk):
            dense_arr = np.asarray(dense_vec, dtype=np.float32)
            cls_vec = dense_arr

            if idx < len(batch_colbert):
                col = np.asarray(batch_colbert[idx], dtype=np.float32)
                if col.ndim == 2 and col.shape[0] > 0:
                    cls_vec = col[0].astype(np.float32)

            time_to_vec[key] = _l2_normalize(cls_vec).astype(np.float32)

        if end % 200 == 0 or end == total:
            print(f" Building cls embeddings: {end}/{total}")
    return time_to_vec


def _build_dense_map(rows):
    time_to_vec = {}
    for key, dense_vec, _text in rows:
        dense_arr = np.asarray(dense_vec, dtype=np.float32)
        time_to_vec[key] = _l2_normalize(dense_arr).astype(np.float32)
    return time_to_vec


def load_time_to_embedding_map(
    models_dir: str,
    source_mode: str = EMBED_SOURCE_MODE,
    force_rebuild: bool = False,
    batch_size: int = 16,
    ensure_complete: bool | None = None,
) -> tuple[dict, str]:
    _ = source_mode
    _ = batch_size
    if ensure_complete is None:
        ensure_complete = os.getenv("EMBED_CACHE_ENSURE_COMPLETE", "1").strip().lower() in {"1", "true", "yes"}

    cache_file = _cache_path(models_dir)
    if not force_rebuild:
        cached = _load_cached_map(cache_file)
        if cached is not None and len(cached) > 0:
            if not ensure_complete:
                return cached, EMBED_SOURCE_MODE

            chroma_keys = _load_chroma_time_keys(models_dir)
            if not chroma_keys:
                return cached, EMBED_SOURCE_MODE

            missing_keys = chroma_keys.difference(cached.keys())
            if not missing_keys:
                return cached, EMBED_SOURCE_MODE

            print(
                f" Embedding cache incomplete: cached={len(cached)} "
                f"chroma={len(chroma_keys)} missing={len(missing_keys)}"
            )
            backfill_mode = os.getenv("EMBED_CACHE_BACKFILL_MODE", "dense").strip().lower()
            if backfill_mode not in {"dense", "cls"}:
                backfill_mode = "dense"
            print(f" Backfilling missing embeddings from ChromaDB... mode={backfill_mode}")
            missing_set = set(missing_keys)
            rows = _load_chroma_rows(models_dir)
            rows_missing = [row for row in rows if row[0] in missing_set]
            if rows_missing:
                if backfill_mode == "cls":
                    filled = _build_cls_map(rows_missing, batch_size=batch_size)
                else:
                    filled = _build_dense_map(rows_missing)
                merged = dict(cached)
                merged.update(filled)
                _save_cached_map(cache_file, merged)
                print(f" Embedding cache updated: {len(merged)}/{len(chroma_keys)}")
                return merged, EMBED_SOURCE_MODE

            print(" Warning: no rows matched missing keys; using cached embeddings as-is.")
            return cached, EMBED_SOURCE_MODE

    rows = _load_chroma_rows(models_dir)
    time_to_vec = _build_cls_map(rows, batch_size=batch_size)
    _save_cached_map(cache_file, time_to_vec)
    return time_to_vec, EMBED_SOURCE_MODE
