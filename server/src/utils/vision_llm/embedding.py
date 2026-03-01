"""
Text-to-CLS-embedding module — converts text into a normalized
1024-dim vector using BGE-M3 ColBERT/dense representations.
"""

import os
from dotenv import load_dotenv
import numpy as np
from FlagEmbedding import BGEM3FlagModel

load_dotenv()

os.environ["TOKENIZERS_PARALLELISM"] = "false"

_DEFAULT_EMBED_MODEL = os.getenv("LLM_EMBED_MODEL", "BAAI/bge-m3")
_embedder: BGEM3FlagModel | None = None


def _get_embedder() -> BGEM3FlagModel:
    """Lazy-load the BGE-M3 embedding model (singleton)."""
    global _embedder
    if _embedder is None:
        _embedder = BGEM3FlagModel(_DEFAULT_EMBED_MODEL, use_fp16=True)
    return _embedder


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    """L2-normalize a vector; returns the original if norm ≈ 0."""
    arr = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-12:
        return arr
    return (arr / norm).astype(np.float32)


def text_to_cls_embedding(text: str) -> np.ndarray:
    """Encode *text* into a 1024-dim CLS vector (ColBERT[0] preferred, dense fallback).

    Parameters
    ----------
    text : str
        Input text to embed.

    Returns
    -------
    np.ndarray
        L2-normalized 1024-dim float32 vector.
    """
    embedder = _get_embedder()
    clean = str(text or "").strip()
    out = embedder.encode(
        [clean],
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=True,
    )

    # Prefer ColBERT CLS token
    colbert = out.get("colbert_vecs", [])
    if len(colbert) > 0:
        token_vecs = np.asarray(colbert[0], dtype=np.float32)
        if token_vecs.ndim == 2 and token_vecs.shape[0] > 0:
            return _l2_normalize(token_vecs[0])

    # Fallback to dense
    dense = out.get("dense_vecs", [])
    if len(dense) > 0:
        return _l2_normalize(np.asarray(dense[0], dtype=np.float32))

    return np.zeros(1024, dtype=np.float32)
