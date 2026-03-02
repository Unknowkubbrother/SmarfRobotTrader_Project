"""
Text-to-CLS-embedding module — converts text into a normalized
1024-dim vector using BGE-M3 ColBERT/dense representations.
"""

import logging
import os
import threading
from dotenv import load_dotenv
import numpy as np
from FlagEmbedding import BGEM3FlagModel

load_dotenv()

os.environ["TOKENIZERS_PARALLELISM"] = "false"

_DEFAULT_EMBED_MODEL = os.getenv("LLM_EMBED_MODEL", "BAAI/bge-m3")
_embedder: BGEM3FlagModel | None = None
_embedder_lock = threading.Lock()
_encode_lock = threading.Lock()
logger = logging.getLogger(__name__)


def _default_device() -> str:
    # Prefer explicit env first; default to CPU for stability in multi-threaded server mode.
    configured = str(os.getenv("LLM_EMBED_DEVICE", "") or "").strip().lower()
    if configured:
        return configured
    return "cpu"


def _is_recoverable_embed_error(exc: Exception) -> bool:
    text = str(exc or "").lower()
    return (
        "meta tensor" in text
        or "expected scalar type float but found half" in text
        or "to_empty()" in text
    )


def _create_embedder(device: str) -> BGEM3FlagModel:
    use_fp16 = device.startswith("cuda")
    return BGEM3FlagModel(
        _DEFAULT_EMBED_MODEL,
        use_fp16=use_fp16,
        device=device,
    )


def _get_embedder() -> BGEM3FlagModel:
    """Lazy-load the BGE-M3 embedding model (singleton)."""
    global _embedder
    if _embedder is None:
        with _embedder_lock:
            if _embedder is None:
                device = _default_device()
                _embedder = _create_embedder(device=device)
    return _embedder


def _reload_embedder_cpu(reason: str = "") -> BGEM3FlagModel:
    global _embedder
    with _embedder_lock:
        if reason:
            logger.warning("Reloading BGE-M3 embedder on CPU due to runtime error: %s", reason)
        _embedder = _create_embedder(device="cpu")
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
    clean = str(text or "").strip()
    out = None

    for attempt in range(2):
        embedder = _get_embedder()
        try:
            # Guard encode() because BGEM3 internal device transfer is not thread-safe.
            with _encode_lock:
                out = embedder.encode(
                    [clean],
                    batch_size=1,
                    max_length=512,
                    return_dense=True,
                    return_sparse=False,
                    return_colbert_vecs=True,
                )
            break
        except Exception as exc:
            if attempt == 0 and _is_recoverable_embed_error(exc):
                _reload_embedder_cpu(reason=str(exc))
                continue
            raise

    if not isinstance(out, dict):
        return np.zeros(1024, dtype=np.float32)

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
