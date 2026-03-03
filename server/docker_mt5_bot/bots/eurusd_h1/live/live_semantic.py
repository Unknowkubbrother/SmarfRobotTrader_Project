import sys

import numpy as np

from .config import CORE_DIR, EMBED_QUALITY_MIN, EMBED_SOURCE_MODE

if CORE_DIR not in sys.path:
    sys.path.insert(0, CORE_DIR)

from embedding_projector import load_projector
from embedding_source import load_time_to_embedding_map, normalize_source_mode


class SemanticRuntime:

    def __init__(self, models_dir):
        self.models_dir = models_dir
        self.embed_source_mode = normalize_source_mode(EMBED_SOURCE_MODE)
        self.cache = {}
        self.quality_cache = {}
        self.stats = {
            "matched": 0,
            "missing_real": 0,
            "latent_errors": 0,
            "quality_sum": 0.0,
            "quality_count": 0,
            "quality_low": 0,
        }

        self.global_time_to_vec = {}
        self.embed_projector = None

        self._load_chroma()
        self._load_projector()

    @property
    def semantic_feature_count(self):
        return int(self.embed_projector.latent_dim) if self.embed_projector is not None else 8

    def _load_chroma(self):
        try:
            time_to_vec, resolved_mode = load_time_to_embedding_map(
                models_dir=self.models_dir,
                source_mode=self.embed_source_mode,
                force_rebuild=False,
            )
            self.global_time_to_vec = {key: np.asarray(vec, dtype=np.float32) for key, vec in time_to_vec.items()}
            self.embed_source_mode = resolved_mode
        except Exception as exc:
            print(f" Could not load ChromaDB: {exc}")
            self.global_time_to_vec = {}

    def _load_projector(self):
        try:
            self.embed_projector = load_projector(self.models_dir)
            if self.embed_projector is None:
                print(" Could not load embedding projector! Fallback to zero semantic features.")
        except Exception as exc:
            print(f" Could not load embedding projector! Fallback to zeros. Error: {exc}")
            self.embed_projector = None

    def _embedding_dim(self):
        if self.global_time_to_vec:
            first = next(iter(self.global_time_to_vec.values()))
            return int(np.asarray(first).shape[-1])
        if self.embed_projector is not None:
            return int(self.embed_projector.input_dim)
        return 1024

    def _to_latent_vector(self, raw_vec):
        if self.embed_projector is None:
            return np.zeros(self.semantic_feature_count, dtype=np.float32)
        try:
            transformed = self.embed_projector.transform(np.asarray(raw_vec, dtype=np.float32).reshape(1, -1))[0]
            return transformed.astype(np.float32)
        except Exception:
            self.stats["latent_errors"] += 1
            return np.zeros(self.semantic_feature_count, dtype=np.float32)

    def _record_quality(self, quality):
        q = float(np.clip(quality, 0.0, 1.0))
        self.stats["quality_sum"] += q
        self.stats["quality_count"] += 1
        if q < EMBED_QUALITY_MIN:
            self.stats["quality_low"] += 1

    def resolve_semantic_pca(self, ts_key):
        cached = self.cache.get(ts_key)
        if cached is not None:
            return cached

        raw_vec = self.global_time_to_vec.get(ts_key)
        if raw_vec is None:
            self.stats["missing_real"] += 1
            raise RuntimeError(f"Missing real semantic embedding for ts={ts_key}")

        self.stats["matched"] += 1
        quality = 1.0

        latent_vec = self._to_latent_vector(raw_vec)
        self.cache[ts_key] = latent_vec
        self.quality_cache[ts_key] = float(np.clip(quality, 0.0, 1.0))
        self._record_quality(quality)
        return latent_vec

    def resolve_semantic_latent(self, ts_key):
        return self.resolve_semantic_pca(ts_key)

    def get_quality(self, ts_key):
        return float(self.quality_cache.get(ts_key, 0.0))

    def quality_summary(self):
        count = int(self.stats.get("quality_count", 0))
        avg = float(self.stats.get("quality_sum", 0.0) / count) if count > 0 else 0.0
        low = int(self.stats.get("quality_low", 0))
        return avg, low, count
