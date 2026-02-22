import os
import sys

import joblib
import numpy as np

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
RL_ROOT = os.path.dirname(TEST_DIR)
CORE_DIR = os.path.join(RL_ROOT, "core")
if CORE_DIR not in sys.path:
    sys.path.insert(0, CORE_DIR)

from chroma_client import ChromaDBClient
from embedding_projector import load_projector
from semantic_embedding import make_regime_key, resolve_from_semantic_map_with_quality
from backtest_config import EMBED_TEST_MODE, EMBED_QUALITY_MIN


class SemanticRuntime:

    def __init__(self, models_dir):
        self.models_dir = models_dir
        self.embed_test_mode = str(EMBED_TEST_MODE).strip().lower()
        if self.embed_test_mode not in {"knn_map", "regime_map", "hour_mean", "global_mean", "zero"}:
            self.embed_test_mode = "knn_map"
        self.cache = {}
        self.quality_cache = {}
        self.stats = {
            "matched": 0,
            "synthetic": 0,
            "knn_fallback": 0,
            "regime_fallback": 0,
            "hour_fallback": 0,
            "global_fallback": 0,
            "zero_fallback": 0,
            "latent_errors": 0,
            "quality_sum": 0.0,
            "quality_count": 0,
            "quality_low": 0,
        }

        self.global_time_to_vec = {}
        self.embed_projector = None
        self.semantic_map = None

        self._load_chroma()
        self._load_projector()
        self._load_semantic_map()

    @property
    def semantic_feature_count(self):
        return int(self.embed_projector.latent_dim) if self.embed_projector is not None else 8

    def _load_chroma(self):
        try:
            print(" Loading embeddings from ChromaDB...")
            client = ChromaDBClient(persist_path=os.path.join(self.models_dir, "chroma_db"))
            docs = client.collection.get(include=["metadatas", "embeddings"])
            for meta, vec in zip(docs.get("metadatas", []), docs.get("embeddings", [])):
                if not meta or not meta.get("symbol_datetime"):
                    continue
                key = meta["symbol_datetime"]
                self.global_time_to_vec[key] = np.asarray(vec, dtype=np.float32)
            print(f" Loaded {len(self.global_time_to_vec)} embeddings globally.")
        except Exception as e:
            print(f" Could not load ChromaDB: {e}")
            self.global_time_to_vec = {}

    def _load_projector(self):
        try:
            self.embed_projector = load_projector(self.models_dir)
            if self.embed_projector is not None:
                print(
                    " Loaded embedding projector successfully. "
                    f"mode={self.embed_projector.mode}, latent_dim={self.embed_projector.latent_dim}"
                )
            else:
                print(" Could not load embedding projector! Fallback to zero semantic features.")
        except Exception as e:
            print(f" Could not load embedding projector! Fallback to zeros. Error: {e}")
            self.embed_projector = None

    def _load_semantic_map(self):
        try:
            self.semantic_map = joblib.load(os.path.join(self.models_dir, "semantic_map.joblib"))
            print(" Loaded semantic_map.joblib for deterministic embedding fallback.")
        except Exception as e:
            print(f" Could not load semantic_map.joblib: {e}")
            self.semantic_map = None

    def _to_latent_vector(self, raw_vec):
        if self.embed_projector is None:
            return np.zeros(self.semantic_feature_count, dtype=np.float32)
        try:
            transformed = self.embed_projector.transform(np.asarray(raw_vec, dtype=np.float32).reshape(1, -1))[0]
            return transformed.astype(np.float32)
        except Exception:
            self.stats["latent_errors"] += 1
            return np.zeros(self.semantic_feature_count, dtype=np.float32)

    def _embedding_dim(self):
        if self.semantic_map is not None:
            return int(self.semantic_map.get("embedding_dim", 1024))
        if self.global_time_to_vec:
            first = next(iter(self.global_time_to_vec.values()))
            return int(np.asarray(first).shape[-1])
        return 1024

    def _zero_raw(self):
        return np.zeros(self._embedding_dim(), dtype=np.float32)

    def _global_mean_raw(self):
        if self.semantic_map is None:
            return self._zero_raw()
        return np.asarray(self.semantic_map.get("global_mean", self._zero_raw()), dtype=np.float32)

    def _hour_mean_raw(self, row_regime):
        if self.semantic_map is None:
            return self._zero_raw()
        hour = int(float(row_regime.get("hour", 0)))
        hour_centroids = self.semantic_map.get("hour_centroids", {})
        vec = hour_centroids.get(hour)
        if vec is None:
            return self._global_mean_raw()
        return np.asarray(vec, dtype=np.float32)

    def _regime_mean_raw(self, row_regime):
        if self.semantic_map is None:
            return self._zero_raw()
        key = make_regime_key(
            trend=int(row_regime.get("trend", 0)),
            momentum=float(row_regime.get("momentum", 0.0)),
            rsi=float(row_regime.get("rsi", 50.0)),
            vol=float(row_regime.get("vol", 0.0)),
            hour=int(float(row_regime.get("hour", 0.0))),
            vol_q1=float(self.semantic_map.get("vol_q1", 0.0)),
            vol_q2=float(self.semantic_map.get("vol_q2", 0.0)),
            mom_thr=float(self.semantic_map.get("mom_thr", 0.0)),
        )
        regime_centroids = self.semantic_map.get("regime_centroids", {})
        vec = regime_centroids.get(key)
        if vec is None:
            return self._global_mean_raw()
        return np.asarray(vec, dtype=np.float32)

    def _resolve_synthetic_raw(self, row_regime):
        if self.embed_test_mode == "knn_map":
            if self.semantic_map is not None:
                self.stats["knn_fallback"] += 1
                return resolve_from_semantic_map_with_quality(row_regime, self.semantic_map)
            self.stats["zero_fallback"] += 1
            return self._zero_raw(), 0.0

        if self.embed_test_mode == "regime_map":
            if self.semantic_map is not None:
                self.stats["regime_fallback"] += 1
                return self._regime_mean_raw(row_regime), 0.45
            self.stats["zero_fallback"] += 1
            return self._zero_raw(), 0.0

        if self.embed_test_mode == "hour_mean":
            if self.semantic_map is not None:
                self.stats["hour_fallback"] += 1
                return self._hour_mean_raw(row_regime), 0.30
            self.stats["zero_fallback"] += 1
            return self._zero_raw(), 0.0

        if self.embed_test_mode == "global_mean":
            if self.semantic_map is not None:
                self.stats["global_fallback"] += 1
                return self._global_mean_raw(), 0.20
            self.stats["zero_fallback"] += 1
            return self._zero_raw(), 0.0

        self.stats["zero_fallback"] += 1
        return self._zero_raw(), 0.0

    def _record_quality(self, quality):
        q = float(np.clip(quality, 0.0, 1.0))
        self.stats["quality_sum"] += q
        self.stats["quality_count"] += 1
        if q < EMBED_QUALITY_MIN:
            self.stats["quality_low"] += 1

    def resolve_semantic_pca(self, ts_key, row_regime):
        cached = self.cache.get(ts_key)
        if cached is not None:
            return cached

        raw_vec = self.global_time_to_vec.get(ts_key)
        if raw_vec is not None:
            self.stats["matched"] += 1
            quality = 1.0
        else:
            self.stats["synthetic"] += 1
            raw_vec, quality = self._resolve_synthetic_raw(row_regime)

        latent_vec = self._to_latent_vector(raw_vec)
        self.cache[ts_key] = latent_vec
        self.quality_cache[ts_key] = float(np.clip(quality, 0.0, 1.0))
        self._record_quality(quality)
        return latent_vec

    def resolve_semantic_latent(self, ts_key, row_regime):
        return self.resolve_semantic_pca(ts_key, row_regime)

    def get_quality(self, ts_key):
        return float(self.quality_cache.get(ts_key, 0.0))

    def quality_summary(self):
        count = int(self.stats.get("quality_count", 0))
        avg = float(self.stats.get("quality_sum", 0.0) / count) if count > 0 else 0.0
        low = int(self.stats.get("quality_low", 0))
        return avg, low, count
