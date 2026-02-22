import numpy as np
import pandas as pd


def regime_to_knn_vector(row_regime: pd.Series) -> np.ndarray:
    hour = float(row_regime.get("hour", 0.0))
    hour_angle = (2.0 * np.pi * hour) / 24.0
    return np.array(
        [
            float(row_regime.get("ret", 0.0)),
            float(row_regime.get("momentum", 0.0)),
            float(row_regime.get("vol", 0.0)),
            float(row_regime.get("trend", 0.0)),
            (float(row_regime.get("rsi", 50.0)) - 50.0) / 50.0,
            np.sin(hour_angle),
            np.cos(hour_angle),
        ],
        dtype=np.float32,
    )


def compute_regime_frame(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]
    ret = close.pct_change().fillna(0.0)
    momentum = ret.rolling(5).sum().fillna(0.0)
    vol = ((df["high"] - df["low"]) / close).replace([np.inf, -np.inf], 0.0).fillna(0.0)

    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    trend = np.where(sma20 > sma50, 1, np.where(sma20 < sma50, -1, 0))

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    rsi = (100 - (100 / (1 + rs))).fillna(50.0)

    if "time" in df.columns:
        hour = pd.to_datetime(df["time"]).dt.hour.fillna(0).astype(int)
    else:
        hour = pd.Series(np.zeros(len(df), dtype=int), index=df.index)

    return pd.DataFrame(
        {
            "ret": ret.astype(np.float32),
            "momentum": momentum.astype(np.float32),
            "vol": vol.astype(np.float32),
            "trend": trend.astype(np.int8),
            "rsi": rsi.astype(np.float32),
            "hour": hour.astype(np.int8),
        },
        index=df.index,
    )


def _safe_quantiles(values: np.ndarray, qs):
    if len(values) == 0:
        return [0.0 for _ in qs]
    return [float(x) for x in np.quantile(values, qs)]


def build_semantic_map(df: pd.DataFrame, raw_embedding_col: str = "raw_embedding") -> dict:
    valid_mask = df[raw_embedding_col].notna()
    if valid_mask.sum() == 0:
        return {
            "embedding_dim": 1024,
            "vol_q1": 0.0,
            "vol_q2": 0.0,
            "mom_thr": 0.0,
            "hour_centroids": {},
            "regime_centroids": {},
            "global_mean": np.zeros(1024, dtype=np.float32),
            "knn_mean": np.zeros(7, dtype=np.float32),
            "knn_std": np.ones(7, dtype=np.float32),
            "knn_features": np.zeros((0, 7), dtype=np.float32),
            "knn_embeddings": np.zeros((0, 1024), dtype=np.float32),
            "knn_default_k": 24,
        }

    regime = compute_regime_frame(df)
    valid_idx = np.where(valid_mask.values)[0]
    valid_embeddings = np.vstack(df.loc[valid_mask, raw_embedding_col].values).astype(np.float32)
    dim = int(valid_embeddings.shape[1])

    vol_vals = regime.loc[valid_mask, "vol"].values.astype(np.float32)
    mom_vals = np.abs(regime.loc[valid_mask, "momentum"].values.astype(np.float32))
    vol_q1, vol_q2 = _safe_quantiles(vol_vals, [0.33, 0.66])
    mom_thr = float(_safe_quantiles(mom_vals, [0.35])[0])

    global_mean = valid_embeddings.mean(axis=0).astype(np.float32)

    hour_centroids = {}
    hours = regime.loc[valid_mask, "hour"].values.astype(int)
    for h in np.unique(hours):
        h_mask = hours == h
        hour_centroids[int(h)] = valid_embeddings[h_mask].mean(axis=0).astype(np.float32)

    regime_centroids = {}
    for pos, row_idx in enumerate(valid_idx):
        r = regime.iloc[row_idx]
        key = make_regime_key(
            trend=int(r["trend"]),
            momentum=float(r["momentum"]),
            rsi=float(r["rsi"]),
            vol=float(r["vol"]),
            hour=int(r["hour"]),
            vol_q1=vol_q1,
            vol_q2=vol_q2,
            mom_thr=mom_thr,
        )
        regime_centroids.setdefault(key, []).append(valid_embeddings[pos])

    regime_centroids = {
        k: np.vstack(v).mean(axis=0).astype(np.float32)
        for k, v in regime_centroids.items()
    }

    knn_raw_features = np.vstack([regime_to_knn_vector(regime.iloc[i]) for i in valid_idx]).astype(np.float32)
    knn_mean = knn_raw_features.mean(axis=0).astype(np.float32)
    knn_std = (knn_raw_features.std(axis=0) + 1e-6).astype(np.float32)
    knn_features = ((knn_raw_features - knn_mean) / knn_std).astype(np.float32)

    return {
        "embedding_dim": dim,
        "vol_q1": vol_q1,
        "vol_q2": vol_q2,
        "mom_thr": mom_thr,
        "hour_centroids": hour_centroids,
        "regime_centroids": regime_centroids,
        "global_mean": global_mean,
        "knn_mean": knn_mean,
        "knn_std": knn_std,
        "knn_features": knn_features,
        "knn_embeddings": valid_embeddings,
        "knn_default_k": 24,
    }


def make_regime_key(
    trend: int,
    momentum: float,
    rsi: float,
    vol: float,
    hour: int,
    vol_q1: float,
    vol_q2: float,
    mom_thr: float,
) -> str:
    if vol < vol_q1:
        vol_bin = 0
    elif vol < vol_q2:
        vol_bin = 1
    else:
        vol_bin = 2

    if momentum > mom_thr:
        mom_bin = 1
    elif momentum < -mom_thr:
        mom_bin = -1
    else:
        mom_bin = 0

    if rsi >= 60:
        rsi_bin = 1
    elif rsi <= 40:
        rsi_bin = -1
    else:
        rsi_bin = 0

    return f"{int(trend)}|{mom_bin}|{rsi_bin}|{vol_bin}|{int(hour)}"


def resolve_from_semantic_map(
    row_regime: pd.Series,
    semantic_map: dict,
) -> np.ndarray:
    vec, _ = resolve_from_semantic_map_with_quality(row_regime, semantic_map)
    return vec


def resolve_from_semantic_map_with_quality(
    row_regime: pd.Series,
    semantic_map: dict,
) -> tuple[np.ndarray, float]:
    dim = int(semantic_map.get("embedding_dim", 1024))
    zero_vec = np.zeros(dim, dtype=np.float32)
    global_mean = np.asarray(semantic_map.get("global_mean", zero_vec), dtype=np.float32)

    knn_features = semantic_map.get("knn_features")
    knn_embeddings = semantic_map.get("knn_embeddings")
    knn_mean = np.asarray(semantic_map.get("knn_mean", np.zeros(7, dtype=np.float32)), dtype=np.float32)
    knn_std = np.asarray(semantic_map.get("knn_std", np.ones(7, dtype=np.float32)), dtype=np.float32)
    if (
        isinstance(knn_features, np.ndarray)
        and isinstance(knn_embeddings, np.ndarray)
        and knn_features.ndim == 2
        and knn_embeddings.ndim == 2
        and len(knn_features) > 0
        and len(knn_features) == len(knn_embeddings)
    ):
        q = regime_to_knn_vector(row_regime)
        qn = (q - knn_mean) / (knn_std + 1e-6)
        d2 = np.sum((knn_features - qn) ** 2, axis=1)
        k = int(min(int(semantic_map.get("knn_default_k", 24)), len(d2)))
        if k > 0:
            idx = np.argpartition(d2, k - 1)[:k]
            local_d = d2[idx]
            local_e = knn_embeddings[idx]
            w = 1.0 / (np.sqrt(local_d) + 1e-6)
            w = w / (w.sum() + 1e-8)
            weighted_d = float(np.sum(local_d * w))
            quality = float(1.0 / (1.0 + np.sqrt(max(weighted_d, 0.0))))
            return (local_e * w[:, None]).sum(axis=0).astype(np.float32), quality

    return global_mean, 0.0
