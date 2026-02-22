import os
import sys

import numpy as np
import pandas as pd

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
RL_ROOT = os.path.dirname(TEST_DIR)
CORE_DIR = os.path.join(RL_ROOT, "core")
if CORE_DIR not in sys.path:
    sys.path.insert(0, CORE_DIR)

from backtest_config import BASE_COLUMNS
from semantic_embedding import compute_regime_frame


def build_feature_columns(semantic_feature_count):
    cols = BASE_COLUMNS.copy()
    for i in range(int(semantic_feature_count)):
        cols.append(f"sem_pca_{i + 1}")
    return cols


def calculate_features(df, semantic_runtime, semantic_feature_count, delta_tick=0, delta_price=0.0):
    df = df.copy()
    regime_frame = compute_regime_frame(df)
    ts_keys = df["time"].dt.strftime("%Y-%m-%d %H:%M:%S").tolist()
    sem_pca = np.vstack(
        [semantic_runtime.resolve_semantic_pca(ts, regime_frame.iloc[i]) for i, ts in enumerate(ts_keys)]
    ).astype(np.float32)
    sem_cols = [f"sem_pca_{i + 1}" for i in range(int(semantic_feature_count))]
    sem_df = pd.DataFrame(sem_pca, columns=sem_cols, index=df.index)
    df = pd.concat([df, sem_df], axis=1)

    df["return"] = df["close"].pct_change().fillna(0)
    df["range"] = (df["high"] - df["low"]) / df["close"]

    full_range = df["high"] - df["low"]
    df["body_ratio"] = np.where(full_range > 0, abs(df["close"] - df["open"]) / full_range, 0)
    df["momentum"] = df["return"].rolling(window=5).sum().fillna(0)

    df["delta_tick"] = 0
    df["delta_price"] = 0.0
    df.loc[df.index[-1], "delta_tick"] = delta_tick
    df.loc[df.index[-1], "delta_price"] = delta_price

    sma20 = df["close"].rolling(20).mean()
    sma50 = df["close"].rolling(50).mean()
    df["sma_cross"] = np.where(sma20 > sma50, 1, np.where(sma20 < sma50, -1, 0))
    df["sma_cross"] = df["sma_cross"].fillna(0)

    delta_c = df["close"].diff()
    gain = delta_c.clip(lower=0).rolling(14).mean()
    loss = (-delta_c.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    df["rsi_norm"] = ((rsi - 50) / 50).fillna(0)

    tr = np.maximum(
        df["high"] - df["low"],
        np.maximum(
            abs(df["high"] - df["close"].shift(1)),
            abs(df["low"] - df["close"].shift(1)),
        ),
    )
    df["atr_norm"] = (tr.rolling(14).mean() / df["close"]).fillna(0)

    df["trend"] = (sma20.pct_change(5) * 100).fillna(0)
    df["trend"] = df["trend"].clip(-2, 2)

    tr_adx = np.maximum(
        df["high"] - df["low"],
        np.maximum(
            abs(df["high"] - df["close"].shift(1)),
            abs(df["low"] - df["close"].shift(1)),
        ),
    )
    plus_dm = np.where(
        (df["high"] - df["high"].shift(1)) > (df["low"].shift(1) - df["low"]),
        np.maximum(df["high"] - df["high"].shift(1), 0),
        0,
    )
    minus_dm = np.where(
        (df["low"].shift(1) - df["low"]) > (df["high"] - df["high"].shift(1)),
        np.maximum(df["low"].shift(1) - df["low"], 0),
        0,
    )
    atr14_adx = pd.Series(tr_adx).rolling(14).mean()
    plus_di = 100 * pd.Series(plus_dm).rolling(14).mean() / (atr14_adx + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).rolling(14).mean() / (atr14_adx + 1e-10)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx_raw = dx.rolling(14).mean()
    df["adx"] = ((adx_raw - 25) / 25).fillna(0).clip(-1, 1)

    return df


def _safe_quantile(values, q, default):
    arr = np.asarray(values, dtype=np.float32)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float(default)
    return float(np.quantile(arr, q))


def build_gate_stats(df):
    close = df["close"]
    high = df["high"]
    low = df["low"]

    tr = np.maximum(
        high - low,
        np.maximum(
            abs(high - close.shift(1)),
            abs(low - close.shift(1)),
        ),
    )
    atr_vals = (tr.rolling(14).mean() / close).fillna(0.0).values

    sma20 = close.rolling(20).mean()
    trend_vals = np.abs((sma20.pct_change(5) * 100).fillna(0.0).clip(-2, 2).values)

    plus_dm = np.where(
        (high - high.shift(1)) > (low.shift(1) - low),
        np.maximum(high - high.shift(1), 0),
        0,
    )
    minus_dm = np.where(
        (low.shift(1) - low) > (high - high.shift(1)),
        np.maximum(low.shift(1) - low, 0),
        0,
    )
    atr14_adx = pd.Series(tr).rolling(14).mean()
    plus_di = 100 * pd.Series(plus_dm).rolling(14).mean() / (atr14_adx + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).rolling(14).mean() / (atr14_adx + 1e-10)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx_vals = ((dx.rolling(14).mean() - 25) / 25).fillna(0.0).clip(-1, 1).values

    return {
        "atr_high": _safe_quantile(atr_vals, 0.70, 0.0015),
        "atr_extreme": _safe_quantile(atr_vals, 0.90, 0.0025),
        "trend_flat": _safe_quantile(trend_vals, 0.35, 0.08),
        "trend_strong": _safe_quantile(trend_vals, 0.75, 0.25),
        "adx_flat": _safe_quantile(adx_vals, 0.40, -0.20),
        "adx_strong": _safe_quantile(adx_vals, 0.75, 0.20),
    }
