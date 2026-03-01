import os
import sys

import pandas as pd

from .config import (
    BAR_HISTORY,
    DATASETS_DIR,
    TEST_DATA_FILE,
    TEST_DATE_FROM,
    TEST_DATE_TO,
    TEST_DIR,
    WINDOW_SIZE,
)

if TEST_DIR not in sys.path:
    sys.path.insert(0, TEST_DIR)

from backtest_features import build_gate_stats


class GateStatsProvider:
    def __init__(self, mode: str = "dataset"):
        self._mode = (mode or "dataset").strip().lower()
        if self._mode not in {"dataset", "dynamic"}:
            self._mode = "dataset"
        self._stats = self._load_from_dataset() if self._mode == "dataset" else None
        self._history = pd.DataFrame(columns=["time", "open", "high", "low", "close"])

    @property
    def mode(self) -> str:
        return "dataset" if self._stats is not None else "dynamic"

    def initial(self):
        return self._stats if self._stats is not None else {}

    def _load_from_dataset(self):
        data_path = os.path.join(DATASETS_DIR, TEST_DATA_FILE)
        if not os.path.exists(data_path):
            print(f" Gate stats dataset not found: {data_path}")
            return None

        try:
            df = pd.read_csv(data_path)
            if "time" not in df.columns:
                print(" Gate stats dataset has no 'time' column; fallback to dynamic.")
                return None

            df["time"] = pd.to_datetime(df["time"])
            if TEST_DATE_FROM:
                df = df[df["time"] >= pd.to_datetime(TEST_DATE_FROM)]
            if TEST_DATE_TO:
                df = df[df["time"] <= pd.to_datetime(TEST_DATE_TO)]
            if "has_delta" in df.columns:
                df = df[df["has_delta"] == 1]

            df = (
                df[["time", "open", "high", "low", "close"]]
                .sort_values("time")
                .drop_duplicates(subset=["time"], keep="last")
                .reset_index(drop=True)
            )
            if len(df) <= max(BAR_HISTORY, WINDOW_SIZE):
                print(
                    f" Gate stats dataset too short ({len(df)} rows); "
                    "fallback to dynamic window-based stats."
                )
                return None

            stats = build_gate_stats(df)
            print(
                " Gate stats loaded from dataset "
                f"({os.path.basename(data_path)}, rows={len(df)})"
            )
            return stats
        except Exception as exc:
            print(f" Gate stats load failed ({exc}); fallback to dynamic.")
            return None

    def update(self, window_df: pd.DataFrame):
        if self._stats is not None:
            return self._stats

        incremental = (
            window_df[["time", "open", "high", "low", "close"]]
            .copy()
            .sort_values("time")
            .drop_duplicates(subset=["time"], keep="last")
        )
        if self._history.empty:
            self._history = incremental.reset_index(drop=True)
        else:
            self._history = (
                pd.concat([self._history, incremental], ignore_index=True)
                .sort_values("time")
                .drop_duplicates(subset=["time"], keep="last")
                .reset_index(drop=True)
            )
        if len(self._history) < WINDOW_SIZE:
            return {}
        return build_gate_stats(self._history)

    def to_records(self, max_rows: int = 800):
        if self._history.empty:
            return []
        tail = self._history.tail(max_rows).copy()
        tail["time"] = pd.to_datetime(tail["time"]).dt.strftime("%Y-%m-%d %H:%M:%S")
        return tail.to_dict(orient="records")

    def load_records(self, rows):
        if not rows:
            self._history = pd.DataFrame(columns=["time", "open", "high", "low", "close"])
            return
        df = pd.DataFrame(rows)
        expected = ["time", "open", "high", "low", "close"]
        missing = [c for c in expected if c not in df.columns]
        if missing:
            self._history = pd.DataFrame(columns=expected)
            return
        df = df[expected].copy()
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df = df.dropna(subset=["time"]).sort_values("time").drop_duplicates(subset=["time"], keep="last").reset_index(drop=True)
        self._history = df
