import pandas as pd

from .config import (
    WINDOW_SIZE,
)
from .live_features import build_gate_stats


class GateStatsProvider:
    def __init__(self):
        self._history = pd.DataFrame(columns=["time", "open", "high", "low", "close"])

    def initial(self):
        return {}

    def update(self, window_df: pd.DataFrame):
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
