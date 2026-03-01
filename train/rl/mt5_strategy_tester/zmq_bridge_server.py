import os
import sys

import pandas as pd
import zmq
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


RL_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CORE_DIR = os.path.join(RL_ROOT, "core")
TEST_DIR = os.path.join(RL_ROOT, "test")
for _path in (CORE_DIR, TEST_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from backtest_bridge import PPOBridge, calc_auto_lot
from backtest_config import (
    BAR_HISTORY,
    DATASETS_DIR,
    INITIAL_BALANCE,
    MODELS_DIR,
    PIP_VALUE,
    RISK_PERCENT,
    SPREAD_PIPS,
    TEST_DATA_FILE,
    TEST_DATE_FROM,
    TEST_DATE_TO,
    WINDOW_SIZE,
)
from backtest_features import build_feature_columns, build_gate_stats
from backtest_semantic import SemanticRuntime
from env_trading import TradingEnv


HOST = os.getenv("ZMQ_HOST", "0.0.0.0")
PORT = int(os.getenv("ZMQ_PORT", "5555"))
MODEL_PATH = os.path.join(MODELS_DIR, "ppo_trading.zip")
VEC_NORM_PATH = os.path.join(MODELS_DIR, "vec_normalize.pkl")
SYNC_EXTERNAL_LOT = os.getenv("MT5_SYNC_EXTERNAL_LOT", "0").strip().lower() in {"1", "true", "yes"}


def _patch_numpy_bitgenerator_compat():
    """
    Handle cross-version NumPy pickle differences.

    Some vec_normalize.pkl files store bit generators as class objects
    (e.g. <class 'numpy.random._pcg64.PCG64'>) while older NumPy expects
    a short string name ("PCG64"). This patch normalizes the input before
    delegating to NumPy's original constructor.
    """
    try:
        import numpy.random._pickle as np_pickle
    except Exception:
        return

    original_ctor = getattr(np_pickle, "__bit_generator_ctor", None)
    if original_ctor is None:
        return
    if getattr(original_ctor, "__name__", "") == "_compat_bit_generator_ctor":
        return

    tolerant_cache = {}

    def _normalize_bg_name(value):
        if isinstance(value, type):
            return value.__name__
        if isinstance(value, str):
            if "PCG64DXSM" in value:
                return "PCG64DXSM"
            if "PCG64" in value:
                return "PCG64"
            if "MT19937" in value:
                return "MT19937"
            if "Philox" in value:
                return "Philox"
            if "SFC64" in value:
                return "SFC64"
            return value
        return str(value)

    def _build_tolerant_bitgen(base_cls):
        cached = tolerant_cache.get(base_cls)
        if cached is not None:
            return cached

        class _TolerantBitGen(base_cls):
            # Ignore incompatible RNG state payloads from different NumPy versions.
            def __setstate__(self, state):
                try:
                    super().__setstate__(state)
                    return
                except Exception:
                    pass

                if isinstance(state, tuple):
                    for candidate in state:
                        if isinstance(candidate, dict):
                            try:
                                super().__setstate__(candidate)
                                return
                            except Exception:
                                continue

                # Keep default initialized RNG state if legacy payload is incompatible.
                return

        _TolerantBitGen.__name__ = f"Compat{base_cls.__name__}"
        tolerant_cache[base_cls] = _TolerantBitGen
        return _TolerantBitGen

    def _compat_bit_generator_ctor(bit_generator_name="MT19937"):
        normalized = _normalize_bg_name(bit_generator_name)

        base_cls = None
        if isinstance(bit_generator_name, type):
            base_cls = bit_generator_name
        elif hasattr(np_pickle, "BitGenerators") and normalized in np_pickle.BitGenerators:
            base_cls = np_pickle.BitGenerators[normalized]

        if base_cls is None:
            return original_ctor(normalized)

        tolerant_cls = _build_tolerant_bitgen(base_cls)
        return tolerant_cls()

    np_pickle.__bit_generator_ctor = _compat_bit_generator_ctor


class GateStatsProvider:
    def __init__(self):
        self._stats = self._load_from_dataset()
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
        self._history = (
            pd.concat([self._history, incremental], ignore_index=True)
            .sort_values("time")
            .drop_duplicates(subset=["time"], keep="last")
            .reset_index(drop=True)
        )
        if len(self._history) < WINDOW_SIZE:
            return {}
        return build_gate_stats(self._history)


def load_model(feature_columns):
    print(" Loading PPO model and VecNormalize...")
    _patch_numpy_bitgenerator_compat()
    dummy_data = {
        "time": [pd.Timestamp.now()] * 80,
        "open": [1.0] * 80,
        "high": [1.0] * 80,
        "low": [1.0] * 80,
        "close": [1.0] * 80,
    }
    for col in feature_columns:
        if col not in dummy_data:
            dummy_data[col] = [0.0] * 80

    mock_df = pd.DataFrame(dummy_data)
    dummy_env = DummyVecEnv(
        [
            lambda: TradingEnv(
                mock_df,
                lot_size=calc_auto_lot(INITIAL_BALANCE),
            )
        ]
    )

    try:
        vec_norm = VecNormalize.load(VEC_NORM_PATH, dummy_env)
    except Exception as exc:
        raise RuntimeError(
            "Failed to load vec_normalize.pkl. "
            "Check NumPy version compatibility between training and runtime environments."
        ) from exc
    vec_norm.training = False
    vec_norm.norm_reward = False

    model = PPO.load(MODEL_PATH)
    print(" Model loaded successfully")
    return model, vec_norm


def _fallback_times(bar_count: int):
    end = pd.Timestamp.utcnow().floor("h")
    return [end - pd.Timedelta(hours=bar_count - i - 1) for i in range(bar_count)]


def parse_mt5_data(data_str: str):
    try:
        parts = data_str.strip().split(";")
        bars_raw = parts[0].split("|") if parts and parts[0] else []
        if not bars_raw:
            return None, 0, 0.0, 0.0

        fallback_times = _fallback_times(len(bars_raw))
        rows = []
        for i, bar in enumerate(bars_raw):
            values = [v.strip() for v in bar.split(",")]
            if len(values) < 4:
                continue

            if len(values) >= 5:
                ts_token = values[0]
                try:
                    ts = pd.to_datetime(int(float(ts_token)), unit="s")
                except Exception:
                    ts = pd.to_datetime(ts_token, errors="coerce")
                if pd.isna(ts):
                    ts = fallback_times[i]
                offset = 1
            else:
                ts = fallback_times[i]
                offset = 0

            rows.append(
                {
                    "time": ts,
                    "open": float(values[offset]),
                    "high": float(values[offset + 1]),
                    "low": float(values[offset + 2]),
                    "close": float(values[offset + 3]),
                }
            )

        if not rows:
            return None, 0, 0.0, 0.0

        df = pd.DataFrame(rows).sort_values("time").reset_index(drop=True)

        state_str = parts[1] if len(parts) > 1 else ""
        state_values = [v.strip() for v in state_str.split(",")] if state_str else []
        delta_tick = int(float(state_values[5])) if len(state_values) > 5 and state_values[5] else 0
        delta_price = float(state_values[6]) if len(state_values) > 6 and state_values[6] else 0.0
        lot_size = float(state_values[7]) if len(state_values) > 7 and state_values[7] else 0.0

        return df, delta_tick, delta_price, lot_size
    except Exception as exc:
        print(f" Parse error: {exc}")
        return None, 0, 0.0, 0.0


def main():
    semantic_runtime = SemanticRuntime(models_dir=MODELS_DIR)
    feature_columns = build_feature_columns(semantic_runtime.semantic_feature_count)

    model, vec_norm = load_model(feature_columns)
    gate_stats_provider = GateStatsProvider()

    bridge = PPOBridge(
        model=model,
        vec_norm=vec_norm,
        feature_columns=feature_columns,
        semantic_runtime=semantic_runtime,
        semantic_feature_count=semantic_runtime.semantic_feature_count,
        gate_stats=gate_stats_provider.initial(),
    )

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    endpoint = f"tcp://{HOST}:{PORT}"
    socket.bind(endpoint)

    bar_count = 0
    action_names = ["HOLD", "BUY", "SELL", "CLOSE"]

    print("\n" + "=" * 70)
    print(" PPO ZMQ Server (MT5) - Test-Logic Aligned")
    print("=" * 70)
    print(f" ZMQ Endpoint: {endpoint}")
    print(f" Model: {MODEL_PATH}")
    print(f" VecNorm: {VEC_NORM_PATH}")
    print(
        f" Risk={RISK_PERCENT}% | InitialBalance={INITIAL_BALANCE}"
    )
    print(f" Feature count: {len(feature_columns)} (semantic={semantic_runtime.semantic_feature_count})")
    print(f" Gate stats mode: {gate_stats_provider.mode}")
    print(f" External lot sync: {SYNC_EXTERNAL_LOT}")
    print(" Waiting for MT5 Strategy Tester...")
    print("=" * 70 + "\n")

    try:
        while True:
            message = socket.recv()
            data_str = message.decode("utf-8").strip()

            if not data_str:
                socket.send_string("0")
                continue

            df, delta_tick, delta_price, lot_from_ea = parse_mt5_data(data_str)
            if df is None or len(df) < WINDOW_SIZE:
                socket.send_string("0")
                continue

            bridge.gate_stats = gate_stats_provider.update(df)

            if SYNC_EXTERNAL_LOT and lot_from_ea > 0:
                bridge.lot_size = lot_from_ea
                bridge.spread_cost = SPREAD_PIPS * PIP_VALUE * lot_from_ea

            try:
                action, current_price = bridge.process_bar(df, delta_tick, delta_price)
                action = int(action)
            except Exception as exc:
                print(f" process_bar error: {exc}")
                socket.send_string("0")
                continue

            bar_count += 1
            win_rate = (bridge.wins / bridge.trades * 100.0) if bridge.trades > 0 else 0.0

            if bar_count % 50 == 0 or action != 0:
                avg_q, low_q, q_count = semantic_runtime.quality_summary()
                print(
                    f" #{bar_count:5d} | {action_names[action]:5s} | "
                    f"Price: {current_price:.5f} | Pos: {bridge.position:+d} | "
                    f"Eq: {bridge.equity:.2f} | Trades: {bridge.trades} (WR:{win_rate:.1f}%) | "
                    f"Qavg:{avg_q:.3f} low:{low_q}/{q_count}"
                )

            socket.send_string(str(action))

    except KeyboardInterrupt:
        ret = (bridge.equity / INITIAL_BALANCE - 1.0) * 100.0
        avg_q, low_q, q_count = semantic_runtime.quality_summary()
        print("\n" + "=" * 70)
        print(" Server stopped")
        print("=" * 70)
        print(f" Bars processed: {bar_count}")
        print(f" Equity: {bridge.equity:.2f} ({ret:+.2f}%)")
        print(f" Trades: {bridge.trades} | WR: {(bridge.wins / bridge.trades * 100.0) if bridge.trades > 0 else 0.0:.1f}%")
        print(f" Fees: {bridge.total_fees:.2f}")
        print(
            " Semantic quality: "
            f"avg={avg_q:.3f}, low={low_q}/{q_count}, "
            f"matched={semantic_runtime.stats['matched']}, synthetic={semantic_runtime.stats['synthetic']}"
        )
        print("=" * 70)
    finally:
        socket.close(0)
        context.term()


if __name__ == "__main__":
    main()
