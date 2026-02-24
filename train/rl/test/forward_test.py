import os
import sys

import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RL_ROOT = os.path.dirname(SCRIPT_DIR)
CORE_DIR = os.path.join(RL_ROOT, "core")
if CORE_DIR not in sys.path:
    sys.path.insert(0, CORE_DIR)

from backtest_bridge import PPOBridge, calc_auto_lot
from backtest_config import (
    ADAPTIVE_GATE,
    BAR_HISTORY,
    DATASETS_DIR,
    EMBED_QUALITY_MIN,
    EMBED_SOURCE_MODE,
    EMBED_TEST_MODE,
    HOLD_EDGE_THRESHOLD,
    INITIAL_BALANCE,
    MIN_ACTION_MARGIN,
    MODELS_DIR,
    OPEN_EDGE_THRESHOLD,
    OPEN_PROB_THRESHOLD,
    OUTPUT_DIR,
    RISK_PERCENT,
    SL_PIPS,
    TEST_DATA_FILE,
    TEST_DATE_FROM,
    TEST_DATE_TO,
    TP_PIPS,
    TRADE_COOLDOWN_BARS,
    WINDOW_SIZE,
)
from backtest_features import build_feature_columns, build_gate_stats
from backtest_semantic import SemanticRuntime
from env_trading import TradingEnv


FORWARD_DATA_FILE = os.getenv("FORWARD_DATA_FILE", TEST_DATA_FILE).strip() or TEST_DATA_FILE
FORWARD_DATE_FROM = os.getenv("FORWARD_DATE_FROM", "").strip()
FORWARD_DATE_TO = os.getenv("FORWARD_DATE_TO", "").strip()
FORWARD_TRAIN_END = os.getenv("FORWARD_TRAIN_END", "").strip()
FORWARD_SPLIT_RATIO = float(os.getenv("FORWARD_SPLIT_RATIO", "0.80"))
FORWARD_DISABLE_PLOT = os.getenv("FORWARD_DISABLE_PLOT", "0").strip().lower() in {"1", "true", "yes"}
FORWARD_OUTPUT_PREFIX = os.getenv("FORWARD_OUTPUT_PREFIX", "forward_test").strip() or "forward_test"


def _patch_numpy_bitgenerator_compat():
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


SEMANTIC_RUNTIME = SemanticRuntime(models_dir=MODELS_DIR)
FEATURE_COLUMNS = build_feature_columns(SEMANTIC_RUNTIME.semantic_feature_count)


def _load_model():
    _patch_numpy_bitgenerator_compat()

    dummy_data = {
        "time": [pd.Timestamp.now()] * 80,
        "open": [1.0] * 80,
        "high": [1.0] * 80,
        "low": [1.0] * 80,
        "close": [1.0] * 80,
    }
    for col in FEATURE_COLUMNS:
        if col not in dummy_data:
            dummy_data[col] = [0] * 80

    mock_df = pd.DataFrame(dummy_data)
    dummy_env = DummyVecEnv(
        [lambda: TradingEnv(mock_df, lot_size=calc_auto_lot(INITIAL_BALANCE), sl_pips=SL_PIPS, tp_pips=TP_PIPS)]
    )

    vec_norm = VecNormalize.load(os.path.join(MODELS_DIR, "vec_normalize.pkl"), dummy_env)
    vec_norm.training = False
    vec_norm.norm_reward = False

    model = PPO.load(os.path.join(MODELS_DIR, "ppo_trading.zip"))
    return model, vec_norm


def _print_forward_config():
    print("=" * 70)
    print(" FORWARD TEST CONFIG")
    print("=" * 70)
    print(f" data_file:          {FORWARD_DATA_FILE}")
    print(f" forward_date_from:  {FORWARD_DATE_FROM or '-'}")
    print(f" forward_date_to:    {FORWARD_DATE_TO or '-'}")
    print(f" forward_train_end:  {FORWARD_TRAIN_END or '-'}")
    print(f" split_ratio:        {FORWARD_SPLIT_RATIO:.2f} (used when no start date provided)")
    print(f" bar_history:        {BAR_HISTORY}")
    print(f" window_size:        {WINDOW_SIZE}")
    print(f" SL/TP:              {SL_PIPS}/{TP_PIPS} pips")
    print(f" risk_percent:       {RISK_PERCENT:.2f}%")
    print("=" * 70 + "\n")


def _load_filtered_data():
    data_path = os.path.join(DATASETS_DIR, FORWARD_DATA_FILE)
    df = pd.read_csv(data_path)
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").drop_duplicates(subset=["time"], keep="last").reset_index(drop=True)

    if TEST_DATE_FROM:
        df = df[df["time"] >= pd.to_datetime(TEST_DATE_FROM)]
    if TEST_DATE_TO:
        df = df[df["time"] <= pd.to_datetime(TEST_DATE_TO)]

    if "has_delta" in df.columns:
        df = df[df["has_delta"] == 1]
    df = df.sort_values("time").reset_index(drop=True)

    if len(df) <= BAR_HISTORY + 2:
        raise ValueError(
            f"Not enough rows for forward test. Need > {BAR_HISTORY + 2}, got {len(df)} after filtering."
        )
    return df, data_path


def _select_forward_indices(df: pd.DataFrame):
    if FORWARD_DATE_TO:
        df = df[df["time"] <= pd.to_datetime(FORWARD_DATE_TO)].reset_index(drop=True)
    if len(df) <= BAR_HISTORY + 2:
        raise ValueError(
            f"Not enough rows after FORWARD_DATE_TO filter. Need > {BAR_HISTORY + 2}, got {len(df)}."
        )

    if FORWARD_DATE_FROM:
        forward_start_time = pd.to_datetime(FORWARD_DATE_FROM)
        start_idx = int(df["time"].searchsorted(forward_start_time))
    elif FORWARD_TRAIN_END:
        train_end_time = pd.to_datetime(FORWARD_TRAIN_END)
        start_idx = int(df["time"].searchsorted(train_end_time, side="right"))
    else:
        split_ratio = min(max(FORWARD_SPLIT_RATIO, 0.05), 0.95)
        start_idx = int(len(df) * split_ratio)

    start_idx = max(start_idx, BAR_HISTORY)
    if start_idx >= len(df) - 1:
        raise ValueError(
            f"Forward start index out of range ({start_idx}) for {len(df)} rows. "
            "Set FORWARD_DATE_FROM/FORWARD_TRAIN_END earlier or lower FORWARD_SPLIT_RATIO."
        )

    warmup_start_idx = max(0, start_idx - BAR_HISTORY)
    run_df = df.iloc[warmup_start_idx:].reset_index(drop=True)
    eval_start_local = start_idx - warmup_start_idx
    loop_start = max(BAR_HISTORY, eval_start_local)
    return df, run_df, start_idx, warmup_start_idx, eval_start_local, loop_start


def _build_gate_stats_reference(df: pd.DataFrame, start_idx: int):
    hist_df = df.iloc[:start_idx]
    if len(hist_df) > WINDOW_SIZE:
        return build_gate_stats(hist_df)
    return build_gate_stats(df)


def _run_forward(run_df, loop_start, eval_start_local, gate_stats):
    model, vec_norm = _load_model()
    bridge = PPOBridge(
        model=model,
        vec_norm=vec_norm,
        feature_columns=FEATURE_COLUMNS,
        semantic_runtime=SEMANTIC_RUNTIME,
        semantic_feature_count=SEMANTIC_RUNTIME.semantic_feature_count,
        gate_stats=gate_stats,
    )

    start_time = run_df["time"].iloc[eval_start_local]
    equity_history = [INITIAL_BALANCE]
    time_history = [start_time]
    buy_signals = []
    sell_signals = []
    close_signals = []

    for i in range(loop_start, len(run_df)):
        window_df = run_df.iloc[i - BAR_HISTORY : i][["time", "open", "high", "low", "close"]].reset_index(drop=True)
        delta_tick = run_df.iloc[i - 1].get("delta_tick", 0)
        delta_price = run_df.iloc[i - 1].get("delta_price", 0.0)
        action, _ = bridge.process_bar(window_df, delta_tick, delta_price)

        if i < eval_start_local:
            continue

        current_time = run_df["time"].iloc[i]
        equity_history.append(bridge.equity)
        time_history.append(current_time)

        if action == 1:
            buy_signals.append((current_time, bridge.equity))
        elif action == 2:
            sell_signals.append((current_time, bridge.equity))
        elif action == 3:
            close_signals.append((current_time, bridge.equity))

    return bridge, equity_history, time_history, buy_signals, sell_signals, close_signals


def _print_results(bridge, df, start_idx, time_history, equity_history, gate_stats):
    max_dd = max((bridge.max_equity - e) / bridge.max_equity for e in equity_history) if equity_history else 0.0
    avg_quality, low_quality, quality_count = SEMANTIC_RUNTIME.quality_summary()

    print("\n" + "=" * 70)
    print(" FORWARD TEST RESULTS")
    print("=" * 70)
    print(f"Period:           {df['time'].iloc[start_idx]} -> {time_history[-1]}")
    print(f"Initial Equity:   ${INITIAL_BALANCE:.2f}")
    print(f"Final Equity:     ${bridge.equity:.2f}")
    print(f"Return:           {((bridge.equity / INITIAL_BALANCE - 1) * 100):.2f}%")
    print(f"Total Trades:     {bridge.trades}")
    print(f"Win Rate:         {(bridge.wins / bridge.trades * 100) if bridge.trades > 0 else 0:.2f}%")
    print(f"Max Drawdown:     {max_dd * 100:.2f}%")
    print(f"Total Fees Paid:  ${bridge.total_fees:.2f}")
    print(f"SL Hits:          {bridge.sl_hits}")
    print(f"TP Hits:          {bridge.tp_hits}")
    print(
        " Semantic cache: "
        f"{len(SEMANTIC_RUNTIME.cache)} timestamps | "
        f"matched={SEMANTIC_RUNTIME.stats['matched']} | synthetic={SEMANTIC_RUNTIME.stats['synthetic']} | "
        f"mode={EMBED_TEST_MODE} | source={EMBED_SOURCE_MODE} | "
        f"knn={SEMANTIC_RUNTIME.stats['knn_fallback']} | "
        f"zero={SEMANTIC_RUNTIME.stats['zero_fallback']} | "
        f"quality_avg={avg_quality:.3f} ({quality_count}) | "
        f"quality_low={low_quality}"
    )
    print(
        f" Trade gate: open_prob>={OPEN_PROB_THRESHOLD:.2f}, edge>={OPEN_EDGE_THRESHOLD:.2f}, "
        f"margin>={MIN_ACTION_MARGIN:.2f}, hold_edge>={HOLD_EDGE_THRESHOLD:.2f}, "
        f"cooldown={TRADE_COOLDOWN_BARS}, adaptive={ADAPTIVE_GATE}, sem_q_min={EMBED_QUALITY_MIN:.2f}, "
        f"skipped={bridge.skipped_signals}, margin_skipped={bridge.margin_skips}, "
        f"semantic_skipped={bridge.semantic_skips}, "
        f"def_skipped={bridge.defensive_skips}, def_triggers={bridge.defensive_triggers}"
    )
    print(
        " Gate stats: "
        f"atr_high={gate_stats['atr_high']:.6f}, atr_extreme={gate_stats['atr_extreme']:.6f}, "
        f"trend_flat={gate_stats['trend_flat']:.4f}, trend_strong={gate_stats['trend_strong']:.4f}, "
        f"adx_flat={gate_stats['adx_flat']:.4f}, adx_strong={gate_stats['adx_strong']:.4f}"
    )
    print("=" * 70 + "\n")


def _save_outputs(time_history, equity_history, buy_signals, sell_signals, close_signals):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    equity_path = os.path.join(OUTPUT_DIR, f"{FORWARD_OUTPUT_PREFIX}_equity.csv")
    signal_path = os.path.join(OUTPUT_DIR, f"{FORWARD_OUTPUT_PREFIX}_signals.csv")

    pd.DataFrame({"time": time_history, "equity": equity_history}).to_csv(equity_path, index=False)

    signal_rows = []
    for t, e in buy_signals:
        signal_rows.append({"time": t, "equity": e, "action": "BUY"})
    for t, e in sell_signals:
        signal_rows.append({"time": t, "equity": e, "action": "SELL"})
    for t, e in close_signals:
        signal_rows.append({"time": t, "equity": e, "action": "CLOSE"})
    signal_df = pd.DataFrame(signal_rows).sort_values("time") if signal_rows else pd.DataFrame(
        columns=["time", "equity", "action"]
    )
    signal_df.to_csv(signal_path, index=False)

    print(f" Saved equity curve: {equity_path}")
    print(f" Saved signals:      {signal_path}")


def _plot_if_enabled(time_history, equity_history, buy_signals, sell_signals, close_signals, final_equity):
    if FORWARD_DISABLE_PLOT:
        print(" Plot disabled by FORWARD_DISABLE_PLOT=1")
        return
    try:
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt
    except ImportError:
        print(" matplotlib not installed; skipping plot.")
        return

    plt.figure(figsize=(14, 7))
    plt.title(f"PPO Forward Test (Return: {((final_equity / INITIAL_BALANCE - 1) * 100):.2f}%)")
    plt.plot(time_history, equity_history, label="Equity", color="blue", linewidth=1.5)

    if buy_signals:
        bx, by = zip(*buy_signals)
        plt.scatter(bx, by, marker="^", color="green", s=50, label="Buy", alpha=0.7)
    if sell_signals:
        sx, sy = zip(*sell_signals)
        plt.scatter(sx, sy, marker="v", color="red", s=50, label="Sell", alpha=0.7)
    if close_signals:
        cx, cy = zip(*close_signals)
        plt.scatter(cx, cy, marker="x", color="black", s=30, label="Close", alpha=0.5)

    plt.axhline(y=INITIAL_BALANCE, color="r", linestyle="--", alpha=0.5, label="Initial Balance")
    plt.xlabel("Time")
    plt.ylabel("Equity (USD)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.gcf().autofmt_xdate()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plot_path = os.path.join(OUTPUT_DIR, f"{FORWARD_OUTPUT_PREFIX}_plot.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f" Saved plot:         {plot_path}")


def main():
    _print_forward_config()

    df, data_path = _load_filtered_data()
    df_all, run_df, start_idx, warmup_start_idx, eval_start_local, loop_start = _select_forward_indices(df)
    gate_stats = _build_gate_stats_reference(df_all, start_idx)

    print("=" * 70)
    print(" FORWARD WINDOW")
    print("=" * 70)
    print(f" data_path:          {data_path}")
    print(f" total_rows:         {len(df_all)}")
    print(f" warmup_start_idx:   {warmup_start_idx}")
    print(f" forward_start_idx:  {start_idx}")
    print(f" eval_start_local:   {eval_start_local}")
    print(f" loop_start:         {loop_start}")
    print(f" start_time:         {df_all['time'].iloc[start_idx]}")
    print(f" end_time:           {df_all['time'].iloc[-1]}")
    print("=" * 70 + "\n")

    bridge, equity_history, time_history, buy_signals, sell_signals, close_signals = _run_forward(
        run_df, loop_start, eval_start_local, gate_stats
    )

    _print_results(bridge, df_all, start_idx, time_history, equity_history, gate_stats)
    _save_outputs(time_history, equity_history, buy_signals, sell_signals, close_signals)
    _plot_if_enabled(time_history, equity_history, buy_signals, sell_signals, close_signals, bridge.equity)


if __name__ == "__main__":
    main()
