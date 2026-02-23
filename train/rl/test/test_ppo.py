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
    MODELS_DIR,
    DATASETS_DIR,
    OUTPUT_DIR,
    DISABLE_PLOT,
    EMBED_SOURCE_MODE,
    EMBED_TEST_MODE,
    EMBED_QUALITY_MIN,
    HOLD_EDGE_THRESHOLD,
    INITIAL_BALANCE,
    MIN_ACTION_MARGIN,
    OPEN_EDGE_THRESHOLD,
    OPEN_PROB_THRESHOLD,
    PIP_SIZE,
    PIP_VALUE,
    RISK_PERCENT,
    SL_PIPS,
    TEST_DATA_FILE,
    TEST_DATE_FROM,
    TEST_DATE_TO,
    TP_PIPS,
    TRADE_COOLDOWN_BARS,
    WINDOW_SIZE,
    MAX_HOLD_STEPS,
)
from backtest_features import build_feature_columns, build_gate_stats, calculate_features as _calculate_features
from backtest_semantic import SemanticRuntime
from env_trading import TradingEnv


SEMANTIC_RUNTIME = SemanticRuntime(
    models_dir=MODELS_DIR,
)
FEATURE_COLUMNS = build_feature_columns(SEMANTIC_RUNTIME.semantic_feature_count)


def calculate_features(df, delta_tick=0, delta_price=0.0):
    return _calculate_features(
        df,
        semantic_runtime=SEMANTIC_RUNTIME,
        semantic_feature_count=SEMANTIC_RUNTIME.semantic_feature_count,
        delta_tick=delta_tick,
        delta_price=delta_price,
    )


def _print_data_filtering(df_full, df, data_path):
    print("=" * 50)
    print(" DATA FILTERING")
    print("=" * 50)
    print(f"ไฟล์ข้อมูล:        {data_path}")
    print(f"ข้อมูลทั้งหมด:     {len(df_full):,} rows")
    print(f"ข้อมูลที่มี delta: {len(df):,} rows")
    print(f"ช่วงเวลา:          {df['time'].iloc[0]} → {df['time'].iloc[-1]}")
    if TEST_DATE_FROM or TEST_DATE_TO:
        print(f"ช่วงที่เลือก:       from={TEST_DATE_FROM or '-'} to={TEST_DATE_TO or '-'}")
    print("=" * 50 + "\n")


def _load_model():
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


def _print_results(df, bridge, equity_history, gate_stats):
    max_dd = max((bridge.max_equity - e) / bridge.max_equity for e in equity_history) if equity_history else 0
    avg_quality, low_quality, quality_count = SEMANTIC_RUNTIME.quality_summary()

    print("\n" + "=" * 50)
    print(" TEST RESULTS (Server-Aligned Mode)")
    print("=" * 50)
    print(f"Final Equity:     ${bridge.equity:.2f}")
    print(f"Return:           {((bridge.equity / INITIAL_BALANCE - 1) * 100):.2f}%")
    print(f"Total Trades:     {bridge.trades}")
    print(f"Win Rate:         {(bridge.wins / bridge.trades * 100) if bridge.trades > 0 else 0:.2f}%")
    print(f"Max Drawdown:     {max_dd * 100:.2f}%")
    print(f"Total Fees Paid:  ${bridge.total_fees:.2f}")
    print(f"SL Hits:          {bridge.sl_hits}")
    print(f"TP Hits:          {bridge.tp_hits}")
    print("=" * 50)
    print(f"Start Time: {df['time'].iloc[BAR_HISTORY]}")
    print(f"End Time:   {df['time'].iloc[-1]}")

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


def _plot_if_enabled(time_history, equity_history, buy_signals, sell_signals, close_signals, final_equity):
    if DISABLE_PLOT:
        print("\n Plot disabled by DISABLE_PLOT=1")
        return

    try:
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt

        print("\n Generating Strategy Tester Graph...")

        plt.figure(figsize=(14, 7))
        plt.title(f"PPO Backtest — Server-Aligned (Return: {((final_equity / INITIAL_BALANCE - 1) * 100):.2f}%)")
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
        graph_path = os.path.join(OUTPUT_DIR, "strategy_tester_results.png")
        plt.savefig(graph_path, dpi=300, bbox_inches="tight")
        print(f" Graph saved as '{graph_path}'")

    except ImportError:
        print("\n  matplotlib not installed.")


def _run_backtest(df):
    gate_stats = build_gate_stats(df)
    model, vec_norm = _load_model()

    bridge = PPOBridge(
        model=model,
        vec_norm=vec_norm,
        feature_columns=FEATURE_COLUMNS,
        semantic_runtime=SEMANTIC_RUNTIME,
        semantic_feature_count=SEMANTIC_RUNTIME.semantic_feature_count,
        gate_stats=gate_stats,
    )

    equity_history = [INITIAL_BALANCE]
    time_history = [df["time"].iloc[BAR_HISTORY]]
    buy_signals = []
    sell_signals = []
    close_signals = []

    print(f" Processing {len(df) - BAR_HISTORY} bars ({BAR_HISTORY}-bar sliding window)...")

    for i in range(BAR_HISTORY, len(df)):
        window_df = df.iloc[i - BAR_HISTORY : i][["time", "open", "high", "low", "close"]].reset_index(drop=True)

        delta_tick = df.iloc[i - 1].get("delta_tick", 0)
        delta_price = df.iloc[i - 1].get("delta_price", 0.0)

        action, _ = bridge.process_bar(window_df, delta_tick, delta_price)
        current_time = df["time"].iloc[i]

        equity_history.append(bridge.equity)
        time_history.append(current_time)

        if action == 1:
            buy_signals.append((current_time, bridge.equity))
        elif action == 2:
            sell_signals.append((current_time, bridge.equity))
        elif action == 3:
            close_signals.append((current_time, bridge.equity))

    return {
        "bridge": bridge,
        "gate_stats": gate_stats,
        "equity_history": equity_history,
        "time_history": time_history,
        "buy_signals": buy_signals,
        "sell_signals": sell_signals,
        "close_signals": close_signals,
    }


def main():
    data_path = os.path.join(DATASETS_DIR, TEST_DATA_FILE)
    df = pd.read_csv(data_path)
    df["time"] = pd.to_datetime(df["time"])

    if TEST_DATE_FROM:
        df = df[df["time"] >= pd.to_datetime(TEST_DATE_FROM)]
    if TEST_DATE_TO:
        df = df[df["time"] <= pd.to_datetime(TEST_DATE_TO)]

    df_full = df.copy()
    df = df[df["has_delta"] == 1].sort_values("time").reset_index(drop=True)

    if len(df) == 0:
        raise ValueError("No rows left after filtering. Check TEST_DATE_FROM/TEST_DATE_TO and has_delta values.")
    if len(df) <= BAR_HISTORY:
        raise ValueError(f"Not enough bars for BAR_HISTORY={BAR_HISTORY}. Need > {BAR_HISTORY}, got {len(df)} after filtering.")

    _print_data_filtering(df_full, df, data_path)
    result = _run_backtest(df)

    _print_results(
        df=df,
        bridge=result["bridge"],
        equity_history=result["equity_history"],
        gate_stats=result["gate_stats"],
    )

    _plot_if_enabled(
        time_history=result["time_history"],
        equity_history=result["equity_history"],
        buy_signals=result["buy_signals"],
        sell_signals=result["sell_signals"],
        close_signals=result["close_signals"],
        final_equity=result["bridge"].equity,
    )


if __name__ == "__main__":
    main()
