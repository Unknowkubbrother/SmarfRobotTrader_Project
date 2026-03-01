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

from backtest_config import (
    BAR_HISTORY,
    MODELS_DIR,
    DATASETS_DIR,
    OUTPUT_DIR,
    DISABLE_PLOT,
    EMBED_SOURCE_MODE,
    EMBED_TEST_MODE,
    EMBED_QUALITY_MIN,
    PIP_SIZE,
    PIP_VALUE,
    SPREAD_PIPS,
    TEST_DATA_FILE,
    TEST_DATE_FROM,
    TEST_DATE_TO,
    TEST_INITIAL_BALANCE,
    TEST_LOT_SIZE,
    RISK_LEVEL,
    RISK_PERCENT,
    WINDOW_SIZE,
)
from backtest_bridge import PPOBridge, calc_auto_lot
from backtest_features import build_feature_columns, build_gate_stats, calculate_features as _calculate_features
from backtest_semantic import SemanticRuntime
from env_trading import TradingEnv


SEMANTIC_RUNTIME = SemanticRuntime(models_dir=MODELS_DIR)
FEATURE_COLUMNS = build_feature_columns(SEMANTIC_RUNTIME.semantic_feature_count)
TEST_MODEL_PATH = os.getenv("TEST_MODEL_PATH", os.path.join(MODELS_DIR, "ppo_trading.zip")).strip() or os.path.join(
    MODELS_DIR, "ppo_trading.zip"
)
TEST_VEC_NORM_PATH = os.getenv("TEST_VEC_NORM_PATH", os.path.join(MODELS_DIR, "vec_normalize.pkl")).strip() or os.path.join(
    MODELS_DIR, "vec_normalize.pkl"
)
TEST_EXECUTION_MODE = os.getenv("TEST_EXECUTION_MODE", "mt5_bridge").strip().lower() or "mt5_bridge"
if TEST_EXECUTION_MODE not in {"mt5_bridge", "train_env"}:
    print(f" Invalid TEST_EXECUTION_MODE={TEST_EXECUTION_MODE!r}, fallback to 'mt5_bridge'")
    TEST_EXECUTION_MODE = "mt5_bridge"
TEST_LOT_MIN = float(os.getenv("TEST_LOT_MIN", os.getenv("LOT_MIN", "0.0001")))
TEST_LOT_STEP = float(os.getenv("TEST_LOT_STEP", os.getenv("LOT_STEP", "0.0001")))
EFFECTIVE_TEST_LOT_SIZE = (
    float(TEST_LOT_SIZE)
    if TEST_LOT_SIZE is not None
    else calc_auto_lot(
        TEST_INITIAL_BALANCE,
        risk_level=RISK_LEVEL,
        risk_pct=RISK_PERCENT,
        min_lot=TEST_LOT_MIN,
        lot_step=TEST_LOT_STEP,
    )
)
EFFECTIVE_TEST_LOT_SOURCE = (
    "manual(TEST_LOT_SIZE)"
    if TEST_LOT_SIZE is not None
    else f"auto(risk_level={RISK_LEVEL}, risk_pct={RISK_PERCENT}%, min={TEST_LOT_MIN}, step={TEST_LOT_STEP})"
)


def calculate_features(df, delta_tick=0, delta_price=0.0, preserve_existing_delta=False):
    return _calculate_features(
        df,
        semantic_runtime=SEMANTIC_RUNTIME,
        semantic_feature_count=SEMANTIC_RUNTIME.semantic_feature_count,
        delta_tick=delta_tick,
        delta_price=delta_price,
        preserve_existing_delta=preserve_existing_delta,
    )


def _print_data_filtering(df_full, df, data_path):
    print("=" * 50)
    print(" DATA FILTERING")
    print("=" * 50)
    print(f"ไฟล์ข้อมูล:        {data_path}")
    print(f"ข้อมูลทั้งหมด:     {len(df_full):,} rows")
    print(f"ข้อมูลที่มี delta: {len(df):,} rows")
    print(f"ช่วงเวลา:          {df['time'].iloc[0]} -> {df['time'].iloc[-1]}")
    if TEST_DATE_FROM or TEST_DATE_TO:
        print(f"ช่วงที่เลือก:       from={TEST_DATE_FROM or '-'} to={TEST_DATE_TO or '-'}")
    print(f"lot config:         {EFFECTIVE_TEST_LOT_SIZE:.4f} ({EFFECTIVE_TEST_LOT_SOURCE})")
    print(f"execution mode:     {TEST_EXECUTION_MODE} ({'MT5/Live parity' if TEST_EXECUTION_MODE == 'mt5_bridge' else 'Train-Env parity'})")
    print(f"model path:         {TEST_MODEL_PATH}")
    print(f"vecnorm path:       {TEST_VEC_NORM_PATH}")
    print("=" * 50 + "\n")


def _load_model():
    rows = max(80, WINDOW_SIZE + 5)
    dummy_data = {
        "time": [pd.Timestamp.now()] * rows,
        "open": [1.0] * rows,
        "high": [1.0] * rows,
        "low": [1.0] * rows,
        "close": [1.0] * rows,
    }
    for col in FEATURE_COLUMNS:
        if col not in dummy_data:
            dummy_data[col] = [0.0] * rows

    mock_df = pd.DataFrame(dummy_data)
    dummy_env = DummyVecEnv(
        [
            lambda: TradingEnv(
                mock_df,
                window_size=WINDOW_SIZE,
                initial_balance=TEST_INITIAL_BALANCE,
                lot_size=EFFECTIVE_TEST_LOT_SIZE,
                pip_size=PIP_SIZE,
                pip_value=PIP_VALUE,
                spread_pips=SPREAD_PIPS,
                random_start=False,
            )
        ]
    )

    vec_norm = VecNormalize.load(TEST_VEC_NORM_PATH, dummy_env)
    vec_norm.training = False
    vec_norm.norm_reward = False

    model = PPO.load(TEST_MODEL_PATH)
    return model, vec_norm


def _print_results(result, gate_stats):
    df = result["df_ref"]
    start_time = result["start_time"]
    end_time = result["end_time"]
    final_equity = float(result["final_equity"])
    trades = int(result["trades"])
    wins = int(result["wins"])
    total_fees = float(result["total_fees"])
    equity_history = result["equity_history"]

    max_dd = max((max(equity_history[: i + 1]) - equity_history[i]) / max(equity_history[: i + 1]) for i in range(len(equity_history)))
    avg_quality, low_quality, quality_count = SEMANTIC_RUNTIME.quality_summary()
    mode_name = "MT5-Bridge Parity" if TEST_EXECUTION_MODE == "mt5_bridge" else "Train-Env Parity"

    print("\n" + "=" * 50)
    print(f" TEST RESULTS ({mode_name})")
    print("=" * 50)
    print(f"Final Equity:     ${final_equity:.2f}")
    print(f"Return:           {((final_equity / TEST_INITIAL_BALANCE - 1) * 100):.2f}%")
    print(f"Total Trades:     {trades}")
    print(f"Win Rate:         {(wins / trades * 100) if trades > 0 else 0:.2f}%")
    print(f"Max Drawdown:     {max_dd * 100:.2f}%")
    print(f"Total Fees Paid:  ${total_fees:.2f}")
    print("=" * 50)
    print(f"Start Time: {start_time}")
    print(f"End Time:   {end_time}")

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
        f" Action policy: model.predict(deterministic=True) on {mode_name} | "
        f"sem_q_min={EMBED_QUALITY_MIN:.2f}"
    )

    print(
        " Gate stats (diagnostic only): "
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
        mode_name = "MT5-Bridge Parity" if TEST_EXECUTION_MODE == "mt5_bridge" else "Train-Env Parity"
        plt.title(f"PPO Backtest - {mode_name} (Return: {((final_equity / TEST_INITIAL_BALANCE - 1) * 100):.2f}%)")
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

        plt.axhline(y=TEST_INITIAL_BALANCE, color="r", linestyle="--", alpha=0.5, label="Initial Balance")
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


def _prepare_feature_frame(df):
    df_features = calculate_features(df, preserve_existing_delta=True)
    # Match train feature warm-up trimming.
    df_features = df_features.iloc[50:].reset_index(drop=True)
    return df_features


def _run_backtest_train_env(df):
    gate_stats = build_gate_stats(df)
    model, vec_norm = _load_model()

    env = TradingEnv(
        df,
        window_size=WINDOW_SIZE,
        initial_balance=TEST_INITIAL_BALANCE,
        lot_size=EFFECTIVE_TEST_LOT_SIZE,
        pip_size=PIP_SIZE,
        pip_value=PIP_VALUE,
        spread_pips=SPREAD_PIPS,
        random_start=False,
    )

    obs, _ = env.reset()

    equity_history = [env.equity]
    time_history = [df["time"].iloc[env.step_idx]]
    buy_signals = []
    sell_signals = []
    close_signals = []

    print(f" Processing {len(df) - env.step_idx} bars (TradingEnv window={WINDOW_SIZE})...")

    done = False
    while not done:
        step_idx = env.step_idx
        step_time = df["time"].iloc[step_idx]

        obs_norm = vec_norm.normalize_obs(obs)
        action, _ = model.predict(obs_norm, deterministic=True)
        action = int(action)

        obs, _, terminated, truncated, info = env.step(action)

        equity_history.append(float(info["equity"]))
        time_history.append(step_time)

        if action == 1:
            buy_signals.append((step_time, float(info["equity"])))
        elif action == 2:
            sell_signals.append((step_time, float(info["equity"])))
        elif action in (3, 4):
            close_signals.append((step_time, float(info["equity"])))

        done = bool(terminated or truncated)

    return {
        "df_ref": df,
        "start_time": df["time"].iloc[WINDOW_SIZE],
        "end_time": df["time"].iloc[min(env.step_idx, len(df) - 1)],
        "final_equity": float(env.equity),
        "trades": int(env.trades),
        "wins": int(env.wins),
        "total_fees": float(env.total_fees),
        "gate_stats": gate_stats,
        "equity_history": equity_history,
        "time_history": time_history,
        "buy_signals": buy_signals,
        "sell_signals": sell_signals,
        "close_signals": close_signals,
    }


def _run_backtest_mt5_bridge(df):
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
    bridge.equity = TEST_INITIAL_BALANCE
    bridge.balance = TEST_INITIAL_BALANCE
    bridge.max_equity = TEST_INITIAL_BALANCE
    bridge.lot_size = EFFECTIVE_TEST_LOT_SIZE
    bridge.spread_cost = SPREAD_PIPS * PIP_VALUE * bridge.lot_size

    min_bars = max(BAR_HISTORY, WINDOW_SIZE)
    if len(df) < min_bars:
        raise ValueError(f"Not enough bars for mt5_bridge mode. Need >= {min_bars}, got {len(df)}.")

    start_idx = min_bars - 1
    print(
        f" Processing {len(df) - start_idx} bars "
        f"(PPOBridge window={WINDOW_SIZE}, bar_history={BAR_HISTORY})..."
    )

    equity_history = [float(bridge.equity)]
    time_history = [df["time"].iloc[start_idx]]
    buy_signals = []
    sell_signals = []
    close_signals = []

    for idx in range(start_idx, len(df)):
        window_df = df.iloc[idx - BAR_HISTORY + 1 : idx + 1].copy()
        step_time = df["time"].iloc[idx]
        row = df.iloc[idx]
        if "delta_tick" in df.columns and pd.notna(row.get("delta_tick")):
            delta_tick = int(float(row.get("delta_tick", 0.0)))
        else:
            delta_tick = 0
        if "delta_price" in df.columns and pd.notna(row.get("delta_price")):
            delta_price = float(row.get("delta_price", 0.0))
        else:
            delta_price = 0.0

        action, _ = bridge.process_bar(window_df, delta_tick=delta_tick, delta_price=delta_price)
        action = int(action)

        equity_history.append(float(bridge.equity))
        time_history.append(step_time)

        if action == 1:
            buy_signals.append((step_time, float(bridge.equity)))
        elif action == 2:
            sell_signals.append((step_time, float(bridge.equity)))
        elif action in (3, 4):
            close_signals.append((step_time, float(bridge.equity)))

    return {
        "df_ref": df,
        "start_time": df["time"].iloc[start_idx],
        "end_time": df["time"].iloc[len(df) - 1],
        "final_equity": float(bridge.equity),
        "trades": int(bridge.trades),
        "wins": int(bridge.wins),
        "total_fees": float(bridge.total_fees),
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

    _print_data_filtering(df_full, df, data_path)

    if "has_delta" in df.columns:
        df = df.drop(columns=["has_delta"])

    if TEST_EXECUTION_MODE == "train_env":
        feature_df = _prepare_feature_frame(df)
        if len(feature_df) <= (WINDOW_SIZE + 1):
            raise ValueError(
                f"Not enough bars after feature prep for WINDOW_SIZE={WINDOW_SIZE}. "
                f"Need > {WINDOW_SIZE + 1}, got {len(feature_df)}."
            )
        result = _run_backtest_train_env(feature_df)
    else:
        result = _run_backtest_mt5_bridge(df)

    _print_results(
        result=result,
        gate_stats=result["gate_stats"],
    )

    _plot_if_enabled(
        time_history=result["time_history"],
        equity_history=result["equity_history"],
        buy_signals=result["buy_signals"],
        sell_signals=result["sell_signals"],
        close_signals=result["close_signals"],
        final_equity=float(result["final_equity"]),
    )


if __name__ == "__main__":
    main()
