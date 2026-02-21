"""
PPO ZMQ Bridge Server v2.0 — Python-Managed State (MT5-Aligned)

Key difference from v1: This server tracks position/equity/PnL internally
(same as env_trading.py) instead of relying on MT5's state. This ensures
the model sees identical observations in both Python test and MT5.

SL/TP is also managed here using bar H/L, matching env_trading.py exactly.
MT5 orders are opened WITHOUT SL/TP — Python sends CLOSE when SL/TP hit.
"""

import zmq
import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from env_trading import TradingEnv
from datetime import datetime

# ==================================================
# CONFIG
# ==================================================
HOST = "0.0.0.0"
PORT = 5555
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
MODEL_PATH = os.path.join(BASE_DIR, "ppo_trading")
VEC_NORM_PATH = os.path.join(BASE_DIR, "vec_normalize.pkl")

# Must match env_trading.py and EA settings
WINDOW_SIZE = 20
INITIAL_BALANCE = 10000
PIP_SIZE = 0.0001
PIP_VALUE = 10.0
SL_PIPS = 30
TP_PIPS = 60
SPREAD_PIPS = 2
MAX_HOLD_STEPS = 30
RISK_PERCENT = 1.0  # Risk % per trade (0 = use fixed LOT_SIZE below)
LOT_SIZE = 0.1      # Fallback fixed lot


def calc_auto_lot(balance, risk_pct=RISK_PERCENT, sl_pips=SL_PIPS,
                  pip_value_per_lot=PIP_VALUE, min_lot=0.01, lot_step=0.01):
    """Risk-based position sizing (same formula as EA)"""
    if risk_pct <= 0:
        return LOT_SIZE
    risk_amount = balance * risk_pct / 100.0
    lot = risk_amount / (sl_pips * pip_value_per_lot)
    lot = max(min_lot, lot_step * int(lot / lot_step))
    return round(lot, 2)

FEATURE_COLUMNS = [
    'return', 'range', 'delta_tick', 'delta_price',
    'body_ratio', 'momentum',
    'sma_cross', 'rsi_norm', 'atr_norm', 'trend'
]


# ==================================================
# LOAD MODEL
# ==================================================
def load_model():
    print("📦 Loading PPO model...")
    dummy_data = {
        'time': [datetime.now()] * 80,
        'open': [1.0]*80, 'high': [1.0]*80, 'low': [1.0]*80, 'close': [1.0]*80,
        'delta_tick': [0]*80, 'delta_price': [0]*80,
        'sma_cross': [0]*80, 'rsi_norm': [0]*80, 'atr_norm': [0]*80, 'trend': [0]*80
    }
    mock_df = pd.DataFrame(dummy_data)
    dummy_env = DummyVecEnv([lambda: TradingEnv(mock_df, lot_size=LOT_SIZE, sl_pips=SL_PIPS, tp_pips=TP_PIPS)])
    vec_norm = VecNormalize.load(VEC_NORM_PATH, dummy_env)
    vec_norm.training = False
    vec_norm.norm_reward = False
    model = PPO.load(MODEL_PATH)
    print("✅ Model loaded successfully")
    return model, vec_norm


# ==================================================
# FEATURE ENGINEERING (same as test_ppo.py)
# ==================================================
def calculate_features(df, delta_tick=0, delta_price=0.0):
    df = df.copy()
    df['return'] = df['close'].pct_change().fillna(0)
    df['range'] = (df['high'] - df['low']) / df['close']

    full_range = df['high'] - df['low']
    df['body_ratio'] = np.where(full_range > 0, abs(df['close'] - df['open']) / full_range, 0)
    df['momentum'] = df['return'].rolling(window=5).sum().fillna(0)

    df['delta_tick'] = 0
    df['delta_price'] = 0.0
    df.loc[df.index[-1], 'delta_tick'] = delta_tick
    df.loc[df.index[-1], 'delta_price'] = delta_price

    sma20 = df['close'].rolling(20).mean()
    sma50 = df['close'].rolling(50).mean()
    df['sma_cross'] = np.where(sma20 > sma50, 1, np.where(sma20 < sma50, -1, 0))
    df['sma_cross'] = df['sma_cross'].fillna(0)

    delta_c = df['close'].diff()
    gain = delta_c.clip(lower=0).rolling(14).mean()
    loss = (-delta_c.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    df['rsi_norm'] = ((rsi - 50) / 50).fillna(0)

    tr = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    df['atr_norm'] = (tr.rolling(14).mean() / df['close']).fillna(0)

    df['trend'] = (sma20.pct_change(5) * 100).fillna(0)
    df['trend'] = df['trend'].clip(-2, 2)

    return df


# ==================================================
# PYTHON-MANAGED STATE (mirrors env_trading.py)
# ==================================================
class PPOBridge:
    """Tracks position/equity/PnL internally — same logic as env_trading.py"""

    def __init__(self, model, vec_norm):
        self.model = model
        self.vec_norm = vec_norm

        # Internal state (same as env_trading.py)
        self.position = 0
        self.entry_price = 0.0
        self.hold_steps = 0
        self.equity = INITIAL_BALANCE
        self.balance = INITIAL_BALANCE
        self.unrealized_pnl = 0.0
        self.trades = 0
        self.wins = 0
        self.total_pnl = 0.0
        self.total_fees = 0.0
        self.sl_hits = 0
        self.tp_hits = 0

        self.lot_size = calc_auto_lot(INITIAL_BALANCE)
        self.spread_cost = SPREAD_PIPS * PIP_VALUE * self.lot_size

        self.prev_bar_high = 0.0
        self.prev_bar_low = 0.0
        self.first_bar = True

    def _calc_pnl(self, entry, exit_price, direction):
        pips = (exit_price - entry) / PIP_SIZE
        return direction * pips * PIP_VALUE * self.lot_size

    def _open(self, direction, price):
        self.position = direction
        self.entry_price = price
        self.hold_steps = 0
        self.unrealized_pnl = 0.0
        self.equity -= self.spread_cost
        self.balance -= self.spread_cost
        self.total_fees += self.spread_cost

    def _close(self, exit_price):
        pnl = self._calc_pnl(self.entry_price, exit_price, self.position)
        self.total_pnl += pnl
        self.balance += pnl
        self.equity = self.balance
        self.trades += 1
        if pnl > 0:
            self.wins += 1
        self.position = 0
        self.entry_price = 0.0
        self.hold_steps = 0
        self.unrealized_pnl = 0.0

    def process_bar(self, df, delta_tick, delta_price):
        """Process one bar — returns action for EA"""
        df = calculate_features(df, delta_tick, delta_price)
        last_bar = df.iloc[-1]
        current_price = last_bar['close']
        bar_high = last_bar['high']
        bar_low = last_bar['low']

        sl_tp_closed = False

        # === SL/TP CHECK on last completed bar's H/L (same as env next_bar check) ===
        if self.position != 0 and self.entry_price > 0 and not self.first_bar:
            if self.position == 1:  # Long
                sl_price = self.entry_price - SL_PIPS * PIP_SIZE
                tp_price = self.entry_price + TP_PIPS * PIP_SIZE
                hit_sl = bar_low <= sl_price
                hit_tp = bar_high >= tp_price
            else:  # Short
                sl_price = self.entry_price + SL_PIPS * PIP_SIZE
                tp_price = self.entry_price - TP_PIPS * PIP_SIZE
                hit_sl = bar_high >= sl_price
                hit_tp = bar_low <= tp_price

            if hit_sl and hit_tp:
                self._close(sl_price)  # Conservative: SL first
                self.sl_hits += 1
                sl_tp_closed = True
            elif hit_sl:
                self._close(sl_price)
                self.sl_hits += 1
                sl_tp_closed = True
            elif hit_tp:
                self._close(tp_price)
                self.tp_hits += 1
                sl_tp_closed = True
            elif self.hold_steps >= MAX_HOLD_STEPS:
                self._close(current_price)
                sl_tp_closed = True

        # Update unrealized PnL (if position still open)
        if self.position != 0 and not sl_tp_closed:
            self.unrealized_pnl = self._calc_pnl(self.entry_price, current_price, self.position)
            self.equity = self.balance + self.unrealized_pnl
            self.hold_steps += 1

        # === BUILD OBSERVATION (same as env_trading.py._get_obs) ===
        obs_window = df[FEATURE_COLUMNS].iloc[-WINDOW_SIZE:].values.flatten().astype(np.float32)

        unrealized_ret = 0.0
        unrealized_pips = 0.0
        if self.position != 0 and self.entry_price > 0:
            unrealized_ret = self.position * (current_price - self.entry_price) / self.entry_price
            unrealized_pips = self.position * (current_price - self.entry_price) / PIP_SIZE
            
        total_pnl_pips = self.total_pnl / (PIP_VALUE * LOT_SIZE) if LOT_SIZE > 0 else 0.0

        state_feat = np.array([
            self.position,
            total_pnl_pips / 1000.0,
            unrealized_pips / 100.0,
            min(self.hold_steps / MAX_HOLD_STEPS, 1.0),
            np.clip(unrealized_ret * 100, -5, 5)
        ], dtype=np.float32)

        full_obs = np.concatenate([obs_window, state_feat])
        obs_norm = self.vec_norm.normalize_obs(full_obs)
        action, _ = self.model.predict(obs_norm, deterministic=True)
        action = int(action)

        # === If SL/TP already closed, skip action (same as env) ===
        if sl_tp_closed:
            self.first_bar = True
            return 3  # Tell EA to close (in case MT5 position is still open)

        # === Execute action internally (same as env) ===
        if action == 1:  # BUY
            if self.position == -1:
                self._close(current_price)
                self._open(1, current_price)
                self.first_bar = True
            elif self.position == 0:
                self._open(1, current_price)
                self.first_bar = True
        elif action == 2:  # SELL
            if self.position == 1:
                self._close(current_price)
                self._open(-1, current_price)
                self.first_bar = True
            elif self.position == 0:
                self._open(-1, current_price)
                self.first_bar = True
        elif action == 3:  # CLOSE
            if self.position != 0:
                self._close(current_price)
        else:
            self.first_bar = False

        if action != 1 and action != 2:
            self.first_bar = False

        return action


# ==================================================
# PARSE MT5 DATA
# ==================================================
def parse_mt5_data(data_str):
    try:
        parts = data_str.strip().split(";")
        bars = parts[0].split("|")

        rows = []
        for bar in bars:
            values = bar.split(",")
            rows.append({
                'open': float(values[0]), 'high': float(values[1]),
                'low': float(values[2]), 'close': float(values[3])
            })

        df = pd.DataFrame(rows)

        state_str = parts[1] if len(parts) > 1 else "0,10000,0,0,0,0,0.0,0.1"
        state_values = state_str.split(",")
        delta_tick = int(state_values[5]) if len(state_values) > 5 else 0
        delta_price = float(state_values[6]) if len(state_values) > 6 else 0.0
        lot_size = float(state_values[7]) if len(state_values) > 7 else 0.0

        return df, delta_tick, delta_price, lot_size
    except Exception as e:
        print(f"❌ Parse error: {e}")
        return None, 0, 0.0, 0.0


# ==================================================
# MAIN SERVER
# ==================================================
def main():
    model, vec_norm = load_model()
    bridge = PPOBridge(model, vec_norm)

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    endpoint = f"tcp://{HOST}:{PORT}"
    socket.bind(endpoint)

    bar_count = 0
    action_names = ["HOLD", "BUY", "SELL", "CLOSE"]

    print("\n" + "="*60)
    print("🚀 PPO ZMQ Server v2.0 — Python-Managed State")
    print("="*60)
    print(f"📡 ZMQ Endpoint: {endpoint}")
    print(f"📊 Model: {MODEL_PATH}")
    print(f"🎯 SL={SL_PIPS} pips | TP={TP_PIPS} pips | Lot={LOT_SIZE}")
    print(f"📈 Features: {len(FEATURE_COLUMNS)} | Window: {WINDOW_SIZE}")
    print(f"💡 State tracked internally (same as env_trading.py)")
    print("="*60)
    print("\n⏳ Waiting for MT5 Strategy Tester...\n")

    try:
        while True:
            message = socket.recv()
            data_str = message.decode('utf-8').strip()

            if not data_str:
                socket.send_string("0")
                continue

            df, delta_tick, delta_price, lot_from_ea = parse_mt5_data(data_str)

            if df is None or len(df) < WINDOW_SIZE:
                socket.send_string("0")
                continue

            # Update lot size from EA (if auto-lot is active)
            if lot_from_ea > 0:
                bridge.lot_size = lot_from_ea
                bridge.spread_cost = SPREAD_PIPS * PIP_VALUE * lot_from_ea

            action = bridge.process_bar(df, delta_tick, delta_price)
            bar_count += 1

            p = df['close'].iloc[-1]
            wr = (bridge.wins / bridge.trades * 100) if bridge.trades > 0 else 0

            if bar_count % 50 == 0 or action != 0:
                print(f"  #{bar_count:4d} | {action_names[action]:5s} | "
                      f"Price: {p:.5f} | Pos: {bridge.position} | "
                      f"Eq: ${bridge.equity:.2f} | "
                      f"Trades: {bridge.trades} (WR:{wr:.1f}%) | "
                      f"SL:{bridge.sl_hits} TP:{bridge.tp_hits}")

            socket.send_string(str(action))

    except KeyboardInterrupt:
        ret = (bridge.equity / INITIAL_BALANCE - 1) * 100
        print(f"\n\n{'='*60}")
        print(f"🛑 Server stopped")
        print(f"{'='*60}")
        print(f"📊 Bars processed: {bar_count}")
        print(f"💰 Equity: ${bridge.equity:.2f} ({ret:+.2f}%)")
        print(f"📈 Trades: {bridge.trades} | WR: {(bridge.wins/bridge.trades*100) if bridge.trades > 0 else 0:.1f}%")
        print(f"🔴 SL Hits: {bridge.sl_hits} | 🟢 TP Hits: {bridge.tp_hits}")
        print(f"💸 Fees: ${bridge.total_fees:.2f}")
        print(f"{'='*60}")
        socket.close()
        context.term()


if __name__ == "__main__":
    main()
