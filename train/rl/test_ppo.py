"""
test_ppo.py — Bar-by-Bar Backtester (matches zmq_bridge_server.py EXACTLY)

This script processes data the SAME way as zmq_bridge_server.py:
- 200-bar sliding window for feature calculation
- PPOBridge internal state tracking (position, equity, SL/TP)
- Same observation construction

Result should match MT5 Strategy Tester output.
"""

import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from env_trading import TradingEnv

# ==================================================
# CONFIG (must match zmq_bridge_server.py)
# ==================================================
WINDOW_SIZE = 20
INITIAL_BALANCE = 10000
PIP_SIZE = 0.0001
PIP_VALUE = 10.0
SL_PIPS = 30
TP_PIPS = 60
SPREAD_PIPS = 2
MAX_HOLD_STEPS = 30
BAR_HISTORY = 200  # Same as EA sends
RISK_PERCENT = 1.0  # Risk % per trade


def calc_auto_lot(balance, risk_pct=RISK_PERCENT, sl_pips=SL_PIPS,
                  pip_value_per_lot=PIP_VALUE, min_lot=0.01, lot_step=0.01):
    """Risk-based position sizing (same formula as EA)"""
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
# FEATURE ENGINEERING (same as zmq_bridge_server.py)
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
# PPOBridge (same as zmq_bridge_server.py)
# ==================================================
class PPOBridge:
    def __init__(self, model, vec_norm):
        self.model = model
        self.vec_norm = vec_norm
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
        self.max_equity = INITIAL_BALANCE
        self.lot_size = calc_auto_lot(INITIAL_BALANCE)
        self.spread_cost = SPREAD_PIPS * PIP_VALUE * self.lot_size
        self.first_bar = True
        print(f"\n💰 Auto Lot: {self.lot_size} (Balance: ${INITIAL_BALANCE}, Risk: {RISK_PERCENT}%)")

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

    def process_bar(self, df, delta_tick=0, delta_price=0.0):
        df = calculate_features(df, delta_tick, delta_price)
        last_bar = df.iloc[-1]
        current_price = last_bar['close']
        bar_high = last_bar['high']
        bar_low = last_bar['low']

        sl_tp_closed = False

        # SL/TP check (same as server)
        if self.position != 0 and self.entry_price > 0 and not self.first_bar:
            if self.position == 1:
                sl_price = self.entry_price - SL_PIPS * PIP_SIZE
                tp_price = self.entry_price + TP_PIPS * PIP_SIZE
                hit_sl = bar_low <= sl_price
                hit_tp = bar_high >= tp_price
            else:
                sl_price = self.entry_price + SL_PIPS * PIP_SIZE
                tp_price = self.entry_price - TP_PIPS * PIP_SIZE
                hit_sl = bar_high >= sl_price
                hit_tp = bar_low <= tp_price

            if hit_sl and hit_tp:
                self._close(sl_price)
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

        if self.position != 0 and not sl_tp_closed:
            self.unrealized_pnl = self._calc_pnl(self.entry_price, current_price, self.position)
            self.equity = self.balance + self.unrealized_pnl
            self.hold_steps += 1

        self.max_equity = max(self.max_equity, self.equity)

        # Build observation (same as server)
        obs_window = df[FEATURE_COLUMNS].iloc[-WINDOW_SIZE:].values.flatten().astype(np.float32)
        unrealized_ret = 0.0
        unrealized_pips = 0.0
        if self.position != 0 and self.entry_price > 0:
            unrealized_ret = self.position * (current_price - self.entry_price) / self.entry_price
            unrealized_pips = self.position * (current_price - self.entry_price) / PIP_SIZE
            
        total_pnl_pips = self.total_pnl / (PIP_VALUE * self.lot_size) if self.lot_size > 0 else 0.0

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

        if sl_tp_closed:
            self.first_bar = True
            return 3, current_price

        if action == 1:
            if self.position == -1:
                self._close(current_price)
                self._open(1, current_price)
                self.first_bar = True
            elif self.position == 0:
                self._open(1, current_price)
                self.first_bar = True
        elif action == 2:
            if self.position == 1:
                self._close(current_price)
                self._open(-1, current_price)
                self.first_bar = True
            elif self.position == 0:
                self._open(-1, current_price)
                self.first_bar = True
        elif action == 3:
            if self.position != 0:
                self._close(current_price)
        else:
            self.first_bar = False

        if action != 1 and action != 2:
            self.first_bar = False

        return action, current_price


# ==================================================
# MAIN — Bar-by-Bar Backtest
# ==================================================
df = pd.read_csv("h1_ohlc_delta1.csv")
df['time'] = pd.to_datetime(df['time'])
df_full = df.copy()
df = df[df['has_delta'] == 1].sort_values("time").reset_index(drop=True)

print("="*50)
print("📊 DATA FILTERING")
print("="*50)
print(f"ข้อมูลทั้งหมด:     {len(df_full):,} rows")
print(f"ข้อมูลที่มี delta: {len(df):,} rows")
print(f"ช่วงเวลา:          {df['time'].iloc[0]} → {df['time'].iloc[-1]}")
print("="*50 + "\n")

# Load model (same as server)
dummy_data = {
    'time': [pd.Timestamp.now()] * 80,
    'open': [1.0]*80, 'high': [1.0]*80, 'low': [1.0]*80, 'close': [1.0]*80,
    'delta_tick': [0]*80, 'delta_price': [0]*80,
    'sma_cross': [0]*80, 'rsi_norm': [0]*80, 'atr_norm': [0]*80, 'trend': [0]*80
}
mock_df = pd.DataFrame(dummy_data)
dummy_env = DummyVecEnv([lambda: TradingEnv(mock_df, lot_size=calc_auto_lot(INITIAL_BALANCE), sl_pips=SL_PIPS, tp_pips=TP_PIPS)])
vec_norm = VecNormalize.load("vec_normalize.pkl", dummy_env)
vec_norm.training = False
vec_norm.norm_reward = False

model = PPO.load("ppo_trading.zip")
bridge = PPOBridge(model, vec_norm)

# Bar-by-bar processing (same as server receives from EA)
equity_history = [INITIAL_BALANCE]
time_history = [df['time'].iloc[BAR_HISTORY]]
buy_signals = []
sell_signals = []
close_signals = []

print(f"🔧 Processing {len(df) - BAR_HISTORY} bars (200-bar sliding window)...")

for i in range(BAR_HISTORY, len(df)):
    # Extract 200-bar window (same as EA sends)
    window_df = df.iloc[i - BAR_HISTORY:i][['open', 'high', 'low', 'close']].reset_index(drop=True)

    # Get delta values for current bar
    delta_tick = df.iloc[i - 1].get('delta_tick', 0)
    delta_price = df.iloc[i - 1].get('delta_price', 0.0)

    action, price = bridge.process_bar(window_df, delta_tick, delta_price)

    current_time = df['time'].iloc[i]
    equity_history.append(bridge.equity)
    time_history.append(current_time)

    if action == 1:
        buy_signals.append((current_time, bridge.equity))
    elif action == 2:
        sell_signals.append((current_time, bridge.equity))
    elif action == 3:
        close_signals.append((current_time, bridge.equity))

drawdown = (bridge.max_equity - bridge.equity) / bridge.max_equity if bridge.max_equity > 0 else 0
max_dd = max((bridge.max_equity - e) / bridge.max_equity for e in equity_history) if equity_history else 0

print("\n" + "="*50)
print("📈 TEST RESULTS (Server-Aligned Mode)")
print("="*50)
print(f"Final Equity:     ${bridge.equity:.2f}")
print(f"Return:           {((bridge.equity/INITIAL_BALANCE - 1) * 100):.2f}%")
print(f"Total Trades:     {bridge.trades}")
print(f"Win Rate:         {(bridge.wins/bridge.trades*100) if bridge.trades > 0 else 0:.2f}%")
print(f"Max Drawdown:     {max_dd*100:.2f}%")
print(f"Total Fees Paid:  ${bridge.total_fees:.2f}")
print(f"SL Hits:          {bridge.sl_hits}")
print(f"TP Hits:          {bridge.tp_hits}")
print("="*50)
print(f"Start Time: {df['time'].iloc[BAR_HISTORY]}")
print(f"End Time:   {df['time'].iloc[-1]}")

# ==================================================
# VISUALIZATION
# ==================================================
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    print("\n📊 Generating Strategy Tester Graph...")

    plt.figure(figsize=(14, 7))
    plt.title(f"PPO Backtest — Server-Aligned (Return: {((bridge.equity/INITIAL_BALANCE - 1) * 100):.2f}%)")
    plt.plot(time_history, equity_history, label='Equity', color='blue', linewidth=1.5)

    if buy_signals:
        bx, by = zip(*buy_signals)
        plt.scatter(bx, by, marker='^', color='green', s=50, label='Buy', alpha=0.7)
    if sell_signals:
        sx, sy = zip(*sell_signals)
        plt.scatter(sx, sy, marker='v', color='red', s=50, label='Sell', alpha=0.7)
    if close_signals:
        cx, cy = zip(*close_signals)
        plt.scatter(cx, cy, marker='x', color='black', s=30, label='Close', alpha=0.5)

    plt.axhline(y=INITIAL_BALANCE, color='r', linestyle='--', alpha=0.5, label='Initial Balance')
    plt.xlabel('Time')
    plt.ylabel('Equity (USD)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gcf().autofmt_xdate()

    plt.savefig('strategy_tester_results.png', dpi=300, bbox_inches='tight')
    print("✅ Graph saved as 'strategy_tester_results.png'")

except ImportError:
    print("\n⚠️  matplotlib not installed.")