import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from env_trading import TradingEnv

df = pd.read_csv("h1_ohlc_delta1.csv")

# ==================================================
# FILTER: ใช้เฉพาะข้อมูลที่มี delta (2024-2025)
# ==================================================
df_full = df.copy()
df['time'] = pd.to_datetime(df['time'])
df = df[df['has_delta'] == 1].sort_values("time").reset_index(drop=True)

print("="*50)
print("📊 DATA FILTERING")
print("="*50)
print(f"ข้อมูลทั้งหมด:     {len(df_full):,} rows")
print(f"ข้อมูลที่มี delta: {len(df):,} rows")
print(f"ช่วงเวลา:          {df['time'].iloc[0]} → {df['time'].iloc[-1]}")
print("="*50 + "\n")

# ==================================================
# FEATURE ENGINEERING (เฉพาะที่สำคัญ)
# ==================================================
print("🔧 Creating features...")

# Basic features
df['return'] = df['close'].pct_change().fillna(0)
df['range'] = (df['high'] - df['low']) / df['close']
df['raw_return'] = df['return']

# Body Ratio
full_range = df['high'] - df['low']
df['body_ratio'] = np.where(full_range > 0, abs(df['close'] - df['open']) / full_range, 0)

# Momentum
df['momentum'] = df['return'].rolling(window=5).sum().fillna(0)

# ===== Trend Indicators (ต้องตรงกับ train_ppo.py) =====
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

df = df.iloc[50:].reset_index(drop=True)

print(f"✅ Features: return, range, delta_tick, delta_price, body_ratio, momentum")
print("="*50 + "\n")


test_env  = DummyVecEnv([lambda: TradingEnv(df)])
test_env = VecNormalize.load("vec_normalize.pkl", test_env)
test_env.training = False
test_env.norm_reward = False
test_env.clip_obs = 10.

# Load Model
model = PPO.load("ppo_trading.zip")

obs = test_env.reset()
done = False
total_reward = 0
positions_held = []
trades = 0
wins = 0

# For plotting
equity_history = [10000]
time_history = [df['time'].iloc[0]]
buy_signals = []
sell_signals = []
close_signals = []

step_idx = 0
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = test_env.step(action)
    total_reward += reward[0]
    
    current_time = df['time'].iloc[min(step_idx, len(df)-1)]
    equity_history.append(info[0]['equity'])
    time_history.append(current_time)
    
    # Track actions for plotting
    action_val = action[0]
    price = df['close'].iloc[min(step_idx, len(df)-1)]
    
    if action_val == 1: # BUY
        buy_signals.append((current_time, equity_history[-1]))
    elif action_val == 2: # SELL
        sell_signals.append((current_time, equity_history[-1]))
    elif action_val == 3: # CLOSE
        close_signals.append((current_time, equity_history[-1]))

    if info[0]['position'] != 0:
        positions_held.append(info[0]['hold_steps'])
    
    if info[0]['trades'] > trades:
        trades = info[0]['trades']
        wins = info[0]['wins']
        
    step_idx += 1

avg_hold = np.mean(positions_held) if positions_held else 0

print("\n" + "="*50)
print("📈 TEST RESULTS (Strategy Tester Mode)")
print("="*50)
print(f"Total Reward:     {total_reward:.4f}")
print(f"Final Equity:     ${info[0]['equity']:.2f}")
print(f"Return:           {((info[0]['equity']/10000 - 1) * 100):.2f}%")
print(f"Total Trades:     {trades}")
print(f"Win Rate:         {(wins/trades*100) if trades > 0 else 0:.2f}%")
print(f"Max Drawdown:     {info[0]['drawdown']*100:.2f}%")
print(f"Avg Hold Steps:   {avg_hold:.1f}")
print(f"Total Fees Paid:  ${info[0]['fees']:.2f}")
print(f"SL Hits:          {info[0].get('sl_hits', 0)}")
print(f"TP Hits:          {info[0].get('tp_hits', 0)}")
print("="*50)

# Sharpe Ratio calculation
if len(df) > 0:
    returns = df['return'].values
    sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
    print(f"Sharpe Ratio:     {sharpe:.2f}")
    print("="*50)

print(f"Start Time: {df['time'].iloc[0]}")
print(f"End Time:   {df['time'].iloc[-1]}")

# ==================================================
# VISUALIZATION (เหมือน MT5 Strategy Tester)
# ==================================================
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    
    print("\n📊 Generating Strategy Tester Graph...")
    
    plt.figure(figsize=(14, 7))
    plt.title(f"PPO Backtest Equity Curve (Return: {((info[0]['equity']/10000 - 1) * 100):.2f}%)")
    plt.plot(time_history, equity_history, label='Equity', color='blue', linewidth=1.5)
    
    # Plot trade signals on equity curve
    if buy_signals:
        bx, by = zip(*buy_signals)
        plt.scatter(bx, by, marker='^', color='green', s=50, label='Buy', alpha=0.7)
    if sell_signals:
        sx, sy = zip(*sell_signals)
        plt.scatter(sx, sy, marker='v', color='red', s=50, label='Sell', alpha=0.7)
    if close_signals:
        cx, cy = zip(*close_signals)
        plt.scatter(cx, cy, marker='x', color='black', s=30, label='Close', alpha=0.5)
        
    plt.axhline(y=10000, color='r', linestyle='--', alpha=0.5, label='Initial Balance')
    
    plt.xlabel('Time')
    plt.ylabel('Equity (USD)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Formatter
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gcf().autofmt_xdate()
    
    # Save & Show
    plt.savefig('strategy_tester_results.png', dpi=300, bbox_inches='tight')
    print("✅ Graph saved as 'strategy_tester_results.png'")
    # plt.show() # Uncomment if running in notebook/GUI
    
except ImportError:
    print("\n⚠️  matplotlib not installed. Graph not generated.")
    print("👉 Install with: pip install matplotlib")