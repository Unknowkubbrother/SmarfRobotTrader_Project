import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from env_trading import TradingEnv

df = pd.read_csv("h1_ohlc_delta_eurusd.csv")

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

# Body Ratio - วัด strength ของแท่งเทียน
full_range = df['high'] - df['low']
df['body_ratio'] = np.where(full_range > 0, abs(df['close'] - df['open']) / full_range, 0)

# Momentum (sum of recent returns)
df['momentum'] = df['return'].rolling(window=5).sum().fillna(0)

print(f"✅ Features: return, range, delta_tick, delta_price, has_delta, body_ratio, momentum")
print("="*50 + "\n")

# split 70/30
split = int(len(df) * 0.7)
train_df = df.iloc[:split].copy()
test_df  = df.iloc[split:].copy()

print(f"📈 Train: {len(test_df):,} rows | Test: {len(test_df):,} rows\n")
test_env  = DummyVecEnv([lambda: TradingEnv(test_df)])
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

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = test_env.step(action)
    total_reward += reward[0]

    if info[0]['position'] != 0:
        positions_held.append(info[0]['hold_steps'])
    
    if info[0]['trades'] > trades:
        trades = info[0]['trades']
        wins = info[0]['wins']

avg_hold = np.mean(positions_held) if positions_held else 0

print("\n" + "="*50)
print("📈 TEST RESULTS")
print("="*50)
print(f"Total Reward:     {total_reward:.4f}")
print(f"Final Equity:     ${info[0]['equity']:.2f}")
print(f"Return:           {((info[0]['equity']/10000 - 1) * 100):.2f}%")
print(f"Total Trades:     {trades}")
print(f"Win Rate:         {(wins/trades*100) if trades > 0 else 0:.2f}%")
print(f"Max Drawdown:     {info[0]['drawdown']*100:.2f}%")
print(f"Avg Hold Steps:   {avg_hold:.1f}")
print(f"Total Fees Paid:  ${info[0]['fees']:.2f}")
print("="*50)

# Sharpe Ratio calculation
if len(test_df) > 0:
    returns = test_df['return'].values
    sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
    print(f"Sharpe Ratio:     {sharpe:.2f}")
    print("="*50)

print(f"start time = {test_df.head(1)}")
print(f"end time = {test_df.tail(1)}")