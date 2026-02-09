import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from env_trading import TradingEnv


# ==================================================
# CUSTOM TENSORBOARD CALLBACK
# ==================================================
class TradingMetricsCallback(BaseCallback):
    """
    Log trading metrics ลง TensorBoard:
    - Equity/Profit/Loss
    - Actions: Hold, Buy(ทำนายขึ้น), Sell(ทำนายลง), Close
    - Win rate, Accuracy
    """
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_equities = []
        self.episode_trades = []
        self.episode_wins = []
        self.episode_drawdowns = []
        self.episode_accuracies = []
        
        # Action counters: 0=Hold, 1=Buy, 2=Sell, 3=Close
        self.action_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        self.total_actions = 0
        
        self.current_equity = 10000
        self.current_accuracy = 0
        self.step_count = 0
    
    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [{}])
        actions = self.locals.get("actions", [])
        
        for i, info in enumerate(infos):
            if info:
                self.current_equity = info.get("equity", 10000)
                self.current_accuracy = info.get("accuracy", 0)
                
                if len(actions) > i:
                    action = int(actions[i])
                    self.action_counts[action] = self.action_counts.get(action, 0) + 1
                    self.total_actions += 1
        
        self.step_count += 1
        if self.step_count % 500 == 0:
            self._log_metrics()
        
        return True
    
    def _on_rollout_end(self) -> None:
        infos = self.locals.get("infos", [{}])
        for info in infos:
            if info:
                self.episode_equities.append(info.get("equity", 10000))
                self.episode_trades.append(info.get("trades", 0))
                self.episode_wins.append(info.get("wins", 0))
                self.episode_drawdowns.append(info.get("drawdown", 0))
                self.episode_accuracies.append(info.get("accuracy", 0))
        self._log_metrics()
    
    def _log_metrics(self):
        # Trading metrics
        self.logger.record("trading/equity", self.current_equity)
        self.logger.record("trading/profit_loss", self.current_equity - 10000)
        self.logger.record("trading/return_pct", ((self.current_equity / 10000) - 1) * 100)
        self.logger.record("trading/accuracy", self.current_accuracy)
        
        # Action Distribution (4 Actions)
        if self.total_actions > 0:
            hold_pct = (self.action_counts.get(0, 0) / self.total_actions) * 100
            buy_pct = (self.action_counts.get(1, 0) / self.total_actions) * 100
            sell_pct = (self.action_counts.get(2, 0) / self.total_actions) * 100
            close_pct = (self.action_counts.get(3, 0) / self.total_actions) * 100
            
            self.logger.record("actions/hold_pct", hold_pct)
            self.logger.record("actions/buy_pct", buy_pct)
            self.logger.record("actions/sell_pct", sell_pct)
            self.logger.record("actions/close_pct", close_pct)
        
        # Statistics
        if self.episode_equities:
            self.logger.record("trading/avg_equity", np.mean(self.episode_equities[-100:]))
            self.logger.record("trading/max_equity", np.max(self.episode_equities[-100:]))
        
        if self.episode_trades:
            total_trades = sum(self.episode_trades[-100:])
            total_wins = sum(self.episode_wins[-100:])
            win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0
            self.logger.record("trading/total_trades", total_trades)
            self.logger.record("trading/win_rate", win_rate)
        
        if self.episode_accuracies:
            self.logger.record("trading/avg_accuracy", np.mean(self.episode_accuracies[-100:]))
        
        if self.episode_drawdowns:
            self.logger.record("trading/max_drawdown", np.max(self.episode_drawdowns[-100:]) * 100)

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

print(f"📈 Train: {len(train_df):,} rows | Test: {len(test_df):,} rows\n")

train_env = DummyVecEnv([lambda: TradingEnv(train_df)])
train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.)

test_env  = DummyVecEnv([lambda: TradingEnv(test_df)])
# For test env, we initialize it but will copy stats later
test_env = VecNormalize(test_env, training=False, norm_reward=False, clip_obs=10.)

model = PPO(
    "MlpPolicy",
    train_env,
    learning_rate=1e-4,          # ลดลง ให้เรียนช้าๆ
    n_steps=2048,                # เพิ่มขึ้น
    batch_size=128,              # เพิ่มขึ้น
    n_epochs=5,                  # ลดลง (ลด overfitting)
    gamma=0.95,                  # ลดลง (focus short-term)
    gae_lambda=0.9,              # ลดลง
    clip_range=0.3,              # เพิ่มขึ้น (ให้ยืดหยุ่นมากขึ้น)
    ent_coef=0.02,               # เพิ่มขึ้นมาก (explore มากขึ้น)
    vf_coef=0.5,
    max_grad_norm=0.5,
    seed=42,                
    policy_kwargs=dict(net_arch=[32, 32]),  # Network เล็กมาก (ลด overfitting)
    verbose=1,
    tensorboard_log="./tensorboard/"
)

# สร้าง callback สำหรับ log trading metrics
trading_callback = TradingMetricsCallback(verbose=1)

print("\n" + "="*50)
print("🚀 Starting Training (Anti-Overfitting Mode)")
print("="*50)
print("📊 TensorBoard: tensorboard --logdir=./tensorboard/")
print("="*50 + "\n")

# ลด timesteps เพื่อไม่ให้ overfit
model.learn(total_timesteps=300_000, callback=trading_callback)
model.save("ppo_trading")
train_env.save("vec_normalize.pkl")

# Sync normalization stats from train to test
test_env.obs_rms = train_env.obs_rms

# evaluate
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