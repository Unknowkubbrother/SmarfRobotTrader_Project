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
        
        # SL/TP hits
        infos = self.locals.get("infos", [{}])
        for info in infos:
            if info:
                self.logger.record("trading/sl_hits", info.get("sl_hits", 0))
                self.logger.record("trading/tp_hits", info.get("tp_hits", 0))

df = pd.read_csv("h1_ohlc_delta.csv")

# ==================================================
# FILTER: ใช้เฉพาะข้อมูลที่มี delta + ข้อมูลล่าสุด
# ==================================================
df_full = df.copy()
df['time'] = pd.to_datetime(df['time'])
df = df[df['has_delta'] == 1].sort_values("time").reset_index(drop=True)

# ใช้เฉพาะข้อมูลตั้งแต่ 2020 — ตลาดเก่าเกินไปไม่ช่วย
df = df[df['time'] >= '2020-01-01'].reset_index(drop=True)

print("="*50)
print("📊 DATA FILTERING (Recent Only)")
print("="*50)
print(f"ข้อมูลทั้งหมด:     {len(df_full):,} rows")
print(f"ข้อมูลที่ใช้ train: {len(df):,} rows (2020+, has_delta)")
print(f"ช่วงเวลา:          {df['time'].iloc[0]} → {df['time'].iloc[-1]}")
print("="*50 + "\n")

# ==================================================
# FEATURE ENGINEERING (รวม Trend Indicators)
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

# ===== NEW: Trend Indicators =====
# SMA Cross — ราคาอยู่เหนือ/ต่ำกว่า SMA20
sma20 = df['close'].rolling(20).mean()
sma50 = df['close'].rolling(50).mean()
df['sma_cross'] = np.where(sma20 > sma50, 1, np.where(sma20 < sma50, -1, 0))
df['sma_cross'] = df['sma_cross'].fillna(0)

# RSI (normalized to -1 to 1)
delta = df['close'].diff()
gain = delta.clip(lower=0).rolling(14).mean()
loss = (-delta.clip(upper=0)).rolling(14).mean()
rs = gain / (loss + 1e-10)
rsi = 100 - (100 / (1 + rs))
df['rsi_norm'] = ((rsi - 50) / 50).fillna(0)  # -1 to 1

# ATR (normalized by price)
tr = np.maximum(
    df['high'] - df['low'],
    np.maximum(
        abs(df['high'] - df['close'].shift(1)),
        abs(df['low'] - df['close'].shift(1))
    )
)
df['atr_norm'] = (tr.rolling(14).mean() / df['close']).fillna(0)

# Trend direction — momentum of SMA20
df['trend'] = (sma20.pct_change(5) * 100).fillna(0)  # % change over 5 bars
df['trend'] = df['trend'].clip(-2, 2)  # Clip extremes

# ADX (Average Directional Index) — วัดความแรงของเทรนด์
tr_adx = np.maximum(
    df['high'] - df['low'],
    np.maximum(
        abs(df['high'] - df['close'].shift(1)),
        abs(df['low'] - df['close'].shift(1))
    )
)
plus_dm = np.where((df['high'] - df['high'].shift(1)) > (df['low'].shift(1) - df['low']),
                    np.maximum(df['high'] - df['high'].shift(1), 0), 0)
minus_dm = np.where((df['low'].shift(1) - df['low']) > (df['high'] - df['high'].shift(1)),
                     np.maximum(df['low'].shift(1) - df['low'], 0), 0)
atr14_adx = pd.Series(tr_adx).rolling(14).mean()
plus_di = 100 * pd.Series(plus_dm).rolling(14).mean() / (atr14_adx + 1e-10)
minus_di = 100 * pd.Series(minus_dm).rolling(14).mean() / (atr14_adx + 1e-10)
dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
adx_raw = dx.rolling(14).mean()
df['adx'] = ((adx_raw - 25) / 25).fillna(0).clip(-1, 1)  # Normalize: <25=ranging(-), >25=trending(+)

# Drop rows with NaN from rolling calculations
df = df.iloc[50:].reset_index(drop=True)

FEATURE_LIST = TradingEnv._get_feature_columns()
print(f"✅ Features ({len(FEATURE_LIST)}): {', '.join(FEATURE_LIST)}")
print("="*50 + "\n")

train_env = DummyVecEnv([lambda: TradingEnv(df, random_start=True, lot_size=0.1, sl_pips=50, tp_pips=50)])
train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.)

model = PPO(
    "MlpPolicy",
    train_env,
    learning_rate=2e-4,
    n_steps=4096,
    batch_size=256,
    n_epochs=8,
    gamma=0.97,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.005,
    vf_coef=0.5,
    max_grad_norm=0.5,
    seed=42,
    policy_kwargs=dict(net_arch=[64, 64]),  # ใหญ่ขึ้นเพราะ features เยอะขึ้น
    verbose=1,
    tensorboard_log="./tensorboard/"
)

trading_callback = TradingMetricsCallback(verbose=1)

print("\n" + "="*50)
print("🚀 Starting Training (Trend-Aware Mode)")
print("="*50)
print("📊 TensorBoard: tensorboard --logdir=./tensorboard/")
print(f"🎯 SL=50 pips | TP=50 pips | Lot=0.1 | MaxHold=30 | RandomStart=ON")
print(f"📈 Features: {len(FEATURE_LIST)} (incl. SMA, RSI, ATR, Trend, ADX)")
print("="*50 + "\n")

model.learn(total_timesteps=1_000_000, callback=trading_callback)
model.save("ppo_trading")
train_env.save("vec_normalize.pkl")