import pandas as pd
from stable_baselines3 import PPO
from env_trading import TradingEnv

df = pd.read_csv("h1_ohlc_delta.csv")

df['return'] = df['close'].pct_change().fillna(0)
df['range'] = (df['high'] - df['low']) / df['close']

env = TradingEnv(df)

model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
    verbose=1
)

model.learn(total_timesteps=500_000)
model.save("ppo_trading")
