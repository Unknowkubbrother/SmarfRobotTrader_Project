import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from env_trading import TradingEnv

df = pd.read_csv("h1_ohlc_delta1.csv")

df['return'] = df['close'].pct_change().fillna(0)
df['range'] = (df['high'] - df['low']) / df['close']
df['raw_return'] = df['return']

# split
split = int(len(df) * 0.7)
train_df = df.iloc[:split].copy()
test_df  = df.iloc[split:].copy()

train_env = DummyVecEnv([lambda: TradingEnv(train_df)])
train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.)

test_env  = DummyVecEnv([lambda: TradingEnv(test_df)])
# For test env, we initialize it but will copy stats later
test_env = VecNormalize(test_env, training=False, norm_reward=False, clip_obs=10.)

model = PPO(
    "MlpPolicy",
    train_env,
    learning_rate=1e-4,
    n_steps=4096,          
    batch_size=128,        
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01, 
    seed=42,                
    policy_kwargs=dict(net_arch=[128, 128]), # Larger network
    verbose=1,
    tensorboard_log="./tensorboard/"
)

model.learn(total_timesteps=500_000)
model.save("ppo_trading")

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
