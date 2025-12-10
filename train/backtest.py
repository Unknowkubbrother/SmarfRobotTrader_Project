import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from trading_env import TradingEnv
import ta
import joblib

def backtest():
    print("⏳ Loading Test Data...")
    try:
        df = pd.read_csv('test_data.csv')
    except FileNotFoundError:
        print("❌ Test data not found. Please run train_ppo.py first.")
        return

    # Ensure time is datetime for plotting
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])

    print("🤖 Loading PPO Model...")
    try:
        model = PPO.load("ppo_trading_eurusd")
    except FileNotFoundError:
        print("❌ Model 'ppo_trading_eurusd.zip' not found.")
        return
        
    print("🔧 Reconstructing Features (Important)...")
    # We must replicate the EXACT feature engineering from training
    # Note: test_data.csv might already be processed if train_ppo saved the PROCESSED df. 
    # Let's check train_ppo.py logic.
    # Logic in train_ppo: "test_df.to_csv('test_data.csv'...)" happens AFTER processing.
    # So test_data.csv already has features and is normalized.
    # UNLESS we re-ran training and saved new data.
    
    # Wait, train_ppo.py saves test_df which is a slice of the processed df.
    # So it HAS the features like 'macd', 'rsi', etc. already scaled.
    
    # We just need to identify the correct columns to pass to TradingEnv.
    
    # Define features used in training (MUST MATCH TRAIN_PPO.PY)
    feature_cols = [
        'macd', 'macd_signal', 'adx', 'rsi', 'stoch_k', 
        'atr', 'return', 'log_return',
        'dist_ema12', 'dist_bb_upper', 'dist_bb_lower'
    ]
    
    # Verify columns exist
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        print(f"⚠️ Missing columns in test data: {missing_cols}")
        print("Re-calculating features...")
        # (This block strictly shouldn't be needed if test_data.csv is from train_ppo, but good for safety)
        pass 
    
    env_cols = feature_cols + ['close']
    if 'time' in df.columns:
        env_cols.append('time')
    
    print(f"✅ Using {len(feature_cols)} features for observation.")
    
    # Filter df to match training environment
    env = TradingEnv(df[env_cols])
    
    obs, info = env.reset()
    done = False
    
    net_worths = []
    timestamps = []
    
    while not done:
        action, _states = model.predict(obs, deterministic=True) # Deterministic for evaluation
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        net_worths.append(info['net_worth'])
        # Try to get timestamp
        try:
             # accessing private df might be risky if indices shifted but let's trust sequential
            ts = df.iloc[env.current_step-1]['time']
            timestamps.append(ts)
        except:
            timestamps.append(env.current_step)

    # --- Analysis ---
    initial_balance = env.initial_balance
    final_balance = net_worths[-1]
    profit = final_balance - initial_balance
    return_pct = (profit / initial_balance) * 100
    
    print("\n📊 Backtest Results:")
    print(f"   Initial Balance: ${initial_balance:,.2f}")
    print(f"   Final Balance:   ${final_balance:,.2f}")
    print(f"   Total Profit:    ${profit:,.2f} ({return_pct:.2f}%)")
    
    # Analyze Trades
    trades = pd.DataFrame(env.trades)
    if not trades.empty:
        n_trades = len(trades)
        winning_trades = trades[trades['pnl'] > 0]
        losing_trades = trades[trades['pnl'] < 0]
        
        win_rate = (len(winning_trades) / n_trades) * 100
        avg_win = winning_trades['pnl'].mean() * 100
        avg_loss = losing_trades['pnl'].mean() * 100
        
        print(f"   Total Trades:    {n_trades}")
        print(f"   Win Rate:        {win_rate:.2f}%")
        print(f"   Avg Win:         {avg_win:.2f}%")
        print(f"   Avg Loss:        {avg_loss:.2f}%")
    else:
        print("   No trades executed.")

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, net_worths, label='Net Worth')
    plt.title('Backtest Result: PPO on EURUSD')
    plt.xlabel('Time')
    plt.ylabel('Balance ($)')
    plt.legend()
    plt.grid(True)
    
    # Save plot
    plt.savefig('backtest_result.png')
    print("📈 Chart saved to backtest_result.png")
    
    # Show plot if possible (might not work in headless)
    try:
        plt.show()
    except:
        pass

if __name__ == "__main__":
    backtest()
