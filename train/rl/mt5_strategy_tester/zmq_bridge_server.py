"""
PPO ZMQ Bridge Server — Python Server for MT5 Strategy Tester

วิธีใช้ (ZMQ Mode):
1. ตรวจสอบว่าติดตั้ง pyzmq แล้ว: pip install pyzmq
2. รัน script นี้: python zmq_bridge_server.py
3. เปิด MT5 → Strategy Tester → เลือก ZmqBridgeEA
4. ตั้งค่า: Symbol=EURUSD, Period=H1, Date range ที่ต้องการ
5. กด Start

หมายเหตุ: ฝั่ง MT5 ต้องลง "mql-zmq" library ก่อน (libzmq.dll และ Include/Zmq)
"""

import zmq
import sys
import os
import numpy as np
import pandas as pd
import time

# Add parent directory for env_trading import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from env_trading import TradingEnv
from datetime import datetime

# ==================================================
# CONFIG
# ==================================================
HOST = "0.0.0.0"      # Listen on all interfaces
PORT = 5555            # Must match EA setting

# Paths relative to this script's location
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
MODEL_PATH = os.path.join(BASE_DIR, "ppo_trading")
VEC_NORM_PATH = os.path.join(BASE_DIR, "vec_normalize.pkl")
WINDOW_SIZE = 20
INITIAL_BALANCE = 10000

# ==================================================
# LOAD MODEL
# ==================================================
def load_model():
    print("📦 Loading PPO model...")
    
    dummy_data = {
        'time': [datetime.now()] * 80,
        'open': [1.0]*80, 'high': [1.0]*80, 'low': [1.0]*80, 'close': [1.0]*80,
        'delta_tick': [0]*80, 'delta_price': [0]*80, 'has_delta': [0]*80,
        'sma_cross': [0]*80, 'rsi_norm': [0]*80, 'atr_norm': [0]*80, 'trend': [0]*80
    }
    mock_df = pd.DataFrame(dummy_data)
    dummy_env = DummyVecEnv([lambda: TradingEnv(mock_df)])
    
    vec_norm = VecNormalize.load(VEC_NORM_PATH, dummy_env)
    vec_norm.training = False
    vec_norm.norm_reward = False
    
    model = PPO.load(MODEL_PATH)
    
    print("✅ Model loaded successfully")
    return model, vec_norm

# ==================================================
# FEATURE ENGINEERING
# ==================================================
def calculate_features(df):
    """Calculate all 11 features from OHLC data"""
    df['return'] = df['close'].pct_change().fillna(0)
    df['range'] = (df['high'] - df['low']) / df['close']
    
    full_range = df['high'] - df['low']
    df['body_ratio'] = np.where(full_range > 0, abs(df['close'] - df['open']) / full_range, 0)
    df['momentum'] = df['return'].rolling(window=5).sum().fillna(0)
    
    df['delta_tick'] = 0
    df['delta_price'] = 0.0
    df['has_delta'] = 0
    
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

FEATURE_COLUMNS = [
    'return', 'range', 'delta_tick', 'delta_price', 'has_delta',
    'body_ratio', 'momentum',
    'sma_cross', 'rsi_norm', 'atr_norm', 'trend'
]

# ==================================================
# PARSE DATA & PREDICT
# ==================================================
def parse_mt5_data(data_str):
    try:
        parts = data_str.strip().split(";")
        bars_str = parts[0]
        bars = bars_str.split("|")
        
        rows = []
        for bar in bars:
            values = bar.split(",")
            rows.append({
                'open': float(values[0]), 'high': float(values[1]),
                'low': float(values[2]), 'close': float(values[3])
            })
        
        df = pd.DataFrame(rows)
        
        state_str = parts[1] if len(parts) > 1 else "0,10000,0,0,0"
        state_values = state_str.split(",")
        position, equity, unrealized_pnl, hold_steps, entry_price = (
            int(state_values[0]), float(state_values[1]), float(state_values[2]),
            int(state_values[3]), float(state_values[4])
        )
        return df, position, equity, unrealized_pnl, hold_steps, entry_price
    except Exception as e:
        print(f"❌ Parse error: {e}")
        return None, 0, INITIAL_BALANCE, 0, 0, 0

def predict_action(model, vec_norm, df, position, equity, unrealized_pnl, hold_steps, entry_price):
    df = calculate_features(df)
    obs_window = df[FEATURE_COLUMNS].iloc[-WINDOW_SIZE:].values.flatten().astype(np.float32)
    
    unrealized_ret = 0.0
    if position != 0 and entry_price > 0:
        current_price = df['close'].iloc[-1]
        unrealized_ret = position * (current_price - entry_price) / entry_price
    
    state_feat = np.array([
        position, equity / INITIAL_BALANCE if INITIAL_BALANCE > 0 else 1.0,
        unrealized_pnl / INITIAL_BALANCE if INITIAL_BALANCE > 0 else 0,
        min(hold_steps / 30.0, 1.0), np.clip(unrealized_ret * 100, -5, 5)
    ], dtype=np.float32)
    
    full_obs = np.concatenate([obs_window, state_feat])
    obs_norm = vec_norm.normalize_obs(full_obs)
    action, _ = model.predict(obs_norm, deterministic=True)
    return int(action)

# ==================================================
# MAIN SERVER
# ==================================================
def main():
    model, vec_norm = load_model()
    
    context = zmq.Context()
    socket = context.socket(zmq.REP)  # Reply socket
    endpoint = f"tcp://{HOST}:{PORT}"
    socket.bind(endpoint)
    
    trade_count = 0
    action_names = ["HOLD", "BUY", "SELL", "CLOSE"]
    
    print("\n" + "="*50)
    print("🚀 PPO ZMQ Server Running")
    print("="*50)
    print(f"📡 ZMQ Endpoint: {endpoint}")
    print(f"📊 Model: {MODEL_PATH}")
    print(f"🎯 Features: {len(FEATURE_COLUMNS)} | Window: {WINDOW_SIZE}")
    print("="*50)
    print("\n⏳ Waiting for ZMQ connections from MT5 Strategy Tester...")
    
    try:
        while True:
            # Wait for next request from client
            message = socket.recv()
            data_str = message.decode('utf-8').strip()
            
            if not data_str:
                socket.send_string("0")
                continue
                
            df, position, equity, unrealized_pnl, hold_steps, entry_price = parse_mt5_data(data_str)
            
            if df is None or len(df) < WINDOW_SIZE:
                socket.send_string("0")
                continue
                
            action = predict_action(model, vec_norm, df, position, equity, unrealized_pnl, hold_steps, entry_price)
            trade_count += 1
            
            p = df['close'].iloc[-1]
            if action != 0:
                print(f"  #{trade_count:4d} | {action_names[action]:5s} | Price: {p:.5f} | Pos: {position} | Equity: ${equity:.2f}")
                
            # Send reply back to client
            socket.send_string(str(action))
            
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped")
        print(f"📊 Total signals processed: {trade_count}")
        socket.close()
        context.term()

if __name__ == "__main__":
    main()
