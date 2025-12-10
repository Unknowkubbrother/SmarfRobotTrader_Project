# ==========================================
# ไฟล์: train_ppo.py
# วัตถุประสงค์: เทรนโมเดล PPO (Proximal Policy Optimization) สำหรับเทรด Forex
# ==========================================

import torch
import gymnasium as gym
import pandas as pd
import numpy as np
from stable_baselines3 import PPO  # อัลกอริทึม Reinforcement Learning
from stable_baselines3.common.vec_env import DummyVecEnv  # สำหรับห่อ environment
import os
from trading_env import TradingEnv  # Custom trading environment ที่เราสร้างเอง
import ta  # ไลบรารีสำหรับคำนวณ Technical Indicators
from sklearn.preprocessing import StandardScaler  # สำหรับ normalize ข้อมูล
import joblib  # สำหรับบันทึก scaler

def load_and_process_data(filepath, save_scaler=True):
    """
    ฟังก์ชันโหลดและประมวลผลข้อมูล
    - โหลดข้อมูลจาก CSV
    - คำนวณ Technical Indicators
    - Normalize ข้อมูล
    - บันทึก scaler สำหรับใช้ในการ backtest
    """
    print("⏳ Loading data...")
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print("❌ CSV file not found.")
        return None, None

    # แปลงคอลัมน์ time เป็น datetime และเรียงตามเวลา
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])
        df = df.sort_values('time').reset_index(drop=True)
    
    print("🔧 Advanced Feature Engineering...")
    
    # ==========================================
    # 1. Trend Indicators (ตัวบอกแนวโน้ม)
    # ==========================================
    # EMA (Exponential Moving Average) - ค่าเฉลี่ยเคลื่อนที่แบบเอ็กซ์โพเนนเชียล
    df['ema_12'] = ta.trend.EMAIndicator(close=df['close'], window=12).ema_indicator()
    df['ema_26'] = ta.trend.EMAIndicator(close=df['close'], window=26).ema_indicator()
    
    # MACD (Moving Average Convergence Divergence) - ตัวบอกความแตกต่างของค่าเฉลี่ยเคลื่อนที่
    df['macd'] = ta.trend.MACD(close=df['close']).macd()
    df['macd_signal'] = ta.trend.MACD(close=df['close']).macd_signal()
    
    # ADX (Average Directional Index) - ตัวบอกความแรงของแนวโน้ม
    df['adx'] = ta.trend.ADXIndicator(high=df['high'], low=df['low'], close=df['close']).adx()
    
    # ==========================================
    # 2. Momentum Indicators (ตัวบอกโมเมนตัม)
    # ==========================================
    # RSI (Relative Strength Index) - ตัวบอกความแรงสัมพัทธ์ (0-100)
    df['rsi'] = ta.momentum.RSIIndicator(close=df['close']).rsi()
    
    # Stochastic Oscillator - ตัวบอกโมเมนตัมแบบ Stochastic
    df['stoch_k'] = ta.momentum.StochasticOscillator(high=df['high'], low=df['low'], close=df['close']).stoch()
    
    # ==========================================
    # 3. Volatility Indicators (ตัวบอกความผันผวน)
    # ==========================================
    # Bollinger Bands - แถบบอลลิงเจอร์ (บอกช่วงราคาที่คาดว่าจะเคลื่อนไหว)
    df['bb_upper'] = ta.volatility.BollingerBands(close=df['close']).bollinger_hband()
    df['bb_lower'] = ta.volatility.BollingerBands(close=df['close']).bollinger_lband()
    
    # ATR (Average True Range) - ค่าเฉลี่ยของช่วงราคาที่แท้จริง (บอกความผันผวน)
    df['atr'] = ta.volatility.AverageTrueRange(high=df['high'], low=df['low'], close=df['close']).average_true_range()
    
    # ==========================================
    # 4. Custom Features (ฟีเจอร์ที่สร้างเอง)
    # ==========================================
    # Return - ผลตอบแทนแบบเปอร์เซ็นต์
    df['return'] = df['close'].pct_change()
    
    # Log Return - ผลตอบแทนแบบ logarithm (ดีกว่าสำหรับการคำนวณทางสถิติ)
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    
    # คำนวณระยะห่างของราคาจาก indicators (ทำให้ข้อมูลเป็น stationary)
    # ระยะห่างจาก EMA 12
    df['dist_ema12'] = (df['close'] - df['ema_12']) / df['ema_12']
    # ระยะห่างจาก Bollinger Band บน
    df['dist_bb_upper'] = (df['bb_upper'] - df['close']) / df['close']
    # ระยะห่างจาก Bollinger Band ล่าง
    df['dist_bb_lower'] = (df['close'] - df['bb_lower']) / df['close']
    
    # ลบแถวที่มีค่า NaN (เกิดจากการคำนวณ indicators)
    df = df.dropna().reset_index(drop=True)
    
    # กำหนดฟีเจอร์ที่จะใช้ในการเทรน (ไม่รวม close, time, open, high, low)
    feature_cols = [
        'macd', 'macd_signal', 'adx', 'rsi', 'stoch_k', 
        'atr', 'return', 'log_return',
        'dist_ema12', 'dist_bb_upper', 'dist_bb_lower'
    ]
    
    print(f"   Selected {len(feature_cols)} features: {feature_cols}")

    # ==========================================
    # Standardize Features (ปรับมาตรฐานข้อมูล)
    # ==========================================
    # ทำให้ข้อมูลมีค่าเฉลี่ย = 0 และ standard deviation = 1
    # สำคัญมากสำหรับ Neural Network เพื่อให้เทรนได้เร็วและดีขึ้น
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    # บันทึก Scaler เพื่อใช้ในการ Backtest (ต้องใช้ scaler ตัวเดียวกัน)
    if save_scaler:
        joblib.dump(scaler, 'scaler.pkl')
    
    return df, feature_cols

def train_ppo():
    """
    ฟังก์ชันหลักสำหรับเทรนโมเดล PPO
    - โหลดและประมวลผลข้อมูล
    - แบ่งข้อมูลเป็น Train/Test
    - สร้าง Environment
    - เทรนโมเดล
    - บันทึกโมเดลและข้อมูลทดสอบ
    """
    # โหลดและประมวลผลข้อมูล
    df, feature_cols = load_and_process_data('EURUSD_2009_to_present.csv')
    if df is None: return

    # ==========================================
    # แบ่งข้อมูล: 80% สำหรับเทรน, 20% สำหรับทดสอบ
    # ==========================================
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].reset_index(drop=True)  # 80% แรก
    test_df = df.iloc[split_idx:].reset_index(drop=True)   # 20% หลัง
    
    print(f"📊 Total Data: {len(df)}")
    print(f"   - Train: {len(train_df)}")
    print(f"   - Test:  {len(test_df)}")
    
    # ==========================================
    # สร้าง Trading Environment
    # ==========================================
    # ส่งฟีเจอร์ที่ต้องการ + close (สำหรับคำนวณ PnL) + time (สำหรับ logging)
    env_cols = feature_cols + ['close', 'time']
    train_env = DummyVecEnv([lambda: TradingEnv(train_df[env_cols])])
    
    # ==========================================
    # สร้างโมเดล PPO
    # ==========================================
    print("🤖 Initializing PPO model...")
    # MlpPolicy = Multi-Layer Perceptron (Neural Network แบบธรรมดา)
    # verbose=1 = แสดงความคืบหน้าระหว่างเทรน
    # tensorboard_log = บันทึก log สำหรับดูใน TensorBoard
    model = PPO("MlpPolicy", train_env, verbose=1, tensorboard_log="./ppo_trading_tensorboard/")
    
    # ==========================================
    # เริ่มเทรนโมเดล
    # ==========================================
    print("🚀 Starting training...")
    try:
        # total_timesteps = จำนวน step ทั้งหมดที่จะเทรน
        # ยิ่งมากยิ่งดี แต่ใช้เวลานาน (50,000 เป็นค่าที่ใช้สำหรับ demo)
        model.learn(total_timesteps=50000)
        print("✅ Training finished!")
    except KeyboardInterrupt:
        print("\n⏹️ Training stopped manually.")
        
    # ==========================================
    # บันทึกโมเดลที่เทรนเสร็จแล้ว
    # ==========================================
    model.save("ppo_trading_eurusd")
    print("💾 Model saved to ppo_trading_eurusd.zip")
    
    # ==========================================
    # บันทึกข้อมูลทดสอบสำหรับ Backtesting
    # ==========================================
    test_df.to_csv('test_data.csv', index=False)
    print("💾 Saved test data for backtesting.")

# เริ่มต้นโปรแกรม
if __name__ == "__main__":
    train_ppo()
