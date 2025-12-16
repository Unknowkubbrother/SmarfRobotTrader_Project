import pandas as pd
from stable_baselines3 import PPO
from trading_env import TradingEnv # ต้องแน่ใจว่าไฟล์นี้คือเวอร์ชันล่าสุด
import numpy as np
from tqdm import tqdm

# --- 1. โหลดข้อมูล Features และแบ่งเป็น Train/Test ---
print("กำลังโหลดข้อมูล Features เพื่อทดสอบ...")
df = pd.read_csv('features_for_rl.csv', index_col='time', parse_dates=['time'])

numeric_cols = [
    'log_return', 'volatility_60', 'mom_60', 
    'rel_vol', 'rsi_14', 'dist_sma_60', 'mid_price'
]
available_cols = [c for c in numeric_cols if c in df.columns]
if not available_cols:
    raise ValueError("ไม่พบคอลัมน์ Features ที่ต้องการในไฟล์ CSV (available_cols is empty). \n"
                     "สาเหตุที่เป็นไปได้: ไฟล์ CSV ไม่มี Header หรือชื่อคอลัมน์ไม่ถูกต้อง \n"
                     "คำแนะนำ: ตรวจสอบไฟล์ features_for_rl.csv หรือลองรัน script create_features ใหม่")

df = df[available_cols]

# แบ่งข้อมูล: 80% สำหรับ Train, 20% สำหรับ Test (ทดสอบ)
train_size = int(len(df) * 0.8)
test_df = df.iloc[train_size:] # *** สำคัญ: ใช้ส่วนที่เหลือ 20% มาทดสอบ ***

print(f"ข้อมูลทั้งหมด: {len(df)} แถว")
print(f"ข้อมูลสำหรับทดสอบ (Test Set): {len(test_df)} แถว")

# --- 2. สร้าง Environment และโหลดโมเดล ---
print("กำลังโหลดโมเดลที่เทรนไว้...")
env = TradingEnv(test_df)
model_name = "ppo_trading_bot"
model = PPO.load(model_name)

# --- 3. เตรียมตัวแปรทั้งหมดก่อนเริ่ม Loop ---
obs, info = env.reset()
done = False
truncated = False

# ตัวแปรสำหรับเก็บสถิติ
initial_balance = info['balance']
previous_balance = initial_balance

total_trades = 0
winning_trades = 0
losing_trades = 0
# *** สร้างตัวแปร action_counts ก่อนใช้ ***
action_counts = {0: 0, 1: 0, 2: 0}

# --- 4. รัน Simulation ด้วย tqdm ---
# สร้าง tqdm object สำหรับ loop นี้
pbar = tqdm(total=len(test_df), desc="กำลังประมวลผล Backtest")

while not done and not truncated:
    # อัพเดท progress bar ทีละ 1 ก้าว
    pbar.update(1)
    
    action, _states = model.predict(obs, deterministic=True)
    action_counts[action.item()] += 1 # ใช้ตัวแปรที่สร้างไว้แล้ว
    
    obs, reward, done, truncated, info = env.step(action)
    
    current_balance = info['balance']
    
    # ตรวจสอบว่ามีการปิดออเดอร์หรือไม่
    if current_balance != previous_balance:
        trade_pnl = current_balance - previous_balance
        total_trades += 1
        
        if trade_pnl > 0:
            winning_trades += 1
        else:
            losing_trades += 1
            
        previous_balance = current_balance

# --- 5. ปิด Progress Bar เมื่อทำงานเสร็จ ---
pbar.close()

# --- 6. สรุปผล Performance ---
final_balance = info['balance']
profit = final_balance - initial_balance
win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

print("\n" + "="*30)
print("     รายงานผลการทดสอบ (Backtest Report)")
print("="*30)
print(f"ยอดเงินเริ่มต้น (Initial Balance) : {initial_balance:,.2f}")
print(f"ยอดเงินสิ้นสุด (Final Balance)   : {final_balance:,.2f}")
print(f"กำไร/ขาดทุนสุดทธิ (Net Profit)      : {profit:,.2f} ({(profit/initial_balance)*100:.2f}%)")
print("-" * 30)
print(f"จำนวนการเทรดทั้งหมด (Total Trades) : {total_trades}")
print(f"การเทรดที่กำไร (Winning Trades)   : {winning_trades}")
print(f"การเทรดที่ขาดทุน (Losing Trades) : {losing_trades}")
print(f"อัตราการชนะ (Win Rate)               : {win_rate:.2f}%")
print(f"การกระจายแอคชัน (Action Dist.)    : Hold={action_counts[0]}, Buy={action_counts[1]}, Sell={action_counts[2]}")
print("="*30)