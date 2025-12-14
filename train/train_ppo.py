import pandas as pd
from stable_baselines3 import PPO
from trading_env import TradingEnv

# --- 1. โหลดข้อมูล Features ---
print("กำลังโหลดข้อมูล Features...")
df = pd.read_csv('features_for_rl.csv')

# แบ่งข้อมูลเป็น Train และ Validation (สำคัญมาก!)
train_size = int(len(df) * 0.8)
train_df = df.iloc[:train_size]
# val_df = df.iloc[train_size:] # ใช้สำหรับทดสอบโมเดล

print(f"โหลดข้อมูลสำเร็จ! มีข้อมูลทั้งหมด {len(df)} แถว")
print(f"แบ่งข้อมูลสำหรับเทรน {len(train_df)} แถว")

# --- 2. สร้าง Environment ---
env = TradingEnv(train_df)

# --- 3. สร้างและเทรนโมเดล PPO ---
# MlpPolicy คือ Neural Network แบบธรรมดาที่เหมาะกับข้อมูลตัวเลข
# verbose=1 จะแสดงความคืบหน้าขณะเทรน
# tensorboard_log จะบันทึก log ไว้ให้ดูผ่าน TensorBoard
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_trading_tensorboard/")

print("\nเริ่มการเทรนโมเดล PPO...")
# total_timesteps คือจำนวนก้าวทั้งหมดที่จะให้ Agent เรียนรู้
# เริ่มต้นที่ 50,000 ก้าว แล้วค่อยๆ เพิ่มขึ้นได้
model.learn(total_timesteps=50_000)

# --- 4. บันทึกโมเดลที่เทรนเสร็จ ---
model_name = "ppo_trading_bot"
model.save(model_name)

print(f"\nเทรนโมเดลเสร็จสิ้น! บันทึกเป็นไฟล์ {model_name}.zip")