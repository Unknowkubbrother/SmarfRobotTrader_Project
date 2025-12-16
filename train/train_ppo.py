import pandas as pd
from stable_baselines3 import PPO
from trading_env import TradingEnv

# --- 1. โหลดข้อมูล Features ---
print("กำลังโหลดข้อมูล Features...")
# บรรทัดที่สำคัญที่สุด: บอกให้ pandas ใช้คอลัมน์ 'time' เป็น index และแปลงเป็นวันที่
df = pd.read_csv('features_for_rl.csv', index_col='time', parse_dates=['time'])

numeric_cols = [
    'log_return', 'volatility_60', 'mom_60', 
    'rel_vol', 'rsi_14', 'dist_sma_60', 'mid_price'
]
# กรองเอาเฉพาะคอลัมน์ที่ต้องการและมีอยู่จริง
available_cols = [c for c in numeric_cols if c in df.columns]
if not available_cols:
    raise ValueError("ไม่พบคอลัมน์ Features ที่ต้องการในไฟล์ CSV (available_cols is empty). \n"
                     "สาเหตุที่เป็นไปได้: ไฟล์ CSV ไม่มี Header หรือชื่อคอลัมน์ไม่ถูกต้อง \n"
                     "คำแนะนำ: ตรวจสอบไฟล์ features_for_rl.csv หรือลองรัน script create_features ใหม่")

df = df[available_cols]
df.dropna(inplace=True)

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

# เพิ่ม entropy coefficient เพื่อ encourage exploration
model = PPO(
    "MlpPolicy", 
    env, 
    verbose=1,
    ent_coef=0.01,  # Entropy coefficient - ยิ่งสูงยิ่ง explore มากขึ้น
    learning_rate=3e-4
)

print("\nเริ่มการเทรนโมเดล PPO...")
print("ใช้ entropy coefficient = 0.01 เพื่อ encourage exploration")
# total_timesteps คือจำนวนก้าวทั้งหมดที่จะให้ Agent เรียนรู้
# เริ่มต้นที่ 50,000 ก้าว แล้วค่อยเพิ่มขึ้นได้
model.learn(total_timesteps=500_000)

# --- 4. บันทึกโมเดลที่เทรนเสร็จ ---
model_name = "ppo_trading_bot"
model.save(model_name)

print(f"\nเทรนโมเดลเสร็จสิ้น! บันทึกเป็นไฟล์ {model_name}.zip")