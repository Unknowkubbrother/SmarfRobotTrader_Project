import pandas as pd
import numpy as np
import os

# --- ตั้งค่าพารามิเตอร์ ---
# คราวนี้เราจะอ่านไฟล์ทีละส่วน (Chunk) เพราะไฟล์อาจจะใหญ่เกิน RAM
input_file = 'EURUSD_1sec_bars.csv' 
output_file = 'features_for_rl.csv'
chunk_size = 100000 # จำนวนแถวที่จะอ่านและประมวลผลต่อรอบ

print(f"กำลังเริ่มประมวลผลไฟล์ {input_file} แบบ Chunk (ทีละ {chunk_size} แถว)...")

# ตัวแปรสำหรับเก็บค่าสะสม (State) ระหว่าง Chunk
last_cumsum_vals = {
    'price_up': 0.0,
    'price_dn': 0.0,
    'tick_up': 0,
    'tick_dn': 0,
    'up_vol': 0.0,
    'dn_vol': 0.0
}

# Buffer สำหรับข้อมูลย้อนหลัง (ใช้คำนวณ indicator ที่ต้องใช้ค่าก่อนหน้า เช่น Rolling, RSI)
# เราต้องการ Buffer อย่างน้อยเท่ากับ window ที่ใหญ่ที่สุด (RSI 14) + diff (1) = 15
# เผื่อไว้สัก 100 แถว
buffer_df = None
buffer_size = 100

# ลบไฟล์ output เก่าถ้ามีอยู่ เพื่อเขียนใหม่
if os.path.exists(output_file):
    os.remove(output_file)

chunk_iter = pd.read_csv(input_file, chunksize=chunk_size, parse_dates=['time'])
first_chunk = True
total_rows_processed = 0

for i, chunk in enumerate(chunk_iter):
    # ตั้ง Time เป็น index
    chunk.set_index('time', inplace=True)
    
    # 1. รวม data กับ buffer ของ chunk ก่อนหน้า (เพื่อคำนวณ Rolling ได้ต่อเนื่อง)
    if buffer_df is not None:
        # เอา buffer มาต่อข้างหน้า chunk ปัจจุบัน
        working_df = pd.concat([buffer_df, chunk])
        start_idx = len(buffer_df) # index ที่ข้อมูลของ chunk ปัจจุบันเริ่ม
    else:
        working_df = chunk
        start_idx = 0
        
    # เก็บข้อมูลส่วนท้ายของ chunk นี้ไว้เป็น buffer สำหรับรอบถัดไป
    buffer_df = chunk.iloc[-buffer_size:].copy()
    
    # --- Part 1: คำนวณ Features พื้นฐาน (ไม่ต้องใช้ state ข้าม chunk หรือใช้แค่แถวตัวเอง) ---
    working_df['price_change'] = working_df['close'] - working_df['open']
    
    working_df['buy_volume'] = np.where(working_df['price_change'] > 0, working_df['volume'], 0)
    working_df['sell_volume'] = np.where(working_df['price_change'] < 0, working_df['volume'], 0)

    # working_df['spread'] = working_df['ask'] - working_df['bid'] # ข้อมูลไม่มี ask/bid แยก มีแค่ OHLC
    working_df['mid_price'] = (working_df['high'] + working_df['low']) / 2

    # --- Part 2: คำนวณ Rolling Features (SMA, RSI) ---
    # คำนวณบน working_df ซึ่งมี buffer แปะอยู่ข้างหน้าแล้ว ทำให้ค่าต้น chunk ถูกต้อง
    
    # SMA 10
    working_df['sma_10'] = working_df['close'].rolling(window=10).mean()
    
    # RSI 14
    delta = working_df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss.replace(0, np.nan)) # กันหาร 0
    # ถ้า loss เป็น 0 คือราคาขึ้นตลอด rs จะเป็น inf -> rsi เป็น 100
    rsi = 100 - (100 / (1 + rs))
    working_df['rsi_14'] = rsi.fillna(100) # handle case loss=0 appropriately if needed, or fillna

    # --- Part 3: คำนวณ Cumulative Features (สะสมค่า) ---
    # เราต้องตัดเอาเฉพาะส่วนที่เป็น chunk ปัจจุบันออกมาคำนวณและบวกค่าสะสมจากรอบก่อน
    # เพราะถ้าคำนวณบน working_df ค่าสะสมจะเริ่มใหม่ หรือเพี้ยนตรงรอยต่อ buffer
    
    # ตัดเอาเฉพาะข้อมูลใหม่ (New Data)
    new_data = working_df.iloc[start_idx:].copy()
    
    # คำนวณค่าที่จะนำมาสะสม (Increments) เฉพาะในส่วน new_data
    # ใช้วิธีคำนวณ vector ใน new_data แล้วค่อย cumsum
    
    # Helper สำหรับคำนวณ cumsum ต่อเนื่อง
    def apply_continuous_cumsum(df, col_name, increment_series, key_state):
        current_cumsum = increment_series.cumsum()
        df[col_name] = current_cumsum + last_cumsum_vals[key_state]
        # อัปเดต state ด้วยค่าสุดท้ายของ chunk นี้
        last_cumsum_vals[key_state] += current_cumsum.iloc[-1]

    # เตรียม Increments
    inc_price_up = new_data['price_change'].where(new_data['price_change'] > 0, 0)
    inc_price_dn = new_data['price_change'].where(new_data['price_change'] < 0, 0).abs()
    
    inc_tick_up = np.where(new_data['price_change'] > 0, 1, 0)
    inc_tick_up_series = pd.Series(inc_tick_up, index=new_data.index) # ทำให้เป็น Series เพื่อใช้ cumsum
    
    inc_tick_dn = np.where(new_data['price_change'] < 0, 1, 0)
    inc_tick_dn_series = pd.Series(inc_tick_dn, index=new_data.index)

    inc_up_vol = new_data['buy_volume']
    inc_dn_vol = new_data['sell_volume']

    # Apply State
    apply_continuous_cumsum(new_data, 'price_up', inc_price_up, 'price_up')
    apply_continuous_cumsum(new_data, 'price_dn', inc_price_dn, 'price_dn')
    apply_continuous_cumsum(new_data, 'tick_up', inc_tick_up_series, 'tick_up')
    apply_continuous_cumsum(new_data, 'tick_dn', inc_tick_dn_series, 'tick_dn')
    apply_continuous_cumsum(new_data, 'up_vol', inc_up_vol, 'up_vol')
    apply_continuous_cumsum(new_data, 'dn_vol', inc_dn_vol, 'dn_vol')

    # --- Clean up & Save ---
    
    # เลือกคอลัมน์
    features_cols = [
        'price_up', 'price_dn', 'tick_up', 'tick_dn', 
        'up_vol', 'dn_vol', 'mid_price',
        'sma_10', 'rsi_14'
    ]
    
    final_chunk = new_data[features_cols].copy()
    
    # ลบแถวที่มี NaN (มักจะเกิดใน Chunk แรกช่วงเริ่มต้นที่ indicators ยังคำนวณไม่ได้)
    if first_chunk:
        final_chunk.dropna(inplace=True)
    else:
        # Chunk หลังๆ ไม่ควรมี NaN จาก rolling เพราะเราใช้ buffer 
        # แต่อาจมี NaN จากข้อมูลดิบ (ถ้ามี) ก็ควร drop หรือ handle
        final_chunk.dropna(inplace=True)
    
    if not final_chunk.empty:
        # บันทึกทีละ chunk แบบ append (mode='a')
        final_chunk.to_csv(output_file, mode='a', header=first_chunk)
        total_rows_processed += len(final_chunk)
    
    first_chunk = False
    print(f"Processed chunk {i+1}: {len(final_chunk)} rows saved.")

print(f"\nประมวลผลเสร็จสิ้นทั้งหมด! ข้อมูลรวม {total_rows_processed} แถว")
print(f"ไฟล์บันทึกที่: {output_file}")