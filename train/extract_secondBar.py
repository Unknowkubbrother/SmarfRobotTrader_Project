import pandas as pd

input_file = 'EURUSD_ticks_2009_to_2025.csv' # แก้ไขชื่อไฟล์ตามที่คุณมี
output_file = 'EURUSD_1sec_bars.csv'

# อ่านไฟล์ทีละ chunk เพื่อไม่ให้ RAM เต็ม
chunk_iterator = pd.read_csv(input_file, chunksize=500_000)

first_chunk = True
for chunk_df in chunk_iterator:
    print(f"กำลังประมวลผล Chunk ที่มี {len(chunk_df)} แถว...")
    
    # แปลงคอลัมน์ time เป็น datetime โดยตรง (ไม่ใช้ unit='s')
    chunk_df['time'] = pd.to_datetime(chunk_df['time'], format='mixed')
    
    # ตั้งคอลัมน์ time เป็น index เพื่อการ resample
    chunk_df.set_index('time', inplace=True)
    
    # ใช้ resample โดยกลุ่มข้อมูลทุกๆ 1 วินาที ('1s')
    # และหาค่า OHLCV ของแต่ละกลุ่ม
    ohlcv_bars = chunk_df.resample('1s').agg({
        'bid': ['first', 'max', 'min', 'last'], # Open, High, Low, Close จาก bid
        'volume': 'sum'                         # Volume คือผลรวม
    })
    
    # จัดรูปแบบคอลัมน์ใหม่ที่เกิดจาก aggregation
    ohlcv_bars.columns = ['open', 'high', 'low', 'close', 'volume']
    
    # เขียนผลลัพธ์ลงไฟล์
    # ใช้ mode 'a' (append) สำหรับ chunk ถัดๆ ไป และ header=False เพื่อไม่ให้เขียนหัวตารางซ้ำ
    ohlcv_bars.to_csv(output_file, mode='a', header=first_chunk)
    
    # หลังจาก chunk แรกแล้ว ให้ตั้งค่า first_chunk เป็น False
    if first_chunk:
        first_chunk = False

print(f"สร้างไฟล์ {output_file} เสร็จสิ้น!")