import pandas as pd

chunk_iterator = pd.read_csv('EURUSD_ticks_2009_to_2025.csv', chunksize=500_000)

df = pd.DataFrame()

first_chunk = True
for chunk_df in chunk_iterator:
    print(f"กำลังประมวลผล Chunk ที่มี {len(chunk_df)} แถว...")
    
    # แปลงคอลัมน์ time เป็น datetime โดยตรง (ไม่ใช้ unit='s')
    chunk_df['time'] = pd.to_datetime(chunk_df['time'], format='mixed')
    
    # ตั้งคอลัมน์ time เป็น index เพื่อการ resample
    chunk_df.set_index('time', inplace=True)
    
    print(chunk_df)

    df = chunk_df

print(df)