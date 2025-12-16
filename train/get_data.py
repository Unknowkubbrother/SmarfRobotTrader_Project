import pandas as pd
from datetime import datetime, timedelta, timezone
from mt5linux import MetaTrader5
import os

# --- ตั้งค่าการเชื่อมต่อ ---
# connect to mt5linux server running in Docker
mt5 = MetaTrader5(host='192.168.0.105', port=8001)

# Fix for NameError: name 'datetime' is not defined on server
try:
    mt5._MetaTrader5__conn.execute("import datetime")
    # เพิ่ม Timeout เป็น 600 วินาที (10 นาที) เพื่อรอ MT5 download history
    mt5._MetaTrader5__conn._config['sync_request_timeout'] = 600
    print("✅ Configured RPyC timeout to 600s")
except Exception as e:
    print(f"⚠️ Could not import datetime on server: {e}")

# --- ตั้งค่าพารามิเตอร์ ---
symbol = "EURUSD"
start_date = datetime(2009, 1, 1, tzinfo=timezone.utc)
end_date = datetime(2025, 12, 10, tzinfo=timezone.utc) 
output_filename = f"{symbol}_ticks_2009_to_2025.csv"

# --- เริ่มการทำงาน ---
print(f"กำลังเชื่อมต่อกับ MetaTrader 5...")

if not mt5.initialize():
    print("เชื่อมต่อกับ MetaTrader 5 ล้มเหลว, error code =", mt5.last_error())
    quit()

print("เชื่อมต่อสำเร็จ!")
print(f"กำลังดึงข้อมูล Tick สำหรับ {symbol} ตั้งแต่ {start_date.date()} ถึง {end_date.date()}...")
print(f"บันทึกข้อมูลลงไฟล์: {output_filename}")

# ลบไฟล์เก่าทิ้งก่อนถ้ามี (เพื่อเริ่มเขียนใหม่)
if os.path.exists(output_filename):
    os.remove(output_filename)
    print("🗑️ ลบไฟล์เก่าเรียบร้อย")

current_date = start_date
delta = timedelta(days=1) 
total_ticks = 0

try:
    while current_date < end_date:
        next_date = current_date + delta
        # อย่าให้เกิน end_date
        if next_date > end_date:
            next_date = end_date
            
        print(f"⏳ กำลังดึง {current_date.date()} ...", end="", flush=True)
        
        ticks = mt5.copy_ticks_range(
            symbol, 
            current_date, 
            next_date, 
            mt5.COPY_TICKS_ALL
        )
        
        if ticks is not None and len(ticks) > 0:
            count = len(ticks)
            total_ticks += count
            
        
            df = pd.DataFrame(ticks)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            
        
            header = not os.path.exists(output_filename) or os.path.getsize(output_filename) == 0
            df.to_csv(output_filename, mode='a', index=False, header=header)
            
            print(f" ✅ ได้ข้อมูล {count:,} Ticks")
        else:
            print(" ⚠️ ไม่มีข้อมูล")
            
        # เลื่อนไปวันถัดไป
        current_date = next_date

except KeyboardInterrupt:
    print("\n🛑 ผู้ใช้กดหยุดการทำงาน")
except Exception as e:
    print(f"\n❌ เกิดข้อผิดพลาด: {e}")
finally:
    mt5.shutdown()
    print("\n" + "="*50)
    print(f"สรุป: ดึงข้อมูลเสร็จสิ้น/หยุดการทำงาน")
    print(f"รวมทั้งหมด: {total_ticks:,} Ticks")
    print(f"บันทึกไฟล์ที่: {os.path.abspath(output_filename)}")