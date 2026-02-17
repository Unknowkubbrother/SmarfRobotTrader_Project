from mt5linux import MetaTrader5
import pandas as pd
import mplfinance as mpf
from datetime import datetime, timedelta
import matplotlib.ticker as ticker

# --- 1. เชื่อมต่อ ---
mt5 = MetaTrader5(host="localhost", port=8001)
mt5.initialize()

# --- 2. ตั้งค่าช่วงเวลา (พิมพ์เลขไหน ต้องได้เลขนั้นในรูป) ---
symbol = "EURUSD"
timeframe = mt5.TIMEFRAME_M1

# ใส่ 22:00 เพื่อให้ในรูปขึ้น 22:00
start_input = datetime(2025, 1, 23, 22, 0) 
end_input   = datetime(2025, 1, 23, 23, 0)

# --- 3. ส่วนต่างเวลา (Timezone Configuration) ---
# Raw Date from MT5 is UTC (08:00)
# Broker Server Time is likely UTC+7 (15:00)
# We want to display 22:00 in local/target time.
# So we need to shift: UTC (08:00) -> Broker (15:00) -> Target (22:00)
# Total Shift = 14 hours

UTC_TO_BROKER = 7
BROKER_TO_TARGET = 7 # This is original OFFSET_HOURS (15:00 -> 22:00)

TOTAL_OFFSET = UTC_TO_BROKER + BROKER_TO_TARGET

# --- 4. ดึงข้อมูล (ลบ 7 เพื่อไปเอาแท่ง 15:00 ของโบรกเกอร์มา) ---
broker_start = start_input - timedelta(hours=BROKER_TO_TARGET)
broker_end   = end_input   - timedelta(hours=BROKER_TO_TARGET)

rates = mt5.copy_rates_range(symbol, timeframe, broker_start, broker_end)
mt5.shutdown()

if rates is None or len(rates) == 0:
    print("❌ ไม่พบข้อมูลในระบบ")
    quit()

# --- 5. จัดเตรียมข้อมูล (บวก 14 กลับเข้าไปเพื่อให้ Index เป็น 22:00) ---
df = pd.DataFrame(rates)
df['time'] = pd.to_datetime(df['time'], unit='s')

# [จุดสำคัญ] ปรับเวลาจาก UTC (08:00) -> 22:00 โดยบวก 14 ชั่วโมง
df['time'] = df['time'] + timedelta(hours=TOTAL_OFFSET)

df.set_index('time', inplace=True)
df = df.rename(columns={'open':'Open','high':'High','low':'Low','close':'Close','tick_volume':'Volume'})

# --- 6. ตั้งค่าการแสดงผล ---
mc = mpf.make_marketcolors(up='#089981', down='#F23645', edge='inherit', wick='inherit', ohlc='i')
s = mpf.make_mpf_style(base_mpf_style='nightclouds', marketcolors=mc, gridstyle='', facecolor='#131722', y_on_right=True)

# ฟังก์ชันดึงเวลาจาก Index ที่บวก 7 แล้วมาโชว์ในรูป
def format_date(x, pos=None):
    if x < 0 or x >= len(df): return ''
    return df.index[int(x)].strftime('%H:%M')

output_filename = f"Final_Chart_2200.png"

# --- 7. สร้างกราฟ ---
fig, axlist = mpf.plot(
    df,
    type='candle',
    style=s,
    volume=False,
    show_nontrading=False,
    tight_layout=True,
    figratio=(16, 9),
    returnfig=True
)

# บังคับป้ายกำกับแกน X ให้ดึงเวลา 22:00 มาแปะ
axlist[0].xaxis.set_major_formatter(ticker.FuncFormatter(format_date))

if len(df) > 1:
    axlist[0].xaxis.set_major_locator(ticker.MaxNLocator(nbins=7)) 
else:
    # กรณีมีแท่งเดียว ไม่ต้องแบ่ง bin เดี๋ยว Error
    axlist[0].xaxis.set_major_locator(ticker.IndexLocator(base=1, offset=0))

fig.savefig(output_filename, bbox_inches='tight', dpi=150)

print(f"✅ บันทึกสำเร็จ! ตรวจสอบไฟล์: {output_filename}")
print(f"⏰ เวลาในภาพเริ่มที่: {df.index[0].strftime('%H:%M')} น. (Total Offset +{TOTAL_OFFSET}h)")