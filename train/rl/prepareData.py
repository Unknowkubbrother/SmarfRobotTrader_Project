from mt5linux import MetaTrader5
import pandas as pd
from datetime import datetime, timedelta
import pytz
from tqdm import tqdm

# ==================================================
# CONNECT MT5
# ==================================================
mt5 = MetaTrader5(host="localhost", port=8001)

if not mt5.initialize():
    raise RuntimeError("MT5 init failed")

print("✓ MT5 initialized")

# ==================================================
# CONFIG
# ==================================================
symbol = "EURUSD"
timezone = pytz.UTC

# TIMEFRAME CONFIGURATION
TIMEFRAME_CONFIG = {
    "M1":  {"mt5": mt5.TIMEFRAME_M1,  "resample": "1min"},
    "M5":  {"mt5": mt5.TIMEFRAME_M5,  "resample": "5min"},
    "M15": {"mt5": mt5.TIMEFRAME_M15, "resample": "15min"},
    "M30": {"mt5": mt5.TIMEFRAME_M30, "resample": "30min"},
    "H1":  {"mt5": mt5.TIMEFRAME_H1,  "resample": "1h"},
    "H4":  {"mt5": mt5.TIMEFRAME_H4,  "resample": "4h"},
    "D1":  {"mt5": mt5.TIMEFRAME_D1,  "resample": "1D"},
}

# SELECT TIMEFRAME HERE
SELECTED_TIMEFRAME = "H1"  # Change this to "M5", "M15", etc.
tf_config = TIMEFRAME_CONFIG[SELECTED_TIMEFRAME]

ohlc_start = datetime(2025, 1, 1, tzinfo=timezone)
ohlc_end   = datetime(2026, 2, 20, 23, 59, 59, tzinfo=timezone)


# ==================================================
# CHECK SYMBOL
# ==================================================
symbol_info = mt5.symbol_info(symbol)
if symbol_info is None:
    mt5.shutdown()
    raise RuntimeError(f"Symbol {symbol} not found")

if not symbol_info.visible:
    mt5.symbol_select(symbol, True)

# ==================================================
# FUNCTION: FETCH OHLC (CHUNK SAFE)
# ==================================================
def fetch_ohlc(mt5, symbol, start, end, timeframe_mt5):
    chunks = []
    current = start

    # คำนวณจำนวน chunk ทั้งหมด (แต่ละ chunk = 365 วัน)
    total_days = (end - start).days
    total_chunks = (total_days // 365) + 1

    print(f"  📊 OHLC ({SELECTED_TIMEFRAME}): จะโหลด ~{total_chunks} chunks (ปีละ 1 chunk)")

    with tqdm(total=total_chunks, desc="📥 Loading OHLC", unit="chunk", ncols=80) as pbar:
        while current < end:
            chunk_end = min(current + timedelta(days=365), end)

            # แสดง log ละเอียด
            pbar.set_postfix_str(f"{current.strftime('%Y-%m-%d')} → {chunk_end.strftime('%Y-%m-%d')}")

            rates = mt5.copy_rates_range(
                symbol,
                timeframe_mt5,
                current,
                chunk_end
            )

            if rates is not None and len(rates) > 0:
                chunks.append(pd.DataFrame(rates))

            current = chunk_end
            pbar.update(1)

    if not chunks:
        return None

    df = pd.concat(chunks, ignore_index=True)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)

    return df[['open', 'high', 'low', 'close']].sort_index()

# ==================================================
# FUNCTION: FETCH DELTA FROM TICKS (CHUNK DAILY)
# ==================================================
def fetch_delta_from_ticks(mt5, symbol, start, end, resample_rule):
    all_deltas = []
    current = start

    # คำนวณจำนวนวันทั้งหมด
    total_days = (end - start).days

    print(f"  🎯 Tick Delta ({SELECTED_TIMEFRAME}): จะโหลด {total_days} วัน (วันละ 1 chunk)")
    print(f"  ⚠️  ส่วนนี้อาจใช้เวลานานมาก เนื่องจากต้องโหลด tick data รายวัน\n")

    processed_days = 0
    with tqdm(total=total_days, desc="📥 Loading Ticks", unit="day", ncols=80) as pbar:
        while current < end:
            chunk_end = min(current + timedelta(days=1), end)

            # แสดง log วันที่กำลังโหลด
            pbar.set_postfix_str(f"{current.strftime('%Y-%m-%d')}")

            ticks = mt5.copy_ticks_range(
                symbol,
                current,
                chunk_end,
                mt5.COPY_TICKS_ALL
            )

            if ticks is not None and len(ticks) > 0:
                df = pd.DataFrame(ticks)
                df['time'] = pd.to_datetime(df['time'], unit='s')
                df = df[['time', 'bid', 'ask']]
                df = df.sort_values('time').reset_index(drop=True)
                df = df.set_index('time')

                # ===== OnTick logic =====
                df['prev_bid'] = df['bid'].shift(1)
                df['prev_ask'] = df['ask'].shift(1)

                buy = (df['bid'] > df['prev_bid']) | (
                    (df['bid'] == df['prev_bid']) & (df['ask'] > df['prev_ask'])
                )
                sell = (df['bid'] < df['prev_bid']) | (
                    (df['bid'] == df['prev_bid']) & (df['ask'] < df['prev_ask'])
                )

                df['bid_diff'] = df['bid'] - df['prev_bid']

                df['delta_tick'] = 0
                df.loc[buy, 'delta_tick'] = 1
                df.loc[sell, 'delta_tick'] = -1

                df['delta_price'] = 0.0
                df.loc[buy | sell, 'delta_price'] = df.loc[buy | sell, 'bid_diff']

                delta_resampled = df.resample(resample_rule).agg({
                    'delta_tick': 'sum',
                    'delta_price': 'sum'
                })

                all_deltas.append(delta_resampled)

            current = chunk_end
            processed_days += 1
            pbar.update(1)

            # ทุก 100 วัน แสดงสถานะเพิ่มเติม
            if processed_days % 100 == 0:
                tqdm.write(f"  ✅ ประมวลผลแล้ว {processed_days}/{total_days} วัน ({(processed_days/total_days)*100:.1f}%)")

    if not all_deltas:
        return pd.DataFrame(columns=['delta_tick', 'delta_price'])

    # รวมซ้ำ (กรณี tick วันต่อวัน)
    return pd.concat(all_deltas).groupby(level=0).sum()

# ==================================================
# FETCH OHLC
# ==================================================
print(f"Fetching OHLC {SELECTED_TIMEFRAME} for {symbol}")
df_ohlc = fetch_ohlc(mt5, symbol, ohlc_start, ohlc_end, tf_config["mt5"])

if df_ohlc is None or len(df_ohlc) == 0:
    mt5.shutdown()
    raise RuntimeError("No OHLC data available")

print(f"✓ Got {len(df_ohlc)} {SELECTED_TIMEFRAME} candles")
print("OHLC range:", df_ohlc.index.min(), "→", df_ohlc.index.max())

# ==================================================
# FETCH DELTA FROM TICKS - ดึงทั้งหมดที่มี
# ==================================================
tick_end   = df_ohlc.index.max().to_pydatetime()
# ดึง tick ทั้งหมดที่ MT5 มี (ไม่ limit 2 ปีแล้ว)
tick_start = df_ohlc.index.min().to_pydatetime()

print("\nFetching Tick Delta (chunked) - ALL AVAILABLE DATA")
print("Tick range:", tick_start, "→", tick_end)
print(f"⏱️  ประมาณ {(tick_end - tick_start).days} วัน")
print("⚠️  อาจใช้เวลานานมาก! กรุณารอ...")

df_delta = fetch_delta_from_ticks(
    mt5,
    symbol,
    tick_start,
    tick_end,
    tf_config["resample"]
)

mt5.shutdown()

print("✓ Tick delta done")
print(df_delta.tail())

# ==================================================
# MERGE HYBRID DATASET
# ==================================================
df_merged = df_ohlc.join(df_delta, how='left')

df_merged['delta_tick'] = df_merged['delta_tick'].fillna(0)
df_merged['delta_price'] = df_merged['delta_price'].fillna(0.0)

df_merged['has_delta'] = (
    (df_merged['delta_tick'] != 0) | (df_merged['delta_price'] != 0)
).astype(int)

print("\nFinal dataset sample:")
print(df_merged.tail())

# ==================================================
# SAVE
# ==================================================
filename = f"{SELECTED_TIMEFRAME.lower()}_ohlc_delta1.csv"
df_merged.to_csv(filename)
print(f"\n✓ Saved: {filename}")
