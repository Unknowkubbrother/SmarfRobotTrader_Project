from mt5linux import MetaTrader5
import pandas as pd
from datetime import datetime, timedelta
import pytz

# ==================================================
# CONFIG
# ==================================================
symbol = "USDTHB"
timezone = pytz.UTC

ohlc_start = datetime(2000, 1, 1, tzinfo=timezone)
ohlc_end   = datetime(2025, 12, 31, 23, 59, 59, tzinfo=timezone)

# ==================================================
# CONNECT MT5
# ==================================================
mt5 = MetaTrader5(host="localhost", port=8001)

if not mt5.initialize():
    raise RuntimeError("MT5 init failed")

print("✓ MT5 initialized")

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
# FUNCTION: FETCH OHLC H1 (CHUNK SAFE)
# ==================================================
def fetch_ohlc_h1(mt5, symbol, start, end):
    chunks = []
    current = start

    while current < end:
        chunk_end = min(current + timedelta(days=365), end)

        rates = mt5.copy_rates_range(
            symbol,
            mt5.TIMEFRAME_H1,
            current,
            chunk_end
        )

        if rates is not None and len(rates) > 0:
            chunks.append(pd.DataFrame(rates))

        current = chunk_end

    if not chunks:
        return None

    df = pd.concat(chunks, ignore_index=True)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)

    return df[['open', 'high', 'low', 'close']].sort_index()

# ==================================================
# FUNCTION: FETCH H1 DELTA FROM TICKS (CHUNK DAILY)
# ==================================================
def fetch_h1_delta_from_ticks(mt5, symbol, start, end):
    all_h1 = []
    current = start

    while current < end:
        chunk_end = min(current + timedelta(days=1), end)

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

            h1 = df.resample('1h').agg({
                'delta_tick': 'sum',
                'delta_price': 'sum'
            })

            all_h1.append(h1)

        current = chunk_end

    if not all_h1:
        return pd.DataFrame(columns=['delta_tick', 'delta_price'])

    # รวมซ้ำ H1 (กรณี tick วันต่อวัน)
    return pd.concat(all_h1).groupby(level=0).sum()

# ==================================================
# FETCH OHLC
# ==================================================
print(f"Fetching OHLC H1 for {symbol}")
df_ohlc = fetch_ohlc_h1(mt5, symbol, ohlc_start, ohlc_end)

if df_ohlc is None or len(df_ohlc) == 0:
    mt5.shutdown()
    raise RuntimeError("No OHLC data available")

print(f"✓ Got {len(df_ohlc)} H1 candles")
print("OHLC range:", df_ohlc.index.min(), "→", df_ohlc.index.max())

# ==================================================
# FETCH DELTA FROM TICKS (SAFE)
# ==================================================
tick_start = df_ohlc.index.min().to_pydatetime()
tick_end   = df_ohlc.index.max().to_pydatetime()

print("\nFetching Tick Delta (chunked)")
print("Tick range:", tick_start, "→", tick_end)

h1_delta = fetch_h1_delta_from_ticks(
    mt5,
    symbol,
    tick_start,
    tick_end
)

mt5.shutdown()

print("✓ Tick delta done")
print(h1_delta.tail())

# ==================================================
# MERGE HYBRID DATASET
# ==================================================
df_h1 = df_ohlc.join(h1_delta, how='left')

df_h1['delta_tick'] = df_h1['delta_tick'].fillna(0)
df_h1['delta_price'] = df_h1['delta_price'].fillna(0.0)

df_h1['has_delta'] = (
    (df_h1['delta_tick'] != 0) | (df_h1['delta_price'] != 0)
).astype(int)

print("\nFinal dataset sample:")
print(df_h1.tail())

# ==================================================
# SAVE
# ==================================================
df_h1.to_csv("h1_ohlc_delta.csv")
print("\n✓ Saved: h1_ohlc_delta.csv")
