from mt5linux import MetaTrader5
import pandas as pd
import mplfinance as mpf
from datetime import datetime, timedelta
import matplotlib.ticker as ticker
import os
import time

# --- Configuration ---
CSV_PATH = "h1_ohlc_delta.csv"
OUTPUT_DIR = "images"
SYMBOL = "EURUSD"
TIMEFRAME = MetaTrader5.TIMEFRAME_M1
UTC_TO_BROKER = 7
BROKER_TO_TARGET = 7
TOTAL_OFFSET = UTC_TO_BROKER + BROKER_TO_TARGET

# Limit for testing (set to None for full run)
LIMIT_ROWS = None

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def format_date_func(df):
    # Closure to capture df
    def _format(x, pos=None):
        if x < 0 or x >= len(df): return ''
        return df.index[int(x)].strftime('%H:%M')
    return _format

def generate_chart(mt5_conn, row, output_path):
    try:
        # 1. Parse Target Time from CSV
        target_time_str = row['time'] # Expected format: YYYY-MM-DD HH:MM:SS
        target_time = pd.to_datetime(target_time_str).to_pydatetime()
        
        # Define the range (e.g., 1 hour window: 22:00 to 23:00)
        start_input = target_time
        end_input = target_time + timedelta(hours=1)

        # 2. Convert Target to Broker Time for fetching
        # Target (22:00) - BROKER_TO_TARGET (7h) = Broker (15:00)
        broker_start = start_input - timedelta(hours=BROKER_TO_TARGET)
        broker_end = end_input - timedelta(hours=BROKER_TO_TARGET)

        # 3. Fetch Data
        rates = mt5_conn.copy_rates_range(SYMBOL, TIMEFRAME, broker_start, broker_end)
        
        if rates is None or len(rates) == 0:
            # Silent skip for missing data
            return False

        # 4. Prepare DataFrame
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        
        # Shift Time: UTC -> Target (add TOTAL_OFFSET)
        df['time'] = df['time'] + timedelta(hours=TOTAL_OFFSET)
        
        df.set_index('time', inplace=True)
        df = df.rename(columns={'open':'Open','high':'High','low':'Low','close':'Close','tick_volume':'Volume'})

        # 5. Plot
        mc = mpf.make_marketcolors(up='#089981', down='#F23645', edge='inherit', wick='inherit', ohlc='i')
        s = mpf.make_mpf_style(base_mpf_style='nightclouds', marketcolors=mc, gridstyle='', facecolor='#131722', y_on_right=True)

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

        axlist[0].xaxis.set_major_formatter(ticker.FuncFormatter(format_date_func(df)))

        if len(df) > 1:
            axlist[0].xaxis.set_major_locator(ticker.MaxNLocator(nbins=7)) 
        else:
            axlist[0].xaxis.set_major_locator(ticker.IndexLocator(base=1, offset=0))

        # Save
        fig.savefig(output_path, bbox_inches='tight', dpi=150)
        
        # Close plot to free memory
        import matplotlib.pyplot as plt
        plt.close(fig)
        return True

    except Exception as e:
        print(f"❌ Error generating {output_path}: {e}")
        return False

def main():
    print("🚀 Starting Batch Image Generation...")
    
    # 1. Connect MT5
    mt5 = MetaTrader5(host="localhost", port=8001)
    if not mt5.initialize():
        print("❌ MT5 Initialize Failed")
        return

    # 2. Read CSV
    if not os.path.exists(CSV_PATH):
        print(f"❌ CSV not found: {CSV_PATH}")
        return
    
    print(f"📖 Reading {CSV_PATH}...")
    df_csv = pd.read_csv(CSV_PATH)

    df_csv['time'] = pd.to_datetime(df_csv['time'])
    df_csv = df_csv[df_csv['has_delta'] == 1].sort_values("time").reset_index(drop=True)

    # Resume from specific time
    start_from = pd.to_datetime("2021-03-29 00:00:00")
    df_csv = df_csv[df_csv['time'] >= start_from]
    print(f"Resuming from {start_from}...")

    # 3. Setup Output
    ensure_dir(OUTPUT_DIR)
    
    # 4. Limit Rows
    total_rows = len(df_csv)
    process_rows = df_csv.head(LIMIT_ROWS) if LIMIT_ROWS else df_csv
    
    print(f"Processing {len(process_rows)} / {total_rows} rows...")

    # 5. Loop
    count = 0
    start_time = time.time()
    
    for index, row in process_rows.iterrows():
        # Create filename from timestamp: 2009-10-16 00:00:00 -> EURUSD_2009.10.16 00.00.png
        dt = pd.to_datetime(row['time'])
        ts_str = dt.strftime('%Y.%m.%d %H.%M')
        filename = f"{SYMBOL}_{ts_str}.png"
        output_path = os.path.join(OUTPUT_DIR, filename)
        
        if generate_chart(mt5, row, output_path):
            count += 1
            if count % 10 == 0:
                print(f"✅ Generated {count} images...", end='\r')
        
        if (index + 1) % 100 == 0:
            print(f"⏳ Processed {index + 1}/{len(process_rows)} rows...", end='\r')

    mt5.shutdown()
    elapsed = time.time() - start_time
    print(f"\n🏁 Done! Generated {count} images in {elapsed:.2f}s")

if __name__ == "__main__":
    main()
