"""
MT5 Data Collector สำหรับ Train Reinforcement Learning
รองรับทั้ง Historical Data และ Real-time Tick Data
"""

# import MetaTrader5 as mt5 # Windows only
from mt5linux import MetaTrader5
# connect to the mt5linux server running in Docker
mt5 = MetaTrader5(host='192.168.0.105', port=8001)
# Fix for NameError: name 'datetime' is not defined on server
try:
    mt5._MetaTrader5__conn.execute("import datetime")
except Exception as e:
    print(f"⚠️ Could not import datetime on server: {e}")

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

class MT5DataCollector:
    """ดึงข้อมูลจาก MetaTrader5 สำหรับ train RL model"""
    
    def __init__(self, symbol="EURUSD", timeframe=mt5.TIMEFRAME_M5):
        self.symbol = symbol
        self.timeframe = timeframe
        self.connected = False
        
    def connect(self, login=None, password=None, server=None):
        """เชื่อมต่อกับ MT5"""
        if not mt5.initialize():
            print(f"❌ MT5 initialize() failed, error: {mt5.last_error()}")
            return False
        
        # ถ้ามี login info ให้ login
        if login and password and server:
            if not mt5.login(login, password, server):
                print(f"❌ Login failed, error: {mt5.last_error()}")
                return False
                
        self.connected = True
        print(f"✅ เชื่อมต่อ MT5 สำเร็จ")
        print(f"📊 MT5 version: {mt5.version()}")
        return True
    
    def get_historical_data(self, days=None, from_year=None, save_csv=False, chunk_months=1):  # เปลี่ยนจาก 6 เป็น 1
        """
        ดึงข้อมูลย้อนหลังสำหรับ training
        
        Args:
            days: จำนวนวันที่ต้องการ (ถ้าไม่ระบุ from_year)
            from_year: ดึงตั้งแต่ปีที่ระบุจนถึงปัจจุบัน (เช่น 2009)
            save_csv: บันทึกเป็น CSV หรือไม่
            chunk_months: แบ่งดึงทีละกี่เดือน (default 1 เดือน)
            
        Returns:
            pandas DataFrame
        """
        if not self.connected:
            print("❌ ยังไม่ได้เชื่อมต่อ MT5")
            return None
        
        # คำนวณวันที่
        utc_to = datetime.now().astimezone()
        
        if from_year:
            utc_from = datetime(from_year, 1, 1).astimezone()
            print(f"📅 กำลังดึงข้อมูลตั้งแต่ปี {from_year} ถึงปัจจุบัน...")
            
            # ถ้าช่วงเวลายาวมาก ให้แบ่งดึงเป็นช่วงๆ
            total_days = (utc_to - utc_from).days
            if total_days > 30:  # มากกว่า 1 เดือน
                return self._get_data_in_chunks(utc_from, utc_to, chunk_months, save_csv)
            
        elif days:
            utc_from = utc_to - timedelta(days=days)
        else:
            days = 30
            utc_from = utc_to - timedelta(days=days)
        
        # ดึงข้อมูลแบบปกติ (สำหรับช่วงสั้นๆ)
        end_ts = int(utc_to.timestamp())
        rates = mt5.copy_rates_range(self.symbol, self.timeframe, utc_from, end_ts)
        
        if rates is None:
            print(f"❌ ไม่สามารถดึงข้อมูลได้, error: {mt5.last_error()}")
            return None
        
        # แปลงเป็น DataFrame
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        
        years = (utc_to - utc_from).days / 365.25
        print(f"✅ ดึงข้อมูลได้ {len(df):,} แท่งเทียน ({years:.1f} ปี)")
        print(f"📅 ตั้งแต่ {df['time'].iloc[0]} ถึง {df['time'].iloc[-1]}")
        print(f"💾 ขนาดข้อมูล: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # คำนวณ features เพิ่มเติมสำหรับ RL
        df = self._add_features(df)
        
        if save_csv:
            if from_year:
                filename = f"{self.symbol}_{from_year}_to_present.csv"
            else:
                filename = f"{self.symbol}_{days}days.csv"
            df.to_csv(filename, index=False)
            print(f"💾 บันทึกไฟล์: {filename}")
        
        return df
    
    def _get_data_in_chunks(self, start_date, end_date, chunk_months, save_csv):
        """
        ดึงข้อมูลแบบแบ่งเป็นช่วงๆ เพื่อหลีกเลี่ยง MT5 limit
        """
        print(f"🔄 แบ่งดึงข้อมูลทีละ {chunk_months} เดือน...")
        
        all_data = []
        current_start = start_date
        chunk = 0
        
        while current_start < end_date:
            # คำนวณวันที่สิ้นสุดของ chunk นี้
            current_end = current_start + timedelta(days=chunk_months * 30)
            if current_end > end_date:
                current_end = end_date
            
            chunk += 1
            print(f"📦 Chunk {chunk}: {current_start.strftime('%Y-%m-%d')} ถึง {current_end.strftime('%Y-%m-%d')}", end=" ")
            
            # ดึงข้อมูลสำหรับช่วงนี้
            # mt5linux requires start to be datetime (for .astimezone()) but end to be int/repr-safe
            end_ts = int(current_end.timestamp())
            rates = mt5.copy_rates_range(self.symbol, self.timeframe, current_start, end_ts)
            
            if rates is None or len(rates) == 0:
                print(f"⚠️  ไม่มีข้อมูล")
                current_start = current_end
                continue
            
            print(f"✅ {len(rates):,} แท่ง")
            all_data.append(pd.DataFrame(rates))
            
            # เลื่อนไปช่วงถัดไป
            current_start = current_end
        
        if not all_data:
            print("❌ ไม่สามารถดึงข้อมูลได้เลย")
            return None
        
        # รวมข้อมูลทั้งหมด
        print("\n🔗 กำลังรวมข้อมูลทั้งหมด...")
        df = pd.concat(all_data, ignore_index=True)
        
        # ลบข้อมูลซ้ำ (ถ้ามี)
        df = df.drop_duplicates(subset=['time'], keep='first')
        df = df.sort_values('time').reset_index(drop=True)
        
        # แปลง timestamp
        df['time'] = pd.to_datetime(df['time'], unit='s')
        
        years = (end_date - start_date).days / 365.25
        print(f"\n✅ รวมข้อมูลเสร็จสิ้น: {len(df):,} แท่งเทียน ({years:.1f} ปี)")
        print(f"📅 ตั้งแต่ {df['time'].iloc[0]} ถึง {df['time'].iloc[-1]}")
        print(f"💾 ขนาดข้อมูล: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # คำนวณ features
        print("🔧 กำลังคำนวณ technical indicators...")
        df = self._add_features(df)
        
        if save_csv:
            filename = f"{self.symbol}_{start_date.year}_to_present.csv"
            print(f"💾 กำลังบันทึกไฟล์: {filename}")
            df.to_csv(filename, index=False)
            print(f"✅ บันทึกเสร็จสิ้น!")
        
        return df
    
    def _add_features(self, df):
        """เพิ่ม features สำหรับ RL"""
        # Returns
        df['return'] = df['close'].pct_change()
        
        # Moving Averages
        df['ma_5'] = df['close'].rolling(window=5).mean()
        df['ma_20'] = df['close'].rolling(window=20).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Volatility
        df['volatility'] = df['return'].rolling(window=20).std()
        
        # Spread (เป็น feature สำคัญในการเทรด)
        df['spread'] = df['high'] - df['low']
        
        return df
    
    def get_tick_stream(self, callback_func, buffer_size=1000):
        """
        ดึง real-time tick data สำหรับ online learning
        
        Args:
            callback_func: ฟังก์ชันที่จะถูกเรียกเมื่อมี tick ใหม่
            buffer_size: จำนวน tick ที่เก็บใน buffer
        """
        if not self.connected:
            print("❌ ยังไม่ได้เชื่อมต่อ MT5")
            return
        
        # Subscribe to market data
        if not mt5.symbol_select(self.symbol, True):
            print(f"❌ ไม่สามารถ subscribe {self.symbol}")
            return
        
        print(f"🔄 เริ่มรับ tick data สำหรับ {self.symbol}...")
        print("⏹️  กด Ctrl+C เพื่อหยุด")
        
        tick_buffer = []
        
        try:
            while True:
                # ดึง tick ล่าสุด
                tick = mt5.symbol_info_tick(self.symbol)
                
                if tick is None:
                    continue
                
                # แปลงเป็น dict
                tick_data = {
                    'time': datetime.fromtimestamp(tick.time),
                    'bid': tick.bid,
                    'ask': tick.ask,
                    'last': tick.last,
                    'volume': tick.volume,
                    'spread': tick.ask - tick.bid
                }
                
                # เก็บใน buffer
                tick_buffer.append(tick_data)
                if len(tick_buffer) > buffer_size:
                    tick_buffer.pop(0)
                
                # เรียก callback function
                callback_func(tick_data, tick_buffer)
                
        except KeyboardInterrupt:
            print("\n⏹️  หยุดรับ tick data")
    
    def prepare_rl_training_data(self, df, sequence_length=60):
        """
        เตรียมข้อมูลสำหรับ train RL (PPO)
        
        Args:
            df: DataFrame จาก get_historical_data()
            sequence_length: จำนวนแท่งเทียนที่ใช้เป็น state
            
        Returns:
            states, actions, rewards
        """
        # เลือก features ที่จะใช้ (ใช้ tick_volume แทน volume)
        feature_cols = ['close', 'tick_volume', 'ma_5', 'ma_20', 'rsi', 'volatility', 'spread']
        
        # Normalize ข้อมูล (สำคัญมากสำหรับ RL!)
        df_norm = df[feature_cols].copy()
        for col in feature_cols:
            df_norm[col] = (df_norm[col] - df_norm[col].mean()) / df_norm[col].std()
        
        # ลบ NaN
        df_norm = df_norm.dropna()
        
        # สร้าง sequences
        states = []
        for i in range(len(df_norm) - sequence_length):
            state = df_norm.iloc[i:i+sequence_length].values
            states.append(state)
        
        states = np.array(states)
        
        print(f"✅ เตรียมข้อมูล RL เสร็จสิ้น")
        print(f"📊 Shape: {states.shape}")
        print(f"   - Samples: {states.shape[0]}")
        print(f"   - Sequence length: {states.shape[1]}")
        print(f"   - Features: {states.shape[2]}")
        
        return states
    
    def disconnect(self):
        """ปิดการเชื่อมต่อ MT5"""
        mt5.shutdown()
        self.connected = False
        print("👋 ปิดการเชื่อมต่อ MT5 แล้ว")


# ==================== ตัวอย่างการใช้งาน ====================

if __name__ == "__main__":
    
    # 1. สร้าง collector
    collector = MT5DataCollector(symbol="EURUSD", timeframe=mt5.TIMEFRAME_M5)
    
    # 2. เชื่อมต่อ MT5
    # แบบไม่ต้อง login (ใช้ account ที่ login อยู่แล้วใน MT5)
    if collector.connect():
        
        # 3. ดึงข้อมูลตั้งแต่ปี 2009 ถึงปัจจุบัน แบ่งทีละ 1 เดือน
        print("\n⏳ กำลังดึงข้อมูล 16 ปี... อาจใช้เวลาสักครู่")
        df = collector.get_historical_data(from_year=2009, save_csv=True, chunk_months=1)
        
        if df is not None:
            print("\n📊 ตัวอย่างข้อมูล:")
            print(df.head())
            
            # 4. เตรียมข้อมูลสำหรับ train RL
            states = collector.prepare_rl_training_data(df, sequence_length=60)
            
            print(f"\n✅ พร้อม train RL model แล้ว!")
            print(f"📦 ข้อมูลที่เตรียมไว้: {states.shape[0]} samples")
        
        # 5. (Optional) รับ real-time tick data
        def on_tick(tick, buffer):
            """ฟังก์ชันที่จะถูกเรียกทุกครั้งที่มี tick ใหม่"""
            print(f"📈 {tick['time']} | Bid: {tick['bid']:.5f} | Ask: {tick['ask']:.5f}")
        
        # ถ้าต้องการรับ real-time data ให้ uncomment บรรทัดนี้
        # collector.get_tick_stream(on_tick)
        
        # 6. ปิดการเชื่อมต่อ
        collector.disconnect()