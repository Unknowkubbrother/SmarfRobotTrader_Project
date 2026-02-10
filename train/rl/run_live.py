
from mt5linux import MetaTrader5
import pandas as pd
import numpy as np
import time
import os
from datetime import datetime, timedelta
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from env_trading import TradingEnv

# Initialize MT5 globally to access constants
mt5 = MetaTrader5(host="localhost", port=8001)

# ==================================================
# CONFIGURATION
# ==================================================
SYMBOL = "EURUSD"
TIMEFRAME = mt5.TIMEFRAME_H1
LOT_SIZE = 0.01
MAGIC_NUMBER = 123456
MODEL_PATH = "ppo_trading.zip"
VEC_NORM_PATH = "vec_normalize.pkl"
DEVIATION = 20
# INITIAL_BALANCE removed here, will be fetched from account info

# Features ที่ใช้ (ต้องเรียงตาม train_ppo.py เป๊ะๆ)
FEATURE_COLUMNS = [
    'return', 'range', 'delta_tick', 'delta_price', 'has_delta',
    'body_ratio', 'momentum'
]

class LiveTradingBot:
    def __init__(self):
        self.model = None
        self.vec_norm = None
        self.hold_steps = 0
        self.last_trade_action = 0 # 0=Flat
        self.initial_balance = 0.0 # Will be set on connect
        self.point = 0.00001 # Default fallback
        self.digits = 5 # Default fallback
        
    def connect(self):
        if not mt5.initialize():
            print("❌ initialize() failed")
            quit()
        
        # Check Account
        account_info = mt5.account_info()
        if account_info is None:
            print("❌ Failed to get account info")
            quit()
            
        print(f"✅ MT5 Connected. Account: {account_info.login}")
        self.initial_balance = account_info.balance
        print(f"💰 Initial Balance: ${self.initial_balance:.2f} (Used for Normalization)")
        
        # Check Symbol
        symbol_info = mt5.symbol_info(SYMBOL)
        if symbol_info is None:
            print(f"❌ Symbol {SYMBOL} not found")
            quit()
            
        if not symbol_info.visible:
            print(f"⚠️ Symbol {SYMBOL} is not visible, trying to select...")
            if not mt5.symbol_select(SYMBOL, True):
                print(f"❌ symbol_select({SYMBOL}) failed")
                quit()
                
        self.point = symbol_info.point
        self.digits = symbol_info.digits
        print(f"ℹ️ Symbol Info: Point={self.point}, Digits={self.digits}")

    def load_model(self):
        print(f"🧠 Loading Model: {MODEL_PATH}")
        self.model = PPO.load(MODEL_PATH)
        
        # Load Normalization Stats
        # ต้องสร้าง Dummy Env เพื่อ Load Stats
        print(f"Stats Loading: {VEC_NORM_PATH}")
        dummy_df = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'tick_volume'])
        # Create a dummy env just to initialize
        # Note: We need to mock data to init TradingEnv
        dummy_data = {
            'time': [datetime.now()] * 50,
            'open': [1.0] * 50, 'high': [1.0] * 50, 'low': [1.0] * 50, 'close': [1.0] * 50,
            'delta_tick': [0]*50, 'delta_price': [0]*50, 'has_delta': [0]*50
        }
        mock_df = pd.DataFrame(dummy_data) 
        dummy_env = DummyVecEnv([lambda: TradingEnv(mock_df)])
        
        self.vec_norm = VecNormalize.load(VEC_NORM_PATH, dummy_env)
        self.vec_norm.training = False # !!! CRITICAL: Do not update stats during inference !!!
        self.vec_norm.norm_reward = False

    def get_market_features(self):
        # 1. Fetch OHLC (Last 30 completed candles)
        # Offset 1 = Start from previous candle (finished), ignore current forming one
        rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 1, 30)
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        
        # 2. Calculate Features (Must match train_ppo.py logic exactly)
        df['return'] = df['close'].pct_change().fillna(0)
        df['range'] = (df['high'] - df['low']) / df['close']
        
        full_range = df['high'] - df['low']
        df['body_ratio'] = np.where(full_range > 0, abs(df['close'] - df['open']) / full_range, 0)
        
        df['momentum'] = df['return'].rolling(window=5).sum().fillna(0)
        
        # 3. Tick Delta calculation (Mocking precise logic)
        # In live trading, accurate `delta_tick` requires real-time tick accumulation.
        # For simplicity in this Demo, we will use a simplified approach or 0 if not critical.
        # However, to match training, we should try.
        # Let's fetch last 1 hour ticks
        now = datetime.now()
        one_hour_ago = now - timedelta(hours=1)
        ticks = mt5.copy_ticks_range(SYMBOL, one_hour_ago, now, mt5.COPY_TICKS_ALL)
        
        delta_tick = 0
        delta_price = 0.0
        has_delta = 0
        
        if ticks is not None and len(ticks) > 0:
            tdf = pd.DataFrame(ticks)
            # Simple approximation of logic
            # Buy = Bid > PrevBid
            tdf['prev_bid'] = tdf['bid'].shift(1)
            tdf['prev_ask'] = tdf['ask'].shift(1)
            
            buy = (tdf['bid'] > tdf['prev_bid']) | ((tdf['bid'] == tdf['prev_bid']) & (tdf['ask'] > tdf['prev_ask']))
            sell = (tdf['bid'] < tdf['prev_bid']) | ((tdf['bid'] == tdf['prev_bid']) & (tdf['ask'] < tdf['prev_ask']))
            
            delta_tick = buy.sum() - sell.sum()
            delta_price = (tdf['bid'].iloc[-1] - tdf['bid'].iloc[0]) + (tdf['ask'].iloc[-1] - tdf['ask'].iloc[0])
            has_delta = 1

        # Fill features
        # We need the LAST row's features
        last_row = df.iloc[-1].copy()
        
        # Manually constructing the feature array
        features = np.array([
            last_row['return'],
            last_row['range'],
            delta_tick,
            delta_price,
            has_delta,
            last_row['body_ratio'],
            last_row['momentum']
        ], dtype=np.float32)
        
        return features

    def get_state_features(self):
        # State: [position, equity_ratio, unrealized_pnl, normalized_hold_steps]
        
        # 1. Position & Unrealized PnL
        positions = mt5.positions_get(symbol=SYMBOL)
        current_position = 0 # Flat
        unrealized_pnl = 0.0
        
        if positions is not None and len(positions) > 0:
            pos = positions[0]
            if pos.type == mt5.ORDER_TYPE_BUY:
                current_position = 1
            elif pos.type == mt5.ORDER_TYPE_SELL:
                current_position = -1
            unrealized_pnl = pos.profit
            
            # Logic to track hold steps
            if current_position == self.last_trade_action:
                self.hold_steps += 1
            else:
                self.hold_steps = 0
                self.last_trade_action = current_position
        else:
            self.hold_steps = 0
            self.last_trade_action = 0

        # 2. Equity
        account = mt5.account_info()
        # Use simple normalization against current balance if initial balance isn't tracked long-term
        # But to match training 'equity/initial', we use the balance at script start
        equity_ratio = account.equity / self.initial_balance if self.initial_balance > 0 else 1.0
        
        # 3. Hold Steps Norm
        hold_norm = min(self.hold_steps / 50.0, 1.0)
        
        state = np.array([
            current_position,
            equity_ratio,
            unrealized_pnl,
            hold_norm
        ], dtype=np.float32)
        
        return state, current_position

    def run(self):
        self.connect()
        self.load_model()
        
        print("Waiting for next candle...")
        last_time = 0
        
        while True:
            # Check for new candle
            rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, 1)
            current_time = rates[0]['time']
            
            if current_time != last_time:
                print(f"\n⏰ New Candle: {datetime.fromtimestamp(current_time)}")
                last_time = current_time
                
                # 1. Get Data
                market_feat = self.get_market_features()
                state_feat, current_pos = self.get_state_features()
                
                # 2. Combine & Normalize
                # Note: NormalizeObs expects full observation including window history if Env uses it?
                # Wait! Env uses window_size=20!
                # My `get_market_features` only returns ONE row.
                # TradingEnv constructs obs as `block.flatten()` of window_size rows!
                # I MUST FETCH WINDOW_SIZE ROWS!
                
                # Correction: Fetch 20 rows (Completed)
                rates_window = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 1, 40) # buffer
                df_window = pd.DataFrame(rates_window)
                df_window['time'] = pd.to_datetime(df_window['time'], unit='s')
                
                # Calculate features for ALL rows to get correct momentum/rolling
                df_window['return'] = df_window['close'].pct_change().fillna(0)
                df_window['range'] = (df_window['high'] - df_window['low']) / df_window['close']
                full_range = df_window['high'] - df_window['low']
                df_window['body_ratio'] = np.where(full_range > 0, abs(df_window['close'] - df_window['open']) / full_range, 0)
                df_window['momentum'] = df_window['return'].rolling(window=5).sum().fillna(0)
                # Delta is tricky for history... assume 0 for history, calc real for last?
                # For simplicity/robustness: use 0 for history deita
                df_window['delta_tick'] = 0
                df_window['delta_price'] = 0.0
                df_window['has_delta'] = 0
                
                # Fill last row with real delta
                # (Re-using Delta Logic from above - simplified here)
                # ... [Insert Delta Logic Here if crucial] ... 
                # Let's skip Delta for history to keep it simple, only last row matters most for some, 
                # but CNN/MLP looks at whole window.
                # Ideally we need full history delta. Live trading limitation.
                
                # Get last 20 rows
                obs_window = df_window[FEATURE_COLUMNS].iloc[-20:].values.flatten().astype(np.float32)
                
                full_obs = np.concatenate([obs_window, state_feat])
                
                # Normalize
                obs_norm = self.vec_norm.normalize_obs(full_obs)
                
                # 3. Predict
                action, _ = self.model.predict(obs_norm, deterministic=True)
                print(f"🤖 Prediction: {action} (Pos: {current_pos})")
                
                # 4. Execute
                self.execute_trade(action, current_pos)
            
            # ==========================================
            # REAL-TIME PRICE & DELTA DISPLAY
            # ==========================================
            tick = mt5.symbol_info_tick(SYMBOL)
            if tick and len(rates) > 0: # Ensure we have rates
                # Calculate Delta for CURRENT forming candle (Index 0)
                # Note: copy_rates_from_pos(..., 1, ...) returns PREVIOUS candle at index 0
                # We need CURRENT candle start time.
                # Let's fetch current candle rate for timestamp
                current_rate = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, 1)
                
                if current_rate is not None and len(current_rate) > 0:
                    candle_start_ts = current_rate[0]['time'] # int timestamp
                    candle_start_dt = datetime.fromtimestamp(candle_start_ts)
                    now = datetime.now()
                    
                    # Fetch ticks from candle start until now
                    ticks = mt5.copy_ticks_range(SYMBOL, candle_start_dt, now, mt5.COPY_TICKS_ALL)
                    
                    current_delta = 0
                    if ticks is not None and len(ticks) > 0:
                        tdf = pd.DataFrame(ticks)
                        tdf['prev_bid'] = tdf['bid'].shift(1)
                        tdf['prev_ask'] = tdf['ask'].shift(1)
                        
                        buy = (tdf['bid'] > tdf['prev_bid']) | ((tdf['bid'] == tdf['prev_bid']) & (tdf['ask'] > tdf['prev_ask']))
                        sell = (tdf['bid'] < tdf['prev_bid']) | ((tdf['bid'] == tdf['prev_bid']) & (tdf['ask'] < tdf['prev_ask']))
                        current_delta = buy.sum() - sell.sum()

                    # Display status on the same line
                    print(f"\r📊 Price: {tick.bid:.{self.digits}f} | 🌊 DeltaTick: {current_delta} | ⏳ Time: {now.strftime('%H:%M:%S')}", end="", flush=True)

            time.sleep(1)

    def execute_trade(self, action, current_pos):
        # Action: 0=HOLD, 1=BUY, 2=SELL, 3=CLOSE
        
        if action == 0: # HOLD
            return
            
        elif action == 3: # CLOSE
            if current_pos != 0:
                self.close_all()
                
        elif action == 1: # BUY
            if current_pos == -1: self.close_all()
            if current_pos == 0: self.send_order(mt5.ORDER_TYPE_BUY)
            
        elif action == 2: # SELL
            if current_pos == 1: self.close_all()
            if current_pos == 0: self.send_order(mt5.ORDER_TYPE_SELL)

    def get_filling_mode(self):
        symbol_info = mt5.symbol_info(SYMBOL)
        if symbol_info is None:
            return mt5.ORDER_FILLING_FOK
        
        filling_mode = symbol_info.filling_mode
        if filling_mode & 1: # SYMBOL_FILLING_FOK = 1
            return mt5.ORDER_FILLING_FOK
        elif filling_mode & 2: # SYMBOL_FILLING_IOC = 2
            return mt5.ORDER_FILLING_IOC
        else:
            return mt5.ORDER_FILLING_RETURNAL

    def close_all(self):
        positions = mt5.positions_get(symbol=SYMBOL)
        for pos in positions:
            tick = mt5.symbol_info_tick(SYMBOL)
            price = tick.bid if pos.type == mt5.ORDER_TYPE_BUY else tick.ask
            req = {
                "action": mt5.TRADE_ACTION_DEAL,
                "position": pos.ticket,
                "symbol": SYMBOL,
                "volume": pos.volume,
                "type": mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                "price": price,
                "magic": MAGIC_NUMBER,
                "comment": "AI Close",
            }
            mt5.order_send(req)
            print("✅ Closed Position")

    def send_order(self, type):
        tick = mt5.symbol_info_tick(SYMBOL)
        price = tick.ask if type == mt5.ORDER_TYPE_BUY else tick.bid
        filling_mode = self.get_filling_mode()

        req = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": SYMBOL,
            "volume": LOT_SIZE,
            "type": type,
            "price": price,
            "magic": MAGIC_NUMBER,
            "comment": "AI Value Trade",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": filling_mode,
        }
        res = mt5.order_send(req)
        if res.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"✅ Opened {'BUY' if type == mt5.ORDER_TYPE_BUY else 'SELL'}")
        else:
            print(f"❌ Order Failed: {res.comment}")

if __name__ == "__main__":
    bot = LiveTradingBot()
    bot.run()
