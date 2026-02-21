
from mt5linux import MetaTrader5
import pandas as pd
import numpy as np
import time
import os
import sys
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
LOT_SIZE = 0.01         # Fallback fixed lot (used when RISK_PERCENT=0)
MAGIC_NUMBER = 123456
MODEL_PATH = "ppo_trading.zip"
VEC_NORM_PATH = "vec_normalize.pkl"
DEVIATION = 20
RISK_PERCENT = 1.0      # Risk % per trade (0 = use fixed LOT_SIZE)

# ==================================================
# SL/TP CONFIGURATION
# ==================================================
SL_PIPS = 30    # Stop Loss in pips (30 pips = 0.00030 for 5-digit broker)
TP_PIPS = 60    # Take Profit in pips (60 pips ≈ 2% TP, R:R = 1:2)
PIP_VALUE_PER_LOT = 10.0  # $10 per pip for 1 standard lot on EURUSD


def calc_auto_lot(balance, risk_pct=RISK_PERCENT, sl_pips=SL_PIPS,
                  pip_value_per_lot=PIP_VALUE_PER_LOT, min_lot=0.01, lot_step=0.01):
    """Risk-based position sizing"""
    if risk_pct <= 0:
        return LOT_SIZE
    risk_amount = balance * risk_pct / 100.0
    lot = risk_amount / (sl_pips * pip_value_per_lot)
    lot = max(min_lot, lot_step * int(lot / lot_step))
    return round(lot, 2)

# Features ที่ใช้ (ต้องเรียงตาม train_ppo.py เป๊ะๆ)
FEATURE_COLUMNS = [
    'return', 'range', 'delta_tick', 'delta_price',
    'body_ratio', 'momentum',
    'sma_cross', 'rsi_norm', 'atr_norm', 'trend'
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
        self.current_lot = LOT_SIZE  # Will be recalculated after connect
        
        # SL/TP tracking
        self.sl_hits = 0
        self.tp_hits = 0
        self.entry_price = 0.0
        self.current_sl = 0.0
        self.current_tp = 0.0
        
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
        
        # Auto lot calculation
        self.current_lot = calc_auto_lot(self.initial_balance)
        print(f"💰 Balance: ${self.initial_balance:.2f} | Auto Lot: {self.current_lot}")
        
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
            'time': [datetime.now()] * 80,
            'open': [1.0] * 80, 'high': [1.0] * 80, 'low': [1.0] * 80, 'close': [1.0] * 80,
            'delta_tick': [0]*80, 'delta_price': [0]*80,
            'sma_cross': [0]*80, 'rsi_norm': [0]*80, 'atr_norm': [0]*80, 'trend': [0]*80
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

        # Fill features
        # We need the LAST row's features
        last_row = df.iloc[-1].copy()
        
        # Manually constructing the feature array
        features = np.array([
            last_row['return'],
            last_row['range'],
            delta_tick,
            delta_price,
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
            self.entry_price = pos.price_open
            self.current_sl = pos.sl
            self.current_tp = pos.tp
            
            # Logic to track hold steps
            if current_position == self.last_trade_action:
                self.hold_steps += 1
            else:
                self.hold_steps = 0
                self.last_trade_action = current_position
        else:
            self.hold_steps = 0
            self.last_trade_action = 0
            self.entry_price = 0.0
            self.current_sl = 0.0
            self.current_tp = 0.0

        # 2. Total PnL (Pips)
        account = mt5.account_info()
        pip_size = self.point * 10 # Assuming 5-digit broker (1 pip = 10 points)
        pip_value = 10.0 # Standard $10/pip for EURUSD at 1 lot
        point_value = pip_value * LOT_SIZE
        
        realized_pnl = account.balance - self.initial_balance
        total_pnl_pips = realized_pnl / point_value if point_value > 0 else 0.0
        
        # 3. Hold Steps Norm (ต้องตรงกับ env: max_hold_steps=30)
        hold_norm = min(self.hold_steps / 30.0, 1.0)
        
        # 4. Unrealized Return & Pips
        unrealized_ret = 0.0
        unrealized_pips = 0.0
        if current_position != 0 and self.entry_price > 0:
            tick = mt5.symbol_info_tick(SYMBOL)
            if tick:
                current_price = tick.bid if current_position == 1 else tick.ask
                unrealized_ret = current_position * (current_price - self.entry_price) / self.entry_price
                unrealized_pips = current_position * (current_price - self.entry_price) / pip_size
        
        state = np.array([
            current_position,
            total_pnl_pips / 1000.0,
            unrealized_pips / 100.0,
            hold_norm,
            np.clip(unrealized_ret * 100, -5, 5)  # ตรงกับ env
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
                
                # Correction: Fetch 80 rows for rolling(50) to work correctly
                rates_window = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 1, 80)
                df_window = pd.DataFrame(rates_window)
                df_window['time'] = pd.to_datetime(df_window['time'], unit='s')
                
                # Calculate features for ALL rows
                df_window['return'] = df_window['close'].pct_change().fillna(0)
                df_window['range'] = (df_window['high'] - df_window['low']) / df_window['close']
                full_range = df_window['high'] - df_window['low']
                df_window['body_ratio'] = np.where(full_range > 0, abs(df_window['close'] - df_window['open']) / full_range, 0)
                df_window['momentum'] = df_window['return'].rolling(window=5).sum().fillna(0)
                
                # Delta (0 for history — live limitation)
                df_window['delta_tick'] = 0
                df_window['delta_price'] = 0.0
                
                # ===== Trend Indicators (ต้องตรงกับ train_ppo.py) =====
                sma20 = df_window['close'].rolling(20).mean()
                sma50 = df_window['close'].rolling(50).mean()
                df_window['sma_cross'] = np.where(sma20 > sma50, 1, np.where(sma20 < sma50, -1, 0))
                df_window['sma_cross'] = df_window['sma_cross'].fillna(0)
                
                delta_c = df_window['close'].diff()
                gain = delta_c.clip(lower=0).rolling(14).mean()
                loss_c = (-delta_c.clip(upper=0)).rolling(14).mean()
                rs = gain / (loss_c + 1e-10)
                rsi = 100 - (100 / (1 + rs))
                df_window['rsi_norm'] = ((rsi - 50) / 50).fillna(0)
                
                tr = np.maximum(
                    df_window['high'] - df_window['low'],
                    np.maximum(
                        abs(df_window['high'] - df_window['close'].shift(1)),
                        abs(df_window['low'] - df_window['close'].shift(1))
                    )
                )
                df_window['atr_norm'] = (tr.rolling(14).mean() / df_window['close']).fillna(0)
                
                df_window['trend'] = (sma20.pct_change(5) * 100).fillna(0)
                df_window['trend'] = df_window['trend'].clip(-2, 2)
                
                # Get last 20 rows (after rolling calcs)
                obs_window = df_window[FEATURE_COLUMNS].iloc[-20:].values.flatten().astype(np.float32)
                
                full_obs = np.concatenate([obs_window, state_feat])
                
                # Normalize
                obs_norm = self.vec_norm.normalize_obs(full_obs)
                
                # 3. Predict
                action, _ = self.model.predict(obs_norm, deterministic=True)
                action_names = {0: 'HOLD', 1: 'BUY', 2: 'SELL', 3: 'CLOSE'}
                print(f"🤖 Prediction: {action_names.get(int(action), '?')} (Pos: {current_pos})")
                
                # 4. Execute
                self.execute_trade(action, current_pos)
            
            # ==========================================
            # REAL-TIME PRICE & DELTA & SL/TP DISPLAY
            # ==========================================
            tick = mt5.symbol_info_tick(SYMBOL)
            if tick and len(rates) > 0:
                # ===== SL/TP TICK-LEVEL MONITORING =====
                # เช็คทุกวินาที ไม่ต้องรอแท่ง H1 ใหม่
                positions = mt5.positions_get(symbol=SYMBOL)
                sl_tp_info = ""
                if positions is not None and len(positions) > 0:
                    pos = positions[0]
                    current_pnl = pos.profit
                    sl_tp_info = f" | SL:{pos.sl:.{self.digits}f} TP:{pos.tp:.{self.digits}f} PnL:{current_pnl:+.2f}"
                    
                    # ถ้า broker ไม่ได้ตั้ง SL/TP (sl=0, tp=0) → เช็คเอง
                    if pos.sl == 0.0 or pos.tp == 0.0:
                        bid = tick.bid
                        ask = tick.ask
                        sl_pips_price = SL_PIPS * self.point
                        tp_pips_price = TP_PIPS * self.point
                        
                        should_close = False
                        close_reason = ""
                        
                        if pos.type == mt5.ORDER_TYPE_BUY:  # Long
                            if bid <= pos.price_open - sl_pips_price:
                                should_close = True
                                close_reason = "SL_HIT"
                                self.sl_hits += 1
                            elif bid >= pos.price_open + tp_pips_price:
                                should_close = True
                                close_reason = "TP_HIT"
                                self.tp_hits += 1
                        elif pos.type == mt5.ORDER_TYPE_SELL:  # Short
                            if ask >= pos.price_open + sl_pips_price:
                                should_close = True
                                close_reason = "SL_HIT"
                                self.sl_hits += 1
                            elif ask <= pos.price_open - tp_pips_price:
                                should_close = True
                                close_reason = "TP_HIT"
                                self.tp_hits += 1
                        
                        if should_close:
                            print(f"\n⚡ {close_reason}! Closing position immediately...")
                            print(f"   Entry: {pos.price_open:.{self.digits}f} | Current PnL: {current_pnl:+.2f}")
                            print(f"   SL hits: {self.sl_hits} | TP hits: {self.tp_hits}")
                            self.close_all()
                            continue
                
                # Calculate Delta for current forming candle
                current_rate = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, 1)
                
                current_delta = 0
                current_delta_price = 0.0
                
                if current_rate is not None and len(current_rate) > 0:
                    candle_start_ts = current_rate[0]['time']
                    candle_start_dt = datetime.fromtimestamp(candle_start_ts)
                    
                    ticks = mt5.copy_ticks_from(SYMBOL, candle_start_dt, 10000, mt5.COPY_TICKS_ALL)
                    
                    if ticks is not None and len(ticks) > 0:
                        tdf = pd.DataFrame(ticks)
                        tdf['prev_bid'] = tdf['bid'].shift(1)
                        tdf['prev_ask'] = tdf['ask'].shift(1)
                        
                        buy = (tdf['bid'] > tdf['prev_bid']) | ((tdf['bid'] == tdf['prev_bid']) & (tdf['ask'] > tdf['prev_ask']))
                        sell = (tdf['bid'] < tdf['prev_bid']) | ((tdf['bid'] == tdf['prev_bid']) & (tdf['ask'] < tdf['prev_ask']))
                        current_delta = buy.sum() - sell.sum()
                        current_delta_price = (tdf['bid'].iloc[-1] - tdf['bid'].iloc[0]) + (tdf['ask'].iloc[-1] - tdf['ask'].iloc[0])

                server_time_str = datetime.fromtimestamp(tick.time).strftime('%H:%M:%S')
                local_time_str = datetime.now().strftime('%H:%M')
                status_line = f"📊 Pr: {tick.bid:.{self.digits}f} | 🌊 DT: {current_delta} | 💰 DP: {current_delta_price:.{self.digits}f}{sl_tp_info} | ⏳ {server_time_str} ({local_time_str})      "
                sys.stdout.write(f"\r{status_line}")
                sys.stdout.flush()

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
            if current_pos <= 0: self.send_order(mt5.ORDER_TYPE_BUY)
            
        elif action == 2: # SELL
            if current_pos == 1: self.close_all()
            if current_pos >= 0: self.send_order(mt5.ORDER_TYPE_SELL)

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
        if positions is None:
            return
        for pos in positions:
            tick = mt5.symbol_info_tick(SYMBOL)
            price = tick.bid if pos.type == mt5.ORDER_TYPE_BUY else tick.ask
            filling_mode = self.get_filling_mode()
            req = {
                "action": mt5.TRADE_ACTION_DEAL,
                "position": pos.ticket,
                "symbol": SYMBOL,
                "volume": pos.volume,
                "type": mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                "price": price,
                "deviation": DEVIATION,
                "magic": MAGIC_NUMBER,
                "comment": "AI Close",
                "type_filling": filling_mode,
            }
            res = mt5.order_send(req)
            if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                print(f"\n✅ Closed Position | PnL: {pos.profit:+.2f}")
            else:
                comment = res.comment if res else 'No response'
                print(f"\n❌ Close Failed: {comment}")

    def send_order(self, order_type):
        tick = mt5.symbol_info_tick(SYMBOL)
        price = tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid
        filling_mode = self.get_filling_mode()
        
        # Calculate SL/TP prices
        sl_distance = SL_PIPS * self.point
        tp_distance = TP_PIPS * self.point
        
        if order_type == mt5.ORDER_TYPE_BUY:
            sl_price = round(price - sl_distance, self.digits)
            tp_price = round(price + tp_distance, self.digits)
        else:  # SELL
            sl_price = round(price + sl_distance, self.digits)
            tp_price = round(price - tp_distance, self.digits)

        # Recalculate lot from current balance
        account = mt5.account_info()
        if account:
            self.current_lot = calc_auto_lot(account.balance)
        
        req = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": SYMBOL,
            "volume": self.current_lot,
            "type": order_type,
            "price": price,
            "sl": sl_price,
            "tp": tp_price,
            "deviation": DEVIATION,
            "magic": MAGIC_NUMBER,
            "comment": "AI Trade",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": filling_mode,
        }
        res = mt5.order_send(req)
        if res.retcode == mt5.TRADE_RETCODE_DONE:
            side = 'BUY' if order_type == mt5.ORDER_TYPE_BUY else 'SELL'
            print(f"\n✅ Opened {side} @ {price:.{self.digits}f} | Lot: {self.current_lot} | SL: {sl_price:.{self.digits}f} | TP: {tp_price:.{self.digits}f}")
            self.entry_price = price
            self.current_sl = sl_price
            self.current_tp = tp_price
        else:
            print(f"\n❌ Order Failed: {res.comment}")

if __name__ == "__main__":
    bot = LiveTradingBot()
    bot.run()
