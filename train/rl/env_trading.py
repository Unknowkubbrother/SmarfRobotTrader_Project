import gymnasium as gym
import numpy as np


class TradingEnv(gym.Env):
    """
    Trading Environment - Simple & Clear
    
    Actions:
    - 0: HOLD    → ถือต่อ / ไม่ทำอะไร
    - 1: BUY     → ทำนายว่าราคาจะขึ้น (Open Long)
    - 2: SELL    → ทำนายว่าราคาจะลง (Open Short)
    - 3: CLOSE   → ปิด order (Take Profit / Cut Loss)
    
    Position:
    - 0: Flat (ไม่มี position)
    - 1: Long (ซื้อ, รอราคาขึ้น)
    - -1: Short (ขาย, รอราคาลง)
    """
    
    def __init__(
        self,
        df,
        window_size=20,
        initial_balance=10_000,
        lot_size=1.0,
        spread_cost=0.0001,      # 0.01% spread
        commission=0.00002,      # 0.002% commission
        max_dd=0.15,             # Max drawdown 15%
        stop_loss_pct=0.01,      # 1% SL
        take_profit_pct=0.02,    # 2% TP (R:R = 1:2)
        max_hold_steps=30,       # ถือสูงสุด 30 แท่ง H1
        random_start=False,      # สุ่มเริ่มต้นในแต่ละ episode
    ):
        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.lot_size = lot_size
        self.spread_cost = spread_cost
        self.commission = commission
        self.max_dd = max_dd
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_hold_steps = max_hold_steps
        self.random_start = random_start
        
        self.max_step = len(df) - 2
        
        # 4 Actions: Hold, Buy(Long), Sell(Short), Close
        self.action_space = gym.spaces.Discrete(4)
        
        # Features: 11 market + 5 state (เพิ่ม trend indicators)
        self.market_features = len(self._get_feature_columns())
        self.state_features = 5
        total_features = (self.window_size * self.market_features) + self.state_features
        
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(total_features,), dtype=np.float32
        )
        
        self.reset()
    
    @staticmethod
    def _get_feature_columns():
        return [
            'return', 'range', 'delta_tick', 'delta_price',
            'body_ratio', 'momentum',
            'sma_cross', 'rsi_norm', 'atr_norm', 'trend'
        ]
    
    def seed(self, seed=None):
        np.random.seed(seed)
        return [seed]
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Random start position — ป้องกันการจำ pattern
        if self.random_start:
            max_start = max(self.window_size + 1, self.max_step - 500)  # เหลืออย่างน้อย 500 steps
            self.step_idx = np.random.randint(self.window_size, max_start)
        else:
            self.step_idx = self.window_size
        self.position = 0       # -1: Short, 0: Flat, 1: Long
        self.equity = self.initial_balance
        self.balance = self.initial_balance
        self.max_equity = self.initial_balance
        
        self.entry_price = 0.0
        self.hold_steps = 0
        self.unrealized_pnl = 0.0
        
        self.trades = 0
        self.wins = 0
        self.total_fees = 0.0
        self.total_pnl = 0.0
        
        # SL/TP tracking
        self.sl_hits = 0
        self.tp_hits = 0
        
        # Track predictions
        self.correct_predictions = 0
        self.total_predictions = 0
        
        # Track trade returns for Sharpe-like reward
        self.trade_returns = []
        
        return self._get_obs(), {}
    
    def _get_obs(self):
        start = self.step_idx - self.window_size
        end = self.step_idx
        
        feature_columns = self._get_feature_columns()
        available_cols = [col for col in feature_columns if col in self.df.columns]
        
        block = self.df.iloc[start:end][available_cols].values
        market_data = block.flatten().astype(np.float32)
        
        # State: position, equity ratio, unrealized pnl, hold steps, unrealized return
        unrealized_ret = 0.0
        if self.position != 0 and self.entry_price > 0:
            current_price = self._get_current_price()
            unrealized_ret = self.position * (current_price - self.entry_price) / self.entry_price
        
        state_data = np.array([
            self.position,
            self.equity / self.initial_balance,
            self.unrealized_pnl / self.initial_balance,
            min(self.hold_steps / self.max_hold_steps, 1.0),  # Normalized hold time
            np.clip(unrealized_ret * 100, -5, 5),  # Unrealized return % (clipped)
        ], dtype=np.float32)
        
        return np.concatenate([market_data, state_data])
    
    def _get_current_price(self):
        return self.df.iloc[self.step_idx]['close']
    
    def step(self, action):
        prev_equity = self.equity
        prev_position = self.position
        current_price = self._get_current_price()
        next_return = self.df.iloc[self.step_idx + 1]['raw_return']
        
        reward = 0.0
        trade_executed = False
        sl_tp_closed = False
        close_pnl = 0.0  # PnL เมื่อปิด position
        
        # ===== SL/TP AUTO-CLOSE CHECK (ก่อน action) =====
        if self.position != 0 and self.entry_price > 0:
            unrealized_return = self.position * (current_price - self.entry_price) / self.entry_price
            
            if unrealized_return <= -self.stop_loss_pct:
                # Stop Loss hit → auto close
                close_pnl = self.unrealized_pnl
                self._close_position(current_price)
                self.sl_hits += 1
                sl_tp_closed = True
                trade_executed = True
            elif unrealized_return >= self.take_profit_pct:
                # Take Profit hit → auto close
                close_pnl = self.unrealized_pnl
                self._close_position(current_price)
                self.tp_hits += 1
                sl_tp_closed = True
                trade_executed = True
            elif self.hold_steps >= self.max_hold_steps:
                # ถือนานเกินไป → บังคับปิด
                close_pnl = self.unrealized_pnl
                self._close_position(current_price)
                sl_tp_closed = True
                trade_executed = True
        
        # ===== ACTION LOGIC (skip if SL/TP just closed) =====
        # 0: HOLD - ไม่ทำอะไร
        # 1: BUY - ทำนายว่าขึ้น (Long)
        # 2: SELL - ทำนายว่าลง (Short)
        # 3: CLOSE - ปิด order
        
        if sl_tp_closed:
            pass  # SL/TP ปิดแล้ว ไม่ต้องทำ action
        elif action == 0:  # HOLD
            pass  # ไม่ทำอะไร
            
        elif action == 1:  # BUY (ทำนายว่าขึ้น)
            if self.position == -1:
                close_pnl = self.unrealized_pnl
                self._close_position(current_price)
                self._open_position(1, current_price)
                trade_executed = True
            elif self.position == 0:
                self._open_position(1, current_price)
                trade_executed = True
            
        elif action == 2:  # SELL (ทำนายว่าลง)
            if self.position == 1:
                close_pnl = self.unrealized_pnl
                self._close_position(current_price)
                self._open_position(-1, current_price)
                trade_executed = True
            elif self.position == 0:
                self._open_position(-1, current_price)
                trade_executed = True
            
        elif action == 3:  # CLOSE
            if self.position != 0 and not sl_tp_closed:
                close_pnl = self.unrealized_pnl
                self._close_position(current_price)
                trade_executed = True
        
        # ===== PRICE MOVEMENT & PnL =====
        if self.position != 0:
            pnl = self.position * next_return * self.lot_size * self.equity
            self.unrealized_pnl += pnl
            self.equity += pnl
            self.hold_steps += 1
            
            # Track prediction accuracy
            if trade_executed and prev_position == 0:
                self.total_predictions += 1
                if (self.position == 1 and next_return > 0) or (self.position == -1 and next_return < 0):
                    self.correct_predictions += 1
        
        self.max_equity = max(self.max_equity, self.equity)
        drawdown = (self.max_equity - self.equity) / self.max_equity if self.max_equity > 0 else 0
        
        # ===== REWARD CALCULATION (Balanced v3) =====
        
        # 1. PnL-based reward — core signal
        equity_change = (self.equity - prev_equity) / self.initial_balance
        reward = np.clip(equity_change * 50, -1, 1)
        
        # 2. HOLD winning position bonus
        if self.position != 0 and self.unrealized_pnl > 0:
            reward += 0.02  # ถือกำไร = ดี (เพิ่มขึ้น)
        
        # 3. Trade penalty — STRONG (ค่า fee ทำลาย performance!)
        if trade_executed and not sl_tp_closed:
            reward -= 0.1  # เพิ่มจาก 0.05
        
        # 4. Close trade bonus — only if held minimum time
        if close_pnl != 0:
            close_return = close_pnl / self.initial_balance
            self.trade_returns.append(close_return)
            if close_pnl > 0:
                reward += min(abs(close_return) * 30, 0.8)  # Big win bonus
            else:
                reward -= min(abs(close_return) * 15, 0.4)
        
        # 5. SL/TP bonuses
        if sl_tp_closed and close_pnl > 0:
            reward += 0.3  # TP hit bonus
        
        # 6. Hold too long penalty
        if self.position != 0 and self.hold_steps > self.max_hold_steps * 0.8:
            reward -= 0.005
        
        # 7. Drawdown penalty
        if drawdown > 0.08:
            reward -= (drawdown - 0.08) * 2
        
        # 8. Stay flat bonus — stronger
        if action == 0 and self.position == 0:
            reward += 0.005  # เพิ่มจาก 0.002
        
        # ===== STEP FORWARD =====
        self.step_idx += 1
        
        # ===== DONE CONDITIONS =====
        truncated = self.step_idx >= self.max_step
        terminated = (
            drawdown >= self.max_dd
            or self.equity <= 0
        )
        done = terminated or truncated
        
        # End episode bonus/penalty
        if done:
            final_return = (self.equity - self.initial_balance) / self.initial_balance
            if final_return > 0:
                reward += min(final_return * 5, 1.0)  # Scaled bonus (max 1.0)
            else:
                reward -= 0.2
            
            # Sharpe-like bonus
            if len(self.trade_returns) > 5:
                avg_ret = np.mean(self.trade_returns)
                std_ret = np.std(self.trade_returns) + 1e-8
                sharpe = avg_ret / std_ret
                reward += np.clip(sharpe * 0.3, -0.5, 0.5)
        
        # Prediction accuracy
        accuracy = (self.correct_predictions / self.total_predictions * 100) if self.total_predictions > 0 else 0
        
        info = {
            "equity": self.equity,
            "balance": self.balance,
            "drawdown": drawdown,
            "position": self.position,
            "trades": self.trades,
            "wins": self.wins,
            "win_rate": (self.wins / self.trades * 100) if self.trades > 0 else 0,
            "hold_steps": self.hold_steps,
            "fees": self.total_fees,
            "unrealized_pnl": self.unrealized_pnl,
            "total_pnl": self.total_pnl,
            "accuracy": accuracy,
            "action": action,
            "sl_hits": self.sl_hits,
            "tp_hits": self.tp_hits,
        }
        
        return self._get_obs(), reward, terminated, truncated, info
    
    def _open_position(self, direction, price):
        """เปิด position ใหม่"""
        self.position = direction
        self.entry_price = price
        self.hold_steps = 0
        self.unrealized_pnl = 0.0
        
        # จ่าย spread + commission
        cost = (self.spread_cost + self.commission) * self.equity * self.lot_size
        self.equity -= cost
        self.total_fees += cost
    
    def _close_position(self, price):
        """ปิด position และ realize PnL"""
        if self.position == 0:
            return
        
        # Realize PnL
        realized_pnl = self.unrealized_pnl
        self.total_pnl += realized_pnl
        self.balance = self.equity
        
        # Track win/loss
        self.trades += 1
        if realized_pnl > 0:
            self.wins += 1
        
        # จ่าย commission
        cost = self.commission * self.equity * self.lot_size
        self.equity -= cost
        self.total_fees += cost
        
        # Reset position state
        self.position = 0
        self.entry_price = 0.0
        self.hold_steps = 0
        self.unrealized_pnl = 0.0
    
    def render(self, mode='human'):
        pos_str = {-1: "SHORT", 0: "FLAT", 1: "LONG"}[self.position]
        print(f"Step: {self.step_idx} | {pos_str} | "
              f"Equity: ${self.equity:.2f} | PnL: ${self.total_pnl:.2f} | "
              f"Trades: {self.trades} | WR: {(self.wins/self.trades*100) if self.trades > 0 else 0:.1f}%")
