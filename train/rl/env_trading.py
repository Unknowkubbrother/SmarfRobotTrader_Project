import gymnasium as gym
import numpy as np


class TradingEnv(gym.Env):
    """
    Trading Environment — MT5-Aligned (Pip-Based)
    
    Simulates trading exactly like MT5 Strategy Tester:
    - Fixed pip SL/TP (not percentage)
    - Pip-value PnL (not compound return)
    - Intra-bar SL/TP check via High/Low
    - Fixed spread cost in pips
    
    Actions:
    - 0: HOLD    → ถือต่อ / ไม่ทำอะไร
    - 1: BUY     → Open Long (ซื้อ)
    - 2: SELL    → Open Short (ขาย)
    - 3: CLOSE   → ปิด order
    
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
        # ===== MT5-Aligned Parameters =====
        lot_size=0.1,             # MT5 lot size (0.1 = mini lot)
        pip_size=0.0001,          # EURUSD: 1 pip = 0.0001
        pip_value=10.0,           # $10 per pip for 1.0 standard lot on EURUSD
        sl_pips=30,               # Stop Loss in pips (matches MT5 EA)
        tp_pips=60,               # Take Profit in pips (matches MT5 EA)
        spread_pips=2,            # Typical broker spread in pips
        commission_per_lot=0.0,   # Commission per lot (0 if spread-only broker)
        max_dd=0.30,              # Max drawdown 30%
        max_hold_steps=30,        # ถือสูงสุด 30 แท่ง H1
        random_start=False,       # สุ่มเริ่มต้นในแต่ละ episode
    ):
        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.lot_size = lot_size
        self.pip_size = pip_size
        self.pip_value = pip_value
        self.sl_pips = sl_pips
        self.tp_pips = tp_pips
        self.spread_pips = spread_pips
        self.commission_per_lot = commission_per_lot
        self.max_dd = max_dd
        self.max_hold_steps = max_hold_steps
        self.random_start = random_start
        
        # Derived: cost per trade (spread + commission)
        # e.g., 2 pips * $10/pip * 0.1 lot = $2 per entry
        self.spread_cost = self.spread_pips * self.pip_value * self.lot_size
        self.commission_cost = self.commission_per_lot * self.lot_size
        
        self.max_step = len(df) - 2
        
        # 4 Actions: Hold, Buy(Long), Sell(Short), Close
        self.action_space = gym.spaces.Discrete(4)
        
        # Features: 10 market + 5 state
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
            max_start = max(self.window_size + 1, self.max_step - 500)
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
        
        # State: position, total_pnl_pips, unrealized_pips, hold_steps, unrealized_return
        unrealized_ret = 0.0
        unrealized_pips = 0.0
        if self.position != 0 and self.entry_price > 0:
            current_price = self._get_current_price()
            unrealized_ret = self.position * (current_price - self.entry_price) / self.entry_price
            unrealized_pips = self.position * (current_price - self.entry_price) / self.pip_size
        
        # Calculate total pnl in pips
        total_pnl_pips = self.total_pnl / (self.pip_value * self.lot_size) if self.lot_size > 0 else 0.0
        
        state_data = np.array([
            self.position,
            total_pnl_pips / 1000.0,          # Scale ~1000 pips to reasonable range (1.0~-1.0)
            unrealized_pips / 100.0,          # Scale ~100 pips to reasonable range (1.0~-1.0)
            min(self.hold_steps / self.max_hold_steps, 1.0),
            np.clip(unrealized_ret * 100, -5, 5),
        ], dtype=np.float32)
        
        return np.concatenate([market_data, state_data])
    
    def _get_current_price(self):
        return self.df.iloc[self.step_idx]['close']
    
    def _calc_pnl_pips(self, entry_price, exit_price, direction):
        """Calculate PnL in dollar amount (MT5-style pip-based)"""
        price_diff = exit_price - entry_price
        pips_moved = price_diff / self.pip_size
        pnl = direction * pips_moved * self.pip_value * self.lot_size
        return pnl
    
    def step(self, action):
        prev_equity = self.equity
        prev_position = self.position
        current_price = self._get_current_price()
        
        # Next bar data for intra-bar SL/TP check
        next_bar = self.df.iloc[self.step_idx + 1]
        next_high = next_bar['high']
        next_low = next_bar['low']
        next_close = next_bar['close']
        
        reward = 0.0
        trade_executed = False
        sl_tp_closed = False
        close_pnl = 0.0
        
        # ===== SL/TP AUTO-CLOSE CHECK (Intra-bar using High/Low like MT5) =====
        if self.position != 0 and self.entry_price > 0:
            if self.position == 1:  # Long position
                sl_price = self.entry_price - self.sl_pips * self.pip_size
                tp_price = self.entry_price + self.tp_pips * self.pip_size
                bar_hit_sl = next_low <= sl_price
                bar_hit_tp = next_high >= tp_price
            else:  # Short position (position == -1)
                sl_price = self.entry_price + self.sl_pips * self.pip_size
                tp_price = self.entry_price - self.tp_pips * self.pip_size
                bar_hit_sl = next_high >= sl_price
                bar_hit_tp = next_low <= tp_price
            
            if bar_hit_sl and bar_hit_tp:
                # Both SL and TP hit in same bar — assume SL hit first (conservative)
                close_pnl = self._calc_pnl_pips(self.entry_price, sl_price, self.position)
                self._close_position_at(sl_price)
                self.sl_hits += 1
                sl_tp_closed = True
                trade_executed = True
            elif bar_hit_sl:
                # Stop Loss hit
                close_pnl = self._calc_pnl_pips(self.entry_price, sl_price, self.position)
                self._close_position_at(sl_price)
                self.sl_hits += 1
                sl_tp_closed = True
                trade_executed = True
            elif bar_hit_tp:
                # Take Profit hit
                close_pnl = self._calc_pnl_pips(self.entry_price, tp_price, self.position)
                self._close_position_at(tp_price)
                self.tp_hits += 1
                sl_tp_closed = True
                trade_executed = True
            elif self.hold_steps >= self.max_hold_steps:
                # ถือนานเกินไป → บังคับปิดที่ราคาปิด
                close_pnl = self._calc_pnl_pips(self.entry_price, next_close, self.position)
                self._close_position_at(next_close)
                sl_tp_closed = True
                trade_executed = True
        
        # ===== ACTION LOGIC (skip if SL/TP just closed) =====
        if sl_tp_closed:
            pass
        elif action == 0:  # HOLD
            pass
            
        elif action == 1:  # BUY (Long)
            if self.position == -1:
                # Close short first, then open long
                close_pnl = self._calc_pnl_pips(self.entry_price, current_price, self.position)
                self._close_position_at(current_price)
                self._open_position(1, current_price)
                trade_executed = True
            elif self.position == 0:
                self._open_position(1, current_price)
                trade_executed = True
            
        elif action == 2:  # SELL (Short)
            if self.position == 1:
                close_pnl = self._calc_pnl_pips(self.entry_price, current_price, self.position)
                self._close_position_at(current_price)
                self._open_position(-1, current_price)
                trade_executed = True
            elif self.position == 0:
                self._open_position(-1, current_price)
                trade_executed = True
            
        elif action == 3:  # CLOSE
            if self.position != 0 and not sl_tp_closed:
                close_pnl = self._calc_pnl_pips(self.entry_price, current_price, self.position)
                self._close_position_at(current_price)
                trade_executed = True
        
        # ===== PRICE MOVEMENT & PnL (pip-based, like MT5) =====
        if self.position != 0:
            # Update unrealized PnL based on next bar's close
            self.unrealized_pnl = self._calc_pnl_pips(
                self.entry_price, next_close, self.position
            )
            self.equity = self.balance + self.unrealized_pnl
            self.hold_steps += 1
            
            # Track prediction accuracy
            next_return = next_close - current_price
            if trade_executed and prev_position == 0:
                self.total_predictions += 1
                if (self.position == 1 and next_return > 0) or (self.position == -1 and next_return < 0):
                    self.correct_predictions += 1
        
        self.max_equity = max(self.max_equity, self.equity)
        drawdown = (self.max_equity - self.equity) / self.max_equity if self.max_equity > 0 else 0
        
        # ===== REWARD CALCULATION (Balanced Profitability v5) =====
        
        # 1. Base PnL reward (Pip-based instead of balance-based)
        point_value = self.pip_value * self.lot_size
        pips_change = (self.equity - prev_equity) / point_value if point_value > 0 else 0.0
        
        if pips_change > 0:
            reward = np.clip(pips_change / 20.0, 0, 1.0) # max out reward at 20 pips
        else:
            reward = np.clip(pips_change / 20.0, -1.0, 0)
        
        # 2. Gentle Floating Loss Penalty
        if self.position != 0 and self.unrealized_pnl < 0:
            reward -= 0.02
        elif self.position != 0 and self.unrealized_pnl > 0:
            reward += 0.05  # Encourage holding winners

        # 3. Trade Entry Penalty (Spread awareness)
        if trade_executed and not sl_tp_closed:
            reward -= 0.1
        
        # 4. Close Trade Reward
        if close_pnl != 0:
            close_pips = close_pnl / point_value if point_value > 0 else 0.0
            self.trade_returns.append(close_pips) # Track pips instead of % return for Sharpe
            if close_pnl > 0:
                reward += min(close_pips / 30.0, 1.0) # 30 pips gives 1.0 reward
            else:
                reward -= min(abs(close_pips) / 40.0, 1.0)
        
        # 5. SL/TP specific bonuses/penalties
        if sl_tp_closed:
            if close_pnl > 0:
                reward += 1.0  # Strong reward for hitting TP
            else:
                reward -= 0.5  # Standard penalty for SL
        
        # 6. Hold too long penalty
        if self.position != 0 and self.hold_steps >= self.max_hold_steps * 0.9:
            reward -= 0.1
        
        # 7. Drawdown survival penalty
        if drawdown > 0.08:
            reward -= (drawdown - 0.08) * 3
        
        # 8. Stay flat bonus
        if action == 0 and self.position == 0:
            reward += 0.01
        
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
            total_pips = (self.equity - self.initial_balance) / point_value if point_value > 0 else 0.0
            if total_pips > 0:
                reward += min(total_pips / 200.0, 1.0) # Bonus caps at 200 pips
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
        """เปิด position ใหม่ (MT5-style: จ่าย spread + commission คงที่)"""
        self.position = direction
        self.entry_price = price
        self.hold_steps = 0
        self.unrealized_pnl = 0.0
        
        # Fixed cost: spread + commission (like MT5)
        cost = self.spread_cost + self.commission_cost
        self.equity -= cost
        self.balance -= cost
        self.total_fees += cost
    
    def _close_position_at(self, exit_price):
        """ปิด position และ realize PnL (MT5-style pip-based)"""
        if self.position == 0:
            return
        
        # Realize PnL (already calculated by caller or from unrealized)
        realized_pnl = self._calc_pnl_pips(self.entry_price, exit_price, self.position)
        self.total_pnl += realized_pnl
        self.balance += realized_pnl
        self.equity = self.balance
        
        # Track win/loss
        self.trades += 1
        if realized_pnl > 0:
            self.wins += 1
        
        # Commission on close (if applicable)
        if self.commission_cost > 0:
            self.equity -= self.commission_cost
            self.balance -= self.commission_cost
            self.total_fees += self.commission_cost
        
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
