import gym
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
        max_dd=0.20,             # Max drawdown 20%
    ):
        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.lot_size = lot_size
        self.spread_cost = spread_cost
        self.commission = commission
        self.max_dd = max_dd
        
        self.max_step = len(df) - 2
        
        # 4 Actions: Hold, Buy(Long), Sell(Short), Close
        self.action_space = gym.spaces.Discrete(4)
        
        # Features: 7 market + 4 state
        self.market_features = 7
        self.state_features = 4
        total_features = (self.window_size * self.market_features) + self.state_features
        
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(total_features,), dtype=np.float32
        )
        
        self.reset()
    
    def seed(self, seed=None):
        np.random.seed(seed)
        return [seed]
    
    def reset(self):
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
        
        # Track predictions
        self.correct_predictions = 0
        self.total_predictions = 0
        
        return self._get_obs()
    
    def _get_obs(self):
        start = self.step_idx - self.window_size
        end = self.step_idx
        
        feature_columns = [
            'return', 'range', 'delta_tick', 'delta_price', 'has_delta',
            'body_ratio', 'momentum'
        ]
        available_cols = [col for col in feature_columns if col in self.df.columns]
        
        block = self.df.iloc[start:end][available_cols].values
        market_data = block.flatten().astype(np.float32)
        
        # State: position, equity ratio, unrealized pnl, hold steps
        state_data = np.array([
            self.position,
            self.equity / self.initial_balance,
            self.unrealized_pnl / self.initial_balance,
            min(self.hold_steps / 50, 1.0)  # Normalized hold time
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
        
        # ===== ACTION LOGIC =====
        # 0: HOLD - ไม่ทำอะไร
        # 1: BUY - ทำนายว่าขึ้น (Long)
        # 2: SELL - ทำนายว่าลง (Short)
        # 3: CLOSE - ปิด order
        
        if action == 0:  # HOLD
            pass  # ไม่ทำอะไร
            
        elif action == 1:  # BUY (ทำนายว่าขึ้น)
            if self.position == -1:
                # มี Short อยู่ -> ปิด Short ก่อน แล้วเปิด Long
                self._close_position(current_price)
                self._open_position(1, current_price)
                trade_executed = True
            elif self.position == 0:
                # ไม่มี position -> เปิด Long
                self._open_position(1, current_price)
                trade_executed = True
            # ถ้ามี Long อยู่แล้ว -> ถือต่อ (ไม่ทำอะไร)
            
        elif action == 2:  # SELL (ทำนายว่าลง)
            if self.position == 1:
                # มี Long อยู่ -> ปิด Long ก่อน แล้วเปิด Short
                self._close_position(current_price)
                self._open_position(-1, current_price)
                trade_executed = True
            elif self.position == 0:
                # ไม่มี position -> เปิด Short
                self._open_position(-1, current_price)
                trade_executed = True
            # ถ้ามี Short อยู่แล้ว -> ถือต่อ
            
        elif action == 3:  # CLOSE
            if self.position != 0:
                self._close_position(current_price)
                trade_executed = True
        
        # ===== PRICE MOVEMENT & PnL =====
        if self.position != 0:
            # คำนวณ PnL จากการเคลื่อนไหวของราคา
            pnl = self.position * next_return * self.lot_size * self.equity
            self.unrealized_pnl += pnl
            self.equity += pnl
            self.hold_steps += 1
            
            # Track prediction accuracy
            if trade_executed and prev_position == 0:
                self.total_predictions += 1
                # ทำนายถูกถ้า: Long & ราคาขึ้น หรือ Short & ราคาลง
                if (self.position == 1 and next_return > 0) or (self.position == -1 and next_return < 0):
                    self.correct_predictions += 1
        
        self.max_equity = max(self.max_equity, self.equity)
        # ===== REWARD CALCULATION (Simpler & More Robust) =====
        
        # 1. PnL-based reward - ใช้ log scale เพื่อลด extreme rewards
        equity_change = (self.equity - prev_equity) / self.initial_balance
        reward = np.sign(equity_change) * np.sqrt(abs(equity_change)) * 5
        
        # 2. สนับสนุน HOLD (ไม่เทรดเกินไป)
        if action == 0 and self.position == 0:
            reward += 0.0001  # Small bonus for staying flat when uncertain
        
        # 3. Penalty สำหรับการเปลี่ยน position บ่อย
        if trade_executed:
            reward -= 0.001
        
        # 4. Simple win/loss bonus at close
        # (ให้ reward เมื่อปิด position กำไร)
        
        # ===== STEP FORWARD =====
        self.step_idx += 1
        
        # ===== DONE CONDITIONS =====
        drawdown = (self.max_equity - self.equity) / self.max_equity if self.max_equity > 0 else 0
        
        done = (
            self.step_idx >= self.max_step
            or drawdown >= self.max_dd
            or self.equity <= 0
        )
        
        # Simple end bonus
        if done:
            final_return = (self.equity - self.initial_balance) / self.initial_balance
            if final_return > 0:
                reward += 0.1  # Bonus for positive return
            else:
                reward -= 0.05  # Small penalty for loss
        
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
        }
        
        return self._get_obs(), reward, done, info
    
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
