import gym
import numpy as np

class TradingEnv(gym.Env):
    def __init__(
        self,
        df,
        window_size=30,  # Add window size
        initial_balance=10_000,
        lot_size=1.0,
        fee=0.0001,
        max_dd=0.3
    ):
        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.lot_size = lot_size
        self.fee = fee
        self.max_dd = max_dd

        self.max_step = len(df) - 2

        self.action_space = gym.spaces.Discrete(3)  # 0: Neutral, 1: Long, 2: Short

        # Market Features: 5 (return, range, delta_tick, delta_price, has_delta)
        # State Features: 2 (position, equity_ratio)
        self.market_features = 5
        self.state_features = 2
        total_features = (self.window_size * self.market_features) + self.state_features
        
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(total_features,), dtype=np.float32
        )

        self.reset()
        
    def seed(self, seed=None):
        return [seed]

    def reset(self):
        self.step_idx = self.window_size  # Start after enough history
        self.position = 0
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.prev_equity = self.initial_balance
        self.max_equity = self.initial_balance
        self.trades = 0
        self.wins = 0
        self.hold_steps = 0
        self.current_trade_pnl = 0
        self.total_fees = 0.0
        return self._get_obs()

    def _get_obs(self):
        # Get window of data
        # From (step_idx - window_size) to step_idx
        start = self.step_idx - self.window_size
        end = self.step_idx
        
        block = self.df.iloc[start:end][['return', 'range', 'delta_tick', 'delta_price', 'has_delta']].values
        
        # We need to append agent state (position, equity) to EACH step in window? 
        # Or just append current state to the flattened vector?
        # Usually it's better to append current state separately or repeat it.
        # For simplicity and sticking to MlpPolicy, let's just use the market data window 
        # AND append current state features as the last part, 
        # OR attach state to every timestep (redundant but easier for Conv1D).
        # Let's stick to flattened.
        
        # Let's verify shape. 
        # If we stick to 7 features per step, we need to artificially construct history for 'position' and 'equity'?
        # No, 'position' and 'equity' are internal states. 
        # Let's clean this up: Market Data (Window) + Agent State (Current).
        # But to keep observation_space simple (Box), we concatenate.
        
        # Better approach:
        # Obs = Flatten(Market_Window) + [Position, Equity_Ratio]
        # But 'Market_Window' is 5 features. 
        # shape = (window_size * 5) + 2
        
        # Let's adjust observation_space definition if we change feature count.
        # But to keep it consistent with "7 features per step" idea (maybe easier), 
        # we can just attach 0s or current state to history? No that's messy.
        
        # Let's do: Obs = Flatten(Last N rows of 5 features) + [Position, Equity]
        # Total dims = N*5 + 2.
        
        # wait, I need to update __init__ observation_space if I do this.
        # Let's do that.
        
        market_data = block.flatten().astype(np.float32)
        state_data = np.array([self.position, self.equity / self.initial_balance], dtype=np.float32)
        
        return np.concatenate([market_data, state_data])

    def step(self, action):
        prev_position = self.position
        self.prev_equity = self.equity

        # action logic (Target Position)
        # 0: Neutral, 1: Long, 2: Short
        if action == 0:
            self.position = 0
        elif action == 1:
            self.position = 1
        elif action == 2:
            self.position = -1


        # price movement
        ret = self.df.iloc[self.step_idx + 1]['raw_return']
        pnl = self.position * ret * self.lot_size * self.equity
        self.equity += pnl

        if self.position != prev_position:
            # fee logic
            fee_cost = self.fee * self.equity
            self.equity -= fee_cost
            self.total_fees += fee_cost
            
            # Trade Logic
            if prev_position != 0:
                self.trades += 1
                if self.current_trade_pnl > 0:
                    self.wins += 1
            
            self.current_trade_pnl = 0
            self.hold_steps = 0
            
            # Penalize trading to reduce churn
            trade_penalty = 0.0005
        else:
            trade_penalty = 0.0

        self.balance = self.equity
        self.max_equity = max(self.max_equity, self.equity)

        # Update PnL calculation for current trade
        if self.position != 0:
            self.hold_steps += 1
            self.current_trade_pnl += pnl

        # ===== reward (สำคัญมาก) =====
        reward = (self.equity - self.prev_equity) / self.initial_balance
        
        # Apply trade penalty
        reward -= trade_penalty

        # drawdown penalty
        drawdown = (self.max_equity - self.equity) / self.max_equity
        reward -= 0.1 * drawdown

        # step
        self.step_idx += 1

        done = (
            self.step_idx >= self.max_step
            or drawdown >= self.max_dd
        )

        info = {
            "equity": self.equity,
            "drawdown": drawdown,
            "position": self.position,
            "trades": self.trades,
            "wins": self.wins,
            "hold_steps": self.hold_steps,
            "fees": self.total_fees
        }


        return self._get_obs(), reward, done, info
