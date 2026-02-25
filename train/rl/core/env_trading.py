import gymnasium as gym
import numpy as np


class TradingEnv(gym.Env):

    def __init__(
        self,
        df,
        window_size=20,
        initial_balance=10_000,

        lot_size=0.1,
        pip_size=0.0001,
        pip_value=10.0,
        spread_pips=2,
        commission_per_lot=0.0,
        max_dd=0.30,
        max_open_orders=5,
        random_start=False,
        recent_bias=0.0,
        recent_lookback=1200,
        min_episode_bars=250,
    ):
        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.balance_scale = max(float(initial_balance), 1e-9)
        self.lot_size = lot_size
        self.pip_size = pip_size
        self.pip_value = pip_value
        self.spread_pips = spread_pips
        self.commission_per_lot = commission_per_lot
        self.max_dd = max_dd
        self.max_open_orders = int(max(1, max_open_orders))
        self.random_start = random_start
        self.recent_bias = float(np.clip(recent_bias, 0.0, 1.0))
        self.recent_lookback = int(max(recent_lookback, window_size + 1))
        self.min_episode_bars = int(max(min_episode_bars, window_size + 1))

        self.spread_cost = self.spread_pips * self.pip_value * self.lot_size
        self.commission_cost = self.commission_per_lot * self.lot_size

        self.max_step = len(df) - 2

        self.action_space = gym.spaces.Discrete(4)
        self.state_features = 5

        self.market_features = 0
        feature_columns = self._get_feature_columns()
        available_cols = [col for col in feature_columns if col in self.df.columns]

        row0 = self.df.iloc[0]
        for col in available_cols:
            val = row0[col]
            if isinstance(val, (list, np.ndarray)):
                self.market_features += len(val)
            else:
                self.market_features += 1

        total_features = (self.window_size * self.market_features) + self.state_features

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(total_features,), dtype=np.float32
        )

        self.reset()

    def _ratio_to_unit01(self, value, full_scale):
        scale = max(float(full_scale), 1e-9)
        signed = np.tanh(float(value) / scale)
        return float(np.clip(0.5 * (signed + 1.0), 0.0, 1.0))

    def _get_feature_columns(self):
        base_cols = [
            'return', 'range', 'delta_tick', 'delta_price',
            'body_ratio', 'momentum',
            'sma_cross', 'rsi_norm', 'atr_norm', 'trend', 'adx'
        ]
        sem_cols = [c for c in self.df.columns if str(c).startswith("sem_pca_")]
        if not sem_cols:
            sem_cols = [c for c in self.df.columns if str(c).startswith("sem_latent_")]
        return base_cols + sem_cols

    def seed(self, seed=None):
        np.random.seed(seed)
        return [seed]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.random_start:
            start_low = self.window_size
            start_high = max(self.window_size + 1, self.max_step - self.min_episode_bars)

            if self.recent_bias > 0 and np.random.rand() < self.recent_bias:
                recent_low = max(start_low, start_high - self.recent_lookback)
                if recent_low < start_high:
                    self.step_idx = np.random.randint(recent_low, start_high)
                else:
                    self.step_idx = start_low
            else:
                if start_low < start_high:
                    self.step_idx = np.random.randint(start_low, start_high)
                else:
                    self.step_idx = start_low
        else:
            self.step_idx = self.window_size

        self.open_positions = []
        self.net_units = 0
        self.long_units = 0
        self.short_units = 0
        self.open_orders = 0
        self.position = 0
        self.entry_price = 0.0
        self.hold_steps = 0
        self.unrealized_pnl = 0.0

        self.equity = self.initial_balance
        self.balance = self.initial_balance
        self.max_equity = self.initial_balance

        self.trades = 0
        self.wins = 0
        self.total_fees = 0.0
        self.total_pnl = 0.0

        self.correct_predictions = 0
        self.total_predictions = 0

        return self._get_obs(), {}

    def _get_obs(self):
        start = self.step_idx - self.window_size
        end = self.step_idx

        feature_columns = self._get_feature_columns()
        available_cols = [col for col in feature_columns if col in self.df.columns]

        block = self.df.iloc[start:end][available_cols].values
        market_data = block.flatten().astype(np.float32)

        total_pnl_ratio = self.total_pnl / self.balance_scale
        unrealized_pnl_ratio = self.unrealized_pnl / self.balance_scale

        state_data = np.array([
            np.clip((self.net_units / float(self.max_open_orders) + 1.0) * 0.5, 0.0, 1.0),
            self.open_orders / float(self.max_open_orders),
            self._ratio_to_unit01(total_pnl_ratio, full_scale=0.20),
            self._ratio_to_unit01(unrealized_pnl_ratio, full_scale=0.05),
            np.tanh(self.hold_steps / max(float(self.window_size), 1.0)),
        ], dtype=np.float32)

        return np.concatenate([market_data, state_data])

    def _get_current_price(self):
        return self.df.iloc[self.step_idx]['close']

    def _calc_pnl_pips(self, entry_price, exit_price, direction):
        price_diff = exit_price - entry_price
        pips_moved = price_diff / self.pip_size
        pnl = direction * pips_moved * self.pip_value * self.lot_size
        return pnl

    def _refresh_position_summary(self):
        self.long_units = sum(1 for pos in self.open_positions if pos["direction"] == 1)
        self.short_units = sum(1 for pos in self.open_positions if pos["direction"] == -1)
        self.net_units = self.long_units - self.short_units
        self.open_orders = len(self.open_positions)
        self.position = int(np.sign(self.net_units))

        if self.position > 0 and self.long_units > 0:
            self.entry_price = float(np.mean([pos["entry_price"] for pos in self.open_positions if pos["direction"] == 1]))
        elif self.position < 0 and self.short_units > 0:
            self.entry_price = float(np.mean([pos["entry_price"] for pos in self.open_positions if pos["direction"] == -1]))
        else:
            self.entry_price = 0.0

        if self.open_positions:
            avg_hold = float(np.mean([pos["hold_steps"] for pos in self.open_positions]))
            self.hold_steps = int(round(avg_hold))
        else:
            self.hold_steps = 0

    def _calc_unrealized_pnl_at(self, mark_price):
        if not self.open_positions:
            return 0.0
        return float(sum(self._calc_pnl_pips(pos["entry_price"], mark_price, pos["direction"]) for pos in self.open_positions))

    def _mark_to_market(self, mark_price):
        if self.open_positions:
            for pos in self.open_positions:
                pos["hold_steps"] += 1
            self.unrealized_pnl = self._calc_unrealized_pnl_at(mark_price)
            self.equity = self.balance + self.unrealized_pnl
        else:
            self.unrealized_pnl = 0.0
            self.equity = self.balance
        self._refresh_position_summary()

    def step(self, action):
        prev_equity = self.equity
        current_price = self._get_current_price()

        next_bar = self.df.iloc[self.step_idx + 1]
        next_close = next_bar['close']
        next_return = next_close - current_price

        trade_executed = False
        directional_action = 0

        if action == 1:
            directional_action = 1
            trade_executed = self._open_position(1, current_price)
        elif action == 2:
            directional_action = -1
            trade_executed = self._open_position(-1, current_price)
        elif action == 3:
            trade_executed = self._close_positions_at(current_price)

        if directional_action != 0 and trade_executed:
            self.total_predictions += 1
            if (directional_action == 1 and next_return > 0) or (directional_action == -1 and next_return < 0):
                self.correct_predictions += 1

        self._mark_to_market(next_close)

        self.max_equity = max(self.max_equity, self.equity)
        drawdown = (self.max_equity - self.equity) / self.max_equity if self.max_equity > 0 else 0
        reward = float(np.clip((self.equity - prev_equity) / self.balance_scale, -1.0, 1.0))

        self.step_idx += 1

        truncated = self.step_idx >= self.max_step
        terminated = (
            drawdown >= self.max_dd
            or self.equity <= 0
        )
        accuracy = (self.correct_predictions / self.total_predictions * 100) if self.total_predictions > 0 else 0

        info = {
            "equity": self.equity,
            "balance": self.balance,
            "equity_ratio": self.equity / self.balance_scale,
            "balance_ratio": self.balance / self.balance_scale,
            "drawdown": drawdown,
            "position": self.position,
            "net_units": self.net_units,
            "long_units": self.long_units,
            "short_units": self.short_units,
            "open_orders": self.open_orders,
            "trades": self.trades,
            "wins": self.wins,
            "win_rate": (self.wins / self.trades * 100) if self.trades > 0 else 0,
            "hold_steps": self.hold_steps,
            "fees": self.total_fees,
            "unrealized_pnl": self.unrealized_pnl,
            "total_pnl": self.total_pnl,
            "unrealized_pnl_ratio": self.unrealized_pnl / self.balance_scale,
            "total_pnl_ratio": self.total_pnl / self.balance_scale,
            "accuracy": accuracy,
            "action": action,
        }

        return self._get_obs(), reward, terminated, truncated, info

    def _open_position(self, direction, price):
        if len(self.open_positions) >= self.max_open_orders:
            return False

        self.open_positions.append({
            "direction": int(direction),
            "entry_price": float(price),
            "hold_steps": 0,
        })
        cost = self.spread_cost + self.commission_cost
        self.balance -= cost
        self.equity -= cost
        self.total_fees += cost
        self._refresh_position_summary()
        return True

    def _close_positions_at(self, exit_price):
        if not self.open_positions:
            return False

        for pos in self.open_positions:
            realized_pnl = self._calc_pnl_pips(pos["entry_price"], exit_price, pos["direction"])
            self.total_pnl += realized_pnl
            self.balance += realized_pnl
            self.trades += 1
            if realized_pnl > 0:
                self.wins += 1
            if self.commission_cost > 0:
                self.balance -= self.commission_cost
                self.total_fees += self.commission_cost

        self.open_positions = []
        self.unrealized_pnl = 0.0
        self.equity = self.balance
        self._refresh_position_summary()
        return True

    def render(self, mode='human'):
        if self.position > 0:
            pos_str = f"NET LONG ({self.net_units:+d})"
        elif self.position < 0:
            pos_str = f"NET SHORT ({self.net_units:+d})"
        else:
            pos_str = "FLAT" if self.open_orders == 0 else "HEDGED (NET 0)"

        print(f"Step: {self.step_idx} | {pos_str} | "
              f"Equity: ${self.equity:.2f} | PnL: ${self.total_pnl:.2f} | "
              f"Open: {self.open_orders}/{self.max_open_orders} | "
              f"Trades: {self.trades} | WR: {(self.wins/self.trades*100) if self.trades > 0 else 0:.1f}%")
