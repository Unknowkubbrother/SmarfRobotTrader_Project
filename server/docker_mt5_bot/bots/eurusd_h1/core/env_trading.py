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
        max_hold_steps=30,
        random_start=False,
        recent_bias=0.0,
        recent_lookback=1200,
        min_episode_bars=250,
        entry_penalty=0.12,
        ranging_entry_penalty=0.05,
        counter_trend_penalty=0.06,
        trend_entry_bonus=0.03,
        loss_close_scale=35.0,
        win_close_scale=30.0,
    ):
        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.lot_size = lot_size
        self.pip_size = pip_size
        self.pip_value = pip_value
        self.spread_pips = spread_pips
        self.commission_per_lot = commission_per_lot
        self.max_dd = max_dd
        self.max_hold_steps = max_hold_steps
        self.random_start = random_start
        self.recent_bias = float(np.clip(recent_bias, 0.0, 1.0))
        self.recent_lookback = int(max(recent_lookback, window_size + 1))
        self.min_episode_bars = int(max(min_episode_bars, window_size + 1))
        self.entry_penalty = float(entry_penalty)
        self.ranging_entry_penalty = float(ranging_entry_penalty)
        self.counter_trend_penalty = float(counter_trend_penalty)
        self.trend_entry_bonus = float(trend_entry_bonus)
        self.loss_close_scale = float(max(loss_close_scale, 1.0))
        self.win_close_scale = float(max(win_close_scale, 1.0))


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
        self.position = 0
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


        self.correct_predictions = 0
        self.total_predictions = 0


        self.trade_returns = []

        return self._get_obs(), {}

    def _get_obs(self):
        start = self.step_idx - self.window_size
        end = self.step_idx

        feature_columns = self._get_feature_columns()
        available_cols = [col for col in feature_columns if col in self.df.columns]

        block = self.df.iloc[start:end][available_cols].values
        market_data = block.flatten().astype(np.float32)


        unrealized_ret = 0.0
        unrealized_pips = 0.0
        if self.position != 0 and self.entry_price > 0:
            current_price = self._get_current_price()
            unrealized_ret = self.position * (current_price - self.entry_price) / self.entry_price
            unrealized_pips = self.position * (current_price - self.entry_price) / self.pip_size


        total_pnl_pips = self.total_pnl / (self.pip_value * self.lot_size) if self.lot_size > 0 else 0.0

        state_data = np.array([
            self.position,
            total_pnl_pips / 1000.0,
            unrealized_pips / 100.0,
            min(self.hold_steps / self.max_hold_steps, 1.0),
            np.clip(unrealized_ret * 100, -5, 5),
        ], dtype=np.float32)

        return np.concatenate([market_data, state_data])

    def _get_current_price(self):
        return self.df.iloc[self.step_idx]['close']

    def _calc_pnl_pips(self, entry_price, exit_price, direction):
        price_diff = exit_price - entry_price
        pips_moved = price_diff / self.pip_size
        pnl = direction * pips_moved * self.pip_value * self.lot_size
        return pnl

    def step(self, action):
        prev_equity = self.equity
        prev_position = self.position
        current_price = self._get_current_price()


        next_bar = self.df.iloc[self.step_idx + 1]
        next_close = next_bar['close']

        reward = 0.0
        trade_executed = False
        timed_out_closed = False
        close_pnl = 0.0
        opened_new_position = False


        if self.position != 0 and self.entry_price > 0 and self.hold_steps >= self.max_hold_steps:
            close_pnl = self._calc_pnl_pips(self.entry_price, next_close, self.position)
            self._close_position_at(next_close)
            timed_out_closed = True
            trade_executed = True


        if timed_out_closed:
            pass
        elif action == 0:
            pass

        elif action == 1:
            if self.position == -1:

                close_pnl = self._calc_pnl_pips(self.entry_price, current_price, self.position)
                self._close_position_at(current_price)
                self._open_position(1, current_price)
                trade_executed = True
                opened_new_position = True
            elif self.position == 0:
                self._open_position(1, current_price)
                trade_executed = True
                opened_new_position = True

        elif action == 2:
            if self.position == 1:
                close_pnl = self._calc_pnl_pips(self.entry_price, current_price, self.position)
                self._close_position_at(current_price)
                self._open_position(-1, current_price)
                trade_executed = True
                opened_new_position = True
            elif self.position == 0:
                self._open_position(-1, current_price)
                trade_executed = True
                opened_new_position = True

        elif action == 3:
            if self.position != 0:
                close_pnl = self._calc_pnl_pips(self.entry_price, current_price, self.position)
                self._close_position_at(current_price)
                trade_executed = True


        if self.position != 0:

            self.unrealized_pnl = self._calc_pnl_pips(
                self.entry_price, next_close, self.position
            )
            self.equity = self.balance + self.unrealized_pnl
            self.hold_steps += 1


            next_return = next_close - current_price
            if trade_executed and prev_position == 0:
                self.total_predictions += 1
                if (self.position == 1 and next_return > 0) or (self.position == -1 and next_return < 0):
                    self.correct_predictions += 1

        self.max_equity = max(self.max_equity, self.equity)
        drawdown = (self.max_equity - self.equity) / self.max_equity if self.max_equity > 0 else 0


        point_value = self.pip_value * self.lot_size
        pips_change = (self.equity - prev_equity) / point_value if point_value > 0 else 0.0

        if pips_change > 0:
            reward = np.clip(pips_change / 20.0, 0, 1.0)
        else:
            reward = np.clip(pips_change / 20.0, -1.0, 0)


        if self.position != 0 and self.unrealized_pnl < 0:
            reward -= 0.02
        elif self.position != 0 and self.unrealized_pnl > 0:
            reward += 0.05


        if opened_new_position:
            reward -= self.entry_penalty

            row = self.df.iloc[self.step_idx]
            trend_val = float(row['trend']) if 'trend' in self.df.columns else 0.0
            adx_val = float(row['adx']) if 'adx' in self.df.columns else 0.0
            sma_cross = float(row['sma_cross']) if 'sma_cross' in self.df.columns else 0.0
            direction = self.position

            counter_trend = (
                (direction == 1 and (trend_val < -0.05 or sma_cross < 0))
                or (direction == -1 and (trend_val > 0.05 or sma_cross > 0))
            )
            weak_trend = abs(trend_val) < 0.05 and adx_val < -0.05
            strong_aligned = (
                ((direction == 1 and trend_val > 0.12) or (direction == -1 and trend_val < -0.12))
                and adx_val > 0.15
            )

            if counter_trend:
                reward -= self.counter_trend_penalty
            if weak_trend:
                reward -= self.ranging_entry_penalty
            if strong_aligned:
                reward += self.trend_entry_bonus


        if close_pnl != 0:
            close_pips = close_pnl / point_value if point_value > 0 else 0.0
            self.trade_returns.append(close_pips)
            if close_pnl > 0:
                reward += min(close_pips / self.win_close_scale, 1.0)
            else:
                reward -= min(abs(close_pips) / self.loss_close_scale, 1.2)


        if self.position != 0 and self.hold_steps >= self.max_hold_steps * 0.9:
            reward -= 0.1


        if drawdown > 0.08:
            reward -= (drawdown - 0.08) * 3


        if action == 0 and self.position == 0:
            reward += 0.01


        self.step_idx += 1


        truncated = self.step_idx >= self.max_step
        terminated = (
            drawdown >= self.max_dd
            or self.equity <= 0
        )
        done = terminated or truncated


        if done:
            total_pips = (self.equity - self.initial_balance) / point_value if point_value > 0 else 0.0
            if total_pips > 0:
                reward += min(total_pips / 200.0, 1.0)
            else:
                reward -= 0.2


            if len(self.trade_returns) > 5:
                avg_ret = np.mean(self.trade_returns)
                std_ret = np.std(self.trade_returns) + 1e-8
                sharpe = avg_ret / std_ret
                reward += np.clip(sharpe * 0.3, -0.5, 0.5)


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

        return self._get_obs(), reward, terminated, truncated, info

    def _open_position(self, direction, price):
        self.position = direction
        self.entry_price = price
        self.hold_steps = 0
        self.unrealized_pnl = 0.0


        cost = self.spread_cost + self.commission_cost
        self.equity -= cost
        self.balance -= cost
        self.total_fees += cost

    def _close_position_at(self, exit_price):
        if self.position == 0:
            return


        realized_pnl = self._calc_pnl_pips(self.entry_price, exit_price, self.position)
        self.total_pnl += realized_pnl
        self.balance += realized_pnl
        self.equity = self.balance


        self.trades += 1
        if realized_pnl > 0:
            self.wins += 1


        if self.commission_cost > 0:
            self.equity -= self.commission_cost
            self.balance -= self.commission_cost
            self.total_fees += self.commission_cost


        self.position = 0
        self.entry_price = 0.0
        self.hold_steps = 0
        self.unrealized_pnl = 0.0

    def render(self, mode='human'):
        pos_str = {-1: "SHORT", 0: "FLAT", 1: "LONG"}[self.position]
        print(f"Step: {self.step_idx} | {pos_str} | "
              f"Equity: ${self.equity:.2f} | PnL: ${self.total_pnl:.2f} | "
              f"Trades: {self.trades} | WR: {(self.wins/self.trades*100) if self.trades > 0 else 0:.1f}%")
