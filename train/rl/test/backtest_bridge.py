import numpy as np

from risk_lot import calc_auto_lot as _calc_auto_lot_shared
from risk_lot import resolve_risk_percent as _resolve_risk_percent_shared
from backtest_config import (
    INITIAL_BALANCE,
    LOT_RISK_PIPS,
    PIP_SIZE,
    PIP_VALUE,
    RISK_LEVEL,
    RISK_PERCENT_HIGH,
    RISK_PERCENT_LOW,
    RISK_PERCENT_MEDIUM,
    SPREAD_PIPS,
    WINDOW_SIZE,
)
from backtest_features import calculate_features


def _resolve_risk_percent(risk_level=None, risk_pct=None):
    return _resolve_risk_percent_shared(
        risk_level=risk_level if risk_level is not None else RISK_LEVEL,
        risk_pct=risk_pct,
        risk_percent_low=RISK_PERCENT_LOW,
        risk_percent_medium=RISK_PERCENT_MEDIUM,
        risk_percent_high=RISK_PERCENT_HIGH,
    )


def calc_auto_lot(balance, risk_pct=None, risk_level=None, pip_value_per_lot=PIP_VALUE, min_lot=0.01, lot_step=0.01):
    return _calc_auto_lot_shared(
        balance=balance,
        risk_pct=risk_pct,
        risk_level=risk_level if risk_level is not None else RISK_LEVEL,
        lot_risk_pips=LOT_RISK_PIPS,
        pip_value_per_lot=pip_value_per_lot,
        min_lot=min_lot,
        lot_step=lot_step,
        risk_percent_low=RISK_PERCENT_LOW,
        risk_percent_medium=RISK_PERCENT_MEDIUM,
        risk_percent_high=RISK_PERCENT_HIGH,
    )


class PPOBridge:
    def __init__(self, model, vec_norm, feature_columns, semantic_runtime, semantic_feature_count, gate_stats=None):
        self.model = model
        self.vec_norm = vec_norm
        self.feature_columns = feature_columns
        self.semantic_runtime = semantic_runtime
        self.semantic_feature_count = semantic_feature_count
        self.gate_stats = gate_stats or {}

        self.position = 0
        self.entry_price = 0.0
        self.hold_steps = 0
        self.equity = INITIAL_BALANCE
        self.balance = INITIAL_BALANCE
        self.unrealized_pnl = 0.0
        self.trades = 0
        self.wins = 0
        self.total_pnl = 0.0
        self.total_fees = 0.0
        self.max_equity = INITIAL_BALANCE
        self.risk_level = RISK_LEVEL
        self.risk_percent = _resolve_risk_percent(risk_level=self.risk_level)
        self.lot_size = calc_auto_lot(INITIAL_BALANCE, risk_level=self.risk_level)
        self.spread_cost = SPREAD_PIPS * PIP_VALUE * self.lot_size

        print(
            f"\n Auto Lot: {self.lot_size} "
            f"(Balance: ${INITIAL_BALANCE}, RiskLevel: {self.risk_level}, Risk: {self.risk_percent}%)"
        )

    def _calc_pnl(self, entry, exit_price, direction):
        pips = (exit_price - entry) / PIP_SIZE
        return direction * pips * PIP_VALUE * self.lot_size

    def _open(self, direction, price):
        self.position = direction
        self.entry_price = price
        self.hold_steps = 0
        self.unrealized_pnl = 0.0
        self.equity -= self.spread_cost
        self.balance -= self.spread_cost
        self.total_fees += self.spread_cost

    def _close(self, exit_price):
        pnl = self._calc_pnl(self.entry_price, exit_price, self.position)
        self.total_pnl += pnl
        self.balance += pnl
        self.equity = self.balance
        self.trades += 1
        if pnl > 0:
            self.wins += 1
        self.position = 0
        self.entry_price = 0.0
        self.hold_steps = 0
        self.unrealized_pnl = 0.0

    def process_bar(self, df, delta_tick=0, delta_price=0.0):
        df = calculate_features(
            df,
            semantic_runtime=self.semantic_runtime,
            semantic_feature_count=self.semantic_feature_count,
            delta_tick=delta_tick,
            delta_price=delta_price,
        )
        current_price = df.iloc[-1]["close"]

        if self.position != 0:
            self.unrealized_pnl = self._calc_pnl(self.entry_price, current_price, self.position)
            self.equity = self.balance + self.unrealized_pnl
            self.hold_steps += 1

        self.max_equity = max(self.max_equity, self.equity)

        block = df[self.feature_columns].iloc[-WINDOW_SIZE:]
        flattened_rows = []
        for _, row in block.iterrows():
            row_data = []
            for col in self.feature_columns:
                val = row[col]
                if isinstance(val, (list, np.ndarray)):
                    row_data.extend(val)
                else:
                    row_data.append(val)
            flattened_rows.extend(row_data)

        obs_window = np.array(flattened_rows, dtype=np.float32)
        unrealized_ret = 0.0
        unrealized_pips = 0.0
        if self.position != 0 and self.entry_price > 0:
            unrealized_ret = self.position * (current_price - self.entry_price) / self.entry_price
            unrealized_pips = self.position * (current_price - self.entry_price) / PIP_SIZE

        total_pnl_pips = self.total_pnl / (PIP_VALUE * self.lot_size) if self.lot_size > 0 else 0.0

        state_feat = np.array(
            [
                self.position,
                total_pnl_pips / 1000.0,
                unrealized_pips / 100.0,
                np.tanh(self.hold_steps / max(float(WINDOW_SIZE), 1.0)),
                np.clip(unrealized_ret * 100, -5, 5),
            ],
            dtype=np.float32,
        )

        full_obs = np.concatenate([obs_window, state_feat])
        obs_norm = self.vec_norm.normalize_obs(full_obs)
        action, _ = self.model.predict(obs_norm, deterministic=True)
        action = int(action) if np.isscalar(action) else int(np.asarray(action).reshape(-1)[0])

        if action == 3 and self.position == 0:
            action = 0

        if action == 1:
            if self.position == -1:
                self._close(current_price)
                self._open(1, current_price)
            elif self.position == 0:
                self._open(1, current_price)
        elif action == 2:
            if self.position == 1:
                self._close(current_price)
                self._open(-1, current_price)
            elif self.position == 0:
                self._open(-1, current_price)
        elif action == 3:
            if self.position != 0:
                self._close(current_price)

        return action, current_price
