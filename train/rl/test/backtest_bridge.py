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

        self.open_positions = []
        self.position = 0
        self.entry_price = 0.0
        self.hold_steps = 0
        self.net_units = 0
        self.long_units = 0
        self.short_units = 0
        self.open_orders = 0

        self.equity = INITIAL_BALANCE
        self.balance = INITIAL_BALANCE
        self.unrealized_pnl = 0.0
        self.trades = 0
        self.wins = 0
        self.total_pnl = 0.0
        self.total_fees = 0.0
        self.max_equity = INITIAL_BALANCE
        self.balance_scale = max(float(INITIAL_BALANCE), 1e-9)
        self.net_units_obs_scale = 5.0
        self.open_orders_obs_scale = 5.0

        self.risk_level = RISK_LEVEL
        self.risk_percent = _resolve_risk_percent(risk_level=self.risk_level)
        self.lot_size = calc_auto_lot(INITIAL_BALANCE, risk_level=self.risk_level)
        self.spread_cost = self._spread_cost_for_lot(self.lot_size)
        self._refresh_position_summary()

        print(
            f"\n Auto Lot: {self.lot_size} "
            f"(Balance: ${INITIAL_BALANCE}, RiskLevel: {self.risk_level}, Risk: {self.risk_percent}%)"
        )
        print(" Bridge execution mode: rl_direct (no rule-based guard)")

    def _ratio_to_unit01(self, value, full_scale):
        scale = max(float(full_scale), 1e-9)
        signed = np.tanh(float(value) / scale)
        return float(np.clip(0.5 * (signed + 1.0), 0.0, 1.0))

    def _spread_cost_for_lot(self, lot):
        return SPREAD_PIPS * PIP_VALUE * float(max(lot, 0.0))

    def _calc_pnl(self, entry, exit_price, direction, lot):
        pips = (float(exit_price) - float(entry)) / PIP_SIZE
        return float(direction) * pips * PIP_VALUE * float(max(lot, 0.0))

    def _flatten_window(self, block):
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
        return np.array(flattened_rows, dtype=np.float32)

    def _positions_by_direction(self, direction):
        d = int(direction)
        return [pos for pos in self.open_positions if int(pos["direction"]) == d]

    def _refresh_position_summary(self):
        self.long_units = sum(1 for pos in self.open_positions if int(pos["direction"]) == 1)
        self.short_units = sum(1 for pos in self.open_positions if int(pos["direction"]) == -1)
        self.net_units = int(self.long_units - self.short_units)
        self.open_orders = int(len(self.open_positions))
        self.position = int(np.sign(self.net_units))

        if self.position > 0 and self.long_units > 0:
            self.entry_price = float(
                np.mean([float(pos["entry_price"]) for pos in self.open_positions if int(pos["direction"]) == 1])
            )
        elif self.position < 0 and self.short_units > 0:
            self.entry_price = float(
                np.mean([float(pos["entry_price"]) for pos in self.open_positions if int(pos["direction"]) == -1])
            )
        else:
            self.entry_price = 0.0

        if self.open_positions:
            avg_hold = float(np.mean([float(pos.get("hold_steps", 0)) for pos in self.open_positions]))
            self.hold_steps = int(round(avg_hold))
        else:
            self.hold_steps = 0

    def _calc_unrealized_pnl_at(self, mark_price):
        if not self.open_positions:
            return 0.0
        return float(
            sum(
                self._calc_pnl(pos["entry_price"], mark_price, pos["direction"], pos.get("lot", self.lot_size))
                for pos in self.open_positions
            )
        )

    def _mark_to_market(self, mark_price):
        if self.open_positions:
            for pos in self.open_positions:
                pos["hold_steps"] = int(pos.get("hold_steps", 0)) + 1
            self.unrealized_pnl = self._calc_unrealized_pnl_at(mark_price)
            self.equity = self.balance + self.unrealized_pnl
        else:
            self.unrealized_pnl = 0.0
            self.equity = self.balance
        self._refresh_position_summary()

    def sync_from_broker(self, direction, entry_price=0.0, volume=0.0, reset_hold=False, order_lot=None):
        dir_sign = int(np.sign(int(direction)))
        lot = float(max(volume, 0.0))
        if dir_sign == 0 or lot <= 0.0:
            self.open_positions = []
            self.unrealized_pnl = 0.0
            self.hold_steps = 0
        else:
            lot_ref = float(max(order_lot if order_lot is not None else self.lot_size, 1e-9))
            estimated_orders = int(max(1, round(lot / lot_ref)))
            per_order_lot = float(max(lot / estimated_orders, 1e-9))
            hold = 0 if reset_hold else int(max(self.hold_steps, 0))
            self.open_positions = []
            for _ in range(estimated_orders):
                self.open_positions.append(
                    {
                        "direction": dir_sign,
                        "entry_price": float(entry_price),
                        "hold_steps": hold,
                        "lot": per_order_lot,
                    }
                )
        self._refresh_position_summary()

    def _open_position(self, direction, price, lot=None):
        lot_to_use = float(max(self.lot_size if lot is None else lot, 0.0))
        if lot_to_use <= 0.0:
            return False
        self.open_positions.append(
            {
                "direction": int(direction),
                "entry_price": float(price),
                "hold_steps": 0,
                "lot": lot_to_use,
            }
        )
        cost = self._spread_cost_for_lot(lot_to_use)
        self.spread_cost = self._spread_cost_for_lot(self.lot_size)
        self.balance -= cost
        self.equity -= cost
        self.total_fees += cost
        self._refresh_position_summary()
        return True

    def _close_positions_at(self, exit_price):
        if not self.open_positions:
            return False
        closing = list(self.open_positions)
        self.open_positions = []
        for pos in closing:
            pnl = self._calc_pnl(pos["entry_price"], exit_price, pos["direction"], pos.get("lot", self.lot_size))
            self.total_pnl += pnl
            self.balance += pnl
            self.trades += 1
            if pnl > 0:
                self.wins += 1
        self.unrealized_pnl = 0.0
        self.equity = self.balance
        self._refresh_position_summary()
        return True

    def process_bar(self, df, delta_tick=0, delta_price=0.0):
        df = calculate_features(
            df,
            semantic_runtime=self.semantic_runtime,
            semantic_feature_count=self.semantic_feature_count,
            delta_tick=delta_tick,
            delta_price=delta_price,
        )
        current_price = float(df.iloc[-1]["close"])

        self._mark_to_market(current_price)
        self.max_equity = max(self.max_equity, self.equity)

        if len(df) >= WINDOW_SIZE + 1:
            block = df[self.feature_columns].iloc[-(WINDOW_SIZE + 1) : -1]
        else:
            block = df[self.feature_columns].iloc[-WINDOW_SIZE:]
        if len(block) != WINDOW_SIZE:
            raise ValueError(f"Insufficient window for observation: need={WINDOW_SIZE}, got={len(block)}")

        obs_window = self._flatten_window(block)
        total_pnl_ratio = self.total_pnl / self.balance_scale
        unrealized_pnl_ratio = self.unrealized_pnl / self.balance_scale

        state_feat = np.array(
            [
                self._ratio_to_unit01(self.net_units, full_scale=self.net_units_obs_scale),
                np.tanh(self.open_orders / self.open_orders_obs_scale),
                self._ratio_to_unit01(total_pnl_ratio, full_scale=0.20),
                self._ratio_to_unit01(unrealized_pnl_ratio, full_scale=0.05),
                np.tanh(self.hold_steps / max(float(WINDOW_SIZE), 1.0)),
            ],
            dtype=np.float32,
        )

        full_obs = np.concatenate([obs_window, state_feat])
        obs_norm = self.vec_norm.normalize_obs(full_obs)
        action, _ = self.model.predict(obs_norm, deterministic=True)
        action = int(action) if np.isscalar(action) else int(np.asarray(action).reshape(-1)[0])

        exec_action = int(action)

        broker_action = exec_action

        if exec_action in (1, 2):
            direction = 1 if exec_action == 1 else -1
            opposite = self._positions_by_direction(-direction)
            if opposite:
                closed = self._close_positions_at(current_price)
                if closed:
                    opened = self._open_position(direction, current_price)
                    broker_action = exec_action if opened else 3
                else:
                    broker_action = 0
            else:
                opened = self._open_position(direction, current_price)
                broker_action = exec_action if opened else 0
        elif exec_action in (3, 4):
            closed = self._close_positions_at(current_price)
            broker_action = 3 if closed else 0

        return int(broker_action), current_price
