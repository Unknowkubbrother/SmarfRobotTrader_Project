from collections import deque

import numpy as np

from backtest_config import (
    ADAPTIVE_GATE,
    COUNTER_TREND_CONF_BONUS,
    COUNTER_TREND_EDGE_BONUS,
    COUNTER_TREND_HOLD_EDGE_BONUS,
    COUNTER_TREND_MARGIN_BONUS,
    DEF_BARS,
    DEF_CONF_BONUS,
    DEF_COOLDOWN_BONUS,
    DEF_EDGE_BONUS,
    DEF_HOLD_EDGE_BONUS,
    DEF_LOOKBACK_TRADES,
    DEF_MARGIN_BONUS,
    DEF_MAX_LOSS_STREAK,
    DEF_MIN_AVG_PIPS,
    DEF_MIN_WINRATE,
    FLAT_CONF_BONUS,
    FLAT_EDGE_BONUS,
    FLAT_HOLD_EDGE_BONUS,
    FLAT_MARGIN_BONUS,
    EMBED_QUALITY_COOLDOWN_BONUS,
    EMBED_QUALITY_CONF_BONUS,
    EMBED_QUALITY_EDGE_BONUS,
    EMBED_QUALITY_HOLD_EDGE_BONUS,
    EMBED_QUALITY_MARGIN_BONUS,
    EMBED_QUALITY_MIN,
    HOLD_EDGE_THRESHOLD,
    INITIAL_BALANCE,
    MAX_HOLD_STEPS,
    MIN_ACTION_MARGIN,
    OPEN_EDGE_THRESHOLD,
    OPEN_PROB_THRESHOLD,
    PIP_SIZE,
    PIP_VALUE,
    RISK_PIPS,
    RISK_PERCENT,
    SPREAD_PIPS,
    TRADE_COOLDOWN_BARS,
    TREND_HOLD_RELAX,
    TREND_RELAX,
    VOL_CONF_BONUS,
    VOL_EDGE_BONUS,
    VOL_HOLD_EDGE_BONUS,
    VOL_MARGIN_BONUS,
    WINDOW_SIZE,
)
from backtest_features import calculate_features


def calc_auto_lot(balance, risk_pct=RISK_PERCENT, risk_pips=RISK_PIPS, pip_value_per_lot=PIP_VALUE, min_lot=0.01, lot_step=0.01):
    risk_amount = balance * risk_pct / 100.0
    lot = risk_amount / (risk_pips * pip_value_per_lot)
    lot = max(min_lot, lot_step * int(lot / lot_step))
    return round(lot, 2)


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
        self.lot_size = calc_auto_lot(INITIAL_BALANCE)
        self.spread_cost = SPREAD_PIPS * PIP_VALUE * self.lot_size
        self.first_bar = True
        self.trade_cooldown = 0
        self.skipped_signals = 0
        self.margin_skips = 0
        self.defensive_skips = 0
        self.defensive_mode_bars = 0
        self.defensive_triggers = 0
        self.loss_streak = 0
        self.recent_trade_pips = deque(maxlen=max(DEF_LOOKBACK_TRADES, 5))
        self.semantic_skips = 0

        print(f"\n Auto Lot: {self.lot_size} (Balance: ${INITIAL_BALANCE}, Risk: {RISK_PERCENT}%)")

    def _calc_pnl(self, entry, exit_price, direction):
        pips = (exit_price - entry) / PIP_SIZE
        return direction * pips * PIP_VALUE * self.lot_size

    def _to_pips(self, pnl):
        denom = PIP_VALUE * self.lot_size
        if denom <= 0:
            return 0.0
        return float(pnl / denom)

    def _update_defensive_mode(self, pnl):
        trade_pips = self._to_pips(pnl)
        self.recent_trade_pips.append(trade_pips)

        if trade_pips <= 0:
            self.loss_streak += 1
        else:
            self.loss_streak = 0

        if not ADAPTIVE_GATE:
            return

        min_trades = min(max(6, DEF_LOOKBACK_TRADES // 2), len(self.recent_trade_pips))
        if min_trades > 0:
            recent = np.array(list(self.recent_trade_pips)[-min_trades:], dtype=np.float32)
            recent_winrate = float(np.mean(recent > 0))
            recent_avg_pips = float(np.mean(recent))
        else:
            recent_winrate = 1.0
            recent_avg_pips = 0.0

        should_defend = (
            self.loss_streak >= DEF_MAX_LOSS_STREAK
            or (min_trades >= 6 and recent_winrate < DEF_MIN_WINRATE)
            or (min_trades >= 6 and recent_avg_pips < DEF_MIN_AVG_PIPS)
        )
        if should_defend:
            self.defensive_mode_bars = max(self.defensive_mode_bars, DEF_BARS)
            self.defensive_triggers += 1

    def _gate_thresholds(self, action, probs, last_bar, semantic_quality):
        conf_thr = OPEN_PROB_THRESHOLD
        edge_thr = OPEN_EDGE_THRESHOLD
        margin_thr = MIN_ACTION_MARGIN
        hold_edge_thr = HOLD_EDGE_THRESHOLD
        cooldown_after_trade = TRADE_COOLDOWN_BARS

        if not ADAPTIVE_GATE:
            return conf_thr, edge_thr, margin_thr, hold_edge_thr, cooldown_after_trade

        atr_norm = float(last_bar.get("atr_norm", 0.0))
        trend = float(last_bar.get("trend", 0.0))
        adx = float(last_bar.get("adx", 0.0))
        sma_cross = float(last_bar.get("sma_cross", 0.0))
        abs_trend = abs(trend)

        atr_high = float(self.gate_stats.get("atr_high", 0.0015))
        atr_extreme = float(self.gate_stats.get("atr_extreme", 0.0025))
        trend_flat = float(self.gate_stats.get("trend_flat", 0.08))
        trend_strong = float(self.gate_stats.get("trend_strong", 0.25))
        adx_flat = float(self.gate_stats.get("adx_flat", -0.20))
        adx_strong = float(self.gate_stats.get("adx_strong", 0.20))

        if self.defensive_mode_bars > 0:
            conf_thr += DEF_CONF_BONUS
            edge_thr += DEF_EDGE_BONUS
            margin_thr += DEF_MARGIN_BONUS
            hold_edge_thr += DEF_HOLD_EDGE_BONUS
            cooldown_after_trade += DEF_COOLDOWN_BONUS

        if atr_norm >= atr_extreme:
            conf_thr += VOL_CONF_BONUS * 1.5
            edge_thr += VOL_EDGE_BONUS * 1.5
            margin_thr += VOL_MARGIN_BONUS * 1.5
            hold_edge_thr += VOL_HOLD_EDGE_BONUS * 1.5
            cooldown_after_trade += 1
        elif atr_norm >= atr_high:
            conf_thr += VOL_CONF_BONUS
            edge_thr += VOL_EDGE_BONUS
            margin_thr += VOL_MARGIN_BONUS
            hold_edge_thr += VOL_HOLD_EDGE_BONUS

        if abs_trend <= trend_flat and adx <= adx_flat:
            conf_thr += FLAT_CONF_BONUS
            edge_thr += FLAT_EDGE_BONUS
            margin_thr += FLAT_MARGIN_BONUS
            hold_edge_thr += FLAT_HOLD_EDGE_BONUS

        direction = 1 if action == 1 else -1
        counter_trend = (
            (direction == 1 and (trend < -trend_flat or sma_cross < 0))
            or (direction == -1 and (trend > trend_flat or sma_cross > 0))
        )
        if counter_trend:
            conf_thr += COUNTER_TREND_CONF_BONUS
            edge_thr += COUNTER_TREND_EDGE_BONUS
            margin_thr += COUNTER_TREND_MARGIN_BONUS
            hold_edge_thr += COUNTER_TREND_HOLD_EDGE_BONUS

        if abs_trend >= trend_strong and adx >= adx_strong and np.sign(trend) == direction:
            conf_thr = max(OPEN_PROB_THRESHOLD, conf_thr - TREND_RELAX)
            edge_thr = max(OPEN_EDGE_THRESHOLD, edge_thr - TREND_RELAX * 0.6)
            margin_thr = max(MIN_ACTION_MARGIN * 0.6, margin_thr - TREND_RELAX * 0.6)
            hold_edge_thr = max(HOLD_EDGE_THRESHOLD * 0.5, hold_edge_thr - TREND_HOLD_RELAX)

        sq = float(np.clip(semantic_quality, 0.0, 1.0))
        if sq < EMBED_QUALITY_MIN:
            deficit = (EMBED_QUALITY_MIN - sq) / max(EMBED_QUALITY_MIN, 1e-6)
            conf_thr += EMBED_QUALITY_CONF_BONUS * deficit
            edge_thr += EMBED_QUALITY_EDGE_BONUS * deficit
            margin_thr += EMBED_QUALITY_MARGIN_BONUS * deficit
            hold_edge_thr += EMBED_QUALITY_HOLD_EDGE_BONUS * deficit
            cooldown_after_trade += int(np.ceil(EMBED_QUALITY_COOLDOWN_BONUS * deficit))

        conf_thr = float(np.clip(conf_thr, OPEN_PROB_THRESHOLD, 0.99))
        edge_thr = float(np.clip(edge_thr, OPEN_EDGE_THRESHOLD, 0.50))
        margin_thr = float(np.clip(margin_thr, MIN_ACTION_MARGIN * 0.5, 0.60))
        hold_edge_thr = float(np.clip(hold_edge_thr, HOLD_EDGE_THRESHOLD * 0.5, 0.60))
        return conf_thr, edge_thr, margin_thr, hold_edge_thr, int(max(cooldown_after_trade, TRADE_COOLDOWN_BARS))

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
        self._update_defensive_mode(pnl)
        self.position = 0
        self.entry_price = 0.0
        self.hold_steps = 0
        self.unrealized_pnl = 0.0

    def process_bar(self, df, delta_tick=0, delta_price=0.0):
        if self.trade_cooldown > 0:
            self.trade_cooldown -= 1
        if self.defensive_mode_bars > 0:
            self.defensive_mode_bars -= 1

        df = calculate_features(
            df,
            semantic_runtime=self.semantic_runtime,
            semantic_feature_count=self.semantic_feature_count,
            delta_tick=delta_tick,
            delta_price=delta_price,
        )
        last_bar = df.iloc[-1]
        ts_key = last_bar["time"].strftime("%Y-%m-%d %H:%M:%S")
        semantic_quality = self.semantic_runtime.get_quality(ts_key)
        current_price = last_bar["close"]

        max_hold_closed = False

        if self.position != 0 and self.entry_price > 0 and not self.first_bar and self.hold_steps >= MAX_HOLD_STEPS:
            self._close(current_price)
            max_hold_closed = True

        if self.position != 0 and not max_hold_closed:
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
                min(self.hold_steps / MAX_HOLD_STEPS, 1.0),
                np.clip(unrealized_ret * 100, -5, 5),
            ],
            dtype=np.float32,
        )

        full_obs = np.concatenate([obs_window, state_feat])
        obs_norm = self.vec_norm.normalize_obs(full_obs)
        obs_input = np.array([obs_norm], dtype=np.float32)
        obs_tensor = self.model.policy.obs_to_tensor(obs_input)[0]
        probs = self.model.policy.get_distribution(obs_tensor).distribution.probs.detach().cpu().numpy()[0]
        action = int(np.argmax(probs))
        cooldown_after_trade = TRADE_COOLDOWN_BARS
        if ADAPTIVE_GATE and self.defensive_mode_bars > 0:
            cooldown_after_trade += DEF_COOLDOWN_BONUS

        if action in (1, 2):
            opposite_action = 2 if action == 1 else 1
            confidence = float(probs[action])
            edge = confidence - float(probs[opposite_action])
            hold_edge = confidence - float(probs[0])
            sorted_probs = np.sort(probs)
            margin = float(sorted_probs[-1] - sorted_probs[-2]) if len(sorted_probs) > 1 else 0.0
            conf_thr, edge_thr, margin_thr, hold_edge_thr, cooldown_after_trade = self._gate_thresholds(
                action,
                probs,
                last_bar,
                semantic_quality,
            )
            if (
                self.trade_cooldown > 0
                or confidence < conf_thr
                or edge < edge_thr
                or margin < margin_thr
                or hold_edge < hold_edge_thr
            ):
                self.skipped_signals += 1
                if margin < margin_thr:
                    self.margin_skips += 1
                if semantic_quality < EMBED_QUALITY_MIN:
                    self.semantic_skips += 1
                if self.defensive_mode_bars > 0:
                    self.defensive_skips += 1
                action = 0
        elif action == 3 and self.position == 0:
            action = 0

        if max_hold_closed:
            self.trade_cooldown = max(self.trade_cooldown, cooldown_after_trade)
            self.first_bar = True
            return 3, current_price

        trade_changed = False
        if action == 1:
            if self.position == -1:
                self._close(current_price)
                self._open(1, current_price)
                trade_changed = True
                self.first_bar = True
            elif self.position == 0:
                self._open(1, current_price)
                trade_changed = True
                self.first_bar = True
        elif action == 2:
            if self.position == 1:
                self._close(current_price)
                self._open(-1, current_price)
                trade_changed = True
                self.first_bar = True
            elif self.position == 0:
                self._open(-1, current_price)
                trade_changed = True
                self.first_bar = True
        elif action == 3:
            if self.position != 0:
                self._close(current_price)
                trade_changed = True
        else:
            self.first_bar = False

        if trade_changed:
            self.trade_cooldown = max(self.trade_cooldown, cooldown_after_trade)

        if action != 1 and action != 2:
            self.first_bar = False

        return action, current_price
