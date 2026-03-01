import json
import os
import sys
import time
import threading
from datetime import datetime, timedelta, timezone

import joblib
import numpy as np
import pandas as pd
from mt5linux import MetaTrader5
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from .config import (
    ALIGN_TEST_LOGIC,
    BAR_HISTORY,
    CORE_DIR,
    DEVIATION,
    ENABLE_CATCHUP_REPLAY,
    EVAL_ON_START,
    EXECUTE_STALE_REPLAY_ORDERS,
    LIVE_DYNAMIC_LOT,
    LIVE_GATE_STATS_MODE,
    LIVE_SYNC_ACCOUNT_STATE,
    LLM_DATASET_JSON,
    LLM_SEMANTIC_CACHE_FILE,
    LLM_SEMANTIC_CACHE_SCHEMA,
    LLM_TEXT_LOG_FILE,
    LLM_DIR,
    MAGIC_NUMBER,
    MAX_CATCHUP_BARS,
    MODEL_PATH,
    MODELS_DIR,
    MT5_HOST,
    MT5_PORT,
    ORDER_TICK_RETRIES,
    ORDER_TICK_RETRY_SEC,
    PIP_VALUE,
    POLL_SECONDS,
    RISK_PERCENT,
    RISK_LEVEL,
    RISK_PROFILE_MAP,
    SPREAD_PIPS,
    STATE_FILE,
    SYMBOL,
    SYNC_EXTERNAL_LOT,
    TEST_DIR,
    TRADING_SCHEDULE_DEFAULT,
    TIMEFRAME_NAME,
    TIMEFRAME_SECONDS_MAP,
    USE_LLM_SEMANTIC,
    VEC_NORM_PATH,
    BOT_WS_URL,
    BOT_CONFIG_ID,
    WINDOW_SIZE,
)
from .gate_stats import GateStatsProvider
from .numpy_compat import patch_numpy_bitgenerator_compat as _patch_numpy_bitgenerator_compat

for _path in (CORE_DIR, TEST_DIR, LLM_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from backtest_bridge import PPOBridge, calc_auto_lot
from backtest_features import build_feature_columns
from backtest_semantic import SemanticRuntime
from env_trading import TradingEnv

try:
    from use_llm import generate_llm_cls_for_bar
except Exception:
    generate_llm_cls_for_bar = None


mt5 = MetaTrader5(host=MT5_HOST, port=MT5_PORT)
class LiveTradingBot:
    def __init__(self):
        self.model = None
        self.vec_norm = None
        self.bridge = None
        self.semantic_runtime = None
        self.feature_columns = None
        self.gate_provider = GateStatsProvider(mode=LIVE_GATE_STATS_MODE)

        self.initial_balance = 0.0
        self.current_lot = 0.01
        self.point = 0.00001
        self.digits = 5
        self.pip_size = 0.0001
        self.last_bar_time = 0
        self.last_known_ticket = 0
        self.state_file = STATE_FILE
        self.state_loaded = False
        self.use_llm_semantic = bool(USE_LLM_SEMANTIC)
        self.align_test_logic = bool(ALIGN_TEST_LOGIC)
        self.sync_account_state = bool(LIVE_SYNC_ACCOUNT_STATE)
        self.dynamic_lot = bool(LIVE_DYNAMIC_LOT)
        self.llm_semantic_cache_file = LLM_SEMANTIC_CACHE_FILE
        self.llm_text_log_file = LLM_TEXT_LOG_FILE
        self.llm_semantic_cache = {}
        self._ws_cache = {}       # ts_key -> {llm_text, cls_vec} from WebSocket
        self._ws_connected = False
        self.last_action = "HOLD"
        self.risk_profile_map = dict(RISK_PROFILE_MAP or {"low": 0.5, "medium": 1.0, "high": 1.5})
        self.risk_level = str(RISK_LEVEL or "medium").lower()
        self.risk_percent = float(RISK_PERCENT)
        if self.risk_level not in self.risk_profile_map:
            self.risk_level = "medium"
        self.trading_schedule = dict(TRADING_SCHEDULE_DEFAULT)

        tf_attr = f"TIMEFRAME_{TIMEFRAME_NAME}"
        self.timeframe = getattr(mt5, tf_attr, mt5.TIMEFRAME_H1)
        self.timeframe_seconds = int(TIMEFRAME_SECONDS_MAP.get(TIMEFRAME_NAME, 3600))

    def connect(self):
        if not mt5.initialize():
            raise RuntimeError("initialize() failed")

        account_info = mt5.account_info()
        if account_info is None:
            raise RuntimeError("Failed to get account info")

        symbol_info = mt5.symbol_info(SYMBOL)
        if symbol_info is None:
            raise RuntimeError(f"Symbol {SYMBOL} not found")

        if not symbol_info.visible and not mt5.symbol_select(SYMBOL, True):
            raise RuntimeError(f"symbol_select({SYMBOL}) failed")

        self.initial_balance = float(account_info.balance)
        self.current_lot = max(
            0.01,
            calc_auto_lot(self.initial_balance, risk_pct=self._resolve_runtime_risk_percent()),
        )
        self.point = float(symbol_info.point)
        self.digits = int(symbol_info.digits)
        self.pip_size = self.point * 10 if self.digits in (3, 5) else self.point

        print(f" MT5 Connected. Account: {account_info.login}")
        print(f" Symbol={SYMBOL} | Timeframe={TIMEFRAME_NAME} | Point={self.point} | Digits={self.digits}")
        print(
            f" Balance={self.initial_balance:.2f} | AutoLot={self.current_lot} | "
            f"Risk={self.risk_level}:{self.risk_percent:.2f}%"
        )
        self._add_log(
            "success",
            f"MT5 connected | account={account_info.login} | risk={self.risk_level}:{self.risk_percent:.2f}%",
        )

    def _load_llm_semantic_cache(self):
        if not self.use_llm_semantic:
            return
        self.llm_semantic_cache = {}
        cache_file = self.llm_semantic_cache_file
        if not cache_file or not os.path.exists(cache_file):
            return
        try:
            payload = joblib.load(cache_file)
        except Exception as exc:
            print(f" LLM semantic cache skipped (invalid file): {exc}")
            return
        if not isinstance(payload, dict):
            return

        rows = None
        if payload.get("schema") == LLM_SEMANTIC_CACHE_SCHEMA and isinstance(payload.get("rows"), dict):
            rows = payload.get("rows")
        elif payload and all(isinstance(k, str) for k in payload.keys()):
            print(
                " LLM semantic cache ignored: legacy schema detected "
                "(pre-UTC fix). Cache will be rebuilt."
            )
            return
        else:
            return

        restored = {}
        for key, vec in rows.items():
            if not isinstance(key, str):
                continue
            arr = np.asarray(vec, dtype=np.float32).reshape(-1)
            if arr.size == 0:
                continue
            restored[key] = arr
        self.llm_semantic_cache = restored
        if restored:
            print(f" Loaded LLM semantic cache: {len(restored)} rows")

    def _save_llm_semantic_cache(self, reason: str = "periodic"):
        if not self.use_llm_semantic:
            return
        if not self.llm_semantic_cache:
            return
        cache_file = self.llm_semantic_cache_file
        try:
            cache_dir = os.path.dirname(cache_file)
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
            tmp_path = f"{cache_file}.tmp"
            serializable = {k: np.asarray(v, dtype=np.float32) for k, v in self.llm_semantic_cache.items()}
            payload = {
                "schema": LLM_SEMANTIC_CACHE_SCHEMA,
                "saved_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                "rows": serializable,
            }
            joblib.dump(payload, tmp_path)
            os.replace(tmp_path, cache_file)
        except Exception as exc:
            print(f" LLM semantic cache save failed ({reason}): {exc}")

    def _append_llm_text_log(self, ts_key: str, llm_text: str):
        if not self.use_llm_semantic:
            return
        if not self.llm_text_log_file:
            return
        try:
            log_dir = os.path.dirname(self.llm_text_log_file)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            record = {
                "time": ts_key,
                "symbol": SYMBOL,
                "timeframe": TIMEFRAME_NAME,
                "text": str(llm_text or "").strip(),
                "saved_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            }
            with open(self.llm_text_log_file, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as exc:
            print(f" LLM text log failed: {exc}")

    def _load_model(self):
        self.semantic_runtime = SemanticRuntime(models_dir=MODELS_DIR)
        self.feature_columns = build_feature_columns(self.semantic_runtime.semantic_feature_count)

        _patch_numpy_bitgenerator_compat()

        dummy_data = {
            "time": [pd.Timestamp.now()] * 80,
            "open": [1.0] * 80,
            "high": [1.0] * 80,
            "low": [1.0] * 80,
            "close": [1.0] * 80,
        }
        for col in self.feature_columns:
            if col not in dummy_data:
                dummy_data[col] = [0.0] * 80

        mock_df = pd.DataFrame(dummy_data)
        dummy_env = DummyVecEnv(
            [
                lambda: TradingEnv(
                    mock_df,
                    lot_size=max(
                        0.01,
                        calc_auto_lot(
                            self.initial_balance or 100.0,
                            risk_pct=self._resolve_runtime_risk_percent(),
                        ),
                    ),
                )
            ]
        )

        self.vec_norm = VecNormalize.load(VEC_NORM_PATH, dummy_env)
        self.vec_norm.training = False
        self.vec_norm.norm_reward = False

        self.model = PPO.load(MODEL_PATH)

        self.bridge = PPOBridge(
            model=self.model,
            vec_norm=self.vec_norm,
            feature_columns=self.feature_columns,
            semantic_runtime=self.semantic_runtime,
            semantic_feature_count=self.semantic_runtime.semantic_feature_count,
            gate_stats=self.gate_provider.initial(),
        )

        if self.use_llm_semantic:
            self._load_llm_semantic_cache()
            if self.llm_semantic_cache:
                self.semantic_runtime.global_time_to_vec.update(self.llm_semantic_cache)
                print(
                    " Merged LLM semantic cache into runtime embeddings: "
                    f"{len(self.llm_semantic_cache)}"
                )

        print(" Model + VecNormalize loaded")
        if self.use_llm_semantic:
            print(
                " Pipeline: live (LLM-CLS semantic + adaptive gate) | "
                f"features={len(self.feature_columns)}"
            )
            print(" LLM semantic mode enabled (best-effort): raw text -> cls embedding")
        else:
            print(
                " Pipeline: test_ppo-aligned (semantic_map fallback + adaptive gate) | "
                f"features={len(self.feature_columns)}"
            )
        print(
            " Live alignment: "
            f"align={self.align_test_logic} "
            f"gate_mode={self.gate_provider.mode} "
            f"sync_account_state={self.sync_account_state} "
            f"dynamic_lot={self.dynamic_lot} "
            f"bridge_lot={self.bridge.lot_size:.2f}"
        )

    def _start_ws_listener(self):
        """Start a background thread: unified WebSocket to BotHub.

        1. Connect to server WS
        2. Send register message (bot_config_id, symbol, timeframe)
        3. Listen for llm_result pushes
        """
        if not BOT_WS_URL or not BOT_CONFIG_ID:
            print(" WS skipped: BOT_WS_URL or BOT_CONFIG_ID not set")
            return

        def _listener():
            import websockets.sync.client as ws_sync
            while True:
                try:
                    print(f" WS connecting to {BOT_WS_URL}")
                    with ws_sync.connect(BOT_WS_URL) as ws:
                        self._ws = ws
                        self._ws_connected = True
                        # Register
                        ws.send(json.dumps({
                            "type": "register",
                            "bot_config_id": BOT_CONFIG_ID,
                            "symbol": SYMBOL.upper(),
                            "timeframe": TIMEFRAME_NAME.upper(),
                        }))
                        print(f" WS registered: {BOT_CONFIG_ID} {SYMBOL}/{TIMEFRAME_NAME}")
                        while True:
                            raw = ws.recv()
                            try:
                                msg = json.loads(raw)
                            except Exception:
                                continue
                            msg_type = msg.get("type")
                            if msg_type == "llm_result":
                                ts_key = msg.get("date_time", "")
                                cls_vec = msg.get("cls_vec")
                                llm_text = msg.get("llm_text", "")
                                if ts_key and cls_vec is not None:
                                    self._ws_cache[ts_key] = {
                                        "cls_vec": np.asarray(cls_vec, dtype=np.float32),
                                        "llm_text": llm_text,
                                    }
                                    print(f" WS received LLM result for {ts_key}")
                            elif msg_type == "bot_config":
                                self._apply_runtime_config(msg, source="ws")
                except Exception as exc:
                    self._ws = None
                    self._ws_connected = False
                    print(f" WS disconnected: {exc} | reconnecting in 5s")
                    time.sleep(5)

        t = threading.Thread(target=_listener, daemon=True, name="ws-bot-hub")
        t.start()

    def _resolve_runtime_risk_percent(self, risk_level: str | None = None) -> float:
        level = str(risk_level or self.risk_level or "medium").strip().lower()
        if level in self.risk_profile_map:
            pct = float(self.risk_profile_map[level])
            if pct > 0:
                return pct
        if self.risk_percent > 0:
            return float(self.risk_percent)
        return 1.0

    def _refresh_lot_from_account(self):
        if not self.dynamic_lot:
            return
        account = mt5.account_info()
        if account is None:
            return
        next_lot = max(
            0.01,
            calc_auto_lot(float(account.balance), risk_pct=self._resolve_runtime_risk_percent()),
        )
        self.current_lot = float(next_lot)
        if self.bridge is not None:
            self.bridge.lot_size = float(next_lot)
            self.bridge.spread_cost = SPREAD_PIPS * PIP_VALUE * self.bridge.lot_size

    def _normalize_schedule(self, schedule) -> dict[str, bool]:
        defaults = {
            "mon": True,
            "tue": True,
            "wed": True,
            "thu": True,
            "fri": True,
            "sat": False,
            "sun": False,
        }
        alias_to_key = {
            "mon": "mon",
            "monday": "mon",
            "tue": "tue",
            "tues": "tue",
            "tuesday": "tue",
            "wed": "wed",
            "weds": "wed",
            "wednesday": "wed",
            "thu": "thu",
            "thur": "thu",
            "thurs": "thu",
            "thursday": "thu",
            "fri": "fri",
            "friday": "fri",
            "sat": "sat",
            "saturday": "sat",
            "sun": "sun",
            "sunday": "sun",
        }
        if isinstance(schedule, str):
            try:
                parsed = json.loads(schedule)
            except Exception:
                parsed = None
            schedule = parsed if isinstance(parsed, dict) else None
        if not isinstance(schedule, dict):
            return defaults
        for raw_key, raw_value in schedule.items():
            key = alias_to_key.get(str(raw_key).strip().lower())
            if key:
                defaults[key] = bool(raw_value)
        return defaults

    def _is_trading_day_enabled(self, bar_end_ts: int) -> bool:
        dt = datetime.fromtimestamp(int(bar_end_ts), tz=timezone.utc)
        day_key = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")[dt.weekday()]
        return bool(self.trading_schedule.get(day_key, True))

    def _apply_runtime_config(self, payload: dict, source: str = "ws"):
        if not isinstance(payload, dict):
            return

        changed = False

        incoming_map = payload.get("risk_profile_map")
        if isinstance(incoming_map, dict):
            merged = dict(self.risk_profile_map)
            for key, value in incoming_map.items():
                lvl = str(key).strip().lower()
                if lvl not in {"low", "medium", "high"}:
                    continue
                try:
                    pct = float(value)
                except Exception:
                    continue
                if pct > 0:
                    merged[lvl] = pct
                    changed = True
            self.risk_profile_map = merged

        incoming_level = payload.get("risk_level")
        if incoming_level is not None:
            lvl = str(incoming_level).strip().lower()
            if lvl in {"low", "medium", "high"} and lvl != self.risk_level:
                self.risk_level = lvl
                changed = True

        incoming_percent = payload.get("risk_percent")
        if incoming_percent is not None:
            try:
                pct = float(incoming_percent)
                if pct > 0 and pct != self.risk_percent:
                    self.risk_percent = pct
                    changed = True
            except Exception:
                pass

        incoming_schedule = payload.get("trading_schedule")
        if incoming_schedule is not None:
            next_schedule = self._normalize_schedule(incoming_schedule)
            if next_schedule != self.trading_schedule:
                self.trading_schedule = next_schedule
                changed = True

        if changed:
            effective_risk = self._resolve_runtime_risk_percent()
            self.risk_percent = float(effective_risk)
            self._refresh_lot_from_account()
            self._add_log(
                "info",
                f"Runtime config updated ({source}) | risk={self.risk_level}:{self.risk_percent:.2f}%",
            )
            print(
                " Runtime config applied | "
                f"source={source} risk_level={self.risk_level} risk_percent={self.risk_percent:.2f}"
            )

    def _push_state_to_server(self, action_name: str = ""):
        """Push full MT5 state to server via WebSocket."""
        if not self._ws_connected or not hasattr(self, '_ws') or self._ws is None:
            return
        try:
            resolved_action = str(action_name or self.last_action or "").strip() or "HOLD"
            self.last_action = resolved_action
            current_pos, pos = self._get_mt5_position()

            # ── MT5 Account info ──
            account = mt5.account_info()
            account_data = {}
            if account is not None:
                account_data = {
                    "balance": float(account.balance),
                    "equity": float(account.equity),
                    "margin": float(account.margin),
                    "free_margin": float(account.margin_free),
                    "margin_level": float(account.margin_level) if account.margin_level else 0.0,
                    "leverage": int(account.leverage),
                    "profit": float(account.profit),
                    "currency": str(account.currency),
                    "server": str(account.server),
                    "login": int(account.login),
                }

            # ── MT5 Active positions ──
            all_positions = mt5.positions_get()
            positions_data = []
            if all_positions:
                for p in all_positions:
                    opened_at_utc = datetime.fromtimestamp(p.time, tz=timezone.utc)
                    positions_data.append({
                        "ticket": int(p.ticket),
                        "symbol": str(p.symbol),
                        "type": "BUY" if p.type == mt5.ORDER_TYPE_BUY else "SELL",
                        "volume": float(p.volume),
                        "price_open": float(p.price_open),
                        "price_current": float(p.price_current),
                        "profit": float(p.profit),
                        "swap": float(p.swap),
                        "sl": float(p.sl),
                        "tp": float(p.tp),
                        "opened_at": opened_at_utc.strftime("%Y-%m-%d %H:%M:%S"),
                        "opened_at_ts": int(p.time),
                        "time": datetime.fromtimestamp(
                            p.time, tz=timezone.utc
                        ).strftime("%H:%M:%S"),
                        "comment": str(p.comment) if p.comment else "",
                    })

            state = {
                "type": "state",
                "bot_config_id": BOT_CONFIG_ID,
                "symbol": SYMBOL,
                "timeframe": TIMEFRAME_NAME,
                # Bot model state
                "position": int(current_pos),
                "entry_price": float(self.bridge.entry_price) if self.bridge else 0.0,
                "total_pnl": float(self.bridge.total_pnl) if self.bridge else 0.0,
                "trades": int(self.bridge.trades) if self.bridge else 0,
                "wins": int(self.bridge.wins) if self.bridge else 0,
                "loss_streak": int(self.bridge.loss_streak) if self.bridge else 0,
                "last_action": resolved_action,
                "last_bar_time": datetime.fromtimestamp(
                    max(1, self.last_bar_time), tz=timezone.utc
                ).strftime("%Y-%m-%d %H:%M:%S") if self.last_bar_time else "",
                "lot_size": float(self.current_lot),
                "unrealized_pnl": float(pos.profit) if pos is not None else 0.0,
                "risk_level": self.risk_level,
                "risk_percent": float(self.risk_percent),
                "risk_profile_map": self.risk_profile_map,
                "trading_schedule": self.trading_schedule,
                # MT5 Account
                **account_data,
                # MT5 Positions
                "positions": positions_data,
                # Logs
                "llm_text": getattr(self, "last_llm_text", ""),
                "recent_logs": getattr(self, "recent_logs", []),
                "ws_connected": True,
            }
            self._ws.send(json.dumps(state, ensure_ascii=False))
        except Exception as exc:
            print(f" WS state push failed: {exc}")

    def _add_log(self, log_type: str, message: str):
        """Add a log entry to be sent to the dashboard Activity Log."""
        if not hasattr(self, "recent_logs"):
            self.recent_logs = []
        now_str = datetime.now().strftime("%H:%M:%S")
        self.recent_logs.append({
            "timestamp": now_str,
            "type": log_type,
            "message": message
        })
        # Keep only the last 50 logs
        if len(self.recent_logs) > 50:
            self.recent_logs.pop(0)

    def _resolve_live_llm_semantic(self, ts_key: str):
        if not self.use_llm_semantic:
            return
        if self.semantic_runtime is None:
            return
        if ts_key in self.semantic_runtime.global_time_to_vec:
            return

        # 1. Check local disk cache
        cached_vec = self.llm_semantic_cache.get(ts_key)
        if cached_vec is not None:
            self.semantic_runtime.global_time_to_vec[ts_key] = np.asarray(cached_vec, dtype=np.float32)
            return

        # 2. Check WebSocket push cache
        ws_data = self._ws_cache.pop(ts_key, None)
        if ws_data is not None:
            cls_vec = np.asarray(ws_data["cls_vec"], dtype=np.float32).reshape(-1)
            expected_dim = int(self.semantic_runtime._embedding_dim())
            if cls_vec.size == expected_dim:
                self.llm_semantic_cache[ts_key] = cls_vec
                self.semantic_runtime.global_time_to_vec[ts_key] = cls_vec
                self.last_llm_text = ws_data.get("llm_text", "")
                self._add_log("analysis", f"Received AI Analysis for {ts_key[-8:]}")
                self._append_llm_text_log(ts_key, self.last_llm_text)
                self._save_llm_semantic_cache(reason="ws_push")
                print(f" LLM semantic ready (WS push) | ts={ts_key} | dim={cls_vec.size}")
                return
            else:
                print(f" WS CLS dim mismatch expected={expected_dim} got={cls_vec.size}")

        # 3. Fallback: compute directly (if module available)
        if generate_llm_cls_for_bar is None:
            print(" LLM semantic skipped: use_llm module unavailable & no WS data")
            return

        dataset_json = LLM_DATASET_JSON if LLM_DATASET_JSON else None
        bar_dt = datetime.strptime(ts_key, "%Y-%m-%d %H:%M:%S")
        print(f" LLM semantic: building cls for {ts_key} (direct fallback)")
        try:
            llm_text, llm_cls = generate_llm_cls_for_bar(
                date_time=bar_dt,
                symbol=SYMBOL,
                dataset_json=dataset_json,
            )
            cls_vec = np.asarray(llm_cls, dtype=np.float32).reshape(-1)
            expected_dim = int(self.semantic_runtime._embedding_dim())
            if cls_vec.size != expected_dim:
                raise RuntimeError(f"CLS dim mismatch expected={expected_dim} got={cls_vec.size}")

            self.llm_semantic_cache[ts_key] = cls_vec
            self.semantic_runtime.global_time_to_vec[ts_key] = cls_vec
            self.last_llm_text = llm_text
            self._add_log("analysis", f"Generated AI Analysis for {ts_key[-8:]}")
            self._append_llm_text_log(ts_key, llm_text)
            self._save_llm_semantic_cache(reason="llm_update")
            print(f" LLM semantic ready | ts={ts_key} | dim={cls_vec.size}")
        except Exception as exc:
            print(f" LLM semantic failed for {ts_key}: {exc} | fallback=semantic_map")

    def _ensure_window_real_semantic(self, window_df: pd.DataFrame):
        if not self.use_llm_semantic:
            return
        if self.semantic_runtime is None:
            return
        tail_df = window_df.tail(int(max(1, WINDOW_SIZE)))
        ts_keys = pd.to_datetime(tail_df["time"]).dt.strftime("%Y-%m-%d %H:%M:%S").tolist()
        missing = [ts for ts in ts_keys if ts not in self.semantic_runtime.global_time_to_vec]
        if missing:
            print(f" LLM semantic: resolving missing window embeddings={len(missing)}")
        for ts_key in missing:
            self._resolve_live_llm_semantic(ts_key)

    def _runtime_state_payload(self):
        if self.bridge is None:
            return {}
        current_pos, pos = self._get_mt5_position()
        current_ticket = int(pos.ticket) if pos is not None else 0
        return {
            "version": 1,
            "saved_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            "symbol": SYMBOL,
            "timeframe": TIMEFRAME_NAME,
            "runtime_config": {
                "risk_level": self.risk_level,
                "risk_percent": float(self.risk_percent),
                "risk_profile_map": dict(self.risk_profile_map),
                "trading_schedule": dict(self.trading_schedule),
            },
            "last_bar_time": int(self.last_bar_time),
            "last_known_ticket": int(current_ticket or self.last_known_ticket),
            "bridge": {
                "position": int(self.bridge.position),
                "entry_price": float(self.bridge.entry_price),
                "hold_steps": int(self.bridge.hold_steps),
                "first_bar": bool(self.bridge.first_bar),
                "trade_cooldown": int(self.bridge.trade_cooldown),
                "defensive_mode_bars": int(self.bridge.defensive_mode_bars),
                "defensive_triggers": int(self.bridge.defensive_triggers),
                "loss_streak": int(self.bridge.loss_streak),
                "skipped_signals": int(self.bridge.skipped_signals),
                "margin_skips": int(self.bridge.margin_skips),
                "defensive_skips": int(self.bridge.defensive_skips),
                "semantic_skips": int(self.bridge.semantic_skips),
                "trades": int(self.bridge.trades),
                "wins": int(self.bridge.wins),
                "total_pnl": float(self.bridge.total_pnl),
                "total_fees": float(self.bridge.total_fees),
                "max_equity": float(self.bridge.max_equity),
                "recent_trade_pips": [float(x) for x in list(self.bridge.recent_trade_pips)],
                "gate_stats": {k: float(v) for k, v in dict(self.bridge.gate_stats or {}).items()},
            },
            "gate_history": self.gate_provider.to_records(max_rows=max(BAR_HISTORY * 2, 400)),
        }

    def _save_runtime_state(self, reason: str = "periodic"):
        if self.bridge is None:
            return
        payload = self._runtime_state_payload()
        if not payload:
            return
        try:
            state_dir = os.path.dirname(self.state_file)
            if state_dir:
                os.makedirs(state_dir, exist_ok=True)
            tmp_path = f"{self.state_file}.tmp"
            with open(tmp_path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=True)
            os.replace(tmp_path, self.state_file)
        except Exception as exc:
            print(f" State save failed ({reason}): {exc}")

    def _load_runtime_state(self):
        if self.bridge is None:
            return
        if not os.path.exists(self.state_file):
            return

        try:
            with open(self.state_file, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except Exception as exc:
            print(f" State load skipped (invalid file): {exc}")
            return

        if str(payload.get("symbol", "")).upper() != SYMBOL.upper():
            print(" State file symbol mismatch; ignored.")
            return
        if str(payload.get("timeframe", "")).upper() != TIMEFRAME_NAME.upper():
            print(" State file timeframe mismatch; ignored.")
            return

        bridge_state = payload.get("bridge", {})
        if isinstance(bridge_state, dict):
            self.bridge.position = int(bridge_state.get("position", self.bridge.position))
            self.bridge.entry_price = float(bridge_state.get("entry_price", self.bridge.entry_price))
            self.bridge.hold_steps = int(max(0, bridge_state.get("hold_steps", self.bridge.hold_steps)))
            self.bridge.first_bar = bool(bridge_state.get("first_bar", self.bridge.first_bar))
            self.bridge.trade_cooldown = int(max(0, bridge_state.get("trade_cooldown", self.bridge.trade_cooldown)))
            self.bridge.defensive_mode_bars = int(max(0, bridge_state.get("defensive_mode_bars", self.bridge.defensive_mode_bars)))
            self.bridge.defensive_triggers = int(max(0, bridge_state.get("defensive_triggers", self.bridge.defensive_triggers)))
            self.bridge.loss_streak = int(max(0, bridge_state.get("loss_streak", self.bridge.loss_streak)))
            self.bridge.skipped_signals = int(max(0, bridge_state.get("skipped_signals", self.bridge.skipped_signals)))
            self.bridge.margin_skips = int(max(0, bridge_state.get("margin_skips", self.bridge.margin_skips)))
            self.bridge.defensive_skips = int(max(0, bridge_state.get("defensive_skips", self.bridge.defensive_skips)))
            self.bridge.semantic_skips = int(max(0, bridge_state.get("semantic_skips", self.bridge.semantic_skips)))
            self.bridge.trades = int(max(0, bridge_state.get("trades", self.bridge.trades)))
            self.bridge.wins = int(max(0, bridge_state.get("wins", self.bridge.wins)))
            self.bridge.total_pnl = float(bridge_state.get("total_pnl", self.bridge.total_pnl))
            self.bridge.total_fees = float(bridge_state.get("total_fees", self.bridge.total_fees))
            self.bridge.max_equity = float(bridge_state.get("max_equity", self.bridge.max_equity))
            self.bridge.gate_stats = dict(bridge_state.get("gate_stats", self.bridge.gate_stats or {}))

            recent = bridge_state.get("recent_trade_pips", [])
            try:
                self.bridge.recent_trade_pips.clear()
                maxlen = int(self.bridge.recent_trade_pips.maxlen or 20)
                for val in list(recent)[-maxlen:]:
                    self.bridge.recent_trade_pips.append(float(val))
            except Exception:
                pass

        self.last_bar_time = int(max(0, payload.get("last_bar_time", self.last_bar_time)))
        self.last_known_ticket = int(max(0, payload.get("last_known_ticket", self.last_known_ticket)))
        self.gate_provider.load_records(payload.get("gate_history", []))

        runtime_cfg = payload.get("runtime_config", {})
        if isinstance(runtime_cfg, dict):
            self._apply_runtime_config(runtime_cfg, source="state")

        self.state_loaded = True
        print(
            " Runtime state restored | "
            f"cooldown={self.bridge.trade_cooldown} "
            f"def_mode={self.bridge.defensive_mode_bars} "
            f"loss_streak={self.bridge.loss_streak} "
            f"hold_steps={self.bridge.hold_steps}"
        )

    def _get_filling_mode(self):
        symbol_info = mt5.symbol_info(SYMBOL)
        if symbol_info is None:
            return mt5.ORDER_FILLING_FOK

        filling_mode = int(symbol_info.filling_mode)
        if filling_mode & 1:
            return mt5.ORDER_FILLING_FOK
        if filling_mode & 2:
            return mt5.ORDER_FILLING_IOC
        return mt5.ORDER_FILLING_RETURNAL

    def _get_mt5_position(self):
        positions = mt5.positions_get(symbol=SYMBOL)
        if positions is None or len(positions) == 0:
            return 0, None
        pos = positions[0]
        if pos.type == mt5.ORDER_TYPE_BUY:
            return 1, pos
        if pos.type == mt5.ORDER_TYPE_SELL:
            return -1, pos
        return 0, pos

    def _sync_bridge_from_mt5(self):
        if self.bridge is None:
            return

        prev_pos = int(self.bridge.position)
        current_pos, pos = self._get_mt5_position()
        current_ticket = int(pos.ticket) if pos is not None else 0

        self.bridge.position = current_pos
        if current_pos != 0 and pos is not None:
            position_changed = (
                prev_pos != current_pos
                or (self.last_known_ticket != 0 and self.last_known_ticket != current_ticket)
            )
            if position_changed or float(self.bridge.entry_price) <= 0.0:
                self.bridge.entry_price = float(pos.price_open)
                self.bridge.hold_steps = 0
                self.bridge.first_bar = True
        else:
            self.bridge.entry_price = 0.0
            self.bridge.hold_steps = 0
            self.bridge.unrealized_pnl = 0.0
            self.bridge.first_bar = False

        account = mt5.account_info()
        if self.sync_account_state and account is not None:
            self.bridge.balance = float(account.balance)
            self.bridge.equity = float(account.equity)
            self.bridge.total_pnl = float(account.balance - self.initial_balance)
            self.bridge.max_equity = max(float(self.bridge.max_equity), float(self.bridge.equity))
        else:
            self.bridge.max_equity = max(float(self.bridge.max_equity), float(self.bridge.equity))

        if SYNC_EXTERNAL_LOT and pos is not None and float(pos.volume) > 0:
            self.bridge.lot_size = float(pos.volume)
        elif self.dynamic_lot:
            if account is not None:
                self.bridge.lot_size = max(
                    0.01,
                    calc_auto_lot(
                        float(account.balance),
                        risk_pct=self._resolve_runtime_risk_percent(),
                    ),
                )
            else:
                self.bridge.lot_size = max(0.01, float(self.bridge.lot_size))
        else:
            self.bridge.lot_size = max(0.01, float(self.bridge.lot_size))
        self.bridge.spread_cost = SPREAD_PIPS * PIP_VALUE * self.bridge.lot_size

        self.current_lot = self.bridge.lot_size
        self.last_known_ticket = current_ticket

    def _fetch_window(self, bar_end_ts: int):
        rates = None
        anchor_dt = datetime.fromtimestamp(max(1, int(bar_end_ts) - 1), tz=timezone.utc)

        if hasattr(mt5, "copy_rates_from"):
            try:
                rates = mt5.copy_rates_from(SYMBOL, self.timeframe, anchor_dt, BAR_HISTORY)
            except Exception:
                rates = None

        if rates is None and hasattr(mt5, "copy_rates_range"):
            try:
                lookback_secs = max(self.timeframe_seconds * BAR_HISTORY * 3, 86400)
                start_dt = anchor_dt - timedelta(seconds=lookback_secs)
                rates = mt5.copy_rates_range(SYMBOL, self.timeframe, start_dt, anchor_dt)
            except Exception:
                rates = None

        if rates is None:
            rates = mt5.copy_rates_from_pos(SYMBOL, self.timeframe, 1, BAR_HISTORY)

        if rates is None or len(rates) < BAR_HISTORY:
            return None

        df = pd.DataFrame(rates)
        if len(df) < BAR_HISTORY:
            return None

        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True).dt.tz_convert(None)
        df = df[["time", "open", "high", "low", "close"]].sort_values("time").reset_index(drop=True)
        if len(df) > BAR_HISTORY:
            df = df.tail(BAR_HISTORY).reset_index(drop=True)
        return df

    def _calc_delta_for_closed_bar(self, current_bar_ts: int):
        end_dt = datetime.fromtimestamp(current_bar_ts, tz=timezone.utc)
        start_dt = end_dt - timedelta(seconds=self.timeframe_seconds)
        ticks = mt5.copy_ticks_range(SYMBOL, start_dt, end_dt, mt5.COPY_TICKS_ALL)

        if ticks is None or len(ticks) <= 1:
            return 0, 0.0

        tdf = pd.DataFrame(ticks)
        tdf["prev_bid"] = tdf["bid"].shift(1)
        tdf["prev_ask"] = tdf["ask"].shift(1)

        buy = (tdf["bid"] > tdf["prev_bid"]) | (
            (tdf["bid"] == tdf["prev_bid"]) & (tdf["ask"] > tdf["prev_ask"])
        )
        sell = (tdf["bid"] < tdf["prev_bid"]) | (
            (tdf["bid"] == tdf["prev_bid"]) & (tdf["ask"] < tdf["prev_ask"])
        )

        delta_tick = int(buy.sum() - sell.sum())
        delta_price = float((tdf["bid"].iloc[-1] - tdf["bid"].iloc[0]) + (tdf["ask"].iloc[-1] - tdf["ask"].iloc[0]))
        return delta_tick, delta_price

    def _current_bar_time(self) -> int:
        latest = mt5.copy_rates_from_pos(SYMBOL, self.timeframe, 0, 1)
        if latest is None or len(latest) == 0:
            return 0
        return int(latest[0]["time"])

    def _discover_missed_bar_ends(self, prev_bar_time: int, current_bar_time: int):
        if current_bar_time <= prev_bar_time:
            return []

        bar_times = []
        if hasattr(mt5, "copy_rates_range"):
            try:
                start_dt = datetime.fromtimestamp(prev_bar_time, tz=timezone.utc)
                end_dt = datetime.fromtimestamp(current_bar_time, tz=timezone.utc)
                rates = mt5.copy_rates_range(SYMBOL, self.timeframe, start_dt, end_dt)
                if rates is not None and len(rates) > 0:
                    rdf = pd.DataFrame(rates)
                    if "time" in rdf.columns:
                        bar_times = sorted(
                            int(ts)
                            for ts in rdf["time"].tolist()
                            if int(ts) > prev_bar_time and int(ts) <= current_bar_time
                        )
            except Exception:
                bar_times = []

        if bar_times:
            return bar_times

        if self.timeframe_seconds <= 0:
            return [current_bar_time]

        ts = int(prev_bar_time + self.timeframe_seconds)
        fallback = []
        while ts <= current_bar_time:
            fallback.append(ts)
            ts += self.timeframe_seconds
        return fallback

    def _replay_missed_bars_if_any(self, current_bar_time: int):
        if not ENABLE_CATCHUP_REPLAY:
            return
        if self.last_bar_time <= 0 or current_bar_time <= self.last_bar_time:
            return

        missed_bar_ends = self._discover_missed_bar_ends(self.last_bar_time, current_bar_time)
        if not missed_bar_ends:
            return

        skipped = 0
        if MAX_CATCHUP_BARS > 0 and len(missed_bar_ends) > MAX_CATCHUP_BARS:
            skipped = len(missed_bar_ends) - MAX_CATCHUP_BARS
            missed_bar_ends = missed_bar_ends[-MAX_CATCHUP_BARS:]

        print(
            " Catch-up replay | "
            f"bars={len(missed_bar_ends)} skipped={skipped} "
            f"execute_stale={int(EXECUTE_STALE_REPLAY_ORDERS)}"
        )

        total = len(missed_bar_ends)
        for idx, bar_end_ts in enumerate(missed_bar_ends, start=1):
            is_latest = idx == total
            execute_orders = is_latest or EXECUTE_STALE_REPLAY_ORDERS
            mode = f"Catch-up {idx}/{total}"
            self.last_bar_time = int(bar_end_ts)
            self._process_closed_bar(
                bar_end_ts=int(bar_end_ts),
                mode=mode,
                execute_orders=execute_orders,
            )

    def close_all(self):
        positions = mt5.positions_get(symbol=SYMBOL)
        if positions is None:
            self._add_log("warning", "Close skipped: positions_get returned None")
            return False
        if len(positions) == 0:
            self._add_log("info", "Close requested: no open positions")
            return True

        all_ok = True
        for pos in positions:
            tick = self._get_trade_tick()
            if tick is None:
                print(" Close Skipped: no live tick")
                self._add_log("warning", f"Close skipped: no live tick for ticket {pos.ticket}")
                all_ok = False
                continue

            price = tick.bid if pos.type == mt5.ORDER_TYPE_BUY else tick.ask
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
                "type_filling": self._get_filling_mode(),
            }
            res = mt5.order_send(req)
            if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                print(f" Closed Position | ticket={pos.ticket} | PnL={pos.profit:+.2f}")
                self._add_log("action", f"Closed position #{pos.ticket} | PnL {pos.profit:+.2f}")
            else:
                comment = res.comment if res else "No response"
                print(f" Close Failed | ticket={pos.ticket} | {comment}")
                self._add_log("warning", f"Close failed #{pos.ticket}: {comment}")
                all_ok = False
        return all_ok

    def _get_trade_tick(self):
        retries = max(1, int(ORDER_TICK_RETRIES))
        wait_sec = max(0.0, float(ORDER_TICK_RETRY_SEC))

        for _ in range(retries):
            if not mt5.symbol_select(SYMBOL, True):
                time.sleep(wait_sec)
                continue

            tick = mt5.symbol_info_tick(SYMBOL)
            if tick is not None and float(getattr(tick, "bid", 0.0)) > 0.0 and float(getattr(tick, "ask", 0.0)) > 0.0:
                return tick
            time.sleep(wait_sec)
        return None

    def send_order(self, order_type):
        tick = self._get_trade_tick()
        if tick is None:
            print(" Order Skipped: no live tick (market closed or quote unavailable)")
            return False

        price = tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid

        if self.dynamic_lot:
            account = mt5.account_info()
            if account is not None:
                self.current_lot = max(
                    0.01,
                    calc_auto_lot(
                        float(account.balance),
                        risk_pct=self._resolve_runtime_risk_percent(),
                    ),
                )
        elif self.bridge is not None:
            self.current_lot = max(0.01, float(self.bridge.lot_size))

        req = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": SYMBOL,
            "volume": self.current_lot,
            "type": order_type,
            "price": price,
            "deviation": DEVIATION,
            "magic": MAGIC_NUMBER,
            "comment": "AI Trade",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self._get_filling_mode(),
        }

        res = mt5.order_send(req)
        if res and res.retcode == mt5.TRADE_RETCODE_DONE:
            side = "BUY" if order_type == mt5.ORDER_TYPE_BUY else "SELL"
            if self.bridge is not None:
                self.bridge.lot_size = float(self.current_lot)
                self.bridge.spread_cost = SPREAD_PIPS * PIP_VALUE * self.bridge.lot_size
            print(
                f" Opened {side} @ {price:.{self.digits}f} | "
                f"Lot={self.current_lot}"
            )
            self._add_log("action", f"Opened {side} @ {price:.{self.digits}f} | lot={self.current_lot:.2f}")
            return True
        else:
            comment = res.comment if res else "No response"
            print(f" Order Failed: {comment}")
            self._add_log("warning", f"Order failed: {comment}")
            return False

    def execute_action(self, action):
        current_pos, _ = self._get_mt5_position()

        if action == 0:
            return True

        if action == 3:
            if current_pos != 0:
                return self.close_all()
            return True

        if action == 1:
            if current_pos == -1:
                closed_ok = self.close_all()
                time.sleep(0.2)
                current_pos, _ = self._get_mt5_position()
                if not closed_ok and current_pos == -1:
                    return False
            if current_pos <= 0:
                return self.send_order(mt5.ORDER_TYPE_BUY)
            return True

        if action == 2:
            if current_pos == 1:
                closed_ok = self.close_all()
                time.sleep(0.2)
                current_pos, _ = self._get_mt5_position()
                if not closed_ok and current_pos == 1:
                    return False
            if current_pos >= 0:
                return self.send_order(mt5.ORDER_TYPE_SELL)
            return True
        return False

    def _reconcile_broker_execution(
        self,
        action: int,
        execute_orders: bool,
        order_ok: bool,
        broker_pos_before: int,
        broker_pos_after: int,
    ):
        if self.bridge is None or not execute_orders or action not in (1, 2, 3):
            return

        expected_after = broker_pos_before
        if action == 1:
            expected_after = 1
        elif action == 2:
            expected_after = -1
        elif action == 3:
            expected_after = 0

        mismatch = broker_pos_after != expected_after
        if order_ok and not mismatch:
            return

        print(
            " Broker reconcile: action not reflected on broker "
            f"(action={action} expected_pos={expected_after} actual_pos={broker_pos_after})"
        )
        self._add_log(
            "warning",
            f"Broker reconcile mismatch | action={action} expected={expected_after} actual={broker_pos_after}",
        )

        # Prevent virtual bridge cooldown from blocking entries when order failed.
        self.bridge.trade_cooldown = 0
        self.bridge.first_bar = False

        if broker_pos_after == 0:
            self.bridge.position = 0
            self.bridge.entry_price = 0.0
            self.bridge.hold_steps = 0
            self.bridge.unrealized_pnl = 0.0

        # Roll back synthetic spread fee if an open from flat did not fill.
        if action in (1, 2) and broker_pos_before == 0 and broker_pos_after == 0:
            self.bridge.total_fees = max(0.0, float(self.bridge.total_fees) - float(self.bridge.spread_cost))

    def _print_status_line(self):
        tick = mt5.symbol_info_tick(SYMBOL)
        if tick is None:
            return

        current_pos, pos = self._get_mt5_position()
        pos_txt = {1: "LONG", -1: "SHORT", 0: "FLAT"}[current_pos]
        pnl_txt = f"{pos.profit:+.2f}" if pos is not None else "0.00"
        server_time_utc = datetime.fromtimestamp(tick.time, tz=timezone.utc).strftime("%H:%M:%SZ")

        line = (
            f"\r Pr:{tick.bid:.{self.digits}f} | Pos:{pos_txt:5s} | PnL:{pnl_txt:>8s} | "
            f"Eq:{self.bridge.equity:8.2f} | T_UTC:{server_time_utc}      "
        )
        sys.stdout.write(line)
        sys.stdout.flush()

    def _process_closed_bar(self, bar_end_ts: int, mode: str = "New Candle", execute_orders: bool = True):
        bar_end_utc = datetime.fromtimestamp(bar_end_ts, tz=timezone.utc)
        bar_open_utc = bar_end_utc - timedelta(seconds=self.timeframe_seconds)
        print(
            f"\n {mode}: processing closed bar -> "
            f"open={bar_open_utc.strftime('%Y-%m-%d %H:%M:%SZ')} "
            f"end={bar_end_utc.strftime('%Y-%m-%d %H:%M:%SZ')}"
        )

        window_df = self._fetch_window(bar_end_ts)
        if window_df is None:
            print(" Not enough bars yet for model window")
            return

        self._ensure_window_real_semantic(window_df)

        delta_tick, delta_price = self._calc_delta_for_closed_bar(bar_end_ts)
        self.bridge.gate_stats = self.gate_provider.update(window_df)

        self._sync_bridge_from_mt5()
        action, model_price = self.bridge.process_bar(window_df, delta_tick, delta_price)
        action = int(action)

        action_name = {0: "HOLD", 1: "BUY", 2: "SELL", 3: "CLOSE"}.get(action, "?")
        self.last_action = action_name
        print(
            f" Model Action: {action_name} | Price={model_price:.5f} | "
            f"dTick={delta_tick} | dPrice={delta_price:.5f}"
        )
        self._add_log(
            "analysis",
            f"{mode} | action={action_name} | price={model_price:.5f} | dTick={delta_tick}",
        )

        if execute_orders and not self._is_trading_day_enabled(bar_end_ts):
            if action in (1, 2):
                self._add_log(
                    "warning",
                    f"Schedule blocked new order ({action_name}) on disabled trading day",
                )
                action = 0
                action_name = "HOLD"
        self.last_action = action_name

        broker_pos_before, _ = self._get_mt5_position()
        order_ok = True
        if execute_orders:
            order_ok = bool(self.execute_action(action))
        elif action != 0:
            print(" No-order mode: action skipped")
            self._add_log("info", f"{mode}: action {action_name} skipped in no-order mode")
        self._sync_bridge_from_mt5()
        broker_pos_after, _ = self._get_mt5_position()
        self._reconcile_broker_execution(
            action=action,
            execute_orders=execute_orders,
            order_ok=order_ok,
            broker_pos_before=int(broker_pos_before),
            broker_pos_after=int(broker_pos_after),
        )
        self._save_runtime_state(reason="bar_close")
        self._push_state_to_server(action_name=action_name)

    def run(self):
        self.connect()
        self._load_model()
        self._load_runtime_state()
        self._sync_bridge_from_mt5()
        self._start_ws_listener()
        startup_eval_pending = bool(EVAL_ON_START)

        print(f" Waiting for new {TIMEFRAME_NAME} candles...")

        try:
            while True:
                current_bar_time = self._current_bar_time()
                if current_bar_time <= 0:
                    time.sleep(POLL_SECONDS)
                    continue

                if startup_eval_pending:
                    startup_eval_pending = False
                    if self.last_bar_time == 0:
                        self.last_bar_time = current_bar_time
                        self._process_closed_bar(current_bar_time, mode="Startup", execute_orders=True)
                        self._sync_bridge_from_mt5()
                        self._push_state_to_server()
                        self._print_status_line()
                        time.sleep(POLL_SECONDS)
                        continue

                    if self.last_bar_time == current_bar_time:
                        print("\n Startup eval: current bar already processed; running no-order refresh")
                        self._process_closed_bar(current_bar_time, mode="Startup", execute_orders=False)
                        self._sync_bridge_from_mt5()
                        self._push_state_to_server()
                        self._print_status_line()
                        time.sleep(POLL_SECONDS)
                        continue

                    print("\n Startup eval deferred: missed bars detected; catch-up replay will process first")

                if self.last_bar_time == 0:
                    self.last_bar_time = current_bar_time
                    self._sync_bridge_from_mt5()
                    self._push_state_to_server()
                    self._print_status_line()
                    time.sleep(POLL_SECONDS)
                    continue

                if current_bar_time != self.last_bar_time:
                    self._replay_missed_bars_if_any(current_bar_time)
                    if self.last_bar_time != current_bar_time:
                        self.last_bar_time = current_bar_time
                        self._process_closed_bar(current_bar_time, mode="New Candle", execute_orders=True)

                self._sync_bridge_from_mt5()
                self._push_state_to_server()
                self._print_status_line()
                time.sleep(POLL_SECONDS)

        except KeyboardInterrupt:
            print("\n Stopped by user")
        finally:
            self._save_runtime_state(reason="shutdown")
            self._save_llm_semantic_cache(reason="shutdown")
            try:
                mt5.shutdown()
            except Exception:
                pass

def main():
    bot = LiveTradingBot()
    bot.run()


if __name__ == "__main__":
    main()
