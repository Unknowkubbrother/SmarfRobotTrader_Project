import json
import os
import shutil
import sys
import time
import threading
from types import SimpleNamespace
from datetime import datetime, timedelta, timezone
from urllib import error as urlerror
from urllib import request as urlrequest

import joblib
import numpy as np
import pandas as pd
from mt5linux import MetaTrader5
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from .config import (
    BAR_HISTORY,
    CORE_DIR,
    DEVIATION,
    ENABLE_CATCHUP_REPLAY,
    EVAL_ON_START,
    EXECUTE_STALE_REPLAY_ORDERS,
    LIVE_DYNAMIC_LOT,
    LIVE_SYNC_ACCOUNT_STATE,
    LLM_SEMANTIC_CACHE_FILE,
    LLM_SEMANTIC_CACHE_SCHEMA,
    LLM_TEXT_LOG_FILE,
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
    TRADING_SCHEDULE_DEFAULT,
    TIMEFRAME_NAME,
    TIMEFRAME_SECONDS_MAP,
    VEC_NORM_PATH,
    VISION_LLM_API_URL,
    VISION_LLM_TIMEOUT_SEC,
    BOT_WS_URL,
    BOT_CONFIG_ID,
)
from .gate_stats import GateStatsProvider
from .numpy_compat import patch_numpy_bitgenerator_compat as _patch_numpy_bitgenerator_compat

if CORE_DIR not in sys.path:
    sys.path.insert(0, CORE_DIR)

from env_trading import TradingEnv
from .live_bridge import PPOBridge, calc_auto_lot
from .live_features import build_feature_columns
from .live_semantic import SemanticRuntime


mt5 = MetaTrader5(host=MT5_HOST, port=MT5_PORT)
class LiveTradingBot:
    def __init__(self):
        self.model = None
        self.vec_norm = None
        self.bridge = None
        self.semantic_runtime = None
        self.feature_columns = None
        self.gate_provider = GateStatsProvider()

        self.initial_balance = 0.0
        self.current_lot = 0.01
        self.point = 0.00001
        self.digits = 5
        self.pip_size = 0.0001
        self.last_bar_time = 0
        self.last_known_ticket = 0
        self.state_file = STATE_FILE
        self.state_loaded = False
        self.sync_account_state = bool(LIVE_SYNC_ACCOUNT_STATE)
        self.dynamic_lot = bool(LIVE_DYNAMIC_LOT)
        self.llm_semantic_cache_file = LLM_SEMANTIC_CACHE_FILE
        self.llm_text_log_file = LLM_TEXT_LOG_FILE
        self.llm_semantic_cache = {}
        self._ws_connected = False
        self._ws_send_lock = threading.Lock()
        self._ws_state_lock = threading.Lock()
        self._ws_pending_state_payload = None
        self._ws_last_enqueued_at = 0.0
        self._sem_retry_not_before = {}  # ts_key -> epoch seconds
        self.last_action = "HOLD"
        self._last_status_tick = None  # {"bid": float, "ask": float, "time": int}
        self._rt_delta_cache = {
            "bar_open_ts": 0,
            "tick_time_ts": 0,
            "delta_tick": 0,
            "delta_price": 0.0,
            "delta_note": "init",
        }
        self.trade_tick_wait_timeout_sec = max(
            20.0,
            float(max(1, int(ORDER_TICK_RETRIES)) * max(0.05, float(ORDER_TICK_RETRY_SEC))),
        )
        self.vision_llm_api_url = str(VISION_LLM_API_URL or "").strip()
        # Live real-only semantic can take long on model warmup; avoid false timeout loops.
        self.vision_llm_timeout_sec = max(420.0, float(VISION_LLM_TIMEOUT_SEC or 420.0))
        self.risk_profile_map = dict(RISK_PROFILE_MAP or {"low": 0.5, "medium": 1.0, "high": 1.5})
        self.risk_level = str(RISK_LEVEL or "medium").lower()
        self.risk_percent = float(RISK_PERCENT)
        if self.risk_level not in self.risk_profile_map:
            self.risk_level = "medium"
        self.trading_schedule = dict(TRADING_SCHEDULE_DEFAULT)
        self._mt5_reconnect_lock = threading.Lock()
        self._mt5_last_reconnect_at = 0.0

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

        print(
            " [MT5] "
            f"account={account_info.login} symbol={SYMBOL} tf={TIMEFRAME_NAME} "
            f"balance={self.initial_balance:.2f} lot={self.current_lot:.2f} "
            f"risk={self.risk_level}:{self.risk_percent:.2f}%"
        )
        self._add_log(
            "success",
            f"MT5 connected | account={account_info.login} | risk={self.risk_level}:{self.risk_percent:.2f}%",
            phase="boot",
            event="mt5_connected",
            meta={
                "account": int(account_info.login),
                "symbol": SYMBOL,
                "timeframe": TIMEFRAME_NAME,
                "balance": float(self.initial_balance),
                "lot": float(self.current_lot),
                "risk_level": self.risk_level,
                "risk_percent": float(self.risk_percent),
            },
        )

    def _safe_last_error(self):
        try:
            return mt5.last_error()
        except Exception:
            return None

    def _is_mt5_ipc_error(self, err=None) -> bool:
        if err is None:
            err = self._safe_last_error()
        if not isinstance(err, tuple) or len(err) < 2:
            return False
        try:
            code = int(err[0])
        except Exception:
            code = 0
        msg = str(err[1] or "").strip().lower()
        if code in (-10004, 10004, -10006, 10006):
            return True
        return "ipc" in msg and "connection" in msg

    def _try_reconnect_mt5(self, reason: str = "runtime", force: bool = False) -> bool:
        now_epoch = time.time()
        if not force and (now_epoch - float(self._mt5_last_reconnect_at)) < 1.0:
            return False
        if not self._mt5_reconnect_lock.acquire(blocking=False):
            return False

        try:
            now_epoch = time.time()
            if not force and (now_epoch - float(self._mt5_last_reconnect_at)) < 1.0:
                return False
            self._mt5_last_reconnect_at = now_epoch
            print(f"\n [MT5] reconnecting ({reason})...")
            self._add_log(
                "warning",
                f"MT5 reconnecting ({reason})",
                phase="mt5",
                event="reconnect_start",
                meta={"reason": str(reason)},
            )

            max_attempts = 3
            last_err = None
            for attempt in range(1, max_attempts + 1):
                try:
                    mt5.shutdown()
                except Exception:
                    pass
                time.sleep(0.05)

                ok = False
                try:
                    ok = bool(mt5.initialize())
                except Exception:
                    ok = False
                if not ok:
                    last_err = self._safe_last_error()
                    time.sleep(min(0.25 * attempt, 0.8))
                    continue

                try:
                    mt5.symbol_select(SYMBOL, True)
                except Exception:
                    pass

                account_info = mt5.account_info()
                if account_info is not None:
                    symbol_info = mt5.symbol_info(SYMBOL)
                    if symbol_info is not None:
                        self.point = float(symbol_info.point)
                        self.digits = int(symbol_info.digits)
                        self.pip_size = self.point * 10 if self.digits in (3, 5) else self.point
                    print(f"\n [MT5] reconnected account={account_info.login} attempt={attempt}")
                    self._add_log(
                        "success",
                        f"MT5 reconnected on attempt {attempt}",
                        phase="mt5",
                        event="reconnect_ok",
                        meta={"attempt": int(attempt), "account": int(account_info.login)},
                    )
                    return True

                last_err = self._safe_last_error()
                time.sleep(min(0.25 * attempt, 0.8))

            err_text = str(last_err) if last_err is not None else "unknown"
            print(f"\n [MT5] reconnect failed ({reason}) | last_error={err_text}")
            self._add_log(
                "warning",
                "MT5 reconnect failed",
                phase="mt5",
                event="reconnect_failed",
                meta={"reason": str(reason), "last_error": err_text},
            )
            return False
        finally:
            self._mt5_reconnect_lock.release()

    def _order_send_with_ipc_retry(self, req: dict, reason: str, refresh_price_fn=None):
        res = mt5.order_send(req)
        retried = False
        if res and res.retcode == mt5.TRADE_RETCODE_DONE:
            return res, retried

        last_err = self._safe_last_error()
        if not self._is_mt5_ipc_error(last_err):
            return res, retried

        retried = True
        err_text = str(last_err)
        print(f" [MT5] order_send IPC error -> reconnect ({reason}) | err={err_text}")
        self._add_log(
            "warning",
            f"order_send IPC error; reconnecting ({reason})",
            phase="order",
            event="order_send_ipc_error",
            meta={"reason": str(reason), "last_error": err_text},
        )
        if not self._try_reconnect_mt5(reason=f"order_send:{reason}"):
            return res, retried

        if callable(refresh_price_fn):
            try:
                next_price = refresh_price_fn()
            except Exception:
                next_price = None
            if next_price is None:
                return res, retried
            req["price"] = float(next_price)

        try:
            res = mt5.order_send(req)
        except Exception:
            res = None
        return res, retried

    def _load_llm_semantic_cache(self):
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

    def _save_llm_semantic_cache(self, reason: str = "periodic"):
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

        self._load_llm_semantic_cache()
        llm_cache_rows = 0
        # Strict real-only semantic for live:
        # do not keep any non-LLM preloaded map in runtime.
        self.semantic_runtime.global_time_to_vec = {}
        self.semantic_runtime.cache = {}
        self.semantic_runtime.quality_cache = {}
        if self.llm_semantic_cache:
            self.semantic_runtime.global_time_to_vec = {
                key: np.asarray(vec, dtype=np.float32)
                for key, vec in self.llm_semantic_cache.items()
            }
            llm_cache_rows = int(len(self.semantic_runtime.global_time_to_vec))

        print(
            " [MODEL] "
            f"ready features={len(self.feature_columns)} sem_dim={self.semantic_runtime.semantic_feature_count} "
            f"llm_cache_rows={llm_cache_rows}"
        )
        self._add_log(
            "success",
            "Model pipeline ready",
            phase="model",
            event="ready",
            meta={
                "features": int(len(self.feature_columns)),
                "semantic_dim": int(self.semantic_runtime.semantic_feature_count),
                "llm_cache_rows": int(llm_cache_rows),
            },
        )

    def _start_ws_listener(self):
        """Start a background thread: unified WebSocket to BotHub.

        1. Connect to server WS
        2. Send register message (bot_config_id, symbol, timeframe)
        3. Listen for llm_result pushes
        """
        if not BOT_WS_URL or not BOT_CONFIG_ID:
            print(" WS skipped: BOT_WS_URL or BOT_CONFIG_ID not set")
            self._add_log("warning", "WS skipped: BOT_WS_URL or BOT_CONFIG_ID not set", phase="ws", event="disabled")
            return

        def _listener():
            import websockets.sync.client as ws_sync
            while True:
                try:
                    print(f"\n [WS] connecting to {BOT_WS_URL}")
                    with ws_sync.connect(BOT_WS_URL) as ws:
                        self._ws = ws
                        self._ws_connected = False
                        self._add_log("info", "WS connected", phase="ws", event="connected", meta={"url": BOT_WS_URL})
                        # Register
                        with self._ws_send_lock:
                            ws.send(json.dumps({
                                "type": "register",
                                "bot_config_id": BOT_CONFIG_ID,
                                "symbol": SYMBOL.upper(),
                                "timeframe": TIMEFRAME_NAME.upper(),
                            }))
                        self._ws_connected = True
                        print(f"\n [WS] registered: {BOT_CONFIG_ID} {SYMBOL}/{TIMEFRAME_NAME}")
                        self._add_log(
                            "success",
                            "WS registered with BotHub",
                            phase="ws",
                            event="registered",
                            meta={"bot_config_id": BOT_CONFIG_ID, "symbol": SYMBOL, "timeframe": TIMEFRAME_NAME},
                        )
                        try:
                            self._sync_bridge_from_mt5()
                            self._push_state_to_server(action_name=self.last_action or "HOLD")
                            self._flush_pending_state_to_server()
                        except Exception:
                            pass
                        while True:
                            try:
                                raw = ws.recv(timeout=max(0.25, min(float(POLL_SECONDS), 1.0)))
                            except TimeoutError:
                                self._flush_pending_state_to_server()
                                continue
                            try:
                                msg = json.loads(raw)
                            except Exception:
                                self._flush_pending_state_to_server()
                                continue
                            msg_type = msg.get("type")
                            if msg_type == "llm_result":
                                # Live semantic path is HTTP endpoint + shared cache.
                                # Ignore ws-pushed llm_result to keep one deterministic source.
                                pass
                            elif msg_type == "bot_config":
                                self._apply_runtime_config(msg, source="ws")
                            self._flush_pending_state_to_server()
                except Exception as exc:
                    self._ws = None
                    self._ws_connected = False
                    with self._ws_state_lock:
                        self._ws_pending_state_payload = None
                    print(f"\n [WS] disconnected: {exc} | reconnecting in 5s")
                    self._add_log(
                        "warning",
                        "WS disconnected; reconnecting in 5s",
                        phase="ws",
                        event="disconnected",
                        meta={"error": str(exc or ""), "retry_sec": 5},
                    )
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
                phase="config",
                event="runtime_updated",
                meta={"source": source, "risk_level": self.risk_level, "risk_percent": float(self.risk_percent)},
            )

    def _flush_pending_state_to_server(self):
        """Flush queued state payload from WS thread."""
        if not self._ws_connected or not hasattr(self, "_ws") or self._ws is None:
            return

        payload = None
        with self._ws_state_lock:
            payload = self._ws_pending_state_payload
            self._ws_pending_state_payload = None
        if not payload:
            return

        with self._ws_send_lock:
            self._ws.send(payload)

    def _push_state_to_server(self, action_name: str = ""):
        """Queue full MT5 state for WS thread (non-blocking for trading loop)."""
        if not self._ws_connected or not hasattr(self, "_ws") or self._ws is None:
            return
        now = time.time()
        # Avoid flooding queue with HOLD heartbeats.
        if now - float(self._ws_last_enqueued_at) < 0.25 and str(action_name or "").upper() == "HOLD":
            return

        resolved_action = str(action_name or self.last_action or "").strip() or "HOLD"
        self.last_action = resolved_action

        current_pos = int(self.bridge.position) if self.bridge is not None else 0
        pos = None
        try:
            current_pos, pos = self._get_mt5_position()
        except Exception as exc:
            print(f" WS state: position read failed: {exc}")

        # ── MT5 Account info ──
        account_data = {}
        try:
            account = mt5.account_info()
        except Exception:
            account = None
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
        elif self.bridge is not None:
            # Keep dashboard alive even when account_info() is temporarily unavailable.
            account_data = {
                "balance": float(self.bridge.balance),
                "equity": float(self.bridge.equity),
                "margin": 0.0,
                "free_margin": 0.0,
                "margin_level": 0.0,
                "leverage": 0,
                "profit": float(self.bridge.unrealized_pnl),
                "currency": "",
                "server": "",
                "login": 0,
            }

        # ── MT5 Active positions ──
        positions_data = []
        all_positions = None
        try:
            all_positions = mt5.positions_get()
        except Exception:
            all_positions = None
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

        last_err = self._safe_last_error()
        if self._is_mt5_ipc_error(last_err):
            self._try_reconnect_mt5(reason="ws_state_snapshot")

        llm_text = str(getattr(self, "last_llm_text", "") or "")
        if len(llm_text) > 800:
            llm_text = llm_text[:800] + " ..."

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
            "llm_text": llm_text,
            "recent_logs": list(getattr(self, "recent_logs", [])[-40:]),
            "ws_connected": bool(self._ws_connected),
        }
        try:
            payload = json.dumps(state, ensure_ascii=False)
        except Exception as exc:
            print(f" WS state serialize failed: {exc}")
            return
        with self._ws_state_lock:
            self._ws_pending_state_payload = payload
            self._ws_last_enqueued_at = now

    def _sanitize_log_meta(self, meta):
        if not isinstance(meta, dict):
            return None
        clean = {}
        for raw_key, raw_value in meta.items():
            key = str(raw_key or "").strip()
            if not key:
                continue
            if isinstance(raw_value, (str, int, float, bool)) or raw_value is None:
                clean[key] = raw_value
            elif isinstance(raw_value, np.generic):
                clean[key] = raw_value.item()
            elif isinstance(raw_value, datetime):
                clean[key] = raw_value.strftime("%Y-%m-%d %H:%M:%S")
            else:
                clean[key] = str(raw_value)
        return clean or None

    def _add_log(
        self,
        log_type: str,
        message: str,
        phase: str = "",
        event: str = "",
        severity: str = "",
        meta: dict | None = None,
    ):
        """Add a log entry to be sent to the dashboard Activity Log."""
        if not hasattr(self, "recent_logs"):
            self.recent_logs = []
        now_str = datetime.now().strftime("%H:%M:%S")
        entry = {
            "timestamp": now_str,
            "type": str(log_type or "info"),
            "message": str(message or "").strip(),
        }
        phase_txt = str(phase or "").strip().upper()
        if phase_txt:
            entry["phase"] = phase_txt
        event_txt = str(event or "").strip().lower()
        if event_txt:
            entry["event"] = event_txt
        severity_txt = str(severity or "").strip().lower()
        if not severity_txt:
            severity_txt = {
                "warning": "warning",
                "success": "success",
                "action": "success",
                "analysis": "info",
            }.get(entry["type"], "info")
        entry["severity"] = severity_txt
        clean_meta = self._sanitize_log_meta(meta)
        if clean_meta:
            entry["meta"] = clean_meta

        self.recent_logs.append(entry)
        # Keep only recent logs for dashboard
        if len(self.recent_logs) > 120:
            self.recent_logs.pop(0)

    def _save_llm_semantic_entry(self, ts_key: str, cls_vec: np.ndarray, llm_text: str, source: str):
        if self.semantic_runtime is None:
            return
        expected_dim = int(self.semantic_runtime._embedding_dim())
        vec = np.asarray(cls_vec, dtype=np.float32).reshape(-1)
        if vec.size != expected_dim:
            raise RuntimeError(f"CLS dim mismatch expected={expected_dim} got={vec.size}")

        self.llm_semantic_cache[ts_key] = vec
        self.semantic_runtime.global_time_to_vec[ts_key] = vec
        self._sem_retry_not_before.pop(ts_key, None)
        self.last_llm_text = str(llm_text or "")
        if self.last_llm_text:
            self._append_llm_text_log(ts_key, self.last_llm_text)
        self._add_log(
            "analysis",
            f"AI Analysis ready for {ts_key[-8:]} ({source})",
            phase="sem",
            event="embedding_ready",
            meta={"ts": ts_key, "source": source, "dim": int(vec.size)},
        )
        self._save_llm_semantic_cache(reason=source)
        print(f"\n [SEM] ready ({source}) ts={ts_key} dim={vec.size}")

    def _request_llm_semantic_from_server(self, ts_key: str):
        if not self.vision_llm_api_url:
            raise RuntimeError("VISION_LLM_API_URL is empty")

        payload = {
            "date_time": ts_key,
            "symbol": SYMBOL.upper(),
            "timeframe": TIMEFRAME_NAME.upper(),
        }
        body = json.dumps(payload).encode("utf-8")
        req = urlrequest.Request(
            self.vision_llm_api_url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlrequest.urlopen(req, timeout=self.vision_llm_timeout_sec) as resp:
                raw = resp.read().decode("utf-8")
        except urlerror.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8")
            except Exception:
                detail = ""
            raise RuntimeError(
                f"server returned HTTP {exc.code}"
                + (f" | {detail}" if detail else "")
            ) from exc
        except urlerror.URLError as exc:
            raise RuntimeError(f"server unavailable: {exc}") from exc

        try:
            data = json.loads(raw) if raw else {}
        except Exception as exc:
            raise RuntimeError(f"invalid server JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise RuntimeError("invalid server payload type")
        cls_raw = data.get("cls_vec")
        if cls_raw is None:
            raise RuntimeError("server payload missing cls_vec")
        llm_text = str(data.get("llm_text", "") or "")
        cls_vec = np.asarray(cls_raw, dtype=np.float32).reshape(-1)
        return llm_text, cls_vec

    def _request_llm_semantic_from_server_with_heartbeat(self, ts_key: str):
        result = {}
        error = {}
        done = threading.Event()

        def _worker():
            try:
                llm_text, cls_vec = self._request_llm_semantic_from_server(ts_key)
                result["llm_text"] = llm_text
                result["cls_vec"] = cls_vec
            except Exception as exc:
                error["exc"] = exc
            finally:
                done.set()

        worker = threading.Thread(
            target=_worker,
            daemon=True,
            name=f"llm-http-{ts_key[-8:].replace(':', '')}",
        )
        worker.start()

        beat_sec = max(0.5, min(float(POLL_SECONDS), 2.0))
        while not done.wait(timeout=beat_sec):
            self.last_action = "HOLD"
            self._sync_bridge_from_mt5()
            self._push_state_to_server(action_name="HOLD")
            active_bar_ts = self._current_bar_time()
            if active_bar_ts <= 0:
                active_bar_ts = self.last_bar_time
            self._print_status_line(current_bar_time=active_bar_ts)

        if "exc" in error:
            raise error["exc"]
        return result["llm_text"], result["cls_vec"]

    def _resolve_live_llm_semantic(self, ts_key: str):
        if self.semantic_runtime is None:
            return
        if ts_key in self.semantic_runtime.global_time_to_vec:
            return

        now_epoch = time.time()
        retry_not_before = float(self._sem_retry_not_before.get(ts_key, 0.0) or 0.0)
        if now_epoch < retry_not_before:
            return

        # 1. Check local disk cache
        cached_vec = self.llm_semantic_cache.get(ts_key)
        if cached_vec is not None:
            self.semantic_runtime.global_time_to_vec[ts_key] = np.asarray(cached_vec, dtype=np.float32)
            self._add_log(
                "analysis",
                f"Semantic cache hit for {ts_key}",
                phase="sem",
                event="cache_hit",
                meta={"ts": ts_key, "source": "disk_cache"},
            )
            return

        # 2. Request from server HTTP endpoint
        print(f"\n [SEM] requesting server ts={ts_key}")
        self._add_log(
            "analysis",
            f"Requesting semantic from server for {ts_key}",
            phase="sem",
            event="server_request",
            meta={"ts": ts_key, "source": "server_post"},
        )
        try:
            llm_text, cls_vec = self._request_llm_semantic_from_server_with_heartbeat(ts_key)
            self._save_llm_semantic_entry(ts_key, cls_vec, llm_text, source="server_post")
        except Exception as exc:
            # Avoid hammering the endpoint immediately after a timeout/failure.
            self._sem_retry_not_before[ts_key] = time.time() + max(2.0, min(float(POLL_SECONDS) * 4.0, 12.0))
            print(f"\n [SEM] request failed ts={ts_key}: {exc} | fallback=blocked_real_only")
            self._add_log(
                "warning",
                f"Semantic request failed for {ts_key}",
                phase="sem",
                event="server_request_failed",
                meta={"ts": ts_key, "error": str(exc)},
            )

    def _ensure_window_real_semantic(self, window_df: pd.DataFrame):
        if self.semantic_runtime is None:
            return False

        ts_keys = pd.to_datetime(window_df["time"]).dt.strftime("%Y-%m-%d %H:%M:%S").tolist()
        missing = [ts for ts in ts_keys if ts not in self.semantic_runtime.global_time_to_vec]
        if missing:
            print(f"\n [SEM] resolving missing window embeddings={len(missing)}")
            self._add_log(
                "analysis",
                f"Resolving missing window embeddings={len(missing)}",
                phase="sem",
                event="window_missing_embeddings",
                meta={"missing_count": int(len(missing)), "oldest": missing[0], "newest": missing[-1]},
            )
        for ts_key in missing:
            self._resolve_live_llm_semantic(ts_key)
        unresolved = [ts for ts in ts_keys if ts not in self.semantic_runtime.global_time_to_vec]
        if unresolved:
            now_epoch = time.time()
            retry_waits = []
            for ts_key in unresolved:
                retry_not_before = float(self._sem_retry_not_before.get(ts_key, 0.0) or 0.0)
                if retry_not_before > now_epoch:
                    retry_waits.append(retry_not_before - now_epoch)
            next_retry_sec = min(retry_waits) if retry_waits else 0.0
            retry_text = f" | next_retry_in={next_retry_sec:.1f}s" if next_retry_sec > 0 else ""
            print(
                "\n [SEM] pending unresolved real embeddings="
                f"{len(unresolved)} | oldest={unresolved[0]} newest={unresolved[-1]}"
                f"{retry_text}"
            )
            self._add_log(
                "warning",
                f"Pending unresolved real embeddings={len(unresolved)}",
                phase="sem",
                event="window_unresolved_embeddings",
                meta={
                    "missing_count": int(len(unresolved)),
                    "oldest": unresolved[0],
                    "newest": unresolved[-1],
                    "next_retry_sec": float(next_retry_sec),
                },
            )
            return False
        return True

    def _recommended_semantic_wait(self, window_df: pd.DataFrame) -> float:
        base_wait = max(0.5, min(float(POLL_SECONDS), 5.0))
        if self.semantic_runtime is None:
            return base_wait
        ts_keys = pd.to_datetime(window_df["time"]).dt.strftime("%Y-%m-%d %H:%M:%S").tolist()
        unresolved = [ts for ts in ts_keys if ts not in self.semantic_runtime.global_time_to_vec]
        if not unresolved:
            return base_wait
        now_epoch = time.time()
        retry_waits = []
        for ts_key in unresolved:
            retry_not_before = float(self._sem_retry_not_before.get(ts_key, 0.0) or 0.0)
            if retry_not_before > now_epoch:
                retry_waits.append(retry_not_before - now_epoch)
        if not retry_waits:
            return base_wait
        return max(base_wait, min(min(retry_waits) + 0.05, 8.0))

    def _wait_until_real_semantic_ready(self, window_df: pd.DataFrame):
        attempt = 0
        total_needed = int(len(window_df))
        while True:
            attempt += 1
            ready = self._ensure_window_real_semantic(window_df)
            if ready:
                if attempt > 1:
                    print(f"\n [SEM] ready after retry={attempt}")
                    self._add_log(
                        "success",
                        f"Semantic window ready after retry={attempt}",
                        phase="sem",
                        event="window_ready_after_retry",
                        meta={"attempt": int(attempt)},
                    )
                print(f"\n [SEM] window ready: {total_needed}/{total_needed} real embeddings")
                self._add_log(
                    "success",
                    f"Semantic window ready ({total_needed}/{total_needed})",
                    phase="sem",
                    event="window_ready",
                    meta={"count": int(total_needed), "attempt": int(attempt)},
                )
                return True

            wait_sec = self._recommended_semantic_wait(window_df)
            self.last_action = "HOLD"
            self._sync_bridge_from_mt5()
            self._add_log(
                "warning",
                "Waiting for real LLM semantic (retrying; no synthetic fallback)",
                phase="sem",
                event="waiting_real_embedding",
                meta={"attempt": int(attempt), "wait_sec": float(wait_sec)},
            )
            self._push_state_to_server(action_name="HOLD")
            active_bar_ts = self._current_bar_time()
            if active_bar_ts <= 0:
                active_bar_ts = self.last_bar_time
            self._print_status_line(current_bar_time=active_bar_ts)
            time.sleep(wait_sec)

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

    def _get_symbol_positions_safe(self):
        retries = 2
        for attempt in range(retries + 1):
            positions = mt5.positions_get(symbol=SYMBOL)
            if positions is None:
                if attempt < retries:
                    time.sleep(0.05)
                continue
            if len(positions) == 0 and self.last_known_ticket != 0 and attempt < retries:
                time.sleep(0.05)
                continue
            return list(positions)
        last_err = self._safe_last_error()
        if self._is_mt5_ipc_error(last_err):
            self._try_reconnect_mt5(reason="positions_get")
            try:
                positions = mt5.positions_get(symbol=SYMBOL)
                if positions is not None:
                    return list(positions)
            except Exception:
                pass
        try:
            all_positions = mt5.positions_get()
            if all_positions:
                filtered = [
                    p
                    for p in all_positions
                    if str(getattr(p, "symbol", "")).upper() == SYMBOL.upper()
                ]
                return list(filtered)
        except Exception:
            pass
        return []

    def _get_mt5_position(self):
        positions = self._get_symbol_positions_safe()
        if not positions:
            return 0, None
        pos = max(positions, key=lambda p: int(getattr(p, "time", 0) or 0))
        if pos.type == mt5.ORDER_TYPE_BUY:
            return 1, pos
        if pos.type == mt5.ORDER_TYPE_SELL:
            return -1, pos
        return 0, pos

    def _broker_exposure_summary(self):
        pos_count = 0
        order_count = 0
        try:
            pos_count = int(len(self._get_symbol_positions_safe()))
        except Exception:
            pos_count = 0
        try:
            orders = mt5.orders_get(symbol=SYMBOL)
            if orders is not None:
                order_count = int(len(orders))
        except Exception:
            order_count = 0
        has_exposure = pos_count > 0 or order_count > 0
        return has_exposure, pos_count, order_count

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

    def _calc_delta_between(self, start_dt: datetime, end_dt: datetime, quick: bool = False):
        if end_dt <= start_dt:
            return 0, 0.0, "range_empty"
        if quick:
            # Realtime status heartbeat should stay light; avoid heavy retry loops.
            tick_retries = 1
            tick_wait = 0.0
        else:
            tick_retries = max(1, int(ORDER_TICK_RETRIES))
            tick_wait = max(0.0, float(ORDER_TICK_RETRY_SEC))
        ticks = None
        tick_source = "ticks"

        for attempt in range(1, tick_retries + 1):
            try:
                mt5.symbol_select(SYMBOL, True)
            except Exception:
                pass
            try:
                ticks = mt5.copy_ticks_range(SYMBOL, start_dt, end_dt, mt5.COPY_TICKS_ALL)
            except Exception:
                ticks = None

            if ticks is not None and len(ticks) > 1:
                break

            # Some bridges can miss the range-end boundary tick; expand by 1 second before giving up.
            try:
                ticks_padded = mt5.copy_ticks_range(
                    SYMBOL,
                    start_dt,
                    end_dt + timedelta(seconds=1),
                    mt5.COPY_TICKS_ALL,
                )
            except Exception:
                ticks_padded = None

            if ticks_padded is not None and len(ticks_padded) > 1:
                ticks = ticks_padded
                tick_source = "ticks_edge_padded"
                break

            if attempt < tick_retries and tick_wait > 0.0:
                time.sleep(tick_wait)

        def _calc_from_ohlc(open_px: float, close_px: float, note: str):
            open_px = float(open_px)
            close_px = float(close_px)
            if open_px <= 0.0 or close_px <= 0.0:
                return None
            move = float(close_px - open_px)
            if abs(move) <= max(self.point * 0.1, 1e-10):
                delta_tick_tf = 0
            else:
                delta_tick_tf = int(round(move / max(self.point, 1e-10)))
            delta_price_tf = float(move * 2.0)
            return delta_tick_tf, delta_price_tf, note

        def _fallback_from_m1():
            tf_m1 = getattr(mt5, "TIMEFRAME_M1", None)
            if tf_m1 is None:
                return None

            try:
                mt5.symbol_select(SYMBOL, True)
            except Exception:
                pass

            try:
                rates_m1 = mt5.copy_rates_range(
                    SYMBOL,
                    tf_m1,
                    start_dt,
                    end_dt + timedelta(minutes=1),
                )
            except Exception:
                rates_m1 = None

            if rates_m1 is None or len(rates_m1) < 2:
                # Fallback path for bridges where copy_rates_range is intermittently empty.
                try:
                    m1_count = max(3, int((end_dt - start_dt).total_seconds() // 60) + 3)
                    anchor_dt = end_dt - timedelta(seconds=1)
                    rates_m1 = mt5.copy_rates_from(SYMBOL, tf_m1, anchor_dt, m1_count)
                except Exception:
                    rates_m1 = None

            if rates_m1 is None:
                return None
            mdf = pd.DataFrame(rates_m1)
            if len(mdf) < 2 or "close" not in mdf.columns or "time" not in mdf.columns:
                return None
            try:
                mdf["time"] = pd.to_datetime(mdf["time"], unit="s", utc=True)
            except Exception:
                return None
            start_ts = int(start_dt.timestamp())
            end_ts = int(end_dt.timestamp())
            row_ts = (mdf["time"].astype("int64") // 10**9).astype(int)
            mdf = mdf[(row_ts >= start_ts) & (row_ts < end_ts)].copy()
            if len(mdf) < 2:
                return None
            close_diff = mdf["close"].astype(float).diff().fillna(0.0)
            buy = int((close_diff > 0).sum())
            sell = int((close_diff < 0).sum())
            delta_tick_m1 = int(buy - sell)
            price_move = float(mdf["close"].iloc[-1] - mdf["close"].iloc[0])
            delta_price_m1 = float(price_move * 2.0)
            return delta_tick_m1, delta_price_m1, f"m1_fallback(n={len(mdf)})"

        def _fallback_from_tf_bar():
            try:
                mt5.symbol_select(SYMBOL, True)
            except Exception:
                pass

            try:
                rates_tf = mt5.copy_rates_range(
                    SYMBOL,
                    self.timeframe,
                    start_dt,
                    end_dt + timedelta(seconds=1),
                )
            except Exception:
                rates_tf = None

            if rates_tf is None or len(rates_tf) == 0:
                try:
                    anchor_dt = end_dt - timedelta(seconds=1)
                    rates_tf = mt5.copy_rates_from(SYMBOL, self.timeframe, anchor_dt, 4)
                except Exception:
                    rates_tf = None

            if rates_tf is None or len(rates_tf) == 0:
                return None
            tdf = pd.DataFrame(rates_tf)
            if len(tdf) == 0 or "open" not in tdf.columns or "close" not in tdf.columns:
                return None
            if "time" not in tdf.columns:
                return None
            try:
                tdf["time"] = pd.to_datetime(tdf["time"], unit="s", utc=True)
            except Exception:
                return None
            tdf = tdf.sort_values("time").reset_index(drop=True)
            start_ts = int(start_dt.timestamp())
            row_ts = (tdf["time"].astype("int64") // 10**9).astype(int)
            selected = tdf[row_ts == start_ts]
            if len(selected) == 0:
                selected = tdf[row_ts < int(end_dt.timestamp())]
                if len(selected) == 0:
                    selected = tdf.iloc[[0]]
                else:
                    selected = selected.iloc[[-1]]

            row = selected.iloc[-1]
            open_px = float(row.get("open", 0.0))
            close_px = float(row.get("close", 0.0))
            return _calc_from_ohlc(open_px, close_px, f"tf_fallback({TIMEFRAME_NAME},n={len(tdf)})")

        if ticks is None:
            fallback = _fallback_from_m1()
            if fallback is not None:
                return fallback
            fallback = _fallback_from_tf_bar()
            if fallback is not None:
                return fallback
            err = None
            try:
                err = mt5.last_error()
            except Exception:
                err = None
            if isinstance(err, tuple) and len(err) >= 2:
                return 0, 0.0, f"ticks_unavailable(err={err[0]}:{err[1]})"
            return 0, 0.0, "ticks_unavailable"

        tick_count = len(ticks)
        if tick_count <= 1:
            fallback = _fallback_from_m1()
            if fallback is not None:
                return fallback
            fallback = _fallback_from_tf_bar()
            if fallback is not None:
                return fallback
            err = None
            try:
                err = mt5.last_error()
            except Exception:
                err = None
            if isinstance(err, tuple) and len(err) >= 2:
                return 0, 0.0, f"ticks_insufficient(n={tick_count},err={err[0]}:{err[1]})"
            return 0, 0.0, f"ticks_insufficient(n={tick_count})"

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
        return delta_tick, delta_price, f"{tick_source}(n={tick_count})"

    def _calc_delta_for_closed_bar(self, current_bar_ts: int, window_df: pd.DataFrame | None = None):
        end_dt = datetime.fromtimestamp(current_bar_ts, tz=timezone.utc)
        start_dt = end_dt - timedelta(seconds=self.timeframe_seconds)
        delta_tick, delta_price, delta_note = self._calc_delta_between(start_dt, end_dt)

        # Last-resort fallback: derive delta from the actual closed bar OHLC in the model window.
        if (
            (delta_note.startswith("ticks_unavailable") or delta_note.startswith("ticks_insufficient"))
            and window_df is not None
            and len(window_df) > 0
            and "open" in window_df.columns
            and "close" in window_df.columns
        ):
            try:
                row = window_df.iloc[-1]
                open_px = float(row.get("open", 0.0))
                close_px = float(row.get("close", 0.0))
                if open_px > 0.0 and close_px > 0.0:
                    move = float(close_px - open_px)
                    if abs(move) <= max(self.point * 0.1, 1e-10):
                        delta_tick = 0
                    else:
                        delta_tick = int(round(move / max(self.point, 1e-10)))
                    delta_price = float(move * 2.0)
                    delta_note = "window_ohlc_fallback"
            except Exception:
                pass

        return delta_tick, delta_price, delta_note

    def _calc_realtime_delta_for_open_bar(
        self,
        tick_time_ts: int,
        current_bar_ts: int | None = None,
        stale_tick: bool = False,
    ):
        bar_open_ts = int(current_bar_ts or 0)
        if bar_open_ts <= 0:
            bar_open_ts = self._current_bar_time()
        if bar_open_ts <= 0:
            return 0, 0.0, "bar_unavailable"

        now_epoch = int(time.time())
        effective_tick_ts = int(tick_time_ts or 0)
        if effective_tick_ts <= 0:
            effective_tick_ts = now_epoch
        if stale_tick:
            # If quote is stale, expand realtime delta window to "now"
            # so fallback paths can still reflect fresh bar movement.
            effective_tick_ts = max(effective_tick_ts, now_epoch)

        cache = self._rt_delta_cache if isinstance(self._rt_delta_cache, dict) else {}
        if (
            int(cache.get("bar_open_ts", 0) or 0) == bar_open_ts
            and int(cache.get("tick_time_ts", 0) or 0) == effective_tick_ts
        ):
            return (
                int(cache.get("delta_tick", 0) or 0),
                float(cache.get("delta_price", 0.0) or 0.0),
                str(cache.get("delta_note", "cached") or "cached"),
            )

        end_ts = int(max(bar_open_ts + 1, effective_tick_ts))
        start_dt = datetime.fromtimestamp(bar_open_ts, tz=timezone.utc)
        end_dt = datetime.fromtimestamp(end_ts, tz=timezone.utc)
        delta_tick, delta_price, delta_note = self._calc_delta_between(start_dt, end_dt, quick=True)
        self._rt_delta_cache = {
            "bar_open_ts": int(bar_open_ts),
            "tick_time_ts": int(effective_tick_ts),
            "delta_tick": int(delta_tick),
            "delta_price": float(delta_price),
            "delta_note": str(delta_note),
        }
        return delta_tick, delta_price, delta_note

    def _current_bar_time(self) -> int:
        try:
            latest = mt5.copy_rates_from_pos(SYMBOL, self.timeframe, 0, 1)
        except Exception:
            latest = None
        if latest is None:
            last_err = self._safe_last_error()
            if self._is_mt5_ipc_error(last_err):
                self._try_reconnect_mt5(reason="current_bar_time")
                try:
                    latest = mt5.copy_rates_from_pos(SYMBOL, self.timeframe, 0, 1)
                except Exception:
                    latest = None
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

        prev_utc = datetime.fromtimestamp(self.last_bar_time, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
        curr_utc = datetime.fromtimestamp(current_bar_time, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
        print(
            " [FLOW] catch-up replay | "
            f"prev_end={prev_utc} current_end={curr_utc} "
            f"bars={len(missed_bar_ends)} skipped={skipped} "
            f"execute_stale={int(EXECUTE_STALE_REPLAY_ORDERS)}"
        )
        self._add_log(
            "info",
            "Catch-up replay",
            phase="flow",
            event="catchup_replay",
            meta={
                "prev_end": prev_utc,
                "current_end": curr_utc,
                "bars": int(len(missed_bar_ends)),
                "skipped": int(skipped),
                "execute_stale": bool(EXECUTE_STALE_REPLAY_ORDERS),
            },
        )

        total = len(missed_bar_ends)
        for idx, bar_end_ts in enumerate(missed_bar_ends, start=1):
            is_latest = idx == total
            execute_orders = is_latest or EXECUTE_STALE_REPLAY_ORDERS
            if total == 1 and is_latest and execute_orders:
                mode = "New Candle"
            else:
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
            last_err = self._safe_last_error()
            if self._is_mt5_ipc_error(last_err):
                self._try_reconnect_mt5(reason="close_all:positions_get")
                positions = mt5.positions_get(symbol=SYMBOL)
        if positions is None:
            self._add_log(
                "warning",
                "Close skipped: positions_get returned None",
                phase="order",
                event="close_skipped",
            )
            return False
        if len(positions) == 0:
            self._add_log(
                "info",
                "Close requested: no open positions",
                phase="order",
                event="close_no_positions",
            )
            return True

        all_ok = True
        for pos in positions:
            tick = self._get_trade_tick()
            if tick is None:
                print(" Close Skipped: no live tick")
                self._add_log(
                    "warning",
                    f"Close skipped: no live tick for ticket {pos.ticket}",
                    phase="order",
                    event="close_skipped_no_tick",
                    meta={"ticket": int(pos.ticket)},
                )
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

            def _refresh_close_price():
                retry_tick = self._get_trade_tick()
                if retry_tick is None:
                    return None
                return retry_tick.bid if pos.type == mt5.ORDER_TYPE_BUY else retry_tick.ask

            res, _ = self._order_send_with_ipc_retry(
                req,
                reason=f"close:{int(pos.ticket)}",
                refresh_price_fn=_refresh_close_price,
            )
            if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                print(f" Closed Position | ticket={pos.ticket} | PnL={pos.profit:+.2f}")
                self._add_log(
                    "action",
                    f"Closed position #{pos.ticket} | PnL {pos.profit:+.2f}",
                    phase="order",
                    event="close_done",
                    meta={"ticket": int(pos.ticket), "pnl": float(pos.profit)},
                )
            else:
                comment = res.comment if res else "No response"
                if comment == "No response":
                    last_err = self._safe_last_error()
                    if isinstance(last_err, tuple) and len(last_err) >= 2:
                        comment = f"{comment} | mt5_error={last_err[0]}:{last_err[1]}"
                print(f" Close Failed | ticket={pos.ticket} | {comment}")
                self._add_log(
                    "warning",
                    f"Close failed #{pos.ticket}: {comment}",
                    phase="order",
                    event="close_failed",
                    meta={"ticket": int(pos.ticket), "reason": str(comment)},
                )
                all_ok = False
        return all_ok

    def _get_trade_tick(self):
        wait_sec = max(0.05, float(ORDER_TICK_RETRY_SEC))
        deadline = time.time() + max(1.0, float(self.trade_tick_wait_timeout_sec))
        next_beat_at = 0.0
        last_err = None

        while time.time() < deadline:
            try:
                mt5.symbol_select(SYMBOL, True)
            except Exception:
                pass

            tick = mt5.symbol_info_tick(SYMBOL)
            if tick is not None and float(getattr(tick, "bid", 0.0)) > 0.0 and float(getattr(tick, "ask", 0.0)) > 0.0:
                self._last_status_tick = {
                    "bid": float(getattr(tick, "bid", 0.0)),
                    "ask": float(getattr(tick, "ask", 0.0)),
                    "time": int(getattr(tick, "time", 0) or 0),
                }
                return tick

            try:
                last_err = mt5.last_error()
            except Exception:
                last_err = None

            if self._is_mt5_ipc_error(last_err):
                if self._try_reconnect_mt5(reason="get_trade_tick"):
                    continue

            now_epoch = time.time()
            if now_epoch >= next_beat_at:
                self._sync_bridge_from_mt5()
                self._push_state_to_server(action_name=self.last_action or "HOLD")
                active_bar_ts = self._current_bar_time()
                if active_bar_ts <= 0:
                    active_bar_ts = self.last_bar_time
                self._print_status_line(current_bar_time=active_bar_ts)
                next_beat_at = now_epoch + 1.0

            time.sleep(wait_sec)

        # Final fallback: use very recent cached status tick to avoid skipping entry on transient quote gaps.
        if isinstance(self._last_status_tick, dict):
            bid = float(self._last_status_tick.get("bid", 0.0) or 0.0)
            ask = float(self._last_status_tick.get("ask", 0.0) or 0.0)
            ts = int(self._last_status_tick.get("time", 0) or 0)
            age_sec = time.time() - float(ts) if ts > 0 else 9999.0
            if bid > 0.0 and ask > 0.0 and age_sec <= 10.0:
                print(f"\n [ORDER] using cached tick fallback age={age_sec:.1f}s")
                self._add_log(
                    "warning",
                    "Using cached tick fallback for order send",
                    phase="order",
                    event="tick_fallback",
                    meta={"age_sec": float(age_sec)},
                )
                return SimpleNamespace(bid=bid, ask=ask, time=ts)

        err_text = ""
        if isinstance(last_err, tuple) and len(last_err) >= 2:
            err_text = f" | mt5_error={last_err[0]}:{last_err[1]}"
        print(
            "\n [ORDER] trade tick unavailable after "
            f"{self.trade_tick_wait_timeout_sec:.1f}s{err_text}"
        )
        self._add_log(
            "warning",
            "Trade tick unavailable after timeout",
            phase="order",
            event="no_live_tick_timeout",
            meta={
                "timeout_sec": float(self.trade_tick_wait_timeout_sec),
                "last_error": str(last_err),
            },
        )
        return None

    def _get_status_tick(self):
        try:
            mt5.symbol_select(SYMBOL, True)
        except Exception:
            pass
        tick = mt5.symbol_info_tick(SYMBOL)
        if tick is not None and float(getattr(tick, "bid", 0.0)) > 0.0:
            self._last_status_tick = {
                "bid": float(getattr(tick, "bid", 0.0)),
                "ask": float(getattr(tick, "ask", 0.0)),
                "time": int(getattr(tick, "time", 0) or 0),
            }
            return self._last_status_tick, False
        if isinstance(self._last_status_tick, dict):
            return self._last_status_tick, True
        return None, False

    def send_order(self, order_type):
        tick = self._get_trade_tick()
        if tick is None:
            print(" Order Skipped: no live tick (market closed or quote unavailable)")
            self._add_log(
                "warning",
                "Order skipped: no live tick",
                phase="order",
                event="open_skipped_no_tick",
                meta={"order_type": "BUY" if order_type == mt5.ORDER_TYPE_BUY else "SELL"},
            )
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

        def _refresh_open_price():
            retry_tick = self._get_trade_tick()
            if retry_tick is None:
                return None
            return retry_tick.ask if order_type == mt5.ORDER_TYPE_BUY else retry_tick.bid

        res, _ = self._order_send_with_ipc_retry(
            req,
            reason="open_buy" if order_type == mt5.ORDER_TYPE_BUY else "open_sell",
            refresh_price_fn=_refresh_open_price,
        )
        if res and res.retcode == mt5.TRADE_RETCODE_DONE:
            side = "BUY" if order_type == mt5.ORDER_TYPE_BUY else "SELL"
            price = float(req.get("price", price))
            if self.bridge is not None:
                self.bridge.lot_size = float(self.current_lot)
                self.bridge.spread_cost = SPREAD_PIPS * PIP_VALUE * self.bridge.lot_size
            print(
                f" Opened {side} @ {price:.{self.digits}f} | "
                f"Lot={self.current_lot}"
            )
            self._add_log(
                "action",
                f"Opened {side} @ {price:.{self.digits}f} | lot={self.current_lot:.2f}",
                phase="order",
                event="open_done",
                meta={"side": side, "price": float(price), "lot": float(self.current_lot)},
            )
            return True
        else:
            comment = res.comment if res else "No response"
            if comment == "No response":
                last_err = self._safe_last_error()
                if isinstance(last_err, tuple) and len(last_err) >= 2:
                    comment = f"{comment} | mt5_error={last_err[0]}:{last_err[1]}"
            print(f" Order Failed: {comment}")
            self._add_log(
                "warning",
                f"Order failed: {comment}",
                phase="order",
                event="open_failed",
                meta={"reason": str(comment)},
            )
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
            phase="order",
            event="broker_reconcile_mismatch",
            meta={
                "action": int(action),
                "expected_pos": int(expected_after),
                "actual_pos": int(broker_pos_after),
                "order_ok": bool(order_ok),
            },
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

    def _print_status_line(self, current_bar_time: int | None = None):
        tick_data, stale_tick = self._get_status_tick()
        if tick_data is None:
            return

        current_pos, pos = self._get_mt5_position()
        pos_txt = {1: "LONG", -1: "SHORT", 0: "FLAT"}[current_pos]
        pnl_txt = f"{pos.profit:+.2f}" if pos is not None else "0.00"
        tick_time_ts = int(tick_data.get("time", 0) or 0)
        now_epoch = int(time.time())
        stale_age_sec = max(0, now_epoch - tick_time_ts) if tick_time_ts > 0 else 0
        if tick_time_ts > 0:
            if stale_tick:
                # Show runtime clock while tick is stale so operator sees loop is alive.
                server_time_utc = datetime.fromtimestamp(now_epoch, tz=timezone.utc).strftime("%H:%M:%SZ")
            else:
                server_time_utc = datetime.fromtimestamp(tick_time_ts, tz=timezone.utc).strftime("%H:%M:%SZ")
        else:
            server_time_utc = "--:--:--Z"
        rt_delta_tick, rt_delta_price, _ = self._calc_realtime_delta_for_open_bar(
            tick_time_ts=tick_time_ts,
            current_bar_ts=current_bar_time,
            stale_tick=stale_tick,
        )

        # Keep the status line compact to avoid terminal wrap.
        stale_note = f" | stale_tick({stale_age_sec}s)" if stale_tick else ""
        line = (
            f"Pr:{float(tick_data.get('bid', 0.0)):.{self.digits}f} | Pos:{pos_txt} | PnL:{pnl_txt} | "
            f"Eq:{self.bridge.equity:.2f} | dT:{rt_delta_tick:d} | dP:{rt_delta_price:.5f} | "
            f"{server_time_utc}{stale_note}"
        )
        width = max(40, int(shutil.get_terminal_size(fallback=(120, 20)).columns) - 1)
        if len(line) > width:
            line = line[:width]
        # Clear whole current line before writing next snapshot.
        sys.stdout.write("\r\033[2K" + line)
        sys.stdout.flush()

    def _phase_icon(self, mode: str) -> str:
        text = str(mode or "").strip().lower()
        if "startup" in text:
            return "🚀"
        if "new candle" in text:
            return "🕐"
        if "catch-up" in text:
            return "⏩"
        return "📊"

    def _process_closed_bar(
        self,
        bar_end_ts: int,
        mode: str = "New Candle",
        execute_orders: bool = True,
        persist_state: bool = True,
        allow_entry_orders: bool = True,
        preserve_timers_on_hold: bool = False,
    ):
        bar_end_utc = datetime.fromtimestamp(bar_end_ts, tz=timezone.utc)
        bar_open_utc = bar_end_utc - timedelta(seconds=self.timeframe_seconds)
        icon = self._phase_icon(mode)
        print(
            f"\n {icon} {'=' * 18} [BAR] {mode} {'=' * 18}\n"
            f" {icon} [BAR] open={bar_open_utc.strftime('%Y-%m-%d %H:%M:%SZ')} "
            f"end={bar_end_utc.strftime('%Y-%m-%d %H:%M:%SZ')}"
        )
        self._add_log(
            "info",
            f"{mode}: processing closed bar",
            phase="bar",
            event="bar_start",
            meta={
                "mode": mode,
                "open_utc": bar_open_utc.strftime("%Y-%m-%d %H:%M:%SZ"),
                "end_utc": bar_end_utc.strftime("%Y-%m-%d %H:%M:%SZ"),
            },
        )

        window_df = self._fetch_window(bar_end_ts)
        if window_df is None:
            print(" Not enough bars yet for model window")
            self._add_log(
                "warning",
                "Not enough bars yet for model window",
                phase="bar",
                event="window_not_ready",
                meta={"required": int(BAR_HISTORY)},
            )
            return

        self._wait_until_real_semantic_ready(window_df)

        delta_tick, delta_price, delta_note = self._calc_delta_for_closed_bar(bar_end_ts, window_df=window_df)
        self.bridge.gate_stats = self.gate_provider.update(window_df)

        self._sync_bridge_from_mt5()
        pre_trade_cooldown = int(self.bridge.trade_cooldown) if self.bridge is not None else 0
        pre_defensive_mode = int(self.bridge.defensive_mode_bars) if self.bridge is not None else 0
        action, model_price = self.bridge.process_bar(window_df, delta_tick, delta_price)
        action = int(action)
        decision = dict(getattr(self.bridge, "last_decision", {}) or {})

        # Startup replay on an already-processed bar should not decay cooldown/defensive timers
        # when the model ends up HOLD (prevents state drift from stop/start loops).
        if preserve_timers_on_hold and action == 0 and self.bridge is not None:
            self.bridge.trade_cooldown = int(pre_trade_cooldown)
            self.bridge.defensive_mode_bars = int(pre_defensive_mode)
            if isinstance(decision, dict):
                decision["cooldown_before"] = int(pre_trade_cooldown)
                decision["cooldown_after"] = int(pre_trade_cooldown)
                decision["defensive_mode_before"] = int(pre_defensive_mode)
                decision["defensive_mode_after"] = int(pre_defensive_mode)

        action_name = {0: "HOLD", 1: "BUY", 2: "SELL", 3: "CLOSE"}.get(action, "?")
        raw_action = int(decision.get("raw_action", action))
        raw_action_name = {0: "HOLD", 1: "BUY", 2: "SELL", 3: "CLOSE"}.get(raw_action, "?")
        self.last_action = action_name
        print(
            f" [MODEL] action={action_name} raw={raw_action_name} | price={model_price:.5f} | "
            f"dTick={delta_tick} | dPrice={delta_price:.5f} | {delta_note}"
        )
        probs = decision.get("probs")
        if isinstance(probs, list) and len(probs) >= 4:
            print(
                " [MODEL] probs: "
                f"HOLD={float(probs[0]):.3f} BUY={float(probs[1]):.3f} "
                f"SELL={float(probs[2]):.3f} CLOSE={float(probs[3]):.3f}"
            )
        gate_reasons = [str(x) for x in decision.get("gate_reasons", []) if str(x).strip()]
        if action == 0 and raw_action != 0 and gate_reasons:
            print(f" [GATE] blocked -> HOLD | reasons: {', '.join(gate_reasons)}")
        self._add_log(
            "analysis",
            f"{mode} | action={action_name} raw={raw_action_name} | price={model_price:.5f} | dTick={delta_tick}",
            phase="model",
            event="bar_inference",
            meta={
                "mode": mode,
                "final_action": action_name,
                "raw_action": raw_action_name,
                "price": float(model_price),
                "delta_tick": int(delta_tick),
                "delta_price": float(delta_price),
                "delta_note": delta_note,
            },
        )
        if isinstance(probs, list) and len(probs) >= 4:
            self._add_log(
                "analysis",
                "Model probabilities",
                phase="model",
                event="probabilities",
                meta={
                    "hold": float(probs[0]),
                    "buy": float(probs[1]),
                    "sell": float(probs[2]),
                    "close": float(probs[3]),
                },
            )
        if action == 0 and raw_action != 0 and gate_reasons:
            self._add_log(
                "warning",
                "Gate blocked signal -> HOLD",
                phase="gate",
                event="blocked",
                meta={"raw_action": raw_action_name, "reasons": " | ".join(gate_reasons)},
            )

        schedule_blocked = False
        startup_entry_blocked = False
        if execute_orders and not self._is_trading_day_enabled(bar_end_ts):
            if action in (1, 2):
                self._add_log(
                    "warning",
                    f"Schedule blocked new order ({action_name}) on disabled trading day",
                    phase="schedule",
                    event="blocked",
                    meta={"action": action_name, "bar_end_ts": int(bar_end_ts)},
                )
                action = 0
                action_name = "HOLD"
                schedule_blocked = True
        if execute_orders and not allow_entry_orders and action in (1, 2):
            self._add_log(
                "info",
                f"{mode}: startup exposure mode blocked new entry ({action_name})",
                phase="startup",
                event="startup_entry_blocked",
                meta={"mode": mode, "action": action_name},
            )
            action = 0
            action_name = "HOLD"
            startup_entry_blocked = True
        self.last_action = action_name

        decision_status = "pass"
        if raw_action == 0 and action == 0:
            decision_status = "model_hold"
        elif raw_action != 0 and action == 0:
            if schedule_blocked:
                decision_status = "schedule_blocked"
            elif startup_entry_blocked:
                decision_status = "startup_entry_blocked"
            elif gate_reasons:
                decision_status = "gate_blocked"
            else:
                decision_status = "suppressed"
        elif raw_action != action:
            decision_status = "adjusted"

        reason_parts = []
        if gate_reasons:
            reason_parts.append("gate:" + ", ".join(gate_reasons))
        if schedule_blocked:
            reason_parts.append("schedule")
        if startup_entry_blocked:
            reason_parts.append("startup_entry_blocked")
        if not reason_parts:
            reason_parts.append("-")
        sem_q = float(decision.get("semantic_quality", 0.0))
        cooldown_after = int(decision.get("cooldown_after", 0))
        print(
            " [DECISION] "
            f"raw={raw_action_name} final={action_name} "
            f"status={decision_status} sem_q={sem_q:.2f} "
            f"cooldown={cooldown_after} reasons={' | '.join(reason_parts)}"
        )
        self._add_log(
            "analysis",
            f"Decision {decision_status}: raw={raw_action_name} final={action_name}",
            phase="decision",
            event="summary",
            meta={
                "status": decision_status,
                "raw_action": raw_action_name,
                "final_action": action_name,
                "semantic_quality": float(sem_q),
                "cooldown_after": int(cooldown_after),
                "reasons": " | ".join(reason_parts),
            },
        )

        broker_pos_before, _ = self._get_mt5_position()
        order_ok = True
        if execute_orders:
            order_ok = bool(self.execute_action(action))
        elif action != 0:
            print(" No-order mode: action skipped")
            self._add_log(
                "info",
                f"{mode}: action {action_name} skipped in no-order mode",
                phase="order",
                event="skipped_no_order_mode",
                meta={"mode": mode, "action": action_name},
            )
        self._sync_bridge_from_mt5()
        broker_pos_after, _ = self._get_mt5_position()
        self._reconcile_broker_execution(
            action=action,
            execute_orders=execute_orders,
            order_ok=order_ok,
            broker_pos_before=int(broker_pos_before),
            broker_pos_after=int(broker_pos_after),
        )
        # Re-sync account/equity after reconcile corrections so dashboard and status line stay aligned.
        self._sync_bridge_from_mt5()
        if persist_state:
            self._save_runtime_state(reason="bar_close")
        summary_type = "info"
        if action_name in ("BUY", "SELL", "CLOSE"):
            summary_type = "action" if (not execute_orders or order_ok) else "warning"
        summary_suffix = f" ({decision_status})" if decision_status else ""
        if not execute_orders and action_name != "HOLD":
            summary_suffix = f"{summary_suffix} (no-order-mode)" if summary_suffix else " (no-order-mode)"
        self._add_log(
            summary_type,
            f"Bar evaluation completed: {action_name}{summary_suffix}",
            phase="bar",
            event="bar_complete",
            meta={
                "mode": mode,
                "raw_action": raw_action_name,
                "final_action": action_name,
                "status": decision_status,
                "execute_orders": bool(execute_orders),
                "order_ok": bool(order_ok),
            },
        )
        self._push_state_to_server(action_name=action_name)

    def run(self):
        print("\n 🚀 [BOOT] run_live starting...")
        self._add_log("info", "run_live starting", phase="boot", event="start")
        self.connect()
        self._load_model()
        self._load_runtime_state()
        self._sync_bridge_from_mt5()
        if self.bridge is not None:
            print(
                " [READY] "
                f"balance={self.bridge.balance:.2f} equity={self.bridge.equity:.2f} "
                f"lot={self.bridge.lot_size:.2f} last_bar_time={self.last_bar_time}"
            )
            self._add_log(
                "success",
                "Live runtime ready",
                phase="boot",
                event="ready",
                meta={
                    "balance": float(self.bridge.balance),
                    "equity": float(self.bridge.equity),
                    "lot": float(self.bridge.lot_size),
                    "last_bar_time": int(self.last_bar_time),
                },
            )
        self._start_ws_listener()
        startup_eval_pending = bool(EVAL_ON_START)

        print(f" 🕐 [CLOCK] Waiting for new {TIMEFRAME_NAME} candles...")
        self._add_log(
            "info",
            f"Waiting for new {TIMEFRAME_NAME} candles",
            phase="clock",
            event="waiting_new_candle",
            meta={"timeframe": TIMEFRAME_NAME},
        )

        try:
            while True:
                current_bar_time = self._current_bar_time()
                if current_bar_time <= 0:
                    last_err = self._safe_last_error()
                    if self._is_mt5_ipc_error(last_err):
                        self._try_reconnect_mt5(reason="clock_wait")
                    time.sleep(POLL_SECONDS)
                    continue

                if startup_eval_pending:
                    startup_eval_pending = False
                    if self.last_bar_time == 0:
                        print("\n 🚀 [STARTUP] no last_bar_time in state -> process latest closed bar with live orders")
                        self._add_log(
                            "info",
                            "Startup: no last_bar_time -> execute latest closed bar",
                            phase="startup",
                            event="no_state_bar",
                            meta={"execute_orders": True},
                        )
                        self.last_bar_time = current_bar_time
                        self._process_closed_bar(current_bar_time, mode="Startup", execute_orders=True)
                        self._sync_bridge_from_mt5()
                        self._push_state_to_server()
                        self._print_status_line(current_bar_time=current_bar_time)
                        time.sleep(POLL_SECONDS)
                        continue

                    if self.last_bar_time == current_bar_time:
                        has_exposure, pos_count, order_count = self._broker_exposure_summary()
                        if has_exposure:
                            print(
                                "\n 🚀 [STARTUP] current bar already processed; "
                                f"broker exposure detected (positions={pos_count}, orders={order_count}) "
                                "-> exposure sync (allow CLOSE, block new entry)"
                            )
                            self._add_log(
                                "info",
                                "Startup exposure sync: allow CLOSE, block new BUY/SELL",
                                phase="startup",
                                event="startup_exposure_sync",
                                meta={"positions": int(pos_count), "orders": int(order_count)},
                            )
                            self._process_closed_bar(
                                current_bar_time,
                                mode="Startup Exposure Sync",
                                execute_orders=True,
                                persist_state=True,
                                allow_entry_orders=False,
                                preserve_timers_on_hold=True,
                            )
                        else:
                            print(
                                "\n 🚀 [STARTUP] current bar already processed; "
                                "no open exposure -> execute startup action now"
                            )
                            self._add_log(
                                "info",
                                "Startup immediate: no exposure -> execute action",
                                phase="startup",
                                event="startup_immediate",
                                meta={"execute_orders": True},
                            )
                            self._process_closed_bar(
                                current_bar_time,
                                mode="Startup Immediate",
                                execute_orders=True,
                                persist_state=True,
                                allow_entry_orders=True,
                                preserve_timers_on_hold=True,
                            )
                        self._sync_bridge_from_mt5()
                        self._push_state_to_server()
                        self._print_status_line(current_bar_time=current_bar_time)
                        time.sleep(POLL_SECONDS)
                        continue

                    print("\n Startup eval deferred: missed bars detected; catch-up replay will process first")

                if self.last_bar_time == 0:
                    self.last_bar_time = current_bar_time
                    self._sync_bridge_from_mt5()
                    self._push_state_to_server()
                    self._print_status_line(current_bar_time=current_bar_time)
                    time.sleep(POLL_SECONDS)
                    continue

                if current_bar_time != self.last_bar_time:
                    closed_utc = datetime.fromtimestamp(current_bar_time, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
                    print(f"\n 🕐 [CLOCK] New closed candle detected at {closed_utc}")
                    self._add_log(
                        "info",
                        "New closed candle detected",
                        phase="clock",
                        event="new_closed_candle",
                        meta={"closed_utc": closed_utc},
                    )
                    self._replay_missed_bars_if_any(current_bar_time)
                    if self.last_bar_time != current_bar_time:
                        self.last_bar_time = current_bar_time
                        self._process_closed_bar(current_bar_time, mode="New Candle", execute_orders=True)

                self._sync_bridge_from_mt5()
                self._push_state_to_server()
                self._print_status_line(current_bar_time=current_bar_time)
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
