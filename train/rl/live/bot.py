import json
import os
import shutil
import sys
import time
import threading
from contextlib import contextmanager
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

try:
    import fcntl
except Exception:  # pragma: no cover - non-POSIX runtime
    fcntl = None

from .config import (
    BAR_HISTORY,
    CORE_DIR,
    DEVIATION,
    ENABLE_CATCHUP_REPLAY,
    EVAL_ON_START,
    EMBED_QUALITY_MIN,
    EXECUTE_STALE_REPLAY_ORDERS,
    FIXED_LOT,
    INTRABAR_TAKE_PROFIT_CHANGE_PCT,
    INTRABAR_TAKE_PROFIT_MONEY,
    INTRABAR_TAKE_PROFIT_PIPS,
    INTRABAR_TRAILING_ENABLED,
    INTRABAR_TRAIL_ARM_BUFFER_RATIO,
    INTRABAR_TRAIL_CONFIRM_POLLS,
    INTRABAR_TRAIL_KEEP_RATIO_NORMAL,
    INTRABAR_TRAIL_KEEP_RATIO_TIGHT,
    INTRABAR_TRAIL_KEEP_RATIO_TREND,
    LIVE_DYNAMIC_LOT,
    LIVE_SYNC_ACCOUNT_STATE,
    LLM_SEMANTIC_CACHE_FILE,
    LLM_SEMANTIC_CACHE_SCHEMA,
    LLM_TEXT_LOG_FILE,
    MAGIC_NUMBER,
    MAX_CATCHUP_BARS,
    MODEL_PATH,
    MODELS_DIR,
    LIVE_PREWARM_REQUEST_TIMEOUT_SEC,
    LIVE_PREWARM_SEMANTIC_MAX_MISSING,
    LIVE_PREWARM_SEMANTIC_MAX_SECONDS,
    LIVE_PREWARM_SEMANTIC_ON_START,
    LIVE_PERFORMANCE_BOOT_LOOKBACK_DAYS,
    LIVE_MANAGED_MAGIC_SET,
    LIVE_MT5_HISTORY_END_AHEAD_HOURS,
    LIVE_SEMANTIC_ALIAS_HOURS,
    LIVE_SEMANTIC_NO_DATA_RETRY_SECONDS,
    LIVE_PERFORMANCE_MAGIC_SET,
    LIVE_PERFORMANCE_SCOPE,
    LIVE_PERFORMANCE_SYNC_INTERVAL_SEC,
    MT5_HOST,
    MT5_LOGIN,
    MT5_LOGIN_RETRIES,
    MT5_PASSWORD,
    MT5_PORT,
    MT5_RETRY_SECONDS,
    MT5_RPC_TIMEOUT_MS,
    MT5_SERVER,
    MT5_SERVER_FALLBACKS,
    MT5_STRICT_SERVER_MATCH,
    ORDER_TICK_RETRIES,
    ORDER_TICK_RETRY_SEC,
    PIP_VALUE,
    POLL_SECONDS,
    RISK_PERCENT,
    RISK_LEVEL,
    RISK_MODE,
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
    VISION_LLM_EMBED_TEXT_API_URL,
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
        self.llm_text_cache = {}
        self._ws_connected = False
        self._ws_send_lock = threading.Lock()
        self._ws_state_lock = threading.Lock()
        self._ws_pending_state_payload = None
        self._ws_last_enqueued_at = 0.0
        self._open_position_tickets = set()
        self._closed_deal_cursor_msc = 0
        self._last_closed_deal_poll_at = 0.0
        self._last_closed_deal_reconcile_at = 0.0
        self._closed_deal_retry_payload = []
        self._closed_deal_retry_until = 0.0
        self._perf_last_sync_at = 0.0
        self._perf_cursor_msc = 0
        self._perf_seeded = False
        self._perf_deal_seen = set()
        self._perf_ticket_net = {}
        self.pending_intrabar_reviews = []
        self.recent_intrabar_reviews = []
        self.intrabar_trailing_state = {}
        self.intrabar_regime_snapshot = {}
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
        self.intrabar_take_profit_change_pct = float(INTRABAR_TAKE_PROFIT_CHANGE_PCT)
        self.intrabar_take_profit_pips = float(INTRABAR_TAKE_PROFIT_PIPS)
        self.intrabar_take_profit_money = float(INTRABAR_TAKE_PROFIT_MONEY)
        self.intrabar_trailing_enabled = bool(INTRABAR_TRAILING_ENABLED)
        self.intrabar_trail_keep_ratio_trend = float(INTRABAR_TRAIL_KEEP_RATIO_TREND)
        self.intrabar_trail_keep_ratio_normal = float(INTRABAR_TRAIL_KEEP_RATIO_NORMAL)
        self.intrabar_trail_keep_ratio_tight = float(INTRABAR_TRAIL_KEEP_RATIO_TIGHT)
        self.intrabar_trail_arm_buffer_ratio = float(INTRABAR_TRAIL_ARM_BUFFER_RATIO)
        self.intrabar_trail_confirm_polls = max(1, int(INTRABAR_TRAIL_CONFIRM_POLLS))
        self.vision_llm_api_url = str(VISION_LLM_API_URL or "").strip()
        self.vision_llm_embed_text_api_url = str(VISION_LLM_EMBED_TEXT_API_URL or "").strip()
        # Live real-only semantic can take long on model warmup; avoid false timeout loops.
        self.vision_llm_timeout_sec = max(420.0, float(VISION_LLM_TIMEOUT_SEC or 420.0))
        self.risk_profile_map = dict(RISK_PROFILE_MAP or {"low": 0.5, "medium": 1.0, "high": 1.5})
        self.risk_level = str(RISK_LEVEL or "medium").lower()
        self.risk_percent = float(RISK_PERCENT)
        self.risk_mode = str(RISK_MODE or "level").strip().lower()
        self.fixed_lot = self._normalize_runtime_lot(FIXED_LOT, fallback=0.0)
        if self.risk_level not in self.risk_profile_map:
            self.risk_level = "medium"
        if self.risk_mode not in {"level", "custom_lot"}:
            self.risk_mode = "level"
        if self.risk_mode == "custom_lot" and self.fixed_lot < 0.01:
            self.risk_mode = "level"
        self.trading_schedule = dict(TRADING_SCHEDULE_DEFAULT)
        self._mt5_reconnect_lock = threading.Lock()
        self._mt5_last_reconnect_at = 0.0
        self.mt5_login_text = str(MT5_LOGIN or "").strip()
        self.mt5_password = str(MT5_PASSWORD or "").strip()
        self.mt5_server = str(MT5_SERVER or "").strip()
        self.mt5_server_fallbacks = tuple(
            str(item or "").strip()
            for item in (MT5_SERVER_FALLBACKS or [])
            if str(item or "").strip()
        )
        self.mt5_strict_server_match = bool(MT5_STRICT_SERVER_MATCH)
        self.mt5_rpc_timeout_ms = max(30000, int(MT5_RPC_TIMEOUT_MS or 180000))
        self.mt5_login_retries = max(1, int(MT5_LOGIN_RETRIES or 20))
        self.mt5_retry_seconds = max(1.0, float(MT5_RETRY_SECONDS or 5.0))
        self.mt5_history_end_ahead_hours = max(0.0, float(LIVE_MT5_HISTORY_END_AHEAD_HOURS or 0.0))
        def _env_float_runtime(name: str, default: float) -> float:
            raw = str(os.getenv(name, "") or "").strip()
            if not raw:
                return float(default)
            try:
                return float(raw)
            except Exception:
                return float(default)

        self.trade_allowed_wait_timeout_sec = max(
            0.0,
            _env_float_runtime("LIVE_TRADE_ALLOWED_WAIT_TIMEOUT_SEC", 20.0),
        )
        self.trade_allowed_probe_interval_sec = max(
            0.1,
            _env_float_runtime("LIVE_TRADE_ALLOWED_PROBE_INTERVAL_SEC", 1.0),
        )
        self.trade_allowed_warn_cooldown_sec = max(
            1.0,
            _env_float_runtime("LIVE_TRADE_ALLOWED_WARN_COOLDOWN_SEC", 10.0),
        )
        self._last_trade_allowed_blocked_log_at = 0.0
        self.semantic_alias_hours = tuple(int(v) for v in (LIVE_SEMANTIC_ALIAS_HOURS or (0,)))
        if 0 not in self.semantic_alias_hours:
            self.semantic_alias_hours = (0,) + self.semantic_alias_hours
        self.semantic_no_data_retry_seconds = max(10.0, float(LIVE_SEMANTIC_NO_DATA_RETRY_SECONDS or 180.0))
        try:
            self.mt5_login_id = int(self.mt5_login_text) if self.mt5_login_text else None
        except ValueError:
            self.mt5_login_id = None
            print(f" [MT5] invalid MT5_LOGIN value: {self.mt5_login_text!r}")

        tf_attr = f"TIMEFRAME_{TIMEFRAME_NAME}"
        self.timeframe = getattr(mt5, tf_attr, mt5.TIMEFRAME_H1)
        self.timeframe_seconds = int(TIMEFRAME_SECONDS_MAP.get(TIMEFRAME_NAME, 3600))

    def _server_candidates(self) -> list[str]:
        candidates: list[str] = []

        def add(value: str) -> None:
            normalized = " ".join(str(value or "").split()).strip()
            if normalized and normalized not in candidates:
                candidates.append(normalized)

        primary = str(self.mt5_server or "").strip()
        add(primary)

        for fallback in self.mt5_server_fallbacks:
            add(fallback)

        if self.mt5_strict_server_match:
            if len(candidates) == 0:
                return [""]
            return candidates

        if primary:
            if primary.lower().startswith("mt5 "):
                add(primary[4:])
            else:
                add(f"MT5 {primary}")

        if len(candidates) == 0:
            return [""]
        return candidates

    def _build_mt5_initialize_kwargs(
        self,
        server_name: str | None = None,
        include_credentials: bool = True,
    ) -> dict:
        kwargs = {"timeout": int(self.mt5_rpc_timeout_ms)}
        if server_name:
            kwargs["server"] = str(server_name).strip()
        if include_credentials and self.mt5_login_id is not None and self.mt5_password:
            kwargs["login"] = int(self.mt5_login_id)
            kwargs["password"] = self.mt5_password
        return kwargs

    def _build_mt5_login_kwargs(self, server_name: str | None = None) -> dict | None:
        if self.mt5_login_id is None or not self.mt5_password:
            return None
        kwargs = {"login": int(self.mt5_login_id), "password": self.mt5_password}
        if server_name:
            kwargs["server"] = str(server_name).strip()
        return kwargs

    def _account_matches_expected_login(self, account_info) -> bool:
        if account_info is None:
            return False
        if self.mt5_login_id is None:
            return True
        try:
            return int(account_info.login) == int(self.mt5_login_id)
        except Exception:
            return False

    def _remember_mt5_server(self, account_info=None, fallback_server: str | None = None) -> None:
        resolved_server = ""
        if account_info is not None:
            resolved_server = str(getattr(account_info, "server", "") or "").strip()
        if resolved_server:
            self.mt5_server = resolved_server
        elif fallback_server:
            self.mt5_server = str(fallback_server).strip()

    def _initialize_mt5_session(self, prefer_session_reuse: bool = False) -> bool:
        if prefer_session_reuse:
            reuse_candidates: list[str | None] = [None]
            for candidate in self._server_candidates():
                normalized = str(candidate or "").strip()
                if normalized and normalized not in reuse_candidates:
                    reuse_candidates.append(normalized)

            # Reconnect path: avoid credentialed login because some MT5 builds
            # can flip AutoTrading off when login() is called repeatedly.
            for server_name in reuse_candidates:
                try:
                    if not bool(
                        mt5.initialize(
                            **self._build_mt5_initialize_kwargs(
                                server_name=server_name,
                                include_credentials=False,
                            )
                        )
                    ):
                        continue
                except Exception:
                    continue

                try:
                    account_info = mt5.account_info()
                except Exception:
                    account_info = None

                if self._account_matches_expected_login(account_info):
                    self._remember_mt5_server(account_info, server_name)
                    return True
            return False

        for server_name in self._server_candidates():
            try:
                if not bool(
                    mt5.initialize(
                        **self._build_mt5_initialize_kwargs(
                            server_name=server_name,
                            include_credentials=True,
                        )
                    )
                ):
                    continue
            except Exception:
                continue

            login_kwargs = self._build_mt5_login_kwargs(server_name)
            if not login_kwargs:
                if server_name:
                    self.mt5_server = str(server_name).strip()
                return True

            try:
                login_ok = bool(mt5.login(**login_kwargs))
            except Exception:
                login_ok = False

            if login_ok:
                if server_name:
                    self.mt5_server = str(server_name).strip()
                return True

            # Some MT5 builds can return login=False while account session is still active.
            try:
                account_info = mt5.account_info()
            except Exception:
                account_info = None
            if account_info is None:
                continue
            if self._account_matches_expected_login(account_info):
                self._remember_mt5_server(account_info, server_name)
                return True

        return False

    def connect(self):
        account_info = None
        last_err = None
        for attempt in range(1, self.mt5_login_retries + 1):
            if self._initialize_mt5_session(prefer_session_reuse=True) or self._initialize_mt5_session():
                account_info = mt5.account_info()
                if account_info is not None:
                    break

            last_err = self._safe_last_error()
            if attempt < self.mt5_login_retries:
                time.sleep(min(self.mt5_retry_seconds, 10.0))

        if account_info is None:
            raise RuntimeError(f"Failed to get account info after login retries. last_error={last_err}")

        if self.mt5_login_id is not None and int(account_info.login) != int(self.mt5_login_id):
            raise RuntimeError(
                f"Connected to unexpected account {account_info.login}, expected {self.mt5_login_id}"
            )

        symbol_info = mt5.symbol_info(SYMBOL)
        if symbol_info is None:
            raise RuntimeError(f"Symbol {SYMBOL} not found")

        if not symbol_info.visible and not mt5.symbol_select(SYMBOL, True):
            raise RuntimeError(f"symbol_select({SYMBOL}) failed")

        self.initial_balance = float(account_info.balance)
        self.current_lot = self._resolve_runtime_lot(self.initial_balance)
        self.point = float(symbol_info.point)
        self.digits = int(symbol_info.digits)
        self.pip_size = self.point * 10 if self.digits in (3, 5) else self.point

        print(
            " [MT5] "
            f"account={account_info.login} symbol={SYMBOL} tf={TIMEFRAME_NAME} "
            f"balance={self.initial_balance:.2f} lot={self.current_lot:.2f} "
            f"risk={self._runtime_risk_label()}"
        )
        self._add_log(
            "success",
            f"MT5 connected | account={account_info.login} | risk={self._runtime_risk_label()}",
            phase="boot",
            event="mt5_connected",
            meta={
                "account": int(account_info.login),
                "symbol": SYMBOL,
                "timeframe": TIMEFRAME_NAME,
                "balance": float(self.initial_balance),
                "lot": float(self.current_lot),
                "risk_level": self.risk_level,
                "risk_mode": self.risk_mode,
                "risk_percent": float(self.risk_percent),
                "fixed_lot": float(self.fixed_lot),
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

            max_attempts = max(3, min(12, self.mt5_login_retries))
            last_err = None
            for attempt in range(1, max_attempts + 1):
                try:
                    mt5.shutdown()
                except Exception:
                    pass
                time.sleep(0.05)

                ok = self._initialize_mt5_session(prefer_session_reuse=True)
                if not ok:
                    last_err = self._safe_last_error()
                    time.sleep(min(self.mt5_retry_seconds, 5.0))
                    continue

                try:
                    mt5.symbol_select(SYMBOL, True)
                except Exception:
                    pass

                account_info = mt5.account_info()
                if account_info is not None:
                    if self.mt5_login_id is not None and int(account_info.login) != int(self.mt5_login_id):
                        last_err = f"unexpected_account:{account_info.login}"
                        time.sleep(min(self.mt5_retry_seconds, 5.0))
                        continue
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
                        meta={
                            "attempt": int(attempt),
                            "account": int(account_info.login),
                            "session_reuse": bool(attempt == 1),
                        },
                    )
                    return True

                last_err = self._safe_last_error()
                time.sleep(min(self.mt5_retry_seconds, 5.0))

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

        retcode = getattr(res, "retcode", None) if res is not None else None
        comment = str(getattr(res, "comment", "") or "").strip() if res is not None else ""
        if self._is_autotrading_disabled_result(retcode, comment):
            if not self._ensure_trade_allowed_before_order(
                reason=f"order_send:{reason}",
                action_label="Order retry",
                force_log=True,
            ):
                return res, retried
            retried = True
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

    def _trade_retcode_name(self, retcode) -> str:
        try:
            code = int(retcode)
        except Exception:
            return str(retcode or "")
        for attr in dir(mt5):
            if not attr.startswith("TRADE_RETCODE_"):
                continue
            try:
                if int(getattr(mt5, attr)) == code:
                    return attr
            except Exception:
                continue
        return str(code)

    def _is_autotrading_disabled_result(self, retcode=None, text: str = "") -> bool:
        try:
            code = int(retcode) if retcode is not None else None
        except Exception:
            code = None

        client_disabled_code = getattr(mt5, "TRADE_RETCODE_CLIENT_DISABLES_AT", 10027)
        try:
            if code is not None and int(code) == int(client_disabled_code):
                return True
        except Exception:
            pass

        haystack = str(text or "").strip().lower()
        if not haystack:
            return False
        keywords = (
            "autotrading disabled by client",
            "auto trading disabled by client",
            "client disables autotrading",
            "client disables auto trading",
            "disabled by client",
        )
        return any(word in haystack for word in keywords)

    def _probe_trade_allowed_state(self) -> tuple[bool | None, str]:
        try:
            info = mt5.terminal_info()
        except Exception as exc:
            return None, f"terminal_info_exception={exc}"

        if info is None:
            return None, "terminal_info_none"

        trade_allowed = bool(getattr(info, "trade_allowed", False))
        tradeapi_disabled = bool(getattr(info, "tradeapi_disabled", False))
        effective_allowed = bool(trade_allowed and not tradeapi_disabled)
        detail = (
            f"trade_allowed={int(trade_allowed)}"
            f" tradeapi_disabled={int(tradeapi_disabled)}"
            f" effective_allowed={int(effective_allowed)}"
        )
        return effective_allowed, detail

    def _ensure_trade_allowed_before_order(
        self,
        reason: str,
        action_label: str = "Order",
        force_log: bool = False,
    ) -> bool:
        timeout_sec = max(0.0, float(self.trade_allowed_wait_timeout_sec))
        probe_interval = max(0.1, float(self.trade_allowed_probe_interval_sec))
        deadline = time.time() + timeout_sec
        attempt = 0
        last_detail = "trade_allowed_probe_unavailable"
        wait_logged = False

        while True:
            attempt += 1
            trade_allowed, detail = self._probe_trade_allowed_state()
            last_detail = str(detail or last_detail)
            if trade_allowed is True:
                if attempt > 1:
                    print(f" [MT5] AutoTrading recovered ({reason}) | {last_detail}")
                    self._add_log(
                        "info",
                        f"MT5 AutoTrading recovered ({reason})",
                        phase="order",
                        event="autotrading_recovered",
                        meta={
                            "reason": str(reason),
                            "attempt": int(attempt),
                            "detail": last_detail,
                        },
                    )
                return True

            if not wait_logged and timeout_sec > 0:
                wait_logged = True
                self._add_log(
                    "warning",
                    "Waiting for MT5 AutoTrading to be enabled",
                    phase="order",
                    event="autotrading_wait",
                    meta={
                        "reason": str(reason),
                        "timeout_sec": float(timeout_sec),
                    },
                )

            if trade_allowed is None:
                last_err = self._safe_last_error()
                if self._is_mt5_ipc_error(last_err):
                    self._try_reconnect_mt5(reason=f"trade_allowed_probe:{reason}")

            if time.time() >= deadline:
                break
            time.sleep(probe_interval)

        now_epoch = time.time()
        should_log = bool(force_log) or (
            (now_epoch - float(self._last_trade_allowed_blocked_log_at))
            >= float(self.trade_allowed_warn_cooldown_sec)
        )
        if should_log:
            msg = f"{action_label} blocked: AutoTrading disabled by client"
            print(f" {msg} | {last_detail}")
            self._add_log(
                "warning",
                msg,
                phase="order",
                event="autotrading_disabled",
                meta={
                    "reason": str(reason),
                    "detail": last_detail,
                    "timeout_sec": float(timeout_sec),
                },
            )
            self._last_trade_allowed_blocked_log_at = now_epoch
        return False

    def _is_insufficient_funds_result(self, retcode=None, text: str = "") -> bool:
        try:
            code = int(retcode) if retcode is not None else None
        except Exception:
            code = None
        no_money_code = getattr(mt5, "TRADE_RETCODE_NO_MONEY", 10019)
        try:
            if code is not None and int(code) == int(no_money_code):
                return True
        except Exception:
            pass

        haystack = str(text or "").strip().lower()
        if not haystack:
            return False
        keywords = (
            "no money",
            "not enough money",
            "insufficient",
            "insufficient funds",
            "insufficient margin",
            "margin",
            "funds",
        )
        return any(word in haystack for word in keywords)

    def _precheck_open_order_funds(self, req: dict, side: str) -> tuple[bool, dict]:
        side_text = str(side or "").strip().upper() or "ORDER"
        account = None
        try:
            account = mt5.account_info()
        except Exception:
            account = None
        balance = float(getattr(account, "balance", 0.0) or 0.0) if account is not None else 0.0
        free_margin = float(getattr(account, "margin_free", 0.0) or 0.0) if account is not None else 0.0
        if account is not None and free_margin <= 0.0:
            return False, {
                "reason": f"{side_text} blocked: free margin is {free_margin:.2f}",
                "retcode": getattr(mt5, "TRADE_RETCODE_NO_MONEY", 10019),
                "retcode_name": "TRADE_RETCODE_NO_MONEY",
                "balance": balance,
                "free_margin": free_margin,
                "required_margin": 0.0,
            }

        try:
            check = mt5.order_check(req)
        except Exception:
            check = None

        if check is None:
            last_err = self._safe_last_error()
            err_text = str(last_err)
            if self._is_insufficient_funds_result(None, err_text):
                return False, {
                    "reason": f"{side_text} blocked: {err_text}",
                    "retcode": getattr(mt5, "TRADE_RETCODE_NO_MONEY", 10019),
                    "retcode_name": "TRADE_RETCODE_NO_MONEY",
                    "balance": balance,
                    "free_margin": free_margin,
                    "required_margin": 0.0,
                }
            return True, {
                "reason": "",
                "retcode": None,
                "retcode_name": "",
                "balance": balance,
                "free_margin": free_margin,
                "required_margin": 0.0,
            }

        retcode = getattr(check, "retcode", None)
        comment = str(getattr(check, "comment", "") or "").strip()
        required_margin = float(getattr(check, "margin", 0.0) or 0.0)
        done_code = getattr(mt5, "TRADE_RETCODE_DONE", None)
        try:
            if done_code is not None and int(retcode) == int(done_code):
                return True, {
                    "reason": "",
                    "retcode": retcode,
                    "retcode_name": self._trade_retcode_name(retcode),
                    "balance": balance,
                    "free_margin": free_margin,
                    "required_margin": required_margin,
                }
        except Exception:
            pass

        detail = comment or self._trade_retcode_name(retcode)
        if self._is_insufficient_funds_result(retcode, detail):
            return False, {
                "reason": f"{side_text} blocked: {detail}",
                "retcode": retcode,
                "retcode_name": self._trade_retcode_name(retcode),
                "balance": balance,
                "free_margin": free_margin,
                "required_margin": required_margin,
            }

        return True, {
            "reason": "",
            "retcode": retcode,
            "retcode_name": self._trade_retcode_name(retcode),
            "balance": balance,
            "free_margin": free_margin,
            "required_margin": required_margin,
        }

    @contextmanager
    def _exclusive_file_lock(self, target_path: str):
        lock_path = f"{target_path}.lock"
        lock_dir = os.path.dirname(lock_path)
        lock_fh = None
        try:
            if lock_dir:
                os.makedirs(lock_dir, exist_ok=True)
            lock_fh = open(lock_path, "a+", encoding="utf-8")
            if fcntl is not None:
                fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
            yield
        finally:
            if lock_fh is not None:
                if fcntl is not None:
                    try:
                        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
                    except Exception:
                        pass
                try:
                    lock_fh.close()
                except Exception:
                    pass

    def _llm_text_log_entry_exists_locked(self, ts_key: str) -> bool:
        log_file = self.llm_text_log_file
        target_ts = str(ts_key or "").strip()
        if not log_file or not target_ts or not os.path.exists(log_file):
            return False
        target_symbol = str(SYMBOL or "").strip().upper()
        target_timeframe = str(TIMEFRAME_NAME or "").strip().upper()
        try:
            with open(log_file, "r", encoding="utf-8") as fh:
                for raw_line in fh:
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except Exception:
                        continue
                    if str(payload.get("time", "")).strip() != target_ts:
                        continue
                    if str(payload.get("symbol", "")).strip().upper() != target_symbol:
                        continue
                    if str(payload.get("timeframe", "")).strip().upper() != target_timeframe:
                        continue
                    return True
        except FileNotFoundError:
            return False
        except Exception as exc:
            print(f" LLM text log dedupe scan failed: {exc}")
        return False

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

    def _load_llm_text_log_cache(self):
        self.llm_text_cache = {}
        log_file = self.llm_text_log_file
        if not log_file or not os.path.exists(log_file):
            return
        try:
            with self._exclusive_file_lock(log_file):
                with open(log_file, "r", encoding="utf-8") as fh:
                    for raw_line in fh:
                        line = str(raw_line or "").strip()
                        if not line:
                            continue
                        try:
                            payload = json.loads(line)
                        except Exception:
                            continue
                        if not isinstance(payload, dict):
                            continue
                        ts_key = str(payload.get("time", "") or "").strip()
                        llm_text = str(payload.get("text", "") or "").strip()
                        if not ts_key or not llm_text:
                            continue
                        self.llm_text_cache[ts_key] = llm_text
        except Exception as exc:
            print(f" LLM text cache skipped (invalid file): {exc}")

    def _save_llm_semantic_cache(self, reason: str = "periodic"):
        if not self.llm_semantic_cache:
            return
        cache_file = self.llm_semantic_cache_file
        try:
            cache_dir = os.path.dirname(cache_file)
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
            with self._exclusive_file_lock(cache_file):
                merged_rows = {}
                if os.path.exists(cache_file):
                    try:
                        existing_payload = joblib.load(cache_file)
                    except Exception:
                        existing_payload = None
                    existing_rows = None
                    if isinstance(existing_payload, dict):
                        if (
                            existing_payload.get("schema") == LLM_SEMANTIC_CACHE_SCHEMA
                            and isinstance(existing_payload.get("rows"), dict)
                        ):
                            existing_rows = existing_payload.get("rows")
                        elif existing_payload and all(isinstance(k, str) for k in existing_payload.keys()):
                            existing_rows = existing_payload
                    if isinstance(existing_rows, dict):
                        for key, vec in existing_rows.items():
                            if not isinstance(key, str):
                                continue
                            arr = np.asarray(vec, dtype=np.float32).reshape(-1)
                            if arr.size == 0:
                                continue
                            merged_rows[key] = arr

                for key, vec in self.llm_semantic_cache.items():
                    if not isinstance(key, str):
                        continue
                    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
                    if arr.size == 0:
                        continue
                    merged_rows[key] = arr

                if not merged_rows:
                    return

                serializable = {k: np.asarray(v, dtype=np.float32) for k, v in merged_rows.items()}
                payload = {
                    "schema": LLM_SEMANTIC_CACHE_SCHEMA,
                    "saved_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                    "rows": serializable,
                }
                tmp_path = f"{cache_file}.tmp.{os.getpid()}.{int(time.time() * 1000000)}"
                try:
                    joblib.dump(payload, tmp_path)
                    os.replace(tmp_path, cache_file)
                finally:
                    if os.path.exists(tmp_path):
                        try:
                            os.remove(tmp_path)
                        except Exception:
                            pass
                self.llm_semantic_cache = merged_rows
        except Exception as exc:
            print(f" LLM semantic cache save failed ({reason}): {exc}")

    def _resolve_cached_semantic_alias(self, ts_key: str):
        if not ts_key:
            return None, None

        # 1) Exact lookup first.
        vec = self.llm_semantic_cache.get(ts_key)
        if vec is not None:
            return ts_key, np.asarray(vec, dtype=np.float32).reshape(-1)

        # 2) Broker server time can be offset by whole hours.
        try:
            base_dt = datetime.strptime(str(ts_key), "%Y-%m-%d %H:%M:%S")
        except Exception:
            return None, None

        for hour_offset in self.semantic_alias_hours:
            try:
                offset = int(hour_offset)
            except Exception:
                continue
            if offset == 0:
                continue
            candidate_key = (base_dt + timedelta(hours=offset)).strftime("%Y-%m-%d %H:%M:%S")
            vec = self.llm_semantic_cache.get(candidate_key)
            if vec is not None:
                return candidate_key, np.asarray(vec, dtype=np.float32).reshape(-1)

        return None, None

    def _adopt_semantic_alias(self, ts_key: str, alias_key: str, alias_vec: np.ndarray):
        vec = np.asarray(alias_vec, dtype=np.float32).reshape(-1)
        if vec.size == 0:
            return False
        self.llm_semantic_cache[ts_key] = vec
        if self.semantic_runtime is not None:
            self.semantic_runtime.global_time_to_vec[ts_key] = vec
        self._sem_retry_not_before.pop(ts_key, None)
        self._hydrate_last_llm_text_from_cache(ts_key)
        self._save_llm_semantic_cache(reason="cache_alias")
        print(f"\n [SEM] ready (cache_alias) ts={ts_key} <- {alias_key} dim={vec.size}")
        self._add_log(
            "analysis",
            f"Semantic alias hit for {ts_key}",
            phase="sem",
            event="cache_alias_hit",
            meta={"ts": ts_key, "alias_ts": alias_key, "dim": int(vec.size)},
        )
        return True

    def _resolve_cached_llm_text_alias(self, ts_key: str):
        if not ts_key:
            return None, None

        llm_text = str(self.llm_text_cache.get(ts_key, "") or "").strip()
        if llm_text:
            return ts_key, llm_text

        try:
            base_dt = datetime.strptime(str(ts_key), "%Y-%m-%d %H:%M:%S")
        except Exception:
            return None, None

        for hour_offset in self.semantic_alias_hours:
            try:
                offset = int(hour_offset)
            except Exception:
                continue
            if offset == 0:
                continue
            candidate_key = (base_dt + timedelta(hours=offset)).strftime("%Y-%m-%d %H:%M:%S")
            llm_text = str(self.llm_text_cache.get(candidate_key, "") or "").strip()
            if llm_text:
                return candidate_key, llm_text

        return None, None

    def _hydrate_last_llm_text_from_cache(self, ts_key: str) -> bool:
        text_key, llm_text = self._resolve_cached_llm_text_alias(ts_key)
        if not llm_text:
            return False

        self.last_llm_text = str(llm_text or "")
        if text_key and text_key != ts_key:
            self.llm_text_cache[ts_key] = self.last_llm_text
            self._append_llm_text_log(ts_key, self.last_llm_text)
        return bool(self.last_llm_text)

    def _request_llm_text_embedding_from_server(self, llm_text: str, timeout_sec: float | None = None):
        if not self.vision_llm_embed_text_api_url:
            raise RuntimeError("VISION_LLM_EMBED_TEXT_API_URL is empty")

        payload = {"text": str(llm_text or "")}
        body = json.dumps(payload).encode("utf-8")
        req = urlrequest.Request(
            self.vision_llm_embed_text_api_url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        timeout = float(timeout_sec or 0.0)
        if timeout <= 0:
            timeout = max(10.0, min(float(self.vision_llm_timeout_sec), 120.0))
        try:
            with urlrequest.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8")
        except urlerror.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8")
            except Exception:
                detail = ""
            raise RuntimeError(
                f"text-embed server returned HTTP {exc.code}"
                + (f" | {detail}" if detail else "")
            ) from exc
        except urlerror.URLError as exc:
            raise RuntimeError(f"text-embed server unavailable: {exc}") from exc

        try:
            data = json.loads(raw) if raw else {}
        except Exception as exc:
            raise RuntimeError(f"text-embed invalid server JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise RuntimeError("text-embed invalid server payload type")
        cls_raw = data.get("cls_vec")
        if cls_raw is None:
            raise RuntimeError("text-embed server payload missing cls_vec")
        return np.asarray(cls_raw, dtype=np.float32).reshape(-1)

    def _try_resolve_semantic_from_text_log(self, ts_key: str, timeout_sec: float | None = None) -> bool:
        text_key, llm_text = self._resolve_cached_llm_text_alias(ts_key)
        if not llm_text:
            return False

        source = "text_log_embed" if text_key == ts_key else "text_log_alias_embed"
        try:
            cls_vec = self._request_llm_text_embedding_from_server(llm_text, timeout_sec=timeout_sec)
            self._save_llm_semantic_entry(ts_key, cls_vec, llm_text, source=source)
            self._add_log(
                "analysis",
                f"Semantic restored from text log for {ts_key}",
                phase="sem",
                event="text_log_embed_ready",
                meta={
                    "ts": ts_key,
                    "text_ts": text_key,
                    "source": source,
                    "text_len": int(len(llm_text)),
                },
            )
            return True
        except Exception as exc:
            self._add_log(
                "warning",
                f"Text-log semantic restore failed for {ts_key}",
                phase="sem",
                event="text_log_embed_failed",
                meta={"ts": ts_key, "text_ts": text_key, "error": str(exc)},
            )
            return False

    def _semantic_retry_delay(self, exc: Exception) -> float:
        err_text = str(exc or "")
        if "No M1 data" in err_text:
            return max(10.0, float(self.semantic_no_data_retry_seconds))
        return max(2.0, min(float(POLL_SECONDS) * 4.0, 12.0))

    def _append_llm_text_log(self, ts_key: str, llm_text: str):
        if not self.llm_text_log_file:
            return
        ts_val = str(ts_key or "").strip()
        text_val = str(llm_text or "").strip()
        try:
            log_dir = os.path.dirname(self.llm_text_log_file)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            record = {
                "time": ts_val,
                "symbol": SYMBOL,
                "timeframe": TIMEFRAME_NAME,
                "text": text_val,
                "saved_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            }
            with self._exclusive_file_lock(self.llm_text_log_file):
                if self._llm_text_log_entry_exists_locked(ts_val):
                    return
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
                    lot_size=self._resolve_runtime_lot(self.initial_balance or 100.0),
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
        # Keep time_to_llm_text as append-only log; do not load it for semantic restore.
        llm_cache_rows = 0
        llm_text_rows = 0
        base_cache_rows = 0
        merged_cache_rows = 0

        # Use static CLS map as baseline, then overlay live LLM cache.
        base_map = {}
        if isinstance(self.semantic_runtime.global_time_to_vec, dict):
            for key, vec in self.semantic_runtime.global_time_to_vec.items():
                if not isinstance(key, str):
                    continue
                arr = np.asarray(vec, dtype=np.float32).reshape(-1)
                if arr.size == 0:
                    continue
                base_map[key] = arr
        base_cache_rows = int(len(base_map))

        self.semantic_runtime.global_time_to_vec = dict(base_map)
        self.semantic_runtime.cache = {}
        self.semantic_runtime.quality_cache = {}
        if self.llm_semantic_cache:
            llm_applied = 0
            for key, vec in self.llm_semantic_cache.items():
                if not isinstance(key, str):
                    continue
                arr = np.asarray(vec, dtype=np.float32).reshape(-1)
                if arr.size == 0:
                    continue
                # LLM vector overrides baseline CLS when timestamps collide.
                self.semantic_runtime.global_time_to_vec[key] = arr
                llm_applied += 1
            llm_cache_rows = int(llm_applied)
        merged_cache_rows = int(len(self.semantic_runtime.global_time_to_vec))

        print(
            " [MODEL] "
            f"ready features={len(self.feature_columns)} sem_dim={self.semantic_runtime.semantic_feature_count} "
            f"base_cache_rows={base_cache_rows} llm_cache_rows={llm_cache_rows} "
            f"merged_cache_rows={merged_cache_rows} llm_text_rows={llm_text_rows}"
        )
        self._add_log(
            "success",
            "Model pipeline ready",
            phase="model",
            event="ready",
            meta={
                "features": int(len(self.feature_columns)),
                "semantic_dim": int(self.semantic_runtime.semantic_feature_count),
                "base_cache_rows": int(base_cache_rows),
                "llm_cache_rows": int(llm_cache_rows),
                "merged_cache_rows": int(merged_cache_rows),
                "llm_text_rows": int(llm_text_rows),
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
                                # WS llm_result is broadcast by symbol/timeframe only.
                                # Keep semantic ownership on the bot's direct HTTP request path.
                                pass
                            elif msg_type == "bot_config":
                                self._apply_runtime_config(msg, source="ws")
                            elif msg_type == "bot_command":
                                self._handle_runtime_command(msg)
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

    def _normalize_runtime_lot(self, value, fallback: float = 0.01) -> float:
        try:
            lot = float(value)
        except Exception:
            lot = float(fallback)
        return round(max(0.01, lot), 2)

    def _is_custom_lot_mode(self) -> bool:
        return str(self.risk_mode or "").strip().lower() == "custom_lot" and float(self.fixed_lot or 0.0) >= 0.01

    def _resolve_runtime_lot(self, balance: float | None = None) -> float:
        if self._is_custom_lot_mode():
            return self._normalize_runtime_lot(self.fixed_lot)
        try:
            current_balance = float(balance)
        except Exception:
            current_balance = float(self.initial_balance or 0.0)
        return max(
            0.01,
            calc_auto_lot(current_balance, risk_pct=self._resolve_runtime_risk_percent()),
        )

    def _runtime_risk_label(self) -> str:
        if self._is_custom_lot_mode():
            return f"custom_lot:{self._normalize_runtime_lot(self.fixed_lot):.2f}"
        return f"{self.risk_level}:{self.risk_percent:.2f}%"

    def _refresh_lot_from_account(self):
        if self._is_custom_lot_mode():
            next_lot = self._resolve_runtime_lot()
        else:
            if not self.dynamic_lot:
                return
            account = mt5.account_info()
            if account is None:
                return
            next_lot = self._resolve_runtime_lot(float(account.balance))
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

        incoming_mode = payload.get("risk_mode")
        if incoming_mode is not None:
            mode = str(incoming_mode).strip().lower()
            if mode in {"level", "custom_lot"} and mode != self.risk_mode:
                self.risk_mode = mode
                changed = True

        incoming_custom_lot = payload.get("custom_lot")
        if incoming_custom_lot is not None:
            lot = self._normalize_runtime_lot(incoming_custom_lot, fallback=0.0)
            if lot != self.fixed_lot:
                self.fixed_lot = lot
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
            if self.risk_mode == "custom_lot" and self.fixed_lot < 0.01:
                self.risk_mode = "level"
            effective_risk = self._resolve_runtime_risk_percent()
            self.risk_percent = float(effective_risk)
            self._refresh_lot_from_account()
            self._add_log(
                "info",
                f"Runtime config updated ({source}) | risk={self._runtime_risk_label()}",
                phase="config",
                event="runtime_updated",
                meta={
                    "source": source,
                    "risk_level": self.risk_level,
                    "risk_mode": self.risk_mode,
                    "risk_percent": float(self.risk_percent),
                    "fixed_lot": float(self.fixed_lot),
                },
            )

    def _send_ws_payload(self, payload: dict) -> bool:
        if not self._ws_connected or not hasattr(self, "_ws") or self._ws is None:
            return False
        try:
            body = json.dumps(payload, ensure_ascii=False)
        except Exception:
            return False
        try:
            with self._ws_send_lock:
                self._ws.send(body)
            return True
        except Exception:
            return False

    def _send_bot_command_ack(
        self,
        command: str,
        command_id: str,
        ok: bool,
        detail: str,
        meta: dict | None = None,
    ):
        payload = {
            "type": "bot_command_ack",
            "bot_config_id": BOT_CONFIG_ID,
            "command": str(command or "").strip().lower(),
            "command_id": str(command_id or "").strip(),
            "ok": bool(ok),
            "detail": str(detail or "").strip(),
            "ts": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        }
        clean_meta = self._sanitize_log_meta(meta)
        if clean_meta:
            payload["meta"] = clean_meta
        self._send_ws_payload(payload)

    def _handle_runtime_command(self, payload: dict):
        if not isinstance(payload, dict):
            return

        command = str(payload.get("command", "")).strip().lower()
        command_id = str(payload.get("command_id", "")).strip()
        if not command:
            return
        if not command_id:
            command_id = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")

        if command in {"emergency_close_all", "close_all_positions", "close_all"}:
            started_at = time.time()
            self._add_log(
                "warning",
                "Emergency command received: close managed positions",
                phase="order",
                event="emergency_close_requested",
            )
            ok = False
            detail = ""
            try:
                ok = bool(self.close_all())
                detail = "All managed positions were closed" if ok else "Some managed positions could not be closed"
            except Exception as exc:
                detail = f"Emergency close failed: {exc}"
                ok = False

            try:
                self._sync_bridge_from_mt5()
                self._push_state_to_server(action_name=self.last_action or "HOLD")
                self._flush_pending_state_to_server()
            except Exception:
                pass

            elapsed_sec = round(max(0.0, time.time() - started_at), 2)
            self._add_log(
                "success" if ok else "warning",
                detail,
                phase="order",
                event="emergency_close_done" if ok else "emergency_close_partial",
                meta={"elapsed_sec": elapsed_sec},
            )
            self._send_bot_command_ack(
                command=command,
                command_id=command_id,
                ok=ok,
                detail=detail,
                meta={"elapsed_sec": elapsed_sec},
            )
            return

        self._send_bot_command_ack(
            command=command,
            command_id=command_id,
            ok=False,
            detail=f"Unsupported command: {command}",
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
        managed_positions = self._get_symbol_positions_safe()
        if managed_positions:
            for p in managed_positions:
                commission_value = 0.0
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
                    "commission": commission_value,
                    "sl": float(p.sl),
                    "tp": float(p.tp),
                    "opened_at": opened_at_utc.strftime("%Y-%m-%d %H:%M:%S"),
                    "opened_at_ts": int(p.time),
                    "time": datetime.fromtimestamp(
                        p.time, tz=timezone.utc
                    ).strftime("%H:%M:%S"),
                    "comment": str(p.comment) if p.comment else "",
                    "magic": int(getattr(p, "magic", 0) or 0),
                })

        current_open_tickets = {
            int(item.get("ticket", 0) or 0)
            for item in positions_data
            if int(item.get("ticket", 0) or 0) > 0
        }
        previous_open_tickets = set(getattr(self, "_open_position_tickets", set()) or set())
        closed_ticket_candidates = previous_open_tickets - current_open_tickets
        self._open_position_tickets = current_open_tickets

        closed_deals = []
        is_reconcile_due = (
            now - float(getattr(self, "_last_closed_deal_reconcile_at", 0.0) or 0.0)
        ) >= 900.0
        should_poll_closed_deals = bool(closed_ticket_candidates) or is_reconcile_due
        if not should_poll_closed_deals:
            should_poll_closed_deals = (now - float(getattr(self, "_last_closed_deal_poll_at", 0.0) or 0.0)) >= 60.0
        if should_poll_closed_deals:
            self._last_closed_deal_poll_at = now
            try:
                if is_reconcile_due:
                    self._last_closed_deal_reconcile_at = now
                    closed_deals = self._collect_recent_closed_deals(force_lookback_sec=3 * 24 * 60 * 60)
                else:
                    closed_deals = self._collect_recent_closed_deals(
                        expected_tickets=closed_ticket_candidates if closed_ticket_candidates else None
                    )
            except Exception as exc:
                print(f" WS state: closed deals fetch failed: {exc}")
                closed_deals = []

        # Keep recently discovered closed deals in outgoing state for a short window.
        # This gives server-side persistence a few retries if one WS frame is dropped.
        if closed_deals:
            latest_by_ticket = {}
            for row in list(getattr(self, "_closed_deal_retry_payload", []) or []) + list(closed_deals):
                if not isinstance(row, dict):
                    continue
                ticket = int(row.get("ticket", 0) or 0)
                if ticket <= 0:
                    continue
                close_msc = int(row.get("closeTimeMsc", 0) or 0)
                prev = latest_by_ticket.get(ticket)
                prev_msc = int((prev or {}).get("closeTimeMsc", 0) or 0)
                if prev is None or close_msc >= prev_msc:
                    latest_by_ticket[ticket] = row
            retry_rows = sorted(
                list(latest_by_ticket.values()),
                key=lambda row: int(row.get("closeTimeMsc", 0) or 0),
                reverse=True,
            )
            self._closed_deal_retry_payload = retry_rows[:200]
            self._closed_deal_retry_until = now + 20.0
        else:
            retry_rows = list(getattr(self, "_closed_deal_retry_payload", []) or [])
            retry_until = float(getattr(self, "_closed_deal_retry_until", 0.0) or 0.0)
            if retry_rows and now < retry_until:
                closed_deals = retry_rows
            elif retry_rows and now >= retry_until:
                self._closed_deal_retry_payload = []
        if closed_deals:
            self._sync_performance_from_mt5_history(
                force=True,
                full_resync=False,
                reason="closed_deals_snapshot",
            )

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
            "magic_number": int(MAGIC_NUMBER),
            "manage_manual_positions": bool(0 in set(int(v) for v in (LIVE_MANAGED_MAGIC_SET or set()))),
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
            "risk_mode": self.risk_mode,
            "risk_percent": float(self.risk_percent),
            "fixed_lot": float(self.fixed_lot),
            "risk_profile_map": self.risk_profile_map,
            "trading_schedule": self.trading_schedule,
            # MT5 Account
            **account_data,
            # MT5 Positions
            "positions": positions_data,
            "closed_deals": closed_deals,
            # Logs
            "llm_text": llm_text,
            "recent_logs": list(getattr(self, "recent_logs", [])[-40:]),
            "pending_intrabar_review_count": int(len(getattr(self, "pending_intrabar_reviews", []) or [])),
            "recent_intrabar_reviews": list(getattr(self, "recent_intrabar_reviews", [])[-20:]),
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
        now_utc = datetime.now(timezone.utc)
        now_str = now_utc.strftime("%H:%M:%S")
        entry = {
            "timestamp": now_str,
            "timestamp_utc": now_utc.isoformat().replace("+00:00", "Z"),
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

    def _parse_semantic_ts_key(self, ts_key: str) -> datetime:
        text = str(ts_key or "").strip()
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y.%m.%d %H.%M", "%Y-%m-%d %H:%M"):
            try:
                return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                continue
        raise ValueError(f"invalid semantic timestamp: {text}")

    def _build_llm_chart_payload(self, ts_key: str) -> dict:
        target_dt = self._parse_semantic_ts_key(ts_key)
        tf_secs = max(60, int(self.timeframe_seconds or 3600))
        epoch = int(target_dt.astimezone(timezone.utc).timestamp())
        aligned_epoch = epoch - (epoch % tf_secs)
        start_utc = datetime.fromtimestamp(aligned_epoch, tz=timezone.utc)
        end_utc = start_utc + timedelta(seconds=tf_secs)
        tf_m1 = getattr(mt5, "TIMEFRAME_M1", None)
        if tf_m1 is None:
            return {}

        try:
            mt5.symbol_select(SYMBOL, True)
        except Exception:
            pass

        rates = None
        try:
            rates = mt5.copy_rates_range(SYMBOL, tf_m1, start_utc, end_utc)
        except Exception:
            rates = None

        if rates is None or len(rates) == 0:
            try:
                anchor_dt = end_utc - timedelta(seconds=1)
                m1_count = max(3, int((end_utc - start_utc).total_seconds() // 60) + 3)
                rates = mt5.copy_rates_from(SYMBOL, tf_m1, anchor_dt, m1_count)
            except Exception:
                rates = None

        if rates is None or len(rates) == 0:
            return {}

        df = pd.DataFrame(rates)
        required_cols = {"time", "open", "high", "low", "close"}
        if df.empty or not required_cols.issubset(df.columns):
            return {}
        if "tick_volume" not in df.columns:
            df["tick_volume"] = 0.0

        df["time"] = pd.to_numeric(df["time"], errors="coerce").fillna(0).astype(int)
        start_ts = int(start_utc.timestamp())
        end_ts = int(end_utc.timestamp())
        df = df[(df["time"] >= start_ts) & (df["time"] < end_ts)].copy()
        if df.empty:
            return {}

        df = df.sort_values("time").drop_duplicates(subset=["time"], keep="last")
        chart_rates = [
            {
                "time": int(row.time),
                "open": float(row.open),
                "high": float(row.high),
                "low": float(row.low),
                "close": float(row.close),
                "tick_volume": float(getattr(row, "tick_volume", 0.0) or 0.0),
            }
            for row in df.itertuples(index=False)
        ]
        if len(chart_rates) == 0:
            return {}

        payload = {
            "chart_rates": chart_rates,
            "resolved_bar_time": start_utc.strftime("%Y-%m-%d %H:%M:%S"),
        }

        try:
            account = mt5.account_info()
        except Exception:
            account = None
        if account is not None:
            server_name = str(getattr(account, "server", "") or "").strip()
            if server_name:
                payload["source_server"] = server_name
            try:
                login_value = int(getattr(account, "login", 0) or 0)
                if login_value > 0:
                    payload["source_login"] = str(login_value)
            except Exception:
                login_text = str(getattr(account, "login", "") or "").strip()
                if login_text:
                    payload["source_login"] = login_text
        elif self.mt5_server:
            payload["source_server"] = str(self.mt5_server).strip()
            if self.mt5_login_id is not None:
                payload["source_login"] = str(int(self.mt5_login_id))

        return payload

    def _request_llm_semantic_from_server(self, ts_key: str, chart_payload: dict | None = None):
        if not self.vision_llm_api_url:
            raise RuntimeError("VISION_LLM_API_URL is empty")

        payload = {
            "date_time": ts_key,
            "symbol": SYMBOL.upper(),
            "timeframe": TIMEFRAME_NAME.upper(),
            "bot_config_id": BOT_CONFIG_ID,
        }
        if isinstance(chart_payload, dict) and chart_payload:
            payload.update(chart_payload)
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
        chart_payload = {}

        try:
            chart_payload = self._build_llm_chart_payload(ts_key)
        except Exception as exc:
            chart_payload = {}
            print(f"\n [SEM] chart payload unavailable ts={ts_key}: {exc}")
            self._add_log(
                "warning",
                f"Chart payload unavailable for {ts_key}",
                phase="sem",
                event="chart_payload_unavailable",
                meta={"ts": ts_key, "error": str(exc)},
            )

        def _worker():
            try:
                llm_text, cls_vec = self._request_llm_semantic_from_server(ts_key, chart_payload=chart_payload)
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

        # 1. Check local disk cache
        alias_key, alias_vec = self._resolve_cached_semantic_alias(ts_key)
        if alias_vec is not None:
            if alias_key == ts_key:
                self.semantic_runtime.global_time_to_vec[ts_key] = np.asarray(alias_vec, dtype=np.float32)
                self._hydrate_last_llm_text_from_cache(ts_key)
            else:
                self._adopt_semantic_alias(ts_key, alias_key, alias_vec)
            self._add_log(
                "analysis",
                f"Semantic cache hit for {ts_key}",
                phase="sem",
                event="cache_hit" if alias_key == ts_key else "cache_alias_hit",
                meta={
                    "ts": ts_key,
                    "source": "disk_cache" if alias_key == ts_key else "cache_alias",
                    "alias_ts": alias_key if alias_key != ts_key else None,
                },
            )
            return

        now_epoch = time.time()
        retry_not_before = float(self._sem_retry_not_before.get(ts_key, 0.0) or 0.0)
        if now_epoch < retry_not_before:
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
            self._sem_retry_not_before[ts_key] = time.time() + self._semantic_retry_delay(exc)
            print(f"\n [SEM] request failed ts={ts_key}: {exc} | fallback=blocked_real_only")
            self._add_log(
                "warning",
                f"Semantic request failed for {ts_key}",
                phase="sem",
                event="server_request_failed",
                meta={"ts": ts_key, "error": str(exc)},
            )

    def _resolve_live_llm_semantic_quick(self, ts_key: str, timeout_sec: float):
        if self.semantic_runtime is None:
            return False
        if ts_key in self.semantic_runtime.global_time_to_vec:
            return True

        alias_key, alias_vec = self._resolve_cached_semantic_alias(ts_key)
        if alias_vec is not None:
            if alias_key == ts_key:
                self.semantic_runtime.global_time_to_vec[ts_key] = np.asarray(alias_vec, dtype=np.float32)
                self._hydrate_last_llm_text_from_cache(ts_key)
            else:
                self._adopt_semantic_alias(ts_key, alias_key, alias_vec)
            return True

        quick_timeout = max(5.0, float(timeout_sec))

        prev_timeout = float(self.vision_llm_timeout_sec)
        self.vision_llm_timeout_sec = min(prev_timeout, quick_timeout)
        try:
            llm_text, cls_vec = self._request_llm_semantic_from_server(ts_key)
            self._save_llm_semantic_entry(ts_key, cls_vec, llm_text, source="startup_prewarm")
            return True
        except Exception as exc:
            self._sem_retry_not_before[ts_key] = time.time() + self._semantic_retry_delay(exc)
            self._add_log(
                "warning",
                f"Startup prewarm failed for {ts_key}",
                phase="sem",
                event="startup_prewarm_failed",
                meta={"ts": ts_key, "error": str(exc)},
            )
            return False
        finally:
            self.vision_llm_timeout_sec = prev_timeout

    def _prewarm_semantic_on_start(self):
        if not bool(LIVE_PREWARM_SEMANTIC_ON_START):
            return
        if self.semantic_runtime is None:
            return

        max_seconds = max(0.0, float(LIVE_PREWARM_SEMANTIC_MAX_SECONDS))
        if max_seconds <= 0.0:
            return

        current_bar_time = self._current_bar_time()
        if current_bar_time <= 0:
            return
        window_df = self._fetch_window(current_bar_time)
        if window_df is None:
            return

        ts_keys = pd.to_datetime(window_df["time"]).dt.strftime("%Y-%m-%d %H:%M:%S").tolist()
        missing = [ts for ts in ts_keys if ts not in self.semantic_runtime.global_time_to_vec]
        if not missing:
            self._add_log(
                "info",
                "Startup prewarm skipped: semantic cache already warm",
                phase="sem",
                event="startup_prewarm_skip_full",
            )
            return

        max_missing = max(1, int(LIVE_PREWARM_SEMANTIC_MAX_MISSING))
        target_keys = missing[-max_missing:]
        started_at = time.time()
        deadline = time.time() + max_seconds
        timeout_per_request = max(5.0, float(LIVE_PREWARM_REQUEST_TIMEOUT_SEC))
        resolved_count = 0
        attempted_count = 0

        print(
            "\n [SEM] startup prewarm "
            f"targets={len(target_keys)} missing={len(missing)} budget={max_seconds:.1f}s"
        )
        self._add_log(
            "info",
            f"Startup semantic prewarm targets={len(target_keys)}",
            phase="sem",
            event="startup_prewarm_start",
            meta={
                "missing_total": int(len(missing)),
                "targets": int(len(target_keys)),
                "budget_sec": float(max_seconds),
            },
        )

        for ts_key in target_keys:
            if time.time() >= deadline:
                break
            attempted_count += 1
            if self._resolve_live_llm_semantic_quick(ts_key, timeout_sec=timeout_per_request):
                resolved_count += 1

        elapsed = max(0.0, float(time.time() - started_at))
        pending_after = len([ts for ts in target_keys if ts not in self.semantic_runtime.global_time_to_vec])
        print(
            "\n [SEM] startup prewarm done "
            f"resolved={resolved_count}/{attempted_count} pending={pending_after} elapsed={elapsed:.1f}s"
        )
        self._add_log(
            "success" if pending_after == 0 else "info",
            f"Startup semantic prewarm resolved={resolved_count}/{attempted_count}",
            phase="sem",
            event="startup_prewarm_done",
            meta={
                "resolved": int(resolved_count),
                "attempted": int(attempted_count),
                "pending_after": int(pending_after),
                "elapsed_sec": float(round(elapsed, 2)),
            },
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
                "risk_mode": self.risk_mode,
                "risk_percent": float(self.risk_percent),
                "custom_lot": float(self.fixed_lot),
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
            "intrabar": {
                "pending_reviews": list(getattr(self, "pending_intrabar_reviews", [])[-50:]),
                "recent_reviews": list(getattr(self, "recent_intrabar_reviews", [])[-50:]),
                "trailing_state": dict(getattr(self, "intrabar_trailing_state", {}) or {}),
                "regime_snapshot": dict(getattr(self, "intrabar_regime_snapshot", {}) or {}),
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
        intrabar_state = payload.get("intrabar", {})
        if isinstance(intrabar_state, dict):
            pending_reviews = intrabar_state.get("pending_reviews", [])
            recent_reviews = intrabar_state.get("recent_reviews", [])
            trailing_state = intrabar_state.get("trailing_state", {})
            regime_snapshot = intrabar_state.get("regime_snapshot", {})
            if isinstance(pending_reviews, list):
                self.pending_intrabar_reviews = [
                    dict(row) for row in pending_reviews[-50:] if isinstance(row, dict)
                ]
            if isinstance(recent_reviews, list):
                self.recent_intrabar_reviews = [
                    dict(row) for row in recent_reviews[-50:] if isinstance(row, dict)
                ]
            if isinstance(trailing_state, dict):
                self.intrabar_trailing_state = dict(trailing_state)
            if isinstance(regime_snapshot, dict):
                self.intrabar_regime_snapshot = dict(regime_snapshot)

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

    def _is_managed_magic(self, raw_magic) -> bool:
        try:
            magic = int(raw_magic or 0)
        except Exception:
            magic = 0
        allowed = set(int(v) for v in (LIVE_MANAGED_MAGIC_SET or set()))
        if len(allowed) == 0:
            return True
        return int(magic) in allowed

    def _is_performance_magic(self, raw_magic) -> bool:
        try:
            magic = int(raw_magic or 0)
        except Exception:
            magic = 0
        allowed = set(int(v) for v in (LIVE_PERFORMANCE_MAGIC_SET or set()))
        if len(allowed) == 0:
            return True
        return int(magic) in allowed

    def _filter_managed_symbol_positions(self, positions):
        if not positions:
            return []
        out = []
        symbol_upper = str(SYMBOL or "").upper()
        for p in positions:
            if str(getattr(p, "symbol", "")).upper() != symbol_upper:
                continue
            if not self._is_managed_magic(getattr(p, "magic", 0)):
                continue
            out.append(p)
        return out

    def _get_symbol_positions_safe(self):
        retries = 2
        for attempt in range(retries + 1):
            positions = mt5.positions_get(symbol=SYMBOL)
            if positions is None:
                if attempt < retries:
                    time.sleep(0.05)
                continue

            managed = self._filter_managed_symbol_positions(list(positions))
            if len(managed) == 0 and self.last_known_ticket != 0 and attempt < retries:
                time.sleep(0.05)
                continue
            return managed

        last_err = self._safe_last_error()
        if self._is_mt5_ipc_error(last_err):
            self._try_reconnect_mt5(reason="positions_get")
            try:
                positions = mt5.positions_get(symbol=SYMBOL)
                if positions is not None:
                    return self._filter_managed_symbol_positions(list(positions))
            except Exception:
                pass
        try:
            all_positions = mt5.positions_get()
            if all_positions:
                return self._filter_managed_symbol_positions(list(all_positions))
        except Exception:
            pass
        return []

    def _collect_recent_closed_deals(self, expected_tickets=None, force_lookback_sec: int | None = None):
        now_utc = datetime.now(timezone.utc)
        history_end_utc = self._history_deals_end_utc(now_utc)
        cursor_msc = int(getattr(self, "_closed_deal_cursor_msc", 0) or 0)
        use_cursor = True
        if isinstance(force_lookback_sec, int) and force_lookback_sec > 0:
            use_cursor = False
            start_utc = now_utc - timedelta(seconds=int(force_lookback_sec))
        elif cursor_msc > 0:
            start_utc = datetime.fromtimestamp(max(0, cursor_msc - 5000) / 1000.0, tz=timezone.utc)
        else:
            start_utc = now_utc - timedelta(days=3)

        try:
            deals = mt5.history_deals_get(start_utc, history_end_utc)
        except Exception as exc:
            print(f" WS state: history_deals_get failed: {exc}")
            deals = None

        if deals is None:
            last_err = self._safe_last_error()
            if self._is_mt5_ipc_error(last_err):
                self._try_reconnect_mt5(reason="history_deals_get")
            return []

        expected = set()
        if expected_tickets:
            for raw_ticket in expected_tickets:
                try:
                    ticket = int(raw_ticket or 0)
                except Exception:
                    ticket = 0
                if ticket > 0:
                    expected.add(ticket)

        entry_out_values = set()
        for name in ("DEAL_ENTRY_OUT", "DEAL_ENTRY_OUT_BY", "DEAL_ENTRY_INOUT"):
            value = getattr(mt5, name, None)
            if value is not None:
                entry_out_values.add(int(value))

        latest_by_ticket = {}
        max_seen_msc = cursor_msc
        symbol_upper = str(SYMBOL or "").upper()

        for deal in deals:
            deal_symbol = str(getattr(deal, "symbol", "") or "").upper()
            if deal_symbol != symbol_upper:
                continue
            if not self._is_performance_magic(getattr(deal, "magic", 0)):
                continue

            entry = int(getattr(deal, "entry", -999) or -999)
            if entry_out_values and entry not in entry_out_values:
                continue

            ticket = int(getattr(deal, "position_id", 0) or 0)
            if ticket <= 0:
                ticket = int(getattr(deal, "order", 0) or 0)
            if ticket <= 0:
                ticket = int(getattr(deal, "ticket", 0) or 0)
            if ticket <= 0:
                continue

            if expected and ticket not in expected:
                continue

            close_time_msc = int(getattr(deal, "time_msc", 0) or 0)
            if close_time_msc <= 0:
                close_time_msc = int(getattr(deal, "time", 0) or 0) * 1000
            if close_time_msc <= 0:
                continue
            if use_cursor and cursor_msc > 0 and close_time_msc < cursor_msc:
                continue

            close_dt = datetime.fromtimestamp(close_time_msc / 1000.0, tz=timezone.utc)
            payload = {
                "ticket": ticket,
                "symbol": deal_symbol or symbol_upper,
                "magic": int(getattr(deal, "magic", 0) or 0),
                "volume": float(getattr(deal, "volume", 0.0) or 0.0),
                "closePrice": float(getattr(deal, "price", 0.0) or 0.0),
                "profit": float(getattr(deal, "profit", 0.0) or 0.0),
                "swap": float(getattr(deal, "swap", 0.0) or 0.0),
                "commission": float(getattr(deal, "commission", 0.0) or 0.0),
                "closeTime": close_dt.isoformat(),
                "closeTimeMsc": close_time_msc,
            }

            previous = latest_by_ticket.get(ticket)
            if not previous or int(previous.get("closeTimeMsc", 0) or 0) <= close_time_msc:
                latest_by_ticket[ticket] = payload

            if close_time_msc > max_seen_msc:
                max_seen_msc = close_time_msc

        if use_cursor and max_seen_msc > cursor_msc:
            self._closed_deal_cursor_msc = max_seen_msc + 1

        rows = sorted(
            list(latest_by_ticket.values()),
            key=lambda row: int(row.get("closeTimeMsc", 0) or 0),
            reverse=True,
        )
        if len(rows) > 200:
            rows = rows[:200]
        return rows

    def _deal_entry_out_values(self):
        values = set()
        for name in ("DEAL_ENTRY_OUT", "DEAL_ENTRY_OUT_BY", "DEAL_ENTRY_INOUT"):
            value = getattr(mt5, name, None)
            if value is not None:
                values.add(int(value))
        return values

    def _resolve_deal_ticket(self, deal) -> int:
        ticket = int(getattr(deal, "position_id", 0) or 0)
        if ticket <= 0:
            ticket = int(getattr(deal, "order", 0) or 0)
        if ticket <= 0:
            ticket = int(getattr(deal, "ticket", 0) or 0)
        return int(ticket)

    def _history_deals_end_utc(self, now_utc: datetime | None = None) -> datetime:
        end_utc = now_utc if isinstance(now_utc, datetime) else datetime.now(timezone.utc)
        if end_utc.tzinfo is None:
            end_utc = end_utc.replace(tzinfo=timezone.utc)
        ahead_hours = max(0.0, float(getattr(self, "mt5_history_end_ahead_hours", 0.0) or 0.0))
        if ahead_hours <= 0.0:
            return end_utc
        return end_utc + timedelta(hours=ahead_hours)

    def _fetch_managed_closed_deal_events(self, start_utc: datetime, end_utc: datetime):
        try:
            deals = mt5.history_deals_get(start_utc, end_utc)
        except Exception as exc:
            print(f" Performance sync: history_deals_get failed: {exc}")
            deals = None

        if deals is None:
            last_err = self._safe_last_error()
            if self._is_mt5_ipc_error(last_err):
                self._try_reconnect_mt5(reason="history_perf_sync")
            return None

        entry_out_values = self._deal_entry_out_values()
        symbol_upper = str(SYMBOL or "").upper()
        scope = str(LIVE_PERFORMANCE_SCOPE or "symbol").strip().lower()
        include_all_symbols = scope == "account"
        events = []

        for deal in deals:
            if not include_all_symbols and str(getattr(deal, "symbol", "") or "").upper() != symbol_upper:
                continue
            if not self._is_performance_magic(getattr(deal, "magic", 0)):
                continue

            entry = int(getattr(deal, "entry", -999) or -999)
            if entry_out_values and entry not in entry_out_values:
                continue

            deal_id = int(getattr(deal, "ticket", 0) or 0)
            if deal_id <= 0:
                continue

            ticket = self._resolve_deal_ticket(deal)
            if ticket <= 0:
                continue

            close_time_msc = int(getattr(deal, "time_msc", 0) or 0)
            if close_time_msc <= 0:
                close_time_msc = int(getattr(deal, "time", 0) or 0) * 1000
            if close_time_msc <= 0:
                continue

            net_pnl = (
                float(getattr(deal, "profit", 0.0) or 0.0)
                + float(getattr(deal, "commission", 0.0) or 0.0)
                + float(getattr(deal, "swap", 0.0) or 0.0)
            )
            events.append(
                {
                    "deal_id": int(deal_id),
                    "ticket": int(ticket),
                    "close_time_msc": int(close_time_msc),
                    "net_pnl": float(net_pnl),
                }
            )

        return events

    def _apply_mt5_performance_stats(self):
        if self.bridge is None:
            return
        trade_nets = [float(value) for value in dict(self._perf_ticket_net or {}).values()]
        self.bridge.trades = int(len(trade_nets))
        self.bridge.wins = int(sum(1 for value in trade_nets if value > 1e-9))
        self.bridge.total_pnl = float(sum(trade_nets))

    def _sync_performance_from_mt5_history(
        self,
        force: bool = False,
        full_resync: bool = False,
        reason: str = "periodic",
    ) -> bool:
        if self.bridge is None:
            return False

        now_epoch = time.time()
        interval_sec = max(5.0, float(LIVE_PERFORMANCE_SYNC_INTERVAL_SEC))
        if (
            not force
            and float(getattr(self, "_perf_last_sync_at", 0.0) or 0.0) > 0.0
            and (now_epoch - float(self._perf_last_sync_at)) < interval_sec
        ):
            return False

        prev_stats = (
            int(getattr(self.bridge, "trades", 0) or 0),
            int(getattr(self.bridge, "wins", 0) or 0),
            float(getattr(self.bridge, "total_pnl", 0.0) or 0.0),
        )

        now_utc = datetime.now(timezone.utc)
        history_end_utc = self._history_deals_end_utc(now_utc)
        mode = "incremental"
        updates = 0

        if bool(full_resync) or not bool(getattr(self, "_perf_seeded", False)):
            mode = "full"
            lookback_days = max(30, int(LIVE_PERFORMANCE_BOOT_LOOKBACK_DAYS))
            start_utc = now_utc - timedelta(days=int(lookback_days))
            events = self._fetch_managed_closed_deal_events(start_utc, history_end_utc)
            self._perf_last_sync_at = now_epoch
            if events is None:
                return False

            events = sorted(events, key=lambda row: int(row.get("close_time_msc", 0) or 0))
            next_seen = set()
            next_ticket_net = {}
            max_seen_msc = 0
            for row in events:
                deal_id = int(row.get("deal_id", 0) or 0)
                ticket = int(row.get("ticket", 0) or 0)
                close_time_msc = int(row.get("close_time_msc", 0) or 0)
                net_pnl = float(row.get("net_pnl", 0.0) or 0.0)
                if deal_id <= 0 or ticket <= 0:
                    continue
                next_seen.add(deal_id)
                next_ticket_net[ticket] = float(next_ticket_net.get(ticket, 0.0) or 0.0) + net_pnl
                if close_time_msc > max_seen_msc:
                    max_seen_msc = close_time_msc
                updates += 1

            self._perf_deal_seen = next_seen
            self._perf_ticket_net = next_ticket_net
            self._perf_cursor_msc = int(max_seen_msc + 1) if max_seen_msc > 0 else int(now_utc.timestamp() * 1000) + 1
            self._perf_seeded = True
        else:
            cursor_msc = int(getattr(self, "_perf_cursor_msc", 0) or 0)
            if cursor_msc > 0:
                start_utc = datetime.fromtimestamp(max(0, cursor_msc - 5000) / 1000.0, tz=timezone.utc)
            else:
                start_utc = now_utc - timedelta(days=1)
            events = self._fetch_managed_closed_deal_events(start_utc, history_end_utc)
            self._perf_last_sync_at = now_epoch
            if events is None:
                return False

            events = sorted(events, key=lambda row: int(row.get("close_time_msc", 0) or 0))
            max_seen_msc = int(cursor_msc)
            for row in events:
                deal_id = int(row.get("deal_id", 0) or 0)
                ticket = int(row.get("ticket", 0) or 0)
                close_time_msc = int(row.get("close_time_msc", 0) or 0)
                net_pnl = float(row.get("net_pnl", 0.0) or 0.0)
                if deal_id <= 0 or ticket <= 0:
                    continue
                if deal_id in self._perf_deal_seen:
                    continue
                self._perf_deal_seen.add(deal_id)
                self._perf_ticket_net[ticket] = float(self._perf_ticket_net.get(ticket, 0.0) or 0.0) + net_pnl
                if close_time_msc > max_seen_msc:
                    max_seen_msc = close_time_msc
                updates += 1

            if max_seen_msc > cursor_msc:
                self._perf_cursor_msc = int(max_seen_msc + 1)

        self._apply_mt5_performance_stats()
        next_stats = (
            int(getattr(self.bridge, "trades", 0) or 0),
            int(getattr(self.bridge, "wins", 0) or 0),
            float(getattr(self.bridge, "total_pnl", 0.0) or 0.0),
        )
        changed = bool(prev_stats != next_stats)
        if full_resync or changed or updates > 0:
            self._add_log(
                "info",
                f"Performance sync ({mode}) trades={next_stats[0]} wins={next_stats[1]} pnl={next_stats[2]:+.2f}",
                phase="mt5",
                event="performance_sync",
                meta={
                    "mode": mode,
                    "reason": str(reason),
                    "scope": str(LIVE_PERFORMANCE_SCOPE),
                    "magic_set": ",".join(str(int(v)) for v in sorted(set(LIVE_PERFORMANCE_MAGIC_SET or set()))),
                    "updates": int(updates),
                    "tickets": int(len(self._perf_ticket_net)),
                    "deals_seen": int(len(self._perf_deal_seen)),
                },
            )
        return changed or updates > 0

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
        pos_tickets = []
        order_tickets = []
        foreign_order_count = 0

        try:
            positions = list(self._get_symbol_positions_safe() or [])
        except Exception:
            positions = []
        active_positions = []
        for p in positions:
            vol = float(getattr(p, "volume", 0.0) or 0.0)
            if vol <= 0.0:
                continue
            active_positions.append(p)
            ticket = int(getattr(p, "ticket", 0) or 0)
            if ticket > 0:
                pos_tickets.append(ticket)
        pos_count = int(len(active_positions))

        pending_types = set()
        for name in (
            "ORDER_TYPE_BUY_LIMIT",
            "ORDER_TYPE_SELL_LIMIT",
            "ORDER_TYPE_BUY_STOP",
            "ORDER_TYPE_SELL_STOP",
            "ORDER_TYPE_BUY_STOP_LIMIT",
            "ORDER_TYPE_SELL_STOP_LIMIT",
        ):
            val = getattr(mt5, name, None)
            if val is not None:
                pending_types.add(int(val))

        inactive_states = set()
        for name in (
            "ORDER_STATE_CANCELED",
            "ORDER_STATE_REJECTED",
            "ORDER_STATE_EXPIRED",
            "ORDER_STATE_FILLED",
        ):
            val = getattr(mt5, name, None)
            if val is not None:
                inactive_states.add(int(val))

        try:
            orders = mt5.orders_get(symbol=SYMBOL)
        except Exception:
            orders = None

        if orders:
            for o in orders:
                try:
                    o_type = int(getattr(o, "type", -1) or -1)
                except Exception:
                    o_type = -1
                if pending_types and o_type not in pending_types:
                    continue

                try:
                    o_state = int(getattr(o, "state", -1) or -1)
                except Exception:
                    o_state = -1
                if inactive_states and o_state in inactive_states:
                    continue

                ticket = int(getattr(o, "ticket", 0) or 0)
                magic = int(getattr(o, "magic", 0) or 0)
                if not self._is_managed_magic(magic):
                    foreign_order_count += 1
                    continue
                if ticket > 0:
                    order_tickets.append(ticket)

        order_count = int(len(order_tickets))
        has_exposure = pos_count > 0 or order_count > 0
        return has_exposure, pos_count, order_count, {
            "pos_tickets": pos_tickets[:8],
            "order_tickets": order_tickets[:8],
            "foreign_order_count": int(foreign_order_count),
        }

    def _sync_bridge_from_mt5(self):
        if self.bridge is None:
            return

        prev_pos = int(self.bridge.position)
        current_pos, pos = self._get_mt5_position()
        current_ticket = int(pos.ticket) if pos is not None else 0
        prev_ticket = int(self.last_known_ticket)
        cleared_trailing = False

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
            self.bridge.max_equity = max(float(self.bridge.max_equity), float(self.bridge.equity))
        else:
            self.bridge.max_equity = max(float(self.bridge.max_equity), float(self.bridge.equity))

        if SYNC_EXTERNAL_LOT and pos is not None and float(pos.volume) > 0:
            self.bridge.lot_size = float(pos.volume)
        elif self._is_custom_lot_mode():
            self.bridge.lot_size = self._resolve_runtime_lot()
        elif self.dynamic_lot:
            if account is not None:
                self.bridge.lot_size = self._resolve_runtime_lot(float(account.balance))
            else:
                self.bridge.lot_size = max(0.01, float(self.bridge.lot_size))
        else:
            self.bridge.lot_size = max(0.01, float(self.bridge.lot_size))
        self.bridge.spread_cost = SPREAD_PIPS * PIP_VALUE * self.bridge.lot_size

        if current_pos == 0:
            cleared_trailing = self._clear_intrabar_trailing_state() or cleared_trailing
        else:
            trailing_state = dict(getattr(self, "intrabar_trailing_state", {}) or {})
            trailing_ticket = int(trailing_state.get("ticket", 0) or 0)
            trailing_side = str(trailing_state.get("side", "") or "").upper()
            actual_side = "LONG" if current_pos == 1 else "SHORT"
            ticket_changed = prev_ticket != 0 and prev_ticket != current_ticket
            side_changed = trailing_side and trailing_side != actual_side
            if ticket_changed or (trailing_ticket > 0 and trailing_ticket != current_ticket) or side_changed:
                cleared_trailing = self._clear_intrabar_trailing_state() or cleared_trailing

        self.current_lot = self.bridge.lot_size
        self.last_known_ticket = current_ticket
        if cleared_trailing:
            self._save_runtime_state(reason="intrabar_trail_sync_reset")
        self._sync_performance_from_mt5_history(force=False, full_resync=False, reason="bridge_heartbeat")

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
        raw_positions = mt5.positions_get(symbol=SYMBOL)
        if raw_positions is None:
            last_err = self._safe_last_error()
            if self._is_mt5_ipc_error(last_err):
                self._try_reconnect_mt5(reason="close_all:positions_get")
                raw_positions = mt5.positions_get(symbol=SYMBOL)
        if raw_positions is None:
            self._add_log(
                "warning",
                "Close skipped: positions_get returned None",
                phase="order",
                event="close_skipped",
            )
            return False

        positions = self._filter_managed_symbol_positions(list(raw_positions))
        if len(positions) == 0:
            self._add_log(
                "info",
                "Close requested: no open positions",
                phase="order",
                event="close_no_positions",
            )
            return True

        if not self._ensure_trade_allowed_before_order(reason="close_all", action_label="Close order"):
            return False

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

    def _clear_intrabar_trailing_state(self) -> bool:
        state = dict(getattr(self, "intrabar_trailing_state", {}) or {})
        if not state:
            return False
        self.intrabar_trailing_state = {}
        return True

    def _update_intrabar_regime_snapshot(self, bar_end_ts: int, decision: dict | None = None) -> None:
        decision = decision if isinstance(decision, dict) else {}
        bar_context = decision.get("bar_context", {})
        snapshot = {
            "bar_end_ts": int(bar_end_ts),
            "bar_end_utc": datetime.fromtimestamp(int(bar_end_ts), tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ"),
            "semantic_quality": float(decision.get("semantic_quality", 0.0) or 0.0),
            "defensive_mode_bars": int(getattr(getattr(self, "bridge", None), "defensive_mode_bars", 0) or 0),
            "atr_norm": 0.0,
            "trend": 0.0,
            "adx": 0.0,
            "sma_cross": 0.0,
        }
        if isinstance(bar_context, dict):
            for key in ("atr_norm", "trend", "adx", "sma_cross"):
                snapshot[key] = float(bar_context.get(key, 0.0) or 0.0)
        self.intrabar_regime_snapshot = snapshot

    def _intrabar_threshold_targets(self) -> dict[str, float]:
        targets: dict[str, float] = {}
        if self.intrabar_take_profit_change_pct > 0.0:
            targets["change_pct"] = float(self.intrabar_take_profit_change_pct)
        if self.intrabar_take_profit_pips > 0.0:
            targets["pnl_pips"] = float(self.intrabar_take_profit_pips)
        if self.intrabar_take_profit_money > 0.0:
            targets["pnl_money"] = float(self.intrabar_take_profit_money)
        return targets

    def _intrabar_trail_buffer_value(self, arm_value: float) -> float:
        arm_value = float(arm_value or 0.0)
        if arm_value <= 0.0:
            return 0.0
        return float(abs(arm_value) * max(0.0, float(self.intrabar_trail_arm_buffer_ratio)))

    def _compute_intrabar_floor_state(
        self,
        arm_value: float,
        peak_value: float,
        keep_ratio: float,
    ) -> tuple[float, float, float, bool]:
        arm_value = float(arm_value or 0.0)
        peak_value = float(peak_value or 0.0)
        keep_ratio = float(np.clip(keep_ratio, 0.0, 1.0))
        if arm_value <= 0.0:
            return 0.0, 0.0, 0.0, False

        buffer_value = self._intrabar_trail_buffer_value(arm_value)
        initial_floor = max(0.0, arm_value - buffer_value)
        activation_peak = arm_value + buffer_value
        if peak_value < activation_peak:
            return float(initial_floor), float(initial_floor), float(activation_peak), False

        dynamic_floor = arm_value + max(0.0, peak_value - arm_value) * keep_ratio
        floor_value = max(initial_floor, dynamic_floor)
        return float(floor_value), float(initial_floor), float(activation_peak), True

    def _compute_intrabar_floor(self, arm_value: float, peak_value: float, keep_ratio: float) -> float:
        floor_value, _, _, _ = self._compute_intrabar_floor_state(arm_value, peak_value, keep_ratio)
        return float(floor_value)

    def _intrabar_regime(self, current_pos: int) -> tuple[str, float, dict]:
        snapshot = dict(getattr(self, "intrabar_regime_snapshot", {}) or {})
        gate_stats = dict(getattr(getattr(self, "bridge", None), "gate_stats", {}) or {})

        atr_norm = float(snapshot.get("atr_norm", 0.0) or 0.0)
        trend = float(snapshot.get("trend", 0.0) or 0.0)
        adx = float(snapshot.get("adx", 0.0) or 0.0)
        sma_cross = float(snapshot.get("sma_cross", 0.0) or 0.0)
        semantic_quality = float(snapshot.get("semantic_quality", 0.0) or 0.0)
        defensive_mode_bars = int(snapshot.get("defensive_mode_bars", 0) or 0)

        direction = 1 if int(current_pos) >= 0 else -1
        abs_trend = abs(trend)
        trend_flat = float(gate_stats.get("trend_flat", 0.08) or 0.08)
        trend_strong = float(gate_stats.get("trend_strong", 0.25) or 0.25)
        adx_flat = float(gate_stats.get("adx_flat", -0.20) or -0.20)
        adx_strong = float(gate_stats.get("adx_strong", 0.20) or 0.20)
        atr_extreme = float(gate_stats.get("atr_extreme", 0.0025) or 0.0025)

        aligned_trend = abs_trend >= trend_strong and adx >= adx_strong and np.sign(trend) == direction
        flat_market = abs_trend <= trend_flat and adx <= adx_flat
        counter_trend = (
            (direction == 1 and (trend < -trend_flat or sma_cross < 0.0))
            or (direction == -1 and (trend > trend_flat or sma_cross > 0.0))
        )
        low_semantic = semantic_quality < max(float(EMBED_QUALITY_MIN) + 0.05, 0.35)

        if defensive_mode_bars > 0 or low_semantic or flat_market or counter_trend or atr_norm >= atr_extreme:
            regime = "tight"
            keep_ratio = float(self.intrabar_trail_keep_ratio_tight)
        elif aligned_trend:
            regime = "trend"
            keep_ratio = float(self.intrabar_trail_keep_ratio_trend)
        else:
            regime = "normal"
            keep_ratio = float(self.intrabar_trail_keep_ratio_normal)

        return regime, keep_ratio, {
            "atr_norm": float(atr_norm),
            "trend": float(trend),
            "adx": float(adx),
            "sma_cross": float(sma_cross),
            "semantic_quality": float(semantic_quality),
            "defensive_mode_bars": int(defensive_mode_bars),
            "aligned_trend": bool(aligned_trend),
            "flat_market": bool(flat_market),
            "counter_trend": bool(counter_trend),
        }

    def _build_intrabar_metrics(self, current_pos: int, pos, tick_data: dict | None) -> dict | None:
        if current_pos == 0 or pos is None or not isinstance(tick_data, dict):
            return None

        entry_price = float(getattr(pos, "price_open", 0.0) or 0.0)
        if entry_price <= 0.0:
            return None

        exit_price = float(tick_data.get("bid", 0.0) if current_pos == 1 else tick_data.get("ask", 0.0))
        if exit_price <= 0.0:
            return None

        change_pct = float(current_pos * (exit_price - entry_price) / max(entry_price, 1e-10) * 100.0)
        pnl_pips = float(current_pos * (exit_price - entry_price) / max(self.pip_size, 1e-10))
        pnl_money = float(getattr(pos, "profit", 0.0) or 0.0)
        return {
            "ticket": int(getattr(pos, "ticket", 0) or 0),
            "side": "LONG" if current_pos == 1 else "SHORT",
            "entry_price": float(entry_price),
            "exit_price": float(exit_price),
            "change_pct": float(change_pct),
            "pnl_pips": float(pnl_pips),
            "pnl_money": float(pnl_money),
            "volume": float(getattr(pos, "volume", 0.0) or 0.0),
        }

    def _intrabar_threshold_hit_reasons(self, metrics: dict, targets: dict[str, float]) -> list[str]:
        reasons: list[str] = []
        change_pct = float(metrics.get("change_pct", 0.0) or 0.0)
        pnl_pips = float(metrics.get("pnl_pips", 0.0) or 0.0)
        pnl_money = float(metrics.get("pnl_money", 0.0) or 0.0)
        if float(targets.get("change_pct", 0.0) or 0.0) > 0.0 and change_pct >= float(targets["change_pct"]):
            reasons.append(f"change {change_pct:.3f}%>={float(targets['change_pct']):.3f}%")
        if float(targets.get("pnl_pips", 0.0) or 0.0) > 0.0 and pnl_pips >= float(targets["pnl_pips"]):
            reasons.append(f"pips {pnl_pips:.1f}>={float(targets['pnl_pips']):.1f}")
        if float(targets.get("pnl_money", 0.0) or 0.0) > 0.0 and pnl_money >= float(targets["pnl_money"]):
            reasons.append(f"money {pnl_money:.2f}>={float(targets['pnl_money']):.2f}")
        return reasons

    def _resolve_bar_close_exit_price(
        self,
        *,
        bar_end_ts: int,
        side: str,
        fallback_close_price: float,
    ) -> tuple[float, str, int]:
        side_upper = str(side or "").upper()
        close_price = float(fallback_close_price or 0.0)
        tick_price_key = "bid" if side_upper == "LONG" else "ask"

        try:
            mt5.symbol_select(SYMBOL, True)
        except Exception:
            pass

        ticks = None
        try:
            end_dt = datetime.fromtimestamp(int(bar_end_ts), tz=timezone.utc)
            start_dt = end_dt - timedelta(seconds=5)
            scan_end_dt = end_dt + timedelta(seconds=2)
            ticks = mt5.copy_ticks_range(SYMBOL, start_dt, scan_end_dt, mt5.COPY_TICKS_ALL)
        except Exception:
            ticks = None

        if ticks is not None and len(ticks) > 0:
            try:
                tdf = pd.DataFrame(ticks)
                if "time_msc" in tdf.columns:
                    tdf["tick_msc"] = pd.to_numeric(tdf["time_msc"], errors="coerce").fillna(0).astype("int64")
                elif "time" in tdf.columns:
                    tdf["tick_msc"] = (pd.to_numeric(tdf["time"], errors="coerce").fillna(0) * 1000).astype("int64")
                else:
                    tdf["tick_msc"] = 0
                tdf = tdf.sort_values("tick_msc").reset_index(drop=True)
                cutoff_msc = int(bar_end_ts) * 1000
                usable = tdf[pd.to_numeric(tdf.get(tick_price_key), errors="coerce").fillna(0.0) > 0.0].copy()
                if len(usable) > 0:
                    at_or_before = usable[usable["tick_msc"] <= cutoff_msc]
                    if len(at_or_before) > 0:
                        row = at_or_before.iloc[-1]
                        return float(row[tick_price_key]), f"tick_{tick_price_key}_at_or_before_close", int(row["tick_msc"])
                    after = usable[usable["tick_msc"] > cutoff_msc]
                    if len(after) > 0:
                        row = after.iloc[0]
                        return float(row[tick_price_key]), f"tick_{tick_price_key}_after_close", int(row["tick_msc"])
            except Exception:
                pass

        if close_price > 0.0:
            if side_upper == "SHORT":
                return float(close_price + (self.pip_size * SPREAD_PIPS)), "bar_close_plus_spread_estimate", 0
            return float(close_price), "bar_close_bid_fallback", 0

        return 0.0, "unavailable", 0

    def _queue_intrabar_review(
        self,
        *,
        pos,
        current_bar_time: int,
        exit_time_ts: int,
        side: str,
        entry_price: float,
        exit_price: float,
        change_pct: float,
        pnl_pips: float,
        pnl_money: float,
        trigger_reasons: list[str],
        exit_mode: str = "hard_take_profit",
        exit_meta: dict | None = None,
    ) -> None:
        review_bar_end_ts = int(current_bar_time + self.timeframe_seconds)
        event_ts = int(exit_time_ts or time.time())
        record = {
            "review_id": f"{int(getattr(pos, 'ticket', 0) or 0)}:{review_bar_end_ts}:{event_ts}",
            "ticket": int(getattr(pos, "ticket", 0) or 0),
            "side": str(side),
            "volume": float(getattr(pos, "volume", 0.0) or 0.0),
            "bar_open_ts": int(current_bar_time),
            "bar_open_utc": datetime.fromtimestamp(current_bar_time, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ"),
            "bar_end_ts": int(review_bar_end_ts),
            "bar_end_utc": datetime.fromtimestamp(review_bar_end_ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ"),
            "triggered_at_ts": int(event_ts),
            "triggered_at_utc": datetime.fromtimestamp(event_ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ"),
            "entry_price": float(entry_price),
            "actual_exit_price": float(exit_price),
            "actual_change_pct": float(change_pct),
            "actual_pnl_pips": float(pnl_pips),
            "actual_pnl_money": float(pnl_money),
            "exit_mode": str(exit_mode or "hard_take_profit"),
            "trigger_reasons": [str(x) for x in list(trigger_reasons or []) if str(x).strip()],
        }
        clean_exit_meta = self._sanitize_log_meta(exit_meta)
        if clean_exit_meta:
            record["exit_meta"] = dict(clean_exit_meta)
        pending = list(getattr(self, "pending_intrabar_reviews", []) or [])
        pending.append(record)
        self.pending_intrabar_reviews = pending[-50:]
        self._add_log(
            "info",
            "Intrabar exit queued for candle-close review",
            phase="intrabar",
            event="review_pending",
            meta={
                "ticket": int(getattr(pos, "ticket", 0) or 0),
                "side": str(side),
                "bar_end_utc": str(record.get("bar_end_utc", "")),
                "triggered_at_utc": str(record.get("triggered_at_utc", "")),
                "actual_change_pct": float(change_pct),
                "actual_pnl_pips": float(pnl_pips),
                "actual_pnl_money": float(pnl_money),
                "exit_mode": str(record.get("exit_mode", "")),
                "reasons": " | ".join(record["trigger_reasons"]),
            },
        )
        self._save_runtime_state(reason="intrabar_review_pending")

    def _review_intrabar_exits_at_bar_close(self, bar_end_ts: int, window_df: pd.DataFrame) -> None:
        pending = list(getattr(self, "pending_intrabar_reviews", []) or [])
        if not pending or window_df is None or len(window_df) <= 0:
            return

        try:
            bar_close_price = float(window_df.iloc[-1]["close"])
        except Exception:
            return
        if bar_close_price <= 0.0:
            return

        still_pending = []
        completed = list(getattr(self, "recent_intrabar_reviews", []) or [])
        bar_end_utc = datetime.fromtimestamp(int(bar_end_ts), tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")

        for row in pending:
            if not isinstance(row, dict):
                continue
            target_bar_end_ts = int(row.get("bar_end_ts", 0) or 0)
            if target_bar_end_ts <= 0:
                continue
            if target_bar_end_ts > int(bar_end_ts):
                still_pending.append(dict(row))
                continue

            side = str(row.get("side", "")).upper()
            direction = 1.0 if side == "LONG" else -1.0
            entry_price = float(row.get("entry_price", 0.0) or 0.0)
            if entry_price <= 0.0:
                continue

            volume = max(0.0, float(row.get("volume", 0.0) or 0.0))
            hold_exit_price, hold_exit_source, hold_exit_tick_msc = self._resolve_bar_close_exit_price(
                bar_end_ts=target_bar_end_ts,
                side=side,
                fallback_close_price=bar_close_price,
            )
            if hold_exit_price <= 0.0:
                continue
            hold_change_pct = float(
                direction * (hold_exit_price - entry_price) / max(entry_price, 1e-10) * 100.0
            )
            hold_pnl_pips = float(direction * (hold_exit_price - entry_price) / max(self.pip_size, 1e-10))
            hold_pnl_money = float(hold_pnl_pips * PIP_VALUE * volume)

            actual_change_pct = float(row.get("actual_change_pct", 0.0) or 0.0)
            actual_pnl_pips = float(row.get("actual_pnl_pips", 0.0) or 0.0)
            actual_pnl_money = float(row.get("actual_pnl_money", 0.0) or 0.0)

            delta_change_pct = float(hold_change_pct - actual_change_pct)
            delta_pips = float(hold_pnl_pips - actual_pnl_pips)
            delta_money = float(hold_pnl_money - actual_pnl_money)

            if delta_money > 1e-9:
                outcome = "hold_better"
            elif delta_money < -1e-9:
                outcome = "intrabar_better"
            else:
                outcome = "flat"

            reviewed = dict(row)
            reviewed.update(
                {
                    "reviewed_at_bar_end_ts": int(bar_end_ts),
                    "reviewed_at_bar_end_utc": bar_end_utc,
                    "review_late": bool(target_bar_end_ts < int(bar_end_ts)),
                    "bar_close_price": float(bar_close_price),
                    "bar_close_exit_price": float(hold_exit_price),
                    "bar_close_exit_source": str(hold_exit_source),
                    "bar_close_exit_tick_msc": int(hold_exit_tick_msc),
                    "hold_to_close_change_pct": float(hold_change_pct),
                    "hold_to_close_pnl_pips": float(hold_pnl_pips),
                    "hold_to_close_pnl_money": float(hold_pnl_money),
                    "delta_vs_hold_change_pct": float(delta_change_pct),
                    "delta_vs_hold_pips": float(delta_pips),
                    "delta_vs_hold_money": float(delta_money),
                    "review_outcome": str(outcome),
                }
            )
            completed.append(reviewed)

            print(
                "\n [INTRABAR][REVIEW] "
                f"ticket={int(reviewed.get('ticket', 0) or 0)} "
                f"source={hold_exit_source} "
                f"actual={actual_change_pct:+.3f}%/{actual_pnl_money:+.2f} "
                f"bar_close={hold_change_pct:+.3f}%/{hold_pnl_money:+.2f} "
                f"delta={delta_money:+.2f} outcome={outcome}"
            )
            self._add_log(
                "analysis",
                f"Intrabar review {outcome}: actual={actual_pnl_money:+.2f} vs close={hold_pnl_money:+.2f}",
                phase="intrabar",
                event="review",
                meta={
                    "ticket": int(reviewed.get("ticket", 0) or 0),
                    "bar_close_exit_source": str(hold_exit_source),
                    "bar_close_exit_tick_msc": int(hold_exit_tick_msc),
                    "actual_change_pct": float(actual_change_pct),
                    "actual_pnl_money": float(actual_pnl_money),
                    "hold_to_close_change_pct": float(hold_change_pct),
                    "hold_to_close_pnl_money": float(hold_pnl_money),
                    "delta_vs_hold_money": float(delta_money),
                    "delta_vs_hold_pips": float(delta_pips),
                    "outcome": str(outcome),
                    "review_late": bool(reviewed.get("review_late", False)),
                },
            )

        self.pending_intrabar_reviews = still_pending[-50:]
        self.recent_intrabar_reviews = completed[-50:]

    def _maybe_take_profit_intrabar(self, current_bar_time: int) -> bool:
        if current_bar_time <= 0 or current_bar_time != self.last_bar_time:
            return False
        targets = self._intrabar_threshold_targets()
        if len(targets) == 0:
            self._clear_intrabar_trailing_state()
            return False

        current_pos, pos = self._get_mt5_position()
        if current_pos == 0 or pos is None:
            self._clear_intrabar_trailing_state()
            return False

        tick_data, stale_tick = self._get_status_tick()
        if tick_data is None:
            return False

        tick_time_ts = int(tick_data.get("time", 0) or 0)
        now_epoch = int(time.time())
        stale_age_sec = max(0, now_epoch - tick_time_ts) if tick_time_ts > 0 else 999999
        max_stale_sec = max(5, int(round(float(POLL_SECONDS) * 3.0)))
        if stale_tick and stale_age_sec > max_stale_sec:
            return False

        metrics = self._build_intrabar_metrics(current_pos, pos, tick_data)
        if metrics is None:
            return False

        ticket = int(metrics.get("ticket", 0) or 0)
        trailing_state = dict(getattr(self, "intrabar_trailing_state", {}) or {})
        if trailing_state and int(trailing_state.get("ticket", 0) or 0) != ticket:
            self._clear_intrabar_trailing_state()
            trailing_state = {}

        hit_reasons = self._intrabar_threshold_hit_reasons(metrics, targets)
        if not hit_reasons:
            if not self.intrabar_trailing_enabled:
                self._clear_intrabar_trailing_state()
                return False
        if not self.intrabar_trailing_enabled:
            if not hit_reasons:
                return False

            side = str(metrics.get("side", ""))
            print(
                "\n [INTRABAR] take-profit hit -> close "
                f"| side={side} | price={float(metrics['exit_price']):.{self.digits}f} "
                f"| change={float(metrics['change_pct']):+.3f}% | pips={float(metrics['pnl_pips']):+.1f} "
                f"| pnl={float(metrics['pnl_money']):+.2f} | trigger={' ; '.join(hit_reasons)}"
            )
            self._add_log(
                "action",
                "Intrabar take-profit hit -> closing position",
                phase="intrabar",
                event="take_profit_hit",
                meta={
                    "side": side,
                    "ticket": int(ticket),
                    "price": float(metrics["exit_price"]),
                    "change_pct": float(metrics["change_pct"]),
                    "pnl_pips": float(metrics["pnl_pips"]),
                    "pnl_money": float(metrics["pnl_money"]),
                    "reasons": " | ".join(hit_reasons),
                },
            )
            if not self.close_all():
                return False

            self._queue_intrabar_review(
                pos=pos,
                current_bar_time=int(current_bar_time),
                exit_time_ts=int(tick_time_ts or now_epoch),
                side=side,
                entry_price=float(metrics["entry_price"]),
                exit_price=float(metrics["exit_price"]),
                change_pct=float(metrics["change_pct"]),
                pnl_pips=float(metrics["pnl_pips"]),
                pnl_money=float(metrics["pnl_money"]),
                trigger_reasons=hit_reasons,
                exit_mode="hard_take_profit",
            )
            self.last_action = "CLOSE"
            return True

        regime, keep_ratio, regime_meta = self._intrabar_regime(current_pos)
        side = str(metrics.get("side", ""))
        state_changed = False

        if not trailing_state:
            if not hit_reasons:
                return False
            trailing_state = {
                "ticket": int(ticket),
                "side": side,
                "bar_open_ts": int(current_bar_time),
                "bar_open_utc": datetime.fromtimestamp(int(current_bar_time), tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ"),
                "armed_at_ts": int(tick_time_ts or now_epoch),
                "armed_at_utc": datetime.fromtimestamp(int(tick_time_ts or now_epoch), tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ"),
                "regime": str(regime),
                "keep_ratio": float(keep_ratio),
                "arm_targets": {key: float(val) for key, val in targets.items()},
                "peak_change_pct": float(metrics["change_pct"]),
                "peak_pnl_pips": float(metrics["pnl_pips"]),
                "peak_pnl_money": float(metrics["pnl_money"]),
                "last_change_pct": float(metrics["change_pct"]),
                "last_pnl_pips": float(metrics["pnl_pips"]),
                "last_pnl_money": float(metrics["pnl_money"]),
                "armed_reasons": [str(x) for x in hit_reasons],
                "regime_meta": dict(regime_meta),
                "floor_breach_count": 0,
                "floor_confirm_polls": int(self.intrabar_trail_confirm_polls),
            }
            for metric_key, arm_value in targets.items():
                floor_value, initial_floor, activation_peak, floor_active = self._compute_intrabar_floor_state(
                    float(arm_value),
                    float(metrics[metric_key]),
                    keep_ratio,
                )
                trailing_state[f"floor_{metric_key}"] = float(floor_value)
                trailing_state[f"initial_floor_{metric_key}"] = float(initial_floor)
                trailing_state[f"activation_peak_{metric_key}"] = float(activation_peak)
                trailing_state[f"floor_active_{metric_key}"] = bool(floor_active)
            self.intrabar_trailing_state = dict(trailing_state)
            print(
                "\n [INTRABAR] trailing armed "
                f"| side={side} | regime={regime} | keep={keep_ratio:.2f} "
                f"| change={float(metrics['change_pct']):+.3f}% | pips={float(metrics['pnl_pips']):+.1f} "
                f"| pnl={float(metrics['pnl_money']):+.2f} | trigger={' ; '.join(hit_reasons)}"
            )
            self._add_log(
                "info",
                "Intrabar trailing armed",
                phase="intrabar",
                event="trail_armed",
                meta={
                    "ticket": int(ticket),
                    "side": side,
                    "regime": str(regime),
                    "keep_ratio": float(keep_ratio),
                    "change_pct": float(metrics["change_pct"]),
                    "pnl_pips": float(metrics["pnl_pips"]),
                    "pnl_money": float(metrics["pnl_money"]),
                    "buffer_ratio": float(self.intrabar_trail_arm_buffer_ratio),
                    "confirm_polls": int(self.intrabar_trail_confirm_polls),
                    "initial_floor_change_pct": float(trailing_state.get("initial_floor_change_pct", 0.0) or 0.0),
                    "activation_peak_change_pct": float(trailing_state.get("activation_peak_change_pct", 0.0) or 0.0),
                    "reasons": " | ".join(hit_reasons),
                },
            )
            self._save_runtime_state(reason="intrabar_trail_armed")
            return False

        prev_regime = str(trailing_state.get("regime", "") or "")
        prev_keep_ratio = float(trailing_state.get("keep_ratio", keep_ratio) or keep_ratio)
        if prev_regime != regime or abs(prev_keep_ratio - float(keep_ratio)) > 1e-9:
            trailing_state["regime"] = str(regime)
            trailing_state["keep_ratio"] = float(keep_ratio)
            trailing_state["regime_meta"] = dict(regime_meta)
            state_changed = True

        trailing_state["last_change_pct"] = float(metrics["change_pct"])
        trailing_state["last_pnl_pips"] = float(metrics["pnl_pips"])
        trailing_state["last_pnl_money"] = float(metrics["pnl_money"])

        for metric_key, arm_value in targets.items():
            peak_key = f"peak_{metric_key}"
            floor_key = f"floor_{metric_key}"
            current_value = float(metrics.get(metric_key, 0.0) or 0.0)
            peak_value = float(trailing_state.get(peak_key, arm_value) or arm_value)
            if current_value > peak_value:
                peak_value = current_value
                trailing_state[peak_key] = float(peak_value)
                state_changed = True
            floor_value, initial_floor, activation_peak, floor_active = self._compute_intrabar_floor_state(
                float(arm_value),
                float(peak_value),
                keep_ratio,
            )
            if abs(float(trailing_state.get(floor_key, 0.0) or 0.0) - float(floor_value)) > 1e-9:
                trailing_state[floor_key] = float(floor_value)
                state_changed = True
            if abs(
                float(trailing_state.get(f"initial_floor_{metric_key}", 0.0) or 0.0) - float(initial_floor)
            ) > 1e-9:
                trailing_state[f"initial_floor_{metric_key}"] = float(initial_floor)
                state_changed = True
            if abs(
                float(trailing_state.get(f"activation_peak_{metric_key}", 0.0) or 0.0) - float(activation_peak)
            ) > 1e-9:
                trailing_state[f"activation_peak_{metric_key}"] = float(activation_peak)
                state_changed = True
            if bool(trailing_state.get(f"floor_active_{metric_key}", False)) != bool(floor_active):
                trailing_state[f"floor_active_{metric_key}"] = bool(floor_active)
                state_changed = True

        prev_breach_count = int(trailing_state.get("floor_breach_count", 0) or 0)
        confirm_required = max(1, int(trailing_state.get("floor_confirm_polls", self.intrabar_trail_confirm_polls) or 1))
        floor_hit_reasons: list[str] = []
        if float(targets.get("change_pct", 0.0) or 0.0) > 0.0:
            floor_value = float(trailing_state.get("floor_change_pct", 0.0) or 0.0)
            if float(metrics["change_pct"]) <= floor_value:
                floor_hit_reasons.append(
                    f"change {float(metrics['change_pct']):.3f}%<=floor {floor_value:.3f}%"
                )
        if float(targets.get("pnl_pips", 0.0) or 0.0) > 0.0:
            floor_value = float(trailing_state.get("floor_pnl_pips", 0.0) or 0.0)
            if float(metrics["pnl_pips"]) <= floor_value:
                floor_hit_reasons.append(
                    f"pips {float(metrics['pnl_pips']):.1f}<=floor {floor_value:.1f}"
                )
        if float(targets.get("pnl_money", 0.0) or 0.0) > 0.0:
            floor_value = float(trailing_state.get("floor_pnl_money", 0.0) or 0.0)
            if float(metrics["pnl_money"]) <= floor_value:
                floor_hit_reasons.append(
                    f"money {float(metrics['pnl_money']):.2f}<=floor {floor_value:.2f}"
                )
        if not floor_hit_reasons:
            if prev_breach_count > 0:
                trailing_state["floor_breach_count"] = 0
                trailing_state.pop("last_floor_hit_reasons", None)
                state_changed = True
                self._add_log(
                    "info",
                    "Intrabar trailing recovered above floor",
                    phase="intrabar",
                    event="trail_floor_recovered",
                    meta={
                        "ticket": int(ticket),
                        "side": side,
                        "regime": str(regime),
                        "keep_ratio": float(keep_ratio),
                        "confirm_polls": int(confirm_required),
                        "recovered_after": int(prev_breach_count),
                        "change_pct": float(metrics["change_pct"]),
                        "floor_change_pct": float(trailing_state.get("floor_change_pct", 0.0) or 0.0),
                    },
                )
                self._save_runtime_state(reason="intrabar_trail_recovered")
            self.intrabar_trailing_state = dict(trailing_state)
            if state_changed and prev_regime != regime:
                self._add_log(
                    "info",
                    "Intrabar trailing regime updated",
                    phase="intrabar",
                    event="trail_regime",
                    meta={
                        "ticket": int(ticket),
                        "side": side,
                        "previous_regime": str(prev_regime),
                        "regime": str(regime),
                        "keep_ratio": float(keep_ratio),
                        "peak_change_pct": float(trailing_state.get("peak_change_pct", metrics["change_pct"]) or metrics["change_pct"]),
                        "floor_change_pct": float(trailing_state.get("floor_change_pct", 0.0) or 0.0),
                        "initial_floor_change_pct": float(trailing_state.get("initial_floor_change_pct", 0.0) or 0.0),
                    },
                )
            return False

        breach_count = prev_breach_count + 1
        trailing_state["floor_breach_count"] = int(breach_count)
        trailing_state["last_floor_hit_reasons"] = [str(x) for x in floor_hit_reasons]
        self.intrabar_trailing_state = dict(trailing_state)
        if confirm_required > 1 and breach_count < confirm_required:
            print(
                "\n [INTRABAR] trailing floor pending "
                f"| side={side} | regime={regime} | confirm={breach_count}/{confirm_required} "
                f"| change={float(metrics['change_pct']):+.3f}% | floor={float(trailing_state.get('floor_change_pct', 0.0) or 0.0):+.3f}% "
                f"| trigger={' ; '.join(floor_hit_reasons)}"
            )
            self._add_log(
                "info",
                "Intrabar trailing floor breached, waiting for confirmation",
                phase="intrabar",
                event="trail_floor_pending",
                meta={
                    "ticket": int(ticket),
                    "side": side,
                    "regime": str(regime),
                    "keep_ratio": float(keep_ratio),
                    "confirm_count": int(breach_count),
                    "confirm_polls": int(confirm_required),
                    "change_pct": float(metrics["change_pct"]),
                    "floor_change_pct": float(trailing_state.get("floor_change_pct", 0.0) or 0.0),
                    "reasons": " | ".join(floor_hit_reasons),
                },
            )
            if breach_count == 1:
                self._save_runtime_state(reason="intrabar_trail_pending")
            return False

        close_reasons = [f"confirm {breach_count}/{confirm_required}"] + list(floor_hit_reasons)

        print(
            "\n [INTRABAR] trailing floor hit -> close "
            f"| side={side} | regime={regime} | keep={keep_ratio:.2f} "
            f"| price={float(metrics['exit_price']):.{self.digits}f} "
            f"| change={float(metrics['change_pct']):+.3f}% | pips={float(metrics['pnl_pips']):+.1f} "
            f"| pnl={float(metrics['pnl_money']):+.2f} | trigger={' ; '.join(close_reasons)}"
        )
        self._add_log(
            "action",
            "Intrabar trailing floor hit -> closing position",
            phase="intrabar",
            event="trail_floor_hit",
            meta={
                "ticket": int(ticket),
                "side": side,
                "regime": str(regime),
                "keep_ratio": float(keep_ratio),
                "price": float(metrics["exit_price"]),
                "change_pct": float(metrics["change_pct"]),
                "pnl_pips": float(metrics["pnl_pips"]),
                "pnl_money": float(metrics["pnl_money"]),
                "peak_change_pct": float(trailing_state.get("peak_change_pct", metrics["change_pct"]) or metrics["change_pct"]),
                "peak_pnl_pips": float(trailing_state.get("peak_pnl_pips", metrics["pnl_pips"]) or metrics["pnl_pips"]),
                "peak_pnl_money": float(trailing_state.get("peak_pnl_money", metrics["pnl_money"]) or metrics["pnl_money"]),
                "floor_change_pct": float(trailing_state.get("floor_change_pct", 0.0) or 0.0),
                "floor_pnl_pips": float(trailing_state.get("floor_pnl_pips", 0.0) or 0.0),
                "floor_pnl_money": float(trailing_state.get("floor_pnl_money", 0.0) or 0.0),
                "initial_floor_change_pct": float(trailing_state.get("initial_floor_change_pct", 0.0) or 0.0),
                "activation_peak_change_pct": float(trailing_state.get("activation_peak_change_pct", 0.0) or 0.0),
                "confirm_count": int(breach_count),
                "confirm_polls": int(confirm_required),
                "reasons": " | ".join(close_reasons),
            },
        )
        if not self.close_all():
            return False

        self._queue_intrabar_review(
            pos=pos,
            current_bar_time=int(current_bar_time),
            exit_time_ts=int(tick_time_ts or now_epoch),
            side=side,
            entry_price=float(metrics["entry_price"]),
            exit_price=float(metrics["exit_price"]),
            change_pct=float(metrics["change_pct"]),
            pnl_pips=float(metrics["pnl_pips"]),
            pnl_money=float(metrics["pnl_money"]),
            trigger_reasons=close_reasons,
            exit_mode="trailing_floor_hit",
            exit_meta={
                "regime": str(regime),
                "keep_ratio": float(keep_ratio),
                "peak_change_pct": float(trailing_state.get("peak_change_pct", metrics["change_pct"]) or metrics["change_pct"]),
                "peak_pnl_pips": float(trailing_state.get("peak_pnl_pips", metrics["pnl_pips"]) or metrics["pnl_pips"]),
                "peak_pnl_money": float(trailing_state.get("peak_pnl_money", metrics["pnl_money"]) or metrics["pnl_money"]),
                "floor_change_pct": float(trailing_state.get("floor_change_pct", 0.0) or 0.0),
                "floor_pnl_pips": float(trailing_state.get("floor_pnl_pips", 0.0) or 0.0),
                "floor_pnl_money": float(trailing_state.get("floor_pnl_money", 0.0) or 0.0),
                "initial_floor_change_pct": float(trailing_state.get("initial_floor_change_pct", 0.0) or 0.0),
                "activation_peak_change_pct": float(trailing_state.get("activation_peak_change_pct", 0.0) or 0.0),
                "confirm_count": int(breach_count),
                "confirm_polls": int(confirm_required),
            },
        )
        self._clear_intrabar_trailing_state()
        self.last_action = "CLOSE"
        return True

    def _reset_intrabar_cooldown_if_flattened(self, prev_bridge_pos: int, current_bar_time: int):
        if self.bridge is None:
            return
        if prev_bridge_pos == 0 or int(self.bridge.position) != 0:
            return
        if current_bar_time <= 0 or current_bar_time != self.last_bar_time:
            return

        if int(self.bridge.trade_cooldown) != 0 or bool(self.bridge.first_bar):
            print("\n [INTRABAR] flat before next candle -> reset cooldown for next prediction")
            self._add_log(
                "info",
                "Intrabar exit detected -> reset cooldown for next candle",
                phase="intrabar",
                event="cooldown_reset",
                meta={"bar_time": int(current_bar_time)},
            )

        self.bridge.trade_cooldown = 0
        self.bridge.first_bar = False
        self._clear_intrabar_trailing_state()
        self.last_action = "CLOSE"
        self._save_runtime_state(reason="intrabar_flat")

    def send_order(self, order_type):
        if not self._ensure_trade_allowed_before_order(
            reason="open_buy" if order_type == mt5.ORDER_TYPE_BUY else "open_sell",
            action_label="Open order",
        ):
            return False

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

        side = "BUY" if order_type == mt5.ORDER_TYPE_BUY else "SELL"
        price = tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid

        if self._is_custom_lot_mode():
            self.current_lot = self._resolve_runtime_lot()
        elif self.dynamic_lot:
            account = mt5.account_info()
            if account is not None:
                self.current_lot = self._resolve_runtime_lot(float(account.balance))
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

        funds_ok, funds_meta = self._precheck_open_order_funds(req, side=side)
        if not funds_ok:
            reason = str((funds_meta or {}).get("reason") or "Insufficient funds/margin").strip()
            retcode_name = str((funds_meta or {}).get("retcode_name") or "").strip()
            print(f" Order Blocked: {reason}")
            self._add_log(
                "warning",
                reason,
                phase="order",
                event="open_blocked_insufficient_funds",
                meta={
                    "side": side,
                    "lot": float(self.current_lot),
                    "price": float(price),
                    "retcode": str((funds_meta or {}).get("retcode") or ""),
                    "retcode_name": retcode_name,
                    "balance": float((funds_meta or {}).get("balance") or 0.0),
                    "free_margin": float((funds_meta or {}).get("free_margin") or 0.0),
                    "required_margin": float((funds_meta or {}).get("required_margin") or 0.0),
                    "reason": reason,
                },
            )
            return False

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
            retcode = getattr(res, "retcode", None) if res is not None else None
            retcode_name = self._trade_retcode_name(retcode)
            is_no_money = self._is_insufficient_funds_result(retcode, comment)
            is_autotrading_disabled = self._is_autotrading_disabled_result(retcode, comment)
            if is_no_money:
                event_name = "open_failed_insufficient_funds"
            elif is_autotrading_disabled:
                event_name = "open_failed_autotrading_disabled"
            else:
                event_name = "open_failed"
            self._add_log(
                "warning",
                f"Order failed: {comment}",
                phase="order",
                event=event_name,
                meta={
                    "side": side,
                    "lot": float(self.current_lot),
                    "price": float(price),
                    "retcode": str(retcode or ""),
                    "retcode_name": retcode_name,
                    "reason": str(comment),
                },
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

        self._review_intrabar_exits_at_bar_close(bar_end_ts, window_df)
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

        self._update_intrabar_regime_snapshot(bar_end_ts, decision)

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
        self._sync_performance_from_mt5_history(force=True, full_resync=True, reason="startup")
        self._sync_bridge_from_mt5()
        self._prewarm_semantic_on_start()
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
        if (
            self.intrabar_take_profit_change_pct > 0.0
            or self.intrabar_take_profit_pips > 0.0
            or self.intrabar_take_profit_money > 0.0
        ):
            intrabar_parts = []
            intrabar_mode = "trailing" if self.intrabar_trailing_enabled else "hard_close"
            intrabar_parts.append(f"mode={intrabar_mode}")
            if self.intrabar_take_profit_change_pct > 0.0:
                intrabar_parts.append(f"change={self.intrabar_take_profit_change_pct:.3f}%")
            if self.intrabar_take_profit_pips > 0.0:
                intrabar_parts.append(f"pips={self.intrabar_take_profit_pips:.1f}")
            if self.intrabar_take_profit_money > 0.0:
                intrabar_parts.append(f"money={self.intrabar_take_profit_money:.2f}")
            if self.intrabar_trailing_enabled:
                intrabar_parts.append(
                    "keep="
                    f"{self.intrabar_trail_keep_ratio_trend:.2f}/"
                    f"{self.intrabar_trail_keep_ratio_normal:.2f}/"
                    f"{self.intrabar_trail_keep_ratio_tight:.2f}"
                )
                intrabar_parts.append(f"buffer={self.intrabar_trail_arm_buffer_ratio:.2f}")
                intrabar_parts.append(f"confirm={self.intrabar_trail_confirm_polls}")
            print(f" [INTRABAR] realtime close active | {' | '.join(intrabar_parts)}")
            self._add_log(
                "info",
                "Intrabar take-profit monitoring active",
                phase="intrabar",
                event="enabled",
                meta={
                    "take_profit_change_pct": float(self.intrabar_take_profit_change_pct),
                    "take_profit_pips": float(self.intrabar_take_profit_pips),
                    "take_profit_money": float(self.intrabar_take_profit_money),
                    "trailing_enabled": bool(self.intrabar_trailing_enabled),
                    "trail_keep_ratio_trend": float(self.intrabar_trail_keep_ratio_trend),
                    "trail_keep_ratio_normal": float(self.intrabar_trail_keep_ratio_normal),
                    "trail_keep_ratio_tight": float(self.intrabar_trail_keep_ratio_tight),
                    "trail_arm_buffer_ratio": float(self.intrabar_trail_arm_buffer_ratio),
                    "trail_confirm_polls": int(self.intrabar_trail_confirm_polls),
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
                        has_exposure, pos_count, order_count, exposure_meta = self._broker_exposure_summary()
                        if has_exposure:
                            pos_tickets = list(exposure_meta.get("pos_tickets", []) or [])
                            order_tickets = list(exposure_meta.get("order_tickets", []) or [])
                            foreign_order_count = int(exposure_meta.get("foreign_order_count", 0) or 0)
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
                                meta={
                                    "positions": int(pos_count),
                                    "orders": int(order_count),
                                    "pos_tickets": ",".join(str(x) for x in pos_tickets) if pos_tickets else "",
                                    "order_tickets": ",".join(str(x) for x in order_tickets) if order_tickets else "",
                                    "ignored_foreign_orders": int(foreign_order_count),
                                },
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
                else:
                    self._maybe_take_profit_intrabar(current_bar_time)

                prev_bridge_pos = int(self.bridge.position) if self.bridge is not None else 0
                self._sync_bridge_from_mt5()
                self._reset_intrabar_cooldown_if_flattened(prev_bridge_pos, current_bar_time)
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
