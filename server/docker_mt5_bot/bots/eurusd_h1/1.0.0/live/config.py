import json
import os
from urllib.parse import urlparse


RL_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CORE_DIR = os.path.join(RL_ROOT, "core")
MODELS_DIR = os.getenv("LIVE_MODELS_DIR", os.path.join(RL_ROOT, "models")).strip() or os.path.join(
    RL_ROOT,
    "models",
)


def _env_first(*names: str) -> str:
    for name in names:
        raw = os.getenv(name, "").strip()
        if raw:
            return raw
    return ""


def _env_float(default: float, *names: str) -> float:
    raw = _env_first(*names)
    if not raw:
        return float(default)
    try:
        return float(raw)
    except ValueError:
        return float(default)


def _env_int(default: int, *names: str) -> int:
    raw = _env_first(*names)
    if not raw:
        return int(default)
    try:
        return int(float(raw))
    except ValueError:
        return int(default)


def _env_bool(default: bool, *names: str) -> bool:
    raw = _env_first(*names).lower()
    if not raw:
        return bool(default)
    return raw in {"1", "true", "yes", "on"}


def _env_int_set(default_values: set[int], *names: str) -> set[int]:
    raw = _env_first(*names)
    if not raw:
        return set(int(v) for v in default_values)

    out: set[int] = set()
    for chunk in str(raw).replace(";", ",").split(","):
        text = str(chunk or "").strip()
        if not text:
            continue
        try:
            out.add(int(float(text)))
        except Exception:
            continue
    if len(out) == 0:
        return set(int(v) for v in default_values)
    return out


def _env_int_tuple(default_values: tuple[int, ...], *names: str) -> tuple[int, ...]:
    raw = _env_first(*names)
    if not raw:
        return tuple(int(v) for v in default_values)

    out: list[int] = []
    seen: set[int] = set()
    for chunk in str(raw).replace(";", ",").split(","):
        text = str(chunk or "").strip()
        if not text:
            continue
        try:
            value = int(float(text))
        except Exception:
            continue
        if value in seen:
            continue
        out.append(value)
        seen.add(value)
    if len(out) == 0:
        return tuple(int(v) for v in default_values)
    return tuple(out)


def _safe_token(raw: str, fallback: str = "default") -> str:
    text = str(raw or "").strip().lower()
    if not text:
        return fallback
    out = []
    for ch in text:
        if ch.isalnum() or ch in {"_", "-"}:
            out.append(ch)
        else:
            out.append("_")
    token = "".join(out).strip("_")
    return token or fallback


def _parse_risk_profile(raw: str) -> dict[str, float]:
    profile = {"low": 0.5, "medium": 1.0, "high": 1.5}
    if not raw:
        return profile
    try:
        payload = json.loads(raw)
    except Exception:
        return profile
    if not isinstance(payload, dict):
        return profile

    for key, value in payload.items():
        lvl = str(key).strip().lower()
        if lvl not in profile:
            continue
        try:
            pct = float(value)
        except Exception:
            continue
        if pct > 0:
            profile[lvl] = pct
    return profile


def _parse_trading_schedule(raw: str) -> dict[str, bool]:
    default = {
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
    if not raw:
        return default
    try:
        payload = json.loads(raw)
    except Exception:
        return default
    if not isinstance(payload, dict):
        return default
    for raw_key, raw_value in payload.items():
        key = alias_to_key.get(str(raw_key).strip().lower())
        if key:
            default[key] = bool(raw_value)
    return default


# ---- Trading profile / runtime controls
_risk_profile_raw = _env_first("LIVE_RISK_PROFILE_JSON", "LIVE_RISK_MAP_JSON")
RISK_PROFILE_MAP = _parse_risk_profile(_risk_profile_raw)
RISK_LEVEL = (_env_first("LIVE_RISK_LEVEL") or "medium").lower()
if RISK_LEVEL not in RISK_PROFILE_MAP:
    RISK_LEVEL = "medium"

_default_risk = float(RISK_PROFILE_MAP.get(RISK_LEVEL, 1.0))
RISK_PERCENT = _env_float(_default_risk, "LIVE_RISK_PERCENT", "RISK_PERCENT")
if RISK_PERCENT <= 0:
    RISK_PERCENT = _default_risk

TRADING_SCHEDULE_DEFAULT = _parse_trading_schedule(_env_first("LIVE_TRADING_SCHEDULE_JSON"))


# ---- MT5 / bot runtime
MT5_HOST = _env_first("MT5_HOST") or "localhost"
MT5_PORT = _env_int(8001, "MT5_PORT")
MT5_LOGIN = _env_first("MT5_LOGIN")
MT5_PASSWORD = _env_first("MT5_PASSWORD")
MT5_SERVER = _env_first("MT5_SERVER")
MT5_SERVER_FALLBACKS = [
    chunk.strip()
    for chunk in _env_first("MT5_SERVER_FALLBACKS").replace(";", ",").split(",")
    if chunk.strip()
]
MT5_STRICT_SERVER_MATCH = _env_bool(False, "MT5_STRICT_SERVER_MATCH")
MT5_RPC_TIMEOUT_MS = max(30000, _env_int(60000, "MT5_RPC_TIMEOUT_MS"))
MT5_LOGIN_RETRIES = max(1, _env_int(20, "MT5_LOGIN_RETRIES"))
MT5_RETRY_SECONDS = max(1.0, _env_float(5.0, "MT5_RETRY_SECONDS"))
SYMBOL = _env_first("LIVE_SYMBOL") or "EURUSD"
TIMEFRAME_NAME = (_env_first("LIVE_TIMEFRAME") or "H1").upper()
MAGIC_NUMBER = _env_int(123456, "LIVE_MAGIC_NUMBER")
DEVIATION = _env_int(20, "LIVE_DEVIATION")
POLL_SECONDS = _env_float(1.0, "LIVE_POLL_SECONDS")
ORDER_TICK_RETRIES = _env_int(8, "LIVE_ORDER_TICK_RETRIES")
ORDER_TICK_RETRY_SEC = _env_float(0.25, "LIVE_ORDER_TICK_RETRY_SEC")
SYNC_EXTERNAL_LOT = _env_bool(True, "LIVE_SYNC_EXTERNAL_LOT")
EVAL_ON_START = _env_bool(True, "LIVE_EVAL_ON_START")
ENABLE_CATCHUP_REPLAY = _env_bool(True, "LIVE_ENABLE_CATCHUP_REPLAY")
MAX_CATCHUP_BARS = _env_int(24, "LIVE_CATCHUP_MAX_BARS")
EXECUTE_STALE_REPLAY_ORDERS = _env_bool(False, "LIVE_CATCHUP_EXECUTE_STALE")
LIVE_SYNC_ACCOUNT_STATE = _env_bool(True, "LIVE_SYNC_ACCOUNT_STATE")
LIVE_DYNAMIC_LOT = _env_bool(True, "LIVE_DYNAMIC_LOT")
LIVE_MANAGE_MANUAL_POSITIONS = _env_bool(False, "LIVE_MANAGE_MANUAL_POSITIONS")
LIVE_STATUS_LINE_ENABLED = _env_bool(True, "LIVE_STATUS_LINE_ENABLED")
LIVE_STATUS_LOG_INTERVAL_SEC = max(0.0, _env_float(60.0, "LIVE_STATUS_LOG_INTERVAL_SEC"))
LIVE_SEMANTIC_NO_DATA_RETRY_SECONDS = max(
    10.0,
    _env_float(180.0, "LIVE_SEMANTIC_NO_DATA_RETRY_SECONDS"),
)
LIVE_SEMANTIC_ALIAS_HOURS = _env_int_tuple(
    (0,),
    "LIVE_SEMANTIC_ALIAS_HOURS",
)

BOT_WS_URL = _env_first("BOT_WS_URL") or "ws://localhost:8000/bot/ws"
BOT_CONFIG_ID = _env_first("BOT_CONFIG_ID") or ""


def _derive_vision_llm_api_url(bot_ws_url: str) -> str:
    explicit = _env_first("VISION_LLM_API_URL")
    if explicit:
        return explicit

    try:
        parsed = urlparse(bot_ws_url)
        if parsed.netloc:
            scheme = "https" if parsed.scheme == "wss" else "http"
            return f"{scheme}://{parsed.netloc}/vision_llm/"
    except Exception:
        pass
    return "http://localhost:8000/vision_llm/"


def _derive_vision_llm_embed_text_api_url(vision_llm_api_url: str) -> str:
    explicit = _env_first("VISION_LLM_EMBED_TEXT_API_URL")
    if explicit:
        return explicit

    base = str(vision_llm_api_url or "").strip()
    if not base:
        return "http://localhost:8000/vision_llm/embed_text"
    if base.endswith("/"):
        return f"{base}embed_text"
    return f"{base}/embed_text"


VISION_LLM_API_URL = _derive_vision_llm_api_url(BOT_WS_URL)
VISION_LLM_EMBED_TEXT_API_URL = _derive_vision_llm_embed_text_api_url(VISION_LLM_API_URL)
VISION_LLM_TIMEOUT_SEC = _env_float(420.00, "VISION_LLM_TIMEOUT_SEC")
LIVE_PERFORMANCE_SYNC_INTERVAL_SEC = max(
    5.0,
    _env_float(120.0, "LIVE_PERFORMANCE_SYNC_INTERVAL_SEC"),
)
LIVE_PERFORMANCE_BOOT_LOOKBACK_DAYS = max(
    30,
    _env_int(3650, "LIVE_PERFORMANCE_BOOT_LOOKBACK_DAYS"),
)
LIVE_MT5_HISTORY_END_AHEAD_HOURS = min(
    24.0,
    max(0.0, _env_float(2.0, "LIVE_MT5_HISTORY_END_AHEAD_HOURS")),
)
LIVE_PERFORMANCE_SCOPE = (_env_first("LIVE_PERFORMANCE_SCOPE") or "symbol").strip().lower()
if LIVE_PERFORMANCE_SCOPE not in {"managed", "symbol", "account"}:
    LIVE_PERFORMANCE_SCOPE = "symbol"
_default_managed_magic_set = {int(MAGIC_NUMBER), 123456, 12345}
if LIVE_MANAGE_MANUAL_POSITIONS:
    _default_managed_magic_set.add(0)
LIVE_MANAGED_MAGIC_SET = _env_int_set(
    _default_managed_magic_set,
    "LIVE_MANAGED_MAGIC_SET",
    "LIVE_MANAGED_MAGIC_NUMBERS",
    "LIVE_MAGIC_SET",
    "LIVE_MAGIC_NUMBERS",
)
LIVE_PERFORMANCE_MAGIC_SET = _env_int_set(
    set(LIVE_MANAGED_MAGIC_SET),
    "LIVE_PERFORMANCE_MAGIC_SET",
    "LIVE_PERFORMANCE_MAGIC_NUMBERS",
)
LIVE_PREWARM_SEMANTIC_ON_START = _env_bool(True, "LIVE_PREWARM_SEMANTIC_ON_START")
LIVE_PREWARM_SEMANTIC_MAX_SECONDS = max(
    0.0,
    _env_float(45.0, "LIVE_PREWARM_SEMANTIC_MAX_SECONDS"),
)
LIVE_PREWARM_SEMANTIC_MAX_MISSING = max(
    1,
    _env_int(8, "LIVE_PREWARM_SEMANTIC_MAX_MISSING"),
)
LIVE_PREWARM_REQUEST_TIMEOUT_SEC = max(
    5.0,
    _env_float(20.0, "LIVE_PREWARM_REQUEST_TIMEOUT_SEC"),
)


# ---- Live feature / gate / bridge parameters (owned by live config)
WINDOW_SIZE = _env_int(20, "LIVE_WINDOW_SIZE", "WINDOW_SIZE")
BAR_HISTORY = _env_int(167, "LIVE_BAR_HISTORY", "BAR_HISTORY")
INITIAL_BALANCE = _env_float(100.0, "LIVE_INITIAL_BALANCE", "INITIAL_BALANCE")
PIP_SIZE = _env_float(0.0001, "LIVE_PIP_SIZE", "PIP_SIZE")
PIP_VALUE = _env_float(10.0, "LIVE_PIP_VALUE", "PIP_VALUE")
RISK_PIPS = _env_int(50, "LIVE_RISK_PIPS", "RISK_PIPS")
SPREAD_PIPS = _env_float(2.0, "LIVE_SPREAD_PIPS", "SPREAD_PIPS")
MAX_HOLD_STEPS = _env_int(16, "LIVE_MAX_HOLD_STEPS", "MAX_HOLD_STEPS")
INTRABAR_TAKE_PROFIT_PIPS = max(
    0.0,
    _env_float(0.0, "LIVE_INTRABAR_TAKE_PROFIT_PIPS"),
)
INTRABAR_TAKE_PROFIT_MONEY = max(
    0.0,
    _env_float(0.0, "LIVE_INTRABAR_TAKE_PROFIT_MONEY"),
)
INTRABAR_TAKE_PROFIT_CHANGE_PCT = max(
    0.0,
    _env_float(0.0, "LIVE_INTRABAR_TAKE_PROFIT_CHANGE_PCT"),
)
INTRABAR_TRAILING_ENABLED = _env_bool(True, "LIVE_INTRABAR_TRAILING_ENABLED")
INTRABAR_TRAIL_KEEP_RATIO_TREND = min(
    1.0,
    max(0.0, _env_float(0.35, "LIVE_INTRABAR_TRAIL_KEEP_RATIO_TREND")),
)
INTRABAR_TRAIL_KEEP_RATIO_NORMAL = min(
    1.0,
    max(0.0, _env_float(0.60, "LIVE_INTRABAR_TRAIL_KEEP_RATIO_NORMAL")),
)
INTRABAR_TRAIL_KEEP_RATIO_TIGHT = min(
    1.0,
    max(0.0, _env_float(0.85, "LIVE_INTRABAR_TRAIL_KEEP_RATIO_TIGHT")),
)
INTRABAR_TRAIL_ARM_BUFFER_RATIO = min(
    0.95,
    max(0.0, _env_float(0.10, "LIVE_INTRABAR_TRAIL_ARM_BUFFER_RATIO")),
)
INTRABAR_TRAIL_CONFIRM_POLLS = max(
    1,
    _env_int(3, "LIVE_INTRABAR_TRAIL_CONFIRM_POLLS"),
)
INTRABAR_MIN_BAR_AGE_SEC = max(
    0,
    _env_int(300, "LIVE_INTRABAR_MIN_BAR_AGE_SEC"),
)
INTRABAR_DISABLE_LAST_SEC = max(
    0,
    _env_int(180, "LIVE_INTRABAR_DISABLE_LAST_SEC"),
)
EMBED_SOURCE_MODE = (_env_first("LIVE_EMBED_SOURCE_MODE", "EMBED_SOURCE_MODE") or "cls").strip().lower()

OPEN_PROB_THRESHOLD = _env_float(0.75, "LIVE_OPEN_PROB_THRESHOLD", "OPEN_PROB_THRESHOLD")
OPEN_EDGE_THRESHOLD = _env_float(0.16, "LIVE_OPEN_EDGE_THRESHOLD", "OPEN_EDGE_THRESHOLD")
MIN_ACTION_MARGIN = _env_float(0.14, "LIVE_MIN_ACTION_MARGIN", "MIN_ACTION_MARGIN")
HOLD_EDGE_THRESHOLD = _env_float(0.05, "LIVE_HOLD_EDGE_THRESHOLD", "HOLD_EDGE_THRESHOLD")
TRADE_COOLDOWN_BARS = _env_int(2, "LIVE_TRADE_COOLDOWN_BARS", "TRADE_COOLDOWN_BARS")
MASK_CLOSE_WHEN_FLAT = _env_bool(True, "LIVE_MASK_CLOSE_WHEN_FLAT", "MASK_CLOSE_WHEN_FLAT")

ADAPTIVE_GATE = _env_bool(True, "LIVE_ADAPTIVE_GATE", "ADAPTIVE_GATE")
DEF_LOOKBACK_TRADES = _env_int(20, "LIVE_DEF_LOOKBACK_TRADES", "DEF_LOOKBACK_TRADES")
DEF_MIN_WINRATE = _env_float(0.50, "LIVE_DEF_MIN_WINRATE", "DEF_MIN_WINRATE")
DEF_MIN_AVG_PIPS = _env_float(1.0, "LIVE_DEF_MIN_AVG_PIPS", "DEF_MIN_AVG_PIPS")
DEF_MAX_LOSS_STREAK = _env_int(3, "LIVE_DEF_MAX_LOSS_STREAK", "DEF_MAX_LOSS_STREAK")
DEF_BARS = _env_int(48, "LIVE_DEF_BARS", "DEF_BARS")
DEF_CONF_BONUS = _env_float(0.04, "LIVE_DEF_CONF_BONUS", "DEF_CONF_BONUS")
DEF_EDGE_BONUS = _env_float(0.05, "LIVE_DEF_EDGE_BONUS", "DEF_EDGE_BONUS")
DEF_MARGIN_BONUS = _env_float(0.04, "LIVE_DEF_MARGIN_BONUS", "DEF_MARGIN_BONUS")
DEF_COOLDOWN_BONUS = _env_int(2, "LIVE_DEF_COOLDOWN_BONUS", "DEF_COOLDOWN_BONUS")
VOL_CONF_BONUS = _env_float(0.02, "LIVE_VOL_CONF_BONUS", "VOL_CONF_BONUS")
VOL_EDGE_BONUS = _env_float(0.03, "LIVE_VOL_EDGE_BONUS", "VOL_EDGE_BONUS")
VOL_MARGIN_BONUS = _env_float(0.02, "LIVE_VOL_MARGIN_BONUS", "VOL_MARGIN_BONUS")
FLAT_CONF_BONUS = _env_float(0.015, "LIVE_FLAT_CONF_BONUS", "FLAT_CONF_BONUS")
FLAT_EDGE_BONUS = _env_float(0.02, "LIVE_FLAT_EDGE_BONUS", "FLAT_EDGE_BONUS")
FLAT_MARGIN_BONUS = _env_float(0.03, "LIVE_FLAT_MARGIN_BONUS", "FLAT_MARGIN_BONUS")
TREND_RELAX = _env_float(0.02, "LIVE_TREND_RELAX", "TREND_RELAX")
COUNTER_TREND_CONF_BONUS = _env_float(0.02, "LIVE_COUNTER_TREND_CONF_BONUS", "COUNTER_TREND_CONF_BONUS")
COUNTER_TREND_EDGE_BONUS = _env_float(0.04, "LIVE_COUNTER_TREND_EDGE_BONUS", "COUNTER_TREND_EDGE_BONUS")
COUNTER_TREND_MARGIN_BONUS = _env_float(0.04, "LIVE_COUNTER_TREND_MARGIN_BONUS", "COUNTER_TREND_MARGIN_BONUS")
COUNTER_TREND_HOLD_EDGE_BONUS = _env_float(
    0.03,
    "LIVE_COUNTER_TREND_HOLD_EDGE_BONUS",
    "COUNTER_TREND_HOLD_EDGE_BONUS",
)
DEF_HOLD_EDGE_BONUS = _env_float(0.03, "LIVE_DEF_HOLD_EDGE_BONUS", "DEF_HOLD_EDGE_BONUS")
VOL_HOLD_EDGE_BONUS = _env_float(0.02, "LIVE_VOL_HOLD_EDGE_BONUS", "VOL_HOLD_EDGE_BONUS")
FLAT_HOLD_EDGE_BONUS = _env_float(0.02, "LIVE_FLAT_HOLD_EDGE_BONUS", "FLAT_HOLD_EDGE_BONUS")
TREND_HOLD_RELAX = _env_float(0.01, "LIVE_TREND_HOLD_RELAX", "TREND_HOLD_RELAX")

EMBED_QUALITY_MIN = _env_float(0.25, "LIVE_EMBED_QUALITY_MIN", "EMBED_QUALITY_MIN")
EMBED_QUALITY_CONF_BONUS = _env_float(0.10, "LIVE_EMBED_QUALITY_CONF_BONUS", "EMBED_QUALITY_CONF_BONUS")
EMBED_QUALITY_EDGE_BONUS = _env_float(0.08, "LIVE_EMBED_QUALITY_EDGE_BONUS", "EMBED_QUALITY_EDGE_BONUS")
EMBED_QUALITY_MARGIN_BONUS = _env_float(0.06, "LIVE_EMBED_QUALITY_MARGIN_BONUS", "EMBED_QUALITY_MARGIN_BONUS")
EMBED_QUALITY_HOLD_EDGE_BONUS = _env_float(
    0.05,
    "LIVE_EMBED_QUALITY_HOLD_EDGE_BONUS",
    "EMBED_QUALITY_HOLD_EDGE_BONUS",
)
EMBED_QUALITY_COOLDOWN_BONUS = _env_int(
    2,
    "LIVE_EMBED_QUALITY_COOLDOWN_BONUS",
    "EMBED_QUALITY_COOLDOWN_BONUS",
)

BASE_COLUMNS = [
    "return",
    "range",
    "delta_tick",
    "delta_price",
    "body_ratio",
    "momentum",
    "sma_cross",
    "rsi_norm",
    "atr_norm",
    "trend",
    "adx",
]


# ---- Runtime artifacts
cache_identity = _safe_token(_env_first("MT5_SERVER"), fallback="")
if not cache_identity:
    cache_identity = _safe_token(f"{SYMBOL}_{TIMEFRAME_NAME}")

cache_dir = os.path.join(MODELS_DIR, "cache", f"broker-{cache_identity}")

LLM_SEMANTIC_CACHE_FILE = (
    _env_first("LIVE_LLM_SEMANTIC_CACHE_FILE")
    or os.path.join(cache_dir, "time_to_embedding_llm_cls.joblib")
)
LLM_TEXT_LOG_FILE = (
    _env_first("LIVE_LLM_TEXT_LOG_FILE")
    or os.path.join(cache_dir, "time_to_llm_text.jsonl")
)
LLM_SEMANTIC_CACHE_SCHEMA = "utc_v2"
STATE_FILE = _env_first("LIVE_STATE_FILE")
if not STATE_FILE:
    state_identity = (
        _safe_token(BOT_CONFIG_ID)
        if BOT_CONFIG_ID
        else _safe_token(
            f"{_env_first('MT5_LOGIN')}|{_env_first('MT5_SERVER')}|{SYMBOL}_{TIMEFRAME_NAME}",
            fallback="",
        )
    )
    if not state_identity:
        state_identity = _safe_token(f"{SYMBOL}_{TIMEFRAME_NAME}")
    STATE_FILE = os.path.join(MODELS_DIR, f"run_live_state_{state_identity}.json")
MODEL_PATH = os.path.join(MODELS_DIR, "ppo_trading.zip")
VEC_NORM_PATH = os.path.join(MODELS_DIR, "vec_normalize.pkl")


TIMEFRAME_SECONDS_MAP = {
    "M1": 60,
    "M2": 120,
    "M3": 180,
    "M4": 240,
    "M5": 300,
    "M6": 360,
    "M10": 600,
    "M12": 720,
    "M15": 900,
    "M20": 1200,
    "M30": 1800,
    "H1": 3600,
    "H2": 7200,
    "H3": 10800,
    "H4": 14400,
    "H6": 21600,
    "H8": 28800,
    "H12": 43200,
    "D1": 86400,
    "W1": 604800,
}
