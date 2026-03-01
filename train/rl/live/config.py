import json
import os
import sys


RL_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CORE_DIR = os.path.join(RL_ROOT, "core")
TEST_DIR = os.path.join(RL_ROOT, "test")
LLM_DIR = os.path.join(os.path.dirname(RL_ROOT), "llm")
if TEST_DIR not in sys.path:
    sys.path.insert(0, TEST_DIR)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return float(default)
    try:
        return float(raw)
    except ValueError:
        return float(default)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return int(default)
    try:
        return int(float(raw))
    except ValueError:
        return int(default)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return bool(default)
    return raw in {"1", "true", "yes", "on"}


def _forward_live_override(base_name: str) -> None:
    """Map LIVE_<NAME> to <NAME> for shared backtest modules."""
    base_raw = os.getenv(base_name, "").strip()
    live_raw = os.getenv(f"LIVE_{base_name}", "").strip()
    if not base_raw and live_raw:
        os.environ[base_name] = live_raw


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


# Forward selected live-only env vars to shared backtest config
for _name in (
    "BAR_HISTORY",
    "WINDOW_SIZE",
    "SPREAD_PIPS",
    "RISK_PIPS",
    "TEST_DATA_FILE",
    "TEST_DATE_FROM",
    "TEST_DATE_TO",
    "OPEN_PROB_THRESHOLD",
    "OPEN_EDGE_THRESHOLD",
    "MIN_ACTION_MARGIN",
    "HOLD_EDGE_THRESHOLD",
    "TRADE_COOLDOWN_BARS",
):
    _forward_live_override(_name)


_risk_profile_raw = (
    os.getenv("LIVE_RISK_PROFILE_JSON", "").strip()
    or os.getenv("LIVE_RISK_MAP_JSON", "").strip()
)
RISK_PROFILE_MAP = _parse_risk_profile(_risk_profile_raw)
RISK_LEVEL = (os.getenv("LIVE_RISK_LEVEL", "medium").strip().lower() or "medium")
if RISK_LEVEL not in RISK_PROFILE_MAP:
    RISK_LEVEL = "medium"

_default_risk = float(RISK_PROFILE_MAP.get(RISK_LEVEL, 1.0))
RISK_PERCENT = _env_float("LIVE_RISK_PERCENT", _default_risk)
if RISK_PERCENT <= 0:
    RISK_PERCENT = _default_risk

# Keep shared backtest modules (imported by live runtime) aligned with live risk.
os.environ["RISK_PERCENT"] = str(RISK_PERCENT)
TRADING_SCHEDULE_DEFAULT = _parse_trading_schedule(
    os.getenv("LIVE_TRADING_SCHEDULE_JSON", "").strip()
)

MT5_HOST = os.getenv("MT5_HOST", "localhost").strip() or "localhost"
MT5_PORT = int(os.getenv("MT5_PORT", "8001"))
SYMBOL = os.getenv("LIVE_SYMBOL", "EURUSD").strip() or "EURUSD"
TIMEFRAME_NAME = os.getenv("LIVE_TIMEFRAME", "H1").strip().upper()
MAGIC_NUMBER = int(os.getenv("LIVE_MAGIC_NUMBER", "123456"))
DEVIATION = int(os.getenv("LIVE_DEVIATION", "20"))
POLL_SECONDS = float(os.getenv("LIVE_POLL_SECONDS", "1"))
ORDER_TICK_RETRIES = int(os.getenv("LIVE_ORDER_TICK_RETRIES", "8"))
ORDER_TICK_RETRY_SEC = float(os.getenv("LIVE_ORDER_TICK_RETRY_SEC", "0.25"))
SYNC_EXTERNAL_LOT = os.getenv("LIVE_SYNC_EXTERNAL_LOT", "1").strip().lower() in {"1", "true", "yes"}
EVAL_ON_START = os.getenv("LIVE_EVAL_ON_START", "1").strip().lower() in {"1", "true", "yes"}
ENABLE_CATCHUP_REPLAY = os.getenv("LIVE_ENABLE_CATCHUP_REPLAY", "1").strip().lower() in {"1", "true", "yes"}
MAX_CATCHUP_BARS = int(os.getenv("LIVE_CATCHUP_MAX_BARS", "0"))
EXECUTE_STALE_REPLAY_ORDERS = os.getenv("LIVE_CATCHUP_EXECUTE_STALE", "0").strip().lower() in {
    "1",
    "true",
    "yes",
}
USE_LLM_SEMANTIC = os.getenv("LIVE_USE_LLM_SEMANTIC", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
ALIGN_TEST_LOGIC = os.getenv("LIVE_ALIGN_TEST_LOGIC", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
LIVE_GATE_STATS_MODE = (
    os.getenv("LIVE_GATE_STATS_MODE", "dataset" if ALIGN_TEST_LOGIC else "dynamic").strip().lower()
    or ("dataset" if ALIGN_TEST_LOGIC else "dynamic")
)
LIVE_SYNC_ACCOUNT_STATE = os.getenv("LIVE_SYNC_ACCOUNT_STATE", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
LIVE_DYNAMIC_LOT = os.getenv("LIVE_DYNAMIC_LOT", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
LLM_DATASET_JSON = os.getenv("LIVE_LLM_DATASET_JSON", "").strip()
VISION_LLM_WS_URL = os.getenv(
    "VISION_LLM_WS_URL",
    "ws://localhost:8000/vision_llm/ws",
).strip()
BOT_WS_URL = os.getenv(
    "BOT_WS_URL",
    "ws://localhost:8000/bot/ws",
).strip()
BOT_CONFIG_ID = os.getenv("BOT_CONFIG_ID", "182bdab8-9274-4a4e-922f-700645086705").strip()


from backtest_config import (  # noqa: E402
    BAR_HISTORY,
    DATASETS_DIR,
    MODELS_DIR,
    PIP_VALUE,
    SPREAD_PIPS,
    TEST_DATA_FILE,
    TEST_DATE_FROM,
    TEST_DATE_TO,
    WINDOW_SIZE,
)


LLM_SEMANTIC_CACHE_FILE = os.getenv(
    "LIVE_LLM_SEMANTIC_CACHE_FILE",
    os.path.join(MODELS_DIR, "time_to_embedding_llm_cls.joblib"),
).strip() or os.path.join(MODELS_DIR, "time_to_embedding_llm_cls.joblib")
LLM_TEXT_LOG_FILE = os.getenv(
    "LIVE_LLM_TEXT_LOG_FILE",
    os.path.join(MODELS_DIR, "time_to_llm_text.jsonl"),
).strip() or os.path.join(MODELS_DIR, "time_to_llm_text.jsonl")
LLM_SEMANTIC_CACHE_SCHEMA = "utc_v2"
STATE_FILE = os.getenv("LIVE_STATE_FILE", os.path.join(MODELS_DIR, "run_live_state.json")).strip() or os.path.join(
    MODELS_DIR,
    "run_live_state.json",
)

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
