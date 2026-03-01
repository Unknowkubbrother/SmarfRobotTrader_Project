import os
import sys


RL_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CORE_DIR = os.path.join(RL_ROOT, "core")
TEST_DIR = os.path.join(RL_ROOT, "test")
LLM_DIR = os.path.join(os.path.dirname(RL_ROOT), "llm")
if TEST_DIR not in sys.path:
    sys.path.insert(0, TEST_DIR)


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


from backtest_config import (  # noqa: E402
    BAR_HISTORY,
    DATASETS_DIR,
    MODELS_DIR,
    PIP_VALUE,
    RISK_PERCENT,
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
