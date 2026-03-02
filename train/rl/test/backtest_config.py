import os


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


WINDOW_SIZE = 20
INITIAL_BALANCE = 100
PIP_SIZE = 0.0001
PIP_VALUE = 10.0
RISK_PIPS = _env_int("RISK_PIPS", 50)
SPREAD_PIPS = _env_float("SPREAD_PIPS", 2)
MAX_HOLD_STEPS = _env_int("MAX_HOLD_STEPS", 16)
BAR_HISTORY = 167
RISK_PERCENT = _env_float("RISK_PERCENT", 1.0)

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
RL_ROOT = os.path.dirname(TEST_DIR)
MODELS_DIR = os.path.join(RL_ROOT, "models")
DATASETS_DIR = os.path.join(RL_ROOT, "datasets")
OUTPUT_DIR = os.path.join(TEST_DIR, "outputs")
TEST_DATA_FILE = os.getenv("TEST_DATA_FILE", "h1_ohlc_delta1.csv").strip() or "h1_ohlc_delta1.csv"
TEST_DATE_FROM = os.getenv("TEST_DATE_FROM", "2025-01-01 00:00:00").strip()
TEST_DATE_TO = os.getenv("TEST_DATE_TO", "2026-02-27 23:00:00").strip()
DISABLE_PLOT = os.getenv("DISABLE_PLOT", "0").strip().lower() in {"1", "true", "yes"}
EMBED_TEST_MODE = "knn_map"
EMBED_SOURCE_MODE = "cls"

OPEN_PROB_THRESHOLD = _env_float("OPEN_PROB_THRESHOLD", 0.75)
OPEN_EDGE_THRESHOLD = _env_float("OPEN_EDGE_THRESHOLD", 0.16)
MIN_ACTION_MARGIN = _env_float("MIN_ACTION_MARGIN", 0.14)
HOLD_EDGE_THRESHOLD = _env_float("HOLD_EDGE_THRESHOLD", 0.05)
TRADE_COOLDOWN_BARS = _env_int("TRADE_COOLDOWN_BARS", 2)

ADAPTIVE_GATE = _env_bool("ADAPTIVE_GATE", True)
DEF_LOOKBACK_TRADES = _env_int("DEF_LOOKBACK_TRADES", 20)
DEF_MIN_WINRATE = _env_float("DEF_MIN_WINRATE", 0.50)
DEF_MIN_AVG_PIPS = _env_float("DEF_MIN_AVG_PIPS", 1.0)
DEF_MAX_LOSS_STREAK = _env_int("DEF_MAX_LOSS_STREAK", 3)
DEF_BARS = _env_int("DEF_BARS", 48)
DEF_CONF_BONUS = _env_float("DEF_CONF_BONUS", 0.04)
DEF_EDGE_BONUS = _env_float("DEF_EDGE_BONUS", 0.05)
DEF_MARGIN_BONUS = _env_float("DEF_MARGIN_BONUS", 0.04)
DEF_COOLDOWN_BONUS = _env_int("DEF_COOLDOWN_BONUS", 2)
VOL_CONF_BONUS = _env_float("VOL_CONF_BONUS", 0.02)
VOL_EDGE_BONUS = _env_float("VOL_EDGE_BONUS", 0.03)
VOL_MARGIN_BONUS = _env_float("VOL_MARGIN_BONUS", 0.02)
FLAT_CONF_BONUS = _env_float("FLAT_CONF_BONUS", 0.015)
FLAT_EDGE_BONUS = _env_float("FLAT_EDGE_BONUS", 0.02)
FLAT_MARGIN_BONUS = _env_float("FLAT_MARGIN_BONUS", 0.03)
TREND_RELAX = _env_float("TREND_RELAX", 0.02)
COUNTER_TREND_CONF_BONUS = _env_float("COUNTER_TREND_CONF_BONUS", 0.02)
COUNTER_TREND_EDGE_BONUS = _env_float("COUNTER_TREND_EDGE_BONUS", 0.04)
COUNTER_TREND_MARGIN_BONUS = _env_float("COUNTER_TREND_MARGIN_BONUS", 0.04)
COUNTER_TREND_HOLD_EDGE_BONUS = _env_float("COUNTER_TREND_HOLD_EDGE_BONUS", 0.03)
DEF_HOLD_EDGE_BONUS = _env_float("DEF_HOLD_EDGE_BONUS", 0.03)
VOL_HOLD_EDGE_BONUS = _env_float("VOL_HOLD_EDGE_BONUS", 0.02)
FLAT_HOLD_EDGE_BONUS = _env_float("FLAT_HOLD_EDGE_BONUS", 0.02)
TREND_HOLD_RELAX = _env_float("TREND_HOLD_RELAX", 0.01)

EMBED_QUALITY_MIN = _env_float("EMBED_QUALITY_MIN", 0.25)
EMBED_QUALITY_CONF_BONUS = _env_float("EMBED_QUALITY_CONF_BONUS", 0.10)
EMBED_QUALITY_EDGE_BONUS = _env_float("EMBED_QUALITY_EDGE_BONUS", 0.08)
EMBED_QUALITY_MARGIN_BONUS = _env_float("EMBED_QUALITY_MARGIN_BONUS", 0.06)
EMBED_QUALITY_HOLD_EDGE_BONUS = _env_float("EMBED_QUALITY_HOLD_EDGE_BONUS", 0.05)
EMBED_QUALITY_COOLDOWN_BONUS = _env_int("EMBED_QUALITY_COOLDOWN_BONUS", 2)

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
