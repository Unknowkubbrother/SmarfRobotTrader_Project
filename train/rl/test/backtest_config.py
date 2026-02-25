import os


WINDOW_SIZE = 20
INITIAL_BALANCE = 100
PIP_SIZE = 0.0001
PIP_VALUE = 10.0
LOT_RISK_PIPS = 50
SPREAD_PIPS = 2
BAR_HISTORY = 167

# Train/Test parity defaults for TradingEnv-based offline backtest.
# Override with env vars when needed.
TEST_INITIAL_BALANCE = float(os.getenv("TEST_INITIAL_BALANCE", "10000"))
_test_lot_size_raw = os.getenv("TEST_LOT_SIZE", "").strip()
TEST_LOT_SIZE = float(_test_lot_size_raw) if _test_lot_size_raw else None

RISK_LEVEL_LOW = "low"
RISK_LEVEL_MEDIUM = "medium"
RISK_LEVEL_HIGH = "high"

RISK_PERCENT_LOW = 0.5
RISK_PERCENT_MEDIUM = 1.0
RISK_PERCENT_HIGH = 2.0

_risk_level_raw = os.getenv("LIVE_RISK_LEVEL", os.getenv("RISK_LEVEL", "2")).strip().lower()
if _risk_level_raw in {"1", "low"}:
    RISK_LEVEL = RISK_LEVEL_LOW
elif _risk_level_raw in {"3", "high"}:
    RISK_LEVEL = RISK_LEVEL_HIGH
else:
    RISK_LEVEL = RISK_LEVEL_MEDIUM

RISK_PERCENT_BY_LEVEL = {
    RISK_LEVEL_LOW: RISK_PERCENT_LOW,
    RISK_LEVEL_MEDIUM: RISK_PERCENT_MEDIUM,
    RISK_LEVEL_HIGH: RISK_PERCENT_HIGH,
}
RISK_PERCENT = float(RISK_PERCENT_BY_LEVEL.get(RISK_LEVEL, RISK_PERCENT_MEDIUM))

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
RL_ROOT = os.path.dirname(TEST_DIR)
MODELS_DIR = os.path.join(RL_ROOT, "models")
DATASETS_DIR = os.path.join(RL_ROOT, "datasets")
OUTPUT_DIR = os.path.join(TEST_DIR, "outputs")
TEST_DATA_FILE = os.getenv("TEST_DATA_FILE", "h1_ohlc_delta1.csv").strip() or "h1_ohlc_delta1.csv"
TEST_DATE_FROM = os.getenv("TEST_DATE_FROM", "").strip()
TEST_DATE_TO = os.getenv("TEST_DATE_TO", "").strip()
DISABLE_PLOT = os.getenv("DISABLE_PLOT", "0").strip().lower() in {"1", "true", "yes"}
EMBED_TEST_MODE = "knn_map"
EMBED_SOURCE_MODE = "cls"

EMBED_QUALITY_MIN = 0.25

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
