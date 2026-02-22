import os


WINDOW_SIZE = 20
INITIAL_BALANCE = 100
PIP_SIZE = 0.0001
PIP_VALUE = 10.0
SL_PIPS = 50
TP_PIPS = 50
SPREAD_PIPS = 2
MAX_HOLD_STEPS = 30
BAR_HISTORY = 200
RISK_PERCENT = 1.0

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
RL_ROOT = os.path.dirname(TEST_DIR)
MODELS_DIR = os.path.join(RL_ROOT, "models")
DATASETS_DIR = os.path.join(RL_ROOT, "datasets")
OUTPUT_DIR = os.path.join(TEST_DIR, "outputs")
TEST_DATA_FILE = os.getenv("TEST_DATA_FILE", "h1_ohlc_delta1.csv").strip() or "h1_ohlc_delta1.csv"
TEST_DATE_FROM = os.getenv("TEST_DATE_FROM", "").strip()
TEST_DATE_TO = os.getenv("TEST_DATE_TO", "").strip()
DISABLE_PLOT = os.getenv("DISABLE_PLOT", "0").strip().lower() in {"1", "true", "yes"}
EMBED_TEST_MODE = os.getenv("EMBED_TEST_MODE", "knn_map").strip().lower() or "knn_map"

OPEN_PROB_THRESHOLD = float(os.getenv("OPEN_PROB_THRESHOLD", "0.86"))
OPEN_EDGE_THRESHOLD = float(os.getenv("OPEN_EDGE_THRESHOLD", "0.16"))
MIN_ACTION_MARGIN = float(os.getenv("MIN_ACTION_MARGIN", "0.20"))
HOLD_EDGE_THRESHOLD = float(os.getenv("HOLD_EDGE_THRESHOLD", "0.08"))
TRADE_COOLDOWN_BARS = int(os.getenv("TRADE_COOLDOWN_BARS", "10"))

ADAPTIVE_GATE = os.getenv("ADAPTIVE_GATE", "1").strip().lower() not in {"0", "false", "no"}
DEF_LOOKBACK_TRADES = int(os.getenv("DEF_LOOKBACK_TRADES", "20"))
DEF_MIN_WINRATE = float(os.getenv("DEF_MIN_WINRATE", "0.50"))
DEF_MIN_AVG_PIPS = float(os.getenv("DEF_MIN_AVG_PIPS", "1.0"))
DEF_MAX_LOSS_STREAK = int(os.getenv("DEF_MAX_LOSS_STREAK", "3"))
DEF_BARS = int(os.getenv("DEF_BARS", "24"))
DEF_CONF_BONUS = float(os.getenv("DEF_CONF_BONUS", "0.08"))
DEF_EDGE_BONUS = float(os.getenv("DEF_EDGE_BONUS", "0.05"))
DEF_MARGIN_BONUS = float(os.getenv("DEF_MARGIN_BONUS", "0.04"))
DEF_COOLDOWN_BONUS = int(os.getenv("DEF_COOLDOWN_BONUS", "2"))
VOL_CONF_BONUS = float(os.getenv("VOL_CONF_BONUS", "0.04"))
VOL_EDGE_BONUS = float(os.getenv("VOL_EDGE_BONUS", "0.03"))
VOL_MARGIN_BONUS = float(os.getenv("VOL_MARGIN_BONUS", "0.02"))
FLAT_CONF_BONUS = float(os.getenv("FLAT_CONF_BONUS", "0.03"))
FLAT_EDGE_BONUS = float(os.getenv("FLAT_EDGE_BONUS", "0.02"))
FLAT_MARGIN_BONUS = float(os.getenv("FLAT_MARGIN_BONUS", "0.03"))
TREND_RELAX = float(os.getenv("TREND_RELAX", "0.02"))
COUNTER_TREND_CONF_BONUS = float(os.getenv("COUNTER_TREND_CONF_BONUS", "0.05"))
COUNTER_TREND_EDGE_BONUS = float(os.getenv("COUNTER_TREND_EDGE_BONUS", "0.04"))
COUNTER_TREND_MARGIN_BONUS = float(os.getenv("COUNTER_TREND_MARGIN_BONUS", "0.04"))
COUNTER_TREND_HOLD_EDGE_BONUS = float(os.getenv("COUNTER_TREND_HOLD_EDGE_BONUS", "0.03"))
DEF_HOLD_EDGE_BONUS = float(os.getenv("DEF_HOLD_EDGE_BONUS", "0.03"))
VOL_HOLD_EDGE_BONUS = float(os.getenv("VOL_HOLD_EDGE_BONUS", "0.02"))
FLAT_HOLD_EDGE_BONUS = float(os.getenv("FLAT_HOLD_EDGE_BONUS", "0.02"))
TREND_HOLD_RELAX = float(os.getenv("TREND_HOLD_RELAX", "0.01"))

EMBED_QUALITY_MIN = float(os.getenv("EMBED_QUALITY_MIN", "0.30"))
EMBED_QUALITY_CONF_BONUS = float(os.getenv("EMBED_QUALITY_CONF_BONUS", "0.10"))
EMBED_QUALITY_EDGE_BONUS = float(os.getenv("EMBED_QUALITY_EDGE_BONUS", "0.08"))
EMBED_QUALITY_MARGIN_BONUS = float(os.getenv("EMBED_QUALITY_MARGIN_BONUS", "0.06"))
EMBED_QUALITY_HOLD_EDGE_BONUS = float(os.getenv("EMBED_QUALITY_HOLD_EDGE_BONUS", "0.05"))
EMBED_QUALITY_COOLDOWN_BONUS = int(os.getenv("EMBED_QUALITY_COOLDOWN_BONUS", "2"))

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
