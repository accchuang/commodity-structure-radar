"""
config.py — Central configuration for commodity-structure-radar.

All tunable parameters in one place. Every module imports from here.
No computation, no I/O — just constants.
"""

import os

# ── Project Root ─────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Monitored Symbols ────────────────────────────────────────────
# User manually updates contract codes before each trading cycle.
# Exchange codes: DCE 大连, CZCE 郑州, SHFE 上海, INE 上海国际能源
MONITORED_SYMBOLS = {
    # ── 黑色系 (Ferrous) ──
    "i":  {"name": "铁矿石",   "exchange": "DCE",  "contract": "I2609"},
    "rb": {"name": "螺纹钢",   "exchange": "SHFE", "contract": "RB2610"},
    "j":  {"name": "焦炭",     "exchange": "DCE",  "contract": "J2609"},
    "hc": {"name": "热卷",     "exchange": "SHFE", "contract": "HC2610"},
    "jm": {"name": "焦煤",     "exchange": "DCE",  "contract": "JM2609"},
    "fg": {"name": "玻璃",     "exchange": "CZCE", "contract": "FG2609"},
    "sa": {"name": "纯碱",     "exchange": "CZCE", "contract": "SA2609"},
    "sm": {"name": "锰硅",     "exchange": "CZCE", "contract": "SM2609"},
    "si": {"name": "硅铁",     "exchange": "CZCE", "contract": "SF2609"},
    # ── 油脂油料 (Oils & Oilseeds) ──
    "y":  {"name": "豆油",     "exchange": "DCE",  "contract": "Y2609"},
    "p":  {"name": "棕榈油",   "exchange": "DCE",  "contract": "P2609"},
    "oi": {"name": "菜籽油",   "exchange": "CZCE", "contract": "OI2609"},
    # ── 农产品 (Agricultural) ──
    "m":  {"name": "豆粕",     "exchange": "DCE",  "contract": "M2609"},
    "rm": {"name": "菜粕",     "exchange": "CZCE", "contract": "RM2609"},
    "a":  {"name": "黄大豆1号","exchange": "DCE",  "contract": "A2609"},
    "b":  {"name": "黄大豆2号","exchange": "DCE",  "contract": "B2609"},
    "c":  {"name": "玉米",     "exchange": "DCE",  "contract": "C2609"},
    "cs": {"name": "玉米淀粉", "exchange": "DCE",  "contract": "CS2609"},
    "jd": {"name": "鸡蛋",     "exchange": "DCE",  "contract": "JD2609"},
    "lh": {"name": "生猪",     "exchange": "DCE",  "contract": "LH2609"},
    # ── 软商品 (Soft Commodities) ──
    "sr": {"name": "白糖",     "exchange": "CZCE", "contract": "SR2609"},
    "cf": {"name": "棉花",     "exchange": "CZCE", "contract": "CF2609"},
    "ap": {"name": "苹果",     "exchange": "CZCE", "contract": "AP2610"},
    "ru": {"name": "橡胶",     "exchange": "SHFE", "contract": "RU2609"},
    "nr": {"name": "20号胶",   "exchange": "INE",  "contract": "NR2609"},
    "sp": {"name": "纸浆",     "exchange": "SHFE", "contract": "SP2609"},
    "ur": {"name": "尿素",     "exchange": "CZCE", "contract": "UR2609"},
    # ── 化工 (Chemicals) ──
    "ma": {"name": "甲醇",     "exchange": "CZCE", "contract": "MA2609"},
    "ta": {"name": "PTA",      "exchange": "CZCE", "contract": "TA2609"},
    "eg": {"name": "乙二醇",   "exchange": "DCE",  "contract": "EG2609"},
    "eb": {"name": "苯乙烯",   "exchange": "DCE",  "contract": "EB2609"},
    "pp": {"name": "聚丙烯",   "exchange": "DCE",  "contract": "PP2609"},
    "l":  {"name": "塑料",     "exchange": "DCE",  "contract": "L2609"},
    "v":  {"name": "PVC",      "exchange": "DCE",  "contract": "V2609"},
    # ── 能源 (Energy) ──
    "sc": {"name": "原油",     "exchange": "INE",  "contract": "SC2609"},
    "fu": {"name": "燃料油",   "exchange": "INE",  "contract": "FU2609"},
    "pg": {"name": "液化气",   "exchange": "DCE",  "contract": "PG2609"},
}

# Human-readable labels for the dropdown
SYMBOL_NAMES = {k: v["name"] for k, v in MONITORED_SYMBOLS.items()}

# ── Trading Sessions (CST / Shanghai time) ───────────────────────
# Each tuple is (start_time, end_time) as "HH:MM"
MORNING_SESSION_1 = ("09:00", "10:15")
MORNING_SESSION_2 = ("10:30", "11:30")
AFTERNOON_SESSION = ("13:30", "15:00")
NIGHT_SESSION = ("21:00", "23:00")

ALL_SESSIONS = [
    MORNING_SESSION_1,
    MORNING_SESSION_2,
    AFTERNOON_SESSION,
    NIGHT_SESSION,
]

# ── Stop-Run Detection Thresholds ─────────────────────────────────
STOPRUN_CONFIRM_WINDOW = 2          # max candles after pierce for rejection
DEFAULT_WICK_BODY_RATIO = 0.6       # min wick/(high-low) for valid rejection
KEY_LEVEL_PIERCE_TICKS = 3          # tick proximity to consider "at level"
STOPRUN_MAX_DURATION_MINUTES = 90   # hard cap on monitoring window

# ── Institutional Data Parameters ─────────────────────────────────
TOP_N_CONCENTRATION = 3             # top-N brokers for CR calculation
CR_HIGH_THRESHOLD = 0.40            # CR above this = concentrated
CR_LOW_THRESHOLD = 0.20            # CR below this = fragmented (skip)

# ── Key Levels ───────────────────────────────────────────────────
POC_LOOKBACK_BARS = 20              # hourly bars for volume profile POC
POC_PRICE_BUCKETS = 50              # number of price buckets for profile
HVN_VOLUME_RATIO = 0.70             # % of max volume for High Volume Node

# ── Data Paths (relative to project root) ─────────────────────────
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
BIAS_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "bias")
BIAS_FILE_NAME = "daily_bias.json"
ALERT_LOG_PATH = os.path.join(PROJECT_ROOT, "data", "alerts.jsonl")

# ── AkShare API Configuration ────────────────────────────────────
AKSHARE_REQUEST_DELAY = 0.5         # seconds between API calls
AKSHARE_MAX_RETRIES = 2             # retry on transient errors

# ── TqSdk Configuration ───────────────────────────────────────────
# K-line durations in seconds
TQSDK_KLINE_30M = 1800
TQSDK_KLINE_1H = 3600
TQSDK_RECONNECT_INTERVAL = 10       # seconds between reconnect attempts
TQSDK_QUOTE_TIMEOUT = 5             # wait_update timeout seconds

# TqSdk account (optional — anonymous mode for delayed data)
# Set TQSDK_USERNAME and TQSDK_PASSWORD env vars, or fill here.
TQSDK_USERNAME = os.environ.get("TQSDK_USERNAME", "")
TQSDK_PASSWORD = os.environ.get("TQSDK_PASSWORD", "")

# ── Alert Configuration ───────────────────────────────────────────
# suppress same (symbol, level) within N min
ALERT_DEDUP_WINDOW_MINUTES = 120
ALERT_MAX_ACTIVE = 50               # max alerts retained in memory

# ── Structural Analysis Configuration ────────────────────────────
STRUCTURE_LOOKBACK_DAYS = 20        # max historical days to load for analysis
CR_TREND_WINDOW = 10               # rolling window for CR slope calculation
CR_LOOSENING_THRESHOLD = -0.02     # CR slope below this = "loosening" (per day)
CR_TIGHTENING_THRESHOLD = 0.02     # CR slope above this = "tightening" (per day)
HHI_HIGH_CONCENTRATION = 0.25      # HHI above this = highly concentrated
BROKER_RANK_LOOKBACK = 5           # days to track broker rank changes

# ── Broker Defection & Price Correlation Configuration ───────────
DEFECTION_NET_FLIP_THRESHOLD = 0    # net_pos crosses 0 → direction flip
DEFECTION_MOMENTUM_WINDOW = 5       # days to assess "consistent" direction
DEFECTION_SHRINK_THRESHOLD = 0.30   # 30%+ reduction in net position = unwind
PRICE_CORR_LAG_DAYS = [1, 3, 5]    # lag windows for price correlation analysis
PRICE_CORR_MIN_DAYS = 8            # min overlapping days for valid correlation
# NOTE: keep below (available_overlap - max_lag) to leave enough
# valid rows after forward-return shift. With 10 days: lag=1→9 valid, lag=5→5 valid.
SMART_MONEY_CORR_THRESHOLD = 0.30  # |correlation| above this = "smart money"
PRICE_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "price")  # daily price parquet

# ── P2: Microstructure Depth Indicators ─────────────────────────
TURNOVER_HIGH_THRESHOLD = 0.30     # turnover rate > 30% = high churn
TURNOVER_SPIKE_THRESHOLD = 0.50    # turnover > 50% = violent repositioning
NEW_ENTRANT_LOOKBACK = 5           # days to track cumulative new entrants
NEW_ENTRANT_TOP_N = 20             # how many top brokers to track for entry/exit
DIVERGENCE_SHRINK_THRESHOLD = 0.05 # CR_diff shrinking >5% = balance shifting

# ── UI Configuration ─────────────────────────────────────────────
UI_REFRESH_INTERVAL_SECONDS = 3     # Streamlit rerun interval
UI_DARK_THEME = True                # use plotly_dark + dark sidebar
UI_CHART_CANDLES = 100             # number of candles to display in chart
