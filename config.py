import os
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")

TICKERS = ["NVDA", "AMD", "INTC", "QCOM", "AVGO", "TSM", "ASML", "MU", "AMAT", "LRCX"]
ETFS = ["SOXX", "SMH", "SOXS", "SOXL"]
ALL_SYMBOLS = TICKERS + ETFS

# Inverse ETFs — sell signal flips to buy and vice versa for underlying
INVERSE_ETFS = {"SOXS"}
LEVERAGED_LONG_ETFS = {"SOXL"}

FACTOR_CATEGORIES = [
    "earnings",
    "tariffs",
    "geopolitics",
    "supply_chain",
    "ai_demand",
    "inventory",
    "rates",
    "china_risk",
    "capex",
    "analyst_actions",
    "macro",
    "regulatory",
]

CATEGORY_WEIGHTS = {
    "earnings": 1.4,
    "tariffs": 1.2,
    "geopolitics": 1.2,
    "ai_demand": 1.2,
    "supply_chain": 1.1,
    "inventory": 1.1,
    "china_risk": 1.1,
    "capex": 1.0,
    "analyst_actions": 1.0,
    "rates": 0.9,
    "macro": 0.9,
    "regulatory": 1.0,
}

SCHEDULE_HOURS = int(os.getenv("SCHEDULE_HOURS", "6"))
LOOKBACK_DAYS = 14
FORWARD_WINDOW_DAYS = 14

BUY_THRESHOLD = 0.35
SELL_THRESHOLD = -0.35

DB_PATH = os.getenv("DB_PATH", "data/factor_history.db")
REPORT_OUTPUT_DIR = os.getenv("REPORT_OUTPUT_DIR", "data/reports")
HISTORY_RETENTION_DAYS = 90
