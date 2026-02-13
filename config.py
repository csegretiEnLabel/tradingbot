"""
Configuration for the trading agent.
All tunable parameters in one place.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── API Keys ──────────────────────────────────────────────
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# ── Trading Mode ──────────────────────────────────────────
TRADING_MODE = os.getenv("TRADING_MODE", "paper")  # "paper" or "live"

ALPACA_BASE_URL = {
    "paper": "https://paper-api.alpaca.markets",
    "live": "https://api.alpaca.markets",
}[TRADING_MODE]

ALPACA_DATA_URL = "https://data.alpaca.markets"

# ── Risk Management ──────────────────────────────────────
MAX_POSITION_PCT = 0.20          # Max 20% of portfolio per position
MAX_POSITIONS = 3                # Max concurrent open positions
DAILY_LOSS_LIMIT_PCT = 0.03     # Stop trading after 3% daily loss
STOP_LOSS_PCT = 0.025           # 2.5% stop-loss per trade
TAKE_PROFIT_PCT = 0.05          # 5% take-profit per trade
MIN_TRADE_VALUE = 1.00          # Minimum trade value in USD (Alpaca supports fractional)
MAX_TRADE_VALUE_PCT = 0.20      # Max single trade as % of equity

# ── Strategy Parameters ──────────────────────────────────
# Universe: liquid, low-price stocks suitable for a $50 account
WATCHLIST = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",   # mega-cap tech (fractional)
    "NVDA", "AMD", "INTC",                       # semiconductors
    "SPY", "QQQ", "IWM",                          # ETFs
    "SOFI", "PLTR", "NIO", "RIVN",               # lower-priced momentum
    "F", "BAC", "T", "SNAP",                      # affordable stocks
]

# Technical indicator settings
RSI_PERIOD = 14
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
SMA_SHORT = 10
SMA_LONG = 30
VOLUME_SPIKE_MULTIPLIER = 1.5   # Volume must be 1.5x average

# ── Scheduling ────────────────────────────────────────────
ANALYSIS_INTERVAL_MIN = 60      # Run analysis every 60 minutes
MARKET_OPEN_HOUR = 9            # EST
MARKET_OPEN_MIN = 30
MARKET_CLOSE_HOUR = 16
PRE_CLOSE_MIN = 15              # Stop new trades 15 min before close

# ── Cost Management ──────────────────────────────────────
DAILY_API_BUDGET = float(os.getenv("DAILY_API_BUDGET", "0.20"))

# Claude model pricing (per 1M tokens)
MODEL_COSTS = {
    "claude-3-5-haiku-20241022": {"input": 1.00, "output": 5.00},
    "claude-sonnet-4-5-20250929": {"input": 3.00, "output": 15.00},
}

# Use Haiku for routine work, Sonnet for critical decisions
MODEL_SCAN = "claude-3-5-haiku-20241022"       # Market scanning (cheap)
MODEL_DECIDE = "claude-sonnet-4-5-20250929"    # Trade decisions (smarter)

# Kill switch: if API costs > trading profits for N consecutive days, shut down
COST_KILL_THRESHOLD_DAYS = 5

# "Rule of 10" — portfolio-based kill/upgrade thresholds (for live trading)
HARD_KILL_EQUITY = 45.0         # If equity drops below this, shut down permanently
MODEL_UPGRADE_EQUITY = 60.0     # If equity exceeds this, upgrade scan model to Sonnet

# ── Logging ───────────────────────────────────────────────
LOG_DIR = "logs"
LOG_LEVEL = "INFO"
