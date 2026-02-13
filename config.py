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
MAX_POSITION_PCT = 0.30          # Max 30% of portfolio per position
MAX_POSITIONS = 3                # Max concurrent open positions
DAILY_LOSS_LIMIT_PCT = 0.03     # Stop trading after 3% daily loss
STOP_LOSS_PCT = 0.025           # 2.5% stop-loss per trade
TAKE_PROFIT_PCT = 0.05          # 5% take-profit per trade
MIN_TRADE_VALUE = 1.00          # Minimum trade value in USD (Alpaca supports fractional)
MAX_TRADE_VALUE_PCT = 0.30      # Max single trade as % of equity

# ── Trailing Stop ─────────────────────────────────────────
TRAILING_BREAKEVEN_PCT = 0.03   # Move stop to breakeven when up 3%
TRAILING_TRAIL_PCT = 0.02       # Trail stop 2% behind highest price once up 5%
TRAILING_ACTIVATE_PCT = 0.05    # Activate trailing stop when up 5%

# ── Strategy Parameters ──────────────────────────────────
# Universe: liquid, low-price stocks suitable for small accounts
WATCHLIST = [
    "SOFI", "PLTR", "NIO", "RIVN", "LCID",      # momentum / growth (< $20)
    "F", "BAC", "T", "SNAP", "WBD",              # affordable value stocks
    "AMD", "INTC", "MU", "HOOD",                  # semi / fintech (moderate price)
    "MARA", "RIOT", "COIN",                        # crypto-adjacent volatility
    "SPY", "QQQ", "IWM",                           # ETFs — for context & trading
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

# Portfolio-based kill/upgrade thresholds (for live trading)
# These are percentages of STARTING equity (fetched on startup)
HARD_KILL_DRAWDOWN_PCT = 0.10   # If equity drops 10% from start, shut down permanently
MODEL_UPGRADE_GAIN_PCT = 0.20   # If equity grows 20% from start, upgrade scan model

# ── Logging ───────────────────────────────────────────────
LOG_DIR = "logs"
LOG_LEVEL = "INFO"
