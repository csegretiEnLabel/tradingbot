"""
Technical analysis and signal generation.
Computes indicators on DAILY bars (primary) and uses intraday for timing context.
Includes signal caching to reduce redundant API calls.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

try:
    import ta
    HAS_TA = True
except ImportError:
    HAS_TA = False

import config

logger = logging.getLogger(__name__)

# -- Signal Cache -----------------------------------------
# Caches bar data for 15 minutes to avoid redundant Alpaca calls
_bar_cache: dict[str, tuple[float, pd.DataFrame]] = {}
CACHE_TTL_SECONDS = 900  # 15 minutes


def _get_cached_bars(trader, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    """Fetch bars with caching. Returns cached data if within TTL. Evicts stale entries."""
    cache_key = f"{symbol}_{timeframe}_{limit}"
    now = time.time()

    if cache_key in _bar_cache:
        cached_time, cached_df = _bar_cache[cache_key]
        if now - cached_time < CACHE_TTL_SECONDS:
            return cached_df

    # Evict stale entries (older than 2x TTL) to prevent unbounded growth
    stale_keys = [k for k, (t, _) in _bar_cache.items() if now - t > CACHE_TTL_SECONDS * 2]
    for k in stale_keys:
        del _bar_cache[k]

    df = trader.get_bars(symbol, timeframe=timeframe, limit=limit)
    _bar_cache[cache_key] = (now, df)
    return df


@dataclass
class Signal:
    symbol: str
    action: str          # "buy", "sell", "hold"
    strength: float      # 0.0 to 1.0
    reasons: list[str]
    indicators: dict
    price: float
    atr_pct: float       # Average True Range as % of price (volatility)


def compute_indicators(df: pd.DataFrame) -> dict:
    """
    Compute all technical indicators for a price DataFrame.
    Returns a dict of current values.
    """
    if df.empty or len(df) < config.SMA_LONG:
        return {}

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    indicators = {}

    # -- Moving Averages ------------------------------
    indicators["sma_short"] = close.rolling(config.SMA_SHORT).mean().iloc[-1]
    indicators["sma_long"] = close.rolling(config.SMA_LONG).mean().iloc[-1]
    indicators["price"] = close.iloc[-1]
    indicators["sma_crossover"] = indicators["sma_short"] > indicators["sma_long"]

    # -- RSI -------------------------------------------
    if HAS_TA:
        rsi_series = ta.momentum.RSIIndicator(close, window=config.RSI_PERIOD).rsi()
        indicators["rsi"] = rsi_series.iloc[-1] if not rsi_series.empty else 50.0

        # MACD
        macd = ta.trend.MACD(
            close,
            window_slow=config.MACD_SLOW,
            window_fast=config.MACD_FAST,
            window_sign=config.MACD_SIGNAL,
        )
        indicators["macd"] = macd.macd().iloc[-1]
        indicators["macd_signal"] = macd.macd_signal().iloc[-1]
        indicators["macd_histogram"] = macd.macd_diff().iloc[-1]
        indicators["macd_bullish"] = indicators["macd"] > indicators["macd_signal"]

        # Bollinger Bands
        bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
        indicators["bb_upper"] = bb.bollinger_hband().iloc[-1]
        indicators["bb_lower"] = bb.bollinger_lband().iloc[-1]
        indicators["bb_mid"] = bb.bollinger_mavg().iloc[-1]
        indicators["bb_pct"] = (close.iloc[-1] - indicators["bb_lower"]) / (
            indicators["bb_upper"] - indicators["bb_lower"]
        ) if indicators["bb_upper"] != indicators["bb_lower"] else 0.5

        # ATR for volatility
        atr = ta.volatility.AverageTrueRange(high, low, close, window=14)
        indicators["atr"] = atr.average_true_range().iloc[-1]
        indicators["atr_pct"] = indicators["atr"] / close.iloc[-1] if close.iloc[-1] > 0 else 0

        # ADX for trend strength / regime detection
        adx_indicator = ta.trend.ADXIndicator(high, low, close, window=14)
        indicators["adx"] = adx_indicator.adx().iloc[-1]
        indicators["di_plus"] = adx_indicator.adx_pos().iloc[-1]
        indicators["di_minus"] = adx_indicator.adx_neg().iloc[-1]

    else:
        # Fallback RSI calculation
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(config.RSI_PERIOD).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(config.RSI_PERIOD).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        indicators["rsi"] = rsi.iloc[-1] if not rsi.empty else 50.0
        indicators["atr_pct"] = 0.02  # default
        indicators["macd_bullish"] = False
        indicators["adx"] = 25.0  # default

    # -- Volume Analysis -------------------------------
    avg_volume = volume.rolling(20).mean().iloc[-1]
    current_volume = volume.iloc[-1]
    indicators["volume_ratio"] = current_volume / avg_volume if avg_volume > 0 else 1.0
    indicators["volume_spike"] = indicators["volume_ratio"] > config.VOLUME_SPIKE_MULTIPLIER

    # -- Price Change ----------------------------------
    indicators["change_1d"] = (close.iloc[-1] / close.iloc[-2] - 1) if len(close) >= 2 else 0
    indicators["change_5d"] = (close.iloc[-1] / close.iloc[-5] - 1) if len(close) >= 5 else 0
    indicators["change_20d"] = (close.iloc[-1] / close.iloc[-20] - 1) if len(close) >= 20 else 0

    return indicators


def detect_regime(indicators: dict) -> str:
    """
    Simple market regime detection using ADX.
    Returns 'trending', 'ranging', or 'volatile'.
    """
    adx = indicators.get("adx", 25)
    atr_pct = indicators.get("atr_pct", 0.02)

    if adx > 30:
        return "trending"
    elif adx < 20 and atr_pct < 0.025:
        return "ranging"
    else:
        return "volatile"


def generate_signal(symbol: str, indicators: dict, price: float) -> Signal:
    """
    Generate a trading signal based on technical indicators.
    Adapts signal weights based on market regime.
    This is a rule-based pre-filter; the AI agent makes the final call.
    """
    if not indicators:
        return Signal(symbol, "hold", 0.0, ["Insufficient data"], {}, price, 0.02)

    reasons = []
    buy_score = 0.0
    sell_score = 0.0

    rsi = indicators.get("rsi", 50)
    sma_cross = indicators.get("sma_crossover", False)
    macd_bull = indicators.get("macd_bullish", False)
    volume_spike = indicators.get("volume_spike", False)
    bb_pct = indicators.get("bb_pct", 0.5)
    change_5d = indicators.get("change_5d", 0)

    # Detect regime and adjust weights
    regime = detect_regime(indicators)
    indicators["regime"] = regime

    # -- Buy Signals ----------------------------------
    if regime == "trending":
        # In trends: favor momentum (SMA, MACD) over mean-reversion (RSI)
        if sma_cross:
            buy_score += 0.3
            reasons.append("SMA bullish crossover (trending)")
        if macd_bull:
            buy_score += 0.25
            reasons.append("MACD bullish (trending)")
        if rsi < config.RSI_OVERSOLD:
            buy_score += 0.15
            reasons.append(f"RSI oversold ({rsi:.1f})")
    elif regime == "ranging":
        # In ranges: favor mean-reversion (RSI, BB) over momentum
        if rsi < config.RSI_OVERSOLD:
            buy_score += 0.35
            reasons.append(f"RSI oversold ({rsi:.1f}) (ranging)")
        if bb_pct < 0.15:
            buy_score += 0.25
            reasons.append(f"Near lower BB ({bb_pct:.2f}) (ranging)")
        if macd_bull:
            buy_score += 0.1
            reasons.append("MACD bullish")
    else:
        # Volatile: standard weights but require stronger signals
        if rsi < config.RSI_OVERSOLD:
            buy_score += 0.25
            reasons.append(f"RSI oversold ({rsi:.1f})")
        if sma_cross:
            buy_score += 0.2
            reasons.append("SMA short > SMA long (bullish)")
        if macd_bull:
            buy_score += 0.2
            reasons.append("MACD bullish crossover")
        if bb_pct < 0.2:
            buy_score += 0.1
            reasons.append(f"Near lower Bollinger Band ({bb_pct:.2f})")

    # Volume confirmation (all regimes)
    if volume_spike and change_5d > 0:
        buy_score += 0.15
        reasons.append("Volume spike with positive momentum")

    # Daily trend alignment bonus
    daily_trend = indicators.get("daily_trend_up")
    if daily_trend is True:
        buy_score += 0.1
        reasons.append("Aligned with daily uptrend")
    elif daily_trend is False:
        buy_score -= 0.15  # Stronger penalty for fighting trend
        reasons.append("Against daily trend (caution)")

    # -- Sell Signals ---------------------------------
    if rsi > config.RSI_OVERBOUGHT:
        sell_score += 0.3
        reasons.append(f"RSI overbought ({rsi:.1f})")

    if not sma_cross:
        sell_score += 0.15
        reasons.append("SMA short < SMA long (bearish)")

    if not macd_bull:
        sell_score += 0.15
        reasons.append("MACD bearish")

    if bb_pct > 0.8:
        sell_score += 0.15
        reasons.append(f"Near upper Bollinger Band ({bb_pct:.2f})")

    # -- Determine Action -----------------------------
    atr_pct = indicators.get("atr_pct", 0.02)

    # Require higher threshold in volatile regime
    # Require higher threshold in volatile regime
    if config.STRATEGY_MODE == "aggressive":
        min_score = 0.25 # Aggressive "Apex Predator" mode
    else:
        # Default "Preservation" mode
        min_score = 0.45 if regime == "volatile" else 0.4

    if buy_score > sell_score and buy_score >= min_score:
        return Signal(symbol, "buy", buy_score, reasons, indicators, price, atr_pct)
    elif sell_score > buy_score and sell_score >= min_score:
        return Signal(symbol, "sell", sell_score, reasons, indicators, price, atr_pct)
    else:
        return Signal(symbol, "hold", max(buy_score, sell_score), reasons or ["No strong signal"], indicators, price, atr_pct)


def scan_universe(trader) -> list[Signal]:
    """
    Scan the entire watchlist and return ranked signals.
    Uses DAILY bars as the primary signal source (correct timeframe for RSI/MACD/SMA).
    Overlays intraday bars for timing context.
    """
    signals = []

    for symbol in config.WATCHLIST:
        try:
            # Daily bars: primary signal source (RSI-14, SMA-10/30, MACD all on daily)
            df_daily = _get_cached_bars(trader, symbol, timeframe="1Day", limit=config.HISTORY_BARS_LIMIT)
            # Intraday bars for timing context
            df_intraday = _get_cached_bars(trader, symbol, timeframe="1Hour", limit=24)

            # Use daily for indicator calculation (correct timeframe for RSI-14 etc.)
            if df_daily.empty:
                continue

            indicators = compute_indicators(df_daily)
            if not indicators:
                continue

            # Overlay intraday timing context
            if not df_intraday.empty and len(df_intraday) >= 5:
                intraday_close = df_intraday["close"]
                # Intraday momentum: how is the stock moving today?
                indicators["intraday_change"] = (intraday_close.iloc[-1] / intraday_close.iloc[0] - 1) \
                    if len(intraday_close) >= 2 else 0
                # Use latest intraday price as "current" price
                indicators["price"] = intraday_close.iloc[-1]

            # Daily trend context (SMA comparison on daily bars)
            if len(df_daily) >= config.SMA_LONG:
                daily_close = df_daily["close"]
                indicators["daily_sma_short"] = daily_close.rolling(config.SMA_SHORT).mean().iloc[-1]
                indicators["daily_sma_long"] = daily_close.rolling(config.SMA_LONG).mean().iloc[-1]
                indicators["daily_trend_up"] = indicators["daily_sma_short"] > indicators["daily_sma_long"]
            else:
                indicators["daily_trend_up"] = None  # Unknown

            price = indicators.get("price", 0)
            signal = generate_signal(symbol, indicators, price)
            signals.append(signal)

        except Exception as e:
            logger.error(f"Error scanning {symbol}: {e}")
            continue

    # Sort by signal strength, buy signals first
    buy_signals = sorted(
        [s for s in signals if s.action == "buy"],
        key=lambda s: s.strength,
        reverse=True,
    )
    sell_signals = sorted(
        [s for s in signals if s.action == "sell"],
        key=lambda s: s.strength,
        reverse=True,
    )

    logger.info(f"Scan complete: {len(buy_signals)} buy, {len(sell_signals)} sell, {len(signals) - len(buy_signals) - len(sell_signals)} hold")

    return buy_signals + sell_signals
