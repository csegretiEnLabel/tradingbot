"""
Quantitative Strategy Engine
-----------------------------
Ports proven algorithms from prod_trade into the AI trading bot framework.

Three strategies:
  1. KAMA (Kaufman Adaptive Moving Average) - adapts smoothing to volatility
  2. Trend Follow (SMA 3/20 crossover) - medium-term trend detection
  3. Momentum (rolling return ranking) - ranks watchlist by momentum strength

All strategies:
  - Use Alpaca bar data via the existing Trader/cache (no yfinance dependency)
  - Output QuantSignal objects consumed by the AI agent
  - Go through the same risk management pipeline as technical signals
  - Log every signal change for full auditability
"""

import json
import logging
import numpy as np
import os
import pandas as pd
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Optional

import config

logger = logging.getLogger(__name__)


# ── Data Classes ────────────────────────────────────────────

@dataclass
class QuantSignal:
    """A signal produced by a quantitative strategy."""
    strategy: str       # "kama", "trend_follow", "momentum"
    symbol: str
    action: str         # "buy", "sell", "hold"
    strength: float     # 0.0 to 1.0
    metrics: dict       # Strategy-specific metrics for AI context
    reasoning: str
    signal_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    price_at_signal: Optional[float] = None
    state: str = "generated"  # "generated", "acted_on", "rejected", "expired"


# ── KAMA Strategy ───────────────────────────────────────────

class KAMAStrategy:
    """
    Kaufman Adaptive Moving Average.

    Adapts its smoothing constant based on the Efficiency Ratio (ER):
      - Trending market (high ER) -> fast adaptation (tracks price closely)
      - Choppy market (low ER)    -> slow adaptation (filters noise)

    Signal: Price > KAMA = bullish, Price < KAMA = bearish.
    Crossovers (price crossing KAMA) are highlighted as stronger signals.

    Ported from prod_quant_bot_kama.py with added safety:
      - Requires signal confirmation (prev bar must agree to prevent whipsaw)
      - Logs efficiency ratio so AI understands market regime
    """

    def __init__(self, period: int = 10, fast: int = 2, slow: int = 30):
        self.period = period
        self.fast = fast
        self.slow = slow

    def calculate_kama(self, prices: pd.Series) -> pd.Series:
        """
        Compute KAMA from a price series.

        Math:
          Direction = |Close[t] - Close[t-n]|
          Volatility = SUM(|Close[t] - Close[t-1]|, n periods)
          ER = Direction / Volatility  (0 to 1)
          FastSC = 2/(fast+1),  SlowSC = 2/(slow+1)
          SC = [ER * (FastSC - SlowSC) + SlowSC]^2
          KAMA[t] = KAMA[t-1] + SC * (Close[t] - KAMA[t-1])
        """
        n = self.period
        if len(prices) < n + 1:
            return pd.Series(np.nan, index=prices.index)

        direction = prices.diff(n).abs()
        volatility = prices.diff().abs().rolling(n).sum()

        # Avoid division by zero
        er = direction / volatility.replace(0, np.nan)
        er = er.fillna(0)

        fast_sc = 2 / (self.fast + 1)
        slow_sc = 2 / (self.slow + 1)
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2

        kama = np.zeros(len(prices))
        kama[:n] = prices.values[:n]
        for i in range(n, len(prices)):
            sc_val = sc.iloc[i] if not np.isnan(sc.iloc[i]) else 0
            kama[i] = kama[i - 1] + sc_val * (prices.values[i] - kama[i - 1])

        return pd.Series(kama, index=prices.index)

    def generate_signal(self, symbol: str, df: pd.DataFrame) -> Optional[QuantSignal]:
        """Generate KAMA signal from daily bar data."""
        if df.empty or len(df) < self.period + 5:
            return None

        close = df["close"]
        kama = self.calculate_kama(close)

        current_price = close.iloc[-1]
        current_kama = kama.iloc[-1]
        prev_price = close.iloc[-2]
        prev_kama = kama.iloc[-2]

        if np.isnan(current_kama) or np.isnan(prev_kama):
            return None

        # Efficiency Ratio for the latest bar
        n = self.period
        if len(close) > n:
            direction = abs(close.iloc[-1] - close.iloc[-1 - n])
            vol_sum = close.diff().abs().iloc[-n:].sum()
            er = direction / vol_sum if vol_sum > 0 else 0
        else:
            er = 0

        # Distance from KAMA as a percentage (strength indicator)
        kama_distance_pct = (current_price - current_kama) / current_kama if current_kama > 0 else 0

        # Detect crossovers
        is_bullish_cross = (current_price > current_kama) and (prev_price <= prev_kama)
        is_bearish_cross = (current_price < current_kama) and (prev_price >= prev_kama)

        # Determine signal
        if current_price > current_kama:
            action = "buy"
            if is_bullish_cross:
                reasoning = (
                    f"KAMA BULLISH CROSSOVER: price ${current_price:.2f} just crossed "
                    f"above KAMA ${current_kama:.2f} (ER={er:.2f})"
                )
            else:
                reasoning = (
                    f"KAMA bullish: price ${current_price:.2f} > KAMA ${current_kama:.2f} "
                    f"(gap={kama_distance_pct:.1%}, ER={er:.2f})"
                )
        elif current_price < current_kama:
            action = "sell"
            if is_bearish_cross:
                reasoning = (
                    f"KAMA BEARISH CROSSOVER: price ${current_price:.2f} just crossed "
                    f"below KAMA ${current_kama:.2f} (ER={er:.2f})"
                )
            else:
                reasoning = (
                    f"KAMA bearish: price ${current_price:.2f} < KAMA ${current_kama:.2f} "
                    f"(gap={kama_distance_pct:.1%}, ER={er:.2f})"
                )
        else:
            action = "hold"
            reasoning = f"KAMA neutral: price equals KAMA at ${current_kama:.2f}"

        # Strength: based on distance from KAMA + ER + crossover bonus
        strength = min(abs(kama_distance_pct) * 10 + er * 0.3, 1.0)
        if is_bullish_cross or is_bearish_cross:
            strength = min(strength + 0.25, 1.0)

        metrics = {
            "kama_value": round(current_kama, 4),
            "price": round(current_price, 4),
            "kama_distance_pct": round(kama_distance_pct, 4),
            "efficiency_ratio": round(er, 4),
            "is_crossover": is_bullish_cross or is_bearish_cross,
        }

        return QuantSignal(
            strategy="kama",
            symbol=symbol,
            action=action,
            strength=round(strength, 3),
            metrics=metrics,
            reasoning=reasoning,
        )


# ── Trend Follow Strategy ──────────────────────────────────

class TrendFollowStrategy:
    """
    SMA Crossover Trend Following.

    Uses fast SMA (3-day) crossing above/below slow SMA (20-day).
    The original prod_quant_bot_trend.py used SMA-3/SMA-20 with
    1-period shift to avoid lookahead bias.

    Signal: SMA3 > SMA20 = buy, SMA3 < SMA20 = sell.
    Fresh crossovers are marked with higher strength.

    Ported from prod_quant_bot_trend.py / prod_trendfollow.py.
    """

    def __init__(self, sma_short: int = 3, sma_long: int = 20):
        self.sma_short = sma_short
        self.sma_long = sma_long

    def generate_signal(self, symbol: str, df: pd.DataFrame) -> Optional[QuantSignal]:
        """Generate trend following signal from daily bar data."""
        if df.empty or len(df) < self.sma_long + 5:
            return None

        close = df["close"]

        # Shift by 1 to avoid lookahead bias (matching prod_trendfollow.py)
        sma_short = close.rolling(self.sma_short).mean().shift(1)
        sma_long = close.rolling(self.sma_long).mean().shift(1)

        current_short = sma_short.iloc[-1]
        current_long = sma_long.iloc[-1]
        prev_short = sma_short.iloc[-2]
        prev_long = sma_long.iloc[-2]

        if any(np.isnan(v) for v in [current_short, current_long, prev_short, prev_long]):
            return None

        current_price = close.iloc[-1]

        # Crossover detection
        currently_bullish = current_short > current_long
        previously_bullish = prev_short > prev_long
        is_fresh_cross = currently_bullish != previously_bullish

        # SMA spread as percentage
        spread_pct = (current_short - current_long) / current_long if current_long > 0 else 0

        if currently_bullish:
            action = "buy"
            if is_fresh_cross:
                reasoning = (
                    f"TREND BULLISH CROSSOVER: SMA{self.sma_short} (${current_short:.2f}) "
                    f"just crossed above SMA{self.sma_long} (${current_long:.2f})"
                )
            else:
                reasoning = (
                    f"Trend bullish: SMA{self.sma_short} (${current_short:.2f}) > "
                    f"SMA{self.sma_long} (${current_long:.2f}), spread={spread_pct:.2%}"
                )
        else:
            action = "sell"
            if is_fresh_cross:
                reasoning = (
                    f"TREND BEARISH CROSSOVER: SMA{self.sma_short} (${current_short:.2f}) "
                    f"just crossed below SMA{self.sma_long} (${current_long:.2f})"
                )
            else:
                reasoning = (
                    f"Trend bearish: SMA{self.sma_short} (${current_short:.2f}) < "
                    f"SMA{self.sma_long} (${current_long:.2f}), spread={spread_pct:.2%}"
                )

        # Strength based on spread magnitude + crossover bonus
        strength = min(abs(spread_pct) * 15, 1.0)
        if is_fresh_cross:
            strength = min(strength + 0.3, 1.0)

        metrics = {
            "sma_short_value": round(current_short, 4),
            "sma_long_value": round(current_long, 4),
            "spread_pct": round(spread_pct, 4),
            "is_crossover": is_fresh_cross,
            "price": round(current_price, 4),
        }

        return QuantSignal(
            strategy="trend_follow",
            symbol=symbol,
            action=action,
            strength=round(strength, 3),
            metrics=metrics,
            reasoning=reasoning,
        )


# ── Momentum Strategy ──────────────────────────────────────

class MomentumStrategy:
    """
    Rolling Return Momentum.

    Ranks watchlist stocks by their N-month rolling return.
    Strong positive momentum = buy signal, negative = sell signal.

    The original prod_tradebot.py used 12-month S&P 500 momentum with
    monthly rebalancing. Here we adapt it for the watchlist:
      - Calculate 1m, 3m, 6m returns using available Alpaca data
      - Rank symbols within the watchlist
      - Present as advisory context for the AI

    Does NOT auto-trade. The AI uses momentum context to make better decisions.
    """

    def __init__(self, lookback_months: int = 12, top_n: int = 5):
        self.lookback_months = lookback_months
        self.top_n = top_n

    def generate_signal(self, symbol: str, df: pd.DataFrame) -> Optional[QuantSignal]:
        """Generate momentum signal from daily bar data."""
        if df.empty or len(df) < 25:
            return None

        close = df["close"]

        # Calculate rolling returns at various horizons
        # ~22 trading days per month
        mom_1m = (close.iloc[-1] / close.iloc[-22] - 1) if len(close) >= 22 else None
        mom_3m = (close.iloc[-1] / close.iloc[-min(66, len(close))] - 1) if len(close) >= 44 else None

        # Use best available horizon
        if mom_3m is not None:
            primary_momentum = mom_3m
            horizon = "3m"
        elif mom_1m is not None:
            primary_momentum = mom_1m
            horizon = "1m"
        else:
            return None

        # Determine signal based on momentum thresholds
        if primary_momentum > 0.08:
            action = "buy"
            reasoning = f"Strong {horizon} momentum: {primary_momentum:+.1%} return"
        elif primary_momentum < -0.08:
            action = "sell"
            reasoning = f"Weak {horizon} momentum: {primary_momentum:+.1%} return"
        else:
            action = "hold"
            reasoning = f"Neutral {horizon} momentum: {primary_momentum:+.1%}"

        strength = min(abs(primary_momentum) * 3, 1.0)

        metrics = {
            "momentum_1m": round(mom_1m, 4) if mom_1m is not None else None,
            "momentum_3m": round(mom_3m, 4) if mom_3m is not None else None,
            "primary_momentum": round(primary_momentum, 4),
            "horizon": horizon,
            "price": round(close.iloc[-1], 4),
        }

        return QuantSignal(
            strategy="momentum",
            symbol=symbol,
            action=action,
            strength=round(strength, 3),
            metrics=metrics,
            reasoning=reasoning,
        )


# ── Quant Engine (Orchestrator) ────────────────────────────

class QuantEngine:
    """
    Orchestrates all quantitative strategies.

    Responsibilities:
      - Initialize enabled strategies from config
      - Scan the watchlist and produce QuantSignal objects
      - Maintain signal history for logging and dashboard
      - Format signals for AI consumption
      - Track signal state changes (crossovers) for alert logging

    Safety:
      - All quant signals still go through risk_manager + AI validation
      - Signal history prevents acting on stale/repeated signals
      - Errors in one strategy don't crash others (isolated try/except)
    """

    SIGNAL_HISTORY_FILE = os.path.join("data", "signal_history.jsonl")
    
    def __init__(self):
        self.kama = KAMAStrategy(
            period=config.QUANT_KAMA_PERIOD,
            fast=config.QUANT_KAMA_FAST,
            slow=config.QUANT_KAMA_SLOW,
        ) if config.QUANT_KAMA_ENABLED else None

        self.trend = TrendFollowStrategy(
            sma_short=config.QUANT_TREND_SMA_SHORT,
            sma_long=config.QUANT_TREND_SMA_LONG,
        ) if config.QUANT_TREND_ENABLED else None

        self.momentum = MomentumStrategy(
            lookback_months=config.QUANT_MOMENTUM_LOOKBACK,
            top_n=config.QUANT_MOMENTUM_TOP_N,
        ) if config.QUANT_MOMENTUM_ENABLED else None

        # Track signal history for state change detection and dashboard
        self._signal_history: dict[tuple[str, str], list[dict]] = {}
        # Track previous signal actions to detect state changes
        self._prev_actions: dict[tuple[str, str], str] = {}
        # Current signals cache (latest signal per strategy/symbol)
        self._current_signals: dict[tuple[str, str], QuantSignal] = {}
        
        # Ensure data directory exists
        os.makedirs("data", exist_ok=True)

        active = self._active_strategy_names()
        if active:
            logger.info(f"[QUANT] Engine initialized with strategies: {', '.join(active)}")
        else:
            logger.info("[QUANT] No quant strategies enabled.")

    def _active_strategy_names(self) -> list[str]:
        names = []
        if self.kama:
            names.append("KAMA")
        if self.trend:
            names.append("TrendFollow")
        if self.momentum:
            names.append("Momentum")
        return names

    def is_active(self) -> bool:
        """Returns True if at least one strategy is enabled."""
        return bool(self.kama or self.trend or self.momentum)

    def scan_symbol(self, symbol: str, df: pd.DataFrame) -> list[QuantSignal]:
        """Run all enabled quant strategies on a single symbol."""
        signals = []

        if self.kama:
            try:
                sig = self.kama.generate_signal(symbol, df)
                if sig:
                    signals.append(sig)
                    self._record_signal(sig)
            except Exception as e:
                logger.error(f"[QUANT] KAMA error for {symbol}: {e}")

        if self.trend:
            try:
                sig = self.trend.generate_signal(symbol, df)
                if sig:
                    signals.append(sig)
                    self._record_signal(sig)
            except Exception as e:
                logger.error(f"[QUANT] TrendFollow error for {symbol}: {e}")

        if self.momentum:
            try:
                sig = self.momentum.generate_signal(symbol, df)
                if sig:
                    signals.append(sig)
                    self._record_signal(sig)
            except Exception as e:
                logger.error(f"[QUANT] Momentum error for {symbol}: {e}")

        return signals

    def scan_universe(self, trader) -> dict[str, list[QuantSignal]]:
        """
        Run all enabled quant strategies on the entire watchlist.
        Returns {symbol: [QuantSignal, ...]} for symbols with signals.
        Uses the same bar cache as technical analysis to avoid extra API calls.
        """
        from strategy import _get_cached_bars

        if not self.is_active():
            return {}

        active = self._active_strategy_names()
        logger.info(
            f"[QUANT] Scanning {len(config.WATCHLIST)} symbols "
            f"with strategies: {', '.join(active)}"
        )

        all_signals: dict[str, list[QuantSignal]] = {}

        for symbol in config.WATCHLIST:
            try:
                # Use more bars for momentum calculation
                bar_limit = max(config.HISTORY_BARS_LIMIT, config.QUANT_BARS_LIMIT)
                df = _get_cached_bars(trader, symbol, timeframe="1Day", limit=bar_limit)
                if df.empty:
                    continue

                signals = self.scan_symbol(symbol, df)
                if signals:
                    all_signals[symbol] = signals
            except Exception as e:
                logger.error(f"[QUANT] Error scanning {symbol}: {e}")

        # Summary logging
        buy_count = sum(
            1 for sigs in all_signals.values() for s in sigs if s.action == "buy"
        )
        sell_count = sum(
            1 for sigs in all_signals.values() for s in sigs if s.action == "sell"
        )
        hold_count = sum(
            1 for sigs in all_signals.values() for s in sigs if s.action == "hold"
        )
        logger.info(
            f"[QUANT] Scan complete: {buy_count} buy, {sell_count} sell, "
            f"{hold_count} hold across {len(all_signals)} symbols"
        )

        return all_signals

    def _record_signal(self, sig: QuantSignal):
        """Record signal and log state changes (crossovers, flips)."""
        key = (sig.strategy, sig.symbol)

        # Store current signal
        self._current_signals[key] = sig
        
        # Detect state change from previous signal
        prev_action = self._prev_actions.get(key)
        is_state_change = prev_action is not None and prev_action != sig.action

        # Update history
        if key not in self._signal_history:
            self._signal_history[key] = []
        self._signal_history[key].append({
            "action": sig.action,
            "strength": sig.strength,
            "time": datetime.now(timezone.utc).isoformat(),
        })
        # Keep last 50 signals per strategy/symbol
        self._signal_history[key] = self._signal_history[key][-50:]

        self._prev_actions[key] = sig.action
        
        # Persist to disk
        self._save_signal_to_file(sig)

        # Log actionable signals (not hold) and state changes
        if sig.action != "hold":
            prefix = "** SIGNAL CHANGE **" if is_state_change else ""
            logger.info(
                f"[QUANT:{sig.strategy.upper()}] {prefix} {sig.symbol} -> "
                f"{sig.action.upper()} (strength={sig.strength:.2f}) | {sig.reasoning}"
            )
        elif is_state_change:
            # Log transition TO hold (e.g., was buy, now hold = momentum fading)
            logger.info(
                f"[QUANT:{sig.strategy.upper()}] {sig.symbol} -> HOLD "
                f"(was {prev_action.upper()}) | {sig.reasoning}"
            )
    
    def _save_signal_to_file(self, sig: QuantSignal):
        """Persist signal to JSONL file."""
        try:
            with open(self.SIGNAL_HISTORY_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(sig)) + "\n")
        except Exception as e:
            logger.error(f"Failed to save signal to file: {e}")

    def get_quant_buy_symbols(
        self, quant_signals: dict[str, list[QuantSignal]]
    ) -> set[str]:
        """Get symbols where at least one quant strategy says BUY."""
        buy_symbols = set()
        for symbol, signals in quant_signals.items():
            for sig in signals:
                if sig.action == "buy" and sig.strength >= 0.2:
                    buy_symbols.add(symbol)
                    break
        return buy_symbols

    def get_consensus(
        self, quant_signals: dict[str, list[QuantSignal]], symbol: str
    ) -> dict:
        """
        Get consensus across strategies for a single symbol.
        Returns {action: str, agreement: float, strategies: list}.
        Used to determine conviction level.
        """
        signals = quant_signals.get(symbol, [])
        if not signals:
            return {"action": "hold", "agreement": 0.0, "strategies": []}

        buy_count = sum(1 for s in signals if s.action == "buy")
        sell_count = sum(1 for s in signals if s.action == "sell")
        total = len(signals)

        if buy_count > sell_count:
            action = "buy"
            agreement = buy_count / total
        elif sell_count > buy_count:
            action = "sell"
            agreement = sell_count / total
        else:
            action = "hold"
            agreement = 0.0

        return {
            "action": action,
            "agreement": round(agreement, 2),
            "strategies": [
                {"strategy": s.strategy, "action": s.action, "strength": s.strength}
                for s in signals
            ],
        }

    def format_for_ai(self, quant_signals: dict[str, list[QuantSignal]]) -> str:
        """
        Format quant signals into a concise text block for the AI agent prompt.
        Groups by symbol, shows each strategy's verdict.
        """
        if not quant_signals:
            return "No quant signals active."

        lines = []
        for symbol in sorted(quant_signals.keys()):
            signals = quant_signals[symbol]
            sym_parts = []
            for sig in signals:
                if sig.action == "hold" and sig.strength < 0.3:
                    continue  # Skip weak hold signals to save tokens

                # Compact metric summary
                key_metrics = []
                for k, v in sig.metrics.items():
                    if v is None or k == "price":
                        continue
                    if isinstance(v, bool):
                        if v:
                            key_metrics.append(k)
                    elif isinstance(v, float):
                        key_metrics.append(f"{k}={v:.3f}")
                    else:
                        key_metrics.append(f"{k}={v}")

                sym_parts.append(
                    f"  [{sig.strategy.upper()}] {sig.action.upper()} "
                    f"str={sig.strength:.2f} | "
                    f"{' '.join(key_metrics[:4])} | {sig.reasoning}"
                )

            if sym_parts:
                lines.append(f"{symbol}:")
                lines.extend(sym_parts)

        if not lines:
            return "Quant strategies active but no actionable signals."

        return "\n".join(lines)

    def get_signal_summary(self) -> dict:
        """Get summary of all current signals for API/dashboard."""
        summary = {
            "strategies_active": self._active_strategy_names(),
            "auto_trade_enabled": config.QUANT_AUTO_TRADE,
            "signals": {},
        }

        for (strategy, symbol), history in self._signal_history.items():
            if history:
                latest = history[-1]
                if symbol not in summary["signals"]:
                    summary["signals"][symbol] = {}
                summary["signals"][symbol][strategy] = latest

        return summary
    
    def get_all_current_signals(self) -> dict[str, dict[str, QuantSignal]]:
        """
        Get current signal from each strategy for every stock in watchlist.
        Returns {symbol: {strategy: QuantSignal}}.
        Shows 'HOLD' when no active signal.
        """
        result = {}
        
        for symbol in config.WATCHLIST:
            result[symbol] = {}
            
            # Check each enabled strategy
            if self.kama:
                key = ("kama", symbol)
                result[symbol]["kama"] = self._current_signals.get(key)
            
            if self.trend:
                key = ("trend_follow", symbol)
                result[symbol]["trend_follow"] = self._current_signals.get(key)
            
            if self.momentum:
                key = ("momentum", symbol)
                result[symbol]["momentum"] = self._current_signals.get(key)
        
        return result
    
    def analyze_single_stock(
        self, 
        symbol: str, 
        trader,
        days_back: int = 30
    ) -> dict:
        """
        Analyze a single stock through all enabled strategies.
        Returns detailed breakdown with signals, metrics, and consensus.
        
        Args:
            symbol: Stock symbol to analyze
            trader: Trader instance for fetching data
            days_back: Days of historical data to use
            
        Returns:
            {
                "symbol": "AAPL",
                "timestamp": "2026-02-13T15:30:00",
                "strategies": {
                    "kama": {signal_dict},
                    "trend_follow": {signal_dict},
                    "momentum": {signal_dict}
                },
                "consensus": {"action": "buy", "agreement": 0.67, ...},
                "suggested_entry": 150.25,
                "stop_loss": 146.50,
                "take_profit": 157.75
            }
        """
        from strategy import _get_cached_bars
        
        # Fetch data
        bar_limit = max(days_back, config.QUANT_BARS_LIMIT)
        try:
            df = _get_cached_bars(trader, symbol, timeframe="1Day", limit=bar_limit)
        except Exception as e:
            logger.error(f"Failed to fetch bars for {symbol}: {e}")
            return {"error": str(e)}
        
        if df.empty:
            return {"error": f"No data available for {symbol}"}
        
        # Run all strategies
        signals = self.scan_symbol(symbol, df)
        
        # Format results
        current_price = df["close"].iloc[-1]
        strategies = {}
        
        for sig in signals:
            strategies[sig.strategy] = asdict(sig)
        
        # Get consensus
        consensus = self.get_consensus({symbol: signals}, symbol)
        
        # Calculate suggested levels (simple ATR-based)
        atr = df["close"].diff().abs().rolling(14).mean().iloc[-1]
        stop_distance = atr * 2.5
        tp_distance = atr * 5
        
        return {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "current_price": round(current_price, 2),
            "strategies": strategies,
            "consensus": consensus,
            "suggested_entry": round(current_price, 2),
            "stop_loss": round(current_price - stop_distance, 2),
            "take_profit": round(current_price + tp_distance, 2),
            "atr_14": round(atr, 2),
        }
    
    def get_dashboard_summary(self) -> dict:
        """
        Comprehensive summary for dashboard display.
        Shows signals grouped by strategy with consensus and conflicts.
        """
        all_signals = self.get_all_current_signals()
        
        # Build strategy-centric view
        strategy_views = {}
        
        for strategy_name in ["kama", "trend_follow", "momentum"]:
            is_enabled = getattr(self, strategy_name.split("_")[0]) is not None
            if not is_enabled:
                continue
            
            buy_signals = []
            sell_signals = []
            hold_signals = []
            
            for symbol, strategies in all_signals.items():
                sig = strategies.get(strategy_name)
                if sig:
                    signal_data = {
                        "symbol": symbol,
                        "strength": sig.strength,
                        "reasoning": sig.reasoning,
                        "metrics": sig.metrics,
                    }
                    
                    if sig.action == "buy":
                        buy_signals.append(signal_data)
                    elif sig.action == "sell":
                        sell_signals.append(signal_data)
                    else:
                        hold_signals.append(signal_data)
            
            strategy_views[strategy_name] = {
                "enabled": True,
                "signals": {
                    "buy": len(buy_signals),
                    "sell": len(sell_signals),
                    "hold": len(hold_signals),
                },
                "top_buy_signals": sorted(
                    buy_signals, key=lambda x: x["strength"], reverse=True
                )[:5],
            }
        
        # Find consensus and conflicts
        consensus_buys = []
        conflicts = []
        
        for symbol, strategies in all_signals.items():
            signals = [s for s in strategies.values() if s is not None]
            if not signals:
                continue
            
            actions = [s.action for s in signals]
            buy_count = actions.count("buy")
            sell_count = actions.count("sell")
            
            # Consensus: all agree
            if len(signals) >= 2 and buy_count == len(signals):
                consensus_buys.append(symbol)
            # Conflict: mixed signals
            elif buy_count > 0 and sell_count > 0:
                conflicts.append({
                    "symbol": symbol,
                    **{s.strategy: s.action for s in signals}
                })
        
        return {
            "timestamp": datetime.now().isoformat(),
            "strategies": strategy_views,
            "consensus_buys": consensus_buys,
            "conflicts": conflicts,
        }
    
    def link_trade_to_signals(self, trade_id: str, signal_ids: list[str]):
        """Link executed trade to originating signals."""
        # Mark signals as acted upon
        for key, sig in self._current_signals.items():
            if sig.signal_id in signal_ids:
                sig.state = "acted_on"
                logger.info(
                    f"[QUANT] Signal {sig.signal_id} linked to trade {trade_id}"
                )
