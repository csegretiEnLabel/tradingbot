"""
AI Agent Brain — "Skeptical Rationalist" v3.0
──────────────────────────────────────────────
Uses Claude to analyze markets with a paranoid, reject-first mindset.
Takes only A+ setups, validates with Devil's Advocate pre-mortem,
but lets winners run via trailing stops once in a position.

Two-tier model usage:
  - Haiku for routine scans (cheap, high throughput)
  - Sonnet for critical trade decisions (smarter, pre-mortem validation)
"""

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

import anthropic

import config
from cost_tracker import CostTracker
from strategy import Signal

logger = logging.getLogger(__name__)

# ── System Prompt ────────────────────────────────────────
# Paranoid risk manager framing: reduces overconfidence and FOMO.
# Equity is injected at runtime so the prompt scales with any account size.
SYSTEM_PROMPT_TEMPLATE = """You are a paranoid Risk Manager guarding a ${equity:.0f} trading account.
Your primary directive is CAPITAL PRESERVATION. Profit is secondary.
Do not waste expensive compute on mediocre setups. Only A+ setups justify the cost of analysis.

Core Philosophy:
- The market is efficient. Most signals are noise. Assume every signal is wrong until proven otherwise.
- "Missing out" (FOMO) is acceptable. Losing money is unacceptable.
- If you are unsure, the answer is ALWAYS NO.
- Cash is a position. Holding cash beats a mediocre trade every single time.
- You do not predict the market. You wait for extreme mispricings and act decisively.

Rules:
- REJECT any setup with < 3:1 Reward/Risk ratio.
- REJECT trades if SPY/market context opposes the signal direction.
- MAX 1 new position per analysis cycle. Never over-trade.
- After entering: aggressively move stops to breakeven once profitable. Protect gains at all costs.
- Cut losers immediately. If a trade doesn't work within 2-3 bars, it's wrong.
- Skip the first 15 minutes after market open.
- Adapt to regime: momentum in trends, mean-reversion in ranges, NOTHING in chop.

Respond in strictly valid JSON when making decisions. Be extremely terse."""


class Agent:
    TRADE_HISTORY_FILE = os.path.join("data", "trade_history.json")
    WASH_SALE_DAYS = 31  # IRS wash sale window

    def __init__(self, cost_tracker: CostTracker, initial_equity: float = 100.0):
        self.client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        self.cost_tracker = cost_tracker
        self.equity = initial_equity  # Updated each cycle from live Alpaca data
        self.recent_trades: list[dict] = []  # Track recent trades to avoid re-entry
        self.entry_times: dict[str, datetime] = {}  # symbol -> entry time
        self.trade_history: list[dict] = self._load_trade_history()  # Persistent

    def _load_trade_history(self) -> list[dict]:
        """Load persistent trade history for wash sale tracking."""
        os.makedirs("data", exist_ok=True)
        if os.path.exists(self.TRADE_HISTORY_FILE):
            try:
                with open(self.TRADE_HISTORY_FILE, "r") as f:
                    history = json.load(f)
                logger.info(f"Loaded {len(history)} trades from history")
                return history
            except Exception as e:
                logger.error(f"Error loading trade history: {e}")
        return []

    def _save_trade_history(self):
        """Persist trade history to disk."""
        try:
            with open(self.TRADE_HISTORY_FILE, "w") as f:
                json.dump(self.trade_history, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving trade history: {e}")

    def get_wash_sale_blacklist(self) -> set[str]:
        """
        Returns symbols sold at a LOSS in the last 31 days.
        These are banned from re-entry to avoid IRS wash sale rule.
        """
        blacklist = set()
        cutoff = (datetime.now(timezone.utc) - timedelta(days=self.WASH_SALE_DAYS)).isoformat()
        for trade in self.trade_history:
            if trade.get("close_time", "") > cutoff and trade.get("pnl", 0) < 0:
                blacklist.add(trade["symbol"])
        return blacklist

    def record_closed_trade(self, symbol: str, pnl: float):
        """
        Record a completed trade (with P&L) for wash sale tracking.
        Called when a position is closed.
        """
        self.trade_history.append({
            "symbol": symbol,
            "pnl": round(pnl, 4),
            "close_time": datetime.now(timezone.utc).isoformat(),
        })
        # Prune history older than 60 days
        cutoff = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()
        self.trade_history = [t for t in self.trade_history if t.get("close_time", "") > cutoff]
        self._save_trade_history()
        if pnl < 0:
            logger.info(f"WASH SALE GUARD: {symbol} sold at loss (${pnl:.2f}). Banned for 31 days.")

    def _get_system_prompt(self) -> str:
        """Generate system prompt with current equity."""
        return SYSTEM_PROMPT_TEMPLATE.format(equity=self.equity)

    def _call_claude(self, prompt: str, model: str = config.MODEL_SCAN, max_tokens: int = 1024) -> Optional[str]:
        """Make a Claude API call and track costs."""
        if not self.cost_tracker.within_daily_budget():
            logger.warning("Daily API budget exceeded. Skipping AI call.")
            return None

        try:
            response = self.client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=self._get_system_prompt(),
                messages=[{"role": "user", "content": prompt}],
            )

            # Track costs
            usage = response.usage
            self.cost_tracker.log_api_call(
                model=model,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
            )

            return response.content[0].text

        except Exception as e:
            logger.error(f"Claude API error: {e}")
            return None

    def _parse_json(self, text: str) -> Optional[any]:
        """Safely parse JSON from Claude response, handling markdown code blocks."""
        if not text:
            return None
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0]
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON: {text[:200]}")
            return None

    def record_trade(self, symbol: str, action: str, result: str):
        """Record a trade for context in future AI calls."""
        now = datetime.now(timezone.utc)
        self.recent_trades.append({
            "symbol": symbol,
            "action": action,
            "result": result,
            "time": now.isoformat(),
        })
        # Track entry time for position duration
        if action == "buy" and result == "filled":
            self.entry_times[symbol] = now
        elif action == "sell" and symbol in self.entry_times:
            del self.entry_times[symbol]

        # Keep only last 10 trades
        self.recent_trades = self.recent_trades[-10:]

    def _get_bars_held(self, symbol: str, bar_minutes: int = 30) -> int:
        """Calculate how many bars a position has been held."""
        entry_time = self.entry_times.get(symbol)
        if not entry_time:
            return -1  # Unknown (position opened before agent started)
        elapsed = (datetime.now(timezone.utc) - entry_time).total_seconds()
        return max(1, int(elapsed / (bar_minutes * 60)))

    # ─────────────────────────────────────────────────────
    # STEP 1: Skeptical Screener (Haiku — cheap)
    # Framed as REJECTION task, not selection. Forces the AI
    # to look for reasons to discard signals before it can pick one.
    # Max 1 trade returned to prevent over-trading.
    # ─────────────────────────────────────────────────────
    def analyze_signals(
        self,
        signals: list[Signal],
        account: dict,
        positions: list[dict],
        market_context: Optional[dict] = None,
    ) -> list[dict]:
        """
        Skeptical signal screener. Rejects most signals; returns at most 1 trade.
        Filters out wash-sale-blacklisted symbols BEFORE Claude sees them.
        Uses Haiku for cost efficiency.
        """
        if not signals:
            return []

        # ── Wash Sale Filter (hard-coded, not AI-managed) ──
        banned = self.get_wash_sale_blacklist()
        clean_signals = []
        for s in signals:
            if s.symbol in banned:
                logger.info(f"⛔ WASH SALE: Skipping {s.symbol} (sold at loss within 31 days)")
            else:
                clean_signals.append(s)

        if not clean_signals:
            logger.info("All signals blocked by wash sale filter.")
            return []

        # Compact signal format
        signal_lines = []
        for s in clean_signals[:8]:
            regime = s.indicators.get("regime", "?")
            signal_lines.append(
                f"{s.symbol} | {s.action} | str={s.strength:.2f} | ${s.price:.2f} | "
                f"RSI={s.indicators.get('rsi', 0):.0f} vol={s.indicators.get('volume_ratio', 1):.1f}x "
                f"1d={s.indicators.get('change_1d', 0):.1%} 5d={s.indicators.get('change_5d', 0):.1%} "
                f"atr={s.atr_pct:.1%} regime={regime} adx={s.indicators.get('adx', 0):.0f} | "
                f"{'; '.join(s.reasons[:3])}"
            )

        # Position summary (with hold duration)
        pos_lines = []
        for p in positions:
            bars = self._get_bars_held(p['symbol'])
            held_str = f"held={bars}bars" if bars >= 0 else "held=?"
            pos_lines.append(
                f"{p['symbol']} ${p['current_price']:.2f} "
                f"entry=${p['avg_entry_price']:.2f} PL={p['unrealized_plpc']:.1%} {held_str}"
            )

        # Market context
        ctx_line = ""
        if market_context:
            ctx_line = (
                f"\nMarket: SPY {market_context['spy_trend']} "
                f"1d={market_context['spy_change_1d']:.1%} "
                f"5d={market_context['spy_change_5d']:.1%} "
                f"vol={market_context['market_volatility']}"
            )

        # Recent trade history
        trade_line = ""
        if self.recent_trades:
            recent = [f"{t['symbol']}({t['action']}→{t['result']})" for t in self.recent_trades[-5:]]
            trade_line = f"\nRecent: {' | '.join(recent)}"

        prompt = f"""Review these signals with extreme skepticism. Most should be REJECTED.

Acct: ${account['equity']:.2f} cash=${account['cash']:.2f} positions={len(positions)} dayPL={account['daily_pnl_pct']:.1%}{ctx_line}
{"Holding: " + " | ".join(pos_lines) if pos_lines else "No positions"}{trade_line}

Signals:
{chr(10).join(signal_lines)}

REJECTION CRITERIA (discard if ANY apply):
- Volume < 1.2x relative average
- Fighting the 5-day trend direction
- RSI in no-man's land (40-60)
- Against SPY/market direction
- Similar trade recently failed
- Regime is "volatile" with ADX < 20 (choppy, no edge)
- Signal strength < 0.5

Return the SINGLE best A+ setup, or empty [] if nothing qualifies.
MAX 1 trade. Quality over quantity.

JSON array: [{{"symbol","action","conviction":"high","reasoning":"<15 words why this won't fail","suggested_notional":$}}]
Empty [] if nothing is A+ quality."""

        response = self._call_claude(prompt, model=config.MODEL_SCAN)
        if not response:
            return []

        result = self._parse_json(response)
        if not isinstance(result, list):
            return []

        # Enforce max 1 trade per cycle
        return result[:1]

    # ─────────────────────────────────────────────────────
    # STEP 2: Devil's Advocate Validator (Sonnet — smart)
    # Uses the Pre-Mortem technique: assumes the trade has
    # already FAILED and asks Sonnet to explain why.
    # Only approves if bullish evidence overwhelms skepticism.
    # ─────────────────────────────────────────────────────
    def validate_trade(self, symbol: str, signal: Signal, account: dict) -> Optional[dict]:
        """
        Devil's Advocate pre-mortem validation using Sonnet.
        Assumes the trade will fail, and asks for evidence to override that assumption.
        Only called for high-conviction trades that passed the skeptical screener.
        """
        prompt = f"""CRITIQUE this trade proposal. Act as Devil's Advocate.

Proposed: BUY {symbol} @ ${signal.price:.2f}
RSI={signal.indicators.get('rsi', 0):.0f} MACD_bull={signal.indicators.get('macd_bullish', '?')}
SMA_cross={signal.indicators.get('sma_crossover', '?')} ADX={signal.indicators.get('adx', 0):.0f}
Vol={signal.indicators.get('volume_ratio', 1):.1f}x 1d={signal.indicators.get('change_1d', 0):.1%}
5d={signal.indicators.get('change_5d', 0):.1%} ATR={signal.atr_pct:.1%}
Regime={signal.indicators.get('regime', '?')}
Acct: ${account['equity']:.2f}

Task:
1. Assume this trade has already FAILED. List 3 reasons why it lost money.
2. Now consider: does the bullish evidence overwhelm those failure scenarios?
3. Only approve if you genuinely believe the setup is exceptional.
4. If approved, set a TIGHT stop loss (ATR-based, 1.5-2.5%).
5. Take profit should be >= 3x the stop distance (enforce 3:1 R:R minimum).
6. Remember: ~40% of gains go to short-term capital gains tax. A small win is breakeven after tax.

JSON only:
{{
  "approved": bool,
  "confidence": 0.0-1.0,
  "failure_reasons": ["reason1", "reason2", "reason3"],
  "stop_loss_pct": float,
  "take_profit_pct": float,
  "position_size_pct": 0.05-0.20,
  "reasoning": "bears vs bulls verdict in <20 words"
}}

Minimum confidence to approve: 0.80. When in doubt, reject."""

        response = self._call_claude(prompt, model=config.MODEL_DECIDE, max_tokens=512)
        if not response:
            return None

        result = self._parse_json(response)
        if not isinstance(result, dict):
            return None

        # Hard gate: reject if confidence below threshold
        confidence = result.get("confidence", 0)
        if confidence < 0.80:
            result["approved"] = False
            result["reasoning"] = f"Confidence {confidence:.0%} below 80% threshold"
            logger.info(f"Sonnet rejected {symbol}: confidence {confidence:.0%} < 80%")

        return result

    # ─────────────────────────────────────────────────────
    # Position Manager
    # NOT the naive "close at 1.5%" — that destroys expectancy.
    # Instead: relies on trailing stops for profitable exits,
    # but aggressively cuts positions showing weakness.
    # ─────────────────────────────────────────────────────
    def should_close_position(self, position: dict, account: dict) -> Optional[dict]:
        """
        Decide what to do with an open position.
        Returns one of: {action: "hold"}, {action: "close"}, or {action: "update_stop", new_stop_price: X}
        AI can now actually control stops, not just hallucinate that it did.
        """
        pnl_pct = position["unrealized_plpc"]
        bars = self._get_bars_held(position["symbol"])
        held_str = f"{bars} bars (~{bars * 30}min)" if bars >= 0 else "unknown duration"

        prompt = f"""Review position: {position['symbol']}
Entry: ${position['avg_entry_price']:.2f} | Now: ${position['current_price']:.2f} | P&L: {pnl_pct:.1%}
Held: {held_str} | Equity: ${account['equity']:.2f}

Capital Preservation Rules:
- LOSING & held > 3 bars: the thesis is dead. CLOSE to stop bleeding.
- LOSING & held <= 2 bars: give it one more bar, but move stop to limit damage.
- FLAT after 4+ bars: momentum stalled. CLOSE at breakeven.
- WINNING: move stop to breakeven (entry price) or tighter. Let it run.

JSON — pick exactly one:
  {{"action": "hold", "reasoning": "..."}}  
  {{"action": "close", "reasoning": "..."}}  
  {{"action": "update_stop", "new_stop_price": <number>, "reasoning": "..."}}"""

        response = self._call_claude(prompt, model=config.MODEL_SCAN, max_tokens=256)
        if not response:
            if pnl_pct < -0.01:
                return {"action": "close", "reasoning": "AI unavailable + losing, closing for safety"}
            return {"action": "hold", "reasoning": "AI unavailable but not losing, holding"}

        result = self._parse_json(response)
        if not isinstance(result, dict):
            if pnl_pct < -0.01:
                return {"action": "close", "reasoning": "Parse error + losing, closing for safety"}
            return {"action": "hold", "reasoning": "Parse error but not losing, holding"}

        # Normalize: support both old {close: bool} and new {action: str} format
        if "close" in result and "action" not in result:
            result["action"] = "close" if result["close"] else "hold"

        return result

    # ─────────────────────────────────────────────────────
    # Daily Review — End of Day Self-Assessment
    # ─────────────────────────────────────────────────────
    def daily_review(self, account: dict, positions: list[dict], cost_summary: dict) -> str:
        """End-of-day self-assessment and learning."""
        pos_summary = " | ".join(
            f"{p['symbol']} PL={p['unrealized_plpc']:.1%}" for p in positions
        ) if positions else "None"

        trade_summary = ""
        if self.recent_trades:
            trade_summary = f"\nTrades today: {len(self.recent_trades)} — " + \
                " | ".join(f"{t['symbol']}({t['result']})" for t in self.recent_trades)

        prompt = f"""EOD Review. Be brutally honest. 4-5 sentences max.
Equity=${account['equity']:.2f} dayPL={account['daily_pnl_pct']:.1%}
YTD P&L=${cost_summary['ytd_pnl']:.2f} Est.Tax=${cost_summary['estimated_tax']:.2f} Net(AfterTax)=${cost_summary['ytd_net_after_tax']:.2f}
Positions: {pos_summary}{trade_summary}
API cost today=${cost_summary['today_api_cost']:.4f} total_net=${cost_summary['total_net']:.4f} sustaining={cost_summary['self_sustaining']}

Cover:
1. Grade today A-F. Were you disciplined or did you deviate?
2. Any positions to close overnight? (holding overnight = extra risk)
3. What regime do you expect tomorrow? Plan accordingly.
4. Are API costs justified by trading quality? If not, trade LESS."""

        response = self._call_claude(prompt, model=config.MODEL_SCAN)
        return response or "No review generated."
