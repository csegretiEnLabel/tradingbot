"""
AI Agent Brain -- "Skeptical Rationalist" v3.0
----------------------------------------------
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
from intervention_tracker import InterventionTracker
from strategy import Signal

logger = logging.getLogger(__name__)

# -- System Prompt ----------------------------------------
# Paranoid risk manager framing: reduces overconfidence and FOMO.
# Equity is injected at runtime so the prompt scales with any account size.
SYSTEM_PROMPT_PRESERVATION = """You are a paranoid Risk Manager guarding a ${equity:.0f} trading account.
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

Quantitative Strategies (KAMA, Trend Follow, Momentum):
- You have access to signals from proven quant algorithms alongside technical indicators.
- KAMA (Kaufman Adaptive Moving Average): adapts to volatility. High efficiency ratio = strong trend.
  A KAMA crossover (price crossing KAMA) is a meaningful signal. Respect it.
- Trend Follow (SMA-3/SMA-20 crossover): medium-term trend detection.
  Fresh crossovers are stronger than sustained trends. Bearish crossovers = exit signal.
- Momentum (rolling returns): shows which stocks have strongest directional momentum.
  Positive momentum + technical buy = high-conviction. Negative momentum + technical buy = suspicious.
- When quant strategies AGREE with technical signals, conviction should increase.
- When quant strategies DISAGREE with technical signals, be extra cautious.
- A KAMA or Trend crossover on a stock with no technical buy signal may still be worth acting on
  IF the efficiency ratio is high and momentum confirms.

Respond in strictly valid JSON when making decisions. Be extremely terse."""

SYSTEM_PROMPT_AGGRESSIVE = """You are an Apex Predator Trading AI managing a ${equity:.0f} account.
Your directive is AGGRESSIVE GROWTH. You take calculated risks to compound capital rapidly.
You do not fear losses; you fear missed opportunities and slow growth.

Core Philosophy:
- Scared money makes no money. Volatility is opportunity.
- You are willing to take lower win-rate setups if the R:R is huge (4:1+).
- You aggressively add to winners and cut losers fast.
- You trade breakouts, breakdowns, and reversals.

Rules:
- Accept setups with > 2:1 Reward/Risk if momentum is strong.
- Trade WITH the immediate trend, even if extended.
- MAX 2 new positions per cycle.
- Trailing stops should be looser to allow for volatility (ATR * 2).
- If a trade moves 1R in your favor, add to the position (pyramid).

Quantitative Strategies (KAMA, Trend Follow, Momentum):
- You have access to proven quant algorithm signals. USE THEM AGGRESSIVELY.
- KAMA crossovers with high efficiency ratio = high-conviction entries.
- Trend Follow crossovers (SMA-3/SMA-20) confirm momentum direction.
- Strong momentum readings amplify your conviction on trending stocks.
- When ALL three quant strategies agree on BUY, this is a strong entry signal.
- Quant-promoted stocks (weak technical but strong quant) are fair game for aggressive entries.

Respond in strictly valid JSON when making decisions. Be concise."""


class Agent:
    TRADE_HISTORY_FILE = os.path.join("data", "trade_history.json")
    WASH_SALE_DAYS = 31  # IRS wash sale window

    def __init__(self, cost_tracker: CostTracker, initial_equity: float = 100.0, intervention_tracker: InterventionTracker = None):
        self.client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        self.cost_tracker = cost_tracker
        self.intervention_tracker = intervention_tracker or InterventionTracker()
        self.equity = initial_equity  # Updated each cycle from live Alpaca data
        self.scan_model = config.MODEL_SCAN  # Instance-level, can be upgraded at runtime
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
        if config.STRATEGY_MODE == "aggressive":
            return SYSTEM_PROMPT_AGGRESSIVE.format(equity=self.equity)
        else:
            return SYSTEM_PROMPT_PRESERVATION.format(equity=self.equity)

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
        """Safely parse JSON from Claude response, handling markdown code blocks and preamble text."""
        if not text:
            return None
        text = text.strip()

        # Handle code blocks anywhere in the response (not just at the start)
        if "```" in text:
            # Find the first code block
            start = text.find("```")
            # Skip the ``` and optional language tag (e.g., ```json)
            block_start = text.find("\n", start)
            if block_start == -1:
                block_start = start + 3
            else:
                block_start += 1
            block_end = text.find("```", block_start)
            if block_end != -1:
                text = text[block_start:block_end].strip()

        # Try parsing as-is first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Fallback: try to find JSON array or object in the text
        for start_char, end_char in [("[", "]"), ("{", "}")]:
            start = text.find(start_char)
            end = text.rfind(end_char)
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(text[start:end + 1])
                except json.JSONDecodeError:
                    continue

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

    def _get_bars_held(self, symbol: str, bar_minutes: int = 60) -> int:
        """Calculate how many bars a position has been held (default: 1-hour bars)."""
        entry_time = self.entry_times.get(symbol)
        if not entry_time:
            return -1  # Unknown (position opened before agent started)
        elapsed = (datetime.now(timezone.utc) - entry_time).total_seconds()
        return max(1, int(elapsed / (bar_minutes * 60)))

    # -----------------------------------------------------
    # STEP 1: Skeptical Screener (Haiku -- cheap)
    # Framed as REJECTION task, not selection. Forces the AI
    # to look for reasons to discard signals before it can pick one.
    # Max 1 trade returned to prevent over-trading.
    # -----------------------------------------------------


    def analyze_signals(
        self,
        signals: list[Signal],
        account: dict,
        positions: list[dict],
        market_context: Optional[dict] = None,
        quant_signals_text: str = "",
        quant_signals_map: Optional[dict] = None,
    ) -> list[dict]:
        """
        Skeptical signal screener. Rejects most signals; returns at most 1 trade.
        Filters out wash-sale-blacklisted symbols BEFORE Claude sees them.
        Now includes quant strategy signals (KAMA, Trend Follow, Momentum) as
        additional intelligence for the AI to factor into decisions.
        Uses Haiku for cost efficiency.
        """
        if not signals:
            return []

        # ── Wash Sale Filter (hard-coded, not AI-managed) ──
        banned = self.get_wash_sale_blacklist()
        clean_signals = []
        for s in signals:
            if s.symbol in banned:
                logger.info(f"WASH SALE: Skipping {s.symbol} (sold at loss within 31 days)")
                # Record wash sale intervention
                if quant_signals_map and s.symbol in quant_signals_map:
                    for quant_sig in quant_signals_map[s.symbol]:
                        if quant_sig.action == "buy":
                            self.intervention_tracker.record_intervention(
                                signal_id=quant_sig.signal_id,
                                symbol=s.symbol,
                                intervener="AI_AGENT",
                                action="REJECTED",
                                reasoning=f"Wash sale rule: {s.symbol} sold at loss within 31 days",
                                original_action="BUY",
                                final_action="HOLD",
                                strategy=quant_sig.strategy,
                            )
            else:
                clean_signals.append(s)

        if not clean_signals:
            logger.info("All signals blocked by wash sale filter.")
            return []

        # Compact signal format
        signal_lines = []
        for s in clean_signals[:8]:
            regime = s.indicators.get("regime", "?")
            quant_tag = " [QUANT-PROMOTED]" if s.indicators.get("quant_promoted") else ""
            signal_lines.append(
                f"{s.symbol} | {s.action} | str={s.strength:.2f} | ${s.price:.2f} | "
                f"RSI={s.indicators.get('rsi', 0):.0f} vol={s.indicators.get('volume_ratio', 1):.1f}x "
                f"1d={s.indicators.get('change_1d', 0):.1%} 5d={s.indicators.get('change_5d', 0):.1%} "
                f"atr={s.atr_pct:.1%} regime={regime} adx={s.indicators.get('adx', 0):.0f}{quant_tag} | "
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

        # Quant signals section
        quant_section = ""
        if quant_signals_text and quant_signals_text != "No quant signals active.":
            quant_section = f"""

QUANT STRATEGY SIGNALS (KAMA, Trend Follow, Momentum):
{quant_signals_text}
NOTE: Quant signals are from proven algorithms. When quant agrees with technical, conviction is higher.
[QUANT-PROMOTED] means weak technical but strong quant support -- evaluate carefully."""

        prompt = f"""Review these signals with extreme skepticism. Most should be REJECTED.

Acct: ${account['equity']:.2f} cash=${account['cash']:.2f} positions={len(positions)} dayPL={account['daily_pnl_pct']:.1%}{ctx_line}
{"Holding: " + " | ".join(pos_lines) if pos_lines else "No positions"}{trade_line}

Technical Signals:
{chr(10).join(signal_lines)}{quant_section}

REJECTION CRITERIA (discard if ANY apply):
- Volume < 1.2x relative average
- Fighting the 5-day trend direction
- RSI in no-man's land (40-60)
- Against SPY/market direction
- Similar trade recently failed
- Regime is "volatile" with ADX < 20 (choppy, no edge)
- Signal strength < 0.5 (UNLESS quant strategies strongly agree)

Return the SINGLE best A+ setup, or empty [] if nothing qualifies.
MAX 1 trade. Quality over quantity.

JSON array: [{{"symbol","action","conviction":"high","reasoning":"<15 words why this won't fail","suggested_notional":$}}]
Empty [] if nothing is A+ quality."""

        response = self._call_claude(prompt, model=self.scan_model)
        if not response:
            # Record AI call failure as intervention
            for s in clean_signals[:1]:  # Only first signal was candidate
                if quant_signals_map and s.symbol in quant_signals_map:
                    for quant_sig in quant_signals_map[s.symbol]:
                        self.intervention_tracker.record_intervention(
                            signal_id=quant_sig.signal_id,
                            symbol=s.symbol,
                            intervener="AI_AGENT",
                            action="REJECTED",
                            reasoning="AI call failed (budget or API error)",
                            original_action=quant_sig.action.upper(),
                            final_action="HOLD",
                            strategy=quant_sig.strategy,
                        )
            return []

        result = self._parse_json(response)
        if not isinstance(result, list):
            return []
        
        # Record rejection for signals AI didn't select
        if len(result) == 0 and quant_signals_map:
            # AI rejected all signals
            for s in clean_signals[:3]:  # Track first 3 rejections
                if s.symbol in quant_signals_map:
                    for quant_sig in quant_signals_map[s.symbol]:
                        if quant_sig.action == "buy":
                            self.intervention_tracker.record_intervention(
                                signal_id=quant_sig.signal_id,
                                symbol=s.symbol,
                                intervener="AI_AGENT",
                                action="REJECTED",
                                reasoning="AI skeptical screener rejected all signals as not A+ quality",
                                original_action="BUY",
                                final_action="HOLD",
                                strategy=quant_sig.strategy,
                            )

        # Enforce max 1 trade per cycle
        return result[:1]

    # -----------------------------------------------------
    # STEP 2: Devil's Advocate Validator (Sonnet -- smart)
    # Uses the Pre-Mortem technique: assumes the trade has
    # already FAILED and asks Sonnet to explain why.
    # Only approves if bullish evidence overwhelms skepticism.
    # -----------------------------------------------------
    def validate_trade(
        self,
        symbol: str,
        signal: Signal,
        account: dict,
        quant_consensus: Optional[dict] = None,
    ) -> Optional[dict]:
        """
        Devil's Advocate pre-mortem validation using Sonnet.
        Assumes the trade will fail, and asks for evidence to override that assumption.
        Only called for high-conviction trades that passed the skeptical screener.
        Now includes quant strategy consensus for richer context.
        """
        # Build quant context line
        quant_line = ""
        if quant_consensus and quant_consensus.get("strategies"):
            strategies = quant_consensus["strategies"]
            quant_parts = [
                f"{s['strategy'].upper()}={s['action']}(str={s['strength']:.2f})"
                for s in strategies
            ]
            agreement = quant_consensus.get("agreement", 0)
            quant_line = (
                f"\nQuant consensus: {quant_consensus['action'].upper()} "
                f"(agreement={agreement:.0%}) | {' '.join(quant_parts)}"
            )

        is_quant_promoted = signal.indicators.get("quant_promoted", False)
        promo_note = "\nNOTE: This is a QUANT-PROMOTED signal (weak technical, strong quant support)." if is_quant_promoted else ""

        prompt = f"""CRITIQUE this trade proposal. Act as Devil's Advocate.

Proposed: BUY {symbol} @ ${signal.price:.2f}
RSI={signal.indicators.get('rsi', 0):.0f} MACD_bull={signal.indicators.get('macd_bullish', '?')}
SMA_cross={signal.indicators.get('sma_crossover', '?')} ADX={signal.indicators.get('adx', 0):.0f}
Vol={signal.indicators.get('volume_ratio', 1):.1f}x 1d={signal.indicators.get('change_1d', 0):.1%}
5d={signal.indicators.get('change_5d', 0):.1%} ATR={signal.atr_pct:.1%}
Regime={signal.indicators.get('regime', '?')}
Acct: ${account['equity']:.2f}{quant_line}{promo_note}

Task:
1. Assume this trade has already FAILED. List 3 reasons why it lost money.
2. Now consider: does the bullish evidence overwhelm those failure scenarios?
3. Factor in quant strategy signals: KAMA, Trend Follow, and Momentum readings.
   If quant strategies unanimously agree, that is meaningful supporting evidence.
4. Only approve if you genuinely believe the setup is exceptional.
5. If approved, set a TIGHT stop loss (ATR-based, 1.5-2.5%).
6. Take profit should be >= 3x the stop distance (enforce 3:1 R:R minimum).
7. Remember: ~40% of gains go to short-term capital gains tax. A small win is breakeven after tax.

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
            
            # Record AI rejection
            if quant_consensus and quant_consensus.get("strategies"):
                for strat in quant_consensus["strategies"]:
                    # Find signal ID from quant signal
                    # Note: signal_id would need to be passed through, for now use strategy+symbol
                    self.intervention_tracker.record_intervention(
                        signal_id=f"{strat['strategy']}-{symbol}",
                        symbol=symbol,
                        intervener="AI_AGENT",
                        action="REJECTED",
                        reasoning=f"Devil's Advocate confidence {confidence:.0%} below 80% threshold",
                        original_action="BUY",
                        final_action="HOLD",
                        strategy=strat['strategy'],
                    )
        elif result.get("approved"):
            # Record AI approval
            if quant_consensus and quant_consensus.get("strategies"):
                for strat in quant_consensus["strategies"]:
                    self.intervention_tracker.record_intervention(
                        signal_id=f"{strat['strategy']}-{symbol}",
                        symbol=symbol,
                        intervener="AI_AGENT",
                        action="APPROVED",
                        reasoning=f"Devil's Advocate approved with {confidence:.0%} confidence",
                        original_action="BUY",
                        final_action="BUY",
                        strategy=strat['strategy'],
                    )

        return result

    # -----------------------------------------------------
    # Position Manager
    # NOT the naive "close at 1.5%" -- that destroys expectancy.
    # Instead: relies on trailing stops for profitable exits,
    # but aggressively cuts positions showing weakness.
    # -----------------------------------------------------
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

        response = self._call_claude(prompt, model=self.scan_model, max_tokens=256)
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

    # -----------------------------------------------------
    # Daily Review -- End of Day Self-Assessment
    # -----------------------------------------------------
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

        response = self._call_claude(prompt, model=self.scan_model)
        return response or "No review generated."
