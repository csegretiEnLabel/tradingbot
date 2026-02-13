"""
AI Agent Brain — uses Claude to analyze markets and make trading decisions.
Two-tier model usage:
  - Haiku for routine scans (cheap)
  - Sonnet for critical trade decisions (smarter but costlier)
"""

import json
import logging
from typing import Optional

import anthropic

import config
from cost_tracker import CostTracker
from strategy import Signal

logger = logging.getLogger(__name__)

# System prompt template — equity is injected at runtime
SYSTEM_PROMPT_TEMPLATE = """You are a disciplined trading agent managing a ${equity:.0f} account. Survival = profitability.
Your trading profits MUST exceed your API costs or you will be shut down.

Rules:
- Capital preservation first. No trade > bad trade.
- Only high-conviction setups with risk/reward > 2:1.
- Cut losers fast, let winners run. Always use stops.
- Max 3 positions. Fractional shares OK.
- Don't chase. Don't average down. Don't fight the trend.
- Skip first 15min of market open.
- Adapt to market regime: trade momentum in trends, mean-reversion in ranges.
- Avoid re-entering positions that were just stopped out.

Respond in valid JSON only when making decisions. Be extremely terse."""


class Agent:
    def __init__(self, cost_tracker: CostTracker, initial_equity: float = 100.0):
        self.client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        self.cost_tracker = cost_tracker
        self.equity = initial_equity  # Updated each cycle
        self.recent_trades: list[dict] = []  # Track recent trades to avoid re-entry

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

    def record_trade(self, symbol: str, action: str, result: str):
        """Record a trade for context in future AI calls."""
        self.recent_trades.append({
            "symbol": symbol,
            "action": action,
            "result": result,
        })
        # Keep only last 10 trades
        self.recent_trades = self.recent_trades[-10:]

    def analyze_signals(
        self,
        signals: list[Signal],
        account: dict,
        positions: list[dict],
        market_context: Optional[dict] = None,
    ) -> list[dict]:
        """
        Given technical signals, use AI to select the best trades.
        Uses Haiku for cost efficiency. Prompts include market context.
        """
        if not signals:
            return []

        # Compact signal format — only what Claude needs, nothing more
        signal_lines = []
        for s in signals[:8]:  # Top 8 only
            regime = s.indicators.get("regime", "?")
            signal_lines.append(
                f"{s.symbol} | {s.action} | str={s.strength:.1f} | ${s.price:.2f} | "
                f"RSI={s.indicators.get('rsi', 0):.0f} vol={s.indicators.get('volume_ratio', 1):.1f}x "
                f"1d={s.indicators.get('change_1d', 0):.1%} 5d={s.indicators.get('change_5d', 0):.1%} "
                f"atr={s.atr_pct:.1%} regime={regime} | {'; '.join(s.reasons[:2])}"
            )

        # Compact position format
        pos_lines = []
        for p in positions:
            pos_lines.append(f"{p['symbol']} ${p['current_price']:.2f} entry=${p['avg_entry_price']:.2f} PL={p['unrealized_plpc']:.1%}")

        # Market context line
        ctx_line = ""
        if market_context:
            ctx_line = (
                f"\nMarket: SPY {market_context['spy_trend']} "
                f"1d={market_context['spy_change_1d']:.1%} "
                f"5d={market_context['spy_change_5d']:.1%} "
                f"vol={market_context['market_volatility']}"
            )

        # Recent trade history (avoid re-entries)
        trade_line = ""
        if self.recent_trades:
            recent = [f"{t['symbol']}({t['action']}→{t['result']})" for t in self.recent_trades[-5:]]
            trade_line = f"\nRecent: {' | '.join(recent)}"

        prompt = f"""Pick best trade(s) from signals. Be VERY selective.

Acct: ${account['equity']:.2f} cash=${account['cash']:.2f} positions={len(positions)} dayPL={account['daily_pnl_pct']:.1%}{ctx_line}
{"Holding: " + " | ".join(pos_lines) if pos_lines else "No positions"}{trade_line}

Signals:
{chr(10).join(signal_lines)}

JSON array. Per trade: {{"symbol","action","conviction":"high"/"medium","reasoning":"<10 words","suggested_notional":$}}
Empty [] if nothing compelling. Avoid symbols recently stopped out."""

        response = self._call_claude(prompt, model=config.MODEL_SCAN)
        if not response:
            return []

        try:
            # Parse JSON from response (handle markdown code blocks)
            text = response.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0]
            trades = json.loads(text)
            return trades if isinstance(trades, list) else []
        except json.JSONDecodeError:
            logger.error(f"Failed to parse AI response: {response[:200]}")
            return []

    def validate_trade(self, symbol: str, signal: Signal, account: dict) -> Optional[dict]:
        """
        Final validation using Sonnet (smarter model).
        Only called for high-conviction trades that passed screening.
        """
        prompt = f"""Validate trade: BUY {symbol} @ ${signal.price:.2f}
RSI={signal.indicators.get('rsi', 0):.0f} MACD_bull={signal.indicators.get('macd_bullish', '?')} SMA_cross={signal.indicators.get('sma_crossover', '?')}
Vol={signal.indicators.get('volume_ratio', 1):.1f}x 1d={signal.indicators.get('change_1d', 0):.1%} 5d={signal.indicators.get('change_5d', 0):.1%} ATR={signal.atr_pct:.1%}
Regime={signal.indicators.get('regime', '?')} ADX={signal.indicators.get('adx', 0):.0f}
Acct: ${account['equity']:.2f}

JSON only: {{"approved":bool,"confidence":0-1,"stop_loss_pct":float,"take_profit_pct":float,"position_size_pct":0.05-0.30,"reasoning":"<10 words"}}"""

        response = self._call_claude(prompt, model=config.MODEL_DECIDE, max_tokens=512)
        if not response:
            return None

        try:
            text = response.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0]
            return json.loads(text)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse validation response: {response[:200]}")
            return None

    def daily_review(self, account: dict, positions: list[dict], cost_summary: dict) -> str:
        """End-of-day self-assessment."""
        pos_summary = " | ".join(
            f"{p['symbol']} PL={p['unrealized_plpc']:.1%}" for p in positions
        ) if positions else "None"

        trade_summary = ""
        if self.recent_trades:
            trade_summary = f"\nTrades today: {len(self.recent_trades)} — " + \
                " | ".join(f"{t['symbol']}({t['result']})" for t in self.recent_trades)

        prompt = f"""EOD Review. 3-4 sentences max.
Equity=${account['equity']:.2f} dayPL={account['daily_pnl_pct']:.1%}
Positions: {pos_summary}{trade_summary}
API cost today=${cost_summary['today_api_cost']:.4f} total_net=${cost_summary['total_net']:.4f} sustaining={cost_summary['self_sustaining']}

Cover: what happened, hold/close overnight, tomorrow's plan, sustainability outlook."""

        response = self._call_claude(prompt, model=config.MODEL_SCAN)
        return response or "No review generated."

    def should_close_position(self, position: dict, account: dict) -> Optional[dict]:
        """Quick check: should we close this position?"""
        prompt = f"""Close {position['symbol']}? Entry=${position['avg_entry_price']:.2f} now=${position['current_price']:.2f} PL={position['unrealized_plpc']:.1%} Equity=${account['equity']:.2f}
JSON: {{"close":bool,"reasoning":"<8 words"}}"""

        response = self._call_claude(prompt, model=config.MODEL_SCAN, max_tokens=256)
        if not response:
            return {"close": True, "reasoning": "AI unavailable, closing for safety"}

        try:
            text = response.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0]
            return json.loads(text)
        except json.JSONDecodeError:
            return {"close": True, "reasoning": "Could not parse AI response, closing for safety"}
