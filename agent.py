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

# System prompt that defines the agent's personality and constraints
SYSTEM_PROMPT = """You are a disciplined trading agent for a $50 account. Survival = profitability.

Rules:
- Capital preservation first. No trade > bad trade.
- Only high-conviction setups with risk/reward > 2:1.
- Cut losers fast, let winners run. Always use stops.
- Max 3 positions. Fractional shares OK.
- Don't chase. Don't average down. Don't fight the trend.
- Skip first 15min of market open.

Respond in valid JSON only when making decisions. Be extremely terse."""


class Agent:
    def __init__(self, cost_tracker: CostTracker):
        self.client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        self.cost_tracker = cost_tracker

    def _call_claude(self, prompt: str, model: str = config.MODEL_SCAN, max_tokens: int = 1024) -> Optional[str]:
        """Make a Claude API call and track costs."""
        if not self.cost_tracker.within_daily_budget():
            logger.warning("Daily API budget exceeded. Skipping AI call.")
            return None

        try:
            response = self.client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=SYSTEM_PROMPT,
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

    def analyze_signals(self, signals: list[Signal], account: dict, positions: list[dict]) -> list[dict]:
        """
        Given technical signals, use AI to select the best trades.
        Uses Haiku for cost efficiency. Prompts are kept minimal to save tokens.
        """
        if not signals:
            return []

        # Compact signal format — only what Claude needs, nothing more
        signal_lines = []
        for s in signals[:8]:  # Top 8 only
            signal_lines.append(
                f"{s.symbol} | {s.action} | str={s.strength:.1f} | ${s.price:.2f} | "
                f"RSI={s.indicators.get('rsi', 0):.0f} vol={s.indicators.get('volume_ratio', 1):.1f}x "
                f"1d={s.indicators.get('change_1d', 0):.1%} 5d={s.indicators.get('change_5d', 0):.1%} "
                f"atr={s.atr_pct:.1%} | {'; '.join(s.reasons[:2])}"
            )

        # Compact position format
        pos_lines = []
        for p in positions:
            pos_lines.append(f"{p['symbol']} ${p['current_price']:.2f} entry=${p['avg_entry_price']:.2f} PL={p['unrealized_plpc']:.1%}")

        prompt = f"""Pick best trade(s) from signals. Be VERY selective.

Acct: ${account['equity']:.2f} cash=${account['cash']:.2f} positions={len(positions)} dayPL={account['daily_pnl_pct']:.1%}
{"Holding: " + " | ".join(pos_lines) if pos_lines else "No positions"}

Signals:
{chr(10).join(signal_lines)}

JSON array. Per trade: {{"symbol","action","conviction":"high"/"medium","reasoning":"<10 words","suggested_notional":$}}
Empty [] if nothing compelling."""

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
Acct: ${account['equity']:.2f}

JSON only: {{"approved":bool,"confidence":0-1,"stop_loss_pct":float,"take_profit_pct":float,"position_size_pct":0.05-0.20,"reasoning":"<10 words"}}"""

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

        prompt = f"""EOD Review. 3-4 sentences max.
Equity=${account['equity']:.2f} dayPL={account['daily_pnl_pct']:.1%}
Positions: {pos_summary}
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
