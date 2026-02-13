"""
Cost Tracker -- monitors API spending vs trading profits.
If costs consistently exceed profits, triggers the kill switch.
The agent must pay for itself or die.
"""

import json
import logging
import os
from datetime import datetime, date
from pathlib import Path

import config

logger = logging.getLogger(__name__)

COST_LOG_FILE = os.path.join(config.LOG_DIR, "cost_tracker.json")


class CostTracker:
    def __init__(self):
        self.daily_api_cost = 0.0
        self.daily_trading_pnl = 0.0
        self.session_total_cost = 0.0
        self.history: list[dict] = []
        self._load_history()

    def _load_history(self):
        """Load historical cost data."""
        try:
            if os.path.exists(COST_LOG_FILE):
                with open(COST_LOG_FILE, "r") as f:
                    self.history = json.load(f)
                logger.info(f"Loaded {len(self.history)} days of cost history")
        except Exception as e:
            logger.error(f"Error loading cost history: {e}")
            self.history = []

    def _save_history(self):
        """Persist cost data."""
        try:
            Path(config.LOG_DIR).mkdir(exist_ok=True)
            with open(COST_LOG_FILE, "w") as f:
                json.dump(self.history, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving cost history: {e}")

    def log_api_call(self, model: str, input_tokens: int, output_tokens: int):
        """
        Log an API call and its estimated cost.
        """
        costs = config.MODEL_COSTS.get(model, {"input": 3.0, "output": 15.0})
        cost = (input_tokens * costs["input"] + output_tokens * costs["output"]) / 1_000_000
        self.daily_api_cost += cost
        self.session_total_cost += cost

        logger.debug(
            f"API call: {model} | {input_tokens}in/{output_tokens}out | "
            f"${cost:.6f} | Daily total: ${self.daily_api_cost:.4f}"
        )

    def update_trading_pnl(self, pnl: float):
        """Update the daily trading P&L."""
        self.daily_trading_pnl = pnl

    def close_day(self):
        """End-of-day: record daily summary and reset. Deduplicates by date."""
        today = str(date.today())

        # Prevent duplicate entries for the same date
        if self.history and self.history[-1].get("date") == today:
            logger.info(f"Day already closed for {today}, updating existing entry.")
            # Update the existing entry with latest data (merge costs)
            existing = self.history[-1]
            existing["api_cost"] = round(existing["api_cost"] + self.daily_api_cost, 6)
            existing["trading_pnl"] = round(self.daily_trading_pnl, 4)  # P&L is absolute, not additive
            existing["net"] = round(existing["trading_pnl"] - existing["api_cost"], 4)
            self._save_history()
        else:
            record = {
                "date": today,
                "api_cost": round(self.daily_api_cost, 6),
                "trading_pnl": round(self.daily_trading_pnl, 4),
                "net": round(self.daily_trading_pnl - self.daily_api_cost, 4),
            }
            self.history.append(record)
            self._save_history()

            logger.info(
                f"Day closed | API cost: ${record['api_cost']:.4f} | "
                f"Trading P&L: ${record['trading_pnl']:.4f} | "
                f"Net: ${record['net']:.4f}"
            )

        # Reset daily counters
        self.daily_api_cost = 0.0
        self.daily_trading_pnl = 0.0

    def should_kill(self) -> tuple[bool, str]:
        """
        Kill switch: should the agent shut down?
        Triggers if API costs > trading profits for N consecutive days.
        """
        threshold = config.COST_KILL_THRESHOLD_DAYS

        if len(self.history) < threshold:
            return False, f"Not enough history ({len(self.history)}/{threshold} days)"

        recent = self.history[-threshold:]
        consecutive_losses = all(day["net"] < 0 for day in recent)

        if consecutive_losses:
            total_loss = sum(day["net"] for day in recent)
            return True, (
                f"KILL SWITCH: {threshold} consecutive days of net loss. "
                f"Total net loss: ${total_loss:.4f}. Agent should shut down."
            )

        return False, "Agent is viable."

    def within_daily_budget(self) -> bool:
        """Check if we're still within the daily API budget."""
        return self.daily_api_cost < config.DAILY_API_BUDGET

    def get_summary(self) -> dict:
        """Get a summary of costs and profitability."""
        total_api_cost = sum(d["api_cost"] for d in self.history) + self.daily_api_cost
        total_pnl = sum(d["trading_pnl"] for d in self.history) + self.daily_trading_pnl
        total_net = total_pnl - total_api_cost

        profitable_days = sum(1 for d in self.history if d["net"] > 0)
        total_days = len(self.history)

        # -- Tax Estimation ------------------------------
        current_year = str(date.today().year)
        ytd_pnl = sum(d["trading_pnl"] for d in self.history if d["date"].startswith(current_year))
        ytd_pnl += self.daily_trading_pnl
        
        # Conservative estimate: 35% short-term capital gains tax
        estimated_tax = ytd_pnl * 0.35 if ytd_pnl > 0 else 0.0
        
        return {
            "total_api_cost": round(total_api_cost, 4),
            "total_trading_pnl": round(total_pnl, 4),
            "total_net": round(total_net, 4),
            "profitable_days": profitable_days,
            "total_days": total_days,
            "win_rate": profitable_days / total_days if total_days > 0 else 0,
            "today_api_cost": round(self.daily_api_cost, 4),
            "today_budget_remaining": round(config.DAILY_API_BUDGET - self.daily_api_cost, 4),
            "self_sustaining": total_net >= 0,
            "ytd_pnl": round(ytd_pnl, 4),
            "estimated_tax": round(estimated_tax, 4),
            "ytd_net_after_tax": round(ytd_pnl - estimated_tax, 4),
        }
