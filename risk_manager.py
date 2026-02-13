"""
Risk Manager -- the guardrails that keep the account alive.
Every trade must pass through here before execution.
"""

import logging
from dataclasses import dataclass

import config

logger = logging.getLogger(__name__)


@dataclass
class RiskDecision:
    approved: bool
    max_notional: float          # Max dollar amount for this trade
    adjusted_stop_loss: float    # Adjusted stop-loss %
    adjusted_take_profit: float  # Adjusted take-profit %
    reason: str


class RiskManager:
    def __init__(self, trader):
        self.trader = trader
        self._daily_loss_triggered = False

    def check_trade(
        self,
        symbol: str,
        side: str,
        price: float,
        atr_pct: float = 0.02,
    ) -> RiskDecision:
        """
        Evaluate whether a trade should be allowed and with what sizing.
        Returns a RiskDecision with approval status and constraints.
        """
        try:
            account = self.trader.get_account()
            positions = self.trader.get_positions()
        except Exception as e:
            return RiskDecision(False, 0, 0, 0, f"Cannot reach broker: {e}")

        equity = account["equity"]
        cash = account["cash"]

        # -- Check 1: Daily Loss Limit ----------------
        daily_pnl_pct = account["daily_pnl_pct"]
        if daily_pnl_pct < -config.DAILY_LOSS_LIMIT_PCT:
            self._daily_loss_triggered = True
            return RiskDecision(
                False, 0, 0, 0,
                f"Daily loss limit hit ({daily_pnl_pct:.2%}). No more trades today."
            )

        # -- Check 2: Max Positions -------------------
        if side == "buy" and len(positions) >= config.MAX_POSITIONS:
            return RiskDecision(
                False, 0, 0, 0,
                f"Max positions ({config.MAX_POSITIONS}) reached."
            )

        # -- Check 3: Not Already Holding -------------
        if side == "buy":
            held_symbols = [p["symbol"] for p in positions]
            if symbol in held_symbols:
                return RiskDecision(
                    False, 0, 0, 0,
                    f"Already holding {symbol}."
                )

        # -- Check 4: Minimum Account Value -----------
        if equity < 5.0:
            return RiskDecision(
                False, 0, 0, 0,
                f"Account equity too low (${equity:.2f}). Agent should shut down."
            )

        # -- Position Sizing --------------------------
        # Risk-based sizing: risk no more than 2% of equity per trade
        risk_per_trade = equity * 0.02  # 2% of current equity
        stop_distance = max(atr_pct * 1.5, config.STOP_LOSS_PCT)  # ATR-based or minimum

        # Position size = risk / stop_distance
        position_size = risk_per_trade / stop_distance if stop_distance > 0 else 0

        # Cap at maximum position size
        max_position = equity * config.MAX_POSITION_PCT
        position_size = min(position_size, max_position)

        # Cap at available cash
        position_size = min(position_size, cash * 0.95)  # Keep 5% cash buffer

        # Minimum viable trade
        if position_size < config.MIN_TRADE_VALUE:
            return RiskDecision(
                False, 0, 0, 0,
                f"Calculated position (${position_size:.2f}) below minimum (${config.MIN_TRADE_VALUE})."
            )

        # -- Dynamic Stop/Take-Profit -----------------
        # Wider stops for volatile stocks, tighter for calm ones
        adjusted_stop = max(stop_distance, 0.015)  # At least 1.5%
        adjusted_stop = min(adjusted_stop, 0.05)    # At most 5%

        # Risk-reward ratio of at least 2:1
        adjusted_take_profit = max(adjusted_stop * 2, config.TAKE_PROFIT_PCT)

        logger.info(
            f"Risk check APPROVED: {symbol} | "
            f"Size: ${position_size:.2f} | "
            f"SL: {adjusted_stop:.1%} | TP: {adjusted_take_profit:.1%}"
        )

        return RiskDecision(
            approved=True,
            max_notional=round(position_size, 2),
            adjusted_stop_loss=adjusted_stop,
            adjusted_take_profit=adjusted_take_profit,
            reason="Trade approved within risk parameters.",
        )

    def check_existing_positions(self) -> list[dict]:
        """
        Review existing positions and return actionable trailing stop / emergency close decisions.
        Implements a real trailing stop:
          - At +3%: move stop to breakeven (entry price)
          - At +5%: trail stop at 2% below current price
          - At -4%: emergency close
        """
        positions = self.trader.get_positions()
        actions = []

        for pos in positions:
            pnl_pct = pos["unrealized_plpc"]
            entry = pos["avg_entry_price"]
            current = pos["current_price"]

            # Emergency stop: if any position loses more than 4%, flag for close
            if pnl_pct < -0.04:
                actions.append({
                    "symbol": pos["symbol"],
                    "action": "emergency_close",
                    "reason": f"Position down {pnl_pct:.1%}, exceeds emergency threshold",
                    "unrealized_pl": pos["unrealized_pl"],
                })

            # Trailing stop: position up >= activate threshold
            elif pnl_pct >= config.TRAILING_ACTIVATE_PCT:
                # Trail stop at TRAILING_TRAIL_PCT below current price
                new_stop = round(current * (1 - config.TRAILING_TRAIL_PCT), 2)
                actions.append({
                    "symbol": pos["symbol"],
                    "action": "update_trailing_stop",
                    "stop_price": new_stop,
                    "reason": f"Position up {pnl_pct:.1%}, trailing stop at ${new_stop:.2f}",
                    "unrealized_pl": pos["unrealized_pl"],
                })

            # Breakeven stop: position up >= breakeven threshold but < full trail
            elif pnl_pct >= config.TRAILING_BREAKEVEN_PCT:
                breakeven_stop = round(entry * 1.001, 2)  # Tiny buffer above entry
                actions.append({
                    "symbol": pos["symbol"],
                    "action": "update_trailing_stop",
                    "stop_price": breakeven_stop,
                    "reason": f"Position up {pnl_pct:.1%}, moving stop to breakeven ${breakeven_stop:.2f}",
                    "unrealized_pl": pos["unrealized_pl"],
                })

        return actions

    def should_trade_today(self) -> tuple[bool, str]:
        """Pre-market check: should we trade at all today?"""
        try:
            account = self.trader.get_account()
        except Exception as e:
            return False, f"Cannot reach broker: {e}"

        equity = account["equity"]

        if equity < 5.0:
            return False, f"Equity too low (${equity:.2f}). Consider shutting down."

        if self._daily_loss_triggered:
            return False, "Daily loss limit was triggered. Waiting for next day."

        return True, f"Ready to trade. Equity: ${equity:.2f}"

    def reset_daily(self):
        """Reset daily flags. Call at start of each trading day."""
        self._daily_loss_triggered = False
        logger.info("Risk manager daily reset complete.")
