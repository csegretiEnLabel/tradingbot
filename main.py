"""
Main entry point for the trading agent.
Runs the trading loop during market hours.

Usage:
  python main.py              # Run the full trading loop
  python main.py --status     # Check account status and exit
  python main.py --review     # Run daily review and exit
  python main.py --once       # Run one analysis cycle and exit
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import config
from agent import Agent
from cost_tracker import CostTracker
from risk_manager import RiskManager
from strategy import scan_universe
from trader import Trader

# ── Logging Setup ─────────────────────────────────────────
Path(config.LOG_DIR).mkdir(exist_ok=True)

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            os.path.join(config.LOG_DIR, f"trading_{datetime.now().strftime('%Y%m%d')}.log")
        ),
    ],
)
logger = logging.getLogger("main")

ET = ZoneInfo("America/New_York")


def print_banner():
    mode = config.TRADING_MODE.upper()
    banner = f"""
╔══════════════════════════════════════════════════╗
║        🤖 AI TRADING AGENT v1.0                 ║
║        Mode: {mode:<8s}                            ║
║        {'⚠️  REAL MONEY' if mode == 'LIVE' else '📝 Paper Trading'}                          ║
╚══════════════════════════════════════════════════╝
"""
    print(banner)


def check_status(trader: Trader, cost_tracker: CostTracker):
    """Print current account status."""
    account = trader.get_account()
    positions = trader.get_positions()
    cost_summary = cost_tracker.get_summary()

    print("\n📊 ACCOUNT STATUS")
    print(f"  Equity:      ${account['equity']:.2f}")
    print(f"  Cash:        ${account['cash']:.2f}")
    print(f"  Daily P&L:   ${account['daily_pnl']:.2f} ({account['daily_pnl_pct']:.2%})")

    if positions:
        print(f"\n📈 OPEN POSITIONS ({len(positions)})")
        for p in positions:
            emoji = "🟢" if p["unrealized_pl"] >= 0 else "🔴"
            print(
                f"  {emoji} {p['symbol']:6s} | "
                f"Qty: {p['qty']:.4f} | "
                f"P&L: ${p['unrealized_pl']:.2f} ({p['unrealized_plpc']:.2%}) | "
                f"Value: ${p['market_value']:.2f}"
            )
    else:
        print("\n  No open positions.")

    print(f"\n💰 COST TRACKING")
    print(f"  Total API cost:    ${cost_summary['total_api_cost']:.4f}")
    print(f"  Total trading P&L: ${cost_summary['total_trading_pnl']:.4f}")
    print(f"  Net:               ${cost_summary['total_net']:.4f}")
    print(f"  Self-sustaining:   {'✅ Yes' if cost_summary['self_sustaining'] else '❌ No'}")
    print(f"  Today's API spend: ${cost_summary['today_api_cost']:.4f} / ${config.DAILY_API_BUDGET:.2f}")
    print()


def run_cycle(trader: Trader, agent: Agent, risk_manager: RiskManager, cost_tracker: CostTracker):
    """
    Run one complete analysis → decision → execution cycle.
    """
    logger.info("=" * 50)
    logger.info("Starting analysis cycle")

    # ── Pre-flight Checks ────────────────────────────
    can_trade, reason = risk_manager.should_trade_today()
    if not can_trade:
        logger.warning(f"Cannot trade: {reason}")
        return

    kill, kill_reason = cost_tracker.should_kill()
    if kill:
        logger.critical(kill_reason)
        logger.critical("AGENT SHUTTING DOWN — not self-sustaining.")
        trader.close_all_positions()
        sys.exit(1)

    if not cost_tracker.within_daily_budget():
        logger.warning("Daily API budget exhausted. Skipping cycle.")
        return

    account = trader.get_account()
    positions = trader.get_positions()

    # ── Rule of 10: Portfolio-Based Kill/Upgrade ─────
    if config.TRADING_MODE == "live":
        if account["equity"] < config.HARD_KILL_EQUITY:
            logger.critical(
                f"HARD KILL: Equity ${account['equity']:.2f} < ${config.HARD_KILL_EQUITY}. "
                f"Experiment failed. Closing all positions and shutting down."
            )
            trader.close_all_positions()
            cost_tracker.close_day()
            sys.exit(1)

        if account["equity"] > config.MODEL_UPGRADE_EQUITY:
            if config.MODEL_SCAN != config.MODEL_DECIDE:
                logger.info(
                    f"🎉 Equity ${account['equity']:.2f} > ${config.MODEL_UPGRADE_EQUITY}. "
                    f"Upgrading scan model to Sonnet for better decisions."
                )
                config.MODEL_SCAN = config.MODEL_DECIDE  # Upgrade to Sonnet

    # ── Step 1: Check Existing Positions ─────────────
    risk_actions = risk_manager.check_existing_positions()
    for action in risk_actions:
        if action["action"] == "emergency_close":
            logger.warning(f"Emergency close: {action['symbol']} — {action['reason']}")
            decision = agent.should_close_position(
                next(p for p in positions if p["symbol"] == action["symbol"]),
                account,
            )
            if decision and decision.get("close", True):
                trader.sell(action["symbol"])
                logger.info(f"Closed {action['symbol']}: {decision.get('reasoning', '')}")
        elif action["action"] == "tighten_stop":
            logger.info(f"Consider tightening stop on {action['symbol']}: {action['reason']}")

    # ── Step 2: Scan for New Opportunities ───────────
    logger.info(f"Scanning {len(config.WATCHLIST)} symbols...")
    signals = scan_universe(trader)
    buy_signals = [s for s in signals if s.action == "buy"]

    if not buy_signals:
        logger.info("No buy signals found. Sitting tight.")
        return

    logger.info(f"Found {len(buy_signals)} buy signals. Consulting AI...")

    # ── Step 3: AI Analysis (Haiku - cheap) ──────────
    trade_ideas = agent.analyze_signals(signals, account, positions)

    if not trade_ideas:
        logger.info("AI found no compelling trades. Staying in cash.")
        return

    logger.info(f"AI suggested {len(trade_ideas)} trade(s)")

    # ── Step 4: Execute Trades ───────────────────────
    for idea in trade_ideas:
        symbol = idea.get("symbol")
        action = idea.get("action", "buy")
        conviction = idea.get("conviction", "medium")

        if action != "buy":
            continue  # Only handle buys for now; sells handled by bracket orders

        # Find matching signal
        matching_signal = next((s for s in signals if s.symbol == symbol), None)
        if not matching_signal:
            continue

        # Risk check
        risk = risk_manager.check_trade(
            symbol=symbol,
            side="buy",
            price=matching_signal.price,
            atr_pct=matching_signal.atr_pct,
        )

        if not risk.approved:
            logger.info(f"Risk rejected {symbol}: {risk.reason}")
            continue

        # For high-conviction trades, do final validation with Sonnet
        if conviction == "high" and risk.max_notional >= 5.0:
            logger.info(f"High-conviction trade: validating {symbol} with Sonnet...")
            validation = agent.validate_trade(matching_signal, matching_signal, account)
            if validation and not validation.get("approved", False):
                logger.info(f"Sonnet rejected {symbol}: {validation.get('reasoning', '')}")
                continue
            if validation:
                # Use Sonnet's refined parameters
                risk.adjusted_stop_loss = validation.get("stop_loss_pct", risk.adjusted_stop_loss)
                risk.adjusted_take_profit = validation.get("take_profit_pct", risk.adjusted_take_profit)

        # Execute!
        notional = min(risk.max_notional, idea.get("suggested_notional", risk.max_notional))
        notional = max(notional, config.MIN_TRADE_VALUE)

        logger.info(
            f"EXECUTING: BUY {symbol} | ${notional:.2f} | "
            f"SL: {risk.adjusted_stop_loss:.1%} | TP: {risk.adjusted_take_profit:.1%}"
        )

        result = trader.buy(
            symbol=symbol,
            notional=notional,
            stop_loss_pct=risk.adjusted_stop_loss,
            take_profit_pct=risk.adjusted_take_profit,
        )

        if result:
            logger.info(f"Order placed: {result}")
        else:
            logger.error(f"Order failed for {symbol}")

    # ── Update Cost Tracker ──────────────────────────
    cost_tracker.update_trading_pnl(account["daily_pnl"])


def run_daily_review(trader: Trader, agent: Agent, cost_tracker: CostTracker):
    """End of day review and bookkeeping."""
    account = trader.get_account()
    positions = trader.get_positions()
    cost_summary = cost_tracker.get_summary()

    review = agent.daily_review(account, positions, cost_summary)

    logger.info("=" * 50)
    logger.info("DAILY REVIEW")
    logger.info(review)
    logger.info("=" * 50)

    # Update and close the day
    cost_tracker.update_trading_pnl(account["daily_pnl"])
    cost_tracker.close_day()

    print(f"\n📋 DAILY REVIEW:\n{review}\n")


def trading_loop(trader: Trader, agent: Agent, risk_manager: RiskManager, cost_tracker: CostTracker):
    """
    Main trading loop. Runs during market hours.
    """
    logger.info("Entering trading loop...")
    risk_manager.reset_daily()

    last_cycle_time = None
    interval = timedelta(minutes=config.ANALYSIS_INTERVAL_MIN)
    reviewed_today = False
    consecutive_errors = 0

    while True:
        now = datetime.now(ET)

        # Check if market is open
        if not trader.is_market_open():
            # If market just closed and we haven't reviewed
            if now.hour >= config.MARKET_CLOSE_HOUR and not reviewed_today:
                run_daily_review(trader, agent, cost_tracker)
                reviewed_today = True

            # Wait for market to open
            next_open = trader.get_next_open()
            if next_open:
                logger.info(f"Market closed. Next open: {next_open}")
            else:
                logger.info("Market closed. Waiting...")

            time.sleep(300)  # Check every 5 minutes
            continue

        # Reset daily flags at market open
        if now.hour == config.MARKET_OPEN_HOUR and now.minute < 35:
            if reviewed_today:
                reviewed_today = False
                risk_manager.reset_daily()

        # Skip first 15 minutes (too volatile)
        if now.hour == config.MARKET_OPEN_HOUR and now.minute < 45:
            logger.info("Waiting for market to settle (first 15 min)...")
            time.sleep(60)
            continue

        # Skip last 15 minutes (no new trades)
        if now.hour == config.MARKET_CLOSE_HOUR - 1 and now.minute > (60 - config.PRE_CLOSE_MIN):
            logger.info("Too close to market close. No new trades.")
            time.sleep(60)
            continue

        # Run cycle at configured interval
        if last_cycle_time is None or (now - last_cycle_time) >= interval:
            try:
                run_cycle(trader, agent, risk_manager, cost_tracker)
                last_cycle_time = now
                consecutive_errors = 0
            except KeyboardInterrupt:
                raise
            except Exception as e:
                consecutive_errors += 1
                logger.error(f"Cycle error ({consecutive_errors}/5): {e}", exc_info=True)
                if consecutive_errors >= 5:
                    logger.critical("5 consecutive errors. Pausing 30 min.")
                    time.sleep(1800)
                    consecutive_errors = 0
                last_cycle_time = now  # Don't hammer on failure

            check_status(trader, cost_tracker)

        time.sleep(30)  # Check every 30 seconds


def main():
    parser = argparse.ArgumentParser(description="AI Trading Agent")
    parser.add_argument("--status", action="store_true", help="Show account status")
    parser.add_argument("--review", action="store_true", help="Run daily review")
    parser.add_argument("--once", action="store_true", help="Run one cycle and exit")
    args = parser.parse_args()

    print_banner()

    # Validate config
    if not config.ALPACA_API_KEY or config.ALPACA_API_KEY == "your_paper_api_key_here":
        print("❌ Please set ALPACA_API_KEY in .env")
        sys.exit(1)
    if not config.ANTHROPIC_API_KEY or config.ANTHROPIC_API_KEY == "your_anthropic_key_here":
        print("❌ Please set ANTHROPIC_API_KEY in .env")
        sys.exit(1)

    if config.TRADING_MODE == "live":
        print("⚠️  LIVE TRADING MODE — REAL MONEY AT RISK")
        confirm = input("Type 'YES' to confirm: ")
        if confirm != "YES":
            print("Aborted.")
            sys.exit(0)

    # Initialize components
    trader = Trader()
    cost_tracker = CostTracker()
    risk_manager = RiskManager(trader)
    agent_brain = Agent(cost_tracker)

    if args.status:
        check_status(trader, cost_tracker)
        return

    if args.review:
        run_daily_review(trader, agent_brain, cost_tracker)
        return

    if args.once:
        if trader.is_market_open():
            run_cycle(trader, agent_brain, risk_manager, cost_tracker)
            check_status(trader, cost_tracker)
        else:
            print("Market is closed. Showing status instead.")
            check_status(trader, cost_tracker)
        return

    # Full trading loop
    try:
        trading_loop(trader, agent_brain, risk_manager, cost_tracker)
    except KeyboardInterrupt:
        logger.info("Shutting down gracefully...")
        check_status(trader, cost_tracker)
        print("\n👋 Agent stopped. Positions remain open.")


if __name__ == "__main__":
    main()
