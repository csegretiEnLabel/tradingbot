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

# Force UTF-8 encoding for Windows consoles
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import config
from agent import Agent
from cost_tracker import CostTracker
from risk_manager import RiskManager
from strategy import scan_universe
from trader import Trader

# -- Logging Setup -----------------------------------------
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
==================================================
         AI TRADING AGENT v3.0
         "Skeptical Rationalist"
         Mode: {mode:<8s}
         {'(!) REAL MONEY' if mode == 'LIVE' else '(?) Paper Trading'}
==================================================
"""
    print(banner)


def check_status(trader: Trader, cost_tracker: CostTracker):
    """Print current account status."""
    account = trader.get_account()
    positions = trader.get_positions()
    cost_summary = cost_tracker.get_summary()

    print("\n[ ACCOUNT STATUS ]")
    print(f"  Equity:      ${account['equity']:.2f}")
    print(f"  Cash:        ${account['cash']:.2f}")
    print(f"  Daily P&L:   ${account['daily_pnl']:.2f} ({account['daily_pnl_pct']:.2%})")

    if positions:
        print(f"\n[ OPEN POSITIONS ({len(positions)}) ]")
        for p in positions:
            tag = "[+]" if p["unrealized_pl"] >= 0 else "[-]"
            print(
                f"  {tag} {p['symbol']:6s} | "
                f"Qty: {p['qty']:.4f} | "
                f"P&L: ${p['unrealized_pl']:.2f} ({p['unrealized_plpc']:.2%}) | "
                f"Value: ${p['market_value']:.2f}"
            )
    else:
        print("\n  No open positions.")

    print(f"\n[ COST TRACKING ]")
    print(f"  Total API cost:    ${cost_summary['total_api_cost']:.4f}")
    print(f"  Total trading P&L: ${cost_summary['total_trading_pnl']:.4f}")
    print(f"  Net:               ${cost_summary['total_net']:.4f}")
    print(f"  Self-sustaining:   {'Yes' if cost_summary['self_sustaining'] else 'No'}")
    print(f"  Today's API spend: ${cost_summary['today_api_cost']:.4f} / ${config.DAILY_API_BUDGET:.2f}")
    print()


# Track positions across cycles to detect broker-triggered closes
_known_positions: dict[str, dict] = {}  # symbol -> {avg_entry_price, qty}
_last_reconcile_time: str = ""  # ISO timestamp of last reconciliation
SNAPSHOT_FILE = os.path.join("data", "positions_snapshot.json")


def load_known_positions():
    """Load known positions from disk to survive restarts."""
    global _known_positions, _last_reconcile_time
    try:
        if os.path.exists(SNAPSHOT_FILE):
            with open(SNAPSHOT_FILE, "r") as f:
                data = json.load(f)
                _known_positions = data.get("positions", {})
                _last_reconcile_time = data.get("last_reconcile_time", "")
                logger.info(f"Loaded {len(_known_positions)} known positions from snapshot.")
    except Exception as e:
        logger.error(f"Failed to load position snapshot: {e}")


def save_known_positions():
    """Save known positions to disk."""
    global _known_positions, _last_reconcile_time
    try:
        os.makedirs("data", exist_ok=True)
        with open(SNAPSHOT_FILE, "w") as f:
            json.dump({
                "positions": _known_positions,
                "last_reconcile_time": _last_reconcile_time
            }, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save position snapshot: {e}")



def reconcile_vanished_positions(trader: Trader, agent: Agent, current_positions: list[dict]):
    """
    Detect positions that disappeared between cycles (broker-triggered stop/TP fills).
    Records P&L for wash sale tracking so vanished losses still blacklist the symbol.
    """
    global _known_positions, _last_reconcile_time

    current_symbols = {p["symbol"] for p in current_positions}
    vanished_symbols = set(_known_positions.keys()) - current_symbols

    if vanished_symbols:
        logger.info(f"[INFO] Position reconciliation: {vanished_symbols} vanished since last cycle")

        # Check Alpaca closed orders to find what happened
        closed_orders = trader.get_closed_orders_since(
            after=_last_reconcile_time or datetime.now(timezone.utc).replace(hour=0, minute=0).isoformat()
        )

        for symbol in vanished_symbols:
            entry_info = _known_positions[symbol]
            entry_price = entry_info["avg_entry_price"]

            # Find the matching fill from closed orders
            fill = next((o for o in closed_orders if o["symbol"] == symbol), None)
            if fill:
                exit_price = fill["filled_avg_price"]
                qty = fill["filled_qty"]
                pnl = (exit_price - entry_price) * qty
                logger.info(
                    f"[INFO] Reconciled {symbol}: entry=${entry_price:.2f} -> "
                    f"exit=${exit_price:.2f} ({fill['type']}) PL=${pnl:.2f}"
                )
            else:
                # Can't find the order -- estimate from last known data
                pnl = entry_info.get("unrealized_pl", 0)
                logger.warning(
                    f"[WARN] Cannot find close order for {symbol}. Estimating PL=${pnl:.2f}"
                )

            agent.record_closed_trade(symbol, pnl)
            agent.record_trade(symbol, "sell", f"broker_closed_pl{pnl:+.2f}")

    # Update known positions for next cycle
    _known_positions = {
        p["symbol"]: {
            "avg_entry_price": p["avg_entry_price"],
            "qty": p["qty"],
            "unrealized_pl": p.get("unrealized_pl", 0),
        }
        for p in current_positions
    }
    _last_reconcile_time = datetime.now(timezone.utc).isoformat()
    save_known_positions()



_model_upgraded = False  # Track whether scan model has been upgraded to Sonnet


def run_cycle(trader: Trader, agent: Agent, risk_manager: RiskManager, cost_tracker: CostTracker, starting_equity: float):
    """
    Run one complete analysis → decision → execution cycle.
    """
    logger.info("=" * 50)
    logger.info("Starting analysis cycle")

    # -- Pre-flight Checks ----------------------------
    can_trade, reason = risk_manager.should_trade_today()
    if not can_trade:
        logger.warning(f"Cannot trade: {reason}")
        return

    kill, kill_reason = cost_tracker.should_kill()
    if kill:
        logger.critical(kill_reason)
        logger.critical("AGENT SHUTTING DOWN -- not self-sustaining.")
        trader.close_all_positions()
        cost_tracker.update_trading_pnl(trader.get_account()["daily_pnl"])
        cost_tracker.close_day()
        sys.exit(1)

    if not cost_tracker.within_daily_budget():
        logger.warning("Daily API budget exhausted. Skipping cycle.")
        return

    account = trader.get_account()
    positions = trader.get_positions()

    # -- Reconcile vanished positions (broker-triggered stop/TP fills) --
    reconcile_vanished_positions(trader, agent, positions)

    # Update agent's equity awareness for dynamic prompts
    agent.equity = account["equity"]

    # -- Portfolio-Based Kill/Upgrade (percentage of starting equity) --
    if config.TRADING_MODE == "live":
        kill_threshold = starting_equity * (1 - config.HARD_KILL_DRAWDOWN_PCT)
        upgrade_threshold = starting_equity * (1 + config.MODEL_UPGRADE_GAIN_PCT)

        if account["equity"] < kill_threshold:
            logger.critical(
                f"HARD KILL: Equity ${account['equity']:.2f} < ${kill_threshold:.2f} "
                f"({config.HARD_KILL_DRAWDOWN_PCT:.0%} drawdown from ${starting_equity:.2f}). "
                f"Experiment failed. Closing all positions and shutting down."
            )
            trader.close_all_positions()
            cost_tracker.close_day()
            sys.exit(1)

        if account["equity"] > upgrade_threshold:
            global _model_upgraded
            if not _model_upgraded:
                logger.info(
                    f"[UPGRADE] Equity ${account['equity']:.2f} > ${upgrade_threshold:.2f} "
                    f"({config.MODEL_UPGRADE_GAIN_PCT:.0%} gain). "
                    f"Upgrading scan model to Sonnet for better decisions."
                )
                agent.scan_model = config.MODEL_DECIDE  # Upgrade via agent, not global config
                _model_upgraded = True

    # -- Step 1: Check Existing Positions (Trailing Stops) --
    risk_actions = risk_manager.check_existing_positions()
    for action in risk_actions:
        if action["action"] == "emergency_close":
            logger.warning(f"Emergency close: {action['symbol']} -- {action['reason']}")
            pos = next((p for p in positions if p["symbol"] == action["symbol"]), None)
            if not pos:
                continue
            decision = agent.should_close_position(pos, account)
            if decision:
                act = decision.get("action", "close")
                reason = decision.get("reasoning", "")
                if act == "close":
                    # Capture P&L before selling for wash sale tracking
                    pnl = float(pos.get("unrealized_pl", 0))
                    trader.sell(action["symbol"])
                    agent.record_trade(action["symbol"], "sell", "emergency_close")
                    agent.record_closed_trade(action["symbol"], pnl)
                    logger.info(f"Closed {action['symbol']} (PL=${pnl:.2f}): {reason}")
                elif act == "update_stop":
                    new_stop = decision.get("new_stop_price")
                    if new_stop:
                        trader.update_stop_loss(action["symbol"], new_stop)
                        logger.info(f"AI moved stop for {action['symbol']} to ${new_stop:.2f}: {reason}")
                else:
                    logger.info(f"AI chose to hold {action['symbol']} despite emergency flag: {reason}")

        elif action["action"] == "update_trailing_stop":
            stop_price = action.get("stop_price")
            if stop_price:
                logger.info(f"Updating trailing stop: {action['symbol']} -> ${stop_price:.2f} ({action['reason']})")
                trader.update_stop_loss(action["symbol"], stop_price)

    # -- Step 2: Get Market Context -------------------
    market_context = trader.get_market_context()
    logger.info(
        f"Market context: SPY {market_context['spy_trend']} "
        f"1d={market_context['spy_change_1d']:.1%} "
        f"volatility={market_context['market_volatility']}"
    )

    # -- Step 3: Scan for New Opportunities -----------
    logger.info(f"Scanning {len(config.WATCHLIST)} symbols...")
    signals = scan_universe(trader)

    # Filter: don't show AI signals for stocks we already hold (no pyramiding)
    current_holdings = {p["symbol"] for p in positions}
    buy_signals = [s for s in signals if s.action == "buy" and s.symbol not in current_holdings]

    if not buy_signals:
        logger.info("No buy signals found. Cash is a position. Sitting tight.")
        return

    logger.info(f"Found {len(buy_signals)} buy signal(s). Consulting skeptical screener...")

    # -- Step 4: Skeptical Screener (Haiku -- cheap) ---
    # Returns at most 1 trade idea (max 1 per cycle enforced in agent)
    trade_ideas = agent.analyze_signals(signals, account, positions, market_context)

    if not trade_ideas:
        logger.info("Screener rejected all signals. No A+ setups. Staying in cash.")
        return

    # Only process the single best idea (enforced)
    idea = trade_ideas[0]
    symbol = idea.get("symbol")
    action = idea.get("action", "buy")

    if action != "buy":
        logger.info(f"Screener suggested non-buy action for {symbol}: {action}. Skipping.")
        return

    # Find matching signal
    matching_signal = next((s for s in signals if s.symbol == symbol), None)
    if not matching_signal:
        logger.warning(f"Signal not found for {symbol}. Skipping.")
        return

    # -- Step 5: Risk Manager Check -------------------
    risk = risk_manager.check_trade(
        symbol=symbol,
        side="buy",
        price=matching_signal.price,
        atr_pct=matching_signal.atr_pct,
    )

    if not risk.approved:
        logger.info(f"Risk Manager rejected {symbol}: {risk.reason}")
        return

    # -- Step 6: Devil's Advocate Validation (Sonnet) -
    # EVERY trade goes through pre-mortem. No exceptions.
    # This is the most important gate -- Sonnet is smarter and catches what Haiku misses.
    logger.info(f"Validating {symbol} with Devil's Advocate (Sonnet)...")
    validation = agent.validate_trade(symbol, matching_signal, account)

    if not validation or not validation.get("approved", False):
        reasons = validation.get("failure_reasons", []) if validation else ["API error"]
        reasoning = validation.get("reasoning", "No reason given") if validation else "Validation failed"
        logger.info(
            f"Devil's Advocate KILLED {symbol}: {reasoning}\n"
            f"  Failure scenarios: {'; '.join(reasons)}"
        )
        agent.record_trade(symbol, "buy", "sonnet_rejected")
        return

    # Use Sonnet's refined parameters (tighter stops, enforced R:R)
    sl = validation.get("stop_loss_pct", risk.adjusted_stop_loss)
    tp = validation.get("take_profit_pct", risk.adjusted_take_profit)
    size_pct = validation.get("position_size_pct", 0.10)

    # Enforce 3:1 R:R minimum
    if tp < sl * 3:
        tp = round(sl * 3, 4)
        logger.info(f"Adjusted TP to {tp:.1%} to enforce 3:1 R:R (SL={sl:.1%})")

    risk.adjusted_stop_loss = sl
    risk.adjusted_take_profit = tp

    # -- Step 7: Execute ------------------------------
    # Use Sonnet's recommended size, capped by risk manager
    notional = min(account["equity"] * size_pct, risk.max_notional)
    notional = min(notional, idea.get("suggested_notional", notional))
    notional = max(notional, config.MIN_TRADE_VALUE)

    logger.info(
        f"[EXEC] EXECUTING: BUY {symbol} | ${notional:.2f} | "
        f"SL: {sl:.1%} | TP: {tp:.1%} | "
        f"Confidence: {validation.get('confidence', '?')} | "
        f"R:R = 1:{tp/sl:.1f}"
    )

    result = trader.buy(
        symbol=symbol,
        notional=notional,
        stop_loss_pct=sl,
        take_profit_pct=tp,
    )

    if result:
        fill = trader.verify_order_fill(result["id"])
        if fill:
            logger.info(f"[SUCCESS] Order FILLED: {fill}")
            agent.record_trade(symbol, "buy", "filled")
        else:
            logger.warning(f"Order placed but fill not confirmed: {result}")
            agent.record_trade(symbol, "buy", "unconfirmed")
    else:
        logger.error(f"[ERROR] Order failed for {symbol}")
        agent.record_trade(symbol, "buy", "failed")

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


# -- Status File for UI -----------------------------------
STATUS_FILE = os.path.join("data", "bot_status.json")


def save_bot_status(last_run: datetime, next_run: datetime):
    """Save execution times for the UI countdown."""
    try:
        data = {
            "last_run_time": last_run.isoformat(),
            "next_run_time": next_run.isoformat(),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        with open(STATUS_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save bot status: {e}")


def trading_loop(trader: Trader, agent: Agent, risk_manager: RiskManager, cost_tracker: CostTracker, starting_equity: float):
    """
    Main trading loop. Runs during market hours.
    """
    logger.info("Entering trading loop...")
    risk_manager.reset_daily()

    last_cycle_time = None
    interval = timedelta(minutes=config.ANALYSIS_INTERVAL_MIN)
    reviewed_today = False
    consecutive_errors = 0

    # Initialize status file
    save_bot_status(
        datetime.now(ET), 
        datetime.now(ET) + timedelta(seconds=10) # Initial short delay
    )

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
        market_open_minute = config.MARKET_OPEN_MIN + config.PRE_CLOSE_MIN  # 30 + 15 = 45
        if now.hour == config.MARKET_OPEN_HOUR and now.minute < market_open_minute:
            logger.info(f"Waiting for market to settle (until {config.MARKET_OPEN_HOUR}:{market_open_minute:02d})...")
            time.sleep(60)
            continue

        # Skip last 15 minutes (no new trades)
        close_cutoff_hour = config.MARKET_CLOSE_HOUR
        close_cutoff_min = 0 - config.PRE_CLOSE_MIN  # e.g., 16:00 - 15min = 15:45
        if close_cutoff_min < 0:
            close_cutoff_hour -= 1
            close_cutoff_min += 60
        if (now.hour > close_cutoff_hour) or (now.hour == close_cutoff_hour and now.minute >= close_cutoff_min):
            logger.info("Too close to market close. No new trades.")
            time.sleep(60)
            continue

        # Run cycle at configured interval
        config.reload_config()
        interval = timedelta(minutes=config.ANALYSIS_INTERVAL_MIN)
        
        if last_cycle_time is None or (now - last_cycle_time) >= interval:
            try:
                run_cycle(trader, agent, risk_manager, cost_tracker, starting_equity)
                last_cycle_time = now
                consecutive_errors = 0
                
                # Update status with next run time
                next_run = now + interval
                save_bot_status(now, next_run)
                logger.info(f"Next cycle at {next_run.strftime('%H:%M:%S')}")

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
        print("[ERROR] Please set ALPACA_API_KEY in .env")
        sys.exit(1)
    if not config.ALPACA_SECRET_KEY or config.ALPACA_SECRET_KEY == "your_paper_secret_key_here":
        print("[ERROR] Please set ALPACA_SECRET_KEY in .env")
        sys.exit(1)
    if not config.ANTHROPIC_API_KEY or config.ANTHROPIC_API_KEY == "your_anthropic_key_here":
        print("[ERROR] Please set ANTHROPIC_API_KEY in .env")
        sys.exit(1)

    if config.TRADING_MODE == "live":
        print("[WARN] LIVE TRADING MODE -- REAL MONEY AT RISK")
        confirm = input("Type 'YES' to confirm: ")
        if confirm != "YES":
            print("Aborted.")
            sys.exit(0)

    # Initialize components
    trader = Trader()
    cost_tracker = CostTracker()
    risk_manager = RiskManager(trader)

    # Fetch starting equity from Alpaca (dynamic, not hardcoded)
    starting_equity = trader.get_account()["equity"]
    logger.info(f"Starting equity: ${starting_equity:.2f} ({config.TRADING_MODE} mode)")
    print(f"  [ALL] Starting equity: ${starting_equity:.2f}")

    # Load persistent state
    load_known_positions()

    agent_brain = Agent(cost_tracker, initial_equity=starting_equity)


    if args.status:
        check_status(trader, cost_tracker)
        return

    if args.review:
        run_daily_review(trader, agent_brain, cost_tracker)
        return

    if args.once:
        if trader.is_market_open():
            run_cycle(trader, agent_brain, risk_manager, cost_tracker, starting_equity)
            check_status(trader, cost_tracker)
        else:
            print("Market is closed. Showing status instead.")
            check_status(trader, cost_tracker)
        return

    # Full trading loop
    try:
        trading_loop(trader, agent_brain, risk_manager, cost_tracker, starting_equity)
    except KeyboardInterrupt:
        logger.info("Shutting down gracefully...")
        check_status(trader, cost_tracker)
        print("\n[STOP] Agent stopped. Positions remain open.")


if __name__ == "__main__":
    main()
