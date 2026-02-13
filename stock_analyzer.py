"""
Stock Analyzer
---------------
Command-line and programmatic interface for analyzing individual stocks
through all enabled quant strategies.

Usage:
    python stock_analyzer.py AAPL
    python stock_analyzer.py TSLA --days 60
    python stock_analyzer.py PLTR --strategies kama,trend
"""

import argparse
import json
import logging
import sys
from typing import Optional

from trader import Trader
from quant_engine import QuantEngine
import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def analyze_stock(
    symbol: str,
    days_back: int = 30,
    strategies: Optional[list[str]] = None,
    json_output: bool = False,
) -> dict:
    """
    Analyze a single stock through all enabled quant strategies.
    
    Args:
        symbol: Stock symbol to analyze
        days_back: Days of historical data to use (default: 30)
        strategies: List of specific strategies to run (default: all enabled)
        json_output: Return JSON instead of pretty-printed text
        
    Returns:
        Analysis results dictionary
    """
    # Initialize trader and quant engine
    trader = Trader()
    quant_engine = QuantEngine()
    
    if not quant_engine.is_active():
        logger.error("No quant strategies are enabled. Check your .env file.")
        return {"error": "No strategies enabled"}
    
    # Analyze the stock
    logger.info(f"Analyzing {symbol} with {days_back} days of data...")
    result = quant_engine.analyze_single_stock(symbol, trader, days_back=days_back)
    
    if "error" in result:
        logger.error(f"Analysis failed: {result['error']}")
        return result
    
    # Filter by requested strategies if specified
    if strategies:
        result["strategies"] = {
            k: v for k, v in result["strategies"].items()
            if k in strategies
        }
    
    return result


def print_analysis(result: dict):
    """Pretty-print analysis results to console."""
    if "error" in result:
        print(f"\n❌ Error: {result['error']}\n")
        return
    
    print(f"\n{'='*60}")
    print(f"  QUANT ANALYSIS: {result['symbol']}")
    print(f"{'='*60}")
    print(f"Current Price: ${result['current_price']:.2f}")
    print(f"Timestamp: {result['timestamp']}")
    print(f"\n{'-'*60}")
    print("STRATEGY SIGNALS:")
    print(f"{'-'*60}")
    
    if not result["strategies"]:
        print("  No signals available")
    else:
        for strategy_name, signal in result["strategies"].items():
            action = signal["action"].upper()
            strength = signal["strength"]
            
            # Color coding
            if action == "BUY":
                action_display = f"🟢 {action}"
            elif action == "SELL":
                action_display = f"🔴 {action}"
            else:
                action_display = f"⚪ {action}"
            
            print(f"\n  [{strategy_name.upper()}] {action_display} (strength: {strength:.2f})")
            print(f"  Reasoning: {signal['reasoning']}")
            
            # Show key metrics
            if signal.get("metrics"):
                metrics_display = []
                for k, v in signal["metrics"].items():
                    if v is not None and k != "price":
                        if isinstance(v, float):
                            metrics_display.append(f"{k}={v:.3f}")
                        else:
                            metrics_display.append(f"{k}={v}")
                if metrics_display:
                    print(f"  Metrics: {', '.join(metrics_display)}")
    
    # Consensus
    print(f"\n{'-'*60}")
    print("CONSENSUS:")
    print(f"{'-'*60}")
    consensus = result["consensus"]
    action = consensus["action"].upper()
    agreement = consensus["agreement"]
    
    if action == "BUY":
        action_display = f"🟢 {action}"
    elif action == "SELL":
        action_display = f"🔴 {action}"
    else:
        action_display = f"⚪ {action}"
    
    print(f"  {action_display} (agreement: {agreement:.0%})")
    
    if consensus.get("strategies"):
        print(f"  Strategy votes:")
        for strat in consensus["strategies"]:
            vote = strat["action"].upper()
            print(f"    - {strat['strategy']}: {vote} (strength: {strat['strength']:.2f})")
    
    # Suggested levels
    print(f"\n{'-'*60}")
    print("SUGGESTED LEVELS:")
    print(f"{'-'*60}")
    print(f"  Entry:       ${result['suggested_entry']:.2f}")
    print(f"  Stop Loss:   ${result['stop_loss']:.2f}  (risk: {((result['suggested_entry'] - result['stop_loss']) / result['suggested_entry'] * 100):.1f}%)")
    print(f"  Take Profit: ${result['take_profit']:.2f}  (reward: {((result['take_profit'] - result['suggested_entry']) / result['suggested_entry'] * 100):.1f}%)")
    print(f"  R:R Ratio:   {((result['take_profit'] - result['suggested_entry']) / (result['suggested_entry'] - result['stop_loss'])):.2f}:1")
    print(f"  ATR (14):    ${result['atr_14']:.2f}")
    print(f"\n{'='*60}\n")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze a stock through quant strategies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python stock_analyzer.py AAPL
  python stock_analyzer.py TSLA --days 60
  python stock_analyzer.py PLTR --strategies kama,trend --json
        """,
    )
    parser.add_argument("symbol", help="Stock symbol to analyze")
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Days of historical data to use (default: 30)",
    )
    parser.add_argument(
        "--strategies",
        help="Comma-separated list of strategies (kama,trend_follow,momentum)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output JSON instead of pretty-printed text",
    )
    
    args = parser.parse_args()
    
    # Parse strategies
    strategies = None
    if args.strategies:
        strategies = [s.strip() for s in args.strategies.split(",")]
    
    # Run analysis
    result = analyze_stock(
        symbol=args.symbol.upper(),
        days_back=args.days,
        strategies=strategies,
        json_output=args.json,
    )
    
    # Output
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print_analysis(result)


if __name__ == "__main__":
    main()
