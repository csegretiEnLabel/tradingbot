"""
Alpaca API wrapper for order execution, position management, and market data.
Supports both paper and live trading via config.
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Optional

import alpaca_trade_api as tradeapi
import pandas as pd

import config

logger = logging.getLogger(__name__)


class Trader:
    def __init__(self):
        self.api = tradeapi.REST(
            key_id=config.ALPACA_API_KEY,
            secret_key=config.ALPACA_SECRET_KEY,
            base_url=config.ALPACA_BASE_URL,
            api_version="v2",
        )
        self.mode = config.TRADING_MODE
        logger.info(f"Trader initialized in {self.mode.upper()} mode → {config.ALPACA_BASE_URL}")

    # ── Account Info ─────────────────────────────────────

    def get_account(self) -> dict:
        """Return account summary."""
        acct = self.api.get_account()
        return {
            "equity": float(acct.equity),
            "cash": float(acct.cash),
            "buying_power": float(acct.buying_power),
            "portfolio_value": float(acct.portfolio_value),
            "daily_pnl": float(acct.equity) - float(acct.last_equity),
            "daily_pnl_pct": (float(acct.equity) - float(acct.last_equity)) / float(acct.last_equity)
            if float(acct.last_equity) > 0 else 0,
        }

    def get_positions(self) -> list[dict]:
        """Return all open positions."""
        positions = self.api.list_positions()
        return [
            {
                "symbol": p.symbol,
                "qty": float(p.qty),
                "side": p.side,
                "market_value": float(p.market_value),
                "cost_basis": float(p.cost_basis),
                "unrealized_pl": float(p.unrealized_pl),
                "unrealized_plpc": float(p.unrealized_plpc),
                "current_price": float(p.current_price),
                "avg_entry_price": float(p.avg_entry_price),
            }
            for p in positions
        ]

    def get_position_count(self) -> int:
        return len(self.api.list_positions())

    # ── Market Data ──────────────────────────────────────

    def get_bars(self, symbol: str, timeframe: str = "1Day", limit: int = 60) -> pd.DataFrame:
        """
        Fetch historical bars for a symbol.
        timeframe: '1Min', '5Min', '15Min', '1Hour', '1Day'
        """
        try:
            bars = self.api.get_bars(
                symbol,
                timeframe,
                limit=limit,
            ).df

            if bars.empty:
                return pd.DataFrame()

            # Flatten multi-index if present
            if isinstance(bars.index, pd.MultiIndex):
                bars = bars.reset_index(level=0, drop=True)

            bars.index = pd.to_datetime(bars.index)
            return bars
        except Exception as e:
            logger.error(f"Error fetching bars for {symbol}: {e}")
            return pd.DataFrame()

    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get latest trade price for a symbol."""
        try:
            trade = self.api.get_latest_trade(symbol)
            return float(trade.price)
        except Exception as e:
            logger.error(f"Error getting price for {symbol}: {e}")
            return None

    def get_snapshot(self, symbol: str) -> Optional[dict]:
        """Get snapshot with latest trade, quote, and bar."""
        try:
            snap = self.api.get_snapshot(symbol)
            return {
                "latest_price": float(snap.latest_trade.price),
                "latest_volume": int(snap.latest_trade.size) if snap.latest_trade else 0,
                "daily_bar": {
                    "open": float(snap.daily_bar.open),
                    "high": float(snap.daily_bar.high),
                    "low": float(snap.daily_bar.low),
                    "close": float(snap.daily_bar.close),
                    "volume": int(snap.daily_bar.volume),
                } if snap.daily_bar else None,
                "prev_daily_bar": {
                    "close": float(snap.prev_daily_bar.close),
                    "volume": int(snap.prev_daily_bar.volume),
                } if snap.prev_daily_bar else None,
            }
        except Exception as e:
            logger.error(f"Error getting snapshot for {symbol}: {e}")
            return None

    # ── Order Execution ──────────────────────────────────

    def buy(
        self,
        symbol: str,
        notional: Optional[float] = None,
        qty: Optional[float] = None,
        stop_loss_pct: float = config.STOP_LOSS_PCT,
        take_profit_pct: float = config.TAKE_PROFIT_PCT,
    ) -> Optional[dict]:
        """
        Place a buy order with bracket (stop-loss + take-profit).
        Use 'notional' for dollar amount or 'qty' for share count.
        Alpaca supports fractional shares, so even $5 trades work.
        """
        try:
            price = self.get_latest_price(symbol)
            if not price:
                return None

            stop_price = round(price * (1 - stop_loss_pct), 2)
            take_profit_price = round(price * (1 + take_profit_pct), 2)

            order_params = {
                "symbol": symbol,
                "side": "buy",
                "type": "market",
                "time_in_force": "day",
                "order_class": "bracket",
                "stop_loss": {"stop_price": str(stop_price)},
                "take_profit": {"limit_price": str(take_profit_price)},
            }

            if notional:
                order_params["notional"] = str(round(notional, 2))
            elif qty:
                order_params["qty"] = str(qty)
            else:
                return None

            order = self.api.submit_order(**order_params)

            result = {
                "id": order.id,
                "symbol": symbol,
                "side": "buy",
                "notional": notional,
                "qty": qty,
                "price_at_order": price,
                "stop_loss": stop_price,
                "take_profit": take_profit_price,
                "status": order.status,
            }
            logger.info(f"BUY ORDER: {result}")
            return result

        except Exception as e:
            logger.error(f"Error placing buy order for {symbol}: {e}")
            return None

    def sell(self, symbol: str, qty: Optional[float] = None) -> Optional[dict]:
        """Sell a position. If qty is None, sells entire position."""
        try:
            if qty is None:
                # Close entire position
                self.api.close_position(symbol)
                logger.info(f"CLOSED position: {symbol}")
                return {"symbol": symbol, "action": "close_all"}
            else:
                order = self.api.submit_order(
                    symbol=symbol,
                    qty=str(qty),
                    side="sell",
                    type="market",
                    time_in_force="day",
                )
                logger.info(f"SELL ORDER: {symbol} qty={qty}")
                return {"id": order.id, "symbol": symbol, "qty": qty, "status": order.status}
        except Exception as e:
            logger.error(f"Error selling {symbol}: {e}")
            return None

    def close_all_positions(self):
        """Emergency: close everything."""
        try:
            self.api.close_all_positions()
            logger.warning("CLOSED ALL POSITIONS")
        except Exception as e:
            logger.error(f"Error closing all positions: {e}")

    def cancel_all_orders(self):
        """Cancel all open orders."""
        try:
            self.api.cancel_all_orders()
            logger.info("Cancelled all open orders")
        except Exception as e:
            logger.error(f"Error cancelling orders: {e}")

    # ── Market Status ────────────────────────────────────

    def is_market_open(self) -> bool:
        """Check if the market is currently open."""
        try:
            clock = self.api.get_clock()
            return clock.is_open
        except Exception as e:
            logger.error(f"Error checking market status: {e}")
            return False

    def get_next_open(self) -> Optional[datetime]:
        """Get next market open time."""
        try:
            clock = self.api.get_clock()
            return clock.next_open
        except Exception:
            return None

    def get_recent_orders(self, limit: int = 10) -> list[dict]:
        """Get recent orders for review."""
        try:
            orders = self.api.list_orders(status="all", limit=limit)
            return [
                {
                    "symbol": o.symbol,
                    "side": o.side,
                    "qty": o.qty,
                    "filled_qty": o.filled_qty,
                    "type": o.type,
                    "status": o.status,
                    "filled_avg_price": o.filled_avg_price,
                    "created_at": str(o.created_at),
                }
                for o in orders
            ]
        except Exception as e:
            logger.error(f"Error fetching orders: {e}")
            return []

    def get_closed_orders_since(self, after: str, limit: int = 50) -> list[dict]:
        """
        Fetch filled sell orders since a given ISO timestamp.
        Used to detect broker-triggered stop/TP fills that happened while the agent slept.
        """
        try:
            orders = self.api.list_orders(
                status="closed",
                limit=limit,
                direction="desc",
                after=after,
            )
            results = []
            for o in orders:
                if o.side == "sell" and o.status == "filled" and o.filled_avg_price:
                    results.append({
                        "symbol": o.symbol,
                        "side": o.side,
                        "filled_qty": float(o.filled_qty or 0),
                        "filled_avg_price": float(o.filled_avg_price),
                        "type": o.type,  # 'stop', 'limit', 'market'
                        "filled_at": str(o.filled_at or o.updated_at),
                    })
            return results
        except Exception as e:
            logger.error(f"Error fetching closed orders: {e}")
            return []

    def update_stop_loss(self, symbol: str, new_stop_price: float) -> bool:
        """
        Cancel existing stop/take-profit orders for a symbol and place a new stop.
        Cancels ALL legs of any bracket/OCO group to avoid partial-bracket issues,
        then places a standalone stop order.
        """
        try:
            # Get current position qty FIRST (before cancelling anything)
            try:
                position = self.api.get_position(symbol)
                qty = position.qty
            except Exception:
                logger.warning(f"No position found for {symbol}, skipping stop update")
                return False

            # Cancel ALL open orders for this symbol (stop, limit, and any OCO legs)
            orders = self.api.list_orders(status="open")
            cancelled_ids = []
            for order in orders:
                if order.symbol == symbol:
                    try:
                        self.api.cancel_order(order.id)
                        cancelled_ids.append(order.id)
                        logger.info(f"Cancelled order for {symbol}: {order.id} (type={order.type})")
                    except Exception as cancel_err:
                        logger.warning(f"Failed to cancel order {order.id}: {cancel_err}")

            # Brief pause to let cancellations settle
            if cancelled_ids:
                time.sleep(0.5)

            # Place new standalone stop order
            self.api.submit_order(
                symbol=symbol,
                qty=str(qty),
                side="sell",
                type="stop",
                time_in_force="gtc",
                stop_price=str(new_stop_price),
            )
            logger.info(f"TRAILING STOP: {symbol} new stop at ${new_stop_price:.2f}")
            return True

        except Exception as e:
            logger.error(f"Error updating stop loss for {symbol}: {e}")
            # Emergency fallback: try to place a stop even if something above failed
            try:
                position = self.api.get_position(symbol)
                self.api.submit_order(
                    symbol=symbol,
                    qty=str(position.qty),
                    side="sell",
                    type="stop",
                    time_in_force="gtc",
                    stop_price=str(new_stop_price),
                )
                logger.info(f"FALLBACK STOP placed for {symbol} at ${new_stop_price:.2f}")
                return True
            except Exception as fallback_err:
                logger.error(f"CRITICAL: Failed to place fallback stop for {symbol}: {fallback_err}")
                return False

    def verify_order_fill(self, order_id: str, timeout: int = 10) -> Optional[dict]:
        """
        Wait for an order to be filled, up to timeout seconds.
        Returns filled order details or None.
        """
        start = time.time()
        while time.time() - start < timeout:
            try:
                order = self.api.get_order(order_id)
                if order.status == "filled":
                    return {
                        "id": order.id,
                        "symbol": order.symbol,
                        "filled_qty": float(order.filled_qty),
                        "filled_avg_price": float(order.filled_avg_price),
                        "status": order.status,
                    }
                elif order.status in ("cancelled", "expired", "rejected"):
                    logger.warning(f"Order {order_id} ended with status: {order.status}")
                    return None
            except Exception as e:
                logger.error(f"Error checking order {order_id}: {e}")
            time.sleep(1)

        logger.warning(f"Order {order_id} not filled within {timeout}s")
        return None

    def get_market_context(self) -> dict:
        """
        Get broad market context (SPY trend, volatility proxy) for AI prompts.
        """
        context = {
            "spy_trend": "unknown",
            "spy_change_1d": 0.0,
            "spy_change_5d": 0.0,
            "market_volatility": "normal",
        }
        try:
            bars = self.get_bars("SPY", timeframe="1Day", limit=30)
            if not bars.empty and len(bars) >= 5:
                close = bars["close"]
                context["spy_change_1d"] = (close.iloc[-1] / close.iloc[-2] - 1)
                context["spy_change_5d"] = (close.iloc[-1] / close.iloc[-5] - 1)

                sma_10 = close.rolling(10).mean().iloc[-1]
                sma_20 = close.rolling(20).mean().iloc[-1]
                if sma_10 > sma_20:
                    context["spy_trend"] = "bullish"
                else:
                    context["spy_trend"] = "bearish"

                # Volatility proxy: recent ATR as % of price
                high = bars["high"]
                low = bars["low"]
                tr = pd.concat([
                    high - low,
                    (high - close.shift(1)).abs(),
                    (low - close.shift(1)).abs(),
                ], axis=1).max(axis=1)
                atr_pct = tr.rolling(14).mean().iloc[-1] / close.iloc[-1]
                if atr_pct > 0.02:
                    context["market_volatility"] = "high"
                elif atr_pct < 0.008:
                    context["market_volatility"] = "low"

        except Exception as e:
            logger.error(f"Error getting market context: {e}")

        return context
