from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import json
import os
import glob
from pathlib import Path
from datetime import datetime

from typing import Optional
from pydantic import BaseModel
import config
from trader import Trader
from cost_tracker import CostTracker
from quant_engine import QuantEngine
from intervention_tracker import InterventionTracker

class SettingsUpdate(BaseModel):
    strategy: str

class QuantSettingsUpdate(BaseModel):
    kama_enabled: Optional[bool] = None
    trend_enabled: Optional[bool] = None
    momentum_enabled: Optional[bool] = None
    auto_trade: Optional[bool] = None

class StockAnalysisRequest(BaseModel):
    symbol: str
    days_back: Optional[int] = 30

app = FastAPI(title="Trading Bot API")

# Enable CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = "data"
LOG_DIR = "logs"
SNAPSHOT_FILE = os.path.join(DATA_DIR, "positions_snapshot.json")
COST_FILE = os.path.join(LOG_DIR, "cost_tracker.json")
HISTORY_FILE = os.path.join(DATA_DIR, "trade_history.json")
QUANT_STATUS_FILE = os.path.join(DATA_DIR, "quant_status.json")

def load_json(path):
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            return None
    return None

@app.get("/api/status")
async def get_status():
    """Fetch live account status from Alpaca and local stats."""
    try:
        trader = Trader()
        account = trader.get_account()
        positions = trader.get_positions()

        cost_tracker = CostTracker()
        cost_summary = cost_tracker.get_summary()

        # Load bot status (execution times)
        bot_status = load_json(os.path.join(DATA_DIR, "bot_status.json"))

        # Load quant status if available
        quant_status = load_json(QUANT_STATUS_FILE)

        return {
            "account": {
                "equity": account["equity"],
                "cash": account["cash"],
                "daily_pnl": account["daily_pnl"],
                "daily_pnl_pct": account["daily_pnl_pct"],
                "buying_power": account["buying_power"],
            },
            "positions": positions,
            "costs": cost_summary,
            "bot_status": bot_status,
            "settings": {
                "strategy": config.STRATEGY_MODE,
                "quant_kama_enabled": config.QUANT_KAMA_ENABLED,
                "quant_trend_enabled": config.QUANT_TREND_ENABLED,
                "quant_momentum_enabled": config.QUANT_MOMENTUM_ENABLED,
                "quant_auto_trade": config.QUANT_AUTO_TRADE,
            },
            "quant_status": quant_status,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/history")
async def get_history():
    """Return historical trade data."""
    history = load_json(HISTORY_FILE) or []
    return history

@app.get("/api/logs")
async def get_logs(limit: int = 50):
    """Return the latest N lines from the most recent log file."""
    try:
        log_files = glob.glob(os.path.join(LOG_DIR, "trading_*.log"))
        if not log_files:
            return {"logs": []}
            
        latest_log = max(log_files, key=os.path.getmtime)
        with open(latest_log, "r") as f:
            lines = f.readlines()
            
        # Reverse to show latest on top
        lines.reverse()
        return {"logs": [line.strip() for line in lines[:limit]]}
    except Exception as e:
        return {"logs": [f"Error reading logs: {str(e)}"]}

@app.post("/api/settings")
async def update_settings(settings: SettingsUpdate):
    """Update trading strategy and persist to .env."""
    try:
        new_mode = settings.strategy.lower()
        if new_mode not in ["preservation", "aggressive"]:
            raise HTTPException(status_code=400, detail="Invalid strategy mode")

        _update_env_var("STRATEGY_MODE", new_mode)
        config.reload_config()

        return {"status": "success", "mode": new_mode}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/quant/status")
async def get_quant_status():
    """Return current quant strategy status and recent signals."""
    quant_status = load_json(QUANT_STATUS_FILE)
    return {
        "settings": {
            "kama_enabled": config.QUANT_KAMA_ENABLED,
            "trend_enabled": config.QUANT_TREND_ENABLED,
            "momentum_enabled": config.QUANT_MOMENTUM_ENABLED,
            "auto_trade": config.QUANT_AUTO_TRADE,
        },
        "kama_params": {
            "period": config.QUANT_KAMA_PERIOD,
            "fast": config.QUANT_KAMA_FAST,
            "slow": config.QUANT_KAMA_SLOW,
        },
        "trend_params": {
            "sma_short": config.QUANT_TREND_SMA_SHORT,
            "sma_long": config.QUANT_TREND_SMA_LONG,
        },
        "momentum_params": {
            "lookback_months": config.QUANT_MOMENTUM_LOOKBACK,
            "top_n": config.QUANT_MOMENTUM_TOP_N,
        },
        "signals": quant_status.get("signals", {}) if quant_status else {},
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/api/quant/settings")
async def update_quant_settings(settings: QuantSettingsUpdate):
    """Update quant strategy settings and persist to .env."""
    try:
        updates = {}
        if settings.kama_enabled is not None:
            updates["QUANT_KAMA_ENABLED"] = str(settings.kama_enabled).lower()
        if settings.trend_enabled is not None:
            updates["QUANT_TREND_ENABLED"] = str(settings.trend_enabled).lower()
        if settings.momentum_enabled is not None:
            updates["QUANT_MOMENTUM_ENABLED"] = str(settings.momentum_enabled).lower()
        if settings.auto_trade is not None:
            updates["QUANT_AUTO_TRADE"] = str(settings.auto_trade).lower()

        for key, value in updates.items():
            _update_env_var(key, value)

        config.reload_config()

        return {
            "status": "success",
            "settings": {
                "kama_enabled": config.QUANT_KAMA_ENABLED,
                "trend_enabled": config.QUANT_TREND_ENABLED,
                "momentum_enabled": config.QUANT_MOMENTUM_ENABLED,
                "auto_trade": config.QUANT_AUTO_TRADE,
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _update_env_var(key: str, value: str):
    """Update or append a key=value pair in the .env file."""
    env_path = ".env"
    lines = []
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            lines = f.readlines()

    found = False
    for i, line in enumerate(lines):
        if line.startswith(f"{key}="):
            lines[i] = f"{key}={value}\n"
            found = True
            break

    if not found:
        lines.append(f"{key}={value}\n")

    with open(env_path, "w") as f:
        f.writelines(lines)

@app.get("/api/quant/signals")
async def get_quant_signals():
    """Get current signals for all strategies and all symbols."""
    try:
        quant_engine = QuantEngine()
        if not quant_engine.is_active():
            return {"error": "No quant strategies enabled"}
        
        all_signals = quant_engine.get_all_current_signals()
        
        # Convert to JSON-serializable format
        result = {}
        for symbol, strategies in all_signals.items():
            result[symbol] = {}
            for strategy_name, signal in strategies.items():
                if signal:
                    result[symbol][strategy_name] = {
                        "action": signal.action,
                        "strength": signal.strength,
                        "reasoning": signal.reasoning,
                        "metrics": signal.metrics,
                        "timestamp": signal.timestamp,
                    }
                else:
                    result[symbol][strategy_name] = None
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/quant/signals/{symbol}")
async def get_quant_signals_for_symbol(symbol: str):
    """Get signals for a specific symbol."""
    try:
        quant_engine = QuantEngine()
        if not quant_engine.is_active():
            return {"error": "No quant strategies enabled"}
        
        all_signals = quant_engine.get_all_current_signals()
        
        if symbol not in all_signals:
            raise HTTPException(status_code=404, detail=f"Symbol {symbol} not in watchlist")
        
        strategies = all_signals[symbol]
        result = {}
        for strategy_name, signal in strategies.items():
            if signal:
                result[strategy_name] = {
                    "action": signal.action,
                    "strength": signal.strength,
                    "reasoning": signal.reasoning,
                    "metrics": signal.metrics,
                    "timestamp": signal.timestamp,
                }
            else:
                result[strategy_name] = None
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/quant/analyze")
async def analyze_stock(request: StockAnalysisRequest):
    """Analyze a custom stock through all strategies."""
    try:
        trader = Trader()
        quant_engine = QuantEngine()
        
        if not quant_engine.is_active():
            return {"error": "No quant strategies enabled"}
        
        result = quant_engine.analyze_single_stock(
            request.symbol.upper(),
            trader,
            days_back=request.days_back
        )
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/quant/interventions")
async def get_interventions(limit: int = 50, symbol: Optional[str] = None):
    """Get recent interventions (AI/risk manager decisions)."""
    try:
        tracker = InterventionTracker()
        interventions = tracker.get_intervention_history(
            symbol=symbol,
            limit=limit
        )
        
        # Convert to dict for JSON serialization
        return [
            {
                "signal_id": i.signal_id,
                "symbol": i.symbol,
                "intervener": i.intervener,
                "action": i.action,
                "reasoning": i.reasoning,
                "timestamp": i.timestamp,
                "original_action": i.original_action,
                "final_action": i.final_action,
                "strategy": i.strategy,
            }
            for i in interventions
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/quant/interventions/stats")
async def get_intervention_stats(days: int = 7):
    """Get rejection statistics."""
    try:
        tracker = InterventionTracker()
        stats = tracker.get_rejection_stats(days=days)
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/quant/dashboard")
async def get_quant_dashboard():
    """Get comprehensive dashboard summary."""
    try:
        quant_engine = QuantEngine()
        if not quant_engine.is_active():
            return {"error": "No quant strategies enabled"}
        
        summary = quant_engine.get_dashboard_summary()
        return summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/quant/performance")
async def get_strategy_performance():
    """Get performance metrics for each strategy."""
    try:
        tracker = InterventionTracker()
        approval_rates = tracker.get_strategy_approval_rates()
        return approval_rates
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
