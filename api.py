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

class SettingsUpdate(BaseModel):
    strategy: str

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
            "costs": cost_summary,
            "bot_status": bot_status,
            "settings": {
                "strategy": config.STRATEGY_MODE
            },
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
            
        # Update .env file
        env_path = ".env"
        lines = []
        if os.path.exists(env_path):
            with open(env_path, "r") as f:
                lines = f.readlines()
        
        # Check if STRATEGY_MODE exists, update or append
        found = False
        for i, line in enumerate(lines):
            if line.startswith("STRATEGY_MODE="):
                lines[i] = f"STRATEGY_MODE={new_mode}\n"
                found = True
                break
        
        if not found:
            lines.append(f"\nSTRATEGY_MODE={new_mode}\n")
            
        with open(env_path, "w") as f:
            f.writelines(lines)
            
        # Reload config in this process
        config.reload_config()
        
        return {"status": "success", "mode": new_mode}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
