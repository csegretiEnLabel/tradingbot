# 🤖 Self-Sustaining AI Trading Agent

An AI-powered trading agent that uses Claude for market analysis and Alpaca for execution.
The agent starts on paper trading, then graduates to live trading with $50 — and must
earn enough to pay for its own API costs or it dies.

## Architecture

```
┌─────────────────────────────────────────────────┐
│                   main.py                        │
│              (Scheduler / Loop)                   │
├─────────────┬───────────────┬───────────────────┤
│  agent.py   │  trader.py    │  risk_manager.py  │
│  (Brain)    │  (Execution)  │  (Guard Rails)    │
│  Claude API │  Alpaca API   │  Position Limits  │
├─────────────┴───────────────┴───────────────────┤
│              cost_tracker.py                      │
│         (API cost accounting & kill switch)       │
├─────────────────────────────────────────────────┤
│              strategy.py                          │
│     (Momentum / Mean-Reversion / Sentiment)      │
└─────────────────────────────────────────────────┘
```

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set up environment variables
```bash
cp .env.example .env
# Edit .env with your API keys
```

### 3. Get API Keys
- **Alpaca**: Sign up at https://alpaca.markets → Paper Trading → API Keys
- **Anthropic**: Get key at https://console.anthropic.com

### 4. Run in paper mode (default)
```bash
python main.py
```

### 5. Graduate to live trading
Once you're confident in paper results:
```bash
# In .env, change:
TRADING_MODE=live
# Fund your Alpaca account with $50
python main.py
```

## How It Works

1. **Market Scan**: Every cycle, the agent scans for opportunities using technical indicators
2. **AI Analysis**: Claude analyzes the top candidates with price action, volume, and momentum
3. **Risk Check**: Every trade passes through the risk manager (max position size, daily loss limit, etc.)
4. **Execution**: Trades are placed via Alpaca with bracket orders (take-profit + stop-loss)
5. **Cost Tracking**: Every Claude API call is logged; if costs exceed profits, the agent slows down
6. **Self-Assessment**: Daily P&L review determines if the agent should continue or shut down

## Risk Management (Critical for $50 account)

- Max 20% of portfolio per position ($10 on a $50 account)
- Max 3 concurrent positions
- Daily loss limit: 3% of portfolio
- Always uses stop-losses (2-3% per trade)
- No overnight holds on volatile stocks (configurable)
- Kill switch if cumulative API costs > cumulative profits for 5 consecutive days

## Cost Optimization

The agent is designed to minimize API costs:
- Uses Claude Haiku for routine scans (~$0.0003/call)
- Only escalates to Sonnet for high-conviction trade decisions (~$0.003/call)
- Caches market analysis for 15 minutes
- Limits to 4 analysis cycles per trading day
- Estimated daily API cost: $0.05-0.15

## Configuration

All tunable parameters are in `config.py`. Key settings:

| Parameter | Default | Description |
|-----------|---------|-------------|
| MAX_POSITION_PCT | 0.20 | Max % of portfolio per trade |
| MAX_POSITIONS | 3 | Max concurrent positions |
| DAILY_LOSS_LIMIT_PCT | 0.03 | Stop trading after this daily loss |
| STOP_LOSS_PCT | 0.025 | Default stop-loss per trade |
| TAKE_PROFIT_PCT | 0.05 | Default take-profit per trade |
| ANALYSIS_INTERVAL_MIN | 60 | Minutes between market scans |
| COST_KILL_THRESHOLD_DAYS | 5 | Days of net loss before shutdown |

## File Structure

```
trading-agent/
├── main.py              # Entry point & scheduler
├── config.py            # All configuration
├── agent.py             # AI brain (Claude integration)
├── trader.py            # Alpaca API wrapper
├── strategy.py          # Technical analysis & signals
├── risk_manager.py      # Position sizing & limits
├── cost_tracker.py      # API cost tracking & kill switch
├── requirements.txt     # Python dependencies
├── .env.example         # Environment template
└── logs/                # Trading logs (auto-created)
```

## Disclaimer

This is an experimental trading bot. Trading involves risk of loss.
The $50 constraint is intentional — treat it as tuition, not investment.
Never fund more than you're willing to lose.
