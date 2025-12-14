# FTMO 200K Live Trading Checklist

## Pre-Launch Verification

### 1. Environment Setup
- [ ] Windows VM is running with stable internet connection
- [ ] MetaTrader 5 is installed and can connect to FTMO servers
- [ ] Python environment is set up with all dependencies
- [ ] `.env` file is configured with correct credentials:
  - `MT5_SERVER`: FTMO server name (e.g., "FTMO-Demo" or "FTMO-Server")
  - `MT5_LOGIN`: Your account login number
  - `MT5_PASSWORD`: Your account password
  - `SCAN_INTERVAL_HOURS`: Set to 4 (recommended)

### 2. Account Configuration Verification
Confirm these settings in `ftmo_config.py`:
- [ ] `account_size`: 200000.0 (FTMO 200K)
- [ ] `risk_per_trade_pct`: 0.95 (matches backtest)
- [ ] `min_confluence_score`: 3 (optimized setting)
- [ ] `weekend_close_enabled`: False (Swing account)
- [ ] `max_concurrent_trades`: 10

### 3. FTMO Rules Compliance
Your bot is configured for these limits:
| Rule | FTMO Limit | Bot Safety Buffer |
|------|------------|-------------------|
| Max Daily Loss | 5% ($10,000) | Halts at 4.2% ($8,400) |
| Max Total Drawdown | 10% ($20,000) | Emergency at 7% ($14,000) |
| Min Trading Days | 4 days | Tracked automatically |
| Phase 1 Target | 10% ($20,000) | Ultra-safe mode at 9% |
| Phase 2 Target | 5% ($10,000) | Ultra-safe mode at 4% |

### 4. Spread Limits Verification
Maximum allowed spreads (in pips) before trade rejection:
- Major pairs (EUR/USD, USD/JPY): 2.0-2.5 pips
- Cross pairs (GBP/JPY): 3.0-5.0 pips
- Gold (XAU/USD): 40 pips
- Default for unlisted: 5 pips

---

## Launch Day Procedure

### Step 1: Pre-Market Checks (Before 5:00 AM UTC Sunday)
1. [ ] Verify no pending orders from previous session
2. [ ] Check `trading_days.json` for correct challenge dates
3. [ ] Review `pending_setups.json` is empty or has valid setups
4. [ ] Confirm challenge state in `challenge_state.json`

### Step 2: Start the Bot
```bash
# Navigate to bot directory
cd /path/to/trading-bot

# Start the bot
python main_live_bot.py
```

### Step 3: Verify Successful Start
Watch for these log messages:
- [ ] "Connected: [login] @ [server]"
- [ ] "Balance: $200,000.00" (or current balance)
- [ ] "Mapped XX/34 symbols"
- [ ] "Challenge Risk Manager initialized with ELITE PROTECTION"
- [ ] Verify `challenge_state.json` shows correct phase and drawdown figures after MT5 sync

### Step 4: First Scan Verification
After the first 4-hour scan completes:
- [ ] Check `logs/tradr_live.log` for scan results
- [ ] Verify no error messages in logs
- [ ] Confirm pending orders (if any) appear in MT5

---

## Daily Monitoring Checklist

### Morning Check (9:00 AM UTC)
- [ ] Verify bot is running (check process or logs)
- [ ] Review overnight trading activity in `tradr_live.log`
- [ ] Check current positions in MT5 terminal
- [ ] Verify equity is within acceptable range

### End of Day Check (9:00 PM UTC)
- [ ] Review daily P&L in MT5
- [ ] Check trading days count: `python -c "import json; print(json.load(open('trading_days.json'))['trading_days'])"`
- [ ] Verify no stuck pending orders
- [ ] Confirm bot is still running

### Weekly Check (Every Sunday)
- [ ] Review weekly performance
- [ ] Check minimum trading days progress
- [ ] Verify challenge end date timeline
- [ ] Review and archive old log files

---

## Trading Days Tracking

### Current Status Command
```python
# Check trading days status
import json
with open('trading_days.json', 'r') as f:
    data = json.load(f)
    print(f"Trading Days: {len(data['trading_days'])}/4")
    print(f"Days Traded: {data['trading_days']}")
    print(f"Challenge End: {data.get('challenge_end_date', 'Not set')}")
```

### Starting a New Challenge
```python
# When starting Phase 1, Phase 2, or resetting
# The bot auto-starts if no challenge is active
# To manually reset:
from main_live_bot import LiveTradingBot
bot = LiveTradingBot()
bot.start_new_challenge(duration_days=30)
```

---

## Risk Management Thresholds

### Automatic Risk Reduction
| Daily Loss Level | Action Taken |
|------------------|--------------|
| 0-2.5% | Full risk (0.95%) |
| 2.5-3.5% | Reduced risk (0.5%) |
| 3.5-4.2% | Ultra-safe (0.25%) |
| >4.2% | HALT - No new trades |

### Total Drawdown Protection
| Drawdown Level | Action Taken |
|----------------|--------------|
| 0-5% | Normal trading |
| 5-7% | Warning + reduced risk |
| >7% | Emergency mode (0.25% risk) |

---

## Emergency Procedures

### If Bot Stops Unexpectedly
1. Check the log file: `logs/tradr_live.log`
2. Check for Python errors or MT5 connection issues
3. Restart the bot: `python main_live_bot.py`
4. Verify all pending orders are still valid

### If Approaching Daily Loss Limit
The bot automatically:
1. Reduces position sizes at 2.5% daily loss
2. Uses ultra-safe mode at 3.5% daily loss
3. Halts all new trades at 4.2% daily loss

Manual intervention:
1. Check positions in MT5
2. Consider manually closing losing positions
3. Do NOT restart trading until next day

### If Approaching Total Drawdown Limit
At 7% total drawdown:
1. Bot enters emergency mode
2. All new trades use 0.25% risk only
3. Consider reducing exposure manually
4. Review strategy performance

### Manual Order Cancellation
```bash
# Cancel all pending orders via MT5 terminal
# Or use Python:
python -c "from tradr.mt5.client import MT5Client; c = MT5Client(); c.connect(); c.cancel_all_pending_orders(); c.disconnect()"
```

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `main_live_bot.py` | Main trading bot executable |
| `ftmo_config.py` | FTMO-specific configuration |
| `strategy_core.py` | Core strategy logic (same as backtest) |
| `pending_setups.json` | Active pending trade setups |
| `trading_days.json` | Trading days tracking for FTMO |
| `challenge_state.json` | Challenge phase and status |
| `logs/tradr_live.log` | Trading activity logs |

---

## Configuration Summary

### Account Type: FTMO Swing Account
- **Can hold positions over weekends**: YES
- **News trading**: ALLOWED
- **Weekend close**: DISABLED

### Strategy Settings (Optimized)
| Parameter | Value |
|-----------|-------|
| Risk per trade | 0.95% |
| Min confluence | 3/7 |
| Min quality factors | 1 |
| Max concurrent trades | 10 |
| Scan interval | 4 hours |
| Pending order expiry | 24 hours |

### Live Market Safeguards
| Safeguard | Setting |
|-----------|---------|
| Slippage buffer | 2 pips |
| Spread validation | Enabled |
| Symbol mapping | OANDA -> FTMO |

---

## Final Pre-Live Verification

Before going live, verify:
1. [ ] Tested on FTMO Demo account for at least 1 week
2. [ ] All symbol mappings working correctly
3. [ ] Orders execute at expected prices
4. [ ] Partial closes and trailing stops working
5. [ ] Trading days tracking records correctly
6. [ ] Log files generating properly
7. [ ] Emergency procedures understood
8. [ ] FTMO rules memorized

---

## Support Commands

### Check Bot Status
```bash
# View recent log entries
tail -100 logs/tradr_live.log
```

### View Current Positions
```python
from tradr.mt5.client import MT5Client
client = MT5Client()
client.connect()
positions = client.get_my_positions()
for p in positions:
    print(f"{p.symbol}: {p.volume} lots, P/L: {p.profit}")
client.disconnect()
```

### View Pending Setups
```python
import json
with open('pending_setups.json', 'r') as f:
    setups = json.load(f)
    for symbol, setup in setups.items():
        print(f"{symbol}: {setup['direction']} @ {setup['entry_price']}")
```

---

## Windows VM Auto-Restart Setup

### Setting Up Automatic Recovery (Task Scheduler)
To ensure the bot restarts after VM reboots:

1. Open Task Scheduler on Windows VM
2. Create new task: "FTMO Trading Bot"
3. Trigger: "At startup"
4. Action: Start a program
   - Program: `python`
   - Arguments: `main_live_bot.py`
   - Start in: `C:\path\to\trading-bot`
5. Settings:
   - Run whether user is logged on or not
   - Run with highest privileges
   - If task fails, restart every 5 minutes

### Security Best Practices
- [ ] Store `.env` file securely (not in shared folders)
- [ ] Set restrictive file permissions on `.env` (only your user can read)
- [ ] Never commit `.env` to version control
- [ ] Use Windows Credential Manager for extra security if desired

---

**Good luck with your FTMO challenge!**

*Bot configured for $200K Swing Account with optimized settings.*
*Remember: The bot is designed to protect your account first, profits second.*
