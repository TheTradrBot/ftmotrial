# PROJECT OVERVIEW - FTMO 200K Trading Bot

**START HERE** - This is the first file to read to understand the entire project.

---

## 1. Project Summary

This is an automated **MetaTrader 5 Trading Bot** designed specifically for **FTMO 200K Challenge accounts**, paired with a **Walk-Forward Optimization System**.

- **Account Size**: $200,000 USD
- **Platform**: MetaTrader 5 (MT5)
- **Strategy**: 7 Confluence Pillars (multi-timeframe analysis)
- **Optimization**: Parameter optimization saves to JSON config (no source code mutation)

### Post-Fix Status: All Critical Issues Resolved

| Issue | Status |
|-------|--------|
| Hardcoded pip values | **FIXED** - Symbol-specific pip values for all 34 assets |
| Source code mutation | **FIXED** - Optimizer saves to `params/current_params.json` only |
| Transaction costs | **FIXED** - Spread + slippage deducted in backtests |
| Look-ahead bias | **FIXED** - Timestamp-based multi-timeframe alignment |
| MT5 connectivity | **FIXED** - Exponential backoff reconnection with heartbeat |
| Spread validation | **FIXED** - Pre-trade checks enforced |

---

## 2. Two Main Components

### A. `main_live_bot.py` - Live Trading Bot
**Runs on: Windows VM with MetaTrader 5**

The primary live trading bot that:
- Executes trades 24/7 using the "7 Confluence Pillars" strategy
- Connects directly to MT5 for order execution
- **Loads all tunable parameters from `params/current_params.json` at startup**
- Includes comprehensive FTMO-compliant risk management
- Supports 34 assets (Forex, Metals, Crypto, Indices)
- Features robust auto-reconnection with exponential backoff

```bash
# Run on Windows VM with MT5 installed
python main_live_bot.py
```

### B. `ftmo_challenge_analyzer.py` - Walk-Forward Optimizer
**Runs on: Replit (no MT5 required)**

The optimization engine that:
- Backtests the strategy using 2024 historical OHLCV data
- Training Period: Jan-Sep 2024
- Validation Period: Oct-Dec 2024
- Runs multiple optimization iterations
- **Saves optimized parameters to `params/current_params.json`** (no source code modification)
- Deducts realistic transaction costs (spread + slippage)
- Generates detailed performance reports and trade logs

```bash
# Run on Replit
python ftmo_challenge_analyzer.py
```

---

## 3. How They Work Together

```
┌─────────────────────────────────────────────────────────────────┐
│                        REPLIT                                    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │           ftmo_challenge_analyzer.py                      │   │
│  │                                                           │   │
│  │   1. Load historical data (data/ohlcv/2024)               │   │
│  │   2. Run backtests with different parameters              │   │
│  │   3. Score results (win rate, R-multiple, drawdown)       │   │
│  │   4. Find optimal parameters                              │   │
│  │   5. SAVE PARAMETERS TO:                                  │   │
│  │      - params/current_params.json (single source of truth)│   │
│  │   6. Save backups to ftmo_optimization_backups/           │   │
│  │                                                           │   │
│  │   NOTE: Does NOT modify main_live_bot.py or other source  │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ Git sync / manual copy
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      WINDOWS VM                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              main_live_bot.py                             │   │
│  │                                                           │   │
│  │   1. LOAD params from params/current_params.json          │   │
│  │   2. Connect to MT5 broker (FTMO)                         │   │
│  │   3. Scan 34 assets every 4 hours                         │   │
│  │   4. Use loaded parameters for strategy decisions         │   │
│  │   5. Place pending orders when setup detected             │   │
│  │   6. Manage positions (partial TPs, trailing SL)          │   │
│  │   7. Enforce FTMO risk rules                              │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

**Key Parameter Flow:**
- All tunable parameters are stored in `params/current_params.json`
- Optimizer writes to this file after finding best parameters
- Live bot reads from this file at startup
- No source code is ever modified by the optimizer

**Parameters Controlled:**
- `min_confluence` - Minimum confluence score required (currently 5-6)
- `risk_per_trade_pct` - Risk per trade (0.5-1.0% = $1,000-$2,000)
- `min_quality_factors` - Minimum quality factors required
- TP levels (atr_tp1_multiplier, atr_tp2_multiplier, etc.)
- Transaction costs (spread_pips, slippage_pips, commission_per_lot)

---

## 4. Trading Strategy Overview

### The 7 Confluence Pillars

Each trade setup is scored 0-7 based on how many pillars align:

| # | Pillar | Description |
|---|--------|-------------|
| 1 | **HTF Bias** | Monthly/Weekly/Daily trend alignment |
| 2 | **Location** | Price at significant S/R zones |
| 3 | **Fibonacci** | Price in Golden Pocket (0.382-0.886 retracement) |
| 4 | **Liquidity** | Sweep of equal highs/lows (liquidity grab) |
| 5 | **Structure** | Break of Structure (BOS) or Change of Character (CHoCH) |
| 6 | **Confirmation** | 4H candle pattern confirmation |
| 7 | **Risk:Reward** | Minimum 1:1 R:R ratio |

**Trade Entry Requirement**: Minimum 5/7 confluence (configurable via params)

### Multi-Timeframe Analysis (Look-Ahead Bias Fixed)

- **Monthly (MN)**: Long-term bias
- **Weekly (W1)**: Intermediate bias  
- **Daily (D1)**: Primary trend direction
- **4-Hour (H4)**: Entry timeframe

**Important**: Timestamp-based alignment ensures no future data leaks into signals.

### 5 Take-Profit Levels with Partial Closes

| Level | R-Multiple | Close % |
|-------|------------|---------|
| TP1 | 1.5R | 10% |
| TP2 | 3.0R | 10% |
| TP3 | 5.0R | 15% |
| TP4 | 7.0R | 20% |
| TP5 | 10.0R | 45% |

---

## 5. Key Files Reference

### Core Trading Files
| File | Purpose |
|------|---------|
| `main_live_bot.py` | Live trading bot (runs on Windows VM, loads params from JSON) |
| `ftmo_challenge_analyzer.py` | Walk-forward optimizer (saves params to JSON) |
| `strategy_core.py` | Core strategy logic (7 Confluence Pillars) |
| `ftmo_config.py` | FTMO-specific configuration and risk parameters |

### Parameter Management
| File | Purpose |
|------|---------|
| `params/current_params.json` | **Single source of truth** for all tunable parameters |
| `params/params_loader.py` | Utility functions to load/save parameters |

### Data Files
| Directory/File | Purpose |
|----------------|---------|
| `data/ohlcv/` | Historical OHLCV data (2023-2024) for 34 assets |
| `tradr/mt5/client.py` | MT5 connection and order execution |
| `tradr/risk/manager.py` | Risk management logic |
| `tradr/risk/position_sizing.py` | Symbol-specific lot size calculations |
| `tradr/data/oanda.py` | OANDA API data fetching |

### Output Files
| Directory/File | Purpose |
|----------------|---------|
| `ftmo_analysis_output/` | Analysis results and trade logs |
| `ftmo_analysis_output/all_trades_jan_dec_2024.csv` | Complete trade history with all details |
| `ftmo_analysis_output/monthly_performance.csv` | Monthly breakdown |
| `ftmo_analysis_output/symbol_performance.csv` | Per-symbol performance |
| `ftmo_optimization_backups/` | Backup copies of each optimization iteration |

### Configuration Files
| File | Purpose |
|------|---------|
| `config.py` | General configuration and asset lists |
| `challenge_rules.py` | FTMO challenge rule definitions |
| `symbol_mapping.py` | Symbol format conversion (OANDA ↔ FTMO/MT5) |

---

## 6. Position Sizing (FIXED)

### Account Parameters
- **Account Size**: $200,000 USD
- **Risk Per Trade**: 0.5-1.0% ($1,000-$2,000 per trade)
- **Max Daily Loss**: 5% ($10,000)
- **Max Total Drawdown**: 10% ($20,000)

### Lot Size Calculation (Symbol-Specific Pip Values)

The formula (from `tradr/risk/position_sizing.py`):

```
lot_size = risk_usd / (stop_pips × pip_value_per_lot)
```

**Pip values are now correctly calculated per symbol:**

| Asset Type | Pip Size | Example |
|------------|----------|---------|
| Standard Forex | 0.0001 | EURUSD, GBPUSD |
| JPY Pairs | 0.01 | USDJPY, EURJPY |
| Gold (XAUUSD) | 0.01 | ~$1/pip/lot |
| Silver (XAGUSD) | 0.01 | |
| BTC/ETH | 1.0 | |
| Indices | 0.1 | SPX500, NAS100 |

**Example Calculations:**

| Asset | Risk | Stop Pips | Pip Value | Lot Size |
|-------|------|-----------|-----------|----------|
| EURUSD | $1,000 | 50 pips | $10/lot | 2.0 lots |
| USDJPY | $1,000 | 80 pips | $6.67/lot | 1.87 lots |
| XAUUSD | $1,500 | 300 pips | $1/lot | 5.0 lots |

---

## 7. How to Use

### Step 1: Run the Optimizer (Replit)
```bash
python ftmo_challenge_analyzer.py
```
This will:
1. Load 2024 historical data
2. Run backtests with various parameter combinations
3. Deduct realistic transaction costs
4. Output results to `ftmo_analysis_output/`
5. **Save optimal parameters to `params/current_params.json`**
6. Save backups to `ftmo_optimization_backups/`

### Step 2: Review Generated Parameters
Check `params/current_params.json` for the optimized settings.

### Step 3: Run the Live Bot (Windows VM)
```bash
python main_live_bot.py
```
Requires:
- MetaTrader 5 installed and running
- `.env` file with credentials:
```
MT5_SERVER=FTMO-Demo
MT5_LOGIN=your_account_number
MT5_PASSWORD=your_password
```
- The bot will automatically load parameters from `params/current_params.json`

### View Status (Replit Web Server)
```bash
python main.py
```
Runs a web server showing bot status and performance.

---

## 8. For New Replit Projects

### Getting Started Checklist

1. **Read this file first** (PROJECT_OVERVIEW.md)

2. **Run optimizer to generate parameters**:
   ```bash
   python ftmo_challenge_analyzer.py
   ```

3. **Check generated parameters** in `params/current_params.json`:
   - `min_confluence`: Minimum confluence score
   - `risk_per_trade_pct`: Risk per trade percentage
   - `max_concurrent_trades`: Maximum open trades
   - `transaction_costs`: Spread, slippage, commission settings

4. **Review backtest results**:
   - `ftmo_analysis_output/all_trades_jan_dec_2024.csv` - All trades with details
   - `ftmo_analysis_output/monthly_performance.csv` - Monthly breakdown

5. **Key metrics to monitor**:
   - Win Rate: Target 60%+
   - Average R-Multiple: Target 1.5R+
   - Max Drawdown: Must stay under 10%
   - Challenge Pass Rate: Target 70%+

### Environment Variables Required

```bash
# OANDA API (for data fetching on Replit)
OANDA_API_KEY=your_oanda_key
OANDA_ACCOUNT_ID=your_account_id

# MT5 (for live trading on Windows VM)
MT5_SERVER=FTMO-Demo
MT5_LOGIN=your_account_number
MT5_PASSWORD=your_password
```

---

## 9. FTMO Challenge Rules Summary

| Rule | Phase 1 | Phase 2 | Funded |
|------|---------|---------|--------|
| Profit Target | 10% ($20,000) | 5% ($10,000) | N/A |
| Max Daily Loss | 5% ($10,000) | 5% ($10,000) | 5% |
| Max Total DD | 10% ($20,000) | 10% ($20,000) | 10% |
| Min Trading Days | 4 | 4 | N/A |
| Time Limit | Unlimited | Unlimited | N/A |

---

## 10. Supported Assets (34 Total)

### Forex Majors (7)
EUR_USD, GBP_USD, USD_JPY, USD_CHF, USD_CAD, AUD_USD, NZD_USD

### Forex Crosses (21)
EUR_GBP, EUR_JPY, EUR_CHF, EUR_AUD, EUR_CAD, EUR_NZD,
GBP_JPY, GBP_CHF, GBP_AUD, GBP_CAD, GBP_NZD,
AUD_JPY, AUD_CHF, AUD_CAD, AUD_NZD,
NZD_JPY, NZD_CHF, NZD_CAD,
CAD_JPY, CAD_CHF, CHF_JPY

### Metals (2)
XAU_USD (Gold), XAG_USD (Silver)

### Indices (2)
SPX500_USD (S&P 500), NAS100_USD (Nasdaq 100)

### Crypto (2)
BTC_USD (Bitcoin), ETH_USD (Ethereum)

---

## Quick Reference Card

```
┌────────────────────────────────────────────────────────────┐
│                    QUICK COMMANDS                          │
├────────────────────────────────────────────────────────────┤
│ Generate parameters:   python ftmo_challenge_analyzer.py   │
│ Run live bot:          python main_live_bot.py             │
│ View web status:       python main.py                      │
├────────────────────────────────────────────────────────────┤
│                    KEY PARAMETERS                          │
├────────────────────────────────────────────────────────────┤
│ Parameters file:       params/current_params.json          │
│ Min Confluence:        5-6 (out of 7)                      │
│ Risk Per Trade:        0.5-1.0% ($1,000-$2,000)            │
│ Max Concurrent Trades: 3-7                                 │
│ Take Profits:          1.5R, 3R, 5R, 7R, 10R               │
├────────────────────────────────────────────────────────────┤
│                    KEY FILES                               │
├────────────────────────────────────────────────────────────┤
│ Parameter Config:      params/current_params.json          │
│ Strategy Logic:        strategy_core.py                    │
│ FTMO Config:           ftmo_config.py                      │
│ Trade Results:         ftmo_analysis_output/               │
│ Historical Data:       data/ohlcv/                         │
└────────────────────────────────────────────────────────────┘
```

---

## 11. Assessment

**Current Rating**: 7.5-8/10  
**Estimated FTMO Pass Probability**: 50-70%

**Status**: Ready for paper trading; monitor closely on live challenge.

**Recommendation**: Complete 1-2 weeks of demo testing before starting a real FTMO challenge.

---

*Last Updated: December 2024*
