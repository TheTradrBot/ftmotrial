# MT5 5ers Trading Bot - Complete System Architecture

**Last Updated**: 2026-01-03  
**Version**: 4.1 (H1 Exit Correction + Architecture Parity Verified)

---

## Table of Contents
1. [System Overview](#system-overview)
2. [Component Architecture](#component-architecture)
3. [Data Flow](#data-flow)
4. [H1 Exit Correction](#h1-exit-correction)
5. [Optimization System](#optimization-system)
6. [Live Bot Features (Dec 2025)](#live-bot-features-dec-2025)
7. [Parameter Management](#parameter-management)
8. [Risk Management](#risk-management)
9. [Multi-Broker Support](#multi-broker-support)
10. [Output Management](#output-management)
11. [Deployment Architecture](#deployment-architecture)

---

## System Overview

### Purpose
Automated trading system for **5ers 60K High Stakes Challenge** accounts using a 7-Pillar Confluence strategy with ADX regime detection and professional quantitative optimization.

### Key Metrics (Current Performance)
- **Account Size**: $60,000 USD (5ers High Stakes)
- **Win Rate**: Target 55%+
- **Max Drawdown**: <10% (5ers hard limit)
- **Daily Loss Limit**: <5% (5ers rule)
- **Risk per Trade**: 0.6% = $360 per R
- **Symbols**: 25 tradable assets

### Technology Stack
- **Language**: Python 3.11+
- **Trading Platform**: MetaTrader 5 (Windows VM)
- **Brokers**: Forex.com Demo (testing), 5ers Live (production)
- **Optimization**: Optuna 3.x (TPE + NSGA-II)
- **Data Storage**: SQLite (optimization state), CSV (historical OHLCV)
- **Deployment**: Windows Server 2016 VM (live bot) + Linux (optimizer)

### Two-Environment Architecture
```
┌─────────────────────────────────────────────────────────────┐
│ LIVE BOT (Windows VM)                                       │
│ main_live_bot.py - MT5 required                             │
│ - Runs 24/7 via Task Scheduler                              │
│ - Scans at 22:05 UTC (after daily close)                    │
│ - Spread monitoring every 10 min                            │
│ - 3-tier graduated risk management                          │
│ - Partial take profits via market orders                    │
└─────────────────────────────────────────────────────────────┘
                        ↕ Git sync
┌─────────────────────────────────────────────────────────────┐
│ OPTIMIZER (Linux/Replit)                                    │
│ ftmo_challenge_analyzer.py - NO MT5 required                │
│ - Runs anywhere (Replit, local, dev container)              │
│ - TPE or NSGA-II optimization                               │
│ - Saves best params to params/current_params.json           │
└─────────────────────────────────────────────────────────────┘
```

---

## Component Architecture

```
ftmotrial/
├── main_live_bot.py              # PRODUCTION: Live MT5 trading execution
├── ftmo_challenge_analyzer.py    # OPTIMIZATION: Backtest & parameter tuning
├── strategy_core.py              # CORE: Trading strategy logic (7 pillars)
├── broker_config.py              # BROKERS: Multi-broker configuration
├── config.py                     # SETTINGS: Account, symbols, contract specs
├── ftmo_config.py                # 5ERS: Challenge rules, risk limits
├── symbol_mapping.py             # UTILS: Internal ↔ Broker symbol conversion
│
├── params/                       # PARAMETER MANAGEMENT
│   ├── optimization_config.py    # Unified config loader
│   ├── optimization_config.json  # Configuration file (DB, modes, trials)
│   ├── current_params.json       # Active strategy parameters
│   ├── params_loader.py          # Load/save parameter utilities
│   └── history/                  # Parameter version history
│
├── tradr/                        # CORE MODULES
│   ├── mt5/
│   │   ├── client.py             # MT5 API wrapper (Windows only)
│   │   └── reconnect.py          # Exponential backoff reconnection
│   ├── risk/
│   │   ├── manager.py            # Drawdown tracking + graduated risk
│   │   └── position_sizing.py    # Lot size calculations
│   ├── strategy/
│   │   └── confluence.py         # 7-Pillar confluence system
│   └── utils/
│       └── output_manager.py     # Centralized output file management
│
├── data/                         # HISTORICAL DATA
│   ├── ohlcv/                    # OHLCV CSV files (2003-2025)
│   │   ├── EUR_USD_D1_2003_2025.csv
│   │   ├── XAU_USD_H4_2003_2025.csv
│   │   └── ... (25 assets × 4 timeframes)
│   └── sr_levels/                # S/R level database (unused)
│
├── ftmo_analysis_output/         # OPTIMIZATION RESULTS
│   ├── NSGA/                     # Multi-objective runs
│   ├── NSGA_H4/                  # H4 timeframe runs
│   ├── TPE/                      # Single-objective runs
│   ├── TPE_H4/                   # H4 timeframe runs
│   └── VALIDATE/                 # Validation outputs
│
├── docs/                         # DOCUMENTATION
│   ├── ARCHITECTURE.md           # This file (system design)
│   ├── STRATEGY_GUIDE.md         # Trading strategy deep dive
│   ├── OPTIMIZATION_FLOW.md      # Optimization process
│   ├── API_REFERENCE.md          # Code API documentation
│   ├── DEPLOYMENT_GUIDE.md       # Setup & deployment instructions
│   └── CHANGELOG.md              # Version history
│
├── scripts/                      # UTILITIES
│   ├── download_oanda_data.py    # Download historical data
│   ├── monitor_optimization.sh   # Watch optimization progress
│   └── validate_broker_symbols.py # Validate symbol mappings
│
└── .github/
    └── copilot-instructions.md   # AI assistant context
```

---

## Data Flow

### 1. Optimization → Live Bot Flow

```
┌─────────────────────────────────────────────────────────────┐
│ STEP 1: OPTIMIZATION (Linux/Replit)                         │
├─────────────────────────────────────────────────────────────┤
│ ftmo_challenge_analyzer.py                                  │
│   ├── Load params/optimization_config.json                  │
│   ├── Initialize Optuna study (SQLite DB)                   │
│   ├── Run trials (TPE or NSGA-II)                           │
│   │   ├── Suggest parameters                                │
│   │   ├── Backtest Jan-Sep 2024 (training)                  │
│   │   ├── Validate Oct-Dec 2024 (OOS)                       │
│   │   └── Score: R + Sharpe + Win Rate                      │
│   ├── Select best trial (OOS validation)                    │
│   └── SAVE → params/current_params.json                     │
└─────────────────────────────────────────────────────────────┘
                        ↓ Git sync
┌─────────────────────────────────────────────────────────────┐
│ STEP 2: LIVE TRADING (Windows VM)                           │
├─────────────────────────────────────────────────────────────┤
│ main_live_bot.py                                            │
│   ├── LOAD params/current_params.json (startup)             │
│   ├── Connect to MT5 (broker-specific server)               │
│   ├── At 22:05 UTC: Scan 25 assets (daily close)            │
│   │   ├── compute_confluence() with loaded params           │
│   │   ├── quality_factors = max(1, confluence_score // 3)   │
│   │   ├── apply_volatile_asset_boost()                      │
│   │   └── Check spread, if wide → save to awaiting_spread   │
│   ├── Every 10 min: Check awaiting_spread.json              │
│   │   └── Good spread → Execute with market order           │
│   ├── Every 30 sec: Manage positions                        │
│   │   ├── Partial exits (45% at 0.8R, 30% at 2R, 25% at 3R) │
│   │   ├── Move SL to BE after TP1                           │
│   │   └── 3-tier risk checks                                │
│   └── Log to logs/tradr_live.log                            │
└─────────────────────────────────────────────────────────────┘
```

### 2. Signal Flow (Daily Close Scan)

```
22:05 UTC Daily Close
         ↓
scan_all_symbols()
         ↓
For each symbol:
    ├── Get MT5 data (24mo, 104wk, 500d)
    ├── compute_confluence(data, params)
    │   ├── Check ADX regime (Trend/Range/Transition)
    │   ├── Calculate 7-pillar confluence score
    │   ├── quality_factors = max(1, confluence_score // 3)
    │   └── apply_volatile_asset_boost(symbol, score, quality)
    ├── If confluence >= MIN_CONFLUENCE:
    │   ├── Check spread
    │   ├── If spread OK → Execute with market order
    │   └── If spread wide → Save to awaiting_spread.json
    └── Log signal details
         ↓
Every 10 minutes:
    ├── Load awaiting_spread.json
    ├── For each pending signal:
    │   ├── Check current spread
    │   ├── If spread improved → Execute market order
    │   └── If expired (12h) → Remove
    └── Save updated awaiting_spread.json
```

---

## H1 Exit Correction

### Problem Statement
D1 backtests cannot determine if SL or TP hit first when both are breached on the same daily candle. This led to 467 trades in run_009 being incorrectly classified as LOSS.

### Solution: Timestamp-Based H1 Lookup
```python
# In strategy_core.py _correct_trades_with_h1()
# Filter H1 candles to the trade's exit window
for candle in h1_candles:
    if entry_dt < candle['time'] <= exit_dt:
        # Check if TP1 hit before SL
        if candle['high'] >= tp1:
            return True  # Trade is actually a WIN
        if candle['low'] <= sl:
            return False  # Trade is correctly a LOSS
```

### Key Implementation Details
1. **Flat sorted H1 list**: Pre-sort by timestamp for efficient lookup
2. **Entry-based filter**: `entry_dt < time <= exit_dt` (exclusive of entry time)
3. **D1 candle period**: 22:00 UTC to 22:00 UTC next day (not calendar date)
4. **Date range fix**: Use `end=2026-01-01` to include 2025 H1 data

### Impact
| Metric | Before H1 Correction | After H1 Correction |
|--------|---------------------|--------------------|
| Win Rate (run_009) | 49.2% | 71.0% |
| Trades Corrected | 0 | 467 |
| Current Simulation | 42.8% | 45.1% (4 corrections) |

---

## Optimization System

### Dual-Mode Architecture

#### Mode 1: TPE Single-Objective (Faster)
```bash
python ftmo_challenge_analyzer.py --single --trials 100
```

**Algorithm**: Tree-structured Parzen Estimator (Bayesian optimization)  
**Objective**: Composite score = R + Sharpe_bonus + PF_bonus + WR_bonus - penalties  
**Speed**: ~1.5 min/trial  
**Output**: `ftmo_analysis_output/TPE/`

#### Mode 2: NSGA-II Multi-Objective (Recommended)
```bash
python ftmo_challenge_analyzer.py --multi --trials 100
```

**Algorithm**: Non-dominated Sorting Genetic Algorithm II  
**Objectives**:
1. **Maximize Total R** (profit in risk units)
2. **Maximize Sharpe Ratio** (risk-adjusted returns)
3. **Maximize Win Rate** (consistency)

**Speed**: ~1.5 min/trial  
**Output**: `ftmo_analysis_output/NSGA/`

### Configuration File: `params/optimization_config.json`

```json
{
  "db_path": "sqlite:///ftmo_optimization.db",
  "study_name": "ftmo_unified_study",
  "use_multi_objective": true,
  "use_adx_regime_filter": true,
  "n_trials": 500,
  "training_start": "2024-01-01",
  "training_end": "2024-09-30",
  "validation_start": "2024-10-01",
  "validation_end": "2024-12-31"
}
```

---

## Live Bot Features (Dec 2025)

### Daily Close Scanning
- **Scan Time**: Only at 22:05 UTC (after NY close)
- **Why**: Ensures complete daily candles, matches backtest exactly
- **Benefit**: No partial candle analysis, consistent with TPE optimizer

### Spread Monitoring System
```python
# After daily close scan:
if spread > MAX_SPREAD[symbol]:
    save_to_awaiting_spread(signal)  # Check again later

# Every 10 minutes:
for signal in awaiting_spread.json:
    if spread_improved():
        execute_market_order(signal)
    elif expired(12h):
        remove(signal)
```

### Entry Filter: Spread Quality Only
- **No session filter** - all signals checked for spread only
- **Spread OK** → Execute immediately with market order
- **Spread wide** → Save to `awaiting_spread.json` for retry
- **Signal expiry**: 12 hours after creation

### 3-Tier Graduated Risk Management

| Tier | Daily DD | Action |
|------|----------|--------|
| 1 | ≥2.0% | Reduce risk: 0.6% → 0.4% (33% reduction) |
| 2 | ≥3.5% | Cancel all pending orders |
| 3 | ≥4.5% | Emergency close ALL positions |

### Partial Take Profits (Market Orders)

| Level | Profit Target | Close % | Action |
|-------|---------------|---------|--------|
| TP1 | 0.8-1R | 45% | Move SL to breakeven + buffer |
| TP2 | 2R | 30% | Trail remaining position |
| TP3 | 3-4R | 25% | Close final portion |

### Live Bot Synced with TPE Optimizer
**CRITICAL**: Both use IDENTICAL quality factors calculation:
```python
# BOTH use this formula (strategy_core.py generate_signals):
quality_factors = max(1, confluence_score // 3)

# BOTH apply volatile asset boost:
boosted_confluence, boosted_quality = apply_volatile_asset_boost(
    symbol, confluence_score, quality_factors, params.volatile_asset_boost
)

# BOTH use same active threshold:
min_quality_for_active = max(1, params.min_quality_factors - 1)
if boosted_confluence >= MIN_CONFLUENCE and boosted_quality >= min_quality_for_active:
    is_active = True
```

---

## Parameter Management

### File Structure

```
params/
├── optimization_config.py        # Config loader class
├── optimization_config.json      # Runtime configuration
├── current_params.json           # Active parameters (live bot)
├── params_loader.py              # Utility functions
└── history/                      # Version history (timestamped backups)
```

### Loading in Live Bot

```python
# main_live_bot.py
from params.params_loader import load_strategy_params

params = load_strategy_params()  # Loads current_params.json
min_conf = params.min_confluence_score  # Use in strategy logic
```

**CRITICAL**: Parameters are NEVER hardcoded in source files. Always load from JSON.

---

## Risk Management

### 5ers Challenge Rules (Hardcoded Limits)

**ftmo_config.py**:
```python
FTMO_CONFIG = {
    "account_size": 60000,
    "max_daily_loss_pct": 5.0,      # $3,000 max daily loss
    "max_total_drawdown_pct": 10.0,  # $6,000 max total drawdown
    "phase_1_target_pct": 8.0,       # $4,800 profit target (Step 1)
    "phase_2_target_pct": 5.0,       # $3,000 profit target (Step 2)
    "emergency_stop_pct": 7.0,       # Emergency halt at 7% DD
    "daily_halt_pct": 4.5            # Halt trading at 4.5% daily loss
}
```

### Pre-Trade Risk Checks

```python
class RiskManager:
    def can_trade(self, symbol: str, risk_pct: float) -> Tuple[bool, str]:
        # Check 1: Daily loss limit (graduated)
        if daily_loss_pct >= 4.5:
            return False, "Emergency stop - close all positions"
        if daily_loss_pct >= 3.5:
            return False, "Cancel pending orders"
        if daily_loss_pct >= 2.0:
            risk_pct *= 0.67  # Reduce to 0.4%
        
        # Check 2: Total drawdown
        if total_drawdown_pct > 7.0:
            return False, "Emergency drawdown stop"
        
        # Check 3: Spread validation
        if spread > MAX_SPREAD[symbol]:
            return False, "Spread too wide"
        
        return True, "OK"
```

### Position Sizing

```python
def calculate_lot_size(
    account_size: float,    # $60,000
    risk_pct: float,        # 0.6%
    sl_pips: float,         # Stop loss distance
    symbol: str             # For pip value lookup
) -> float:
    """
    Example:
        Account: $60,000
        Risk: 0.6% = $360
        SL: 50 pips
        EURUSD pip value: $10/pip (standard lot)
        
        Lot size = $360 / (50 pips × $10) = 0.72 lots
    """
    risk_amount = account_size * (risk_pct / 100)
    pip_value = get_contract_specs(symbol)['pip_value']
    return round(risk_amount / (sl_pips * pip_value), 2)
```

---

## Multi-Broker Support

### Supported Brokers

| Broker | Account | Purpose | Leverage |
|--------|---------|---------|----------|
| Forex.com Demo | $50,000 | Testing before live | 1:100 |
| 5ers Live | $60,000 | Production trading | 1:100 |

### Configuration

Set in `.env`:
```env
BROKER_TYPE=forexcom_demo  # or fiveers_live
MT5_SERVER=Forex.comGlobal-Demo
MT5_LOGIN=22936023
MT5_PASSWORD=xxx
```

### Symbol Mapping

```python
from symbol_mapping import get_broker_symbol, get_internal_symbol

# Internal (OANDA format) → Broker-specific
get_broker_symbol("EUR_USD", "forexcom")   # → "EURUSD"
get_broker_symbol("XAU_USD", "forexcom")   # → "XAUUSD"
get_broker_symbol("SPX500_USD", "forexcom") # → "SPX500"
get_broker_symbol("NAS100_USD", "forexcom") # → "NAS100"

# Broker-specific → Internal
get_internal_symbol("EURUSD", "forexcom")   # → "EUR_USD"
```

### Tradable Symbols (25 total)

**Forex (20)**:
EUR_USD, GBP_USD, USD_JPY, USD_CHF, AUD_USD, NZD_USD, USD_CAD,
EUR_GBP, EUR_JPY, GBP_JPY, EUR_CHF, EUR_AUD, EUR_CAD, EUR_NZD,
GBP_AUD, GBP_CAD, GBP_NZD, AUD_CAD, AUD_NZD, NZD_CAD

**Metals (1)**:
XAU_USD (Gold)

**Indices (3)**:
SPX500_USD, NAS100_USD, UK100_GBP

**Crypto (1)**:
BTC_USD

---

## Output Management

### Directory Structure

```
ftmo_analysis_output/
├── NSGA/                          # Multi-objective optimization runs
│   ├── optimization.log           # Real-time progress
│   ├── run.log                    # Full console output
│   ├── best_trades_*.csv          # Trade exports
│   └── optimization_report.csv    # Final summary
│
├── TPE/                           # Single-objective runs
│   └── ... (same structure)
│
└── DIRECTORY_GUIDE.md             # Explains output structure
```

### Live Bot State Files

```
# Working directory on Windows VM
C:\Users\Administrator\ftmotrial\
├── pending_setups.json            # Awaiting activation
├── awaiting_spread.json           # Signals waiting for good spread
├── partial_exits.json             # TP tracking
└── logs/
    └── tradr_live.log             # Trading activity log
```

---

## Deployment Architecture

### Windows VM (Production)
```
OS: Windows Server 2016
Python: 3.11.7 (C:\Users\Administrator\AppData\Local\Programs\Python\Python311)
Virtual Environment: C:\Users\Administrator\ftmotrial\venv
MT5: MetaTrader 5 with broker terminal running
Scheduler: Task Scheduler (FTMO_Live_Bot task)

Key Commands:
  cd C:\Users\Administrator\ftmotrial
  git pull
  .\venv\Scripts\Activate.ps1
  python main_live_bot.py
```

### Task Scheduler Configuration
- **Task Name**: FTMO_Live_Bot
- **Trigger**: At startup, then every 4 hours
- **Action**: Run main_live_bot.py
- **Settings**: Restart on failure, run on battery

### Development Environment (Linux)
```
OS: Ubuntu 24.04 LTS (dev container)
Python: 3.11+
Purpose: Optimization, testing, code development

Key Commands:
  python ftmo_challenge_analyzer.py --single --trials 100
  ./run_optimization.sh --single --trials 100
```

---

## Troubleshooting

### Common Issues

**Issue**: "No module named 'MetaTrader5'"  
**Solution**: Windows only - install via `pip install MetaTrader5`

**Issue**: "Symbol not found" on MT5  
**Solution**: Check symbol_mapping.py for correct broker mapping

**Issue**: "Spread too wide"  
**Solution**: Signal saved to awaiting_spread.json, will retry

**Issue**: "Daily loss limit exceeded"  
**Solution**: Bot auto-reduces risk or halts per graduated tiers

### Debug Commands

```bash
# Check optimization status
python ftmo_challenge_analyzer.py --status

# Validate symbol mappings
python scripts/validate_broker_symbols.py

# Test MT5 connection (Windows)
python -c "from tradr.mt5.client import MT5Client; MT5Client().initialize()"
```

---

## References

### Internal Documentation
- [STRATEGY_GUIDE.md](STRATEGY_GUIDE.md) - 7-Pillar Confluence system
- [OPTIMIZATION_FLOW.md](OPTIMIZATION_FLOW.md) - Optimization process
- [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) - Setup instructions
- [CHANGELOG.md](CHANGELOG.md) - Version history

### External Resources
- [Optuna Documentation](https://optuna.readthedocs.io/)
- [MetaTrader 5 Python API](https://www.mql5.com/en/docs/integration/python_metatrader5)
- [5ers Challenge Rules](https://www.the5ers.com/)

---

**Maintained by**: TheTradrBot  
**Last Updated**: 2025-12-31
