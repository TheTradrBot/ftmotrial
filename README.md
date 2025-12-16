# FTMO 200K Trading Bot

A robust, production-hardened MetaTrader 5 automated trading bot designed for FTMO 200K challenge accounts, with professional-grade parameter optimization and live safety features.

## Key Improvements (Post-Fix Status)

This system has undergone major critical upgrades to ensure production safety:

| Issue | Status | Description |
|-------|--------|-------------|
| Position Sizing | **RESOLVED** | Symbol-specific pip values for all 34 assets (JPY pairs, Gold, BTC, indices) |
| Parameter Management | **RESOLVED** | Optimizer saves to `params/current_params.json` - no source code mutation |
| Transaction Costs | **RESOLVED** | Realistic spread + slippage deducted in all backtests |
| Look-Ahead Bias | **RESOLVED** | Timestamp-based multi-timeframe alignment |
| MT5 Connectivity | **RESOLVED** | Exponential backoff reconnection, heartbeat monitoring, partial fill handling |
| Spread Validation | **RESOLVED** | Pre-trade spread checks enforced |

## Components

### 1. main_live_bot.py
The primary live trading bot that:
- Runs 24/7 on a Windows VM with MetaTrader 5
- Executes trades using the "7 Confluence Pillars" strategy
- **Loads all parameters from `params/current_params.json` at startup**
- Includes comprehensive risk management for FTMO compliance
- Supports 34 assets (Forex, Metals, Crypto, Indices)
- Features robust MT5 auto-reconnection with exponential backoff

### 2. ftmo_challenge_analyzer.py
The optimization engine that:
- Backtests strategy using 2024 historical data (Training: Jan-Sep, Validation: Oct-Dec)
- Runs walk-forward optimization iterations
- **Saves optimized parameters to `params/current_params.json`** (does NOT modify source code)
- Deducts realistic transaction costs (spread + slippage + commission)
- Generates detailed performance reports in `ftmo_analysis_output/`

### 3. params/current_params.json
The single source of truth for all tunable strategy parameters:
- Loaded by both backtest and live trading systems
- Updated only by the optimizer
- Versioned with timestamps for traceability
- Includes transaction cost definitions

## Trading Strategy

The bot employs a "7 Confluence Pillars" strategy:
1. **HTF Bias** - Monthly/Weekly/Daily trend alignment
2. **Location** - Price at significant S/R zones
3. **Fibonacci** - Golden Pocket (0.382-0.886 retracement)
4. **Liquidity** - Sweep of equal highs/lows (liquidity grab)
5. **Structure** - Break of Structure (BOS) or Change of Character (CHoCH)
6. **Confirmation** - 4H candle pattern confirmation
7. **Risk:Reward** - Minimum 1:1 R:R ratio

## Risk Management

- Accurate symbol-specific position sizing (all 34 assets safe)
- Dynamic risk per trade (0.5-1.0% = $1,000-$2,000)
- Maximum concurrent trades limit
- Pre-trade FTMO rule violation checks
- Pre-trade spread validation
- 5 risk modes: Aggressive, Normal, Conservative, Ultra-Safe, Halted

## Setup

### Step 1: Run Optimizer (Generate Parameters)
```bash
python ftmo_challenge_analyzer.py
```
This generates optimized parameters and saves them to `params/current_params.json`.

### Step 2: Environment Variables
Create a `.env` file with:
```
MT5_SERVER=FTMO-Demo
MT5_LOGIN=your_account_number
MT5_PASSWORD=your_password
OANDA_API_KEY=your_oanda_key
OANDA_ACCOUNT_ID=your_account_id
```

### Step 3: Dependencies
```bash
pip install -r requirements.txt
```

## Usage

### Live Trading (Windows VM with MT5)
```bash
python main_live_bot.py
```
The bot automatically loads parameters from `params/current_params.json`.

### Run Optimization
```bash
python ftmo_challenge_analyzer.py
```
Outputs optimized parameters to `params/current_params.json` and creates backups.

### Status Server
```bash
python main.py
```

## Project Structure

```
├── main_live_bot.py              # Live trading bot (loads params from JSON)
├── ftmo_challenge_analyzer.py    # Optimization engine (saves params to JSON)
├── strategy_core.py              # Core strategy logic
├── ftmo_config.py                # FTMO configuration
├── params/
│   ├── current_params.json       # Single source of truth for parameters
│   └── params_loader.py          # Parameter loading utilities
├── tradr/                        # Trading infrastructure
│   ├── mt5/                      # MT5 client with auto-reconnection
│   ├── risk/                     # Risk management & position sizing
│   └── data/                     # Data providers
├── data/ohlcv/                   # Historical data (2023-2024)
├── ftmo_analysis_output/         # Analysis results
└── ftmo_optimization_backups/    # Parameter iteration backups
```

## Robustness Features

### Position Sizing Safety
- Correct pip values for all asset classes (standard forex, JPY pairs, metals, crypto, indices)
- Validated against contract specifications

### Backtesting Accuracy
- Transaction costs (spread + slippage) deducted from all simulated trades
- Timestamp-based multi-timeframe alignment prevents look-ahead bias
- Weekend filtering on entry signals

### Live Trading Safety
- Pre-trade spread validation before order placement
- MT5 connection monitoring with heartbeat
- Exponential backoff reconnection (handles network interruptions)
- Partial fill handling for large orders
- Graceful shutdown on system signals

### Professional Separation
- Optimization never modifies production source code
- Parameters stored in versioned JSON files
- Clear audit trail with timestamped backups

## Assessment

**Current Rating**: 7.5-8/10  
**Estimated FTMO Pass Probability**: 50-70%

**Status**: Ready for paper trading; monitor closely on live challenge. Recommend 1-2 weeks of demo testing before real challenge.

## License

Private - For personal use only.


## Optimization & Backtesting

The optimizer uses professional quant best practices:

- TRAINING PERIOD: January 1, 2024 – September 30, 2024 (in-sample optimization)
- VALIDATION PERIOD: October 1, 2024 – December 31, 2024 (out-of-sample test)
- FINAL BACKTEST: Full year 2024 with best parameters

All trades from the final full-year backtest are exported to:
`ftmo_analysis_output/all_trades_2024_full.csv`

Parameters are saved to `params/current_params.json`
