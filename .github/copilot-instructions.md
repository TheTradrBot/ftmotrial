# 5ers 60K High Stakes Trading Bot - AI Agent Instructions

## Project Overview
Automated MetaTrader 5 trading bot for **5ers 60K High Stakes** Challenge accounts. Two-environment architecture:
- **Live Bot** (`main_live_bot.py`): Runs on Windows VM with MT5 installed
- **Optimizer** (`ftmo_challenge_analyzer.py`): Runs anywhere (Replit/local) - no MT5 required

### Multi-Broker Support (NEW)
The bot now supports multiple brokers for testing and production:
- **Forex.com Demo** ($50K): For testing before 5ers live
- **5ers Live** ($60K): Production trading

Set `BROKER_TYPE=forexcom_demo` or `BROKER_TYPE=fiveers_live` in `.env`

## Architecture & Data Flow

```
broker_config.py                 ← Multi-broker configuration (Forex.com, 5ers)
params/optimization_config.json  ← Optimization mode settings (multi-obj, ADX, etc.)
params/current_params.json       ← Optimized strategy parameters
         ↑                            ↓
ftmo_challenge_analyzer.py      main_live_bot.py
(Optuna optimization)           (loads params at startup)
         ↑
data/ohlcv/{SYMBOL}_{TF}_2003_2025.csv  (historical data)
```

### Key Modules
| File | Purpose |
|------|---------|
| `strategy_core.py` | Trading strategy logic - 7 Confluence Pillars, regime detection |
| `broker_config.py` | Multi-broker configuration (Forex.com, 5ers) |
| `params/params_loader.py` | Load/save optimized parameters from JSON |
| `params/optimization_config.py` | Unified optimization config (DB path, mode toggles) |
| `config.py` | Account settings, CONTRACT_SPECS (pip values), tradable symbols |
| `ftmo_config.py` | 5ers challenge rules, risk limits, TP/SL settings |
| `symbol_mapping.py` | Multi-broker symbol conversion (`EUR_USD` → broker-specific) |
| `tradr/mt5/client.py` | MT5 API wrapper (Windows only) |
| `tradr/risk/manager.py` | 5ers drawdown tracking, pre-trade risk checks |

## Critical Conventions

### Multi-Broker Symbol Mapping (Dec 31, 2025)
Symbol mapping is now broker-aware:
```python
from symbol_mapping import get_broker_symbol, get_internal_symbol

# Convert internal -> broker
broker_sym = get_broker_symbol("EUR_USD", "forexcom")  # -> "EURUSD"
broker_sym = get_broker_symbol("SPX500_USD", "fiveers")  # -> "US500.cash"
broker_sym = get_broker_symbol("SPX500_USD", "forexcom")  # -> "US500"
```

### Recent Bug Fixes (Jan 3, 2026)
**CRITICAL H1 EXIT CORRECTION**: Fixed D1 backtest same-candle SL/TP issue:
1. **Problem**: D1 can't determine if SL or TP hit first on same candle
2. **Impact**: 467 trades in run_009 incorrectly classified as LOSS (49.2% → 71.0% WR)
3. **Solution**: `_correct_trades_with_h1()` uses timestamp-based lookup:
   - Flat sorted H1 list per symbol
   - Filter: `entry_dt < time <= exit_dt`
   - D1 period: 22:00 UTC to 22:00 UTC (not calendar date)
4. **Date fix**: Use `end=2026-01-01` to include 2025 H1 data

### Architecture Parity (Jan 3, 2026)
**TPE Optimizer & Live Bot use IDENTICAL setup finding:**
| Component | TPE Optimizer | Live Bot |
|-----------|---------------|----------|
| Setup logic | `compute_confluence()` | `compute_confluence()` ✅ |
| Trend inference | `_infer_trend()` | `_infer_trend()` ✅ |
| Direction | `_pick_direction_from_bias()` | `_pick_direction_from_bias()` ✅ |
| Timeframes | MN1, W1, D1, H4 | MN1, W1, D1, H4 ✅ |
| Entry | Candle close (simulation) | Pending order + spread check |

### Previous Bug Fixes (Dec 28, 2025)
**IMPORTANT**: The following bugs were recently fixed - avoid reintroducing:

1. **ComplianceTracker**: Implemented compliance tracking class with daily DD (4.5%), total DD (9%), streak halt (999)
   - Metrics-only mode for backtesting (no trade filtering)
   - Returns (trades, compliance_report) tuple from run_full_period_backtest
   - Hard constraints: TP ordering (tp1<tp2<tp3), close-sum ≤85%, ADX threshold ordering
2. **Parameter expansion**: Expanded search space from 17→25+ parameters:
   - TP scaling: tp1/2/3_r_multiple (1.0-6.0R) and tp1/2/3_close_pct (0.15-0.40)
   - Filter toggles: 6 new filters (HTF, structure, Fibonacci, confirmation, displacement, candle rejection)
   - All filter toggles HARD-CODED to False during optimization (baseline establishment)
3. **0-trade bug fix**: Initial implementation filtered all trades due to:
   - Aggressive filter toggles set to True
   - Compliance penalty rejecting trials with DD breaches
   - Streak halt (7) filtering 889/897 trades
   - FIX: Filters disabled, compliance penalty removed, streak halt set to 999
4. **params_loader.py**: Removed `liquidity_sweep_lookback` parameter (doesn't exist in StrategyParams)
5. **professional_quant_suite.py**:
   - Win rate: Remove duplicate `* 100` (already percentage)
   - Calmar ratio: Use `max_drawdown_pct` not `max_drawdown` (USD)
   - Total return: Return USD value, not percentage
6. **ftmo_challenge_analyzer.py**:
   - Quarterly stats must be calculated BEFORE early return for losing trials
   - Use `overall_stats['r_total']` not `user_attrs.get('total_r')` for logging
   - ADX filter disabled: `require_adx_filter=False` everywhere

### Symbol Format
- **Config/data files**: OANDA format with underscores (`EUR_USD`, `XAU_USD`)
- **MT5 execution**: FTMO format (`EURUSD`, `XAUUSD`, `US500.cash`)
- Always use `symbol_mapping.py` for conversions

### Parameters - NEVER Hardcode
```python
# ✅ CORRECT: Load from params loader
from params.params_loader import load_strategy_params
params = load_strategy_params()

# ❌ WRONG: Hardcoding in source files
MIN_CONFLUENCE = 5  # Don't do this
```

### Pip Values - Symbol-Specific
Different instruments have different pip sizes. Always use `get_contract_specs()`:
- Standard forex: `0.0001` (4 decimal)
- JPY pairs: `0.01` (2 decimal)
- Gold (XAUUSD): `0.01`
- Crypto (BTCUSD): `1.0`

### Multi-Timeframe Data
Prevent look-ahead bias by slicing HTF data to reference timestamp:
```python
# strategy_core.py pattern - always use _slice_htf_by_timestamp()
htf_candles = _slice_htf_by_timestamp(weekly_candles, current_daily_dt)
```

## Development Commands

### Run Optimization (resumable)

**Recommended: Use helper script for background runs**
```bash
./run_optimization.sh --single --trials 100  # TPE (logs to ftmo_analysis_output/TPE/run.log)
./run_optimization.sh --multi --trials 100   # NSGA-II (logs to ftmo_analysis_output/NSGA/run.log)
tail -f ftmo_analysis_output/TPE/run.log     # Monitor complete output
```

**Direct Python execution**
```bash
python ftmo_challenge_analyzer.py             # Run/resume optimization
python ftmo_challenge_analyzer.py --status    # Check progress
python ftmo_challenge_analyzer.py --config    # Show current configuration
python ftmo_challenge_analyzer.py --trials 100  # Set trial count
python ftmo_challenge_analyzer.py --multi     # Use NSGA-II multi-objective
python ftmo_challenge_analyzer.py --single    # Use TPE single-objective
python ftmo_challenge_analyzer.py --adx       # Enable ADX regime filtering
```
Uses Optuna with SQLite storage (`ftmo_optimization.db`) for crash-resistant optimization.
Configuration loaded from `params/optimization_config.json`.

**Output Structure:**
- NSGA-II runs: `ftmo_analysis_output/NSGA/` (run.log + optimization.log + CSVs)
- TPE runs: `ftmo_analysis_output/TPE/` (run.log + optimization.log + CSVs)
- `run.log`: Complete console output (all debug info, asset processing)
- `optimization.log`: Trial results only (clean, structured)
- Each mode has its own optimization.log and CSV files

### Run Live Bot (Windows VM only)
```bash
# Requires .env with MT5_SERVER, MT5_LOGIN, MT5_PASSWORD
python main_live_bot.py
```

### Background Optimization
```bash
# Recommended: Use helper script
./run_optimization.sh --single --trials 100  # Auto-logs to ftmo_analysis_output/TPE/run.log

# Manual nohup
nohup python ftmo_challenge_analyzer.py > ftmo_analysis_output/TPE/run.log 2>&1 &
tail -f ftmo_analysis_output/TPE/optimization.log  # Monitor TPE progress
tail -f ftmo_analysis_output/NSGA/optimization.log # Monitor NSGA-II progress
```

## 5ers Challenge Rules (hardcoded limits)
- Max daily loss: **5%** (halt at 4.2%)
- Max total drawdown: **10%** (emergency at 7%)
- Step 1 target: **8%**, Step 2: **5%**
- Min profitable days: **3**
- Risk per trade: 0.6% = $360 per R (on 60K account)

## File Locations
- Historical data: `data/ohlcv/{SYMBOL}_{TF}_2003_2025.csv`
- Optimized params: `params/current_params.json`
- Backtest output: `ftmo_analysis_output/`
- Logs: `logs/tradr_live.log`
- Documentation: `docs/` (system guide, strategy analysis, compliance tracking)
- Utility scripts: `scripts/` (optimization monitoring, debug tools)
- H1 correction: `scripts/apply_h1_correction.py` (post-process trades for accurate SL/TP classification)
- Compliance: `docs/COMPLIANCE_TRACKING_IMPLEMENTATION.md` (FTMOComplianceTracker guide)

## Testing Strategy Changes
1. Modify `strategy_core.py` (contains `compute_confluence()`, `simulate_trades()`)
2. Run optimizer: `python ftmo_challenge_analyzer.py --trials 50`
3. Check `ftmo_analysis_output/` for trade CSVs and performance metrics
4. Verify OOS (out-of-sample) performance matches training period

## Common Patterns

### Adding a New Indicator Filter
```python
# In strategy_core.py StrategyParams dataclass
use_my_filter: bool = False
my_threshold: float = 0.5

# In compute_confluence() function
if params.use_my_filter and my_indicator < params.my_threshold:
    return Signal(...)  # Skip or adjust
```

### Adding to Optimization
```python
# In ftmo_challenge_analyzer.py objective function
my_param = trial.suggest_float("my_param", 0.1, 2.0)
```
