# Changelog

**Last Updated**: 2026-01-03
**Auto-generated**: From git commits

---

## v1.2.1 - Explicit Disabled Parameters Fix (Jan 3, 2026)

### Critical Parameter Fixes
- 🐛 **Missing Disabled Parameters**: Fixed parameters not being passed to main_live_bot
  - **Bug**: `use_session_filter`, `use_graduated_risk`, `use_atr_trailing` had `default=True` but were NOT in JSON
  - **Impact**: Live bot would enable filters that run_009 had DISABLED
  - **Fix**: All disabled parameters now EXPLICIT in all param files

### Parameter File Updates
- ✅ **PRODUCTION_PARAMS.json**: Added 6 missing disabled parameters
- ✅ **current_params.json**: Added 6 missing disabled parameters  
- ✅ **validation_params.json**: Added 6 missing disabled parameters
- ✅ **run_009/best_params.json**: Added 6 missing disabled parameters

### Default Value Corrections
- ✅ **StrategyParams**: Changed defaults to match run_009 baseline:
  - `use_session_filter: False` (was True)
  - `use_graduated_risk: False` (was True)
  - `use_atr_trailing: False` (was True)
- ✅ **run_full_period_backtest()**: Same default changes for consistency

### Documentation Updates
- ✅ **copilot-instructions.md**: Added explicit list of disabled parameters
- ✅ **CHANGELOG.md**: Documented this fix

---

## v1.2.0 - H1 Exit Correction & Architecture Parity (Jan 3, 2026)

### Critical Bug Fixes
- 🐛 **H1 Exit Detection**: Fixed timestamp-based H1 lookup for accurate SL/TP classification
  - **Bug**: D1 backtests couldn't determine if SL or TP hit first on same candle
  - **Impact**: 467 trades in run_009 were incorrectly classified as LOSS
  - **Fix**: `_correct_trades_with_h1()` now uses flat sorted H1 list with `entry_dt < time <= exit_dt` filter
  - **Result**: 49.2% → 71.0% win rate improvement for run_009

- 🐛 **Date Range Loading**: Fixed `load_ohlcv_data()` to include 2025 data
  - **Bug**: Using `end=2025-01-01` filtered out all 2025 data
  - **Fix**: Use `end=2026-01-01` to include all available H1 data

### Architecture Verification
- ✅ **TPE Optimizer & Live Bot Parity**: Confirmed both use IDENTICAL setup finding logic:
  - Same `compute_confluence()` function from `strategy_core.py`
  - Same `_infer_trend()` and `_pick_direction_from_bias()` for bias calculation
  - Same weekly/monthly data for trend inference
  - Same H4 data for confirmation
  - Same `StrategyParams` class with run_009 defaults
  
- ✅ **Difference**: Live bot adds entry validation (spread, margin, distance checks)

### Parameter Updates
- ✅ **StrategyParams Defaults**: Now baked in with run_009 optimized values:
  - `tp1_r_multiple=1.7`, `tp2_r_multiple=2.7`, `tp3_r_multiple=6.0`
  - `trail_activation_r=0.65`, `min_confluence=2`, `risk_per_trade_pct=0.65`

### Utility Scripts
- ✨ **apply_h1_correction.py**: Post-process trades CSV to correct SL/TP classification
  - Reads trades from run_009 output
  - Applies H1-based exit detection
  - Outputs corrected CSV with accurate win/loss classification

---

## v1.1.0 - Live Bot Enhancements (Dec 31, 2025)

### Live Bot Features
- ✨ **Daily Close Scanning**: Scan only at 22:05 UTC (after NY close)
  - Ensures complete daily candles, matches backtest exactly
  - No more 4-hour interval scanning with partial candles
  
- ✨ **Spread Monitoring**: Every 10 min for signals awaiting better spread
  - Fresh signals saved to `awaiting_spread.json` if spread too wide
  - When spread improves → Execute with MARKET ORDER immediately
  - Signals expire after 12 hours
  
- ✨ **Spread-Only Entry Filter**: No session filter needed
  - All signals check spread quality only
  - Spread OK → Execute immediately
  - Spread wide → Save for retry every 10 min
  
- ✨ **3-Tier Graduated Risk Management**:
  | Tier | Daily DD | Action |
  |------|----------|--------|
  | 1 | ≥2.0% | Reduce risk: 0.6% → 0.4% |
  | 2 | ≥3.5% | Cancel all pending orders |
  | 3 | ≥4.5% | Emergency close positions |

### Live Bot Sync with TPE Optimizer
- ✅ **Quality Factors**: Now uses `max(1, confluence_score // 3)` (identical to backtest)
- ✅ **Volatile Asset Boost**: Applied via `apply_volatile_asset_boost()` for XAU_USD, NAS100_USD, GBP_JPY, BTC_USD
- ✅ **Active Status Check**: Uses boosted scores with `min_quality_for_active = max(1, min_quality_factors - 1)`

### Multi-Broker Support
- ✨ **Forex.com Demo** ($50K): For testing before 5ers live
- ✨ **5ers Live** ($60K): Production trading
- ✨ **broker_config.py**: New multi-broker configuration
- ✨ **symbol_mapping.py**: Fixed index symbols (SPX500, NAS100, UK100 for Forex.com)

### Bug Fixes
- ✅ **Symbol mapping**: Fixed Forex.com indices (was US500 → now SPX500)
- ✅ **Excluded symbols**: JPY pairs and XAG_USD removed from Forex.com demo (min lot issues)
- ✅ **Session filter**: Only applies to intraday data, not daily backtests

---

## v1.0.0 - Production Release (Dec 31, 2025)

### Highlights
- Finalized 0.6% risk parameters; synced `best_params.json` and `params/current_params.json`.
- 12-year robustness validation (2014-2025): +2,766.3R, $995K, ~48.6% WR.
- FTMO compliance: daily DD <3.8% (limit 5%), total DD <3% (limit 10%).
- 5ers challenge speed: Step 1 (8%) in 18 dagen; Step 2 (5%) in 10 dagen; total 28 dagen.

### Parameters (Dec 31, 2025)
- Risk 0.6% per trade; min_confluence/score 2; min_quality_factors 3.
- ADX trend/range: 22 / 11; trend_min_confluence 6; range_min_confluence 2.
- ATR/trail: trail_activation_r 0.8; atr_trail_multiplier 1.6; atr_min_percentile 42; atr_volatility_ratio 0.95.
- TP ladder: 1.7R / 2.6R / 5.4R with closes 38% / 16% / 30%; partial_exit_at_1r true; partial_exit_pct 0.7.
- DD guards: daily_loss_halt_pct 3.8; max_total_dd_warning 7.9; consecutive_loss_halt 10.
- Filters baseline: HTF/structure/confirmation/fib/displacement/candle_rejection disabled.

---

## Previous Changes (Dec 30, 2025)

### New Features
- **NSGA-II Directory Structure**: Added dedicated output directories for multi-objective optimization
  - `ftmo_analysis_output/NSGA/` - NSGA-II optimization results
  - `ftmo_analysis_output/VALIDATE_NSGA/` - NSGA-II parameter validation
  - Automatic mode detection: `--multi` flag routes to NSGA directories

### Documentation
- Added `ftmo_analysis_output/NSGA/README.md` - NSGA-II optimization guide
- Added `ftmo_analysis_output/VALIDATE_NSGA/README.md` - NSGA-II validation guide  
- Added `ftmo_analysis_output/DIRECTORY_GUIDE.md` - Complete directory structure reference

---

## Previous Changes (Dec 29, 2025)

### New Features
- **Validation Mode**: Test existing parameters on different date ranges without running optimization
  ```bash
  python ftmo_challenge_analyzer.py --validate --start 2020-01-01 --end 2022-12-31 --params-file best_params.json
  ```

### Bug Fixes
- **CRITICAL**: Fixed parameter saving bug - ALL 30+ Optuna parameters now saved correctly
- **CRITICAL**: Fixed date handling in validation mode (datetime vs date objects)
- **Archive improvements**: Added missing files to history archives

---

## Previous Changes (Dec 28, 2025)

### New Features
- **FTMOComplianceTracker**: Compliance tracking with daily DD (4.5%), total DD (9%), streak halt
- **Parameter expansion**: 25+ optimizable parameters (TP scaling, 6 filter toggles, ADX regime)
- **TP scaling**: tp1/2/3_r_multiple (1.0-6.0R) and tp1/2/3_close_pct (0.15-0.40)
- **Filter toggles**: 6 new filters (HTF, structure, Fibonacci, confirmation, displacement, candle rejection)

### Critical Bug Fixes
- **0-trade bug**: Fixed aggressive filters/compliance penalties causing 0 trades
- **params_loader.py**: Removed obsolete `liquidity_sweep_lookback` parameter
- **Metric calculations**: Fixed win_rate (4700%→47%), Calmar ratio, total_return units
- **Optimization logs**: Fixed R=0.0 display bug for losing trials
- **Trade exports**: All 34 symbols now appear in CSV outputs
