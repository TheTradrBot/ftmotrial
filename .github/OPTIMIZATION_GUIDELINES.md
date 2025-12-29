# Optimization Guidelines - Critical Rules

## NEVER Add These to progress_callback()

The `progress_callback()` function in `OptunaOptimizer.run_optimization()` runs **during** optimization after each trial.

### ❌ FORBIDDEN in progress_callback:
1. **Validation backtests** - `run_full_period_backtest(VALIDATION_START, VALIDATION_END, ...)`
2. **Final backtests** - `run_full_period_backtest(FULL_PERIOD_START, FULL_PERIOD_END, ...)`
3. **CSV exports** - `export_trades_to_csv()` or `output_mgr.save_best_trial_trades()`
4. **Params saving** - `save_optimized_params()` or `save_best_params_persistent()`

### ✅ ALLOWED in progress_callback:
1. **Trial logging** - `output_mgr.log_trial()`
2. **Console output** - `print()` statements
3. **Score tracking** - Update `best_value_before_run`
4. **User attributes** - Access `trial.user_attrs`

## Correct Flow

### During Optimization (progress_callback):
```python
def progress_callback(study, trial):
    # ✅ Log trial
    output_mgr.log_trial(trial_number=trial.number, score=trial.value, ...)
    
    # ✅ Print summary
    print(f"Trial #{trial.number}: Score={trial.value}")
    
    # ❌ NO VALIDATION RUNS!
    # ❌ NO CSV EXPORTS!
```

### After Optimization (main):
```python
# After study.optimize() completes:

# 1. Top 5 validation
top_5_results = validate_top_trials(study, top_n=5)

# 2. Best OOS selection
best_oos = top_5_results[0]

# 3. Final full period backtest
full_trades = run_full_period_backtest(FULL_PERIOD_START, FULL_PERIOD_END, ...)

# 4. CSV exports
export_trades_to_csv(training_trades, "all_trades_jan_dec_2024.csv")
export_trades_to_csv(validation_trades, "all_trades_2024_full.csv")
export_trades_to_csv(full_trades, "all_trades_2023_2025_full.csv")

# 5. Save params
save_optimized_params(best_params)
```

## Why This Matters

**Problem:** Running validation/final backtests in progress_callback:
- Multiplies runtime by 3x (training + validation + final per trial)
- 5 trials = 5×3 = 15 backtests instead of 5 training + 5 validation
- Wastes compute on non-competitive trials

**Solution:** Defer validation to end:
- 5 trials = 5 training backtests (fast)
- Then 5 validation backtests for top 5 only
- Then 1 final backtest for best OOS trial
- Total: 11 backtests instead of 15

## Multi-Objective (NSGA-II)

Same rules apply - no early validation in the optimization loop.

## Last Modified
2025-12-29 - Added after accidental re-introduction of early CSV exports
