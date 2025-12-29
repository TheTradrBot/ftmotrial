# Validation Results

Test optimized parameters on different date ranges without re-optimization.

## How to Run Validation

```bash
# Basic usage (uses best_params.json)
python ftmo_challenge_analyzer.py --validate --start 2020-01-01 --end 2022-12-31

# With specific parameters file
python ftmo_challenge_analyzer.py --validate --start 2018-01-01 --end 2020-12-31 --params-file ftmo_analysis_output/TPE/history/run_006/best_params.json

# Test different periods
python ftmo_challenge_analyzer.py --validate --start 2015-01-01 --end 2018-12-31
python ftmo_challenge_analyzer.py --validate --start 2019-01-01 --end 2021-12-31
```

## Parameters

| Flag | Description | Example |
|------|-------------|---------|
| `--validate` | Enable validation mode (required) | |
| `--start` | Start date (YYYY-MM-DD) | `2020-01-01` |
| `--end` | End date (YYYY-MM-DD) | `2022-12-31` |
| `--params-file` | Path to parameters JSON | `best_params.json` |

## Output Structure

```
VALIDATE/
├── best_trades_training.csv      # 70% of period
├── best_trades_validation.csv    # 30% of period
├── best_trades_final.csv         # Full period
├── monthly_stats_final.csv
├── symbol_performance.csv
├── best_params.json              # Parameters used
├── professional_backtest_report.txt
└── history/
    ├── val_2020_2022_001/        # First run: 2020-2022
    ├── val_2020_2022_002/        # Second run: same period
    ├── val_2018_2020_001/        # Different period
    └── val_2015_2018_001/        # Another period
```

## Validation Results Summary

| Period | Parameters | Total R | Trades | Win Rate | Profit |
|--------|------------|---------|--------|----------|--------|
| 2020-2022 | run_006 (+701R) | +614.96R | 2,602 | 48.3% | $737,947 |

## Tips

**Compare parameters across periods:**
```bash
# Run same period with different params
python ftmo_challenge_analyzer.py --validate --start 2020-01-01 --end 2022-12-31 --params-file ftmo_analysis_output/TPE/history/run_005/best_params.json
python ftmo_challenge_analyzer.py --validate --start 2020-01-01 --end 2022-12-31 --params-file ftmo_analysis_output/TPE/history/run_006/best_params.json
```

**Test strategy robustness:**
```bash
# Test on multiple non-overlapping periods
python ftmo_challenge_analyzer.py --validate --start 2015-01-01 --end 2017-12-31
python ftmo_challenge_analyzer.py --validate --start 2018-01-01 --end 2020-12-31
python ftmo_challenge_analyzer.py --validate --start 2021-01-01 --end 2023-12-31
```
