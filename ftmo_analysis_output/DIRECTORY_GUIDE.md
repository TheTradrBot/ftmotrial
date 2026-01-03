# FTMO Analysis Output Directory Structure

This directory contains all optimization and validation results organized by mode.

## 🏗️ Directory Overview

```
ftmo_analysis_output/
├── TPE/                    # TPE (single-objective) optimization results
│   ├── history/
│   ├── best_params.json    ← Source for production deployment
│   ├── best_trades_*.csv
│   ├── professional_backtest_report.txt
│   └── optimization.log
│
├── NSGA/                   # NSGA-II (multi-objective) optimization results
│   ├── history/
│   ├── best_params.json
│   └── optimization.log
│
├── VALIDATE/               # TPE parameter validation on different periods
│   └── history/
│
└── VALIDATE_NSGA/          # NSGA-II parameter validation
    └── history/
```

## 🔐 Production Deployment

After optimization, promote best params to production:

```bash
# Promote TPE results to production
python -m params.promote_to_production --source TPE

# Or promote NSGA results
python -m params.promote_to_production --source NSGA
```

The `best_params.json` in each directory is the **source** for production deployment.
The actual production parameters are stored in `params/PRODUCTION_PARAMS.json`.

## Usage by Mode

### 1. TPE Optimization (Default)
**Command:**
```bash
python ftmo_challenge_analyzer.py --trials 50
# or
python ftmo_challenge_analyzer.py --single --trials 50
```

**Output:** `TPE/` directory
- Fast single-objective optimization
- Maximizes composite score (R + Sharpe + WinRate bonuses)
- Recommended for quick parameter tuning

### 2. NSGA-II Multi-Objective Optimization
**Command:**
```bash
python ftmo_challenge_analyzer.py --multi --trials 100
```

**Output:** `NSGA/` directory
- Slow multi-objective optimization
- Finds Pareto frontier (trade-offs between R, Sharpe, WinRate)
- Recommended for balanced strategy development

### 3. TPE Parameter Validation
**Command:**
```bash
python ftmo_challenge_analyzer.py --validate --start 2020-01-01 --end 2022-12-31
```

**Output:** `VALIDATE/` directory
- Tests TPE-optimized parameters on different date ranges
- No re-optimization
- Verifies generalization to other periods

### 4. NSGA-II Parameter Validation
**Command:**
```bash
python ftmo_challenge_analyzer.py --validate --multi --start 2020-01-01 --end 2022-12-31
```

**Output:** `VALIDATE_NSGA/` directory
- Tests NSGA-II-optimized parameters on different date ranges
- No re-optimization
- Verifies Pareto solution robustness

## File Structure in Each Directory

### Optimization Directories (TPE, NSGA)
```
MODE/
├── history/
│   └── run_XXX/                      # Archived optimization runs
│       ├── best_trades_training.csv
│       ├── best_trades_validation.csv
│       ├── best_trades_final.csv
│       ├── monthly_stats.csv
│       ├── symbol_performance.csv
│       ├── best_params.json
│       ├── analysis_summary_*.txt
│       ├── professional_backtest_report.txt
│       └── optimization.log
│
├── best_trades_training.csv          # Latest run (VALIDATE mode only)
├── best_trades_validation.csv
├── best_trades_final.csv
├── monthly_stats.csv
├── symbol_performance.csv
├── optimization.log                  # Trial-by-trial log
└── run.log                           # Complete debug output
```

### Validation Directories (VALIDATE, VALIDATE_NSGA)
```
VALIDATE_MODE/
├── history/
│   └── val_YYYY_YYYY_XXX/            # Archived validation runs
│       ├── best_trades_training.csv  # 70% of period
│       ├── best_trades_validation.csv # 30% of period
│       ├── best_trades_final.csv     # Full period
│       ├── monthly_stats.csv
│       ├── symbol_performance.csv
│       ├── best_params.json
│       ├── analysis_summary_*.txt
│       └── professional_backtest_report.txt
│
├── best_trades_training.csv          # Latest validation
├── best_trades_validation.csv
├── best_trades_final.csv
├── monthly_stats.csv
└── symbol_performance.csv
```

## Choosing the Right Mode

| Goal | Mode | Directory | Command |
|------|------|-----------|---------|
| Fast optimization | TPE | `TPE/` | `--single` or default |
| Balanced strategy | NSGA-II | `NSGA/` | `--multi` |
| Test TPE params | Validation | `VALIDATE/` | `--validate` |
| Test NSGA params | Validation NSGA | `VALIDATE_NSGA/` | `--validate --multi` |

## Workflow Examples

### Standard Workflow (TPE)
```bash
# 1. Optimize with TPE
python ftmo_challenge_analyzer.py --trials 50

# 2. Validate on different periods
python ftmo_challenge_analyzer.py --validate --start 2015-01-01 --end 2017-12-31
python ftmo_challenge_analyzer.py --validate --start 2018-01-01 --end 2020-12-31
python ftmo_challenge_analyzer.py --validate --start 2021-01-01 --end 2023-12-31

# Results in:
# - TPE/history/run_001/
# - VALIDATE/history/val_2015_2017_001/
# - VALIDATE/history/val_2018_2020_001/
# - VALIDATE/history/val_2021_2023_001/
```

### Multi-Objective Workflow (NSGA-II)
```bash
# 1. Optimize with NSGA-II
python ftmo_challenge_analyzer.py --multi --trials 100

# 2. Validate Pareto solutions
python ftmo_challenge_analyzer.py --validate --multi --start 2015-01-01 --end 2017-12-31
python ftmo_challenge_analyzer.py --validate --multi --start 2018-01-01 --end 2020-12-31

# Results in:
# - NSGA/history/run_001/
# - VALIDATE_NSGA/history/val_2015_2017_001/
# - VALIDATE_NSGA/history/val_2018_2020_001/
```

### Comparison Workflow
```bash
# 1. Run both optimizations
python ftmo_challenge_analyzer.py --single --trials 50  # TPE
python ftmo_challenge_analyzer.py --multi --trials 100  # NSGA-II

# 2. Validate both on same periods
python ftmo_challenge_analyzer.py --validate --start 2020-01-01 --end 2022-12-31
python ftmo_challenge_analyzer.py --validate --multi --start 2020-01-01 --end 2022-12-31

# 3. Compare results
# TPE:  VALIDATE/history/val_2020_2022_001/
# NSGA: VALIDATE_NSGA/history/val_2020_2022_001/
```

## Tips

- **TPE is faster**: Use for initial exploration (20-50 trials)
- **NSGA is thorough**: Use for final refinement (50-100+ trials)
- **Always validate**: Test on multiple non-overlapping periods
- **Archive runs**: history/ subdirectories preserve all optimization attempts
- **Compare modes**: Run same validation on both TPE and NSGA params

## Configuration

Edit `params/optimization_config.json` to change default settings:
- Database paths (separate for TPE and NSGA)
- Trial count defaults
- Optimization mode toggles
