# Parameter Deployment Guide

This document describes the **auditable parameter deployment process** for the 5ers Trading Bot. The system ensures full traceability from optimization to production.

## Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                     PARAMETER LIFECYCLE                               │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   1. OPTIMIZATION                                                    │
│      python ftmo_challenge_analyzer.py --trials 100                  │
│      Output: ftmo_analysis_output/TPE/best_params.json               │
│                                                                      │
│   2. REVIEW                                                          │
│      Check: professional_backtest_report.txt                         │
│      Verify: Sharpe > 1.0, WR > 45%, PF > 1.3, OOS stable           │
│                                                                      │
│   3. PROMOTION                                                       │
│      python -m params.promote_to_production                          │
│      Output: params/PRODUCTION_PARAMS.json                           │
│                                                                      │
│   4. APPROVAL                                                        │
│      Edit PRODUCTION_PARAMS.json:                                    │
│        - Set "PRODUCTION_LOCKED": true                               │
│        - Set "validation.approved": true                             │
│                                                                      │
│   5. DEPLOYMENT                                                      │
│      git commit + push                                               │
│      On Windows VM: git pull && restart bot                          │
│                                                                      │
│   6. AUDIT                                                           │
│      python scripts/audit_production.py --verbose                    │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

## File Structure

### params/PRODUCTION_PARAMS.json (Production - LOCKED)

This is the **single source of truth** for live trading. Contains:

```json
{
  "PRODUCTION_LOCKED": true,
  "production_version": "1.0.20260102",
  "deployed_at": "2026-01-02T12:00:00Z",
  "deployed_by": "promote_to_production()",
  
  "source": {
    "optimization_mode": "TPE",
    "optimization_run": "2026-01-01 01:17:44",
    "best_score": 212.10,
    "source_file": "ftmo_analysis_output/TPE/best_params.json"
  },
  
  "validation": {
    "training_sharpe": 2.92,
    "validation_sharpe": 4.76,
    "win_rate": 49.2,
    "profit_factor": 1.60,
    "training_period": "2023-01-01 to 2024-09-30",
    "validation_period": "2024-10-01 to 2025-12-26",
    "approved": true
  },
  
  "parameters": {
    "min_confluence": 2,
    "tp1_r_multiple": 0.6,
    "tp2_r_multiple": 1.2,
    "tp3_r_multiple": 2.0,
    ...
  },
  
  "checksum": {
    "algorithm": "sha256",
    "params_hash": "5381a982325cd0f1..."
  }
}
```

### params/current_params.json (Optimization Output)

Latest optimization output. **NOT used by live bot** unless production params fail verification.

### ftmo_analysis_output/{MODE}/best_params.json

Raw optimizer output. Used as source for promotion.

## Commands

### 1. Run Optimization

```bash
# TPE (single-objective, recommended)
python ftmo_challenge_analyzer.py --single --trials 100

# NSGA-II (multi-objective)
python ftmo_challenge_analyzer.py --multi --trials 100

# Check status
python ftmo_challenge_analyzer.py --status
```

### 2. Promote to Production

```bash
# Interactive (recommended)
python -m params.promote_to_production

# Specify source
python -m params.promote_to_production --source TPE

# With metrics
python -m params.promote_to_production --source TPE --sharpe-train 2.92 --sharpe-val 4.76

# Verify current production
python -m params.promote_to_production --verify

# Show current production params
python -m params.promote_to_production --show

# List available runs
python -m params.promote_to_production --list
```

### 3. Audit Production

```bash
# Full audit
python scripts/audit_production.py

# Verbose (includes strategy defaults)
python scripts/audit_production.py --verbose
```

## Approval Checklist

Before setting `approved: true`, verify:

- [ ] Training Sharpe Ratio > 1.0
- [ ] Validation Sharpe Ratio > 0.8 (OOS)
- [ ] Win Rate > 45%
- [ ] Profit Factor > 1.3
- [ ] Max Drawdown < 15%
- [ ] OOS degradation < 50%
- [ ] Reviewed trade distribution (monthly_stats.csv)
- [ ] Checked symbol performance (symbol_performance.csv)

## Live Bot Behavior

### On Startup

1. **Verify** production params (`verify_production_params()`)
2. **Load** if valid → `USING_PRODUCTION_PARAMS = True`
3. **Fallback** if invalid → Uses `current_params.json` with warning
4. **Log** audit info (version, hash, source)

### Logging

```
======================================================================
BROKER: 5ers Live
Demo Mode: NO (LIVE)
Account Size: $60,000
✅ PRODUCTION PARAMS ACTIVE (auditable)
   Source: TPE run @ 2026-01-01 01:17:44
   Hash: 5381a982325cd0f1
======================================================================
```

## Parameter Naming Conventions

The optimizer and strategy core use slightly different names. The loader handles this automatically:

| Optimizer Output | StrategyParams | Notes |
|------------------|----------------|-------|
| `min_confluence_score` | `min_confluence` | Auto-mapped |
| `tp1_r_multiple` | `atr_tp1_multiplier` | Auto-mapped |
| `tp2_r_multiple` | `atr_tp2_multiplier` | Auto-mapped |
| `tp3_r_multiple` | `atr_tp3_multiplier` | Auto-mapped |

## Rollback Procedure

If production params cause issues:

1. **Immediate**: Set `PRODUCTION_LOCKED: false` → bot falls back to current_params.json
2. **Revert**: Restore previous PRODUCTION_PARAMS.json from git history
3. **Investigate**: Check audit trail in params/history/

## History & Backup

Every parameter save creates a backup in `params/history/`:

```
params/history/
├── params_20251228_204053.json
├── params_20251229_091042.json
├── params_20260101_011240.json
└── ...
```

## Troubleshooting

### "Production params not found"

```bash
python -m params.promote_to_production
```

### "Production params not approved"

Edit `params/PRODUCTION_PARAMS.json`:
```json
{
  "PRODUCTION_LOCKED": true,
  "validation": {
    "approved": true
  }
}
```

### "Parameter mismatch with source"

This means production params were manually edited and differ from the optimization source. Either:
1. Re-promote from source: `python -m params.promote_to_production --source TPE`
2. Or if intentional, update source_file reference in PRODUCTION_PARAMS.json

### Audit fails

Run verbose audit and fix issues:
```bash
python scripts/audit_production.py --verbose
```

## Security Notes

- Production params should only be modified through `promote_to_production`
- Direct edits to PRODUCTION_PARAMS.json break the audit trail
- Always commit param changes with descriptive messages
- Review audit before each production deployment
