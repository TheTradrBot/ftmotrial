# Audit Procedures

This document describes the audit procedures for the 5ers Trading Bot to ensure production readiness and parameter traceability.

## Quick Audit

```bash
# Run full production audit
python scripts/audit_production.py

# Verbose output (recommended for reviews)
python scripts/audit_production.py --verbose
```

## Audit Checks

### 1. Production Parameters

**What it checks:**
- `params/PRODUCTION_PARAMS.json` exists
- `PRODUCTION_LOCKED` is `true`
- `validation.approved` is `true`
- All required fields present

**Pass criteria:**
```
✅ PRODUCTION_PARAMS.json exists
✅ Verification passed
✅ Validation metrics present
✅ Approval status: APPROVED
```

### 2. Live Bot Configuration

**What it checks:**
- `main_live_bot.py` imports `load_production_params`
- Calls `verify_production_params()` on startup
- Has `USING_PRODUCTION_PARAMS` flag

**Pass criteria:**
```
✅ Imports load_production_params
✅ Calls verify_production_params
✅ Has USING_PRODUCTION_PARAMS flag
```

### 3. Optimization Runs

**What it shows:**
- Available optimization runs (TPE, NSGA, etc.)
- Score and timestamp for each
- Which run is currently in production

**Example output:**
```
📊 OPTIMIZATION RUNS
   TPE          Score:       212.10 @ 2026-01-01 01:17:44 ← PRODUCTION
   NSGA         Score:       195.50 @ 2026-01-01 04:53:19
   TPE_H4       (no results)
   NSGA_H4      (no results)
```

### 4. Parameter Consistency

**What it checks:**
- Production params match source file
- Key parameters haven't been manually changed
- No unexplained mismatches

**Parameters checked:**
- `min_confluence` / `min_confluence_score`
- `tp1_r_multiple`, `tp2_r_multiple`, `tp3_r_multiple`
- `risk_per_trade_pct`

**Warning if:**
```
⚠️ Parameter mismatches with source:
   tp1_r_multiple: prod=0.6, source=1.7
```

### 5. Strategy Defaults (verbose only)

**What it shows:**
- Default values in `StrategyParams` dataclass
- Useful for understanding fallback behavior

## Audit Output Interpretation

### Full Pass

```
======================================================================
PRODUCTION READINESS AUDIT
Date: 2026-01-02T18:04:29Z
======================================================================

📁 PRODUCTION PARAMETERS
   ✅ PRODUCTION_PARAMS.json exists
   ✅ Verification: Production params valid
   ✅ Validation metrics: present
   ✅ Approval status: APPROVED

🤖 LIVE BOT CONFIGURATION
   ✅ Imports load_production_params
   ✅ Calls verify_production_params
   ✅ Has USING_PRODUCTION_PARAMS flag

📊 OPTIMIZATION RUNS
   TPE          Score:       212.10 @ 2026-01-01 01:17:44 ← PRODUCTION

🔗 PARAMETER CONSISTENCY
   ✅ Production params match source file

======================================================================
✅ AUDIT PASSED - System is production ready
======================================================================
```

### Common Issues

#### Issue: No Production Params

```
❌ PRODUCTION_PARAMS.json MISSING
```

**Fix:**
```bash
python -m params.promote_to_production
```

#### Issue: Not Approved

```
❌ Approval status: NOT APPROVED
```

**Fix:**
Edit `params/PRODUCTION_PARAMS.json`:
```json
{
  "PRODUCTION_LOCKED": true,
  "validation": {
    "approved": true
  }
}
```

#### Issue: Parameter Mismatch

```
⚠️ Parameter mismatches with source:
   tp1_r_multiple: prod=0.6, source=1.7
```

**Analysis:**
This means the production params were manually edited after promotion. Could be:
1. **Intentional**: Someone adjusted params for production (document reason!)
2. **Error**: Re-promote from source

**Fix (if error):**
```bash
python -m params.promote_to_production --source TPE --force
```

## Pre-Deployment Checklist

Before deploying to production (Windows VM):

- [ ] Run audit: `python scripts/audit_production.py --verbose`
- [ ] All checks pass (✅)
- [ ] Review `professional_backtest_report.txt`
- [ ] Check `monthly_stats.csv` for consistency
- [ ] Verify validation Sharpe > 1.0
- [ ] Confirm win rate > 45%
- [ ] Review any parameter mismatches
- [ ] Commit changes with descriptive message
- [ ] Push to remote

## Post-Deployment Verification

After deploying to Windows VM:

1. **Check startup logs:**
   ```
   ✅ PRODUCTION PARAMS ACTIVE (auditable)
   ```

2. **Verify params loaded:**
   ```
   Min Confluence: 2/7
   Source: TPE run @ 2026-01-01 01:17:44
   ```

3. **Monitor first trades** for expected behavior

## Scheduled Audits

Recommended audit schedule:

| Frequency | Action |
|-----------|--------|
| Before deployment | Full audit |
| Weekly | Verify production params |
| After optimization | Review + promote if better |
| Monthly | Full system review |

## Audit History

Keep a log of audits in your deployment notes:

```
2026-01-02: Audit PASSED. TPE run 212.10 deployed.
2026-01-15: Audit PASSED. No changes.
2026-02-01: New optimization. NSGA 225.30. Promoted.
```

## Automation

For CI/CD integration:

```yaml
# .github/workflows/audit.yml
- name: Audit Production Params
  run: python scripts/audit_production.py
  
- name: Fail on Audit Issues
  if: ${{ failure() }}
  run: echo "Audit failed - review before deploy"
```
