# 5ers 60K High Stakes Trading Bot

Automated MetaTrader 5 trading bot for **5ers 60K High Stakes** Challenge accounts. Uses a 6-Pillar Confluence system with multi-timeframe analysis. Validated on 12 years (2014-2025) and production-ready.

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    AUDITABLE PARAMETER FLOW                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ftmo_challenge_analyzer.py        ← Optuna Optimization               │
│            │                                                            │
│            ▼                                                            │
│   ftmo_analysis_output/TPE/best_params.json                             │
│            │                                                            │
│            ▼  (Manual Review Required)                                  │
│   python -m params.promote_to_production                                │
│            │                                                            │
│            ▼                                                            │
│   params/PRODUCTION_PARAMS.json     ← LOCKED (full audit trail)         │
│            │                                                            │
│            ▼                                                            │
│   main_live_bot.py                  ← Verifies + loads production       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Run Optimization

```bash
./run_optimization.sh --single --trials 100  # TPE (recommended)
./run_optimization.sh --multi --trials 100   # NSGA-II multi-objective

# Monitor progress
tail -f ftmo_analysis_output/TPE/optimization.log
```

### 2. Promote to Production (After Review)

```bash
# Interactive promotion
python -m params.promote_to_production

# Verify production params
python -m params.promote_to_production --verify
```

### 3. Audit Production Readiness

```bash
python scripts/audit_production.py --verbose
```

### 4. Run Live Bot (Windows VM)

```bash
python main_live_bot.py
```

## 📁 Project Structure

```
ftmotrial/
├── main_live_bot.py              # Live MT5 bot (Windows VM)
├── ftmo_challenge_analyzer.py    # Optuna optimization engine
├── strategy_core.py              # Trading strategy (6 Confluence Pillars)
│
├── params/                       # PARAMETER MANAGEMENT
│   ├── PRODUCTION_PARAMS.json    # 🔒 LOCKED production params (auditable)
│   ├── current_params.json       # Latest optimization output
│   ├── params_loader.py          # Parameter loading logic
│   ├── promote_to_production.py  # CLI to promote params to production
│   └── history/                  # Backup of all param changes
│
├── ftmo_analysis_output/         # OPTIMIZATION RESULTS
│   ├── TPE/                      # TPE results (best_params.json, trades, report)
│   ├── NSGA/                     # NSGA-II multi-objective results
│   └── VALIDATE/                 # Validation on different periods
│
├── scripts/                      # UTILITIES
│   ├── audit_production.py       # 🔍 Production readiness audit
│   └── monitor_optimization.sh   # Monitor running optimization
│
├── docs/                         # DOCUMENTATION
└── data/ohlcv/                   # Historical OHLCV data (2003-2025)
```

## 🔐 Parameter Management

### Production Parameters (params/PRODUCTION_PARAMS.json)

Contains full audit trail:
- **source**: Which optimization run (TPE/NSGA, timestamp, score)
- **validation**: Sharpe ratio, win rate, approval status
- **parameters**: Locked strategy parameters
- **checksum**: SHA256 hash for integrity

### Deployment Workflow

1. **Optimize** → `python ftmo_challenge_analyzer.py`
2. **Review** → Check `professional_backtest_report.txt`
3. **Promote** → `python -m params.promote_to_production`
4. **Approve** → Set `approved: true` in PRODUCTION_PARAMS.json
5. **Audit** → `python scripts/audit_production.py`
6. **Deploy** → Pull on Windows VM, restart bot

## 📊 Latest Results (Jan 2, 2026)

### TPE Optimization (Production)

| Metric | Training | Validation | Full |
|--------|----------|------------|------|
| Sharpe Ratio | 2.92 | 4.76 | 3.53 |
| Total Return | $132,740 | $118,248 | $248,551 |
| Win Rate | 47.4% | 52.8% | 49.2% |
| Max Drawdown | $9,848 | $7,535 | $9,848 |

### 5ers Challenge Projections
- Step 1 (8% = $4,800): ~18 days
- Step 2 (5% = $3,000): ~10 days

## 🛡️ Risk Management

- Max daily loss: **5%** (halt at 4.2%)
- Max total drawdown: **10%** (emergency at 7%)
- Risk per trade: 0.65% = $390 per R
- Graduated risk tiers at 2%, 3.5%, 4.5% DD

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | System architecture |
| [docs/PARAMETER_DEPLOYMENT.md](docs/PARAMETER_DEPLOYMENT.md) | Parameter deployment guide |
| [docs/AUDIT_PROCEDURES.md](docs/AUDIT_PROCEDURES.md) | Audit procedures |
| [docs/STRATEGY_GUIDE.md](docs/STRATEGY_GUIDE.md) | Strategy deep dive |
| [.github/copilot-instructions.md](.github/copilot-instructions.md) | AI context |

---

**Last Updated**: January 2, 2026
