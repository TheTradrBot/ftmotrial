#!/usr/bin/env python3
"""Debug script to understand why validation period has 0 trades"""

from datetime import datetime
from pathlib import Path
import json

# Import the analyzer
from ftmo_challenge_analyzer import (
    run_full_period_backtest,
    TRAINING_START, TRAINING_END,
    VALIDATION_START, VALIDATION_END,
    FULL_PERIOD_START, FULL_PERIOD_END,
)

# Load best parameters
params_file = Path("best_params.json")
if not params_file.exists():
    print("❌ best_params.json not found!")
    exit(1)

with open(params_file) as f:
    best_params = json.load(f)

print(f"{'='*80}")
print("DEBUG: Validation Period Trade Generation")
print(f"{'='*80}\n")

print(f"Date Ranges:")
print(f"  Training:    {TRAINING_START.date()} to {TRAINING_END.date()}")
print(f"  Validation:  {VALIDATION_START.date()} to {VALIDATION_END.date()}")
print(f"  Full Period: {FULL_PERIOD_START.date()} to {FULL_PERIOD_END.date()}\n")

print(f"Best Parameters:")
for k, v in sorted(best_params.items()):
    if isinstance(v, float):
        print(f"  {k}: {v:.4f}" if v != int(v) else f"  {k}: {int(v)}")
    else:
        print(f"  {k}: {v}")
print()

# Test 1: Training Period
print(f"{'='*80}")
print("TEST 1: TRAINING PERIOD (2023-01-01 to 2024-09-30)")
print(f"{'='*80}")
training_trades = run_full_period_backtest(
    start_date=TRAINING_START,
    end_date=TRAINING_END,
    min_confluence=best_params.get('min_confluence_score', 3),
    min_quality_factors=best_params.get('min_quality_factors', 2),
    risk_per_trade_pct=best_params.get('risk_per_trade_pct', 0.5),
    atr_min_percentile=best_params.get('atr_min_percentile', 60.0),
    trail_activation_r=best_params.get('trail_activation_r', 2.2),
    december_atr_multiplier=best_params.get('december_atr_multiplier', 1.5),
    volatile_asset_boost=best_params.get('volatile_asset_boost', 1.5),
    adx_trend_threshold=best_params.get('adx_trend_threshold', 25.0),
    adx_range_threshold=best_params.get('adx_range_threshold', 20.0),
    trend_min_confluence=best_params.get('trend_min_confluence', 6),
    range_min_confluence=best_params.get('range_min_confluence', 5),
    atr_trail_multiplier=best_params.get('atr_trail_multiplier', 1.5),
    partial_exit_at_1r=best_params.get('partial_exit_at_1r', True),
    partial_exit_pct=best_params.get('partial_exit_pct', 0.5),
)

print(f"✓ Generated {len(training_trades)} trades\n")
if training_trades:
    print(f"First 5 trades:")
    for i, t in enumerate(training_trades[:5]):
        print(f"  {i+1}. {t.symbol:8} | Entry: {t.entry_date} | R: {t.rr:+.2f}")

# Test 2: Validation Period
print(f"\n{'='*80}")
print("TEST 2: VALIDATION PERIOD (2024-10-01 to 2025-12-26)")
print(f"{'='*80}")
validation_trades = run_full_period_backtest(
    start_date=VALIDATION_START,
    end_date=VALIDATION_END,
    min_confluence=best_params.get('min_confluence_score', 3),
    min_quality_factors=best_params.get('min_quality_factors', 2),
    risk_per_trade_pct=best_params.get('risk_per_trade_pct', 0.5),
    atr_min_percentile=best_params.get('atr_min_percentile', 60.0),
    trail_activation_r=best_params.get('trail_activation_r', 2.2),
    december_atr_multiplier=best_params.get('december_atr_multiplier', 1.5),
    volatile_asset_boost=best_params.get('volatile_asset_boost', 1.5),
    adx_trend_threshold=best_params.get('adx_trend_threshold', 25.0),
    adx_range_threshold=best_params.get('adx_range_threshold', 20.0),
    trend_min_confluence=best_params.get('trend_min_confluence', 6),
    range_min_confluence=best_params.get('range_min_confluence', 5),
    atr_trail_multiplier=best_params.get('atr_trail_multiplier', 1.5),
    partial_exit_at_1r=best_params.get('partial_exit_at_1r', True),
    partial_exit_pct=best_params.get('partial_exit_pct', 0.5),
)

print(f"✓ Generated {len(validation_trades)} trades")
if not validation_trades:
    print(f"\n⚠️  NO TRADES GENERATED IN VALIDATION PERIOD!")
    print(f"\nLikely causes:")
    print(f"  1. Parameters too strict (min_confluence={best_params.get('min_confluence_score')})")
    print(f"  2. Market regime changed in validation period")
    print(f"  3. ADX thresholds filtering out trades")
    print(f"  4. Insufficient data in validation period\n")
else:
    print(f"First 5 trades:")
    for i, t in enumerate(validation_trades[:5]):
        print(f"  {i+1}. {t.symbol:8} | Entry: {t.entry_date} | R: {t.rr:+.2f}")

# Test 3: Full Period
print(f"\n{'='*80}")
print("TEST 3: FULL PERIOD (2023-01-01 to 2025-12-26)")
print(f"{'='*80}")
full_trades = run_full_period_backtest(
    start_date=FULL_PERIOD_START,
    end_date=FULL_PERIOD_END,
    min_confluence=best_params.get('min_confluence_score', 3),
    min_quality_factors=best_params.get('min_quality_factors', 2),
    risk_per_trade_pct=best_params.get('risk_per_trade_pct', 0.5),
    atr_min_percentile=best_params.get('atr_min_percentile', 60.0),
    trail_activation_r=best_params.get('trail_activation_r', 2.2),
    december_atr_multiplier=best_params.get('december_atr_multiplier', 1.5),
    volatile_asset_boost=best_params.get('volatile_asset_boost', 1.5),
    adx_trend_threshold=best_params.get('adx_trend_threshold', 25.0),
    adx_range_threshold=best_params.get('adx_range_threshold', 20.0),
    trend_min_confluence=best_params.get('trend_min_confluence', 6),
    range_min_confluence=best_params.get('range_min_confluence', 5),
    atr_trail_multiplier=best_params.get('atr_trail_multiplier', 1.5),
    partial_exit_at_1r=best_params.get('partial_exit_at_1r', True),
    partial_exit_pct=best_params.get('partial_exit_pct', 0.5),
)

print(f"✓ Generated {len(full_trades)} trades\n")

# Summary
print(f"{'='*80}")
print("SUMMARY")
print(f"{'='*80}")
print(f"Training period:   {len(training_trades):4d} trades")
print(f"Validation period: {len(validation_trades):4d} trades")
print(f"Full period:       {len(full_trades):4d} trades")
print(f"\nTrain/Val Ratio: {len(training_trades)+len(validation_trades)}/{len(full_trades)} = {(len(training_trades)+len(validation_trades))/max(len(full_trades),1)*100:.1f}%")
