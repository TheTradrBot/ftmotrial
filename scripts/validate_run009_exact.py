#!/usr/bin/env python3
"""
Exact Run_009 Validation with Complete Debug Output
====================================================

Replicates TPE run_009 EXACTLY:
1. Same parameters from best_params.json
2. Same period (2023-01-01 to 2025-12-26)
3. Same output format as optimizer
4. Debug output showing why trade counts differ

Expected: ~2008 trades (matching original run_009)
"""

import sys
import json
from datetime import datetime
from pathlib import Path
import pandas as pd

# Import optimizer modules
from ftmo_challenge_analyzer import (
    run_full_period_backtest,
    get_all_trading_assets,
    DEFAULT_EXCLUDED_ASSETS,
    TIMEFRAME_CONFIG,
)
from tradr.utils.output_manager import OutputManager

# Output directory
OUTPUT_DIR = Path("ftmo_analysis_output/TPE/history/run009validation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("RUN_009 EXACT VALIDATION - COMPLETE DEBUG")
print("=" * 80)

# Load run_009 parameters
params_file = Path("ftmo_analysis_output/TPE/history/run_009/best_params.json")
with open(params_file) as f:
    run009_data = json.load(f)

params_dict = run009_data['parameters']

print(f"\nLoaded parameters from: {params_file}")
print(f"Original run_009 results:")
print(f"  Period: {run009_data.get('period', 'Unknown')}")
print(f"  Total trades: {run009_data.get('results', {}).get('total_trades', 'Unknown')}")

print(f"\nKey parameters:")
print(f"  risk_per_trade_pct: {params_dict['risk_per_trade_pct']}")
print(f"  min_confluence_score: {params_dict['min_confluence_score']}")
print(f"  ADX trend/range: {params_dict['adx_trend_threshold']} / {params_dict['adx_range_threshold']}")
print(f"  TP levels: {params_dict['tp1_r_multiple']}R / {params_dict['tp2_r_multiple']}R / {params_dict['tp3_r_multiple']}R")

# Check assets
all_assets = get_all_trading_assets()
excluded = DEFAULT_EXCLUDED_ASSETS
trading_assets = [a for a in all_assets if a not in excluded]

print(f"\nAssets:")
print(f"  Total available: {len(all_assets)}")
print(f"  Excluded: {len(excluded)}")
print(f"  Trading: {len(trading_assets)}")

# Period
start_date = datetime(2023, 1, 1)
end_date = datetime(2025, 12, 26)
print(f"\nBacktest period:")
print(f"  Start: {start_date.date()}")
print(f"  End: {end_date.date()}")

# Timeframe config for TPE
tf_config = TIMEFRAME_CONFIG['TPE']
print(f"\nTimeframe configuration:")
print(f"  Entry TF: {tf_config['entry_tf']}")
print(f"  Confirmation TF: {tf_config['confirmation_tf']}")
print(f"  Bias TF: {tf_config['bias_tf']}")
print(f"  SR TF: {tf_config['sr_tf']}")

print(f"\n{'=' * 80}")
print("STARTING BACKTEST")
print("=" * 80)

# CRITICAL BUGFIX: Run_009 had ALL filters disabled
# Now explicitly passing all disabled filter params
print(f"\n✅ Using corrected parameters with ALL filters explicitly disabled")

# Run backtest with EXACT same parameters as run_009
trades, dd_stats = run_full_period_backtest(
    start_date=start_date,
    end_date=end_date,
    tf_config=tf_config,
    excluded_assets=excluded,
    # Core parameters
    risk_per_trade_pct=params_dict['risk_per_trade_pct'],
    min_confluence=params_dict.get('min_confluence_score', params_dict.get('min_confluence', 2)),
    min_quality_factors=params_dict['min_quality_factors'],
    atr_trail_multiplier=params_dict['atr_trail_multiplier'],
    atr_min_percentile=params_dict['atr_min_percentile'],
    trail_activation_r=params_dict['trail_activation_r'],
    december_atr_multiplier=params_dict['december_atr_multiplier'],
    volatile_asset_boost=params_dict['volatile_asset_boost'],
    # ADX regime parameters
    use_adx_regime_filter=False,  # CRITICAL: run_009 had this DISABLED!
    adx_trend_threshold=params_dict['adx_trend_threshold'],
    adx_range_threshold=params_dict['adx_range_threshold'],
    trend_min_confluence=params_dict['trend_min_confluence'],
    range_min_confluence=params_dict['range_min_confluence'],
    use_adx_slope_rising=False,  # Disabled in run_009
    atr_vol_ratio_range=params_dict.get('atr_vol_ratio_range', 0.9),
    # TP parameters
    tp1_r_multiple=params_dict['tp1_r_multiple'],
    tp2_r_multiple=params_dict['tp2_r_multiple'],
    tp3_r_multiple=params_dict['tp3_r_multiple'],
    tp1_close_pct=params_dict['tp1_close_pct'],
    tp2_close_pct=params_dict['tp2_close_pct'],
    tp3_close_pct=params_dict['tp3_close_pct'],
    # Filter toggles (run_009 had all False)
    use_htf_filter=params_dict.get('use_htf_filter', False),
    use_structure_filter=params_dict.get('use_structure_filter', False),
    use_confirmation_filter=params_dict.get('use_confirmation_filter', False),
    use_fib_filter=params_dict.get('use_fib_filter', False),
    use_displacement_filter=params_dict.get('use_displacement_filter', False),
    use_candle_rejection=params_dict.get('use_candle_rejection', False),
    # Session/Risk filters (CRITICAL: run_009 had these DISABLED!)
    use_session_filter=params_dict.get('use_session_filter', False),
    use_graduated_risk=params_dict.get('use_graduated_risk', False),
)

print(f"\n{'=' * 80}")
print("BACKTEST COMPLETE")
print("=" * 80)

print(f"\nResults:")
print(f"  Total trades: {len(trades)}")
print(f"  Expected (run_009): 2008")
print(f"  Difference: {len(trades) - 2008:+d} ({(len(trades)/2008 - 1)*100:+.1f}%)")

if len(trades) == 0:
    print("\n❌ ERROR: No trades generated!")
    print("This suggests a critical issue with parameter passing or data loading")
    sys.exit(1)

# Analyze trade details
wins = sum(1 for t in trades if t.is_winner)
win_rate = (wins / len(trades)) * 100
h1_corrected = sum(1 for t in trades if "H1 corrected" in str(t.exit_reason))

print(f"\nTrade statistics:")
print(f"  Wins: {wins}")
print(f"  Losses: {len(trades) - wins}")
print(f"  Win rate: {win_rate:.1f}%")
print(f"  H1 corrections: {h1_corrected}")

# Symbol breakdown
symbol_counts = {}
for t in trades:
    symbol_counts[t.symbol] = symbol_counts.get(t.symbol, 0) + 1

print(f"\nSymbol breakdown ({len(symbol_counts)} symbols):")
for symbol in sorted(symbol_counts.keys()):
    print(f"  {symbol}: {symbol_counts[symbol]} trades")

# Compare with run_009 symbols
run009_csv = Path("ftmo_analysis_output/TPE/history/run_009/best_trades_final.csv")
run009_df = pd.read_csv(run009_csv)
run009_symbols = set(run009_df['symbol'].unique())
current_symbols = set(symbol_counts.keys())

missing_symbols = run009_symbols - current_symbols
extra_symbols = current_symbols - run009_symbols

if missing_symbols:
    print(f"\n❌ Missing symbols (in run_009 but not in validation):")
    for s in sorted(missing_symbols):
        count = len(run009_df[run009_df['symbol'] == s])
        print(f"  {s}: {count} trades in run_009")

if extra_symbols:
    print(f"\n⚠️  Extra symbols (in validation but not in run_009):")
    for s in sorted(extra_symbols):
        print(f"  {s}: {symbol_counts[s]} trades")

# Save output files using OutputManager format
print(f"\n{'=' * 80}")
print("SAVING OUTPUT FILES")
print("=" * 80)

# Create best_trades_final.csv
trades_data = []
for i, t in enumerate(trades, 1):
    # Calculate profit USD
    risk_usd = 60000.0 * (params_dict['risk_per_trade_pct'] / 100)
    profit_usd = t.rr * risk_usd
    
    trades_data.append({
        'trade_id': i,
        'symbol': t.symbol,
        'direction': t.direction,
        'entry_date': str(t.entry_date),
        'exit_date': str(t.exit_date),
        'entry_price': t.entry_price,
        'exit_price': t.exit_price,
        'stop_loss': t.stop_loss,
        'take_profit': t.tp1,
        'result_r': t.rr,
        'profit_usd': profit_usd,
        'win': 1 if t.is_winner else 0,
        'confluence_score': t.confluence_score,
        'exit_reason': t.exit_reason or "",
    })

trades_df = pd.DataFrame(trades_data)
csv_file = OUTPUT_DIR / "best_trades_final.csv"
trades_df.to_csv(csv_file, index=False)
print(f"✓ Saved: {csv_file} ({len(trades_df)} trades)")

# Save validation report
report_lines = []
report_lines.append("=" * 80)
report_lines.append("RUN_009 VALIDATION REPORT")
report_lines.append("=" * 80)
report_lines.append(f"\nGenerated: {datetime.now().isoformat()}")
report_lines.append(f"Source: {params_file}")
report_lines.append(f"\nPeriod: {start_date.date()} to {end_date.date()}")
report_lines.append(f"\n{'=' * 80}")
report_lines.append("COMPARISON WITH ORIGINAL RUN_009")
report_lines.append("=" * 80)
report_lines.append(f"\nOriginal run_009 trades: 2008")
report_lines.append(f"Validation trades:       {len(trades)}")
report_lines.append(f"Difference:              {len(trades) - 2008:+d} ({(len(trades)/2008 - 1)*100:+.1f}%)")

if abs(len(trades) - 2008) > 50:
    report_lines.append(f"\n❌ WARNING: Trade count differs significantly!")
    report_lines.append(f"   This suggests parameters or configuration mismatch")
else:
    report_lines.append(f"\n✅ Trade count matches (within 2.5% tolerance)")

report_lines.append(f"\n{'=' * 80}")
report_lines.append("TRADE STATISTICS")
report_lines.append("=" * 80)
report_lines.append(f"Total trades:     {len(trades)}")
report_lines.append(f"Wins:             {wins}")
report_lines.append(f"Losses:           {len(trades) - wins}")
report_lines.append(f"Win rate:         {win_rate:.1f}%")
report_lines.append(f"H1 corrections:   {h1_corrected}")

report_lines.append(f"\n{'=' * 80}")
report_lines.append("SYMBOL BREAKDOWN")
report_lines.append("=" * 80)
report_lines.append(f"Total symbols:    {len(symbol_counts)}")
report_lines.append(f"Run_009 symbols:  {len(run009_symbols)}")

if missing_symbols:
    report_lines.append(f"\nMissing symbols: {len(missing_symbols)}")
    for s in sorted(missing_symbols):
        count = len(run009_df[run009_df['symbol'] == s])
        report_lines.append(f"  {s}: {count} trades in run_009")

if extra_symbols:
    report_lines.append(f"\nExtra symbols: {len(extra_symbols)}")
    for s in sorted(extra_symbols):
        report_lines.append(f"  {s}: {symbol_counts[s]} trades")

report_lines.append(f"\n{'=' * 80}")
report_lines.append("FILES GENERATED")
report_lines.append("=" * 80)
report_lines.append(f"  - best_trades_final.csv")
report_lines.append(f"  - validation_report.txt")
report_lines.append(f"  - validation_params.json")

report_text = "\n".join(report_lines)

report_file = OUTPUT_DIR / "validation_report.txt"
with open(report_file, 'w') as f:
    f.write(report_text)
print(f"✓ Saved: {report_file}")

# Save parameters
params_output = OUTPUT_DIR / "validation_params.json"
with open(params_output, 'w') as f:
    json.dump({
        'source': 'run_009',
        'validation_date': datetime.now().isoformat(),
        'period': {
            'start': str(start_date.date()),
            'end': str(end_date.date()),
        },
        'parameters': params_dict,
        'results': {
            'total_trades': len(trades),
            'wins': wins,
            'losses': len(trades) - wins,
            'win_rate_pct': round(win_rate, 1),
            'h1_corrections': h1_corrected,
            'expected_trades': 2008,
            'difference': len(trades) - 2008,
        }
    }, f, indent=2)
print(f"✓ Saved: {params_output}")

print(f"\n{'=' * 80}")
print("✅ VALIDATION COMPLETE")
print("=" * 80)
print(f"\n📁 Output: {OUTPUT_DIR}")
print("\n" + report_text)
