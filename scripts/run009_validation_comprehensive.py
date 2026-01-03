#!/usr/bin/env python3
"""
Run_009 Comprehensive Validation with Complete Stats Output
============================================================

Replicates Run_009 EXACTLY with full statistical analysis:
1. Same parameters from best_params.json
2. Same period (2023-01-01 to 2025-12-26)
3. Same output format as optimizer
4. Professional backtest report
5. Trade-by-trade CSV analysis
6. Monthly and quarterly breakdowns

Output files:
  - validation_summary.txt: Summary with key metrics
  - validation_trades.csv: All trades with details
  - validation_report.txt: Professional backtest report
  - validation_monthly_stats.csv: Monthly breakdown
  - validation_quarterly_stats.csv: Quarterly breakdown
"""

import sys
import json
from datetime import datetime, date
from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np

# Import optimizer modules
from ftmo_challenge_analyzer import (
    run_full_period_backtest,
    get_all_trading_assets,
    DEFAULT_EXCLUDED_ASSETS,
    TIMEFRAME_CONFIG,
)
from strategy_core import Trade
from professional_quant_suite import calculate_risk_metrics

# Output directory
OUTPUT_DIR = Path("ftmo_analysis_output/TPE/history/run009validation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ACCOUNT_SIZE = 60000.0

def calculate_quarterly_stats(trades: List[Trade], risk_usd: float) -> Dict:
    """Calculate stats by quarter"""
    quarterly = {}
    
    # Define quarters
    quarters = {
        '2023_Q1': (datetime(2023, 1, 1), datetime(2023, 3, 31)),
        '2023_Q2': (datetime(2023, 4, 1), datetime(2023, 6, 30)),
        '2023_Q3': (datetime(2023, 7, 1), datetime(2023, 9, 30)),
        '2023_Q4': (datetime(2023, 10, 1), datetime(2023, 12, 31)),
        '2024_Q1': (datetime(2024, 1, 1), datetime(2024, 3, 31)),
        '2024_Q2': (datetime(2024, 4, 1), datetime(2024, 6, 30)),
        '2024_Q3': (datetime(2024, 7, 1), datetime(2024, 9, 30)),
        '2024_Q4': (datetime(2024, 10, 1), datetime(2024, 12, 31)),
        '2025_Q1': (datetime(2025, 1, 1), datetime(2025, 3, 31)),
        '2025_Q2': (datetime(2025, 4, 1), datetime(2025, 6, 30)),
        '2025_Q3': (datetime(2025, 7, 1), datetime(2025, 9, 30)),
        '2025_Q4': (datetime(2025, 10, 1), datetime(2025, 12, 31)),
    }
    
    for q_name, (q_start, q_end) in quarters.items():
        q_trades = []
        for t in trades:
            entry = t.entry_date if hasattr(t, 'entry_date') else t.entry_datetime
            if isinstance(entry, str):
                entry = datetime.fromisoformat(entry.replace('Z', '+00:00'))
            if hasattr(entry, 'tzinfo') and entry.tzinfo:
                entry = entry.replace(tzinfo=None)
            
            if q_start <= entry <= q_end:
                q_trades.append(t)
        
        if q_trades:
            q_r = sum(getattr(t, 'rr', 0) for t in q_trades)
            q_wins = sum(1 for t in q_trades if getattr(t, 'rr', 0) > 0)
            q_wr = (q_wins / len(q_trades) * 100) if q_trades else 0
            q_profit = q_r * risk_usd
            
            quarterly[q_name] = {
                'trades': len(q_trades),
                'wins': q_wins,
                'losses': len(q_trades) - q_wins,
                'r_total': round(q_r, 2),
                'profit': round(q_profit, 2),
                'win_rate': round(q_wr, 1),
                'avg_r': round(q_r / len(q_trades), 3) if q_trades else 0,
            }
        else:
            quarterly[q_name] = {
                'trades': 0,
                'wins': 0,
                'losses': 0,
                'r_total': 0.0,
                'profit': 0.0,
                'win_rate': 0.0,
                'avg_r': 0.0,
            }
    
    return quarterly


def calculate_monthly_stats(trades: List[Trade], risk_usd: float) -> Dict:
    """Calculate stats by month"""
    monthly = {}
    
    for t in trades:
        entry = t.entry_date if hasattr(t, 'entry_date') else t.entry_datetime
        if isinstance(entry, str):
            entry = datetime.fromisoformat(entry.replace('Z', '+00:00'))
        if hasattr(entry, 'tzinfo') and entry.tzinfo:
            entry = entry.replace(tzinfo=None)
        
        month_key = f"{entry.year}_{entry.month:02d}"
        
        if month_key not in monthly:
            monthly[month_key] = []
        monthly[month_key].append(t)
    
    month_stats = {}
    for month_key in sorted(monthly.keys()):
        m_trades = monthly[month_key]
        m_r = sum(getattr(t, 'rr', 0) for t in m_trades)
        m_wins = sum(1 for t in m_trades if getattr(t, 'rr', 0) > 0)
        m_wr = (m_wins / len(m_trades) * 100) if m_trades else 0
        m_profit = m_r * risk_usd
        
        month_stats[month_key] = {
            'trades': len(m_trades),
            'wins': m_wins,
            'losses': len(m_trades) - m_wins,
            'r_total': round(m_r, 2),
            'profit': round(m_profit, 2),
            'win_rate': round(m_wr, 1),
            'avg_r': round(m_r / len(m_trades), 3) if m_trades else 0,
        }
    
    return month_stats


def main():
    print("=" * 80)
    print("RUN_009 COMPREHENSIVE VALIDATION")
    print("=" * 80)
    
    # Load run_009 parameters
    params_file = Path("ftmo_analysis_output/TPE/history/run_009/best_params.json")
    with open(params_file) as f:
        run009_data = json.load(f)
    
    params_dict = run009_data['parameters']
    
    print(f"\n✅ Loaded parameters from: {params_file}")
    print(f"\nRun_009 original results:")
    print(f"  Period: {run009_data.get('period', 'Unknown')}")
    print(f"  Total trades: {run009_data.get('results', {}).get('total_trades', 'Unknown')}")
    print(f"  Win rate: {run009_data.get('results', {}).get('win_rate', 'Unknown')}%")
    
    # Setup
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2025, 12, 26)
    risk_pct = params_dict['risk_per_trade_pct']
    risk_usd = ACCOUNT_SIZE * (risk_pct / 100)
    
    training_end = datetime(2024, 9, 30)
    validation_start = datetime(2024, 10, 1)
    
    all_assets = get_all_trading_assets()
    excluded = DEFAULT_EXCLUDED_ASSETS
    trading_assets = [a for a in all_assets if a not in excluded]
    
    tf_config = TIMEFRAME_CONFIG['TPE']
    
    print(f"\nSetup:")
    print(f"  Full period: {start_date.date()} to {end_date.date()}")
    print(f"  Training: {start_date.date()} to {training_end.date()}")
    print(f"  Validation: {validation_start.date()} to {end_date.date()}")
    print(f"  Risk per trade: {risk_pct}% = ${risk_usd:.2f}")
    print(f"  Trading assets: {len(trading_assets)}")
    
    print(f"\n{'=' * 80}")
    print("RUNNING BACKTEST")
    print("=" * 80)
    
    # Run full period backtest
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
        use_adx_regime_filter=False,
        use_adx_slope_rising=False,
        adx_trend_threshold=params_dict['adx_trend_threshold'],
        adx_range_threshold=params_dict['adx_range_threshold'],
        trend_min_confluence=params_dict['trend_min_confluence'],
        range_min_confluence=params_dict['range_min_confluence'],
        atr_vol_ratio_range=params_dict.get('atr_vol_ratio_range', 0.9),
        # TP parameters
        tp1_r_multiple=params_dict['tp1_r_multiple'],
        tp2_r_multiple=params_dict['tp2_r_multiple'],
        tp3_r_multiple=params_dict['tp3_r_multiple'],
        tp1_close_pct=params_dict['tp1_close_pct'],
        tp2_close_pct=params_dict['tp2_close_pct'],
        tp3_close_pct=params_dict['tp3_close_pct'],
        # Filter toggles
        use_htf_filter=params_dict.get('use_htf_filter', False),
        use_structure_filter=params_dict.get('use_structure_filter', False),
        use_confirmation_filter=params_dict.get('use_confirmation_filter', False),
        use_fib_filter=params_dict.get('use_fib_filter', False),
        use_displacement_filter=params_dict.get('use_displacement_filter', False),
        use_candle_rejection=params_dict.get('use_candle_rejection', False),
        # Critical disabled filters
        use_session_filter=params_dict.get('use_session_filter', False),
        use_graduated_risk=params_dict.get('use_graduated_risk', False),
    )
    
    print(f"\n✅ Backtest complete")
    print(f"  Total trades: {len(trades)}")
    
    if len(trades) == 0:
        print("\n❌ ERROR: No trades generated!")
        return
    
    # Split trades by period
    def get_entry_date(t):
        """Extract and normalize entry date from trade object."""
        entry = t.entry_date if hasattr(t, 'entry_date') else t.entry_datetime
        if isinstance(entry, str):
            entry = datetime.fromisoformat(entry.replace('Z', '+00:00'))
        if hasattr(entry, 'tzinfo') and entry.tzinfo:
            entry = entry.replace(tzinfo=None)
        return entry
    
    training_trades = [t for t in trades if get_entry_date(t) <= training_end]
    validation_trades = [t for t in trades if get_entry_date(t) > training_end]
    
    # Calculate stats
    total_r = sum(getattr(t, 'rr', 0) for t in trades)
    total_wins = sum(1 for t in trades if getattr(t, 'rr', 0) > 0)
    total_wr = (total_wins / len(trades) * 100) if trades else 0
    total_profit = total_r * risk_usd
    
    train_r = sum(getattr(t, 'rr', 0) for t in training_trades)
    train_wins = sum(1 for t in training_trades if getattr(t, 'rr', 0) > 0)
    train_wr = (train_wins / len(training_trades) * 100) if training_trades else 0
    train_profit = train_r * risk_usd
    
    val_r = sum(getattr(t, 'rr', 0) for t in validation_trades)
    val_wins = sum(1 for t in validation_trades if getattr(t, 'rr', 0) > 0)
    val_wr = (val_wins / len(validation_trades) * 100) if validation_trades else 0
    val_profit = val_r * risk_usd
    
    # Calculate Sharpe ratios (approximate using risk metrics)
    train_metrics = calculate_risk_metrics(training_trades) if training_trades else None
    val_metrics = calculate_risk_metrics(validation_trades) if validation_trades else None
    total_metrics = calculate_risk_metrics(trades) if trades else None
    
    train_sharpe = train_metrics.sharpe_ratio if train_metrics else 0
    val_sharpe = val_metrics.sharpe_ratio if val_metrics else 0
    total_sharpe = total_metrics.sharpe_ratio if total_metrics else 0
    
    print(f"\n  Training: {len(training_trades)} trades, {train_wr:.1f}% WR, ${train_profit:+.2f}")
    print(f"  Validation: {len(validation_trades)} trades, {val_wr:.1f}% WR, ${val_profit:+.2f}")
    print(f"  Total: {len(trades)} trades, {total_wr:.1f}% WR, ${total_profit:+.2f}")
    
    # Calculate quarterly and monthly stats
    quarterly = calculate_quarterly_stats(trades, risk_usd)
    monthly = calculate_monthly_stats(trades, risk_usd)
    
    # Save summary file
    summary_file = OUTPUT_DIR / "validation_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("RUN_009 VALIDATION - COMPREHENSIVE SUMMARY\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("PERIOD BREAKDOWN\n")
        f.write("-" * 80 + "\n")
        f.write(f"Training Period (2023-01-01 to 2024-09-30)\n")
        f.write(f"  Trades: {len(training_trades)}\n")
        f.write(f"  Total R: {train_r:+.2f}R\n")
        f.write(f"  Estimated Profit: ${train_profit:+.2f}\n")
        f.write(f"  Win Rate: {train_wr:.1f}%\n")
        f.write(f"  Avg R per Trade: {train_r/len(training_trades):+.3f}\n")
        f.write(f"  Sharpe Ratio: {train_sharpe:.2f}\n\n")
        
        f.write(f"Validation Period (2024-10-01 to 2025-12-26)\n")
        f.write(f"  Trades: {len(validation_trades)}\n")
        f.write(f"  Total R: {val_r:+.2f}R\n")
        f.write(f"  Estimated Profit: ${val_profit:+.2f}\n")
        f.write(f"  Win Rate: {val_wr:.1f}%\n")
        f.write(f"  Avg R per Trade: {val_r/len(validation_trades):+.3f}\n")
        f.write(f"  Sharpe Ratio: {val_sharpe:.2f}\n\n")
        
        f.write(f"Full Period (2023-01-01 to 2025-12-26)\n")
        f.write(f"  Trades: {len(trades)}\n")
        f.write(f"  Total R: {total_r:+.2f}R\n")
        f.write(f"  Estimated Profit: ${total_profit:+.2f}\n")
        f.write(f"  Win Rate: {total_wr:.1f}%\n")
        f.write(f"  Avg R per Trade: {total_r/len(trades):+.3f}\n")
        f.write(f"  Sharpe Ratio: {total_sharpe:.2f}\n\n")
        
        f.write("QUARTERLY BREAKDOWN\n")
        f.write("-" * 80 + "\n")
        for q_name in sorted(quarterly.keys()):
            q = quarterly[q_name]
            f.write(f"  {q_name}: {q['trades']} trades, {q['r_total']:+.2f}R, {q['win_rate']:.0f}% WR, ${q['profit']:+.2f}\n")
        
        f.write(f"\nTOTAL PROFIT (Full Period 2023-01-01 to 2025-12-26): ${total_profit:+.2f}\n")
        f.write("=" * 80 + "\n")
    
    print(f"\n✅ Summary saved to: {summary_file}")
    
    # Save trades CSV
    trades_csv = OUTPUT_DIR / "validation_trades.csv"
    trades_data = []
    for i, t in enumerate(trades, 1):
        entry = t.entry_date if hasattr(t, 'entry_date') else t.entry_datetime
        if isinstance(entry, str):
            entry = entry.replace('Z', '+00:00')
        exit_time = t.exit_date if hasattr(t, 'exit_date') else t.exit_datetime
        if isinstance(exit_time, str):
            exit_time = exit_time.replace('Z', '+00:00')
        
        rr = getattr(t, 'rr', 0)
        profit = rr * risk_usd
        
        trades_data.append({
            'trade_id': i,
            'symbol': t.symbol,
            'direction': t.direction,
            'entry_datetime': entry,
            'exit_datetime': exit_time,
            'entry_price': getattr(t, 'entry_price', 0),
            'exit_price': getattr(t, 'exit_price', 0),
            'stop_loss': getattr(t, 'stop_loss', 0),
            'r_multiple': rr,
            'profit_usd': profit,
            'is_winner': rr > 0,
            'exit_reason': getattr(t, 'exit_reason', 'Unknown'),
        })
    
    df_trades = pd.DataFrame(trades_data)
    df_trades.to_csv(trades_csv, index=False)
    print(f"✅ Trades saved to: {trades_csv}")
    
    # Save monthly stats
    monthly_csv = OUTPUT_DIR / "validation_monthly_stats.csv"
    monthly_data = []
    for month_key in sorted(monthly.keys()):
        m = monthly[month_key]
        monthly_data.append({
            'month': month_key,
            'trades': m['trades'],
            'wins': m['wins'],
            'losses': m['losses'],
            'r_total': m['r_total'],
            'profit_usd': m['profit'],
            'win_rate': m['win_rate'],
            'avg_r': m['avg_r'],
        })
    
    df_monthly = pd.DataFrame(monthly_data)
    df_monthly.to_csv(monthly_csv, index=False)
    print(f"✅ Monthly stats saved to: {monthly_csv}")
    
    # Save quarterly stats
    quarterly_csv = OUTPUT_DIR / "validation_quarterly_stats.csv"
    quarterly_data = []
    for q_name in sorted(quarterly.keys()):
        q = quarterly[q_name]
        quarterly_data.append({
            'quarter': q_name,
            'trades': q['trades'],
            'wins': q['wins'],
            'losses': q['losses'],
            'r_total': q['r_total'],
            'profit_usd': q['profit'],
            'win_rate': q['win_rate'],
            'avg_r': q['avg_r'],
        })
    
    df_quarterly = pd.DataFrame(quarterly_data)
    df_quarterly.to_csv(quarterly_csv, index=False)
    print(f"✅ Quarterly stats saved to: {quarterly_csv}")
    
    # Comparison with run_009
    print(f"\n{'=' * 80}")
    print("COMPARISON WITH RUN_009")
    print(f"{'=' * 80}")
    
    run009_expected_trades = 2008
    trade_diff = len(trades) - run009_expected_trades
    trade_pct = (len(trades) / run009_expected_trades - 1) * 100
    
    print(f"\nExpected (run_009): {run009_expected_trades} trades")
    print(f"Actual (validation): {len(trades)} trades")
    print(f"Difference: {trade_diff:+d} ({trade_pct:+.1f}%)")
    
    if abs(trade_pct) < 5:
        print(f"✅ Trade count within acceptable range (±5%)")
    elif abs(trade_pct) < 15:
        print(f"⚠️  Trade count variance (acceptable: data/code changes)")
    else:
        print(f"❌ Trade count variance too high (>15%)")
    
    print(f"\n{'=' * 80}")
    print("✅ VALIDATION COMPLETE")
    print(f"{'=' * 80}")
    print(f"\nOutput files:")
    print(f"  {summary_file}")
    print(f"  {trades_csv}")
    print(f"  {monthly_csv}")
    print(f"  {quarterly_csv}")


if __name__ == '__main__':
    main()
