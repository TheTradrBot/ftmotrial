#!/usr/bin/env python3
"""
Apply H1 Correction to Backtest Trades

This script fixes the critical D1 backtest bug where SL/TP order cannot be 
determined when both hit on the same daily candle. The D1 simulation always
checks SL first, causing many trades to be incorrectly classified as LOSS
when TP actually hit first.

Usage:
    python scripts/apply_h1_correction.py [input_csv] [output_csv]
    
    # Default: processes run_009 trades
    python scripts/apply_h1_correction.py
    
    # Custom files
    python scripts/apply_h1_correction.py my_trades.csv my_trades_corrected.csv

The script:
1. Loads trade CSV with SL exits
2. For each SL exit, uses H1 data to check if TP was hit first
3. Corrects WIN/LOSS classification
4. Saves corrected trades to output CSV

Expected improvement: ~22% increase in win rate (49% → 71%)
"""
import sys
import pandas as pd
from pathlib import Path
from datetime import datetime

# Default paths
DEFAULT_INPUT = 'ftmo_analysis_output/TPE/history/run_009/best_trades_final.csv'
DEFAULT_OUTPUT = 'ftmo_analysis_output/TPE/history/run_009/best_trades_h1_corrected.csv'
H1_DATA_DIR = Path('data/ohlcv')


def load_h1_data(symbol: str) -> pd.DataFrame | None:
    """Load H1 data for a symbol."""
    h1_file = H1_DATA_DIR / f"{symbol}_H1_2023_2025.csv"
    if not h1_file.exists():
        return None
    
    df = pd.read_csv(h1_file)
    df.columns = [c.lower() for c in df.columns]
    df['time'] = pd.to_datetime(df['time']).dt.tz_localize(None)
    return df


def check_sl_tp_order(
    h1_data: pd.DataFrame,
    entry_dt: datetime,
    exit_dt: datetime,
    direction: str,
    sl_price: float,
    tp_price: float,
) -> tuple[str | None, str | None]:
    """
    Check which hit first: SL or TP.
    
    Returns:
        (first_hit_time, first_hit_type) where type is 'SL' or 'TP'
    """
    # Get H1 candles AFTER entry and up to exit
    mask = (h1_data['time'] > entry_dt) & (h1_data['time'] <= exit_dt)
    period = h1_data[mask]
    
    if len(period) == 0:
        return None, None
    
    sl_time = None
    tp_time = None
    
    for _, c in period.iterrows():
        if direction == 'bullish':
            if sl_time is None and c['low'] <= sl_price:
                sl_time = c['time']
            if tp_time is None and c['high'] >= tp_price:
                tp_time = c['time']
        else:  # bearish
            if sl_time is None and c['high'] >= sl_price:
                sl_time = c['time']
            if tp_time is None and c['low'] <= tp_price:
                tp_time = c['time']
    
    # Determine which hit first
    if tp_time and (not sl_time or tp_time < sl_time):
        return tp_time, 'TP'
    elif sl_time:
        return sl_time, 'SL'
    else:
        return None, None


def apply_h1_correction(input_csv: str, output_csv: str) -> dict:
    """
    Apply H1 correction to trades CSV.
    
    Returns:
        Statistics dictionary with correction counts
    """
    print(f"Loading trades from: {input_csv}")
    trades = pd.read_csv(input_csv)
    print(f"Loaded {len(trades)} trades")
    
    original_wins = trades['win'].sum()
    original_win_rate = trades['win'].mean() * 100
    print(f"Original: {original_wins} wins ({original_win_rate:.1f}%)")
    
    # Get unique symbols
    symbols = trades['symbol'].unique()
    print(f"\nLoading H1 data for {len(symbols)} symbols...")
    
    # Load H1 data for all symbols
    h1_cache = {}
    for sym in symbols:
        h1 = load_h1_data(sym)
        if h1 is not None:
            h1_cache[sym] = h1
    print(f"Loaded H1 data for {len(h1_cache)}/{len(symbols)} symbols")
    
    # Process trades
    corrected_trades = []
    stats = {
        'loss_to_win': 0,
        'win_to_loss': 0,
        'unchanged': 0,
        'no_h1_data': 0,
    }
    
    for idx, t in trades.iterrows():
        symbol = t['symbol']
        h1 = h1_cache.get(symbol)
        
        if h1 is None:
            stats['no_h1_data'] += 1
            corrected_trades.append(t.to_dict())
            continue
        
        # Parse dates
        entry_dt = pd.to_datetime(t['entry_date']).tz_localize(None)
        exit_dt = pd.to_datetime(t['exit_date']).tz_localize(None)
        
        # Check SL/TP order using H1
        first_hit_time, first_hit_type = check_sl_tp_order(
            h1_data=h1,
            entry_dt=entry_dt,
            exit_dt=exit_dt,
            direction=t['direction'],
            sl_price=t['stop_loss'],
            tp_price=t['take_profit'],  # TP1
        )
        
        original_win = t['win']
        original_result_r = t['result_r']
        
        # Determine correct outcome
        if first_hit_type == 'TP':
            new_win = 1
            if original_win == 0:
                # LOSS → WIN: Recalculate R value based on TP1 hit
                # Run_009 uses partial closes: TP1=34%, TP2=16%, TP3=35%, trail=15%
                entry = t['entry_price']
                tp = t['take_profit']
                sl = t['stop_loss']
                risk = abs(entry - sl)
                if risk > 0:
                    tp_r = abs(tp - entry) / risk
                    # Assume partial close at TP1 (34%) + breakeven trail (66%)
                    new_result_r = tp_r * 0.34 + 0.0 * 0.66  # Trail at breakeven = 0
                    new_result_r = round(new_result_r, 2)
                else:
                    new_result_r = 0.5  # Default small win
            else:
                # Already WIN, keep original R
                new_result_r = original_result_r
        elif first_hit_type == 'SL':
            new_win = 0
            if original_win == 1:
                # WIN → LOSS: Set to -1R (full loss)
                new_result_r = -1.0
            else:
                # Already LOSS, keep original R
                new_result_r = original_result_r
        else:
            # Inconclusive - keep original
            new_win = original_win
            new_result_r = original_result_r
        
        # Track changes
        if original_win == 0 and new_win == 1:
            stats['loss_to_win'] += 1
        elif original_win == 1 and new_win == 0:
            stats['win_to_loss'] += 1
        else:
            stats['unchanged'] += 1
        
        # Create corrected trade
        new_trade = t.to_dict()
        new_trade['win'] = new_win
        new_trade['result_r'] = new_result_r
        corrected_trades.append(new_trade)
    
    # Create corrected DataFrame
    corrected_df = pd.DataFrame(corrected_trades)
    
    # Calculate new stats
    new_wins = corrected_df['win'].sum()
    new_win_rate = corrected_df['win'].mean() * 100
    
    print(f"\n{'='*60}")
    print(f"CORRECTION SUMMARY:")
    print(f"  LOSS → WIN: {stats['loss_to_win']}")
    print(f"  WIN → LOSS: {stats['win_to_loss']}")
    print(f"  Unchanged:  {stats['unchanged']}")
    print(f"  No H1 data: {stats['no_h1_data']}")
    print(f"\nOriginal: {original_wins} wins ({original_win_rate:.1f}%)")
    print(f"Corrected: {new_wins} wins ({new_win_rate:.1f}%)")
    print(f"Net change: {stats['loss_to_win'] - stats['win_to_loss']:+d} wins")
    
    # Save corrected trades
    corrected_df.to_csv(output_csv, index=False)
    print(f"\nSaved to: {output_csv}")
    
    return {
        'original_trades': len(trades),
        'original_wins': int(original_wins),
        'original_win_rate': original_win_rate,
        'corrected_wins': int(new_wins),
        'corrected_win_rate': new_win_rate,
        **stats,
    }


def main():
    # Parse arguments
    if len(sys.argv) >= 3:
        input_csv = sys.argv[1]
        output_csv = sys.argv[2]
    elif len(sys.argv) == 2:
        input_csv = sys.argv[1]
        output_csv = input_csv.replace('.csv', '_h1_corrected.csv')
    else:
        input_csv = DEFAULT_INPUT
        output_csv = DEFAULT_OUTPUT
    
    # Run correction
    stats = apply_h1_correction(input_csv, output_csv)
    
    return stats


if __name__ == '__main__':
    main()
