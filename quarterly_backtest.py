#!/usr/bin/env python3
"""
Quarterly Backtest Report - Shows Q1/Q2/Q3/Q4 stats with live updates
"""

from datetime import datetime, date
from typing import Dict, List, Tuple
from collections import defaultdict
import sys

from backtest import run_backtest
from config import FOREX_PAIRS, METALS, INDICES, CRYPTO_ASSETS

def get_quarter_from_date(dt) -> str:
    """Get quarter label from datetime/date"""
    if hasattr(dt, 'month'):
        month = dt.month
    else:
        month = dt
    
    if month in [1, 2, 3]:
        return "Q1"
    elif month in [4, 5, 6]:
        return "Q2"
    elif month in [7, 8, 9]:
        return "Q3"
    else:
        return "Q4"

def extract_quarter_stats(trades: List[Dict]) -> Dict[str, Dict]:
    """Extract stats by quarter from trades"""
    quarterly_stats = defaultdict(lambda: {
        "trades": [],
        "total_trades": 0,
        "wins": 0,
        "losses": 0,
        "win_rate": 0.0,
        "total_profit": 0.0,
    })
    
    for trade in trades:
        if "entry_date" not in trade:
            continue
        
        entry_dt = trade["entry_date"]
        if isinstance(entry_dt, str):
            try:
                entry_dt = datetime.fromisoformat(entry_dt.replace('Z', '+00:00'))
            except:
                continue
        
        if hasattr(entry_dt, 'year'):
            quarter_key = f"{entry_dt.year}_{get_quarter_from_date(entry_dt)}"
        else:
            quarter_key = "Unknown"
        
        quarterly_stats[quarter_key]["trades"].append(trade)
        quarterly_stats[quarter_key]["total_trades"] += 1
        
        if trade.get("rr", 0) > 0:
            quarterly_stats[quarter_key]["wins"] += 1
        else:
            quarterly_stats[quarter_key]["losses"] += 1
        
        # Calculate profit based on RR
        rr = trade.get("rr", 0)
        quarterly_stats[quarter_key]["total_profit"] += rr
    
    # Calculate win rates
    for q in quarterly_stats:
        total = quarterly_stats[q]["total_trades"]
        if total > 0:
            quarterly_stats[q]["win_rate"] = (quarterly_stats[q]["wins"] / total) * 100
    
    return quarterly_stats

def print_quarterly_report(quarterly_stats: Dict) -> Tuple[int, float]:
    """Print formatted quarterly report, return total trades and profit"""
    print("\n" + "=" * 90)
    print("QUARTERLY BACKTEST REPORT - ALL SYMBOLS".center(90))
    print("=" * 90)
    
    sorted_quarters = sorted(quarterly_stats.keys())
    
    total_all_trades = 0
    total_all_profit = 0
    
    for quarter in sorted_quarters:
        stats = quarterly_stats[quarter]
        trades = stats["total_trades"]
        win_rate = stats["win_rate"]
        profit_r = stats["total_profit"]
        
        total_all_trades += trades
        total_all_profit += profit_r
        
        # Calculate profit in USD (assuming 100K account, 1% risk = $1000 per R)
        profit_usd = profit_r * 1000  # 1R = $1000 per trade
        
        status = "✓ PASS" if (trades >= 20 and win_rate >= 30 and profit_r > 0) else "✗"
        
        print(f"\n{quarter:>12} | Trades: {trades:3d} | Win Rate: {win_rate:5.1f}% | "
              f"Profit: {profit_r:+7.1f}R (${profit_usd:+10,.0f}) {status}")
    
    print("\n" + "=" * 90)
    total_usd = total_all_profit * 1000
    
    # Check targets: 20+ trades/Q, 30%+ WR, 30% total profit
    all_pass = True
    quarterly_pass = all(
        (quarterly_stats[q]["total_trades"] >= 20 and 
         quarterly_stats[q]["win_rate"] >= 30 and 
         quarterly_stats[q]["total_profit"] > 0)
        for q in quarterly_stats if quarterly_stats[q]["total_trades"] > 0
    )
    
    total_profit_pct = ((total_all_profit * 1000) / 100000) * 100  # %return on $100k
    
    print(f"TOTAL          | Trades: {total_all_trades:3d} | "
          f"Profit: {total_all_profit:+7.1f}R (${total_usd:+12,.0f}) | "
          f"Return: {total_profit_pct:+.1f}% | TARGET: +$150,000")
    
    if quarterly_pass:
        print("\n✓ ALL QUARTERS PASS: 20+ trades, 30%+ win rate, positive profit per quarter")
    else:
        print("\n✗ QUARTERS BELOW TARGET - Optimization needed")
    
    print("=" * 90 + "\n")
    
    return total_all_trades, total_all_profit

def run_quarterly_backtest():
    """Run backtest on all symbols and show quarterly breakdown"""
    all_symbols = FOREX_PAIRS + METALS + INDICES + CRYPTO_ASSETS
    combined_trades = []
    
    print(f"\n[*] Running quarterly backtest on {len(all_symbols)} symbols...")
    
    for i, symbol in enumerate(all_symbols):
        try:
            result = run_backtest(symbol, "Jan 2023 - Now")
            if result.get("trades"):
                combined_trades.extend(result["trades"])
                print(f"    [{i+1}/{len(all_symbols)}] {symbol:12} -> {len(result['trades']):3d} trades")
        except Exception as e:
            print(f"    [{i+1}/{len(all_symbols)}] {symbol:12} -> ERROR: {str(e)[:40]}")
            continue
    
    if not combined_trades:
        print("\n[!] No trades generated. Check data and strategy parameters.")
        return
    
    # Extract and display quarterly stats
    quarterly_stats = extract_quarter_stats(combined_trades)
    total_trades, total_profit = print_quarterly_report(quarterly_stats)
    
    # Summary
    print(f"[✓] Backtest Complete: {total_trades} total trades | {total_profit:+.1f}R total profit")

if __name__ == "__main__":
    run_quarterly_backtest()
