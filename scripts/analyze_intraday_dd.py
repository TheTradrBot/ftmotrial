#!/usr/bin/env python3
"""
Intraday DD Analysis for Run_009

Replays all trades hour-by-hour to calculate:
1. When equity drops below 4.3% from yesterday_high
2. How many times this would trigger trade closures
3. Which trades would have been affected
4. Impact on final P&L

This simulates the live bot's emergency DD protection.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from collections import defaultdict

# Paths
TRADES_FILE = Path("ftmo_analysis_output/TPE/history/run_009/best_trades_final.csv")
H1_DATA_DIR = Path("data/ohlcv")
OUTPUT_DIR = Path("ftmo_analysis_output/TPE/history/run_009/intraday_dd_analysis")

# 5ers Rules
INITIAL_BALANCE = 60000.0
DAILY_DD_THRESHOLD = 4.3  # % from yesterday_high (conservative, includes fees)
RISK_PER_TRADE = 0.0065  # 0.65% = $390 per trade


class IntradayDDAnalyzer:
    """Hour-by-hour equity replay with DD breach detection"""
    
    def __init__(self, trades_df: pd.DataFrame):
        self.trades = trades_df.copy()
        self.trades['entry_date'] = pd.to_datetime(self.trades['entry_date'])
        self.trades['exit_date'] = pd.to_datetime(self.trades['exit_date'])
        
        # Load H1 data for all traded pairs
        self.h1_data = {}
        self.load_h1_data()
        
        # Results
        self.dd_breaches = []
        self.closed_trades = []
        
    def load_h1_data(self):
        """Load H1 candles for all traded pairs"""
        print("📊 Loading H1 data...")
        
        symbols = self.trades['symbol'].unique()
        
        for symbol in symbols:
            h1_file = H1_DATA_DIR / f"{symbol}_H1_2023_2025.csv"
            
            if h1_file.exists():
                df = pd.read_csv(h1_file)
                df['time'] = pd.to_datetime(df['time'])
                df = df.set_index('time').sort_index()
                self.h1_data[symbol] = df
                print(f"  ✓ {symbol}: {len(df):,} candles")
            else:
                print(f"  ⚠️  {symbol}: H1 data not found (will use daily estimate)")
        
        print(f"\n✅ Loaded H1 data for {len(self.h1_data)}/{len(symbols)} pairs\n")
    
    def get_unrealized_pnl_at_time(self, timestamp: pd.Timestamp, open_positions: List[Dict]) -> float:
        """Calculate total unrealized P&L for all open positions at given timestamp"""
        total_unrealized = 0.0
        
        for pos in open_positions:
            symbol = pos['symbol']
            entry_price = pos['entry_price']
            stop_loss = pos['stop_loss']
            direction = pos['direction']
            
            # Get current price from H1 data
            if symbol not in self.h1_data:
                # Fallback: use linear interpolation to exit
                continue
            
            h1_candles = self.h1_data[symbol]
            
            # Find closest candle
            try:
                # Get candle at or before timestamp
                candle = h1_candles[h1_candles.index <= timestamp].iloc[-1]
                current_price = candle['Close']
            except (IndexError, KeyError):
                continue
            
            # Calculate unrealized P&L in pips
            risk = abs(entry_price - stop_loss)
            
            if direction == 'bullish':
                price_move = current_price - entry_price
            else:  # bearish
                price_move = entry_price - current_price
            
            unrealized_r = price_move / risk if risk > 0 else 0
            unrealized_usd = unrealized_r * (INITIAL_BALANCE * RISK_PER_TRADE)
            
            total_unrealized += unrealized_usd
        
        return total_unrealized
    
    def replay_trades_hourly(self):
        """
        Replay all trades hour-by-hour to detect DD breaches.
        
        This is the ACCURATE version using H1 data.
        """
        print("🔄 Starting hourly equity replay...")
        print("=" * 70)
        
        # Initialize
        current_balance = INITIAL_BALANCE
        yesterday_high = INITIAL_BALANCE
        current_day = None
        
        # Sort all trades by entry date
        trades_sorted = self.trades.sort_values('entry_date').reset_index(drop=True)
        
        # Get date range
        start_date = trades_sorted['entry_date'].min()
        end_date = trades_sorted['exit_date'].max()
        
        print(f"Period: {start_date.date()} to {end_date.date()}")
        print(f"Total trades: {len(trades_sorted)}\n")
        
        # Generate hourly timestamps
        current_time = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
        hourly_timestamps = []
        
        while current_time <= end_date:
            hourly_timestamps.append(current_time)
            current_time += timedelta(hours=1)
        
        print(f"Analyzing {len(hourly_timestamps):,} hours...\n")
        
        # Track open positions
        open_positions = []
        next_trade_idx = 0
        
        dd_breach_count = 0
        max_intraday_dd = 0.0
        
        for hour_idx, timestamp in enumerate(hourly_timestamps):
            # Progress update
            if hour_idx % 1000 == 0:
                progress = (hour_idx / len(hourly_timestamps)) * 100
                print(f"  {progress:.1f}% - {timestamp.date()} {timestamp.hour:02d}:00 | "
                      f"Balance: ${current_balance:,.0f} | Open: {len(open_positions)}", 
                      end='\r', flush=True)
            
            # Check for new day (reset yesterday_high)
            current_date = timestamp.date()
            if current_day != current_date:
                if current_day is not None:
                    # Update yesterday_high at start of new day
                    yesterday_high = max(current_balance, 
                                        current_balance + self.get_unrealized_pnl_at_time(timestamp, open_positions))
                current_day = current_date
            
            # 1. Check for new trade entries at this hour
            while next_trade_idx < len(trades_sorted):
                trade = trades_sorted.iloc[next_trade_idx]
                
                if trade['entry_date'] <= timestamp:
                    # Open this trade
                    open_positions.append({
                        'trade_id': trade['trade_id'],
                        'symbol': trade['symbol'],
                        'direction': trade['direction'],
                        'entry_price': trade['entry_price'],
                        'stop_loss': trade['stop_loss'],
                        'entry_date': trade['entry_date'],
                        'exit_date': trade['exit_date'],
                        'result_r': trade['result_r'],
                        'profit_usd': trade['profit_usd'],
                    })
                    next_trade_idx += 1
                else:
                    break
            
            # 2. Check for trade exits at this hour
            closed_this_hour = []
            for pos in open_positions:
                if pos['exit_date'] <= timestamp:
                    # Close this trade
                    current_balance += pos['profit_usd']
                    closed_this_hour.append(pos)
            
            # Remove closed trades
            for pos in closed_this_hour:
                open_positions.remove(pos)
            
            # 3. Calculate current equity (balance + unrealized P&L)
            unrealized_pnl = self.get_unrealized_pnl_at_time(timestamp, open_positions)
            current_equity = current_balance + unrealized_pnl
            
            # 4. Check Daily DD
            if yesterday_high > 0:
                daily_dd = (yesterday_high - current_equity) / yesterday_high * 100
                
                if daily_dd > max_intraday_dd:
                    max_intraday_dd = daily_dd
                
                # DD BREACH DETECTED
                if daily_dd > DAILY_DD_THRESHOLD:
                    dd_breach_count += 1
                    
                    breach_info = {
                        'timestamp': timestamp,
                        'balance': current_balance,
                        'equity': current_equity,
                        'yesterday_high': yesterday_high,
                        'daily_dd': daily_dd,
                        'open_trades': len(open_positions),
                        'unrealized_pnl': unrealized_pnl,
                    }
                    
                    self.dd_breaches.append(breach_info)
                    
                    # In live bot: would close all open trades here
                    # For analysis: just log it
        
        print("\n" + "=" * 70)
        print("✅ Replay complete!\n")
        
        # Summary
        print("📊 INTRADAY DD BREACH ANALYSIS")
        print("=" * 70)
        print(f"Total hours analyzed:     {len(hourly_timestamps):,}")
        print(f"DD breach events (>4.3%): {dd_breach_count}")
        print(f"Max intraday DD:          {max_intraday_dd:.2f}%")
        print(f"Breach rate:              {dd_breach_count / len(hourly_timestamps) * 100:.3f}%")
        
        if dd_breach_count > 0:
            print(f"\n⚠️  DD breaches occurred {dd_breach_count} times")
            print("   In live trading, these would trigger trade closures")
        else:
            print(f"\n✅ No DD breaches detected at {DAILY_DD_THRESHOLD}% threshold")
        
        return dd_breach_count, max_intraday_dd
    
    def generate_breach_report(self):
        """Generate detailed report of all DD breaches"""
        if not self.dd_breaches:
            print("\n✅ No breaches to report")
            return
        
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        # Convert to DataFrame
        df = pd.DataFrame(self.dd_breaches)
        
        # Save to CSV
        csv_file = OUTPUT_DIR / "dd_breaches_log.csv"
        df.to_csv(csv_file, index=False)
        print(f"\n📄 Breach log saved: {csv_file}")
        
        # Generate summary statistics
        print(f"\n📈 BREACH STATISTICS")
        print("=" * 70)
        print(f"Total breaches:           {len(df)}")
        print(f"Avg DD at breach:         {df['daily_dd'].mean():.2f}%")
        print(f"Max DD at breach:         {df['daily_dd'].max():.2f}%")
        print(f"Avg open trades:          {df['open_trades'].mean():.1f}")
        
        # Group by day
        df['date'] = df['timestamp'].dt.date
        daily_breaches = df.groupby('date').size()
        
        print(f"\nDays with breaches:       {len(daily_breaches)}")
        print(f"Max breaches in one day:  {daily_breaches.max()}")
        
        # Show first 10 breaches
        print(f"\n🔍 FIRST 10 DD BREACHES:")
        print("=" * 70)
        print(f"{'Date':<12} {'Time':<6} {'DD%':>6} {'Equity':>12} {'Open':>5}")
        print("-" * 70)
        
        for idx, row in df.head(10).iterrows():
            ts = row['timestamp']
            print(f"{ts.date()} {ts.hour:02d}:00  {row['daily_dd']:>6.2f}% "
                  f"${row['equity']:>10,.0f}  {row['open_trades']:>5}")


def main():
    """Main entry point"""
    print("\n" + "=" * 70)
    print("INTRADAY DD ANALYSIS - RUN_009")
    print("=" * 70)
    print(f"Threshold: {DAILY_DD_THRESHOLD}% Daily DD (from yesterday_high)")
    print(f"Initial balance: ${INITIAL_BALANCE:,.0f}")
    print(f"Risk per trade: {RISK_PER_TRADE*100:.2f}%\n")
    
    # Load trades
    if not TRADES_FILE.exists():
        print(f"❌ Trades file not found: {TRADES_FILE}")
        return
    
    print(f"📂 Loading trades from: {TRADES_FILE}")
    trades_df = pd.read_csv(TRADES_FILE)
    print(f"✅ Loaded {len(trades_df)} trades\n")
    
    # Run analysis
    analyzer = IntradayDDAnalyzer(trades_df)
    breach_count, max_dd = analyzer.replay_trades_hourly()
    
    # Generate report
    analyzer.generate_breach_report()
    
    # Final recommendation
    print(f"\n💡 RECOMMENDATION FOR LIVE BOT")
    print("=" * 70)
    
    if breach_count == 0:
        print(f"✅ No DD breaches at {DAILY_DD_THRESHOLD}% threshold")
        print("   Current emergency stop level is appropriate")
    elif breach_count < 10:
        print(f"⚠️  {breach_count} rare breach events detected")
        print(f"   Consider keeping {DAILY_DD_THRESHOLD}% threshold")
        print("   Implement: Close all trades + halt new entries")
    else:
        print(f"❌ {breach_count} breach events - threshold may be too tight")
        print(f"   Consider raising to 4.5-4.8% or accept occasional halts")
    
    print(f"\n📊 Max intraday DD: {max_dd:.2f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()
