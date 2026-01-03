#!/usr/bin/env python3
"""
5ers Realistic Intraday DD Analysis

Simulates the ACTUAL 5ers drawdown rules with hourly worst-case equity calculation:

1. DAILY DD (5% rule):
   - Resets every day at 00:00 server time
   - Based on MAX(previous_day_balance, previous_day_equity)
   - Uses FLOATING equity (balance + unrealized P&L)
   - Breach = account terminated

2. TOTAL DD (10% rule):
   - Fixed stop-out at initial_balance * 0.90 = $54,000
   - NEVER changes, regardless of peak equity
   - Breach = account terminated

3. WORST-CASE HOURLY CALCULATION:
   - Long trades: use hourly LOW (worst case for longs)
   - Short trades: use hourly HIGH (worst case for shorts)
   - This simulates real-time mark-to-market exposure

4. PROTECTIVE CLOSE at 4.3% daily DD:
   - Close ALL open trades when daily DD hits 4.3%
   - This prevents hitting the 5% hard limit
   - Simulates what the live bot would do

Output:
- Number of DD breach events
- Trades closed by protective halts
- Adjusted final P&L accounting for forced closes
- Comparison: Original P&L vs Protected P&L
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import json

# Paths
TRADES_FILE = Path("ftmo_analysis_output/TPE/history/run_009/best_trades_final.csv")
H1_DATA_DIR = Path("data/ohlcv")
OUTPUT_DIR = Path("ftmo_analysis_output/TPE/history/run_009/5ers_dd_analysis")

# 5ers Rules (CORRECT IMPLEMENTATION)
INITIAL_BALANCE = 60_000.0
DAILY_DD_LIMIT_PCT = 5.0           # 5% daily drawdown limit
DAILY_DD_SAFETY_PCT = 4.3          # Close all trades at 4.3% (safety threshold)
TOTAL_DD_LIMIT_PCT = 10.0          # 10% total drawdown limit
STOP_OUT_LEVEL = INITIAL_BALANCE * 0.90  # $54,000 (FIXED!)
RISK_PER_TRADE_PCT = 0.65          # 0.65% per trade = $390


@dataclass
class OpenPosition:
    """Track an open position"""
    trade_id: int
    symbol: str
    direction: str
    entry_price: float
    stop_loss: float
    take_profit: float
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    original_result_r: float
    original_profit_usd: float
    
    # Will be set when force-closed
    forced_close: bool = False
    forced_close_time: Optional[pd.Timestamp] = None
    forced_close_price: Optional[float] = None
    forced_close_pnl: Optional[float] = None


@dataclass
class DayState:
    """Track state for a single trading day"""
    date: datetime
    start_balance: float      # Balance at 00:00
    start_equity: float       # Equity at 00:00 (balance + floating at that moment)
    daily_dd_base: float      # MAX(start_balance, start_equity)
    min_equity_allowed: float # daily_dd_base * 0.95
    
    # Tracked throughout day
    lowest_equity: float = field(default=float('inf'))
    max_daily_dd_pct: float = 0.0
    protective_close_triggered: bool = False
    protective_close_time: Optional[pd.Timestamp] = None


@dataclass 
class DDEvent:
    """Record a DD event"""
    timestamp: pd.Timestamp
    event_type: str  # 'daily_warning', 'daily_protective_close', 'daily_breach', 'total_breach'
    daily_dd_pct: float
    total_dd_pct: float
    equity: float
    balance: float
    open_trades: int
    floating_pnl: float
    trades_closed: List[int] = field(default_factory=list)


class FiveersRealisticDDAnalyzer:
    """
    Accurate 5ers DD analysis using hourly worst-case equity.
    """
    
    def __init__(self, trades_df: pd.DataFrame):
        self.trades = trades_df.copy()
        self.trades['entry_date'] = pd.to_datetime(self.trades['entry_date'])
        self.trades['exit_date'] = pd.to_datetime(self.trades['exit_date'])
        
        # H1 data cache
        self.h1_data: Dict[str, pd.DataFrame] = {}
        self.load_h1_data()
        
        # Results
        self.dd_events: List[DDEvent] = []
        self.closed_by_protection: List[OpenPosition] = []
        self.day_states: List[DayState] = []
        
        # Final stats
        self.original_total_profit = 0.0
        self.protected_total_profit = 0.0
        
    def load_h1_data(self):
        """Load H1 candles for all traded symbols"""
        print("📊 Loading H1 data...")
        
        symbols = self.trades['symbol'].unique()
        loaded = 0
        
        for symbol in symbols:
            h1_file = H1_DATA_DIR / f"{symbol}_H1_2023_2025.csv"
            
            if h1_file.exists():
                df = pd.read_csv(h1_file)
                # Handle different column names
                time_col = 'time' if 'time' in df.columns else 'timestamp'
                df['time'] = pd.to_datetime(df[time_col])
                df = df.set_index('time').sort_index()
                
                # Normalize column names
                df.columns = [c.lower() for c in df.columns]
                
                self.h1_data[symbol] = df
                loaded += 1
                print(f"  ✓ {symbol}: {len(df):,} H1 candles")
            else:
                print(f"  ⚠️  {symbol}: H1 data not found")
        
        print(f"\n✅ Loaded H1 data for {loaded}/{len(symbols)} symbols\n")
    
    def get_h1_candle(self, symbol: str, timestamp: pd.Timestamp) -> Optional[Dict]:
        """Get H1 candle for symbol at given timestamp"""
        if symbol not in self.h1_data:
            return None
        
        h1_df = self.h1_data[symbol]
        
        # Find candle at or before timestamp
        try:
            mask = h1_df.index <= timestamp
            if not mask.any():
                return None
            candle = h1_df[mask].iloc[-1]
            return {
                'open': candle['open'],
                'high': candle['high'],
                'low': candle['low'],
                'close': candle['close'],
            }
        except (IndexError, KeyError):
            return None
    
    def calculate_floating_pnl_at_close(
        self, 
        timestamp: pd.Timestamp, 
        open_positions: List[OpenPosition]
    ) -> Tuple[float, Dict[int, float]]:
        """
        Calculate floating P&L using CLOSE prices (for EOD/00:00 snapshots).
        Used for daily DD base calculation.
        
        Returns: (total_floating_pnl, {trade_id: individual_pnl})
        """
        total_floating = 0.0
        individual_pnl = {}
        
        for pos in open_positions:
            candle = self.get_h1_candle(pos.symbol, timestamp)
            
            if candle is None:
                individual_pnl[pos.trade_id] = 0.0
                continue
            
            # Use CLOSE price (not worst-case)
            if pos.direction == 'bullish':
                price_move = candle['close'] - pos.entry_price
            else:
                price_move = pos.entry_price - candle['close']
            
            risk = abs(pos.entry_price - pos.stop_loss)
            if risk > 0:
                r_move = price_move / risk
            else:
                r_move = 0.0
            
            risk_usd = INITIAL_BALANCE * (RISK_PER_TRADE_PCT / 100)
            pnl_usd = r_move * risk_usd
            
            individual_pnl[pos.trade_id] = pnl_usd
            total_floating += pnl_usd
        
        return total_floating, individual_pnl
    
    def calculate_worst_case_floating_pnl(
        self, 
        timestamp: pd.Timestamp, 
        open_positions: List[OpenPosition]
    ) -> Tuple[float, Dict[int, float]]:
        """
        Calculate worst-case floating P&L for all open positions.
        
        Uses:
        - LOW price for LONG positions (worst case)
        - HIGH price for SHORT positions (worst case)
        
        Returns: (total_floating_pnl, {trade_id: individual_pnl})
        """
        total_floating = 0.0
        individual_pnl = {}
        
        for pos in open_positions:
            candle = self.get_h1_candle(pos.symbol, timestamp)
            
            if candle is None:
                # No H1 data - use entry price (assume flat)
                individual_pnl[pos.trade_id] = 0.0
                continue
            
            # Calculate worst-case price for this hour
            if pos.direction == 'bullish':
                # Long position: worst case is LOW
                worst_price = candle['low']
                price_move = worst_price - pos.entry_price
            else:
                # Short position: worst case is HIGH
                worst_price = candle['high']
                price_move = pos.entry_price - worst_price
            
            # Calculate P&L in R
            risk = abs(pos.entry_price - pos.stop_loss)
            if risk > 0:
                r_move = price_move / risk
            else:
                r_move = 0.0
            
            # Convert to USD
            risk_usd = INITIAL_BALANCE * (RISK_PER_TRADE_PCT / 100)
            pnl_usd = r_move * risk_usd
            
            individual_pnl[pos.trade_id] = pnl_usd
            total_floating += pnl_usd
        
        return total_floating, individual_pnl
    
    def calculate_mark_to_market_pnl(
        self, 
        timestamp: pd.Timestamp, 
        pos: OpenPosition
    ) -> float:
        """Calculate current P&L for a single position at close price"""
        candle = self.get_h1_candle(pos.symbol, timestamp)
        
        if candle is None:
            return 0.0
        
        if pos.direction == 'bullish':
            price_move = candle['close'] - pos.entry_price
        else:
            price_move = pos.entry_price - candle['close']
        
        risk = abs(pos.entry_price - pos.stop_loss)
        if risk > 0:
            r_move = price_move / risk
        else:
            r_move = 0.0
        
        risk_usd = INITIAL_BALANCE * (RISK_PER_TRADE_PCT / 100)
        return r_move * risk_usd
    
    def run_simulation(self) -> Dict:
        """
        Run the full simulation hour-by-hour.
        
        Returns comprehensive statistics.
        """
        print("=" * 80)
        print("🚀 5ERS REALISTIC DD SIMULATION")
        print("=" * 80)
        print(f"\nInitial Balance: ${INITIAL_BALANCE:,.0f}")
        print(f"Stop-Out Level:  ${STOP_OUT_LEVEL:,.0f} (fixed)")
        print(f"Daily DD Limit:  {DAILY_DD_LIMIT_PCT}% of previous day equity")
        print(f"Safety Close at: {DAILY_DD_SAFETY_PCT}% daily DD")
        print(f"Risk per trade:  {RISK_PER_TRADE_PCT}% = ${INITIAL_BALANCE * RISK_PER_TRADE_PCT / 100:,.0f}")
        print()
        
        # Initialize state
        current_balance = INITIAL_BALANCE
        current_equity = INITIAL_BALANCE
        
        # Daily DD tracking
        current_day: Optional[datetime] = None
        current_day_state: Optional[DayState] = None
        
        # Open positions
        open_positions: List[OpenPosition] = []
        
        # Trade tracking
        trades_sorted = self.trades.sort_values('entry_date').reset_index(drop=True)
        next_trade_idx = 0
        
        # Calculate original profit (no DD protection)
        self.original_total_profit = trades_sorted['profit_usd'].sum()
        
        # Get date range
        start_date = trades_sorted['entry_date'].min().replace(hour=0, minute=0, second=0)
        end_date = trades_sorted['exit_date'].max() + timedelta(days=1)
        
        print(f"Period: {start_date.date()} to {end_date.date()}")
        print(f"Total trades: {len(trades_sorted)}")
        print(f"Original total profit: ${self.original_total_profit:,.2f}")
        print()
        
        # Generate hourly timestamps
        current_time = start_date
        hours_analyzed = 0
        
        # Track account status
        account_terminated = False
        termination_reason = None
        
        # Progress tracking
        total_hours = int((end_date - start_date).total_seconds() / 3600)
        
        while current_time <= end_date and not account_terminated:
            hours_analyzed += 1
            
            # Progress update
            if hours_analyzed % 500 == 0:
                pct = (hours_analyzed / total_hours) * 100
                print(f"  {pct:.1f}% | {current_time.date()} | "
                      f"Balance: ${current_balance:,.0f} | "
                      f"Open: {len(open_positions)} | "
                      f"Events: {len(self.dd_events)}", end='\r', flush=True)
            
            # === NEW DAY CHECK (00:00 server time) ===
            this_day = current_time.date()
            if current_day != this_day:
                # Save previous day state
                if current_day_state is not None:
                    self.day_states.append(current_day_state)
                
                # Calculate equity at 00:00 using CLOSE prices
                # This represents the snapshot at day transition
                floating_at_00_00, _ = self.calculate_floating_pnl_at_close(
                    current_time, 
                    open_positions
                )
                equity_at_00_00 = current_balance + floating_at_00_00
                
                # Daily DD base = MAX(balance, equity) at 00:00
                # This is exactly how 5ers calculates it
                daily_dd_base = max(current_balance, equity_at_00_00)
                
                # Calculate minimum allowed equity for today
                min_equity_allowed = daily_dd_base * (1 - DAILY_DD_LIMIT_PCT / 100)
                
                # Create new day state
                current_day_state = DayState(
                    date=this_day,
                    start_balance=current_balance,
                    start_equity=equity_at_00_00,
                    daily_dd_base=daily_dd_base,
                    min_equity_allowed=min_equity_allowed,
                    lowest_equity=equity_at_00_00,
                )
                
                current_day = this_day
            
            # === OPEN NEW TRADES ===
            while next_trade_idx < len(trades_sorted):
                trade = trades_sorted.iloc[next_trade_idx]
                
                if trade['entry_date'] <= current_time:
                    pos = OpenPosition(
                        trade_id=int(trade['trade_id']),
                        symbol=trade['symbol'],
                        direction=trade['direction'],
                        entry_price=trade['entry_price'],
                        stop_loss=trade['stop_loss'],
                        take_profit=trade['take_profit'],
                        entry_date=trade['entry_date'],
                        exit_date=trade['exit_date'],
                        original_result_r=trade['result_r'],
                        original_profit_usd=trade['profit_usd'],
                    )
                    open_positions.append(pos)
                    next_trade_idx += 1
                else:
                    break
            
            # === CLOSE TRADES THAT REACHED EXIT DATE (normal exit) ===
            positions_to_close = []
            for pos in open_positions:
                if not pos.forced_close and pos.exit_date <= current_time:
                    # Normal exit - use original profit
                    current_balance += pos.original_profit_usd
                    positions_to_close.append(pos)
            
            for pos in positions_to_close:
                open_positions.remove(pos)
            
            # === CALCULATE WORST-CASE FLOATING P&L ===
            floating_pnl, individual_pnl = self.calculate_worst_case_floating_pnl(
                current_time, open_positions
            )
            current_equity = current_balance + floating_pnl
            
            # Track lowest equity of the day
            if current_day_state and current_equity < current_day_state.lowest_equity:
                current_day_state.lowest_equity = current_equity
            
            # === CHECK TOTAL DD (HARD STOP) ===
            if current_equity < STOP_OUT_LEVEL:
                account_terminated = True
                termination_reason = f"TOTAL DD BREACH: Equity ${current_equity:,.0f} < ${STOP_OUT_LEVEL:,.0f}"
                
                total_dd_pct = ((INITIAL_BALANCE - current_equity) / INITIAL_BALANCE) * 100
                daily_dd_pct = ((current_day_state.daily_dd_base - current_equity) / current_day_state.daily_dd_base) * 100 if current_day_state else 0
                
                self.dd_events.append(DDEvent(
                    timestamp=current_time,
                    event_type='total_breach',
                    daily_dd_pct=daily_dd_pct,
                    total_dd_pct=total_dd_pct,
                    equity=current_equity,
                    balance=current_balance,
                    open_trades=len(open_positions),
                    floating_pnl=floating_pnl,
                    trades_closed=[p.trade_id for p in open_positions],
                ))
                
                print(f"\n\n❌ {termination_reason}")
                break
            
            # === CHECK DAILY DD ===
            if current_day_state and not current_day_state.protective_close_triggered:
                daily_dd_pct = ((current_day_state.daily_dd_base - current_equity) / 
                               current_day_state.daily_dd_base) * 100
                
                if daily_dd_pct > current_day_state.max_daily_dd_pct:
                    current_day_state.max_daily_dd_pct = daily_dd_pct
                
                # Check for protective close threshold (4.3%)
                if daily_dd_pct >= DAILY_DD_SAFETY_PCT and len(open_positions) > 0:
                    current_day_state.protective_close_triggered = True
                    current_day_state.protective_close_time = current_time
                    
                    total_dd_pct = ((INITIAL_BALANCE - current_equity) / INITIAL_BALANCE) * 100
                    
                    # Close ALL open positions at current market price
                    closed_trade_ids = []
                    close_pnl_total = 0.0
                    
                    for pos in open_positions:
                        # Mark-to-market close
                        close_pnl = self.calculate_mark_to_market_pnl(current_time, pos)
                        
                        pos.forced_close = True
                        pos.forced_close_time = current_time
                        pos.forced_close_pnl = close_pnl
                        
                        current_balance += close_pnl
                        close_pnl_total += close_pnl
                        closed_trade_ids.append(pos.trade_id)
                        
                        self.closed_by_protection.append(pos)
                    
                    # Record event
                    self.dd_events.append(DDEvent(
                        timestamp=current_time,
                        event_type='daily_protective_close',
                        daily_dd_pct=daily_dd_pct,
                        total_dd_pct=total_dd_pct,
                        equity=current_equity,
                        balance=current_balance,
                        open_trades=len(open_positions),
                        floating_pnl=floating_pnl,
                        trades_closed=closed_trade_ids,
                    ))
                    
                    # Clear all positions
                    open_positions.clear()
                    
                    # Recalculate equity after close
                    current_equity = current_balance
                
                # Check for hard breach (5%)
                elif daily_dd_pct >= DAILY_DD_LIMIT_PCT:
                    account_terminated = True
                    termination_reason = f"DAILY DD BREACH: {daily_dd_pct:.2f}% >= {DAILY_DD_LIMIT_PCT}%"
                    
                    total_dd_pct = ((INITIAL_BALANCE - current_equity) / INITIAL_BALANCE) * 100
                    
                    self.dd_events.append(DDEvent(
                        timestamp=current_time,
                        event_type='daily_breach',
                        daily_dd_pct=daily_dd_pct,
                        total_dd_pct=total_dd_pct,
                        equity=current_equity,
                        balance=current_balance,
                        open_trades=len(open_positions),
                        floating_pnl=floating_pnl,
                        trades_closed=[p.trade_id for p in open_positions],
                    ))
                    
                    print(f"\n\n❌ {termination_reason}")
                    break
            
            # Next hour
            current_time += timedelta(hours=1)
        
        # Final day state
        if current_day_state:
            self.day_states.append(current_day_state)
        
        # Calculate final protected profit
        self.protected_total_profit = current_balance - INITIAL_BALANCE
        
        # Print results
        print("\n" + "=" * 80)
        self.print_results(account_terminated, termination_reason)
        
        return self.generate_report()
    
    def print_results(self, terminated: bool, reason: Optional[str]):
        """Print analysis results"""
        print("📊 5ERS DD ANALYSIS RESULTS")
        print("=" * 80)
        
        print(f"\n{'❌ ACCOUNT TERMINATED' if terminated else '✅ ACCOUNT SURVIVED'}")
        if reason:
            print(f"   Reason: {reason}")
        
        print(f"\n💰 PROFIT COMPARISON:")
        print(f"   Original (no protection):  ${self.original_total_profit:,.2f}")
        print(f"   With DD protection:        ${self.protected_total_profit:,.2f}")
        
        diff = self.protected_total_profit - self.original_total_profit
        diff_pct = (diff / abs(self.original_total_profit)) * 100 if self.original_total_profit != 0 else 0
        print(f"   Difference:                ${diff:,.2f} ({diff_pct:+.1f}%)")
        
        print(f"\n📅 PROTECTIVE CLOSE EVENTS: {len([e for e in self.dd_events if e.event_type == 'daily_protective_close'])}")
        
        # Show each protective close event
        for event in self.dd_events:
            if event.event_type == 'daily_protective_close':
                print(f"   {event.timestamp}: Daily DD {event.daily_dd_pct:.2f}% | "
                      f"Closed {len(event.trades_closed)} trades | "
                      f"Balance after: ${event.balance:,.0f}")
        
        print(f"\n🔄 TRADES FORCE-CLOSED: {len(self.closed_by_protection)}")
        
        # Calculate P&L difference from force-closed trades
        original_pnl_closed = sum(p.original_profit_usd for p in self.closed_by_protection)
        forced_pnl_closed = sum(p.forced_close_pnl or 0 for p in self.closed_by_protection)
        
        print(f"   Original P&L of those trades: ${original_pnl_closed:,.2f}")
        print(f"   Actual P&L (force-closed):    ${forced_pnl_closed:,.2f}")
        print(f"   P&L saved by protection:      ${forced_pnl_closed - original_pnl_closed:,.2f}")
        
        # Day analysis
        if self.day_states:
            worst_day = max(self.day_states, key=lambda d: d.max_daily_dd_pct)
            print(f"\n📉 WORST DAILY DD:")
            print(f"   Date: {worst_day.date}")
            print(f"   Max DD: {worst_day.max_daily_dd_pct:.2f}%")
            print(f"   DD Base: ${worst_day.daily_dd_base:,.0f}")
            print(f"   Min Equity: ${worst_day.lowest_equity:,.0f}")
        
        # Count days with high DD
        high_dd_days = [d for d in self.day_states if d.max_daily_dd_pct >= 3.0]
        print(f"\n⚠️  DAYS WITH DD >= 3%: {len(high_dd_days)}")
        for day in sorted(high_dd_days, key=lambda d: d.max_daily_dd_pct, reverse=True)[:10]:
            print(f"   {day.date}: {day.max_daily_dd_pct:.2f}%")
    
    def generate_report(self) -> Dict:
        """Generate comprehensive report dictionary"""
        return {
            'initial_balance': INITIAL_BALANCE,
            'stop_out_level': STOP_OUT_LEVEL,
            'original_profit': self.original_total_profit,
            'protected_profit': self.protected_total_profit,
            'profit_difference': self.protected_total_profit - self.original_total_profit,
            'protective_close_events': len([e for e in self.dd_events if e.event_type == 'daily_protective_close']),
            'trades_force_closed': len(self.closed_by_protection),
            'total_dd_breaches': len([e for e in self.dd_events if e.event_type == 'total_breach']),
            'daily_dd_breaches': len([e for e in self.dd_events if e.event_type == 'daily_breach']),
            'days_analyzed': len(self.day_states),
            'worst_daily_dd_pct': max((d.max_daily_dd_pct for d in self.day_states), default=0),
        }
    
    def save_results(self):
        """Save detailed results to CSV files"""
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        # 1. DD Events
        if self.dd_events:
            events_data = []
            for e in self.dd_events:
                events_data.append({
                    'timestamp': e.timestamp,
                    'event_type': e.event_type,
                    'daily_dd_pct': e.daily_dd_pct,
                    'total_dd_pct': e.total_dd_pct,
                    'equity': e.equity,
                    'balance': e.balance,
                    'open_trades': e.open_trades,
                    'floating_pnl': e.floating_pnl,
                    'trades_closed_count': len(e.trades_closed),
                })
            pd.DataFrame(events_data).to_csv(OUTPUT_DIR / 'dd_events.csv', index=False)
            print(f"\n💾 Saved: {OUTPUT_DIR / 'dd_events.csv'}")
        
        # 2. Force-closed trades
        if self.closed_by_protection:
            closed_data = []
            for p in self.closed_by_protection:
                closed_data.append({
                    'trade_id': p.trade_id,
                    'symbol': p.symbol,
                    'direction': p.direction,
                    'entry_date': p.entry_date,
                    'original_exit_date': p.exit_date,
                    'forced_close_time': p.forced_close_time,
                    'entry_price': p.entry_price,
                    'original_profit_usd': p.original_profit_usd,
                    'forced_close_pnl': p.forced_close_pnl,
                    'pnl_difference': (p.forced_close_pnl or 0) - p.original_profit_usd,
                })
            pd.DataFrame(closed_data).to_csv(OUTPUT_DIR / 'force_closed_trades.csv', index=False)
            print(f"💾 Saved: {OUTPUT_DIR / 'force_closed_trades.csv'}")
        
        # 3. Daily stats
        if self.day_states:
            day_data = []
            for d in self.day_states:
                day_data.append({
                    'date': d.date,
                    'start_balance': d.start_balance,
                    'daily_dd_base': d.daily_dd_base,
                    'min_equity_allowed': d.min_equity_allowed,
                    'lowest_equity': d.lowest_equity,
                    'max_daily_dd_pct': d.max_daily_dd_pct,
                    'protective_close': d.protective_close_triggered,
                })
            pd.DataFrame(day_data).to_csv(OUTPUT_DIR / 'daily_dd_stats.csv', index=False)
            print(f"💾 Saved: {OUTPUT_DIR / 'daily_dd_stats.csv'}")
        
        # 4. Summary JSON
        report = self.generate_report()
        with open(OUTPUT_DIR / 'analysis_summary.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"💾 Saved: {OUTPUT_DIR / 'analysis_summary.json'}")


def main():
    """Main entry point"""
    print("\n" + "=" * 80)
    print("5ERS REALISTIC INTRADAY DD ANALYSIS")
    print("Using worst-case hourly equity (Low for longs, High for shorts)")
    print("=" * 80 + "\n")
    
    # Load trades
    if not TRADES_FILE.exists():
        print(f"❌ Trades file not found: {TRADES_FILE}")
        return
    
    print(f"📂 Loading trades from: {TRADES_FILE}")
    trades_df = pd.read_csv(TRADES_FILE)
    print(f"   Found {len(trades_df)} trades\n")
    
    # Run analysis
    analyzer = FiveersRealisticDDAnalyzer(trades_df)
    report = analyzer.run_simulation()
    
    # Save results
    analyzer.save_results()
    
    print("\n" + "=" * 80)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
