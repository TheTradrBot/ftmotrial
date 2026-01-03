#!/usr/bin/env python3
"""
5ers REALISTIC Live Trading Simulation

This simulates ACTUAL live trading with proper SL/TP detection:

1. SL/TP DETECTION:
   - Checks EVERY H1 candle for SL/TP hits
   - Exits trades IMMEDIATELY when price touches SL or TP
   - Uses conservative assumptions (if both hit, assume SL first)

2. DAILY DD (5% rule):
   - Resets at 00:00 server time
   - Based on MAX(balance, equity) at 00:00 using CLOSE prices
   - Uses WORST-CASE floating P&L for intraday monitoring
   - Breach = account terminated (NO protective close)

3. TOTAL DD (10% rule):
   - Fixed at $54,000 (never changes)
   - Breach = account terminated

4. REALISTIC COSTS:
   - Spread: ~2 pips average
   - Commission: $3.50 per lot per side
   - Total cost per trade: ~$27

Key Differences from Backtest:
- Backtest assumes perfect exits at predetermined times
- Live trading exits at SL/TP whenever they're hit
- This can DRAMATICALLY change results and DD patterns

Output:
- Exact breach events (if any)
- Trade-by-trade execution details
- Comparison: Backtest P&L vs Live P&L
- Daily DD statistics
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import json
import argparse

# Paths
TRADES_FILE = Path("ftmo_analysis_output/TPE/history/run_009/best_trades_final.csv")
H1_DATA_DIR = Path("data/ohlcv")
OUTPUT_DIR = Path("ftmo_analysis_output/TPE/history/run_009/5ers_live_simulation")

# 5ers Rules
INITIAL_BALANCE = 60_000.0
DAILY_DD_LIMIT_PCT = 5.0           # Hard breach
TOTAL_DD_LIMIT_PCT = 10.0          # Hard breach
STOP_OUT_LEVEL = INITIAL_BALANCE * 0.90  # $54,000 (FIXED!)
RISK_PER_TRADE_PCT = 0.65          # Risk per trade

# Trading Costs (realistic)
SPREAD_PIPS = 2.0                  # Average spread
COMMISSION_PER_LOT = 3.5           # Per side
PIP_VALUE = 10.0                   # $10 per pip for standard lot


@dataclass
class LiveTrade:
    """Track a live trade execution"""
    trade_id: int
    symbol: str
    direction: str
    entry_price: float
    stop_loss: float
    take_profit: float
    entry_time: pd.Timestamp
    risk_usd: float
    
    # Backtest reference (for comparison)
    backtest_exit_date: pd.Timestamp
    backtest_result_r: float
    backtest_profit_usd: float
    
    # Live execution results
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None  # 'sl', 'tp', 'forced'
    live_result_r: Optional[float] = None
    live_profit_usd: Optional[float] = None
    
    def is_closed(self) -> bool:
        return self.exit_time is not None


@dataclass
class DayState:
    """Daily DD state"""
    date: datetime
    start_balance: float
    start_equity: float
    daily_dd_base: float              # MAX(balance, equity) at 00:00
    min_equity_allowed: float         # 95% of dd_base
    
    # Tracked throughout day
    lowest_equity: float = field(default=float('inf'))
    max_daily_dd_pct: float = 0.0


@dataclass
class BreachEvent:
    """Record a DD breach"""
    timestamp: pd.Timestamp
    breach_type: str  # 'daily' or 'total'
    daily_dd_pct: float
    total_dd_pct: float
    equity: float
    balance: float
    open_trades: int
    floating_pnl: float


class Realistic5ersSimulator:
    """
    Realistic live trading simulator with proper SL/TP detection.
    """
    
    def __init__(self, trades_file: Path):
        print(f"📂 Loading trades from: {trades_file}")
        self.trades = pd.read_csv(trades_file)
        self.trades['entry_date'] = pd.to_datetime(self.trades['entry_date'])
        self.trades['exit_date'] = pd.to_datetime(self.trades['exit_date'])
        print(f"   ✓ Loaded {len(self.trades)} trade signals\n")
        
        # H1 data cache
        self.h1_data: Dict[str, pd.DataFrame] = {}
        self.load_h1_data()
        
        # Results
        self.executed_trades: List[LiveTrade] = []
        self.day_states: List[DayState] = []
        self.breaches: List[BreachEvent] = []
        
        # Stats
        self.backtest_total_profit = self.trades['profit_usd'].sum()
        self.live_total_profit = 0.0
        
    def load_h1_data(self):
        """Load H1 candles for all symbols"""
        print("📊 Loading H1 data...")
        
        symbols = self.trades['symbol'].unique()
        loaded = 0
        missing = []
        
        for symbol in symbols:
            h1_file = H1_DATA_DIR / f"{symbol}_H1_2023_2025.csv"
            
            if h1_file.exists():
                df = pd.read_csv(h1_file)
                time_col = 'time' if 'time' in df.columns else 'timestamp'
                df['time'] = pd.to_datetime(df[time_col])
                df = df.set_index('time').sort_index()
                df.columns = [c.lower() for c in df.columns]
                
                self.h1_data[symbol] = df
                loaded += 1
                print(f"  ✓ {symbol}: {len(df):,} candles")
            else:
                missing.append(symbol)
                print(f"  ⚠️  {symbol}: H1 data NOT FOUND")
        
        if missing:
            print(f"\n⚠️  WARNING: Missing H1 data for {len(missing)} symbols!")
            print(f"   Trades for these symbols will use backtest exits (less accurate)")
        
        print(f"\n✅ Loaded {loaded}/{len(symbols)} symbols\n")
    
    def get_candle(self, symbol: str, timestamp: pd.Timestamp) -> Optional[Dict]:
        """Get H1 candle at or before timestamp"""
        if symbol not in self.h1_data:
            return None
        
        h1_df = self.h1_data[symbol]
        
        try:
            # Find candle at or before timestamp
            mask = h1_df.index <= timestamp
            if not mask.any():
                return None
            
            candle = h1_df[mask].iloc[-1]
            
            return {
                'timestamp': candle.name,
                'open': candle['open'],
                'high': candle['high'],
                'low': candle['low'],
                'close': candle['close'],
            }
        except (IndexError, KeyError):
            return None
    
    def check_sl_tp_hit(
        self, 
        trade: LiveTrade, 
        candle: Dict
    ) -> Optional[Tuple[float, str]]:
        """
        Check if SL or TP was hit in this candle.
        
        Returns: (exit_price, exit_reason) or None
        
        Logic:
        - LONG: SL hit if low <= SL, TP hit if high >= TP
        - SHORT: SL hit if high >= SL, TP hit if low <= TP
        - If BOTH hit in same candle: conservatively assume SL first
          (unless open price gapped through TP)
        """
        if trade.direction == 'bullish':
            # Long trade
            sl_hit = candle['low'] <= trade.stop_loss
            tp_hit = candle['high'] >= trade.take_profit
            
            if sl_hit and tp_hit:
                # Both levels touched - which happened first?
                # Conservative: assume SL unless we gapped up
                if candle['open'] >= trade.take_profit:
                    return (trade.take_profit, 'tp')
                else:
                    return (trade.stop_loss, 'sl')
            elif sl_hit:
                return (trade.stop_loss, 'sl')
            elif tp_hit:
                return (trade.take_profit, 'tp')
        
        else:
            # Short trade
            sl_hit = candle['high'] >= trade.stop_loss
            tp_hit = candle['low'] <= trade.take_profit
            
            if sl_hit and tp_hit:
                if candle['open'] <= trade.take_profit:
                    return (trade.take_profit, 'tp')
                else:
                    return (trade.stop_loss, 'sl')
            elif sl_hit:
                return (trade.stop_loss, 'sl')
            elif tp_hit:
                return (trade.take_profit, 'tp')
        
        return None
    
    def calculate_trade_pnl(
        self, 
        trade: LiveTrade, 
        exit_price: float,
        include_costs: bool = True
    ) -> Tuple[float, float]:
        """
        Calculate P&L for a trade.
        
        Returns: (result_r, profit_usd)
        """
        # Calculate price movement
        if trade.direction == 'bullish':
            price_move = exit_price - trade.entry_price
        else:
            price_move = trade.entry_price - exit_price
        
        # Calculate R
        risk_distance = abs(trade.entry_price - trade.stop_loss)
        result_r = price_move / risk_distance if risk_distance > 0 else 0.0
        
        # Calculate USD P&L
        profit_usd = result_r * trade.risk_usd
        
        # Apply trading costs
        if include_costs:
            # Simplified: assume 1 standard lot per trade
            spread_cost = SPREAD_PIPS * PIP_VALUE
            commission_cost = COMMISSION_PER_LOT * 2  # Round-trip
            total_cost = spread_cost + commission_cost
            
            profit_usd -= total_cost
        
        return result_r, profit_usd
    
    def calculate_floating_pnl_close(
        self,
        timestamp: pd.Timestamp,
        open_trades: List[LiveTrade]
    ) -> float:
        """
        Calculate floating P&L using CLOSE prices.
        Used for EOD equity snapshots (daily DD base).
        """
        total_floating = 0.0
        
        for trade in open_trades:
            candle = self.get_candle(trade.symbol, timestamp)
            if not candle:
                continue
            
            _, pnl = self.calculate_trade_pnl(trade, candle['close'], include_costs=False)
            total_floating += pnl
        
        return total_floating
    
    def calculate_floating_pnl_worst_case(
        self,
        timestamp: pd.Timestamp,
        open_trades: List[LiveTrade]
    ) -> float:
        """
        Calculate worst-case floating P&L.
        - LONG: use LOW price
        - SHORT: use HIGH price
        
        Used for intraday DD monitoring.
        """
        total_floating = 0.0
        
        for trade in open_trades:
            candle = self.get_candle(trade.symbol, timestamp)
            if not candle:
                continue
            
            # Worst-case price
            if trade.direction == 'bullish':
                worst_price = candle['low']
            else:
                worst_price = candle['high']
            
            _, pnl = self.calculate_trade_pnl(trade, worst_price, include_costs=False)
            total_floating += pnl
        
        return total_floating
    
    def run_simulation(self, continue_after_breach: bool = False) -> Dict:
        """Run the complete live trading simulation
        
        Args:
            continue_after_breach: If True, don't stop at breaches - count all of them
        """
        print("=" * 80)
        print("🚀 5ERS REALISTIC LIVE TRADING SIMULATION")
        if continue_after_breach:
            print("📊 MODE: COUNT ALL BREACHES (no account reset)")
        print("=" * 80)
        print(f"\nInitial Balance:    ${INITIAL_BALANCE:,.0f}")
        print(f"Stop-Out Level:     ${STOP_OUT_LEVEL:,.0f} (FIXED)")
        print(f"Daily DD Limit:     {DAILY_DD_LIMIT_PCT}% (HARD BREACH)")
        print(f"Total DD Limit:     {TOTAL_DD_LIMIT_PCT}% (HARD BREACH)")
        print(f"Risk per Trade:     {RISK_PER_TRADE_PCT}% = ${INITIAL_BALANCE * RISK_PER_TRADE_PCT / 100:,.0f}")
        print(f"Spread Cost:        {SPREAD_PIPS} pips = ${SPREAD_PIPS * PIP_VALUE:.0f}")
        print(f"Commission:         ${COMMISSION_PER_LOT * 2:.0f} per trade")
        print(f"Total Cost/Trade:   ~${(SPREAD_PIPS * PIP_VALUE) + (COMMISSION_PER_LOT * 2):.0f}\n")
        
        # Initialize state
        current_balance = INITIAL_BALANCE
        current_day: Optional[datetime] = None
        current_day_state: Optional[DayState] = None
        
        # Open trades
        open_trades: List[LiveTrade] = []
        
        # Trade signals queue
        trades_sorted = self.trades.sort_values('entry_date').reset_index(drop=True)
        next_trade_idx = 0
        
        # Simulation period
        start_date = trades_sorted['entry_date'].min().replace(hour=0, minute=0, second=0)
        end_date = trades_sorted['exit_date'].max() + timedelta(days=1)
        
        print(f"📅 Period: {start_date.date()} → {end_date.date()}")
        print(f"📊 Trade Signals: {len(trades_sorted)}")
        print(f"💰 Backtest P&L: ${self.backtest_total_profit:,.2f}\n")
        print("Starting hourly simulation...\n")
        
        # Hourly loop
        current_time = start_date
        hours_simulated = 0
        total_hours = int((end_date - start_date).total_seconds() / 3600)
        
        account_terminated = False
        termination_reason = None
        
        while current_time <= end_date and not account_terminated:
            hours_simulated += 1
            
            # Progress indicator
            if hours_simulated % 500 == 0:
                progress = (hours_simulated / total_hours) * 100
                print(f"  {progress:.1f}% | {current_time.date()} | "
                      f"Balance: ${current_balance:,.0f} | "
                      f"Open: {len(open_trades)} | "
                      f"Executed: {len(self.executed_trades)}", 
                      end='\r', flush=True)
            
            # === NEW DAY (00:00 server time) ===
            this_day = current_time.date()
            if current_day != this_day:
                # Save previous day
                if current_day_state is not None:
                    self.day_states.append(current_day_state)
                
                # Calculate equity at 00:00 (using CLOSE prices)
                floating_at_midnight = self.calculate_floating_pnl_close(
                    current_time, 
                    open_trades
                )
                equity_at_midnight = current_balance + floating_at_midnight
                
                # Daily DD base = MAX(balance, equity) at 00:00
                daily_dd_base = max(current_balance, equity_at_midnight)
                min_equity_allowed = daily_dd_base * (1 - DAILY_DD_LIMIT_PCT / 100)
                
                # Create new day state
                current_day_state = DayState(
                    date=this_day,
                    start_balance=current_balance,
                    start_equity=equity_at_midnight,
                    daily_dd_base=daily_dd_base,
                    min_equity_allowed=min_equity_allowed,
                    lowest_equity=equity_at_midnight,
                )
                
                current_day = this_day
            
            # === OPEN NEW TRADES (at entry_date) ===
            while next_trade_idx < len(trades_sorted):
                signal = trades_sorted.iloc[next_trade_idx]
                
                if signal['entry_date'] <= current_time:
                    # Create live trade
                    trade = LiveTrade(
                        trade_id=int(signal['trade_id']),
                        symbol=signal['symbol'],
                        direction=signal['direction'],
                        entry_price=signal['entry_price'],
                        stop_loss=signal['stop_loss'],
                        take_profit=signal['take_profit'],
                        entry_time=current_time,
                        risk_usd=INITIAL_BALANCE * (RISK_PER_TRADE_PCT / 100),
                        backtest_exit_date=signal['exit_date'],
                        backtest_result_r=signal['result_r'],
                        backtest_profit_usd=signal['profit_usd'],
                    )
                    open_trades.append(trade)
                    next_trade_idx += 1
                else:
                    break
            
            # === CHECK SL/TP HITS ===
            trades_to_close = []
            
            for trade in open_trades:
                # Get current candle
                candle = self.get_candle(trade.symbol, current_time)
                
                if candle:
                    # Check if SL or TP hit
                    hit = self.check_sl_tp_hit(trade, candle)
                    
                    if hit:
                        exit_price, exit_reason = hit
                        
                        # Calculate live results
                        result_r, profit_usd = self.calculate_trade_pnl(
                            trade, 
                            exit_price
                        )
                        
                        # Update trade
                        trade.exit_time = current_time
                        trade.exit_price = exit_price
                        trade.exit_reason = exit_reason
                        trade.live_result_r = result_r
                        trade.live_profit_usd = profit_usd
                        
                        # Update balance
                        current_balance += profit_usd
                        
                        # Mark for closure
                        trades_to_close.append(trade)
                        self.executed_trades.append(trade)
                
                # Also check if we've reached backtest exit date
                # (fallback for trades without H1 data)
                elif not candle and trade.backtest_exit_date <= current_time:
                    # No H1 data - use backtest exit
                    trade.exit_time = current_time
                    trade.exit_price = None  # Unknown
                    trade.exit_reason = 'backtest_fallback'
                    trade.live_result_r = trade.backtest_result_r
                    trade.live_profit_usd = trade.backtest_profit_usd
                    
                    current_balance += trade.backtest_profit_usd
                    trades_to_close.append(trade)
                    self.executed_trades.append(trade)
            
            # Remove closed trades
            for trade in trades_to_close:
                open_trades.remove(trade)
            
            # === CALCULATE CURRENT EQUITY (worst-case for DD monitoring) ===
            floating_pnl = self.calculate_floating_pnl_worst_case(
                current_time, 
                open_trades
            )
            current_equity = current_balance + floating_pnl
            
            # Track lowest equity of the day
            if current_day_state and current_equity < current_day_state.lowest_equity:
                current_day_state.lowest_equity = current_equity
            
            # === CHECK TOTAL DD (10%) ===
            if current_equity < STOP_OUT_LEVEL:
                total_dd_pct = ((INITIAL_BALANCE - current_equity) / INITIAL_BALANCE) * 100
                daily_dd_pct = ((current_day_state.daily_dd_base - current_equity) / 
                               current_day_state.daily_dd_base) * 100 if current_day_state else 0
                
                breach = BreachEvent(
                    timestamp=current_time,
                    breach_type='total',
                    daily_dd_pct=daily_dd_pct,
                    total_dd_pct=total_dd_pct,
                    equity=current_equity,
                    balance=current_balance,
                    open_trades=len(open_trades),
                    floating_pnl=floating_pnl,
                )
                self.breaches.append(breach)
                
                print(f"\n\n❌ TOTAL DD BREACH #{len(self.breaches)}!")
                print(f"   Time: {current_time}")
                print(f"   Equity: ${current_equity:,.2f} < ${STOP_OUT_LEVEL:,.2f}")
                print(f"   Total DD: {total_dd_pct:.2f}%")
                
                if not continue_after_breach:
                    account_terminated = True
                    termination_reason = "TOTAL DD BREACH"
                    break
            
            # === CHECK DAILY DD (5%) ===
            if current_day_state:
                daily_dd_pct = ((current_day_state.daily_dd_base - current_equity) / 
                               current_day_state.daily_dd_base) * 100
                
                if daily_dd_pct > current_day_state.max_daily_dd_pct:
                    current_day_state.max_daily_dd_pct = daily_dd_pct
                
                # Hard breach at 5%
                if daily_dd_pct >= DAILY_DD_LIMIT_PCT:
                    total_dd_pct = ((INITIAL_BALANCE - current_equity) / INITIAL_BALANCE) * 100
                    
                    breach = BreachEvent(
                        timestamp=current_time,
                        breach_type='daily',
                        daily_dd_pct=daily_dd_pct,
                        total_dd_pct=total_dd_pct,
                        equity=current_equity,
                        balance=current_balance,
                        open_trades=len(open_trades),
                        floating_pnl=floating_pnl,
                    )
                    self.breaches.append(breach)
                    
                    print(f"\n\n❌ DAILY DD BREACH #{len(self.breaches)}!")
                    print(f"   Time: {current_time}")
                    print(f"   Daily DD: {daily_dd_pct:.2f}% >= {DAILY_DD_LIMIT_PCT}%")
                    print(f"   DD Base: ${current_day_state.daily_dd_base:,.2f}")
                    print(f"   Equity: ${current_equity:,.2f}")
                    
                    if not continue_after_breach:
                        account_terminated = True
                        termination_reason = "DAILY DD BREACH"
                        break
            
            # Next hour
            current_time += timedelta(hours=1)
        
        # Save final day state
        if current_day_state:
            self.day_states.append(current_day_state)
        
        # Calculate final P&L
        self.live_total_profit = current_balance - INITIAL_BALANCE
        
        # Print results
        print("\n" + "=" * 80)
        self.print_results(account_terminated, termination_reason, current_balance)
        
        return self.generate_report(account_terminated, current_balance)
    
    def print_results(self, terminated: bool, reason: Optional[str], final_balance: float):
        """Print comprehensive results"""
        print("📊 SIMULATION RESULTS")
        print("=" * 80)
        
        # Account status
        print(f"\n{'❌ ACCOUNT TERMINATED' if terminated else '✅ ACCOUNT SURVIVED'}")
        if reason:
            print(f"   Reason: {reason}")
        
        # P&L comparison
        print(f"\n💰 PROFIT & LOSS:")
        print(f"   Backtest (ideal):    ${self.backtest_total_profit:,.2f}")
        print(f"   Live (realistic):    ${self.live_total_profit:,.2f}")
        
        diff = self.live_total_profit - self.backtest_total_profit
        diff_pct = (diff / abs(self.backtest_total_profit)) * 100 if self.backtest_total_profit != 0 else 0
        print(f"   Difference:          ${diff:,.2f} ({diff_pct:+.1f}%)")
        
        # Trade execution
        print(f"\n📈 TRADE EXECUTION:")
        total_signals = len(self.trades)
        executed = len(self.executed_trades)
        
        print(f"   Signals:       {total_signals}")
        print(f"   Executed:      {executed}")
        print(f"   Not executed:  {total_signals - executed}")
        
        if executed > 0:
            sl_hits = [t for t in self.executed_trades if t.exit_reason == 'sl']
            tp_hits = [t for t in self.executed_trades if t.exit_reason == 'tp']
            fallback = [t for t in self.executed_trades if t.exit_reason == 'backtest_fallback']
            
            print(f"\n   Exit Breakdown:")
            print(f"     SL hits:     {len(sl_hits)} ({len(sl_hits)/executed*100:.1f}%)")
            print(f"     TP hits:     {len(tp_hits)} ({len(tp_hits)/executed*100:.1f}%)")
            print(f"     Fallback:    {len(fallback)} ({len(fallback)/executed*100:.1f}%)")
            
            # P&L stats
            winners = [t for t in self.executed_trades if t.live_profit_usd and t.live_profit_usd > 0]
            losers = [t for t in self.executed_trades if t.live_profit_usd and t.live_profit_usd < 0]
            
            if winners:
                avg_win = sum(t.live_profit_usd for t in winners) / len(winners)
                print(f"\n   Avg Winner: ${avg_win:,.2f}")
            if losers:
                avg_loss = sum(t.live_profit_usd for t in losers) / len(losers)
                print(f"   Avg Loser:  ${avg_loss:,.2f}")
            
            win_rate = len(winners) / executed * 100
            print(f"   Win Rate:   {win_rate:.1f}%")
        
        # Breach events
        if self.breaches:
            print(f"\n⚠️  BREACH EVENTS: {len(self.breaches)}")
            for breach in self.breaches:
                print(f"\n   {breach.breach_type.upper()} BREACH:")
                print(f"     Time:      {breach.timestamp}")
                print(f"     Daily DD:  {breach.daily_dd_pct:.2f}%")
                print(f"     Total DD:  {breach.total_dd_pct:.2f}%")
                print(f"     Equity:    ${breach.equity:,.2f}")
                print(f"     Open:      {breach.open_trades} trades")
        
        # Daily DD stats
        if self.day_states:
            worst_day = max(self.day_states, key=lambda d: d.max_daily_dd_pct)
            
            print(f"\n📉 DAILY DD STATISTICS:")
            print(f"   Days Analyzed:   {len(self.day_states)}")
            print(f"   Worst Daily DD:  {worst_day.max_daily_dd_pct:.2f}%")
            print(f"     Date:          {worst_day.date}")
            print(f"     DD Base:       ${worst_day.daily_dd_base:,.0f}")
            print(f"     Lowest Equity: ${worst_day.lowest_equity:,.0f}")
            
            # Days with high DD
            high_dd_days = [d for d in self.day_states if d.max_daily_dd_pct >= 3.0]
            print(f"\n   Days with DD >= 3%: {len(high_dd_days)}")
            
            if high_dd_days:
                print(f"\n   Top 10 Worst Days:")
                for i, day in enumerate(sorted(high_dd_days, 
                                              key=lambda d: d.max_daily_dd_pct, 
                                              reverse=True)[:10], 1):
                    print(f"     {i}. {day.date}: {day.max_daily_dd_pct:.2f}%")
    
    def generate_report(self, terminated: bool, final_balance: float) -> Dict:
        """Generate comprehensive report"""
        return {
            'account_terminated': terminated,
            'initial_balance': INITIAL_BALANCE,
            'final_balance': final_balance,
            'live_profit': self.live_total_profit,
            'backtest_profit': self.backtest_total_profit,
            'profit_difference': self.live_total_profit - self.backtest_total_profit,
            'signals_total': len(self.trades),
            'trades_executed': len(self.executed_trades),
            'breach_events': len(self.breaches),
            'days_analyzed': len(self.day_states),
            'worst_daily_dd_pct': max((d.max_daily_dd_pct for d in self.day_states), default=0),
        }
    
    def save_results(self):
        """Save detailed results to files"""
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        print(f"\n💾 Saving results to {OUTPUT_DIR}/")
        
        # 1. Executed trades
        if self.executed_trades:
            trades_data = []
            for t in self.executed_trades:
                trades_data.append({
                    'trade_id': t.trade_id,
                    'symbol': t.symbol,
                    'direction': t.direction,
                    'entry_time': t.entry_time,
                    'entry_price': t.entry_price,
                    'stop_loss': t.stop_loss,
                    'take_profit': t.take_profit,
                    'exit_time': t.exit_time,
                    'exit_price': t.exit_price,
                    'exit_reason': t.exit_reason,
                    'live_result_r': t.live_result_r,
                    'live_profit_usd': t.live_profit_usd,
                    'backtest_result_r': t.backtest_result_r,
                    'backtest_profit_usd': t.backtest_profit_usd,
                    'difference_usd': (t.live_profit_usd or 0) - t.backtest_profit_usd,
                })
            pd.DataFrame(trades_data).to_csv(OUTPUT_DIR / 'executed_trades.csv', index=False)
            print(f"  ✓ executed_trades.csv")
        
        # 2. Daily DD stats
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
                })
            pd.DataFrame(day_data).to_csv(OUTPUT_DIR / 'daily_dd_stats.csv', index=False)
            print(f"  ✓ daily_dd_stats.csv")
        
        # 3. Breach events
        if self.breaches:
            breach_data = []
            for b in self.breaches:
                breach_data.append({
                    'timestamp': b.timestamp,
                    'breach_type': b.breach_type,
                    'daily_dd_pct': b.daily_dd_pct,
                    'total_dd_pct': b.total_dd_pct,
                    'equity': b.equity,
                    'balance': b.balance,
                    'open_trades': b.open_trades,
                    'floating_pnl': b.floating_pnl,
                })
            pd.DataFrame(breach_data).to_csv(OUTPUT_DIR / 'breaches.csv', index=False)
            print(f"  ✓ breaches.csv")
        
        # 4. Summary JSON
        report = self.generate_report(
            len(self.breaches) > 0,
            INITIAL_BALANCE + self.live_total_profit
        )
        with open(OUTPUT_DIR / 'summary.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"  ✓ summary.json")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='5ers Realistic Live Trading Simulation'
    )
    parser.add_argument(
        '--count-all-breaches', '-c',
        action='store_true',
        help='Continue after breaches and count all breaches over full period'
    )
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("5ERS REALISTIC LIVE TRADING SIMULATION")
    print("With proper SL/TP detection and realistic costs")
    if args.count_all_breaches:
        print("MODE: Count ALL breaches (no account reset)")
    print("=" * 80 + "\n")
    
    # Check files exist
    if not TRADES_FILE.exists():
        print(f"❌ Trades file not found: {TRADES_FILE}")
        print(f"   Please ensure file exists")
        return
    
    if not H1_DATA_DIR.exists():
        print(f"⚠️  WARNING: H1 data directory not found: {H1_DATA_DIR}")
        print(f"   Simulation will use backtest exits (less accurate)")
    
    # Run simulation
    simulator = Realistic5ersSimulator(TRADES_FILE)
    report = simulator.run_simulation(continue_after_breach=args.count_all_breaches)
    
    # Save results
    simulator.save_results()
    
    print("\n" + "=" * 80)
    print("✅ SIMULATION COMPLETE")
    print("=" * 80)
    
    # Final summary
    print(f"\n📊 FINAL VERDICT:")
    if report['account_terminated']:
        print(f"   ❌ Account would have FAILED 5ers challenge")
        print(f"   Breach events: {report['breach_events']}")
    else:
        print(f"   ✅ Account SURVIVED 3-year period!")
        print(f"   No breaches detected")
    
    print(f"\n   Worst daily DD: {report['worst_daily_dd_pct']:.2f}%")
    print(f"   Days analyzed: {report['days_analyzed']}")
    print(f"   Live profit: ${report['live_profit']:,.2f}")


if __name__ == "__main__":
    main()
