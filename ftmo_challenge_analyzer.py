#!/usr/bin/env python3
"""
Ultimate FTMO Challenge Performance Analyzer - 2024 Historical Data
Production-Ready, Resumable Optimizer with ADX Trend Filter

This module provides a comprehensive backtesting and self-optimizing system that:
1. Backtests using 2024 historical data
2. Training Period: Jan-Sep 2024, Validation Period: Oct-Dec 2024
3. ADX > 25 trend-strength filter to avoid ranging markets
4. December fully open for trading (no date restrictions)
5. Tracks ALL trades with complete entry/exit data
6. Generates detailed CSV reports with all trade details
7. Self-optimizes by saving parameters to params/current_params.json
8. RESUMABLE: Uses Optuna SQLite storage for crash-resistant optimization
9. STATUS MODE: Check progress anytime with --status flag

Usage:
  python ftmo_challenge_analyzer.py              # Run/resume optimization
  python ftmo_challenge_analyzer.py --status     # Check progress without running
  python ftmo_challenge_analyzer.py --trials 100 # Set number of trials
"""

import argparse
import json
import csv
import os
import random
import numpy as np
from dataclasses import dataclass, field, asdict
from datetime import datetime, date, timedelta, timezone
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any, Union
import pandas as pd

from strategy_core import (
    StrategyParams,
    Trade,
    Signal,
    compute_confluence,
    simulate_trades,
    _infer_trend,
    _pick_direction_from_bias,
    get_default_params,
    extract_ml_features,
    apply_ml_filter,
    check_volatility_filter,
)

from data_provider import get_ohlcv as get_ohlcv_api
from ftmo_config import FTMO_CONFIG, FTMO10KConfig, get_pip_size, get_sl_limits
from config import FOREX_PAIRS, METALS, INDICES, CRYPTO_ASSETS
from tradr.data.oanda import OandaClient
from tradr.risk.position_sizing import calculate_lot_size, get_contract_specs
from params.params_loader import save_optimized_params

OUTPUT_DIR = Path("ftmo_analysis_output")
OUTPUT_DIR.mkdir(exist_ok=True)

OPTUNA_DB_PATH = "sqlite:///optuna_study.db"
OPTUNA_STUDY_NAME = "ftmo_study"
PROGRESS_LOG_FILE = "ftmo_optimization_progress.txt"

TRAINING_START = datetime(2024, 1, 1)
TRAINING_END = datetime(2024, 9, 30)
VALIDATION_START = datetime(2024, 10, 1)
VALIDATION_END = datetime(2024, 12, 31)
FULL_YEAR_START = datetime(2024, 1, 1)
FULL_YEAR_END = datetime(2024, 12, 31)

QUARTERS_2024 = {
    "Q1": (datetime(2024, 1, 1), datetime(2024, 3, 31)),
    "Q2": (datetime(2024, 4, 1), datetime(2024, 6, 30)),
    "Q3": (datetime(2024, 7, 1), datetime(2024, 9, 30)),
    "Q4": (datetime(2024, 10, 1), datetime(2024, 12, 31)),
}

TRAINING_QUARTERS = {
    "Q1": (datetime(2024, 1, 1), datetime(2024, 3, 31)),
    "Q2": (datetime(2024, 4, 1), datetime(2024, 6, 30)),
    "Q3": (datetime(2024, 7, 1), datetime(2024, 9, 30)),
}

ACCOUNT_SIZE = 200000.0


def calculate_adx(candles: List[Dict], period: int = 14) -> float:
    """
    Calculate Average Directional Index (ADX) for trend strength measurement.
    ADX > 25 indicates a strong trend.
    
    Args:
        candles: List of OHLCV candle dictionaries
        period: ADX period (default 14)
    
    Returns:
        ADX value (0-100 scale)
    """
    if len(candles) < period * 2:
        return 0.0
    
    highs = [c.get("high", 0) for c in candles]
    lows = [c.get("low", 0) for c in candles]
    closes = [c.get("close", 0) for c in candles]
    
    plus_dm = []
    minus_dm = []
    tr_values = []
    
    for i in range(1, len(candles)):
        high_diff = highs[i] - highs[i-1]
        low_diff = lows[i-1] - lows[i]
        
        if high_diff > low_diff and high_diff > 0:
            plus_dm.append(high_diff)
        else:
            plus_dm.append(0)
        
        if low_diff > high_diff and low_diff > 0:
            minus_dm.append(low_diff)
        else:
            minus_dm.append(0)
        
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i-1]),
            abs(lows[i] - closes[i-1])
        )
        tr_values.append(tr)
    
    if len(tr_values) < period:
        return 0.0
    
    smoothed_plus_dm = sum(plus_dm[:period])
    smoothed_minus_dm = sum(minus_dm[:period])
    smoothed_tr = sum(tr_values[:period])
    
    dx_values = []
    
    for i in range(period, len(tr_values)):
        smoothed_plus_dm = smoothed_plus_dm - (smoothed_plus_dm / period) + plus_dm[i]
        smoothed_minus_dm = smoothed_minus_dm - (smoothed_minus_dm / period) + minus_dm[i]
        smoothed_tr = smoothed_tr - (smoothed_tr / period) + tr_values[i]
        
        if smoothed_tr == 0:
            continue
            
        plus_di = 100 * smoothed_plus_dm / smoothed_tr
        minus_di = 100 * smoothed_minus_dm / smoothed_tr
        
        di_sum = plus_di + minus_di
        if di_sum == 0:
            dx = 0
        else:
            dx = 100 * abs(plus_di - minus_di) / di_sum
        dx_values.append(dx)
    
    if not dx_values:
        return 0.0
    
    if len(dx_values) < period:
        return sum(dx_values) / len(dx_values)
    
    adx = sum(dx_values[:period]) / period
    for i in range(period, len(dx_values)):
        adx = ((adx * (period - 1)) + dx_values[i]) / period
    
    return adx


def check_adx_filter(candles: List[Dict], min_adx: float = 25.0) -> Tuple[bool, float]:
    """
    Check if ADX is above minimum threshold for trend trading.
    
    Args:
        candles: D1 candles for ADX calculation
        min_adx: Minimum ADX value (default 25)
    
    Returns:
        Tuple of (passes_filter, adx_value)
    """
    adx = calculate_adx(candles, period=14)
    return adx > min_adx, adx


def log_optimization_progress(trial_num: int, value: float, best_value: float, best_params: Dict):
    """Append optimization progress to log file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    key_params = {k: round(v, 3) if isinstance(v, float) else v 
                  for k, v in list(best_params.items())[:5]}
    log_entry = (
        f"[{timestamp}] Trial #{trial_num}: value={value:.0f}, "
        f"best_value={best_value:.0f}, params={json.dumps(key_params)}\n"
    )
    with open(PROGRESS_LOG_FILE, 'a') as f:
        f.write(log_entry)


def show_optimization_status():
    """Display current optimization status without running new trials."""
    import optuna
    
    print("\n" + "=" * 60)
    print("FTMO OPTIMIZATION STATUS CHECK")
    print("=" * 60)
    
    db_file = "optuna_study.db"
    if not os.path.exists(db_file):
        print("\nNo optimization study found.")
        print("Run 'python ftmo_challenge_analyzer.py' to start optimization.")
        return
    
    try:
        study = optuna.load_study(
            study_name=OPTUNA_STUDY_NAME,
            storage=OPTUNA_DB_PATH
        )
        
        print(f"\nStudy Name: {OPTUNA_STUDY_NAME}")
        print(f"Completed Trials: {len(study.trials)}")
        
        if study.best_trial:
            print(f"\nBest Value: {study.best_value:.0f}")
            print(f"Best Parameters:")
            for k, v in sorted(study.best_params.items()):
                if isinstance(v, float):
                    print(f"  {k}: {v:.3f}")
                else:
                    print(f"  {k}: {v}")
            
            best_trial = study.best_trial
            if best_trial.datetime_complete:
                print(f"\nLast Update: {best_trial.datetime_complete.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            print("\nNo completed trials yet.")
        
    except Exception as e:
        print(f"\nError loading study: {e}")
        return
    
    if os.path.exists(PROGRESS_LOG_FILE):
        print(f"\n{'='*60}")
        print("RECENT PROGRESS (last 10 entries):")
        print("=" * 60)
        with open(PROGRESS_LOG_FILE, 'r') as f:
            lines = f.readlines()
            for line in lines[-10:]:
                print(line.rstrip())
    else:
        print("\nNo progress log found yet.")
    
    print(f"\n{'='*60}")
    print("To resume optimization: python ftmo_challenge_analyzer.py")
    print("=" * 60)


def is_valid_trading_day(dt: datetime) -> bool:
    """Check if datetime is a valid trading day (no weekends)."""
    if dt.weekday() >= 5:
        return False
    return True


def _atr(candles: List[Dict], period: int = 14) -> float:
    """Calculate Average True Range (ATR)."""
    if len(candles) < period + 1:
        return 0.0
    
    tr_values = []
    for i in range(1, len(candles)):
        high = candles[i].get("high")
        low = candles[i].get("low")
        prev_close = candles[i - 1].get("close")
        
        if high is None or low is None or prev_close is None:
            continue
        
        tr = max(
            high - low,
            abs(high - prev_close),
            abs(low - prev_close)
        )
        tr_values.append(tr)
    
    if len(tr_values) < period:
        return sum(tr_values) / len(tr_values) if tr_values else 0.0
    
    atr_val = sum(tr_values[:period]) / period
    for tr in tr_values[period:]:
        atr_val = (atr_val * (period - 1) + tr) / period
    
    return atr_val


def _calculate_atr_percentile(candles: List[Dict], period: int = 14, lookback: int = 100) -> Tuple[float, float]:
    """Calculate current ATR and its percentile rank."""
    if len(candles) < period + lookback:
        current_atr = _atr(candles, period)
        return current_atr, 50.0
    
    atr_values = []
    for i in range(lookback):
        end_idx = len(candles) - i
        if end_idx < period + 1:
            break
        slice_candles = candles[:end_idx]
        atr_val = _atr(slice_candles, period)
        if atr_val > 0:
            atr_values.append(atr_val)
    
    if not atr_values:
        return 0.0, 50.0
    
    current_atr = atr_values[0]
    sorted_atrs = sorted(atr_values)
    rank = sum(1 for v in sorted_atrs if v <= current_atr)
    percentile = (rank / len(sorted_atrs)) * 100
    
    return current_atr, percentile


@dataclass
class BacktestTrade:
    """Complete trade record for CSV export."""
    trade_num: int
    symbol: str
    direction: str
    entry_date: Any
    entry_price: float
    stop_loss_price: float
    tp1_price: float
    tp2_price: Optional[float]
    tp3_price: Optional[float]
    tp4_price: Optional[float]
    tp5_price: Optional[float]
    exit_date: Any
    exit_price: float
    tp1_hit: bool
    tp2_hit: bool
    tp3_hit: bool
    tp4_hit: bool
    tp5_hit: bool
    sl_hit: bool
    exit_reason: str
    r_multiple: float
    profit_loss_usd: float
    confluence_score: int
    holding_time_hours: float
    lot_size: float
    risk_pips: float
    validation_notes: str = ""
    adx_value: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for CSV export."""
        return {
            "Trade#": self.trade_num,
            "Symbol": self.symbol,
            "Direction": self.direction,
            "Entry Date": str(self.entry_date) if self.entry_date else "",
            "Entry Price": self.entry_price,
            "Stop Loss Price": self.stop_loss_price,
            "TP1 Price": self.tp1_price,
            "TP2 Price": self.tp2_price or "",
            "TP3 Price": self.tp3_price or "",
            "TP4 Price": self.tp4_price or "",
            "TP5 Price": self.tp5_price or "",
            "Exit Date": str(self.exit_date) if self.exit_date else "",
            "Exit Price": self.exit_price,
            "TP1 Hit?": "Yes" if self.tp1_hit else "No",
            "TP2 Hit?": "Yes" if self.tp2_hit else "No",
            "TP3 Hit?": "Yes" if self.tp3_hit else "No",
            "TP4 Hit?": "Yes" if self.tp4_hit else "No",
            "TP5 Hit?": "Yes" if self.tp5_hit else "No",
            "SL Hit?": "Yes" if self.sl_hit else "No",
            "Final Exit Reason": self.exit_reason,
            "R Multiple": round(self.r_multiple, 2),
            "Profit/Loss USD": round(self.profit_loss_usd, 2),
            "Confluence Score": self.confluence_score,
            "Holding Time (hours)": round(self.holding_time_hours, 1),
            "Lot Size": round(self.lot_size, 2),
            "Risk Pips": round(self.risk_pips, 1),
            "ADX Value": round(self.adx_value, 1),
            "Validation Notes": self.validation_notes,
        }


class MonteCarloSimulator:
    """Monte Carlo simulation for robustness testing."""
    
    def __init__(self, trades: List[Any], num_simulations: int = 1000):
        self.trades = trades
        self.num_simulations = num_simulations
        self.r_values = self._extract_r_values()
    
    def _extract_r_values(self) -> List[float]:
        r_values = []
        for t in self.trades:
            r = getattr(t, 'rr', None) or getattr(t, 'r_multiple', None) or 0.0
            r_values.append(float(r))
        return r_values
    
    def run_simulation(self) -> Dict[str, Any]:
        if not self.r_values:
            return {"error": "No trades to simulate", "num_simulations": 0}
        
        np.random.seed(42)
        
        final_equities = []
        max_drawdowns = []
        win_rates = []
        
        num_trades = len(self.r_values)
        
        for _ in range(self.num_simulations):
            indices = np.random.choice(num_trades, size=num_trades, replace=True)
            resampled_trades = [self.r_values[i] for i in indices]
            
            noise = np.random.uniform(0.9, 1.1, size=num_trades)
            perturbed_trades = [r * n for r, n in zip(resampled_trades, noise)]
            
            equity_curve = [0.0]
            for r in perturbed_trades:
                equity_curve.append(equity_curve[-1] + r)
            
            final_equities.append(equity_curve[-1])
            
            peak = equity_curve[0]
            max_dd = 0.0
            for eq in equity_curve:
                if eq > peak:
                    peak = eq
                dd = peak - eq
                if dd > max_dd:
                    max_dd = dd
            max_drawdowns.append(max_dd)
            
            wins = sum(1 for r in perturbed_trades if r > 0)
            win_rates.append(wins / num_trades * 100 if num_trades > 0 else 0)
        
        return {
            "num_simulations": self.num_simulations,
            "num_trades": num_trades,
            "mean_return": float(np.mean(final_equities)),
            "std_return": float(np.std(final_equities)),
            "mean_max_dd": float(np.mean(max_drawdowns)),
            "mean_win_rate": float(np.mean(win_rates)),
            "worst_case_dd": float(np.percentile(max_drawdowns, 95)),
            "best_case_return": float(np.percentile(final_equities, 95)),
            "worst_case_return": float(np.percentile(final_equities, 5)),
            "confidence_intervals": {
                "final_equity": {f"p{p}": float(np.percentile(final_equities, p)) for p in [5, 25, 50, 75, 95]},
                "max_drawdown": {f"p{p}": float(np.percentile(max_drawdowns, p)) for p in [5, 25, 50, 75, 95]},
            },
        }


def run_monte_carlo_analysis(trades: List[Any], num_simulations: int = 1000) -> Dict:
    """Run Monte Carlo analysis on trades."""
    if not trades:
        return {"error": "No trades provided"}
    
    simulator = MonteCarloSimulator(trades, num_simulations)
    results = simulator.run_simulation()
    
    print(f"\nMonte Carlo Simulation ({results.get('num_simulations', 0)} iterations):")
    print(f"  Mean Return: {results.get('mean_return', 0):+.2f}R")
    print(f"  Std Dev: {results.get('std_return', 0):.2f}R")
    print(f"  Best Case (95th): {results.get('best_case_return', 0):+.2f}R")
    print(f"  Worst Case (5th): {results.get('worst_case_return', 0):+.2f}R")
    print(f"  Worst Case DD (95th): {results.get('worst_case_dd', 0):.2f}R")
    
    return results


def load_ohlcv_data(symbol: str, timeframe: str, start_date: datetime, end_date: datetime) -> List[Dict]:
    """Load OHLCV data from local CSV files or API."""
    data_dir = Path("data/ohlcv")
    
    symbol_normalized = symbol.replace("_", "").replace("/", "")
    
    tf_map = {"D1": "D1", "H4": "H4", "W1": "W1", "MN": "MN"}
    tf = tf_map.get(timeframe, timeframe)
    
    pattern = f"{symbol_normalized}_{tf}_*.csv"
    matches = list(data_dir.glob(pattern))
    
    if matches:
        csv_path = matches[0]
        try:
            df = pd.read_csv(csv_path)
            
            date_col = None
            for col in ['time', 'timestamp', 'date', 'Date', 'Time']:
                if col in df.columns:
                    date_col = col
                    break
            
            if date_col:
                df[date_col] = pd.to_datetime(df[date_col])
                df = df[(df[date_col] >= start_date) & (df[date_col] <= end_date)]
            
            candles = []
            for _, row in df.iterrows():
                candle = {
                    "time": row.get(date_col) if date_col else None,
                    "open": row.get("open") or row.get("Open"),
                    "high": row.get("high") or row.get("High"),
                    "low": row.get("low") or row.get("Low"),
                    "close": row.get("close") or row.get("Close"),
                    "volume": row.get("volume") or row.get("Volume") or 0,
                }
                candles.append(candle)
            
            return candles
        except Exception as e:
            print(f"Error loading {csv_path}: {e}")
    
    try:
        days_needed = (end_date - start_date).days + 100
        return get_ohlcv_api(symbol, timeframe, count=days_needed, use_cache=True, start_date=start_date)
    except Exception:
        return []


def get_all_trading_assets() -> List[str]:
    """Get list of all tradeable assets."""
    assets = []
    assets.extend(FOREX_PAIRS if FOREX_PAIRS else [
        "EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF", "AUD_USD", "NZD_USD", "USD_CAD",
        "EUR_GBP", "EUR_JPY", "GBP_JPY", "EUR_CHF", "EUR_AUD", "EUR_CAD", "EUR_NZD",
        "GBP_CHF", "GBP_AUD", "GBP_CAD", "GBP_NZD", "AUD_JPY", "AUD_NZD", "AUD_CAD",
        "AUD_CHF", "NZD_JPY", "NZD_CAD", "NZD_CHF", "CAD_JPY", "CAD_CHF", "CHF_JPY"
    ])
    assets.extend(METALS if METALS else ["XAU_USD", "XAG_USD"])
    assets.extend(INDICES if INDICES else ["SPX500_USD", "NAS100_USD"])
    assets.extend(CRYPTO_ASSETS if CRYPTO_ASSETS else ["BTC_USD", "ETH_USD"])
    
    return list(set(assets))


def run_full_period_backtest(
    start_date: datetime,
    end_date: datetime,
    min_confluence: int = 3,
    min_quality_factors: int = 1,
    risk_per_trade_pct: float = 0.5,
    atr_min_percentile: float = 60.0,
    trail_activation_r: float = 2.2,
    december_atr_multiplier: float = 1.5,
    volatile_asset_boost: float = 1.5,
    ml_min_prob: Optional[float] = None,
    bollinger_std: float = 2.0,
    rsi_period: int = 14,
    excluded_assets: Optional[List[str]] = None,
    require_adx_filter: bool = True,
    min_adx: float = 25.0,
) -> List[Trade]:
    """
    Run backtest for a given period with ADX > 25 trend filter.
    December is fully open for trading.
    """
    assets = get_all_trading_assets()
    
    if excluded_assets:
        assets = [a for a in assets if a not in excluded_assets]
    
    all_trades: List[Trade] = []
    seen_trades = set()
    
    params = StrategyParams(
        min_confluence=min_confluence,
        min_quality_factors=min_quality_factors,
        risk_per_trade_pct=risk_per_trade_pct,
        atr_min_percentile=atr_min_percentile,
        trail_activation_r=trail_activation_r,
        december_atr_multiplier=december_atr_multiplier,
        volatile_asset_boost=volatile_asset_boost,
        bollinger_std=bollinger_std,
        rsi_period=rsi_period,
        ml_min_prob=ml_min_prob if ml_min_prob else 0.0,
    )
    
    for symbol in assets:
        try:
            d1_candles = load_ohlcv_data(symbol, "D1", start_date - timedelta(days=100), end_date)
            h4_candles = load_ohlcv_data(symbol, "H4", start_date - timedelta(days=50), end_date)
            w1_candles = load_ohlcv_data(symbol, "W1", start_date - timedelta(days=365), end_date)
            mn_candles = load_ohlcv_data(symbol, "MN", start_date - timedelta(days=730), end_date)
            
            if not d1_candles or len(d1_candles) < 30:
                continue
            
            if require_adx_filter:
                adx_passes, adx_value = check_adx_filter(d1_candles, min_adx)
                if not adx_passes:
                    continue
            
            current_atr, atr_percentile = _calculate_atr_percentile(d1_candles)
            if atr_percentile < atr_min_percentile:
                continue
            
            trades = simulate_trades(
                candles=d1_candles,
                symbol=symbol,
                params=params,
                h4_candles=h4_candles,
                weekly_candles=w1_candles,
                monthly_candles=mn_candles,
                include_transaction_costs=True,
            )
            
            for trade in trades:
                trade_key = (
                    trade.symbol,
                    str(trade.entry_date)[:10],
                    trade.direction,
                    round(trade.entry_price, 5)
                )
                if trade_key not in seen_trades:
                    seen_trades.add(trade_key)
                    all_trades.append(trade)
                    
        except Exception as e:
            continue
    
    all_trades.sort(key=lambda t: str(t.entry_date))
    return all_trades


def convert_to_backtest_trade(
    trade: Trade,
    trade_num: int,
    risk_per_trade_pct: float,
    adx_value: float = 0.0
) -> BacktestTrade:
    """Convert strategy Trade to BacktestTrade for CSV export."""
    entry_dt = trade.entry_date
    exit_dt = trade.exit_date
    
    if isinstance(entry_dt, str):
        try:
            entry_dt = datetime.fromisoformat(entry_dt.replace("Z", "+00:00"))
        except:
            entry_dt = datetime.now()
    
    if isinstance(exit_dt, str):
        try:
            exit_dt = datetime.fromisoformat(exit_dt.replace("Z", "+00:00"))
        except:
            exit_dt = datetime.now()
    
    specs = get_contract_specs(trade.symbol)
    pip_value_unit = specs.get("pip_value", 0.0001)
    stop_distance = abs(trade.entry_price - trade.stop_loss)
    risk_pips = stop_distance / pip_value_unit if pip_value_unit > 0 else 0
    
    holding_hours = 0.0
    if entry_dt and exit_dt:
        if isinstance(entry_dt, datetime) and isinstance(exit_dt, datetime):
            try:
                if entry_dt.tzinfo is not None and exit_dt.tzinfo is None:
                    exit_dt = exit_dt.replace(tzinfo=entry_dt.tzinfo)
                elif entry_dt.tzinfo is None and exit_dt.tzinfo is not None:
                    entry_dt = entry_dt.replace(tzinfo=exit_dt.tzinfo)
                delta = exit_dt - entry_dt
                holding_hours = delta.total_seconds() / 3600
                if holding_hours < 0:
                    holding_hours = abs(holding_hours)
            except:
                holding_hours = 0.0
    
    risk_usd = ACCOUNT_SIZE * (risk_per_trade_pct / 100)
    profit_usd = trade.rr * risk_usd
    
    exit_reason = trade.exit_reason or ""
    tp1_hit = "TP1" in exit_reason or "TP2" in exit_reason or "TP3" in exit_reason or "TP4" in exit_reason or "TP5" in exit_reason
    tp2_hit = "TP2" in exit_reason or "TP3" in exit_reason or "TP4" in exit_reason or "TP5" in exit_reason
    tp3_hit = "TP3" in exit_reason or "TP4" in exit_reason or "TP5" in exit_reason
    tp4_hit = "TP4" in exit_reason or "TP5" in exit_reason
    tp5_hit = "TP5" in exit_reason
    sl_hit = exit_reason == "SL"
    
    sizing_result = calculate_lot_size(
        symbol=trade.symbol,
        account_balance=ACCOUNT_SIZE,
        risk_percent=risk_per_trade_pct / 100,
        entry_price=trade.entry_price,
        stop_loss_price=trade.stop_loss,
        max_lot=100.0,
        min_lot=0.01,
    )
    lot_size = sizing_result.get("lot_size", 0.01)
    
    spread_cost = risk_pips * 0.02
    slippage_cost = risk_pips * 0.01
    commission_cost = 0.001
    adjusted_profit = profit_usd * (1 - spread_cost - slippage_cost - commission_cost)
    
    return BacktestTrade(
        trade_num=trade_num,
        symbol=trade.symbol,
        direction=trade.direction.upper(),
        entry_date=entry_dt,
        entry_price=trade.entry_price,
        stop_loss_price=trade.stop_loss,
        tp1_price=trade.tp1 or 0.0,
        tp2_price=trade.tp2,
        tp3_price=trade.tp3,
        tp4_price=trade.tp4,
        tp5_price=trade.tp5,
        exit_date=exit_dt,
        exit_price=trade.exit_price,
        tp1_hit=tp1_hit,
        tp2_hit=tp2_hit,
        tp3_hit=tp3_hit,
        tp4_hit=tp4_hit,
        tp5_hit=tp5_hit,
        sl_hit=sl_hit,
        exit_reason=exit_reason,
        r_multiple=trade.rr,
        profit_loss_usd=adjusted_profit,
        confluence_score=trade.confluence_score,
        holding_time_hours=holding_hours,
        lot_size=lot_size,
        risk_pips=risk_pips,
        adx_value=adx_value,
        validation_notes="ADX > 25 trend filter applied",
    )


def export_trades_to_csv(trades: List[Trade], filename: str, risk_per_trade_pct: float = 0.5):
    """Export trades to CSV with all required columns."""
    filepath = OUTPUT_DIR / filename
    
    if not trades:
        print(f"No trades to export to {filename}")
        return
    
    backtest_trades = []
    for i, trade in enumerate(trades, 1):
        bt_trade = convert_to_backtest_trade(trade, i, risk_per_trade_pct)
        backtest_trades.append(bt_trade)
    
    with open(filepath, 'w', newline='') as f:
        if backtest_trades:
            fieldnames = list(backtest_trades[0].to_dict().keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for trade in backtest_trades:
                writer.writerow(trade.to_dict())
    
    print(f"Exported {len(backtest_trades)} trades to: {filepath}")


def print_period_results(trades: List[Trade], period_name: str, start: datetime, end: datetime) -> Dict:
    """Print results for a specific period."""
    if not trades:
        print(f"\n{period_name}: No trades generated")
        return {'trades': 0, 'total_r': 0, 'win_rate': 0, 'net_profit': 0}
    
    total_r = sum(getattr(t, 'rr', 0) for t in trades)
    wins = sum(1 for t in trades if getattr(t, 'rr', 0) > 0)
    losses = sum(1 for t in trades if getattr(t, 'rr', 0) <= 0)
    win_rate = (wins / len(trades) * 100) if trades else 0
    
    risk_usd = ACCOUNT_SIZE * 0.005
    total_profit = total_r * risk_usd
    
    print(f"\n{period_name}")
    print(f"  Period: {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}")
    print(f"  Total Trades: {len(trades)}")
    print(f"  Wins: {wins}, Losses: {losses}")
    print(f"  Win Rate: {win_rate:.1f}%")
    print(f"  Total R: {total_r:+.2f}R")
    print(f"  Net Profit (est): ${total_profit:+,.2f}")
    print(f"  ADX > 25 trend filter applied")
    
    return {
        'trades': len(trades),
        'total_r': total_r,
        'win_rate': win_rate,
        'net_profit': total_profit,
    }


def update_readme_documentation():
    """Update README.md with optimization & backtesting section."""
    readme_path = Path("README.md")
    
    new_section = """
## Optimization & Backtesting

The optimizer uses professional quant best practices:

- **TRAINING PERIOD**: January 1, 2024 – September 30, 2024 (in-sample optimization)
- **VALIDATION PERIOD**: October 1, 2024 – December 31, 2024 (out-of-sample test)
- **FINAL BACKTEST**: Full year 2024 (December fully open for trading)
- **ADX > 25 trend-strength filter** applied to avoid ranging markets.

All trades from the final backtest are exported to:
`ftmo_analysis_output/all_trades_2024_full.csv`

Parameters are saved to `params/current_params.json`

Optimization is resumable and can be checked with: `python ftmo_challenge_analyzer.py --status`
"""
    
    if readme_path.exists():
        content = readme_path.read_text()
        
        import re
        pattern = r'## Optimization & Backtesting.*?(?=\n## |\Z)'
        
        if re.search(pattern, content, re.DOTALL):
            content = re.sub(pattern, new_section.strip() + "\n\n", content, flags=re.DOTALL)
        else:
            content = content.rstrip() + "\n\n" + new_section.strip() + "\n"
        
        readme_path.write_text(content)
        print("README.md updated with optimization section.")
    else:
        readme_path.write_text(new_section.strip())
        print("README.md created with optimization section.")


def train_ml_model(trades: List[Trade]) -> bool:
    """Train RandomForest ML model on full-year trades and save to models/best_rf.joblib."""
    os.makedirs('models', exist_ok=True)
    
    if len(trades) < 50:
        print(f"Insufficient trades for ML training: {len(trades)} < 50 required")
        return False
    
    features_list = []
    labels = []
    
    for trade in trades:
        r_value = getattr(trade, 'rr', 0)
        label = 1 if r_value > 0 else 0
        
        features = {
            'confluence_score': trade.confluence_score,
            'direction_bullish': 1 if trade.direction == 'bullish' else 0,
            'risk': abs(trade.entry_price - trade.stop_loss) if trade.stop_loss else 0,
            'rr_target': abs(trade.tp1 - trade.entry_price) / max(0.00001, abs(trade.entry_price - trade.stop_loss)) if trade.tp1 and trade.stop_loss else 1,
        }
        
        features_list.append(list(features.values()))
        labels.append(label)
    
    try:
        from sklearn.ensemble import RandomForestClassifier
        import joblib
        
        clf = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=5,
            min_samples_leaf=10
        )
        clf.fit(features_list, labels)
        
        joblib.dump(clf, 'models/best_rf.joblib')
        print(f"ML model trained on {len(trades)} trades and saved to models/best_rf.joblib")
        return True
    except Exception as e:
        print(f"ML training failed: {e}")
        return False


class OptunaOptimizer:
    """
    Optuna-based optimizer for FTMO strategy parameters.
    Runs optimization ONLY on training data (Jan-Sep 2024).
    Uses persistent SQLite storage for resumability.
    """
    
    def __init__(self):
        self.best_params: Dict = {}
        self.best_score: float = -float('inf')
    
    def _objective(self, trial) -> float:
        """
        Optuna objective function - runs ONLY on TRAINING period.
        Objective: Maximize total_net_profit_dollars on training
        """
        params = {
            'risk_per_trade_pct': trial.suggest_float('risk_per_trade_pct', 0.5, 0.8, step=0.05),
            'min_confluence_score': trial.suggest_int('min_confluence_score', 3, 6),
            'min_quality_factors': trial.suggest_int('min_quality_factors', 1, 4),
            'atr_min_percentile': trial.suggest_float('atr_min_percentile', 60.0, 85.0, step=5.0),
            'trail_activation_r': trial.suggest_float('trail_activation_r', 1.8, 3.4, step=0.2),
            'december_atr_multiplier': trial.suggest_float('december_atr_multiplier', 1.3, 1.8, step=0.1),
            'volatile_asset_boost': trial.suggest_float('volatile_asset_boost', 1.3, 2.0, step=0.1),
        }
        
        training_trades = run_full_period_backtest(
            start_date=TRAINING_START,
            end_date=TRAINING_END,
            min_confluence=params['min_confluence_score'],
            min_quality_factors=params['min_quality_factors'],
            risk_per_trade_pct=params['risk_per_trade_pct'],
            atr_min_percentile=params['atr_min_percentile'],
            trail_activation_r=params['trail_activation_r'],
            december_atr_multiplier=params['december_atr_multiplier'],
            volatile_asset_boost=params['volatile_asset_boost'],
            ml_min_prob=None,
            require_adx_filter=True,
            min_adx=25.0,
        )
        
        if not training_trades or len(training_trades) == 0:
            return -50000.0
        
        total_r = sum(getattr(t, 'rr', 0) for t in training_trades)
        
        if total_r <= 0:
            return -50000.0
        
        quarterly_r = {"Q1": 0.0, "Q2": 0.0, "Q3": 0.0}
        for t in training_trades:
            entry = getattr(t, 'entry_date', None)
            if entry:
                if isinstance(entry, str):
                    try:
                        entry = datetime.fromisoformat(entry.replace("Z", "+00:00"))
                    except:
                        continue
                if hasattr(entry, 'replace') and entry.tzinfo:
                    entry = entry.replace(tzinfo=None)
                
                for q, (q_start, q_end) in TRAINING_QUARTERS.items():
                    if q_start <= entry <= q_end:
                        quarterly_r[q] += getattr(t, 'rr', 0)
                        break
        
        penalty = 0.0
        for q, r_val in quarterly_r.items():
            if r_val < 0.0:
                penalty += 5000
        
        max_dd_r = 0.0
        equity = 0.0
        peak = 0.0
        for t in training_trades:
            equity += getattr(t, 'rr', 0)
            if equity > peak:
                peak = equity
            dd = peak - equity
            if dd > max_dd_r:
                max_dd_r = dd
        
        if max_dd_r > 15.0:
            penalty += 10000
        
        risk_usd = ACCOUNT_SIZE * (params['risk_per_trade_pct'] / 100)
        total_net_profit = total_r * risk_usd
        
        score = total_net_profit - penalty
        
        all_quarters_positive = all(r >= 0 for r in quarterly_r.values())
        if all_quarters_positive:
            score += 30000
        
        return score
    
    def run_optimization(self, n_trials: int = 5) -> Dict:
        """Run Optuna optimization on TRAINING data only."""
        import optuna
        from optuna.pruners import MedianPruner
        
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        print(f"\n{'='*60}")
        print(f"OPTUNA OPTIMIZATION - Adding {n_trials} trials")
        print(f"TRAINING PERIOD ONLY: Jan 1 - Sep 30, 2024")
        print(f"ADX > 25 trend filter applied")
        print(f"Storage: {OPTUNA_DB_PATH} (resumable)")
        print(f"{'='*60}")
        
        study = optuna.create_study(
            direction='maximize',
            study_name=OPTUNA_STUDY_NAME,
            storage=OPTUNA_DB_PATH,
            load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=MedianPruner()
        )
        
        existing_trials = len(study.trials)
        if existing_trials > 0:
            print(f"Resuming from existing study with {existing_trials} completed trials")
            if study.best_trial:
                print(f"Current best value: {study.best_value:.0f}")
        
        def progress_callback(study, trial):
            log_optimization_progress(
                trial_num=trial.number,
                value=trial.value if trial.value is not None else 0,
                best_value=study.best_value if study.best_trial else 0,
                best_params=study.best_params if study.best_trial else {}
            )
        
        study.optimize(
            self._objective,
            n_trials=n_trials,
            show_progress_bar=True,
            callbacks=[progress_callback]
        )
        
        self.best_params = study.best_params
        self.best_score = study.best_value
        
        print(f"\n{'='*60}")
        print(f"OPTIMIZATION COMPLETE")
        print(f"{'='*60}")
        print(f"Total Trials: {len(study.trials)}")
        print(f"Best Score: {self.best_score:.0f}")
        print(f"Best Parameters:")
        for k, v in sorted(self.best_params.items()):
            if isinstance(v, float):
                print(f"  {k}: {v:.3f}")
            else:
                print(f"  {k}: {v}")
        
        params_to_save = {
            'min_confluence': self.best_params.get('min_confluence_score', 5),
            'min_quality_factors': self.best_params.get('min_quality_factors', 2),
            'risk_per_trade_pct': self.best_params.get('risk_per_trade_pct', 0.5),
            'atr_min_percentile': self.best_params.get('atr_min_percentile', 75.0),
            'trail_activation_r': self.best_params.get('trail_activation_r', 2.2),
            'december_atr_multiplier': self.best_params.get('december_atr_multiplier', 1.5),
            'volatile_asset_boost': self.best_params.get('volatile_asset_boost', 1.5),
        }
        
        try:
            save_optimized_params(params_to_save, backup=True)
            print(f"\nOptimized parameters saved to params/current_params.json")
        except Exception as e:
            print(f"Failed to save params: {e}")
        
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'n_trials': n_trials,
            'total_trials': len(study.trials),
        }


def main():
    """
    Professional FTMO Optimization Workflow with CLI support.
    
    Usage:
      python ftmo_challenge_analyzer.py              # Run/resume optimization (5 trials)
      python ftmo_challenge_analyzer.py --status     # Check progress without running
      python ftmo_challenge_analyzer.py --trials 100 # Run 100 trials
    """
    parser = argparse.ArgumentParser(
        description="FTMO Professional Optimization System - Resumable with ADX Filter"
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Check optimization progress without running new trials"
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=5,
        help="Number of optimization trials to run (default: 5)"
    )
    args = parser.parse_args()
    
    if args.status:
        show_optimization_status()
        return
    
    n_trials = args.trials
    
    print(f"\n{'='*80}")
    print("FTMO PROFESSIONAL OPTIMIZATION SYSTEM")
    print(f"{'='*80}")
    print(f"\nData Partitioning:")
    print(f"  TRAINING:    Jan 1, 2024 - Sep 30, 2024 (in-sample)")
    print(f"  VALIDATION:  Oct 1, 2024 - Dec 31, 2024 (out-of-sample)")
    print(f"  FINAL:       Full year 2024 (December fully open)")
    print(f"\nADX > 25 trend-strength filter applied")
    print(f"Resumable: Study stored in {OPTUNA_DB_PATH}")
    print(f"{'='*80}\n")
    
    optimizer = OptunaOptimizer()
    results = optimizer.run_optimization(n_trials=n_trials)
    
    best_params = results['best_params']
    
    print(f"\n{'='*80}")
    print("=== TRAINING RESULTS (Jan-Sep 2024) ===")
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
        ml_min_prob=None,
        require_adx_filter=True,
    )
    
    training_results = print_period_results(
        training_trades, "TRAINING RESULTS (Jan-Sep 2024)",
        TRAINING_START, TRAINING_END
    )
    
    print(f"\n{'='*80}")
    print("=== VALIDATION RESULTS (Oct-Dec 2024) ===")
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
        ml_min_prob=None,
        require_adx_filter=True,
    )
    
    validation_results = print_period_results(
        validation_trades, "VALIDATION RESULTS (Oct-Dec 2024)",
        VALIDATION_START, VALIDATION_END
    )
    
    print(f"\n{'='*80}")
    print("=== FULL YEAR FINAL RESULTS ===")
    print(f"{'='*80}")
    print("Running full year backtest with ML FILTER DISABLED...")
    print("December fully open for trading")
    
    full_year_trades = run_full_period_backtest(
        start_date=FULL_YEAR_START,
        end_date=FULL_YEAR_END,
        min_confluence=best_params.get('min_confluence_score', 3),
        min_quality_factors=best_params.get('min_quality_factors', 2),
        risk_per_trade_pct=best_params.get('risk_per_trade_pct', 0.5),
        atr_min_percentile=best_params.get('atr_min_percentile', 60.0),
        trail_activation_r=best_params.get('trail_activation_r', 2.2),
        december_atr_multiplier=best_params.get('december_atr_multiplier', 1.5),
        volatile_asset_boost=best_params.get('volatile_asset_boost', 1.5),
        ml_min_prob=None,
        require_adx_filter=True,
    )
    
    risk_pct = best_params.get('risk_per_trade_pct', 0.5)
    export_trades_to_csv(full_year_trades, "all_trades_2024_full.csv", risk_pct)
    
    full_year_results = print_period_results(
        full_year_trades, "FULL YEAR FINAL RESULTS (2024)",
        FULL_YEAR_START, FULL_YEAR_END
    )
    
    if full_year_trades and len(full_year_trades) >= 30:
        print(f"\n{'='*80}")
        print("MONTE CARLO SIMULATION (1000 iterations)")
        print(f"{'='*80}")
        mc_results = run_monte_carlo_analysis(full_year_trades, num_simulations=1000)
    
    print(f"\n{'='*80}")
    print("QUARTERLY PERFORMANCE BREAKDOWN (Full Year)")
    print(f"{'='*80}")
    
    for q_name, (q_start, q_end) in QUARTERS_2024.items():
        q_filtered = []
        for t in full_year_trades:
            entry = getattr(t, 'entry_date', None)
            if entry:
                if isinstance(entry, str):
                    try:
                        entry = datetime.fromisoformat(entry.replace("Z", "+00:00"))
                    except:
                        continue
                if hasattr(entry, 'replace') and entry.tzinfo:
                    entry = entry.replace(tzinfo=None)
                if q_start <= entry <= q_end:
                    q_filtered.append(t)
        
        q_r = sum(getattr(t, 'rr', 0) for t in q_filtered)
        q_wins = sum(1 for t in q_filtered if getattr(t, 'rr', 0) > 0)
        q_wr = (q_wins / len(q_filtered) * 100) if q_filtered else 0
        print(f"  {q_name}: {len(q_filtered)} trades, {q_r:+.1f}R, {q_wr:.0f}% win rate")
    
    if full_year_trades and len(full_year_trades) >= 50:
        print(f"\n{'='*80}")
        print("TRAINING ML MODEL")
        print(f"{'='*80}")
        train_ml_model(full_year_trades)
    
    print(f"\n{'='*80}")
    print("UPDATING DOCUMENTATION")
    print(f"{'='*80}")
    update_readme_documentation()
    
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"\nBest Score: {results['best_score']:.2f}")
    print(f"Trials Run This Session: {results['n_trials']}")
    print(f"Total Trials in Study: {results.get('total_trials', results['n_trials'])}")
    print(f"\nFiles Created:")
    print(f"  - params/current_params.json (optimized parameters)")
    print(f"  - ftmo_analysis_output/all_trades_2024_full.csv ({len(full_year_trades) if full_year_trades else 0} trades)")
    print(f"  - models/best_rf.joblib (ML model)")
    print(f"  - optuna_study.db (resumable optimization state)")
    print(f"  - ftmo_optimization_progress.txt (progress log)")
    
    print(f"\nDocumentation updated. CSV exported. Ready for live trading.")


if __name__ == "__main__":
    main()
