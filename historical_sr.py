"""
Historical Support/Resistance Level Detection

Loads historical OHLCV data from local CSV files and calculates S/R levels
based on the Blueprint strategy specification:
- S/R level = price where multiple candle HIGHs and LOWs converge
- A level is confirmed when there are 3+ "touches" (hits)
- A touch = when a candle's LOW hits near another candle's HIGH (or vice versa)

This provides long-term S/R zones from 2003+ data for Monthly and Weekly timeframes.
"""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import pandas as pd

DATA_DIR = Path("data/ohlcv")
CACHE_DIR = Path("data/sr_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

SYMBOL_MAP = {
    "EUR_USD": "EURUSD",
    "GBP_USD": "GBPUSD",
    "USD_JPY": "USDJPY",
    "USD_CHF": "USDCHF",
    "USD_CAD": "USDCAD",
    "AUD_USD": "AUDUSD",
    "NZD_USD": "NZDUSD",
    "EUR_GBP": "EURGBP",
    "EUR_JPY": "EURJPY",
    "EUR_CHF": "EURCHF",
    "EUR_AUD": "EURAUD",
    "EUR_CAD": "EURCAD",
    "EUR_NZD": "EURNZD",
    "GBP_JPY": "GBPJPY",
    "GBP_CHF": "GBPCHF",
    "GBP_AUD": "GBPAUD",
    "GBP_CAD": "GBPCAD",
    "GBP_NZD": "GBPNZD",
    "AUD_JPY": "AUDJPY",
    "AUD_CHF": "AUDCHF",
    "AUD_CAD": "AUDCAD",
    "AUD_NZD": "AUDNZD",
    "NZD_JPY": "NZDJPY",
    "NZD_CHF": "NZDCHF",
    "NZD_CAD": "NZDCAD",
    "CAD_JPY": "CADJPY",
    "CAD_CHF": "CADCHF",
    "CHF_JPY": "CHFJPY",
    "XAU_USD": "XAUUSD",
    "XAG_USD": "XAGUSD",
    "BTC_USD": "BTCUSD",
    "ETH_USD": "ETHUSD",
    "SPX500_USD": "SPX500",
    "NAS100_USD": "NAS100",
}


def _get_pip_tolerance(symbol: str) -> float:
    """
    Get the tolerance for S/R level matching based on instrument type.
    Returns tolerance as a decimal (e.g., 0.0005 for 5 pips on EURUSD).
    """
    symbol_upper = symbol.upper().replace("_", "")
    
    if "JPY" in symbol_upper:
        return 0.05  # 5 pips for JPY pairs
    elif "XAU" in symbol_upper:
        return 5.0  # $5 for gold
    elif "XAG" in symbol_upper:
        return 0.10  # $0.10 for silver
    elif "BTC" in symbol_upper:
        return 500.0  # $500 for BTC
    elif "ETH" in symbol_upper:
        return 50.0  # $50 for ETH
    elif "SPX" in symbol_upper or "NAS" in symbol_upper or "US30" in symbol_upper:
        return 50.0  # 50 points for indices
    else:
        return 0.0010  # 10 pips for standard forex


def load_historical_data(symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
    """
    Load historical OHLCV data from local CSV files.
    
    Args:
        symbol: Symbol in MT5 format (e.g., "EUR_USD") or clean format ("EURUSD")
        timeframe: "MN", "W1", "D1", or "H4"
    
    Returns:
        DataFrame with OHLCV data or None if not found
    """
    clean_symbol = symbol.replace("_", "")
    if symbol in SYMBOL_MAP:
        clean_symbol = SYMBOL_MAP[symbol]
    
    patterns = [
        f"{clean_symbol}_{timeframe}_*.csv",
        f"{clean_symbol}_{timeframe}.csv",
    ]
    
    for pattern in patterns:
        matches = list(DATA_DIR.glob(pattern))
        if matches:
            matches.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            try:
                df = pd.read_csv(matches[0], index_col=0, parse_dates=True)
                df.columns = ['Open', 'High', 'Low', 'Close', 'Volume'] if len(df.columns) == 5 else df.columns
                return df
            except Exception as e:
                print(f"[historical_sr] Error loading {matches[0]}: {e}")
                continue
    
    print(f"[historical_sr] No data found for {clean_symbol} {timeframe}")
    return None


def find_sr_levels(
    candles: pd.DataFrame,
    min_touches: int = 3,
    tolerance_pct: float = 0.002,
    tolerance_absolute: Optional[float] = None
) -> List[Dict]:
    """
    Find S/R levels where candle HIGHs and LOWs converge.
    
    An S/R level is confirmed when:
    - A price level has been touched by multiple candle highs AND lows
    - There are at least `min_touches` total touches at that level
    - A "touch" is when a HIGH or LOW is within tolerance of the level
    
    Args:
        candles: DataFrame with High, Low columns
        min_touches: Minimum number of touches to confirm S/R (default 3)
        tolerance_pct: Percentage tolerance for level matching (default 0.2%)
        tolerance_absolute: Absolute tolerance (overrides percentage if provided)
    
    Returns:
        List of S/R level dicts with: level, touches, first_touch, last_touch, type
    """
    if candles is None or len(candles) < 10:
        return []
    
    highs = candles['High'].values
    lows = candles['Low'].values
    times = candles.index.tolist()
    
    all_levels = []
    for i, (h, l) in enumerate(zip(highs, lows)):
        all_levels.append({'price': h, 'type': 'high', 'idx': i, 'time': times[i]})
        all_levels.append({'price': l, 'type': 'low', 'idx': i, 'time': times[i]})
    
    all_levels.sort(key=lambda x: x['price'])
    
    sr_zones = []
    used = set()
    
    for i, level in enumerate(all_levels):
        if i in used:
            continue
        
        price = level['price']
        
        if tolerance_absolute:
            tol = tolerance_absolute
        else:
            tol = price * tolerance_pct
        
        touches = []
        high_touches = 0
        low_touches = 0
        
        for j, other in enumerate(all_levels):
            if abs(other['price'] - price) <= tol:
                touches.append(other)
                used.add(j)
                if other['type'] == 'high':
                    high_touches += 1
                else:
                    low_touches += 1
        
        if len(touches) >= min_touches and high_touches >= 1 and low_touches >= 1:
            avg_price = sum(t['price'] for t in touches) / len(touches)
            first_time = min(t['time'] for t in touches)
            last_time = max(t['time'] for t in touches)
            
            sr_zones.append({
                'level': round(avg_price, 5),
                'touches': len(touches),
                'high_touches': high_touches,
                'low_touches': low_touches,
                'first_touch': str(first_time),
                'last_touch': str(last_time),
                'strength': 'strong' if len(touches) >= 5 else 'moderate',
            })
    
    sr_zones.sort(key=lambda x: x['touches'], reverse=True)
    
    return sr_zones


def find_sr_from_high_low_convergence(
    candles: pd.DataFrame,
    min_touches: int = 3,
    symbol: str = "EURUSD"
) -> List[Dict]:
    """
    Find S/R levels specifically where a candle's LOW hits another candle's HIGH.
    
    This is the core S/R detection logic per the strategy spec:
    - A monthly/weekly HIGH that later gets touched by another candle's LOW = resistance
    - A monthly/weekly LOW that later gets touched by another candle's HIGH = support
    - 3+ such touches confirms the level
    
    Args:
        candles: DataFrame with High, Low columns
        min_touches: Minimum touches to confirm (default 3)
        symbol: Symbol for tolerance calculation
    
    Returns:
        List of confirmed S/R levels
    """
    if candles is None or len(candles) < 10:
        return []
    
    tolerance = _get_pip_tolerance(symbol)
    
    highs = candles['High'].values
    lows = candles['Low'].values
    times = candles.index.tolist()
    
    level_hits = {}
    
    for i in range(len(candles)):
        current_high = highs[i]
        current_low = lows[i]
        
        for j in range(len(candles)):
            if i == j:
                continue
            
            if abs(current_low - highs[j]) <= tolerance:
                level_key = round((current_low + highs[j]) / 2, 5)
                if level_key not in level_hits:
                    level_hits[level_key] = {
                        'level': level_key,
                        'touches': [],
                        'first_time': None,
                        'last_time': None,
                    }
                
                hit_info = {
                    'time': times[i],
                    'type': 'low_hits_high',
                    'low_idx': i,
                    'high_idx': j,
                    'low_price': current_low,
                    'high_price': highs[j],
                }
                
                is_duplicate = any(
                    t['low_idx'] == i and t['high_idx'] == j 
                    for t in level_hits[level_key]['touches']
                )
                if not is_duplicate:
                    level_hits[level_key]['touches'].append(hit_info)
            
            if abs(current_high - lows[j]) <= tolerance:
                level_key = round((current_high + lows[j]) / 2, 5)
                if level_key not in level_hits:
                    level_hits[level_key] = {
                        'level': level_key,
                        'touches': [],
                        'first_time': None,
                        'last_time': None,
                    }
                
                hit_info = {
                    'time': times[i],
                    'type': 'high_hits_low',
                    'high_idx': i,
                    'low_idx': j,
                    'high_price': current_high,
                    'low_price': lows[j],
                }
                
                is_duplicate = any(
                    t.get('high_idx') == i and t.get('low_idx') == j 
                    for t in level_hits[level_key]['touches']
                )
                if not is_duplicate:
                    level_hits[level_key]['touches'].append(hit_info)
    
    merged_levels = _merge_nearby_levels(level_hits, tolerance)
    
    confirmed_sr = []
    for level_key, data in merged_levels.items():
        touch_count = len(data['touches'])
        if touch_count >= min_touches:
            all_times = [t['time'] for t in data['touches']]
            confirmed_sr.append({
                'level': data['level'],
                'touches': touch_count,
                'first_touch': str(min(all_times)),
                'last_touch': str(max(all_times)),
                'strength': 'very_strong' if touch_count >= 5 else 'strong' if touch_count >= 4 else 'moderate',
            })
    
    confirmed_sr.sort(key=lambda x: x['touches'], reverse=True)
    
    return confirmed_sr


def _merge_nearby_levels(level_hits: Dict, tolerance: float) -> Dict:
    """Merge S/R levels that are within tolerance of each other."""
    sorted_levels = sorted(level_hits.keys())
    merged = {}
    used = set()
    
    for level in sorted_levels:
        if level in used:
            continue
        
        nearby_levels = [level]
        for other in sorted_levels:
            if other != level and other not in used and abs(other - level) <= tolerance * 2:
                nearby_levels.append(other)
                used.add(other)
        
        all_touches = []
        for nl in nearby_levels:
            all_touches.extend(level_hits[nl]['touches'])
        
        avg_level = sum(nearby_levels) / len(nearby_levels)
        merged[round(avg_level, 5)] = {
            'level': round(avg_level, 5),
            'touches': all_touches,
        }
        used.add(level)
    
    return merged


def get_historical_sr_levels(
    symbol: str,
    timeframe: str = "MN",
    min_touches: int = 3,
    use_cache: bool = True
) -> List[Dict]:
    """
    Get historical S/R levels for a symbol from local data.
    
    Args:
        symbol: Symbol (e.g., "EUR_USD" or "EURUSD")
        timeframe: "MN" for monthly, "W1" for weekly
        min_touches: Minimum touches to confirm S/R
        use_cache: Whether to use cached results
    
    Returns:
        List of S/R levels with level, touches, strength
    """
    clean_symbol = symbol.replace("_", "")
    cache_file = CACHE_DIR / f"{clean_symbol}_{timeframe}_sr.json"
    
    if use_cache and cache_file.exists():
        try:
            cache_stat = cache_file.stat()
            cache_age_hours = (datetime.now().timestamp() - cache_stat.st_mtime) / 3600
            if cache_age_hours < 24:
                with open(cache_file, 'r') as f:
                    return json.load(f)
        except Exception:
            pass
    
    df = load_historical_data(symbol, timeframe)
    if df is None:
        return []
    
    sr_levels = find_sr_from_high_low_convergence(df, min_touches=min_touches, symbol=symbol)
    
    if sr_levels:
        try:
            with open(cache_file, 'w') as f:
                json.dump(sr_levels, f, indent=2)
        except Exception as e:
            print(f"[historical_sr] Warning: Could not cache S/R levels: {e}")
    
    return sr_levels


def get_all_htf_sr_levels(symbol: str, min_touches: int = 3) -> Dict[str, List[Dict]]:
    """
    Get S/R levels for both Monthly and Weekly timeframes.
    
    Returns:
        Dict with 'monthly' and 'weekly' keys containing S/R level lists
    """
    return {
        'monthly': get_historical_sr_levels(symbol, 'MN', min_touches),
        'weekly': get_historical_sr_levels(symbol, 'W1', min_touches),
    }


def is_price_at_historical_sr(
    price: float,
    symbol: str,
    tolerance_pct: float = 0.005
) -> Tuple[bool, Optional[Dict]]:
    """
    Check if current price is at a historical S/R level.
    
    Args:
        price: Current price
        symbol: Symbol
        tolerance_pct: How close price must be to level (default 0.5%)
    
    Returns:
        Tuple of (is_at_sr, sr_level_info)
    """
    sr_data = get_all_htf_sr_levels(symbol)
    
    all_levels = sr_data['monthly'] + sr_data['weekly']
    
    for sr in all_levels:
        level = sr['level']
        tolerance = level * tolerance_pct
        if abs(price - level) <= tolerance:
            return True, sr
    
    return False, None


def print_sr_summary(symbol: str):
    """Print a summary of S/R levels for a symbol."""
    sr_data = get_all_htf_sr_levels(symbol)
    
    print(f"\n{'='*60}")
    print(f"S/R LEVELS FOR {symbol}")
    print(f"{'='*60}")
    
    print(f"\n--- MONTHLY S/R ({len(sr_data['monthly'])} levels) ---")
    for sr in sr_data['monthly'][:10]:
        print(f"  Level: {sr['level']:.5f} | Touches: {sr['touches']} | Strength: {sr['strength']}")
    
    print(f"\n--- WEEKLY S/R ({len(sr_data['weekly'])} levels) ---")
    for sr in sr_data['weekly'][:10]:
        print(f"  Level: {sr['level']:.5f} | Touches: {sr['touches']} | Strength: {sr['strength']}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        symbol = sys.argv[1]
    else:
        symbol = "EURUSD"
    
    print(f"Loading historical S/R for {symbol}...")
    print_sr_summary(symbol)
