"""
Data access layer for Blueprint Trader AI.

Uses OANDA v20 REST API for OHLCV candles with intelligent caching
to reduce latency and API calls.
"""

import datetime as dt
import os
from typing import List, Dict, Any, Optional

import requests

from config import OANDA_API_URL, GRANULARITY_MAP
from cache import get_cache


def _get_api_key() -> str:
    """Get OANDA API key from environment."""
    return os.getenv("OANDA_API_KEY", "").strip()


def _oanda_headers() -> Optional[Dict[str, str]]:
    """Get OANDA API headers, or None if API key not configured."""
    api_key = _get_api_key()
    if not api_key:
        print("[data] OANDA_API_KEY not configured")
        return None
    return {"Authorization": f"Bearer {api_key}"}


def get_ohlcv(
    instrument: str,
    timeframe: str = "D",
    count: int = 200,
    use_cache: bool = True,
    start_date: Optional[dt.datetime] = None,
    end_date: Optional[dt.datetime] = None,
) -> List[Dict[str, Any]]:
    """
    Fetch OHLCV candles from OANDA for a given instrument and timeframe.
    
    Args:
        instrument: OANDA instrument name (e.g. EUR_USD)
        timeframe: Candle timeframe - "D", "H4", "W", "M"
        count: Number of candles to fetch (ignored if start_date is provided)
        use_cache: Whether to use caching (default True)
        start_date: Optional start date for historical data (e.g., datetime(2023, 1, 1))
        end_date: Optional end date for historical data (defaults to now)

    Returns:
        List of candle dicts with keys: time, open, high, low, close, volume
    """
    if start_date is not None:
        return _get_ohlcv_date_range(instrument, timeframe, start_date, end_date, use_cache)
    
    cache = get_cache()
    
    if use_cache:
        cached = cache.get(instrument, timeframe, count)
        if cached is not None:
            return cached

    headers = _oanda_headers()
    if headers is None:
        print(f"[data.get_ohlcv] OANDA_API_KEY not configured. Set it in Replit Secrets.")
        return []

    granularity = GRANULARITY_MAP.get(timeframe, timeframe)
    url = f"{OANDA_API_URL}/v3/instruments/{instrument}/candles"

    params = {
        "granularity": granularity,
        "count": count,
        "price": "M",
    }

    try:
        resp = requests.get(url, headers=headers, params=params, timeout=15)
    except requests.exceptions.RequestException as e:
        print(f"[data.get_ohlcv] Network error for {instrument}, {timeframe}: {e}")
        return []
    
    if resp.status_code != 200:
        print(f"[data.get_ohlcv] Error {resp.status_code} for {instrument}, {timeframe}: {resp.text}")
        return []

    data = resp.json()
    candles = []

    for c in data.get("candles", []):
        if not c.get("complete", True):
            continue
        time_str = c["time"]
        t = time_str.split(".")[0].replace("Z", "")
        time_dt = dt.datetime.fromisoformat(t)

        mid = c["mid"]
        candles.append({
            "time": time_dt,
            "open": float(mid["o"]),
            "high": float(mid["h"]),
            "low": float(mid["l"]),
            "close": float(mid["c"]),
            "volume": float(c.get("volume", 0)),
        })

    if use_cache and candles:
        cache.set(instrument, timeframe, count, candles)

    return candles


def _get_ohlcv_date_range(
    instrument: str,
    timeframe: str,
    start_date: dt.datetime,
    end_date: Optional[dt.datetime],
    use_cache: bool = True,
) -> List[Dict[str, Any]]:
    """
    Fetch OHLCV candles for a specific date range, handling OANDA's 5000 candle limit.
    
    Args:
        instrument: OANDA instrument name
        timeframe: Candle timeframe
        start_date: Start datetime
        end_date: End datetime (defaults to now)
        use_cache: Whether to use caching
    
    Returns:
        List of candle dicts
    """
    if end_date is None:
        end_date = dt.datetime.now(dt.timezone.utc)
    
    if start_date.tzinfo is None:
        start_date = start_date.replace(tzinfo=dt.timezone.utc)
    if end_date.tzinfo is None:
        end_date = end_date.replace(tzinfo=dt.timezone.utc)
    
    cache = get_cache()
    cache_key = f"{instrument}_{timeframe}_{start_date.isoformat()}_{end_date.isoformat()}"
    
    if use_cache:
        cached = cache.get(instrument, f"{timeframe}_range", hash(cache_key) % 1000000)
        if cached is not None:
            return cached
    
    headers = _oanda_headers()
    if headers is None:
        print(f"[data._get_ohlcv_date_range] OANDA_API_KEY not configured.")
        return []
    
    granularity = GRANULARITY_MAP.get(timeframe, timeframe)
    url = f"{OANDA_API_URL}/v3/instruments/{instrument}/candles"
    
    all_candles = []
    current_start = start_date
    max_candles_per_request = 5000
    
    timeframe_deltas = {
        "M1": dt.timedelta(minutes=1),
        "M5": dt.timedelta(minutes=5),
        "M15": dt.timedelta(minutes=15),
        "M30": dt.timedelta(minutes=30),
        "H1": dt.timedelta(hours=1),
        "H4": dt.timedelta(hours=4),
        "D": dt.timedelta(days=1),
        "W": dt.timedelta(weeks=1),
        "M": dt.timedelta(days=30),
    }
    
    candle_duration = timeframe_deltas.get(granularity, dt.timedelta(days=1))
    batch_duration = candle_duration * max_candles_per_request
    
    while current_start < end_date:
        batch_end = min(current_start + batch_duration, end_date)
        
        params = {
            "granularity": granularity,
            "from": current_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "to": batch_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "price": "M",
        }
        
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=30)
        except requests.exceptions.RequestException as e:
            print(f"[data._get_ohlcv_date_range] Network error for {instrument}: {e}")
            break
        
        if resp.status_code != 200:
            print(f"[data._get_ohlcv_date_range] Error {resp.status_code}: {resp.text}")
            break
        
        data = resp.json()
        batch_candles = []
        
        for c in data.get("candles", []):
            if not c.get("complete", True):
                continue
            time_str = c["time"]
            t = time_str.split(".")[0].replace("Z", "")
            time_dt = dt.datetime.fromisoformat(t)
            
            mid = c["mid"]
            batch_candles.append({
                "time": time_dt,
                "open": float(mid["o"]),
                "high": float(mid["h"]),
                "low": float(mid["l"]),
                "close": float(mid["c"]),
                "volume": float(c.get("volume", 0)),
            })
        
        all_candles.extend(batch_candles)
        
        if not batch_candles:
            current_start = batch_end
        else:
            last_time = batch_candles[-1]["time"]
            if last_time.tzinfo is None:
                last_time = last_time.replace(tzinfo=dt.timezone.utc)
            current_start = last_time + candle_duration
        
        if current_start >= end_date:
            break
    
    if use_cache and all_candles:
        cache.set(instrument, f"{timeframe}_range", hash(cache_key) % 1000000, all_candles)
    
    print(f"[data._get_ohlcv_date_range] Fetched {len(all_candles)} candles for {instrument} from {start_date.date()} to {end_date.date()}")
    
    return all_candles


def get_cache_stats() -> Dict[str, Any]:
    """Get statistics about the data cache."""
    return get_cache().get_stats()


def clear_cache() -> None:
    """Clear all cached data."""
    get_cache().clear()


def clear_instrument_cache(instrument: str) -> None:
    """Clear cache for a specific instrument."""
    get_cache().clear_instrument(instrument)


def get_current_prices(instruments: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Get current bid/ask prices from OANDA.
    
    Returns dict like: {"EUR_USD": {"bid": 1.0950, "ask": 1.0952, "mid": 1.0951}}
    """
    headers = _oanda_headers()
    if headers is None:
        return {}
    
    if not instruments:
        return {}
    
    account_id = _get_account_id()
    if not account_id:
        print("[data.get_current_prices] No OANDA_ACCOUNT_ID configured")
        return {}
    
    # OANDA requires comma-separated instrument list
    instruments_str = ",".join(instruments)
    url = f"{OANDA_API_URL}/v3/accounts/{account_id}/pricing"
    
    params = {"instruments": instruments_str}
    
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=10)
    except requests.exceptions.RequestException as e:
        print(f"[data.get_current_prices] Network error: {e}")
        return {}
    
    if resp.status_code != 200:
        print(f"[data.get_current_prices] Error {resp.status_code}: {resp.text}")
        return {}
    
    result = {}
    data = resp.json()
    
    for price in data.get("prices", []):
        instrument = price.get("instrument")
        bid = float(price.get("bids", [{}])[0].get("price", 0)) if price.get("bids") else 0
        ask = float(price.get("asks", [{}])[0].get("price", 0)) if price.get("asks") else 0
        mid = (bid + ask) / 2 if bid and ask else 0
        
        if instrument and mid:
            result[instrument] = {"bid": bid, "ask": ask, "mid": mid}
    
    return result


def _get_account_id() -> str:
    """Get OANDA account ID from environment."""
    import os
    account_id = os.getenv("OANDA_ACCOUNT_ID", "").strip()
    if not account_id:
        print("[data._get_account_id] OANDA_ACCOUNT_ID not set or empty")
    return account_id
