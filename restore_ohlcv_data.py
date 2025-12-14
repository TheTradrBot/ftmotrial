"""
Re-download all OHLCV data from OANDA to restore weekend candles.
The previous data was corrupted by removing Fridays and Sundays incorrectly.
"""

import os
import datetime as dt
from pathlib import Path
import pandas as pd
import requests
import time

OANDA_API_URL = "https://api-fxpractice.oanda.com"
OHLCV_DIR = Path("data/ohlcv")

SYMBOLS = [
    "EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF", "USD_CAD", "AUD_USD", "NZD_USD",
    "EUR_GBP", "EUR_JPY", "EUR_CHF", "EUR_AUD", "EUR_CAD", "EUR_NZD",
    "GBP_JPY", "GBP_CHF", "GBP_AUD", "GBP_CAD", "GBP_NZD",
    "AUD_JPY", "AUD_CHF", "AUD_CAD", "AUD_NZD",
    "NZD_JPY", "NZD_CHF", "NZD_CAD",
    "CAD_JPY", "CAD_CHF", "CHF_JPY",
    "XAU_USD", "XAG_USD",
    "BTC_USD", "ETH_USD",
    "SPX500_USD", "NAS100_USD",
]

TIMEFRAMES = {
    "D": "D1",
    "H4": "H4",
    "W": "W1",
    "M": "MN",
}

def get_oanda_candles(symbol, granularity, start_date, end_date):
    api_key = os.getenv("OANDA_API_KEY", "").strip()
    if not api_key:
        print("ERROR: OANDA_API_KEY not set")
        return []
    
    headers = {"Authorization": f"Bearer {api_key}"}
    url = f"{OANDA_API_URL}/v3/instruments/{symbol}/candles"
    
    all_candles = []
    current_start = start_date
    
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
    max_candles = 5000
    batch_duration = candle_duration * max_candles
    
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
            if resp.status_code != 200:
                print(f"  Error {resp.status_code} for {symbol}: {resp.text[:100]}")
                break
            
            data = resp.json()
            batch_candles = []
            
            for c in data.get("candles", []):
                if not c.get("complete", True):
                    continue
                time_str = c["time"]
                t = time_str.split(".")[0].replace("Z", "")
                time_dt = dt.datetime.fromisoformat(t)
                if time_dt.tzinfo is None:
                    time_dt = time_dt.replace(tzinfo=dt.timezone.utc)
                
                mid = c["mid"]
                batch_candles.append({
                    "time": time_dt,
                    "Open": float(mid["o"]),
                    "High": float(mid["h"]),
                    "Low": float(mid["l"]),
                    "Close": float(mid["c"]),
                    "Volume": float(c.get("volume", 0)),
                })
            
            all_candles.extend(batch_candles)
            
            if not batch_candles:
                current_start = batch_end
            else:
                last_time = batch_candles[-1]["time"]
                current_start = last_time + candle_duration
            
        except requests.exceptions.RequestException as e:
            print(f"  Network error for {symbol}: {e}")
            break
        
        time.sleep(0.1)
    
    return all_candles


def save_to_csv(candles, symbol, tf_file):
    if not candles:
        return False
    
    norm_symbol = symbol.replace("_", "")
    filename = f"{norm_symbol}_{tf_file}_2023_2024.csv"
    filepath = OHLCV_DIR / filename
    
    df = pd.DataFrame(candles)
    df.set_index("time", inplace=True)
    df.sort_index(inplace=True)
    df = df[~df.index.duplicated(keep='first')]
    df.to_csv(filepath)
    
    return True


def main():
    OHLCV_DIR.mkdir(parents=True, exist_ok=True)
    
    start_date = dt.datetime(2023, 1, 1, tzinfo=dt.timezone.utc)
    end_date = dt.datetime(2024, 12, 31, 23, 59, 59, tzinfo=dt.timezone.utc)
    
    total = len(SYMBOLS) * len(TIMEFRAMES)
    count = 0
    
    print(f"Downloading {total} files from OANDA...")
    print(f"Date range: {start_date.date()} to {end_date.date()}")
    print("=" * 50)
    
    for symbol in SYMBOLS:
        for granularity, tf_file in TIMEFRAMES.items():
            count += 1
            print(f"[{count}/{total}] {symbol} {granularity}...", end=" ", flush=True)
            
            candles = get_oanda_candles(symbol, granularity, start_date, end_date)
            
            if candles:
                save_to_csv(candles, symbol, tf_file)
                print(f"OK ({len(candles)} candles)")
            else:
                print("FAILED (0 candles)")
            
            time.sleep(0.2)
    
    print("=" * 50)
    print("Download complete!")
    
    sample_file = OHLCV_DIR / "EURUSD_D1_2023_2024.csv"
    if sample_file.exists():
        df = pd.read_csv(sample_file)
        print(f"\nSample verification - EURUSD D1:")
        print(f"  Total rows: {len(df)}")
        print(f"  First 10 dates:")
        for i, row in df.head(10).iterrows():
            date_str = row.iloc[0] if 'time' not in df.columns else row['time']
            print(f"    {date_str}")


if __name__ == "__main__":
    main()
