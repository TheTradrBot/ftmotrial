"""
OANDA Historical Data Downloader

Downloads OHLCV data for all 34 trading assets from OANDA for 2003-2025.
Derives monthly (MN) data from daily candles since OANDA API is unreliable for monthly.
Skips already downloaded files. Run in Shell tab for long downloads.

Usage: python download_oanda_data.py
"""

import os
from datetime import datetime, timezone
from pathlib import Path
from data import get_ohlcv
import pandas as pd

OUTPUT_DIR = Path("data/ohlcv")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ASSETS = [
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

TIMEFRAMES = ["D", "H4", "W"]
TF_MAP = {"D": "D1", "H4": "H4", "W": "W1"}

START_YEAR = 2003
END_YEAR = 2025


def derive_monthly_from_daily(symbol: str) -> bool:
    """
    Derive monthly OHLCV data from daily data.
    Returns True if successful, False otherwise.
    """
    daily_path = OUTPUT_DIR / f"{symbol}_D1_{START_YEAR}_{END_YEAR}.csv"
    monthly_path = OUTPUT_DIR / f"{symbol}_MN_{START_YEAR}_{END_YEAR}.csv"
    
    if monthly_path.exists():
        print(f"  MN: Already exists, skipping")
        return True
    
    if not daily_path.exists():
        print(f"  MN: Cannot derive - daily data not found")
        return False
    
    try:
        df = pd.read_csv(daily_path, index_col=0, parse_dates=True)
        
        monthly = df.resample('ME').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()
        
        monthly.to_csv(monthly_path)
        print(f"  MN: Derived {len(monthly)} monthly candles from daily data")
        return True
    except Exception as e:
        print(f"  MN: Error deriving monthly - {e}")
        return False


def main():
    print("=" * 70)
    print("OANDA HISTORICAL DATA DOWNLOADER")
    print("=" * 70)
    print(f"Assets: {len(ASSETS)}")
    print(f"Date range: {START_YEAR}-01-01 to {END_YEAR}-12-31")
    print(f"Timeframes: D1, H4, W1 (Monthly derived from Daily)")
    print(f"Skipping already downloaded files")
    print("=" * 70)

    start_date = datetime(START_YEAR, 1, 1, tzinfo=timezone.utc)
    end_date = datetime(END_YEAR, 12, 31, tzinfo=timezone.utc)

    successful = 0
    skipped = 0
    failed = 0
    monthly_derived = 0

    for i, asset in enumerate(ASSETS, 1):
        print(f"\n[{i}/{len(ASSETS)}] {asset}")
        symbol = asset.replace("_", "")
        
        for tf in TIMEFRAMES:
            tf_name = TF_MAP[tf]
            output_path = OUTPUT_DIR / f"{symbol}_{tf_name}_{START_YEAR}_{END_YEAR}.csv"
            
            if output_path.exists():
                print(f"  {tf_name}: Already exists, skipping")
                skipped += 1
                continue
            
            try:
                candles = get_ohlcv(
                    instrument=asset,
                    timeframe=tf,
                    start_date=start_date,
                    end_date=end_date,
                    use_cache=False
                )
                
                if candles:
                    df = pd.DataFrame(candles)
                    df.set_index('time', inplace=True)
                    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                    df.to_csv(output_path)
                    print(f"  {tf_name}: {len(candles)} candles saved")
                    successful += 1
                else:
                    print(f"  {tf_name}: No data")
                    failed += 1
            except Exception as e:
                print(f"  {tf_name}: Error - {e}")
                failed += 1
        
        if derive_monthly_from_daily(symbol):
            monthly_derived += 1

    print("\n" + "=" * 70)
    print("DOWNLOAD COMPLETE")
    print("=" * 70)
    print(f"Successful downloads: {successful} files")
    print(f"Skipped (already exist): {skipped} files")
    print(f"Failed: {failed} files")
    print(f"Monthly derived from daily: {monthly_derived} files")
    print("=" * 70)

if __name__ == "__main__":
    main()
