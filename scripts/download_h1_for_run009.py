#!/usr/bin/env python3
"""
Download H1 data for all pairs traded in run_009

This script downloads hourly (H1) OHLCV data from OANDA API
for accurate intraday drawdown analysis.

Usage:
    1. Set OANDA_API_KEY environment variable or create .env file
    2. python scripts/download_h1_for_run009.py
"""

import os
import sys
import time
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import requests
from dotenv import load_dotenv

# Load .env if exists
load_dotenv()

# OANDA API Configuration
OANDA_API_KEY = os.getenv("OANDA_API_KEY", "")
OANDA_API_URL = "https://api-fxpractice.oanda.com"

# Traded pairs from run_009 (forex only - skip crypto/metals/indices)
FOREX_PAIRS = [
    "AUD_CAD", "AUD_CHF", "AUD_JPY", "AUD_NZD", "AUD_USD",
    "CAD_CHF", "CAD_JPY", "CHF_JPY",
    "EUR_AUD", "EUR_CAD", "EUR_CHF", "EUR_GBP", "EUR_JPY", "EUR_NZD", "EUR_USD",
    "GBP_AUD", "GBP_CAD", "GBP_CHF", "GBP_JPY", "GBP_NZD", "GBP_USD",
    "NZD_CAD", "NZD_CHF", "NZD_JPY", "NZD_USD",
    "USD_CAD", "USD_CHF", "USD_JPY",
]

# Not available on OANDA Practice (need alternative source)
SPECIAL_PAIRS = {
    "BTC_USD": "Crypto - not on OANDA",
    "ETH_USD": "Crypto - not on OANDA", 
    "XAU_USD": "Metal - only on OANDA Live",
    "XAG_USD": "Metal - only on OANDA Live",
    "SPX500_USD": "Index - only on OANDA Live",
    "NAS100_USD": "Index - only on OANDA Live",
}

OUTPUT_DIR = Path("data/ohlcv")


def download_h1_chunk(instrument: str, from_date: datetime, to_date: datetime) -> pd.DataFrame:
    """Download H1 candles for a date range (max ~5000 candles per request)"""
    if not OANDA_API_KEY:
        raise ValueError("OANDA_API_KEY not set")
    
    url = f"{OANDA_API_URL}/v3/instruments/{instrument}/candles"
    
    headers = {
        "Authorization": f"Bearer {OANDA_API_KEY}",
        "Content-Type": "application/json"
    }
    
    params = {
        "granularity": "H1",
        "from": from_date.strftime("%Y-%m-%dT%H:%M:%S.000000000Z"),
        "to": to_date.strftime("%Y-%m-%dT%H:%M:%S.000000000Z"),
        "price": "M"  # Mid prices
    }
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        if "candles" not in data or len(data["candles"]) == 0:
            return pd.DataFrame()
        
        candles = []
        for candle in data["candles"]:
            if not candle.get("complete", False):
                continue
                
            candles.append({
                "time": pd.to_datetime(candle["time"]),
                "Open": float(candle["mid"]["o"]),
                "High": float(candle["mid"]["h"]),
                "Low": float(candle["mid"]["l"]),
                "Close": float(candle["mid"]["c"]),
                "Volume": int(candle["volume"])
            })
        
        return pd.DataFrame(candles)
        
    except Exception as e:
        print(f"    ❌ Error: {e}")
        return pd.DataFrame()


def download_full_h1(instrument: str, start_year: int = 2023, end_year: int = 2025) -> pd.DataFrame:
    """
    Download full H1 history in monthly chunks.
    
    H1 candles: ~720 per month (30 days × 24 hours)
    OANDA limit: ~5000 candles per request
    Strategy: Download 1 month at a time
    """
    all_candles = []
    
    current = datetime(start_year, 1, 1)
    end = datetime(end_year, 12, 31, 23, 59, 59)
    
    month_count = 0
    while current < end:
        # Download 1 month at a time
        chunk_end = min(current + timedelta(days=31), end)
        
        print(f"    {current.strftime('%Y-%m')}...", end=" ", flush=True)
        
        df = download_h1_chunk(instrument, current, chunk_end)
        
        if not df.empty:
            all_candles.append(df)
            print(f"✓ {len(df)} candles")
            month_count += 1
        else:
            print("⚠️  no data")
        
        current = chunk_end
        time.sleep(0.5)  # Rate limit protection
    
    if all_candles:
        combined = pd.concat(all_candles, ignore_index=True)
        # Remove duplicates (overlapping month boundaries)
        combined = combined.drop_duplicates(subset=['time']).sort_values('time')
        return combined
    
    return pd.DataFrame()


def main():
    """Main download orchestrator"""
    
    print("=" * 70)
    print("H1 DATA DOWNLOADER FOR RUN_009 INTRADAY DD ANALYSIS")
    print("=" * 70)
    print(f"\nPeriod: 2023-01-01 to 2025-12-31")
    print(f"Granularity: H1 (hourly)")
    print(f"Forex pairs: {len(FOREX_PAIRS)}")
    print(f"Special pairs (manual): {len(SPECIAL_PAIRS)}\n")
    
    # Check API key
    if not OANDA_API_KEY:
        print("❌ OANDA_API_KEY not found!")
        print("\nOptions:")
        print("1. Create .env file with: OANDA_API_KEY=your-key-here")
        print("2. Export env var: export OANDA_API_KEY='your-key-here'")
        print("\nGet API key: https://www.oanda.com/account/tpa/personal_token")
        sys.exit(1)
    
    print(f"✅ OANDA API key found: {OANDA_API_KEY[:8]}...")
    print(f"📁 Output directory: {OUTPUT_DIR}\n")
    
    # Stats
    downloaded = 0
    skipped = 0
    failed = 0
    
    # Download forex pairs
    for i, pair in enumerate(FOREX_PAIRS, 1):
        print(f"[{i}/{len(FOREX_PAIRS)}] {pair}")
        
        # Check if already exists
        output_file = OUTPUT_DIR / f"{pair}_H1_2023_2025.csv"
        if output_file.exists():
            existing = pd.read_csv(output_file)
            print(f"  ⏭️  Already exists ({len(existing)} candles), skipping")
            skipped += 1
            continue
        
        # Download
        try:
            df = download_full_h1(pair, 2023, 2025)
            
            if not df.empty:
                # Save
                df.to_csv(output_file, index=False)
                print(f"  ✅ Saved {len(df):,} candles to {output_file.name}")
                downloaded += 1
            else:
                print(f"  ❌ No data received")
                failed += 1
        
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            failed += 1
        
        # Rate limit: 1 request per second
        if i < len(FOREX_PAIRS):
            time.sleep(1)
    
    # Special pairs note
    if SPECIAL_PAIRS:
        print(f"\n{'─' * 70}")
        print("⚠️  SPECIAL PAIRS (Require alternative data source):")
        print("─" * 70)
        for pair, note in SPECIAL_PAIRS.items():
            print(f"  {pair}: {note}")
        print("\n💡 For these pairs, consider:")
        print("  - Yahoo Finance (free but limited)")
        print("  - Polygon.io (crypto/stocks)")
        print("  - Binance API (BTC/ETH)")
        print("  - MT5 broker demo account")
    
    # Summary
    print(f"\n{'=' * 70}")
    print("DOWNLOAD SUMMARY")
    print("=" * 70)
    print(f"✅ Downloaded: {downloaded}")
    print(f"⏭️  Skipped:    {skipped}")
    print(f"❌ Failed:     {failed}")
    print(f"📊 Total forex pairs: {downloaded + skipped} / {len(FOREX_PAIRS)}")
    
    if downloaded > 0:
        print(f"\n✅ Ready for intraday DD analysis!")
        print(f"   Next: python scripts/analyze_intraday_dd.py")


if __name__ == "__main__":
    main()
