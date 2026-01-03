#!/usr/bin/env python3
"""
Download H1 data for special assets (crypto/metals/indices) via Yahoo Finance

Assets not available on OANDA Practice:
- BTC_USD, ETH_USD (crypto)
- XAU_USD, XAG_USD (metals)
- SPX500_USD, NAS100_USD (indices)
"""

import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime
import time

OUTPUT_DIR = Path("data/ohlcv")

# Yahoo Finance ticker mapping
YAHOO_TICKERS = {
    # Crypto (Yahoo uses -USD suffix)
    "BTC_USD": "BTC-USD",
    "ETH_USD": "ETH-USD",
    
    # Metals (use futures or ETFs as proxy)
    "XAU_USD": "GC=F",      # Gold futures
    "XAG_USD": "SI=F",      # Silver futures
    
    # Indices (Yahoo symbols)
    "SPX500_USD": "^GSPC",  # S&P 500
    "NAS100_USD": "^NDX",   # Nasdaq 100
}


def download_yahoo_h1(ticker: str, our_symbol: str) -> pd.DataFrame:
    """
    Download hourly data from Yahoo Finance.
    
    Yahoo limits: H1 data only available for last 730 days.
    Strategy: Download in chunks (last 2 years in 6-month periods)
    """
    print(f"\n[{our_symbol}] Downloading from Yahoo Finance...")
    print(f"  Ticker: {ticker}")
    
    all_data = []
    
    # Yahoo Finance limitation: H1 data only for last ~730 days
    # So we need to download recent period only
    periods = [
        ("2024-06-01", "2025-01-01"),  # Last 6 months of 2024
        ("2025-01-01", "2026-01-02"),  # All of 2025
    ]
    
    try:
        for start, end in periods:
            print(f"  Fetching {start} to {end}...", end=" ")
            
            data = yf.download(
                ticker,
                start=start,
                end=end,
                interval="1h",
                progress=False,
                auto_adjust=True
            )
            
            if not data.empty:
                all_data.append(data)
                print(f"✓ {len(data)} candles")
            else:
                print("⚠️  empty")
        
        if not all_data:
            print(f"  ❌ No data from any period")
            
            # Fallback: Try daily data and interpolate to hourly
            print(f"  🔄 Trying daily data fallback...")
            return download_daily_interpolate(ticker, our_symbol)
        
        # Combine all periods
        combined = pd.concat(all_data)
        combined = combined.reset_index()
        
        # Rename columns
        if 'Datetime' in combined.columns:
            combined.rename(columns={'Datetime': 'time'}, inplace=True)
        elif 'Date' in combined.columns:
            combined.rename(columns={'Date': 'time'}, inplace=True)
        
        combined.columns = ['time', 'Open', 'High', 'Low', 'Close', 'Volume']
        combined['time'] = pd.to_datetime(combined['time'])
        
        # Remove duplicates and sort
        combined = combined.drop_duplicates(subset=['time']).sort_values('time')
        
        print(f"  ✅ Total: {len(combined):,} hourly candles")
        return combined
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        print(f"  🔄 Trying daily data fallback...")
        return download_daily_interpolate(ticker, our_symbol)


def download_daily_interpolate(ticker: str, our_symbol: str) -> pd.DataFrame:
    """
    Fallback: Download daily data and create synthetic hourly candles.
    
    This is less accurate but allows analysis when H1 not available.
    Each daily candle is split into 24 hourly candles with interpolated prices.
    """
    try:
        print(f"  📊 Downloading daily data for interpolation...")
        
        data = yf.download(
            ticker,
            start="2023-01-01",
            end="2026-01-01",
            interval="1d",
            progress=False,
            auto_adjust=True
        )
        
        if data.empty:
            print(f"  ❌ No daily data either")
            return pd.DataFrame()
        
        data = data.reset_index()
        
        # Rename columns
        if 'Date' in data.columns:
            data.rename(columns={'Date': 'time'}, inplace=True)
        
        data.columns = ['time', 'Open', 'High', 'Low', 'Close', 'Volume']
        data['time'] = pd.to_datetime(data['time'])
        
        print(f"  ✓ Got {len(data)} daily candles")
        print(f"  🔄 Interpolating to hourly (synthetic)...")
        
        # Create hourly candles from daily
        hourly_candles = []
        
        for idx, row in data.iterrows():
            day_start = row['time']
            
            # Create 24 hourly candles for this day
            for hour in range(24):
                hour_time = day_start + pd.Timedelta(hours=hour)
                
                # Simple interpolation: spread daily range across hours
                # Hour 0: open, Hour 23: close, middle hours interpolated
                progress = hour / 23 if hour < 23 else 1.0
                
                interpolated_price = row['Open'] + (row['Close'] - row['Open']) * progress
                
                hourly_candles.append({
                    'time': hour_time,
                    'Open': interpolated_price,
                    'High': min(row['High'], interpolated_price * 1.002),  # Small variance
                    'Low': max(row['Low'], interpolated_price * 0.998),
                    'Close': interpolated_price,
                    'Volume': row['Volume'] / 24  # Distribute volume
                })
        
        hourly_df = pd.DataFrame(hourly_candles)
        print(f"  ✅ Created {len(hourly_df):,} synthetic hourly candles")
        print(f"  ⚠️  NOTE: Interpolated data - less accurate than real H1")
        
        return hourly_df
        
    except Exception as e:
        print(f"  ❌ Interpolation failed: {e}")
        return pd.DataFrame()


def main():
    print("=" * 70)
    print("YAHOO FINANCE H1 DOWNLOADER - SPECIAL ASSETS")
    print("=" * 70)
    print("\nAssets: crypto, metals, indices")
    print(f"Period: 2023-01-01 to 2025-12-31")
    print(f"Output: {OUTPUT_DIR}\n")
    
    downloaded = 0
    skipped = 0
    failed = 0
    
    for our_symbol, yahoo_ticker in YAHOO_TICKERS.items():
        output_file = OUTPUT_DIR / f"{our_symbol}_H1_2023_2025.csv"
        
        # Check if already exists
        if output_file.exists():
            existing = pd.read_csv(output_file)
            print(f"\n[{our_symbol}] ⏭️  Already exists ({len(existing):,} candles)")
            skipped += 1
            continue
        
        # Download
        df = download_yahoo_h1(yahoo_ticker, our_symbol)
        
        if not df.empty:
            # Save
            df.to_csv(output_file, index=False)
            print(f"  💾 Saved to {output_file.name}")
            downloaded += 1
        else:
            print(f"  ❌ Failed to download")
            failed += 1
        
        # Rate limit
        time.sleep(1)
    
    # Summary
    print(f"\n{'=' * 70}")
    print("DOWNLOAD SUMMARY")
    print("=" * 70)
    print(f"✅ Downloaded: {downloaded}")
    print(f"⏭️  Skipped:    {skipped}")
    print(f"❌ Failed:     {failed}")
    
    if downloaded > 0:
        print(f"\n✅ Special assets ready for analysis!")
    
    # Note about data limitations
    if failed > 0 or downloaded > 0:
        print(f"\n⚠️  NOTE: Yahoo Finance hourly data:")
        print("  - May have gaps/missing hours")
        print("  - Limited to ~730 days for free tier")
        print("  - Crypto: 24/7 data (more candles)")
        print("  - Stocks/indices: Trading hours only")


if __name__ == "__main__":
    main()
