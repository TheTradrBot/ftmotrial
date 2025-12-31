"""
Broker Configuration Module - Multi-Broker Support

Supports switching between different MT5 brokers:
- Forex.com Demo (testing)
- 5ers Live (production)

Usage:
    from broker_config import get_broker_config, BrokerType
    
    config = get_broker_config()  # Auto-detects from environment
    # or
    config = get_broker_config(BrokerType.FOREXCOM_DEMO)
"""

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class BrokerType(Enum):
    """Supported broker types."""
    FOREXCOM_DEMO = "forexcom_demo"
    FIVEERS_LIVE = "fiveers_live"
    FIVEERS_DEMO = "fiveers_demo"


@dataclass
class BrokerConfig:
    """
    Broker-specific configuration.
    
    All trading parameters adjusted per broker/account type.
    """
    
    # =================================================================
    # BROKER IDENTIFICATION
    # =================================================================
    broker_type: BrokerType
    broker_name: str
    is_demo: bool
    
    # =================================================================
    # MT5 CONNECTION
    # =================================================================
    mt5_login: int = 0
    mt5_password: str = ""
    mt5_server: str = ""
    mt5_path: Optional[str] = None  # Path to terminal64.exe (Windows)
    
    # =================================================================
    # ACCOUNT SETTINGS
    # =================================================================
    account_size: float = 50000.0
    account_currency: str = "USD"
    
    # =================================================================
    # RISK LIMITS
    # =================================================================
    max_daily_dd_pct: float = 5.0           # Max daily drawdown
    max_total_dd_pct: float = 10.0          # Max total drawdown
    internal_daily_halt_pct: float = 3.8    # Bot halts at this level
    internal_dd_warning_pct: float = 7.9    # Bot warns at this level
    risk_per_trade_pct: float = 0.6         # Risk per trade
    
    # =================================================================
    # TRADING PARAMETERS
    # =================================================================
    signal_check_time: str = "22:05"        # UTC, after NY close
    price_check_interval: float = 2.0       # Seconds between price checks
    position_sync_interval: int = 60        # Sync every N seconds
    scan_interval_hours: int = 4            # Full scan interval
    
    # Safety limits
    max_trades_per_day: int = 15
    max_open_positions: int = 10
    max_spread_pips: float = 3.0            # Max allowed spread
    
    # Bot identification (different per broker to distinguish trades)
    magic_number: int = 50000001
    
    # =================================================================
    # CHALLENGE TARGETS (if applicable)
    # =================================================================
    step1_target_pct: float = 8.0           # Phase 1 profit target
    step2_target_pct: float = 5.0           # Phase 2 profit target
    min_profitable_days: int = 3            # Minimum profitable days
    
    # =================================================================
    # SYMBOL SETTINGS
    # =================================================================
    # Which asset classes to trade (can be limited for testing)
    trade_forex: bool = True
    trade_metals: bool = True
    trade_indices: bool = True
    trade_crypto: bool = True               # May not be available on all brokers
    
    # Symbols to exclude (broker doesn't offer or testing purposes)
    excluded_symbols: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Calculate derived values."""
        self.risk_amount = self.account_size * (self.risk_per_trade_pct / 100)
        self.max_daily_dd_usd = self.account_size * (self.max_daily_dd_pct / 100)
        self.max_total_dd_usd = self.account_size * (self.max_total_dd_pct / 100)
    
    def get_tradable_symbols(self) -> List[str]:
        """Get list of tradable symbols based on config."""
        from config import FOREX_PAIRS, METALS, INDICES, CRYPTO_ASSETS
        
        symbols = []
        if self.trade_forex:
            symbols.extend(FOREX_PAIRS)
        if self.trade_metals:
            symbols.extend(METALS)
        if self.trade_indices:
            symbols.extend(INDICES)
        if self.trade_crypto:
            symbols.extend(CRYPTO_ASSETS)
        
        # Remove excluded symbols
        symbols = [s for s in symbols if s not in self.excluded_symbols]
        
        # Remove duplicates while preserving order
        seen = set()
        unique_symbols = []
        for s in symbols:
            if s not in seen:
                seen.add(s)
                unique_symbols.append(s)
        
        return unique_symbols
    
    def print_summary(self):
        """Print configuration summary."""
        print("=" * 70)
        print(f"BROKER CONFIGURATION: {self.broker_name}")
        print("=" * 70)
        print(f"  Broker Type:     {self.broker_type.value}")
        print(f"  Demo Mode:       {'YES ⚠️' if self.is_demo else 'NO (LIVE)'}")
        print(f"  MT5 Server:      {self.mt5_server}")
        print(f"  MT5 Login:       {self.mt5_login}")
        print()
        print(f"  Account Size:    ${self.account_size:,.0f}")
        print(f"  Risk per Trade:  {self.risk_per_trade_pct}% = ${self.risk_amount:.0f}")
        print(f"  Max Daily DD:    {self.max_daily_dd_pct}% = ${self.max_daily_dd_usd:.0f}")
        print(f"  Max Total DD:    {self.max_total_dd_pct}% = ${self.max_total_dd_usd:.0f}")
        print()
        print(f"  Magic Number:    {self.magic_number}")
        print(f"  Symbols:         {len(self.get_tradable_symbols())}")
        print("=" * 70)


def get_forexcom_demo_config() -> BrokerConfig:
    """
    Get Forex.com Demo configuration.
    
    ⚠️ DEMO MODE - For testing before 5ers live!
    """
    return BrokerConfig(
        broker_type=BrokerType.FOREXCOM_DEMO,
        broker_name="Forex.com Demo",
        is_demo=True,
        
        # MT5 Connection (from environment)
        mt5_login=int(os.getenv("MT5_LOGIN", "0")),
        mt5_password=os.getenv("MT5_PASSWORD", ""),
        mt5_server=os.getenv("MT5_SERVER", "FOREX.com-Demo"),
        mt5_path=os.getenv("MT5_PATH"),
        
        # Account
        account_size=float(os.getenv("ACCOUNT_SIZE", "50000")),
        account_currency="USD",
        
        # Risk (same rules as 5ers for testing)
        max_daily_dd_pct=5.0,
        max_total_dd_pct=10.0,
        internal_daily_halt_pct=3.8,
        internal_dd_warning_pct=7.9,
        risk_per_trade_pct=0.6,
        
        # Trading
        signal_check_time="22:05",
        max_trades_per_day=15,
        max_open_positions=10,
        max_spread_pips=5.0,  # Wider spreads on demo
        magic_number=50000001,
        scan_interval_hours=int(os.getenv("SCAN_INTERVAL_HOURS", "4")),
        
        # Challenge targets (simulate 5ers)
        step1_target_pct=8.0,
        step2_target_pct=5.0,
        min_profitable_days=3,
        
        # Symbols - crypto may not be available
        trade_forex=True,
        trade_metals=True,
        trade_indices=True,
        trade_crypto=False,  # Forex.com demo may not have crypto
        excluded_symbols=[],
    )


def get_fiveers_live_config() -> BrokerConfig:
    """
    Get 5ers 60K High Stakes Live configuration.
    
    ⚠️ LIVE TRADING - Real money!
    """
    return BrokerConfig(
        broker_type=BrokerType.FIVEERS_LIVE,
        broker_name="5ers 60K High Stakes",
        is_demo=False,
        
        # MT5 Connection (from environment)
        mt5_login=int(os.getenv("MT5_LOGIN", "0")),
        mt5_password=os.getenv("MT5_PASSWORD", ""),
        mt5_server=os.getenv("MT5_SERVER", "5ersLtd-Server"),
        mt5_path=os.getenv("MT5_PATH"),
        
        # Account
        account_size=60000.0,
        account_currency="USD",
        
        # Risk
        max_daily_dd_pct=5.0,
        max_total_dd_pct=10.0,
        internal_daily_halt_pct=3.8,
        internal_dd_warning_pct=7.9,
        risk_per_trade_pct=0.6,
        
        # Trading
        signal_check_time="22:05",
        max_trades_per_day=15,
        max_open_positions=10,
        max_spread_pips=3.0,  # Tighter spread requirement for live
        magic_number=60000001,
        scan_interval_hours=int(os.getenv("SCAN_INTERVAL_HOURS", "4")),
        
        # Challenge targets
        step1_target_pct=8.0,
        step2_target_pct=5.0,
        min_profitable_days=3,
        
        # All symbols
        trade_forex=True,
        trade_metals=True,
        trade_indices=True,
        trade_crypto=True,
        excluded_symbols=[],
    )


def get_fiveers_demo_config() -> BrokerConfig:
    """Get 5ers Demo configuration (if available)."""
    config = get_fiveers_live_config()
    config.broker_type = BrokerType.FIVEERS_DEMO
    config.broker_name = "5ers Demo"
    config.is_demo = True
    config.magic_number = 60000002
    return config


def get_broker_config(broker_type: Optional[BrokerType] = None) -> BrokerConfig:
    """
    Get broker configuration.
    
    If broker_type is None, auto-detects from BROKER_TYPE environment variable.
    
    Args:
        broker_type: Explicit broker type, or None to auto-detect
    
    Returns:
        BrokerConfig for the specified/detected broker
    """
    if broker_type is None:
        # Auto-detect from environment
        broker_env = os.getenv("BROKER_TYPE", "forexcom_demo").lower()
        
        if broker_env in ("forexcom_demo", "forexcom", "forex.com"):
            broker_type = BrokerType.FOREXCOM_DEMO
        elif broker_env in ("fiveers_live", "5ers_live", "5ers", "fiveers"):
            broker_type = BrokerType.FIVEERS_LIVE
        elif broker_env in ("fiveers_demo", "5ers_demo"):
            broker_type = BrokerType.FIVEERS_DEMO
        else:
            print(f"[broker_config] Unknown BROKER_TYPE: {broker_env}, defaulting to forexcom_demo")
            broker_type = BrokerType.FOREXCOM_DEMO
    
    if broker_type == BrokerType.FOREXCOM_DEMO:
        return get_forexcom_demo_config()
    elif broker_type == BrokerType.FIVEERS_LIVE:
        return get_fiveers_live_config()
    elif broker_type == BrokerType.FIVEERS_DEMO:
        return get_fiveers_demo_config()
    else:
        raise ValueError(f"Unknown broker type: {broker_type}")


def print_broker_comparison():
    """Print comparison of all broker configurations."""
    print("\n" + "=" * 80)
    print("BROKER CONFIGURATION COMPARISON")
    print("=" * 80)
    
    configs = [
        get_forexcom_demo_config(),
        get_fiveers_live_config(),
    ]
    
    # Header
    print(f"{'Metric':<25} | {'Forex.com Demo':>18} | {'5ers Live':>18}")
    print("-" * 70)
    
    # Comparison rows
    rows = [
        ("Account Size", lambda c: f"${c.account_size:,.0f}"),
        ("Is Demo", lambda c: "YES" if c.is_demo else "NO"),
        ("Risk per Trade", lambda c: f"{c.risk_per_trade_pct}%"),
        ("Risk Amount", lambda c: f"${c.risk_amount:.0f}"),
        ("Max Daily DD", lambda c: f"{c.max_daily_dd_pct}% (${c.max_daily_dd_usd:.0f})"),
        ("Max Total DD", lambda c: f"{c.max_total_dd_pct}% (${c.max_total_dd_usd:.0f})"),
        ("Halt Level", lambda c: f"{c.internal_daily_halt_pct}%"),
        ("Max Spread (pips)", lambda c: f"{c.max_spread_pips}"),
        ("Magic Number", lambda c: str(c.magic_number)),
        ("Trade Crypto", lambda c: "YES" if c.trade_crypto else "NO"),
    ]
    
    for label, getter in rows:
        vals = [getter(c) for c in configs]
        print(f"{label:<25} | {vals[0]:>18} | {vals[1]:>18}")
    
    print("=" * 80)


if __name__ == "__main__":
    print("\n🔧 Broker Configuration Module\n")
    
    # Show current config
    config = get_broker_config()
    config.print_summary()
    
    print("\n📊 Broker Comparison:")
    print_broker_comparison()
    
    print("\n📋 Tradable Symbols:")
    symbols = config.get_tradable_symbols()
    print(f"   Total: {len(symbols)} symbols")
    for i, s in enumerate(symbols, 1):
        if i <= 10:
            print(f"   {i:2d}. {s}")
        elif i == 11:
            print(f"   ... and {len(symbols) - 10} more")
            break
