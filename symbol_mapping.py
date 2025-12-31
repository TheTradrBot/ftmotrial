"""
Symbol Mapping for Multi-Broker Support

Maps internal OANDA-style symbols (EUR_USD) to broker-specific symbols.

Supported Brokers:
- 5ers (FTMO-style): EURUSD, US500.cash
- Forex.com: EURUSD, US500

Usage:
    from symbol_mapping import get_broker_symbol, get_internal_symbol
    
    # Convert internal -> broker
    broker_sym = get_broker_symbol("EUR_USD", "forexcom")  # -> "EURUSD"
    
    # Convert broker -> internal
    internal_sym = get_internal_symbol("EURUSD", "forexcom")  # -> "EUR_USD"
"""

from typing import Dict, List, Tuple, Optional
from enum import Enum


class BrokerName(Enum):
    """Supported broker names for symbol mapping."""
    FIVEERS = "fiveers"      # 5ers (FTMO-style symbols)
    FOREXCOM = "forexcom"    # Forex.com


# =============================================================================
# INTERNAL (OANDA-STYLE) SYMBOL LISTS
# =============================================================================

ALL_FOREX_PAIRS_OANDA: List[str] = [
    "EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF", "USD_CAD", "AUD_USD", "NZD_USD",
    "EUR_GBP", "EUR_JPY", "EUR_CHF", "EUR_AUD", "EUR_CAD", "EUR_NZD",
    "GBP_JPY", "GBP_CHF", "GBP_AUD", "GBP_CAD", "GBP_NZD",
    "AUD_JPY", "AUD_CHF", "AUD_CAD", "AUD_NZD",
    "NZD_JPY", "NZD_CHF", "NZD_CAD",
    "CAD_JPY", "CAD_CHF", "CHF_JPY",
]

ALL_METALS_OANDA: List[str] = ["XAU_USD", "XAG_USD"]
ALL_CRYPTO_OANDA: List[str] = ["BTC_USD", "ETH_USD"]
ALL_INDICES_OANDA: List[str] = ["SPX500_USD", "NAS100_USD"]

ALL_TRADABLE_OANDA: List[str] = (
    ALL_FOREX_PAIRS_OANDA + ALL_METALS_OANDA + ALL_CRYPTO_OANDA + ALL_INDICES_OANDA
)


# =============================================================================
# 5ERS (FTMO-STYLE) SYMBOL MAPPING
# =============================================================================

OANDA_TO_FIVEERS: Dict[str, str] = {
    # ============ FOREX MAJORS (7) ============
    "EUR_USD": "EURUSD",
    "GBP_USD": "GBPUSD",
    "USD_JPY": "USDJPY",
    "USD_CHF": "USDCHF",
    "USD_CAD": "USDCAD",
    "AUD_USD": "AUDUSD",
    "NZD_USD": "NZDUSD",
    
    # ============ EUR CROSSES (6) ============
    "EUR_GBP": "EURGBP",
    "EUR_JPY": "EURJPY",
    "EUR_CHF": "EURCHF",
    "EUR_AUD": "EURAUD",
    "EUR_CAD": "EURCAD",
    "EUR_NZD": "EURNZD",
    
    # ============ GBP CROSSES (5) ============
    "GBP_JPY": "GBPJPY",
    "GBP_CHF": "GBPCHF",
    "GBP_AUD": "GBPAUD",
    "GBP_CAD": "GBPCAD",
    "GBP_NZD": "GBPNZD",
    
    # ============ AUD CROSSES (4) ============
    "AUD_JPY": "AUDJPY",
    "AUD_CHF": "AUDCHF",
    "AUD_CAD": "AUDCAD",
    "AUD_NZD": "AUDNZD",
    
    # ============ NZD CROSSES (3) ============
    "NZD_JPY": "NZDJPY",
    "NZD_CHF": "NZDCHF",
    "NZD_CAD": "NZDCAD",
    
    # ============ CAD/CHF CROSSES (3) ============
    "CAD_JPY": "CADJPY",
    "CAD_CHF": "CADCHF",
    "CHF_JPY": "CHFJPY",
    
    # ============ METALS (2) ============
    "XAU_USD": "XAUUSD",
    "XAG_USD": "XAGUSD",
    
    # ============ CRYPTO (2) ============
    "BTC_USD": "BTCUSD",
    "ETH_USD": "ETHUSD",
    
    # ============ INDICES (2) ============
    "SPX500_USD": "US500.cash",
    "NAS100_USD": "US100.cash",
}


# =============================================================================
# FOREX.COM SYMBOL MAPPING
# =============================================================================
# ⚠️ VERIFY THESE BY RUNNING validate_broker_symbols.py FIRST!
# Forex.com symbol names may vary by account type/region.

OANDA_TO_FOREXCOM: Dict[str, str] = {
    # ============ FOREX MAJORS (7) ============
    "EUR_USD": "EURUSD",
    "GBP_USD": "GBPUSD",
    "USD_JPY": "USDJPY",
    "USD_CHF": "USDCHF",
    "USD_CAD": "USDCAD",
    "AUD_USD": "AUDUSD",
    "NZD_USD": "NZDUSD",
    
    # ============ EUR CROSSES (6) ============
    "EUR_GBP": "EURGBP",
    "EUR_JPY": "EURJPY",
    "EUR_CHF": "EURCHF",
    "EUR_AUD": "EURAUD",
    "EUR_CAD": "EURCAD",
    "EUR_NZD": "EURNZD",
    
    # ============ GBP CROSSES (5) ============
    "GBP_JPY": "GBPJPY",
    "GBP_CHF": "GBPCHF",
    "GBP_AUD": "GBPAUD",
    "GBP_CAD": "GBPCAD",
    "GBP_NZD": "GBPNZD",
    
    # ============ AUD CROSSES (4) ============
    "AUD_JPY": "AUDJPY",
    "AUD_CHF": "AUDCHF",
    "AUD_CAD": "AUDCAD",
    "AUD_NZD": "AUDNZD",
    
    # ============ NZD CROSSES (3) ============
    "NZD_JPY": "NZDJPY",
    "NZD_CHF": "NZDCHF",
    "NZD_CAD": "NZDCAD",
    
    # ============ CAD/CHF CROSSES (3) ============
    "CAD_JPY": "CADJPY",
    "CAD_CHF": "CADCHF",
    "CHF_JPY": "CHFJPY",
    
    # ============ METALS (2) ============
    # Forex.com variations: XAUUSD, XAU/USD, Gold - verify!
    "XAU_USD": "XAUUSD",
    "XAG_USD": "XAGUSD",
    
    # ============ CRYPTO (2) ============
    # ⚠️ Forex.com demo may NOT offer crypto!
    "BTC_USD": "BTCUSD",
    "ETH_USD": "ETHUSD",
    
    # ============ INDICES (2) ============
    # Forex.com variations: US500, SPX500, USA500 - verify!
    "SPX500_USD": "US500",
    "NAS100_USD": "USTEC",  # or NAS100, US100, USTEC100
}


# =============================================================================
# BROKER MAPPING REGISTRY
# =============================================================================

BROKER_MAPPINGS: Dict[str, Dict[str, str]] = {
    "fiveers": OANDA_TO_FIVEERS,
    "5ers": OANDA_TO_FIVEERS,
    "ftmo": OANDA_TO_FIVEERS,
    "forexcom": OANDA_TO_FOREXCOM,
    "forex.com": OANDA_TO_FOREXCOM,
}


# =============================================================================
# LEGACY COMPATIBILITY (FTMO/5ers as default)
# =============================================================================

OANDA_TO_FTMO: Dict[str, str] = OANDA_TO_FIVEERS  # Alias for backward compatibility

FTMO_TO_OANDA: Dict[str, str] = {v: k for k, v in OANDA_TO_FIVEERS.items()}

# Reverse mappings for all brokers
FIVEERS_TO_OANDA: Dict[str, str] = {v: k for k, v in OANDA_TO_FIVEERS.items()}
FOREXCOM_TO_OANDA: Dict[str, str] = {v: k for k, v in OANDA_TO_FOREXCOM.items()}


# =============================================================================
# BROKER-AWARE CONVERSION FUNCTIONS
# =============================================================================

def get_broker_symbol(internal_symbol: str, broker: str = "fiveers") -> str:
    """
    Convert internal (OANDA-style) symbol to broker-specific symbol.
    
    Args:
        internal_symbol: Internal symbol (e.g., "EUR_USD")
        broker: Broker name ("fiveers", "forexcom", etc.)
    
    Returns:
        Broker-specific symbol (e.g., "EURUSD")
    """
    broker_lower = broker.lower()
    mapping = BROKER_MAPPINGS.get(broker_lower, OANDA_TO_FIVEERS)
    
    if internal_symbol in mapping:
        return mapping[internal_symbol]
    
    # Fallback: remove underscores
    return internal_symbol.replace("_", "")


def get_internal_symbol(broker_symbol: str, broker: str = "fiveers") -> str:
    """
    Convert broker-specific symbol to internal (OANDA-style) symbol.
    
    Args:
        broker_symbol: Broker symbol (e.g., "EURUSD")
        broker: Broker name ("fiveers", "forexcom", etc.)
    
    Returns:
        Internal symbol (e.g., "EUR_USD")
    """
    broker_lower = broker.lower()
    
    # Get reverse mapping
    if broker_lower in ("fiveers", "5ers", "ftmo"):
        reverse_map = FIVEERS_TO_OANDA
    elif broker_lower in ("forexcom", "forex.com"):
        reverse_map = FOREXCOM_TO_OANDA
    else:
        reverse_map = FIVEERS_TO_OANDA
    
    if broker_symbol in reverse_map:
        return reverse_map[broker_symbol]
    
    # Fallback: try to parse 6-char forex
    if len(broker_symbol) == 6:
        return f"{broker_symbol[:3]}_{broker_symbol[3:]}"
    
    return broker_symbol


def get_symbol_map_for_broker(broker: str = "fiveers") -> Dict[str, str]:
    """
    Get the complete symbol mapping for a broker.
    
    Args:
        broker: Broker name ("fiveers", "forexcom", etc.)
    
    Returns:
        Dict mapping internal -> broker symbols
    """
    broker_lower = broker.lower()
    return BROKER_MAPPINGS.get(broker_lower, OANDA_TO_FIVEERS).copy()


# =============================================================================
# LEGACY SYMBOL LISTS (for backward compatibility)
# =============================================================================

ALL_FOREX_PAIRS_FTMO: List[str] = [OANDA_TO_FIVEERS[s] for s in ALL_FOREX_PAIRS_OANDA]
ALL_METALS_FTMO: List[str] = ["XAUUSD", "XAGUSD"]
ALL_CRYPTO_FTMO: List[str] = ["BTCUSD", "ETHUSD"]
ALL_INDICES_FTMO: List[str] = ["US500.cash", "US100.cash"]

ALL_TRADABLE_FTMO: List[str] = (
    ALL_FOREX_PAIRS_FTMO + ALL_METALS_FTMO + ALL_CRYPTO_FTMO + ALL_INDICES_FTMO
)


def oanda_to_ftmo(symbol: str) -> str:
    """
    Convert OANDA symbol name to FTMO MT5 symbol name.
    LEGACY: Use get_broker_symbol() for multi-broker support.
    """
    return get_broker_symbol(symbol, "fiveers")


def ftmo_to_oanda(symbol: str) -> str:
    """
    Convert FTMO MT5 symbol name to OANDA symbol name.
    LEGACY: Use get_internal_symbol() for multi-broker support.
    """
    return get_internal_symbol(symbol, "fiveers")


def get_contract_specs() -> Dict[str, Dict]:
    """Get contract specifications for all tradable symbols (OANDA format)."""
    return {
        "EUR_USD": {"pip_value": 0.0001, "contract_size": 100000, "pip_location": 4},
        "GBP_USD": {"pip_value": 0.0001, "contract_size": 100000, "pip_location": 4},
        "USD_JPY": {"pip_value": 0.01, "contract_size": 100000, "pip_location": 2},
        "USD_CHF": {"pip_value": 0.0001, "contract_size": 100000, "pip_location": 4},
        "USD_CAD": {"pip_value": 0.0001, "contract_size": 100000, "pip_location": 4},
        "AUD_USD": {"pip_value": 0.0001, "contract_size": 100000, "pip_location": 4},
        "NZD_USD": {"pip_value": 0.0001, "contract_size": 100000, "pip_location": 4},
        "EUR_GBP": {"pip_value": 0.0001, "contract_size": 100000, "pip_location": 4},
        "EUR_JPY": {"pip_value": 0.01, "contract_size": 100000, "pip_location": 2},
        "EUR_CHF": {"pip_value": 0.0001, "contract_size": 100000, "pip_location": 4},
        "EUR_AUD": {"pip_value": 0.0001, "contract_size": 100000, "pip_location": 4},
        "EUR_CAD": {"pip_value": 0.0001, "contract_size": 100000, "pip_location": 4},
        "EUR_NZD": {"pip_value": 0.0001, "contract_size": 100000, "pip_location": 4},
        "GBP_JPY": {"pip_value": 0.01, "contract_size": 100000, "pip_location": 2},
        "GBP_CHF": {"pip_value": 0.0001, "contract_size": 100000, "pip_location": 4},
        "GBP_AUD": {"pip_value": 0.0001, "contract_size": 100000, "pip_location": 4},
        "GBP_CAD": {"pip_value": 0.0001, "contract_size": 100000, "pip_location": 4},
        "GBP_NZD": {"pip_value": 0.0001, "contract_size": 100000, "pip_location": 4},
        "AUD_JPY": {"pip_value": 0.01, "contract_size": 100000, "pip_location": 2},
        "AUD_CHF": {"pip_value": 0.0001, "contract_size": 100000, "pip_location": 4},
        "AUD_CAD": {"pip_value": 0.0001, "contract_size": 100000, "pip_location": 4},
        "AUD_NZD": {"pip_value": 0.0001, "contract_size": 100000, "pip_location": 4},
        "NZD_JPY": {"pip_value": 0.01, "contract_size": 100000, "pip_location": 2},
        "NZD_CHF": {"pip_value": 0.0001, "contract_size": 100000, "pip_location": 4},
        "NZD_CAD": {"pip_value": 0.0001, "contract_size": 100000, "pip_location": 4},
        "CAD_JPY": {"pip_value": 0.01, "contract_size": 100000, "pip_location": 2},
        "CAD_CHF": {"pip_value": 0.0001, "contract_size": 100000, "pip_location": 4},
        "CHF_JPY": {"pip_value": 0.01, "contract_size": 100000, "pip_location": 2},
        "XAU_USD": {"pip_value": 0.01, "contract_size": 100, "pip_location": 2},
        "XAG_USD": {"pip_value": 0.001, "contract_size": 5000, "pip_location": 3},
        "BTC_USD": {"pip_value": 1.0, "contract_size": 1, "pip_location": 0},
        "ETH_USD": {"pip_value": 0.01, "contract_size": 1, "pip_location": 2},
        "SPX500_USD": {"pip_value": 0.1, "contract_size": 1, "pip_location": 1},
        "NAS100_USD": {"pip_value": 0.1, "contract_size": 1, "pip_location": 1},
    }


def print_summary(broker: str = "fiveers"):
    """Print a summary of all tradable symbols for a specific broker."""
    print("=" * 70)
    print(f"TRADABLE SYMBOLS SUMMARY - {broker.upper()}")
    print("=" * 70)
    
    print(f"\nForex Pairs: {len(ALL_FOREX_PAIRS_OANDA)}")
    for i, oanda in enumerate(ALL_FOREX_PAIRS_OANDA, 1):
        broker_sym = get_broker_symbol(oanda, broker)
        print(f"  {i:2d}. {oanda:10s} -> {broker_sym}")
    
    print(f"\nMetals: {len(ALL_METALS_OANDA)}")
    for oanda in ALL_METALS_OANDA:
        broker_sym = get_broker_symbol(oanda, broker)
        print(f"      {oanda:10s} -> {broker_sym}")
    
    print(f"\nCrypto: {len(ALL_CRYPTO_OANDA)}")
    for oanda in ALL_CRYPTO_OANDA:
        broker_sym = get_broker_symbol(oanda, broker)
        print(f"      {oanda:10s} -> {broker_sym}")
    
    print(f"\nIndices: {len(ALL_INDICES_OANDA)}")
    for oanda in ALL_INDICES_OANDA:
        broker_sym = get_broker_symbol(oanda, broker)
        print(f"      {oanda:15s} -> {broker_sym}")
    
    print(f"\nTotal: {len(ALL_TRADABLE_OANDA)} symbols")
    print("=" * 70)


def print_broker_comparison():
    """Print symbol mapping comparison between brokers."""
    print("\n" + "=" * 70)
    print("SYMBOL MAPPING COMPARISON: 5ERS vs FOREX.COM")
    print("=" * 70)
    
    # Symbols that differ
    differences = []
    for oanda in ALL_TRADABLE_OANDA:
        fiveers_sym = get_broker_symbol(oanda, "fiveers")
        forexcom_sym = get_broker_symbol(oanda, "forexcom")
        if fiveers_sym != forexcom_sym:
            differences.append((oanda, fiveers_sym, forexcom_sym))
    
    if differences:
        print(f"\n{'Internal':<15} | {'5ers':<15} | {'Forex.com':<15}")
        print("-" * 50)
        for oanda, fiveers, forexcom in differences:
            print(f"{oanda:<15} | {fiveers:<15} | {forexcom:<15}")
    else:
        print("\nNo differences - all symbols map identically!")
    
    print("=" * 70)


if __name__ == "__main__":
    import sys
    broker = sys.argv[1] if len(sys.argv) > 1 else "fiveers"
    print_summary(broker)
    print_broker_comparison()
