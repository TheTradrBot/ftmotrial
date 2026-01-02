"""
Parameter Loader - Single Source of Truth for Strategy Parameters.

This module loads optimized parameters from params/current_params.json.
All trading components (live bot, backtests) must use this loader.

Usage:
    from params.params_loader import load_strategy_params, get_transaction_costs
    
    params = load_strategy_params()  # Returns StrategyParams object
    costs = get_transaction_costs("EURUSD")  # Returns spread, slippage, commission
"""

import json
from pathlib import Path
from typing import Dict, Optional, Any, Tuple
from dataclasses import dataclass


PARAMS_FILE = Path(__file__).parent / "current_params.json"


class ParamsNotFoundError(Exception):
    """Raised when params file doesn't exist. Run optimizer first."""
    pass


def load_params_dict() -> Dict[str, Any]:
    """
    Load raw parameters dictionary from JSON file.
    
    Returns:
        Dict with all parameters (extracted from 'parameters' key if present)
        
    Raises:
        ParamsNotFoundError: If params file doesn't exist
    """
    if not PARAMS_FILE.exists():
        raise ParamsNotFoundError(
            f"Parameters file not found: {PARAMS_FILE}\n"
            "Run the optimizer first: python ftmo_challenge_analyzer.py"
        )
    
    with open(PARAMS_FILE, 'r') as f:
        data = json.load(f)
    
    # Handle new format: params are under 'parameters' key
    if "parameters" in data:
        return data["parameters"]
    
    # Old format: params at root level
    return data


def load_strategy_params():
    """
    Load optimized strategy parameters.
    
    Returns:
        StrategyParams object with optimized values
        
    Raises:
        ParamsNotFoundError: If params file doesn't exist
    """
    from strategy_core import StrategyParams
    
    data = load_params_dict()
    
    return StrategyParams(
        min_confluence=data.get("min_confluence", 5),
        min_quality_factors=data.get("min_quality_factors", 3),
        atr_sl_multiplier=data.get("atr_sl_multiplier", 1.5),
        atr_tp1_multiplier=data.get("atr_tp1_multiplier", 0.6),
        atr_tp2_multiplier=data.get("atr_tp2_multiplier", 1.2),
        atr_tp3_multiplier=data.get("atr_tp3_multiplier", 2.0),
        atr_tp4_multiplier=data.get("atr_tp4_multiplier", 3.0),
        atr_tp5_multiplier=data.get("atr_tp5_multiplier", 4.0),
        fib_low=data.get("fib_low", 0.382),
        fib_high=data.get("fib_high", 0.886),
        structure_sl_lookback=data.get("structure_sl_lookback", 35),
        # REMOVED: liquidity_sweep_lookback - doesn't exist in StrategyParams
        use_htf_filter=data.get("use_htf_filter", False),  # DISABLED by default for baseline
        use_structure_filter=data.get("use_structure_filter", False),  # DISABLED by default
        # REMOVED: use_liquidity_filter - doesn't exist in StrategyParams
        use_fib_filter=data.get("use_fib_filter", False),  # DISABLED by default
        use_confirmation_filter=data.get("use_confirmation_filter", False),  # DISABLED by default
        require_htf_alignment=data.get("require_htf_alignment", False),
        require_confirmation_for_active=data.get("require_confirmation_for_active", True),
        require_rr_for_active=data.get("require_rr_for_active", True),
        min_rr_ratio=data.get("min_rr_ratio", 1.0),
        risk_per_trade_pct=data.get("risk_per_trade_pct", 0.5),
        cooldown_bars=data.get("cooldown_bars", 0),
        max_open_trades=data.get("max_open_trades", 3),
        tp1_close_pct=data.get("tp1_close_pct", 0.10),
        tp2_close_pct=data.get("tp2_close_pct", 0.10),
        tp3_close_pct=data.get("tp3_close_pct", 0.15),
        tp4_close_pct=data.get("tp4_close_pct", 0.20),
        tp5_close_pct=data.get("tp5_close_pct", 0.45),
        use_atr_regime_filter=data.get("use_atr_regime_filter", False),  # DISABLED by default
        atr_min_percentile=data.get("atr_min_percentile", 60.0),
        use_zscore_filter=data.get("use_zscore_filter", False),  # DISABLED by default
        zscore_threshold=data.get("zscore_threshold", 1.5),
        use_pattern_filter=data.get("use_pattern_filter", False),  # DISABLED by default
        # Regime Adaptive V2 parameters
        adx_trend_threshold=data.get("adx_trend_threshold", 25.0),
        adx_range_threshold=data.get("adx_range_threshold", 20.0),
        trend_min_confluence=data.get("trend_min_confluence", 4),
        range_min_confluence=data.get("range_min_confluence", 3),
        atr_volatility_ratio=data.get("atr_volatility_ratio", 0.8),
        partial_exit_at_1r=data.get("partial_exit_at_1r", True),
        partial_exit_pct=data.get("partial_exit_pct", 0.5),
        atr_trail_multiplier=data.get("atr_trail_multiplier", 1.5),
        trail_activation_r=data.get("trail_activation_r", 2.2),
        december_atr_multiplier=data.get("december_atr_multiplier", 1.5),
        volatile_asset_boost=data.get("volatile_asset_boost", 1.5),
        # ADX slope toggle
        use_adx_slope_rising=data.get("use_adx_slope_rising", False),
        # Blueprint V2 filters
        use_mitigated_sr=data.get("use_mitigated_sr", False),
        use_structural_framework=data.get("use_structural_framework", False),
        use_displacement_filter=data.get("use_displacement_filter", False),
        use_candle_rejection=data.get("use_candle_rejection", False),
        # Session filter & graduated risk
        use_session_filter=data.get("use_session_filter", True),
        session_start_utc=data.get("session_start_utc", 8),
        session_end_utc=data.get("session_end_utc", 22),
        use_graduated_risk=data.get("use_graduated_risk", True),
        tier1_dd_pct=data.get("tier1_dd_pct", 2.0),
        tier1_risk_factor=data.get("tier1_risk_factor", 0.67),
        tier2_dd_pct=data.get("tier2_dd_pct", 3.5),
        tier3_dd_pct=data.get("tier3_dd_pct", 4.5),
    )


def get_min_confluence() -> int:
    """Get minimum confluence score from params."""
    data = load_params_dict()
    return data.get("min_confluence", 5)


def get_max_concurrent_trades() -> int:
    """Get maximum concurrent trades from params."""
    data = load_params_dict()
    return data.get("max_concurrent_trades", 7)


def get_risk_per_trade_pct() -> float:
    """Get risk per trade percentage from params."""
    data = load_params_dict()
    return data.get("risk_per_trade_pct", 0.5)


def get_transaction_costs(symbol: str) -> Tuple[float, float, float]:
    """
    Get transaction costs for a symbol.
    
    Args:
        symbol: Trading symbol (any format - EURUSD, EUR_USD, etc)
        
    Returns:
        Tuple of (spread_pips, slippage_pips, commission_per_lot)
    """
    data = load_params_dict()
    costs = data.get("transaction_costs", {})
    
    normalized = symbol.replace("_", "").replace(".", "").replace("/", "").upper()
    
    spread_config = costs.get("spread_pips", {})
    spread = spread_config.get(normalized, spread_config.get("default", 2.5))
    slippage = costs.get("slippage_pips", 5.0)  # OPTIMIZED: Increased from 1.0 to 5.0 pips for realistic execution
    commission = costs.get("commission_per_lot", 7.0)
    
    return spread, slippage, commission


def save_optimized_params(
    params_dict: Dict[str, Any],
    backup: bool = True
) -> Path:
    """
    Save optimized parameters to JSON file.
    
    Args:
        params_dict: Dictionary of optimized parameters
        backup: Whether to create backup in history folder
        
    Returns:
        Path to saved file
    """
    from datetime import datetime
    
    params_dict["generated_at"] = datetime.utcnow().isoformat() + "Z"
    params_dict["generated_by"] = "ftmo_challenge_analyzer.py"
    
    if "version" not in params_dict:
        params_dict["version"] = "1.0.0"
    
    with open(PARAMS_FILE, 'w') as f:
        json.dump(params_dict, f, indent=2)
    
    if backup:
        history_dir = Path(__file__).parent / "history"
        history_dir.mkdir(exist_ok=True)
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        backup_path = history_dir / f"params_{timestamp}.json"
        with open(backup_path, 'w') as f:
            json.dump(params_dict, f, indent=2)
    
    return PARAMS_FILE
