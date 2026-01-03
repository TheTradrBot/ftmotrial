"""
Parameter Loader - Single Source of Truth for Strategy Parameters.

This module provides the CANONICAL parameter loading mechanism for ALL trading components.
It enforces a strict separation between optimization results and production deployment.

ARCHITECTURE:
============
1. PRODUCTION_PARAMS.json  - LOCKED production parameters (used by main_live_bot.py)
2. current_params.json     - Latest optimization output (NOT auto-deployed to production)
3. ftmo_analysis_output/   - Full optimization history with audit trail

AUDIT TRAIL:
============
- PRODUCTION_PARAMS.json contains full provenance: source run, validation metrics, approval
- Production params are NEVER auto-updated by optimizer
- Manual review + promote_to_production() required for changes
- Checksum verification available via verify_production_params()

Usage:
    # For LIVE TRADING (production mode):
    from params.params_loader import load_production_params
    params = load_production_params()  # Fails if not properly configured
    
    # For BACKTESTING/OPTIMIZATION:
    from params.params_loader import load_strategy_params
    params = load_strategy_params()  # Uses current_params.json
    
    # For AUDIT:
    from params.params_loader import get_production_audit_info
    audit = get_production_audit_info()  # Full provenance
"""

import json
import hashlib
from pathlib import Path
from typing import Dict, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime


# File paths
PARAMS_DIR = Path(__file__).parent
PARAMS_FILE = PARAMS_DIR / "current_params.json"
PRODUCTION_PARAMS_FILE = PARAMS_DIR / "PRODUCTION_PARAMS.json"
HISTORY_DIR = PARAMS_DIR / "history"


class ParamsNotFoundError(Exception):
    """Raised when params file doesn't exist. Run optimizer first."""
    pass


class ProductionParamsError(Exception):
    """Raised when production params are invalid or not properly configured."""
    pass


# =============================================================================
# PRODUCTION PARAMETER LOADING (for main_live_bot.py)
# =============================================================================

def load_production_params():
    """
    Load PRODUCTION parameters for live trading.
    
    This is the ONLY function that should be used by main_live_bot.py.
    It enforces strict validation and provides full audit trail.
    
    Returns:
        StrategyParams object configured for production
        
    Raises:
        ProductionParamsError: If production params missing or invalid
    """
    from strategy_core import StrategyParams
    
    if not PRODUCTION_PARAMS_FILE.exists():
        raise ProductionParamsError(
            f"Production parameters not found: {PRODUCTION_PARAMS_FILE}\n"
            "Run: python -m params.promote_to_production to deploy optimized params"
        )
    
    with open(PRODUCTION_PARAMS_FILE, 'r') as f:
        data = json.load(f)
    
    # Verify production lock
    if not data.get("PRODUCTION_LOCKED"):
        raise ProductionParamsError(
            "Production params not locked! Set PRODUCTION_LOCKED: true after review."
        )
    
    # Verify approval
    if not data.get("validation", {}).get("approved"):
        raise ProductionParamsError(
            "Production params not approved! Review validation metrics first."
        )
    
    params = data.get("parameters", {})
    
    # Filter out documentation keys (starting with _)
    clean_params = {k: v for k, v in params.items() if not k.startswith("_")}
    
    return _build_strategy_params(clean_params)


def get_production_audit_info() -> Dict[str, Any]:
    """
    Get full audit information for production parameters.
    
    Returns:
        Dict with source, validation, deployment info
    """
    if not PRODUCTION_PARAMS_FILE.exists():
        return {"error": "No production params configured"}
    
    with open(PRODUCTION_PARAMS_FILE, 'r') as f:
        data = json.load(f)
    
    return {
        "production_version": data.get("production_version"),
        "deployed_at": data.get("deployed_at"),
        "source": data.get("source", {}),
        "validation": data.get("validation", {}),
        "params_hash": calculate_params_hash(data.get("parameters", {}))
    }


def verify_production_params() -> Tuple[bool, str]:
    """
    Verify production params integrity.
    
    Returns:
        (valid, message) tuple
    """
    try:
        audit = get_production_audit_info()
        if "error" in audit:
            return False, audit["error"]
        
        # Load and verify
        load_production_params()
        
        return True, f"Production params valid. Version: {audit['production_version']}, Hash: {audit['params_hash'][:16]}"
    except Exception as e:
        return False, f"Verification failed: {e}"


def calculate_params_hash(params: Dict) -> str:
    """Calculate SHA256 hash of parameters for integrity verification."""
    # Filter documentation keys
    clean = {k: v for k, v in params.items() if not k.startswith("_")}
    params_str = json.dumps(clean, sort_keys=True)
    return hashlib.sha256(params_str.encode()).hexdigest()


# =============================================================================
# OPTIMIZATION PARAMETER LOADING (for backtesting/development)
# =============================================================================

def load_params_dict() -> Dict[str, Any]:
    """
    Load raw parameters dictionary from current_params.json (optimization output).
    
    NOTE: For live trading, use load_production_params() instead!
    
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


def _build_strategy_params(data: Dict[str, Any]):
    """
    Build StrategyParams from dictionary - shared logic for all loaders.
    
    Handles naming mismatches (min_confluence_score → min_confluence, etc.)
    """
    from strategy_core import StrategyParams
    
    # Handle naming mismatch from optimizer
    if 'min_confluence_score' in data and 'min_confluence' not in data:
        data = data.copy()
        data['min_confluence'] = data.pop('min_confluence_score')
    
    return StrategyParams(
        min_confluence=data.get("min_confluence", 5),
        min_quality_factors=data.get("min_quality_factors", 3),
        atr_sl_multiplier=data.get("atr_sl_multiplier", 1.5),
        atr_tp1_multiplier=data.get("tp1_r_multiple", data.get("atr_tp1_multiplier", 0.6)),
        atr_tp2_multiplier=data.get("tp2_r_multiple", data.get("atr_tp2_multiplier", 1.2)),
        atr_tp3_multiplier=data.get("tp3_r_multiple", data.get("atr_tp3_multiplier", 2.0)),
        atr_tp4_multiplier=data.get("atr_tp4_multiplier", 3.0),
        atr_tp5_multiplier=data.get("atr_tp5_multiplier", 4.0),
        fib_low=data.get("fib_low", 0.382),
        fib_high=data.get("fib_high", 0.886),
        structure_sl_lookback=data.get("structure_sl_lookback", 35),
        use_htf_filter=data.get("use_htf_filter", False),
        use_structure_filter=data.get("use_structure_filter", False),
        use_fib_filter=data.get("use_fib_filter", False),
        use_confirmation_filter=data.get("use_confirmation_filter", False),
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
        use_atr_regime_filter=data.get("use_atr_regime_filter", False),
        atr_min_percentile=data.get("atr_min_percentile", 60.0),
        use_zscore_filter=data.get("use_zscore_filter", False),
        zscore_threshold=data.get("zscore_threshold", 1.5),
        use_pattern_filter=data.get("use_pattern_filter", False),
        adx_trend_threshold=data.get("adx_trend_threshold", 25.0),
        adx_range_threshold=data.get("adx_range_threshold", 20.0),
        trend_min_confluence=data.get("trend_min_confluence", 4),
        range_min_confluence=data.get("range_min_confluence", 3),
        atr_volatility_ratio=data.get("atr_vol_ratio_range", data.get("atr_volatility_ratio", 0.8)),
        atr_vol_ratio_range=data.get("atr_vol_ratio_range", 0.8),
        atr_trail_multiplier=data.get("atr_trail_multiplier", 1.5),
        trail_activation_r=data.get("trail_activation_r", 2.2),
        december_atr_multiplier=data.get("december_atr_multiplier", 1.5),
        volatile_asset_boost=data.get("volatile_asset_boost", 1.5),
        use_adx_slope_rising=data.get("use_adx_slope_rising", False),
        use_mitigated_sr=data.get("use_mitigated_sr", False),
        use_structural_framework=data.get("use_structural_framework", False),
        use_displacement_filter=data.get("use_displacement_filter", False),
        use_candle_rejection=data.get("use_candle_rejection", False),
        use_session_filter=data.get("use_session_filter", True),
        session_start_utc=data.get("session_start_utc", 8),
        session_end_utc=data.get("session_end_utc", 22),
        use_graduated_risk=data.get("use_graduated_risk", True),
        tier1_dd_pct=data.get("tier1_dd_pct", 2.0),
        tier1_risk_factor=data.get("tier1_risk_factor", 0.67),
        tier2_dd_pct=data.get("tier2_dd_pct", 3.5),
        tier3_dd_pct=data.get("tier3_dd_pct", 4.5),
    )


def load_strategy_params():
    """
    Load strategy parameters for BACKTESTING/OPTIMIZATION.
    
    NOTE: For live trading, use load_production_params() instead!
    """
    data = load_params_dict()
    return _build_strategy_params(data)


def get_min_confluence() -> int:
    """Get minimum confluence score from params."""
    data = load_params_dict()
    # Handle naming mismatch
    return data.get("min_confluence", data.get("min_confluence_score", 5))


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
    slippage = costs.get("slippage_pips", 5.0)
    commission = costs.get("commission_per_lot", 7.0)
    
    return spread, slippage, commission


def save_optimized_params(
    params_dict: Dict[str, Any],
    backup: bool = True
) -> Path:
    """
    Save optimized parameters to current_params.json (NOT production!).
    
    NOTE: This does NOT update PRODUCTION_PARAMS.json.
    Use promote_to_production() to deploy optimized params to production.
    
    Args:
        params_dict: Dictionary of optimized parameters
        backup: Whether to create backup in history folder
        
    Returns:
        Path to saved file
    """
    params_dict["generated_at"] = datetime.utcnow().isoformat() + "Z"
    params_dict["generated_by"] = "ftmo_challenge_analyzer.py"
    
    if "version" not in params_dict:
        params_dict["version"] = "1.0.0"
    
    with open(PARAMS_FILE, 'w') as f:
        json.dump(params_dict, f, indent=2)
    
    if backup:
        HISTORY_DIR.mkdir(exist_ok=True)
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        backup_path = HISTORY_DIR / f"params_{timestamp}.json"
        with open(backup_path, 'w') as f:
            json.dump(params_dict, f, indent=2)
    
    return PARAMS_FILE


# =============================================================================
# PRODUCTION DEPLOYMENT (manual review required)
# =============================================================================

def promote_to_production(
    source_file: Path = None,
    validation_metrics: Dict = None,
    force: bool = False
) -> Path:
    """
    Promote optimized parameters to production after manual review.
    
    This creates PRODUCTION_PARAMS.json with full audit trail.
    
    Args:
        source_file: Path to source params (default: ftmo_analysis_output/TPE/best_params.json)
        validation_metrics: Dict with training/validation Sharpe, WR, etc.
        force: If True, skip confirmation prompt
        
    Returns:
        Path to production params file
        
    Example:
        from params.params_loader import promote_to_production
        promote_to_production(
            source_file=Path("ftmo_analysis_output/TPE/best_params.json"),
            validation_metrics={
                "training_sharpe": 2.92,
                "validation_sharpe": 4.76,
                "win_rate": 49.2
            }
        )
    """
    if source_file is None:
        # Default to TPE best params
        source_file = Path("ftmo_analysis_output/TPE/best_params.json")
    
    if not source_file.exists():
        raise FileNotFoundError(f"Source file not found: {source_file}")
    
    with open(source_file, 'r') as f:
        source_data = json.load(f)
    
    # Extract parameters
    if "parameters" in source_data:
        params = source_data["parameters"]
    else:
        params = source_data
    
    # Handle naming mismatch
    if 'min_confluence_score' in params:
        params = params.copy()
        params['min_confluence'] = params.pop('min_confluence_score')
    
    # Build production structure
    production = {
        "PRODUCTION_LOCKED": not force,  # Set to True after review
        "production_version": f"1.0.{datetime.utcnow().strftime('%Y%m%d')}",
        "deployed_at": datetime.utcnow().isoformat() + "Z",
        "deployed_by": "promote_to_production()",
        
        "source": {
            "optimization_mode": source_data.get("optimization_mode", "unknown"),
            "optimization_run": source_data.get("timestamp", "unknown"),
            "best_score": source_data.get("best_score"),
            "source_file": str(source_file),
        },
        
        "validation": validation_metrics or {
            "approved": False,
            "note": "Add validation metrics and set approved=True after review"
        },
        
        "parameters": params,
        
        "checksum": {
            "algorithm": "sha256",
            "params_hash": calculate_params_hash(params)
        }
    }
    
    with open(PRODUCTION_PARAMS_FILE, 'w') as f:
        json.dump(production, f, indent=2)
    
    print(f"✅ Production params written to: {PRODUCTION_PARAMS_FILE}")
    print(f"   Hash: {production['checksum']['params_hash'][:16]}...")
    print(f"   ⚠️  Set PRODUCTION_LOCKED=true and approved=true after review!")
    
    return PRODUCTION_PARAMS_FILE
