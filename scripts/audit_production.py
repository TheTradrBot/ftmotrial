#!/usr/bin/env python3
"""
Audit Script - Verify Production Readiness & Parameter Consistency.

This script performs a comprehensive audit of the trading system to ensure:
1. Production parameters are properly configured and locked
2. Live bot will use the correct parameters
3. Full traceability from optimization run to production

Usage:
    python scripts/audit_production.py
    python scripts/audit_production.py --verbose
    python scripts/audit_production.py --fix  # Attempt to fix issues
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import fields

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def check_file_exists(path: Path, description: str) -> tuple[bool, str]:
    """Check if file exists."""
    if path.exists():
        return True, f"✅ {description}: {path}"
    return False, f"❌ {description} MISSING: {path}"


def audit_production_params():
    """Audit production parameters."""
    from params.params_loader import (
        PRODUCTION_PARAMS_FILE,
        verify_production_params,
        get_production_audit_info,
        calculate_params_hash,
    )
    
    results = []
    
    # Check file exists
    ok, msg = check_file_exists(PRODUCTION_PARAMS_FILE, "PRODUCTION_PARAMS.json")
    results.append((ok, msg))
    
    if not ok:
        return results, False
    
    # Verify params
    valid, verify_msg = verify_production_params()
    results.append((valid, f"{'✅' if valid else '❌'} Verification: {verify_msg}"))
    
    if not valid:
        return results, False
    
    # Get audit info
    audit = get_production_audit_info()
    
    results.append((True, f"   Version: {audit.get('production_version', 'N/A')}"))
    results.append((True, f"   Deployed: {audit.get('deployed_at', 'N/A')}"))
    results.append((True, f"   Source: {audit.get('source', {}).get('optimization_mode', '?')} run"))
    results.append((True, f"   Hash: {audit.get('params_hash', 'N/A')[:16]}..."))
    
    # Check validation metrics
    validation = audit.get('validation', {})
    has_sharpe = validation.get('training_sharpe') is not None
    is_approved = validation.get('approved', False)
    
    results.append((has_sharpe, f"{'✅' if has_sharpe else '⚠️'} Validation metrics: {'present' if has_sharpe else 'MISSING'}"))
    results.append((is_approved, f"{'✅' if is_approved else '❌'} Approval status: {'APPROVED' if is_approved else 'NOT APPROVED'}"))
    
    return results, all(r[0] for r in results)


def audit_strategy_core():
    """Audit strategy_core.py StrategyParams defaults."""
    from strategy_core import StrategyParams
    
    results = []
    
    # Get default values
    default_params = StrategyParams()
    
    results.append((True, "📋 StrategyParams Default Values:"))
    
    # Key parameters to check
    key_params = [
        ('min_confluence', 4),
        ('atr_tp1_multiplier', 0.6),
        ('atr_tp2_multiplier', 1.2),
        ('atr_tp3_multiplier', 2.0),
        ('risk_per_trade_pct', 1.0),
        ('trail_activation_r', 2.2),
    ]
    
    for param, expected_default in key_params:
        actual = getattr(default_params, param, 'MISSING')
        match = actual == expected_default
        results.append((True, f"   {param}: {actual}"))
    
    return results, True


def audit_live_bot_imports():
    """Audit that live bot correctly imports production params."""
    main_bot = Path("main_live_bot.py")
    
    if not main_bot.exists():
        return [(False, "❌ main_live_bot.py not found")], False
    
    content = main_bot.read_text()
    
    results = []
    
    # Check for production params import
    has_prod_import = "load_production_params" in content
    results.append((has_prod_import, f"{'✅' if has_prod_import else '❌'} Imports load_production_params"))
    
    # Check for verify call
    has_verify = "verify_production_params" in content
    results.append((has_verify, f"{'✅' if has_verify else '⚠️'} Calls verify_production_params"))
    
    # Check for USING_PRODUCTION_PARAMS flag
    has_flag = "USING_PRODUCTION_PARAMS" in content
    results.append((has_flag, f"{'✅' if has_flag else '❌'} Has USING_PRODUCTION_PARAMS flag"))
    
    return results, all(r[0] for r in results)


def audit_optimization_output():
    """Audit optimization output directories."""
    output_dir = Path("ftmo_analysis_output")
    
    results = []
    
    for mode in ["TPE", "NSGA", "TPE_H4", "NSGA_H4"]:
        mode_dir = output_dir / mode
        best_params = mode_dir / "best_params.json"
        
        if best_params.exists():
            with open(best_params) as f:
                data = json.load(f)
            score = data.get("best_score", "N/A")
            timestamp = data.get("timestamp", "N/A")
            
            # Check if this matches production
            from params.params_loader import PRODUCTION_PARAMS_FILE
            is_production = False
            if PRODUCTION_PARAMS_FILE.exists():
                with open(PRODUCTION_PARAMS_FILE) as f:
                    prod_data = json.load(f)
                prod_source = prod_data.get("source", {}).get("source_file", "")
                is_production = str(best_params) in prod_source or mode in prod_source
            
            marker = " ← PRODUCTION" if is_production else ""
            score_str = f"{score:.2f}" if isinstance(score, (int, float)) else str(score)
            results.append((True, f"   {mode:12} Score: {score_str:>12} @ {timestamp}{marker}"))
        else:
            results.append((True, f"   {mode:12} (no results)"))
    
    return results, True


def audit_param_consistency():
    """Check consistency between production params and optimizer output."""
    from params.params_loader import PRODUCTION_PARAMS_FILE, PARAMS_FILE
    
    results = []
    
    if not PRODUCTION_PARAMS_FILE.exists():
        results.append((False, "❌ Cannot check consistency - no production params"))
        return results, False
    
    with open(PRODUCTION_PARAMS_FILE) as f:
        prod_data = json.load(f)
    
    prod_params = prod_data.get("parameters", {})
    source_file = prod_data.get("source", {}).get("source_file")
    
    if source_file and Path(source_file).exists():
        with open(source_file) as f:
            source_data = json.load(f)
        source_params = source_data.get("parameters", source_data)
        
        # Compare key parameters
        mismatches = []
        for key in ["min_confluence", "min_confluence_score", "tp1_r_multiple", "tp2_r_multiple", "tp3_r_multiple"]:
            prod_val = prod_params.get(key)
            source_val = source_params.get(key)
            
            # Handle naming mismatch
            if key == "min_confluence" and prod_val is None:
                prod_val = prod_params.get("min_confluence_score")
            if key == "min_confluence_score" and source_val is None:
                source_val = source_params.get("min_confluence")
            
            if prod_val != source_val and prod_val is not None and source_val is not None:
                mismatches.append(f"{key}: prod={prod_val}, source={source_val}")
        
        if mismatches:
            results.append((False, f"⚠️ Parameter mismatches with source:"))
            for m in mismatches:
                results.append((False, f"   {m}"))
        else:
            results.append((True, f"✅ Production params match source file"))
    else:
        results.append((False, f"⚠️ Source file not found: {source_file}"))
    
    return results, all(r[0] for r in results)


def run_full_audit(verbose: bool = False):
    """Run complete audit."""
    print("=" * 70)
    print("PRODUCTION READINESS AUDIT")
    print(f"Date: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 70)
    
    all_passed = True
    
    # 1. Production params
    print("\n📁 PRODUCTION PARAMETERS")
    results, passed = audit_production_params()
    all_passed &= passed
    for ok, msg in results:
        print(f"   {msg}")
    
    # 2. Live bot imports
    print("\n🤖 LIVE BOT CONFIGURATION")
    results, passed = audit_live_bot_imports()
    all_passed &= passed
    for ok, msg in results:
        print(f"   {msg}")
    
    # 3. Optimization output
    print("\n📊 OPTIMIZATION RUNS")
    results, _ = audit_optimization_output()
    for ok, msg in results:
        print(f"{msg}")
    
    # 4. Param consistency
    print("\n🔗 PARAMETER CONSISTENCY")
    results, passed = audit_param_consistency()
    all_passed &= passed
    for ok, msg in results:
        print(f"   {msg}")
    
    # 5. Strategy core (verbose only)
    if verbose:
        print("\n📋 STRATEGY DEFAULTS")
        results, _ = audit_strategy_core()
        for ok, msg in results:
            print(f"{msg}")
    
    # Summary
    print("\n" + "=" * 70)
    if all_passed:
        print("✅ AUDIT PASSED - System is production ready")
    else:
        print("❌ AUDIT FAILED - Review issues above")
    print("=" * 70)
    
    return all_passed


def main():
    parser = argparse.ArgumentParser(description="Audit production readiness")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output")
    parser.add_argument("--fix", action="store_true", help="Attempt to fix issues")
    
    args = parser.parse_args()
    
    passed = run_full_audit(verbose=args.verbose)
    
    if not passed and args.fix:
        print("\n🔧 Attempting fixes...")
        print("   Run: python -m params.promote_to_production")
    
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
