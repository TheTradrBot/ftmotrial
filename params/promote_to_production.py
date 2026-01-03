#!/usr/bin/env python3
"""
Promote Optimized Parameters to Production.

This script takes the best parameters from an optimization run and promotes
them to PRODUCTION_PARAMS.json with full audit trail for compliance.

Usage:
    # Interactive mode (recommended):
    python -m params.promote_to_production
    
    # Direct mode (specify source):
    python -m params.promote_to_production --source ftmo_analysis_output/TPE/best_params.json
    
    # With validation metrics:
    python -m params.promote_to_production --source TPE --sharpe-train 2.92 --sharpe-val 4.76
    
    # Verify current production params:
    python -m params.promote_to_production --verify
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from params.params_loader import (
    promote_to_production,
    verify_production_params,
    get_production_audit_info,
    calculate_params_hash,
    PRODUCTION_PARAMS_FILE,
)


def list_available_runs():
    """List available optimization runs."""
    output_dir = Path("ftmo_analysis_output")
    
    runs = []
    for mode in ["TPE", "TPE_H4", "NSGA", "NSGA_H4"]:
        best_params = output_dir / mode / "best_params.json"
        if best_params.exists():
            with open(best_params) as f:
                data = json.load(f)
            runs.append({
                "mode": mode,
                "path": str(best_params),
                "score": data.get("best_score", "N/A"),
                "timestamp": data.get("timestamp", "N/A"),
            })
    
    return runs


def interactive_select():
    """Interactive selection of optimization run to promote."""
    runs = list_available_runs()
    
    if not runs:
        print("❌ No optimization runs found in ftmo_analysis_output/")
        print("   Run the optimizer first: python ftmo_challenge_analyzer.py --trials 50")
        return None
    
    print("\n" + "=" * 70)
    print("AVAILABLE OPTIMIZATION RUNS")
    print("=" * 70)
    
    for i, run in enumerate(runs, 1):
        score = run['score']
        if isinstance(score, float):
            score_str = f"{score:.2f}"
        else:
            score_str = str(score)
        print(f"  [{i}] {run['mode']:12} Score: {score_str:12} @ {run['timestamp']}")
    
    print("  [0] Cancel")
    print()
    
    while True:
        try:
            choice = input("Select run to promote [1]: ").strip()
            if choice == "":
                choice = 1
            elif choice == "0":
                print("Cancelled.")
                return None
            else:
                choice = int(choice)
            
            if 1 <= choice <= len(runs):
                return runs[choice - 1]
            else:
                print(f"Invalid choice. Enter 1-{len(runs)} or 0 to cancel.")
        except ValueError:
            print("Invalid input. Enter a number.")


def collect_validation_metrics(source_path: Path):
    """Collect validation metrics from professional_backtest_report.txt or interactively."""
    report_path = source_path.parent / "professional_backtest_report.txt"
    
    metrics = {}
    
    if report_path.exists():
        print(f"\n📊 Found validation report: {report_path}")
        with open(report_path) as f:
            content = f.read()
        
        # Try to parse key metrics
        import re
        
        # Training Sharpe
        match = re.search(r"Training Sharpe Ratio:\s*\+?([\d.]+)", content)
        if match:
            metrics["training_sharpe"] = float(match.group(1))
        
        # Validation Sharpe
        match = re.search(r"Validation Sharpe Ratio:\s*\+?([\d.]+)", content)
        if match:
            metrics["validation_sharpe"] = float(match.group(1))
        
        # Win Rate
        match = re.search(r"Win Rate\s+[\d.]+%\s+[\d.]+%\s+([\d.]+)%", content)
        if match:
            metrics["win_rate_full"] = float(match.group(1))
        
        # Max Drawdown
        match = re.search(r"Max Drawdown\s+[\d.]+\$\s+[\d.]+\$\s+([\d.]+)\$", content)
        if match:
            metrics["max_drawdown_usd"] = float(match.group(1))
        
        # Profit Factor
        match = re.search(r"Profit Factor\s+[\d.]+\s+[\d.]+\s+([\d.]+)", content)
        if match:
            metrics["profit_factor"] = float(match.group(1))
        
        if metrics:
            print(f"   Parsed metrics: {json.dumps(metrics, indent=2)}")
    
    # Allow manual override/addition
    print("\n📝 Enter validation metrics (press Enter to skip/use parsed value):")
    
    def get_float(prompt, default=None):
        default_str = f" [{default}]" if default else ""
        val = input(f"   {prompt}{default_str}: ").strip()
        if val == "":
            return default
        try:
            return float(val)
        except ValueError:
            return default
    
    metrics["training_sharpe"] = get_float("Training Sharpe", metrics.get("training_sharpe"))
    metrics["validation_sharpe"] = get_float("Validation Sharpe", metrics.get("validation_sharpe"))
    metrics["win_rate"] = get_float("Win Rate (%)", metrics.get("win_rate_full"))
    metrics["profit_factor"] = get_float("Profit Factor", metrics.get("profit_factor"))
    
    # Training/validation periods
    metrics["training_period"] = "2023-01-01 to 2024-09-30"
    metrics["validation_period"] = "2024-10-01 to 2025-12-26"
    
    # Approval
    print()
    approve = input("   Approve for production? [y/N]: ").strip().lower()
    metrics["approved"] = approve in ("y", "yes")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Promote optimization results to production")
    parser.add_argument("--source", type=str, help="Path to best_params.json or mode name (TPE/NSGA)")
    parser.add_argument("--verify", action="store_true", help="Verify current production params")
    parser.add_argument("--show", action="store_true", help="Show current production params")
    parser.add_argument("--list", action="store_true", help="List available optimization runs")
    parser.add_argument("--sharpe-train", type=float, help="Training Sharpe ratio")
    parser.add_argument("--sharpe-val", type=float, help="Validation Sharpe ratio")
    parser.add_argument("--force", action="store_true", help="Skip confirmation prompts")
    
    args = parser.parse_args()
    
    # --verify: Check current production params
    if args.verify:
        valid, msg = verify_production_params()
        if valid:
            print(f"✅ {msg}")
            audit = get_production_audit_info()
            print(f"   Source: {audit.get('source', {}).get('optimization_mode')} @ {audit.get('source', {}).get('optimization_run')}")
            print(f"   Hash: {audit.get('params_hash', 'N/A')[:16]}...")
            sys.exit(0)
        else:
            print(f"❌ {msg}")
            sys.exit(1)
    
    # --show: Display current production params
    if args.show:
        if not PRODUCTION_PARAMS_FILE.exists():
            print("❌ No production params configured")
            sys.exit(1)
        
        with open(PRODUCTION_PARAMS_FILE) as f:
            data = json.load(f)
        
        print(json.dumps(data, indent=2))
        sys.exit(0)
    
    # --list: Show available runs
    if args.list:
        runs = list_available_runs()
        if not runs:
            print("No optimization runs found.")
            sys.exit(0)
        
        print("\nAvailable optimization runs:")
        for run in runs:
            print(f"  {run['mode']:12} Score: {run['score']:.2f} @ {run['timestamp']}")
        sys.exit(0)
    
    # Determine source file
    if args.source:
        source_path = Path(args.source)
        if not source_path.exists():
            # Try as mode name
            source_path = Path(f"ftmo_analysis_output/{args.source}/best_params.json")
        
        if not source_path.exists():
            print(f"❌ Source not found: {args.source}")
            sys.exit(1)
    else:
        # Interactive selection
        run = interactive_select()
        if run is None:
            sys.exit(0)
        source_path = Path(run["path"])
    
    print(f"\n📂 Source: {source_path}")
    
    # Collect validation metrics
    if args.sharpe_train or args.sharpe_val:
        metrics = {
            "training_sharpe": args.sharpe_train,
            "validation_sharpe": args.sharpe_val,
            "approved": args.force,
        }
    else:
        metrics = collect_validation_metrics(source_path)
    
    # Confirm
    if not args.force:
        print(f"\n" + "=" * 70)
        print("PROMOTION SUMMARY")
        print("=" * 70)
        print(f"Source: {source_path}")
        print(f"Metrics: {json.dumps(metrics, indent=2)}")
        print()
        confirm = input("Proceed with promotion? [y/N]: ").strip().lower()
        if confirm not in ("y", "yes"):
            print("Cancelled.")
            sys.exit(0)
    
    # Promote
    try:
        result = promote_to_production(
            source_file=source_path,
            validation_metrics=metrics,
            force=args.force
        )
        print(f"\n✅ Production params written to: {result}")
        
        if not metrics.get("approved"):
            print("\n⚠️  IMPORTANT: Parameters are NOT yet approved!")
            print("   1. Review PRODUCTION_PARAMS.json")
            print("   2. Set 'PRODUCTION_LOCKED': true")
            print("   3. Set 'validation.approved': true")
            print("   4. Commit and deploy to Windows VM")
    except Exception as e:
        print(f"❌ Promotion failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
