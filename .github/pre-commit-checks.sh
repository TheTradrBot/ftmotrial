#!/bin/bash
# Pre-commit checks to prevent regression of critical fixes

echo "🔍 Running pre-commit checks..."

# Check 1: Ensure no validation runs in progress_callback
if grep -n "run_full_period_backtest.*VALIDATION" ftmo_challenge_analyzer.py | grep -A 20 "def progress_callback" > /dev/null; then
    echo "❌ ERROR: Validation backtest found in progress_callback!"
    echo "   Validation should only run in validate_top_trials() after optimization."
    exit 1
fi

# Check 2: Ensure OutputManager is used for CSV exports in main()
if grep -n "export_trades_to_csv.*training_trades" ftmo_challenge_analyzer.py | grep -A 50 "def main" > /dev/null; then
    echo "❌ ERROR: Direct export_trades_to_csv() found in main()!"
    echo "   Use output_mgr.save_best_trial_trades() instead."
    exit 1
fi

# Check 3: Ensure archive_current_run() is called at end
if ! grep -n "archive_current_run()" ftmo_challenge_analyzer.py > /dev/null; then
    echo "⚠️  WARNING: archive_current_run() not found!"
    echo "   Results may not be archived to history/"
fi

# Check 4: Ensure correct import (set_output_manager not reset_output_manager)
if grep -n "reset_output_manager" ftmo_challenge_analyzer.py > /dev/null; then
    echo "❌ ERROR: reset_output_manager import found!"
    echo "   Use set_output_manager instead."
    exit 1
fi

echo "✅ All pre-commit checks passed!"
exit 0
