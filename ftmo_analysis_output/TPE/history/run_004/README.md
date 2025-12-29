# Optimization Run Archive - run_004

## Run Details
- **Date**: 2025-12-29 09:04:53
- **Mode**: TPE (Single-Objective)
- **Trials**: 5
- **Best Score**: 47.54

## Best Parameters
```json
{
  "min_confluence_score": 3,
  "min_quality_factors": 1,
  "risk_per_trade_pct": 0.45,
  "atr_min_percentile": 55.0,
  "trail_activation_r": 1.0,
  "december_atr_multiplier": 1.6,
  "volatile_asset_boost": 1.1,
  "adx_trend_threshold": 17.0,
  "adx_range_threshold": 15.0,
  "trend_min_confluence": 3,
  "range_min_confluence": 3,
  "atr_trail_multiplier": 2.0,
  "partial_exit_at_1r": true,
  "partial_exit_pct": 0.55
}
```

## Results

### Training (2023-01-01 to 2024-09-30)
- Trades: 1,687
- Total R: +31.84
- Win Rate: 46.6%
- Profit: $28,656

### Validation (2024-10-01 to 2025-12-26)
- Trades: 1,085  
- Total R: +148.31
- Win Rate: 51.7%
- Profit: $133,475

### Full Period (2023-2025)
- Trades: 2,726
- Total R: +190.65
- Win Rate: 48.8%
- Profit: $171,589

## Risk Metrics
- Training Sharpe: 0.26
- Validation Sharpe: 1.88 ⭐ (excellent OOS performance)
- Degradation: -1.61 (improved!)
- FTMO DD: 3.1% ✅ PASS

## Notes
This was the first run after critical fixes:
- Removed early validation from progress_callback
- Fixed OutputManager integration
- Added archiving functionality

Manual archiving performed after run completion.
