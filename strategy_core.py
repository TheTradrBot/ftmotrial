"""
Strategy Core Module for Blueprint Trader AI.

This module provides the single source of truth for trading rules,
used by both backtests and live scanning/Discord outputs.

The strategy is parameterized to allow optimization while staying
faithful to the Blueprint confluence-based approach.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple, Any

from indicators import calculate_adx_with_slope, check_di_crossover

try:
    from fibonacci_strategy import analyze_fib_setup
except ImportError:
    analyze_fib_setup = None


def _get_candle_datetime(candle: Dict) -> Optional[datetime]:
    """Extract datetime from candle dictionary."""
    time_val = candle.get("time") or candle.get("timestamp") or candle.get("date")
    if time_val is None:
        return None
    
    try:
        if isinstance(time_val, str):
            return datetime.fromisoformat(time_val.replace("Z", "+00:00"))
        elif isinstance(time_val, datetime):
            return time_val
        else:
            return None
    except (ValueError, TypeError):
        return None


def get_atr_scaling_factor(entry_tf: str) -> float:
    """
    Get ATR scaling factor for different entry timeframes.
    
    When using H4 or H1 entries instead of D1, ATR values are proportionally
    smaller. This function provides the multiplier to normalize ATR across
    timeframes for consistent SL/TP sizing.
    
    Args:
        entry_tf: Entry timeframe ('D1', 'H4', 'H1', etc.)
    
    Returns:
        float: Scaling factor (1.0 for D1, 0.4 for H4, 0.2 for H1)
    
    Example:
        >>> atr_d1 = 100 pips
        >>> atr_h4 = 40 pips (40% of D1)
        >>> get_atr_scaling_factor('H4')  # Returns 0.4
        >>> sl = atr_h4 * 2.0  # Same logic, scaled result
    """
    scaling_map = {
        'D1': 1.0,      # Baseline (daily)
        'H4': 0.4,      # 4-hour is ~40% of daily
        'H1': 0.2,      # 1-hour is ~20% of daily
        'M15': 0.1,     # 15-min is ~10% of daily
        'W1': 2.5,      # Weekly is ~2.5x daily
        'MN': 10.0      # Monthly is ~10x daily
    }
    
    return scaling_map.get(entry_tf, 1.0)


def _slice_htf_by_timestamp(
    htf_candles: Optional[List[Dict]],
    reference_dt: datetime
) -> Optional[List[Dict]]:
    """
    Slice higher-timeframe candles to only include those with timestamp <= reference.
    
    This prevents look-ahead bias by ensuring we only use HTF data
    that would have been available at the time of the reference candle.
    
    Args:
        htf_candles: List of higher-timeframe candles (weekly, monthly, etc)
        reference_dt: The reference datetime (current daily candle time)
    
    Returns:
        Sliced list of candles or None if input is None/empty
    """
    if not htf_candles:
        return None
    
    result = []
    for candle in htf_candles:
        candle_dt = _get_candle_datetime(candle)
        if candle_dt is None:
            result.append(candle)
        elif candle_dt <= reference_dt:
            result.append(candle)
        else:
            break
    
    return result if result else None


def _is_weekend(candle: Dict) -> bool:
    """Check if a candle falls on a weekend (Saturday=5, Sunday=6)."""
    dt = _get_candle_datetime(candle)
    if dt is None:
        return False
    return dt.weekday() >= 5  # 5=Saturday, 6=Sunday


@dataclass
class StrategyParams:
    """
    Strategy parameters that can be optimized.
    
    DEFAULTS: These are the BEST OPTIMIZED PARAMS from TPE run_009.
    All components (strategy_core, ftmo_challenge_analyzer, main_live_bot) 
    should use these defaults for consistency.
    
    Source: ftmo_analysis_output/TPE/history/run_009/best_params.json
    Score: 212.09, Sharpe: 2.92 (train) / 4.76 (validation)
    """
    # === CONFLUENCE THRESHOLDS (run_009 optimized) ===
    min_confluence: int = 2  # run_009: min_confluence_score = 2
    min_quality_factors: int = 3  # run_009: 3
    
    atr_sl_multiplier: float = 1.5
    
    # ============================================================================
    # TP R-MULTIPLES (run_009 optimized values)
    # ============================================================================
    atr_tp1_multiplier: float = 1.7  # TP1 at 1.7R (run_009)
    atr_tp2_multiplier: float = 2.7  # TP2 at 2.7R (run_009)
    atr_tp3_multiplier: float = 6.0  # TP3 at 6.0R (run_009)
    atr_tp4_multiplier: float = 3.0
    atr_tp5_multiplier: float = 4.0
    
    fib_low: float = 0.382
    fib_high: float = 0.886
    
    structure_sl_lookback: int = 35
    
    # === FILTER TOGGLES (all disabled per run_009) ===
    use_htf_filter: bool = False
    use_structure_filter: bool = False
    use_fib_filter: bool = False
    use_confirmation_filter: bool = False
    
    require_htf_alignment: bool = False
    require_confirmation_for_active: bool = False
    require_rr_for_active: bool = False
    
    min_rr_ratio: float = 1.0
    risk_per_trade_pct: float = 0.65  # run_009: 0.65%
    
    cooldown_bars: int = 0
    max_open_trades: int = 3
    
    # === TP CLOSE PERCENTAGES (run_009 optimized) ===
    tp1_close_pct: float = 0.34  # run_009: 34% at TP1
    tp2_close_pct: float = 0.16  # run_009: 16% at TP2
    tp3_close_pct: float = 0.35  # run_009: 35% at TP3
    tp4_close_pct: float = 0.10  # run_009: 10% at TP4
    tp5_close_pct: float = 0.05  # run_009: 5% at TP5
    
    # Quantitative enhancement filters - DISABLED per run_009
    use_atr_regime_filter: bool = False
    atr_min_percentile: float = 42.0  # run_009: 42.0
    use_zscore_filter: bool = False
    zscore_threshold: float = 1.5
    use_pattern_filter: bool = False
    
    # Blueprint V2 enhancements - DISABLED per run_009
    use_mitigated_sr: bool = False
    sr_proximity_pct: float = 0.02
    use_structural_framework: bool = False
    use_displacement_filter: bool = False  # run_009: false
    displacement_atr_mult: float = 1.5
    use_candle_rejection: bool = False  # run_009: false
    
    # Advanced quant filters - DISABLED
    use_momentum_filter: bool = False
    momentum_lookback: int = 10
    use_mean_reversion: bool = False
    
    # ML filter parameters
    ml_min_prob: float = 0.6
    
    # === FTMO CHALLENGE PARAMETERS (run_009 optimized) ===
    trail_activation_r: float = 0.65  # run_009: 0.65 (was 2.2)
    december_atr_multiplier: float = 1.7  # run_009: 1.7 (was 1.5)
    volatile_asset_boost: float = 1.35  # run_009: 1.35 (was 1.5)
    
    # ============================================================================
    # REGIME-ADAPTIVE V2 PARAMETERS (run_009 optimized)
    # ============================================================================
    
    # Regime Detection Thresholds (run_009 values)
    adx_trend_threshold: float = 20.0  # run_009: 20.0 (was 25.0)
    adx_range_threshold: float = 13.0  # run_009: 13.0 (was 20.0)
    
    # Range Mode Filters
    range_min_confluence: int = 3  # run_009: 3
    atr_volatility_ratio: float = 0.8
    fib_range_target: float = 0.786
    
    # Trend Mode Parameters
    trend_min_confluence: int = 6  # run_009: 6 (was 4)
    
    # Trail Management
    atr_trail_multiplier: float = 1.6  # run_009: 1.6 (was 1.5)
    
    # ============================================================================
    # REGIME-ADAPTIVE V2 ENHANCED PARAMETERS
    # ============================================================================
    
    # ADX Regime Filter Master Toggle
    use_adx_regime_filter: bool = False  # run_009: disabled

    # ADX Slope-Based Early Trend Entry
    use_adx_slope_rising: bool = False  # If True, allow Trend Mode entries on rising ADX (even slightly below threshold) combined with strong +DI/-DI crossover

    # Additional Strategy-Level Toggles
    use_fib_0786_only: bool = False  # True: require 0.786 zone only; False: allow broader 0.618-0.886
    use_market_structure_bos_only: bool = False  # True: require BOS only; False: allow BOS or CHoCH
    use_volatility_sizing_boost: bool = False  # Increase risk % in high ATR periods
    
    # Categorical/Other Parameters
    fib_zone_type: str = 'golden_only'  # Options: 'golden_only', 'extended', 'full_retracement'
    candle_pattern_strictness: str = 'moderate'  # Options: 'strict', 'moderate', 'loose'
    atr_vol_ratio_range: float = 0.8  # For Range Mode low-vol filter (0.6-0.9)
    
    # ============================================================================
    # SESSION FILTER & GRADUATED RISK MANAGEMENT
    # These control trading hours and position sizing during drawdowns
    # DEFAULTS: All disabled to match run_009 baseline
    # ============================================================================
    
    # Session Filter: Only trade during London/NY hours
    use_session_filter: bool = False  # DISABLED: Asian session trading allowed (run_009)
    session_start_utc: int = 8  # London open
    session_end_utc: int = 22  # NY close
    
    # Graduated Risk Management (3-tier system)
    use_graduated_risk: bool = False  # DISABLED: Fixed risk per trade (run_009)
    tier1_dd_pct: float = 2.0  # Tier 1: Reduce risk at this daily DD%
    tier1_risk_factor: float = 0.67  # Risk multiplier (0.6% -> 0.4%)
    tier2_dd_pct: float = 3.5  # Tier 2: Cancel pending at this daily DD%
    tier3_dd_pct: float = 4.5  # Tier 3: Emergency close at this daily DD%
    
    # ATR Trailing Stop
    use_atr_trailing: bool = False  # DISABLED: No trailing stops (run_009)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert parameters to dictionary."""
        return {
            "min_confluence": self.min_confluence,
            "min_quality_factors": self.min_quality_factors,
            "atr_sl_multiplier": self.atr_sl_multiplier,
            "atr_tp1_multiplier": self.atr_tp1_multiplier,
            "atr_tp2_multiplier": self.atr_tp2_multiplier,
            "atr_tp3_multiplier": self.atr_tp3_multiplier,
            "fib_low": self.fib_low,
            "fib_high": self.fib_high,
            "structure_sl_lookback": self.structure_sl_lookback,
            "use_htf_filter": self.use_htf_filter,
            "use_structure_filter": self.use_structure_filter,
            "use_fib_filter": self.use_fib_filter,
            "use_confirmation_filter": self.use_confirmation_filter,
            "require_htf_alignment": self.require_htf_alignment,
            "require_confirmation_for_active": self.require_confirmation_for_active,
            "require_rr_for_active": self.require_rr_for_active,
            "min_rr_ratio": self.min_rr_ratio,
            "risk_per_trade_pct": self.risk_per_trade_pct,
            "cooldown_bars": self.cooldown_bars,
            "max_open_trades": self.max_open_trades,
            "tp1_close_pct": self.tp1_close_pct,
            "tp2_close_pct": self.tp2_close_pct,
            "tp3_close_pct": self.tp3_close_pct,
            "tp4_close_pct": self.tp4_close_pct,
            "tp5_close_pct": self.tp5_close_pct,
            "use_atr_regime_filter": self.use_atr_regime_filter,
            "atr_min_percentile": self.atr_min_percentile,
            "use_zscore_filter": self.use_zscore_filter,
            "zscore_threshold": self.zscore_threshold,
            "use_pattern_filter": self.use_pattern_filter,
            "use_mitigated_sr": self.use_mitigated_sr,
            "sr_proximity_pct": self.sr_proximity_pct,
            "use_structural_framework": self.use_structural_framework,
            "use_displacement_filter": self.use_displacement_filter,
            "displacement_atr_mult": self.displacement_atr_mult,
            "use_candle_rejection": self.use_candle_rejection,
            "use_momentum_filter": self.use_momentum_filter,
            "momentum_lookback": self.momentum_lookback,
            "use_mean_reversion": self.use_mean_reversion,
            "ml_min_prob": self.ml_min_prob,
            "trail_activation_r": self.trail_activation_r,
            "december_atr_multiplier": self.december_atr_multiplier,
            "volatile_asset_boost": self.volatile_asset_boost,
            "adx_trend_threshold": self.adx_trend_threshold,
            "adx_range_threshold": self.adx_range_threshold,
            "range_min_confluence": self.range_min_confluence,
            "atr_volatility_ratio": self.atr_volatility_ratio,
            "fib_range_target": self.fib_range_target,
            "trend_min_confluence": self.trend_min_confluence,
            "atr_trail_multiplier": self.atr_trail_multiplier,
            # REGIME-ADAPTIVE V2 ENHANCED PARAMETERS
            "use_adx_regime_filter": self.use_adx_regime_filter,
            "use_adx_slope_rising": self.use_adx_slope_rising,
            "use_fib_0786_only": self.use_fib_0786_only,
            "use_market_structure_bos_only": self.use_market_structure_bos_only,
            "use_atr_trailing": self.use_atr_trailing,
            "use_volatility_sizing_boost": self.use_volatility_sizing_boost,
            "fib_zone_type": self.fib_zone_type,
            "candle_pattern_strictness": self.candle_pattern_strictness,
            "atr_vol_ratio_range": self.atr_vol_ratio_range,
            # SESSION FILTER & GRADUATED RISK MANAGEMENT
            "use_session_filter": self.use_session_filter,
            "session_start_utc": self.session_start_utc,
            "session_end_utc": self.session_end_utc,
            "use_graduated_risk": self.use_graduated_risk,
            "tier1_dd_pct": self.tier1_dd_pct,
            "tier1_risk_factor": self.tier1_risk_factor,
            "tier2_dd_pct": self.tier2_dd_pct,
            "tier3_dd_pct": self.tier3_dd_pct,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StrategyParams":
        """Create parameters from dictionary."""
        return cls(**{k: v for k, v in d.items() if hasattr(cls, k)})


@dataclass
class Signal:
    """Represents a trading signal/setup."""
    symbol: str
    direction: str
    bar_index: int
    timestamp: Any
    
    confluence_score: int = 0
    quality_factors: int = 0
    
    entry: Optional[float] = None
    stop_loss: Optional[float] = None
    tp1: Optional[float] = None
    tp2: Optional[float] = None
    tp3: Optional[float] = None
    tp4: Optional[float] = None
    tp5: Optional[float] = None
    
    is_active: bool = False
    is_watching: bool = False
    
    flags: Dict[str, bool] = field(default_factory=dict)
    notes: Dict[str, str] = field(default_factory=dict)


@dataclass
class Trade:
    """Represents a completed trade for backtest analysis."""
    symbol: str
    direction: str
    entry_date: Any
    exit_date: Any
    entry_price: float
    exit_price: float
    stop_loss: float
    tp1: Optional[float] = None
    tp2: Optional[float] = None
    tp3: Optional[float] = None
    tp4: Optional[float] = None
    tp5: Optional[float] = None
    
    risk: float = 0.0
    reward: float = 0.0
    rr: float = 0.0
    
    is_winner: bool = False
    exit_reason: str = ""
    
    confluence_score: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert trade to dictionary."""
        return {
            "symbol": self.symbol,
            "direction": self.direction,
            "entry_date": str(self.entry_date),
            "exit_date": str(self.exit_date),
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "stop_loss": self.stop_loss,
            "tp1": self.tp1,
            "tp2": self.tp2,
            "tp3": self.tp3,
            "tp4": self.tp4,
            "tp5": self.tp5,
            "risk": self.risk,
            "reward": self.reward,
            "rr": self.rr,
            "is_winner": self.is_winner,
            "exit_reason": self.exit_reason,
            "confluence_score": self.confluence_score,
        }


def _is_tradable_session(dt: datetime, params: StrategyParams) -> bool:
    """
    Check if timestamp is during tradable session (London/NY hours).
    
    Args:
        dt: Datetime to check (should be UTC)
        params: Strategy parameters with session settings
    
    Returns:
        True if during tradable session, False if in Asian session (skip)
    """
    if not params.use_session_filter:
        return True  # Filter disabled, all hours tradable
    
    hour_utc = dt.hour
    return params.session_start_utc <= hour_utc < params.session_end_utc


def _atr(candles: List[Dict], period: int = 14) -> float:
    """
    Calculate Average True Range (ATR).
    
    Args:
        candles: List of OHLCV candle dictionaries
        period: ATR period (default 14)
    
    Returns:
        ATR value or 0 if insufficient data
    """
    if len(candles) < period + 1:
        return 0.0
    
    tr_values = []
    for i in range(1, len(candles)):
        high = candles[i].get("high")
        low = candles[i].get("low")
        prev_close = candles[i - 1].get("close")
        
        if high is None or low is None or prev_close is None:
            continue
        
        tr = max(
            high - low,
            abs(high - prev_close),
            abs(low - prev_close)
        )
        tr_values.append(tr)
    
    if len(tr_values) < period:
        return sum(tr_values) / len(tr_values) if tr_values else 0.0
    
    atr_val = sum(tr_values[:period]) / period
    for tr in tr_values[period:]:
        atr_val = (atr_val * (period - 1) + tr) / period
    
    return atr_val


def _calculate_atr_percentile(candles: List[Dict], period: int = 14, lookback: int = 100) -> Tuple[float, float]:
    """
    Calculate current ATR and its percentile rank over the lookback period.
    
    Args:
        candles: List of OHLCV candle dictionaries
        period: ATR period (default 14)
        lookback: Number of days to calculate percentile over (default 100)
    
    Returns:
        Tuple of (current_atr, percentile_rank 0-100)
    """
    if len(candles) < period + lookback:
        current_atr = _atr(candles, period)
        return current_atr, 50.0
    
    atr_values = []
    for i in range(lookback):
        end_idx = len(candles) - i
        if end_idx < period + 1:
            break
        slice_candles = candles[:end_idx]
        atr_val = _atr(slice_candles, period)
        if atr_val > 0:
            atr_values.append(atr_val)
    
    if not atr_values:
        return 0.0, 50.0
    
    current_atr = atr_values[0]
    
    sorted_atrs = sorted(atr_values)
    rank = sum(1 for v in sorted_atrs if v <= current_atr)
    percentile = (rank / len(sorted_atrs)) * 100
    
    return current_atr, percentile


def calculate_adx(candles: List[Dict], period: int = 14) -> float:
    """
    Calculate Average Directional Index (ADX) for trend strength measurement.
    
    ADX measures trend strength regardless of direction:
    - ADX > 25: Strong trend (good for trend following)
    - ADX 20-25: Moderate trend (transition zone)
    - ADX < 20: Weak trend/ranging market (consider mean reversion)
    
    Args:
        candles: List of OHLCV candle dictionaries
        period: ADX period (default 14)
    
    Returns:
        ADX value (0-100 scale)
    """
    if len(candles) < period * 2:
        return 0.0
    
    highs = [c.get("high", 0) for c in candles]
    lows = [c.get("low", 0) for c in candles]
    closes = [c.get("close", 0) for c in candles]
    
    plus_dm = []
    minus_dm = []
    tr_values = []
    
    for i in range(1, len(candles)):
        high_diff = highs[i] - highs[i-1]
        low_diff = lows[i-1] - lows[i]
        
        if high_diff > low_diff and high_diff > 0:
            plus_dm.append(high_diff)
        else:
            plus_dm.append(0)
        
        if low_diff > high_diff and low_diff > 0:
            minus_dm.append(low_diff)
        else:
            minus_dm.append(0)
        
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i-1]),
            abs(lows[i] - closes[i-1])
        )
        tr_values.append(tr)
    
    if len(tr_values) < period:
        return 0.0
    
    smoothed_plus_dm = sum(plus_dm[:period])
    smoothed_minus_dm = sum(minus_dm[:period])
    smoothed_tr = sum(tr_values[:period])
    
    dx_values = []
    
    for i in range(period, len(tr_values)):
        smoothed_plus_dm = smoothed_plus_dm - (smoothed_plus_dm / period) + plus_dm[i]
        smoothed_minus_dm = smoothed_minus_dm - (smoothed_minus_dm / period) + minus_dm[i]
        smoothed_tr = smoothed_tr - (smoothed_tr / period) + tr_values[i]
        
        if smoothed_tr == 0:
            continue
            
        plus_di = 100 * smoothed_plus_dm / smoothed_tr
        minus_di = 100 * smoothed_minus_dm / smoothed_tr
        
        di_sum = plus_di + minus_di
        if di_sum == 0:
            dx = 0
        else:
            dx = 100 * abs(plus_di - minus_di) / di_sum
        dx_values.append(dx)
    
    if not dx_values:
        return 0.0
    
    if len(dx_values) < period:
        return sum(dx_values) / len(dx_values)
    
    adx = sum(dx_values[:period]) / period
    for i in range(period, len(dx_values)):
        adx = ((adx * (period - 1)) + dx_values[i]) / period
    
    return adx


def detect_regime(
    daily_candles: List[Dict],
    adx_trend_threshold: float = 25.0,
    adx_range_threshold: float = 20.0,
    use_adx_slope_rising: bool = False,
    use_adx_regime_filter: bool = True  # When False, always returns 'Trend' mode with can_trade=True (bypasses ADX filtering)
) -> Dict:
    """
    Detect market regime based on ADX (Average Directional Index).
    
    When use_adx_regime_filter=False, the ADX filter is bypassed and all setups
    are treated as Trend Mode. This allows trading without regime restrictions.
    
    This is the core function for the Regime-Adaptive V2 trading system.
    It classifies the current market into one of three regimes:
    
    1. TREND MODE (ADX >= adx_trend_threshold):
       - Market has strong directional movement
       - Trade with momentum: use trend-following entries
       - Higher confluence requirements, larger position sizing allowed
       
    2. RANGE MODE (ADX < adx_range_threshold):
       - Market is ranging/consolidating
       - Trade mean reversion: fade extremes at S/R zones
       - Ultra-conservative: ALL range mode filters must pass
       - Require RSI extremes, Fib 0.786, H4 rejection candles
       
    3. TRANSITION ZONE (ADX between thresholds):
       - Market is transitioning between regimes
       - OPTIMIZED: Allow trading at 6+/7 confluence (requires high conviction)
       - This captures emerging trends early with strict confirmation
    
    V2 Enhancement: Early Trend Detection
    When use_adx_slope_rising=True, allows Trend Mode entry even when ADX is
    slightly below threshold if:
    - ADX is rising (slope > 0)
    - ADX is within 3 points of trend threshold
    - There's a recent +DI/-DI crossover (trend direction confirmation)
    
    Args:
        daily_candles: List of D1 OHLCV candle dictionaries
        adx_trend_threshold: ADX level for trend mode (default 25.0)
        adx_range_threshold: ADX level for range mode (default 20.0)
        use_adx_slope_rising: Enable early trend detection via ADX slope (default False)
    
    Returns:
        Dict with keys:
            'mode': str - 'Trend', 'Range', or 'Transition'
            'adx': float - Current ADX value
            'can_trade': bool - Whether entries are allowed in this regime
            'description': str - Human-readable regime description
            'adx_slope': float - ADX slope (only if use_adx_slope_rising=True)
            'di_crossover': str - DI crossover info (only if use_adx_slope_rising=True)
            'early_trend_entry': bool - True if triggered by early trend detection
    
    Note:
        - No look-ahead bias: uses only data up to current candle
        - ADX is calculated using standard 14-period smoothing
    """
    adx = calculate_adx(daily_candles, period=14)
    
    # BYPASS ADX REGIME FILTER: When disabled, treat all setups as Trend Mode
    if not use_adx_regime_filter:
        return {
            'mode': 'Trend',
            'adx': adx,
            'can_trade': True,
            'description': f'ADX Filter DISABLED: Trading without regime restrictions (ADX={adx:.1f})',
            'early_trend_entry': False,
            'adx_filter_disabled': True
        }
    
    adx_slope = 0.0
    di_crossover_info = ""
    early_trend_entry = False
    
    if use_adx_slope_rising:
        adx_with_slope, plus_di, minus_di, adx_slope, is_slope_rising = calculate_adx_with_slope(
            daily_candles, period=14, slope_lookback=3
        )
        bullish_cross, bearish_cross, di_crossover_info = check_di_crossover(
            daily_candles, period=14, lookback=3
        )
        has_di_crossover = bullish_cross or bearish_cross
        
        adx_near_threshold = adx >= (adx_trend_threshold - 3.0)
        
        if is_slope_rising and adx_near_threshold and has_di_crossover and adx < adx_trend_threshold:
            early_trend_entry = True
            return {
                'mode': 'Trend',
                'adx': adx,
                'can_trade': True,
                'description': f'Trend Mode (Early Entry): ADX={adx:.1f} rising with DI crossover, near threshold {adx_trend_threshold}',
                'adx_slope': adx_slope,
                'di_crossover': di_crossover_info,
                'early_trend_entry': True,
                'plus_di': plus_di,
                'minus_di': minus_di
            }
    
    if adx >= adx_trend_threshold:
        result = {
            'mode': 'Trend',
            'adx': adx,
            'can_trade': True,
            'description': f'Trend Mode: ADX={adx:.1f} >= {adx_trend_threshold} (momentum trading allowed)',
            'early_trend_entry': False
        }
    elif adx < adx_range_threshold:
        result = {
            'mode': 'Range',
            'adx': adx,
            'can_trade': True,
            'description': f'Range Mode: ADX={adx:.1f} < {adx_range_threshold} (conservative mean reversion only)',
            'early_trend_entry': False
        }
    else:
        result = {
            'mode': 'Transition',
            'adx': adx,
            'can_trade': True,  # OPTIMIZED: Allow trading in transition at 6+/7 confluence for +20-30% more setups
            'description': f'Transition Zone: ADX={adx:.1f} between {adx_range_threshold}-{adx_trend_threshold} (high confluence trading allowed)',
            'early_trend_entry': False
        }
    
    if use_adx_slope_rising:
        result['adx_slope'] = adx_slope
        result['di_crossover'] = di_crossover_info
    
    return result





def _check_h4_rejection_candle(h4_candles: List[Dict], direction: str) -> Tuple[bool, str]:
    """
    Check for H4 rejection candle (engulfing or pinbar) at current level.
    
    Args:
        h4_candles: List of H4 OHLCV candle dictionaries
        direction: 'bullish' or 'bearish'
    
    Returns:
        Tuple of (has_rejection, description)
    """
    if not h4_candles or len(h4_candles) < 3:
        return False, "H4 Rejection: Insufficient data"
    
    last = h4_candles[-1]
    prev = h4_candles[-2]
    
    body_last = abs(last["close"] - last["open"])
    range_last = last["high"] - last["low"]
    
    if direction == "bullish":
        is_engulfing = (
            last["close"] > last["open"] and
            prev["close"] < prev["open"] and
            last["close"] > prev["open"] and
            last["open"] < prev["close"]
        )
        
        lower_wick = min(last["close"], last["open"]) - last["low"]
        upper_wick = last["high"] - max(last["close"], last["open"])
        is_pinbar = lower_wick > body_last * 2 and upper_wick < body_last * 0.5
        
        if is_engulfing:
            return True, "H4 Rejection: Bullish engulfing pattern"
        elif is_pinbar:
            return True, "H4 Rejection: Bullish pinbar (hammer)"
        else:
            return False, "H4 Rejection: No bullish rejection pattern"
    else:
        is_engulfing = (
            last["close"] < last["open"] and
            prev["close"] > prev["open"] and
            last["close"] < prev["open"] and
            last["open"] > prev["close"]
        )
        
        upper_wick = last["high"] - max(last["close"], last["open"])
        lower_wick = min(last["close"], last["open"]) - last["low"]
        is_pinbar = upper_wick > body_last * 2 and lower_wick < body_last * 0.5
        
        if is_engulfing:
            return True, "H4 Rejection: Bearish engulfing pattern"
        elif is_pinbar:
            return True, "H4 Rejection: Bearish pinbar (shooting star)"
        else:
            return False, "H4 Rejection: No bearish rejection pattern"


def _check_fib_786_zone(
    candles: List[Dict],
    price: float,
    direction: str,
    tolerance: float = 0.05
) -> Tuple[bool, str]:
    """
    Check if price is in the Fib 0.786 retracement zone (±tolerance).
    
    This is specifically for Range Mode entries which require price
    to be at deep retracement levels for mean reversion setups.
    
    Args:
        candles: List of OHLCV candle dictionaries
        price: Current price
        direction: 'bullish' or 'bearish'
        tolerance: Tolerance around 0.786 level (default ±0.05 = 0.736-0.836)
    
    Returns:
        Tuple of (is_in_zone, description)
    """
    if not candles or len(candles) < 20:
        return False, "Fib 0.786: Insufficient data"
    
    swing_highs, swing_lows = _find_pivots(candles[-30:], lookback=3)
    
    if not swing_highs or not swing_lows:
        return False, "Fib 0.786: No swing points found"
    
    recent_high = max(swing_highs[-3:]) if len(swing_highs) >= 3 else max(swing_highs)
    recent_low = min(swing_lows[-3:]) if len(swing_lows) >= 3 else min(swing_lows)
    
    swing_range = recent_high - recent_low
    if swing_range <= 0:
        return False, "Fib 0.786: Invalid swing range"
    
    fib_786_target = 0.786
    fib_low = fib_786_target - tolerance
    fib_high = fib_786_target + tolerance
    
    if direction == "bullish":
        fib_low_price = recent_high - swing_range * fib_high
        fib_high_price = recent_high - swing_range * fib_low
        
        if fib_low_price <= price <= fib_high_price:
            retracement = (recent_high - price) / swing_range
            return True, f"Fib 0.786: Price at {retracement:.1%} retracement (target zone)"
        else:
            return False, f"Fib 0.786: Price not in {fib_low:.1%}-{fib_high:.1%} zone"
    else:
        fib_low_price = recent_low + swing_range * fib_low
        fib_high_price = recent_low + swing_range * fib_high
        
        if fib_low_price <= price <= fib_high_price:
            retracement = (price - recent_low) / swing_range
            return True, f"Fib 0.786: Price at {retracement:.1%} retracement (target zone)"
        else:
            return False, f"Fib 0.786: Price not in {fib_low:.1%}-{fib_high:.1%} zone"


def validate_range_mode_entry(
    daily_candles: List[Dict],
    h4_candles: Optional[List[Dict]],
    weekly_candles: Optional[List[Dict]],
    monthly_candles: Optional[List[Dict]],
    price: float,
    direction: str,
    confluence_score: int,
    params: Optional["StrategyParams"] = None,
    historical_sr: Optional[Dict[str, List[Dict]]] = None,
    atr_vol_ratio_range: float = 0.8,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Validate entry for Range Mode (conservative mean reversion).
    
    Range Mode is ultra-conservative and requires ALL of the following conditions:
    
    1. Confluence Score >= min_confluence (5-6)
       - Even in ranging markets, we need multiple confirming factors
    
    2. Price at Major S/R Zone
       - Uses existing location helpers to check for support/resistance
       - Historical HTF levels carry more weight
    
    3. Price in Fib 0.786 Retracement Zone (±0.05 tolerance)
       - Deep retracement = better risk/reward for mean reversion
       - 0.786 is the classic "last chance" Fib level
    
    4. H4 Rejection Candle (Engulfing or Pinbar)
       - Confirms price rejection at the level
       - Without rejection, no entry allowed
    
    5. ATR Volatility Filter
       - Current ATR(14) < atr_volatility_ratio * ATR(50) average
       - Low volatility = ranging market confirmation
       - High volatility = avoid (may break out)
    
    Args:
        daily_candles: D1 OHLCV data
        h4_candles: H4 OHLCV data (for rejection candle check)
        weekly_candles: W1 OHLCV data (for location context)
        monthly_candles: MN OHLCV data (for location context)
        price: Current price
        direction: 'bullish' or 'bearish'
        confluence_score: Pre-calculated confluence score
        params: StrategyParams with range mode thresholds
        historical_sr: Optional historical S/R levels
        atr_vol_ratio_range: ATR volatility ratio threshold (default 0.8)
    
    Returns:
        Tuple of (is_valid, details_dict)
        - is_valid: True only if ALL conditions pass
        - details_dict: Contains each check result and notes
    """
    from dataclasses import dataclass
    
    if params is None:
        params = StrategyParams()
    
    details = {
        'confluence_check': {'passed': False, 'note': ''},
        'location_check': {'passed': False, 'note': ''},
        'fib_786_check': {'passed': False, 'note': ''},
        'h4_rejection_check': {'passed': False, 'note': ''},
        'atr_volatility_check': {'passed': False, 'note': ''},
        'all_passed': False,
        'failed_checks': [],
    }
    
    min_confluence = params.range_min_confluence
    if confluence_score >= min_confluence:
        details['confluence_check'] = {
            'passed': True,
            'note': f'Confluence {confluence_score} >= {min_confluence}'
        }
    else:
        details['confluence_check'] = {
            'passed': False,
            'note': f'Confluence {confluence_score} < {min_confluence}'
        }
        details['failed_checks'].append('confluence')
    
    location_note, is_at_sr = _location_context(
        monthly_candles, weekly_candles, daily_candles, price, direction, historical_sr
    )
    if is_at_sr:
        details['location_check'] = {'passed': True, 'note': location_note}
    else:
        details['location_check'] = {'passed': False, 'note': location_note}
        details['failed_checks'].append('location')
    
    fib_in_zone, fib_note = _check_fib_786_zone(
        daily_candles, price, direction, tolerance=0.05
    )
    if fib_in_zone:
        details['fib_786_check'] = {'passed': True, 'note': fib_note}
    else:
        details['fib_786_check'] = {'passed': False, 'note': fib_note}
        details['failed_checks'].append('fib_786')
    
    h4_data = h4_candles if h4_candles and len(h4_candles) >= 3 else daily_candles[-10:]
    has_rejection, rejection_note = _check_h4_rejection_candle(h4_data, direction)
    if has_rejection:
        details['h4_rejection_check'] = {'passed': True, 'note': rejection_note}
    else:
        details['h4_rejection_check'] = {'passed': False, 'note': rejection_note}
        details['failed_checks'].append('h4_rejection')
    
    current_atr = _atr(daily_candles, period=14)
    long_atr = _atr(daily_candles, period=50) if len(daily_candles) >= 51 else current_atr
    
    if long_atr > 0:
        atr_ratio = current_atr / long_atr
        threshold = atr_vol_ratio_range
        
        if atr_ratio < threshold:
            details['atr_volatility_check'] = {
                'passed': True,
                'note': f'ATR ratio {atr_ratio:.2f} < {threshold} (low volatility confirmed)'
            }
        else:
            details['atr_volatility_check'] = {
                'passed': False,
                'note': f'ATR ratio {atr_ratio:.2f} >= {threshold} (volatility too high)'
            }
            details['failed_checks'].append('atr_volatility')
    else:
        details['atr_volatility_check'] = {
            'passed': False,
            'note': 'ATR: Unable to calculate'
        }
        details['failed_checks'].append('atr_volatility')
    
    all_passed = (
        details['confluence_check']['passed'] and
        details['location_check']['passed'] and
        details['fib_786_check']['passed'] and
        details['h4_rejection_check']['passed'] and
        details['atr_volatility_check']['passed']
    )
    
    details['all_passed'] = all_passed
    
    return all_passed, details


def calculate_volatility_parity_risk(
    atr: float,
    base_risk_pct: float,
    reference_atr: Optional[float] = None,
    min_risk_pct: float = 0.25,
    max_risk_pct: float = 2.0,
) -> float:
    """
    Calculate volatility parity adjusted risk percentage.
    
    Volatility parity ensures equal dollar risk per ATR unit across all symbols,
    normalizing position sizes based on current volatility.
    
    Args:
        atr: Current ATR value for the symbol
        base_risk_pct: Base risk percentage (e.g., 0.5 for 0.5%)
        reference_atr: Reference ATR for normalization. If None or 0, returns base risk.
        min_risk_pct: Minimum risk percentage floor (default 0.25%)
        max_risk_pct: Maximum risk percentage cap (default 2.0%)
    
    Returns:
        Adjusted risk percentage capped between min and max
        
    Example:
        If reference_atr=0.0010 and current atr=0.0020 (2x more volatile):
        adjusted_risk = base_risk * (0.0010 / 0.0020) = base_risk * 0.5
        This reduces position size for more volatile instruments.
    """
    if atr <= 0 or reference_atr is None or reference_atr <= 0:
        return max(min_risk_pct, min(max_risk_pct, base_risk_pct))
    
    adjustment_ratio = reference_atr / atr
    adjusted_risk = base_risk_pct * adjustment_ratio
    
    return max(min_risk_pct, min(max_risk_pct, adjusted_risk))


def _detect_bullish_n_pattern(candles: List[Dict], lookback: int = 10) -> Tuple[bool, str]:
    """
    Detect Bullish N pattern: impulse up, pullback, higher low formation.
    
    Pattern criteria:
    1. Initial impulse up (significant up move)
    2. Pullback/retracement (down move that doesn't break prior low)
    3. Higher low confirmed (new low above previous swing low)
    
    Args:
        candles: List of OHLCV candle dictionaries
        lookback: Number of candles to analyze (default 10)
    
    Returns:
        Tuple of (pattern_detected, description)
    """
    if len(candles) < lookback + 5:
        return False, "Insufficient data for pattern detection"
    
    recent = candles[-(lookback + 5):]
    
    swing_highs, swing_lows = _find_pivots(recent, lookback=2)
    
    if len(swing_lows) < 2 or len(swing_highs) < 1:
        return False, "Not enough swing points"
    
    last_low = swing_lows[-1]
    prev_low = swing_lows[-2]
    last_high = swing_highs[-1] if swing_highs else recent[-1]["high"]
    
    higher_low = last_low > prev_low
    
    impulse_range = last_high - prev_low
    atr = _atr(candles, 14)
    significant_impulse = impulse_range > atr * 1.5 if atr > 0 else False
    
    current_price = recent[-1]["close"]
    pullback_depth = (last_high - current_price) / impulse_range if impulse_range > 0 else 0
    valid_pullback = 0.2 <= pullback_depth <= 0.7
    
    if higher_low and significant_impulse and valid_pullback:
        return True, f"Bullish N: Higher low at {last_low:.5f}, impulse from {prev_low:.5f}"
    elif higher_low and significant_impulse:
        return True, f"Bullish N forming: Higher low confirmed, awaiting pullback"
    elif higher_low:
        return False, "Higher low present but no clear impulse"
    else:
        return False, "No Bullish N pattern detected"


def _detect_bearish_v_pattern(candles: List[Dict], lookback: int = 10) -> Tuple[bool, str]:
    """
    Detect Bearish V pattern: impulse down, pullback, lower high formation.
    
    Pattern criteria:
    1. Initial impulse down (significant down move)
    2. Pullback/retracement (up move that doesn't break prior high)
    3. Lower high confirmed (new high below previous swing high)
    
    Args:
        candles: List of OHLCV candle dictionaries
        lookback: Number of candles to analyze (default 10)
    
    Returns:
        Tuple of (pattern_detected, description)
    """
    if len(candles) < lookback + 5:
        return False, "Insufficient data for pattern detection"
    
    recent = candles[-(lookback + 5):]
    
    swing_highs, swing_lows = _find_pivots(recent, lookback=2)
    
    if len(swing_highs) < 2 or len(swing_lows) < 1:
        return False, "Not enough swing points"
    
    last_high = swing_highs[-1]
    prev_high = swing_highs[-2]
    last_low = swing_lows[-1] if swing_lows else recent[-1]["low"]
    
    lower_high = last_high < prev_high
    
    impulse_range = prev_high - last_low
    atr = _atr(candles, 14)
    significant_impulse = impulse_range > atr * 1.5 if atr > 0 else False
    
    current_price = recent[-1]["close"]
    pullback_depth = (current_price - last_low) / impulse_range if impulse_range > 0 else 0
    valid_pullback = 0.2 <= pullback_depth <= 0.7
    
    if lower_high and significant_impulse and valid_pullback:
        return True, f"Bearish V: Lower high at {last_high:.5f}, impulse from {prev_high:.5f}"
    elif lower_high and significant_impulse:
        return True, f"Bearish V forming: Lower high confirmed, awaiting pullback"
    elif lower_high:
        return False, "Lower high present but no clear impulse"
    else:
        return False, "No Bearish V pattern detected"


def _calculate_zscore(price: float, candles: List[Dict], period: int = 20) -> float:
    """
    Calculate z-score of current price relative to moving average.
    
    Z-score measures how many standard deviations the price is from the mean.
    - Z-score < 0: Price below mean
    - Z-score > 0: Price above mean
    - |Z-score| > 2: Extreme deviation
    
    Args:
        price: Current price
        candles: List of OHLCV candle dictionaries
        period: Period for mean/std calculation (default 20)
    
    Returns:
        Z-score value (0.0 if insufficient data)
    """
    if len(candles) < period:
        return 0.0
    
    closes = [c["close"] for c in candles[-period:] if c.get("close") is not None]
    
    if len(closes) < period:
        return 0.0
    
    mean = sum(closes) / len(closes)
    
    variance = sum((x - mean) ** 2 for x in closes) / len(closes)
    std = variance ** 0.5
    
    if std == 0:
        return 0.0
    
    zscore = (price - mean) / std
    return zscore


def _find_bos_swing_for_bullish_n(candles: List[Dict], lookback: int = 20) -> Optional[Tuple[float, float, int, int]]:
    """
    Find the Bullish N pattern anchor points for Fibonacci:
    After a break of structure UP, find the swing low.
    Return fib anchors from red candle close to green candle open at swing low.
    
    Blueprint rule: For bullish N, take fibs from red candle close to green candle open.
    
    Returns:
        Tuple of (fib_start, fib_end, swing_low_idx, bos_idx) or None
    """
    if len(candles) < lookback + 10:
        return None
    
    recent = candles[-(lookback + 10):]
    
    bos_idx = None
    for i in range(len(recent) - 1, 4, -1):
        candle = recent[i]
        prev_high = max(c["high"] for c in recent[max(0, i-5):i])
        if candle["close"] > prev_high and candle["close"] > candle["open"]:
            is_strong = (candle["high"] - candle["low"]) > 0
            if is_strong:
                bos_idx = i
                break
    
    if bos_idx is None:
        return None
    
    swing_low_idx = None
    swing_low_val = float('inf')
    for i in range(bos_idx - 1, max(0, bos_idx - 10), -1):
        if recent[i]["low"] < swing_low_val:
            swing_low_val = recent[i]["low"]
            swing_low_idx = i
    
    if swing_low_idx is None:
        return None
    
    red_candle_close = None
    green_candle_open = None
    
    for i in range(swing_low_idx, min(len(recent), swing_low_idx + 3)):
        c = recent[i]
        if c["close"] < c["open"]:  # Red/bearish candle
            red_candle_close = c["close"]
        elif c["close"] > c["open"] and red_candle_close is not None:  # Green/bullish after red
            green_candle_open = c["open"]
            break
    
    if red_candle_close is None:
        red_candle_close = swing_low_val
    if green_candle_open is None:
        green_candle_open = recent[min(swing_low_idx + 1, len(recent) - 1)]["open"]
    
    fib_start = min(red_candle_close, green_candle_open)
    bos_candle = recent[bos_idx]
    fib_end = bos_candle["close"]
    
    return (fib_start, fib_end, swing_low_idx, bos_idx)


def _find_bos_swing_for_bearish_v(candles: List[Dict], lookback: int = 20) -> Optional[Tuple[float, float, int, int]]:
    """
    Find the Bearish V pattern anchor points for Fibonacci:
    After a break of structure DOWN, find the swing high.
    Return fib anchors from green candle close to red candle open at swing high.
    
    Blueprint rule: For bearish V (shorts), take fibs from green candle close to red candle open.
    
    Returns:
        Tuple of (fib_start, fib_end, swing_high_idx, bos_idx) or None
    """
    if len(candles) < lookback + 10:
        return None
    
    recent = candles[-(lookback + 10):]
    
    bos_idx = None
    for i in range(len(recent) - 1, 4, -1):
        candle = recent[i]
        prev_low = min(c["low"] for c in recent[max(0, i-5):i])
        if candle["close"] < prev_low and candle["close"] < candle["open"]:
            is_strong = (candle["high"] - candle["low"]) > 0
            if is_strong:
                bos_idx = i
                break
    
    if bos_idx is None:
        return None
    
    swing_high_idx = None
    swing_high_val = float('-inf')
    for i in range(bos_idx - 1, max(0, bos_idx - 10), -1):
        if recent[i]["high"] > swing_high_val:
            swing_high_val = recent[i]["high"]
            swing_high_idx = i
    
    if swing_high_idx is None:
        return None
    
    green_candle_close = None
    red_candle_open = None
    
    for i in range(swing_high_idx, min(len(recent), swing_high_idx + 3)):
        c = recent[i]
        if c["close"] > c["open"]:  # Green/bullish candle
            green_candle_close = c["close"]
        elif c["close"] < c["open"] and green_candle_close is not None:  # Red/bearish after green
            red_candle_open = c["open"]
            break
    
    if green_candle_close is None:
        green_candle_close = swing_high_val
    if red_candle_open is None:
        red_candle_open = recent[min(swing_high_idx + 1, len(recent) - 1)]["open"]
    
    fib_start = max(green_candle_close, red_candle_open)
    bos_candle = recent[bos_idx]
    fib_end = bos_candle["close"]
    
    return (fib_start, fib_end, swing_high_idx, bos_idx)


def _detect_mitigated_sr(candles: List[Dict], price: float, direction: str, 
                         proximity_pct: float = 0.02) -> Tuple[bool, str, Optional[float]]:
    """
    Detect mitigated S/R zones - zones that were broken then retested.
    
    A mitigated zone is more reliable because it shows the level has been
    tested, broken, and is now acting as support (former resistance) or
    resistance (former support).
    
    Args:
        candles: OHLCV candles
        price: Current price
        direction: Trade direction
        proximity_pct: How close price must be to zone (default 2%)
    
    Returns:
        Tuple of (is_at_mitigated_sr, note, sr_level)
    """
    if len(candles) < 50:
        return False, "Mitigated SR: Insufficient data", None
    
    swing_highs_with_idx = []
    swing_lows_with_idx = []
    lookback = 3
    
    for i in range(lookback, len(candles) - lookback):
        high = candles[i]["high"]
        low = candles[i]["low"]
        
        is_swing_high = all(candles[j]["high"] <= high for j in range(i - lookback, i + lookback + 1) if j != i)
        is_swing_low = all(candles[j]["low"] >= low for j in range(i - lookback, i + lookback + 1) if j != i)
        
        if is_swing_high:
            swing_highs_with_idx.append((high, i))
        if is_swing_low:
            swing_lows_with_idx.append((low, i))
    
    mitigated_levels = []
    
    for sr_level, sr_idx in swing_highs_with_idx:
        was_broken = False
        was_retested = False
        for i in range(sr_idx + 1, len(candles)):
            if candles[i]["close"] > sr_level:
                was_broken = True
            if was_broken and candles[i]["low"] <= sr_level <= candles[i]["high"]:
                was_retested = True
                break
        if was_broken and was_retested:
            mitigated_levels.append(("resistance_turned_support", sr_level))
    
    for sr_level, sr_idx in swing_lows_with_idx:
        was_broken = False
        was_retested = False
        for i in range(sr_idx + 1, len(candles)):
            if candles[i]["close"] < sr_level:
                was_broken = True
            if was_broken and candles[i]["low"] <= sr_level <= candles[i]["high"]:
                was_retested = True
                break
        if was_broken and was_retested:
            mitigated_levels.append(("support_turned_resistance", sr_level))
    
    for level_type, level in mitigated_levels:
        distance_pct = abs(price - level) / price if price > 0 else 0
        
        if distance_pct <= proximity_pct:
            if direction == "bullish" and level_type == "resistance_turned_support":
                return True, f"Mitigated SR: At RTS level {level:.5f} (within {distance_pct:.1%})", level
            elif direction == "bearish" and level_type == "support_turned_resistance":
                return True, f"Mitigated SR: At STR level {level:.5f} (within {distance_pct:.1%})", level
    
    return False, "Mitigated SR: No qualified level nearby", None


def _detect_structural_framework(candles: List[Dict], direction: str) -> Tuple[bool, str, Optional[Tuple[float, float]]]:
    """
    Detect ascending/descending channel frameworks on daily timeframe.
    
    Framework detection:
    - Ascending channel: Connect 3+ swing lows (ascending) and 3+ swing highs (ascending)
    - Descending channel: Connect 3+ swing highs (descending) and 3+ swing lows (descending)
    
    Returns:
        Tuple of (is_in_framework, note, (lower_bound, upper_bound) or None)
    """
    if len(candles) < 30:
        return False, "Framework: Insufficient data", None
    
    swing_highs, swing_lows = [], []
    lookback = 3
    
    for i in range(lookback, len(candles) - lookback):
        high = candles[i]["high"]
        low = candles[i]["low"]
        
        is_swing_high = all(candles[j]["high"] <= high for j in range(i - lookback, i + lookback + 1) if j != i)
        is_swing_low = all(candles[j]["low"] >= low for j in range(i - lookback, i + lookback + 1) if j != i)
        
        if is_swing_high:
            swing_highs.append((i, high))
        if is_swing_low:
            swing_lows.append((i, low))
    
    if len(swing_lows) < 3 or len(swing_highs) < 3:
        return False, "Framework: Not enough swing points", None
    
    recent_lows = swing_lows[-5:]
    ascending_lows = all(recent_lows[i][1] <= recent_lows[i+1][1] for i in range(len(recent_lows)-1))
    descending_lows = all(recent_lows[i][1] >= recent_lows[i+1][1] for i in range(len(recent_lows)-1))
    
    recent_highs = swing_highs[-5:]
    ascending_highs = all(recent_highs[i][1] <= recent_highs[i+1][1] for i in range(len(recent_highs)-1))
    descending_highs = all(recent_highs[i][1] >= recent_highs[i+1][1] for i in range(len(recent_highs)-1))
    
    current_price = candles[-1]["close"]
    last_low = swing_lows[-1][1] if swing_lows else candles[-1]["low"]
    last_high = swing_highs[-1][1] if swing_highs else candles[-1]["high"]
    
    if ascending_lows and ascending_highs:
        if direction == "bullish":
            price_position = (current_price - last_low) / (last_high - last_low) if last_high > last_low else 0.5
            if price_position < 0.4:
                return True, f"Framework: Ascending channel - price near lower bound ({price_position:.0%})", (last_low, last_high)
        return False, "Framework: Ascending channel but not at optimal entry", (last_low, last_high)
    
    elif descending_lows and descending_highs:
        if direction == "bearish":
            price_position = (current_price - last_low) / (last_high - last_low) if last_high > last_low else 0.5
            if price_position > 0.6:
                return True, f"Framework: Descending channel - price near upper bound ({price_position:.0%})", (last_low, last_high)
        return False, "Framework: Descending channel but not at optimal entry", (last_low, last_high)
    
    return False, "Framework: No clear channel detected", None


def _detect_displacement(candles: List[Dict], direction: str, atr_mult: float = 1.5) -> Tuple[bool, str]:
    """
    Detect displacement - strong candles beyond structure confirming the move.
    
    Displacement = large body candle that shows institutional order flow.
    Must be at least atr_mult * ATR in body size.
    
    Returns:
        Tuple of (has_displacement, note)
    """
    if len(candles) < 20:
        return False, "Displacement: Insufficient data"
    
    atr = _atr(candles, 14)
    if atr <= 0:
        return False, "Displacement: ATR calculation failed"
    
    min_body = atr * atr_mult
    
    for i in range(-5, 0):
        if abs(i) > len(candles):
            continue
        c = candles[i]
        body = abs(c["close"] - c["open"])
        
        if body >= min_body:
            if direction == "bullish" and c["close"] > c["open"]:
                return True, f"Displacement: Strong bullish candle ({body/atr:.1f}x ATR)"
            elif direction == "bearish" and c["close"] < c["open"]:
                return True, f"Displacement: Strong bearish candle ({body/atr:.1f}x ATR)"
    
    return False, "Displacement: No strong impulse candle found"


def _detect_candle_rejection(candles: List[Dict], direction: str) -> Tuple[bool, str]:
    """
    Detect pinbar or engulfing rejection patterns at current price.
    
    Returns:
        Tuple of (has_rejection, note)
    """
    if len(candles) < 3:
        return False, "Rejection: Insufficient data"
    
    curr = candles[-1]
    prev = candles[-2]
    
    body = abs(curr["close"] - curr["open"])
    full_range = curr["high"] - curr["low"]
    upper_wick = curr["high"] - max(curr["close"], curr["open"])
    lower_wick = min(curr["close"], curr["open"]) - curr["low"]
    
    if full_range > 0:
        body_ratio = body / full_range
        
        if direction == "bullish":
            if lower_wick > body * 2 and lower_wick > upper_wick * 1.5:
                return True, "Rejection: Bullish pinbar (long lower wick)"
            if curr["close"] > curr["open"] and curr["close"] > prev["high"] and curr["open"] < prev["low"]:
                return True, "Rejection: Bullish engulfing pattern"
            if curr["close"] > curr["open"] and lower_wick > body:
                return True, "Rejection: Hammer pattern"
        else:
            if upper_wick > body * 2 and upper_wick > lower_wick * 1.5:
                return True, "Rejection: Bearish pinbar (long upper wick)"
            if curr["close"] < curr["open"] and curr["open"] > prev["high"] and curr["close"] < prev["low"]:
                return True, "Rejection: Bearish engulfing pattern"
            if curr["close"] < curr["open"] and upper_wick > body:
                return True, "Rejection: Shooting star pattern"
    
    return False, "Rejection: No clear rejection pattern"




def _detect_momentum(candles: List[Dict], direction: str, lookback: int = 10) -> Tuple[bool, str]:
    """
    Detect momentum alignment using rate of change.
    
    Returns:
        Tuple of (momentum_aligned, note)
    """
    if len(candles) < lookback + 5:
        return False, "Momentum: Insufficient data"
    
    closes = [c["close"] for c in candles[-(lookback + 5):]]
    
    current = closes[-1]
    past = closes[-lookback]
    roc = ((current - past) / past) * 100 if past > 0 else 0
    
    short_roc = ((closes[-1] - closes[-5]) / closes[-5]) * 100 if closes[-5] > 0 else 0
    
    if direction == "bullish":
        if roc > 0 and short_roc > 0:
            return True, f"Momentum: Bullish aligned (ROC: {roc:.2f}%, Short: {short_roc:.2f}%)"
        elif roc < -2 and short_roc > 0:
            return True, f"Momentum: Mean reversion setup (pulling back in downtrend, bounce starting)"
    else:
        if roc < 0 and short_roc < 0:
            return True, f"Momentum: Bearish aligned (ROC: {roc:.2f}%, Short: {short_roc:.2f}%)"
        elif roc > 2 and short_roc < 0:
            return True, f"Momentum: Mean reversion setup (rallying in uptrend, reversal starting)"
    
    return False, f"Momentum: Not aligned (ROC: {roc:.2f}%, Short: {short_roc:.2f}%)"


def _find_pivots(candles: List[Dict], lookback: int = 5) -> Tuple[List[float], List[float]]:
    """
    Find swing highs and swing lows in candle data.
    
    Args:
        candles: List of OHLCV candle dictionaries
        lookback: Number of bars to look back/forward for pivot identification
    
    Returns:
        Tuple of (swing_highs, swing_lows) as lists of price levels
    """
    if len(candles) < lookback * 2 + 1:
        return [], []
    
    swing_highs = []
    swing_lows = []
    
    for i in range(lookback, len(candles) - lookback):
        high = candles[i]["high"]
        low = candles[i]["low"]
        
        is_swing_high = True
        is_swing_low = True
        
        for j in range(i - lookback, i + lookback + 1):
            if j == i:
                continue
            if candles[j]["high"] > high:
                is_swing_high = False
            if candles[j]["low"] < low:
                is_swing_low = False
        
        if is_swing_high:
            swing_highs.append(high)
        if is_swing_low:
            swing_lows.append(low)
    
    return swing_highs, swing_lows


def _infer_trend(candles: List[Dict], short_lookback: int = 8, long_lookback: int = 21) -> str:
    """
    Infer trend direction without using EMA (EMA removed).

    Simple heuristic: compare short vs long simple averages and
    check recent price action for higher highs / lower lows.

    Returns: "bullish", "bearish" or "mixed".
    """
    if not candles or len(candles) < 5:
        return "mixed"

    closes = [c.get("close") for c in candles if c.get("close") is not None]
    if len(closes) < 5:
        return "mixed"

    short_n = min(short_lookback, len(closes))
    long_n = min(long_lookback, len(closes))

    short_avg = sum(closes[-short_n:]) / short_n
    long_avg = sum(closes[-long_n:]) / long_n

    current_price = closes[-1]

    bullish = 0
    bearish = 0

    if short_avg > long_avg:
        bullish += 1
    else:
        bearish += 1

    if current_price > long_avg:
        bullish += 1
    else:
        bearish += 1

    # simple momentum check: compare last close to recent window
    if len(closes) >= 10:
        recent_max = max(closes[-10:-1])
        recent_min = min(closes[-10:-1])
        if closes[-1] > recent_max:
            bullish += 1
        if closes[-1] < recent_min:
            bearish += 1

    if bullish > bearish:
        return "bullish"
    if bearish > bullish:
        return "bearish"
    return "mixed"


def _pick_direction_from_bias(
    mn_trend: str,
    wk_trend: str,
    d_trend: str
) -> Tuple[str, str, bool]:
    """
    Determine trade direction based on multi-timeframe bias.
    
    Args:
        mn_trend: Monthly trend
        wk_trend: Weekly trend
        d_trend: Daily trend
    
    Returns:
        Tuple of (direction, note, htf_aligned)
    """
    trends = [mn_trend, wk_trend, d_trend]
    bullish_count = sum(1 for t in trends if t == "bullish")
    bearish_count = sum(1 for t in trends if t == "bearish")
    
    if bullish_count >= 2:
        direction = "bullish"
        htf_aligned = mn_trend == "bullish" or wk_trend == "bullish"
        note = f"HTF bias: {mn_trend.upper()[0]}/{wk_trend.upper()[0]}/{d_trend.upper()[0]} -> Bullish"
    elif bearish_count >= 2:
        direction = "bearish"
        htf_aligned = mn_trend == "bearish" or wk_trend == "bearish"
        note = f"HTF bias: {mn_trend.upper()[0]}/{wk_trend.upper()[0]}/{d_trend.upper()[0]} -> Bearish"
    else:
        direction = d_trend if d_trend != "mixed" else "bullish"
        htf_aligned = False
        note = f"HTF bias: Mixed ({mn_trend[0].upper()}/{wk_trend[0].upper()}/{d_trend[0].upper()})"
    
    return direction, note, htf_aligned


def _location_context(
    monthly_candles: List[Dict],
    weekly_candles: List[Dict],
    daily_candles: List[Dict],
    price: float,
    direction: str,
    historical_sr: Optional[Dict[str, List[Dict]]] = None,
) -> Tuple[str, bool]:
    """
    Check if price is at a key location (support/resistance zone).
    
    Now also checks against historical S/R levels from 20+ years of data.
    
    Args:
        monthly_candles: Monthly OHLCV data
        weekly_candles: Weekly OHLCV data  
        daily_candles: Daily OHLCV data
        price: Current price
        direction: Trade direction
        historical_sr: Optional dict with 'monthly' and 'weekly' S/R level lists
    
    Returns:
        Tuple of (note, is_valid_location)
    """
    if not daily_candles or len(daily_candles) < 20:
        return "Location: Insufficient data", False
    
    highs = [c["high"] for c in daily_candles[-50:]] if len(daily_candles) >= 50 else [c["high"] for c in daily_candles]
    lows = [c["low"] for c in daily_candles[-50:]] if len(daily_candles) >= 50 else [c["low"] for c in daily_candles]
    
    recent_high = max(highs[-20:])
    recent_low = min(lows[-20:])
    range_size = recent_high - recent_low
    
    if range_size <= 0:
        return "Location: No range", False
    
    swing_highs, swing_lows = _find_pivots(daily_candles[-50:] if len(daily_candles) >= 50 else daily_candles, lookback=3)
    
    atr = _atr(daily_candles, 14)
    zone_tolerance = atr * 0.5 if atr > 0 else range_size * 0.05
    
    near_historical_sr = False
    historical_sr_note = ""
    if historical_sr:
        sr_tolerance = price * 0.005
        monthly_levels = historical_sr.get('monthly', [])
        for sr in monthly_levels[:20]:
            if abs(price - sr['level']) <= sr_tolerance:
                near_historical_sr = True
                historical_sr_note = f" (HTF MN SR: {sr['level']:.5f}, {sr['touches']} touches)"
                break
        
        if not near_historical_sr:
            weekly_levels = historical_sr.get('weekly', [])
            for sr in weekly_levels[:30]:
                if abs(price - sr['level']) <= sr_tolerance:
                    near_historical_sr = True
                    historical_sr_note = f" (HTF W1 SR: {sr['level']:.5f}, {sr['touches']} touches)"
                    break
    
    if direction == "bullish":
        near_support = any(abs(price - sl) < zone_tolerance for sl in swing_lows[-5:]) if swing_lows else False
        near_range_low = (price - recent_low) < range_size * 0.3
        
        if near_historical_sr:
            return f"Location: At historical S/R zone{historical_sr_note}", True
        elif near_support or near_range_low:
            return "Location: Near support zone", True
        else:
            return "Location: Not at key support", False
    else:
        near_resistance = any(abs(price - sh) < zone_tolerance for sh in swing_highs[-5:]) if swing_highs else False
        near_range_high = (recent_high - price) < range_size * 0.3
        
        if near_historical_sr:
            return f"Location: At historical S/R zone{historical_sr_note}", True
        elif near_resistance or near_range_high:
            return "Location: Near resistance zone", True
        else:
            return "Location: Not at key resistance", False


def _fib_context(
    weekly_candles: List[Dict],
    daily_candles: List[Dict],
    direction: str,
    price: float,
    fib_low: float = 0.382,
    fib_high: float = 0.886,
) -> Tuple[str, bool]:
    """
    Check if price is within a Fibonacci retracement zone using new Fibonacci module.
    
    Returns:
        Tuple of (note, is_in_fib_zone)
    """
    try:
        candles = daily_candles if daily_candles and len(daily_candles) >= 30 else weekly_candles
        
        if not candles or len(candles) < 20:
            return "Fib: Insufficient data", False
        
        # Use new Fibonacci analysis if available
        if analyze_fib_setup:
            fib_analysis = analyze_fib_setup(candles, direction, price)
            if fib_analysis.get("valid"):
                in_zone = fib_analysis.get("in_golden_zone", False)
                pattern_note = fib_analysis.get("pattern_notes", "")
                return f"Fib: Golden Zone {in_zone}, Patterns: {pattern_note}", in_zone
        
        leg = _find_last_swing_leg_for_fib(candles, direction)
        
        if not leg:
            return "Fib: No clear swing leg found", False
        
        lo, hi = leg
        span = hi - lo
        
        if span <= 0:
            return "Fib: Invalid swing range", False
        
        if direction == "bullish":
            fib_382 = hi - span * 0.382
            fib_500 = hi - span * 0.5
            fib_618 = hi - span * 0.618
            fib_786 = hi - span * 0.786
            
            if fib_786 <= price <= fib_382:
                level = round((hi - price) / span, 3)
                return f"Fib: Price at {level:.1%} retracement (Golden Pocket zone)", True
            elif fib_618 <= price <= fib_500:
                return "Fib: Price at 50-61.8% zone", True
            else:
                return "Fib: Price outside retracement zone", False
        else:
            fib_382 = lo + span * 0.382
            fib_500 = lo + span * 0.5
            fib_618 = lo + span * 0.618
            fib_786 = lo + span * 0.786
            
            if fib_382 <= price <= fib_786:
                level = round((price - lo) / span, 3)
                return f"Fib: Price at {level:.1%} retracement (Golden Pocket zone)", True
            elif fib_500 <= price <= fib_618:
                return "Fib: Price at 50-61.8% zone", True
            else:
                return "Fib: Price outside retracement zone", False
    except Exception as e:
        return f"Fib: Error calculating ({type(e).__name__})", False


def _find_last_swing_leg_for_fib(candles: List[Dict], direction: str) -> Optional[Tuple[float, float]]:
    """
    Find the last swing leg for Fibonacci calculation using proper Blueprint anchoring.
    
    Blueprint rules:
    - Bullish N: After BOS up, fibs from red candle close to green candle open at swing low
    - Bearish V: After BOS down, fibs from green candle close to red candle open at swing high
    
    Returns:
        Tuple of (fib_low, fib_high) or None
    """
    if not candles or len(candles) < 20:
        return None
    
    try:
        if direction == "bullish":
            result = _find_bos_swing_for_bullish_n(candles, lookback=20)
            if result:
                fib_start, fib_end, _, _ = result
                return (fib_start, fib_end)
        else:
            result = _find_bos_swing_for_bearish_v(candles, lookback=20)
            if result:
                fib_start, fib_end, _, _ = result
                return (fib_end, fib_start)
    except Exception:
        pass
    
    try:
        swing_highs, swing_lows = _find_pivots(candles, lookback=3)
    except Exception:
        swing_highs, swing_lows = [], []
    
    if not swing_highs or not swing_lows:
        try:
            highs = [c["high"] for c in candles[-30:] if "high" in c]
            lows = [c["low"] for c in candles[-30:] if "low" in c]
            if highs and lows:
                return (min(lows), max(highs))
        except Exception:
            pass
        return None
    
    try:
        recent_highs = swing_highs[-3:] if len(swing_highs) >= 3 else swing_highs
        recent_lows = swing_lows[-3:] if len(swing_lows) >= 3 else swing_lows
        
        if recent_highs and recent_lows:
            hi = max(recent_highs)
            lo = min(recent_lows)
            return (lo, hi)
    except Exception:
        pass
    
    return None


def _structure_context(
    monthly_candles: List[Dict],
    weekly_candles: List[Dict],
    daily_candles: List[Dict],
    direction: str,
) -> Tuple[bool, str]:
    """
    Check market structure alignment (BOS/CHoCH).
    
    Returns:
        Tuple of (is_aligned, note)
    """
    if not daily_candles or len(daily_candles) < 10:
        return False, "Structure: Insufficient data"
    
    swing_highs, swing_lows = _find_pivots(daily_candles[-30:], lookback=3)
    
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return False, "Structure: Not enough swing points"
    
    if direction == "bullish":
        higher_low = swing_lows[-1] > swing_lows[-2] if len(swing_lows) >= 2 else False
        higher_high = swing_highs[-1] > swing_highs[-2] if len(swing_highs) >= 2 else False
        
        bos_up = daily_candles[-1]["close"] > max(swing_highs[-3:]) if swing_highs else False
        
        if bos_up:
            return True, "Structure: BOS up confirmed"
        elif higher_low and higher_high:
            return True, "Structure: HH/HL pattern (bullish)"
        elif higher_low:
            return True, "Structure: Higher low formed"
        else:
            return False, "Structure: No bullish structure"
    else:
        lower_high = swing_highs[-1] < swing_highs[-2] if len(swing_highs) >= 2 else False
        lower_low = swing_lows[-1] < swing_lows[-2] if len(swing_lows) >= 2 else False
        
        bos_down = daily_candles[-1]["close"] < min(swing_lows[-3:]) if swing_lows else False
        
        if bos_down:
            return True, "Structure: BOS down confirmed"
        elif lower_high and lower_low:
            return True, "Structure: LH/LL pattern (bearish)"
        elif lower_high:
            return True, "Structure: Lower high formed"
        else:
            return False, "Structure: No bearish structure"


def _h4_confirmation(
    h4_candles: List[Dict],
    direction: str,
    daily_candles: List[Dict],
) -> Tuple[str, bool]:
    """
    Check for 4H timeframe confirmation (entry trigger).
    
    Returns:
        Tuple of (note, is_confirmed)
    """
    candles = h4_candles if h4_candles and len(h4_candles) >= 5 else daily_candles[-10:]
    
    if not candles or len(candles) < 3:
        return "4H: Insufficient data", False
    
    last = candles[-1]
    prev = candles[-2]
    
    body_last = abs(last["close"] - last["open"])
    range_last = last["high"] - last["low"]
    body_ratio = body_last / range_last if range_last > 0 else 0
    
    if direction == "bullish":
        bullish_candle = last["close"] > last["open"]
        engulfing = (
            last["close"] > last["open"] and
            prev["close"] < prev["open"] and
            last["close"] > prev["open"] and
            last["open"] < prev["close"]
        )
        
        lower_wick = last["open"] - last["low"] if last["close"] > last["open"] else last["close"] - last["low"]
        upper_wick = last["high"] - last["close"] if last["close"] > last["open"] else last["high"] - last["open"]
        pin_bar = lower_wick > body_last * 2 and upper_wick < body_last * 0.5
        
        bos_check = last["high"] > max(c["high"] for c in candles[-5:-1]) if len(candles) >= 5 else False
        
        if engulfing:
            return "4H: Bullish engulfing confirmed", True
        elif pin_bar:
            return "4H: Bullish pin bar (rejection)", True
        elif bos_check and bullish_candle:
            return "4H: Break of structure up", True
        elif bullish_candle and body_ratio > 0.6:
            return "4H: Strong bullish candle", True
        else:
            return "4H: Awaiting bullish confirmation", False
    else:
        bearish_candle = last["close"] < last["open"]
        engulfing = (
            last["close"] < last["open"] and
            prev["close"] > prev["open"] and
            last["close"] < prev["open"] and
            last["open"] > prev["close"]
        )
        
        upper_wick = last["high"] - last["open"] if last["close"] < last["open"] else last["high"] - last["close"]
        lower_wick = last["close"] - last["low"] if last["close"] < last["open"] else last["open"] - last["low"]
        pin_bar = upper_wick > body_last * 2 and lower_wick < body_last * 0.5
        
        bos_check = last["low"] < min(c["low"] for c in candles[-5:-1]) if len(candles) >= 5 else False
        
        if engulfing:
            return "4H: Bearish engulfing confirmed", True
        elif pin_bar:
            return "4H: Bearish pin bar (rejection)", True
        elif bos_check and bearish_candle:
            return "4H: Break of structure down", True
        elif bearish_candle and body_ratio > 0.6:
            return "4H: Strong bearish candle", True
        else:
            return "4H: Awaiting bearish confirmation", False


def _find_structure_sl(candles: List[Dict], direction: str, lookback: int = 35) -> Optional[float]:
    """
    Find structure-based stop loss level.
    
    Returns:
        Stop loss price level or None
    """
    if not candles or len(candles) < 5:
        return None
    
    recent = candles[-lookback:] if len(candles) >= lookback else candles
    swing_highs, swing_lows = _find_pivots(recent, lookback=3)
    
    if direction == "bullish":
        if swing_lows:
            return min(swing_lows[-3:]) if len(swing_lows) >= 3 else min(swing_lows)
        else:
            return min(c["low"] for c in recent[-10:])
    else:
        if swing_highs:
            return max(swing_highs[-3:]) if len(swing_highs) >= 3 else max(swing_highs)
        else:
            return max(c["high"] for c in recent[-10:])


def _compute_confluence_flags(
    monthly_candles: List[Dict],
    weekly_candles: List[Dict],
    daily_candles: List[Dict],
    h4_candles: List[Dict],
    direction: str,
    params: Optional[StrategyParams] = None,
) -> Tuple[Dict[str, bool], Dict[str, str], Tuple]:
    """
    Compute all confluence flags for a trading setup.
    
    This is the main entry point for confluence calculation,
    used by both backtests and live scanning.
    
    Returns:
        Tuple of (flags dict, notes dict, trade_levels tuple)
    """
    return compute_confluence(
        monthly_candles, weekly_candles, daily_candles, h4_candles, direction, params
    )


def compute_confluence(
    monthly_candles: List[Dict],
    weekly_candles: List[Dict],
    daily_candles: List[Dict],
    h4_candles: List[Dict],
    direction: str,
    params: Optional[StrategyParams] = None,
    historical_sr: Optional[Dict[str, List[Dict]]] = None,
) -> Tuple[Dict[str, bool], Dict[str, str], Tuple]:
    """
    Compute confluence flags for a given setup.
    
    Uses the same core logic as strategy.py but with parameterization.
    
    Args:
        monthly_candles: Monthly OHLCV data
        weekly_candles: Weekly OHLCV data
        daily_candles: Daily OHLCV data
        h4_candles: 4H OHLCV data
        direction: Trade direction ("bullish" or "bearish")
        params: Strategy parameters (uses defaults if None)
        historical_sr: Optional dict with 'monthly' and 'weekly' S/R levels from historical data
    
    Returns:
        Tuple of (flags dict, notes dict, trade_levels tuple)
    """
    if params is None:
        params = StrategyParams()
    
    price = daily_candles[-1]["close"] if daily_candles else float("nan")
    
    mn_trend = _infer_trend(monthly_candles) if monthly_candles else "mixed"
    wk_trend = _infer_trend(weekly_candles) if weekly_candles else "mixed"
    d_trend = _infer_trend(daily_candles) if daily_candles else "mixed"
    _, htf_note_text, htf_ok = _pick_direction_from_bias(mn_trend, wk_trend, d_trend)
    
    if params.use_htf_filter:
        loc_note, loc_ok = _location_context(
            monthly_candles, weekly_candles, daily_candles, price, direction, historical_sr
        )
    else:
        loc_note, loc_ok = "Location filter disabled", True
    
    if params.use_fib_filter:
        fib_note, fib_ok = _fib_context(weekly_candles, daily_candles, direction, price)
    else:
        fib_note, fib_ok = "Fib filter disabled", True
    
    # Liquidity sweep / pool checks removed — treat as always OK
    liq_note, liq_ok = "Liquidity filter removed", True
    
    if params.use_structure_filter:
        struct_ok, struct_note = _structure_context(
            monthly_candles, weekly_candles, daily_candles, direction
        )
    else:
        struct_ok, struct_note = True, "Structure filter disabled"
    
    if params.use_confirmation_filter:
        conf_note, conf_ok = _h4_confirmation(h4_candles, direction, daily_candles)
    else:
        conf_note, conf_ok = "Confirmation filter disabled", True
    
    if params.use_atr_regime_filter:
        _, atr_percentile = _calculate_atr_percentile(daily_candles, period=14, lookback=100)
        atr_regime_ok = atr_percentile >= params.atr_min_percentile
        atr_regime_note = f"ATR Regime: {atr_percentile:.1f}th percentile ({'OK' if atr_regime_ok else 'Low volatility'})"
    else:
        atr_regime_ok, atr_regime_note = True, "ATR regime filter disabled"
    
    if params.use_pattern_filter:
        if direction == "bullish":
            pattern_ok, pattern_note = _detect_bullish_n_pattern(daily_candles, lookback=10)
        else:
            pattern_ok, pattern_note = _detect_bearish_v_pattern(daily_candles, lookback=10)
    else:
        pattern_ok, pattern_note = True, "Pattern filter disabled"
    
    if params.use_zscore_filter:
        zscore = _calculate_zscore(price, daily_candles, period=20)
        if direction == "bullish":
            zscore_valid = zscore < -1.0
            zscore_note = f"Z-Score: {zscore:.2f} ({'Valid <-1.0' if zscore_valid else 'Above -1.0, not ideal for long'})"
        else:
            zscore_valid = zscore > 1.0
            zscore_note = f"Z-Score: {zscore:.2f} ({'Valid >1.0' if zscore_valid else 'Below 1.0, not ideal for short'})"
    else:
        zscore_valid, zscore_note = True, "Z-score filter disabled"
    
    # Blueprint V2 enhancements
    if params.use_mitigated_sr:
        mitigated_sr_ok, mitigated_sr_note, _ = _detect_mitigated_sr(
            daily_candles, price, direction, params.sr_proximity_pct
        )
    else:
        mitigated_sr_ok, mitigated_sr_note = True, "Mitigated SR disabled"
    
    if params.use_structural_framework:
        framework_ok, framework_note, _ = _detect_structural_framework(daily_candles, direction)
    else:
        framework_ok, framework_note = True, "Framework disabled"
    
    if params.use_displacement_filter:
        displacement_ok, displacement_note = _detect_displacement(
            daily_candles, direction, params.displacement_atr_mult
        )
    else:
        displacement_ok, displacement_note = True, "Displacement disabled"
    
    if params.use_candle_rejection:
        rejection_ok, rejection_note = _detect_candle_rejection(h4_candles if h4_candles else daily_candles, direction)
    else:
        rejection_ok, rejection_note = True, "Candle rejection disabled"
    
    if params.use_momentum_filter:
        momentum_ok, momentum_note = _detect_momentum(daily_candles, direction, params.momentum_lookback)
    else:
        momentum_ok, momentum_note = True, "Momentum filter disabled"
    
    rr_note, rr_ok, entry, sl, tp1, tp2, tp3, tp4, tp5 = compute_trade_levels(
        daily_candles, direction, params, h4_candles
    )
    
    flags = {
        "htf_bias": htf_ok,
        "location": loc_ok,
        "fib": fib_ok,
        "liquidity": liq_ok,
        "structure": struct_ok,
        "confirmation": conf_ok,
        "rr": rr_ok,
        "atr_regime_ok": atr_regime_ok,
        "pattern_confirmed": pattern_ok,
        "zscore_valid": zscore_valid,
        "mitigated_sr": mitigated_sr_ok,
        "framework": framework_ok,
        "displacement": displacement_ok,
        "rejection": rejection_ok,
        "momentum": momentum_ok,
    }
    
    notes = {
        "htf_bias": htf_note_text,
        "location": loc_note,
        "fib": fib_note,
        "liquidity": liq_note,
        "structure": struct_note,
        "confirmation": conf_note,
        "rr": rr_note,
        "atr_regime": atr_regime_note,
        "pattern": pattern_note,
        "zscore": zscore_note,
        "mitigated_sr": mitigated_sr_note,
        "framework": framework_note,
        "displacement": displacement_note,
        "rejection": rejection_note,
        "momentum": momentum_note,
    }
    
    trade_levels = (entry, sl, tp1, tp2, tp3, tp4, tp5)
    return flags, notes, trade_levels


def compute_trade_levels(
    daily_candles: List[Dict],
    direction: str,
    params: Optional[StrategyParams] = None,
    h4_candles: Optional[List[Dict]] = None,
) -> Tuple[str, bool, Optional[float], Optional[float], Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]]:
    """
    Compute entry, SL, and TP levels using parameterized logic.
    
    Args:
        daily_candles: Daily OHLCV data
        direction: Trade direction
        params: Strategy parameters
        h4_candles: 4H OHLCV data for tighter SL calculation
    
    Returns:
        Tuple of (note, is_valid, entry, sl, tp1, tp2, tp3, tp4, tp5)
    """
    if params is None:
        params = StrategyParams()
    
    if not daily_candles:
        return "R/R: no data.", False, None, None, None, None, None, None, None
    
    current = daily_candles[-1]["close"]
    atr = _atr(daily_candles, 14)
    
    if atr <= 0:
        return "R/R: ATR too small.", False, None, None, None, None, None, None, None
    
    leg = _find_last_swing_leg_for_fib(daily_candles, direction)
    
    sl_candles = h4_candles if h4_candles and len(h4_candles) >= 20 else daily_candles
    h4_lookback = 20
    structure_sl = _find_structure_sl(sl_candles, direction, lookback=h4_lookback)
    
    if leg:
        lo, hi = leg
        span = hi - lo
        if span > 0:
            if direction == "bullish":
                gp_mid = hi - span * 0.618
                entry = current if abs(current - gp_mid) < atr * 0.3 else gp_mid
                
                base_sl = lo - atr * 0.5
                if structure_sl is not None:
                    sl = min(base_sl, structure_sl - atr * 0.4)
                else:
                    sl = base_sl
                
                risk = entry - sl
                
                if risk > 0:
                    tp1 = entry + risk * params.atr_tp1_multiplier
                    tp2 = entry + risk * params.atr_tp2_multiplier
                    tp3 = entry + risk * params.atr_tp3_multiplier
                    tp4 = entry + risk * 2.5
                    tp5 = entry + risk * 3.5
                    
                    note = f"R/R: Entry near {entry:.5f}, SL at {sl:.5f}"
                    return note, True, entry, sl, tp1, tp2, tp3, tp4, tp5
            else:
                gp_mid = lo + span * 0.618
                entry = current if abs(current - gp_mid) < atr * 0.3 else gp_mid
                
                base_sl = hi + atr * 0.5
                if structure_sl is not None:
                    sl = max(base_sl, structure_sl + atr * 0.4)
                else:
                    sl = base_sl
                
                risk = sl - entry
                
                if risk > 0:
                    tp1 = entry - risk * params.atr_tp1_multiplier
                    tp2 = entry - risk * params.atr_tp2_multiplier
                    tp3 = entry - risk * params.atr_tp3_multiplier
                    tp4 = entry - risk * 2.5
                    tp5 = entry - risk * 3.5
                    
                    note = f"R/R: Entry near {entry:.5f}, SL at {sl:.5f}"
                    return note, True, entry, sl, tp1, tp2, tp3, tp4, tp5
    
    entry = current
    sl_mult = params.atr_sl_multiplier
    
    if direction == "bullish":
        if structure_sl is not None:
            sl = min(entry - atr * sl_mult, structure_sl - atr * 0.4)
        else:
            sl = entry - atr * sl_mult
        risk = entry - sl
        tp1 = entry + risk * params.atr_tp1_multiplier
        tp2 = entry + risk * params.atr_tp2_multiplier
        tp3 = entry + risk * params.atr_tp3_multiplier
        tp4 = entry + risk * 2.5
        tp5 = entry + risk * 3.5
    else:
        if structure_sl is not None:
            sl = max(entry + atr * sl_mult, structure_sl + atr * 0.4)
        else:
            sl = entry + atr * sl_mult
        risk = sl - entry
        tp1 = entry - risk * params.atr_tp1_multiplier
        tp2 = entry - risk * params.atr_tp2_multiplier
        tp3 = entry - risk * params.atr_tp3_multiplier
        tp4 = entry - risk * 2.5
        tp5 = entry - risk * 3.5
    
    note = f"R/R: ATR+structure levels"
    return note, True, entry, sl, tp1, tp2, tp3, tp4, tp5


def generate_signals(
    candles: List[Dict],
    symbol: str = "UNKNOWN",
    params: Optional[StrategyParams] = None,
    monthly_candles: Optional[List[Dict]] = None,
    weekly_candles: Optional[List[Dict]] = None,
    h4_candles: Optional[List[Dict]] = None,
) -> List[Signal]:
    """
    Generate trading signals from historical candles.
    
    This function walks through candles sequentially (no look-ahead bias)
    and generates signals based on the Blueprint strategy rules.
    
    Args:
        candles: Daily OHLCV candles (oldest to newest)
        symbol: Asset symbol
        params: Strategy parameters
        monthly_candles: Optional monthly data (derived from daily if not provided)
        weekly_candles: Optional weekly data (derived from daily if not provided)
        h4_candles: Optional 4H data (uses daily for confirmation if not provided)
    
    Returns:
        List of Signal objects
    """
    if params is None:
        params = StrategyParams()
    
    if len(candles) < 50:
        return []
    
    signals = []
    
    for i in range(50, len(candles)):
        try:
            daily_slice = candles[:i+1]
            current_candle = candles[i]
            current_dt = _get_candle_datetime(current_candle)
            
            if current_dt is not None:
                weekly_slice = _slice_htf_by_timestamp(weekly_candles, current_dt)
                monthly_slice = _slice_htf_by_timestamp(monthly_candles, current_dt)
                h4_slice = _slice_htf_by_timestamp(h4_candles, current_dt)
            else:
                weekly_slice = weekly_candles[:i//5+1] if weekly_candles else None
                monthly_slice = monthly_candles[:i//20+1] if monthly_candles else None
                h4_slice = h4_candles[:i*6+1] if h4_candles else None
            
            mn_trend = _infer_trend(monthly_slice) if monthly_slice else _infer_trend(daily_slice[-60:])
            wk_trend = _infer_trend(weekly_slice) if weekly_slice else _infer_trend(daily_slice[-20:])
            d_trend = _infer_trend(daily_slice[-10:])
            
            direction, _, _ = _pick_direction_from_bias(mn_trend, wk_trend, d_trend)
            
            flags, notes, trade_levels = compute_confluence(
                monthly_slice or [],
                weekly_slice or [],
                daily_slice,
                h4_slice or daily_slice[-20:],
                direction,
                params,
            )
        except Exception:
            continue
        
        entry, sl, tp1, tp2, tp3, tp4, tp5 = trade_levels
        
        confluence_score = sum(1 for v in flags.values() if v)
        
        # Quality is now simply based on confidence level (confluence score itself)
        # No additional pillar requirement since we've removed RSI, Bollinger, and Liquidity filters
        quality_factors = max(1, confluence_score // 3)  # At least 1 quality factor for any decent confluence
        
        # Apply volatile asset boost for high-volatility instruments BEFORE threshold check
        # This allows volatile assets (XAUUSD, NAS100USD, GBPJPY, BTCUSD) to more easily pass thresholds
        boosted_confluence, boosted_quality = apply_volatile_asset_boost(
            symbol,
            confluence_score,
            quality_factors,
            params.volatile_asset_boost
        )
        
        has_rr = flags.get("rr", False)
        has_confirmation = flags.get("confirmation", False)
        
        is_active = False
        is_watching = False
        
        # Use boosted scores for threshold comparison
        # BUGFIX: Reduce quality threshold by 1 since confluence_score now includes many new filters
        # This ensures signals that pass confluence threshold also pass quality threshold
        min_quality_for_active = max(1, params.min_quality_factors - 1)
        if boosted_confluence >= params.min_confluence and boosted_quality >= min_quality_for_active:
            is_active = True
        elif boosted_confluence >= params.min_confluence - 1:
            is_watching = True
        
        if is_active or is_watching:
            candle = candles[i]
            timestamp = candle.get("time") or candle.get("timestamp") or candle.get("date")
            
            signal = Signal(
                symbol=symbol,
                direction=direction,
                bar_index=i,
                timestamp=timestamp,
                confluence_score=confluence_score,
                quality_factors=quality_factors,
                entry=entry,
                stop_loss=sl,
                tp1=tp1,
                tp2=tp2,
                tp3=tp3,
                tp4=tp4,
                tp5=tp5,
                is_active=is_active,
                is_watching=is_watching,
                flags=flags,
                notes=notes,
            )
            signals.append(signal)
    
    return signals


def _validate_and_find_entry(
    candles: List[Dict],
    signal_bar: int,
    theoretical_entry: float,
    direction: str,
    max_wait_bars: int = 5,
) -> Tuple[Optional[int], float, bool]:
    """
    Validate entry price against actual candle data and find real entry point.
    
    For limit orders (golden pocket entries), we need to wait for price to 
    actually reach our entry level. This prevents look-ahead bias.
    
    IMPORTANT: Limit order fill logic:
    - Bullish limit order: Placed BELOW current price, fills when price drops TO or THROUGH it
      -> Entry is valid if theoretical_entry >= candle_low (price reached down to our level)
    - Bearish limit order: Placed ABOVE current price, fills when price rises TO or THROUGH it
      -> Entry is valid if theoretical_entry <= candle_high (price reached up to our level)
    
    If price never reaches the entry level, the trade is SKIPPED (not adjusted).
    
    Args:
        candles: OHLCV candles
        signal_bar: Bar index where signal was generated
        theoretical_entry: Calculated entry price from strategy
        direction: 'bullish' or 'bearish'
        max_wait_bars: Maximum bars to wait for entry (default 5)
    
    Returns:
        Tuple of (actual_entry_bar, actual_entry_price, was_adjusted)
        Returns (None, 0, False) if entry never achieved
    """
    for bar_offset in range(max_wait_bars + 1):
        check_bar = signal_bar + bar_offset
        if check_bar >= len(candles):
            break
            
        candle = candles[check_bar]
        high = candle["high"]
        low = candle["low"]
        
        if direction == "bullish":
            if low <= theoretical_entry <= high:
                return check_bar, theoretical_entry, False
            elif theoretical_entry > high:
                pass
        else:
            if low <= theoretical_entry <= high:
                return check_bar, theoretical_entry, False
            elif theoretical_entry < low:
                pass
    
    return None, 0.0, False


def _check_exit_with_h1_data(
    h1_candles_dict: Dict[str, List[Dict]],
    symbol: str,
    direction: str,
    entry_time: Any,
    sl_price: float,
    tp_prices: List[float],  # [tp1, tp2, tp3, tp4, tp5]
    trailing_sl: float,
    tp_hit_flags: List[bool],  # [tp1_hit, tp2_hit, ...]
) -> Tuple[Optional[str], Optional[float], List[bool]]:
    """
    Check SL/TP hits using H1 data for accurate intraday detection.
    
    This fixes the critical bug where D1 backtest cannot determine
    which (SL or TP) was hit first when both occur on the same day.
    
    Args:
        h1_candles_dict: Dict mapping symbol to list of H1 candles
        symbol: Trading symbol
        direction: 'bullish' or 'bearish'
        entry_time: Entry timestamp (to start checking from)
        sl_price: Current stop loss price (may be trailing)
        tp_prices: List of TP prices [tp1, tp2, tp3, tp4, tp5]
        trailing_sl: Current trailing stop level
        tp_hit_flags: Current state of TP hits [tp1_hit, tp2_hit, ...]
    
    Returns:
        Tuple of (exit_type, exit_price, updated_tp_flags)
        exit_type: 'sl', 'tp1', 'tp2', 'tp3', 'tp4', 'tp5', or None if no exit
        exit_price: Price at which exit occurred
        updated_tp_flags: Updated TP hit flags
    """
    if not h1_candles_dict or symbol not in h1_candles_dict:
        return None, None, tp_hit_flags
    
    h1_candles = h1_candles_dict[symbol]
    if not h1_candles:
        return None, None, tp_hit_flags
    
    # Convert entry_time to comparable format
    entry_dt = _get_candle_datetime({"time": entry_time})
    if entry_dt is None:
        return None, None, tp_hit_flags
    
    # Make timezone-naive for comparison
    if hasattr(entry_dt, 'tzinfo') and entry_dt.tzinfo is not None:
        entry_dt = entry_dt.replace(tzinfo=None)
    
    # Find H1 candles after entry
    updated_flags = list(tp_hit_flags)
    current_sl = trailing_sl
    
    for h1_candle in h1_candles:
        h1_time = _get_candle_datetime(h1_candle)
        if h1_time is None:
            continue
        
        # Make timezone-naive
        if hasattr(h1_time, 'tzinfo') and h1_time.tzinfo is not None:
            h1_time = h1_time.replace(tzinfo=None)
        
        # Only check candles after entry
        if h1_time < entry_dt:
            continue
        
        h1_high = h1_candle.get("high", h1_candle.get("High", 0))
        h1_low = h1_candle.get("low", h1_candle.get("Low", 0))
        
        # Check SL hit first (conservative)
        sl_hit = False
        if direction == "bullish":
            sl_hit = h1_low <= current_sl
        else:
            sl_hit = h1_high >= current_sl
        
        if sl_hit:
            return 'sl', current_sl, updated_flags
        
        # Check TP hits in order
        for i, (tp_price, already_hit) in enumerate(zip(tp_prices, updated_flags)):
            if tp_price is None or already_hit:
                continue
            
            tp_hit = False
            if direction == "bullish":
                tp_hit = h1_high >= tp_price
            else:
                tp_hit = h1_low <= tp_price
            
            if tp_hit:
                updated_flags[i] = True
                # If this is TP5 (final), return exit
                if i == 4:  # TP5
                    return f'tp{i+1}', tp_price, updated_flags
                # Otherwise continue to check for more hits on same candle
    
    return None, None, updated_flags


def simulate_trades(
    candles: List[Dict],
    symbol: str = "UNKNOWN",
    params: Optional[StrategyParams] = None,
    monthly_candles: Optional[List[Dict]] = None,
    weekly_candles: Optional[List[Dict]] = None,
    h4_candles: Optional[List[Dict]] = None,
    h1_candles: Optional[List[Dict]] = None,  # NEW: H1 data for accurate exit detection
    include_transaction_costs: bool = True,
    enforce_5ers_rules: bool = False,  # Changed to False - use penalty system instead
    initial_balance: float = 60000.0,
    track_dd_stats: bool = True,  # Track DD for penalty calculation
) -> Tuple[List[Trade], Optional[Dict]]:
    """
    Simulate trades through historical candles using the Blueprint strategy.
    
    This is a walk-forward simulation with no look-ahead bias.
    Uses the same logic as live trading but runs through historical data.
    
    IMPORTANT: Supports multiple concurrent trades up to params.max_open_trades.
    Entry prices are validated against actual candle data.
    If the theoretical entry price is not available on the signal bar,
    we wait up to 5 bars for price to reach the entry level.
    
    Transaction costs (spread + slippage) are deducted from each trade
    when include_transaction_costs=True to produce realistic backtest results.
    
    5ers Challenge DD Tracking (when track_dd_stats=True):
    - Tracks Daily DD and Total DD but does NOT stop backtest
    - Returns DD stats for penalty calculation in optimizer
    - Live bot has separate enforcement via ChallengeRiskManager
    
    Args:
        candles: Daily OHLCV candles (oldest to newest)
        symbol: Asset symbol
        params: Strategy parameters
        monthly_candles: Optional monthly data
        weekly_candles: Optional weekly data
        h4_candles: Optional 4H data
        include_transaction_costs: Whether to include spread/slippage costs (default True)
        enforce_5ers_rules: DEPRECATED - kept for compatibility (default False)
        initial_balance: Starting balance for DD calculations (default 60000)
        track_dd_stats: Track DD statistics for optimizer penalty (default True)
    
    Returns:
        Tuple of (List of Trade objects, Optional DD stats dict)
    """
    if params is None:
        params = StrategyParams()
    
    transaction_cost_pips = 0.0
    pip_value = 0.0001
    
    if include_transaction_costs:
        try:
            from params.params_loader import get_transaction_costs
            from tradr.risk.position_sizing import get_contract_specs
            
            spread_pips, slippage_pips, _ = get_transaction_costs(symbol)
            transaction_cost_pips = spread_pips + slippage_pips
            
            specs = get_contract_specs(symbol)
            pip_value = specs.get("pip_value", 0.0001)
        except Exception:
            transaction_cost_pips = 2.5
            pip_value = 0.0001
    
    transaction_cost_price = transaction_cost_pips * pip_value
    
    # === BUILD H1 INDEX FOR ACCURATE EXIT DETECTION ===
    # When H1 data is provided, we use it to determine exact SL/TP hit order
    # This fixes the critical bug where D1 data can't distinguish which hit first
    h1_index = {}  # Maps date -> list of H1 candles for that day
    use_h1_for_exits = False
    
    if h1_candles and len(h1_candles) > 0:
        use_h1_for_exits = True
        for h1c in h1_candles:
            h1_time = _get_candle_datetime(h1c)
            if h1_time is None:
                continue
            # Make timezone-naive
            if hasattr(h1_time, 'tzinfo') and h1_time.tzinfo is not None:
                h1_time = h1_time.replace(tzinfo=None)
            day_key = h1_time.date()
            if day_key not in h1_index:
                h1_index[day_key] = []
            h1_index[day_key].append({
                'time': h1_time,
                'high': h1c.get('high', h1c.get('High', 0)),
                'low': h1c.get('low', h1c.get('Low', 0)),
                'close': h1c.get('close', h1c.get('Close', 0)),
            })
    
    # === 5ERS DD TRACKING (for penalty calculation, not enforcement) ===
    dd_stats = None
    if track_dd_stats:
        current_balance = initial_balance
        current_equity = initial_balance
        yesterday_high = initial_balance
        current_day_start_idx = 0
        daily_max_dd = 0.0  # Track max DD within current day
        
        # Track DD statistics per day
        daily_dd_records = []  # List of (date, max_dd_pct) tuples
        max_daily_dd = 0.0
        max_total_dd = 0.0
        days_over_4pct = 0
        days_over_5pct = 0
    
    signals = generate_signals(
        candles, symbol, params,
        monthly_candles, weekly_candles, h4_candles
    )
    
    active_signals = [s for s in signals if s.is_active]
    
    TP1_CLOSE_PCT = params.tp1_close_pct
    TP2_CLOSE_PCT = params.tp2_close_pct
    TP3_CLOSE_PCT = params.tp3_close_pct
    TP4_CLOSE_PCT = params.tp4_close_pct
    TP5_CLOSE_PCT = params.tp5_close_pct
    
    signal_to_pending_entry = {}
    for sig in active_signals:
        if sig.entry is None or sig.stop_loss is None or sig.tp1 is None:
            continue
        risk = abs(sig.entry - sig.stop_loss)
        if risk <= 0:
            continue
        signal_to_pending_entry[id(sig)] = {
            "signal": sig,
            "wait_until_bar": sig.bar_index + 5,
        }
    
    trades = []
    open_trades = []
    entered_signal_ids = set()
    
    for bar_idx in range(len(candles)):
        c = candles[bar_idx]
        high = c["high"]
        low = c["low"]
        bar_timestamp = c.get("time") or c.get("timestamp") or c.get("date")
        
        # === 5ERS DD TRACKING (no enforcement, just stats) ===
        if track_dd_stats:
            # Check for nieuwe dag
            if bar_idx > current_day_start_idx:
                # Update yesterday_high met max van vorige dag
                yesterday_high = max(current_balance, current_equity)
                current_day_start_idx = bar_idx
                
                # Reset daily DD tracking for new day
                daily_max_dd = 0.0
            
            # Bereken equity: balance + unrealized P&L open trades
            unrealized_pnl = 0.0
            for ot in open_trades:
                entry_price = ot["entry_price"]
                risk = ot["risk"]
                direction = ot["direction"]
                current_price = c.get("close", entry_price)
                
                if direction == "bullish":
                    price_move = current_price - entry_price
                else:
                    price_move = entry_price - current_price
                
                unrealized_r = price_move / risk if risk > 0 else 0
                unrealized_pnl += unrealized_r * risk
            
            current_equity = current_balance + unrealized_pnl
            
            # Track Daily DD: 5% van yesterday_high
            if yesterday_high > 0:
                daily_drop = max(0, yesterday_high - current_equity)
                daily_dd_pct = (daily_drop / yesterday_high) * 100
                daily_max_dd = max(daily_max_dd, daily_dd_pct)
                
                if daily_dd_pct > max_daily_dd:
                    max_daily_dd = daily_dd_pct
                
                # Track days over thresholds (will count at end of day)
                if bar_idx == len(candles) - 1 or (bar_idx < len(candles) - 1 and 
                   _get_candle_datetime(candles[bar_idx+1]).date() != _get_candle_datetime(c).date()):
                    if daily_max_dd > 5.0:
                        days_over_5pct += 1
                    elif daily_max_dd > 4.0:
                        days_over_4pct += 1
                    daily_dd_records.append((bar_timestamp, daily_max_dd))
            
            # Track Total DD: 10% van initial balance
            if initial_balance > 0:
                total_drop = max(0, initial_balance - current_equity)
                total_dd_pct = (total_drop / initial_balance) * 100
                
                if total_dd_pct > max_total_dd:
                    max_total_dd = total_dd_pct
        
        trades_to_close = []
        for ot in open_trades:
            direction = ot["direction"]
            entry_price = ot["entry_price"]
            risk = ot["risk"]
            trailing_sl = ot["trailing_sl"]
            tp1 = ot["tp1"]
            tp2 = ot["tp2"]
            tp3 = ot["tp3"]
            tp4 = ot["tp4"]
            tp5 = ot["tp5"]
            tp1_hit = ot["tp1_hit"]
            tp2_hit = ot["tp2_hit"]
            tp3_hit = ot["tp3_hit"]
            tp4_hit = ot["tp4_hit"]
            tp5_hit = ot["tp5_hit"]
            
            tp1_rr = ot["tp1_rr"]
            tp2_rr = ot["tp2_rr"]
            tp3_rr = ot["tp3_rr"]
            tp4_rr = ot["tp4_rr"]
            tp5_rr = ot["tp5_rr"]
            
            trade_closed = False
            rr = 0.0
            is_winner = False
            exit_reason = ""
            reward = 0.0
            
            # === H1-BASED EXIT DETECTION (when available) ===
            # This fixes the critical bug where D1 data can't determine SL vs TP order
            h1_sl_hit_first = None  # True = SL first, False = TP first, None = unknown
            h1_tp_updates = {}  # TP levels hit on H1 before SL
            
            if use_h1_for_exits:
                # Get current day's H1 candles
                bar_dt = _get_candle_datetime(c)
                if bar_dt:
                    if hasattr(bar_dt, 'tzinfo') and bar_dt.tzinfo is not None:
                        bar_dt = bar_dt.replace(tzinfo=None)
                    day_key = bar_dt.date()
                    
                    # Get entry timestamp to filter H1 candles
                    entry_ts = ot.get("entry_timestamp")
                    entry_dt_for_h1 = _get_candle_datetime({"time": entry_ts}) if entry_ts else None
                    if entry_dt_for_h1 and hasattr(entry_dt_for_h1, 'tzinfo') and entry_dt_for_h1.tzinfo is not None:
                        entry_dt_for_h1 = entry_dt_for_h1.replace(tzinfo=None)
                    entry_day = entry_dt_for_h1.date() if entry_dt_for_h1 else None
                    
                    if day_key in h1_index:
                        h1_day_candles = h1_index[day_key]
                        
                        # Track what happens on H1 level
                        temp_sl = trailing_sl
                        temp_tp1_hit = tp1_hit
                        temp_tp2_hit = tp2_hit
                        temp_tp3_hit = tp3_hit
                        temp_tp4_hit = tp4_hit
                        temp_tp5_hit = tp5_hit
                        
                        for h1c in h1_day_candles:
                            # CRITICAL FIX: Skip H1 candles before trade entry
                            # On entry day, only check candles AFTER entry time
                            if entry_day == day_key and entry_dt_for_h1:
                                h1_time = h1c.get('time')
                                if h1_time is not None and h1_time < entry_dt_for_h1:
                                    continue  # Skip candles before entry
                            
                            h1_high = h1c['high']
                            h1_low = h1c['low']
                            
                            # Check SL first
                            if direction == "bullish":
                                if h1_low <= temp_sl:
                                    h1_sl_hit_first = True
                                    break
                            else:
                                if h1_high >= temp_sl:
                                    h1_sl_hit_first = True
                                    break
                            
                            # Check TPs in order
                            if direction == "bullish":
                                if tp1 and not temp_tp1_hit and h1_high >= tp1:
                                    temp_tp1_hit = True
                                    h1_tp_updates['tp1'] = True
                                    # Move trailing SL if trail_activation_r reached
                                    if tp1_rr >= params.trail_activation_r:
                                        temp_sl = entry_price
                                if tp2 and temp_tp1_hit and not temp_tp2_hit and h1_high >= tp2:
                                    temp_tp2_hit = True
                                    h1_tp_updates['tp2'] = True
                                    if tp2_rr >= params.trail_activation_r and tp1:
                                        temp_sl = tp1 + 0.5 * risk
                                if tp3 and temp_tp2_hit and not temp_tp3_hit and h1_high >= tp3:
                                    temp_tp3_hit = True
                                    h1_tp_updates['tp3'] = True
                                    if tp3_rr >= params.trail_activation_r and tp2:
                                        temp_sl = tp2 + 0.5 * risk
                                if tp4 and temp_tp3_hit and not temp_tp4_hit and h1_high >= tp4:
                                    temp_tp4_hit = True
                                    h1_tp_updates['tp4'] = True
                                    if tp3:
                                        temp_sl = tp3 + 0.5 * risk
                                if tp5 and temp_tp4_hit and not temp_tp5_hit and h1_high >= tp5:
                                    h1_sl_hit_first = False  # TP5 hit = full win
                                    h1_tp_updates['tp5'] = True
                                    break
                            else:  # bearish
                                if tp1 and not temp_tp1_hit and h1_low <= tp1:
                                    temp_tp1_hit = True
                                    h1_tp_updates['tp1'] = True
                                    if tp1_rr >= params.trail_activation_r:
                                        temp_sl = entry_price
                                if tp2 and temp_tp1_hit and not temp_tp2_hit and h1_low <= tp2:
                                    temp_tp2_hit = True
                                    h1_tp_updates['tp2'] = True
                                    if tp2_rr >= params.trail_activation_r and tp1:
                                        temp_sl = tp1 - 0.5 * risk
                                if tp3 and temp_tp2_hit and not temp_tp3_hit and h1_low <= tp3:
                                    temp_tp3_hit = True
                                    h1_tp_updates['tp3'] = True
                                    if tp3_rr >= params.trail_activation_r and tp2:
                                        temp_sl = tp2 - 0.5 * risk
                                if tp4 and temp_tp3_hit and not temp_tp4_hit and h1_low <= tp4:
                                    temp_tp4_hit = True
                                    h1_tp_updates['tp4'] = True
                                    if tp3:
                                        temp_sl = tp3 - 0.5 * risk
                                if tp5 and temp_tp4_hit and not temp_tp5_hit and h1_low <= tp5:
                                    h1_sl_hit_first = False  # TP5 hit = full win
                                    h1_tp_updates['tp5'] = True
                                    break
                        
                        # Apply H1-detected TP updates to the trade state
                        if 'tp1' in h1_tp_updates and not tp1_hit:
                            ot["tp1_hit"] = True
                            tp1_hit = True
                            if tp1_rr >= params.trail_activation_r:
                                ot["trailing_sl"] = entry_price
                                trailing_sl = entry_price
                        if 'tp2' in h1_tp_updates and not tp2_hit:
                            ot["tp2_hit"] = True
                            tp2_hit = True
                            if tp2_rr >= params.trail_activation_r and tp1:
                                new_sl = tp1 + (0.5 * risk if direction == "bullish" else -0.5 * risk)
                                ot["trailing_sl"] = new_sl
                                trailing_sl = new_sl
                        if 'tp3' in h1_tp_updates and not tp3_hit:
                            ot["tp3_hit"] = True
                            tp3_hit = True
                            if tp3_rr >= params.trail_activation_r and tp2:
                                new_sl = tp2 + (0.5 * risk if direction == "bullish" else -0.5 * risk)
                                ot["trailing_sl"] = new_sl
                                trailing_sl = new_sl
                        if 'tp4' in h1_tp_updates and not tp4_hit:
                            ot["tp4_hit"] = True
                            tp4_hit = True
                            if tp3:
                                new_sl = tp3 + (0.5 * risk if direction == "bullish" else -0.5 * risk)
                                ot["trailing_sl"] = new_sl
                                trailing_sl = new_sl
                        
                        # CRITICAL FIX: If H1 showed TP hits but NO SL hit on this day,
                        # then TP was hit first (or SL was never hit on this day).
                        # Set h1_sl_hit_first = False to prevent the D1 SL check from triggering.
                        if h1_sl_hit_first is None and len(h1_tp_updates) > 0:
                            # H1 showed TP hit(s) without hitting SL first
                            h1_sl_hit_first = False
                        
                        # If H1 showed SL hit, but we now have updated trailing SL from TPs
                        # Re-check if original SL assessment was correct
                        if h1_sl_hit_first == True and len(h1_tp_updates) > 0:
                            # The trailing SL was moved up by TP hits - SL might still be valid
                            # but at the new trailing level
                            pass  # Continue with updated trailing_sl
            
            if direction == "bullish":
                # Check if SL should be triggered
                # CRITICAL FIX: If H1 data shows TP was hit first, skip SL check
                sl_should_trigger = low <= trailing_sl
                if use_h1_for_exits and h1_sl_hit_first == False:
                    # H1 showed TP hit first (or TP5 completed) - don't trigger SL
                    sl_should_trigger = False
                
                if sl_should_trigger:
                    trail_rr = (trailing_sl - entry_price) / risk
                    if tp4_hit:
                        remaining_pct = TP5_CLOSE_PCT
                        rr = TP1_CLOSE_PCT * tp1_rr + TP2_CLOSE_PCT * tp2_rr + TP3_CLOSE_PCT * tp3_rr + TP4_CLOSE_PCT * tp4_rr + remaining_pct * trail_rr
                        exit_reason = "TP4+Trail"
                        is_winner = True
                    elif tp3_hit:
                        remaining_pct = TP4_CLOSE_PCT + TP5_CLOSE_PCT
                        rr = TP1_CLOSE_PCT * tp1_rr + TP2_CLOSE_PCT * tp2_rr + TP3_CLOSE_PCT * tp3_rr + remaining_pct * trail_rr
                        exit_reason = "TP3+Trail"
                        is_winner = True
                    elif tp2_hit:
                        remaining_pct = TP3_CLOSE_PCT + TP4_CLOSE_PCT + TP5_CLOSE_PCT
                        rr = TP1_CLOSE_PCT * tp1_rr + TP2_CLOSE_PCT * tp2_rr + remaining_pct * trail_rr
                        exit_reason = "TP2+Trail"
                        is_winner = True  # Any trailing stop exit is a WIN
                    elif tp1_hit:
                        remaining_pct = TP2_CLOSE_PCT + TP3_CLOSE_PCT + TP4_CLOSE_PCT + TP5_CLOSE_PCT
                        rr = TP1_CLOSE_PCT * tp1_rr + remaining_pct * trail_rr
                        exit_reason = "TP1+Trail"
                        is_winner = True  # Any trailing stop exit is a WIN
                    else:
                        rr = -1.0
                        exit_reason = "SL"
                        is_winner = False
                    reward = rr * risk
                    trade_closed = True
                
                if not trade_closed and tp1 is not None and high >= tp1 and not tp1_hit:
                    ot["tp1_hit"] = True
                    tp1_hit = True
                    # Delay trailing activation until trail_activation_r is reached
                    if tp1_rr >= params.trail_activation_r:
                        ot["trailing_sl"] = entry_price
                        ot["trailing_activated"] = True
                
                if not trade_closed and tp1_hit and tp2 is not None and high >= tp2 and not tp2_hit:
                    ot["tp2_hit"] = True
                    tp2_hit = True
                    # Only activate trailing if we've reached trail_activation_r
                    if tp2_rr >= params.trail_activation_r and tp1 is not None:
                        ot["trailing_sl"] = tp1 + 0.5 * risk
                        ot["trailing_activated"] = True
                
                if not trade_closed and tp2_hit and tp3 is not None and high >= tp3 and not tp3_hit:
                    ot["tp3_hit"] = True
                    tp3_hit = True
                    # Only activate trailing if we've reached trail_activation_r
                    if tp3_rr >= params.trail_activation_r and tp2 is not None:
                        ot["trailing_sl"] = tp2 + 0.5 * risk
                        ot["trailing_activated"] = True
                
                if not trade_closed and tp3_hit and tp4 is not None and high >= tp4 and not tp4_hit:
                    ot["tp4_hit"] = True
                    tp4_hit = True
                    if tp3 is not None:
                        ot["trailing_sl"] = tp3 + 0.5 * risk
                
                if not trade_closed and tp4_hit and tp5 is not None and high >= tp5 and not tp5_hit:
                    ot["tp5_hit"] = True
                    rr = TP1_CLOSE_PCT * tp1_rr + TP2_CLOSE_PCT * tp2_rr + TP3_CLOSE_PCT * tp3_rr + TP4_CLOSE_PCT * tp4_rr + TP5_CLOSE_PCT * tp5_rr
                    reward = rr * risk
                    exit_reason = "TP5"
                    is_winner = True
                    trade_closed = True
            else:
                # Bearish: Check if SL should be triggered
                # CRITICAL FIX: If H1 data shows TP was hit first, skip SL check
                sl_should_trigger = high >= trailing_sl
                if use_h1_for_exits and h1_sl_hit_first == False:
                    # H1 showed TP hit first (or TP5 completed) - don't trigger SL
                    sl_should_trigger = False
                
                if sl_should_trigger:
                    trail_rr = (entry_price - trailing_sl) / risk
                    if tp4_hit:
                        remaining_pct = TP5_CLOSE_PCT
                        rr = TP1_CLOSE_PCT * tp1_rr + TP2_CLOSE_PCT * tp2_rr + TP3_CLOSE_PCT * tp3_rr + TP4_CLOSE_PCT * tp4_rr + remaining_pct * trail_rr
                        exit_reason = "TP4+Trail"
                        is_winner = True
                    elif tp3_hit:
                        remaining_pct = TP4_CLOSE_PCT + TP5_CLOSE_PCT
                        rr = TP1_CLOSE_PCT * tp1_rr + TP2_CLOSE_PCT * tp2_rr + TP3_CLOSE_PCT * tp3_rr + remaining_pct * trail_rr
                        exit_reason = "TP3+Trail"
                        is_winner = True
                    elif tp2_hit:
                        remaining_pct = TP3_CLOSE_PCT + TP4_CLOSE_PCT + TP5_CLOSE_PCT
                        rr = TP1_CLOSE_PCT * tp1_rr + TP2_CLOSE_PCT * tp2_rr + remaining_pct * trail_rr
                        exit_reason = "TP2+Trail"
                        is_winner = True  # Any trailing stop exit is a WIN
                    elif tp1_hit:
                        remaining_pct = TP2_CLOSE_PCT + TP3_CLOSE_PCT + TP4_CLOSE_PCT + TP5_CLOSE_PCT
                        rr = TP1_CLOSE_PCT * tp1_rr + remaining_pct * trail_rr
                        exit_reason = "TP1+Trail"
                        is_winner = True  # Any trailing stop exit is a WIN
                    else:
                        rr = -1.0
                        exit_reason = "SL"
                        is_winner = False
                    reward = rr * risk
                    trade_closed = True
                
                if not trade_closed and tp1 is not None and low <= tp1 and not tp1_hit:
                    ot["tp1_hit"] = True
                    tp1_hit = True
                    # Delay trailing activation until trail_activation_r is reached
                    if tp1_rr >= params.trail_activation_r:
                        ot["trailing_sl"] = entry_price
                        ot["trailing_activated"] = True
                
                if not trade_closed and tp1_hit and tp2 is not None and low <= tp2 and not tp2_hit:
                    ot["tp2_hit"] = True
                    tp2_hit = True
                    # Only activate trailing if we've reached trail_activation_r
                    if tp2_rr >= params.trail_activation_r and tp1 is not None:
                        ot["trailing_sl"] = tp1 - 0.5 * risk
                        ot["trailing_activated"] = True
                
                if not trade_closed and tp2_hit and tp3 is not None and low <= tp3 and not tp3_hit:
                    ot["tp3_hit"] = True
                    tp3_hit = True
                    # Only activate trailing if we've reached trail_activation_r
                    if tp3_rr >= params.trail_activation_r and tp2 is not None:
                        ot["trailing_sl"] = tp2 - 0.5 * risk
                        ot["trailing_activated"] = True
                
                if not trade_closed and tp3_hit and tp4 is not None and low <= tp4 and not tp4_hit:
                    ot["tp4_hit"] = True
                    tp4_hit = True
                    if tp3 is not None:
                        ot["trailing_sl"] = tp3 - 0.5 * risk
                
                if not trade_closed and tp4_hit and tp5 is not None and low <= tp5 and not tp5_hit:
                    ot["tp5_hit"] = True
                    rr = TP1_CLOSE_PCT * tp1_rr + TP2_CLOSE_PCT * tp2_rr + TP3_CLOSE_PCT * tp3_rr + TP4_CLOSE_PCT * tp4_rr + TP5_CLOSE_PCT * tp5_rr
                    reward = rr * risk
                    exit_reason = "TP5"
                    is_winner = True
                    trade_closed = True
            
            if trade_closed:
                cost_r = ot.get("transaction_cost_r", 0.0)
                adjusted_rr = rr - cost_r
                adjusted_reward = adjusted_rr * risk
                adjusted_is_winner = is_winner and adjusted_rr >= 0
                
                trade = Trade(
                    symbol=symbol,
                    direction=direction,
                    entry_date=ot["entry_timestamp"],
                    exit_date=bar_timestamp,
                    entry_price=entry_price,
                    exit_price=entry_price + adjusted_reward if direction == "bullish" else entry_price - adjusted_reward,
                    stop_loss=ot["sl"],
                    tp1=tp1,
                    tp2=tp2,
                    tp3=tp3,
                    tp4=tp4,
                    tp5=tp5,
                    risk=risk,
                    reward=adjusted_reward,
                    rr=adjusted_rr,
                    is_winner=adjusted_is_winner,
                    exit_reason=exit_reason,
                    confluence_score=ot["confluence_score"],
                )
                trades.append(trade)
                trades_to_close.append(ot)
                
                # Update balance when trade closes (for DD tracking)
                if track_dd_stats:
                    current_balance += adjusted_reward
        
        for ot in trades_to_close:
            open_trades.remove(ot)
        
        if len(open_trades) < params.max_open_trades:
            for sig_id, pending in list(signal_to_pending_entry.items()):
                if sig_id in entered_signal_ids:
                    continue
                if len(open_trades) >= params.max_open_trades:
                    break
                
                sig = pending["signal"]
                if bar_idx < sig.bar_index:
                    continue
                if bar_idx > pending["wait_until_bar"]:
                    entered_signal_ids.add(sig_id)
                    continue
                
                # Session filter: Only applicable for intraday timeframes (H4 or lower)
                # For Daily backtests, candle timestamps are midnight so we skip this check
                # The live bot handles session filtering at scan time instead
                # NOTE: This filter is primarily for future H4/H1 backtests
                if params.use_session_filter:
                    entry_dt = _get_candle_datetime(c)
                    if entry_dt:
                        # Only apply if we have intraday data (hour != 0 or we detect H4 data)
                        # For daily data, timestamps are typically 00:00 so we skip
                        hour = entry_dt.hour
                        # If hour is 0 and all candles have hour 0, this is daily data - skip filter
                        if hour != 0:  # Only filter if we have actual intraday timing
                            if not _is_tradable_session(entry_dt, params):
                                continue  # Skip this bar, but don't expire the signal
                
                theoretical_entry = sig.entry
                direction = sig.direction
                
                if direction == "bullish":
                    if low <= theoretical_entry <= high:
                        entry_price = theoretical_entry
                    else:
                        continue
                else:
                    if low <= theoretical_entry <= high:
                        entry_price = theoretical_entry
                    else:
                        continue
                
                sl = sig.stop_loss
                tp1 = sig.tp1
                tp2 = sig.tp2
                tp3 = sig.tp3
                tp4 = sig.tp4
                tp5 = sig.tp5
                risk = abs(entry_price - sl)
                
                if risk <= 0:
                    entered_signal_ids.add(sig_id)
                    continue
                
                # Hard volatility filter - skip trades in low volatility regimes
                # In December, apply stricter threshold
                if params.use_atr_regime_filter:
                    current_date = _get_candle_datetime(candles[bar_idx])
                    passes_vol, _ = check_volatility_filter(
                        candles[:bar_idx+1], 
                        params.atr_min_percentile,
                        current_date=current_date,
                        december_atr_multiplier=params.december_atr_multiplier
                    )
                    if not passes_vol:
                        entered_signal_ids.add(sig_id)
                        continue
                
                if params.ml_min_prob > 0:
                    features = extract_ml_features(candles[:bar_idx+1], sig.flags, direction, params)
                    if features:
                        should_trade, prob = apply_ml_filter(features, params.ml_min_prob)
                        if not should_trade:
                            entered_signal_ids.add(sig_id)
                            continue
                
                tp1_rr = (tp1 - entry_price) / risk if tp1 and direction == "bullish" else ((entry_price - tp1) / risk if tp1 else 0)
                tp2_rr = (tp2 - entry_price) / risk if tp2 and direction == "bullish" else ((entry_price - tp2) / risk if tp2 else 0)
                tp3_rr = (tp3 - entry_price) / risk if tp3 and direction == "bullish" else ((entry_price - tp3) / risk if tp3 else 0)
                tp4_rr = (tp4 - entry_price) / risk if tp4 and direction == "bullish" else ((entry_price - tp4) / risk if tp4 else 0)
                tp5_rr = (tp5 - entry_price) / risk if tp5 and direction == "bullish" else ((entry_price - tp5) / risk if tp5 else 0)
                
                cost_as_r = transaction_cost_price / risk if risk > 0 else 0.0
                
                # Apply volatile asset boost for high-volatility instruments
                boosted_confluence, _ = apply_volatile_asset_boost(
                    symbol,
                    sig.confluence_score,
                    sig.quality_factors,
                    params.volatile_asset_boost
                )
                
                open_trades.append({
                    "signal_id": sig_id,
                    "direction": direction,
                    "entry_bar": bar_idx,
                    "entry_price": entry_price,
                    "entry_timestamp": bar_timestamp,
                    "sl": sl,
                    "trailing_sl": sl,
                    "trailing_activated": False,  # Flag to track when trail_activation_r threshold is reached
                    "tp1": tp1,
                    "tp2": tp2,
                    "tp3": tp3,
                    "tp4": tp4,
                    "tp5": tp5,
                    "risk": risk,
                    "tp1_hit": False,
                    "tp2_hit": False,
                    "tp3_hit": False,
                    "tp4_hit": False,
                    "tp5_hit": False,
                    "tp1_rr": tp1_rr,
                    "tp2_rr": tp2_rr,
                    "tp3_rr": tp3_rr,
                    "tp4_rr": tp4_rr,
                    "tp5_rr": tp5_rr,
                    "confluence_score": boosted_confluence,
                    "transaction_cost_r": cost_as_r,
                })
                entered_signal_ids.add(sig_id)
    
    # === COMPILE DD STATS ===
    if track_dd_stats:
        dd_stats = {
            'max_daily_dd': max_daily_dd,
            'max_total_dd': max_total_dd,
            'days_over_4pct': days_over_4pct,
            'days_over_5pct': days_over_5pct,
            'daily_dd_records': daily_dd_records,
        }
    
    # === H1 POST-VALIDATION: Correct trade outcomes using H1 data ===
    # The D1/H4 simulation cannot determine if SL or TP hit first on same candle.
    # H1 data provides accurate intraday detection.
    if use_h1_for_exits and len(trades) > 0:
        trades = _correct_trades_with_h1(trades, h1_index)
    
    return trades, dd_stats


def _correct_trades_with_h1(trades: List[Trade], h1_index: Dict) -> List[Trade]:
    """
    Post-process trades to correct SL/TP classification using H1 data.
    
    The D1 simulation often incorrectly classifies trades as LOSS when both
    SL and TP are hit on the same daily candle (because it checks SL first).
    
    This function uses H1 data to determine the TRUE order of events and
    corrects trade outcomes accordingly.
    
    IMPORTANT: D1 candles start at 22:00 UTC and span 24 hours.
    We must collect H1 candles by timestamp range, not by calendar date.
    
    Args:
        trades: List of Trade objects from D1 simulation
        h1_index: Dict mapping date -> list of H1 candles
        
    Returns:
        List of corrected Trade objects
    """
    # Build flat sorted list of all H1 candles for efficient range lookup
    all_h1 = []
    for day_candles in h1_index.values():
        all_h1.extend(day_candles)
    all_h1.sort(key=lambda x: x['time'])
    
    if not all_h1:
        return trades
    
    corrected_trades = []
    
    for trade in trades:
        # Only correct SL exits - these might actually be TP wins
        if trade.exit_reason != "SL":
            corrected_trades.append(trade)
            continue
        
        # Get trade details
        entry_dt = _get_candle_datetime({"time": trade.entry_date})
        exit_dt = _get_candle_datetime({"time": trade.exit_date})
        
        if entry_dt is None or exit_dt is None:
            corrected_trades.append(trade)
            continue
        
        # Make timezone-naive
        if hasattr(entry_dt, 'tzinfo') and entry_dt.tzinfo is not None:
            entry_dt = entry_dt.replace(tzinfo=None)
        if hasattr(exit_dt, 'tzinfo') and exit_dt.tzinfo is not None:
            exit_dt = exit_dt.replace(tzinfo=None)
        
        entry_price = trade.entry_price
        sl_price = trade.stop_loss
        tp1_price = trade.tp1
        direction = trade.direction
        
        if tp1_price is None:
            corrected_trades.append(trade)
            continue
        
        # Collect H1 candles for the trade period using timestamp range
        # Filter: entry_dt < time <= exit_dt (after entry, up to and including exit D1 candle period)
        h1_candles = []
        for h1c in all_h1:
            h1_time = h1c['time']
            if entry_dt < h1_time <= exit_dt:
                h1_candles.append(h1c)
        
        if not h1_candles:
            corrected_trades.append(trade)
            continue
        
        # Check which hit first: SL or TP1
        first_sl_time = None
        first_tp_time = None
        
        for h1c in h1_candles:
            h = h1c['high']
            l = h1c['low']
            t = h1c['time']
            
            if direction == "bullish":
                if first_sl_time is None and l <= sl_price:
                    first_sl_time = t
                if first_tp_time is None and h >= tp1_price:
                    first_tp_time = t
            else:  # bearish
                if first_sl_time is None and h >= sl_price:
                    first_sl_time = t
                if first_tp_time is None and l <= tp1_price:
                    first_tp_time = t
            
            # Stop once both are found
            if first_sl_time and first_tp_time:
                break
        
        # Determine correct outcome
        should_be_win = False
        if first_tp_time and not first_sl_time:
            should_be_win = True  # TP hit, SL never hit
        elif first_tp_time and first_sl_time and first_tp_time < first_sl_time:
            should_be_win = True  # TP hit before SL
        
        if should_be_win and not trade.is_winner:
            # Correct this trade from LOSS to WIN
            # Calculate correct RR based on TP1 hit
            risk = abs(entry_price - sl_price)
            if risk > 0:
                tp1_rr = abs(tp1_price - entry_price) / risk
                # For simplicity, assume TP1+Trail outcome
                corrected_rr = tp1_rr * 0.34  # TP1 close pct
                
                # Create corrected trade
                corrected_trade = Trade(
                    symbol=trade.symbol,
                    direction=trade.direction,
                    entry_date=trade.entry_date,
                    exit_date=trade.exit_date,
                    entry_price=trade.entry_price,
                    exit_price=tp1_price,
                    stop_loss=trade.stop_loss,
                    tp1=trade.tp1,
                    tp2=trade.tp2,
                    tp3=trade.tp3,
                    tp4=trade.tp4,
                    tp5=trade.tp5,
                    risk=trade.risk,
                    reward=corrected_rr * trade.risk,
                    rr=corrected_rr,
                    is_winner=True,
                    exit_reason="TP1+Trail (H1 corrected)",
                    confluence_score=trade.confluence_score,
                )
                corrected_trades.append(corrected_trade)
                continue
        
        # No correction needed
        corrected_trades.append(trade)
    
    return corrected_trades


def get_default_params() -> StrategyParams:
    """Get default strategy parameters - optimized for trade generation."""
    return StrategyParams(
        min_confluence=2,
        min_quality_factors=1,
        require_confirmation_for_active=False,
        require_rr_for_active=False,
        atr_sl_multiplier=1.5,
        atr_tp1_multiplier=0.6,
        atr_tp2_multiplier=1.2,
        atr_tp3_multiplier=2.0,
        atr_tp4_multiplier=3.0,
        atr_tp5_multiplier=4.0,
    )


def get_aggressive_params() -> StrategyParams:
    """Get aggressive (more trades) strategy parameters."""
    return StrategyParams(
        min_confluence=1,
        min_quality_factors=0,
        require_confirmation_for_active=False,
        require_rr_for_active=False,
        atr_sl_multiplier=1.2,
        atr_tp1_multiplier=1.0,
        atr_tp2_multiplier=2.0,
        atr_tp3_multiplier=3.0,
        atr_tp4_multiplier=4.0,
        atr_tp5_multiplier=5.0,
    )


def get_conservative_params() -> StrategyParams:
    """Get conservative (higher quality) strategy parameters."""
    return StrategyParams(
        min_confluence=4,
        min_quality_factors=2,
        require_htf_alignment=True,
        require_confirmation_for_active=True,
        require_rr_for_active=True,
        atr_sl_multiplier=1.8,
        atr_tp1_multiplier=1.0,
        atr_tp2_multiplier=2.0,
        atr_tp3_multiplier=3.0,
        atr_tp4_multiplier=4.0,
        atr_tp5_multiplier=5.0,
    )


def extract_ml_features(
    candles: List[Dict],
    flags: Dict[str, bool],
    direction: str,
    params: Optional[StrategyParams] = None,
) -> Dict[str, float]:
    """
    Extract ML features for trade filtering.
    
    Args:
        candles: List of OHLCV candle dictionaries
        flags: Confluence flags from compute_confluence
        direction: Trade direction ("bullish" or "bearish")
        params: Strategy parameters
    
    Returns:
        Dict with ML features for model input
    """
    if params is None:
        params = StrategyParams()
    
    if not candles or len(candles) < 20:
        return {}
    
    price = candles[-1].get("close", 0)
    
    z_score = _calculate_zscore(price, candles, period=20)
    
    _, atr_percentile = _calculate_atr_percentile(candles, period=14, lookback=100)
    
    momentum_lookback = params.momentum_lookback
    if len(candles) >= momentum_lookback + 1:
        current_close = candles[-1].get("close", 0)
        past_close = candles[-(momentum_lookback + 1)].get("close", 1)
        momentum_roc = ((current_close - past_close) / past_close * 100) if past_close > 0 else 0
    else:
        momentum_roc = 0
    
    features = {
        "htf_aligned": 1 if flags.get("htf_aligned", False) else 0,
        "location_ok": 1 if flags.get("location_ok", False) else 0,
        "fib_ok": 1 if flags.get("fib_ok", False) else 0,
        "structure_ok": 1 if flags.get("structure_ok", False) else 0,
        "liquidity_ok": 1 if flags.get("liquidity_ok", False) else 0,
        "confirmation_ok": 1 if flags.get("confirmation_ok", False) else 0,
        "atr_regime_ok": 1 if flags.get("atr_regime_ok", False) else 0,
        "z_score": z_score,
        "atr_percentile": atr_percentile,
        "momentum_roc": momentum_roc,
        "direction_bullish": 1 if direction == "bullish" else 0,
    }
    
    return features


def apply_ml_filter(
    features: Dict[str, float],
    min_prob: float = 0.6,
) -> Tuple[bool, float]:
    """
    Apply ML model to filter trades.
    
    Args:
        features: Dict of features from extract_ml_features
        min_prob: Minimum probability threshold for trade acceptance
    
    Returns:
        Tuple of (should_trade, probability)
    """
    import os
    
    model_path = "models/best_rf.joblib"
    
    if not os.path.exists(model_path):
        return (True, 1.0)
    
    try:
        import joblib
        model = joblib.load(model_path)
        
        feature_order = [
            "htf_aligned", "location_ok", "fib_ok", "structure_ok",
            "liquidity_ok", "confirmation_ok", "atr_regime_ok",
            "z_score", "atr_percentile", "momentum_roc", "direction_bullish"
        ]
        
        feature_values = [[features.get(f, 0) for f in feature_order]]
        
        probas = model.predict_proba(feature_values)
        prob_profitable = probas[0][1] if len(probas[0]) > 1 else probas[0][0]
        
        should_trade = prob_profitable >= min_prob
        return (should_trade, float(prob_profitable))
        
    except Exception as e:
        return (True, 1.0)


def check_volatility_filter(
    candles: List[Dict],
    atr_min_percentile: float = 60.0,
    current_date: Optional[datetime] = None,
    december_atr_multiplier: float = 1.0,
) -> Tuple[bool, float]:
    """
    Check if current volatility is above minimum threshold.
    In December, applies december_atr_multiplier to be more strict.
    """
    _, atr_percentile = _calculate_atr_percentile(candles, period=14, lookback=100)
    
    effective_threshold = atr_min_percentile
    if current_date and current_date.month == 12:
        effective_threshold = min(95.0, atr_min_percentile * december_atr_multiplier)
    
    passes_filter = atr_percentile >= effective_threshold
    return (passes_filter, atr_percentile)


VOLATILE_ASSETS = ["XAU_USD", "XAUUSD", "NAS100_USD", "NAS100USD", "GBP_JPY", "GBPJPY", "BTC_USD", "BTCUSD"]

def apply_volatile_asset_boost(
    symbol: str,
    confluence_score: int,
    quality_factors: int,
    volatile_asset_boost: float = 1.0,
) -> Tuple[int, int]:
    """
    Apply boost to confluence/quality scores for volatile assets.
    These assets (XAUUSD, NAS100USD, GBPJPY, BTCUSD) have potential for bigger R trades.
    """
    normalized_symbol = symbol.replace("_", "").upper()
    is_volatile = any(v.replace("_", "").upper() == normalized_symbol for v in VOLATILE_ASSETS)
    
    if is_volatile and volatile_asset_boost > 1.0:
        boosted_confluence = int(confluence_score * volatile_asset_boost)
        boosted_quality = int(quality_factors * volatile_asset_boost)
        return (boosted_confluence, boosted_quality)
    
    return (confluence_score, quality_factors)
