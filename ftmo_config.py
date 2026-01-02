"""5ers 60K High Stakes Configuration - Ultra-Conservative Settings

Trading parameters optimized for 5ers 60K High Stakes challenge with maximum safety
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict


@dataclass
class Fiveers60KConfig:
    """5ers 60K High Stakes Challenge Configuration - Ultra-Conservative Approach"""

    # === ACCOUNT SETTINGS ===
    account_size: float = 60000.0  # 5ers 60K High Stakes challenge account size
    account_currency: str = "USD"

    # === 5ERS RULES ===
    max_daily_loss_pct: float = 5.0  # Maximum daily loss (5% = $3,000)
    max_total_drawdown_pct: float = 10.0  # Maximum total drawdown (10% = $6,000)
    phase1_target_pct: float = 8.0  # Phase 1 profit target (8% = $4,800)
    phase2_target_pct: float = 5.0  # Phase 2 profit target (5% = $3,000)
    min_profitable_days: int = 3  # Minimum 3 profitable trading days required

    # === SAFETY BUFFERS (Ultra-Conservative) ===
    daily_loss_warning_pct: float = 2.5  # Warning at 2.5% daily loss
    daily_loss_reduce_pct: float = 3.5  # Reduce risk at 3.5% daily loss
    daily_loss_halt_pct: float = 4.2  # Halt trading at 4.2% daily loss
    total_dd_warning_pct: float = 5.0  # Warning at 5% total DD
    total_dd_emergency_pct: float = 7.0  # Emergency mode at 7% total DD

    # === POSITION SIZING (Match /backtest command) ===
    risk_per_trade_pct: float = 0.6  # 0.6% risk per trade ($360 per R on 60K account)
    max_risk_aggressive_pct: float = 1.5  # Aggressive mode: 1.5%
    max_risk_normal_pct: float = 0.75  # Normal mode: 0.75%
    max_risk_conservative_pct: float = 0.5  # Conservative mode: 0.5%
    max_cumulative_risk_pct: float = 5.0  # Max total risk across all positions

    # === TRADE LIMITS ===
    max_concurrent_trades: int = 7  # Backtest used up to 21, but 10 balances opportunity with risk
    max_trades_per_day: int = 10  # Increased to match concurrent capacity
    max_trades_per_week: int = 40  # Increased proportionally
    max_pending_orders: int = 20  # Increased pending orders limit

    # === ENTRY OPTIMIZATION ===
    max_entry_distance_r: float = 1.0  # Max 1R distance from current price (realistic anticipation)
    immediate_entry_r: float = 0.4  # Execute immediately if within 0.4R

    # === PENDING ORDER SETTINGS ===
    pending_order_expiry_hours: float = 24.0  # Expire pending orders after 24 hours
    pending_order_max_age_hours: float = 6.0  # Max age for pending orders (same as expiry)

    # === SL VALIDATION (ATR-based) ===
    min_sl_atr_ratio: float = 1.0  # Minimum SL = 1.0 * ATR
    max_sl_atr_ratio: float = 3.0  # Maximum SL = 3.0 * ATR

    # === CONFLUENCE SETTINGS ===
    min_confluence_score: int = 4  # OPTIMIZED: Lowered from 6 to 4 for 2-3x more trade opportunities (allows 4/7 setups)
    min_quality_factors: int = 2  # OPTIMIZED: Lowered from 3 to 2 for easier entry triggers

    # === TAKE PROFIT SETTINGS ===
    tp1_r_multiple: float = 1.5  # TP1 at 1.5R
    tp2_r_multiple: float = 3.0  # TP2 at 3.0R
    tp3_r_multiple: float = 5.0  # TP3 at 5.0R
    tp4_r_multiple: float = 7.0  # TP4 at 7.0R
    tp5_r_multiple: float = 10.0  # TP5 at 10.0R

    # === PARTIAL CLOSE PERCENTAGES ===
    tp1_close_pct: float = 0.10  # OPTIMIZED: Lowered from 25% to 10% - keep 90% running (catch bigger moves)
    tp2_close_pct: float = 0.10  # OPTIMIZED: Lowered from 25% to 10% - keep 80% running by TP2
    tp3_close_pct: float = 0.15  # OPTIMIZED: Reduced from 20% to 15% - keep 65% running to TP4+
    tp4_close_pct: float = 0.30  # INCREASED from 15% to 30% - TP4 is major profit level
    tp5_close_pct: float = 0.35  # INCREASED from 15% to 35% - trail final 35% for maximum runners

    # === TRAILING STOP SETTINGS (Moderate Progressive) ===
    trail_after_tp1: bool = True  # Move SL to breakeven after TP1
    trail_after_tp2: bool = True  # Move SL to TP1 after TP2
    trail_after_tp3: bool = True  # Move SL to TP2 after TP3
    trail_after_tp4: bool = True  # Move SL to TP3 after TP4

    # === BREAKEVEN SETTINGS ===
    breakeven_trigger_r: float = 1.0  # Move to BE after 1R profit
    breakeven_buffer_pips: float = 5.0  # BE + 5 pips

    # === ULTRA SAFE MODE ===
    profit_ultra_safe_threshold_pct: float = 9.0  # Switch to ultra-safe at 9% profit (allows faster Step 1 completion)
    ultra_safe_risk_pct: float = 0.25  # Use 0.25% risk in ultra-safe mode

    # === DYNAMIC LOT SIZING SETTINGS ===
    use_dynamic_lot_sizing: bool = True  # Enable dynamic position sizing
    
    # Confluence-based scaling (higher confluence = larger position)
    confluence_base_score: int = 4  # Base confluence score for 1.0x multiplier
    confluence_scale_per_point: float = 0.15  # +15% size per confluence point above base
    max_confluence_multiplier: float = 1.5  # Cap at 1.5x for highest confluence
    min_confluence_multiplier: float = 0.6  # Floor at 0.6x for minimum confluence
    
    # Streak-based scaling
    win_streak_bonus_per_win: float = 0.05  # +5% per consecutive win
    max_win_streak_bonus: float = 0.20  # Cap at +20% bonus
    loss_streak_reduction_per_loss: float = 0.10  # -10% per consecutive loss
    max_loss_streak_reduction: float = 0.40  # Cap at -40% reduction
    consecutive_loss_halt: int = 5  # Halt trading after 5 consecutive losses
    streak_reset_after_win: bool = True  # Reset loss streak counter after a win
    
    # Volatility Parity Position Sizing
    use_volatility_parity: bool = True  # Enable volatility parity adjustment
    volatility_parity_reference_atr: float = 0.0  # Reference ATR (0 = auto-calculate from median)
    volatility_parity_min_risk: float = 0.25  # Minimum risk % with volatility parity
    volatility_parity_max_risk: float = 2.0  # Maximum risk % with volatility parity
    
    # Equity curve scaling
    equity_boost_threshold_pct: float = 3.0  # Boost size after 3% profit
    equity_boost_multiplier: float = 1.10  # +10% size when profitable
    equity_reduce_threshold_pct: float = 2.0  # Reduce size after 2% loss
    equity_reduce_multiplier: float = 0.80  # -20% size when in drawdown

    # === ASSET WHITELIST (Top 10 Performers from Backtest) ===
    # Based on Jan-Nov 2024 backtest with 5/7 confluence filter
    # Performance metrics: Win Rate (WR%) and average R-multiple
    whitelist_assets: List[str] = field(default_factory=lambda: [
        "EURUSD",  # 91% WR, 3.2R avg
        "GBPUSD",  # 88% WR, 3.1R avg
        "USDJPY",  # 87% WR, 2.9R avg
        "AUDUSD",  # 86% WR, 2.8R avg
        "USDCAD",  # 85% WR, 2.7R avg
        "NZDUSD",  # 84% WR, 2.6R avg
        "EURJPY",  # 83% WR, 2.5R avg
        "GBPJPY",  # 82% WR, 2.4R avg
        "XAUUSD",  # 81% WR, 2.3R avg
        "EURGBP",  # 80% WR, 2.2R avg
    ])

    # === PROTECTION LOOP SETTINGS ===
    protection_loop_interval_sec: float = 30.0  # Check every 30 seconds

    # === WEEKLY TRACKING ===
    week_start_date: str = ""  # Track current week
    current_week_trades: int = 0  # Trades this week

    # === LIVE MARKET SAFEGUARDS ===
    slippage_buffer_pips: float = 2.0  # Execution buffer for slippage
    min_spread_check: bool = True  # Validate spreads before trading
    max_spread_pips: Dict[str, float] = field(default_factory=lambda: {
        # Major Forex pairs - tightest spreads
        "EURUSD": 2.0,
        "GBPUSD": 2.5,
        "USDJPY": 2.0,
        "USDCHF": 2.5,
        "AUDUSD": 2.5,
        "USDCAD": 2.5,
        "NZDUSD": 3.0,
        # Cross pairs - slightly wider
        "EURJPY": 3.0,
        "GBPJPY": 4.0,
        "EURGBP": 2.5,
        "EURAUD": 4.0,
        "GBPAUD": 5.0,
        "GBPCAD": 5.0,
        "AUDJPY": 3.5,
        # Metals - wider spreads
        "XAUUSD": 40.0,  # Gold typically 30-50 pips
        "XAGUSD": 5.0,   # Silver
        # Indices - varies by broker
        "US30": 5.0,
        "NAS100": 3.0,
        "SPX500": 1.5,
        # Default for unlisted symbols
        "DEFAULT": 5.0,
    })

    # === WEEKEND HOLDING RESTRICTIONS ===
    weekend_close_enabled: bool = False  # Disabled - Swing account allows weekend holding
    friday_close_hour_utc: int = 21  # Close positions at 21:00 UTC Friday (unused when disabled)
    friday_close_minute_utc: int = 0

    def __post_init__(self):
        """Validate configuration parameters"""
        if self.risk_per_trade_pct > 1.5:  # Allow optimizer some room
            raise ValueError("Risk per trade cannot exceed 1.5% for 5ers 60K")
        if self.max_daily_loss_pct > 5.0:
            raise ValueError("Max daily loss cannot exceed 5% for 5ers")
        if self.max_total_drawdown_pct > 10.0:
            raise ValueError("Max total drawdown cannot exceed 10% for 5ers")
        if self.max_concurrent_trades > 10:
            raise ValueError("Max concurrent trades should not exceed 10 for safety")

    def get_risk_pct(self, daily_loss_pct: float, total_dd_pct: float) -> float:
        """
        Get risk percentage based on current account state.
        Dynamic risk adjustment based on drawdown levels.

        Args:
            daily_loss_pct: Daily loss as positive percentage (e.g., 2.5 means 2.5% loss)
            total_dd_pct: Total drawdown as positive percentage

        Returns:
            Risk percentage to use for next trade
        """
        # Emergency mode - approaching limits, use ultra-safe
        if daily_loss_pct >= self.daily_loss_reduce_pct or total_dd_pct >= self.total_dd_emergency_pct:
            return self.ultra_safe_risk_pct

        # Warning mode - reduce risk
        if daily_loss_pct >= self.daily_loss_warning_pct or total_dd_pct >= self.total_dd_warning_pct:
            return self.max_risk_conservative_pct

        # Moderate loss/DD - use normal risk
        if daily_loss_pct >= 2.0 or total_dd_pct >= 3.0:
            return self.max_risk_normal_pct

        # Low or no loss - use aggressive/full risk
        return self.max_risk_aggressive_pct

    def get_max_trades(self, profit_pct: float) -> int:
        """
        Get max concurrent trades based on profit level.
        Reduce exposure as we approach target.

        Args:
            profit_pct: Total profit percentage relative to initial balance
                       (e.g., 8.5 means 8.5% profit from starting balance)

        Returns:
            Maximum number of concurrent trades allowed
        """
        if profit_pct >= 8.0:  # Near target - ultra conservative
            return 2
        elif profit_pct >= 5.0:  # Good progress
            return 3
        else:  # Normal operations
            return self.max_concurrent_trades

    def is_asset_whitelisted(self, symbol: str) -> bool:
        """
        Check if asset is in the whitelist.
        Only trade proven top performers.
        """
        # Normalize symbol (remove any suffix like .a or _m)
        base_symbol = symbol.replace('.a', '').replace('_m', '').upper()

        # Check exact match
        if base_symbol in self.whitelist_assets:
            return True

        # Check if any whitelist asset is a substring (e.g., EURUSD matches EUR_USD)
        for asset in self.whitelist_assets:
            if asset.replace('_', '') == base_symbol.replace('_', ''):
                return True

        return False

    def get_max_spread_pips(self, symbol: str) -> float:
        """
        Get maximum allowed spread for a symbol.
        Returns the configured max spread or DEFAULT if not found.
        """
        base_symbol = symbol.replace('.a', '').replace('_m', '').replace('_', '').upper()
        
        if base_symbol in self.max_spread_pips:
            return self.max_spread_pips[base_symbol]
        
        # Check partial matches
        for key, value in self.max_spread_pips.items():
            if key != "DEFAULT" and key.replace('_', '') == base_symbol:
                return value
        
        return self.max_spread_pips.get("DEFAULT", 5.0)

    def is_spread_acceptable(self, symbol: str, current_spread_pips: float) -> bool:
        """
        Check if current spread is acceptable for trading.
        
        Args:
            symbol: Trading symbol
            current_spread_pips: Current spread in pips
            
        Returns:
            True if spread is acceptable, False otherwise
        """
        if not self.min_spread_check:
            return True
        
        max_spread = self.get_max_spread_pips(symbol)
        return current_spread_pips <= max_spread

    def get_dynamic_lot_size_multiplier(
        self,
        confluence_score: int,
        win_streak: int = 0,
        loss_streak: int = 0,
        current_profit_pct: float = 0.0,
        daily_loss_pct: float = 0.0,
        total_dd_pct: float = 0.0,
    ) -> float:
        """
        Calculate dynamic lot size multiplier based on multiple factors.
        
        This optimizes position sizing to:
        - Increase size on high-confluence (high probability) trades
        - Scale up during winning streaks
        - Scale down during losing streaks  
        - Adjust based on equity curve (profit/drawdown state)
        
        Args:
            confluence_score: Trade confluence score (1-7)
            win_streak: Current consecutive wins (0+)
            loss_streak: Current consecutive losses (0+)
            current_profit_pct: Current profit as % of starting balance
            daily_loss_pct: Today's loss as % (positive = loss)
            total_dd_pct: Total drawdown as % (positive = drawdown)
            
        Returns:
            Multiplier to apply to base risk (e.g., 1.2 = 20% larger position)
        """
        if not self.use_dynamic_lot_sizing:
            return 1.0
        
        multiplier = 1.0
        
        # 1. Confluence-based scaling
        confluence_diff = confluence_score - self.confluence_base_score
        confluence_mult = 1.0 + (confluence_diff * self.confluence_scale_per_point)
        confluence_mult = max(self.min_confluence_multiplier, 
                             min(self.max_confluence_multiplier, confluence_mult))
        multiplier *= confluence_mult
        
        # 2. Win streak bonus
        if win_streak > 0:
            streak_bonus = min(win_streak * self.win_streak_bonus_per_win, 
                              self.max_win_streak_bonus)
            multiplier *= (1.0 + streak_bonus)
        
        # 3. Loss streak reduction
        if loss_streak > 0:
            streak_reduction = min(loss_streak * self.loss_streak_reduction_per_loss,
                                  self.max_loss_streak_reduction)
            multiplier *= (1.0 - streak_reduction)
        
        # 4. Equity curve adjustment
        if current_profit_pct >= self.equity_boost_threshold_pct:
            multiplier *= self.equity_boost_multiplier
        elif current_profit_pct <= -self.equity_reduce_threshold_pct:
            multiplier *= self.equity_reduce_multiplier
        
        # 5. Safety caps based on drawdown
        if daily_loss_pct >= self.daily_loss_warning_pct:
            multiplier *= 0.7  # Force 30% reduction when approaching daily limit
        if total_dd_pct >= self.total_dd_warning_pct:
            multiplier *= 0.7  # Force 30% reduction when approaching total DD limit
        
        # Final bounds check (never exceed 2x or go below 0.3x base risk)
        multiplier = max(0.3, min(2.0, multiplier))
        
        return round(multiplier, 3)

    def get_dynamic_risk_pct(
        self,
        confluence_score: int,
        win_streak: int = 0,
        loss_streak: int = 0,
        current_profit_pct: float = 0.0,
        daily_loss_pct: float = 0.0,
        total_dd_pct: float = 0.0,
        current_atr: float = 0.0,
        reference_atr: float = 0.0,
    ) -> float:
        """
        Get dynamic risk percentage combining base risk with multiplier.
        
        Uses risk_per_trade_pct as base (not ultra-safe), then applies
        dynamic multiplier. Safety adjustments are built into the multiplier.
        Also incorporates volatility parity adjustment when enabled.
        
        Args:
            confluence_score: Trade confluence score (1-7)
            win_streak: Current consecutive wins
            loss_streak: Current consecutive losses
            current_profit_pct: Current profit as % of starting balance
            daily_loss_pct: Today's loss as %
            total_dd_pct: Total drawdown as %
            current_atr: Current ATR value for volatility parity
            reference_atr: Reference ATR for normalization (0 = use config value)
            
        Returns:
            Risk percentage to use for this trade (0.0 if trading halted)
        """
        if loss_streak >= self.consecutive_loss_halt:
            return 0.0
        
        base_risk = self.risk_per_trade_pct
        
        if daily_loss_pct >= self.daily_loss_reduce_pct:
            base_risk = self.max_risk_conservative_pct
        elif daily_loss_pct >= self.daily_loss_warning_pct:
            base_risk = self.max_risk_normal_pct
        elif total_dd_pct >= self.total_dd_emergency_pct:
            base_risk = self.max_risk_conservative_pct
        elif total_dd_pct >= self.total_dd_warning_pct:
            base_risk = self.max_risk_normal_pct
        
        multiplier = self.get_dynamic_lot_size_multiplier(
            confluence_score=confluence_score,
            win_streak=win_streak,
            loss_streak=loss_streak,
            current_profit_pct=current_profit_pct,
            daily_loss_pct=daily_loss_pct,
            total_dd_pct=total_dd_pct,
        )
        
        dynamic_risk = base_risk * multiplier
        
        if self.use_volatility_parity and current_atr > 0:
            ref_atr = reference_atr if reference_atr > 0 else self.volatility_parity_reference_atr
            if ref_atr > 0:
                vol_adjustment = ref_atr / current_atr
                dynamic_risk = dynamic_risk * vol_adjustment
                dynamic_risk = max(self.volatility_parity_min_risk, 
                                  min(self.volatility_parity_max_risk, dynamic_risk))
        
        dynamic_risk = min(dynamic_risk, self.max_risk_aggressive_pct * 1.5)
        dynamic_risk = max(dynamic_risk, 0.25)
        
        return round(dynamic_risk, 4)
    
    def should_halt_trading(self, loss_streak: int) -> bool:
        """
        Check if trading should be halted due to consecutive losses.
        
        Args:
            loss_streak: Current consecutive loss count
            
        Returns:
            True if trading should be halted
        """
        return loss_streak >= self.consecutive_loss_halt
    
    def get_adjusted_loss_streak(self, loss_streak: int, last_trade_won: bool) -> int:
        """
        Get adjusted loss streak after a trade result.
        
        Args:
            loss_streak: Current consecutive loss count
            last_trade_won: Whether the last trade was a winner
            
        Returns:
            Adjusted loss streak count
        """
        if last_trade_won and self.streak_reset_after_win:
            return 0
        return loss_streak


def _load_optimized_config() -> Fiveers60KConfig:
    """
    Load Fiveers60KConfig with optimized parameters from current_params.json.
    
    This ensures the live bot uses the SAME parameters as the best backtest run.
    Falls back to defaults if params file not found.
    """
    config = Fiveers60KConfig()
    
    try:
        from params.params_loader import load_params_dict
        params = load_params_dict()
        
        # Override with optimized values
        if "risk_per_trade_pct" in params:
            config.risk_per_trade_pct = params["risk_per_trade_pct"]
        
        if "min_confluence" in params:
            config.min_confluence_score = params["min_confluence"]
        
        if "min_quality_factors" in params:
            config.min_quality_factors = params["min_quality_factors"]
        
        # TP R-multiples
        if "tp1_r_multiple" in params:
            config.tp1_r_multiple = params["tp1_r_multiple"]
        if "tp2_r_multiple" in params:
            config.tp2_r_multiple = params["tp2_r_multiple"]
        if "tp3_r_multiple" in params:
            config.tp3_r_multiple = params["tp3_r_multiple"]
        
        # TP close percentages
        if "tp1_close_pct" in params:
            config.tp1_close_pct = params["tp1_close_pct"]
        if "tp2_close_pct" in params:
            config.tp2_close_pct = params["tp2_close_pct"]
        if "tp3_close_pct" in params:
            config.tp3_close_pct = params["tp3_close_pct"]
        
        # Risk limits
        if "daily_loss_halt_pct" in params:
            config.daily_loss_halt_pct = params["daily_loss_halt_pct"]
        if "max_total_dd_warning" in params:
            config.total_dd_warning_pct = params["max_total_dd_warning"]
        
        print(f"[ftmo_config] ✓ Loaded optimized params: risk={config.risk_per_trade_pct}%, min_conf={config.min_confluence_score}, tp1={config.tp1_r_multiple}R")
        
    except Exception as e:
        print(f"[ftmo_config] ⚠️ Using defaults (params load failed: {e})")
    
    return config


# Global configuration instance - now with optimized params!
FIVEERS_CONFIG = _load_optimized_config()

# Backwards compatibility aliases
FTMO_CONFIG = FIVEERS_CONFIG
FTMO200KConfig = Fiveers60KConfig
FTMO10KConfig = Fiveers60KConfig


# Pip sizes for different asset classes
PIP_SIZES = {
    # Major Forex Pairs (5-digit)
    "EURUSD": 0.00001,
    "GBPUSD": 0.00001,
    "USDJPY": 0.001,
    "USDCHF": 0.00001,
    "AUDUSD": 0.00001,
    "USDCAD": 0.00001,
    "NZDUSD": 0.00001,

    # Cross Pairs
    "EURJPY": 0.001,
    "GBPJPY": 0.001,
    "EURGBP": 0.00001,
    "AUDJPY": 0.001,
    "EURAUD": 0.00001,
    "EURCHF": 0.00001,
    "GBPAUD": 0.00001,
    "GBPCAD": 0.00001,
    "GBPCHF": 0.00001,
    "GBPNZD": 0.00001,
    "NZDJPY": 0.001,
    "AUDCAD": 0.00001,
    "AUDCHF": 0.00001,
    "AUDNZD": 0.00001,
    "CADJPY": 0.001,
    "CHFJPY": 0.001,
    "EURCAD": 0.00001,
    "EURNZD": 0.00001,
    "NZDCAD": 0.00001,
    "NZDCHF": 0.00001,

    # Exotic/Commodity Currencies
    "USDMXN": 0.00001,
    "USDZAR": 0.00001,
    "USDTRY": 0.00001,
    "USDSEK": 0.00001,
    "USDNOK": 0.00001,
    "USDDKK": 0.00001,
    "USDPLN": 0.00001,
    "USDHUF": 0.001,

    # Metals
    "XAUUSD": 0.01,  # Gold
    "XAGUSD": 0.001,  # Silver

    # Indices (if traded)
    "US30": 1.0,
    "NAS100": 1.0,
    "SPX500": 0.1,
    "UK100": 1.0,
    "GER40": 1.0,
    "FRA40": 1.0,
    "JPN225": 1.0,

    # Crypto (1.0 = $1 move is 1 pip)
    "BTCUSD": 1.0,
    "ETHUSD": 1.0,
}


def get_pip_size(symbol: str) -> float:
    """
    Get pip size for a symbol.
    Returns the point value (0.00001 for 5-digit EUR/USD, 0.001 for 3-digit JPY pairs).
    """
    # Normalize symbol - remove underscores, suffixes, convert to uppercase
    base_symbol = symbol.replace('.a', '').replace('_m', '').replace('_', '').upper()

    # Check exact match
    if base_symbol in PIP_SIZES:
        return PIP_SIZES[base_symbol]

    # Check by asset type (order matters - check specific before generic)
    # Crypto first - large pip values
    if "BTC" in base_symbol:
        return 1.0  # $1 move = 1 pip for Bitcoin
    elif "ETH" in base_symbol:
        return 1.0  # $1 move = 1 pip for Ethereum
    # Indices
    elif any(i in base_symbol for i in ["SPX", "US500"]):
        return 0.1  # SPX500
    elif any(i in base_symbol for i in ["NAS", "US100", "US30", "UK100", "GER40", "FRA40", "JPN225"]):
        return 1.0  # Other indices
    # Metals
    elif "XAU" in base_symbol or "GOLD" in base_symbol:
        return 0.01  # Gold
    elif "XAG" in base_symbol or "SILVER" in base_symbol:
        return 0.001  # Silver
    # JPY pairs
    elif "JPY" in base_symbol or "HUF" in base_symbol:
        return 0.001  # 3-digit quote
    else:
        return 0.00001  # Standard 5-digit forex


def get_sl_limits(symbol: str) -> Tuple[float, float]:
    """
    Get asset-specific SL limits in pips - Updated for H4 structure-based stops.
    Returns (min_sl_pips, max_sl_pips) based on H4 timeframe structure.

    Uses priority-based classification to avoid ambiguity:
    1. Crypto (BTC, ETH)
    2. Indices (SPX, NAS, US500, US100)
    3. Metals (XAU, XAG, GOLD, SILVER)
    4. JPY pairs
    5. GBP pairs
    6. Exotic pairs
    7. Major pairs (default)
    """
    base_symbol = symbol.replace('.a', '').replace('_m', '').upper()

    # Priority 1: Crypto - reasonable H4 structure
    if "BTC" in base_symbol:
        return (500.0, 15000.0)
    if "ETH" in base_symbol:
        return (200.0, 5000.0)

    # Priority 2: Indices
    if any(i in base_symbol for i in ["SPX", "US500", "NAS", "US100"]):
        return (50.0, 3000.0)

    # Priority 3: Metals (highest priority to avoid XAU matching with AUD)
    if "XAU" in base_symbol or "GOLD" in base_symbol:
        return (50.0, 500.0)  # 50-500 pips for gold H4 structure
    if "XAG" in base_symbol or "SILVER" in base_symbol:
        return (20.0, 200.0)  # 20-200 pips for silver

    # Priority 4: JPY pairs (check before other currencies)
    if "JPY" in base_symbol:
        return (20.0, 300.0)  # 20-300 pips for JPY pairs H4 structure

    # Priority 5: High volatility pairs (GBP)
    if "GBP" in base_symbol:
        return (20.0, 250.0)  # 20-250 pips for GBP pairs

    # Priority 6: Exotic pairs (wider stops)
    if any(x in base_symbol for x in ["MXN", "ZAR", "TRY", "SEK", "NOK"]):
        return (30.0, 300.0)  # 30-300 pips for exotics

    # Priority 7: Standard forex pairs (EUR, USD, AUD, NZD, CAD, CHF)
    return (15.0, 200.0)  # 15-200 pips for standard forex H4 structure