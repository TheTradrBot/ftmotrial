"""
Challenge Risk Manager for 5ers 60K High Stakes

Tracks daily P&L, total drawdown, and enforces risk limits to protect the challenge account.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Dict, Any, Tuple
from datetime import datetime, date
import json
from pathlib import Path
import logging

log = logging.getLogger(__name__)


class RiskMode(Enum):
    """Risk mode levels based on current P&L state."""
    NORMAL = auto()          # Normal trading, full risk
    CONSERVATIVE = auto()    # Warning level, reduced risk
    ULTRA_SAFE = auto()      # Near target, minimal risk
    HALTED = auto()          # Trading halted, too close to limits
    EMERGENCY = auto()       # Emergency stop, close all positions


class ActionType(Enum):
    """Actions the risk manager can recommend."""
    CONTINUE = auto()        # Continue normal trading
    REDUCE_RISK = auto()     # Reduce position sizes
    HALT_NEW_TRADES = auto() # Stop opening new positions
    CLOSE_PENDING = auto()   # Close all pending orders
    CLOSE_ALL = auto()       # Emergency close all positions


@dataclass
class AccountSnapshot:
    """Snapshot of account state."""
    balance: float
    equity: float
    peak_equity: float
    daily_pnl: float
    daily_loss_pct: float
    total_dd_pct: float
    # Additional fields needed by main_live_bot
    daily_pnl_pct: float = 0.0          # Daily P&L as percentage (negative = loss)
    total_drawdown_pct: float = 0.0      # Alias for total_dd_pct
    open_positions: int = 0              # Number of open positions
    total_risk_pct: float = 0.0          # Total risk exposure percentage


@dataclass
class ChallengeConfig:
    """Configuration for challenge risk management."""
    # Core settings
    enabled: bool = True
    phase: int = 1
    account_size: float = 60000.0
    
    # Risk limits (from FIVEERS_CONFIG)
    max_risk_per_trade_pct: float = 0.75
    max_cumulative_risk_pct: float = 5.0
    max_concurrent_trades: int = 7
    max_pending_orders: int = 20
    
    # Take profit percentages
    tp1_close_pct: float = 0.45
    tp2_close_pct: float = 0.30
    tp3_close_pct: float = 0.25
    
    # Daily loss thresholds
    daily_loss_warning_pct: float = 2.5
    daily_loss_reduce_pct: float = 3.5
    daily_loss_halt_pct: float = 4.2
    
    # Total drawdown thresholds
    total_dd_warning_pct: float = 5.0
    total_dd_emergency_pct: float = 7.0
    
    # Protection settings
    protection_loop_interval_sec: float = 30.0
    pending_order_max_age_hours: float = 24.0
    
    # Ultra-safe mode (near target)
    profit_ultra_safe_threshold_pct: float = 9.0
    ultra_safe_risk_pct: float = 0.25
    
    # Challenge rules
    max_daily_loss_pct: float = 5.0
    max_total_drawdown_pct: float = 10.0
    phase1_target_pct: float = 8.0
    phase2_target_pct: float = 5.0
    max_trades_per_day: int = 10
    risk_per_trade_pct: float = 0.6
    conservative_risk_pct: float = 0.4


class ChallengeRiskManager:
    """
    Manages risk for prop firm challenges (5ers, FTMO, etc.)
    
    Tracks:
    - Daily P&L and limits
    - Total drawdown from peak equity
    - Position counts and cumulative risk
    - Profit targets
    """
    
    def __init__(
        self,
        config: ChallengeConfig,
        mt5_client: Any = None,
        state_file: str = "challenge_risk_state.json"
    ):
        self.config = config
        self.mt5 = mt5_client
        self.state_file = Path(state_file)
        
        # State tracking
        self.starting_balance: float = config.account_size
        self.peak_equity: float = config.account_size
        self.current_balance: float = config.account_size
        self.current_equity: float = config.account_size
        
        self.day_start_balance: float = config.account_size
        self.daily_pnl: float = 0.0
        self.total_drawdown: float = 0.0
        self.total_drawdown_pct: float = 0.0
        self.daily_loss_pct: float = 0.0
        
        self.current_date: date = date.today()
        self.trades_today: int = 0
        self.risk_mode: RiskMode = RiskMode.NORMAL
        self.halted: bool = False  # Trading halted flag
        self.halt_reason: str = ""  # Reason for halt
        
        # Load persisted state
        self._load_state()
    
    def _load_state(self):
        """Load persisted state from file."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                
                self.starting_balance = state.get('starting_balance', self.config.account_size)
                self.peak_equity = state.get('peak_equity', self.config.account_size)
                self.day_start_balance = state.get('day_start_balance', self.config.account_size)
                self.trades_today = state.get('trades_today', 0)
                
                saved_date = state.get('current_date')
                if saved_date:
                    self.current_date = date.fromisoformat(saved_date)
                    
                log.info(f"Loaded challenge state: peak_equity=${self.peak_equity:,.2f}")
            except Exception as e:
                log.warning(f"Could not load state file: {e}")
    
    def _save_state(self):
        """Persist state to file."""
        state = {
            'starting_balance': self.starting_balance,
            'peak_equity': self.peak_equity,
            'day_start_balance': self.day_start_balance,
            'current_date': self.current_date.isoformat(),
            'trades_today': self.trades_today,
            'last_update': datetime.now().isoformat()
        }
        try:
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            log.error(f"Could not save state file: {e}")
    
    def sync_with_mt5(self, balance: float, equity: float):
        """
        Sync state with MT5 account data.
        Call this at startup and periodically.
        """
        today = date.today()
        
        # Check for new day
        if today != self.current_date:
            log.info(f"New trading day detected: {today}")
            self.day_start_balance = balance
            self.trades_today = 0
            self.current_date = today
        
        # Update current state
        self.current_balance = balance
        self.current_equity = equity
        
        # Update peak equity (high water mark)
        if equity > self.peak_equity:
            self.peak_equity = equity
            log.info(f"New peak equity: ${self.peak_equity:,.2f}")
        
        # Calculate metrics
        self.daily_pnl = balance - self.day_start_balance
        self.daily_loss_pct = abs(min(0, self.daily_pnl)) / self.day_start_balance * 100
        
        self.total_drawdown = self.peak_equity - equity
        self.total_drawdown_pct = self.total_drawdown / self.peak_equity * 100 if self.peak_equity > 0 else 0
        
        # Determine risk mode
        self._update_risk_mode()
        
        # Persist state
        self._save_state()
    
    def _update_risk_mode(self):
        """Update risk mode based on current metrics."""
        old_mode = self.risk_mode
        
        # Check for emergency conditions
        if self.total_drawdown_pct >= self.config.total_dd_emergency_pct:
            self.risk_mode = RiskMode.EMERGENCY
            log.critical(f"🚨 EMERGENCY: Total DD {self.total_drawdown_pct:.1f}% >= {self.config.total_dd_emergency_pct}%! CLOSING ALL POSITIONS!")
        elif self.daily_loss_pct >= self.config.daily_loss_halt_pct:
            self.risk_mode = RiskMode.HALTED
            log.error(f"🛑 HALT: Daily loss {self.daily_loss_pct:.1f}% >= {self.config.daily_loss_halt_pct}%! NO NEW TRADES!")
        elif self.daily_loss_pct >= self.config.daily_loss_reduce_pct:
            self.risk_mode = RiskMode.CONSERVATIVE
            log.warning(f"⚠️ DE-RISKING: Daily loss {self.daily_loss_pct:.1f}% >= {self.config.daily_loss_reduce_pct}%! Reducing risk to {self.config.conservative_risk_pct}%")
        elif self.daily_loss_pct >= self.config.daily_loss_warning_pct:
            # Warning level - still normal mode but log warning
            log.warning(f"⚠️ WARNING: Daily loss {self.daily_loss_pct:.1f}% approaching limit!")
            self.risk_mode = RiskMode.NORMAL
        elif self.total_drawdown_pct >= self.config.total_dd_warning_pct:
            self.risk_mode = RiskMode.CONSERVATIVE
            log.warning(f"⚠️ DE-RISKING: Total DD {self.total_drawdown_pct:.1f}% >= {self.config.total_dd_warning_pct}%!")
        else:
            # Check for ultra-safe mode (near profit target)
            profit_pct = (self.current_balance - self.starting_balance) / self.starting_balance * 100
            if profit_pct >= self.config.profit_ultra_safe_threshold_pct:
                self.risk_mode = RiskMode.ULTRA_SAFE
            else:
                self.risk_mode = RiskMode.NORMAL
        
        if old_mode != self.risk_mode:
            log.warning(f"Risk mode changed: {old_mode.name} → {self.risk_mode.name}")
    
    def can_trade(self) -> Tuple[bool, str, ActionType]:
        """
        Check if trading is allowed.
        
        Returns:
            Tuple of (allowed, reason, recommended_action)
        """
        # Emergency mode - close everything
        if self.risk_mode == RiskMode.EMERGENCY:
            return False, f"Emergency mode: DD={self.total_drawdown_pct:.1f}%", ActionType.CLOSE_ALL
        
        # Halted mode - no new trades
        if self.risk_mode == RiskMode.HALTED:
            return False, f"Trading halted: Daily loss={self.daily_loss_pct:.1f}%", ActionType.HALT_NEW_TRADES
        
        # Check daily trade limit
        if self.trades_today >= self.config.max_trades_per_day:
            return False, f"Daily trade limit reached: {self.trades_today}", ActionType.HALT_NEW_TRADES
        
        # Check position count if MT5 available
        if self.mt5:
            try:
                positions = self.mt5.get_my_positions()
                if len(positions) >= self.config.max_concurrent_trades:
                    return False, f"Max concurrent trades: {len(positions)}", ActionType.HALT_NEW_TRADES
            except:
                pass
        
        # Conservative mode - reduced risk but allowed
        if self.risk_mode == RiskMode.CONSERVATIVE:
            return True, "Conservative mode - reduced risk", ActionType.REDUCE_RISK
        
        # Ultra safe mode - minimal risk
        if self.risk_mode == RiskMode.ULTRA_SAFE:
            return True, "Ultra-safe mode - protecting profits", ActionType.REDUCE_RISK
        
        return True, "OK", ActionType.CONTINUE
    
    def get_adjusted_risk_pct(self) -> float:
        """Get the appropriate risk percentage for current mode."""
        if self.risk_mode == RiskMode.ULTRA_SAFE:
            return self.config.ultra_safe_risk_pct
        elif self.risk_mode == RiskMode.CONSERVATIVE:
            return self.config.conservative_risk_pct
        else:
            return self.config.risk_per_trade_pct
    
    def record_trade(self):
        """Record that a trade was opened."""
        self.trades_today += 1
        self._save_state()
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status summary."""
        return {
            'risk_mode': self.risk_mode.name,
            'balance': self.current_balance,
            'equity': self.current_equity,
            'peak_equity': self.peak_equity,
            'daily_pnl': self.daily_pnl,
            'daily_loss_pct': self.daily_loss_pct,
            'total_dd_pct': self.total_drawdown_pct,
            'trades_today': self.trades_today,
            'profit_pct': (self.current_balance - self.starting_balance) / self.starting_balance * 100
        }
    
    def run_protection_check(self) -> list:
        """
        Run protection checks and return list of actions to take.
        
        Returns:
            List of ActionType enums
        """
        actions = []
        
        # Sync with MT5 if available
        if self.mt5:
            try:
                balance = self.mt5.get_account_balance()
                equity = self.mt5.get_account_equity()
                self.sync_with_mt5(balance, equity)
            except:
                pass
        
        # Check for emergency
        if self.risk_mode == RiskMode.EMERGENCY:
            actions.append(ActionType.CLOSE_ALL)
            self.halted = True
            self.halt_reason = f"Emergency: Total DD {self.total_drawdown_pct:.1f}% >= {self.config.total_dd_emergency_pct}%"
        
        # Check for halt
        elif self.risk_mode == RiskMode.HALTED:
            actions.append(ActionType.HALT_NEW_TRADES)
            self.halted = True
            self.halt_reason = f"Daily loss {self.daily_loss_pct:.1f}% >= {self.config.daily_loss_halt_pct}%"
        
        # Check for conservative
        elif self.risk_mode == RiskMode.CONSERVATIVE:
            actions.append(ActionType.REDUCE_RISK)
        
        return actions
    
    def get_account_snapshot(self):
        """Get current account snapshot."""
        # Calculate daily P&L percentage (negative = loss)
        daily_pnl_pct = (self.daily_pnl / self.day_start_balance * 100) if self.day_start_balance > 0 else 0.0
        
        # Get open positions count from MT5 if available
        open_positions = 0
        if self.mt5_client:
            try:
                positions = self.mt5_client.get_positions()
                open_positions = len(positions) if positions else 0
            except Exception:
                pass
        
        # Calculate total risk as sum of all position risks
        total_risk_pct = open_positions * self.config.max_risk_per_trade_pct
        
        return AccountSnapshot(
            balance=self.current_balance,
            equity=self.current_equity,
            peak_equity=self.peak_equity,
            daily_pnl=self.daily_pnl,
            daily_loss_pct=self.daily_loss_pct,
            total_dd_pct=self.total_drawdown_pct,
            daily_pnl_pct=daily_pnl_pct,
            total_drawdown_pct=self.total_drawdown_pct,
            open_positions=open_positions,
            total_risk_pct=total_risk_pct
        )
    
    @property
    def initial_balance(self) -> float:
        """Get initial/starting balance."""
        return self.starting_balance
    
    def get_partial_close_volumes(self, total_volume: float) -> Tuple[float, float, float]:
        """
        Calculate volumes for partial closes at TP1, TP2, TP3.
        
        Args:
            total_volume: Total position volume
            
        Returns:
            Tuple of (tp1_volume, tp2_volume, tp3_volume)
        """
        tp1_vol = round(total_volume * self.config.tp1_close_pct, 2)
        tp2_vol = round(total_volume * self.config.tp2_close_pct, 2)
        tp3_vol = round(total_volume * self.config.tp3_close_pct, 2)
        
        # Ensure at least minimum lot size
        min_lot = 0.01
        tp1_vol = max(min_lot, tp1_vol) if tp1_vol > 0 else 0
        tp2_vol = max(min_lot, tp2_vol) if tp2_vol > 0 else 0
        tp3_vol = max(min_lot, tp3_vol) if tp3_vol > 0 else 0
        
        return tp1_vol, tp2_vol, tp3_vol
    
    def __str__(self) -> str:
        status = self.get_status()
        return (
            f"ChallengeRiskManager:\n"
            f"  Mode: {status['risk_mode']}\n"
            f"  Balance: ${status['balance']:,.2f}\n"
            f"  Daily P&L: ${status['daily_pnl']:+,.2f} ({status['daily_loss_pct']:.1f}% loss)\n"
            f"  Total DD: {status['total_dd_pct']:.1f}%\n"
            f"  Profit: {status['profit_pct']:+.1f}%\n"
            f"  Trades today: {status['trades_today']}"
        )


def create_challenge_manager(
    account_size: float = 60000.0,
    mt5_client: Any = None,
    **kwargs
) -> ChallengeRiskManager:
    """
    Factory function to create a ChallengeRiskManager with custom config.
    """
    config = ChallengeConfig(account_size=account_size, **kwargs)
    return ChallengeRiskManager(config=config, mt5_client=mt5_client)
