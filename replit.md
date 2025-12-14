# Blueprint Trader AI

## Overview
Blueprint Trader AI is an automated trading bot designed for FTMO Challenge accounts. It operates 24/7 on a Windows VM with MetaTrader 5, executing trades based on a rigorously backtested strategy. A lightweight Flask web server provides monitoring capabilities, while the core trading operations remain independent on the VM. The project aims to maximize success rates in proprietary trading challenges by adhering strictly to FTMO rules and employing advanced risk management.

## User Preferences
- Preferred communication style: Simple, everyday language
- Strategy must use EXACT SAME logic as backtests
- Bot must trade independently (no Discord dependency for trades)
- Pre-trade risk checks to prevent FTMO rule violations
- Using FTMO demo account for trading
- Using OANDA API for data fetching

## System Architecture

### Core Components
The system is comprised of two main parts:
1.  **Standalone MT5 Bot (`main_live_bot.py`)**: Runs independently on a Windows VM, handling 24/7 trading operations, signal generation via `strategy_core.py`, auto-reconnection, and scheduled tasks for continuous operation.
2.  **Minimal Flask Web Server**: Provides a simple monitoring interface, replacing the previous Discord bot for status updates.

### Trading Strategy
The bot employs a "7 Confluence Pillars" strategy, evaluating setups based on:
1.  HTF Bias (Monthly/Weekly/Daily trend)
2.  Location (S/R zones)
3.  Fibonacci (Golden Pocket)
4.  Liquidity (Sweep near equal highs/lows)
5.  Structure (BOS/CHoCH alignment)
6.  Confirmation (4H candle patterns)
7.  Risk:Reward (Min 1:1)

Trades are executed only if confluence is `ACTIVE` (>= 4 pillars, quality >= 1, valid R:R).

### Risk Management
The system incorporates an "Elite 7-layer safety system" designed for FTMO challenges:
-   **Global Risk Controller**: Real-time P/L tracking, proactive SL adjustment, emergency closes.
-   **Dynamic Position Sizing**: Adaptive sizing (0.75% base risk) based on current drawdown, partial scaling (45% TP1, 30% TP2, 25% TP3).
-   **Smart Concurrent Trade Limit**: Maximum 5 open positions, 6 pending orders.
-   **Pending Order Management**: Risk-based and time-based cancellation of pending orders.
-   **Live Equity Protection Loop**: 30-second monitoring cycle for automatic protective actions.
-   **Challenge-Optimized Behavior**: Five risk modes (Aggressive, Normal, Conservative, Ultra-Safe, Halted) dynamically adjust risk based on drawdown.
-   **Core Strategy Integration**: Safety layers wrap around the core entry logic.

Pre-trade risk simulations prevent breaches of FTMO daily and total drawdown limits.

### Supported Assets & Symbol Mapping
The bot trades 34 assets, including Forex pairs (Majors, EUR Crosses, GBP Crosses, AUD Crosses, NZD Crosses, Other Crosses), Metals (XAUUSD, XAGUSD), Crypto (BTCUSD, ETHUSD), and Indices (US500, US100). A `symbol_mapping.py` handles conversions between OANDA (data source) and FTMO MT5 (trading platform) naming conventions (e.g., `EUR_USD` to `EURUSD`, `SPX500_USD` to `US500`).

### Technical Implementations
-   **Entry Price Validation**: Trades only execute if the price reaches the calculated limit order level within 5 bars, preventing look-ahead bias from backtests.
-   **Dynamic Lot Sizing**: Adapts position size based on confluence scores, win/loss streaks, and equity curve performance, with safety clamps near DD limits.
-   **H4 Stop Loss Calculation**: Uses H4 timeframe structure for tighter stop losses and larger position sizes.
-   **Sequential Partial Take-Profit**: Ensures TP1 hits before TP2, correctly calculating weighted partial profits.
-   **Deployment**: Automated PowerShell scripts (`deploy.ps1`) for Windows VM setup, including Python, Git, virtual environment, and scheduled tasks.

## External Dependencies
-   **MetaTrader5**: Trading platform for execution on Windows VM.
-   **OANDA API**: Used for fetching market data.
-   **Python Libraries**:
    -   `pandas`, `numpy`: Data manipulation.
    -   `requests`: HTTP requests.
    -   `python-dotenv`: Environment variable management.
    -   `flask`, `flask-cors`: Web server for monitoring.
    -   `discord-py`: (Deprecated for live bot, previously used for minimal Discord monitoring).
-   **Dukascopy**: Integrated for historical tick data validation.

## Recent Changes (December 2024)

### FTMO 200K Swing Account Optimizations
1. **Trading Days Tracking**: Added automatic tracking of minimum trading days requirement (4 days) with warnings when approaching deadline without enough days traded.

2. **Configuration Sync**: Synchronized all configuration values between `ftmo_config.py` and optimizer output:
   - `risk_per_trade_pct`: 0.95%
   - `min_confluence_score`: 3/7
   - `min_quality_factors`: 1

3. **Live Market Safeguards**: Added execution protection:
   - `slippage_buffer_pips`: 2 pips execution buffer
   - `max_spread_pips`: Per-symbol spread limits (rejects trades with excessive spreads)
   - `is_spread_acceptable()`: Spread validation before trade entry

4. **Weekend Holding**: Disabled weekend close (`weekend_close_enabled: False`) - Swing account allows holding positions over weekends.

5. **Comprehensive Checklist**: Created `LIVE_TRADING_CHECKLIST.md` with:
   - Pre-launch verification steps
   - Launch day procedures
   - Daily/weekly monitoring checklists
   - Emergency procedures
   - Risk management thresholds
   - Windows Task Scheduler auto-restart setup

### Key Files Updated
- `main_live_bot.py`: Trading days tracking methods added
- `ftmo_config.py`: Live market safeguards, weekend close disabled
- `LIVE_TRADING_CHECKLIST.md`: Comprehensive pre-live checklist (NEW)