# Live Trading Bot -- Architecture and Operations Guide

> **Last updated**: 2026-02-08
> **Applies to**: dual_v3 live trading system (5ers prop account)
> **Entry point**: `dual_v3/main.py`

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Core Components](#2-core-components)
3. [Finite State Machine (FSM)](#3-finite-state-machine-fsm)
4. [Main Loop Data Flow](#4-main-loop-data-flow)
5. [Position Sizing Formula](#5-position-sizing-formula)
6. [Trailing Stop Loss (TSL) Logic](#6-trailing-stop-loss-tsl-logic)
7. [State Restoration on Restart](#7-state-restoration-on-restart)
8. [DRY_RUN Mode](#8-dry_run-mode)
9. [News Filter](#9-news-filter)
10. [Timezone Handling](#10-timezone-handling)
11. [Configuration Reference](#11-configuration-reference)
12. [Magic Numbers and Symbol Mapping](#12-magic-numbers-and-symbol-mapping)
13. [Safety Features and Prop Firm Compliance](#13-safety-features-and-prop-firm-compliance)
14. [Startup and Shutdown Procedure](#14-startup-and-shutdown-procedure)

---

## 1. Architecture Overview

The live trading bot implements a **5-Component FSM (Finite State Machine) Model** defined in the Architecture Specification 3.0. It connects to MetaTrader 5 (MT5) via the official Python API and trades two instruments -- GER40 (DAX40) and XAUUSD (Gold) -- on a 5ers prop firm account.

### High-Level Architecture

```
+-------------------------------------------------------------+
|                      BotController                          |
|  (Main orchestrator -- 2-second polling loop)               |
|                                                             |
|  +------------------+  +------------------+                 |
|  |   MT5Executor    |  |   RiskManager    |                 |
|  | (MT5 API calls)  |  | (Position sizing)|                 |
|  +------------------+  +------------------+                 |
|                                                             |
|  +------------------+  +------------------+                 |
|  |  StateManager    |  |   IBStrategy[]   |                 |
|  | (Daily tracking) |  | (Signal + TSL)   |                 |
|  +------------------+  +------------------+                 |
+-------------------------------------------------------------+
           |
           v
   [MetaTrader 5 Terminal]
           |
           v
   [5ers Prop Firm Server -- Asia/Jerusalem]
```

### Design Principles

- **Single-threaded polling**: No async, no threads. A simple `while True` loop with `time.sleep(2)`.
- **One trade per symbol per day**: Each strategy (GER40, XAUUSD) may open at most one position per calendar day.
- **Separation of concerns**: Each component has a single responsibility with clear interfaces.
- **Fail-safe defaults**: If any component fails, the bot skips the trade rather than risking an invalid order.

### Source Files

| File | Role |
|------|------|
| `src/bot_controller.py` | Main orchestrator, control loop |
| `src/mt5_executor.py` | All MetaTrader5 API interactions |
| `src/risk_manager.py` | Position sizing and risk validation |
| `src/state_manager.py` | Daily state tracking, restart recovery |
| `src/strategies/ib_strategy.py` | IB strategy FSM and TSL logic |
| `src/strategies/base_strategy.py` | Abstract strategy interface |
| `config.py` | Environment-based configuration |
| `main.py` | Entry point |

---

## 2. Core Components

### 2.1 BotController (`src/bot_controller.py`)

The central orchestrator. It creates all other components, connects to MT5, initializes strategies, and runs the main polling loop.

**Responsibilities:**
- Connects to MT5 using credentials from `.env`
- Instantiates MT5Executor, RiskManager, StateManager
- Creates IBStrategy instances for GER40 and XAUUSD
- Runs the 2-second polling loop (Architecture Spec Section 5.0)
- Detects new-day transitions and resets daily states
- Matches open positions to strategies by magic number
- Coordinates signal detection, position sizing, and order placement

**Key Methods:**

| Method | Description |
|--------|-------------|
| `__init__()` | Creates executor, risk manager, state manager |
| `_initialize_strategies()` | Creates IBStrategy instances after MT5 connection |
| `_find_position_for_strategy()` | Matches positions by magic number |
| `_check_and_reset_daily_states()` | Detects day boundary, resets states |
| `run()` | Main control loop (see Section 4) |

### 2.2 MT5Executor (`src/mt5_executor.py`)

The **sole interface** to the MetaTrader5 Python API. No other module imports or calls `MetaTrader5` directly.

**Responsibilities:**
- Connection management with retry logic (10 retries, 5-second delay)
- Auto-reconnect on IPC pipe errors
- Bar data retrieval with MT5 server timezone correction
- Order placement with fill mode auto-detection
- Position modification (SL updates for TSL)
- Position closure (time-based exits)
- Symbol info caching (volume_step, volume_min, volume_max)
- Tick data retrieval

**Connection Resilience:**

```python
# Retry logic: 10 attempts with 5-second delay
# Auto-reconnect on IPC pipe errors (MT5 terminal restart)
def connect(self, login, password, server, max_retries=10, retry_delay=5):
    for attempt in range(max_retries):
        if mt5.initialize(login=login, password=password, server=server):
            return True
        time.sleep(retry_delay)
    return False
```

**Virtual TP Implementation:**
When placing orders, MT5 TP is set to `0.0` (no server-side TP). The actual take-profit is managed in-memory by the IBStrategy's TSL logic. This allows dynamic TP adjustment as the trailing stop progresses.

**Timezone Correction for Bar Data:**
The `get_bars()` method handles MT5's server timezone (Asia/Jerusalem) by converting request times to server-local time, then normalizing returned bar timestamps back to UTC.

### 2.3 RiskManager (`src/risk_manager.py`)

Calculates position sizes and validates trades against risk limits.

**Responsibilities:**
- Position size calculation based on percentage risk
- Margin usage validation (max 40%)
- Daily drawdown enforcement (max 3%)
- Pre-trade validation (lot size bounds, margin check)
- Global risk assessment

**Key Methods:**

| Method | Description |
|--------|-------------|
| `calculate_position_size()` | Computes lot size from risk percentage, SL distance, contract size |
| `check_margin_usage()` | Ensures current margin < 40% of equity |
| `check_global_risk()` | Enforces daily drawdown limit |
| `validate_trade()` | Pre-trade check combining lot size and margin validation |

### 2.4 StateManager (`src/state_manager.py`)

The bot's "operational memory." Tracks which strategies have already traded today to enforce the one-trade-per-symbol-per-day rule.

**Responsibilities:**
- Tracking traded-today status per strategy ID
- Daily state reset on new-day detection
- State restoration from MT5 on bot restart (prevents duplicate trades)

**Strategy ID Format:** `{symbol}_{magic_number}` (e.g., `DAX40_1001`)

**State Restoration Logic:**
On startup, the StateManager queries MT5 for:
1. **Open positions** matching strategy magic numbers -- if found, marks that strategy as "traded today"
2. **Today's deal history** -- if a deal with a strategy's magic number exists in today's history, marks it as "traded today"

This prevents the bot from opening duplicate positions after a restart.

### 2.5 IBStrategy (`src/strategies/ib_strategy.py`)

Implements the Initial Balance (IB) breakout strategy with FSM-based state management and trailing stop loss logic. Extends `BaseStrategy` from `base_strategy.py`.

**Responsibilities:**
- FSM state transitions (see Section 3)
- IB range calculation from M2 bars
- Signal detection across 4 variations
- TSL management for open positions
- News filter integration

**Signal Variations (checked in priority order):**

| Priority | Variation | Description |
|----------|-----------|-------------|
| 1 | Reverse | Price reversal after IB breakout |
| 2 | OCAE | Open-Close Above/Below Extension |
| 3 | TCWE | Two Candles With Extension |
| 4 | REV_RB | Reverse with Rebound confirmation |

Each variation has its own TSL parameters (tsl_target, tsl_sl) controlling how aggressively the trailing stop advances.

---

## 3. Finite State Machine (FSM)

The IBStrategy operates as a daily FSM that resets each trading day.

### State Diagram

```
                    [Bot Start / New Day]
                           |
                           v
              +----------------------------+
              | AWAITING_IB_CALCULATION    |
              | (Waiting for IB window)    |
              +----------------------------+
                           |
                    [IB window ends, IB range calculated]
                           |
                           v
              +----------------------------+
              | AWAITING_TRADE_WINDOW      |
              | (Waiting for trade window) |
              +----------------------------+
                           |
                    [Trade window opens]
                           |
                           v
              +----------------------------+
              | IN_TRADE_WINDOW            |<-----+
              | (Checking for signals)     |      |
              +----------------------------+      |
                      |           |               |
          [Signal found]    [Window ends]    [No signal yet]
                      |           |               |
                      v           v               |
              +------------+  +-----------+       |
              |POSITION_   |  | DAY_ENDED |       |
              |OPEN        |  +-----------+       |
              |(Managing   |                      |
              | TSL)       |----------------------+
              +------------+      [Position closed]
                      |
              [Position closed / Day ends]
                      |
                      v
              +----------------------------+
              | DAY_ENDED                  |
              | (No more activity today)   |
              +----------------------------+
```

### State Descriptions

| State | Behavior |
|-------|----------|
| `AWAITING_IB_CALCULATION` | Waits for the IB time window to pass. During this window, M2 bars are collected to compute the IB High and IB Low. |
| `AWAITING_TRADE_WINDOW` | IB range is known. Waits for the configurable delay period (e.g., 20 minutes) before signal checking begins. |
| `IN_TRADE_WINDOW` | Actively checks each new M2 bar for signal conditions across all 4 variations. Transitions to POSITION_OPEN on trade entry or DAY_ENDED when the trade window expires. |
| `POSITION_OPEN` | A position is live. The TSL logic runs every poll cycle, adjusting SL and virtual TP. Transitions to DAY_ENDED when the position is closed (by SL, virtual TP, or time exit). |
| `DAY_ENDED` | Terminal state for the day. No further action until the next trading day. |

### IB Window Configuration

The IB window is defined per symbol in the strategy parameters:

- **GER40**: 08:00 -- 08:30 Europe/Berlin (Frankfurt open)
- **XAUUSD**: Configured per the optimization results (typically Asian session)

### Signal Detection Guard

The `_is_last_candle_closed()` method ensures signals are only evaluated on **completed** M2 candles, preventing intra-bar false signals. It checks that the current time is at least 2 minutes past the last bar's open time.

---

## 4. Main Loop Data Flow

The main loop in `BotController.run()` implements Architecture Specification Section 5.0. It executes every 2 seconds.

### Pseudocode

```
CONNECT to MT5
INITIALIZE strategies (GER40, XAUUSD)
RESTORE state from MT5 (check existing positions/history)

WHILE True:
    1. now_utc = datetime.now(UTC)

    2. IF new_day(now_utc):
         state_manager.reset_all_daily_states()

    3. open_positions = executor.get_open_positions()

    4. FOR EACH strategy IN strategies:
         strategy_id = "{symbol}_{magic_number}"
         pos = find_position_by_magic_number(open_positions, strategy.magic_number)

         5. IF pos EXISTS:
              tick = executor.get_tick(strategy.symbol)
              strategy.update_position_state(pos, tick)   # TSL logic

         6. ELIF NOT state_manager.has_traded_today(strategy_id):
              signal = strategy.check_signal(now_utc)      # FSM logic

              IF signal IS NOT None:
                  account = mt5.account_info()
                  lots = risk_manager.calculate_position_size(
                      symbol, entry_price, stop_loss, balance
                  )

                  IF lots > 0:
                      validation = risk_manager.validate_trade(symbol, lots)

                      IF validation.valid:
                          result = executor.place_order(symbol, signal, lots, magic)

                          IF result.success:
                              state_manager.set_trade_taken(strategy_id)

    7. time.sleep(2)
```

### Flow Explanation

| Step | Component | Action |
|------|-----------|--------|
| 1 | BotController | Get current UTC time |
| 2 | StateManager | Reset daily flags on day boundary |
| 3 | MT5Executor | Query all open positions |
| 4 | BotController | Iterate over strategies |
| 5 | IBStrategy | If position exists, run TSL updates |
| 6 | IBStrategy | If no position and not traded today, check for signal |
| 6a | RiskManager | Calculate position size from balance and SL distance |
| 6b | RiskManager | Validate trade (margin, lot bounds) |
| 6c | MT5Executor | Place order on MT5 |
| 6d | StateManager | Mark strategy as "traded today" |
| 7 | BotController | Sleep 2 seconds |

### Polling Interval Rationale

The 2-second interval was chosen as a balance between:
- **Signal detection speed**: Maximum delay of 2--4 seconds after M2 candle close
- **MT5 API load**: Low enough to avoid rate limiting or terminal instability
- **Previous value**: Was 5 seconds, which caused unacceptable signal detection delays

---

## 5. Position Sizing Formula

The RiskManager calculates lot sizes using the following formula:

### Formula

```
risk_amount = account_balance * (RISK_PER_TRADE_PCT / 100)

sl_distance_price = abs(entry_price - stop_loss)

raw_lots = risk_amount / (sl_distance_price * trade_contract_size)

# Round down to nearest volume_step
lots = floor(raw_lots / volume_step) * volume_step

# Clamp to broker limits
lots = max(volume_min, min(lots, volume_max))
```

### Example -- XAUUSD

```
Account balance:   $50,000
Risk per trade:    0.9%
Entry price:       $2,650.00
Stop loss:         $2,640.00
Contract size:     100 oz/lot
Volume step:       0.01

risk_amount = 50000 * 0.009 = $450
sl_distance = |2650.00 - 2640.00| = 10.00
raw_lots = 450 / (10.00 * 100) = 0.45
lots = floor(0.45 / 0.01) * 0.01 = 0.45 lots
```

### Example -- GER40 (DAX40)

```
Account balance:   $50,000
Risk per trade:    0.9%
Entry price:       20,500.0
Stop loss:         20,480.0
Contract size:     1.0
Volume step:       0.1

risk_amount = 50000 * 0.009 = $450
sl_distance = |20500.0 - 20480.0| = 20.0
raw_lots = 450 / (20.0 * 1.0) = 22.5
lots = floor(22.5 / 0.1) * 0.1 = 22.5 lots
```

### Safety Checks

After computing the lot size, the RiskManager runs:
1. **Volume bounds**: `volume_min <= lots <= volume_max`
2. **Margin check**: Ensure the required margin does not exceed `MAX_MARGIN_USAGE_PCT` (40%) of account equity
3. **Zero check**: If lots rounds to 0, the trade is skipped entirely

---

## 6. Trailing Stop Loss (TSL) Logic

The TSL system manages open positions by progressively moving the stop loss in the direction of profit while tracking a virtual take-profit target.

### Key Concept: Virtual TP

The MT5 order has **TP set to 0.0** (no server-side TP). The actual TP is tracked in-memory by the IBStrategy. This allows the TSL to dynamically extend the TP as the position moves into profit.

### TSL State Structure

Each open position maintains the following TSL state:

```python
tsl_state = {
    "variation": str,           # Which signal variation opened this trade
    "tsl_target": float,        # Virtual TP multiplier (in R-units)
    "tsl_sl": float,            # SL step size multiplier (in R-units)
    "initial_sl": float,        # Original stop loss price
    "initial_tp": float,        # Original take profit price
    "current_tp": float,        # Current virtual TP (moves with TSL)
    "entry_price": float,       # Entry price
    "tsl_triggered": bool,      # Whether TSL has been activated
    "position_window_end": dt,  # Time-based exit deadline
    "tsl_history": list,        # Log of SL/TP adjustments
}
```

### TSL Algorithm

```
risk = abs(entry_price - initial_sl)

ON EACH TICK:
    IF price reaches current_tp:
        # TSL activates: move SL forward, extend TP
        new_sl = current_sl + (tsl_sl * risk)
        new_tp = current_tp + (tsl_target * risk)
        executor.modify_position(ticket, sl=new_sl)
        current_tp = new_tp
        tsl_triggered = True

    IF current_time >= position_window_end:
        # Time-based exit
        executor.close_position(ticket)
```

### Per-Variation TSL Parameters

Each signal variation defines its own TSL behavior:

| Variation | tsl_target | tsl_sl | Behavior |
|-----------|-----------|--------|----------|
| Reverse | Configurable | Configurable | Standard trailing |
| OCAE | Configurable | Configurable | Standard trailing |
| TCWE | Configurable | Configurable | Standard trailing |
| REV_RB | Configurable | Configurable | Standard trailing |

The actual values are loaded from the strategy parameters (optimized via backtesting).

### TSL Limitation on Restart

**Important**: TSL state is stored **only in-memory**. If the bot restarts while a position is open, the TSL state **cannot be reliably restored** from MT5. The strategy sets a dummy TSL state after restart, which means trailing behavior may be degraded until the position is closed. The `_restore_tsl_state_from_position()` method documents this limitation explicitly.

---

## 7. State Restoration on Restart

When the bot restarts (planned or crash recovery), the StateManager queries MT5 to reconstruct the current state and prevent duplicate trades.

### Restoration Process

```
FOR EACH strategy IN strategies:
    strategy_id = "{symbol}_{magic_number}"

    1. Check open positions:
       positions = mt5.positions_get(symbol=strategy.symbol)
       IF any position has magic == strategy.magic_number:
           mark strategy_id as traded_today
           log: "Found open position for {strategy_id}"

    2. Check today's deal history:
       deals = mt5.history_deals_get(today_start, now)
       IF any deal has magic == strategy.magic_number:
           mark strategy_id as traded_today
           log: "Found deal history for {strategy_id} today"
```

### What This Prevents

- **Duplicate entries**: If GER40 already has an open position or traded earlier today, the bot will not attempt to open another.
- **Ghost positions**: The bot correctly identifies positions it owns (by magic number) vs. positions from other sources.

### What This Does NOT Restore

- **TSL state**: The trailing stop history and current virtual TP are lost. The strategy sets a simplified fallback TSL state.
- **FSM state**: The strategy starts fresh and recalculates its FSM state from market data.

---

## 8. DRY_RUN Mode

When `DRY_RUN=True` is set in the `.env` file, the bot operates normally but **does not execute any trades** on MT5.

### Behavior in DRY_RUN Mode

| Operation | DRY_RUN=False | DRY_RUN=True |
|-----------|--------------|-------------|
| MT5 connection | Real connection | Real connection |
| Bar data retrieval | Real data | Real data |
| Signal detection | Normal | Normal |
| Position sizing | Normal calculation | Normal calculation |
| Order placement | Sent to MT5 | **Logged only, not sent** |
| Position modification | Sent to MT5 | **Logged only** |
| Position closure | Sent to MT5 | **Logged only** |

### How to Enable

In `dual_v3/.env`:
```
DRY_RUN=True
```

### Use Cases

- **Strategy validation**: Run against live market data without risking capital
- **Infrastructure testing**: Verify MT5 connectivity, logging, and state management
- **Pre-deployment verification**: Confirm the bot behaves correctly before switching to live

---

## 9. News Filter

The news filter prevents trading during high-impact economic news events, as required by the 5ers prop firm rules.

### Behavior

- **Blackout window**: No new trades are opened within 15 minutes before or after a high-impact news event
- **Existing positions**: Not affected by the news filter (positions remain open and TSL continues)
- **Scope**: Applied per-symbol based on the currency exposure

### Configuration

The news filter is enabled per-strategy during initialization:

```python
IBStrategy(
    symbol,
    params,
    executor,
    magic_number,
    strategy_label="GER40_007",
    news_filter_enabled=True,  # 5ers compliance
)
```

### Data Source

News events are typically sourced from ForexFactory or equivalent calendars. The filter checks the event schedule before allowing a signal to proceed to order placement.

---

## 10. Timezone Handling

The bot operates across multiple timezones, which is a significant source of complexity.

### Timezone Map

| Entity | Timezone | UTC Offset (approx.) |
|--------|----------|---------------------|
| Bot / User | Asia/Almaty | UTC+5 (no DST) |
| MT5 Server (5ers) | Asia/Jerusalem | UTC+2 / UTC+3 (DST) |
| GER40 Market | Europe/Berlin | UTC+1 / UTC+2 (DST) |
| XAUUSD Reference | Asia/Tokyo | UTC+9 (no DST) |

### How Timezones Are Handled

1. **Internal clock**: All internal timestamps use `datetime.now(pytz.utc)` -- UTC everywhere.

2. **MT5 bar data**: The MT5 server returns bar timestamps in server time (Asia/Jerusalem). The `MT5Executor.get_bars()` method:
   - Converts the request time to server-local time before querying
   - Normalizes returned timestamps back to UTC

3. **IB window calculation**: The IB window (e.g., 08:00--08:30 Europe/Berlin for GER40) is defined in local market time. The IBStrategy converts this to UTC for comparison against `now_utc`.

4. **Daily reset**: Day boundaries are determined in UTC. The `_check_and_reset_daily_states()` method compares `now_utc.date()` against the last known date.

### Common Pitfalls

- **DST transitions**: Asia/Jerusalem and Europe/Berlin observe DST at different dates. The IB window's UTC equivalent shifts accordingly.
- **MT5 weekend gaps**: MT5 returns no data during market closure. The bot must handle empty bar arrays gracefully.
- **Cross-midnight instruments**: XAUUSD trades nearly 24 hours. The "new day" boundary must be consistent with the strategy's definition.

---

## 11. Configuration Reference

### Environment Variables (`.env`)

| Variable | Default | Description |
|----------|---------|-------------|
| `MT5_LOGIN` | (required) | MT5 account number |
| `MT5_PASSWORD` | (required) | MT5 account password |
| `MT5_SERVER` | (required) | MT5 server name |
| `DRY_RUN` | `False` | Enable dry-run mode (no real trades) |
| `RISK_PER_TRADE_PCT` | `0.9` | Risk per trade as % of balance |
| `MAX_DAILY_LOSS_PCT` | `3.0` | Maximum daily loss as % of balance |
| `MAX_MARGIN_USAGE_PCT` | `40.0` | Maximum margin usage as % of equity |
| `LOG_LEVEL` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `LOG_FILE` | `bot.log` | Log file path |

### `config.py` Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `SYMBOL_MAPPING["GER40"]` | `"DAX40"` | MT5 symbol name for GER40 |
| `SYMBOL_MAPPING["XAUUSD"]` | `"XAUUSD"` | MT5 symbol name for XAUUSD |
| `MAGIC_NUMBER_GER40` | `1001` | Magic number for GER40 orders |
| `MAGIC_NUMBER_XAUUSD` | `1002` | Magic number for XAUUSD orders |

### Strategy Parameters

Strategy parameters (`GER40_PARAMS_PROD`, `XAUUSD_PARAMS_PROD`) are imported from `src/utils/strategy_logic.py` and include:

- IB window start/end times and timezone
- Wait period after IB (minutes)
- Trade window duration
- Per-variation enable/disable flags
- Per-variation TSL parameters (tsl_target, tsl_sl)
- Signal-specific thresholds (extension percentages, rebound criteria)

---

## 12. Magic Numbers and Symbol Mapping

### Magic Numbers

Magic numbers are unique integer identifiers embedded in every MT5 order. They allow the bot to:
- Identify which positions belong to which strategy
- Avoid interfering with manually placed trades
- Support state restoration after restart

| Strategy | Magic Number | MT5 Symbol |
|----------|-------------|------------|
| GER40 (DAX) | 1001 | DAX40 |
| XAUUSD (Gold) | 1002 | XAUUSD |

### Symbol Mapping

The bot uses internal names (GER40, XAUUSD) that differ from the MT5 broker names. The `SYMBOL_MAPPING` dictionary in `config.py` translates between them:

```python
SYMBOL_MAPPING = {
    "GER40": "DAX40",     # 5ers uses DAX40 for German index
    "XAUUSD": "XAUUSD"    # Gold vs USD (standard name)
}
```

The BotController uses this mapping when creating strategy instances. The strategies themselves operate using the MT5 symbol names.

---

## 13. Safety Features and Prop Firm Compliance

### Risk Limits

| Parameter | Value | Enforcement |
|-----------|-------|-------------|
| Risk per trade | 0.9% of balance | RiskManager.calculate_position_size() |
| Max daily loss | 3.0% of balance | RiskManager.check_global_risk() |
| Max margin usage | 40% of equity | RiskManager.check_margin_usage() |
| One trade per symbol per day | Enforced | StateManager.has_traded_today() |

### 5ers Prop Firm Rules Enforced

1. **Visible stop loss**: Every order is placed with an explicit SL. The MT5Executor sets SL on the order request.
2. **News filter**: No trades within 15 minutes of high-impact news events (configurable per strategy).
3. **Drawdown protection**: Daily loss limit of 3% prevents catastrophic drawdown.
4. **Margin discipline**: 40% margin cap leaves sufficient buffer for adverse price moves.

### Pre-Trade Validation Chain

Before any order is placed, it must pass through:

```
1. Signal detection (IBStrategy.check_signal)
   |-- News filter check
   |-- IB range validation
   |-- Signal variation conditions

2. Position sizing (RiskManager.calculate_position_size)
   |-- Non-zero lot size

3. Trade validation (RiskManager.validate_trade)
   |-- Lot size within broker limits
   |-- Margin usage below threshold

4. Order placement (MT5Executor.place_order)
   |-- Fill mode auto-detection
   |-- SL set on order
   |-- TP set to 0.0 (virtual)
```

If any step fails, the trade is skipped and the failure is logged.

### Error Handling

- **MT5 connection loss**: Auto-reconnect with 10 retries and 5-second delays. If reconnection fails, the bot exits cleanly.
- **Order rejection**: Logged with MT5 error details. The trade opportunity is marked as used (no retry on the same day).
- **Fatal exceptions**: Caught in the main loop's try/except. The bot disconnects from MT5 and exits.
- **Keyboard interrupt (Ctrl+C)**: Clean shutdown with MT5 disconnection.

---

## 14. Startup and Shutdown Procedure

### Startup Sequence

```
1. main.py
   |-- Configure logging (file + console)
   |-- Create BotController instance

2. BotController.__init__()
   |-- Create MT5Executor (dry_run from config)
   |-- Create RiskManager (risk_pct, margin_pct)
   |-- Create StateManager

3. BotController.run()
   |-- Connect to MT5 (login, password, server)
   |-- Initialize strategies (GER40, XAUUSD)
   |-- Restore state from MT5 (check open positions + history)
   |-- Log account balance and equity
   |-- Enter main loop (2-second polling)
```

### Running the Bot

```bash
cd C:\Trading\ib_trading_bot\dual_v3
python main.py
```

### Shutdown Sequence

The bot shuts down in three scenarios:

1. **Ctrl+C (KeyboardInterrupt)**: Clean exit
2. **Fatal exception**: Error logged, then clean exit
3. **MT5 connection failure**: After exhausting retries

In all cases, the `finally` block ensures:

```
1. Log "Shutting down bot controller..."
2. executor.disconnect()  -- Calls mt5.shutdown()
3. Log "BOT CONTROLLER STOPPED"
```

### Important Notes on Shutdown

- **Open positions are NOT closed on shutdown.** Positions remain on the MT5 server with their stop losses intact.
- **TSL state is lost.** If the bot restarts, TSL history is not preserved (see Section 7).
- **State restoration handles restart.** The bot will detect existing positions and avoid duplicates on the next startup.

### Demo Mode

For testing with a demo account (Forex pairs), use:

```bash
cd C:\Trading\ib_trading_bot\dual_v3
python run_demo.py
```

This runs with different symbols (EURUSD, GBPUSD, USDJPY, USDCHF) and a 5-second polling interval. It is intended for infrastructure testing only.
