# Slow Backtest Engine -- Architecture and Usage Guide

> **Last updated**: 2026-02-08
> **Applies to**: dual_v3 backtest system
> **Entry point**: `dual_v3/backtest/run_backtest_v8_ger40.py` (example), `dual_v3/backtest/run_parallel_backtest.py` (parallel)

---

## Table of Contents

1. [Purpose and Design Philosophy](#1-purpose-and-design-philosophy)
2. [Adapter Pattern Architecture](#2-adapter-pattern-architecture)
3. [MT5 Emulator](#3-mt5-emulator)
4. [Data Pipeline](#4-data-pipeline)
5. [Running Backtests](#5-running-backtests)
6. [Output Format and Reporting](#6-output-format-and-reporting)
7. [Results Directory Structure](#7-results-directory-structure)
8. [Performance Characteristics](#8-performance-characteristics)

---

## 1. Purpose and Design Philosophy

The slow backtest engine is designed to run the **actual IBStrategy code** against historical data with no modifications to the strategy logic. It achieves this by emulating the entire MetaTrader5 Python API, allowing the same `IBStrategy`, `RiskManager`, and supporting code to run identically in both live and backtest environments.

### Why "Slow"?

The engine processes data **tick-by-tick** or **candle-by-candle**, faithfully reproducing the same execution path the live bot would follow. This fidelity comes at the cost of speed -- a single 2-year backtest for one parameter set may take several minutes. For parameter grid search (thousands of combinations), the [Fast Backtest Engine](FAST_BACKTEST_ENGINE.md) is preferred.

### When to Use the Slow Engine

| Use Case | Slow Engine | Fast Engine |
|----------|-------------|-------------|
| Final validation of optimized parameters | Yes | No |
| Detailed trade-by-trade analysis | Yes | No |
| Debugging strategy logic | Yes | No |
| Parameter grid search (1000+ combos) | No | Yes |
| Quick screening of parameter ranges | No | Yes |
| Generating production-quality reports | Yes | No |

### Key Properties

- **Code identity**: Runs the exact same `IBStrategy` class as the live bot
- **Tick-level simulation**: Synthetic 5-second ticks generated from M1 OHLC data
- **Full MT5 API emulation**: Positions, margin, account balance, order fills, deal history
- **Deterministic**: Fixed random seed (default 42) ensures reproducible results
- **Parallel execution**: Multiprocessing support for running multiple parameter groups simultaneously

### Source Files

| File | Role |
|------|------|
| `backtest/emulator/mt5_emulator.py` | MT5 API emulator (singleton) |
| `backtest/adapter.py` | Patches MT5 module, provides BacktestExecutor |
| `backtest/backtest_runner.py` | Main orchestrator (data loading, simulation, results) |
| `backtest/config.py` | Backtest-specific configuration |
| `backtest/data_processor/data_ingestor.py` | CSV data loading and cleaning |
| `backtest/data_processor/tick_generator.py` | M1 -> 5-second tick synthesis |
| `backtest/risk_manager.py` | Backtest-specific risk manager |
| `backtest/analysis/metrics.py` | Performance analytics (Sharpe, drawdown, etc.) |
| `backtest/reporting/report_manager.py` | Output folder and report orchestration |
| `backtest/reporting/excel_report.py` | Excel report generation |
| `backtest/reporting/trade_charts.py` | Individual trade chart generation |
| `backtest/run_backtest_v8_ger40.py` | Example run script (GER40 V8 parameters) |
| `backtest/run_parallel_backtest.py` | Parallel execution runner |

---

## 2. Adapter Pattern Architecture

The core architectural insight is the **adapter pattern**: the backtest engine replaces the real MetaTrader5 module with an emulated version at the Python import level, so the strategy code runs without any awareness that it is in a backtest.

### Architecture Diagram

```
+------------------------------------------------------------------+
|                    Backtest Environment                           |
|                                                                  |
|  +---------------------------+                                   |
|  |   BacktestRunner          |                                   |
|  |   (Orchestrator)          |                                   |
|  +---------------------------+                                   |
|         |                                                        |
|         | Creates & configures                                   |
|         v                                                        |
|  +---------------------------+    +---------------------------+  |
|  |   BacktestExecutor        |    |   MT5Emulator             |  |
|  |   (adapter.py)            |--->|   (mt5_emulator.py)       |  |
|  |                           |    |                           |  |
|  |   Drop-in replacement     |    |   Full MT5 API emulation  |  |
|  |   for MT5Executor         |    |   - initialize/login      |  |
|  +---------------------------+    |   - account_info          |  |
|         |                         |   - symbol_info           |  |
|         | sys.modules patch       |   - copy_rates_from_pos   |  |
|         v                         |   - positions_get         |  |
|  +---------------------------+    |   - order_send            |  |
|  | sys.modules["MetaTrader5"]|    |   - history_deals_get     |  |
|  | = create_mt5_patch_module |    +---------------------------+  |
|  +---------------------------+              ^                    |
|         |                                   |                    |
|         v                                   |                    |
|  +---------------------------+              |                    |
|  |   IBStrategy              |   Calls MT5  |                    |
|  |   (Unmodified live code)  |---functions-->|                    |
|  +---------------------------+                                   |
+------------------------------------------------------------------+
```

### How the Patch Works

The `adapter.py` module provides `create_mt5_patch_module()`, which creates a module-like object that replaces the `MetaTrader5` entry in `sys.modules`:

```python
# In adapter.py
def create_mt5_patch_module(emulator):
    """Create a fake MetaTrader5 module backed by the emulator."""
    module = types.ModuleType("MetaTrader5")
    module.initialize = emulator.initialize
    module.login = emulator.login
    module.account_info = emulator.account_info
    module.symbol_info = emulator.symbol_info
    module.symbol_info_tick = emulator.symbol_info_tick
    module.copy_rates_from_pos = emulator.copy_rates_from_pos
    module.positions_get = emulator.positions_get
    module.order_send = emulator.order_send
    module.history_deals_get = emulator.history_deals_get
    # ... all other MT5 API functions
    return module

# Usage in BacktestRunner:
import sys
patch_module = create_mt5_patch_module(emulator)
sys.modules["MetaTrader5"] = patch_module
```

After this patch, any code that does `import MetaTrader5 as mt5` will receive the emulated module instead of the real one. The IBStrategy, RiskManager, and all other modules operate identically to the live environment.

### BacktestExecutor

The `BacktestExecutor` class mirrors the `MT5Executor` interface but routes all calls through the emulator:

| MT5Executor Method | BacktestExecutor Equivalent |
|-------------------|----------------------------|
| `connect()` | No-op (emulator always "connected") |
| `get_bars()` | Routes to emulator's `copy_rates_from_pos` |
| `get_tick()` | Routes to emulator's `symbol_info_tick` |
| `place_order()` | Routes to emulator's `order_send` |
| `modify_position()` | Routes to emulator's position SL modification |
| `close_position()` | Routes to emulator's position closure |
| `get_open_positions()` | Routes to emulator's `positions_get` |
| `disconnect()` | No-op |

---

## 3. MT5 Emulator

The `MT5Emulator` (`backtest/emulator/mt5_emulator.py`) is the heart of the backtest engine. It is a **thread-safe singleton** that fully emulates the MetaTrader5 Python API.

### Singleton Pattern

```python
class MT5Emulator:
    _instance = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance
```

The singleton ensures that all modules in the backtest share the same emulated MT5 state.

### Emulated State

The emulator maintains:

| State | Description |
|-------|-------------|
| `account_balance` | Current cash balance (updates on trade close) |
| `account_equity` | Balance + unrealized P&L |
| `positions` | Dict of open positions keyed by ticket number |
| `deals_history` | List of completed deals (for history queries) |
| `current_time` | Simulated clock (advanced by the tick feeder) |
| `tick_data` | Current bid/ask prices per symbol |
| `bar_data` | Historical OHLC data per symbol per timeframe |
| `symbol_configs` | Symbol specifications (spread, digits, contract size, etc.) |

### Position Lifecycle in the Emulator

```
1. order_send(BUY/SELL)
   |-- Validate lot size, margin
   |-- Create position record (ticket, entry_price, sl, tp, volume)
   |-- Deduct margin from available funds
   |-- Record deal in history (DEAL_ENTRY_IN)

2. On each tick update:
   |-- Recalculate unrealized P&L for all positions
   |-- Check SL/TP hits:
   |     IF bid <= sl (for BUY) or ask >= sl (for SELL):
   |         Close at exact SL price (not tick price)
   |     IF tp > 0 and bid >= tp (for BUY) or ask <= tp (for SELL):
   |         Close at exact TP price

3. Position close (SL/TP/manual):
   |-- Calculate final P&L
   |-- Update account_balance
   |-- Release margin
   |-- Remove from positions dict
   |-- Record deal in history (DEAL_ENTRY_OUT)
```

### SL/TP Execution Precision

The emulator closes positions at the **exact SL or TP price**, not at the current tick price. This is important for accurate backtest results:

```python
def _close_position_at_price(self, ticket, close_price, reason):
    """Close position at exact price (SL or TP hit)."""
    pos = self.positions[ticket]
    if pos.type == ORDER_TYPE_BUY:
        profit = (close_price - pos.price_open) * pos.volume * contract_size
    else:
        profit = (pos.price_open - close_price) * pos.volume * contract_size
    # ...
```

### Timeframe Resampling

The emulator supports bar data at multiple timeframes. When the strategy requests M2 bars but only M1 data is loaded, the emulator resamples:

```python
def _resample_to_timeframe(self, m1_data, timeframe):
    """Convert M1 bars to M2, M5, M15, etc."""
    # Groups M1 bars by target timeframe periods
    # Aggregates: open=first, high=max, low=min, close=last, volume=sum
```

---

## 4. Data Pipeline

The data pipeline transforms raw CSV files of M1 OHLC data into the tick-level data consumed by the emulator.

### Pipeline Overview

```
[CSV Files]                     [DataIngestor]              [TickGenerator]
  GER40_2023_01.csv    --->   Aggregate, clean,   --->   Generate synthetic
  GER40_2023_02.csv           validate, normalize         5-second ticks
  ...                         to UTC                      from M1 candles
  GER40_2025_10.csv
                                    |                          |
                                    v                          v
                              [Clean M1 DataFrame]      [Tick DataFrame]
                                                              |
                                                              v
                                                    [MT5Emulator.load_data()]
```

### 4.1 DataIngestor (`backtest/data_processor/data_ingestor.py`)

Responsible for loading and cleaning raw CSV data.

**Processing Steps:**

1. **CSV Aggregation**: Reads all CSV files from a data folder and concatenates them into a single DataFrame
2. **Datetime Parsing**: Handles multiple datetime formats automatically
3. **Duplicate Removal**: Drops duplicate timestamps, keeping the last occurrence
4. **OHLC Validation**: Ensures High >= Open, Close, Low and Low <= Open, Close, High for every bar
5. **UTC Normalization**: Converts all timestamps to UTC
6. **Sorting**: Sorts by timestamp ascending
7. **Gap Detection**: Identifies and logs gaps in the time series (weekends, holidays)

**Input Format (CSV):**

```csv
time,open,high,low,close,tick_volume,spread,real_volume
2023-01-02 00:00:00,14071.3,14074.0,14069.3,14072.1,145,15,0
```

### 4.2 TickGenerator (`backtest/data_processor/tick_generator.py`)

Transforms M1 OHLC bars into synthetic 5-second ticks (12 ticks per minute).

**Tick Generation Algorithm:**

For each M1 bar, the generator creates a plausible intra-bar price path:

```
1. Determine path direction:
   IF close > open:
       path = Open -> Low -> High -> Close  (bullish: dip first, then rally)
   ELSE:
       path = Open -> High -> Low -> Close  (bearish: rally first, then dip)

2. Create 12 interpolation points along this path:
   ticks[0] = open
   ticks[1..3] = interpolate toward first extreme
   ticks[4..6] = at first extreme, moving toward second
   ticks[7..9] = interpolate toward second extreme
   ticks[10..11] = interpolate toward close
   ticks[11] = close (exact)

3. Add jitter:
   FOR each tick (except open and close):
       tick += random_uniform(-jitter, +jitter) * (high - low) * jitter_factor
       tick = clamp(tick, low, high)

4. Generate bid/ask:
   bid = tick_price
   ask = tick_price + spread
```

**Configuration:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ticks_per_minute` | 12 | Number of synthetic ticks per M1 bar |
| `jitter_factor` | 0.1 | Noise amplitude as fraction of bar range |
| `random_seed` | 42 | For reproducibility (None for random) |

**Caching:**

Generated tick data is cached as **Parquet files** (snappy compression) to avoid regeneration on subsequent runs. The cache key includes the symbol, date range, and generation parameters.

---

## 5. Running Backtests

### 5.1 Single Backtest (Run Script)

The simplest way to run a backtest is via a dedicated run script. The file `run_backtest_v8_ger40.py` provides a reference example.

**Example run script structure:**

```python
from backtest.backtest_runner import BacktestRunner

# Define strategy parameters (from optimization)
PARAMS = {
    "ib_start_time": "08:00",
    "ib_end_time": "08:30",
    "ib_timezone": "Europe/Berlin",
    "wait_minutes": 20,
    "trade_window_minutes": 360,
    # ... per-variation parameters
}

# Create runner
runner = BacktestRunner(
    symbol="GER40",
    params=PARAMS,
    start_date="2023-01-01",
    end_date="2025-10-31",
    risk_amount=1000.0,       # Fixed $1000 risk per trade
    output_dir="backtest/output",
)

# Run
results = runner.run_with_bot_integration()
```

**Running from command line:**

```bash
cd C:\Trading\ib_trading_bot\dual_v3
python -m backtest.run_backtest_v8_ger40
```

### 5.2 BacktestRunner Modes

The `BacktestRunner` supports two execution modes:

| Mode | Method | Description |
|------|--------|-------------|
| Candle mode | `run()` | Processes M2 bars directly, faster but less precise |
| Tick mode | `run_with_bot_integration()` | Feeds 5-second ticks through actual IBStrategy, highest fidelity |

**Tick mode** is the recommended mode for final validation, as it exercises the exact same code path as the live bot.

### 5.3 BacktestRunner Orchestration (Tick Mode)

```
1. PREPARE DATA
   |-- DataIngestor loads and cleans CSV files
   |-- TickGenerator creates 5-second ticks (or loads from cache)

2. CONFIGURE EMULATOR
   |-- Reset MT5Emulator state
   |-- Load tick data into emulator
   |-- Set initial balance, leverage, symbol configs
   |-- Patch sys.modules["MetaTrader5"]

3. CREATE STRATEGY
   |-- Instantiate IBStrategy with test parameters
   |-- Create BacktestExecutor (adapter)

4. SIMULATION LOOP
   FOR each tick in tick_data:
       |-- Advance emulator clock
       |-- Update bid/ask prices
       |-- Check SL/TP for open positions
       |-- Call strategy.check_signal() or strategy.update_position_state()

5. COLLECT RESULTS
   |-- Extract trade history from emulator
   |-- Calculate performance metrics
   |-- Generate reports
```

### 5.4 Parallel Backtests

For running multiple parameter groups simultaneously, use `run_parallel_backtest.py`.

**Command line usage:**

```bash
# Run all GER40 groups with 20 workers
python -m backtest.run_parallel_backtest --symbol GER40 --workers 20

# Run from a specific groups file
python -m backtest.run_parallel_backtest --groups-file groups_vm11.json --workers 20

# Custom date range and risk
python -m backtest.run_parallel_backtest \
    --symbol XAUUSD \
    --workers 20 \
    --start-date 2023-01-01 \
    --end-date 2025-10-31 \
    --risk 1000.0 \
    --skip-charts
```

**CLI Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--symbol` | (required*) | Symbol to backtest (GER40 or XAUUSD) |
| `--groups-file` | (required*) | Path to JSON file with group definitions |
| `--workers` | 20 | Number of parallel worker processes |
| `--start-date` | 2023-01-01 | Backtest start date |
| `--end-date` | 2025-10-31 | Backtest end date |
| `--risk` | 1000.0 | Fixed risk amount per trade (USD) |
| `--skip-charts` | False | Skip individual trade chart generation |
| `--output-dir` | Auto-generated | Output directory path |

*One of `--symbol` or `--groups-file` is required.

**Parallel Architecture:**

```
[Main Process]
     |
     |-- Load groups from JSON
     |-- Create multiprocessing.Pool(num_workers)
     |
     +--- [Worker 1] -- run_single_group_backtest(group_1)
     +--- [Worker 2] -- run_single_group_backtest(group_2)
     +--- [Worker 3] -- run_single_group_backtest(group_3)
     +--- ...
     +--- [Worker N] -- run_single_group_backtest(group_N)
     |
     |-- Collect results via imap_unordered
     |-- Save summary CSV and JSON
```

Each worker process creates its own MT5Emulator instance (the singleton is per-process due to multiprocessing fork). Worker logging is suppressed to WARNING level to avoid interleaved output.

**Progress Reporting:**

During parallel execution, the main process logs progress with ETA:

```
[15/200] GER40_group_015: R=45.20, Trades=187, ETA: 23.5min
[16/200] GER40_group_016: R=38.70, Trades=165, ETA: 22.8min
```

### 5.5 Backtest Risk Manager

The backtest system uses its own `BacktestRiskManager` (`backtest/risk_manager.py`) with two modes:

| Mode | Description |
|------|-------------|
| **Fixed amount** | Risk a fixed dollar amount per trade (e.g., $1000). Used for standardized comparison across parameter groups. |
| **Percentage** | Risk a percentage of current equity per trade. Mirrors live bot behavior. |

**Margin constraint:**

```python
max_lots = (equity * max_margin_pct / 100 * leverage) / (price * contract_size)
lots = min(calculated_lots, max_lots)
```

---

## 6. Output Format and Reporting

The backtest engine produces comprehensive reports for each run.

### 6.1 Performance Metrics (`backtest/analysis/metrics.py`)

The `PerformanceMetrics` class computes:

| Metric | Description |
|--------|-------------|
| **Total R** | Sum of all trade results in R-multiples (risk units) |
| **Win Rate** | Percentage of winning trades |
| **Profit Factor** | Gross profit / Gross loss |
| **Sharpe Ratio** | Risk-adjusted return (annualized) |
| **Sortino Ratio** | Downside-risk-adjusted return |
| **Calmar Ratio** | Annual return / Max drawdown |
| **Max Drawdown** | Largest peak-to-trough equity decline |
| **Max Drawdown Duration** | Longest period between equity peaks |
| **Monthly Returns** | Return breakdown by calendar month |
| **By-Variation Breakdown** | Metrics computed per signal variation (Reverse, OCAE, TCWE, REV_RB) |

### 6.2 Excel Report (`backtest/reporting/excel_report.py`)

The `ExcelReportGenerator` creates a multi-sheet Excel workbook:

| Sheet | Contents |
|-------|----------|
| **Trades** | Complete trade log with entry/exit times, prices, P&L, R-multiple, variation |
| **Statistics** | Summary metrics (total R, Sharpe, max drawdown, win rate, etc.) |
| **By Variation** | Performance breakdown per signal variation with conditional formatting |

Conditional formatting highlights:
- Green cells for positive P&L / R-values
- Red cells for negative values
- Bold headers and auto-sized columns

### 6.3 Trade Charts (`backtest/reporting/trade_charts.py`)

The `TradeChartGenerator` produces individual PNG charts for each trade showing:

- OHLC candlestick price action around the trade
- IB zone highlighting (shaded rectangle for the Initial Balance range)
- Entry marker (green triangle for buy, red triangle for sell)
- Exit marker (with P&L annotation)
- Stop loss zone (red shading)
- Take profit zone (green shading)

Charts are saved as PNG files in the `trades/` subdirectory of the output folder.

### 6.4 Summary Text

A `summary.txt` file with human-readable results:

```
Backtest Summary
================
Symbol: GER40
Period: 2023-01-01 to 2025-10-31
Total Trades: 187
Win Rate: 52.4%
Total R: 45.20
Sharpe Ratio: 1.85
Max Drawdown: -12.3R
Profit Factor: 1.65
```

---

## 7. Results Directory Structure

Each backtest run creates a timestamped output directory:

```
backtest/output/
  20260208_143022_GER40_v8/
  |
  |-- config.json              # Parameters used for this run
  |-- results.xlsx             # Excel report (Trades, Statistics, By Variation)
  |-- summary.txt              # Human-readable summary
  |-- equity_drawdown.png      # Equity curve + drawdown chart
  |
  +-- trades/
       |-- trade_001_GER40_2023-01-05_BUY_+2.1R.png
       |-- trade_002_GER40_2023-01-12_SELL_-1.0R.png
       |-- ...
```

**Naming Convention:**

- **Directory**: `{timestamp}_{symbol}_{version}/`
- **Trade charts**: `trade_{number}_{symbol}_{date}_{direction}_{result}R.png`

### Parallel Run Output

When running parallel backtests, each group gets its own subdirectory:

```
backtest/output/
  parallel_20260208_150000_GER40/
  |
  |-- results_summary_GER40.csv    # Consolidated results for all groups
  |-- results_full_GER40.json      # Detailed JSON with all metrics
  |-- failed_groups_GER40.json     # Failed groups for retry (if any)
  |
  +-- group_001/
  |    |-- config.json
  |    |-- results.xlsx
  |    +-- ...
  |
  +-- group_002/
       |-- config.json
       |-- results.xlsx
       +-- ...
```

**Summary CSV Columns:**

| Column | Description |
|--------|-------------|
| `group_id` | Unique identifier for the parameter group |
| `total_r` | Total R-multiple result |
| `sharpe` | Sharpe ratio |
| `total_trades` | Number of trades |
| `win_rate` | Win rate percentage |
| `profit_factor` | Profit factor |
| `max_drawdown` | Maximum drawdown in R |
| `r_difference` | Difference between in-sample and control period R |

---

## 8. Performance Characteristics

### Execution Speed

| Configuration | Approximate Time |
|--------------|-----------------|
| Single backtest, tick mode, 2 years, with charts | 3--8 minutes |
| Single backtest, tick mode, 2 years, skip charts | 1--3 minutes |
| Single backtest, candle mode, 2 years | 30--60 seconds |
| 200 groups, 20 workers, tick mode, skip charts | 30--60 minutes |

### Resource Usage

| Resource | Usage |
|----------|-------|
| CPU | One core per worker process (multiprocessing) |
| Memory | ~200--500 MB per worker (tick data + emulator state) |
| Disk | ~50--200 MB per group output (with charts), ~5 MB without |

### Optimization Recommendations

1. **Use `--skip-charts`** for parallel runs. Trade chart generation is the slowest part of reporting.
2. **Cache tick data**: The TickGenerator saves Parquet files. Ensure the cache directory is not cleared between runs.
3. **Match workers to cores**: Use `--workers N` where N is slightly less than the CPU core count (leave 2--4 cores for OS and I/O).
4. **SSD storage**: Tick data I/O is significant. An SSD substantially reduces data loading time.

### Data Requirements

| Symbol | Data Files | Approximate Size |
|--------|-----------|-----------------|
| GER40 | `GER40 1m 01_01_2023-04_11_2025` | ~100--200 MB (CSV) |
| XAUUSD | `XAUUSD 1m 01_01_2023-04_11_2025` | ~100--200 MB (CSV) |
| GER40 Control | `ger40+pepperstone_0411-2001` | ~20--40 MB (CSV) |
| XAUUSD Control | `xauusd_oanda_0411-2001` | ~20--40 MB (CSV) |

Data folders are configured in `backtest/config.py` under the `DATA_FOLDERS` dictionary. The base path defaults to `C:/Trading/ib_trading_bot/dual_v3/data`.

### Reproducibility

The slow backtest engine is fully deterministic when:
- `random_seed` is set (default: 42) in `BacktestConfig`
- The same data files are used
- The same parameters are provided

The random seed controls the tick generator's jitter noise. Setting `random_seed=None` produces different tick paths on each run, useful for robustness testing.
