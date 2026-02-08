# Strategy Optimization Tools

Standalone analytical tools for studying market microstructure to inform strategy design decisions. These tools are independent of the parameter optimizer and operate directly on M1 (1-minute) candle data.

**Last updated**: 2026-02-08

---

## Table of Contents

1. [Overview](#overview)
2. [Volatility Analysis](#volatility-analysis)
3. [Candle Size Analysis](#candle-size-analysis)
4. [Fractal Analysis](#fractal-analysis)
5. [Data Requirements](#data-requirements)
6. [Output Locations](#output-locations)
7. [Use Cases](#use-cases)

---

## Overview

The `strategy_optimization/` directory contains three analysis tools:

| Tool | Purpose | Key Question Answered |
|------|---------|----------------------|
| Volatility | Intraday activity heatmaps | "When is the market most active?" |
| Candle Size | Candle body distribution by time | "How large are directional moves at each time slot?" |
| Fractals | H1 fractal detection and chart generation | "Which fractal levels were swept during each trade?" |

Each tool reads M1 Parquet data, performs its analysis across multiple timeframes, and produces CSV data files and PNG charts in its own `results/` subdirectory.

### Directory Structure

```
strategy_optimization/
    volatility/
        analyze_volatility.py        # Main script
        results/                     # Output: CSVs and PNGs
    candle_size/
        analyze_candle_size.py       # Main script
        results/                     # Output: CSVs and PNGs
    fractals/
        fractals.py                  # Fractal detection library
        generate_fractal_charts.py   # Chart generation with fractal overlays
        results/                     # Output: per-group chart directories
```

---

## Volatility Analysis

**Script**: `strategy_optimization/volatility/analyze_volatility.py`

### What It Does

Calculates intraday price movement (volatility proxy) for each time-of-day interval. The metric is the sum of `|Close[i] - Close[i-1]|` expressed as a percentage of the close price, averaged across all trading days.

This is not true volatility (standard deviation) but rather a measure of absolute price displacement, which is more relevant for breakout strategies like IB.

### Timeframes Analyzed

| Timeframe | Interval | X-axis Tick Step |
|-----------|----------|-----------------|
| 15min | 15-minute buckets | Every 4 bars (= 1 hour) |
| 30min | 30-minute buckets | Every 2 bars (= 1 hour) |
| 1h | 1-hour buckets | Every bar |

### Processing Steps

1. Loads M1 Parquet data for the symbol
2. Converts timestamps from UTC to the symbol's local timezone
3. Computes absolute price change between consecutive M1 candles as a percentage
4. Floors each candle's timestamp to the analysis interval (15min/30min/1h)
5. Groups by time-of-day slot and aggregates:
   - `total_movement_pct`: sum of all |close-change|% values in that slot
   - `trading_days`: number of unique dates with data for that slot
   - `avg_movement_pct`: total / trading_days
6. Generates a color-coded bar chart (YlOrRd colormap: yellow = low, red = high)

### Chart Features

- Color-coded bars: intensity proportional to activity level
- Session marker lines (vertical dashed):
  - **GER40**: EU Open (09:00 green), US Open (14:30 blue), EU Close (17:30 red)
  - **XAUUSD**: Tokyo Open (09:00 red), London Open (16:00 green), NY Open (21:00 blue)
- Title with symbol, timeframe, date range, and trading day count

### Symbol Configuration

| Symbol | Timezone | M1 Data File |
|--------|----------|-------------|
| GER40 | Europe/Berlin | `GER40_m1.parquet` |
| XAUUSD | Asia/Tokyo | `XAUUSD_m1.parquet` |

### Output

For each symbol and timeframe combination:

| File | Description |
|------|-------------|
| `{SYMBOL}_activity_{TF}.csv` | Raw data: time_slot, total_movement_pct, trading_days, candle_count, avg_movement_pct |
| `{SYMBOL}_activity_{TF}.png` | Bar chart (150 DPI, 16x8 inches) |

Example files for GER40:
- `GER40_activity_15min.csv` / `GER40_activity_15min.png`
- `GER40_activity_30min.csv` / `GER40_activity_30min.png`
- `GER40_activity_1h.csv` / `GER40_activity_1h.png`

### How to Run

```bash
cd dual_v3
python -m strategy_optimization.volatility.analyze_volatility
```

Processes both GER40 and XAUUSD automatically. Skips a symbol if its data file is not found.

---

## Candle Size Analysis

**Script**: `strategy_optimization/candle_size/analyze_candle_size.py`

### What It Does

Measures the average directional move (candle body size) at each time-of-day interval. The metric is `|Open - Close|` expressed as a percentage of the open price, after resampling M1 data to the target timeframe.

Unlike volatility analysis (which sums tick-by-tick changes), candle size analysis measures net directional movement per interval. A time slot with high volatility but small candle bodies indicates choppy/ranging price action, while large bodies indicate sustained directional moves.

### Timeframes Analyzed

Same as volatility analysis: 15min, 30min, 1h.

### Processing Steps

1. Loads M1 Parquet data and converts to local timezone index
2. Resamples to the target timeframe using standard OHLC aggregation:
   - `open`: first M1 open in the interval
   - `close`: last M1 close in the interval
3. Calculates body size: `body_pct = |close - open| / open * 100`
4. Groups by time-of-day slot and computes:
   - `total_body_pct`: sum of all body percentages in that slot
   - `trading_days`: unique dates count
   - `avg_body_pct`: total / trading_days
5. Generates a color-coded bar chart

### Chart Features

Identical layout to volatility charts:
- Color-coded bars (YlOrRd colormap)
- Session marker lines for GER40 (EU/US open/close) and XAUUSD (Tokyo/London/NY open)
- Y-axis: "Average |Open - Close| (%)"

### Output

For each symbol and timeframe:

| File | Description |
|------|-------------|
| `{SYMBOL}_candle_size_{TF}.csv` | Raw data: time_slot, total_body_pct, trading_days, candle_count, avg_body_pct |
| `{SYMBOL}_candle_size_{TF}.png` | Bar chart (150 DPI, 16x8 inches) |

Example files for XAUUSD:
- `XAUUSD_candle_size_15min.csv` / `XAUUSD_candle_size_15min.png`
- `XAUUSD_candle_size_30min.csv` / `XAUUSD_candle_size_30min.png`
- `XAUUSD_candle_size_1h.csv` / `XAUUSD_candle_size_1h.png`

### How to Run

```bash
cd dual_v3
python -m strategy_optimization.candle_size.analyze_candle_size
```

---

## Fractal Analysis

### Fractal Detection Library

**Script**: `strategy_optimization/fractals/fractals.py`

A library module (not a standalone script) providing functions for detecting and tracking Williams 3-bar fractals on H1 (1-hour) timeframe data.

#### Williams 3-Bar Fractal Definition

- **High fractal**: The center bar's high is strictly greater than the high of both the preceding and following bars.
- **Low fractal**: The center bar's low is strictly less than the low of both the preceding and following bars.

A fractal is **confirmed** after the confirming bar (the bar after the center bar) closes. The `confirmed_time` is set to 1 hour after the confirming bar's opening time.

#### Key Functions

**`resample_m1_to_h1(m1_data)`**

Resamples M1 candle data to H1 using standard OHLC aggregation. Preserves tick_volume if present. Input must have UTC-aware timestamps.

**`detect_fractals_3bar(h1_data)`**

Scans H1 data and returns a DataFrame of all detected fractals:

| Column | Description |
|--------|-------------|
| time | Center bar's opening time (when the fractal level occurred) |
| price | Fractal level (high of center bar for high fractals, low for low fractals) |
| type | `"high"` or `"low"` |
| confirmed_time | When the fractal is confirmed (after confirming bar closes) |

**`find_unswept_fractals(fractals, before_time, lookback_hours, m1_data)`**

Filters fractals to those that:
1. Formed within the lookback window (center bar time between `before_time - lookback_hours` and `before_time`)
2. Were confirmed before `before_time`
3. Were NOT swept (price did not reach the fractal level) between confirmation and `before_time`

A high fractal is swept when any M1 bar's high reaches or exceeds the fractal price. A low fractal is swept when any M1 bar's low reaches or falls below the fractal price.

**`find_swept_fractals_in_window(unswept, m1_data, window_start, window_end)`**

Given a set of unswept fractals, finds which ones were swept during a specified time window (typically the IB + trade window period). Returns the sweep time (first M1 bar that touches the level).

**`get_swept_fractals_for_trade(m1_data, ib_start, window_end, lookback_hours=48)`**

High-level convenience function that chains the above steps:
1. Extracts relevant M1 data subset
2. Resamples to H1
3. Detects all fractals
4. Finds unswept fractals at IB start
5. Finds which were swept during the trade window
6. Returns a list of swept fractal dicts sorted by sweep time

---

### Fractal Chart Generator

**Script**: `strategy_optimization/fractals/generate_fractal_charts.py`

Regenerates trade charts from backtest results with fractal sweep lines overlaid.

#### What It Does

1. Reads existing trade data from backtest output directories (`backtest/output/parallel_control/`)
2. For each trade, detects unswept H1 fractals using a 48-hour lookback from the chart start
3. Generates per-trade charts with:
   - M2 (2-minute) candlestick chart
   - IB zone (shaded rectangle)
   - Trade markers (entry arrow, exit cross, SL/TP dashed lines, profit/loss zones)
   - **Black horizontal lines for swept fractals** extending from fractal formation time to sweep time
   - Fractal timestamp labels (e.g., "14:00 UTC")

#### FractalChartGenerator Class

Extends `TradeChartGenerator` from the backtest reporting module. Key additions:

- `generate_chart_with_fractals()`: Main method that produces the fractal-enhanced chart
- `_resample_to_m2()`: Resamples M1 data to 2-minute candles for chart display
- `_draw_fractal_sweeps()`: Draws black horizontal lines from fractal source to sweep point

#### Chart Window

- Start: 1 hour before IB start (local time)
- End: 4.5 hours after IB start (local time)
- Fractal lookback: 48 hours (default)

#### Groups Processed

Default configuration processes four specific groups:
- GER40_055
- XAUUSD_059
- GER40_006
- XAUUSD_010

These are defined in the `GROUPS` constant and can be modified.

#### Input Data

| Source | Path |
|--------|------|
| Backtest results | `backtest/output/parallel_control/{GROUP_ID}/` |
| Trade CSV | `{GROUP_DIR}/trades.csv` |
| Trade config | `{GROUP_DIR}/config.json` |
| Results Excel | `{GROUP_DIR}/results.xlsx` (for entry/exit prices, SL, TP) |
| Original charts | `{GROUP_DIR}/trades/*.jpg` |
| M1 Parquet | `data/control/{SYMBOL}_m1.parquet` |

#### Output

Per-group directories containing regenerated trade charts:

```
strategy_optimization/fractals/results/
    GER40_055/
        trades/
            001_2025-11-03_OCAE_long.jpg
            002_2025-11-04_Reverse_short.jpg
            ...
    XAUUSD_059/
        trades/
            ...
```

#### How to Run

```bash
cd dual_v3/strategy_optimization/fractals
python generate_fractal_charts.py
```

Requires:
- M1 Parquet data in `data/control/`
- Existing backtest results in `backtest/output/parallel_control/`
- The `backtest.reporting.trade_charts` module (for base chart generation)

---

## Data Requirements

All three tools read M1 Parquet data from the `data/control/` directory.

### Required Files

| File | Symbol | Description |
|------|--------|-------------|
| `dual_v3/data/control/GER40_m1.parquet` | GER40 | M1 candles with UTC timestamps |
| `dual_v3/data/control/XAUUSD_m1.parquet` | XAUUSD | M1 candles with UTC timestamps |

### Parquet Column Schema

| Column | Type | Description |
|--------|------|-------------|
| time | datetime64[ns, UTC] | Candle open time in UTC |
| open | float64 | Open price |
| high | float64 | High price |
| low | float64 | Low price |
| close | float64 | Close price |
| tick_volume | int64 | Tick volume (optional) |

If the `time` column is timezone-naive, the tools will localize it to UTC automatically.

### Data Preparation

If using data from the parameter optimizer pipeline, the Parquet files are already in the correct format. If using separate control period data, ensure it follows the same schema and is placed in `data/control/`.

---

## Output Locations

Each tool writes to its own `results/` subdirectory:

| Tool | Output Directory |
|------|-----------------|
| Volatility | `strategy_optimization/volatility/results/` |
| Candle Size | `strategy_optimization/candle_size/results/` |
| Fractals | `strategy_optimization/fractals/results/` |

Output directories are created automatically if they do not exist.

### Complete Output Inventory

**Volatility** (6 CSVs + 6 PNGs):
```
results/
    GER40_activity_15min.csv
    GER40_activity_15min.png
    GER40_activity_30min.csv
    GER40_activity_30min.png
    GER40_activity_1h.csv
    GER40_activity_1h.png
    XAUUSD_activity_15min.csv
    XAUUSD_activity_15min.png
    XAUUSD_activity_30min.csv
    XAUUSD_activity_30min.png
    XAUUSD_activity_1h.csv
    XAUUSD_activity_1h.png
```

**Candle Size** (6 CSVs + 6 PNGs):
```
results/
    GER40_candle_size_15min.csv
    GER40_candle_size_15min.png
    GER40_candle_size_30min.csv
    GER40_candle_size_30min.png
    GER40_candle_size_1h.csv
    GER40_candle_size_1h.png
    XAUUSD_candle_size_15min.csv
    XAUUSD_candle_size_15min.png
    XAUUSD_candle_size_30min.csv
    XAUUSD_candle_size_30min.png
    XAUUSD_candle_size_1h.csv
    XAUUSD_candle_size_1h.png
```

**Fractals** (per-trade charts):
```
results/
    GER40_055/trades/*.jpg
    GER40_006/trades/*.jpg
    XAUUSD_059/trades/*.jpg
    XAUUSD_010/trades/*.jpg
```

---

## Use Cases

### Selecting the IB Window

**Tools**: Volatility + Candle Size

The IB window should capture a period of establishing the daily range before the main directional move. Compare the volatility and candle size charts:

1. Look for a period of moderate activity that precedes the highest-activity session.
2. For GER40, the 08:00-09:00 Berlin window captures pre-market and early DAX activity before the main EU session push.
3. For XAUUSD, the 09:00-10:00 Tokyo window captures the Asian session establishing range before the London session creates directional moves.

If volatility is high but candle bodies are small at a given time, the market is choppy -- not ideal for breakout. If both metrics are high, conditions favor IB breakout entries.

### Understanding Market Activity Patterns

**Tools**: Volatility + Candle Size

Identify which trading sessions produce the most movement for each symbol:

- **GER40**: Typically peaks at EU open (09:00 Berlin), shows secondary activity at US equity open (14:30 Berlin), and quiets down after EU close (17:30 Berlin).
- **XAUUSD**: Activity distributed across Asian, London, and NY sessions. Tokyo open, London open (16:00 Tokyo time), and NY open (21:00 Tokyo time) are key activity points.

### Evaluating Fractal-Based Entry/Exit Refinements

**Tool**: Fractals

Swept fractal levels represent key liquidity points. By overlaying fractal sweeps on trade charts:

1. Observe whether winning trades tend to sweep a fractal before reversing in the trade direction (liquidity grab pattern).
2. Assess whether stop-loss placement aligns with nearby fractal levels.
3. Evaluate whether unswept fractals above/below IB levels act as take-profit magnets.

This information can guide enhancements to the IB strategy, such as:
- Requiring a fractal sweep before entry (confirmation)
- Adjusting stop-loss to the nearest unswept fractal
- Using fractal distance to set take-profit targets

### Identifying Optimal Trade Windows

**Tools**: Volatility + Candle Size

The `TRADE_WINDOW_MINUTES` parameter controls how long after the IB period the strategy continues looking for entries. Volatility analysis shows when directional moves typically occur:

- If activity peaks 60-90 minutes after the IB window, a shorter trade window (60-90 min) captures the main move.
- If activity is distributed over several hours, a longer window (180-240 min) may be appropriate.

Cross-reference with candle size: prefer windows where both activity and directional body size are above average.
