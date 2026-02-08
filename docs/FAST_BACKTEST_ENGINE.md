# Fast Backtest Engine -- Architecture and Usage Guide

> **Last updated**: 2026-02-08
> **Applies to**: dual_v3 parameter optimizer
> **Entry point**: `dual_v3/params_optimizer/engine/fast_backtest.py`

---

## Table of Contents

1. [Purpose and Design Philosophy](#1-purpose-and-design-philosophy)
2. [Vectorized Day-by-Day Processing](#2-vectorized-day-by-day-processing)
3. [FastBacktestOptimized](#3-fastbacktestoptimized)
4. [IB Cache Precomputation](#4-ib-cache-precomputation)
5. [Limitations vs. Slow Engine](#5-limitations-vs-slow-engine)
6. [Performance Metrics and Scoring](#6-performance-metrics-and-scoring)
7. [API Reference](#7-api-reference)
8. [Integration with Parameter Optimizer](#8-integration-with-parameter-optimizer)

---

## 1. Purpose and Design Philosophy

The fast backtest engine exists for one reason: **parameter grid search at scale**. While the [Slow Backtest Engine](SLOW_BACKTEST_ENGINE.md) runs the actual IBStrategy code tick-by-tick for maximum fidelity, the fast engine reimplements the signal detection and trade simulation logic using vectorized NumPy/Pandas operations, achieving a **50--75x speedup** over the slow engine.

### Speed Comparison

| Engine | Time per Parameter Set (2 years) | Time for 10,000 Sets |
|--------|--------------------------------|---------------------|
| Slow (tick mode) | 3--8 minutes | ~35--55 days |
| Slow (candle mode) | 30--60 seconds | ~3.5--7 days |
| Fast | 4--6 seconds | ~11--17 hours |
| Fast Optimized | 3--5 seconds | ~8--14 hours |

### When to Use Each Engine

```
Parameter Search (10,000+ combinations)
     |
     v
[Fast Backtest Engine]
     |
     | Top 20-50 candidates identified
     v
[Slow Backtest Engine -- tick mode]
     |
     | Final validation with full fidelity
     v
[Production parameters selected]
```

### Design Principles

- **Day-by-day processing**: Instead of iterating candle-by-candle, the engine processes entire trading days as vectorized operations
- **Signal detection on M2 bars**: Resamples M1 data to M2 for signal detection (matching the live strategy)
- **Trade simulation on M1 bars**: Once a signal is detected, the trade is simulated using M1 price data for SL/TP/TSL resolution
- **No MT5 emulation overhead**: No singleton, no position tracking, no margin calculation -- just pure signal + trade math
- **IB cache support**: Precomputed IB ranges avoid redundant calculations across parameter combinations

### Source Files

| File | Role |
|------|------|
| `params_optimizer/engine/fast_backtest.py` | Core fast backtest implementation |
| `params_optimizer/engine/fast_backtest_optimized.py` | Optimized variant with vectorized timezone ops |
| `params_optimizer/engine/backtest_wrapper.py` | Wrapper running actual IBStrategy (for comparison) |
| `params_optimizer/engine/metrics_calculator.py` | Combined scoring and ranking |
| `params_optimizer/engine/parameter_grid.py` | Parameter combination generator |

---

## 2. Vectorized Day-by-Day Processing

The core speedup comes from processing data **one day at a time** using Pandas/NumPy operations rather than iterating bar-by-bar.

### Processing Flow

```
FOR EACH trading day:

    1. EXTRACT day's M1 bars
       |-- Filter by date from pre-loaded DataFrame

    2. COMPUTE IB RANGE
       |-- Filter bars within IB window (e.g., 08:00-08:30 local time)
       |-- IB_high = max(high), IB_low = min(low)
       |-- (Or read from IB cache if available)

    3. RESAMPLE to M2
       |-- Group consecutive pairs of M1 bars
       |-- Aggregate: open=first, high=max, low=min, close=last

    4. DETECT SIGNALS on M2 bars (vectorized)
       |-- Check each variation in priority order:
       |     1. Reverse
       |     2. OCAE
       |     3. TCWE
       |     4. REV_RB
       |-- First valid signal wins (highest priority)

    5. SIMULATE TRADE on M1 bars (if signal found)
       |-- Entry at signal price
       |-- Walk forward through M1 bars:
       |     Check SL hit -> exit at SL
       |     Check virtual TP hit -> advance TSL
       |     Check time window end -> exit at close
       |-- Record trade result (R-multiple, variation, entry/exit)

CALCULATE aggregate metrics
```

### Signal Detection Details

Each signal variation has its own detection method:

**Reverse (`_check_reverse`):**
- Price breaks above IB_high (or below IB_low)
- Then reverses back through the opposite boundary
- Entry on the reversal confirmation candle

**OCAE -- Open-Close Above Extension (`_check_ocae`):**
- Both open and close of a candle are above the upper extension (IB_high + extension_pct * IB_range)
- Or both below the lower extension
- Entry at candle close

**TCWE -- Two Candles With Extension (`_check_tcwe`):**
- Two consecutive candles with closes beyond the extension level
- Confirms sustained breakout momentum
- Uses pre-computed "EQ touched" cumulative mask via `np.cumsum` for performance

**REV_RB -- Reverse with Rebound (`_check_rev_rb`):**
- Similar to Reverse but requires a rebound confirmation
- Controlled by `REV_RB_PCT` parameter
- Can be disabled entirely (skipped when `REV_RB_enabled=False`)

### Trade Simulation (`_simulate_after_entry`)

Once a signal is detected, the trade is simulated bar-by-bar on M1 data:

```python
def _simulate_after_entry(self, m1_bars, entry_idx, signal):
    """
    Simulate trade from entry point forward.

    Walks through M1 bars checking:
    1. SL hit (using bar low for buys, bar high for sells)
    2. Virtual TP hit (triggers TSL advancement)
    3. Time window expiry (exit at market)

    TSL stepping:
    - When price reaches current_tp:
        new_sl = current_sl + (tsl_sl * risk)
        new_tp = current_tp + (tsl_target * risk)
    - Repeats until SL is hit or time expires
    """
```

The TSL logic in the fast engine mirrors the live strategy's stepped trailing:

```
Initial state:
    sl = signal.stop_loss
    tp = signal.take_profit
    risk = abs(entry - sl)

On each M1 bar:
    IF price reaches tp:
        sl += tsl_sl * risk      # Move SL forward
        tp += tsl_target * risk   # Extend TP target

    IF price hits sl:
        EXIT at sl
        result_r = (sl - entry) / risk  # For buys
```

---

## 3. FastBacktestOptimized

The `FastBacktestOptimized` class (`fast_backtest_optimized.py`) extends `FastBacktest` with additional performance optimizations that provide approximately **1.3x additional speedup**.

### Key Optimization: Vectorized Timezone Conversion

The primary bottleneck in the base `FastBacktest` was timezone conversion. Converting timestamps from UTC to local market time (e.g., Europe/Berlin) for IB window calculation used:

```python
# SLOW: apply + lambda (~20x slower)
df["local_time"] = df["time"].apply(lambda x: x.astimezone(tz))
```

The optimized version uses Pandas' built-in vectorized timezone operations:

```python
# FAST: vectorized dt.tz_convert
df["local_time"] = df["time"].dt.tz_convert(target_tz)
```

This single change produces a ~20x speedup for the timezone conversion step, which translates to ~1.3x overall speedup since timezone conversion is only one part of the total computation.

### Additional Optimizations

**Pre-computed EQ Touched Mask (TCWE and OCAE):**

For the TCWE and OCAE signal variations, the engine needs to know whether price has "touched" the EQ level (midpoint of IB) before checking extension conditions. The optimized version pre-computes a cumulative boolean mask:

```python
# Pre-compute whether EQ has been touched at any point up to each bar
eq_touched_cumsum = np.cumsum(
    (bars["high"] >= eq_level) & (bars["low"] <= eq_level)
)
eq_touched_at_bar = eq_touched_cumsum > 0
```

This replaces a per-bar loop with a single vectorized operation.

**Cache Clearing:**

The optimized class provides a `clear_cache()` method for clearing internal caches when switching between symbols, preventing stale timezone data from affecting results.

---

## 4. IB Cache Precomputation

The IB (Initial Balance) range is the same for all parameter combinations that share the same IB window definition. Computing it once and caching the results eliminates redundant work.

### How It Works

```
BEFORE grid search:

    FOR EACH unique (ib_start, ib_end, ib_timezone) in parameter grid:
        FOR EACH trading day in date range:
            |-- Filter M1 bars within IB window
            |-- IB_high = max(high)
            |-- IB_low = min(low)
            |-- IB_range = IB_high - IB_low
            |-- Store in cache: ib_cache[date] = (IB_high, IB_low, IB_range)

DURING grid search:

    FastBacktest receives ib_cache parameter
    |-- On each day, reads IB values from cache instead of recomputing
    |-- Saves ~10-15% of per-day computation time
```

### Cache Structure

```python
ib_cache = {
    datetime.date(2023, 1, 2): {
        "ib_high": 14125.3,
        "ib_low": 14089.7,
        "ib_range": 35.6,
    },
    datetime.date(2023, 1, 3): {
        "ib_high": 14201.1,
        "ib_low": 14178.4,
        "ib_range": 22.7,
    },
    # ... one entry per trading day
}
```

### When the Cache Is Invalid

The IB cache is specific to a particular IB window configuration. If the grid search varies the IB window parameters (start time, end time, or timezone), separate caches must be built for each unique window.

---

## 5. Limitations vs. Slow Engine

The fast engine trades fidelity for speed. Understanding the differences is critical for interpreting results.

### Comparison Table

| Aspect | Slow Engine | Fast Engine |
|--------|------------|-------------|
| Code executed | Actual IBStrategy class | Reimplemented signal logic |
| Tick resolution | 5-second synthetic ticks | M1 bar-level |
| Position management | Full MT5 emulator | Simplified entry/exit math |
| Margin tracking | Yes | No |
| Account balance updates | Yes (compounding) | No (fixed risk per trade) |
| Spread simulation | Bid/ask spread on every tick | Applied at entry only |
| Slippage | Configurable | Not simulated |
| News filter | Via actual strategy code | Reimplemented (may differ) |
| TSL precision | Tick-level TP/SL checks | M1 bar-level checks |
| Commission | Configurable | Not applied |
| Multi-position | Emulated (one per magic) | One trade per day assumed |

### Where Results May Diverge

1. **Intra-bar execution**: The fast engine checks SL/TP against M1 bar high/low. If both SL and TP would be hit within the same bar, the fast engine uses a heuristic (assumes the adverse direction hit first). The slow engine resolves this at the tick level.

2. **TSL step timing**: In the slow engine, TSL advances the moment a tick crosses the virtual TP. In the fast engine, the check happens at M1 bar granularity, which may delay TSL advancement by up to 59 seconds.

3. **Spread effects**: The slow engine applies spread on every tick (bid for sells, ask for buys). The fast engine applies spread at entry only, which may slightly favor results.

4. **Compounding**: The slow engine updates account balance after each trade, affecting subsequent position sizes. The fast engine uses a fixed risk amount, making results independent of trade sequence.

### Acceptable Divergence

In practice, the fast engine produces results that correlate strongly with the slow engine (R-squared > 0.95 for total R across parameter sets). The ranking of parameter combinations is generally preserved, making the fast engine reliable for screening purposes.

**Recommended workflow**: Use the fast engine to identify the top 20--50 parameter sets, then validate those with the slow engine.

---

## 6. Performance Metrics and Scoring

The `MetricsCalculator` (`metrics_calculator.py`) provides standardized scoring for comparing parameter combinations.

### Computed Metrics

For each backtest result, the following metrics are computed:

| Metric | Description |
|--------|-------------|
| `total_r` | Sum of all trade R-multiples |
| `sharpe` | Sharpe ratio (annualized, based on daily R-returns) |
| `win_rate` | Percentage of trades with R > 0 |
| `total_trades` | Number of trades |
| `profit_factor` | Sum of winning R / abs(Sum of losing R) |
| `max_drawdown` | Maximum peak-to-trough decline in cumulative R |
| `avg_r` | Average R per trade |
| `by_variation` | Breakdown of all metrics per signal variation |

### Combined Score Formula

The MetricsCalculator computes a weighted combined score for ranking:

```
combined_score = (
    0.40 * normalized(total_r) +
    0.35 * normalized(sharpe) +
    0.25 * normalized(win_rate)
)
```

**Normalization**: Min-max normalization across all parameter sets in the batch:

```python
normalized(x) = (x - x_min) / (x_max - x_min)
```

### Ranking Output

The calculator produces a ranked list:

```
Rank  Group ID          Total R  Sharpe  WinRate  Score
----  ----------------  -------  ------  -------  -----
1     GER40_grp_042     67.3     2.15    56.2%    0.891
2     GER40_grp_117     63.8     2.08    54.8%    0.864
3     GER40_grp_089     61.2     1.95    57.1%    0.852
...
```

### Summary Report

The `generate_summary_report()` method produces a human-readable text report with:

- Top N parameter sets by combined score
- Distribution statistics (mean, median, std for each metric)
- Parameter value frequency analysis (which values appear most in top results)
- By-variation performance summary

---

## 7. API Reference

### FastBacktest

```python
class FastBacktest:
    def __init__(
        self,
        symbol: str,
        params: dict,
        m1_data: pd.DataFrame,
        risk_amount: float = 1000.0,
        ib_cache: dict = None,
        news_filter_enabled: bool = False,
    ):
        """
        Initialize the fast backtest engine.

        Args:
            symbol: Trading symbol ("GER40" or "XAUUSD")
            params: Strategy parameters dict containing:
                - ib_start_time: str ("08:00")
                - ib_end_time: str ("08:30")
                - ib_timezone: str ("Europe/Berlin")
                - wait_minutes: int (20)
                - trade_window_minutes: int (360)
                - variations: dict of per-variation params
                - tsl_target: float
                - tsl_sl: float
            m1_data: DataFrame with columns [time, open, high, low, close]
                     Timestamps must be timezone-aware (UTC)
            risk_amount: Fixed risk per trade in USD
            ib_cache: Optional precomputed IB ranges (see Section 4)
            news_filter_enabled: Whether to apply news filter
        """

    def run(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> dict:
        """
        Run the backtest over the specified period.

        Returns:
            dict with keys:
                - "trades": list of trade dicts
                - "total_r": float
                - "total_trades": int
                - "win_rate": float
                - "sharpe": float
                - "profit_factor": float
                - "max_drawdown": float
                - "by_variation": dict of per-variation metrics
        """
```

### FastBacktestOptimized

```python
class FastBacktestOptimized(FastBacktest):
    """
    Optimized variant with vectorized timezone conversion.

    Drop-in replacement for FastBacktest. Same API, ~1.3x faster.
    Call clear_cache() when switching between symbols.
    """

    def clear_cache(self):
        """Clear internal timezone and computation caches."""
```

### BacktestWrapper

```python
class BacktestWrapper:
    """
    Wrapper that runs the ACTUAL IBStrategy for reproducibility testing.

    Much slower than FastBacktest but uses the real strategy code.
    Used to validate that FastBacktest results match the slow engine.
    """

    def __init__(
        self,
        symbol: str,
        params: dict,
        m1_data: pd.DataFrame,
        risk_amount: float = 1000.0,
    ):
        """
        Args:
            symbol: Trading symbol
            params: Strategy parameters
            m1_data: M1 OHLC DataFrame (UTC timestamps)
            risk_amount: Fixed risk per trade
        """

    def run(self, start_date: datetime, end_date: datetime) -> dict:
        """
        Run backtest using actual IBStrategy code.

        Patches MT5 module, creates emulator, feeds candles one by one.
        Returns same format as FastBacktest.run().
        """
```

### MetricsCalculator

```python
class MetricsCalculator:
    """Calculate combined scores and rankings for backtest results."""

    def __init__(
        self,
        weights: dict = None,
    ):
        """
        Args:
            weights: Scoring weights dict. Defaults to:
                {"total_r": 0.40, "sharpe": 0.35, "win_rate": 0.25}
        """

    def calculate_scores(
        self,
        results: list[dict],
    ) -> pd.DataFrame:
        """
        Calculate combined scores for all results.

        Args:
            results: List of backtest result dicts

        Returns:
            DataFrame with original metrics + normalized scores + combined_score,
            sorted by combined_score descending
        """

    def generate_summary_report(
        self,
        scored_df: pd.DataFrame,
        top_n: int = 20,
    ) -> str:
        """
        Generate human-readable summary report.

        Args:
            scored_df: Output from calculate_scores()
            top_n: Number of top results to display

        Returns:
            Formatted string report
        """
```

### ParameterGrid

```python
class ParameterGrid:
    """Generate all valid parameter combinations for grid search."""

    def __init__(
        self,
        param_ranges: dict,
    ):
        """
        Args:
            param_ranges: Dict mapping parameter names to lists of values.
                Example:
                {
                    "ib_start_time": ["07:30", "08:00", "08:30"],
                    "wait_minutes": [10, 15, 20, 30],
                    "tsl_target": [1.0, 1.5, 2.0],
                    "tsl_sl": [0.3, 0.5, 0.7],
                    "REV_RB_enabled": [True, False],
                    "REV_RB_PCT": [0.3, 0.5, 0.7],
                }
        """

    def generate(self) -> list[dict]:
        """
        Generate all valid parameter combinations.

        Smart filtering applied:
        - Skips REV_RB_PCT variations when REV_RB_enabled=False
        - Skips TSL_SL variations when TSL is disabled

        Returns:
            List of parameter dicts, one per combination
        """

    def to_tuples(self) -> list[tuple]:
        """
        Convert parameter dicts to hashable tuples.
        Useful for tracking completed combinations in resume mode.
        """

    def filter_completed(
        self,
        completed: set[tuple],
    ) -> list[dict]:
        """
        Remove already-completed combinations (for resume support).

        Args:
            completed: Set of tuples from previous runs

        Returns:
            Filtered list of remaining parameter dicts
        """

    def estimate_time(
        self,
        seconds_per_combo: float = 5.0,
        num_workers: int = 20,
    ) -> str:
        """
        Estimate total grid search time.

        Returns:
            Formatted string: "10,000 combinations, ~14 hours with 20 workers"
        """
```

---

## 8. Integration with Parameter Optimizer

The fast backtest engine is designed to be used within a larger parameter optimization pipeline.

### Typical Optimization Workflow

```
1. DEFINE PARAMETER RANGES
   |
   |  param_ranges = {
   |      "ib_start_time": ["07:30", "08:00", "08:30"],
   |      "wait_minutes": [10, 15, 20, 25, 30],
   |      "trade_window_minutes": [240, 300, 360, 420],
   |      "tsl_target": [1.0, 1.5, 2.0, 2.5],
   |      "tsl_sl": [0.3, 0.5, 0.7, 1.0],
   |      "REV_RB_enabled": [True, False],
   |      "REV_RB_PCT": [0.3, 0.5, 0.7],
   |      # ... more parameters
   |  }
   |
   v
2. GENERATE GRID
   |
   |  grid = ParameterGrid(param_ranges)
   |  combinations = grid.generate()
   |  print(grid.estimate_time(seconds_per_combo=5, num_workers=20))
   |  # "8,400 combinations, ~11.7 hours with 20 workers"
   |
   v
3. PRECOMPUTE IB CACHE
   |
   |  FOR EACH unique IB window in combinations:
   |      ib_cache = precompute_ib_ranges(m1_data, ib_window)
   |
   v
4. PARALLEL GRID SEARCH (multiprocessing)
   |
   |  FOR EACH combination (distributed across workers):
   |      engine = FastBacktestOptimized(symbol, params, m1_data, ib_cache=cache)
   |      result = engine.run(start_date, end_date)
   |      results.append(result)
   |
   v
5. SCORE AND RANK
   |
   |  calculator = MetricsCalculator()
   |  scored = calculator.calculate_scores(results)
   |  report = calculator.generate_summary_report(scored, top_n=50)
   |
   v
6. VALIDATE TOP CANDIDATES (Slow Engine)
   |
   |  FOR EACH top_50_params:
   |      slow_result = BacktestRunner(symbol, params).run_with_bot_integration()
   |      compare(fast_result, slow_result)
   |
   v
7. SELECT PRODUCTION PARAMETERS
   |
   |  Best params -> config for live bot
```

### Worker Function Example

```python
def run_fast_backtest_worker(args):
    """Worker function for multiprocessing pool."""
    params, m1_data, start_date, end_date, ib_cache = args

    engine = FastBacktestOptimized(
        symbol="GER40",
        params=params,
        m1_data=m1_data,
        risk_amount=1000.0,
        ib_cache=ib_cache,
    )

    result = engine.run(start_date, end_date)
    result["params"] = params
    return result
```

### Resume Support

For long-running grid searches, the `ParameterGrid` supports resume:

```python
# Load completed results from previous run
completed_tuples = set()
for result in previous_results:
    completed_tuples.add(grid.param_dict_to_tuple(result["params"]))

# Get remaining combinations
remaining = grid.filter_completed(completed_tuples)
print(f"Resuming: {len(remaining)} of {len(all_combinations)} remaining")
```

### VM Deployment Pattern

The grid search is designed for deployment on cloud VMs with high core counts:

```
VM 1 (24 cores):
    Groups 1-5000 -> 20 workers

VM 2 (24 cores):
    Groups 5001-10000 -> 20 workers

Each VM:
    - Loads the same M1 data
    - Receives a groups JSON file with its assigned parameter sets
    - Writes results to its own output directory
    - Results are merged post-run
```

### Scoring Weight Customization

The default weights (total_r: 40%, sharpe: 35%, win_rate: 25%) can be adjusted:

```python
# Prioritize consistency over raw returns
calculator = MetricsCalculator(weights={
    "total_r": 0.25,
    "sharpe": 0.50,
    "win_rate": 0.25,
})

# Prioritize raw returns
calculator = MetricsCalculator(weights={
    "total_r": 0.60,
    "sharpe": 0.25,
    "win_rate": 0.15,
})
```

Weights must sum to 1.0. The choice of weights reflects the trader's preference between total profitability (total_r), risk-adjusted returns (sharpe), and consistency (win_rate).
