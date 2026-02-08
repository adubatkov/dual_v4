# Parameter Optimizer

Exhaustive grid search system for the Initial Balance (IB) trading strategy. Evaluates 1.48M+ parameter combinations using a Master-Worker multiprocessing architecture with checkpoint/resume support for crash recovery.

**Last updated**: 2026-02-08

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Parameter Grid](#parameter-grid)
4. [Data Preprocessing](#data-preprocessing)
5. [Configuration](#configuration)
6. [Running the Optimizer](#running-the-optimizer)
7. [Checkpoint and Resume](#checkpoint-and-resume)
8. [Output Files](#output-files)
9. [AWS Deployment](#aws-deployment)
10. [Performance Characteristics](#performance-characteristics)
11. [Troubleshooting](#troubleshooting)

---

## Overview

The parameter optimizer performs a brute-force grid search across all possible combinations of the IB strategy parameters. Each combination is backtested against historical M1 (1-minute) candle data for GER40 (DAX) and XAUUSD (Gold) using a fast vectorized backtest engine.

### Key Metrics

| Metric | Value |
|--------|-------|
| Combinations (expanded grid) | ~1.48M per symbol |
| Combinations (standard grid) | ~20K per symbol |
| Speed per combination | ~4-5 seconds |
| AWS 96-CPU runtime (90 workers) | ~20-25 hours |
| IB cache speedup | ~30% faster |
| Memory per worker | ~500 MB |

### Supported Symbols

| Symbol | Timezone | IB Sessions |
|--------|----------|-------------|
| GER40 | Europe/Berlin | 08:00-08:30, 08:00-09:00, 09:00-09:30 |
| XAUUSD | Asia/Tokyo | 09:00-09:30, 09:00-10:00, 10:00-10:30 |

### Strategy Variations

Each parameter combination is evaluated across four trade entry variations:

- **OCAE** (Once Close At Entry) -- TSL_TARGET > 0
- **TCWE** (Trailing Close With Exit) -- TSL_TARGET > 0 AND RR_TARGET >= 1.0
- **Reverse** -- STOP_MODE == "eq"
- **REV_RB** (Reverse Rebound) -- REV_RB_ENABLED == True

---

## Architecture

The system follows a Master-Worker pattern built on Python's `multiprocessing.Pool`.

```
                    +-------------------+
                    |  run_optimizer.py |
                    |      (CLI)        |
                    +---------+---------+
                              |
                    +---------v---------+
                    |  OptimizerMaster  |
                    |    master.py      |
                    +---------+---------+
                              |
          +-------------------+-------------------+
          |                   |                   |
    +-----v-----+      +-----v-----+       +-----v-----+
    |  Worker 1  |      |  Worker 2  |  ...  | Worker 90  |
    |  worker.py |      |  worker.py |       |  worker.py |
    +-----+-----+      +-----+-----+       +-----+-----+
          |                   |                   |
    +-----v------+     +-----v------+      +-----v------+
    |FastBacktest|     |FastBacktest|      |FastBacktest|
    |+ IB Cache  |     |+ IB Cache  |      |+ IB Cache  |
    +------------+     +------------+      +------------+

    +-------------------+     +--------------------+
    | CheckpointManager |     | VariationAggregator|
    | (crash recovery)  |     | (on-the-fly stats) |
    +-------------------+     +--------------------+
```

### Component Responsibilities

**run_optimizer.py** -- CLI entry point. Parses arguments, dispatches to data preparation, optimization run, or info display.

**OptimizerMaster (master.py)** -- Coordinates the entire optimization. Generates the parameter grid, creates the worker pool, distributes work via `imap_unordered`, tracks progress with ETA, triggers periodic checkpoints, and finalizes results with ranking and variation analytics.

**Worker (worker.py)** -- Each worker process loads M1 data and the IB cache once during initialization (module-level cache). Processes individual parameter combinations by calling `FastBacktestOptimized.run_with_params()`. Returns a dict of metrics (total_r, sharpe_ratio, winrate, trade count, etc.).

**FastWorker (fast_worker.py)** -- Alternative worker implementation using the base `FastBacktest` engine. Parameter tuples are converted to dicts using a fixed key order defined in `ParameterGrid.PARAM_KEYS`.

**CheckpointManager (checkpoint.py)** -- Saves progress atomically at configurable intervals. Supports async writes via `ThreadPoolExecutor` for ~10-17% performance improvement. Stores completed tuple sets in JSON, partial results in Parquet, and VariationAggregator state in pickle.

**VariationAggregator** -- Aggregates statistics per variation in real time as results arrive. Produces per-variation summaries, top-N rankings, and best-parameter-per-variation reports.

### News Filter (5ers Compliance)

The optimizer includes a news filter that blocks trades within a configurable window around high-impact news events. This is enabled by default to match prop firm (The5ers) compliance requirements. Default: 2 minutes before and 2 minutes after high-impact events.

---

## Parameter Grid

Two grid modes are available, selected via `GRID_MODE` in `config.py`.

### Standard Grid (~20K combinations, ~2 hours on 8-core)

| Parameter | Values | Count |
|-----------|--------|-------|
| IB_WAIT_MINUTES | 0, 15 | 2 |
| TRADE_WINDOW_MINUTES | 60, 90, 120 | 3 |
| RR_TARGET | 0.5, 1.0, 1.5, 2.0 | 4 |
| STOP_MODE | ib_start, eq | 2 |
| TSL_TARGET | 0, 0.5, 1.0, 1.5 | 4 |
| TSL_SL | 0.5, 1.0 | 2 |
| MIN_SL_PCT | 0.001 | 1 |
| REV_RB_ENABLED | True, False | 2 |
| REV_RB_PCT | 0.5 | 1 |
| IB_BUFFER_PCT | 0.05, 0.10, 0.15, 0.20, 0.25 | 5 |
| MAX_DISTANCE_PCT | 0.5, 0.75, 1.0, 1.25 | 4 |

Total per IB time config: base combinations * REV_RB factor. Multiplied by 3 IB time configs per symbol.

### Expanded Grid (~1.48M+ combinations, for AWS)

| Parameter | Values | Count |
|-----------|--------|-------|
| IB_WAIT_MINUTES | 0, 10, 15, 20 | 4 |
| TRADE_WINDOW_MINUTES | 40, 60, 90, 120, 150, 180, 210, 240 | 8 |
| RR_TARGET | 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0 | 7 |
| STOP_MODE | ib_start, eq | 2 |
| TSL_TARGET | 0.0, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0 | 7 |
| TSL_SL | 0.5, 0.75, 1.0, 1.25, 1.5, 2.0 | 6 |
| MIN_SL_PCT | 0.0015 | 1 |
| REV_RB_ENABLED | True, False | 2 |
| REV_RB_PCT | 1.0 | 1 |
| IB_BUFFER_PCT | 0.0, 0.05, 0.10, 0.15, 0.20 | 5 |
| MAX_DISTANCE_PCT | 0.5, 0.75, 1.0, 1.25, 1.5 | 5 |

### Parameter Descriptions

| Parameter | Description |
|-----------|-------------|
| IB_WAIT_MINUTES | Minutes to wait after IB period ends before allowing trades |
| TRADE_WINDOW_MINUTES | Maximum window (from IB end) during which trades can be entered |
| RR_TARGET | Risk-to-reward ratio target for take-profit |
| STOP_MODE | Where to place the stop-loss: `ib_start` (IB high/low) or `eq` (IB equilibrium) |
| TSL_TARGET | Trailing stop activation threshold in R (0 = disabled) |
| TSL_SL | Trailing stop-loss distance in R once activated |
| MIN_SL_PCT | Minimum stop-loss distance as percentage of price |
| REV_RB_ENABLED | Enable reverse rebound entry logic |
| REV_RB_PCT | Rebound percentage threshold for reverse entries |
| IB_BUFFER_PCT | Buffer added to IB high/low for breakout confirmation |
| MAX_DISTANCE_PCT | Maximum allowed distance from IB level for trade entry |

### IB Time Configurations

Each symbol has multiple IB window configurations tested independently:

**GER40 (Europe/Berlin)**:
- 08:00 - 08:30 (30-minute IB)
- 08:00 - 09:00 (60-minute IB)
- 09:00 - 09:30 (30-minute IB)

**XAUUSD (Asia/Tokyo)**:
- 09:00 - 09:30 (30-minute IB)
- 09:00 - 10:00 (60-minute IB)
- 10:00 - 10:30 (30-minute IB)

---

## Data Preprocessing

Before running the optimizer, raw M1 CSV data must be converted to optimized Parquet format and the IB cache must be pre-computed.

### Step 1: Prepare Parquet Data

Converts raw CSV files to a single Parquet file with trading hours filter applied.

```bash
cd dual_v3
python -m params_optimizer.run_optimizer --prepare-data --symbol GER40
python -m params_optimizer.run_optimizer --prepare-data --symbol XAUUSD
```

**What it does** (`data/prepare_data.py`):
1. Loads all CSV files from `dual_v3/data/{SYMBOL} 1m 01_01_2023-04_11_2025/`
2. Parses timestamps with UTC timezone awareness
3. Removes duplicate timestamps
4. Filters to trading hours:
   - GER40: 07:00-23:00 Europe/Berlin
   - XAUUSD: Weekdays only (24/5 market)
5. Saves as Snappy-compressed Parquet to `dual_v3/data/optimized/{SYMBOL}_m1.parquet`

**Output**: ~10x faster loading compared to raw CSV.

### Step 2: Pre-compute IB Cache

Pre-calculates Initial Balance values (IBH, IBL, EQ) for every trading day and every IB time configuration.

```bash
python -m params_optimizer.data.ib_precompute --symbol GER40
python -m params_optimizer.data.ib_precompute --symbol XAUUSD
```

**What it does** (`data/ib_precompute.py`):
1. Loads M1 Parquet data
2. For each IB time config (start, end, timezone):
   - Groups data by trading date
   - Computes IBH (IB High), IBL (IB Low), EQ (equilibrium midpoint)
3. Stores as pickle: `dual_v3/data/optimized/{SYMBOL}_ib_cache.pkl`

**Performance impact**: IB calculation accounts for ~30% of per-combination backtest time. The pre-computed cache eliminates this overhead entirely.

### Data Paths

| Type | Path |
|------|------|
| Raw CSV (GER40) | `dual_v3/data/GER40 1m 01_01_2023-04_11_2025/` |
| Raw CSV (XAUUSD) | `dual_v3/data/XAUUSD 1m 01_01_2023-04_11_2025/` |
| Optimized Parquet | `dual_v3/data/optimized/{SYMBOL}_m1.parquet` |
| IB Cache | `dual_v3/data/optimized/{SYMBOL}_ib_cache.pkl` |

---

## Configuration

### config.py

Central configuration file. Key settings:

```python
# Grid mode selection
GRID_MODE = "standard"   # "standard" (~20K) or "expanded" (~1.48M)

# Ranking weights for result scoring
RANKING_WEIGHTS = {
    "Total_R": 0.40,       # Primary: total profit in R
    "Sharpe_Ratio": 0.35,  # Risk-adjusted return
    "Winrate_pct": 0.25,   # Stability
}
```

### OptimizerConfig dataclass

All runtime parameters in a single dataclass:

| Field | Default | Description |
|-------|---------|-------------|
| symbol | (required) | GER40 or XAUUSD |
| num_workers | 90 | Number of parallel worker processes |
| combinations_per_chunk | 10 | imap_unordered chunksize |
| checkpoint_interval | 100 | Save checkpoint every N combinations |
| initial_balance | 100000.0 | Backtest starting balance |
| risk_pct | 1.0 | Risk percentage per trade |
| max_margin_pct | 40.0 | Maximum margin usage |
| news_filter_enabled | True | Enable news filter (5ers compliance) |
| news_before_minutes | 2 | Block trades N minutes before news |
| news_after_minutes | 2 | Block trades N minutes after news |

### optimizer_config.json (Optional)

JSON-based configuration override. Values from this file are used as defaults; CLI arguments take precedence.

```json
{
    "parallelization": {
        "num_workers": 90,
        "combinations_per_chunk": 10
    },
    "checkpoint": {
        "interval": 100,
        "file": "checkpoint.json"
    },
    "ranking_weights": {
        "total_r": 0.40,
        "sharpe_ratio": 0.35,
        "winrate": 0.25
    },
    "news_filter": {
        "enabled": true,
        "before_minutes": 2,
        "after_minutes": 2
    }
}
```

### BLAS Thread Limits

`config.py` sets BLAS thread environment variables to 1 at import time. This prevents NumPy/BLAS from spawning additional threads inside each worker process, which would cause thread contention with the multiprocessing pool:

```python
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
```

---

## Running the Optimizer

All commands are run from the `dual_v3/` directory.

### Show Info (Data Status and Grid Size)

```bash
python -m params_optimizer.run_optimizer --info --symbol GER40
```

Displays: data availability, file sizes, candle counts, date ranges, and total parameter combinations.

### Run Optimization

```bash
# Default workers (CPU count - 6)
python -m params_optimizer.run_optimizer --symbol GER40

# Specify worker count
python -m params_optimizer.run_optimizer --symbol GER40 --workers 4

# Disable news filter
python -m params_optimizer.run_optimizer --symbol GER40 --no-news-filter

# Custom output directory
python -m params_optimizer.run_optimizer --symbol GER40 --output-dir /path/to/results
```

Each run creates a timestamped subdirectory: `output/{SYMBOL}_{N}combos_{YYYYMMDD_HHMMSS}/`

### Resume from Checkpoint

Resume is enabled by default. The optimizer automatically detects and loads the latest checkpoint:

```bash
# Auto-resume (default behavior)
python -m params_optimizer.run_optimizer --symbol GER40

# Resume a specific run directory
python -m params_optimizer.run_optimizer --symbol GER40 --run-dir GER40_144combos_20240107_220100

# Force fresh start (ignore checkpoint)
python -m params_optimizer.run_optimizer --symbol GER40 --no-resume
```

### Generate Excel Report

Generate an Excel report from existing Parquet results without re-running the optimization:

```bash
# From latest run
python -m params_optimizer.run_optimizer --generate-excel --symbol GER40

# From specific run
python -m params_optimizer.run_optimizer --generate-excel --symbol GER40 --run-dir GER40_144combos_20240107_220100
```

---

## Checkpoint and Resume

The checkpoint system enables crash recovery for long-running optimizations (20+ hours on AWS).

### Checkpoint Files

| File | Format | Contents |
|------|--------|----------|
| `checkpoint.json` | JSON | Metadata: symbol, timestamp, completed count, total count, and set of completed parameter tuple strings |
| `results_partial.parquet` | Parquet | All results computed so far (metrics + flattened params) |
| `variation_agg_checkpoint.pkl` | Pickle | Serialized VariationAggregator state |

### Checkpoint Behavior

1. **Periodic saves**: Every `checkpoint_interval` combinations (default: 100), a checkpoint is triggered.
2. **Async writes**: Checkpoint writes happen in a background thread (`ThreadPoolExecutor` with 1 worker) so the main processing loop is not blocked. This provides ~10-17% throughput improvement.
3. **Atomic writes**: All files are written to temporary files first, then atomically renamed. This prevents corruption from mid-write crashes.
4. **Final checkpoint**: After the pool completes, a synchronous final checkpoint is written. Then `results_partial.parquet` is renamed to `results_final.parquet` and checkpoint files are cleaned up.

### Resume Logic

On resume:
1. `checkpoint.json` is loaded. Completed parameter tuples are reconstructed from string representations.
2. `results_partial.parquet` is loaded back into the results list.
3. `variation_agg_checkpoint.pkl` is loaded to restore the VariationAggregator.
4. The full parameter grid is generated, and completed tuples are subtracted to determine remaining work.
5. Remaining combinations are shuffled (deterministic seed=42) and dispatched to workers.

---

## Output Files

After optimization completes, the run directory contains:

| File | Description |
|------|-------------|
| `{SYMBOL}_full_results.parquet` | All combinations with all metrics (~2 GB for expanded grid) |
| `{SYMBOL}_top_100.csv` | Top 100 ranked combinations (human-readable) |
| `{SYMBOL}_best_params.json` | Single best parameter set |
| `{SYMBOL}_summary.txt` | Text report with statistics |
| `{SYMBOL}_variations_summary.csv` | Summary metrics per variation |
| `{SYMBOL}_variation_{VAR}_top500.csv` | Top 500 per variation (OCAE, TCWE, Reverse, REV_RB) |
| `{SYMBOL}_best_by_variation.json` | Best parameters per variation |
| `{SYMBOL}_variation_report.txt` | Detailed variation analytics report |
| `{SYMBOL}_variation_agg.pkl` | Serialized VariationAggregator for later analysis |
| `results_final.parquet` | Renamed from results_partial after completion |

### Ranking Method

Results are ranked using a weighted composite score:

```
Score = 0.40 * normalized(Total_R) + 0.35 * normalized(Sharpe_Ratio) + 0.25 * normalized(Winrate)
```

Only combinations with at least 10 trades (`min_trades=10`) are included in rankings.

### Analyzing Results

For detailed analysis of optimization output, see [RESULTS_ANALYSIS.md](RESULTS_ANALYSIS.md).

---

## AWS Deployment

### Recommended Instance

**EC2 c6i.24xlarge**: 96 vCPU, 192 GB RAM. Use 90 workers to leave 6 CPUs for the OS and master process.

### Setup Steps

```bash
# 1. System setup
sudo apt update && sudo apt upgrade -y
sudo apt install python3.10 python3.10-venv python3-pip screen -y

# 2. Clone and configure
git clone <repository-url>
cd ib-trading-bot
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. Upload data (from local machine)
scp -r "dual_v3/data/optimized/GER40_m1.parquet" user@aws-ip:~/ib-trading-bot/dual_v3/data/optimized/
scp -r "dual_v3/data/optimized/GER40_ib_cache.pkl" user@aws-ip:~/ib-trading-bot/dual_v3/data/optimized/

# 4. Run in screen session
screen -S optimizer
cd dual_v3
python -m params_optimizer.run_optimizer --symbol GER40 --workers 90
# Detach: Ctrl+A, D
# Reattach: screen -r optimizer

# 5. Monitor progress (separate terminal)
cat dual_v3/params_optimizer/output/*/checkpoint.json | python -m json.tool | grep completed_count

# 6. Download results (from local machine)
scp -r user@aws-ip:~/ib-trading-bot/dual_v3/params_optimizer/output ./aws_results/
```

### Crash Recovery on AWS

If the instance crashes or is terminated:

```bash
# Simply re-run the same command -- checkpoint auto-resume is the default
python -m params_optimizer.run_optimizer --symbol GER40 --workers 90
```

The optimizer will detect the checkpoint in the run directory and resume from where it left off.

### Cost Estimation

For the expanded grid (~1.48M combinations) at ~4.5 seconds per combination with 90 workers:

- Wall time: ~20-25 hours
- c6i.24xlarge on-demand (us-east-1): ~$4.08/hour
- Estimated cost: ~$80-$100 per symbol per run

---

## Performance Characteristics

### Worker Initialization

Each worker process initializes once via the Pool initializer:
1. Loads M1 Parquet data (~500 MB per worker, uses OS page cache for shared pages)
2. Loads IB cache from pickle file (each worker loads independently)
3. Optionally initializes the news filter
4. Creates a `FastBacktestOptimized` instance with pre-computed EQ mask

### Backtest Engine

`FastBacktestOptimized` is ~1.6x faster than the base `FastBacktest` and ~75x faster than `BacktestWrapper` (which wraps the full IBStrategy). Key optimizations:
- Vectorized timezone conversion
- Pre-computed EQ mask
- IB cache lookup instead of per-day IB calculation
- Pure NumPy/Pandas operations without Python-level loops

### Memory

- Master process: ~2-4 GB (results accumulation)
- Per worker: ~500 MB (M1 data + IB cache + engine state)
- Total for 90 workers: ~45 GB + master = ~50 GB
- 192 GB instance provides ample headroom

---

## Troubleshooting

### "Data not found for {SYMBOL}"

Run data preparation first:

```bash
python -m params_optimizer.run_optimizer --prepare-data --symbol GER40
```

### "IB cache not found" (Warning)

The optimizer will still run but without the ~30% speedup. Create the cache:

```bash
python -m params_optimizer.data.ib_precompute --symbol GER40
```

### Out of Memory

Reduce the worker count:

```bash
python -m params_optimizer.run_optimizer --symbol GER40 --workers 50
```

### Checkpoint Corruption

Remove checkpoint files and restart:

```bash
# Move to _trash per project protocol
mv dual_v3/params_optimizer/output/*/checkpoint.json dual_v3/_trash/
mv dual_v3/params_optimizer/output/*/results_partial.parquet dual_v3/_trash/
python -m params_optimizer.run_optimizer --symbol GER40 --no-resume
```

### Slow Performance

1. Verify IB cache exists: `ls dual_v3/data/optimized/*_ib_cache.pkl`
2. Use Parquet data (not raw CSV)
3. BLAS thread limits are set in `config.py` -- no manual action needed
4. Check system load: `htop` or `top`

### Checking Data Status

```bash
python -m params_optimizer.run_optimizer --info --symbol GER40
```

---

## File Structure Reference

```
params_optimizer/
    run_optimizer.py             # CLI entry point
    config.py                    # Central configuration and parameter grids
    optimizer_config.json        # Optional JSON config overrides
    orchestrator/
        master.py                # OptimizerMaster (pool coordinator)
        worker.py                # Worker process (FastBacktestOptimized)
        fast_worker.py           # Alternative worker (FastBacktest)
        checkpoint.py            # CheckpointManager (crash recovery)
    engine/
        fast_backtest.py         # Fast vectorized backtest engine
        fast_backtest_optimized.py  # Optimized variant (~1.6x faster)
        parameter_grid.py        # ParameterGrid generation
        metrics_calculator.py    # Ranking and scoring
        backtest_wrapper.py      # Slow wrapper for validation only
    data/
        loader.py                # Data loading with caching
        prepare_data.py          # CSV to Parquet conversion
        ib_precompute.py         # IB cache pre-computation
    analytics/
        variation_aggregator.py  # On-the-fly variation statistics
    reports/
        excel_generator.py       # Excel report generation
    output/                      # Run directories with results
```

---

## Dependencies

```
pandas>=2.0
numpy>=1.24
pyarrow       # Parquet read/write
psutil        # Memory monitoring
pytz          # Timezone handling
openpyxl      # Excel report generation (optional)
```
