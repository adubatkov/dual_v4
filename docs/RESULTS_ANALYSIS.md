# Results Analysis

Pipeline for analyzing parameter optimization output, selecting the best parameter sets, and validating them through slow-engine backtests. This document covers every script in the `dual_v3/analyze/` directory and how they fit together.

**Last updated**: 2026-02-08

---

## Table of Contents

1. [Pipeline Overview](#pipeline-overview)
2. [Analysis Scripts](#analysis-scripts)
   - [analyze_optimization.py (v2, Parquet)](#analyze_optimizationpy-v2-parquet)
   - [analyze_optimization_db.py (v3, SQLite)](#analyze_optimization_dbpy-v3-sqlite)
   - [best_params_GER40_V8.py / best_params_XAUUSD_V8.py](#best-params-files)
   - [generate_backtest_groups.py](#generate_backtest_groupspy)
   - [create_combined_charts.py](#create_combined_chartspy)
   - [create_combined_charts_control.py](#create_combined_charts_controlpy)
   - [day_of_week_analysis.py](#day_of_week_analysispy)
3. [Parameter Selection Criteria](#parameter-selection-criteria)
4. [Chart Generation](#chart-generation)
5. [Backtest Group Generation](#backtest-group-generation)
6. [Day-of-Week Analysis](#day-of-week-analysis)
7. [Directory Structure](#directory-structure)
8. [End-to-End Workflow](#end-to-end-workflow)

---

## Pipeline Overview

The full analysis pipeline has five stages:

```
Stage 1: Optimization
    params_optimizer -> SQLite DBs (~2 GB each) or Parquet files

Stage 2: Analysis
    analyze_optimization_db.py -> Excel reports + best params dicts (V8)

Stage 3: Group Generation
    generate_backtest_groups.py -> backtest_groups_{SYMBOL}.json

Stage 4: Slow Validation
    parallel backtest runner -> parallel_results/{SYMBOL}/

Stage 5: Combined Analysis
    create_combined_charts.py -> charts_combined/
    day_of_week_analysis.py -> per-day performance tables
```

### Data Flow

```
SQLite DBs
    |
    v
analyze_optimization_db.py
    |
    +-> optimization_analysis_{SYMBOL}_v3.xlsx    (Excel report)
    +-> best_params_{SYMBOL}_V8.py                (Python dict, IB+Buffer mode)
    +-> best_params_{SYMBOL}_V8_STRICT.py         (Python dict, IB+Buffer+MaxDist mode)
    |
    v
generate_backtest_groups.py
    |
    +-> backtest_groups_{SYMBOL}.json             (unique param sets for validation)
    |
    v
[Parallel slow backtest runner]
    |
    +-> parallel_results/{SYMBOL}/{GROUP_ID}/trades.csv
    +-> parallel_results/{SYMBOL}/{GROUP_ID}/config.json
    |
    v
create_combined_charts.py
    |
    +-> parallel_results/charts_combined/*.png    (equity + drawdown charts)
    +-> combined_summary.csv / .xlsx

create_combined_charts_control.py
    |
    +-> parallel_results_control/charts_combined/*.png
    +-> combined_summary_control.csv / .xlsx

day_of_week_analysis.py
    |
    +-> Console output: per-day performance tables
```

---

## Analysis Scripts

### analyze_optimization.py (v2, Parquet)

**Purpose**: Analyze raw Parquet output from the parameter optimizer (older format from VM runs).

**Input**: Parquet files in `results_vm*_*` subdirectories:
- `results_vm1_ger40/results_final.parquet`
- `results_vm2_ger40/results_final.parquet`
- `results_vm3_xauusd/results_final.parquet`
- `results_vm4_xauusd_partial/results_partial.parquet`

**Processing**:
1. Loads and merges multiple Parquet files per symbol
2. Parses `params_json` column into individual parameter columns
3. Groups results by two modes:
   - **IB + Buffer**: Groups by (ib_start, ib_end, ib_tz, ib_wait, ib_buffer_pct)
   - **IB + Buffer + MaxDist**: Groups by (ib_start, ib_end, ib_tz, ib_wait, ib_buffer_pct, max_distance_pct)
4. For each group, finds the best parameter set per variation (OCAE, TCWE, Reverse, REV_RB)
5. Ranks groups by combined Total R across all variations
6. Applies max drawdown filter (default: 10 R)

**Output**:
- `optimization_analysis_{SYMBOL}_v2.xlsx` -- Multi-sheet Excel with summary and variation details
- `best_params_{SYMBOL}_V7.py` -- Python dict (IB+Buffer mode)
- `best_params_{SYMBOL}_V7_STRICT.py` -- Python dict (IB+Buffer+MaxDist mode)

**Run**:
```bash
cd dual_v3/analyze
python analyze_optimization.py
```

---

### analyze_optimization_db.py (v3, SQLite)

**Purpose**: Same analysis as v2 but reading from consolidated SQLite databases. This is the current primary analysis script.

**Input**: SQLite databases:
- `GER40_optimization.db` (table: `results`)
- `XAUUSD_optimization.db` (table: `results`)

Each database contains the complete optimization results with `params_json` column and per-variation metric columns (e.g., `ocae_total_r`, `ocae_sharpe_ratio`, `tcwe_trades`, etc.).

**Key differences from v2**:
- Reads from SQLite instead of Parquet
- Max drawdown filter set to 12 R (vs 10 R in v2)
- Top N groups set to 25 (vs 5 in v2)
- Adds TSL constraint validation: `tsl_target == 0 OR tsl_sl < rr_target + 1`
- Generates V8 (not V7) parameter files

**Output**:
- `optimization_analysis_{SYMBOL}_v3.xlsx` -- Multi-sheet Excel report
- `best_params_{SYMBOL}_V8.py` -- Python dict (IB+Buffer mode)
- `best_params_{SYMBOL}_V8_STRICT.py` -- Python dict (IB+Buffer+MaxDist mode)

**Excel Sheets**:

| Sheet | Contents |
|-------|----------|
| IB+Buffer Summary | Top 25 groups ranked by combined Total R |
| IB+Buffer Details | Per-variation parameters for each top group |
| IB+Buffer+MaxDist Summary | Top 25 groups with stricter grouping |
| IB+Buffer+MaxDist Details | Per-variation parameters for strict groups |

**Run**:
```bash
cd dual_v3/analyze
python analyze_optimization_db.py
```

---

### Best Params Files

**`best_params_GER40_V8.py`** and **`best_params_XAUUSD_V8.py`** are generated output files containing the selected parameter dictionaries ready to be copied into `strategy_logic.py` for live trading.

**GER40 V8 example** (IB 08:00-08:30, Europe/Berlin, Wait 15m, Buffer 20%):

| Variation | Total R | Sharpe | Win Rate | Trades | Max DD |
|-----------|---------|--------|----------|--------|--------|
| REV_RB | 35.46 | 7.16 | 61.0% | 41 | 4.0 |
| Reverse | 54.04 | 2.43 | 26.2% | 160 | 11.5 |
| TCWE | 16.77 | 0.79 | 28.5% | 242 | 11.8 |
| OCAE | 47.99 | 1.79 | 41.4% | 297 | 12.0 |
| **Combined** | **154.26** | **1.90 (weighted)** | -- | **740** | -- |

**XAUUSD V8 example** (IB 09:00-09:30, Asia/Tokyo, Wait 20m, Buffer 5%):

| Variation | Total R | Sharpe | Win Rate | Trades | Max DD |
|-----------|---------|--------|----------|--------|--------|
| REV_RB | 0.61 | 0.44 | 42.9% | 20 | 2.7 |
| Reverse | 44.06 | 3.51 | 47.6% | 98 | 11.5 |
| TCWE | 25.10 | 1.20 | 68.3% | 344 | 9.3 |
| OCAE | 37.64 | 2.56 | 55.6% | 225 | 5.2 |
| **Combined** | **107.41** | **1.95 (weighted)** | -- | **687** | -- |

These files are not meant to be run as scripts. They define Python dictionaries that are imported or copied into the strategy configuration.

---

### generate_backtest_groups.py

**Purpose**: Generate a comprehensive set of unique parameter groups for slow-engine backtest validation. Uses multiple grouping modes and ranking strategies to avoid selection bias.

**Input**: SQLite databases (`GER40_optimization.db`, `XAUUSD_optimization.db`)

**Processing**:
1. Applies 3 grouping modes:
   - **IB Only**: Group by (ib_start, ib_end, ib_tz, ib_wait)
   - **IB + Buffer**: Group by (ib_start, ib_end, ib_tz, ib_wait, ib_buffer_pct)
   - **IB + Buffer + MaxDist**: Group by (ib_start, ib_end, ib_tz, ib_wait, ib_buffer_pct, max_distance_pct)

2. Applies 6 ranking strategies to each grouping mode:

| Strategy | Formula |
|----------|---------|
| total_r | Rank by combined Total R (highest first) |
| sharpe_weighted | 50% normalized R + 50% normalized Sharpe |
| calmar | Total R / (Max DD + 1) |
| multi_criteria | 35% R + 35% Sharpe + 15% DD penalty + 15% trades |
| winrate_focus | 50% winrate + 30% R + 20% Sharpe |
| r_sharpe_70_30 | 70% R + 30% Sharpe |

3. Takes top 40 groups per category (3 modes x 6 strategies = 18 categories x 40 = 720 candidates)
4. Deduplicates across categories by full parameter signature
5. Converts each unique group to backtest-ready format with variation-specific parameters

**Output**: `backtest_groups_{SYMBOL}.json`

Each group entry contains:
- Unique ID (e.g., `GER40_001`)
- Source category (which grouping+ranking produced it)
- Combined metrics (total_r, sharpe, max_dd, trades)
- Per-variation parameters in strategy-ready format (IB_START, IB_END, IB_TZ, IB_WAIT, TRADE_WINDOW, RR_TARGET, STOP_MODE, etc.)
- Expected metrics per variation (for comparison with slow backtest results)

**Run**:
```bash
cd dual_v3/analyze
python generate_backtest_groups.py
```

**Typical output**: ~100-200 unique groups per symbol after deduplication.

---

### create_combined_charts.py

**Purpose**: Create combined equity/drawdown charts for GER40 + XAUUSD portfolio combinations from the main backtest period (Jan 2023 - Oct 2025).

**Input**:
- Group configs from `parallel_results/{SYMBOL}/{GROUP_ID}/config.json`
- Trade results from slow backtests (or re-runs them if trades.csv does not exist)

**Processing**:
1. Selects specific top groups per symbol (e.g., GER40 groups [1, 5, 7, 15, 22, 36, 48] and XAUUSD groups [1, 12, 28])
2. Runs backtests for each group using `backtest_single_group.run_single_group_backtest()` to obtain trade logs
3. For each GER40-XAUUSD pair combination:
   - Merges trades from both symbols sorted by exit time
   - Calculates combined cumulative R (equity curve)
   - Calculates peak-to-trough drawdown in R and percentage
   - Generates a two-panel chart: equity curve (top) + drawdown (bottom)

**Output**:
- `parallel_results/charts_combined/GER40_{N}_XAUUSD_{M}.png` -- Chart per combination
- `parallel_results/charts_combined/combined_summary.csv` -- Summary table
- `parallel_results/charts_combined/combined_summary.xlsx` -- Excel summary

**Run**:
```bash
cd dual_v3/analyze
python create_combined_charts.py
```

---

### create_combined_charts_control.py

**Purpose**: Same as `create_combined_charts.py` but for the **control period** (November 2025 - January 2026). Uses existing trades.csv files from `parallel_results_control/`.

**Key differences from the main script**:
- Reads pre-computed trades from control period VM results (vm11-vm14 folders)
- Does not re-run backtests
- Uses different top groups based on control period performance
- Calculates Sharpe-like metric (annualized return/stdev ratio)
- Output filenames prefixed with `control_`

**Output**:
- `parallel_results_control/charts_combined/control_GER40_{N}_XAUUSD_{M}.png`
- `parallel_results_control/charts_combined/combined_summary_control.csv`
- `parallel_results_control/charts_combined/combined_summary_control.xlsx`

**Run**:
```bash
cd dual_v3/analyze
python create_combined_charts_control.py
```

---

### day_of_week_analysis.py

**Purpose**: Analyze trading performance broken down by day of the week (Monday through Friday). Compares TSL (main) and Control periods side by side.

**Input**: Trade CSV files from two dataset periods:
- **TSL**: `parallel_results_tsl/vm*/` directories
- **Control**: `parallel_results_control/vm*/` directories

**Groups analyzed**:
- Group A: GER40_055 + XAUUSD_059
- Group B: GER40_006 + XAUUSD_010

**Metrics computed per day**:

| Metric | Description |
|--------|-------------|
| trades | Total trade count |
| wins / losses | Win and loss counts |
| winrate% | Win percentage |
| avg_r | Average R per trade |
| total_r | Sum of R |
| avg_profit | Average profit in currency |
| avg_win_r / avg_loss_r | Average R for winners and losers |
| profit_factor | Sum of winning R / sum of losing R |

**Output**: Console tables with per-day breakdowns and TSL vs Control comparison showing deltas in winrate and average R.

**Run**:
```bash
cd dual_v3/analyze
python day_of_week_analysis.py
```

---

## Parameter Selection Criteria

The analysis scripts apply several filters and selection criteria:

### Hard Filters

| Filter | Threshold | Applied In |
|--------|-----------|------------|
| Max drawdown | <= 12 R per variation | analyze_optimization_db.py, generate_backtest_groups.py |
| Min trades | >= 10 (ranking only) | OptimizerMaster._finalize_results() |
| TSL constraint | tsl_target == 0 OR tsl_sl < rr_target + 1 | analyze_optimization_db.py |

### Ranking Metrics

Results are ranked by combined Total R across all four variations (OCAE + TCWE + Reverse + REV_RB). Within each group, the best parameter set per variation is selected by highest total_r that passes the drawdown filter.

### Weighted Sharpe

The weighted average Sharpe ratio is computed as:

```
weighted_sharpe = sum(sharpe_i * trades_i) / sum(trades_i)
```

This weights each variation's Sharpe by its trade count, giving more influence to variations that produce more trades.

### Grouping Modes

Two primary grouping modes ensure common IB parameters across variations:

1. **IB + Buffer**: All four variations share the same IB window, wait time, and buffer percentage. MaxDistance and other params may differ per variation.

2. **IB + Buffer + MaxDist** (Strict): All four variations share IB window, wait time, buffer, AND max distance percentage. More constrained but ensures consistent risk parameters.

---

## Chart Generation

### Combined Equity Charts

Each chart contains two panels:

**Top panel (Equity Curve)**:
- Blue line: cumulative R over time
- Shaded area under curve
- Gray dashed line at y=0
- Title: group IDs, total R, trade counts

**Bottom panel (Drawdown)**:
- Red fill: drawdown from peak in R
- Legend: maximum drawdown in R and percentage

Charts are saved at 150 DPI as PNG files.

### Control Period Charts

Identical format to main charts but cover the out-of-sample control period (Nov 2025 - Jan 2026). Also include Sharpe ratio in the title.

---

## Backtest Group Generation

The group generation process is designed to produce a diverse set of candidate parameter sets for validation, avoiding over-fitting to a single ranking metric.

### Deduplication

Groups are deduplicated across all 18 categories (3 grouping modes x 6 ranking strategies) by comparing the full parameter signature of each variation. Two groups are considered identical if all variation parameters match exactly. Only the first occurrence is kept.

### Output Format

Each group in `backtest_groups_{SYMBOL}.json` has this structure:

```json
{
    "id": "GER40_001",
    "symbol": "GER40",
    "source_category": "ib_buffer_total_r",
    "rank_in_category": 1,
    "combined_total_r": 154.26,
    "weighted_sharpe": 1.90,
    "max_group_dd": 12.0,
    "total_trades": 740,
    "params": {
        "OCAE": {
            "IB_START": "08:00",
            "IB_END": "08:30",
            "IB_TZ": "Europe/Berlin",
            "IB_WAIT": 15,
            "TRADE_WINDOW": 210,
            "RR_TARGET": 1.0,
            ...
            "_expected": {
                "total_r": 47.99,
                "sharpe": 1.79,
                "winrate": 41.4,
                "trades": 297,
                "max_dd": 12.0
            }
        },
        "TCWE": { ... },
        "Reverse": { ... },
        "REV_RB": { ... }
    }
}
```

The `_expected` sub-dict for each variation stores the fast-engine metrics so they can be compared with slow-engine validation results.

---

## Day-of-Week Analysis

### Purpose

Identifies whether certain days of the week consistently underperform. If a day shows negative expected R across both TSL and Control periods, it may be worth excluding from live trading via a day-of-week filter.

### Comparison Table

The script outputs a side-by-side comparison:

```
Day  | TSL WR%   TSL avgR   TSL totR | CTL WR%   CTL avgR   CTL totR | dWR%   dAvgR
Mon  |  52.30%    0.0812     14.62   |  48.00%    0.0543      5.43   | -4.30  -0.0269
Tue  |  ...
```

- `dWR%`: Control winrate minus TSL winrate (positive = control outperforms)
- `dAvgR`: Control average R minus TSL average R

---

## Directory Structure

```
analyze/
    analyze_optimization.py          # v2 analyzer (Parquet input)
    analyze_optimization_db.py       # v3 analyzer (SQLite input) -- primary
    generate_backtest_groups.py      # Group generator for validation
    create_combined_charts.py        # Portfolio equity charts (main period)
    create_combined_charts_control.py  # Portfolio equity charts (control period)
    day_of_week_analysis.py          # Day-of-week performance analysis
    best_params_GER40_V8.py          # Generated: GER40 best params
    best_params_XAUUSD_V8.py         # Generated: XAUUSD best params
    GER40_optimization.db            # Input: SQLite optimization results
    XAUUSD_optimization.db           # Input: SQLite optimization results
    results_vm1_ger40/               # Input: Parquet from VM run 1
    results_vm2_ger40/               # Input: Parquet from VM run 2
    parallel_results/                # Output: slow backtest results
        GER40/
            GER40_001/
                config.json
                trades.csv
            ...
        XAUUSD/
            ...
        charts_combined/
            GER40_001_XAUUSD_001.png
            combined_summary.csv
    parallel_results_control/        # Output: control period results
        vm11_GER40/
        vm12_GER40/
        vm13_XAUUSD/
        vm14_XAUUSD/
        charts_combined/
            control_GER40_037_XAUUSD_010.png
            combined_summary_control.csv
    parallel_results_tsl/            # Output: TSL period results
        vm11_GER40/
        vm13_XAUUSD/
```

---

## End-to-End Workflow

### 1. Run Optimization

See [PARAMETER_OPTIMIZER.md](PARAMETER_OPTIMIZER.md) for full instructions.

```bash
cd dual_v3
python -m params_optimizer.run_optimizer --symbol GER40 --workers 90
```

### 2. Consolidate Results into SQLite

Transfer optimization output Parquet files and convert to SQLite databases in the `analyze/` directory.

### 3. Run Analysis

```bash
cd dual_v3/analyze
python analyze_optimization_db.py
```

Review the generated Excel files and best_params_*.py files.

### 4. Generate Backtest Groups

```bash
python generate_backtest_groups.py
```

Review `backtest_groups_GER40.json` and `backtest_groups_XAUUSD.json`.

### 5. Run Slow Validation Backtests

Use the parallel backtest runner to validate groups with the full IBStrategy engine. This produces per-group trade logs in `parallel_results/`.

### 6. Generate Combined Charts

```bash
python create_combined_charts.py
```

Review equity curves and drawdown charts in `parallel_results/charts_combined/`.

### 7. Run Control Period Validation

Run backtests on the control period (out-of-sample data) and generate charts:

```bash
python create_combined_charts_control.py
```

### 8. Day-of-Week Analysis

```bash
python day_of_week_analysis.py
```

Review per-day performance to identify potential day filters for live trading.

### 9. Select Final Parameters

Based on all analysis results, select the parameter set for live deployment. Copy the appropriate `best_params_*_V8.py` dictionary into the strategy configuration.
