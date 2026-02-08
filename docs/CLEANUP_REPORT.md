# Cleanup Report - dual_v3

**Date:** 2026-02-08

This document summarizes the cleanup performed on the dual_v3 development workspace.

---

## 1. Files Moved to _trash/

### 1.1 Temporary Files (7 files)
- `tmpclaude-1a4a-cwd`, `tmpclaude-6530-cwd`, `tmpclaude-b3ba-cwd`, `tmpclaude-c837-cwd`, `tmpclaude-eb39-cwd`, `tmpclaude-f91a-cwd` (root)
- `backtest/tmpclaude-fb96-cwd`

**Rationale:** Auto-generated Claude Code temp files, no value.

### 1.2 Bug Reports (2 files)
- `bug SL 10.01.2026.txt` (35KB)
- `bug №2 SL 10.01.2026.txt` (9KB)

**Rationale:** Historical bug investigation notes. The bugs were fixed (see CHANGELOG.md). Content preserved in _trash for reference.

### 1.3 Outdated Documentation (2 files + 1 directory)
- `docs/BOT_ARCHITECTURE.md` -- Replaced by `docs/LIVE_TRADING_BOT.md`
- `docs/ROADMAP.md` -- Outdated (news filter is now implemented, items completed or obsolete)
- `docs/claude code/` (4 PDFs) -- Internal Claude Code usage docs, unrelated to the trading bot project

### 1.4 Legacy Directories (3 directories)
- `project_info/` -- Old research scripts and reference materials from early development
- `output/` -- Legacy backtest output from parameter finder (3 subdirectories, superseded by backtest/output/)
- `_temp/` -- Temporary calculation scripts (calc_grid.py and related)

### 1.5 Duplicate Backtest Scripts (10 files)
Kept: `run_backtest.py` (original), `run_backtest_v8_ger40.py`, `run_backtest_v8_xauusd.py` (latest), `run_backtest_prod_*.py` (production validation)

Moved to _trash:
- `backtest/run_backtest_v2.py`
- `backtest/run_backtest_v3.py`
- `backtest/run_backtest_v3_xauusd.py`
- `backtest/run_backtest_v4.py`
- `backtest/run_backtest_v5.py`
- `backtest/run_backtest_v6_ger40.py`
- `backtest/run_backtest_v6_xauusd.py`
- `backtest/run_backtest_v7_ger40.py`
- `backtest/run_backtest_v7_xauusd.py`
- `backtest/run_backtest_xauusd.py`

**Rationale:** Near-identical scripts differing only in parameter dicts. A parameterized template (`run_backtest_template.py`) was created to replace the copy-paste pattern.

### 1.6 Old Analysis Scripts (6 files)
- `analyze/best_params_GER40.py`, `analyze/best_params_XAUUSD.py` (original, superseded by V8)
- `analyze/best_params_GER40_V7.py`, `analyze/best_params_GER40_V7_STRICT.py`
- `analyze/best_params_XAUUSD_V7.py`, `analyze/best_params_XAUUSD_V7_STRICT.py`

Kept: `best_params_*_V8*.py` (latest analysis)

### 1.7 Old Optimization Excel Files (4 files)
- `analyze/optimization_analysis_GER40.xlsx`, `analyze/optimization_analysis_GER40_v2.xlsx`
- `analyze/optimization_analysis_XAUUSD.xlsx`, `analyze/optimization_analysis_XAUUSD_v2.xlsx`

Kept: `optimization_analysis_*_v3.xlsx` (latest analysis)

### 1.8 Duplicate Optimizer Output (2 directories)
- `params_optimizer/output/GER40_AWS_129600 — копия/`
- `params_optimizer/output/XAUUSD_AWS_129600 — копия/`

**Rationale:** Copy ("копия") directories of optimization results. Originals preserved.

### 1.9 Old Optimizer Test Scripts (3 files)
- `params_optimizer/test_quick.py`
- `params_optimizer/test_all_sets.py`
- `params_optimizer/profile_backtest.py`

**Rationale:** Development/debugging scripts used during optimizer creation. No longer needed.

### 1.10 Miscellaneous (3 items)
- `notebooks/.ipynb_checkpoints/` -- Jupyter auto-generated checkpoint directory
- `analyze/nul` -- Windows "nul" artifact (0-byte file)
- `analyze/all_charts/2025_only — копия/` -- Duplicate charts directory

---

## 2. Dead Code Removed from strategy_logic.py

**File:** `src/utils/strategy_logic.py`
**Before:** 1759 lines
**After:** 996 lines
**Removed:** 763 lines

### 2.1 Legacy Parameter Dictionaries (~464 lines)
- `GER40_PARAMS` -- Original V1 parameters
- `GER40_PARAMS_V2` -- Optimized from params_finder
- `GER40_PARAMS_V3` -- AWS optimization (129,600 combinations)
- `GER40_PARAMS_V4` -- Further grid search
- `GER40_PARAMS_V5` -- REV_RB enabled variant
- `XAUUSD_PARAMS_V3` -- AWS optimization for Gold
- `XAU_PARAMS` -- Original Gold parameters

All superseded by `GER40_PARAMS_PROD` and `XAUUSD_PARAMS_PROD` (V9, current production).

### 2.2 Dead Functions (~178 lines)
- `load_all_data()` -- CSV batch loader, only used by run_dual_strategy()
- `format_excel_workbook()` -- Excel formatting, only used by run_dual_strategy()
- `run_dual_strategy()` -- Standalone batch backtester with hardcoded paths to nonexistent directories
- `if __name__ == "__main__"` block

### 2.3 Reference Updates
- `backtest/run_backtest.py`: Updated import from `GER40_PARAMS` to `GER40_PARAMS_PROD`
- `notebooks/debug_tools.py`: Updated import from `GER40_PARAMS_V3`/`XAUUSD_PARAMS_V3` to PROD
- Comments in `config.py` and `src/config.py`: Updated references
- Docstring in `ib_strategy.py`: Updated parameter names

### 2.4 Preserved Code
All removed code saved to `_trash/legacy_params_*.py` for reference.

---

## 3. Cache Cleanup (Deleted)

- 25 `__pycache__/` directories removed recursively
- `.pytest_cache/` directory removed

These are auto-regenerated by Python and should not be in version control.

---

## 4. New Files Created

### 4.1 Documentation (10 files in docs/)
| File | Size | Description |
|------|------|-------------|
| TRADING_STRATEGY.md | 36 KB | Manual trading guide for human traders |
| LIVE_TRADING_BOT.md | 30 KB | Bot architecture and operation |
| SLOW_BACKTEST_ENGINE.md | 26 KB | MT5 emulator-based backtester |
| FAST_BACKTEST_ENGINE.md | 24 KB | Vectorized NumPy backtest engine |
| PARAMETER_OPTIMIZER.md | 22 KB | Grid search optimization framework |
| RESULTS_ANALYSIS.md | 19 KB | Analysis pipeline and tools |
| STRATEGY_OPTIMIZATION_TOOLS.md | 17 KB | Volatility, fractal, candle size analysis |
| NEWS_FILTER.md | 16 KB | ForexFactory economic calendar integration |
| DEPLOYMENT_INFRASTRUCTURE.md | 14 KB | VM setup and deployment scripts |
| DATA_SOURCES.md | 14 KB | Market data formats and sources |
| CLEANUP_REPORT.md | -- | This file |

### 4.2 Backtest Template
- `backtest/run_backtest_template.py` (28 KB) -- Parameterized CLI script replacing the need to copy-paste backtest scripts for new parameter sets

### 4.3 Updated Files
- `PROJECT_STATE.md` -- Updated to reflect production status
- `CLEANUP_TODO.md` -- Replaced with reference to this report
- `.gitignore` -- Added entries for *.db, *.parquet, tmpclaude-*, nul

---

## 5. Files Preserved (Not Touched)

- All `src/` code (production bot)
- `backtest/output/` (all historical backtest results)
- `backtest/run_backtest.py`, `run_backtest_v8_*.py`, `run_backtest_prod_*.py`
- `backtest/run_parallel_backtest*.py`, `backtest_single_group*.py`
- `backtest/` infrastructure (emulator/, data_processor/, analysis/, reporting/)
- `vm_optimizer_key`, `vm_optimizer_key.pub` (SSH keys)
- `analyze/parallel_results*/` (optimization results from VMs)
- `analyze/results_vm*/` (VM-specific results)
- `analyze/GER40_optimization.db`, `XAUUSD_optimization.db` (optimization databases, gitignored)
- `analyze/best_params_*_V8*.py` (latest analysis)
- `analyze/optimization_analysis_*_v3.xlsx` (latest Excel analysis)
- `data/` (all market data, gitignored)
- `strategy_optimization/` (all analysis tools and results)
- `params_optimizer/` (engine, orchestrator, data, analytics, benchmark, reports)
- `CHANGELOG.md` (62KB comprehensive history)
- `docs/strategies/`, `docs/troubleshooting/`, `docs/operations/`, `docs/testing/`, `docs/configuration/`
