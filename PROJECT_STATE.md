# Project State - IB Trading Bot (dual_v4)

**Last Updated:** 2026-02-09

## Purpose

Strategy evolution and backtesting workspace. Based on the production-validated dual_v3 codebase.

**NOT a live trading bot.** For live trading, see `dual_v3/` and the production deployment at `C:\Trading\5ers_ger_xau_trading_bot`.

## Research Directions

Ideas to explore in this workspace:
- 1H-4H fractal levels for TP target placement
- Session volatility analysis for realistic TP assessment
- BOS (Break of Structure) and CISD stop loss variations
- ATR-based dynamic SL/TP sizing
- Multi-timeframe confirmation signals

## Baseline Strategy

Starting from dual_v3 PROD parameters (V9):
- **GER40**: IB 08:00-09:00 Europe/Berlin
- **XAUUSD**: IB 09:00-09:30 Asia/Tokyo
- 4 variations: Reverse > OCAE > TCWE > REV_RB (priority order)

Parameters defined in `src/utils/strategy_logic.py` as `GER40_PARAMS_PROD` and `XAUUSD_PARAMS_PROD`.

## Infrastructure

### Backtesting
- **Slow engine**: `backtest/` (MT5 emulator, runs actual IBStrategy)
- **Fast engine**: `params_optimizer/engine/` (vectorized NumPy, ~100x faster)
- **Day debugger**: `notebooks/debug_strategy.ipynb` (point-specific day analysis)

### VMs for Parallel Backtesting
| VM | IP | Typical Use |
|----|-----|-------------|
| VM11 | 10.10.32.11 | GER40 backtests |
| VM12 | 10.10.32.12 | GER40 backtests |
| VM13 | 10.10.32.13 | XAUUSD backtests |
| VM14 | 10.10.32.14 | XAUUSD backtests |

SSH key: `vm_optimizer_key`
Deploy script: `deploy_parallel_backtest.sh`

### Data
- Raw M1 CSVs: `data/GER40 1m .../`, `data/XAUUSD 1m .../`
- Extended data: `data/ger40+pepperstone_0411-2001/`, `data/xauusd_oanda_0411-2001/`
- Optimized parquet: `data/optimized/`
- News data: `data/news/` (ForexFactory JSON)

## File Structure

```
dual_v4/
├── config.py            # Configuration (no credentials)
├── PROJECT_STATE.md     # This file
├── CHANGELOG.md         # Change history
│
├── src/                 # Strategy source code
│   ├── config.py        # Timezone constants, symbol configs
│   ├── strategies/      # IBStrategy FSM, Signal dataclass
│   ├── utils/           # strategy_logic.py (CORE), time_utils.py
│   └── news_filter/     # ForexFactory integration
│
├── backtest/            # Slow backtest engine (MT5 emulator)
├── params_optimizer/    # Fast backtest + grid search optimizer
├── analyze/             # Results analysis and visualization
├── strategy_optimization/  # Fractals, volatility, candle size analysis
├── notebooks/           # Debug notebooks (day-by-day testing)
├── scripts/             # News data loading utilities
├── data/                # Market data
├── docs/                # Documentation (15+ files)
├── deploy_parallel_backtest.sh
├── vm_optimizer_key     # SSH key for VMs
└── _trash/              # Archived files (gitignored)
```

## Documentation

See `docs/` for comprehensive documentation inherited from dual_v3:
- `TRADING_STRATEGY.md` - Manual trading guide
- `LIVE_TRADING_BOT.md` - Bot architecture reference
- `SLOW_BACKTEST_ENGINE.md` - MT5 emulator backtester
- `FAST_BACKTEST_ENGINE.md` - Vectorized backtest engine
- `PARAMETER_OPTIMIZER.md` - Grid search optimization
- `RESULTS_ANALYSIS.md` - Analysis pipeline
- `STRATEGY_OPTIMIZATION_TOOLS.md` - Volatility, fractal analysis
- `NEWS_FILTER.md` - Economic calendar integration
- `DATA_SOURCES.md` - Market data formats
- `DEPLOYMENT_INFRASTRUCTURE.md` - VM setup and deployment
