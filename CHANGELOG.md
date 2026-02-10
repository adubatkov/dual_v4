# Changelog - dual_v4

## 2026-02-10

### SMC Architecture Planning
- Created `plans/` directory with 6 architecture documents (7955 lines total)
- `SMC_ARCHITECTURE.md` - Master plan: 3-layer architecture (Detectors -> Registry -> Engine)
- `PHASE_1_FOUNDATION.md` - Models, config, timeframe manager, registry, event log, fractal+FVG detectors
- `PHASE_2_ENGINE.md` - CISD detector, market structure detector (BOS/CHoCH/MSS), SMCEngine
- `PHASE_3_INTEGRATION.md` - IBStrategySMC wrapper, debug_smc.ipynb, chart overlays
- `PHASE_4_FAST_BACKTEST.md` - Batch SMC pre-computation for fast backtest engine
- `PHASE_5_VALIDATION.md` - FVG/block/entry/SL/TP/BE validation rules from InValidation.md
- Source material: 64 SMC concept files from notion_export/

## 2026-02-09

### Initial Setup
- Created dual_v4 workspace from dual_v3 production codebase
- Copied strategy core, both backtest engines, optimizer, analysis tools
- Copied research tools: fractals, volatility, candle size analysis
- Copied debug notebook for day-by-day strategy testing
- Copied all documentation (15+ files)
- Copied market data (M1 CSVs, parquet, news JSON)
- Copied VM infrastructure (deploy script, SSH keys)
- Stripped MT5 credentials from config files (R&D workspace, no live trading)
