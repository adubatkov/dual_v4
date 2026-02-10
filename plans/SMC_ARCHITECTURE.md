# SMC (Smart Money Concepts) Architecture for dual_v4

## 1. Context

dual_v4 -- Strategy R&D workspace for IB trading bot.

**Current state**: IB forms -> signal fires (OCAE/Reverse/TCWE/REV_RB) -> trade opens immediately -> closes by SL/TP/TSL/time. No awareness of market structure context.

**Desired state**: Bot evaluates SMC context (FVG, BOS, CISD, fractals, liquidity) before entry. Can delay entry for confirmation. Can modify SL/TP based on active structures. Works in slow backtest, fast backtest, AND live bot.

**Source material**: 64 SMC concept files in `C:\Trading\ib_trading_bot\notion_export\`

---

## 2. Architecture Overview

Three-layer design: **Detectors** (pure functions) -> **Registry** (state tracking) -> **Engine** (decision logic).

```
                    +---------------------+
                    |   TimeframeManager  |  M1 -> M2/M5/M15/M30/H1/H4
                    +----------+----------+
                               |
                    +----------v----------+
                    |     Detectors       |  Pure functions: detect_fvg(), detect_cisd(), ...
                    +----------+----------+
                               |
                    +----------v----------+
                    |    SMCRegistry      |  Tracks active/invalidated structures
                    +----------+----------+
                               |
                    +----------v----------+
                    |     SMCEngine       |  evaluate_signal() -> ENTER/WAIT/REJECT
                    +----------+----------+
                               |
              +----------------+----------------+
              |                                 |
    +---------v---------+            +----------v---------+
    |  IBStrategySMC    |            |  FastBacktest      |
    |  (slow backtest   |            |  (batch SMC        |
    |   + live bot)     |            |   pre-computation) |
    +-------------------+            +--------------------+
```

**Layer 1 - Detectors**: Pure functions. Input: DataFrame + params. Output: list of detected structures. No side effects, no state. Fully testable with synthetic data.

**Layer 2 - Registry**: In-memory container for all active/invalidated structures. Organized by instrument -> timeframe -> type. Manages lifecycle: formation -> activation -> invalidation. Provides query API (e.g., "all active FVGs above price X on H1").

**Layer 3 - Engine**: High-level orchestrator. Calls detectors on each timeframe update, stores results in registry, evaluates signals against SMC context, manages confirmation logic.

---

## 3. File Structure

```
dual_v4/src/smc/                          # SMC engine
|-- __init__.py
|-- models.py                             # Dataclasses: FVG, CISD, BOS, Fractal, SMCDecision...
|-- config.py                             # SMCConfig dataclass (per-instrument params)
|-- registry.py                           # SMCRegistry: tracks active/invalidated structures
|-- event_log.py                          # SMCEventLog: chronological audit trail
|-- engine.py                             # SMCEngine: evaluate_signal(), check_confirmation()
|-- timeframe_manager.py                  # M1 -> all TFs resampling + caching
+-- detectors/
    |-- __init__.py
    |-- fractal_detector.py               # Port from strategy_optimization/fractals/fractals.py
    |-- fvg_detector.py                   # 3-candle FVG detection
    |-- cisd_detector.py                  # Change In State of Delivery
    +-- market_structure_detector.py      # HH/HL/LL/LH + BOS/CHoCH/MSS

dual_v4/src/strategies/
|-- ib_strategy.py                        # UNCHANGED (production reference)
+-- ib_strategy_smc.py                    # NEW: extends IBStrategy with SMC overlay

dual_v4/notebooks/
|-- debug_smc.ipynb                       # NEW: SMC-aware debugging notebook
+-- debug_tools.py                        # MODIFIED: add generate_chart_with_smc()

dual_v4/strategy_optimization/smc_tools/  # Standalone SMC analysis
|-- smc_analyzer.py                       # Batch analysis of SMC structures on historical data
+-- smc_visualizer.py                     # Chart overlays for SMC structures

dual_v4/tests/test_smc/                   # Test suite
|-- test_detectors.py
|-- test_registry.py
|-- test_engine.py
+-- test_integration.py
```

---

## 4. Data Models

### 4.1 SMC Structures

```python
@dataclass
class FVG:
    id: str                               # Unique identifier
    instrument: str                       # "GER40" / "XAUUSD"
    timeframe: str                        # "M2", "M5", "H1", etc.
    direction: str                        # "bullish" / "bearish"
    high: float                           # Upper boundary of gap
    low: float                            # Lower boundary of gap
    formation_time: datetime              # When the 3rd candle closed
    formation_candles: Tuple[int,int,int]  # Indices of 3 candles
    status: str = "active"                # active/partial_fill/full_fill/invalidated/inverted
    fill_pct: float = 0.0                 # How much has been filled (0-1)
    invalidation_time: Optional[datetime] = None
    invalidation_reason: Optional[str] = None

@dataclass
class Fractal:
    id: str
    instrument: str
    timeframe: str                        # Usually "H1" or "H4"
    type: str                             # "high" / "low"
    price: float                          # Fractal level price
    time: datetime                        # Center candle time
    confirmed_time: datetime              # When 3rd candle closed (confirmed)
    swept: bool = False                   # Has price swept this level
    sweep_time: Optional[datetime] = None

@dataclass
class CISD:
    id: str
    instrument: str
    timeframe: str
    direction: str                        # "long" / "short"
    delivery_candle_open: float           # Delivery candle that led to POI
    delivery_candle_close: float
    delivery_candle_body_high: float      # max(open, close) of delivery candle
    delivery_candle_body_low: float       # min(open, close) of delivery candle
    confirmation_time: datetime           # When the confirming candle closed
    confirmation_price: float             # Close price of confirming candle

@dataclass
class BOS:
    id: str
    instrument: str
    timeframe: str
    direction: str                        # "bullish" / "bearish"
    broken_level: float                   # The structure level that was broken
    break_time: datetime                  # When the break candle closed
    break_candle_close: float
    bos_type: str                         # "bos" / "choch" / "mss"
    displacement: bool = False            # Was there displacement (aggressive move)?

@dataclass
class MarketStructure:
    instrument: str
    timeframe: str
    swing_points: List[StructurePoint]    # Ordered list of HH/HL/LL/LH
    current_trend: str                    # "uptrend" / "downtrend" / "ranging"
    last_update: datetime

@dataclass
class StructurePoint:
    time: datetime
    price: float
    type: str                             # "HH" / "HL" / "LL" / "LH"
    is_key: bool = False                  # Key High/Low (builds structure)
```

### 4.2 Decision Objects

```python
@dataclass
class SMCDecision:
    action: str                           # "ENTER" / "WAIT" / "REJECT"
    reason: str                           # Human-readable explanation
    modified_signal: Optional[Signal] = None  # Modified SL/TP/entry if ENTER
    confirmation_criteria: List[ConfirmationCriteria] = field(default_factory=list)
    confluence_score: float = 0.0
    timeout_minutes: int = 30             # Max wait time for WAIT action

@dataclass
class ConfirmationCriteria:
    type: str                             # "CISD" / "FVG_REBALANCE" / "BOS"
    timeframe: str                        # Which TF to check
    direction: str                        # Required direction
    params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ModifiedSignal:
    """Signal with SMC modifications applied."""
    original_signal: Signal               # Original IB signal
    entry_price: float                    # New entry price
    stop_loss: float                      # New SL (e.g., candle low)
    take_profit: float                    # New TP
    modification_reason: str              # What SMC changed and why
    confirming_structure_id: str          # ID of the structure that confirmed

@dataclass
class ConfluenceScore:
    total: float                          # Weighted sum
    breakdown: Dict[str, float]           # {"fractal_support": +1.5, "fvg_gap": +1.0, ...}
    direction_bias: str                   # "long" / "short" / "neutral"
    active_structures: List[str]          # IDs of contributing structures
```

### 4.3 Event Log

```python
@dataclass
class SMCEvent:
    timestamp: datetime
    instrument: str
    event_type: str                       # See event type list below
    timeframe: str
    direction: Optional[str] = None       # "long"/"short" if directional
    price: Optional[float] = None
    structure_id: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    bias_impact: float = 0.0             # +N for long bias, -N for short bias
```

Event types:
- Structure formation: `FVG_FORMED`, `CISD_DETECTED`, `BOS_DETECTED`, `FRACTAL_CONFIRMED`
- Structure lifecycle: `FVG_PARTIAL_FILL`, `FVG_FULL_FILL`, `FVG_INVALIDATED`, `FRACTAL_SWEPT`
- Signal evaluation: `SIGNAL_EVALUATED`, `CONFLUENCE_SCORED`, `CONFIRMATION_STARTED`, `CONFIRMATION_MET`
- Trade actions: `SIGNAL_REJECTED`, `SIGNAL_MODIFIED`, `ENTRY_EXECUTED`

---

## 5. Modified Signal Flow

### Current Flow
```
IB signal -> Enter immediately -> Manage by SL/TP/TSL/time
```

### New Flow
```
IB signal -> SMCEngine.evaluate_signal() -> ENTER / WAIT / REJECT
  |
  |-- ENTER: proceed (possibly with modified SL/TP/entry)
  |-- WAIT:  transition to AWAITING_CONFIRMATION, check criteria each bar
  |-- REJECT: try next signal variation
```

### FSM State Diagram

IBStrategySMC adds one new state to the existing FSM:

```
AWAITING_IB_CALCULATION
    |
    v
AWAITING_TRADE_WINDOW
    |
    v
IN_TRADE_WINDOW
    | (signal detected)
    |-- SMC=ENTER  --> POSITION_OPEN (with possibly modified SL/TP/entry)
    |-- SMC=WAIT   --> AWAITING_CONFIRMATION  <-- NEW STATE
    |-- SMC=REJECT --> try next variation
    |
AWAITING_CONFIRMATION
    |-- criteria met   --> POSITION_OPEN (entry from confirming candle)
    |-- timeout        --> back to IN_TRADE_WINDOW (scan for new signals)
    |
POSITION_OPEN / DAY_ENDED
```

### Example: User's Scenario

1. 10:15 -- OCAE long signal fires (price closed above IB High after EQ touch)
2. SMCEngine.evaluate_signal() runs:
   - Detects unswept 1h high fractal at 22480 (price just closed above it at 22482)
   - Fractal proximity = potential reaction zone
   - Decision: WAIT
   - Criteria: [CISD long on M2, FVG_REBALANCE bullish on M2]
   - Timeout: 30 min
3. State -> AWAITING_CONFIRMATION
4. 10:18 -- M2 FVG forms at 22470-22475 (bearish pullback creates gap)
5. 10:22 -- Price returns to FVG, partially fills it (low touches 22472), closes at 22478 (above FVG)
6. SMCEngine.check_confirmation(): FVG_REBALANCE criteria met
7. ModifiedSignal: entry=22478 (close of confirming candle), SL=22470 (low of confirming candle)
8. State -> POSITION_OPEN with modified parameters

---

## 6. Backtest Integration

### 6.1 Slow Backtest (mt5_emulator.py)

IBStrategySMC replaces IBStrategy in the backtest adapter. No changes to emulator itself.

```python
# In run_backtest_template.py or equivalent:
strategy = IBStrategySMC(
    symbol="GER40",
    params=GER40_PARAMS_PROD,
    executor=backtest_executor,
    magic_number=1001,
    smc_config=SMCConfig(instrument="GER40")
)
```

- SMCEngine.update() called every M1 bar (incremental)
- TimeframeManager resamples M1 -> M2/M5/H1 on the fly
- Registry accumulates structures throughout the day
- Performance impact: <10% slower (detector runs are fast, data is small)

### 6.2 Fast Backtest (fast_backtest.py)

Batch mode: pre-compute all SMC structures ONCE per day, then query during signal evaluation.

```python
def _process_day(self, day_df, day_date, params):
    # STEP 1: Pre-compute SMC context for the day
    smc_ctx = self._build_smc_context(day_df, day_date, params.get("SMC"))

    # STEP 2: Normal IB + signal detection
    signal = self._detect_signal(...)

    # STEP 3: Apply SMC filter
    if smc_ctx and signal:
        final_signal = self._apply_smc_filter(signal, smc_ctx, day_df)
        if final_signal is None:
            return None  # Rejected or no confirmation found
        signal = final_signal

    # STEP 4: Simulate trade
    return self._simulate_trade(signal, day_df, params)
```

Pre-computation strategy:
- H1 fractals: detect from previous 48h of data (requires loading prior days)
- M2 FVGs: detect from trade window M2 data
- H1 FVGs: detect from morning H1 candles
- CISD: scan forward after signal for confirmation

Performance target: <10s per param combo (vs 4-6s current without SMC).

### 6.3 Live Bot

Same SMCEngine, fed by real MT5 bars instead of emulator data. Optional JSON checkpoint for crash recovery (serialize registry state).

---

## 7. Debug Tooling

### 7.1 Debug Notebook (`notebooks/debug_smc.ipynb`)

Side-by-side comparison: run same day WITHOUT SMC vs WITH SMC.

Cells:
1. Setup: imports, paths, params
2. Run without SMC: `SingleDayBacktest(symbol).run(date, params)`
3. Run WITH SMC: `SingleDayBacktestSMC(symbol, smc_config).run(date, params)`
4. Compare results: trade taken?, PnL, R, SMC decision
5. Chart WITHOUT SMC: standard generate_chart()
6. Chart WITH SMC: generate_chart_with_smc() with overlays
7. Event log: table of all SMC events for the day

### 7.2 Chart Overlays

Added to `notebooks/debug_tools.py`:

```python
def generate_chart_with_smc(
    day_date, symbol, strategy_params, smc_config,
    show_structures=["fractals", "fvg", "cisd", "bos"],
    figsize=(20, 12), save_path=None
):
```

Visual elements:
- FVGs: shaded rectangles (green=bullish, red=bearish, alpha=0.2)
- Fractals: triangle markers (up=high, down=low)
- BOS/CISD: vertical dashed lines with direction arrows and labels
- Confirmation candle: highlighted background with annotation
- Modified entry/SL: markers at new levels (distinct from original signal markers)
- Event timeline: small subplot below chart

### 7.3 Standalone Analysis (`strategy_optimization/smc_tools/`)

smc_analyzer.py: Run SMC detection on historical data without any strategy logic. Output: DataFrame of all structures detected per day. Useful for:
- Validating detector accuracy against manual chart analysis
- Statistical analysis (how many FVGs per day? Fractal sweep rate?)
- Finding good example days for debug notebook

---

## 8. SMC Configuration

```python
@dataclass
class SMCConfig:
    instrument: str                       # "GER40" / "XAUUSD"

    # Feature flags (enable incrementally per phase)
    enable_fractals: bool = True
    enable_fvg: bool = True
    enable_cisd: bool = True
    enable_bos: bool = True

    # Fractal params
    fractal_timeframe: str = "H1"         # H1 or H4
    fractal_lookback_hours: int = 48
    fractal_proximity_pct: float = 0.002  # 0.2% -- "near price" threshold

    # FVG params
    fvg_timeframes: List[str] = field(default_factory=lambda: ["M2", "H1"])
    fvg_min_size_pct: float = 0.0005      # Min gap size as fraction of price
    fvg_rebalance_threshold: float = 0.3  # 30% fill = partial rebalance

    # CISD params
    cisd_timeframes: List[str] = field(default_factory=lambda: ["M2"])

    # BOS params
    bos_timeframes: List[str] = field(default_factory=lambda: ["M5", "M15"])

    # Confluence scoring weights
    weight_fractal: float = 1.0
    weight_fvg: float = 1.5
    weight_cisd: float = 2.0
    weight_bos: float = 1.5
    min_confluence_score: float = 2.0     # Min score to approve entry

    # Confirmation settings
    max_wait_minutes: int = 30            # Timeout for AWAITING_CONFIRMATION

    # Context timeframes for multi-TF analysis
    context_tfs: List[str] = field(default_factory=lambda: ["M2", "M5", "H1"])
```

Added to strategy params dict:
```python
GER40_PARAMS_PROD = {
    "OCAE": { ... existing params ... },
    "Reverse": { ... existing params ... },
    "TCWE": { ... existing params ... },
    "REV_RB": { ... existing params ... },
    "SMC": SMCConfig(instrument="GER40", ...)  # None = disabled
}
```

---

## 9. Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Integration pattern | Wrapper (IBStrategySMC extends IBStrategy) | No changes to production code, easy A/B testing, rollback = smc_config=None |
| State storage | In-memory (Python objects) | Backtest speed is critical. SQLite too slow for hot path. Export post-run. |
| Detector design | Pure functions (stateless) | Easy to test with synthetic data, easy to optimize, easy to parallelize |
| Feature activation | Feature flags in SMCConfig | Enable tools incrementally: fractals+FVG first, then CISD, then BOS |
| Fast backtest approach | Pre-compute per day, query during signal | Maintains ~10s speed. Similar to existing IB caching pattern. |
| Multi-TF data | TimeframeManager with caching | Resample once, cache result. Incremental updates for live/slow backtest. |
| Phase 5 scope | Validation rules ONLY from InValidation.md | OB, BB, FVL, Liquidity, Inducement not in scope per user decision |

---

## 10. Phase Overview

### Phase 1: Foundation + First Detectors
See: `PHASE_1_FOUNDATION.md`
- Data models, config, timeframe manager, registry, event log
- Fractal detector (port from existing), FVG detector
- Standalone analysis tool
- Tests

### Phase 2: Core Detectors + Engine
See: `PHASE_2_ENGINE.md`
- CISD detector, market structure detector (HH/HL/LL/LH + BOS/CHoCH)
- SMCEngine (evaluate_signal, check_confirmation, confluence scoring)
- Tests

### Phase 3: Strategy Integration + Debug
See: `PHASE_3_INTEGRATION.md`
- IBStrategySMC wrapper class
- Debug notebook (debug_smc.ipynb)
- Chart overlays in debug_tools.py
- Slow backtest integration test

### Phase 4: Fast Backtest Integration
See: `PHASE_4_FAST_BACKTEST.md`
- Batch SMC methods for fast_backtest.py
- Pre-computation strategy, performance benchmarks
- Parity tests (slow vs fast)

### Phase 5: Validation Rules
See: `PHASE_5_VALIDATION.md`
- FVG validation (6 variants from InValidation.md)
- Block validation via CISD
- Entry/SL/TP/BE validation rules
- Order flow validation

---

## 11. Key Source Files

| File | Role |
|------|------|
| `src/utils/strategy_logic.py` | Current signal detection (4 variations). Add SMC_CONFIG to params. |
| `src/strategies/ib_strategy.py` | Current FSM (40KB). Base class for IBStrategySMC. |
| `src/strategies/base_strategy.py` | Signal dataclass. May extend for SMC fields. |
| `params_optimizer/engine/fast_backtest.py` | Fast engine (1636 lines). Add batch SMC methods. |
| `backtest/emulator/mt5_emulator.py` | Slow engine. No changes needed (IBStrategySMC is a drop-in). |
| `notebooks/debug_tools.py` | Debug tools. Add generate_chart_with_smc(). |
| `strategy_optimization/fractals/fractals.py` | Existing fractal detection. Port to src/smc/detectors/. |
| `notion_export/FVG.md` | FVG detection rules |
| `notion_export/CISD.md` | CISD detection rules |
| `notion_export/BosChochMSSMS range.md` | BOS/CHoCH rules |
| `notion_export/InValidation.md` | Validation rules (Phase 5) |
