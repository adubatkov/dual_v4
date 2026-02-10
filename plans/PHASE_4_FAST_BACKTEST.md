# Phase 4: SMC Integration into Fast Backtest Engine

## Goal

Integrate SMC (Smart Money Concepts) analysis into the fast vectorized backtest engine (`params_optimizer/engine/fast_backtest.py`) while maintaining performance targets and ensuring parity with slow backtest SMC decisions.

**Performance Target**: Less than 10 seconds per parameter combination (current baseline: 4-6 seconds without SMC).

**Success Criteria**:
- SMC filtering logic matches slow backtest behavior (parity testing passes)
- Performance degradation under 2x (4-6s -> under 10s)
- All SMC structures pre-computed once per day (no redundant calculations)
- Cross-day state handled correctly (fractals, active FVGs from yesterday)

---

## 1. Current Fast Backtest Architecture

### 1.1 Processing Model

```python
# Current flow in FastBacktest.run_with_params()
for day_date, day_df in m1_data.groupby("ib_date"):
    trade = _process_day(day_df, day_date, params)
    if trade:
        trades.append(trade)
```

**_process_day() workflow**:
1. Get IB levels (from cache or compute): IBH, IBL, EQ
2. Extract trade window M1 data
3. Resample M1 -> M2 for signal detection
4. Detect signals (priority: Reverse > OCAE > TCWE > REV_RB)
5. Simulate trade execution on M1 data

**Key characteristics**:
- Days processed independently (stateless per day)
- M2 used for signal detection (matches IBStrategy behavior)
- M1 used for precise SL/TP simulation
- IB cache pre-computed for all dates/params combinations

### 1.2 Performance Optimization Strategies

Current optimizations in fast_backtest.py:
- Vectorized operations on pandas DataFrames (no row-by-row iteration)
- IB pre-computation (ib_precompute.py generates cache upfront)
- M1 -> M2 resampling cached per day
- Signal detection on M2 bars (reduced data volume)
- Trade simulation on M1 bars (only for detected signals)

---

## 2. SMC Integration Approach

### 2.1 Core Challenge: Cross-Day State

**Problem**: Fast backtest processes days independently, but SMC structures require cross-day context:
- H1 fractals: need 48h lookback (2 days of prior data)
- Active FVGs: may form yesterday and remain active today
- BOS/CISD: may reference previous day's structure

**Solution**: Load lookback data when building SMC context for each day.

```python
def _get_lookback_data(self, day_date: datetime.date, hours: int) -> pd.DataFrame:
    """
    Get M1 data starting from (day_date - hours) up to day_date start.

    Args:
        day_date: Current trading day
        hours: Number of hours to look back (e.g., 48 for H1 fractals)

    Returns:
        DataFrame with M1 data for lookback period
    """
    tz = pytz.timezone("UTC")
    day_start = datetime.combine(day_date, time(0, 0))
    day_start_ts = pd.Timestamp(tz.localize(day_start))
    lookback_start_ts = day_start_ts - timedelta(hours=hours)

    lookback_df = self.m1_data[
        (self.m1_data["time"] >= lookback_start_ts) &
        (self.m1_data["time"] < day_start_ts)
    ].copy()

    return lookback_df
```

### 2.2 SMCDayContext: Pre-computed Structure Container

**Purpose**: Store all SMC structures detected for a single day. Built once per day, used multiple times during signal evaluation.

```python
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime

@dataclass
class SMCDayContext:
    """
    Pre-computed SMC structures for a single trading day.
    Built once per day in _build_smc_context().
    Used by _apply_smc_filter_batch() for signal evaluation.
    """
    # Metadata
    instrument: str
    day_date: datetime.date
    build_time: datetime

    # Fractals (usually H1 timeframe)
    fractals_high: List['Fractal'] = field(default_factory=list)  # Active high fractals
    fractals_low: List['Fractal'] = field(default_factory=list)   # Active low fractals

    # FVGs by timeframe
    fvgs_m2: List['FVG'] = field(default_factory=list)  # M2 FVGs
    fvgs_h1: List['FVG'] = field(default_factory=list)  # H1 FVGs

    # Market structure
    bos_events: List['BOS'] = field(default_factory=list)          # BOS/CHoCH events
    cisd_events: List['CISD'] = field(default_factory=list)        # CISD confirmations

    # Cached price levels (for quick lookups)
    active_fractal_highs: List[float] = field(default_factory=list)  # Unswept high fractals
    active_fractal_lows: List[float] = field(default_factory=list)   # Unswept low fractals
    active_fvg_zones: List[Dict[str, float]] = field(default_factory=list)  # {"high": x, "low": y, "direction": "bullish"}

    # Performance metrics
    num_fractals_detected: int = 0
    num_fvgs_detected: int = 0
    num_bos_detected: int = 0
    num_cisd_detected: int = 0
```

---

## 3. Implementation: New Methods

### 3.1 _build_smc_context()

**Purpose**: Pre-compute all SMC structures for the day ONCE before signal detection.

**Location**: Add to `FastBacktest` class in `fast_backtest.py`.

**Method Signature**:

```python
def _build_smc_context(
    self,
    day_df: pd.DataFrame,
    day_date: datetime.date,
    smc_config: 'SMCConfig'
) -> SMCDayContext:
    """
    Pre-compute all SMC structures for a single trading day.

    Called once per day in _process_day() BEFORE signal detection.

    Args:
        day_df: M1 data for current day (from groupby)
        day_date: Trading day date
        smc_config: SMC configuration (fractal lookback, FVG params, etc.)

    Returns:
        SMCDayContext with all pre-computed structures

    Performance Note:
        - H1 fractals: requires 48h lookback (load prior days' M1 data)
        - M2 FVGs: computed from trade window M2 data
        - H1 FVGs: computed from H1 resampled data
        - Target overhead: <1s per day
    """
    from src.smc.models import SMCDayContext, Fractal, FVG, BOS, CISD
    from src.smc.detectors.fractal_detector import detect_fractals_vectorized
    from src.smc.detectors.fvg_detector import detect_fvg_vectorized
    from src.smc.detectors.market_structure_detector import detect_bos_vectorized
    from src.smc.detectors.cisd_detector import detect_cisd_vectorized

    context = SMCDayContext(
        instrument=self.symbol,
        day_date=day_date,
        build_time=datetime.utcnow()
    )

    # Step 1: Load lookback data for H1 fractals (48h)
    lookback_df = self._get_lookback_data(day_date, hours=48)

    # Combine lookback + current day for H1 analysis
    h1_analysis_df = pd.concat([lookback_df, day_df], ignore_index=True)
    h1_analysis_df = h1_analysis_df.sort_values("time").reset_index(drop=True)

    # Resample to H1
    h1_df = self._resample_m1_to_h1(h1_analysis_df)

    # Step 2: Detect H1 fractals (5-bar pattern: 2 before, center, 2 after)
    fractals = detect_fractals_vectorized(
        df=h1_df,
        instrument=self.symbol,
        timeframe="H1",
        lookback=smc_config.fractal_lookback  # typically 2
    )

    # Separate high/low fractals
    for fractal in fractals:
        if fractal.type == "high":
            context.fractals_high.append(fractal)
            if not fractal.swept:
                context.active_fractal_highs.append(fractal.price)
        else:  # low
            context.fractals_low.append(fractal)
            if not fractal.swept:
                context.active_fractal_lows.append(fractal.price)

    context.num_fractals_detected = len(fractals)

    # Step 3: Detect H1 FVGs
    h1_fvgs = detect_fvg_vectorized(
        df=h1_df,
        instrument=self.symbol,
        timeframe="H1"
    )
    context.fvgs_h1 = [fvg for fvg in h1_fvgs if fvg.status == "active"]

    # Step 4: Detect M2 FVGs (from trade window M2 data)
    # Note: trade window not yet computed here; will be done in _process_day
    # For now, we'll store M2 FVG detection as deferred until trade window available
    # Alternative: detect M2 FVGs from full day M2 data
    m2_df = self._resample_m1_to_m2(day_df)
    m2_fvgs = detect_fvg_vectorized(
        df=m2_df,
        instrument=self.symbol,
        timeframe="M2"
    )
    context.fvgs_m2 = [fvg for fvg in m2_fvgs if fvg.status == "active"]

    # Step 5: Detect BOS/CHoCH events (on H1 for now)
    bos_events = detect_bos_vectorized(
        df=h1_df,
        instrument=self.symbol,
        timeframe="H1"
    )
    context.bos_events = bos_events
    context.num_bos_detected = len(bos_events)

    # Step 6: Detect CISD events (on M2 for faster reaction)
    cisd_events = detect_cisd_vectorized(
        df=m2_df,
        instrument=self.symbol,
        timeframe="M2"
    )
    context.cisd_events = cisd_events
    context.num_cisd_detected = len(cisd_events)

    # Step 7: Build active FVG zones cache for quick lookups
    for fvg in context.fvgs_h1 + context.fvgs_m2:
        if fvg.status == "active":
            context.active_fvg_zones.append({
                "high": fvg.high,
                "low": fvg.low,
                "midpoint": fvg.midpoint,
                "direction": fvg.direction,
                "timeframe": fvg.timeframe
            })

    context.num_fvgs_detected = len(context.fvgs_h1) + len(context.fvgs_m2)

    return context
```

**Performance Optimization**:
- Use vectorized detector functions (no row-by-row iteration)
- Cache active structure lists (avoid repeated filtering)
- Pre-compute price level lists for binary search
- Limit lookback to minimum required (48h for H1 fractals)

### 3.2 _apply_smc_filter_batch()

**Purpose**: Apply SMC filtering to a detected signal. May modify entry/SL/TP or reject signal entirely.

**Method Signature**:

```python
def _apply_smc_filter_batch(
    self,
    signal: Signal,
    smc_context: SMCDayContext,
    df_trade_m1: pd.DataFrame,
    params: Dict
) -> Optional[Signal]:
    """
    Apply SMC filtering to a detected signal.

    Checks if signal conflicts with active SMC structures.
    If conflict found, scans forward for confirmation (CISD, FVG rebalance, etc.).

    Args:
        signal: Detected signal (Reverse, OCAE, TCWE, REV_RB)
        smc_context: Pre-computed SMC structures for the day
        df_trade_m1: M1 data for trade window (for forward scanning)
        params: Strategy parameters

    Returns:
        Modified signal (with new entry/SL/TP) or None (signal rejected)

    SMC Filtering Logic:
        1. Check for unswept fractals opposing signal direction
        2. Check for active FVGs in entry zone
        3. If conflict: scan forward for SMC confirmation
        4. If confirmation found: modify signal (new entry at CISD/FVG test)
        5. If no confirmation: reject signal (return None)
    """
    # Step 1: Check for opposing fractals
    if signal.direction == "long":
        # Check if unswept low fractal exists above entry (bearish structure)
        opposing_fractals = [
            f for f in smc_context.active_fractal_lows
            if f > signal.entry_price * 0.98  # Within 2% above entry
        ]
    else:  # short
        # Check if unswept high fractal exists below entry (bullish structure)
        opposing_fractals = [
            f for f in smc_context.active_fractal_highs
            if f < signal.entry_price * 1.02  # Within 2% below entry
        ]

    if opposing_fractals:
        # Conflict detected: scan forward for confirmation
        confirmation = self._scan_for_smc_confirmation(
            signal, smc_context, df_trade_m1, params
        )

        if confirmation is None:
            # No confirmation found, reject signal
            return None

        # Modify signal with confirmation details
        signal.entry_price = confirmation["entry_price"]
        signal.stop_price = confirmation["stop_price"]
        signal.entry_time = confirmation["entry_time"]
        signal.extra["smc_confirmed"] = True
        signal.extra["smc_confirmation_type"] = confirmation["type"]

        return signal

    # Step 2: Check for active FVGs in entry zone
    entry_zone_fvgs = [
        fvg for fvg in smc_context.active_fvg_zones
        if (fvg["low"] <= signal.entry_price <= fvg["high"])
    ]

    if entry_zone_fvgs:
        # Entry inside FVG: check if FVG supports signal direction
        supporting_fvgs = [
            fvg for fvg in entry_zone_fvgs
            if (signal.direction == "long" and fvg["direction"] == "bullish") or
               (signal.direction == "short" and fvg["direction"] == "bearish")
        ]

        if not supporting_fvgs:
            # FVG opposes signal direction: scan for confirmation
            confirmation = self._scan_for_smc_confirmation(
                signal, smc_context, df_trade_m1, params
            )

            if confirmation is None:
                return None

            signal.entry_price = confirmation["entry_price"]
            signal.stop_price = confirmation["stop_price"]
            signal.entry_time = confirmation["entry_time"]
            signal.extra["smc_confirmed"] = True
            signal.extra["smc_confirmation_type"] = confirmation["type"]

            return signal

    # No conflict detected, signal passes SMC filter unchanged
    signal.extra["smc_checked"] = True
    return signal
```

### 3.3 _scan_for_smc_confirmation()

**Purpose**: Scan forward in M1 data for SMC confirmation (CISD, FVG rebalance, fractal sweep).

**Method Signature**:

```python
def _scan_for_smc_confirmation(
    self,
    signal: Signal,
    smc_context: SMCDayContext,
    df_trade_m1: pd.DataFrame,
    params: Dict,
    max_bars: int = 30  # Max bars to scan forward (30 M1 bars = 30 minutes)
) -> Optional[Dict[str, Any]]:
    """
    Scan forward for SMC confirmation after signal detection.

    Confirmation types:
        - CISD: Delivery candle reaction at signal level
        - FVG Rebalance: Price returns to fill FVG, then reacts
        - Fractal Sweep: Price sweeps fractal, then reverses

    Args:
        signal: Original detected signal
        smc_context: Pre-computed SMC structures
        df_trade_m1: M1 data for forward scanning
        params: Strategy parameters
        max_bars: Maximum M1 bars to scan forward

    Returns:
        Confirmation dict with new entry/SL/time, or None if no confirmation
    """
    # Find signal entry index in M1 data
    signal_time = signal.entry_time
    signal_idx = df_trade_m1[df_trade_m1["time"] >= signal_time].index[0]

    # Scan forward up to max_bars
    scan_end_idx = min(signal_idx + max_bars, len(df_trade_m1) - 1)

    for i in range(signal_idx + 1, scan_end_idx + 1):
        bar = df_trade_m1.iloc[i]

        # Check for CISD confirmation
        cisd_match = self._check_cisd_confirmation(bar, signal, smc_context)
        if cisd_match:
            return {
                "type": "CISD",
                "entry_price": cisd_match["entry_price"],
                "stop_price": cisd_match["stop_price"],
                "entry_time": bar["time"]
            }

        # Check for FVG rebalance confirmation
        fvg_match = self._check_fvg_rebalance(bar, signal, smc_context)
        if fvg_match:
            return {
                "type": "FVG_REBALANCE",
                "entry_price": fvg_match["entry_price"],
                "stop_price": fvg_match["stop_price"],
                "entry_time": bar["time"]
            }

        # Check for fractal sweep confirmation
        fractal_match = self._check_fractal_sweep(bar, signal, smc_context)
        if fractal_match:
            return {
                "type": "FRACTAL_SWEEP",
                "entry_price": fractal_match["entry_price"],
                "stop_price": fractal_match["stop_price"],
                "entry_time": bar["time"]
            }

    # No confirmation found within max_bars
    return None
```

**Helper Methods**:

```python
def _check_cisd_confirmation(
    self,
    bar: pd.Series,
    signal: Signal,
    smc_context: SMCDayContext
) -> Optional[Dict[str, float]]:
    """
    Check if current bar shows CISD confirmation.

    CISD: Delivery candle reaction at level (close beyond body of delivery candle).
    """
    # Find CISD events near signal entry price
    cisd_events = [
        cisd for cisd in smc_context.cisd_events
        if abs(cisd.confirmation_price - signal.entry_price) / signal.entry_price < 0.002  # Within 0.2%
    ]

    if not cisd_events:
        return None

    # Check if bar reacts from CISD level
    for cisd in cisd_events:
        if signal.direction == "long":
            # Long: bar low touches CISD, closes above delivery body high
            if (bar["low"] <= cisd.delivery_candle_body_low * 1.001 and
                bar["close"] > cisd.delivery_candle_body_high):
                return {
                    "entry_price": float(bar["close"]),
                    "stop_price": float(cisd.delivery_candle_body_low * 0.999)
                }
        else:  # short
            # Short: bar high touches CISD, closes below delivery body low
            if (bar["high"] >= cisd.delivery_candle_body_high * 0.999 and
                bar["close"] < cisd.delivery_candle_body_low):
                return {
                    "entry_price": float(bar["close"]),
                    "stop_price": float(cisd.delivery_candle_body_high * 1.001)
                }

    return None

def _check_fvg_rebalance(
    self,
    bar: pd.Series,
    signal: Signal,
    smc_context: SMCDayContext
) -> Optional[Dict[str, float]]:
    """
    Check if bar shows FVG rebalance confirmation.

    FVG Rebalance: Price enters FVG (fills partially), then reacts away.
    """
    for fvg_zone in smc_context.active_fvg_zones:
        # Check if bar enters FVG zone
        bar_in_fvg = (bar["low"] <= fvg_zone["high"] and bar["high"] >= fvg_zone["low"])

        if not bar_in_fvg:
            continue

        # Check if direction aligns
        if signal.direction == "long" and fvg_zone["direction"] == "bullish":
            # Bar dips into bullish FVG, closes above midpoint
            if bar["low"] <= fvg_zone["midpoint"] and bar["close"] > fvg_zone["midpoint"]:
                return {
                    "entry_price": float(bar["close"]),
                    "stop_price": float(fvg_zone["low"] * 0.999)
                }
        elif signal.direction == "short" and fvg_zone["direction"] == "bearish":
            # Bar rises into bearish FVG, closes below midpoint
            if bar["high"] >= fvg_zone["midpoint"] and bar["close"] < fvg_zone["midpoint"]:
                return {
                    "entry_price": float(bar["close"]),
                    "stop_price": float(fvg_zone["high"] * 1.001)
                }

    return None

def _check_fractal_sweep(
    self,
    bar: pd.Series,
    signal: Signal,
    smc_context: SMCDayContext
) -> Optional[Dict[str, float]]:
    """
    Check if bar shows fractal sweep + reversal confirmation.

    Fractal Sweep: Price sweeps fractal level (wick beyond), then reverses.
    """
    if signal.direction == "long":
        # Long: sweep low fractal (wick below), close above fractal
        for fractal_low in smc_context.active_fractal_lows:
            if bar["low"] < fractal_low * 0.999 and bar["close"] > fractal_low:
                return {
                    "entry_price": float(bar["close"]),
                    "stop_price": float(bar["low"] * 0.998)
                }
    else:  # short
        # Short: sweep high fractal (wick above), close below fractal
        for fractal_high in smc_context.active_fractal_highs:
            if bar["high"] > fractal_high * 1.001 and bar["close"] < fractal_high:
                return {
                    "entry_price": float(bar["close"]),
                    "stop_price": float(bar["high"] * 1.002)
                }

    return None
```

---

## 4. Integration into _process_day()

**Modified _process_day() workflow**:

```python
def _process_day(
    self, day_df: pd.DataFrame, day_date: datetime.date, params: Dict
) -> Optional[Dict[str, Any]]:
    """
    Process single trading day (MODIFIED for SMC integration).

    New workflow:
        1. Get IB from cache/compute
        2. Build SMC context (fractals, FVGs, BOS, CISD)
        3. Get trade window M1/M2 data
        4. Detect signals (Reverse > OCAE > TCWE > REV_RB)
        5. Apply SMC filter to signal (may modify or reject)
        6. Simulate trade execution on M1
    """
    # 1. Get IB from cache or compute
    ib = self._get_ib_cached(day_date, params)
    if ib is None:
        ib = self._compute_ib(day_df, day_date, params)
        self._cache_misses += 1
    else:
        self._cache_hits += 1

    if not ib:
        return None

    ibh, ibl, eq = ib["IBH"], ib["IBL"], ib["EQ"]

    # 2. Build SMC context (NEW STEP)
    smc_config = self._get_smc_config(params)  # Extract SMC params from params dict
    smc_context = self._build_smc_context(day_df, day_date, smc_config)

    # 3. Get trade window - M1 data
    df_trade_m1 = self._get_trade_window(day_df, day_date, params)
    if df_trade_m1.empty:
        return None

    # 4. Resample M1 to M2 for signal detection
    df_trade_m2 = self._resample_m1_to_m2(df_trade_m1)
    if df_trade_m2.empty:
        return None

    # 5. Get pre-context for Reverse signal
    ib_start_ts, ib_end_ts = self._ib_window_on_date(...)
    first_trade_ts = df_trade_m1["time"].iat[0]
    df_pre_context_m1 = day_df[(day_df["time"] >= ib_start_ts) & (day_df["time"] < first_trade_ts)]
    df_pre_context_m2 = self._resample_m1_to_m2(df_pre_context_m1) if not df_pre_context_m1.empty else pd.DataFrame()

    trade_start_price = float(df_trade_m2["open"].iat[0])

    # 6. Check signals in priority order
    _, trade_window_end = self._trade_window_on_date(...)

    def is_signal_valid(sig):
        # Existing news filter + trade window check
        # ... (unchanged)

    # Collect primary signals
    primary_candidates = []

    rev_sig = self._check_reverse(df_trade_m2, df_pre_context_m2, ibh, ibl, eq, params)
    if is_signal_valid(rev_sig):
        primary_candidates.append(rev_sig)

    ocae_sig = self._check_ocae(df_trade_m2, ibh, ibl, eq, trade_start_price, params)
    if is_signal_valid(ocae_sig):
        primary_candidates.append(ocae_sig)

    tcwe_sig = self._check_tcwe(df_trade_m2, ibh, ibl, eq, trade_start_price, params)
    if is_signal_valid(tcwe_sig):
        primary_candidates.append(tcwe_sig)

    # Select earliest primary signal
    if primary_candidates:
        priority = {"Reverse": 0, "OCAE": 1, "TCWE": 2}
        signal = min(primary_candidates, key=lambda s: (s.entry_idx, priority.get(s.signal_type, 99)))
    else:
        # Check REV_RB if no primary signals
        rev_rb_sig = self._check_rev_rb(df_trade_m2, ibh, ibl, eq, params)
        if is_signal_valid(rev_rb_sig):
            signal = rev_rb_sig
        else:
            return None

    # 7. Apply SMC filter (NEW STEP)
    signal = self._apply_smc_filter_batch(signal, smc_context, df_trade_m1, params)
    if signal is None:
        # Signal rejected by SMC filter
        return None

    # 8. Simulate trade execution on M1 data
    return self._simulate_trade_on_m1(df_trade_m1, signal, ib, day_date, params)
```

**New Helper Method**:

```python
def _get_smc_config(self, params: Dict) -> 'SMCConfig':
    """
    Extract SMC configuration from params dict.

    SMC params (add to param optimization space):
        - smc_enabled: bool (default True)
        - smc_fractal_lookback: int (default 2, for 5-bar fractal)
        - smc_fvg_min_gap_pct: float (default 0.001, minimum gap size)
        - smc_confirmation_max_bars: int (default 30, max M1 bars to scan)
    """
    from src.smc.config import SMCConfig

    return SMCConfig(
        enabled=params.get("smc_enabled", True),
        fractal_lookback=params.get("smc_fractal_lookback", 2),
        fvg_min_gap_pct=params.get("smc_fvg_min_gap_pct", 0.001),
        confirmation_max_bars=params.get("smc_confirmation_max_bars", 30),
        # Add more SMC params as needed
    )
```

---

## 5. Performance Optimization Strategies

### 5.1 Vectorized Detectors

All detector functions MUST use vectorized pandas operations:

```python
# BAD: Row-by-row iteration (slow)
fractals = []
for i in range(2, len(df) - 2):
    if df.iloc[i]["high"] > df.iloc[i-1]["high"] and df.iloc[i]["high"] > df.iloc[i-2]["high"]:
        fractals.append(...)

# GOOD: Vectorized operations (fast)
df["is_high_fractal"] = (
    (df["high"] > df["high"].shift(1)) &
    (df["high"] > df["high"].shift(2)) &
    (df["high"] > df["high"].shift(-1)) &
    (df["high"] > df["high"].shift(-2))
)
fractals = df[df["is_high_fractal"]].apply(lambda row: Fractal(...), axis=1).tolist()
```

### 5.2 Caching Strategies

**Per-day caching** (within _build_smc_context):
- Resample M1 -> H1 once, reuse for all detectors
- Pre-compute active structure lists (avoid repeated filtering)
- Store price levels in sorted lists for binary search

**Cross-day caching** (future optimization):
- Cache H1 fractals across days (fractals don't change retroactively)
- Cache FVG invalidations (once filled, always filled)
- Requires modification to stateless day-by-day model

### 5.3 Minimal Lookback

```python
# Only load what's needed
H1_FRACTAL_LOOKBACK_HOURS = 48  # 2 days for 5-bar H1 fractal pattern
M2_FVG_LOOKBACK_HOURS = 0       # Only current day needed

# Adjust based on detector requirements
if smc_config.enable_h4_fractals:
    H4_FRACTAL_LOOKBACK_HOURS = 96  # 4 days for 5-bar H4 fractal
```

### 5.4 Early Termination

```python
# In _scan_for_smc_confirmation, terminate early if signal invalidated
for i in range(signal_idx + 1, scan_end_idx + 1):
    bar = df_trade_m1.iloc[i]

    # Early termination: if price moves against signal beyond threshold, stop scanning
    if signal.direction == "long":
        if bar["low"] < signal.stop_price * 0.995:  # 0.5% below SL
            return None  # Signal invalidated, no confirmation possible
    else:
        if bar["high"] > signal.stop_price * 1.005:
            return None

    # Continue scanning for confirmation...
```

---

## 6. Parity Testing Approach

**Goal**: Ensure fast backtest SMC decisions match slow backtest (IBStrategySMC).

### 6.1 Test Dataset

```python
# Use small representative dataset
test_dates = ["2024-01-15", "2024-02-20", "2024-03-10"]  # 3 days with known signals
test_params = GER40_PARAMS_PROD  # Production params
```

### 6.2 Parity Test Script

```python
# dual_v4/tests/test_smc_parity.py

def test_smc_parity_fast_vs_slow():
    """
    Test that fast backtest SMC decisions match slow backtest.

    For each test day:
        1. Run slow backtest (IBStrategySMC) -> get signal decisions
        2. Run fast backtest with SMC -> get signal decisions
        3. Compare: entry times, entry prices, SL, TP, rejection reasons
    """
    from params_optimizer.engine.fast_backtest import FastBacktest
    from backtest.emulator.mt5_emulator import MT5Emulator
    from src.strategies.ib_strategy_smc import IBStrategySMC

    # Load test data
    m1_data = load_m1_data("GER40", start="2024-01-15", end="2024-01-16")

    # Run slow backtest
    slow_results = run_slow_backtest_smc(m1_data, GER40_PARAMS_PROD)

    # Run fast backtest
    fast_bt = FastBacktest(symbol="GER40", m1_data=m1_data)
    fast_results = fast_bt.run_with_params(GER40_PARAMS_PROD)

    # Compare decisions
    assert len(slow_results["trades"]) == len(fast_results["trades"])

    for slow_trade, fast_trade in zip(slow_results["trades"], fast_results["trades"]):
        # Entry time within 1 minute
        assert abs((slow_trade["entry_time"] - fast_trade["entry_time"]).total_seconds()) < 60

        # Entry price within 0.1%
        assert abs(slow_trade["entry_price"] - fast_trade["entry_price"]) / slow_trade["entry_price"] < 0.001

        # SL/TP within 0.2%
        assert abs(slow_trade["sl"] - fast_trade["sl"]) / slow_trade["sl"] < 0.002
        assert abs(slow_trade["tp"] - fast_trade["tp"]) / slow_trade["tp"] < 0.002

        # Signal rejection reasons match
        if slow_trade["rejected"]:
            assert fast_trade["rejected"]
            assert slow_trade["rejection_reason"] == fast_trade["rejection_reason"]
```

### 6.3 Debugging Mismatches

When parity test fails:

```python
# Add debug logging to fast_backtest.py
def _apply_smc_filter_batch(self, signal, smc_context, df_trade_m1, params):
    logger.debug(f"SMC Filter Input: signal={signal.signal_type}, entry_price={signal.entry_price}")
    logger.debug(f"Active fractals: high={smc_context.active_fractal_highs}, low={smc_context.active_fractal_lows}")

    # ... filtering logic ...

    logger.debug(f"SMC Filter Output: {'PASSED' if result else 'REJECTED'}")
    return result
```

Compare logs from slow vs fast backtest to find divergence point.

---

## 7. Execution Checklist

### Phase 4A: Foundation (Week 1)

- [ ] Create `src/smc/models.py` with `SMCDayContext` dataclass
- [ ] Implement `_get_lookback_data()` in `FastBacktest`
- [ ] Implement `_build_smc_context()` skeleton (no detectors yet)
- [ ] Add unit test for `SMCDayContext` serialization

### Phase 4B: Detectors Integration (Week 2)

- [ ] Port fractal detector to `src/smc/detectors/fractal_detector.py`
- [ ] Implement vectorized `detect_fractals_vectorized()`
- [ ] Implement vectorized `detect_fvg_vectorized()`
- [ ] Implement vectorized `detect_bos_vectorized()`
- [ ] Implement vectorized `detect_cisd_vectorized()`
- [ ] Test each detector on synthetic data (performance: <100ms per day)

### Phase 4C: SMC Filter (Week 3)

- [ ] Implement `_apply_smc_filter_batch()`
- [ ] Implement `_scan_for_smc_confirmation()`
- [ ] Implement helper methods: `_check_cisd_confirmation()`, `_check_fvg_rebalance()`, `_check_fractal_sweep()`
- [ ] Add SMC params to `_get_smc_config()`
- [ ] Test SMC filter on single day with known structures

### Phase 4D: Integration & Testing (Week 4)

- [ ] Modify `_process_day()` to call `_build_smc_context()` and `_apply_smc_filter_batch()`
- [ ] Run fast backtest on 3-day test dataset (performance: <30s total)
- [ ] Implement parity test script (`tests/test_smc_parity.py`)
- [ ] Run parity test, debug mismatches
- [ ] Benchmark performance: confirm <10s per param combo on full dataset

### Phase 4E: Optimization (Week 5)

- [ ] Profile code with `cProfile` to find bottlenecks
- [ ] Optimize detector functions (vectorization, caching)
- [ ] Reduce lookback data loading (only load if needed)
- [ ] Add early termination in `_scan_for_smc_confirmation()`
- [ ] Rerun benchmarks: target <8s per param combo

### Phase 4F: Documentation (Week 6)

- [ ] Update `params_optimizer/README.md` with SMC integration notes
- [ ] Document new params: `smc_enabled`, `smc_fractal_lookback`, etc.
- [ ] Add SMC debugging section to `docs/BACKTEST_DEBUGGING.md`
- [ ] Create performance comparison report (with/without SMC)

---

## 8. Performance Targets Summary

| Metric | Without SMC (Baseline) | With SMC (Target) | With SMC (Stretch) |
|--------|------------------------|-------------------|-------------------|
| Time per param combo | 4-6s | <10s | <8s |
| Time per day (processing) | ~10ms | ~50ms | ~30ms |
| SMC context build time | N/A | <1s | <500ms |
| Lookback data load | N/A | <200ms | <100ms |
| Detector execution (all) | N/A | <500ms | <300ms |
| SMC filter per signal | N/A | <10ms | <5ms |

**Hardware baseline**: Intel i7-10700K, 32GB RAM, SSD (same as current benchmarks).

---

## 9. Risk Mitigation

### 9.1 Fallback Mode

If SMC integration causes performance degradation beyond 2x:

```python
# In params dict, add fallback flag
params["smc_enabled"] = False  # Disable SMC filter, revert to baseline
```

### 9.2 Incremental Rollout

- Week 1-2: Detectors only (no filtering, just logging detected structures)
- Week 3-4: SMC filter with logging (no signal modification)
- Week 5-6: Full SMC filter with signal modification

### 9.3 Regression Testing

Before merging Phase 4:
- [ ] Baseline backtest results (without SMC) must remain unchanged
- [ ] All existing unit tests must pass
- [ ] Performance must be within 2x baseline (4-6s -> <12s acceptable, <10s target)

---

## 10. Next Steps After Phase 4

Once Phase 4 is complete and tested:
- **Phase 5**: Implement validation rules (FVG invalidation, block validation, TP/SL/BE validation)
- **Phase 6**: Integrate SMC into live bot (`src/strategies/ib_strategy_smc.py`)
- **Phase 7**: Production deployment on VPS with monitoring

---

## Appendix A: File Modifications Summary

| File | Change Type | Description |
|------|-------------|-------------|
| `params_optimizer/engine/fast_backtest.py` | MODIFY | Add `_build_smc_context()`, `_apply_smc_filter_batch()`, `_scan_for_smc_confirmation()`, helper methods |
| `src/smc/models.py` | NEW | Add `SMCDayContext` dataclass |
| `src/smc/config.py` | NEW | Add `SMCConfig` dataclass |
| `src/smc/detectors/fractal_detector.py` | NEW | Vectorized fractal detection |
| `src/smc/detectors/fvg_detector.py` | NEW | Vectorized FVG detection |
| `src/smc/detectors/cisd_detector.py` | NEW | Vectorized CISD detection |
| `src/smc/detectors/market_structure_detector.py` | NEW | Vectorized BOS/CHoCH detection |
| `tests/test_smc_parity.py` | NEW | Parity testing script |

---

## Appendix B: Example SMC Filter Flow

```
Day: 2024-01-15 (GER40)

1. _build_smc_context():
   - Load lookback: 2024-01-13 18:00 -> 2024-01-14 23:59 (48h)
   - Detect H1 fractals: 3 high fractals, 2 low fractals
   - Detect M2 FVGs: 5 bullish FVGs, 3 bearish FVGs
   - Detect BOS: 1 bullish BOS at 09:30
   - Detect CISD: 2 CISD events at 10:15 and 11:00
   - Time: 850ms

2. Signal Detection:
   - OCAE long signal at 10:30, entry_price=16500

3. _apply_smc_filter_batch():
   - Check opposing fractals: None found
   - Check FVGs in entry zone: 1 bullish FVG at 16490-16510 (supports signal)
   - Result: PASS (signal unchanged)

4. Trade Execution:
   - Entry: 16500, SL: 16450, TP: 16650
   - Result: +3R (TP hit)
```

---

**END OF PHASE_4_FAST_BACKTEST.md**
