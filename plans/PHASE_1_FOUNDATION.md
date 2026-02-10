# Phase 1: Foundation + First Detectors

## Goal

Build the SMC engine infrastructure and two detectors (fractals + FVG). After this phase, we can detect FVGs and fractals on historical data, store them in a registry, and analyze them with a standalone tool.

## Dependencies

- Existing: `strategy_optimization/fractals/fractals.py` (port to new location)
- Existing: M1 data in `data/` folder (CSV files)
- No changes to existing strategy code in this phase

---

## 1. `src/smc/__init__.py`

```python
"""
SMC (Smart Money Concepts) Engine for dual_v4.

Three-layer architecture:
- Detectors: pure functions that scan DataFrames for SMC structures
- Registry: in-memory state tracking for active/invalidated structures
- Engine: orchestrator that ties detectors + registry + decision logic
"""
```

## 2. `src/smc/models.py`

All data models for SMC structures and decisions. Every model uses `@dataclass` for serialization and type safety.

### 2.1 Structure ID Generation

```python
import hashlib
from datetime import datetime

def make_id(prefix: str, instrument: str, timeframe: str, time: datetime, extra: str = "") -> str:
    """Generate deterministic ID for SMC structures.

    Format: {prefix}_{hash8}
    Hash from: instrument + timeframe + time_iso + extra
    """
    raw = f"{instrument}:{timeframe}:{time.isoformat()}:{extra}"
    h = hashlib.md5(raw.encode()).hexdigest()[:8]
    return f"{prefix}_{h}"
```

### 2.2 FVG

```python
@dataclass
class FVG:
    id: str
    instrument: str
    timeframe: str                        # "M2", "M5", "H1", etc.
    direction: str                        # "bullish" / "bearish"
    high: float                           # Upper boundary of gap
    low: float                            # Lower boundary of gap
    midpoint: float                       # (high + low) / 2 -- Fair Value Level
    formation_time: datetime              # Timestamp of 3rd candle close
    candle1_time: datetime                # 1st candle time (for reference)
    candle2_time: datetime                # 2nd candle time (gap candle)
    candle3_time: datetime                # 3rd candle time (confirming)
    status: str = "active"                # active / partial_fill / full_fill / invalidated / inverted
    fill_pct: float = 0.0                 # How much filled (0.0 - 1.0)
    fill_time: Optional[datetime] = None
    invalidation_time: Optional[datetime] = None
    invalidation_reason: Optional[str] = None
```

**FVG Detection Algorithm** (from notion_export/FVG.md):

Three consecutive candles. Bullish FVG:
- Candle 1 (before): any
- Candle 2 (middle): strong bullish move
- Candle 3 (after): any
- Gap condition: `candle1.high < candle3.low` (the wicks don't overlap)
- FVG zone: from `candle1.high` to `candle3.low`

Bearish FVG:
- Gap condition: `candle1.low > candle3.high`
- FVG zone: from `candle3.high` to `candle1.low`

### 2.3 Fractal

```python
@dataclass
class Fractal:
    id: str
    instrument: str
    timeframe: str                        # "H1", "H4"
    type: str                             # "high" / "low"
    price: float                          # Fractal level
    time: datetime                        # Center bar time
    confirmed_time: datetime              # After confirming (3rd) bar closes
    status: str = "active"                # active / swept
    swept: bool = False
    sweep_time: Optional[datetime] = None
    sweep_price: Optional[float] = None
```

### 2.4 CISD (defined here, implemented in Phase 2)

```python
@dataclass
class CISD:
    id: str
    instrument: str
    timeframe: str
    direction: str                        # "long" / "short"
    delivery_candle_time: datetime
    delivery_candle_body_high: float      # max(open, close)
    delivery_candle_body_low: float       # min(open, close)
    confirmation_time: datetime
    confirmation_close: float
    status: str = "active"                # active / invalidated
```

### 2.5 BOS (defined here, implemented in Phase 2)

```python
@dataclass
class BOS:
    id: str
    instrument: str
    timeframe: str
    direction: str                        # "bullish" / "bearish"
    broken_level: float
    break_time: datetime
    break_candle_close: float
    bos_type: str                         # "bos" / "choch" / "mss"
    displacement: bool = False

@dataclass
class StructurePoint:
    time: datetime
    price: float
    type: str                             # "HH" / "HL" / "LL" / "LH"
    is_key: bool = False

@dataclass
class MarketStructure:
    instrument: str
    timeframe: str
    swing_points: List[StructurePoint]
    current_trend: str                    # "uptrend" / "downtrend" / "ranging"
    last_update: datetime
```

### 2.6 Decision Objects (defined here, used in Phase 2+)

```python
@dataclass
class SMCDecision:
    action: str                           # "ENTER" / "WAIT" / "REJECT"
    reason: str
    modified_signal: Optional[Any] = None # ModifiedSignal (avoids circular import)
    confirmation_criteria: List['ConfirmationCriteria'] = field(default_factory=list)
    confluence_score: float = 0.0
    timeout_minutes: int = 30

@dataclass
class ConfirmationCriteria:
    type: str                             # "CISD" / "FVG_REBALANCE" / "BOS"
    timeframe: str
    direction: str
    params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ConfluenceScore:
    total: float
    breakdown: Dict[str, float]
    direction_bias: str                   # "long" / "short" / "neutral"
    contributing_structures: List[str]    # Structure IDs
```

### 2.7 Event Model

```python
@dataclass
class SMCEvent:
    timestamp: datetime
    instrument: str
    event_type: str
    timeframe: str
    direction: Optional[str] = None
    price: Optional[float] = None
    structure_id: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    bias_impact: float = 0.0
```

---

## 3. `src/smc/config.py`

```python
from dataclasses import dataclass, field
from typing import List

@dataclass
class SMCConfig:
    """Configuration for SMC engine. One per instrument."""

    instrument: str                       # "GER40" / "XAUUSD"

    # --- Feature flags ---
    enable_fractals: bool = True
    enable_fvg: bool = True
    enable_cisd: bool = False             # Phase 2
    enable_bos: bool = False              # Phase 2

    # --- Fractal params ---
    fractal_timeframe: str = "H1"
    fractal_lookback_hours: int = 48      # How far back to look for unswept fractals
    fractal_proximity_pct: float = 0.002  # 0.2% of price = "near" threshold

    # --- FVG params ---
    fvg_timeframes: List[str] = field(default_factory=lambda: ["M2", "H1"])
    fvg_min_size_points: float = 0.0      # Min gap size in price points (0 = any)

    # --- CISD params (Phase 2) ---
    cisd_timeframes: List[str] = field(default_factory=lambda: ["M2"])

    # --- BOS params (Phase 2) ---
    bos_timeframes: List[str] = field(default_factory=lambda: ["M5", "M15"])

    # --- Confluence (Phase 2) ---
    weight_fractal: float = 1.0
    weight_fvg: float = 1.5
    weight_cisd: float = 2.0
    weight_bos: float = 1.5
    min_confluence_score: float = 2.0

    # --- Confirmation (Phase 3) ---
    max_wait_minutes: int = 30

    # --- Timeframes for multi-TF context ---
    context_tfs: List[str] = field(default_factory=lambda: ["M2", "M5", "H1"])

    # --- Registry cleanup ---
    max_structure_age_hours: int = 72     # Remove structures older than this
```

---

## 4. `src/smc/timeframe_manager.py`

Handles resampling M1 data to any higher timeframe and caches results.

```python
class TimeframeManager:
    """Manages multi-timeframe data from M1 source.

    Supports: M1, M2, M5, M15, M30, H1, H4, D1
    """

    # Mapping from TF name to pandas resample rule
    TF_RULES = {
        "M1": "1min",
        "M2": "2min",
        "M5": "5min",
        "M15": "15min",
        "M30": "30min",
        "H1": "1h",
        "H4": "4h",
        "D1": "1D",
    }

    def __init__(self, m1_data: pd.DataFrame, instrument: str):
        """
        Args:
            m1_data: DataFrame with columns [time, open, high, low, close]
                     time must be timezone-aware or naive UTC
            instrument: "GER40" or "XAUUSD"
        """
        self.instrument = instrument
        self._m1_data = m1_data.copy()
        self._cache: Dict[str, pd.DataFrame] = {}

    def get_data(self, timeframe: str, up_to: Optional[datetime] = None) -> pd.DataFrame:
        """Get OHLC data for the requested timeframe.

        Args:
            timeframe: One of TF_RULES keys
            up_to: If set, return only COMPLETED candles before this time

        Returns:
            DataFrame with [time, open, high, low, close] columns.
            Only includes COMPLETED candles (not the current forming candle).
        """
        if timeframe == "M1":
            df = self._m1_data
        elif timeframe in self._cache:
            df = self._cache[timeframe]
        else:
            df = self._resample(timeframe)
            self._cache[timeframe] = df

        if up_to is not None:
            # Only completed candles: time + candle_duration <= up_to
            return df[df["time"] < up_to].copy()
        return df.copy()

    def get_last_n_candles(self, timeframe: str, n: int, before: datetime) -> pd.DataFrame:
        """Get last N completed candles before given time."""
        df = self.get_data(timeframe, up_to=before)
        return df.tail(n).copy()

    def append_m1(self, new_bars: pd.DataFrame):
        """Append new M1 bars and invalidate cache.

        For incremental updates in slow backtest / live trading.
        """
        self._m1_data = pd.concat([self._m1_data, new_bars]).drop_duplicates(subset=["time"])
        self._m1_data = self._m1_data.sort_values("time").reset_index(drop=True)
        self._cache.clear()  # Force re-resample on next access

    def _resample(self, timeframe: str) -> pd.DataFrame:
        """Resample M1 to target timeframe using standard OHLC aggregation."""
        rule = self.TF_RULES[timeframe]
        df = self._m1_data.set_index("time")

        agg = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
        }
        if "volume" in df.columns:
            agg["volume"] = "sum"

        resampled = df.resample(rule).agg(agg).dropna()
        return resampled.reset_index()
```

**Key behaviors:**
- `get_data(up_to=...)` returns only COMPLETED candles (critical for avoiding look-ahead bias)
- Cache invalidated on `append_m1()` (simple but correct; can optimize later)
- Port of `resample_m1_to_h1()` from `strategy_optimization/fractals/fractals.py` but generalized

---

## 5. `src/smc/registry.py`

```python
class SMCRegistry:
    """In-memory registry of all SMC structures for one instrument.

    Supports:
    - Add new structures
    - Query active structures by type/timeframe/direction
    - Update status (active -> partial_fill -> invalidated, etc.)
    - Cleanup old structures
    """

    def __init__(self, instrument: str):
        self.instrument = instrument
        # {structure_type: {structure_id: structure_object}}
        self._store: Dict[str, Dict[str, Any]] = {
            "fvg": {},
            "fractal": {},
            "cisd": {},
            "bos": {},
        }

    def add(self, structure_type: str, structure) -> None:
        """Add a structure to the registry."""
        self._store[structure_type][structure.id] = structure

    def get_by_id(self, structure_type: str, structure_id: str) -> Optional[Any]:
        """Get structure by ID."""
        return self._store.get(structure_type, {}).get(structure_id)

    def get_active(
        self,
        structure_type: str,
        timeframe: Optional[str] = None,
        direction: Optional[str] = None,
        before_time: Optional[datetime] = None,
    ) -> List[Any]:
        """Query active structures with optional filters.

        Args:
            structure_type: "fvg", "fractal", "cisd", "bos"
            timeframe: Filter by timeframe (e.g., "H1")
            direction: Filter by direction (e.g., "bullish")
            before_time: Only structures formed before this time

        Returns:
            List of matching structures with status="active"
        """
        results = []
        for s in self._store.get(structure_type, {}).values():
            if s.status != "active":
                continue
            if timeframe and s.timeframe != timeframe:
                continue
            if direction and s.direction != direction:
                continue
            if before_time:
                # Use formation_time for FVG, confirmed_time for Fractal
                t = getattr(s, "formation_time", None) or getattr(s, "confirmed_time", None)
                if t and t > before_time:
                    continue
            results.append(s)
        return results

    def update_status(self, structure_type: str, structure_id: str,
                      new_status: str, **kwargs) -> bool:
        """Update a structure's status and optional fields.

        kwargs can include: fill_pct, fill_time, invalidation_time,
                           invalidation_reason, swept, sweep_time, etc.
        Returns True if found and updated.
        """
        s = self.get_by_id(structure_type, structure_id)
        if s is None:
            return False
        # Replace with updated copy (dataclasses are mutable)
        for key, value in kwargs.items():
            if hasattr(s, key):
                object.__setattr__(s, key, value)
        object.__setattr__(s, "status", new_status)
        return True

    def get_unswept_fractals(
        self,
        timeframe: str,
        before_time: Optional[datetime] = None,
    ) -> List['Fractal']:
        """Convenience: get all active (unswept) fractals."""
        return [f for f in self.get_active("fractal", timeframe=timeframe, before_time=before_time)
                if not f.swept]

    def get_fvgs_near_price(
        self,
        timeframe: str,
        price: float,
        max_distance_pct: float = 0.005,
        direction: Optional[str] = None,
    ) -> List['FVG']:
        """Get active FVGs within distance of price.

        Args:
            price: Current price
            max_distance_pct: Max distance as fraction of price (0.005 = 0.5%)
        """
        threshold = price * max_distance_pct
        results = []
        for fvg in self.get_active("fvg", timeframe=timeframe, direction=direction):
            # Distance = min distance from price to FVG zone
            if fvg.low <= price <= fvg.high:
                dist = 0  # Price is inside FVG
            else:
                dist = min(abs(price - fvg.high), abs(price - fvg.low))
            if dist <= threshold:
                results.append(fvg)
        return results

    def cleanup(self, before_time: datetime) -> int:
        """Remove structures formed before given time. Returns count removed."""
        removed = 0
        for stype in self._store:
            to_remove = []
            for sid, s in self._store[stype].items():
                t = getattr(s, "formation_time", None) or getattr(s, "confirmed_time", None) or getattr(s, "break_time", None)
                if t and t < before_time:
                    to_remove.append(sid)
            for sid in to_remove:
                del self._store[stype][sid]
                removed += 1
        return removed

    def count(self, structure_type: Optional[str] = None) -> int:
        """Count structures in registry."""
        if structure_type:
            return len(self._store.get(structure_type, {}))
        return sum(len(v) for v in self._store.values())

    def to_dataframe(self, structure_type: str) -> pd.DataFrame:
        """Export structures as DataFrame for analysis."""
        from dataclasses import asdict
        records = [asdict(s) for s in self._store.get(structure_type, {}).values()]
        return pd.DataFrame(records)
```

**Key behaviors:**
- Mutable dataclasses for in-place status updates (performance over frozen)
- Query API for common access patterns (active by type, fractals unswept, FVGs near price)
- `to_dataframe()` for post-analysis export
- `cleanup()` to prevent memory bloat during long backtests

---

## 6. `src/smc/event_log.py`

```python
class SMCEventLog:
    """Chronological audit trail of all SMC events.

    Append-only during execution. Queryable for debugging.
    Exportable to CSV/DataFrame for analysis.
    """

    def __init__(self):
        self.events: List[SMCEvent] = []

    def record(self, event: SMCEvent) -> None:
        """Record a new event."""
        self.events.append(event)

    def record_simple(
        self,
        timestamp: datetime,
        instrument: str,
        event_type: str,
        timeframe: str,
        direction: Optional[str] = None,
        price: Optional[float] = None,
        structure_id: Optional[str] = None,
        bias_impact: float = 0.0,
        **details,
    ) -> None:
        """Convenience method to record an event without constructing SMCEvent."""
        self.record(SMCEvent(
            timestamp=timestamp,
            instrument=instrument,
            event_type=event_type,
            timeframe=timeframe,
            direction=direction,
            price=price,
            structure_id=structure_id,
            details=details,
            bias_impact=bias_impact,
        ))

    def get_events(
        self,
        event_type: Optional[str] = None,
        after: Optional[datetime] = None,
        before: Optional[datetime] = None,
        structure_id: Optional[str] = None,
    ) -> List[SMCEvent]:
        """Query events with filters."""
        result = self.events
        if event_type:
            result = [e for e in result if e.event_type == event_type]
        if after:
            result = [e for e in result if e.timestamp >= after]
        if before:
            result = [e for e in result if e.timestamp < before]
        if structure_id:
            result = [e for e in result if e.structure_id == structure_id]
        return result

    def get_bias_at_time(self, time: datetime) -> float:
        """Sum of bias_impact of all events up to given time.
        Positive = long bias, negative = short bias."""
        return sum(e.bias_impact for e in self.events if e.timestamp <= time)

    def to_dataframe(self) -> pd.DataFrame:
        """Export as DataFrame."""
        from dataclasses import asdict
        return pd.DataFrame([asdict(e) for e in self.events])

    def to_csv(self, path: str) -> None:
        """Export to CSV file."""
        self.to_dataframe().to_csv(path, index=False)

    def clear(self) -> None:
        """Clear all events (for daily reset in backtest)."""
        self.events.clear()

    def __len__(self) -> int:
        return len(self.events)
```

---

## 7. `src/smc/detectors/__init__.py`

```python
"""
SMC Detectors - Pure functions for detecting market structures.

Each detector:
- Takes a DataFrame (OHLC) and configuration params
- Returns a list of detected structures (dataclasses from models.py)
- Has NO side effects and NO state
- Is fully testable with synthetic data
"""
from .fractal_detector import detect_fractals, find_unswept_fractals
from .fvg_detector import detect_fvg
```

---

## 8. `src/smc/detectors/fractal_detector.py`

Port from `strategy_optimization/fractals/fractals.py`. Changes:
- Return `Fractal` dataclass objects (not DataFrame rows)
- Accept instrument parameter for ID generation
- Generalize timeframe (not hardcoded H1)

```python
"""
Fractal detector - Williams 3-bar fractals.

Ported from strategy_optimization/fractals/fractals.py.
Returns Fractal dataclass objects instead of DataFrame rows.
"""

from datetime import datetime, timedelta
from typing import List, Optional
import pandas as pd

from ..models import Fractal, make_id


def detect_fractals(
    ohlc_data: pd.DataFrame,
    instrument: str,
    timeframe: str,
    candle_duration_hours: float = 1.0,
) -> List[Fractal]:
    """
    Detect Williams 3-bar fractals.

    High fractal: center bar high > prev high AND next high
    Low fractal: center bar low < prev low AND next low

    Args:
        ohlc_data: DataFrame with [time, open, high, low, close]
        instrument: "GER40" / "XAUUSD"
        timeframe: "H1", "H4", etc.
        candle_duration_hours: Duration of one candle (1.0 for H1, 4.0 for H4)

    Returns:
        List of Fractal objects. Each fractal is confirmed after the 3rd bar closes.
    """
    if len(ohlc_data) < 3:
        return []

    fractals = []

    for i in range(1, len(ohlc_data) - 1):
        curr = ohlc_data.iloc[i]
        prev = ohlc_data.iloc[i - 1]
        next_ = ohlc_data.iloc[i + 1]

        confirmed_time = next_["time"] + timedelta(hours=candle_duration_hours)

        # High fractal
        if curr["high"] > prev["high"] and curr["high"] > next_["high"]:
            frac = Fractal(
                id=make_id("frac", instrument, timeframe, curr["time"], f"high_{curr['high']}"),
                instrument=instrument,
                timeframe=timeframe,
                type="high",
                price=curr["high"],
                time=curr["time"],
                confirmed_time=confirmed_time,
            )
            fractals.append(frac)

        # Low fractal
        if curr["low"] < prev["low"] and curr["low"] < next_["low"]:
            frac = Fractal(
                id=make_id("frac", instrument, timeframe, curr["time"], f"low_{curr['low']}"),
                instrument=instrument,
                timeframe=timeframe,
                type="low",
                price=curr["low"],
                time=curr["time"],
                confirmed_time=confirmed_time,
            )
            fractals.append(frac)

    return fractals


def find_unswept_fractals(
    fractals: List[Fractal],
    m1_data: pd.DataFrame,
    before_time: datetime,
    lookback_hours: int = 48,
) -> List[Fractal]:
    """
    Filter fractals to only those NOT swept by price before given time.

    A fractal is "unswept" if no M1 bar after its confirmation touched
    the fractal level before the specified time.

    Args:
        fractals: List of Fractal objects from detect_fractals()
        m1_data: M1 OHLC data for sweep checking
        before_time: Check sweep status as of this time
        lookback_hours: Only consider fractals within this many hours

    Returns:
        List of unswept Fractal objects
    """
    lookback_start = before_time - timedelta(hours=lookback_hours)

    candidates = [
        f for f in fractals
        if f.time >= lookback_start
        and f.time < before_time
        and f.confirmed_time <= before_time
    ]

    if not candidates:
        return []

    unswept = []

    for frac in candidates:
        check_mask = (m1_data["time"] >= frac.confirmed_time) & (m1_data["time"] < before_time)
        check_data = m1_data[check_mask]

        swept = False
        if not check_data.empty:
            if frac.type == "high":
                swept = (check_data["high"] >= frac.price).any()
            else:
                swept = (check_data["low"] <= frac.price).any()

        if not swept:
            unswept.append(frac)

    return unswept


def check_fractal_sweep(
    fractal: Fractal,
    m1_data: pd.DataFrame,
    window_start: datetime,
    window_end: datetime,
) -> Optional[datetime]:
    """
    Check if a fractal was swept during a specific time window.

    Args:
        fractal: Fractal to check
        m1_data: M1 OHLC data
        window_start: Start of window
        window_end: End of window

    Returns:
        Sweep time (first M1 bar that touched the level) or None
    """
    mask = (m1_data["time"] >= window_start) & (m1_data["time"] <= window_end)
    window_data = m1_data[mask]

    if window_data.empty:
        return None

    if fractal.type == "high":
        sweep_bars = window_data[window_data["high"] >= fractal.price]
    else:
        sweep_bars = window_data[window_data["low"] <= fractal.price]

    if not sweep_bars.empty:
        return sweep_bars.iloc[0]["time"]
    return None
```

---

## 9. `src/smc/detectors/fvg_detector.py`

New detector implementing FVG detection from notion_export/FVG.md.

```python
"""
FVG (Fair Value Gap) detector.

FVG = 3-candle formation where middle candle body is NOT overlapped
by wicks of first and third candles.

Bullish FVG: candle1.high < candle3.low (gap up)
Bearish FVG: candle1.low > candle3.high (gap down)
"""

from datetime import datetime
from typing import List, Optional
import pandas as pd

from ..models import FVG, make_id


def detect_fvg(
    ohlc_data: pd.DataFrame,
    instrument: str,
    timeframe: str,
    direction: Optional[str] = None,
    min_size_points: float = 0.0,
) -> List[FVG]:
    """
    Detect Fair Value Gaps in OHLC data.

    Args:
        ohlc_data: DataFrame with [time, open, high, low, close].
                   Must contain at least 3 rows.
        instrument: "GER40" / "XAUUSD"
        timeframe: "M2", "M5", "H1", etc.
        direction: "bullish" / "bearish" / None (detect both)
        min_size_points: Minimum gap size in price points (0 = any)

    Returns:
        List of FVG objects. Formation time = 3rd candle time.
    """
    if len(ohlc_data) < 3:
        return []

    fvgs = []

    for i in range(1, len(ohlc_data) - 1):
        c1 = ohlc_data.iloc[i - 1]  # Before
        c2 = ohlc_data.iloc[i]      # Gap candle (middle)
        c3 = ohlc_data.iloc[i + 1]  # After

        # Bullish FVG: gap between c1.high and c3.low
        if direction is None or direction == "bullish":
            if c3["low"] > c1["high"]:
                gap_size = c3["low"] - c1["high"]
                if gap_size >= min_size_points:
                    fvg_low = c1["high"]
                    fvg_high = c3["low"]
                    fvg = FVG(
                        id=make_id("fvg", instrument, timeframe, c3["time"], f"bull_{fvg_low}_{fvg_high}"),
                        instrument=instrument,
                        timeframe=timeframe,
                        direction="bullish",
                        high=fvg_high,
                        low=fvg_low,
                        midpoint=(fvg_high + fvg_low) / 2,
                        formation_time=c3["time"],
                        candle1_time=c1["time"],
                        candle2_time=c2["time"],
                        candle3_time=c3["time"],
                    )
                    fvgs.append(fvg)

        # Bearish FVG: gap between c3.high and c1.low
        if direction is None or direction == "bearish":
            if c1["low"] > c3["high"]:
                gap_size = c1["low"] - c3["high"]
                if gap_size >= min_size_points:
                    fvg_high = c1["low"]
                    fvg_low = c3["high"]
                    fvg = FVG(
                        id=make_id("fvg", instrument, timeframe, c3["time"], f"bear_{fvg_low}_{fvg_high}"),
                        instrument=instrument,
                        timeframe=timeframe,
                        direction="bearish",
                        high=fvg_high,
                        low=fvg_low,
                        midpoint=(fvg_high + fvg_low) / 2,
                        formation_time=c3["time"],
                        candle1_time=c1["time"],
                        candle2_time=c2["time"],
                        candle3_time=c3["time"],
                    )
                    fvgs.append(fvg)

    return fvgs


def check_fvg_fill(
    fvg: FVG,
    m1_data: pd.DataFrame,
    after_time: datetime,
    up_to_time: datetime,
) -> Optional[dict]:
    """
    Check if/how an FVG has been filled by price action.

    Args:
        fvg: FVG to check
        m1_data: M1 OHLC data
        after_time: Start checking from (usually fvg.formation_time)
        up_to_time: Check until this time

    Returns:
        dict with {fill_pct, fill_type, fill_time} or None if untouched
        fill_type: "partial" (<100%) or "full" (100%)
    """
    mask = (m1_data["time"] > after_time) & (m1_data["time"] <= up_to_time)
    data = m1_data[mask]

    if data.empty:
        return None

    gap_size = fvg.high - fvg.low
    if gap_size <= 0:
        return None

    if fvg.direction == "bullish":
        # Bullish FVG filled from above: price drops into gap
        # Fill = how deep price goes into the gap (from high to low)
        min_low = data["low"].min()
        if min_low < fvg.high:
            penetration = fvg.high - max(min_low, fvg.low)
            fill_pct = min(penetration / gap_size, 1.0)
            fill_time = data[data["low"] <= fvg.high].iloc[0]["time"]
            return {
                "fill_pct": fill_pct,
                "fill_type": "full" if fill_pct >= 0.99 else "partial",
                "fill_time": fill_time,
            }
    else:
        # Bearish FVG filled from below: price rises into gap
        max_high = data["high"].max()
        if max_high > fvg.low:
            penetration = min(max_high, fvg.high) - fvg.low
            fill_pct = min(penetration / gap_size, 1.0)
            fill_time = data[data["high"] >= fvg.low].iloc[0]["time"]
            return {
                "fill_pct": fill_pct,
                "fill_type": "full" if fill_pct >= 0.99 else "partial",
                "fill_time": fill_time,
            }

    return None


def check_fvg_rebalance(
    fvg: FVG,
    candle: pd.Series,
) -> bool:
    """
    Check if a candle has "rebalanced" an FVG (entered + closed on the correct side).

    For bullish FVG: candle low entered FVG zone AND candle closed above FVG high
    For bearish FVG: candle high entered FVG zone AND candle closed below FVG low

    Args:
        fvg: FVG to check
        candle: Single OHLC row (pd.Series with high, low, close)

    Returns:
        True if this candle rebalanced the FVG
    """
    if fvg.direction == "bullish":
        # Price dipped into FVG (low <= fvg.high) and closed above (close > fvg.high)
        entered = candle["low"] <= fvg.high
        closed_above = candle["close"] > fvg.high
        return entered and closed_above
    else:
        # Price rose into FVG (high >= fvg.low) and closed below (close < fvg.low)
        entered = candle["high"] >= fvg.low
        closed_below = candle["close"] < fvg.low
        return entered and closed_below
```

---

## 10. `strategy_optimization/smc_tools/smc_analyzer.py`

Standalone analysis tool for validating detectors on historical data.

```python
"""
SMC Analyzer - Standalone tool for analyzing SMC structures on historical data.

Usage:
    analyzer = SMCAnalyzer("GER40")
    analyzer.load_data("2024-06-01", "2024-06-30")
    report = analyzer.analyze_day("2024-06-15")
    analyzer.print_report(report)
"""

import sys
import pandas as pd
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional
from pathlib import Path

# Add parent path
DUAL_V4_PATH = Path("C:/Trading/ib_trading_bot/dual_v4")
sys.path.insert(0, str(DUAL_V4_PATH))

from src.smc.config import SMCConfig
from src.smc.timeframe_manager import TimeframeManager
from src.smc.detectors.fractal_detector import detect_fractals, find_unswept_fractals
from src.smc.detectors.fvg_detector import detect_fvg, check_fvg_fill
from src.smc.models import FVG, Fractal


class SMCAnalyzer:
    """Standalone SMC structure analysis on historical data."""

    def __init__(self, instrument: str, config: Optional[SMCConfig] = None):
        self.instrument = instrument
        self.config = config or SMCConfig(instrument=instrument)
        self.m1_data: Optional[pd.DataFrame] = None
        self.tfm: Optional[TimeframeManager] = None

    def load_data(self, start_date: str, end_date: str) -> None:
        """Load M1 data from CSV files for date range."""
        from backtest.config import DEFAULT_CONFIG, DATA_FOLDERS

        data_folder = DEFAULT_CONFIG.data_base_path / DATA_FOLDERS[self.instrument]
        # Load CSVs... (uses existing data_ingestor pattern)
        # self.m1_data = loaded_data
        # self.tfm = TimeframeManager(self.m1_data, self.instrument)
        pass

    def analyze_day(self, day_date: str) -> Dict:
        """Run SMC analysis for a single day.

        Returns dict with:
        - fractals_h1: List[Fractal] detected on H1
        - unswept_fractals: List[Fractal] unswept at IB start
        - fvgs_m2: List[FVG] detected on M2
        - fvgs_h1: List[FVG] detected on H1
        - summary: Dict with counts and statistics
        """
        pass

    def analyze_range(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Run analysis for date range. Returns summary DataFrame."""
        pass

    def print_report(self, report: Dict) -> None:
        """Print human-readable analysis report."""
        pass
```

Full implementation in Phase 1 execution. The skeleton shows the API.

---

## 11. Tests

### `tests/test_smc/test_detectors.py`

```python
"""Tests for SMC detectors."""
import pytest
import pandas as pd
from datetime import datetime, timedelta
from src.smc.detectors.fractal_detector import detect_fractals, find_unswept_fractals
from src.smc.detectors.fvg_detector import detect_fvg, check_fvg_fill, check_fvg_rebalance


class TestFractalDetector:
    """Test fractal detection with synthetic data."""

    def _make_h1_data(self, ohlc_list):
        """Create H1 DataFrame from list of (time, o, h, l, c) tuples."""
        return pd.DataFrame(ohlc_list, columns=["time", "open", "high", "low", "close"])

    def test_high_fractal_detected(self):
        """Center bar high > neighbors -> high fractal."""
        data = self._make_h1_data([
            (datetime(2024,1,1,10), 100, 105, 99, 103),   # prev
            (datetime(2024,1,1,11), 103, 110, 101, 107),  # center (highest high)
            (datetime(2024,1,1,12), 107, 108, 104, 106),  # next
        ])
        fractals = detect_fractals(data, "GER40", "H1")
        assert len(fractals) == 1
        assert fractals[0].type == "high"
        assert fractals[0].price == 110

    def test_low_fractal_detected(self):
        """Center bar low < neighbors -> low fractal."""
        data = self._make_h1_data([
            (datetime(2024,1,1,10), 100, 105, 98, 103),
            (datetime(2024,1,1,11), 103, 107, 95, 100),   # center (lowest low)
            (datetime(2024,1,1,12), 100, 104, 97, 102),
        ])
        fractals = detect_fractals(data, "GER40", "H1")
        assert len(fractals) == 1
        assert fractals[0].type == "low"
        assert fractals[0].price == 95

    def test_both_high_and_low_on_same_bar(self):
        """One bar can be both high and low fractal."""
        data = self._make_h1_data([
            (datetime(2024,1,1,10), 100, 105, 98, 103),
            (datetime(2024,1,1,11), 103, 115, 90, 100),   # Extreme bar
            (datetime(2024,1,1,12), 100, 104, 97, 102),
        ])
        fractals = detect_fractals(data, "GER40", "H1")
        assert len(fractals) == 2
        types = {f.type for f in fractals}
        assert types == {"high", "low"}

    def test_no_fractal_when_equal_highs(self):
        """Equal highs -> no high fractal (requires strictly greater)."""
        data = self._make_h1_data([
            (datetime(2024,1,1,10), 100, 110, 99, 103),
            (datetime(2024,1,1,11), 103, 110, 101, 107),  # Same high
            (datetime(2024,1,1,12), 107, 108, 104, 106),
        ])
        fractals = detect_fractals(data, "GER40", "H1")
        high_fractals = [f for f in fractals if f.type == "high"]
        assert len(high_fractals) == 0

    def test_confirmed_time(self):
        """Fractal confirmed after 3rd bar closes."""
        data = self._make_h1_data([
            (datetime(2024,1,1,10), 100, 105, 99, 103),
            (datetime(2024,1,1,11), 103, 110, 101, 107),
            (datetime(2024,1,1,12), 107, 108, 104, 106),
        ])
        fractals = detect_fractals(data, "GER40", "H1", candle_duration_hours=1.0)
        assert fractals[0].confirmed_time == datetime(2024, 1, 1, 13)


class TestFVGDetector:
    """Test FVG detection with synthetic data."""

    def _make_data(self, ohlc_list):
        return pd.DataFrame(ohlc_list, columns=["time", "open", "high", "low", "close"])

    def test_bullish_fvg_detected(self):
        """Gap between c1.high and c3.low -> bullish FVG."""
        data = self._make_data([
            (datetime(2024,1,1,10,0), 100, 102, 99, 101),   # c1: high=102
            (datetime(2024,1,1,10,2), 103, 108, 102, 107),  # c2: strong move up
            (datetime(2024,1,1,10,4), 107, 110, 105, 109),  # c3: low=105
        ])
        # Gap: c1.high=102, c3.low=105 -> FVG from 102 to 105
        fvgs = detect_fvg(data, "GER40", "M2")
        assert len(fvgs) == 1
        assert fvgs[0].direction == "bullish"
        assert fvgs[0].low == 102
        assert fvgs[0].high == 105

    def test_bearish_fvg_detected(self):
        """Gap between c3.high and c1.low -> bearish FVG."""
        data = self._make_data([
            (datetime(2024,1,1,10,0), 110, 112, 108, 109),  # c1: low=108
            (datetime(2024,1,1,10,2), 107, 108, 102, 103),  # c2: strong drop
            (datetime(2024,1,1,10,4), 103, 105, 101, 104),  # c3: high=105
        ])
        # Gap: c3.high=105, c1.low=108 -> FVG from 105 to 108
        fvgs = detect_fvg(data, "GER40", "M2")
        assert len(fvgs) == 1
        assert fvgs[0].direction == "bearish"
        assert fvgs[0].low == 105
        assert fvgs[0].high == 108

    def test_no_fvg_when_wicks_overlap(self):
        """If c1 and c3 wicks overlap, no FVG."""
        data = self._make_data([
            (datetime(2024,1,1,10,0), 100, 106, 99, 103),   # c1: high=106
            (datetime(2024,1,1,10,2), 103, 108, 102, 107),  # c2: move up
            (datetime(2024,1,1,10,4), 107, 110, 104, 109),  # c3: low=104
        ])
        # c1.high=106 > c3.low=104 -> wicks overlap -> no bullish FVG
        fvgs = detect_fvg(data, "GER40", "M2", direction="bullish")
        assert len(fvgs) == 0

    def test_fvg_rebalance_bullish(self):
        """Candle dips into bullish FVG and closes above."""
        from src.smc.models import FVG as FVGModel
        fvg = FVGModel(
            id="test", instrument="GER40", timeframe="M2",
            direction="bullish", high=105, low=102, midpoint=103.5,
            formation_time=datetime(2024,1,1,10,4),
            candle1_time=datetime(2024,1,1,10,0),
            candle2_time=datetime(2024,1,1,10,2),
            candle3_time=datetime(2024,1,1,10,4),
        )
        # Candle enters FVG (low=103 <= 105) and closes above (close=106 > 105)
        candle = pd.Series({"high": 108, "low": 103, "close": 106})
        assert check_fvg_rebalance(fvg, candle) is True

        # Candle enters but closes inside FVG (not above)
        candle_inside = pd.Series({"high": 108, "low": 103, "close": 104})
        assert check_fvg_rebalance(fvg, candle_inside) is False


class TestRegistry:
    """Test SMCRegistry operations."""

    def test_add_and_query(self):
        from src.smc.registry import SMCRegistry
        from src.smc.models import FVG

        reg = SMCRegistry("GER40")
        fvg = FVG(
            id="fvg_test1", instrument="GER40", timeframe="M2",
            direction="bullish", high=105, low=102, midpoint=103.5,
            formation_time=datetime(2024,1,1,10,4),
            candle1_time=datetime(2024,1,1,10,0),
            candle2_time=datetime(2024,1,1,10,2),
            candle3_time=datetime(2024,1,1,10,4),
        )
        reg.add("fvg", fvg)

        active = reg.get_active("fvg", timeframe="M2", direction="bullish")
        assert len(active) == 1
        assert active[0].id == "fvg_test1"

    def test_update_status(self):
        from src.smc.registry import SMCRegistry
        from src.smc.models import FVG

        reg = SMCRegistry("GER40")
        fvg = FVG(
            id="fvg_test2", instrument="GER40", timeframe="M2",
            direction="bullish", high=105, low=102, midpoint=103.5,
            formation_time=datetime(2024,1,1,10,4),
            candle1_time=datetime(2024,1,1,10,0),
            candle2_time=datetime(2024,1,1,10,2),
            candle3_time=datetime(2024,1,1,10,4),
        )
        reg.add("fvg", fvg)
        reg.update_status("fvg", "fvg_test2", "partial_fill", fill_pct=0.5)

        # No longer "active"
        active = reg.get_active("fvg")
        assert len(active) == 0
```

---

## 12. Execution Checklist

1. [ ] Create directory structure: `src/smc/`, `src/smc/detectors/`, `strategy_optimization/smc_tools/`, `tests/test_smc/`
2. [ ] Write `src/smc/__init__.py`
3. [ ] Write `src/smc/models.py` (all dataclasses)
4. [ ] Write `src/smc/config.py` (SMCConfig)
5. [ ] Write `src/smc/timeframe_manager.py` (TimeframeManager)
6. [ ] Write `src/smc/registry.py` (SMCRegistry)
7. [ ] Write `src/smc/event_log.py` (SMCEventLog)
8. [ ] Write `src/smc/detectors/__init__.py`
9. [ ] Write `src/smc/detectors/fractal_detector.py` (port + enhance)
10. [ ] Write `src/smc/detectors/fvg_detector.py` (new)
11. [ ] Write `strategy_optimization/smc_tools/__init__.py`
12. [ ] Write `strategy_optimization/smc_tools/smc_analyzer.py`
13. [ ] Write `tests/test_smc/test_detectors.py`
14. [ ] Write `tests/test_smc/test_registry.py`
15. [ ] Verify: `python -c "from src.smc.detectors import detect_fractals, detect_fvg; print('OK')"`
16. [ ] Run: `pytest tests/test_smc/ -v`
17. [ ] Verify: SMCAnalyzer can load data and detect structures for 1 day
