"""
SMC data models - all dataclasses for structures and decisions.
"""

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


def make_id(prefix: str, instrument: str, timeframe: str, time: datetime, extra: str = "") -> str:
    """Generate deterministic ID for SMC structures.

    Format: {prefix}_{hash8}
    Hash from: instrument + timeframe + time_iso + extra
    """
    raw = f"{instrument}:{timeframe}:{time.isoformat()}:{extra}"
    h = hashlib.md5(raw.encode()).hexdigest()[:8]
    return f"{prefix}_{h}"


# ================================
# SMC Structures
# ================================

@dataclass
class FVG:
    """Fair Value Gap - 3-candle price gap."""
    id: str
    instrument: str
    timeframe: str
    direction: str                        # "bullish" / "bearish"
    high: float                           # Upper boundary of gap
    low: float                            # Lower boundary of gap
    midpoint: float                       # (high + low) / 2
    formation_time: datetime              # 3rd candle time
    candle1_time: datetime
    candle2_time: datetime
    candle3_time: datetime
    status: str = "active"                # active / partial_fill / full_fill / invalidated / inverted
    fill_pct: float = 0.0
    fill_time: Optional[datetime] = None
    invalidation_time: Optional[datetime] = None
    invalidation_reason: Optional[str] = None


@dataclass
class Fractal:
    """Williams 3-bar fractal."""
    id: str
    instrument: str
    timeframe: str
    type: str                             # "high" / "low"
    price: float
    time: datetime                        # Center bar time
    confirmed_time: datetime              # After confirming bar closes
    status: str = "active"                # active / swept
    swept: bool = False
    sweep_time: Optional[datetime] = None
    sweep_price: Optional[float] = None


@dataclass
class CISD:
    """Change In State of Delivery."""
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


@dataclass
class BOS:
    """Break of Structure."""
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
    """Single swing point in market structure."""
    time: datetime
    price: float
    type: str                             # "HH" / "HL" / "LL" / "LH"
    is_key: bool = False


@dataclass
class MarketStructure:
    """Current market structure state."""
    instrument: str
    timeframe: str
    swing_points: List[StructurePoint] = field(default_factory=list)
    current_trend: str = "ranging"        # "uptrend" / "downtrend" / "ranging"
    last_update: Optional[datetime] = None


# ================================
# Decision Objects
# ================================

@dataclass
class ConfirmationCriteria:
    """What must happen for a WAIT decision to become ENTER."""
    type: str                             # "CISD" / "FVG_REBALANCE" / "BOS"
    timeframe: str
    direction: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SMCDecision:
    """Result of evaluating a signal through SMC context."""
    action: str                           # "ENTER" / "WAIT" / "REJECT"
    reason: str
    modified_signal: Optional[Any] = None
    confirmation_criteria: List[ConfirmationCriteria] = field(default_factory=list)
    confluence_score: float = 0.0
    timeout_minutes: int = 30


@dataclass
class ConfluenceScore:
    """Weighted confluence from active structures."""
    total: float
    breakdown: Dict[str, float] = field(default_factory=dict)
    direction_bias: str = "neutral"       # "long" / "short" / "neutral"
    contributing_structures: List[str] = field(default_factory=list)


# ================================
# Fast Backtest Context
# ================================

@dataclass
class SMCDayContext:
    """
    Pre-computed SMC structures for a single trading day.

    Built once per day in fast backtest. Used by SMC filter for signal
    evaluation without redundant re-computation.
    """
    instrument: str
    day_date: Any                         # datetime.date
    build_time: Optional[datetime] = None

    # Fractals
    fractals_high: List[Fractal] = field(default_factory=list)
    fractals_low: List[Fractal] = field(default_factory=list)

    # FVGs by timeframe
    fvgs_m2: List[FVG] = field(default_factory=list)
    fvgs_h1: List[FVG] = field(default_factory=list)

    # BOS and CISD events
    bos_events: List[BOS] = field(default_factory=list)
    cisd_events: List[CISD] = field(default_factory=list)

    # Cached price levels (sorted, for fast lookup)
    active_fractal_highs: List[float] = field(default_factory=list)
    active_fractal_lows: List[float] = field(default_factory=list)
    active_fvg_zones: List[Dict[str, Any]] = field(default_factory=list)

    # Counts
    num_fractals: int = 0
    num_fvgs: int = 0
    num_bos: int = 0
    num_cisds: int = 0


# ================================
# Event Log
# ================================

@dataclass
class SMCEvent:
    """Single chronological event in the SMC event log."""
    timestamp: datetime
    instrument: str
    event_type: str
    timeframe: str
    direction: Optional[str] = None
    price: Optional[float] = None
    structure_id: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    bias_impact: float = 0.0
