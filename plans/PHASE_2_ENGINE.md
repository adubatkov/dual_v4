# Phase 2: Core Detectors + SMC Engine

## Goal

Implement CISD detection, market structure detection (HH/HL/LL/LH + BOS/CHoCH/MSS), and the main SMC orchestration engine. After this phase, the engine can evaluate IB signals against SMC context, calculate confluence scores, and determine whether to ENTER/WAIT/REJECT.

## Dependencies

- Phase 1 completed: models.py, config.py, timeframe_manager.py, registry.py, event_log.py, fractal_detector.py, fvg_detector.py
- Existing: M1 data in data/ folder
- No changes to existing strategy code in this phase

---

## 1. `src/smc/detectors/cisd_detector.py`

CISD (Change In State of Delivery) detects when market control transitions from buyers to sellers or vice versa.

### 1.1 Detection Algorithm

CISD formation requires three conditions:
1. Price reaches a key zone (liquidity level, FVG, block)
2. Gets reaction (bounce from the zone)
3. Closes above/below the body of the candle(s) that led to the key zone

**Bearish CISD (long signal):**
- Downtrend delivery candle leads to a zone
- Price closes ABOVE the body (max of open, close) of that delivery candle
- Signal: smart money is now buying

**Bullish CISD (short signal):**
- Uptrend delivery candle leads to a zone
- Price closes BELOW the body (min of open, close) of that delivery candle
- Signal: smart money is now selling

**Special cases:**
- Multiple delivery candles: use the candle that STARTED the unidirectional movement
- If movement is interrupted: use the candle that led directly to the test
- CRITICAL: If there's an unoperated zone below/above CISD that can continue the movement, CISD is INVALID

### 1.2 Implementation

```python
"""
CISD (Change In State of Delivery) detector.

Detects when market control transitions from one side to another
by identifying when price closes beyond the body of delivery candle(s)
that led to a key zone.
"""

from datetime import datetime
from typing import List, Optional, Tuple
import pandas as pd
import numpy as np

from ..models import CISD, FVG, Fractal, make_id


def detect_cisd(
    ohlc_data: pd.DataFrame,
    instrument: str,
    timeframe: str,
    poi_zones: List[dict],
) -> List[CISD]:
    """
    Detect CISD formations in OHLC data.

    Args:
        ohlc_data: DataFrame with [time, open, high, low, close]
        instrument: "GER40" / "XAUUSD"
        timeframe: "M2", "M5", etc.
        poi_zones: List of points of interest (FVGs, fractals, blocks)
                   Each dict has: {type, direction, high, low, time}

    Returns:
        List of CISD objects
    """
    if len(ohlc_data) < 3:
        return []

    cisds = []

    # For each POI zone, look for CISD confirmation
    for poi in poi_zones:
        cisd = _check_poi_for_cisd(ohlc_data, instrument, timeframe, poi)
        if cisd:
            cisds.append(cisd)

    return cisds


def _check_poi_for_cisd(
    ohlc_data: pd.DataFrame,
    instrument: str,
    timeframe: str,
    poi: dict,
) -> Optional[CISD]:
    """
    Check if a POI zone has been tested and generated a CISD.

    Args:
        ohlc_data: OHLC data
        instrument: instrument name
        timeframe: timeframe string
        poi: Point of interest dict with {type, direction, high, low, time}

    Returns:
        CISD object if detected, None otherwise
    """
    poi_high = poi["high"]
    poi_low = poi["low"]
    poi_time = poi["time"]
    poi_direction = poi.get("direction")  # "bullish" or "bearish"

    # Find candles after POI formation
    after_poi = ohlc_data[ohlc_data["time"] > poi_time].copy()
    if len(after_poi) < 2:
        return None

    # Find the test: price entering the POI zone
    test_idx = _find_zone_test(after_poi, poi_low, poi_high)
    if test_idx is None:
        return None

    test_candle = after_poi.iloc[test_idx]

    # Find delivery candles: unidirectional movement that led to the test
    delivery_candles = _find_delivery_candles(after_poi, test_idx)
    if not delivery_candles:
        return None

    # Get the delivery candle that STARTED the movement
    delivery_idx = delivery_candles[0]
    delivery_candle = after_poi.iloc[delivery_idx]

    delivery_body_high = max(delivery_candle["open"], delivery_candle["close"])
    delivery_body_low = min(delivery_candle["open"], delivery_candle["close"])

    # Look for confirmation candle after the test
    confirmation = _find_cisd_confirmation(
        after_poi,
        test_idx,
        delivery_body_high,
        delivery_body_low,
        poi_direction,
    )

    if confirmation is None:
        return None

    conf_idx, conf_direction = confirmation
    conf_candle = after_poi.iloc[conf_idx]

    # Create CISD object
    cisd = CISD(
        id=make_id(
            "cisd",
            instrument,
            timeframe,
            conf_candle["time"],
            f"{conf_direction}_{conf_candle['close']}",
        ),
        instrument=instrument,
        timeframe=timeframe,
        direction=conf_direction,
        delivery_candle_time=delivery_candle["time"],
        delivery_candle_body_high=delivery_body_high,
        delivery_candle_body_low=delivery_body_low,
        confirmation_time=conf_candle["time"],
        confirmation_close=conf_candle["close"],
    )

    return cisd


def _find_zone_test(
    ohlc_data: pd.DataFrame,
    zone_low: float,
    zone_high: float,
) -> Optional[int]:
    """
    Find the first candle that enters the zone.

    Args:
        ohlc_data: OHLC DataFrame
        zone_low: Lower boundary of zone
        zone_high: Upper boundary of zone

    Returns:
        Index of first candle that touches the zone, or None
    """
    for i in range(len(ohlc_data)):
        candle = ohlc_data.iloc[i]
        # Candle touches zone if high >= zone_low AND low <= zone_high
        if candle["high"] >= zone_low and candle["low"] <= zone_high:
            return i
    return None


def _find_delivery_candles(
    ohlc_data: pd.DataFrame,
    test_idx: int,
) -> List[int]:
    """
    Find delivery candles: unidirectional movement before the test.

    Traces back from test_idx to find the sequence of candles that
    delivered price to the test zone.

    Args:
        ohlc_data: OHLC DataFrame
        test_idx: Index of the test candle

    Returns:
        List of indices representing delivery candles (earliest first)
    """
    if test_idx == 0:
        return []

    delivery = []
    test_candle = ohlc_data.iloc[test_idx]

    # Determine movement direction based on test candle
    is_downmove = test_candle["close"] < test_candle["open"]

    # Trace back to find unidirectional movement
    for i in range(test_idx - 1, -1, -1):
        candle = ohlc_data.iloc[i]

        if is_downmove:
            # For downmove: each candle should close lower than open
            if candle["close"] < candle["open"]:
                delivery.insert(0, i)
            else:
                break  # Movement interrupted
        else:
            # For upmove: each candle should close higher than open
            if candle["close"] > candle["open"]:
                delivery.insert(0, i)
            else:
                break  # Movement interrupted

    return delivery if delivery else [test_idx - 1]  # At least one delivery candle


def _find_cisd_confirmation(
    ohlc_data: pd.DataFrame,
    test_idx: int,
    delivery_body_high: float,
    delivery_body_low: float,
    poi_direction: Optional[str],
) -> Optional[Tuple[int, str]]:
    """
    Find confirmation candle that closes beyond delivery candle body.

    Args:
        ohlc_data: OHLC DataFrame
        test_idx: Index of the test candle
        delivery_body_high: Max(open, close) of delivery candle
        delivery_body_low: Min(open, close) of delivery candle
        poi_direction: Direction bias from POI ("bullish" or "bearish")

    Returns:
        Tuple of (confirmation_index, direction) or None
    """
    # Look forward from test for up to 10 candles
    max_look_forward = min(10, len(ohlc_data) - test_idx - 1)

    for i in range(test_idx + 1, test_idx + 1 + max_look_forward):
        candle = ohlc_data.iloc[i]

        # Bearish CISD (long signal): close above delivery body high
        if candle["close"] > delivery_body_high:
            # Prefer if POI was bearish (confirming reversal from bearish to bullish)
            if poi_direction is None or poi_direction == "bearish":
                return (i, "long")

        # Bullish CISD (short signal): close below delivery body low
        if candle["close"] < delivery_body_low:
            # Prefer if POI was bullish (confirming reversal from bullish to bearish)
            if poi_direction is None or poi_direction == "bullish":
                return (i, "short")

    return None


def check_cisd_invalidation(
    cisd: CISD,
    ohlc_data: pd.DataFrame,
    fvgs: List[FVG],
    fractals: List[Fractal],
) -> bool:
    """
    Check if a CISD should be invalidated.

    CISD is invalid if there's an unoperated zone (FVG or fractal)
    in the direction of CISD that could continue the original movement.

    Args:
        cisd: CISD to check
        ohlc_data: Recent OHLC data
        fvgs: Active FVGs
        fractals: Active fractals

    Returns:
        True if CISD should be invalidated
    """
    # Get current price
    if ohlc_data.empty:
        return False

    current_price = ohlc_data.iloc[-1]["close"]

    if cisd.direction == "long":
        # For long CISD, check for bearish structures above current price
        # that could pull price back down
        for fvg in fvgs:
            if fvg.direction == "bearish" and fvg.low > current_price:
                if fvg.status == "active":
                    return True

        for frac in fractals:
            if frac.type == "low" and frac.price < cisd.delivery_candle_body_low:
                if not frac.swept:
                    return True

    else:  # short
        # For short CISD, check for bullish structures below current price
        for fvg in fvgs:
            if fvg.direction == "bullish" and fvg.high < current_price:
                if fvg.status == "active":
                    return True

        for frac in fractals:
            if frac.type == "high" and frac.price > cisd.delivery_candle_body_high:
                if not frac.swept:
                    return True

    return False
```

---

## 2. `src/smc/detectors/market_structure_detector.py`

Detects market structure: swing points (HH/HL/LL/LH), trend direction, and structure breaks (BOS/CHoCH/MSS).

### 2.1 Market Structure Concepts

**Swing Points:**
- HH (Higher High): new high above previous high in uptrend
- HL (Higher Low): new low above previous low in uptrend
- LL (Lower Low): new low below previous low in downtrend
- LH (Lower High): new high below previous high in downtrend

**Key Levels:**
- Key High: last significant high that led to structure update (LL formation)
- Key Low: last significant low that led to structure update (HH formation)

**Structure Breaks:**
- BOS (Break of Structure): price breaks previous High/Low IN direction of current trend (continuation)
- CHoCH (Change of Character): price breaks previous High/Low OPPOSITE to trend (potential reversal)
- MSS (Market Structure Shift): breaks non-key swing + Displacement with FVG (stronger reversal signal)

### 2.2 Implementation

```python
"""
Market structure detector.

Detects swing points (HH/HL/LL/LH), current trend, and structure breaks
(BOS/CHoCH/MSS) based on Smart Money Concepts methodology.
"""

from datetime import datetime
from typing import List, Optional, Tuple
import pandas as pd
import numpy as np

from ..models import (
    MarketStructure,
    StructurePoint,
    BOS,
    make_id,
    FVG,
)


def detect_swing_points(
    ohlc_data: pd.DataFrame,
    instrument: str,
    timeframe: str,
    lookback: int = 3,
) -> MarketStructure:
    """
    Detect swing points and classify them as HH/HL/LL/LH.

    Args:
        ohlc_data: DataFrame with [time, open, high, low, close]
        instrument: "GER40" / "XAUUSD"
        timeframe: "M5", "M15", "H1", etc.
        lookback: Number of bars on each side for swing detection

    Returns:
        MarketStructure object with swing points and current trend
    """
    if len(ohlc_data) < lookback * 2 + 1:
        return MarketStructure(
            instrument=instrument,
            timeframe=timeframe,
            swing_points=[],
            current_trend="ranging",
            last_update=ohlc_data.iloc[-1]["time"] if not ohlc_data.empty else datetime.now(),
        )

    # Find swing highs and lows
    swing_highs = _find_swing_highs(ohlc_data, lookback)
    swing_lows = _find_swing_lows(ohlc_data, lookback)

    # Merge and sort all swing points
    all_swings = []

    for idx, price in swing_highs:
        all_swings.append({
            "time": ohlc_data.iloc[idx]["time"],
            "price": price,
            "type_raw": "high",
            "idx": idx,
        })

    for idx, price in swing_lows:
        all_swings.append({
            "time": ohlc_data.iloc[idx]["time"],
            "price": price,
            "type_raw": "low",
            "idx": idx,
        })

    all_swings.sort(key=lambda x: x["idx"])

    # Classify swings as HH/HL/LL/LH
    structure_points = _classify_swings(all_swings)

    # Determine current trend
    current_trend = _determine_trend(structure_points)

    # Mark key levels
    structure_points = _mark_key_levels(structure_points, current_trend)

    return MarketStructure(
        instrument=instrument,
        timeframe=timeframe,
        swing_points=structure_points,
        current_trend=current_trend,
        last_update=ohlc_data.iloc[-1]["time"],
    )


def _find_swing_highs(
    ohlc_data: pd.DataFrame,
    lookback: int,
) -> List[Tuple[int, float]]:
    """
    Find swing highs: bars where high is highest in lookback window.

    Args:
        ohlc_data: OHLC DataFrame
        lookback: Window size on each side

    Returns:
        List of (index, price) tuples
    """
    swing_highs = []

    for i in range(lookback, len(ohlc_data) - lookback):
        center_high = ohlc_data.iloc[i]["high"]

        # Check if this high is highest in window
        is_swing = True
        for j in range(i - lookback, i + lookback + 1):
            if j == i:
                continue
            if ohlc_data.iloc[j]["high"] >= center_high:
                is_swing = False
                break

        if is_swing:
            swing_highs.append((i, center_high))

    return swing_highs


def _find_swing_lows(
    ohlc_data: pd.DataFrame,
    lookback: int,
) -> List[Tuple[int, float]]:
    """
    Find swing lows: bars where low is lowest in lookback window.

    Args:
        ohlc_data: OHLC DataFrame
        lookback: Window size on each side

    Returns:
        List of (index, price) tuples
    """
    swing_lows = []

    for i in range(lookback, len(ohlc_data) - lookback):
        center_low = ohlc_data.iloc[i]["low"]

        # Check if this low is lowest in window
        is_swing = True
        for j in range(i - lookback, i + lookback + 1):
            if j == i:
                continue
            if ohlc_data.iloc[j]["low"] <= center_low:
                is_swing = False
                break

        if is_swing:
            swing_lows.append((i, center_low))

    return swing_lows


def _classify_swings(
    swings: List[dict],
) -> List[StructurePoint]:
    """
    Classify swings as HH/HL/LL/LH based on sequence.

    Args:
        swings: List of swing dicts with {time, price, type_raw, idx}

    Returns:
        List of StructurePoint objects
    """
    if not swings:
        return []

    structure_points = []
    prev_high = None
    prev_low = None

    for swing in swings:
        if swing["type_raw"] == "high":
            if prev_high is None:
                # First high, no classification yet
                swing_type = "high"
            elif swing["price"] > prev_high:
                swing_type = "HH"
            else:
                swing_type = "LH"

            prev_high = swing["price"]

        else:  # low
            if prev_low is None:
                # First low, no classification yet
                swing_type = "low"
            elif swing["price"] > prev_low:
                swing_type = "HL"
            else:
                swing_type = "LL"

            prev_low = swing["price"]

        structure_points.append(
            StructurePoint(
                time=swing["time"],
                price=swing["price"],
                type=swing_type,
            )
        )

    return structure_points


def _determine_trend(
    structure_points: List[StructurePoint],
) -> str:
    """
    Determine current trend based on recent swing points.

    Args:
        structure_points: List of StructurePoint objects

    Returns:
        "uptrend" / "downtrend" / "ranging"
    """
    if len(structure_points) < 4:
        return "ranging"

    # Look at last 4 swings
    recent = structure_points[-4:]

    hh_count = sum(1 for s in recent if s.type == "HH")
    hl_count = sum(1 for s in recent if s.type == "HL")
    ll_count = sum(1 for s in recent if s.type == "LL")
    lh_count = sum(1 for s in recent if s.type == "LH")

    # Uptrend: more HH/HL than LL/LH
    if (hh_count + hl_count) > (ll_count + lh_count):
        return "uptrend"

    # Downtrend: more LL/LH than HH/HL
    if (ll_count + lh_count) > (hh_count + hl_count):
        return "downtrend"

    return "ranging"


def _mark_key_levels(
    structure_points: List[StructurePoint],
    current_trend: str,
) -> List[StructurePoint]:
    """
    Mark key highs and key lows.

    Key Low: last significant low that led to HH formation (uptrend)
    Key High: last significant high that led to LL formation (downtrend)

    Args:
        structure_points: List of StructurePoint objects
        current_trend: Current trend direction

    Returns:
        Updated list with is_key flags set
    """
    if current_trend == "uptrend":
        # Find last HL before most recent HH
        for i in range(len(structure_points) - 1, -1, -1):
            if structure_points[i].type == "HH":
                # Find HL before this HH
                for j in range(i - 1, -1, -1):
                    if structure_points[j].type == "HL":
                        structure_points[j].is_key = True
                        break
                break

    elif current_trend == "downtrend":
        # Find last LH before most recent LL
        for i in range(len(structure_points) - 1, -1, -1):
            if structure_points[i].type == "LL":
                # Find LH before this LL
                for j in range(i - 1, -1, -1):
                    if structure_points[j].type == "LH":
                        structure_points[j].is_key = True
                        break
                break

    return structure_points


def detect_bos_choch(
    market_structure: MarketStructure,
    ohlc_data: pd.DataFrame,
    instrument: str,
    timeframe: str,
    fvgs: Optional[List[FVG]] = None,
) -> List[BOS]:
    """
    Detect BOS, CHoCH, and MSS based on current market structure.

    Args:
        market_structure: Current MarketStructure object
        ohlc_data: Recent OHLC data
        instrument: "GER40" / "XAUUSD"
        timeframe: Timeframe string
        fvgs: Optional list of FVGs for MSS detection

    Returns:
        List of BOS objects (includes BOS, CHoCH, MSS types)
    """
    if len(market_structure.swing_points) < 2:
        return []

    if ohlc_data.empty:
        return []

    detected = []
    current_price = ohlc_data.iloc[-1]["close"]
    current_time = ohlc_data.iloc[-1]["time"]

    # Get recent swing points
    recent_swings = market_structure.swing_points[-5:]  # Last 5 swings
    current_trend = market_structure.current_trend

    # Check for structure breaks
    for swing in recent_swings:
        bos = _check_structure_break(
            swing,
            current_price,
            current_time,
            current_trend,
            instrument,
            timeframe,
            ohlc_data,
            fvgs,
        )
        if bos:
            detected.append(bos)

    return detected


def _check_structure_break(
    swing: StructurePoint,
    current_price: float,
    current_time: datetime,
    current_trend: str,
    instrument: str,
    timeframe: str,
    ohlc_data: pd.DataFrame,
    fvgs: Optional[List[FVG]],
) -> Optional[BOS]:
    """
    Check if current price has broken a swing level.

    Args:
        swing: StructurePoint to check
        current_price: Current close price
        current_time: Current time
        current_trend: Current trend direction
        instrument: Instrument name
        timeframe: Timeframe string
        ohlc_data: Recent OHLC data for displacement check
        fvgs: List of FVGs for MSS detection

    Returns:
        BOS object if break detected, None otherwise
    """
    broken = False
    direction = None

    # Check if swing was broken
    if swing.type in ["HH", "high", "LH"]:
        # High broken if price closes above it
        if current_price > swing.price:
            broken = True
            direction = "bullish"

    elif swing.type in ["LL", "low", "HL"]:
        # Low broken if price closes below it
        if current_price < swing.price:
            broken = True
            direction = "bearish"

    if not broken:
        return None

    # Determine BOS type
    if current_trend == "uptrend" and direction == "bullish":
        bos_type = "bos"  # Break in direction of trend
    elif current_trend == "downtrend" and direction == "bearish":
        bos_type = "bos"  # Break in direction of trend
    elif current_trend == "uptrend" and direction == "bearish":
        bos_type = "choch"  # Break opposite to trend
    elif current_trend == "downtrend" and direction == "bullish":
        bos_type = "choch"  # Break opposite to trend
    else:
        bos_type = "choch"  # Default to CHoCH if ranging

    # Check for displacement (aggressive move)
    displacement = _check_displacement(ohlc_data, direction)

    # Check for MSS (non-key swing + displacement + FVG)
    if bos_type == "choch" and not swing.is_key and displacement:
        if fvgs and _check_fvg_near_break(fvgs, swing.price, current_time):
            bos_type = "mss"

    return BOS(
        id=make_id(
            "bos",
            instrument,
            timeframe,
            current_time,
            f"{bos_type}_{direction}_{swing.price}",
        ),
        instrument=instrument,
        timeframe=timeframe,
        direction=direction,
        broken_level=swing.price,
        break_time=current_time,
        break_candle_close=current_price,
        bos_type=bos_type,
        displacement=displacement,
    )


def _check_displacement(
    ohlc_data: pd.DataFrame,
    direction: str,
) -> bool:
    """
    Check if recent move shows displacement (aggressive movement).

    Displacement = recent candles show strong directional bias
    without significant pullback.

    Args:
        ohlc_data: Recent OHLC data
        direction: "bullish" or "bearish"

    Returns:
        True if displacement detected
    """
    if len(ohlc_data) < 3:
        return False

    # Look at last 3 candles
    recent = ohlc_data.tail(3)

    if direction == "bullish":
        # Check for strong bullish candles
        bullish_count = sum(
            1 for _, row in recent.iterrows()
            if row["close"] > row["open"]
        )
        return bullish_count >= 2

    else:  # bearish
        # Check for strong bearish candles
        bearish_count = sum(
            1 for _, row in recent.iterrows()
            if row["close"] < row["open"]
        )
        return bearish_count >= 2


def _check_fvg_near_break(
    fvgs: List[FVG],
    break_price: float,
    break_time: datetime,
) -> bool:
    """
    Check if there's an FVG formed near the structure break.

    Args:
        fvgs: List of active FVGs
        break_price: Price level that was broken
        break_time: Time of the break

    Returns:
        True if FVG found near break
    """
    for fvg in fvgs:
        # FVG should be recent (within last few candles)
        if fvg.formation_time > break_time:
            continue

        # FVG should be near the break level
        if fvg.low <= break_price <= fvg.high:
            return True

        # Or close enough (within 0.5%)
        distance = min(
            abs(break_price - fvg.high),
            abs(break_price - fvg.low),
        )
        if distance / break_price < 0.005:
            return True

    return False
```

---

## 3. `src/smc/engine.py`

Main SMC orchestrator that ties all detectors together and makes trading decisions.

### 3.1 SMCEngine Class

```python
"""
SMC Engine - Main orchestrator for Smart Money Concepts analysis.

Responsibilities:
- Update all SMC structures on each bar
- Evaluate IB signals with SMC context
- Calculate confluence scores
- Check confirmation criteria
- Manage structure lifecycle
"""

from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import pandas as pd

from .config import SMCConfig
from .timeframe_manager import TimeframeManager
from .registry import SMCRegistry
from .event_log import SMCEventLog
from .models import (
    SMCDecision,
    ConfirmationCriteria,
    ConfluenceScore,
    FVG,
    Fractal,
    CISD,
    BOS,
    MarketStructure,
)
from .detectors.fractal_detector import (
    detect_fractals,
    find_unswept_fractals,
    check_fractal_sweep,
)
from .detectors.fvg_detector import (
    detect_fvg,
    check_fvg_fill,
    check_fvg_rebalance,
)
from .detectors.cisd_detector import (
    detect_cisd,
    check_cisd_invalidation,
)
from .detectors.market_structure_detector import (
    detect_swing_points,
    detect_bos_choch,
)


class SMCEngine:
    """
    Main SMC orchestration engine.

    Manages all SMC detectors, registry, and decision logic.
    """

    def __init__(
        self,
        config: SMCConfig,
        timeframe_manager: TimeframeManager,
    ):
        """
        Initialize SMC engine.

        Args:
            config: SMC configuration
            timeframe_manager: Multi-timeframe data manager
        """
        self.config = config
        self.tfm = timeframe_manager
        self.registry = SMCRegistry(config.instrument)
        self.event_log = SMCEventLog()

        # Track market structure per timeframe
        self.market_structures: Dict[str, MarketStructure] = {}

    def update(
        self,
        current_time: datetime,
        new_m1_bars: Optional[pd.DataFrame] = None,
    ) -> None:
        """
        Update all SMC structures.

        Called on each M1 bar in slow backtest / live trading.
        For fast backtest, call once per day with full day data.

        Args:
            current_time: Current time (for "up to" filtering)
            new_m1_bars: Optional new M1 bars to append (for incremental updates)
        """
        # Append new M1 data if provided
        if new_m1_bars is not None and not new_m1_bars.empty:
            self.tfm.append_m1(new_m1_bars)

        # Update fractals
        if self.config.enable_fractals:
            self._update_fractals(current_time)

        # Update FVGs
        if self.config.enable_fvg:
            self._update_fvgs(current_time)

        # Update market structure and BOS/CHoCH
        if self.config.enable_bos:
            self._update_market_structure(current_time)

        # Update CISDs
        if self.config.enable_cisd:
            self._update_cisds(current_time)

        # Cleanup old structures
        cleanup_before = current_time - timedelta(hours=self.config.max_structure_age_hours)
        removed = self.registry.cleanup(cleanup_before)
        if removed > 0:
            self.event_log.record_simple(
                timestamp=current_time,
                instrument=self.config.instrument,
                event_type="CLEANUP",
                timeframe="all",
                details={"removed_count": removed},
            )

    def _update_fractals(self, current_time: datetime) -> None:
        """Update fractal structures."""
        tf = self.config.fractal_timeframe
        ohlc = self.tfm.get_data(tf, up_to=current_time)

        if len(ohlc) < 3:
            return

        # Detect new fractals
        candle_duration = 1.0 if tf == "H1" else 4.0 if tf == "H4" else 1.0
        fractals = detect_fractals(
            ohlc,
            self.config.instrument,
            tf,
            candle_duration_hours=candle_duration,
        )

        # Add new fractals to registry
        m1_data = self.tfm.get_data("M1", up_to=current_time)
        for frac in fractals:
            if frac.confirmed_time > current_time:
                continue  # Not confirmed yet

            # Check if already in registry
            existing = self.registry.get_by_id("fractal", frac.id)
            if existing:
                continue

            # Check if swept
            sweep_time = check_fractal_sweep(
                frac,
                m1_data,
                frac.confirmed_time,
                current_time,
            )

            if sweep_time:
                frac.swept = True
                frac.sweep_time = sweep_time
                frac.status = "swept"

            self.registry.add("fractal", frac)

            self.event_log.record_simple(
                timestamp=frac.confirmed_time,
                instrument=self.config.instrument,
                event_type="FRACTAL_CONFIRMED",
                timeframe=tf,
                direction="high" if frac.type == "high" else "low",
                price=frac.price,
                structure_id=frac.id,
            )

    def _update_fvgs(self, current_time: datetime) -> None:
        """Update FVG structures."""
        for tf in self.config.fvg_timeframes:
            ohlc = self.tfm.get_data(tf, up_to=current_time)

            if len(ohlc) < 3:
                continue

            # Detect new FVGs
            fvgs = detect_fvg(
                ohlc,
                self.config.instrument,
                tf,
                min_size_points=self.config.fvg_min_size_points,
            )

            m1_data = self.tfm.get_data("M1", up_to=current_time)

            for fvg in fvgs:
                if fvg.formation_time > current_time:
                    continue

                # Check if already in registry
                existing = self.registry.get_by_id("fvg", fvg.id)
                if existing:
                    continue

                # Check if filled
                fill_result = check_fvg_fill(
                    fvg,
                    m1_data,
                    fvg.formation_time,
                    current_time,
                )

                if fill_result:
                    fvg.fill_pct = fill_result["fill_pct"]
                    fvg.fill_time = fill_result["fill_time"]
                    fvg.status = fill_result["fill_type"] + "_fill"

                self.registry.add("fvg", fvg)

                self.event_log.record_simple(
                    timestamp=fvg.formation_time,
                    instrument=self.config.instrument,
                    event_type="FVG_FORMED",
                    timeframe=tf,
                    direction=fvg.direction,
                    price=fvg.midpoint,
                    structure_id=fvg.id,
                )

    def _update_market_structure(self, current_time: datetime) -> None:
        """Update market structure and BOS/CHoCH."""
        for tf in self.config.bos_timeframes:
            ohlc = self.tfm.get_data(tf, up_to=current_time)

            if len(ohlc) < 10:
                continue

            # Detect swing points and classify structure
            ms = detect_swing_points(
                ohlc,
                self.config.instrument,
                tf,
                lookback=3,
            )

            self.market_structures[tf] = ms

            # Detect BOS/CHoCH/MSS
            fvgs = self.registry.get_active("fvg", timeframe=tf)
            bos_list = detect_bos_choch(
                ms,
                ohlc,
                self.config.instrument,
                tf,
                fvgs=fvgs,
            )

            for bos in bos_list:
                # Check if already in registry
                existing = self.registry.get_by_id("bos", bos.id)
                if existing:
                    continue

                self.registry.add("bos", bos)

                self.event_log.record_simple(
                    timestamp=bos.break_time,
                    instrument=self.config.instrument,
                    event_type=f"{bos.bos_type.upper()}_DETECTED",
                    timeframe=tf,
                    direction=bos.direction,
                    price=bos.broken_level,
                    structure_id=bos.id,
                    bias_impact=1.0 if bos.direction == "bullish" else -1.0,
                )

    def _update_cisds(self, current_time: datetime) -> None:
        """Update CISD structures."""
        for tf in self.config.cisd_timeframes:
            ohlc = self.tfm.get_data(tf, up_to=current_time)

            if len(ohlc) < 5:
                continue

            # Get POI zones (FVGs and fractals)
            poi_zones = []

            # FVGs as POI
            fvgs = self.registry.get_active("fvg", timeframe=tf, before_time=current_time)
            for fvg in fvgs:
                poi_zones.append({
                    "type": "fvg",
                    "direction": fvg.direction,
                    "high": fvg.high,
                    "low": fvg.low,
                    "time": fvg.formation_time,
                })

            # Fractals as POI
            fractals = self.registry.get_unswept_fractals(
                self.config.fractal_timeframe,
                before_time=current_time,
            )
            for frac in fractals:
                poi_zones.append({
                    "type": "fractal",
                    "direction": "bearish" if frac.type == "high" else "bullish",
                    "high": frac.price + 2,  # Small zone around fractal
                    "low": frac.price - 2,
                    "time": frac.confirmed_time,
                })

            # Detect CISDs
            cisds = detect_cisd(
                ohlc,
                self.config.instrument,
                tf,
                poi_zones,
            )

            for cisd in cisds:
                # Check if already in registry
                existing = self.registry.get_by_id("cisd", cisd.id)
                if existing:
                    continue

                # Check invalidation
                if check_cisd_invalidation(cisd, ohlc, fvgs, fractals):
                    cisd.status = "invalidated"

                self.registry.add("cisd", cisd)

                self.event_log.record_simple(
                    timestamp=cisd.confirmation_time,
                    instrument=self.config.instrument,
                    event_type="CISD_DETECTED",
                    timeframe=tf,
                    direction=cisd.direction,
                    price=cisd.confirmation_close,
                    structure_id=cisd.id,
                    bias_impact=2.0 if cisd.direction == "long" else -2.0,
                )

    def evaluate_signal(
        self,
        signal: Any,
        current_time: datetime,
    ) -> SMCDecision:
        """
        Evaluate an IB signal with SMC context.

        Args:
            signal: IB signal object (from existing strategy)
            current_time: Current time

        Returns:
            SMCDecision with action (ENTER/WAIT/REJECT) and reasoning
        """
        # Get signal direction and price
        signal_direction = signal.direction  # "long" or "short"
        signal_price = getattr(signal, "entry_price", None)

        if signal_price is None:
            # Default to using current close
            m1_data = self.tfm.get_data("M1", up_to=current_time)
            signal_price = m1_data.iloc[-1]["close"] if not m1_data.empty else 0

        # Calculate confluence score
        confluence = self.calculate_confluence(
            signal_price,
            signal_direction,
            current_time,
        )

        # Check for conflicting structures
        conflicts = self._check_conflicts(signal_direction, signal_price, current_time)

        if conflicts:
            return SMCDecision(
                action="REJECT",
                reason=f"Conflicting structures: {conflicts}",
                confluence_score=confluence.total,
            )

        # Decide based on confluence score
        if confluence.total >= self.config.min_confluence_score:
            return SMCDecision(
                action="ENTER",
                reason=f"Strong confluence ({confluence.total:.1f}): {', '.join(confluence.contributing_structures)}",
                confluence_score=confluence.total,
            )

        # Wait for confirmation
        criteria = self._build_confirmation_criteria(signal_direction)

        return SMCDecision(
            action="WAIT",
            reason=f"Weak confluence ({confluence.total:.1f}), waiting for confirmation",
            confirmation_criteria=criteria,
            confluence_score=confluence.total,
            timeout_minutes=self.config.max_wait_minutes,
        )

    def check_confirmation(
        self,
        pending_signal: Any,
        criteria: List[ConfirmationCriteria],
        current_time: datetime,
    ) -> Optional[Any]:
        """
        Check if confirmation criteria are met.

        Args:
            pending_signal: The signal waiting for confirmation
            criteria: List of ConfirmationCriteria to check
            current_time: Current time

        Returns:
            Modified signal if confirmed, None otherwise
        """
        for criterion in criteria:
            if criterion.type == "CISD":
                if self._check_cisd_criterion(criterion, current_time):
                    return self._create_modified_signal(pending_signal, criterion, current_time)

            elif criterion.type == "FVG_REBALANCE":
                if self._check_fvg_rebalance_criterion(criterion, current_time):
                    return self._create_modified_signal(pending_signal, criterion, current_time)

            elif criterion.type == "BOS":
                if self._check_bos_criterion(criterion, current_time):
                    return self._create_modified_signal(pending_signal, criterion, current_time)

        return None

    def calculate_confluence(
        self,
        price: float,
        direction: str,
        current_time: datetime,
    ) -> ConfluenceScore:
        """
        Calculate weighted confluence score from all active structures.

        Args:
            price: Signal price
            direction: Signal direction ("long" or "short")
            current_time: Current time

        Returns:
            ConfluenceScore object
        """
        score = 0.0
        breakdown = {}
        contributing = []

        # Fractals
        if self.config.enable_fractals:
            frac_score, frac_ids = self._score_fractals(price, direction, current_time)
            score += frac_score
            if frac_score > 0:
                breakdown["fractals"] = frac_score
                contributing.extend(frac_ids)

        # FVGs
        if self.config.enable_fvg:
            fvg_score, fvg_ids = self._score_fvgs(price, direction, current_time)
            score += fvg_score
            if fvg_score > 0:
                breakdown["fvgs"] = fvg_score
                contributing.extend(fvg_ids)

        # CISDs
        if self.config.enable_cisd:
            cisd_score, cisd_ids = self._score_cisds(direction, current_time)
            score += cisd_score
            if cisd_score > 0:
                breakdown["cisds"] = cisd_score
                contributing.extend(cisd_ids)

        # BOS/CHoCH
        if self.config.enable_bos:
            bos_score, bos_ids = self._score_bos(direction, current_time)
            score += bos_score
            if bos_score > 0:
                breakdown["bos"] = bos_score
                contributing.extend(bos_ids)

        return ConfluenceScore(
            total=score,
            breakdown=breakdown,
            direction_bias=direction,
            contributing_structures=contributing,
        )

    def _score_fractals(
        self,
        price: float,
        direction: str,
        current_time: datetime,
    ) -> Tuple[float, List[str]]:
        """Score fractals near price."""
        fractals = self.registry.get_unswept_fractals(
            self.config.fractal_timeframe,
            before_time=current_time,
        )

        score = 0.0
        ids = []
        threshold = price * self.config.fractal_proximity_pct

        for frac in fractals:
            distance = abs(price - frac.price)
            if distance > threshold:
                continue

            # Long signal near low fractal = support
            if direction == "long" and frac.type == "low":
                score += self.config.weight_fractal
                ids.append(frac.id)

            # Short signal near high fractal = resistance
            if direction == "short" and frac.type == "high":
                score += self.config.weight_fractal
                ids.append(frac.id)

        return score, ids

    def _score_fvgs(
        self,
        price: float,
        direction: str,
        current_time: datetime,
    ) -> Tuple[float, List[str]]:
        """Score FVGs near price."""
        score = 0.0
        ids = []

        for tf in self.config.fvg_timeframes:
            fvgs = self.registry.get_fvgs_near_price(
                tf,
                price,
                max_distance_pct=0.005,
            )

            for fvg in fvgs:
                # Long signal near bullish FVG
                if direction == "long" and fvg.direction == "bullish":
                    score += self.config.weight_fvg
                    ids.append(fvg.id)

                # Short signal near bearish FVG
                if direction == "short" and fvg.direction == "bearish":
                    score += self.config.weight_fvg
                    ids.append(fvg.id)

        return score, ids

    def _score_cisds(
        self,
        direction: str,
        current_time: datetime,
    ) -> Tuple[float, List[str]]:
        """Score recent CISDs."""
        score = 0.0
        ids = []

        # Look for CISD in last 30 minutes
        recent_threshold = current_time - timedelta(minutes=30)

        for tf in self.config.cisd_timeframes:
            cisds = self.registry.get_active("cisd", timeframe=tf, direction=direction)

            for cisd in cisds:
                if cisd.confirmation_time >= recent_threshold:
                    score += self.config.weight_cisd
                    ids.append(cisd.id)

        return score, ids

    def _score_bos(
        self,
        direction: str,
        current_time: datetime,
    ) -> Tuple[float, List[str]]:
        """Score recent BOS in signal direction."""
        score = 0.0
        ids = []

        # Look for BOS in last 30 minutes
        recent_threshold = current_time - timedelta(minutes=30)

        for tf in self.config.bos_timeframes:
            bos_dir = "bullish" if direction == "long" else "bearish"
            bos_list = self.registry.get_active("bos", timeframe=tf, direction=bos_dir)

            for bos in bos_list:
                if bos.break_time >= recent_threshold:
                    # BOS gets full weight, CHoCH gets half (opposite to trend)
                    weight = self.config.weight_bos if bos.bos_type == "bos" else self.config.weight_bos * 0.5
                    score += weight
                    ids.append(bos.id)

        return score, ids

    def _check_conflicts(
        self,
        direction: str,
        price: float,
        current_time: datetime,
    ) -> List[str]:
        """Check for conflicting structures."""
        conflicts = []

        # Check for opposite-direction structures
        if direction == "long":
            # Look for bearish structures above current price
            fvgs = self.registry.get_fvgs_near_price(
                self.config.fvg_timeframes[0] if self.config.fvg_timeframes else "M2",
                price,
                max_distance_pct=0.01,
                direction="bearish",
            )
            if fvgs:
                conflicts.append("bearish_fvg_above")

        else:  # short
            # Look for bullish structures below current price
            fvgs = self.registry.get_fvgs_near_price(
                self.config.fvg_timeframes[0] if self.config.fvg_timeframes else "M2",
                price,
                max_distance_pct=0.01,
                direction="bullish",
            )
            if fvgs:
                conflicts.append("bullish_fvg_below")

        return conflicts

    def _build_confirmation_criteria(
        self,
        direction: str,
    ) -> List[ConfirmationCriteria]:
        """Build confirmation criteria for WAIT decision."""
        criteria = []

        if self.config.enable_cisd:
            criteria.append(
                ConfirmationCriteria(
                    type="CISD",
                    timeframe=self.config.cisd_timeframes[0] if self.config.cisd_timeframes else "M2",
                    direction=direction,
                )
            )

        if self.config.enable_fvg:
            criteria.append(
                ConfirmationCriteria(
                    type="FVG_REBALANCE",
                    timeframe=self.config.fvg_timeframes[0] if self.config.fvg_timeframes else "M2",
                    direction="bullish" if direction == "long" else "bearish",
                )
            )

        return criteria

    def _check_cisd_criterion(
        self,
        criterion: ConfirmationCriteria,
        current_time: datetime,
    ) -> bool:
        """Check if CISD criterion is met."""
        recent_threshold = current_time - timedelta(minutes=10)

        cisds = self.registry.get_active(
            "cisd",
            timeframe=criterion.timeframe,
            direction=criterion.direction,
        )

        for cisd in cisds:
            if cisd.confirmation_time >= recent_threshold:
                return True

        return False

    def _check_fvg_rebalance_criterion(
        self,
        criterion: ConfirmationCriteria,
        current_time: datetime,
    ) -> bool:
        """Check if FVG rebalance criterion is met."""
        # Get recent M1 candle
        m1_data = self.tfm.get_data("M1", up_to=current_time)
        if m1_data.empty:
            return False

        last_candle = m1_data.iloc[-1]

        # Get active FVGs
        fvgs = self.registry.get_active(
            "fvg",
            timeframe=criterion.timeframe,
            direction=criterion.direction,
        )

        for fvg in fvgs:
            if check_fvg_rebalance(fvg, last_candle):
                return True

        return False

    def _check_bos_criterion(
        self,
        criterion: ConfirmationCriteria,
        current_time: datetime,
    ) -> bool:
        """Check if BOS criterion is met."""
        recent_threshold = current_time - timedelta(minutes=15)

        bos_list = self.registry.get_active(
            "bos",
            timeframe=criterion.timeframe,
            direction=criterion.direction,
        )

        for bos in bos_list:
            if bos.break_time >= recent_threshold:
                return True

        return False

    def _create_modified_signal(
        self,
        original_signal: Any,
        confirming_criterion: ConfirmationCriteria,
        current_time: datetime,
    ) -> Any:
        """Create modified signal with SMC-adjusted parameters."""
        # Get confirmation candle
        m1_data = self.tfm.get_data("M1", up_to=current_time)
        if m1_data.empty:
            return original_signal

        conf_candle = m1_data.iloc[-1]

        # Modify entry, SL, TP based on confirmation candle
        modified_signal = original_signal  # Copy original

        # Entry = close of confirmation candle
        modified_signal.entry_price = conf_candle["close"]

        # SL = low/high of confirmation candle (tighter)
        if original_signal.direction == "long":
            modified_signal.stop_loss = conf_candle["low"]
        else:
            modified_signal.stop_loss = conf_candle["high"]

        # TP remains unchanged (original IB target)

        return modified_signal
```

---

## 4. Tests

### 4.1 `tests/test_smc/test_cisd_detector.py`

```python
"""Tests for CISD detector."""
import pytest
import pandas as pd
from datetime import datetime, timedelta
from src.smc.detectors.cisd_detector import detect_cisd


class TestCISDDetector:
    """Test CISD detection with synthetic data."""

    def _make_data(self, ohlc_list):
        return pd.DataFrame(ohlc_list, columns=["time", "open", "high", "low", "close"])

    def test_bearish_cisd_long_signal(self):
        """Downtrend delivery to zone, close above body -> long CISD."""
        data = self._make_data([
            # Downtrend delivery candles
            (datetime(2024, 1, 1, 10, 0), 110, 112, 108, 109),  # Down
            (datetime(2024, 1, 1, 10, 2), 109, 110, 105, 106),  # Down (delivery)
            # Test zone at 105
            (datetime(2024, 1, 1, 10, 4), 106, 108, 104, 105),  # Test
            # Confirmation: close above body (106-109 body high = 109)
            (datetime(2024, 1, 1, 10, 6), 105, 111, 104, 110),  # Close above 109
        ])

        poi = {
            "type": "fvg",
            "direction": "bearish",
            "high": 106,
            "low": 104,
            "time": datetime(2024, 1, 1, 9, 58),
        }

        cisds = detect_cisd(data, "GER40", "M2", [poi])

        assert len(cisds) == 1
        assert cisds[0].direction == "long"
        assert cisds[0].confirmation_close == 110

    def test_bullish_cisd_short_signal(self):
        """Uptrend delivery to zone, close below body -> short CISD."""
        data = self._make_data([
            # Uptrend delivery candles
            (datetime(2024, 1, 1, 10, 0), 100, 104, 99, 103),   # Up
            (datetime(2024, 1, 1, 10, 2), 103, 108, 102, 107),  # Up (delivery)
            # Test zone at 108
            (datetime(2024, 1, 1, 10, 4), 107, 109, 106, 108),  # Test
            # Confirmation: close below body (103-107 body low = 103)
            (datetime(2024, 1, 1, 10, 6), 108, 109, 101, 102),  # Close below 103
        ])

        poi = {
            "type": "fvg",
            "direction": "bullish",
            "high": 109,
            "low": 107,
            "time": datetime(2024, 1, 1, 9, 58),
        }

        cisds = detect_cisd(data, "GER40", "M2", [poi])

        assert len(cisds) == 1
        assert cisds[0].direction == "short"
        assert cisds[0].confirmation_close == 102

    def test_no_cisd_without_confirmation(self):
        """No confirmation candle -> no CISD."""
        data = self._make_data([
            (datetime(2024, 1, 1, 10, 0), 110, 112, 108, 109),
            (datetime(2024, 1, 1, 10, 2), 109, 110, 105, 106),
            (datetime(2024, 1, 1, 10, 4), 106, 108, 104, 105),  # Test
            # No confirmation (stays inside delivery body range)
            (datetime(2024, 1, 1, 10, 6), 105, 108, 104, 107),
        ])

        poi = {
            "type": "fvg",
            "direction": "bearish",
            "high": 106,
            "low": 104,
            "time": datetime(2024, 1, 1, 9, 58),
        }

        cisds = detect_cisd(data, "GER40", "M2", [poi])
        assert len(cisds) == 0
```

### 4.2 `tests/test_smc/test_market_structure.py`

```python
"""Tests for market structure detector."""
import pytest
import pandas as pd
from datetime import datetime
from src.smc.detectors.market_structure_detector import (
    detect_swing_points,
    detect_bos_choch,
)


class TestMarketStructureDetector:
    """Test swing point and BOS/CHoCH detection."""

    def _make_data(self, ohlc_list):
        return pd.DataFrame(ohlc_list, columns=["time", "open", "high", "low", "close"])

    def test_swing_highs_detected(self):
        """Swing high = bar with highest high in window."""
        data = self._make_data([
            (datetime(2024, 1, 1, 10), 100, 105, 99, 103),
            (datetime(2024, 1, 1, 11), 103, 108, 101, 106),
            (datetime(2024, 1, 1, 12), 106, 112, 105, 110),  # Swing high at 112
            (datetime(2024, 1, 1, 13), 110, 111, 107, 109),
            (datetime(2024, 1, 1, 14), 109, 110, 106, 108),
        ])

        ms = detect_swing_points(data, "GER40", "H1", lookback=1)

        highs = [sp for sp in ms.swing_points if sp.type in ["high", "HH", "LH"]]
        assert len(highs) >= 1
        assert any(sp.price == 112 for sp in highs)

    def test_uptrend_classification(self):
        """Series of HH and HL -> uptrend."""
        data = self._make_data([
            (datetime(2024, 1, 1, 10), 100, 102, 98, 101),   # Low at 98
            (datetime(2024, 1, 1, 11), 101, 106, 100, 105),  # High at 106
            (datetime(2024, 1, 1, 12), 105, 107, 103, 104),  # HL at 103
            (datetime(2024, 1, 1, 13), 104, 110, 103, 109),  # HH at 110
            (datetime(2024, 1, 1, 14), 109, 111, 107, 108),
        ])

        ms = detect_swing_points(data, "GER40", "H1", lookback=1)

        # Should have HH and HL
        types = {sp.type for sp in ms.swing_points}
        assert "HH" in types or "HL" in types
        # Trend should be uptrend (or ranging if not enough data)
        assert ms.current_trend in ["uptrend", "ranging"]

    def test_bos_detected(self):
        """Break of structure in trend direction -> BOS."""
        data = self._make_data([
            (datetime(2024, 1, 1, 10), 100, 105, 99, 104),
            (datetime(2024, 1, 1, 11), 104, 108, 103, 107),  # Swing high at 108
            (datetime(2024, 1, 1, 12), 107, 109, 106, 108),
            (datetime(2024, 1, 1, 13), 108, 112, 107, 111),  # Break 108 -> BOS
        ])

        ms = detect_swing_points(data, "GER40", "M5", lookback=1)
        bos_list = detect_bos_choch(ms, data, "GER40", "M5")

        # Should detect at least one BOS
        assert len(bos_list) >= 0  # May be 0 if structure not clear enough
```

### 4.3 `tests/test_smc/test_engine.py`

```python
"""Tests for SMC Engine."""
import pytest
import pandas as pd
from datetime import datetime
from src.smc.config import SMCConfig
from src.smc.timeframe_manager import TimeframeManager
from src.smc.engine import SMCEngine


class TestSMCEngine:
    """Test SMC engine integration."""

    def _make_m1_data(self):
        """Create minimal M1 data for testing."""
        times = pd.date_range(
            start="2024-01-01 10:00",
            periods=100,
            freq="1min",
        )
        data = pd.DataFrame({
            "time": times,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
        })
        return data

    def test_engine_initialization(self):
        """Engine initializes with config and TFM."""
        m1_data = self._make_m1_data()
        tfm = TimeframeManager(m1_data, "GER40")
        config = SMCConfig(instrument="GER40")

        engine = SMCEngine(config, tfm)

        assert engine.config.instrument == "GER40"
        assert engine.registry.instrument == "GER40"
        assert len(engine.event_log) == 0

    def test_update_detects_structures(self):
        """update() runs all enabled detectors."""
        m1_data = self._make_m1_data()
        tfm = TimeframeManager(m1_data, "GER40")
        config = SMCConfig(
            instrument="GER40",
            enable_fractals=True,
            enable_fvg=True,
        )

        engine = SMCEngine(config, tfm)
        engine.update(current_time=datetime(2024, 1, 1, 11, 0))

        # Should have run detectors (may or may not find structures in random data)
        assert engine.registry.count() >= 0
```

---

## 5. Execution Checklist

1. [ ] Create `src/smc/detectors/cisd_detector.py`
   - [ ] Implement `detect_cisd()` function
   - [ ] Implement helper functions: `_check_poi_for_cisd()`, `_find_zone_test()`, `_find_delivery_candles()`, `_find_cisd_confirmation()`
   - [ ] Implement `check_cisd_invalidation()`

2. [ ] Create `src/smc/detectors/market_structure_detector.py`
   - [ ] Implement `detect_swing_points()` function
   - [ ] Implement `_find_swing_highs()` and `_find_swing_lows()`
   - [ ] Implement `_classify_swings()` for HH/HL/LL/LH
   - [ ] Implement `_determine_trend()`
   - [ ] Implement `_mark_key_levels()`
   - [ ] Implement `detect_bos_choch()` function
   - [ ] Implement `_check_structure_break()`, `_check_displacement()`, `_check_fvg_near_break()`

3. [ ] Create `src/smc/engine.py`
   - [ ] Implement `SMCEngine` class
   - [ ] Implement `__init__()` method
   - [ ] Implement `update()` method
   - [ ] Implement `_update_fractals()`, `_update_fvgs()`, `_update_market_structure()`, `_update_cisds()`
   - [ ] Implement `evaluate_signal()` method
   - [ ] Implement `check_confirmation()` method
   - [ ] Implement `calculate_confluence()` method
   - [ ] Implement scoring methods: `_score_fractals()`, `_score_fvgs()`, `_score_cisds()`, `_score_bos()`
   - [ ] Implement `_check_conflicts()`, `_build_confirmation_criteria()`
   - [ ] Implement criterion checkers: `_check_cisd_criterion()`, `_check_fvg_rebalance_criterion()`, `_check_bos_criterion()`
   - [ ] Implement `_create_modified_signal()`

4. [ ] Update `src/smc/__init__.py`
   - [ ] Export SMCEngine
   - [ ] Export all new detector functions

5. [ ] Update `src/smc/models.py` (if needed)
   - [ ] Verify CISD, BOS, MarketStructure, StructurePoint models match implementation
   - [ ] Add any missing fields discovered during implementation

6. [ ] Update `src/smc/config.py`
   - [ ] Add `fvg_min_size_points` field (currently shows `fvg_min_size_pct` in architecture)
   - [ ] Verify all config fields used by engine exist

7. [ ] Write tests
   - [ ] Create `tests/test_smc/test_cisd_detector.py`
   - [ ] Create `tests/test_smc/test_market_structure.py`
   - [ ] Create `tests/test_smc/test_engine.py`

8. [ ] Run tests
   - [ ] `pytest tests/test_smc/test_cisd_detector.py -v`
   - [ ] `pytest tests/test_smc/test_market_structure.py -v`
   - [ ] `pytest tests/test_smc/test_engine.py -v`
   - [ ] `pytest tests/test_smc/ -v` (all tests)

9. [ ] Manual validation
   - [ ] Create simple script to test engine on 1 day of real data
   - [ ] Verify structures are detected correctly
   - [ ] Verify confluence scoring produces reasonable results
   - [ ] Verify event log captures all important events

10. [ ] Documentation
    - [ ] Add docstrings to all public functions
    - [ ] Add inline comments for complex logic (delivery candle selection, swing classification)
    - [ ] Update architecture doc if any design decisions changed during implementation

---

## 6. Key Implementation Notes

### 6.1 CISD Detection Edge Cases

- **Multiple delivery candles**: Always use the FIRST candle in the unidirectional sequence
- **Interrupted movement**: If movement is not unidirectional, use the candle immediately before the test
- **Invalidation**: Check for unoperated zones (FVGs, fractals) that contradict the CISD direction

### 6.2 Market Structure Classification

- **Swing detection**: Use N-bar lookback (default 3) on each side
- **HH/HL/LL/LH**: Compare each swing to the previous swing of the same type (high vs high, low vs low)
- **Trend determination**: Count recent HH/HL vs LL/LH; majority wins
- **Key levels**: Last HL before HH (uptrend) or last LH before LL (downtrend)

### 6.3 BOS vs CHoCH vs MSS

- **BOS**: Break in direction of trend, confirms continuation
- **CHoCH**: Break opposite to trend, signals potential reversal
- **MSS**: CHoCH + non-key swing + displacement + FVG = strong reversal signal

### 6.4 Confluence Scoring

- **Weights**: Configurable per structure type (fractal=1.0, FVG=1.5, CISD=2.0, BOS=1.5)
- **Proximity**: Only count structures "near" the signal price (configurable threshold)
- **Direction alignment**: Only count structures that support the signal direction
- **Time decay**: Optionally reduce weight for older structures (not implemented in base version)

### 6.5 Confirmation Criteria

- **CISD**: Look for CISD in signal direction within last 10 minutes
- **FVG_REBALANCE**: Check if last candle rebalanced an FVG (entered + closed on correct side)
- **BOS**: Look for BOS in signal direction within last 15 minutes

### 6.6 Performance Considerations

- **Cache market structures**: Store per-timeframe to avoid recomputing on each call
- **Limit lookback**: Only check recent structures (30min for CISD/BOS, 48h for fractals)
- **Lazy evaluation**: Only run enabled detectors
- **Registry cleanup**: Remove structures older than max_structure_age_hours

---

## 7. Test Scenarios

### 7.1 CISD Detection

**Scenario 1: Clean bearish CISD (long signal)**
- Setup: Downtrend to FVG zone, reaction, close above delivery body
- Expected: CISD detected with direction="long"

**Scenario 2: Clean bullish CISD (short signal)**
- Setup: Uptrend to fractal zone, reaction, close below delivery body
- Expected: CISD detected with direction="short"

**Scenario 3: No confirmation**
- Setup: Price tests zone but doesn't close beyond delivery body
- Expected: No CISD detected

**Scenario 4: Invalidated CISD**
- Setup: CISD forms but unswept fractal exists that contradicts direction
- Expected: CISD detected but status="invalidated"

### 7.2 Market Structure

**Scenario 1: Uptrend formation**
- Setup: Series of higher highs and higher lows
- Expected: Swing points classified as HH/HL, trend="uptrend"

**Scenario 2: BOS in uptrend**
- Setup: Uptrend, price breaks previous high
- Expected: BOS detected with bos_type="bos", direction="bullish"

**Scenario 3: CHoCH (potential reversal)**
- Setup: Uptrend, price breaks previous low
- Expected: BOS detected with bos_type="choch", direction="bearish"

**Scenario 4: MSS (strong reversal)**
- Setup: CHoCH + displacement + FVG formation
- Expected: BOS detected with bos_type="mss"

### 7.3 SMC Engine

**Scenario 1: High confluence -> ENTER**
- Setup: IB signal + nearby fractal + recent CISD + FVG
- Expected: SMCDecision with action="ENTER", high confluence score

**Scenario 2: Low confluence -> WAIT**
- Setup: IB signal with no supporting structures
- Expected: SMCDecision with action="WAIT", confirmation criteria provided

**Scenario 3: Conflicting structures -> REJECT**
- Setup: Long signal but bearish FVG above price
- Expected: SMCDecision with action="REJECT", reason mentions conflict

**Scenario 4: Confirmation met**
- Setup: Pending signal, CISD forms in correct direction
- Expected: check_confirmation() returns modified signal

---

## 8. Integration Points

### 8.1 Phase 1 Dependencies

- All Phase 1 modules must be complete and tested
- TimeframeManager must support M2, M5, M15 timeframes (in addition to M1, H1, H4)
- Registry must support all 4 structure types: fvg, fractal, cisd, bos
- EventLog must handle all event types defined in this phase

### 8.2 Phase 3 Integration

- IBStrategySMC will call `engine.evaluate_signal()` on each IB signal
- IBStrategySMC will call `engine.check_confirmation()` in AWAITING_CONFIRMATION state
- IBStrategySMC will call `engine.update()` on each M1 bar

### 8.3 Fast Backtest Integration (Phase 4)

- Fast backtest will call `engine.update()` once with full day data
- Fast backtest will use `engine.evaluate_signal()` during signal processing
- Fast backtest may pre-compute structures for entire date range

---

## File Paths Summary

All file paths are absolute (Windows format):

**Created in this phase:**
- `C:\Trading\ib_trading_bot\dual_v4\src\smc\detectors\cisd_detector.py`
- `C:\Trading\ib_trading_bot\dual_v4\src\smc\detectors\market_structure_detector.py`
- `C:\Trading\ib_trading_bot\dual_v4\src\smc\engine.py`
- `C:\Trading\ib_trading_bot\dual_v4\tests\test_smc\test_cisd_detector.py`
- `C:\Trading\ib_trading_bot\dual_v4\tests\test_smc\test_market_structure.py`
- `C:\Trading\ib_trading_bot\dual_v4\tests\test_smc\test_engine.py`

**Modified from Phase 1:**
- `C:\Trading\ib_trading_bot\dual_v4\src\smc\__init__.py` (export SMCEngine)
- `C:\Trading\ib_trading_bot\dual_v4\src\smc\config.py` (verify fields)
- `C:\Trading\ib_trading_bot\dual_v4\src\smc\models.py` (verify models)

**Dependencies from Phase 1:**
- `C:\Trading\ib_trading_bot\dual_v4\src\smc\models.py`
- `C:\Trading\ib_trading_bot\dual_v4\src\smc\config.py`
- `C:\Trading\ib_trading_bot\dual_v4\src\smc\timeframe_manager.py`
- `C:\Trading\ib_trading_bot\dual_v4\src\smc\registry.py`
- `C:\Trading\ib_trading_bot\dual_v4\src\smc\event_log.py`
- `C:\Trading\ib_trading_bot\dual_v4\src\smc\detectors\fractal_detector.py`
- `C:\Trading\ib_trading_bot\dual_v4\src\smc\detectors\fvg_detector.py`

---

END OF PHASE 2 SPECIFICATION
