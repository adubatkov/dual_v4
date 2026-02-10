"""
Market structure detector.

Detects swing points (HH/HL/LL/LH), current trend, and structure breaks
(BOS/CHoCH/MSS) based on Smart Money Concepts methodology.
"""

from datetime import datetime
from typing import List, Optional, Tuple

import pandas as pd

from ..models import BOS, FVG, MarketStructure, StructurePoint, make_id


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

    Returns:
        List of (index, price) tuples
    """
    swing_highs = []

    for i in range(lookback, len(ohlc_data) - lookback):
        center_high = ohlc_data.iloc[i]["high"]

        is_swing = True
        for j in range(i - lookback, i + lookback + 1):
            if j == i:
                continue
            if ohlc_data.iloc[j]["high"] >= center_high:
                is_swing = False
                break

        if is_swing:
            swing_highs.append((i, float(center_high)))

    return swing_highs


def _find_swing_lows(
    ohlc_data: pd.DataFrame,
    lookback: int,
) -> List[Tuple[int, float]]:
    """
    Find swing lows: bars where low is lowest in lookback window.

    Returns:
        List of (index, price) tuples
    """
    swing_lows = []

    for i in range(lookback, len(ohlc_data) - lookback):
        center_low = ohlc_data.iloc[i]["low"]

        is_swing = True
        for j in range(i - lookback, i + lookback + 1):
            if j == i:
                continue
            if ohlc_data.iloc[j]["low"] <= center_low:
                is_swing = False
                break

        if is_swing:
            swing_lows.append((i, float(center_low)))

    return swing_lows


def _classify_swings(
    swings: List[dict],
) -> List[StructurePoint]:
    """
    Classify swings as HH/HL/LL/LH based on sequence.

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
                swing_type = "high"
            elif swing["price"] > prev_high:
                swing_type = "HH"
            else:
                swing_type = "LH"

            prev_high = swing["price"]

        else:  # low
            if prev_low is None:
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

    if (hh_count + hl_count) > (ll_count + lh_count):
        return "uptrend"

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
    """
    if current_trend == "uptrend":
        for i in range(len(structure_points) - 1, -1, -1):
            if structure_points[i].type == "HH":
                for j in range(i - 1, -1, -1):
                    if structure_points[j].type == "HL":
                        structure_points[j].is_key = True
                        break
                break

    elif current_trend == "downtrend":
        for i in range(len(structure_points) - 1, -1, -1):
            if structure_points[i].type == "LL":
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
    current_price = float(ohlc_data.iloc[-1]["close"])
    current_time = ohlc_data.iloc[-1]["time"]

    recent_swings = market_structure.swing_points[-5:]
    current_trend = market_structure.current_trend

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
    """Check if current price has broken a swing level."""
    broken = False
    direction = None

    if swing.type in ["HH", "high", "LH"]:
        if current_price > swing.price:
            broken = True
            direction = "bullish"

    elif swing.type in ["LL", "low", "HL"]:
        if current_price < swing.price:
            broken = True
            direction = "bearish"

    if not broken:
        return None

    # Determine BOS type
    if current_trend == "uptrend" and direction == "bullish":
        bos_type = "bos"
    elif current_trend == "downtrend" and direction == "bearish":
        bos_type = "bos"
    elif current_trend == "uptrend" and direction == "bearish":
        bos_type = "choch"
    elif current_trend == "downtrend" and direction == "bullish":
        bos_type = "choch"
    else:
        bos_type = "choch"

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

    Displacement = recent candles show strong directional bias.
    """
    if len(ohlc_data) < 3:
        return False

    recent = ohlc_data.tail(3)

    if direction == "bullish":
        bullish_count = sum(
            1 for _, row in recent.iterrows()
            if row["close"] > row["open"]
        )
        return bullish_count >= 2

    else:
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
    """Check if there's an FVG formed near the structure break."""
    for fvg in fvgs:
        if fvg.formation_time > break_time:
            continue

        # FVG contains break level
        if fvg.low <= break_price <= fvg.high:
            return True

        # Or close enough (within 0.5%)
        distance = min(
            abs(break_price - fvg.high),
            abs(break_price - fvg.low),
        )
        if break_price > 0 and distance / break_price < 0.005:
            return True

    return False
