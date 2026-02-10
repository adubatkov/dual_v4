"""
CISD (Change In State of Delivery) detector.

Detects when market control transitions from one side to another
by identifying when price closes beyond the body of delivery candle(s)
that led to a key zone.
"""

from datetime import datetime
from typing import List, Optional, Tuple

import pandas as pd

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
    poi_direction = poi.get("direction")

    # Find candles after POI formation
    after_poi = ohlc_data[ohlc_data["time"] > poi_time].copy()
    if len(after_poi) < 2:
        return None

    # Find the test: price entering the POI zone
    test_idx = _find_zone_test(after_poi, poi_low, poi_high)
    if test_idx is None:
        return None

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

    return CISD(
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


def _find_zone_test(
    ohlc_data: pd.DataFrame,
    zone_low: float,
    zone_high: float,
) -> Optional[int]:
    """
    Find the first candle that enters the zone.

    Returns:
        Index of first candle that touches the zone, or None
    """
    for i in range(len(ohlc_data)):
        candle = ohlc_data.iloc[i]
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
            if candle["close"] < candle["open"]:
                delivery.insert(0, i)
            else:
                break
        else:
            if candle["close"] > candle["open"]:
                delivery.insert(0, i)
            else:
                break

    return delivery if delivery else [test_idx - 1]


def _find_cisd_confirmation(
    ohlc_data: pd.DataFrame,
    test_idx: int,
    delivery_body_high: float,
    delivery_body_low: float,
    poi_direction: Optional[str],
) -> Optional[Tuple[int, str]]:
    """
    Find confirmation candle that closes beyond delivery candle body.

    Returns:
        Tuple of (confirmation_index, direction) or None
    """
    max_look_forward = min(10, len(ohlc_data) - test_idx - 1)

    for i in range(test_idx + 1, test_idx + 1 + max_look_forward):
        candle = ohlc_data.iloc[i]

        # Bearish CISD (long signal): close above delivery body high
        if candle["close"] > delivery_body_high:
            if poi_direction is None or poi_direction == "bearish":
                return (i, "long")

        # Bullish CISD (short signal): close below delivery body low
        if candle["close"] < delivery_body_low:
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

    Returns:
        True if CISD should be invalidated
    """
    if ohlc_data.empty:
        return False

    current_price = ohlc_data.iloc[-1]["close"]

    if cisd.direction == "long":
        for fvg in fvgs:
            if fvg.direction == "bearish" and fvg.low > current_price:
                if fvg.status == "active":
                    return True

        for frac in fractals:
            if frac.type == "low" and frac.price < cisd.delivery_candle_body_low:
                if not frac.swept:
                    return True

    else:  # short
        for fvg in fvgs:
            if fvg.direction == "bullish" and fvg.high < current_price:
                if fvg.status == "active":
                    return True

        for frac in fractals:
            if frac.type == "high" and frac.price > cisd.delivery_candle_body_high:
                if not frac.swept:
                    return True

    return False
