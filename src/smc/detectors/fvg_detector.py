"""
FVG (Fair Value Gap) detector.

FVG = 3-candle formation where wicks of 1st and 3rd candles don't overlap.

Bullish FVG: candle1.high < candle3.low (gap up)
Bearish FVG: candle1.low > candle3.high (gap down)
"""

from datetime import datetime
from typing import Dict, List, Optional

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
        ohlc_data: DataFrame with [time, open, high, low, close]. Min 3 rows.
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
        c1 = ohlc_data.iloc[i - 1]
        c2 = ohlc_data.iloc[i]
        c3 = ohlc_data.iloc[i + 1]

        # Bullish FVG: gap between c1.high and c3.low
        if direction is None or direction == "bullish":
            if c3["low"] > c1["high"]:
                gap_size = c3["low"] - c1["high"]
                if gap_size >= min_size_points:
                    fvg_low = c1["high"]
                    fvg_high = c3["low"]
                    fvgs.append(FVG(
                        id=make_id("fvg", instrument, timeframe, c3["time"],
                                   f"bull_{fvg_low:.4f}_{fvg_high:.4f}"),
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
                    ))

        # Bearish FVG: gap between c3.high and c1.low
        if direction is None or direction == "bearish":
            if c1["low"] > c3["high"]:
                gap_size = c1["low"] - c3["high"]
                if gap_size >= min_size_points:
                    fvg_high = c1["low"]
                    fvg_low = c3["high"]
                    fvgs.append(FVG(
                        id=make_id("fvg", instrument, timeframe, c3["time"],
                                   f"bear_{fvg_low:.4f}_{fvg_high:.4f}"),
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
                    ))

    return fvgs


def check_fvg_fill(
    fvg: FVG,
    m1_data: pd.DataFrame,
    after_time: datetime,
    up_to_time: datetime,
) -> Optional[Dict]:
    """
    Check if/how an FVG has been filled by price action.

    Returns:
        dict with {fill_pct, fill_type, fill_time} or None if untouched.
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
        min_low = data["low"].min()
        if min_low < fvg.high:
            penetration = fvg.high - max(min_low, fvg.low)
            fill_pct = min(penetration / gap_size, 1.0)
            fill_time = data[data["low"] < fvg.high].iloc[0]["time"]
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
            fill_time = data[data["high"] > fvg.low].iloc[0]["time"]
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
    Check if a candle has "rebalanced" an FVG.

    For bullish FVG: candle enters FVG zone AND closes above FVG.high
    For bearish FVG: candle enters FVG zone AND closes below FVG.low

    Args:
        fvg: FVG to check
        candle: Single OHLC row (pd.Series with high, low, close)

    Returns:
        True if this candle rebalanced the FVG
    """
    if fvg.direction == "bullish":
        entered = candle["low"] <= fvg.high
        closed_above = candle["close"] > fvg.high
        return bool(entered and closed_above)
    else:
        entered = candle["high"] >= fvg.low
        closed_below = candle["close"] < fvg.low
        return bool(entered and closed_below)
