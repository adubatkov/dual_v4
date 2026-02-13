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
        candle_duration_hours: Duration of one candle for confirmation time

    Returns:
        List of Fractal objects. Each fractal confirmed after 3rd bar closes.
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
            fractals.append(Fractal(
                id=make_id("frac", instrument, timeframe, curr["time"], f"high_{curr['high']}"),
                instrument=instrument,
                timeframe=timeframe,
                type="high",
                price=curr["high"],
                time=curr["time"],
                confirmed_time=confirmed_time,
                candle_close=float(curr["close"]),
            ))

        # Low fractal
        if curr["low"] < prev["low"] and curr["low"] < next_["low"]:
            fractals.append(Fractal(
                id=make_id("frac", instrument, timeframe, curr["time"], f"low_{curr['low']}"),
                instrument=instrument,
                timeframe=timeframe,
                type="low",
                price=curr["low"],
                time=curr["time"],
                confirmed_time=confirmed_time,
                candle_close=float(curr["close"]),
            ))

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
                swept = bool((check_data["high"] >= frac.price).any())
            else:
                swept = bool((check_data["low"] <= frac.price).any())

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

    Returns:
        Sweep time (first M1 bar that touched the level) or None.
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
