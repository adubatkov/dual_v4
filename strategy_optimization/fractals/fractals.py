"""
Fractal detection utilities for 1H timeframe.

Williams 3-bar fractal definition:
- High fractal: center bar's high > high of bar before AND high of bar after
- Low fractal: center bar's low < low of bar before AND low of bar after
"""

from datetime import datetime, timedelta
from typing import List, Dict, Optional

import pandas as pd
import pytz


def resample_m1_to_h1(m1_data: pd.DataFrame) -> pd.DataFrame:
    """
    Resample M1 data to H1 candles.

    Args:
        m1_data: DataFrame with columns [time, open, high, low, close]
                 time must be timezone-aware UTC

    Returns:
        H1 DataFrame with same columns
    """
    df = m1_data.copy()
    df = df.set_index("time")

    agg_dict = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
    }

    if "tick_volume" in df.columns:
        agg_dict["tick_volume"] = "sum"
    elif "volume" in df.columns:
        agg_dict["volume"] = "sum"

    resampled = df.resample("1h").agg(agg_dict).dropna()
    resampled = resampled.reset_index()
    return resampled


def detect_fractals_3bar(h1_data: pd.DataFrame) -> pd.DataFrame:
    """
    Detect Williams 3-bar fractals in H1 data.

    High fractal: high[i] > high[i-1] AND high[i] > high[i+1]
    Low fractal: low[i] < low[i-1] AND low[i] < low[i+1]

    Args:
        h1_data: H1 DataFrame with columns [time, open, high, low, close]

    Returns:
        DataFrame with columns [time, price, type, confirmed_time]
        - time: center bar's opening time (when the fractal level occurred)
        - confirmed_time: when the fractal is confirmed (after confirming bar closes)
        - type is 'high' or 'low'
    """
    if len(h1_data) < 3:
        return pd.DataFrame(columns=["time", "price", "type", "confirmed_time"])

    fractals = []

    for i in range(1, len(h1_data) - 1):
        curr = h1_data.iloc[i]
        prev = h1_data.iloc[i - 1]
        next_ = h1_data.iloc[i + 1]

        # Confirmed time = end of confirming bar (next_ bar's time + 1 hour)
        confirmed_time = next_["time"] + timedelta(hours=1)

        # High fractal
        if curr["high"] > prev["high"] and curr["high"] > next_["high"]:
            fractals.append({
                "time": curr["time"],
                "price": curr["high"],
                "type": "high",
                "confirmed_time": confirmed_time,
            })

        # Low fractal
        if curr["low"] < prev["low"] and curr["low"] < next_["low"]:
            fractals.append({
                "time": curr["time"],
                "price": curr["low"],
                "type": "low",
                "confirmed_time": confirmed_time,
            })

    return pd.DataFrame(fractals)


def find_unswept_fractals(
    fractals: pd.DataFrame,
    before_time: datetime,
    lookback_hours: int,
    m1_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Find fractals that formed in lookback period and were NOT yet swept by before_time.

    A fractal is "unswept" if no M1 bar after its CONFIRMATION touched the fractal level
    before the specified time.

    Args:
        fractals: DataFrame with [time, price, type, confirmed_time] from detect_fractals_3bar
        before_time: Time to check unswept status (typically IB start)
        lookback_hours: How many hours back to look for fractals
        m1_data: M1 data for checking sweeps

    Returns:
        DataFrame of unswept fractals
    """
    if fractals.empty:
        return fractals

    lookback_start = before_time - timedelta(hours=lookback_hours)

    # Filter fractals that:
    # 1. Formed in lookback window (center bar time in window)
    # 2. Were CONFIRMED before before_time (confirmed_time < before_time)
    mask = (
        (fractals["time"] >= lookback_start) &
        (fractals["time"] < before_time) &
        (fractals["confirmed_time"] <= before_time)
    )
    candidates = fractals[mask].copy()

    if candidates.empty:
        return candidates

    unswept = []

    for _, frac in candidates.iterrows():
        frac_confirmed = frac["confirmed_time"]
        frac_price = frac["price"]
        frac_type = frac["type"]

        # Check M1 bars AFTER fractal confirmation until before_time
        # A fractal can only be swept after it's confirmed
        check_mask = (m1_data["time"] >= frac_confirmed) & (m1_data["time"] < before_time)
        check_data = m1_data[check_mask]

        swept = False
        if not check_data.empty:
            if frac_type == "high":
                # High fractal swept if any bar's high >= fractal price
                if (check_data["high"] >= frac_price).any():
                    swept = True
            else:  # low
                # Low fractal swept if any bar's low <= fractal price
                if (check_data["low"] <= frac_price).any():
                    swept = True

        if not swept:
            unswept.append(frac.to_dict())

    return pd.DataFrame(unswept)


def find_swept_fractals_in_window(
    unswept: pd.DataFrame,
    m1_data: pd.DataFrame,
    window_start: datetime,
    window_end: datetime,
) -> List[Dict]:
    """
    Find which unswept fractals were swept during the trading window.

    Args:
        unswept: DataFrame of unswept fractals [time, price, type]
        m1_data: M1 data for the symbol
        window_start: Start of IB (or trading window)
        window_end: End of trade window

    Returns:
        List of dicts: [{fractal_time, fractal_price, fractal_type, sweep_time}, ...]
        Sorted by sweep_time
    """
    if unswept.empty:
        return []

    # Filter M1 data to window
    window_mask = (m1_data["time"] >= window_start) & (m1_data["time"] <= window_end)
    window_data = m1_data[window_mask].copy()

    if window_data.empty:
        return []

    swept = []

    for _, frac in unswept.iterrows():
        frac_price = frac["price"]
        frac_type = frac["type"]
        frac_time = frac["time"]

        # Find first bar that sweeps this fractal
        if frac_type == "high":
            sweep_mask = window_data["high"] >= frac_price
        else:
            sweep_mask = window_data["low"] <= frac_price

        sweep_bars = window_data[sweep_mask]

        if not sweep_bars.empty:
            first_sweep = sweep_bars.iloc[0]
            swept.append({
                "fractal_time": frac_time,
                "fractal_price": frac_price,
                "fractal_type": frac_type,
                "sweep_time": first_sweep["time"],
            })

    # Sort by sweep time
    swept.sort(key=lambda x: x["sweep_time"])

    return swept


def get_swept_fractals_for_trade(
    m1_data: pd.DataFrame,
    ib_start: datetime,
    window_end: datetime,
    lookback_hours: int = 48,
) -> List[Dict]:
    """
    High-level function to get all fractals swept during a trade's IB+window period.

    Args:
        m1_data: Full M1 data for the symbol (UTC timezone)
        ib_start: IB period start time (UTC)
        window_end: End of trade window (UTC)
        lookback_hours: How far back to look for fractals

    Returns:
        List of swept fractal dicts with fractal_time, fractal_price, fractal_type, sweep_time
    """
    # Get M1 data for lookback + window period
    data_start = ib_start - timedelta(hours=lookback_hours + 1)
    data_end = window_end + timedelta(hours=1)

    subset_mask = (m1_data["time"] >= data_start) & (m1_data["time"] <= data_end)
    subset = m1_data[subset_mask].copy()

    if subset.empty:
        return []

    # Resample to H1
    h1_data = resample_m1_to_h1(subset)

    if h1_data.empty:
        return []

    # Detect all fractals
    fractals = detect_fractals_3bar(h1_data)

    if fractals.empty:
        return []

    # Find unswept fractals at IB start
    unswept = find_unswept_fractals(
        fractals=fractals,
        before_time=ib_start,
        lookback_hours=lookback_hours,
        m1_data=subset,
    )

    if unswept.empty:
        return []

    # Find which were swept during IB + trade window
    swept = find_swept_fractals_in_window(
        unswept=unswept,
        m1_data=subset,
        window_start=ib_start,
        window_end=window_end,
    )

    return swept
