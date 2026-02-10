"""
Multi-timeframe data management - resamples M1 to any higher timeframe.
"""

from datetime import datetime
from typing import Dict, Optional

import pandas as pd


class TimeframeManager:
    """Manages multi-timeframe data from M1 source.

    Supports: M1, M2, M5, M15, M30, H1, H4, D1.
    Caches resampled data, invalidates on append.
    """

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

    # Candle duration in hours (for fractal confirmation time)
    TF_HOURS = {
        "M1": 1 / 60,
        "M2": 2 / 60,
        "M5": 5 / 60,
        "M15": 0.25,
        "M30": 0.5,
        "H1": 1.0,
        "H4": 4.0,
        "D1": 24.0,
    }

    def __init__(self, m1_data: pd.DataFrame, instrument: str):
        """
        Args:
            m1_data: DataFrame with columns [time, open, high, low, close].
                     Optionally [volume]. Time should be datetime.
            instrument: "GER40" or "XAUUSD"
        """
        self.instrument = instrument
        self._m1_data = m1_data.copy()
        self._cache: Dict[str, pd.DataFrame] = {}

    @property
    def m1_data(self) -> pd.DataFrame:
        return self._m1_data

    def get_data(self, timeframe: str, up_to: Optional[datetime] = None) -> pd.DataFrame:
        """Get OHLC data for the requested timeframe.

        Args:
            timeframe: One of TF_RULES keys (M1, M2, ..., D1)
            up_to: If set, return only COMPLETED candles before this time

        Returns:
            DataFrame with [time, open, high, low, close] columns.
        """
        if timeframe not in self.TF_RULES:
            raise ValueError(f"Unsupported timeframe: {timeframe}. Supported: {list(self.TF_RULES.keys())}")

        if timeframe == "M1":
            df = self._m1_data
        elif timeframe in self._cache:
            df = self._cache[timeframe]
        else:
            df = self._resample(timeframe)
            self._cache[timeframe] = df

        if up_to is not None:
            return df[df["time"] < up_to].copy()
        return df.copy()

    def get_last_n_candles(self, timeframe: str, n: int, before: datetime) -> pd.DataFrame:
        """Get last N completed candles before given time."""
        df = self.get_data(timeframe, up_to=before)
        return df.tail(n).copy()

    def append_m1(self, new_bars: pd.DataFrame) -> None:
        """Append new M1 bars and invalidate cache.

        For incremental updates in slow backtest / live trading.
        """
        self._m1_data = pd.concat([self._m1_data, new_bars]).drop_duplicates(subset=["time"])
        self._m1_data = self._m1_data.sort_values("time").reset_index(drop=True)
        self._cache.clear()

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

    def get_candle_duration_hours(self, timeframe: str) -> float:
        """Get candle duration in hours for a timeframe."""
        return self.TF_HOURS.get(timeframe, 1.0)
