#!/usr/bin/env python3
"""
Optimized Fast Vectorized Backtest Engine.

Performance improvements over fast_backtest.py:
1. Vectorized timezone conversion (~20x faster)

Expected overall speedup: ~1.3x (from timezone optimization alone)
"""

import numpy as np
import pandas as pd
import pytz
from datetime import datetime, time, timedelta
from typing import Optional, Dict, Any, Tuple, List

from .fast_backtest import FastBacktest, Signal, SPREAD_POINTS

# Type checking import for NewsFilter
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.news_filter import NewsFilter


class FastBacktestOptimized(FastBacktest):
    """
    Optimized version of FastBacktest with performance improvements.

    Key optimizations:
    1. Pre-compute timezone dates (vectorized instead of apply+lambda)

    All signal detection methods inherited from parent class for correctness.
    """

    def __init__(
        self,
        symbol: str,
        m1_data: pd.DataFrame,
        ib_cache: Optional[Dict[Tuple[str, str, str], Dict[Any, Dict[str, float]]]] = None,
        news_filter: Optional["NewsFilter"] = None,
    ):
        """Initialize with pre-loaded data."""
        # Call parent init
        super().__init__(symbol, m1_data, ib_cache, news_filter)

        # Pre-computed timezone date caches (populated on demand)
        self._tz_date_cache: Dict[str, pd.Series] = {}

    def _get_tz_dates_vectorized(self, timezone_str: str) -> pd.Series:
        """
        Get pre-computed dates for timezone using vectorized operations.

        Uses caching to avoid recomputation for same timezone.

        Performance: ~20x faster than apply(lambda x: x.astimezone(tz).date())
        """
        if timezone_str in self._tz_date_cache:
            return self._tz_date_cache[timezone_str]

        # Vectorized conversion - MUCH faster than apply+lambda
        # Step 1: Convert to target timezone
        time_in_tz = self.m1_data["time"].dt.tz_convert(timezone_str)

        # Step 2: Extract date (vectorized)
        dates = time_in_tz.dt.date

        # Cache the result
        self._tz_date_cache[timezone_str] = dates

        return dates

    def run_with_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run backtest with given parameters. Optimized version.

        Only optimizes timezone conversion - all other logic inherited from parent.
        """
        try:
            timezone_str = params.get("ib_timezone", "Europe/Berlin")

            # OPTIMIZATION: Use vectorized timezone conversion instead of apply+lambda
            self.m1_data["ib_date"] = self._get_tz_dates_vectorized(timezone_str)

            trades = []

            # Process day-by-day (same as parent)
            for day_date, day_df in self.m1_data.groupby("ib_date"):
                day_df = day_df.sort_values("time").reset_index(drop=True)
                trade = self._process_day(day_df, day_date, params)
                if trade:
                    trades.append(trade)

            return self._calculate_metrics(trades, params)

        except Exception as e:
            return {
                "error": str(e),
                "total_r": 0.0,
                "total_profit": 0.0,
                "total_trades": 0,
                "winrate": 0.0,
                "sharpe_ratio": 0.0,
                "profit_factor": 0.0,
                "max_drawdown": 0.0,
                "avg_trade_r": 0.0,
            }

    def _check_tcwe(
        self, df_trade: pd.DataFrame, ibh: float, ibl: float, eq: float,
        trade_start_price: float, params: Dict
    ) -> Optional[Signal]:
        """
        TCWE - two candles without equilibrium (second candle goes further than first).

        OPTIMIZED VERSION: Pre-compute EQ touched mask once instead of calling
        _eq_touched_before_idx() multiple times.
        """
        n = len(df_trade)
        ib_range = ibh - ibl
        if ib_range <= 0:
            return None

        ib_buffer_pct = params.get("ib_buffer_pct", 0.0)
        max_distance_pct = params.get("max_distance_pct", 1.0)
        buffer = ib_buffer_pct * ib_range
        max_dist = max_distance_pct * ib_range

        # OPTIMIZATION: Pre-compute EQ touched cumulative mask
        # eq_touched[i] = True if EQ was touched at any index from 0 to i (inclusive)
        lows = df_trade["low"].values
        highs = df_trade["high"].values
        closes = df_trade["close"].values

        eq_touched_per_bar = (lows <= eq) & (highs >= eq)
        eq_touched = np.cumsum(eq_touched_per_bar) > 0

        first_idx_long = None
        first_idx_short = None

        # Find first breakout without EQ touch
        for i in range(n):
            c = float(closes[i])
            # Check long breakout with buffer
            if (c > ibh + buffer) and not eq_touched[i]:
                distance = c - ibh
                if distance <= max_dist:
                    first_idx_long = i
                    break
            # Check short breakout with buffer
            if (c < ibl - buffer) and not eq_touched[i]:
                distance = ibl - c
                if distance <= max_dist:
                    first_idx_short = i
                    break

        if first_idx_long is not None:
            c1 = float(closes[first_idx_long])
            for j in range(first_idx_long + 1, n):
                if eq_touched[j]:
                    break
                c2 = float(closes[j])
                # Second candle must also pass buffer and distance checks
                if (c2 > ibh + buffer) and (c2 > c1):
                    distance = c2 - ibh
                    if distance <= max_dist:
                        base_price = c2
                        entry_price = self._apply_spread(base_price, "long")
                        stop_price = self._get_stop_price(
                            df_trade, j, "long", entry_price, trade_start_price, eq, params
                        )
                        entry_time = df_trade["time"].iat[j]
                        return Signal(
                            signal_type="TCWE",
                            direction="long",
                            entry_idx=j,
                            entry_price=entry_price,
                            stop_price=stop_price,
                            entry_time=entry_time
                        )

        if first_idx_short is not None:
            c1 = float(closes[first_idx_short])
            for j in range(first_idx_short + 1, n):
                if eq_touched[j]:
                    break
                c2 = float(closes[j])
                if (c2 < ibl - buffer) and (c2 < c1):
                    distance = ibl - c2
                    if distance <= max_dist:
                        base_price = c2
                        entry_price = self._apply_spread(base_price, "short")
                        stop_price = self._get_stop_price(
                            df_trade, j, "short", entry_price, trade_start_price, eq, params
                        )
                        entry_time = df_trade["time"].iat[j]
                        return Signal(
                            signal_type="TCWE",
                            direction="short",
                            entry_idx=j,
                            entry_price=entry_price,
                            stop_price=stop_price,
                            entry_time=entry_time
                        )

        return None

    def _check_ocae(
        self, df_trade: pd.DataFrame, ibh: float, ibl: float, eq: float,
        trade_start_price: float, params: Dict
    ) -> Optional[Signal]:
        """
        OCAE - first breakout after EQ touch.

        OPTIMIZED VERSION: Pre-compute EQ touched mask once.
        """
        ib_range = ibh - ibl
        if ib_range <= 0:
            return None

        ib_buffer_pct = params.get("ib_buffer_pct", 0.0)
        max_distance_pct = params.get("max_distance_pct", 1.0)
        buffer = ib_buffer_pct * ib_range
        max_dist = max_distance_pct * ib_range

        # OPTIMIZATION: Pre-compute arrays
        lows = df_trade["low"].values
        highs = df_trade["high"].values
        closes = df_trade["close"].values

        eq_touched_per_bar = (lows <= eq) & (highs >= eq)
        eq_touched = np.cumsum(eq_touched_per_bar) > 0

        # Find first valid breakout
        first_breakout_idx = None
        first_breakout_direction = None

        n = len(df_trade)
        for i in range(n):
            c = float(closes[i])
            # Check long breakout with buffer and distance
            if c > ibh + buffer:
                distance = c - ibh
                if distance <= max_dist:
                    first_breakout_idx = i
                    first_breakout_direction = "long"
                    break
            # Check short breakout with buffer and distance
            if c < ibl - buffer:
                distance = ibl - c
                if distance <= max_dist:
                    first_breakout_idx = i
                    first_breakout_direction = "short"
                    break

        if first_breakout_idx is None:
            return None

        # Check if EQ was touched BEFORE first breakout
        if not eq_touched[first_breakout_idx]:
            return None

        # Return OCAE signal
        c = float(closes[first_breakout_idx])
        base_price = c
        entry_price = self._apply_spread(base_price, first_breakout_direction)
        stop_price = self._get_stop_price(
            df_trade, first_breakout_idx, first_breakout_direction,
            entry_price, trade_start_price, eq, params
        )
        entry_time = df_trade["time"].iat[first_breakout_idx]

        return Signal(
            signal_type="OCAE",
            direction=first_breakout_direction,
            entry_idx=first_breakout_idx,
            entry_price=entry_price,
            stop_price=stop_price,
            entry_time=entry_time
        )

    def clear_cache(self):
        """Clear cached timezone dates (useful when switching symbols)."""
        self._tz_date_cache.clear()
