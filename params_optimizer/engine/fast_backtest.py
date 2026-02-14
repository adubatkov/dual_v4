#!/usr/bin/env python3
"""
Fast Vectorized Backtest Engine for Parameter Optimization.

Processes data day-by-day instead of candle-by-candle for ~50-75x speedup.
Target: ~4-6 seconds per parameter combination on full dataset.

Ported from strategy_logic.py with same logic but optimized for parameter search.
"""

import pandas as pd
import numpy as np
import pytz
from datetime import datetime, time, timedelta
from typing import Optional, Dict, Any, Tuple, List, TYPE_CHECKING
from dataclasses import dataclass, field

if TYPE_CHECKING:
    from src.news_filter import NewsFilter


# Symbol spreads in points (same as MT5Emulator config)
SPREAD_POINTS = {
    "GER40": 1.5,
    "XAUUSD": 0.30,  # Fixed: was 0.20, should match config.py
    "NAS100": 1.5,
    "UK100": 1.0,
}


@dataclass
class Signal:
    """Trading signal detected by signal detection functions."""
    signal_type: str  # "Reverse", "OCAE", "TCWE", "REV_RB"
    direction: str    # "long" or "short"
    entry_idx: int    # Index in df_trade (M2) for entry
    entry_price: float  # Raw price for SL/TP calculation (like IBStrategy signal.entry_price)
    stop_price: float
    entry_time: Optional[pd.Timestamp] = None  # M2 candle time for M1 mapping
    tick_price: Optional[float] = None  # Tick price for execution (CLOSE of previous candle for Reverse)
    extra: Dict[str, Any] = field(default_factory=dict)  # Additional signal-specific data


class FastBacktest:
    """
    Vectorized backtest engine for parameter optimization.
    Processes data day-by-day instead of candle-by-candle.
    """

    def __init__(
        self,
        symbol: str,
        m1_data: pd.DataFrame,
        ib_cache: Optional[Dict[Tuple[str, str, str], Dict[Any, Dict[str, float]]]] = None,
        news_filter: Optional["NewsFilter"] = None,
    ):
        """
        Initialize with pre-loaded data.

        Args:
            symbol: Trading symbol (GER40, XAUUSD, etc.)
            m1_data: M1 candle data with columns: time, open, high, low, close
            ib_cache: Pre-computed IB cache from ib_precompute.py
                      Format: {(start, end, tz): {date: {IBH, IBL, EQ}}}
            news_filter: Optional NewsFilter for filtering trades during high-impact news
        """
        self.symbol = symbol
        self.m1_data = m1_data.copy()
        self.ib_cache = ib_cache
        self.news_filter = news_filter
        self._precompute_dates()
        self._cache_hits = 0
        self._cache_misses = 0
        self._news_filtered_count = 0  # Track trades filtered by news

    def get_cache_stats(self) -> Dict[str, int]:
        """Get IB cache hit/miss statistics."""
        total = self._cache_hits + self._cache_misses
        return {
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "total_lookups": total,
            "hit_rate_pct": (self._cache_hits / total * 100) if total > 0 else 0.0,
        }

    def get_news_filter_stats(self) -> Dict[str, Any]:
        """Get news filter statistics."""
        return {
            "news_filter_enabled": self.news_filter is not None,
            "trades_filtered_by_news": self._news_filtered_count,
        }

    def reset_cache_stats(self) -> None:
        """Reset cache statistics."""
        self._cache_hits = 0
        self._cache_misses = 0

    def reset_news_filter_stats(self) -> None:
        """Reset news filter statistics."""
        self._news_filtered_count = 0

    def _precompute_dates(self):
        """Precompute IB dates for faster groupby operations."""
        # Ensure time column is datetime
        if not pd.api.types.is_datetime64_any_dtype(self.m1_data["time"]):
            self.m1_data["time"] = pd.to_datetime(self.m1_data["time"], utc=True)

        # Make sure timezone aware
        if self.m1_data["time"].dt.tz is None:
            self.m1_data["time"] = self.m1_data["time"].dt.tz_localize("UTC")

    def _resample_full(self, freq: str) -> pd.DataFrame:
        """Resample full M1 data to higher timeframe (matching slow engine _resample_m1)."""
        df = self.m1_data[["time", "open", "high", "low", "close"]].copy()
        df = df.set_index("time")
        resampled = df.resample(freq).agg({
            "open": "first", "high": "max", "low": "min", "close": "last"
        }).dropna().reset_index()
        return resampled

    def _precompute_fractals(self, params: Dict) -> Optional[Dict]:
        """Pre-compute all H1, H4, M2 fractals for the entire M1 dataset.

        Matches slow engine (run_backtest_template.py lines 694-714):
        1. Resample M1 to H1, H4, M2
        2. Detect fractals on each
        3. H4 dedup: H4 overrides H1 at same (type, round(price, 2))
        4. Sort by confirmed_time
        5. Pre-sweep: simulate slow engine's global sweep tracking to mark
           which fractals were swept before any trade opens
        """
        be_enabled = params.get("fractal_be_enabled", False)
        tsl_enabled = params.get("fractal_tsl_enabled", False)

        if not be_enabled and not tsl_enabled:
            return None

        from src.smc.detectors.fractal_detector import detect_fractals

        h1_data = self._resample_full("1h")
        h4_data = self._resample_full("4h")
        m2_data = self._resample_full("2min")

        h1_fractals = detect_fractals(h1_data, self.symbol, "H1", candle_duration_hours=1.0)
        h4_fractals = detect_fractals(h4_data, self.symbol, "H4", candle_duration_hours=4.0)
        m2_fractals = detect_fractals(m2_data, self.symbol, "M2", candle_duration_hours=2 / 60)

        # H4 dedup: remove H1 fractals that overlap with H4 at same (type, round(price, 2))
        h4_keys = {(f.type, round(f.price, 2)) for f in h4_fractals}
        filtered_h1 = [f for f in h1_fractals if (f.type, round(f.price, 2)) not in h4_keys]

        h1h4_sorted = sorted(filtered_h1 + h4_fractals, key=lambda f: f.confirmed_time)
        m2_sorted = sorted(m2_fractals, key=lambda f: f.confirmed_time)

        # Pre-sweep: simulate the slow engine's global fractal sweep processing.
        # The slow engine checks sweeps on EVERY M1 candle (even without open positions)
        # and removes swept fractals. By trade entry time, many nearby fractals
        # are already gone. Without this, the fast engine has too many active fractals.
        self._presweep_fractals(h1h4_sorted)

        return {
            "h1h4_fractals": h1h4_sorted,
            "m2_fractals": m2_sorted,
            "be_enabled": be_enabled,
            "tsl_enabled": tsl_enabled,
        }

    def _presweep_fractals(self, h1h4_sorted: list) -> None:
        """Simulate slow engine's global sweep tracking on all M1 data.

        Sets sweep_time on each fractal to the M1 candle time when it was
        first touched by price action. This allows _simulate_after_entry
        to exclude pre-swept fractals from the initial active list.
        """
        # Use .tolist() to get tz-aware pd.Timestamp objects (not numpy datetime64)
        # so comparisons with fractal.confirmed_time (tz-aware UTC) work correctly.
        times = self.m1_data["time"].tolist()
        highs = self.m1_data["high"].values.astype(float)
        lows = self.m1_data["low"].values.astype(float)

        exp_delta_h1 = timedelta(hours=48)
        exp_delta_h4 = timedelta(hours=96)

        ptr = 0
        active = []

        for j in range(len(times)):
            t = times[j]
            hi = highs[j]
            lo = lows[j]

            # Advance pointer (newly confirmed)
            while ptr < len(h1h4_sorted) and h1h4_sorted[ptr].confirmed_time <= t:
                active.append(h1h4_sorted[ptr])
                ptr += 1

            # Expire stale (H1: 48h, H4: 96h from fractal.time)
            if active:
                exp_h1 = t - exp_delta_h1
                exp_h4 = t - exp_delta_h4
                active = [
                    f for f in active
                    if (f.timeframe == "H1" and f.time >= exp_h1) or
                       (f.timeframe == "H4" and f.time >= exp_h4)
                ]

            # Check sweeps
            swept_idx = []
            for i, frac in enumerate(active):
                if frac.type == "high" and hi >= frac.price:
                    frac.swept = True
                    frac.sweep_time = t
                    swept_idx.append(i)
                elif frac.type == "low" and lo <= frac.price:
                    frac.swept = True
                    frac.sweep_time = t
                    swept_idx.append(i)

            for i in sorted(swept_idx, reverse=True):
                active.pop(i)

    def run_with_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run backtest with given parameters. Target: ~4-6 sec.

        Args:
            params: Dictionary with keys:
                - ib_start: "HH:MM" format
                - ib_end: "HH:MM" format
                - ib_timezone: e.g., "Europe/Berlin"
                - ib_wait_minutes: int
                - trade_window_minutes: int
                - rr_target: float
                - stop_mode: "ib_start", "eq", or "cisd"
                - tsl_target: float (0 = disabled)
                - tsl_sl: float
                - min_sl_pct: float
                - rev_rb_enabled: bool
                - rev_rb_pct: float
                - ib_buffer_pct: float
                - max_distance_pct: float

        Returns:
            Dictionary with backtest results
        """
        try:
            timezone_str = params.get("ib_timezone", "Europe/Berlin")

            # Precompute IB date column for this timezone
            tz = pytz.timezone(timezone_str)
            self.m1_data["ib_date"] = self.m1_data["time"].apply(
                lambda x: x.astimezone(tz).date()
            )

            # Pre-compute fractals once for entire period (if enabled)
            fractal_ctx = self._precompute_fractals(params)

            trades = []

            # Process day-by-day
            for day_date, day_df in self.m1_data.groupby("ib_date"):
                day_df = day_df.sort_values("time").reset_index(drop=True)
                trade = self._process_day(day_df, day_date, params, fractal_ctx)
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

    def _process_day(
        self, day_df: pd.DataFrame, day_date: datetime.date, params: Dict,
        fractal_ctx: Optional[Dict] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Process single trading day.

        Signal detection on M2 bars (like ib_strategy.py).
        Trade simulation on M1 bars (for precise SL/TP detection).

        Returns trade result dict or None if no trade.
        """
        # Day-level news skip (matches IBStrategy.check_signal day-level filter)
        news_skip_events = params.get("news_skip_events", [])
        if news_skip_events and self.news_filter is not None:
            should_skip, reason = self.news_filter.should_skip_day(day_date, news_skip_events)
            if should_skip:
                return None

        # 1. Get IB from cache or compute
        ib = self._get_ib_cached(day_date, params)
        if ib is None:
            # Fallback to computing IB if not in cache
            ib = self._compute_ib(day_df, day_date, params)
            self._cache_misses += 1
        else:
            self._cache_hits += 1

        if not ib:
            return None

        ibh, ibl, eq = ib["IBH"], ib["IBL"], ib["EQ"]

        # 2. Get trade window - M1 data
        df_trade_m1 = self._get_trade_window(day_df, day_date, params)
        if df_trade_m1.empty:
            return None

        # 3. Resample M1 to M2 for signal detection (like ib_strategy.get_bars("M2"))
        df_trade_m2 = self._resample_m1_to_m2(df_trade_m1)
        if df_trade_m2.empty:
            return None

        # 4. Get pre-context for Reverse signal (also in M2)
        ib_start_ts, ib_end_ts = self._ib_window_on_date(
            day_date,
            params["ib_start"],
            params["ib_end"],
            params["ib_timezone"]
        )
        first_trade_ts = df_trade_m1["time"].iat[0]
        df_pre_context_m1 = day_df[
            (day_df["time"] >= ib_start_ts) & (day_df["time"] < first_trade_ts)
        ][["time", "open", "high", "low", "close"]].copy()

        # Resample pre-context to M2 as well
        df_pre_context_m2 = self._resample_m1_to_m2(df_pre_context_m1) if not df_pre_context_m1.empty else pd.DataFrame()

        # Trade start price for stop calculation (first M2 candle open)
        trade_start_price = float(df_trade_m2["open"].iat[0])

        # 5. Check signals in IBStrategy priority order: Reverse > OCAE > TCWE > REV_RB
        # IMPORTANT: REV_RB only triggers if all other signals fail (not based on entry_idx)
        # This matches IBStrategy's sequential signal checking behavior

        # Get trade window end time for filtering
        _, trade_window_end = self._trade_window_on_date(
            day_date,
            params["ib_end"],
            params["ib_timezone"],
            params["ib_wait_minutes"],
            params["trade_window_minutes"]
        )

        def is_signal_valid(sig):
            """Check if signal can be detected before trade window ends and not blocked by news."""
            if sig is None or sig.entry_time is None:
                return sig is not None

            if sig.signal_type in ("OCAE", "TCWE"):
                # Must wait for signal candle to close
                signal_close_time = sig.entry_time + timedelta(minutes=2)
                if signal_close_time > trade_window_end:
                    return False
                entry_time_for_news = signal_close_time
            else:
                # Reverse and REV_RB - entry_time is when we can enter
                if sig.entry_time > trade_window_end:
                    return False
                entry_time_for_news = sig.entry_time

            # Check news filter if enabled
            if self.news_filter is not None:
                # Convert to UTC for news filter
                if hasattr(entry_time_for_news, 'tz') and entry_time_for_news.tz is not None:
                    entry_utc = entry_time_for_news.tz_convert('UTC')
                elif hasattr(entry_time_for_news, 'tzinfo') and entry_time_for_news.tzinfo is not None:
                    entry_utc = entry_time_for_news.astimezone(pytz.UTC)
                else:
                    # Assume already UTC
                    entry_utc = entry_time_for_news

                allowed, _ = self.news_filter.is_trade_allowed(entry_utc)
                if not allowed:
                    self._news_filtered_count += 1
                    return False

            return True

        # Check primary signals: Reverse, OCAE, TCWE
        # Collect all valid primary signals and pick earliest by entry_idx
        primary_candidates = []

        rev_sig = self._check_reverse(df_trade_m2, df_pre_context_m2, ibh, ibl, eq, params)
        if is_signal_valid(rev_sig):
            primary_candidates.append(rev_sig)

        ocae_sig = self._check_ocae(df_trade_m2, ibh, ibl, eq, trade_start_price, params)
        if is_signal_valid(ocae_sig):
            primary_candidates.append(ocae_sig)

        tcwe_sig = self._check_tcwe(df_trade_m2, ibh, ibl, eq, trade_start_price, params)
        if is_signal_valid(tcwe_sig):
            primary_candidates.append(tcwe_sig)

        # If any primary signal found, select earliest by entry_idx
        if primary_candidates:
            # Priority for tie-breaking: Reverse > OCAE > TCWE
            priority = {"Reverse": 0, "OCAE": 1, "TCWE": 2}
            signal = min(primary_candidates, key=lambda s: (s.entry_idx, priority.get(s.signal_type, 99)))
        else:
            # Only check REV_RB if NO primary signals triggered (matches IBStrategy behavior)
            rev_rb_sig = self._check_rev_rb(df_trade_m2, ibh, ibl, eq, params)
            if is_signal_valid(rev_rb_sig):
                signal = rev_rb_sig
            else:
                return None

        # 6. Apply SMC filter (if enabled)
        if params.get("smc_enabled", False):
            signal = self._apply_smc_filter(signal, day_df, day_date, df_trade_m1, params)
            if signal is None:
                return None

        # 7. Simulate trade execution on M1 data for precise SL/TP
        return self._simulate_trade_on_m1(df_trade_m1, signal, ib, day_date, params, fractal_ctx)

    def _apply_smc_filter(
        self, signal, day_df: pd.DataFrame, day_date, df_trade_m1: pd.DataFrame, params: Dict
    ) -> Optional[Any]:
        """Apply SMC filter to detected signal. Returns modified signal or None."""
        try:
            from src.smc.fast_filter import build_smc_day_context, apply_smc_filter

            lookback_df = self._get_lookback_data(day_date, hours=params.get("smc_lookback_hours", 48))
            smc_params = {
                "enable_fractals": params.get("smc_fractals", True),
                "enable_fvg": params.get("smc_fvg", True),
                "enable_bos": params.get("smc_bos", False),
                "enable_cisd": params.get("smc_cisd", False),
                "fvg_min_size_points": params.get("smc_fvg_min_size", 0.0),
                "smc_confirmation_max_bars": params.get("smc_confirmation_max_bars", 30),
            }

            ctx = build_smc_day_context(day_df, lookback_df, self.symbol, smc_params)
            return apply_smc_filter(signal, ctx, df_trade_m1, smc_params)
        except Exception:
            # SMC filter failure should not block trade execution
            return signal

    def _get_lookback_data(self, day_date, hours: int = 48) -> pd.DataFrame:
        """Get M1 data for lookback period (prior days) for cross-day SMC context."""
        import pytz
        day_start = datetime.combine(day_date, time(0, 0))
        # Handle timezone-aware data
        first_time = self.m1_data["time"].iloc[0]
        if hasattr(first_time, 'tzinfo') and first_time.tzinfo is not None:
            day_start = pd.Timestamp(day_start, tz="UTC")

        lookback_start = day_start - timedelta(hours=hours)
        return self.m1_data[
            (self.m1_data["time"] >= lookback_start) &
            (self.m1_data["time"] < day_start)
        ].copy()

    # ========================================
    # IB and Trade Window Functions
    # ========================================

    def _parse_session_time(self, t_str: str) -> time:
        """Parse time string "HH:MM" to time object."""
        return datetime.strptime(t_str, "%H:%M").time()

    def _ib_window_on_date(
        self, local_date: datetime.date, start_str: str, end_str: str, timezone_str: str
    ) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """Get IB window timestamps for given date."""
        tz = pytz.timezone(timezone_str)
        start_naive = datetime.combine(local_date, self._parse_session_time(start_str))
        end_naive = datetime.combine(local_date, self._parse_session_time(end_str))
        start = tz.localize(start_naive)
        end = tz.localize(end_naive)
        return (pd.Timestamp(start), pd.Timestamp(end))

    def _trade_window_on_date(
        self, local_date: datetime.date, ib_end_str: str, timezone_str: str,
        wait_minutes: int, duration_minutes: int
    ) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """Get trade window timestamps after IB + wait."""
        tz = pytz.timezone(timezone_str)
        ib_end_naive = datetime.combine(local_date, self._parse_session_time(ib_end_str))
        ib_end = tz.localize(ib_end_naive)
        trade_start = ib_end + timedelta(minutes=wait_minutes)
        trade_end = trade_start + timedelta(minutes=duration_minutes)
        return (pd.Timestamp(trade_start), pd.Timestamp(trade_end))

    def _get_ib_cached(
        self, day_date: datetime.date, params: Dict
    ) -> Optional[Dict[str, float]]:
        """
        Get IB from pre-computed cache if available.

        Args:
            day_date: Trading day
            params: Parameters containing ib_start, ib_end, ib_timezone

        Returns:
            Dict with IBH, IBL, EQ or None if not in cache
        """
        if self.ib_cache is None:
            return None

        cache_key = (
            params["ib_start"],
            params["ib_end"],
            params.get("ib_timezone", "Europe/Berlin")
        )

        config_cache = self.ib_cache.get(cache_key)
        if config_cache is None:
            return None

        return config_cache.get(day_date)

    def _compute_ib(
        self, day_df: pd.DataFrame, day_date: datetime.date, params: Dict
    ) -> Optional[Dict[str, float]]:
        """Compute Initial Balance (IBH, IBL, EQ) for given day."""
        start_ib, end_ib = self._ib_window_on_date(
            day_date,
            params["ib_start"],
            params["ib_end"],
            params["ib_timezone"]
        )
        df_ib = day_df[(day_df["time"] >= start_ib) & (day_df["time"] < end_ib)]
        if df_ib.empty:
            return None
        ib_high = float(df_ib["high"].max())
        ib_low = float(df_ib["low"].min())
        eq = (ib_high + ib_low) / 2.0
        return {"IBH": ib_high, "IBL": ib_low, "EQ": eq}

    def _get_trade_window(
        self, day_df: pd.DataFrame, day_date: datetime.date, params: Dict
    ) -> pd.DataFrame:
        """Get trade window data for given day.

        Note: Uses <= end_trade to include the candle at window end.
        BacktestWrapper closes position when current_time >= position_window_end,
        which means the candle at window end IS processed and used for time exit.
        """
        start_trade, end_trade = self._trade_window_on_date(
            day_date,
            params["ib_end"],
            params["ib_timezone"],
            params["ib_wait_minutes"],
            params["trade_window_minutes"]
        )
        return day_df[
            (day_df["time"] >= start_trade) & (day_df["time"] <= end_trade)
        ].copy().reset_index(drop=True)

    def _resample_m1_to_m2(self, df_m1: pd.DataFrame) -> pd.DataFrame:
        """
        Resample M1 data to M2 for signal detection.

        Same logic as MT5Emulator.get_bars("M2", ...):
        - time: first timestamp in 2-min window
        - open: first open
        - high: max high
        - low: min low
        - close: last close

        IMPORTANT: Excludes partial M2 bars at the start where the M2 timestamp
        is before the first M1 bar. This matches MT5Emulator behavior which
        only returns complete bars.

        Args:
            df_m1: M1 DataFrame with columns: time, open, high, low, close

        Returns:
            M2 DataFrame with same columns
        """
        if df_m1.empty:
            return df_m1

        # Store first M1 bar time for filtering partial bars
        first_m1_time = df_m1["time"].iloc[0]

        df = df_m1.set_index("time")

        # Resample to 2-minute bars
        agg_dict = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
        }

        # Add tick_volume if present
        if "tick_volume" in df.columns:
            agg_dict["tick_volume"] = "sum"

        m2 = df.resample("2min", label="left", closed="left").agg(agg_dict)
        m2 = m2.dropna().reset_index()

        # Filter out partial M2 bars at the start
        # M2 bar is partial if its timestamp < first M1 bar time
        # (means the M2 bar doesn't have data for its first minute)
        m2 = m2[m2["time"] >= first_m1_time].reset_index(drop=True)

        return m2

    def _apply_spread(self, price: float, direction: str) -> float:
        """
        Return price unchanged for signal creation.

        IBStrategy calculates SL/TP from raw candle prices.
        Spread is applied later at execution time (in _simulate_trade).

        Args:
            price: Base price (candle close/open)
            direction: "long" or "short"

        Returns:
            Price unchanged (spread applied separately)
        """
        # Don't apply spread here - it's applied in _simulate_trade
        # after SL/TP are calculated from raw price
        return price

    def _get_execution_price(self, raw_price: float, direction: str) -> float:
        """
        Get execution price with half-spread applied.

        Matches MT5Emulator behavior:
          BUY: tick.ask = mid_price + half_spread
          SELL: tick.bid = mid_price - half_spread

        Args:
            raw_price: Signal price (candle close/open)
            direction: "long" or "short"

        Returns:
            Execution price with spread
        """
        spread = SPREAD_POINTS.get(self.symbol, 0.0)
        half_spread = spread / 2

        if direction == "long":
            return raw_price + half_spread  # BUY at ASK
        else:
            return raw_price - half_spread  # SELL at BID

    def _find_m1_idx_for_m2_time(
        self, df_m1: pd.DataFrame, m2_time: pd.Timestamp, for_entry: bool = True
    ) -> int:
        """
        Find M1 index that corresponds to M2 candle time.

        For entry (for_entry=True):
            M2 candle at 09:00 covers 09:00-09:02.
            Entry happens AFTER M2 closes = at 09:02.
            Find first M1 candle at or after 09:02.

        For other operations (for_entry=False):
            Find M1 candle that corresponds to M2 start time.

        Args:
            df_m1: M1 DataFrame
            m2_time: M2 candle time
            for_entry: If True, find entry point (after M2 close)

        Returns:
            Index in df_m1
        """
        if for_entry:
            # Entry after M2 candle closes
            m2_close_time = m2_time + timedelta(minutes=2)
            mask = df_m1["time"] >= m2_close_time
        else:
            # M2 candle start time
            mask = df_m1["time"] >= m2_time

        idx_arr = df_m1[mask].index
        if len(idx_arr) == 0:
            return len(df_m1) - 1
        return idx_arr[0]

    # ========================================
    # Signal Detection Functions
    # ========================================

    def _check_reverse(
        self, df_trade: pd.DataFrame, df_context_before: pd.DataFrame,
        ibh: float, ibl: float, eq: float, params: Dict
    ) -> Optional[Signal]:
        """
        Reverse signal - sweep detection with CISD.

        Ported from process_reverse_signal_fixed() in strategy_logic.py
        """
        ib_range = float(ibh - ibl)
        if ib_range <= 0 or df_trade.empty:
            return None

        ib_buffer_pct = params.get("ib_buffer_pct", 0.0)
        buffer = ib_buffer_pct * ib_range
        n = len(df_trade)

        # 1) Collect sweeps until first invalidation
        sweeps: List[Dict[str, Any]] = []
        invalid_at: Optional[int] = None

        for i in range(n):
            o = float(df_trade["open"].iat[i])
            c = float(df_trade["close"].iat[i])
            h = float(df_trade["high"].iat[i])
            lo = float(df_trade["low"].iat[i])

            # Invalidation: both open and close outside IB+buffer on same side
            if (o > ibh + buffer and c > ibh + buffer) or (o < ibl - buffer and c < ibl - buffer):
                invalid_at = i
                break

            # Upper sweep (short scenario): shadow above IBH but close within buffer zone
            if h > ibh and o <= ibh + buffer and c <= ibh + buffer:
                sweeps.append({"dir": "upper", "idx": i, "ext": h})

            # Lower sweep (long scenario): shadow below IBL but close within buffer zone
            if lo < ibl and o >= ibl - buffer and c >= ibl - buffer:
                sweeps.append({"dir": "lower", "idx": i, "ext": lo})

        if invalid_at is not None:
            sweeps = [s for s in sweeps if s["idx"] < invalid_at]

        if not sweeps:
            return None

        # 2) Group consecutive sweeps of same direction
        groups: List[Dict[str, Any]] = []
        cur = None
        for s in sweeps:
            if cur is None:
                cur = {"dir": s["dir"], "start": s["idx"], "end": s["idx"], "ext": s["ext"]}
            else:
                if s["dir"] == cur["dir"] and s["idx"] == cur["end"] + 1:
                    cur["end"] = s["idx"]
                    if s["dir"] == "upper":
                        cur["ext"] = max(cur["ext"], s["ext"])
                    else:
                        cur["ext"] = min(cur["ext"], s["ext"])
                else:
                    groups.append(cur)
                    cur = {"dir": s["dir"], "start": s["idx"], "end": s["idx"], "ext": s["ext"]}
        if cur is not None:
            groups.append(cur)

        # 3) For each group, find CISD
        for g in groups:
            prev_trade = df_trade.iloc[:g["start"]][["time", "open", "high", "low", "close"]]
            if df_context_before is not None and not df_context_before.empty:
                ctx = pd.concat([df_context_before, prev_trade], ignore_index=True)
            else:
                ctx = prev_trade.reset_index(drop=True)

            # Find last opposite candle
            last_opp = None
            for j in range(len(ctx) - 1, -1, -1):
                ro = float(ctx["open"].iat[j])
                rc = float(ctx["close"].iat[j])
                if g["dir"] == "upper":
                    if rc < ro:  # bearish
                        last_opp = {"low": float(ctx["low"].iat[j]), "high": float(ctx["high"].iat[j])}
                        break
                else:
                    if rc > ro:  # bullish
                        last_opp = {"low": float(ctx["low"].iat[j]), "high": float(ctx["high"].iat[j])}
                        break

            if last_opp is None:
                continue

            # 4) Find CISD AFTER sweep group ends
            for k in range(g["end"] + 1, n):
                ck = float(df_trade["close"].iat[k])

                if g["dir"] == "upper":  # SHORT
                    if ck < last_opp["low"]:
                        distance = ibh - ck
                        if distance >= 0.5 * ib_range:
                            break

                        # Entry on next candle open (after CISD candle closes)
                        # IBStrategy uses OPEN of entry candle for SL/TP calculation
                        # But MT5Emulator executes at tick based on CLOSE of CISD candle
                        if k + 1 < n:
                            entry_idx = k + 1
                            # entry_price = OPEN of entry candle (for SL/TP calc, like IBStrategy)
                            entry_price = float(df_trade["open"].iat[entry_idx])
                            # tick_price = CLOSE of CISD candle (for execution, like MT5Emulator)
                            tick_price = float(df_trade["close"].iat[k])
                            stop_price = float(g["ext"])  # Sweep extreme
                            entry_time = df_trade["time"].iat[entry_idx]

                            return Signal(
                                signal_type="Reverse",
                                direction="short",
                                entry_idx=entry_idx,
                                entry_price=entry_price,
                                stop_price=stop_price,
                                entry_time=entry_time,
                                tick_price=tick_price,
                                extra={"sweep_extreme": float(g["ext"]), "cisd_idx": k}
                            )

                else:  # LONG
                    if ck > last_opp["high"]:
                        distance = ck - ibl
                        if distance >= 0.5 * ib_range:
                            break

                        # Entry on next candle open (after CISD candle closes)
                        # IBStrategy uses OPEN of entry candle for SL/TP calculation
                        # But MT5Emulator executes at tick based on CLOSE of CISD candle
                        if k + 1 < n:
                            entry_idx = k + 1
                            # entry_price = OPEN of entry candle (for SL/TP calc, like IBStrategy)
                            entry_price = float(df_trade["open"].iat[entry_idx])
                            # tick_price = CLOSE of CISD candle (for execution, like MT5Emulator)
                            tick_price = float(df_trade["close"].iat[k])
                            stop_price = float(g["ext"])
                            entry_time = df_trade["time"].iat[entry_idx]

                            return Signal(
                                signal_type="Reverse",
                                direction="long",
                                entry_idx=entry_idx,
                                entry_price=entry_price,
                                stop_price=stop_price,
                                entry_time=entry_time,
                                tick_price=tick_price,
                                extra={"sweep_extreme": float(g["ext"]), "cisd_idx": k}
                            )

        return None

    def _eq_touched_before_idx(self, df: pd.DataFrame, eq: float, idx: int) -> bool:
        """Check if EQ was touched before given index."""
        sub = df.iloc[:idx + 1]
        return bool(np.any((sub["low"] <= eq) & (sub["high"] >= eq)))

    def _check_ocae(
        self, df_trade: pd.DataFrame, ibh: float, ibl: float, eq: float,
        trade_start_price: float, params: Dict
    ) -> Optional[Signal]:
        """
        OCAE - first breakout after EQ touch.

        IMPORTANT: Matches IBStrategy logic (first_breakout_bar + eq_touched_before_idx):
        1. Find the FIRST valid breakout (regardless of EQ touch)
        2. If EQ was touched before that first breakout, return OCAE signal
        3. If EQ was NOT touched before first breakout, return None (pattern invalid)

        This is different from "find first breakout WITH EQ touch" which would
        continue looking for later breakouts if first one has no EQ touch.
        """
        ib_range = ibh - ibl
        if ib_range <= 0 or df_trade.empty:
            return None

        # Per-variation override: OCAE may have different buffer than Reverse
        ib_buffer_pct = params.get("ocae_ib_buffer_pct", params.get("ib_buffer_pct", 0.0))
        max_distance_pct = params.get("ocae_max_distance_pct", params.get("max_distance_pct", 1.0))
        buffer = ib_buffer_pct * ib_range
        max_dist = max_distance_pct * ib_range

        # Step 1: Find FIRST valid breakout (matching first_breakout_bar behavior)
        first_breakout_idx = None
        first_breakout_direction = None

        for i in range(len(df_trade)):
            c = float(df_trade["close"].iat[i])

            # Long breakout
            if c > ibh + buffer:
                distance = c - ibh
                if distance <= max_dist:
                    first_breakout_idx = i
                    first_breakout_direction = "long"
                    break

            # Short breakout
            if c < ibl - buffer:
                distance = ibl - c
                if distance <= max_dist:
                    first_breakout_idx = i
                    first_breakout_direction = "short"
                    break

        # No breakout found
        if first_breakout_idx is None:
            return None

        # Step 2: Check if EQ was touched BEFORE the first breakout
        if not self._eq_touched_before_idx(df_trade, eq, first_breakout_idx):
            return None  # Pattern invalid - first breakout without EQ touch

        # Step 3: Return OCAE signal
        c = float(df_trade["close"].iat[first_breakout_idx])
        base_price = c
        entry_price = self._apply_spread(base_price, first_breakout_direction)
        ocae_stop_mode = params.get("ocae_stop_mode", params.get("stop_mode", "ib_start"))
        stop_price = self._get_stop_price(
            df_trade, first_breakout_idx, first_breakout_direction,
            entry_price, trade_start_price, eq, params, stop_mode=ocae_stop_mode
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

    def _check_tcwe(
        self, df_trade: pd.DataFrame, ibh: float, ibl: float, eq: float,
        trade_start_price: float, params: Dict
    ) -> Optional[Signal]:
        """
        TCWE - two candles without equilibrium (second candle goes further than first).

        Ported from tcwe_second_further_idx() in strategy_logic.py
        """
        n = len(df_trade)
        ib_range = ibh - ibl
        if ib_range <= 0:
            return None

        # Per-variation override: TCWE may have different buffer/distance/stop than Reverse
        ib_buffer_pct = params.get("tcwe_ib_buffer_pct", params.get("ib_buffer_pct", 0.0))
        max_distance_pct = params.get("tcwe_max_distance_pct", params.get("max_distance_pct", 1.0))
        tcwe_stop_mode = params.get("tcwe_stop_mode", params.get("stop_mode", "ib_start"))
        buffer = ib_buffer_pct * ib_range
        max_dist = max_distance_pct * ib_range

        first_idx_long = None
        first_idx_short = None

        # Find first breakout without EQ touch
        for i in range(n):
            c = float(df_trade["close"].iat[i])
            # Check long breakout with buffer
            if (c > ibh + buffer) and not self._eq_touched_before_idx(df_trade, eq, i):
                distance = c - ibh
                if distance <= max_dist:
                    first_idx_long = i
                    break
            # Check short breakout with buffer
            if (c < ibl - buffer) and not self._eq_touched_before_idx(df_trade, eq, i):
                distance = ibl - c
                if distance <= max_dist:
                    first_idx_short = i
                    break

        if first_idx_long is not None:
            c1 = float(df_trade["close"].iat[first_idx_long])
            for j in range(first_idx_long + 1, n):
                c2 = float(df_trade["close"].iat[j])
                if self._eq_touched_before_idx(df_trade, eq, j):
                    break
                # Second candle must also pass buffer and distance checks
                if (c2 > ibh + buffer) and (c2 > c1):
                    distance = c2 - ibh
                    if distance <= max_dist:
                        # Entry on close of second candle (after candle closes)
                        base_price = c2
                        entry_price = self._apply_spread(base_price, "long")
                        stop_price = self._get_stop_price(
                            df_trade, j, "long", entry_price, trade_start_price, eq, params,
                            stop_mode=tcwe_stop_mode
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
            c1 = float(df_trade["close"].iat[first_idx_short])
            for j in range(first_idx_short + 1, n):
                c2 = float(df_trade["close"].iat[j])
                if self._eq_touched_before_idx(df_trade, eq, j):
                    break
                # Second candle must also pass buffer and distance checks
                if (c2 < ibl - buffer) and (c2 < c1):
                    distance = ibl - c2
                    if distance <= max_dist:
                        # Entry on close of second candle (after candle closes)
                        base_price = c2
                        entry_price = self._apply_spread(base_price, "short")
                        stop_price = self._get_stop_price(
                            df_trade, j, "short", entry_price, trade_start_price, eq, params,
                            stop_mode=tcwe_stop_mode
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

    def _check_rev_rb(
        self, df_trade: pd.DataFrame, ibh: float, ibl: float, eq: float, params: Dict
    ) -> Optional[Signal]:
        """
        REV_RB - limit order after extended zone trigger.

        Ported from simulate_reverse_limit_both_sides() in strategy_logic.py
        """
        if not params.get("rev_rb_enabled", False) or df_trade.empty:
            return None

        ib_range = float(ibh - ibl)
        if ib_range <= 0:
            return None

        rev_rb_pct = params.get("rev_rb_pct", 0.5)
        ext_up = ibh + rev_rb_pct * ib_range
        ext_dn = ibl - rev_rb_pct * ib_range

        # Find first trigger on both sides (starting from beginning of trade window)
        j_up = None
        j_dn = None
        for j in range(len(df_trade)):
            if j_up is None and float(df_trade["high"].iat[j]) >= ext_up:
                j_up = j
            if j_dn is None and float(df_trade["low"].iat[j]) <= ext_dn:
                j_dn = j
            if j_up is not None and j_dn is not None:
                break

        # Choose earlier trigger
        chosen = None
        if j_up is not None and j_dn is not None:
            chosen = ("long", j_up) if j_up <= j_dn else ("short", j_dn)
        elif j_up is not None:
            chosen = ("long", j_up)
        elif j_dn is not None:
            chosen = ("short", j_dn)
        else:
            return None

        direction, trig_idx = chosen

        rev_rb_min_sl_pct = params.get("rev_rb_min_sl_pct", params.get("min_sl_pct", 0.001))

        if direction == "long":
            entry_level = float(ibh)
            stop = float(eq)  # SL=EQ for REV_RB
            risk = abs(entry_level - stop)

            # Check minimum SL size
            min_sl_size = entry_level * rev_rb_min_sl_pct
            if risk < min_sl_size:
                stop = entry_level - min_sl_size

            # Wait for price return to ibh for limit fill
            fill_idx = None
            for k in range(trig_idx + 1, len(df_trade)):
                if float(df_trade["low"].iat[k]) <= entry_level:
                    fill_idx = k
                    break
            if fill_idx is None:
                return None

            # Apply spread to limit entry price
            entry_price = self._apply_spread(entry_level, "long")
            entry_time = df_trade["time"].iat[fill_idx]

            return Signal(
                signal_type="REV_RB",
                direction="long",
                entry_idx=fill_idx,
                entry_price=entry_price,
                stop_price=stop,
                entry_time=entry_time,
                extra={"trigger_idx": trig_idx}
            )

        else:  # short
            entry_level = float(ibl)
            stop = float(eq)
            risk = abs(entry_level - stop)

            min_sl_size = entry_level * rev_rb_min_sl_pct
            if risk < min_sl_size:
                stop = entry_level + min_sl_size

            fill_idx = None
            for k in range(trig_idx + 1, len(df_trade)):
                if float(df_trade["high"].iat[k]) >= entry_level:
                    fill_idx = k
                    break
            if fill_idx is None:
                return None

            # Apply spread to limit entry price
            entry_price = self._apply_spread(entry_level, "short")
            entry_time = df_trade["time"].iat[fill_idx]

            return Signal(
                signal_type="REV_RB",
                direction="short",
                entry_idx=fill_idx,
                entry_price=entry_price,
                stop_price=stop,
                entry_time=entry_time,
                extra={"trigger_idx": trig_idx}
            )

    # ========================================
    # Stop Price and Trade Simulation
    # ========================================

    def _find_cisd_level(
        self, df: pd.DataFrame, direction: str, current_idx: int, entry_price: float
    ) -> Optional[float]:
        """Find CISD level for stop loss."""
        if current_idx <= 0:
            return None

        for i in range(current_idx - 1, -1, -1):
            row = df.iloc[i]
            if direction == "long":
                if row["close"] < row["open"]:
                    return float(row["low"])
            else:
                if row["close"] > row["open"]:
                    return float(row["high"])
        return None

    def _get_stop_price(
        self, df_trade: pd.DataFrame, entry_idx: int, direction: str,
        entry_price: float, trade_start_price: float, eq: float, params: Dict,
        stop_mode: Optional[str] = None
    ) -> float:
        """Get initial stop price based on STOP_MODE."""
        if stop_mode is None:
            stop_mode = params.get("stop_mode", "ib_start")

        if stop_mode.lower() == "eq":
            return float(eq)
        elif stop_mode.lower() == "cisd":
            cisd = self._find_cisd_level(df_trade, direction, entry_idx, entry_price)
            if cisd is not None:
                return float(cisd)
            return float(trade_start_price)
        else:  # "ib_start"
            return float(trade_start_price)

    def _place_sl_tp_with_min_size(
        self, direction: str, entry_price: float, stop_price: float,
        rr_target: float, min_sl_pct: float
    ) -> Tuple[float, float, bool]:
        """
        Place SL and TP with minimum SL size enforcement.

        CRITICAL: Validates that SL is on correct side of entry:
        - LONG: SL must be BELOW entry
        - SHORT: SL must be ABOVE entry

        If SL is on wrong side, it is flipped to correct side with same distance.
        """
        min_sl_size = entry_price * min_sl_pct
        adjusted = False

        # Step 1: Validate SL is on correct side for direction
        if direction == "long":
            # For LONG, SL must be BELOW entry
            if stop_price >= entry_price:
                # SL is above or at entry - flip to correct side
                stop_price = entry_price - abs(entry_price - stop_price)
                adjusted = True
        else:
            # For SHORT, SL must be ABOVE entry
            if stop_price <= entry_price:
                # SL is below or at entry - flip to correct side
                stop_price = entry_price + abs(entry_price - stop_price)
                adjusted = True

        # Step 2: Calculate risk (now guaranteed positive in correct direction)
        risk = abs(entry_price - stop_price)

        # Step 3: Apply minimum SL size if needed
        if risk < min_sl_size:
            adjusted = True
            if direction == "long":
                stop_price = entry_price - min_sl_size
            else:
                stop_price = entry_price + min_sl_size
            risk = min_sl_size

        # Step 4: Calculate TP
        tp = entry_price + rr_target * risk if direction == "long" else entry_price - rr_target * risk

        return float(stop_price), float(tp), adjusted

    def _simulate_trade(
        self, df_trade: pd.DataFrame, signal: Signal,
        ib: Dict[str, float], day_date: datetime.date, params: Dict
    ) -> Dict[str, Any]:
        """
        Simulate trade entry and exit with TSL logic.

        Matches BacktestWrapper/IBStrategy behavior:
        1. SL/TP calculated from raw signal price (signal.entry_price)
        2. Execution at tick price (signal.tick_price if available, else candle close)
        3. R calculated from execution price

        Ported from simulate_after_entry() in strategy_logic.py
        """
        direction = signal.direction
        entry_idx = signal.entry_idx
        raw_entry_price = signal.entry_price  # Raw price for SL/TP calculation

        # Resolve per-variation params
        variation = signal.signal_type.lower()
        rr_target = params.get(f"{variation}_rr_target", params.get("rr_target", 1.0))
        min_sl_pct = params.get(f"{variation}_min_sl_pct", params.get("min_sl_pct", 0.001))
        tsl_target = params.get(f"{variation}_tsl_target", params.get("tsl_target", 0.0))
        tsl_sl = params.get(f"{variation}_tsl_sl", params.get("tsl_sl", 0.5))

        # Calculate SL/TP from RAW price (like IBStrategy does)
        stop, tp, sl_adjusted = self._place_sl_tp_with_min_size(
            direction,
            raw_entry_price,  # Use raw price for SL/TP calc
            signal.stop_price,
            rr_target,
            min_sl_pct
        )

        # Get tick price for execution
        # For Reverse: tick_price is CLOSE of CISD candle (set in signal)
        # For OCAE/TCWE: tick_price should be signal.entry_price (candle CLOSE of signal candle)
        #   because entry happens right after signal candle closes, tick = signal candle close
        if signal.tick_price is not None:
            tick_mid_price = signal.tick_price
        else:
            # For OCAE/TCWE: signal.entry_price IS the tick mid-price (candle close)
            tick_mid_price = raw_entry_price

        # Apply spread to get execution price (like Emulator does)
        entry_price = self._get_execution_price(tick_mid_price, direction)

        # Simulate exit using EXECUTION price (with spread)
        exit_result = self._simulate_after_entry(
            df_trade, entry_idx, direction, entry_price, stop, tp, tsl_target, tsl_sl,
            raw_entry_price=raw_entry_price
        )

        # Risk based on execution price (matches BacktestWrapper)
        risk = abs(entry_price - stop)

        return {
            "date": day_date,
            "variation": signal.signal_type,
            "direction": direction,
            "entry_time": df_trade["time"].iat[entry_idx],
            "entry_price": entry_price,  # Execution price (with spread)
            "stop": stop,
            "tp": tp,
            "sl_adjusted": sl_adjusted,
            "exit_time": exit_result["exit_time"],
            "exit_price": exit_result["exit_price"],
            "exit_reason": exit_result["exit_reason"],
            "R": exit_result["R"],
            "profit": exit_result["R"] * risk if risk > 0 else 0,
            "ibh": ib["IBH"],
            "ibl": ib["IBL"],
            "eq": ib["EQ"],
        }

    def _simulate_trade_on_m1(
        self, df_m1: pd.DataFrame, signal: Signal,
        ib: Dict[str, float], day_date: datetime.date, params: Dict,
        fractal_ctx: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Simulate trade on M1 data for precise SL/TP detection.

        Signal was detected on M2, but execution is simulated on M1
        for more accurate SL/TP hit detection (like run_backtest_v2.py).

        Matches BacktestWrapper/IBStrategy behavior:
        1. SL/TP calculated from raw signal price (no spread)
        2. Execution at tick price (with spread)
        3. R calculated from execution price

        Args:
            df_m1: M1 trade window data
            signal: Signal detected on M2 (contains entry_time for mapping)
            ib: IB levels
            day_date: Trading day
            params: Strategy parameters

        Returns:
            Trade result dict
        """
        direction = signal.direction
        raw_entry_price = signal.entry_price  # Raw price for SL/TP calculation

        # Map M2 entry time to M1 index
        # For Reverse: entry is on OPEN of next M2 candle (entry_time is that candle's time)
        # For OCAE/TCWE: entry is after signal candle CLOSES
        # For REV_RB: entry is when limit order fills
        if signal.entry_time is not None:
            # Different logic for Reverse vs OCAE/TCWE
            if signal.signal_type == "Reverse":
                # Reverse enters on OPEN of entry_time candle
                # Find M1 candle at or after entry_time (which is M2 open time)
                m1_entry_idx = self._find_m1_idx_for_m2_time(df_m1, signal.entry_time, for_entry=False)
            else:
                # OCAE/TCWE/REV_RB enter after signal candle closes
                m1_entry_idx = self._find_m1_idx_for_m2_time(df_m1, signal.entry_time, for_entry=True)
        else:
            # Fallback: use entry_idx directly (shouldn't happen)
            m1_entry_idx = min(signal.entry_idx * 2, len(df_m1) - 1)

        # Resolve per-variation params (XAUUSD has different RR/TSL/STOP per variation)
        variation = signal.signal_type.lower()  # "reverse", "ocae", "tcwe", "rev_rb"
        rr_target = params.get(f"{variation}_rr_target", params.get("rr_target", 1.0))
        min_sl_pct = params.get(f"{variation}_min_sl_pct", params.get("min_sl_pct", 0.001))
        tsl_target = params.get(f"{variation}_tsl_target", params.get("tsl_target", 0.0))
        tsl_sl = params.get(f"{variation}_tsl_sl", params.get("tsl_sl", 0.5))

        # Calculate SL/TP from RAW price (like IBStrategy does)
        stop, tp, sl_adjusted = self._place_sl_tp_with_min_size(
            direction,
            raw_entry_price,  # Use raw price for SL/TP calc
            signal.stop_price,
            rr_target,
            min_sl_pct
        )

        # Get tick price for execution
        # For Reverse: tick_price is CLOSE of CISD candle (set in signal)
        # For OCAE/TCWE: tick_price should be signal.entry_price (candle CLOSE of signal candle)
        #   because entry happens right after signal candle closes, tick = signal candle close
        if signal.tick_price is not None:
            tick_mid_price = signal.tick_price
        else:
            # For OCAE/TCWE: signal.entry_price IS the tick mid-price (candle close)
            tick_mid_price = raw_entry_price

        # Apply spread to get execution price (like Emulator does)
        entry_price = self._get_execution_price(tick_mid_price, direction)

        # Simulate exit using EXECUTION price (with spread)
        # Pass raw_entry_price so TSL uses correct risk base (matches slow engine)
        exit_result = self._simulate_after_entry(
            df_m1, m1_entry_idx, direction, entry_price, stop, tp, tsl_target, tsl_sl,
            fractal_ctx, raw_entry_price=raw_entry_price
        )

        # Risk based on execution price (matches BacktestWrapper R calculation)
        risk = abs(entry_price - stop)

        # Get entry time from M1 (more accurate)
        entry_time_m1 = df_m1["time"].iat[m1_entry_idx] if m1_entry_idx < len(df_m1) else signal.entry_time

        return {
            "date": day_date,
            "variation": signal.signal_type,
            "direction": direction,
            "entry_time": entry_time_m1,
            "entry_price": entry_price,
            "stop": stop,
            "tp": tp,
            "sl_adjusted": sl_adjusted,
            "exit_time": exit_result["exit_time"],
            "exit_price": exit_result["exit_price"],
            "exit_reason": exit_result["exit_reason"],
            "R": exit_result["R"],
            "profit": exit_result["R"] * risk if risk > 0 else 0,
            "ibh": ib["IBH"],
            "ibl": ib["IBL"],
            "eq": ib["EQ"],
        }

    def _simulate_after_entry(
        self, df_trade: pd.DataFrame, start_idx: int, direction: str,
        entry_price: float, stop: float, tp: float, tsl_target: float, tsl_sl: float,
        fractal_ctx: Optional[Dict] = None,
        raw_entry_price: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Trade exit simulation with organic TSL + fractal BE/TSL.

        Matches slow engine (run_backtest_template.py) order of operations per M1 candle:
        1. Organic TSL update (using prev candle close, one step per candle)
        2. Fractal activation/expiry and sweep detection
        3. Fractal BE/TSL activation on sweep
        4. Effective SL = best_of(organic, fractal_be, fractal_tsl)
        5. Same-candle SL prevention: if SL changed, use prev candle's SL
        6. SL/TP hit check
        """
        risk = abs(entry_price - stop)
        if risk <= 0:
            return {
                "exit_reason": "invalid",
                "exit_time": df_trade["time"].iat[start_idx],
                "exit_price": entry_price,
                "R": 0.0
            }

        # TSL risk: slow engine uses raw_entry_price for TSL step size,
        # not execution price. SL/TP were computed from raw_entry, so
        # TSL steps must use the same risk base.
        tsl_risk = abs(raw_entry_price - stop) if raw_entry_price is not None else risk

        curr_stop = float(stop)
        curr_tp = float(tp)

        tsl_target_val = 0.0 if tsl_target is None else float(tsl_target)
        tsl_sl_val = 0.0 if tsl_sl is None else float(tsl_sl)
        no_trail = (tsl_target_val <= 0.0)

        half_spread = SPREAD_POINTS.get(self.symbol, 0.0) / 2
        is_long = direction == "long"

        # --- Fractal state initialization ---
        use_fractals = fractal_ctx is not None
        fractal_be_sl = None    # entry_price when BE activated (None = inactive)
        fractal_tsl_active = False
        fractal_tsl_sl = None   # current M2 fractal SL level

        if use_fractals:
            entry_time = df_trade["time"].iat[start_idx]
            h1h4_list = fractal_ctx["h1h4_fractals"]
            m2_list = fractal_ctx["m2_fractals"]
            be_enabled = fractal_ctx["be_enabled"]
            tsl_enabled = fractal_ctx["tsl_enabled"]

            # Find starting pointer positions (first fractal not yet confirmed at entry)
            h1h4_ptr = 0
            while h1h4_ptr < len(h1h4_list) and h1h4_list[h1h4_ptr].confirmed_time <= entry_time:
                h1h4_ptr += 1

            m2_ptr = 0
            last_m2_high = None  # price only
            last_m2_low = None   # price only
            while m2_ptr < len(m2_list) and m2_list[m2_ptr].confirmed_time <= entry_time:
                mf = m2_list[m2_ptr]
                if mf.type == "high":
                    last_m2_high = mf.price
                else:
                    last_m2_low = mf.price
                m2_ptr += 1

            # Build initial active H1/H4 fractals:
            # confirmed before entry, not expired, not pre-swept.
            # Use strict < for sweep_time: fractals swept ON the entry candle
            # are still available (slow engine sweeps them during trade, not before).
            expiry_h1 = entry_time - timedelta(hours=48)
            expiry_h4 = entry_time - timedelta(hours=96)
            active_fractals = []
            for k in range(h1h4_ptr):
                f = h1h4_list[k]
                # Skip fractals swept BEFORE entry candle (strict <)
                if f.swept and f.sweep_time < entry_time:
                    continue
                if (f.timeframe == "H1" and f.time >= expiry_h1) or \
                   (f.timeframe == "H4" and f.time >= expiry_h4):
                    active_fractals.append(f)

            # --- Entry-candle fractal sweep processing ---
            # The slow engine processes fractal sweeps on the entry candle itself
            # (same candle as position open). Replicate that here.
            entry_hi = float(df_trade["high"].iat[start_idx])
            entry_lo = float(df_trade["low"].iat[start_idx])
            swept_on_entry = []
            for fi, frac in enumerate(active_fractals):
                if frac.type == "high" and entry_hi >= frac.price:
                    swept_on_entry.append(fi)
                elif frac.type == "low" and entry_lo <= frac.price:
                    swept_on_entry.append(fi)

            if swept_on_entry:
                # Fractal BE: activate if SL in negative zone
                if be_enabled and fractal_be_sl is None:
                    sl_negative = (is_long and curr_stop < entry_price) or \
                                  (not is_long and curr_stop > entry_price)
                    if sl_negative:
                        fractal_be_sl = entry_price

                # Fractal TSL: activate
                if tsl_enabled and not fractal_tsl_active:
                    fractal_tsl_active = True

                # Remove swept fractals
                for idx in sorted(swept_on_entry, reverse=True):
                    active_fractals.pop(idx)

        # Track effective SL for same-candle prevention
        effective_sl = curr_stop
        effective_sl_prev = curr_stop

        # If fractal BE activated on entry candle, update effective_sl
        # (same-candle prevention: entry candle SL check uses original stop,
        # but effective_sl_prev is set for the NEXT candle)
        if use_fractals:
            candidates = [curr_stop]
            if fractal_be_sl is not None:
                candidates.append(fractal_be_sl)
            if fractal_tsl_active and ((is_long and last_m2_low is not None) or
                                       (not is_long and last_m2_high is not None)):
                fractal_tsl_sl = last_m2_low if is_long else last_m2_high
            effective_sl = max(candidates) if is_long else min(candidates)
            if fractal_tsl_sl is not None:
                effective_sl = max(effective_sl, fractal_tsl_sl) if is_long else min(effective_sl, fractal_tsl_sl)
            # effective_sl_prev stays as curr_stop (original) for same-candle prevention
            # Update it to effective_sl so the NEXT candle uses the BE-modified SL
            effective_sl_prev = effective_sl

        for i in range(start_idx + 1, len(df_trade)):
            lo = float(df_trade["low"].iat[i])
            hi = float(df_trade["high"].iat[i])
            t = df_trade["time"].iat[i]

            # === Step 1: TP check (no_trail only - actual TP exit) ===
            if no_trail:
                if is_long:
                    if hi >= curr_tp:
                        tp_r = (curr_tp - entry_price) / risk
                        return {"exit_reason": "tp", "exit_time": t,
                                "exit_price": curr_tp, "R": tp_r}
                else:
                    if lo <= curr_tp:
                        tp_r = (entry_price - curr_tp) / risk
                        return {"exit_reason": "tp", "exit_time": t,
                                "exit_price": curr_tp, "R": tp_r}

            # === Step 2: Organic TSL update (trail mode only) ===
            # Uses tsl_risk (raw entry based) for step size, matching slow engine's
            # IBStrategy.update_position_state() which uses tsl_state["entry_price"]
            organic_sl = curr_stop
            if not no_trail:
                prev_close = float(df_trade["close"].iat[i - 1])
                if is_long:
                    tsl_price = prev_close + half_spread
                    if tsl_price >= curr_tp:
                        new_stop = curr_stop + tsl_sl_val * tsl_risk
                        curr_stop = max(curr_stop, new_stop)
                        curr_tp = curr_tp + tsl_target_val * tsl_risk
                else:
                    tsl_price = prev_close - half_spread
                    if tsl_price <= curr_tp:
                        new_stop = curr_stop - tsl_sl_val * tsl_risk
                        curr_stop = min(curr_stop, new_stop)
                        curr_tp = curr_tp - tsl_target_val * tsl_risk
                organic_sl = curr_stop

            # === Step 3: Fractal logic ===
            if use_fractals:
                # 3a. Advance H1/H4 pointer (newly confirmed fractals)
                while h1h4_ptr < len(h1h4_list) and h1h4_list[h1h4_ptr].confirmed_time <= t:
                    active_fractals.append(h1h4_list[h1h4_ptr])
                    h1h4_ptr += 1

                # 3b. Expire stale H1/H4 fractals
                if active_fractals:
                    exp_h1 = t - timedelta(hours=48)
                    exp_h4 = t - timedelta(hours=96)
                    active_fractals = [
                        f for f in active_fractals
                        if (f.timeframe == "H1" and f.time >= exp_h1) or
                           (f.timeframe == "H4" and f.time >= exp_h4)
                    ]

                # 3c. Advance M2 pointer, update last_m2_high/low
                while m2_ptr < len(m2_list) and m2_list[m2_ptr].confirmed_time <= t:
                    mf = m2_list[m2_ptr]
                    if mf.type == "high":
                        last_m2_high = mf.price
                    else:
                        last_m2_low = mf.price
                    m2_ptr += 1

                # 3d. Check H1/H4 fractal sweeps
                swept_indices = []
                for fi, frac in enumerate(active_fractals):
                    if frac.type == "high" and hi >= frac.price:
                        swept_indices.append(fi)
                    elif frac.type == "low" and lo <= frac.price:
                        swept_indices.append(fi)

                if swept_indices:
                    # 3e. Fractal BE activation (once, if SL negative)
                    if be_enabled and fractal_be_sl is None:
                        sl_negative = (is_long and organic_sl < entry_price) or \
                                      (not is_long and organic_sl > entry_price)
                        if sl_negative:
                            fractal_be_sl = entry_price

                    # 3f. Fractal TSL activation (once)
                    if tsl_enabled and not fractal_tsl_active:
                        fractal_tsl_active = True

                    # Remove swept fractals
                    for idx in sorted(swept_indices, reverse=True):
                        active_fractals.pop(idx)

                # 3g. Update fractal TSL SL value (every candle if active)
                if fractal_tsl_active:
                    if is_long and last_m2_low is not None:
                        fractal_tsl_sl = last_m2_low
                    elif not is_long and last_m2_high is not None:
                        fractal_tsl_sl = last_m2_high

            # === Step 4: Compute effective SL ===
            candidates = [organic_sl]
            if fractal_be_sl is not None:
                candidates.append(fractal_be_sl)
            if fractal_tsl_sl is not None:
                candidates.append(fractal_tsl_sl)

            effective_sl = max(candidates) if is_long else min(candidates)

            # === Step 5: Same-candle SL prevention ===
            # If SL changed this candle (by organic TSL or fractal), use prev candle's SL
            if abs(effective_sl - effective_sl_prev) > 0.001:
                check_sl = effective_sl_prev
            else:
                check_sl = effective_sl

            # === Step 6: SL hit check ===
            if is_long:
                if lo <= check_sl:
                    exit_price = check_sl if hi >= check_sl else hi
                    r = (exit_price - entry_price) / risk
                    return {"exit_reason": "stop", "exit_time": t, "exit_price": exit_price, "R": r}
            else:
                if hi >= check_sl:
                    exit_price = check_sl if lo <= check_sl else lo
                    r = (entry_price - exit_price) / risk
                    return {"exit_reason": "stop", "exit_time": t, "exit_price": exit_price, "R": r}

            # Update prev for next candle
            effective_sl_prev = effective_sl

        # End of window - close at emulator price
        # BacktestWrapper closes at window end using tick from PREVIOUS candle:
        # - At 08:30:00, the 08:30 candle is just opening
        # - Emulator uses last CLOSED candle (08:29) close price + spread
        # - For LONG close: bid = prev_close - half_spread
        # - For SHORT close: ask = prev_close + half_spread
        half_spread = SPREAD_POINTS.get(self.symbol, 0.0) / 2
        if len(df_trade) >= 2:
            # Use second-to-last candle's close (the last CLOSED candle at window end time)
            prev_close = float(df_trade["close"].iat[-2])
        else:
            prev_close = float(df_trade["close"].iat[-1])

        if direction == "long":
            exit_price = prev_close - half_spread  # Close at bid
        else:
            exit_price = prev_close + half_spread  # Close at ask

        r = (exit_price - entry_price) / risk if direction == "long" else (entry_price - exit_price) / risk
        return {"exit_reason": "time", "exit_time": df_trade["time"].iat[-1], "exit_price": exit_price, "R": r}

    # ========================================
    # Metrics Calculation
    # ========================================

    def _calculate_metrics(self, trades: List[Dict], params: Dict) -> Dict[str, Any]:
        """Calculate backtest metrics from trade list."""
        empty_variation = {
            "total_r": 0.0,
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "winrate": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "calmar_ratio": 0.0,
            "avg_trade_r": 0.0,
            "avg_r_wins": 0.0,
            "avg_r_losses": 0.0,
            "max_consec_wins": 0,
            "max_consec_losses": 0,
        }

        if not trades:
            return {
                "total_r": 0.0,
                "total_profit": 0.0,
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "winrate": 0.0,
                "sharpe_ratio": 0.0,
                "profit_factor": 0.0,
                "max_drawdown": 0.0,
                "avg_trade_r": 0.0,
                "by_variation": {
                    "OCAE": empty_variation.copy(),
                    "TCWE": empty_variation.copy(),
                    "Reverse": empty_variation.copy(),
                    "REV_RB": empty_variation.copy(),
                },
            }

        r_values = [t["R"] for t in trades]
        profits = [t.get("profit", 0) for t in trades]

        total_trades = len(trades)
        total_r = sum(r_values)
        total_profit = sum(profits)

        wins = sum(1 for r in r_values if r > 0)
        losses = sum(1 for r in r_values if r < 0)
        counted = wins + losses
        winrate = (wins / counted * 100.0) if counted > 0 else 0.0

        avg_trade_r = total_r / total_trades if total_trades > 0 else 0.0

        # Sharpe ratio
        sharpe_ratio = self._calculate_sharpe(r_values)

        # Profit factor
        gross_profit = sum(r for r in r_values if r > 0)
        gross_loss = abs(sum(r for r in r_values if r < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

        # Max drawdown (simplified - based on cumulative R)
        max_drawdown = self._calculate_max_drawdown(r_values)

        # Calculate metrics BY VARIATION
        by_variation = self._calculate_metrics_by_variation(trades)

        # Create simple trades_by_type structure for variation_aggregator
        # (aggregator expects "trades_by_type" with {count, r} structure)
        trades_by_type = {}
        for var_name, var_stats in by_variation.items():
            trades_by_type[var_name] = {
                "count": var_stats.get("trades", 0),
                "r": var_stats.get("total_r", 0.0),
            }

        return {
            "total_r": round(total_r, 4),
            "total_profit": round(total_profit, 2),
            "total_trades": total_trades,
            "winning_trades": wins,
            "losing_trades": losses,
            "winrate": round(winrate, 2),
            "sharpe_ratio": round(sharpe_ratio, 4),
            "profit_factor": round(profit_factor, 4),
            "max_drawdown": round(max_drawdown, 2),
            "avg_trade_r": round(avg_trade_r, 4),
            "by_variation": by_variation,
            "trades_by_type": trades_by_type,
            "trades_detail": trades,
        }

    def _calculate_metrics_by_variation(self, trades: List[Dict]) -> Dict[str, Dict[str, Any]]:
        """
        Calculate full metrics for each variation type.

        Returns dict with keys: OCAE, TCWE, Reverse, REV_RB
        Each contains: total_r, trades, wins, losses, winrate, sharpe, sortino,
                       profit_factor, max_drawdown, calmar, avg_r, avg_r_wins,
                       avg_r_losses, max_consec_wins, max_consec_losses
        """
        variations = ["OCAE", "TCWE", "Reverse", "REV_RB"]
        result = {}

        for var in variations:
            var_trades = [t for t in trades if t.get("variation") == var]

            if not var_trades:
                result[var] = {
                    "total_r": 0.0,
                    "trades": 0,
                    "wins": 0,
                    "losses": 0,
                    "winrate": 0.0,
                    "sharpe_ratio": 0.0,
                    "sortino_ratio": 0.0,
                    "profit_factor": 0.0,
                    "max_drawdown": 0.0,
                    "calmar_ratio": 0.0,
                    "avg_trade_r": 0.0,
                    "avg_r_wins": 0.0,
                    "avg_r_losses": 0.0,
                    "max_consec_wins": 0,
                    "max_consec_losses": 0,
                }
                continue

            r_values = [t["R"] for t in var_trades]
            n_trades = len(var_trades)
            total_r = sum(r_values)

            # Wins/Losses
            wins = sum(1 for r in r_values if r > 0)
            losses = sum(1 for r in r_values if r < 0)
            counted = wins + losses
            winrate = (wins / counted * 100.0) if counted > 0 else 0.0

            # Avg R
            avg_trade_r = total_r / n_trades if n_trades > 0 else 0.0

            # Avg R wins/losses separately
            r_wins = [r for r in r_values if r > 0]
            r_losses = [r for r in r_values if r < 0]
            avg_r_wins = sum(r_wins) / len(r_wins) if r_wins else 0.0
            avg_r_losses = sum(r_losses) / len(r_losses) if r_losses else 0.0

            # Sharpe ratio
            sharpe_ratio = self._calculate_sharpe(r_values)

            # Sortino ratio
            sortino_ratio = self._calculate_sortino(r_values)

            # Profit factor
            gross_profit = sum(r for r in r_values if r > 0)
            gross_loss = abs(sum(r for r in r_values if r < 0))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

            # Max drawdown
            max_drawdown = self._calculate_max_drawdown(r_values)

            # Calmar ratio (total_r / max_drawdown)
            calmar_ratio = total_r / max_drawdown if max_drawdown > 0 else 0.0

            # Consecutive wins/losses
            max_consec_wins, max_consec_losses = self._calculate_consecutive_streaks(r_values)

            result[var] = {
                "total_r": round(total_r, 4),
                "trades": n_trades,
                "wins": wins,
                "losses": losses,
                "winrate": round(winrate, 2),
                "sharpe_ratio": round(sharpe_ratio, 4),
                "sortino_ratio": round(sortino_ratio, 4),
                "profit_factor": round(profit_factor, 4),
                "max_drawdown": round(max_drawdown, 4),
                "calmar_ratio": round(calmar_ratio, 4),
                "avg_trade_r": round(avg_trade_r, 4),
                "avg_r_wins": round(avg_r_wins, 4),
                "avg_r_losses": round(avg_r_losses, 4),
                "max_consec_wins": max_consec_wins,
                "max_consec_losses": max_consec_losses,
            }

        return result

    def _calculate_sortino(self, returns: List[float], risk_free_rate: float = 0.0) -> float:
        """Calculate Sortino ratio (uses only downside deviation)."""
        if len(returns) < 2:
            return 0.0

        returns_arr = np.array(returns)
        mean_return = np.mean(returns_arr) - risk_free_rate

        # Downside deviation - only negative returns
        negative_returns = returns_arr[returns_arr < 0]
        if len(negative_returns) == 0:
            return 0.0 if mean_return <= 0 else 999.0  # No losses = infinite Sortino (capped)

        downside_std = np.std(negative_returns, ddof=1)
        if downside_std == 0:
            return 0.0

        # Annualized
        return mean_return / downside_std * np.sqrt(252)

    def _calculate_consecutive_streaks(self, r_values: List[float]) -> Tuple[int, int]:
        """Calculate max consecutive wins and losses."""
        if not r_values:
            return 0, 0

        max_wins = 0
        max_losses = 0
        curr_wins = 0
        curr_losses = 0

        for r in r_values:
            if r > 0:
                curr_wins += 1
                curr_losses = 0
                max_wins = max(max_wins, curr_wins)
            elif r < 0:
                curr_losses += 1
                curr_wins = 0
                max_losses = max(max_losses, curr_losses)
            else:
                # R == 0, reset both
                curr_wins = 0
                curr_losses = 0

        return max_wins, max_losses

    def _calculate_sharpe(self, returns: List[float], risk_free_rate: float = 0.0) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) < 2:
            return 0.0

        returns_arr = np.array(returns)
        mean_return = np.mean(returns_arr) - risk_free_rate
        std_return = np.std(returns_arr, ddof=1)

        if std_return == 0:
            return 0.0

        # Annualized (assuming daily returns, ~252 trading days)
        return mean_return / std_return * np.sqrt(252)

    def _calculate_max_drawdown(self, r_values: List[float]) -> float:
        """Calculate maximum drawdown as absolute R loss from peak."""
        if not r_values:
            return 0.0

        cumulative = np.cumsum(r_values)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = running_max - cumulative

        return float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0
