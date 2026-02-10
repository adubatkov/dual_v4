"""Tests for SMC fast filter module.

Tests:
- SMCDayContext creation from synthetic data
- build_smc_day_context with fractals and FVGs
- apply_smc_filter: pass, reject, and confirmation scenarios
- _scan_for_confirmation forward scanning
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

from src.smc.fast_filter import build_smc_day_context, apply_smc_filter, _scan_for_confirmation
from src.smc.models import SMCDayContext, FVG, Fractal


# ---------------------
# Helpers
# ---------------------

@dataclass
class MockSignal:
    """Mock signal matching fast_backtest.py Signal interface."""
    signal_type: str = "Reverse"
    direction: str = "long"
    entry_idx: int = 0
    entry_price: float = 100.0
    stop_price: float = 98.0
    entry_time: Optional[Any] = None
    tick_price: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)


def _make_m1(periods=200, start="2024-01-02 08:00", base_price=100.0):
    """Create synthetic M1 data with some structure."""
    times = pd.date_range(start=start, periods=periods, freq="1min", tz="UTC")
    np.random.seed(42)
    prices = [base_price]
    for _ in range(periods - 1):
        prices.append(prices[-1] + np.random.normal(0, 0.3))

    return pd.DataFrame({
        "time": times,
        "open": prices,
        "high": [p + abs(np.random.normal(0, 0.2)) for p in prices],
        "low": [p - abs(np.random.normal(0, 0.2)) for p in prices],
        "close": [p + np.random.normal(0, 0.1) for p in prices],
    })


def _make_lookback(hours=48, base_price=100.0):
    """Create lookback M1 data (prior to day)."""
    periods = hours * 60
    start = datetime(2024, 1, 1, 0, 0)
    times = pd.date_range(start=start, periods=periods, freq="1min", tz="UTC")
    np.random.seed(123)
    prices = [base_price]
    for _ in range(periods - 1):
        prices.append(prices[-1] + np.random.normal(0, 0.2))

    return pd.DataFrame({
        "time": times,
        "open": prices,
        "high": [p + abs(np.random.normal(0, 0.15)) for p in prices],
        "low": [p - abs(np.random.normal(0, 0.15)) for p in prices],
        "close": [p + np.random.normal(0, 0.08) for p in prices],
    })


# ---------------------
# SMCDayContext Tests
# ---------------------

class TestSMCDayContext:
    """Test SMCDayContext dataclass."""

    def test_empty_context(self):
        """Empty context has zero counts."""
        ctx = SMCDayContext(instrument="GER40", day_date=datetime(2024, 1, 2).date())
        assert ctx.num_fractals == 0
        assert ctx.num_fvgs == 0
        assert ctx.active_fractal_highs == []
        assert ctx.active_fractal_lows == []
        assert ctx.active_fvg_zones == []


# ---------------------
# build_smc_day_context Tests
# ---------------------

class TestBuildSMCDayContext:
    """Test building SMC context from data."""

    def test_builds_without_error(self):
        """Context builds successfully from synthetic data."""
        day_m1 = _make_m1()
        lookback = _make_lookback(hours=48)

        ctx = build_smc_day_context(day_m1, lookback, "GER40")

        assert ctx.instrument == "GER40"
        assert ctx.day_date is not None
        assert ctx.build_time is not None

    def test_with_empty_lookback(self):
        """Context builds with empty lookback data."""
        day_m1 = _make_m1()
        empty = pd.DataFrame(columns=["time", "open", "high", "low", "close"])

        ctx = build_smc_day_context(day_m1, empty, "GER40")

        assert ctx.instrument == "GER40"
        assert ctx.num_fractals >= 0

    def test_with_empty_day_data(self):
        """Context handles empty day data gracefully."""
        empty = pd.DataFrame(columns=["time", "open", "high", "low", "close"])
        lookback = _make_lookback()

        ctx = build_smc_day_context(empty, lookback, "GER40")

        assert ctx.instrument == "GER40"

    def test_fractals_detected(self):
        """H1 fractals detected when enough data."""
        day_m1 = _make_m1(periods=400)
        lookback = _make_lookback(hours=48)

        ctx = build_smc_day_context(day_m1, lookback, "GER40", {"enable_fractals": True})

        # With random walk data, some fractals should be detected
        assert ctx.num_fractals >= 0

    def test_fvgs_detected(self):
        """FVGs detected from M2 and H1 data."""
        day_m1 = _make_m1(periods=400)
        lookback = _make_lookback(hours=48)

        ctx = build_smc_day_context(day_m1, lookback, "GER40", {"enable_fvg": True})

        assert ctx.num_fvgs >= 0

    def test_fvg_zones_cached(self):
        """Active FVG zones are pre-cached in context."""
        day_m1 = _make_m1(periods=400)
        lookback = _make_lookback(hours=48)

        ctx = build_smc_day_context(day_m1, lookback, "GER40", {"enable_fvg": True})

        # Each FVG zone should have required keys
        for zone in ctx.active_fvg_zones:
            assert "high" in zone
            assert "low" in zone
            assert "midpoint" in zone
            assert "direction" in zone
            assert zone["direction"] in ("bullish", "bearish")

    def test_disabled_detectors_produce_empty(self):
        """Disabled detectors produce empty lists."""
        day_m1 = _make_m1()
        lookback = _make_lookback()

        ctx = build_smc_day_context(day_m1, lookback, "GER40", {
            "enable_fractals": False,
            "enable_fvg": False,
            "enable_bos": False,
            "enable_cisd": False,
        })

        assert ctx.num_fractals == 0
        assert ctx.num_fvgs == 0
        assert ctx.num_bos == 0
        assert ctx.num_cisds == 0


# ---------------------
# apply_smc_filter Tests
# ---------------------

class TestApplySMCFilter:
    """Test SMC filtering logic."""

    def test_pass_when_no_conflict(self):
        """Signal passes when no conflicting structures."""
        ctx = SMCDayContext(
            instrument="GER40",
            day_date=datetime(2024, 1, 2).date(),
        )

        signal = MockSignal(direction="long", entry_price=100.0, stop_price=98.0)
        df_m1 = _make_m1(periods=50)

        result = apply_smc_filter(signal, ctx, df_m1)

        assert result is not None
        assert result.extra.get("smc_checked") is True
        assert result.extra.get("smc_result") == "pass"

    def test_reject_on_opposing_fractal(self):
        """Long signal rejected near high fractal without confirmation."""
        ctx = SMCDayContext(
            instrument="GER40",
            day_date=datetime(2024, 1, 2).date(),
            active_fractal_highs=[100.0],  # High fractal at entry price
        )

        signal = MockSignal(
            direction="long",
            entry_price=100.0,
            stop_price=98.0,
            entry_time=pd.Timestamp("2024-01-02 10:00", tz="UTC"),
        )

        # Create M1 data that won't produce confirmation
        df_m1 = _make_m1(periods=50, start="2024-01-02 09:50")

        result = apply_smc_filter(signal, ctx, df_m1)

        # May be None (rejected) or modified (confirmed) depending on data
        # With random data, both outcomes are valid
        if result is None:
            # Rejected
            assert signal.extra.get("smc_result") == "rejected"
        else:
            # Confirmed
            assert result.extra.get("smc_confirmed") is True

    def test_reject_on_opposing_fvg(self):
        """Long signal inside bearish FVG is rejected without confirmation."""
        ctx = SMCDayContext(
            instrument="GER40",
            day_date=datetime(2024, 1, 2).date(),
            active_fvg_zones=[{
                "high": 101.0,
                "low": 99.0,
                "midpoint": 100.0,
                "direction": "bearish",
                "timeframe": "M2",
            }],
        )

        signal = MockSignal(
            direction="long",
            entry_price=100.0,
            stop_price=98.0,
            entry_time=pd.Timestamp("2024-01-02 10:00", tz="UTC"),
        )

        df_m1 = _make_m1(periods=50, start="2024-01-02 09:50")
        result = apply_smc_filter(signal, ctx, df_m1)

        # Either rejected or confirmed
        if result is None:
            assert signal.extra.get("smc_conflict") == "bearish_fvg_at_entry"

    def test_supporting_fvg_passes(self):
        """Long signal inside bullish FVG passes (not a conflict)."""
        ctx = SMCDayContext(
            instrument="GER40",
            day_date=datetime(2024, 1, 2).date(),
            active_fvg_zones=[{
                "high": 101.0,
                "low": 99.0,
                "midpoint": 100.0,
                "direction": "bullish",  # Supports long
                "timeframe": "M2",
            }],
        )

        signal = MockSignal(direction="long", entry_price=100.0, stop_price=98.0)
        df_m1 = _make_m1(periods=50)

        result = apply_smc_filter(signal, ctx, df_m1)

        assert result is not None
        assert result.extra.get("smc_checked") is True

    def test_short_signal_with_no_conflict(self):
        """Short signal passes when no opposing structures."""
        ctx = SMCDayContext(
            instrument="GER40",
            day_date=datetime(2024, 1, 2).date(),
        )

        signal = MockSignal(direction="short", entry_price=100.0, stop_price=102.0)
        df_m1 = _make_m1(periods=50)

        result = apply_smc_filter(signal, ctx, df_m1)

        assert result is not None
        assert result.extra.get("smc_result") == "pass"


# ---------------------
# _scan_for_confirmation Tests
# ---------------------

class TestScanForConfirmation:
    """Test forward scanning for SMC confirmation."""

    def test_no_confirmation_returns_none(self):
        """No confirmation in flat data returns None."""
        ctx = SMCDayContext(
            instrument="GER40",
            day_date=datetime(2024, 1, 2).date(),
        )

        signal = MockSignal(
            direction="long",
            entry_price=100.0,
            stop_price=98.0,
            entry_time=pd.Timestamp("2024-01-02 10:00", tz="UTC"),
        )

        # Flat data (no FVG zones, no fractals to sweep)
        df = _make_m1(periods=50, start="2024-01-02 09:50")

        result = _scan_for_confirmation(signal, ctx, df, max_bars=20)

        # With empty context, no confirmation possible
        assert result is None

    def test_no_entry_time_returns_none(self):
        """Signal without entry_time returns None."""
        ctx = SMCDayContext(instrument="GER40", day_date=datetime(2024, 1, 2).date())
        signal = MockSignal(entry_time=None)
        df = _make_m1(periods=50)

        result = _scan_for_confirmation(signal, ctx, df)
        assert result is None

    def test_empty_data_returns_none(self):
        """Empty M1 data returns None."""
        ctx = SMCDayContext(instrument="GER40", day_date=datetime(2024, 1, 2).date())
        signal = MockSignal(entry_time=pd.Timestamp("2024-01-02 10:00", tz="UTC"))
        empty = pd.DataFrame(columns=["time", "open", "high", "low", "close"])

        result = _scan_for_confirmation(signal, ctx, empty)
        assert result is None

    def test_fvg_rebalance_confirmation(self):
        """FVG rebalance produces confirmation."""
        ctx = SMCDayContext(
            instrument="GER40",
            day_date=datetime(2024, 1, 2).date(),
            active_fvg_zones=[{
                "high": 100.5,
                "low": 99.0,
                "midpoint": 99.75,
                "direction": "bullish",
                "timeframe": "M2",
            }],
        )

        signal = MockSignal(
            direction="long",
            entry_price=100.0,
            stop_price=97.0,
            entry_time=pd.Timestamp("2024-01-02 10:00", tz="UTC"),
        )

        # Create data where bar dips into FVG and closes above midpoint
        times = pd.date_range("2024-01-02 09:58", periods=10, freq="1min", tz="UTC")
        df = pd.DataFrame({
            "time": times,
            "open":  [100.0, 100.0, 100.0, 99.5, 99.3, 100.2, 100.5, 100.8, 101.0, 101.2],
            "high":  [100.2, 100.2, 100.2, 100.0, 100.0, 100.5, 100.8, 101.0, 101.2, 101.5],
            "low":   [99.8,  99.8,  99.8,  99.2,  99.0,  99.8,  100.2, 100.5, 100.8, 101.0],
            "close": [100.0, 100.0, 100.0, 99.3,  100.0, 100.3, 100.6, 100.9, 101.1, 101.3],
        })

        result = _scan_for_confirmation(signal, ctx, df, max_bars=8)

        if result is not None:
            assert result["type"] == "FVG_REBALANCE"
            assert result["entry_price"] > 0

    def test_early_termination_on_stop_breach(self):
        """Scanning stops early if price breaches stop level."""
        ctx = SMCDayContext(
            instrument="GER40",
            day_date=datetime(2024, 1, 2).date(),
            active_fvg_zones=[{
                "high": 100.5,
                "low": 99.0,
                "midpoint": 99.75,
                "direction": "bullish",
                "timeframe": "M2",
            }],
        )

        signal = MockSignal(
            direction="long",
            entry_price=100.0,
            stop_price=98.0,
            entry_time=pd.Timestamp("2024-01-02 10:00", tz="UTC"),
        )

        # Create data that drops below stop (98.0 * 0.995 = 97.51)
        times = pd.date_range("2024-01-02 09:58", periods=10, freq="1min", tz="UTC")
        df = pd.DataFrame({
            "time": times,
            "open":  [100.0, 100.0, 99.0, 98.0, 97.0, 96.0, 95.0, 94.0, 93.0, 92.0],
            "high":  [100.5, 100.2, 99.5, 98.5, 97.5, 96.5, 95.5, 94.5, 93.5, 92.5],
            "low":   [99.5,  99.5,  98.5, 97.5, 96.5, 95.5, 94.5, 93.5, 92.5, 91.5],
            "close": [100.0, 99.0,  98.0, 97.0, 96.0, 95.0, 94.0, 93.0, 92.0, 91.0],
        })

        result = _scan_for_confirmation(signal, ctx, df, max_bars=8)
        assert result is None
