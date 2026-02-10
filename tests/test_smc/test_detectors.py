"""Tests for SMC detectors - fractals and FVGs."""
import pytest
import pandas as pd
from datetime import datetime, timedelta

from src.smc.detectors.fractal_detector import (
    check_fractal_sweep,
    detect_fractals,
    find_unswept_fractals,
)
from src.smc.detectors.fvg_detector import (
    check_fvg_fill,
    check_fvg_rebalance,
    detect_fvg,
)
from src.smc.models import FVG, Fractal


def _make_ohlc(ohlc_list):
    """Create DataFrame from list of (time, o, h, l, c) tuples."""
    return pd.DataFrame(ohlc_list, columns=["time", "open", "high", "low", "close"])


# ==========================================
# Fractal Detector Tests
# ==========================================


class TestFractalDetection:
    """Test Williams 3-bar fractal detection."""

    def test_high_fractal_detected(self):
        """Center bar high > neighbors -> high fractal."""
        data = _make_ohlc([
            (datetime(2024, 1, 1, 10), 100, 105, 99, 103),
            (datetime(2024, 1, 1, 11), 103, 110, 101, 107),  # highest high
            (datetime(2024, 1, 1, 12), 107, 108, 104, 106),
        ])
        fractals = detect_fractals(data, "GER40", "H1")
        assert len(fractals) == 1
        assert fractals[0].type == "high"
        assert fractals[0].price == 110

    def test_low_fractal_detected(self):
        """Center bar low < neighbors -> low fractal."""
        data = _make_ohlc([
            (datetime(2024, 1, 1, 10), 100, 108, 98, 103),  # high=108 > center
            (datetime(2024, 1, 1, 11), 103, 107, 95, 100),  # lowest low, high < prev
            (datetime(2024, 1, 1, 12), 100, 108, 97, 102),  # high=108 > center
        ])
        fractals = detect_fractals(data, "GER40", "H1")
        assert len(fractals) == 1
        assert fractals[0].type == "low"
        assert fractals[0].price == 95

    def test_both_high_and_low_on_same_bar(self):
        """One bar can be both high and low fractal."""
        data = _make_ohlc([
            (datetime(2024, 1, 1, 10), 100, 105, 98, 103),
            (datetime(2024, 1, 1, 11), 103, 115, 90, 100),  # extreme bar
            (datetime(2024, 1, 1, 12), 100, 104, 97, 102),
        ])
        fractals = detect_fractals(data, "GER40", "H1")
        assert len(fractals) == 2
        types = {f.type for f in fractals}
        assert types == {"high", "low"}

    def test_no_fractal_when_equal_highs(self):
        """Equal highs -> no high fractal (requires strictly greater)."""
        data = _make_ohlc([
            (datetime(2024, 1, 1, 10), 100, 110, 99, 103),
            (datetime(2024, 1, 1, 11), 103, 110, 101, 107),  # same high
            (datetime(2024, 1, 1, 12), 107, 108, 104, 106),
        ])
        fractals = detect_fractals(data, "GER40", "H1")
        high_fractals = [f for f in fractals if f.type == "high"]
        assert len(high_fractals) == 0

    def test_no_fractal_when_equal_lows(self):
        """Equal lows -> no low fractal (requires strictly less)."""
        data = _make_ohlc([
            (datetime(2024, 1, 1, 10), 100, 105, 95, 103),
            (datetime(2024, 1, 1, 11), 103, 107, 95, 100),  # same low
            (datetime(2024, 1, 1, 12), 100, 104, 97, 102),
        ])
        fractals = detect_fractals(data, "GER40", "H1")
        low_fractals = [f for f in fractals if f.type == "low"]
        assert len(low_fractals) == 0

    def test_confirmed_time(self):
        """Fractal confirmed after 3rd bar closes."""
        data = _make_ohlc([
            (datetime(2024, 1, 1, 10), 100, 105, 99, 103),
            (datetime(2024, 1, 1, 11), 103, 110, 101, 107),
            (datetime(2024, 1, 1, 12), 107, 108, 104, 106),
        ])
        fractals = detect_fractals(data, "GER40", "H1", candle_duration_hours=1.0)
        assert fractals[0].confirmed_time == datetime(2024, 1, 1, 13)

    def test_multiple_fractals_in_series(self):
        """Detect multiple fractals in longer data series."""
        data = _make_ohlc([
            (datetime(2024, 1, 1, 10), 100, 105, 99, 103),
            (datetime(2024, 1, 1, 11), 103, 110, 101, 107),  # high fractal
            (datetime(2024, 1, 1, 12), 107, 108, 104, 106),
            (datetime(2024, 1, 1, 13), 106, 107, 98, 99),   # low fractal
            (datetime(2024, 1, 1, 14), 99, 103, 99, 102),
        ])
        fractals = detect_fractals(data, "GER40", "H1")
        assert len(fractals) == 2
        assert fractals[0].type == "high"
        assert fractals[1].type == "low"

    def test_empty_data_returns_empty(self):
        """Less than 3 bars returns no fractals."""
        data = _make_ohlc([
            (datetime(2024, 1, 1, 10), 100, 105, 99, 103),
            (datetime(2024, 1, 1, 11), 103, 110, 101, 107),
        ])
        assert detect_fractals(data, "GER40", "H1") == []

    def test_fractal_id_is_deterministic(self):
        """Same data produces same fractal ID."""
        data = _make_ohlc([
            (datetime(2024, 1, 1, 10), 100, 105, 99, 103),
            (datetime(2024, 1, 1, 11), 103, 110, 101, 107),
            (datetime(2024, 1, 1, 12), 107, 108, 104, 106),
        ])
        f1 = detect_fractals(data, "GER40", "H1")
        f2 = detect_fractals(data, "GER40", "H1")
        assert f1[0].id == f2[0].id


class TestFindUnsweptFractals:
    """Test unswept fractal filtering."""

    def test_unswept_fractal_returned(self):
        """Fractal not touched by price remains unswept."""
        fractals = [
            Fractal(
                id="f1", instrument="GER40", timeframe="H1",
                type="high", price=110.0,
                time=datetime(2024, 1, 1, 11),
                confirmed_time=datetime(2024, 1, 1, 13),
            )
        ]
        # M1 data that never reaches 110
        m1 = _make_ohlc([
            (datetime(2024, 1, 1, 13, 0), 105, 108, 104, 107),
            (datetime(2024, 1, 1, 13, 1), 107, 109, 106, 108),
        ])
        result = find_unswept_fractals(fractals, m1, before_time=datetime(2024, 1, 1, 14))
        assert len(result) == 1
        assert result[0].id == "f1"

    def test_swept_fractal_excluded(self):
        """Fractal touched by price is swept -> excluded."""
        fractals = [
            Fractal(
                id="f1", instrument="GER40", timeframe="H1",
                type="high", price=110.0,
                time=datetime(2024, 1, 1, 11),
                confirmed_time=datetime(2024, 1, 1, 13),
            )
        ]
        # M1 data that reaches 110
        m1 = _make_ohlc([
            (datetime(2024, 1, 1, 13, 0), 105, 108, 104, 107),
            (datetime(2024, 1, 1, 13, 1), 107, 111, 106, 110),  # high >= 110
        ])
        result = find_unswept_fractals(fractals, m1, before_time=datetime(2024, 1, 1, 14))
        assert len(result) == 0

    def test_low_fractal_swept(self):
        """Low fractal swept when price reaches below its level."""
        fractals = [
            Fractal(
                id="f2", instrument="GER40", timeframe="H1",
                type="low", price=95.0,
                time=datetime(2024, 1, 1, 11),
                confirmed_time=datetime(2024, 1, 1, 13),
            )
        ]
        m1 = _make_ohlc([
            (datetime(2024, 1, 1, 13, 0), 98, 99, 94, 97),  # low <= 95
        ])
        result = find_unswept_fractals(fractals, m1, before_time=datetime(2024, 1, 1, 14))
        assert len(result) == 0


class TestCheckFractalSweep:
    """Test fractal sweep detection in time window."""

    def test_sweep_detected(self):
        """High fractal swept returns sweep time."""
        frac = Fractal(
            id="f1", instrument="GER40", timeframe="H1",
            type="high", price=110.0,
            time=datetime(2024, 1, 1, 11),
            confirmed_time=datetime(2024, 1, 1, 13),
        )
        m1 = _make_ohlc([
            (datetime(2024, 1, 1, 14, 0), 108, 109, 107, 108),
            (datetime(2024, 1, 1, 14, 1), 109, 111, 108, 110),  # swept here
            (datetime(2024, 1, 1, 14, 2), 110, 112, 109, 111),
        ])
        result = check_fractal_sweep(
            frac, m1,
            window_start=datetime(2024, 1, 1, 14),
            window_end=datetime(2024, 1, 1, 15),
        )
        assert result == datetime(2024, 1, 1, 14, 1)

    def test_no_sweep_returns_none(self):
        """No sweep returns None."""
        frac = Fractal(
            id="f1", instrument="GER40", timeframe="H1",
            type="high", price=115.0,
            time=datetime(2024, 1, 1, 11),
            confirmed_time=datetime(2024, 1, 1, 13),
        )
        m1 = _make_ohlc([
            (datetime(2024, 1, 1, 14, 0), 108, 109, 107, 108),
            (datetime(2024, 1, 1, 14, 1), 109, 111, 108, 110),
        ])
        result = check_fractal_sweep(
            frac, m1,
            window_start=datetime(2024, 1, 1, 14),
            window_end=datetime(2024, 1, 1, 15),
        )
        assert result is None


# ==========================================
# FVG Detector Tests
# ==========================================


class TestFVGDetection:
    """Test Fair Value Gap detection."""

    def test_bullish_fvg_detected(self):
        """Gap between c1.high and c3.low -> bullish FVG."""
        data = _make_ohlc([
            (datetime(2024, 1, 1, 10, 0), 100, 102, 99, 101),   # c1: high=102
            (datetime(2024, 1, 1, 10, 2), 103, 108, 102, 107),  # c2: strong up
            (datetime(2024, 1, 1, 10, 4), 107, 110, 105, 109),  # c3: low=105
        ])
        fvgs = detect_fvg(data, "GER40", "M2")
        assert len(fvgs) == 1
        assert fvgs[0].direction == "bullish"
        assert fvgs[0].low == 102
        assert fvgs[0].high == 105

    def test_bearish_fvg_detected(self):
        """Gap between c3.high and c1.low -> bearish FVG."""
        data = _make_ohlc([
            (datetime(2024, 1, 1, 10, 0), 110, 112, 108, 109),  # c1: low=108
            (datetime(2024, 1, 1, 10, 2), 107, 108, 102, 103),  # c2: strong drop
            (datetime(2024, 1, 1, 10, 4), 103, 105, 101, 104),  # c3: high=105
        ])
        fvgs = detect_fvg(data, "GER40", "M2")
        assert len(fvgs) == 1
        assert fvgs[0].direction == "bearish"
        assert fvgs[0].low == 105
        assert fvgs[0].high == 108

    def test_no_fvg_when_wicks_overlap(self):
        """Overlapping c1/c3 wicks -> no FVG."""
        data = _make_ohlc([
            (datetime(2024, 1, 1, 10, 0), 100, 106, 99, 103),
            (datetime(2024, 1, 1, 10, 2), 103, 108, 102, 107),
            (datetime(2024, 1, 1, 10, 4), 107, 110, 104, 109),  # c3.low=104 < c1.high=106
        ])
        fvgs = detect_fvg(data, "GER40", "M2", direction="bullish")
        assert len(fvgs) == 0

    def test_direction_filter(self):
        """Filter by direction returns only matching FVGs."""
        # This data has a bearish FVG
        data = _make_ohlc([
            (datetime(2024, 1, 1, 10, 0), 110, 112, 108, 109),
            (datetime(2024, 1, 1, 10, 2), 107, 108, 102, 103),
            (datetime(2024, 1, 1, 10, 4), 103, 105, 101, 104),
        ])
        assert len(detect_fvg(data, "GER40", "M2", direction="bearish")) == 1
        assert len(detect_fvg(data, "GER40", "M2", direction="bullish")) == 0

    def test_min_size_filter(self):
        """FVGs smaller than min_size_points are excluded."""
        data = _make_ohlc([
            (datetime(2024, 1, 1, 10, 0), 100, 102, 99, 101),
            (datetime(2024, 1, 1, 10, 2), 103, 108, 102, 107),
            (datetime(2024, 1, 1, 10, 4), 107, 110, 105, 109),
        ])
        # Gap size = 105 - 102 = 3 points
        assert len(detect_fvg(data, "GER40", "M2", min_size_points=2.0)) == 1
        assert len(detect_fvg(data, "GER40", "M2", min_size_points=5.0)) == 0

    def test_fvg_midpoint(self):
        """Midpoint is correctly calculated."""
        data = _make_ohlc([
            (datetime(2024, 1, 1, 10, 0), 100, 102, 99, 101),
            (datetime(2024, 1, 1, 10, 2), 103, 108, 102, 107),
            (datetime(2024, 1, 1, 10, 4), 107, 110, 105, 109),
        ])
        fvgs = detect_fvg(data, "GER40", "M2")
        assert fvgs[0].midpoint == pytest.approx(103.5)

    def test_fvg_formation_time(self):
        """Formation time = 3rd candle time."""
        data = _make_ohlc([
            (datetime(2024, 1, 1, 10, 0), 100, 102, 99, 101),
            (datetime(2024, 1, 1, 10, 2), 103, 108, 102, 107),
            (datetime(2024, 1, 1, 10, 4), 107, 110, 105, 109),
        ])
        fvgs = detect_fvg(data, "GER40", "M2")
        assert fvgs[0].formation_time == datetime(2024, 1, 1, 10, 4)

    def test_empty_data_returns_empty(self):
        """Less than 3 bars returns no FVGs."""
        data = _make_ohlc([
            (datetime(2024, 1, 1, 10, 0), 100, 102, 99, 101),
        ])
        assert detect_fvg(data, "GER40", "M2") == []

    def test_fvg_id_deterministic(self):
        """Same data produces same FVG ID."""
        data = _make_ohlc([
            (datetime(2024, 1, 1, 10, 0), 100, 102, 99, 101),
            (datetime(2024, 1, 1, 10, 2), 103, 108, 102, 107),
            (datetime(2024, 1, 1, 10, 4), 107, 110, 105, 109),
        ])
        f1 = detect_fvg(data, "GER40", "M2")
        f2 = detect_fvg(data, "GER40", "M2")
        assert f1[0].id == f2[0].id


class TestFVGFill:
    """Test FVG fill checking."""

    def test_bullish_fvg_partial_fill(self):
        """Price enters bullish FVG but doesn't fill completely."""
        fvg = FVG(
            id="test_fvg", instrument="GER40", timeframe="M2",
            direction="bullish", high=105, low=102, midpoint=103.5,
            formation_time=datetime(2024, 1, 1, 10, 4),
            candle1_time=datetime(2024, 1, 1, 10, 0),
            candle2_time=datetime(2024, 1, 1, 10, 2),
            candle3_time=datetime(2024, 1, 1, 10, 4),
        )
        # Price dips to 103 (into the gap but not to 102)
        m1 = _make_ohlc([
            (datetime(2024, 1, 1, 10, 5), 106, 107, 103, 104),
        ])
        result = check_fvg_fill(
            fvg, m1,
            after_time=datetime(2024, 1, 1, 10, 4),
            up_to_time=datetime(2024, 1, 1, 11),
        )
        assert result is not None
        assert result["fill_type"] == "partial"
        # penetration = 105 - max(103, 102) = 2, gap = 3, fill = 2/3
        assert result["fill_pct"] == pytest.approx(2 / 3)

    def test_bullish_fvg_full_fill(self):
        """Price fills bullish FVG completely."""
        fvg = FVG(
            id="test_fvg", instrument="GER40", timeframe="M2",
            direction="bullish", high=105, low=102, midpoint=103.5,
            formation_time=datetime(2024, 1, 1, 10, 4),
            candle1_time=datetime(2024, 1, 1, 10, 0),
            candle2_time=datetime(2024, 1, 1, 10, 2),
            candle3_time=datetime(2024, 1, 1, 10, 4),
        )
        m1 = _make_ohlc([
            (datetime(2024, 1, 1, 10, 5), 106, 107, 101, 103),  # low=101 < 102
        ])
        result = check_fvg_fill(
            fvg, m1,
            after_time=datetime(2024, 1, 1, 10, 4),
            up_to_time=datetime(2024, 1, 1, 11),
        )
        assert result is not None
        assert result["fill_type"] == "full"
        assert result["fill_pct"] == pytest.approx(1.0)

    def test_bearish_fvg_partial_fill(self):
        """Price rises into bearish FVG partially."""
        fvg = FVG(
            id="test_fvg", instrument="GER40", timeframe="M2",
            direction="bearish", high=108, low=105, midpoint=106.5,
            formation_time=datetime(2024, 1, 1, 10, 4),
            candle1_time=datetime(2024, 1, 1, 10, 0),
            candle2_time=datetime(2024, 1, 1, 10, 2),
            candle3_time=datetime(2024, 1, 1, 10, 4),
        )
        m1 = _make_ohlc([
            (datetime(2024, 1, 1, 10, 5), 103, 107, 102, 106),  # high=107 into gap
        ])
        result = check_fvg_fill(
            fvg, m1,
            after_time=datetime(2024, 1, 1, 10, 4),
            up_to_time=datetime(2024, 1, 1, 11),
        )
        assert result is not None
        assert result["fill_type"] == "partial"
        # penetration = min(107, 108) - 105 = 2, gap = 3, fill = 2/3
        assert result["fill_pct"] == pytest.approx(2 / 3)

    def test_no_fill_returns_none(self):
        """Price doesn't touch FVG returns None."""
        fvg = FVG(
            id="test_fvg", instrument="GER40", timeframe="M2",
            direction="bullish", high=105, low=102, midpoint=103.5,
            formation_time=datetime(2024, 1, 1, 10, 4),
            candle1_time=datetime(2024, 1, 1, 10, 0),
            candle2_time=datetime(2024, 1, 1, 10, 2),
            candle3_time=datetime(2024, 1, 1, 10, 4),
        )
        m1 = _make_ohlc([
            (datetime(2024, 1, 1, 10, 5), 106, 108, 105, 107),  # stays above
        ])
        result = check_fvg_fill(
            fvg, m1,
            after_time=datetime(2024, 1, 1, 10, 4),
            up_to_time=datetime(2024, 1, 1, 11),
        )
        assert result is None


class TestFVGRebalance:
    """Test FVG rebalance detection."""

    def test_bullish_rebalance(self):
        """Candle dips into bullish FVG and closes above."""
        fvg = FVG(
            id="test", instrument="GER40", timeframe="M2",
            direction="bullish", high=105, low=102, midpoint=103.5,
            formation_time=datetime(2024, 1, 1, 10, 4),
            candle1_time=datetime(2024, 1, 1, 10, 0),
            candle2_time=datetime(2024, 1, 1, 10, 2),
            candle3_time=datetime(2024, 1, 1, 10, 4),
        )
        candle = pd.Series({"high": 108, "low": 103, "close": 106})
        assert check_fvg_rebalance(fvg, candle) is True

    def test_bullish_no_rebalance_close_inside(self):
        """Candle enters but closes inside FVG -> no rebalance."""
        fvg = FVG(
            id="test", instrument="GER40", timeframe="M2",
            direction="bullish", high=105, low=102, midpoint=103.5,
            formation_time=datetime(2024, 1, 1, 10, 4),
            candle1_time=datetime(2024, 1, 1, 10, 0),
            candle2_time=datetime(2024, 1, 1, 10, 2),
            candle3_time=datetime(2024, 1, 1, 10, 4),
        )
        candle = pd.Series({"high": 108, "low": 103, "close": 104})
        assert check_fvg_rebalance(fvg, candle) is False

    def test_bullish_no_rebalance_no_entry(self):
        """Candle stays above FVG -> no entry -> no rebalance."""
        fvg = FVG(
            id="test", instrument="GER40", timeframe="M2",
            direction="bullish", high=105, low=102, midpoint=103.5,
            formation_time=datetime(2024, 1, 1, 10, 4),
            candle1_time=datetime(2024, 1, 1, 10, 0),
            candle2_time=datetime(2024, 1, 1, 10, 2),
            candle3_time=datetime(2024, 1, 1, 10, 4),
        )
        candle = pd.Series({"high": 112, "low": 106, "close": 110})
        assert check_fvg_rebalance(fvg, candle) is False

    def test_bearish_rebalance(self):
        """Candle rises into bearish FVG and closes below."""
        fvg = FVG(
            id="test", instrument="GER40", timeframe="M2",
            direction="bearish", high=108, low=105, midpoint=106.5,
            formation_time=datetime(2024, 1, 1, 10, 4),
            candle1_time=datetime(2024, 1, 1, 10, 0),
            candle2_time=datetime(2024, 1, 1, 10, 2),
            candle3_time=datetime(2024, 1, 1, 10, 4),
        )
        candle = pd.Series({"high": 107, "low": 102, "close": 103})
        assert check_fvg_rebalance(fvg, candle) is True

    def test_bearish_no_rebalance_close_inside(self):
        """Candle enters bearish FVG but closes inside -> no rebalance."""
        fvg = FVG(
            id="test", instrument="GER40", timeframe="M2",
            direction="bearish", high=108, low=105, midpoint=106.5,
            formation_time=datetime(2024, 1, 1, 10, 4),
            candle1_time=datetime(2024, 1, 1, 10, 0),
            candle2_time=datetime(2024, 1, 1, 10, 2),
            candle3_time=datetime(2024, 1, 1, 10, 4),
        )
        candle = pd.Series({"high": 107, "low": 102, "close": 106})
        assert check_fvg_rebalance(fvg, candle) is False
