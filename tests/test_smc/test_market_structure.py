"""Tests for market structure detector."""
import pytest
import pandas as pd
from datetime import datetime

from src.smc.detectors.market_structure_detector import (
    detect_swing_points,
    detect_bos_choch,
    _find_swing_highs,
    _find_swing_lows,
    _classify_swings,
    _determine_trend,
)
from src.smc.models import FVG


def _make_data(ohlc_list):
    return pd.DataFrame(ohlc_list, columns=["time", "open", "high", "low", "close"])


class TestSwingDetection:
    """Test swing point detection."""

    def test_swing_high_detected_lookback_1(self):
        """Swing high with lookback=1."""
        data = _make_data([
            (datetime(2024, 1, 1, 10), 100, 105, 99, 103),
            (datetime(2024, 1, 1, 11), 103, 112, 101, 110),  # Highest
            (datetime(2024, 1, 1, 12), 110, 108, 105, 107),
        ])

        highs = _find_swing_highs(data, lookback=1)
        assert len(highs) == 1
        assert highs[0][1] == 112

    def test_swing_low_detected_lookback_1(self):
        """Swing low with lookback=1."""
        data = _make_data([
            (datetime(2024, 1, 1, 10), 100, 105, 98, 103),
            (datetime(2024, 1, 1, 11), 103, 107, 90, 100),  # Lowest
            (datetime(2024, 1, 1, 12), 100, 104, 95, 102),
        ])

        lows = _find_swing_lows(data, lookback=1)
        assert len(lows) == 1
        assert lows[0][1] == 90

    def test_no_swing_with_equal_highs(self):
        """Equal highs -> no swing (requires strictly greater)."""
        data = _make_data([
            (datetime(2024, 1, 1, 10), 100, 110, 99, 103),
            (datetime(2024, 1, 1, 11), 103, 110, 101, 107),  # Same
            (datetime(2024, 1, 1, 12), 107, 108, 104, 106),
        ])

        highs = _find_swing_highs(data, lookback=1)
        assert len(highs) == 0

    def test_multiple_swing_highs(self):
        """Multiple swing highs in longer series."""
        data = _make_data([
            (datetime(2024, 1, 1, 10), 100, 102, 99, 101),
            (datetime(2024, 1, 1, 11), 101, 110, 100, 108),  # SH
            (datetime(2024, 1, 1, 12), 108, 105, 103, 104),
            (datetime(2024, 1, 1, 13), 104, 106, 102, 105),
            (datetime(2024, 1, 1, 14), 105, 115, 104, 113),  # SH
            (datetime(2024, 1, 1, 15), 113, 112, 109, 111),
        ])

        highs = _find_swing_highs(data, lookback=1)
        assert len(highs) == 2


class TestSwingClassification:
    """Test swing point classification (HH/HL/LL/LH)."""

    def test_hh_classification(self):
        """Second high above first -> HH."""
        swings = [
            {"time": datetime(2024, 1, 1, 10), "price": 100, "type_raw": "high", "idx": 0},
            {"time": datetime(2024, 1, 1, 11), "price": 95, "type_raw": "low", "idx": 1},
            {"time": datetime(2024, 1, 1, 12), "price": 110, "type_raw": "high", "idx": 2},
        ]

        points = _classify_swings(swings)
        assert points[0].type == "high"  # First high, unclassified
        assert points[2].type == "HH"

    def test_lh_classification(self):
        """Second high below first -> LH."""
        swings = [
            {"time": datetime(2024, 1, 1, 10), "price": 110, "type_raw": "high", "idx": 0},
            {"time": datetime(2024, 1, 1, 11), "price": 95, "type_raw": "low", "idx": 1},
            {"time": datetime(2024, 1, 1, 12), "price": 105, "type_raw": "high", "idx": 2},
        ]

        points = _classify_swings(swings)
        assert points[2].type == "LH"

    def test_hl_classification(self):
        """Second low above first -> HL."""
        swings = [
            {"time": datetime(2024, 1, 1, 10), "price": 90, "type_raw": "low", "idx": 0},
            {"time": datetime(2024, 1, 1, 11), "price": 110, "type_raw": "high", "idx": 1},
            {"time": datetime(2024, 1, 1, 12), "price": 95, "type_raw": "low", "idx": 2},
        ]

        points = _classify_swings(swings)
        assert points[2].type == "HL"

    def test_ll_classification(self):
        """Second low below first -> LL."""
        swings = [
            {"time": datetime(2024, 1, 1, 10), "price": 95, "type_raw": "low", "idx": 0},
            {"time": datetime(2024, 1, 1, 11), "price": 110, "type_raw": "high", "idx": 1},
            {"time": datetime(2024, 1, 1, 12), "price": 90, "type_raw": "low", "idx": 2},
        ]

        points = _classify_swings(swings)
        assert points[2].type == "LL"

    def test_empty_swings(self):
        """Empty swings returns empty list."""
        assert _classify_swings([]) == []


class TestTrendDetermination:
    """Test trend determination from structure points."""

    def test_uptrend_from_hh_hl(self):
        """HH and HL dominant -> uptrend."""
        from src.smc.models import StructurePoint

        points = [
            StructurePoint(time=datetime(2024, 1, 1, 10), price=100, type="high"),
            StructurePoint(time=datetime(2024, 1, 1, 11), price=95, type="low"),
            StructurePoint(time=datetime(2024, 1, 1, 12), price=110, type="HH"),
            StructurePoint(time=datetime(2024, 1, 1, 13), price=100, type="HL"),
            StructurePoint(time=datetime(2024, 1, 1, 14), price=120, type="HH"),
            StructurePoint(time=datetime(2024, 1, 1, 15), price=108, type="HL"),
        ]

        assert _determine_trend(points) == "uptrend"

    def test_downtrend_from_ll_lh(self):
        """LL and LH dominant -> downtrend."""
        from src.smc.models import StructurePoint

        points = [
            StructurePoint(time=datetime(2024, 1, 1, 10), price=110, type="high"),
            StructurePoint(time=datetime(2024, 1, 1, 11), price=100, type="low"),
            StructurePoint(time=datetime(2024, 1, 1, 12), price=105, type="LH"),
            StructurePoint(time=datetime(2024, 1, 1, 13), price=95, type="LL"),
            StructurePoint(time=datetime(2024, 1, 1, 14), price=100, type="LH"),
            StructurePoint(time=datetime(2024, 1, 1, 15), price=88, type="LL"),
        ]

        assert _determine_trend(points) == "downtrend"

    def test_ranging_with_few_points(self):
        """Less than 4 points -> ranging."""
        from src.smc.models import StructurePoint

        points = [
            StructurePoint(time=datetime(2024, 1, 1, 10), price=100, type="high"),
            StructurePoint(time=datetime(2024, 1, 1, 11), price=95, type="low"),
        ]

        assert _determine_trend(points) == "ranging"


class TestDetectSwingPoints:
    """Test full detect_swing_points function."""

    def test_returns_market_structure(self):
        """Returns MarketStructure with swing points."""
        data = _make_data([
            (datetime(2024, 1, 1, 10), 100, 102, 98, 101),
            (datetime(2024, 1, 1, 11), 101, 106, 100, 105),  # Swing high
            (datetime(2024, 1, 1, 12), 105, 104, 99, 100),
            (datetime(2024, 1, 1, 13), 100, 103, 97, 102),   # Swing low
            (datetime(2024, 1, 1, 14), 102, 108, 101, 107),
        ])

        ms = detect_swing_points(data, "GER40", "H1", lookback=1)
        assert ms.instrument == "GER40"
        assert ms.timeframe == "H1"
        assert len(ms.swing_points) > 0

    def test_insufficient_data_returns_ranging(self):
        """Not enough data -> ranging."""
        data = _make_data([
            (datetime(2024, 1, 1, 10), 100, 102, 98, 101),
        ])

        ms = detect_swing_points(data, "GER40", "H1", lookback=3)
        assert ms.current_trend == "ranging"
        assert ms.swing_points == []


class TestBOSDetection:
    """Test BOS/CHoCH/MSS detection."""

    def test_bos_in_uptrend(self):
        """Break above swing high in uptrend -> BOS."""
        # Build uptrend data with clear swing points
        data = _make_data([
            (datetime(2024, 1, 1, 10), 100, 102, 98, 101),
            (datetime(2024, 1, 1, 11), 101, 108, 100, 107),  # SH 108
            (datetime(2024, 1, 1, 12), 107, 106, 103, 104),
            (datetime(2024, 1, 1, 13), 104, 105, 101, 102),  # SL 101
            (datetime(2024, 1, 1, 14), 102, 112, 101, 111),  # SH 112 (HH)
            (datetime(2024, 1, 1, 15), 111, 110, 107, 108),
            (datetime(2024, 1, 1, 16), 108, 109, 105, 106),  # SL 105 (HL)
            (datetime(2024, 1, 1, 17), 106, 115, 105, 114),  # Break above 112 -> BOS
        ])

        ms = detect_swing_points(data, "GER40", "M5", lookback=1)
        bos_list = detect_bos_choch(ms, data, "GER40", "M5")

        # Should detect breaks
        bullish_breaks = [b for b in bos_list if b.direction == "bullish"]
        assert len(bullish_breaks) >= 1

    def test_choch_reversal(self):
        """Break below swing low in uptrend -> CHoCH."""
        # Data with clear alternating swing highs and lows
        # Lookback=1: SH at [i] needs h[i] > h[i-1] AND h[i] > h[i+1]
        #             SL at [i] needs l[i] < l[i-1] AND l[i] < l[i+1]
        data = _make_data([
            (datetime(2024, 1, 1, 10), 100, 105, 99, 103),
            (datetime(2024, 1, 1, 11), 103, 115, 102, 112),  # SH=115
            (datetime(2024, 1, 1, 12), 112, 108, 104, 106),
            (datetime(2024, 1, 1, 13), 106, 107, 93, 95),    # SL=93 (104>93, check [4])
            (datetime(2024, 1, 1, 14), 95, 120, 96, 118),    # SH=120 (93<96: SL confirmed)
            (datetime(2024, 1, 1, 15), 118, 116, 110, 112),
            (datetime(2024, 1, 1, 16), 112, 114, 103, 105),  # SL=103 (110>103, check [7])
            (datetime(2024, 1, 1, 17), 105, 107, 104, 106),  # 103<104: SL[6] confirmed
            (datetime(2024, 1, 1, 18), 106, 108, 80, 82),    # Break below 93 -> bearish
        ])

        ms = detect_swing_points(data, "GER40", "M5", lookback=1)
        bos_list = detect_bos_choch(ms, data, "GER40", "M5")

        bearish_breaks = [b for b in bos_list if b.direction == "bearish"]
        assert len(bearish_breaks) >= 1

    def test_empty_structure_returns_empty(self):
        """No swing points -> no BOS."""
        from src.smc.models import MarketStructure

        ms = MarketStructure(
            instrument="GER40",
            timeframe="M5",
            swing_points=[],
            current_trend="ranging",
        )

        data = _make_data([
            (datetime(2024, 1, 1, 10), 100, 102, 98, 101),
        ])

        assert detect_bos_choch(ms, data, "GER40", "M5") == []

    def test_bos_type_field(self):
        """BOS objects have correct bos_type field."""
        data = _make_data([
            (datetime(2024, 1, 1, 10), 100, 102, 98, 101),
            (datetime(2024, 1, 1, 11), 101, 108, 100, 107),
            (datetime(2024, 1, 1, 12), 107, 106, 103, 104),
            (datetime(2024, 1, 1, 13), 104, 105, 101, 102),
            (datetime(2024, 1, 1, 14), 102, 112, 101, 111),
            (datetime(2024, 1, 1, 15), 111, 110, 107, 108),
            (datetime(2024, 1, 1, 16), 108, 109, 105, 106),
            (datetime(2024, 1, 1, 17), 106, 115, 105, 114),
        ])

        ms = detect_swing_points(data, "GER40", "M5", lookback=1)
        bos_list = detect_bos_choch(ms, data, "GER40", "M5")

        for bos in bos_list:
            assert bos.bos_type in ("bos", "choch", "mss")
            assert bos.direction in ("bullish", "bearish")
            assert bos.id.startswith("bos_")
