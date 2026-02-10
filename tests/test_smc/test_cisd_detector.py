"""Tests for CISD detector."""
import pytest
import pandas as pd
from datetime import datetime, timedelta

from src.smc.detectors.cisd_detector import (
    detect_cisd,
    check_cisd_invalidation,
)
from src.smc.models import CISD, FVG, Fractal


def _make_data(ohlc_list):
    return pd.DataFrame(ohlc_list, columns=["time", "open", "high", "low", "close"])


class TestCISDDetector:
    """Test CISD detection with synthetic data."""

    def test_bearish_cisd_long_signal(self):
        """Downtrend delivery to zone, close above body -> long CISD."""
        data = _make_data([
            # Downtrend delivery candles (body_high of first = max(112,111) = 112)
            (datetime(2024, 1, 1, 10, 0), 112, 114, 110, 111),  # Down (delivery start)
            (datetime(2024, 1, 1, 10, 2), 111, 113, 107, 108),  # Down (delivery)
            # Test candle enters zone [101, 103]
            (datetime(2024, 1, 1, 10, 4), 108, 109, 101, 102),  # Test
            # Confirmation: close > 112 (delivery start body high)
            (datetime(2024, 1, 1, 10, 6), 102, 115, 101, 114),  # Close 114 > 112
        ])

        poi = {
            "type": "fvg",
            "direction": "bearish",
            "high": 103,
            "low": 101,
            "time": datetime(2024, 1, 1, 9, 58),
        }

        cisds = detect_cisd(data, "GER40", "M2", [poi])

        assert len(cisds) == 1
        assert cisds[0].direction == "long"
        assert cisds[0].confirmation_close == 114

    def test_bullish_cisd_short_signal(self):
        """Uptrend delivery to zone, close below body -> short CISD."""
        data = _make_data([
            # Uptrend delivery candles (body_low of first = min(98,101) = 98)
            (datetime(2024, 1, 1, 10, 0), 98, 102, 97, 101),    # Up (delivery start)
            (datetime(2024, 1, 1, 10, 2), 101, 106, 100, 105),   # Up (delivery)
            # Test candle enters zone [108, 110]
            (datetime(2024, 1, 1, 10, 4), 105, 110, 104, 109),   # Test
            # Confirmation: close < 98 (delivery start body low)
            (datetime(2024, 1, 1, 10, 6), 109, 110, 95, 96),     # Close 96 < 98
        ])

        poi = {
            "type": "fvg",
            "direction": "bullish",
            "high": 110,
            "low": 108,
            "time": datetime(2024, 1, 1, 9, 58),
        }

        cisds = detect_cisd(data, "GER40", "M2", [poi])

        assert len(cisds) == 1
        assert cisds[0].direction == "short"
        assert cisds[0].confirmation_close == 96

    def test_no_cisd_without_confirmation(self):
        """No confirmation candle -> no CISD."""
        data = _make_data([
            (datetime(2024, 1, 1, 10, 0), 110, 112, 108, 109),
            (datetime(2024, 1, 1, 10, 2), 109, 110, 105, 106),
            (datetime(2024, 1, 1, 10, 4), 106, 108, 104, 105),  # Test
            # No confirmation (stays inside delivery body range)
            (datetime(2024, 1, 1, 10, 6), 105, 108, 104, 107),
        ])

        poi = {
            "type": "fvg",
            "direction": "bearish",
            "high": 106,
            "low": 104,
            "time": datetime(2024, 1, 1, 9, 58),
        }

        cisds = detect_cisd(data, "GER40", "M2", [poi])
        assert len(cisds) == 0

    def test_empty_data(self):
        """Less than 3 bars returns no CISDs."""
        data = _make_data([
            (datetime(2024, 1, 1, 10, 0), 100, 105, 99, 103),
        ])
        cisds = detect_cisd(data, "GER40", "M2", [])
        assert len(cisds) == 0

    def test_no_poi_zones(self):
        """No POI zones returns no CISDs."""
        data = _make_data([
            (datetime(2024, 1, 1, 10, 0), 100, 105, 99, 103),
            (datetime(2024, 1, 1, 10, 2), 103, 108, 102, 107),
            (datetime(2024, 1, 1, 10, 4), 107, 110, 105, 109),
        ])
        cisds = detect_cisd(data, "GER40", "M2", [])
        assert len(cisds) == 0

    def test_cisd_id_contains_prefix(self):
        """CISD IDs start with cisd_ prefix."""
        data = _make_data([
            (datetime(2024, 1, 1, 10, 0), 112, 114, 110, 111),
            (datetime(2024, 1, 1, 10, 2), 111, 113, 107, 108),
            (datetime(2024, 1, 1, 10, 4), 108, 109, 101, 102),
            (datetime(2024, 1, 1, 10, 6), 102, 115, 101, 114),
        ])

        poi = {
            "type": "fvg",
            "direction": "bearish",
            "high": 103,
            "low": 101,
            "time": datetime(2024, 1, 1, 9, 58),
        }

        cisds = detect_cisd(data, "GER40", "M2", [poi])
        assert len(cisds) == 1
        assert cisds[0].id.startswith("cisd_")


class TestCISDInvalidation:
    """Test CISD invalidation logic."""

    def test_long_cisd_invalidated_by_bearish_fvg(self):
        """Long CISD invalidated if active bearish FVG above current price."""
        cisd = CISD(
            id="cisd_test",
            instrument="GER40",
            timeframe="M2",
            direction="long",
            delivery_candle_time=datetime(2024, 1, 1, 10),
            delivery_candle_body_high=109,
            delivery_candle_body_low=106,
            confirmation_time=datetime(2024, 1, 1, 10, 6),
            confirmation_close=110,
        )

        ohlc = _make_data([
            (datetime(2024, 1, 1, 10, 8), 110, 112, 109, 111),
        ])

        fvgs = [
            FVG(
                id="fvg_test", instrument="GER40", timeframe="M2",
                direction="bearish", high=120, low=115, midpoint=117.5,
                formation_time=datetime(2024, 1, 1, 10, 2),
                candle1_time=datetime(2024, 1, 1, 9, 58),
                candle2_time=datetime(2024, 1, 1, 10, 0),
                candle3_time=datetime(2024, 1, 1, 10, 2),
                status="active",
            ),
        ]

        assert check_cisd_invalidation(cisd, ohlc, fvgs, []) is True

    def test_long_cisd_not_invalidated_when_no_conflicts(self):
        """Long CISD valid when no conflicting structures."""
        cisd = CISD(
            id="cisd_test",
            instrument="GER40",
            timeframe="M2",
            direction="long",
            delivery_candle_time=datetime(2024, 1, 1, 10),
            delivery_candle_body_high=109,
            delivery_candle_body_low=106,
            confirmation_time=datetime(2024, 1, 1, 10, 6),
            confirmation_close=110,
        )

        ohlc = _make_data([
            (datetime(2024, 1, 1, 10, 8), 110, 112, 109, 111),
        ])

        assert check_cisd_invalidation(cisd, ohlc, [], []) is False

    def test_empty_ohlc_returns_false(self):
        """Empty OHLC data -> no invalidation."""
        cisd = CISD(
            id="cisd_test",
            instrument="GER40",
            timeframe="M2",
            direction="long",
            delivery_candle_time=datetime(2024, 1, 1, 10),
            delivery_candle_body_high=109,
            delivery_candle_body_low=106,
            confirmation_time=datetime(2024, 1, 1, 10, 6),
            confirmation_close=110,
        )

        empty_ohlc = pd.DataFrame(columns=["time", "open", "high", "low", "close"])
        assert check_cisd_invalidation(cisd, empty_ohlc, [], []) is False
