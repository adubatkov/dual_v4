"""Tests for SMC Registry and Event Log."""
import pytest
import pandas as pd
from datetime import datetime, timedelta

from src.smc.models import FVG, Fractal, SMCEvent
from src.smc.registry import SMCRegistry
from src.smc.event_log import SMCEventLog


def _make_fvg(id_str="fvg_t1", direction="bullish", timeframe="M2",
              high=105.0, low=102.0, status="active",
              formation_time=None) -> FVG:
    """Helper to create FVG for testing."""
    ft = formation_time or datetime(2024, 1, 1, 10, 4)
    return FVG(
        id=id_str, instrument="GER40", timeframe=timeframe,
        direction=direction, high=high, low=low,
        midpoint=(high + low) / 2,
        formation_time=ft,
        candle1_time=ft - timedelta(minutes=4),
        candle2_time=ft - timedelta(minutes=2),
        candle3_time=ft,
        status=status,
    )


def _make_fractal(id_str="frac_t1", ftype="high", price=110.0,
                  timeframe="H1", time=None, swept=False) -> Fractal:
    """Helper to create Fractal for testing."""
    t = time or datetime(2024, 1, 1, 11)
    return Fractal(
        id=id_str, instrument="GER40", timeframe=timeframe,
        type=ftype, price=price, time=t,
        confirmed_time=t + timedelta(hours=2),
        swept=swept,
    )


# ==========================================
# SMCRegistry Tests
# ==========================================


class TestSMCRegistry:
    """Test SMCRegistry operations."""

    def test_add_and_get_by_id(self):
        """Add structure and retrieve by ID."""
        reg = SMCRegistry("GER40")
        fvg = _make_fvg()
        reg.add("fvg", fvg)
        assert reg.get_by_id("fvg", "fvg_t1") is fvg

    def test_get_by_id_not_found(self):
        """Non-existent ID returns None."""
        reg = SMCRegistry("GER40")
        assert reg.get_by_id("fvg", "nonexistent") is None

    def test_add_unknown_type_raises(self):
        """Adding to unknown structure type raises ValueError."""
        reg = SMCRegistry("GER40")
        fvg = _make_fvg()
        with pytest.raises(ValueError, match="Unknown structure type"):
            reg.add("unknown_type", fvg)

    def test_get_active_filters_by_status(self):
        """Only active structures returned."""
        reg = SMCRegistry("GER40")
        reg.add("fvg", _make_fvg("f1", status="active"))
        reg.add("fvg", _make_fvg("f2", status="invalidated"))

        active = reg.get_active("fvg")
        assert len(active) == 1
        assert active[0].id == "f1"

    def test_get_active_filters_by_timeframe(self):
        """Filter active by timeframe."""
        reg = SMCRegistry("GER40")
        reg.add("fvg", _make_fvg("f1", timeframe="M2"))
        reg.add("fvg", _make_fvg("f2", timeframe="H1"))

        m2_fvgs = reg.get_active("fvg", timeframe="M2")
        assert len(m2_fvgs) == 1
        assert m2_fvgs[0].id == "f1"

    def test_get_active_filters_by_direction(self):
        """Filter active by direction."""
        reg = SMCRegistry("GER40")
        reg.add("fvg", _make_fvg("f1", direction="bullish"))
        reg.add("fvg", _make_fvg("f2", direction="bearish"))

        bull = reg.get_active("fvg", direction="bullish")
        assert len(bull) == 1
        assert bull[0].id == "f1"

    def test_get_active_filters_by_time(self):
        """Filter active by before_time."""
        reg = SMCRegistry("GER40")
        reg.add("fvg", _make_fvg("f1", formation_time=datetime(2024, 1, 1, 10)))
        reg.add("fvg", _make_fvg("f2", formation_time=datetime(2024, 1, 1, 14)))

        before_noon = reg.get_active("fvg", before_time=datetime(2024, 1, 1, 12))
        assert len(before_noon) == 1
        assert before_noon[0].id == "f1"

    def test_update_status(self):
        """Update structure status and extra fields."""
        reg = SMCRegistry("GER40")
        fvg = _make_fvg("f1")
        reg.add("fvg", fvg)

        result = reg.update_status("fvg", "f1", "partial_fill", fill_pct=0.5)
        assert result is True
        assert fvg.status == "partial_fill"
        assert fvg.fill_pct == 0.5

    def test_update_status_not_found(self):
        """Updating non-existent returns False."""
        reg = SMCRegistry("GER40")
        assert reg.update_status("fvg", "nonexistent", "active") is False

    def test_update_status_removes_from_active(self):
        """After status change, no longer returned by get_active."""
        reg = SMCRegistry("GER40")
        reg.add("fvg", _make_fvg("f1"))
        reg.update_status("fvg", "f1", "invalidated")

        active = reg.get_active("fvg")
        assert len(active) == 0

    def test_get_unswept_fractals(self):
        """Filter active fractals that are not swept."""
        reg = SMCRegistry("GER40")
        reg.add("fractal", _make_fractal("fr1", swept=False))
        reg.add("fractal", _make_fractal("fr2", swept=True))

        unswept = reg.get_unswept_fractals()
        assert len(unswept) == 1
        assert unswept[0].id == "fr1"

    def test_get_fvgs_near_price_inside(self):
        """FVG containing the price is returned with 0 distance."""
        reg = SMCRegistry("GER40")
        reg.add("fvg", _make_fvg("f1", high=105, low=102))

        near = reg.get_fvgs_near_price("M2", price=103.5)
        assert len(near) == 1

    def test_get_fvgs_near_price_outside_close(self):
        """FVG close to price (within threshold) is returned."""
        reg = SMCRegistry("GER40")
        reg.add("fvg", _make_fvg("f1", high=105, low=102))

        # Price at 105.2, distance = 0.2, threshold = 105.2 * 0.005 = 0.526
        near = reg.get_fvgs_near_price("M2", price=105.2)
        assert len(near) == 1

    def test_get_fvgs_near_price_too_far(self):
        """FVG too far from price is excluded."""
        reg = SMCRegistry("GER40")
        reg.add("fvg", _make_fvg("f1", high=105, low=102))

        # Price at 120, distance = 15, threshold = 120 * 0.005 = 0.6
        near = reg.get_fvgs_near_price("M2", price=120)
        assert len(near) == 0

    def test_cleanup_removes_old(self):
        """Cleanup removes structures formed before cutoff."""
        reg = SMCRegistry("GER40")
        reg.add("fvg", _make_fvg("old", formation_time=datetime(2024, 1, 1, 8)))
        reg.add("fvg", _make_fvg("new", formation_time=datetime(2024, 1, 1, 14)))

        removed = reg.cleanup(before_time=datetime(2024, 1, 1, 12))
        assert removed == 1
        assert reg.get_by_id("fvg", "old") is None
        assert reg.get_by_id("fvg", "new") is not None

    def test_count(self):
        """Count structures by type and total."""
        reg = SMCRegistry("GER40")
        reg.add("fvg", _make_fvg("f1"))
        reg.add("fvg", _make_fvg("f2"))
        reg.add("fractal", _make_fractal("fr1"))

        assert reg.count("fvg") == 2
        assert reg.count("fractal") == 1
        assert reg.count() == 3

    def test_clear(self):
        """Clear removes everything."""
        reg = SMCRegistry("GER40")
        reg.add("fvg", _make_fvg("f1"))
        reg.add("fractal", _make_fractal("fr1"))
        reg.clear()
        assert reg.count() == 0

    def test_to_dataframe(self):
        """Export to DataFrame."""
        reg = SMCRegistry("GER40")
        reg.add("fvg", _make_fvg("f1"))
        reg.add("fvg", _make_fvg("f2"))

        df = reg.to_dataframe("fvg")
        assert len(df) == 2
        assert "id" in df.columns
        assert "direction" in df.columns

    def test_to_dataframe_empty(self):
        """Export empty type returns empty DataFrame."""
        reg = SMCRegistry("GER40")
        df = reg.to_dataframe("fvg")
        assert df.empty


# ==========================================
# SMCEventLog Tests
# ==========================================


class TestSMCEventLog:
    """Test SMCEventLog operations."""

    def test_record_and_len(self):
        """Record events and check count."""
        log = SMCEventLog()
        assert len(log) == 0

        log.record(SMCEvent(
            timestamp=datetime(2024, 1, 1, 10),
            instrument="GER40",
            event_type="fractal_detected",
            timeframe="H1",
        ))
        assert len(log) == 1

    def test_record_simple(self):
        """Convenience method records event."""
        log = SMCEventLog()
        log.record_simple(
            timestamp=datetime(2024, 1, 1, 10),
            instrument="GER40",
            event_type="fvg_detected",
            timeframe="M2",
            direction="bullish",
            price=103.5,
        )
        assert len(log) == 1
        assert log.events[0].event_type == "fvg_detected"
        assert log.events[0].direction == "bullish"

    def test_get_events_by_type(self):
        """Filter events by type."""
        log = SMCEventLog()
        log.record_simple(timestamp=datetime(2024, 1, 1, 10), instrument="GER40",
                          event_type="fractal_detected", timeframe="H1")
        log.record_simple(timestamp=datetime(2024, 1, 1, 11), instrument="GER40",
                          event_type="fvg_detected", timeframe="M2")

        fractals = log.get_events(event_type="fractal_detected")
        assert len(fractals) == 1

    def test_get_events_by_time_range(self):
        """Filter events by time range."""
        log = SMCEventLog()
        log.record_simple(timestamp=datetime(2024, 1, 1, 10), instrument="GER40",
                          event_type="e1", timeframe="H1")
        log.record_simple(timestamp=datetime(2024, 1, 1, 12), instrument="GER40",
                          event_type="e2", timeframe="H1")
        log.record_simple(timestamp=datetime(2024, 1, 1, 14), instrument="GER40",
                          event_type="e3", timeframe="H1")

        result = log.get_events(after=datetime(2024, 1, 1, 11), before=datetime(2024, 1, 1, 13))
        assert len(result) == 1
        assert result[0].event_type == "e2"

    def test_get_events_by_structure_id(self):
        """Filter events by structure ID."""
        log = SMCEventLog()
        log.record_simple(timestamp=datetime(2024, 1, 1, 10), instrument="GER40",
                          event_type="e1", timeframe="H1", structure_id="fvg_abc")
        log.record_simple(timestamp=datetime(2024, 1, 1, 11), instrument="GER40",
                          event_type="e2", timeframe="H1", structure_id="fvg_xyz")

        result = log.get_events(structure_id="fvg_abc")
        assert len(result) == 1
        assert result[0].structure_id == "fvg_abc"

    def test_get_bias_at_time(self):
        """Bias is sum of bias_impact up to time."""
        log = SMCEventLog()
        log.record_simple(timestamp=datetime(2024, 1, 1, 10), instrument="GER40",
                          event_type="fractal_detected", timeframe="H1", bias_impact=1.0)
        log.record_simple(timestamp=datetime(2024, 1, 1, 11), instrument="GER40",
                          event_type="bos_detected", timeframe="M5", bias_impact=-0.5)
        log.record_simple(timestamp=datetime(2024, 1, 1, 13), instrument="GER40",
                          event_type="fvg_detected", timeframe="M2", bias_impact=0.5)

        # Before last event
        bias_noon = log.get_bias_at_time(datetime(2024, 1, 1, 12))
        assert bias_noon == pytest.approx(0.5)  # 1.0 + (-0.5)

        # After all events
        bias_end = log.get_bias_at_time(datetime(2024, 1, 1, 14))
        assert bias_end == pytest.approx(1.0)  # 1.0 + (-0.5) + 0.5

    def test_to_dataframe(self):
        """Export to DataFrame."""
        log = SMCEventLog()
        log.record_simple(timestamp=datetime(2024, 1, 1, 10), instrument="GER40",
                          event_type="test", timeframe="H1")
        df = log.to_dataframe()
        assert len(df) == 1
        assert "event_type" in df.columns

    def test_to_dataframe_empty(self):
        """Empty log returns empty DataFrame."""
        log = SMCEventLog()
        df = log.to_dataframe()
        assert df.empty

    def test_clear(self):
        """Clear removes all events."""
        log = SMCEventLog()
        log.record_simple(timestamp=datetime(2024, 1, 1, 10), instrument="GER40",
                          event_type="test", timeframe="H1")
        log.clear()
        assert len(log) == 0

    def test_record_simple_with_details(self):
        """Extra kwargs stored in details dict."""
        log = SMCEventLog()
        log.record_simple(
            timestamp=datetime(2024, 1, 1, 10),
            instrument="GER40",
            event_type="custom",
            timeframe="H1",
            note="something important",
            value=42,
        )
        assert log.events[0].details["note"] == "something important"
        assert log.events[0].details["value"] == 42
