"""Tests for Phase 5 SMC validation methods.

Tests:
- FVG validation (structural_update, zone_to_zone, full_fill, inverted)
- BOS validation (opposing zones check)
- CISD validation (opposing zones check)
- OrderFlow validation (step confirmations, invalidation)
- Entry validation (Model 1 / Model 2)
- TP target validation
- SL placement validation
- BE move validation (FTA, idea invalidation, structural, POI block)
"""
import pytest
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

from src.smc.models import (
    BOS, CISD, FVG, Fractal, OrderFlow, SMCDayContext,
)
from src.smc.config import SMCConfig
from src.smc.engine import SMCEngine
from src.smc.timeframe_manager import TimeframeManager


# ---------------------
# Helpers
# ---------------------

@dataclass
class MockSignal:
    """Mock signal for validation tests."""
    signal_type: str = "Reverse"
    direction: str = "long"
    entry_price: float = 100.0
    stop_price: float = 98.0
    entry_time: Optional[Any] = None
    extra: Dict[str, Any] = field(default_factory=dict)


def _make_fvg(direction="bullish", high=101.0, low=99.0, status="active",
              tf="M2", instrument="GER40") -> FVG:
    """Create a test FVG."""
    return FVG(
        id=f"fvg_test_{direction}_{high}",
        instrument=instrument,
        timeframe=tf,
        direction=direction,
        high=high,
        low=low,
        midpoint=(high + low) / 2,
        formation_time=datetime(2024, 1, 2, 9, 0),
        candle1_time=datetime(2024, 1, 2, 8, 56),
        candle2_time=datetime(2024, 1, 2, 8, 58),
        candle3_time=datetime(2024, 1, 2, 9, 0),
        status=status,
    )


def _make_bos(direction="bullish", broken_level=100.0, bos_type="bos") -> BOS:
    """Create a test BOS."""
    return BOS(
        id=f"bos_test_{direction}_{broken_level}",
        instrument="GER40",
        timeframe="M5",
        direction=direction,
        broken_level=broken_level,
        break_time=datetime(2024, 1, 2, 10, 0),
        break_candle_close=broken_level + (1 if direction == "bullish" else -1),
        bos_type=bos_type,
    )


def _make_cisd(direction="long", confirmation_close=100.0) -> CISD:
    """Create a test CISD."""
    return CISD(
        id=f"cisd_test_{direction}_{confirmation_close}",
        instrument="GER40",
        timeframe="M2",
        direction=direction,
        delivery_candle_time=datetime(2024, 1, 2, 9, 56),
        delivery_candle_body_high=100.5,
        delivery_candle_body_low=99.5,
        confirmation_time=datetime(2024, 1, 2, 9, 58),
        confirmation_close=confirmation_close,
    )


def _make_bar(**kwargs) -> pd.Series:
    """Create a price bar as pd.Series."""
    defaults = {
        "time": datetime(2024, 1, 2, 10, 0),
        "open": 100.0,
        "high": 101.0,
        "low": 99.0,
        "close": 100.5,
    }
    defaults.update(kwargs)
    return pd.Series(defaults)


def _make_engine() -> SMCEngine:
    """Create engine with minimal M1 data for testing."""
    times = pd.date_range("2024-01-02 08:00", periods=60, freq="1min", tz="UTC")
    m1 = pd.DataFrame({
        "time": times,
        "open": [100.0] * 60,
        "high": [101.0] * 60,
        "low": [99.0] * 60,
        "close": [100.0] * 60,
    })
    config = SMCConfig(instrument="GER40")
    tfm = TimeframeManager(m1, "GER40")
    return SMCEngine(config=config, timeframe_manager=tfm)


def _empty_context(instrument="GER40") -> SMCDayContext:
    """Create empty SMC context."""
    return SMCDayContext(
        instrument=instrument,
        day_date=datetime(2024, 1, 2).date(),
    )


# ---------------------
# FVG Validation Tests
# ---------------------

class TestValidateFVGStatus:
    """Test FVG validation (4 of 6 variants)."""

    def test_already_invalidated_returns_current(self):
        """Already-invalidated FVG keeps its status."""
        engine = _make_engine()
        fvg = _make_fvg(status="full_fill")
        fvg.invalidation_reason = "test reason"
        bar = _make_bar()

        status, reason = engine.validate_fvg_status(fvg, bar)
        assert status == "full_fill"
        assert reason == "test reason"

    def test_structural_update_by_bos(self):
        """FVG invalidated when same-direction BOS occurs after formation."""
        engine = _make_engine()
        fvg = _make_fvg(direction="bullish")
        bos = _make_bos(direction="bullish")
        bos.break_time = fvg.formation_time + timedelta(hours=1)
        bar = _make_bar()

        status, reason = engine.validate_fvg_status(fvg, bar, recent_bos=[bos])
        assert status == "structural_update"
        assert "bos" in reason

    def test_no_structural_update_opposing_direction(self):
        """BOS in opposite direction does not invalidate FVG."""
        engine = _make_engine()
        fvg = _make_fvg(direction="bullish")
        bos = _make_bos(direction="bearish")
        bos.break_time = fvg.formation_time + timedelta(hours=1)
        bar = _make_bar(open=100.0, high=100.5, low=99.5, close=100.0)

        status, reason = engine.validate_fvg_status(fvg, bar, recent_bos=[bos])
        assert status == "active"

    def test_zone_to_zone_bullish_passthrough(self):
        """Bullish FVG invalidated by strong pass-through candle."""
        engine = _make_engine()
        fvg = _make_fvg(direction="bullish", high=101.0, low=99.0)  # 2pt gap
        # Strong bullish candle closing above FVG with body > 1.5 * gap
        bar = _make_bar(open=98.0, high=102.5, low=97.5, close=102.0)

        status, reason = engine.validate_fvg_status(fvg, bar)
        assert status == "zone_to_zone"
        assert "Pass-through" in reason

    def test_zone_to_zone_bearish_passthrough(self):
        """Bearish FVG invalidated by strong pass-through candle."""
        engine = _make_engine()
        fvg = _make_fvg(direction="bearish", high=101.0, low=99.0)
        # Strong bearish candle closing below FVG with body > 1.5 * gap
        bar = _make_bar(open=102.0, high=102.5, low=97.0, close=97.5)

        status, reason = engine.validate_fvg_status(fvg, bar)
        assert status == "zone_to_zone"

    def test_full_fill_bullish(self):
        """Bullish FVG fully filled when price drops through entire gap."""
        engine = _make_engine()
        fvg = _make_fvg(direction="bullish", high=101.0, low=99.0)
        # Bar low reaches below FVG low (full fill)
        bar = _make_bar(open=100.5, high=100.5, low=98.5, close=99.0)

        status, reason = engine.validate_fvg_status(fvg, bar)
        assert status == "full_fill"
        assert "filled" in reason.lower()

    def test_full_fill_bearish(self):
        """Bearish FVG fully filled when price rises through entire gap."""
        engine = _make_engine()
        fvg = _make_fvg(direction="bearish", high=101.0, low=99.0)
        # Bar high reaches above FVG high (full fill)
        bar = _make_bar(open=99.5, high=101.5, low=99.5, close=101.0)

        status, reason = engine.validate_fvg_status(fvg, bar)
        assert status == "full_fill"

    def test_partial_fill_stays_active(self):
        """Partially filled FVG stays active."""
        engine = _make_engine()
        fvg = _make_fvg(direction="bullish", high=101.0, low=99.0)
        # Bar only partially enters the gap
        bar = _make_bar(open=100.5, high=100.8, low=100.0, close=100.5)

        status, reason = engine.validate_fvg_status(fvg, bar)
        assert status == "active"
        assert fvg.fill_pct > 0

    def test_inversion_bullish_to_bearish(self):
        """Bullish FVG inverted by strong bearish close below low."""
        engine = _make_engine()
        fvg = _make_fvg(direction="bullish", high=101.0, low=99.0)
        # Strong bearish candle: open above FVG, close well below, body > 0.8 * gap
        bar = _make_bar(open=100.5, high=100.5, low=97.0, close=97.2)

        status, reason = engine.validate_fvg_status(fvg, bar)
        assert status == "inverted"
        assert "Inverted" in reason

    def test_inversion_bearish_to_bullish(self):
        """Bearish FVG inverted by strong bullish close above high."""
        engine = _make_engine()
        fvg = _make_fvg(direction="bearish", high=101.0, low=99.0)
        # Strong bullish candle: open below FVG, close well above
        bar = _make_bar(open=99.5, high=103.0, low=99.5, close=102.8)

        status, reason = engine.validate_fvg_status(fvg, bar)
        assert status == "inverted"

    def test_no_inversion_weak_candle(self):
        """Weak candle beyond FVG does not trigger inversion."""
        engine = _make_engine()
        fvg = _make_fvg(direction="bullish", high=101.0, low=99.0)
        # Small bearish candle below FVG (body < 0.8 * gap)
        bar = _make_bar(open=99.2, high=99.5, low=98.5, close=98.9)

        status, reason = engine.validate_fvg_status(fvg, bar)
        # Should not be inverted (body 0.3 < 1.6 = 0.8 * 2.0)
        assert status != "inverted"


class TestCalculateFVGFill:
    """Test FVG fill calculation."""

    def test_bullish_fill_from_above(self):
        engine = _make_engine()
        fvg = _make_fvg(direction="bullish", high=101.0, low=99.0)
        bar = _make_bar(low=100.0)  # 50% fill

        fill = engine._calculate_fvg_fill(fvg, bar)
        assert abs(fill - 0.5) < 0.01

    def test_bearish_fill_from_below(self):
        engine = _make_engine()
        fvg = _make_fvg(direction="bearish", high=101.0, low=99.0)
        bar = _make_bar(high=100.0)  # 50% fill

        fill = engine._calculate_fvg_fill(fvg, bar)
        assert abs(fill - 0.5) < 0.01

    def test_no_fill_when_bar_outside(self):
        engine = _make_engine()
        fvg = _make_fvg(direction="bullish", high=101.0, low=99.0)
        bar = _make_bar(low=102.0, high=103.0)  # bar entirely above FVG

        fill = engine._calculate_fvg_fill(fvg, bar)
        assert fill == 0.0

    def test_full_fill_capped_at_1(self):
        engine = _make_engine()
        fvg = _make_fvg(direction="bullish", high=101.0, low=99.0)
        bar = _make_bar(low=97.0)  # way below, still 1.0

        fill = engine._calculate_fvg_fill(fvg, bar)
        assert fill == 1.0


# ---------------------
# BOS Validation Tests
# ---------------------

class TestValidateBOS:
    """Test BOS validation."""

    def test_valid_bullish_bos_no_opposing(self):
        """Bullish BOS valid when no opposing zones below."""
        engine = _make_engine()
        bos = _make_bos(direction="bullish", broken_level=100.0)
        ctx = _empty_context()

        valid, reason = engine.validate_bos(bos, ctx)
        assert valid is True

    def test_invalid_bullish_bos_bearish_fvg_below(self):
        """Bullish BOS invalid when bearish FVG below."""
        engine = _make_engine()
        bos = _make_bos(direction="bullish", broken_level=100.0)
        ctx = _empty_context()
        ctx.fvgs_h1 = [_make_fvg(direction="bearish", high=99.5, low=98.5)]

        valid, reason = engine.validate_bos(bos, ctx)
        assert valid is False
        assert "Bearish FVG" in reason

    def test_invalid_bullish_bos_low_fractal_below(self):
        """Bullish BOS invalid when unswept low fractal nearby below."""
        engine = _make_engine()
        bos = _make_bos(direction="bullish", broken_level=100.0)
        ctx = _empty_context()
        ctx.active_fractal_lows = [99.0]

        valid, reason = engine.validate_bos(bos, ctx)
        assert valid is False
        assert "fractal" in reason.lower()

    def test_valid_bearish_bos_no_opposing(self):
        """Bearish BOS valid when no opposing zones above."""
        engine = _make_engine()
        bos = _make_bos(direction="bearish", broken_level=100.0)
        ctx = _empty_context()

        valid, reason = engine.validate_bos(bos, ctx)
        assert valid is True

    def test_invalid_bearish_bos_bullish_fvg_above(self):
        """Bearish BOS invalid when bullish FVG above."""
        engine = _make_engine()
        bos = _make_bos(direction="bearish", broken_level=100.0)
        ctx = _empty_context()
        ctx.fvgs_m2 = [_make_fvg(direction="bullish", high=101.5, low=100.5)]

        valid, reason = engine.validate_bos(bos, ctx)
        assert valid is False
        assert "Bullish FVG" in reason

    def test_far_away_fvg_does_not_invalidate(self):
        """FVG far from BOS level does not invalidate."""
        engine = _make_engine()
        bos = _make_bos(direction="bullish", broken_level=100.0)
        ctx = _empty_context()
        # FVG at 90.0 (10% away, > 2% threshold)
        ctx.fvgs_h1 = [_make_fvg(direction="bearish", high=91.0, low=89.0)]

        valid, reason = engine.validate_bos(bos, ctx)
        assert valid is True


# ---------------------
# CISD Validation Tests
# ---------------------

class TestValidateCISD:
    """Test CISD validation."""

    def test_valid_long_cisd_no_opposing(self):
        engine = _make_engine()
        cisd = _make_cisd(direction="long", confirmation_close=100.0)
        ctx = _empty_context()

        valid, reason = engine.validate_cisd(cisd, ctx)
        assert valid is True

    def test_invalid_long_cisd_bearish_fvg_below(self):
        engine = _make_engine()
        cisd = _make_cisd(direction="long", confirmation_close=100.0)
        ctx = _empty_context()
        ctx.fvgs_h1 = [_make_fvg(direction="bearish", high=99.5, low=98.5)]

        valid, reason = engine.validate_cisd(cisd, ctx)
        assert valid is False
        assert "Bearish FVG" in reason

    def test_invalid_short_cisd_bullish_fvg_above(self):
        engine = _make_engine()
        cisd = _make_cisd(direction="short", confirmation_close=100.0)
        ctx = _empty_context()
        ctx.fvgs_m2 = [_make_fvg(direction="bullish", high=101.5, low=100.5)]

        valid, reason = engine.validate_cisd(cisd, ctx)
        assert valid is False
        assert "Bullish FVG" in reason

    def test_valid_short_cisd_no_opposing(self):
        engine = _make_engine()
        cisd = _make_cisd(direction="short", confirmation_close=100.0)
        ctx = _empty_context()

        valid, reason = engine.validate_cisd(cisd, ctx)
        assert valid is True


# ---------------------
# OrderFlow Validation Tests
# ---------------------

class TestValidateOrderFlow:
    """Test OrderFlow validation."""

    def test_pending_with_step1_becomes_confirmed(self):
        engine = _make_engine()
        of = OrderFlow(
            id="of_1", instrument="GER40", timeframe="M5",
            direction="long", trigger_price=100.0,
            trigger_time=datetime(2024, 1, 2, 9, 0),
            target_price=105.0,
            step1_price=101.0, step1_time=datetime(2024, 1, 2, 9, 30),
            step1_type="zone_test",
        )
        bar = _make_bar(close=102.0)

        status, reason = engine.validate_order_flow(of, bar)
        assert status == "step1_confirmed"

    def test_step1_confirmed_with_step2_becomes_validated(self):
        engine = _make_engine()
        of = OrderFlow(
            id="of_1", instrument="GER40", timeframe="M5",
            direction="long", trigger_price=100.0,
            trigger_time=datetime(2024, 1, 2, 9, 0),
            target_price=105.0,
            step1_price=101.0, step1_time=datetime(2024, 1, 2, 9, 30),
            step2_price=103.0, step2_time=datetime(2024, 1, 2, 10, 0),
            status="step1_confirmed",
        )
        bar = _make_bar(close=104.0)

        status, reason = engine.validate_order_flow(of, bar)
        assert status == "validated"
        assert "Two steps" in reason

    def test_long_invalidated_by_close_below_step1(self):
        engine = _make_engine()
        of = OrderFlow(
            id="of_1", instrument="GER40", timeframe="M5",
            direction="long", trigger_price=100.0,
            trigger_time=datetime(2024, 1, 2, 9, 0),
            target_price=105.0,
            step1_price=101.0,
        )
        bar = _make_bar(close=99.0)  # below step1

        status, reason = engine.validate_order_flow(of, bar)
        assert status == "invalidated"
        assert "step1" in reason

    def test_short_invalidated_by_close_above_step2(self):
        engine = _make_engine()
        of = OrderFlow(
            id="of_1", instrument="GER40", timeframe="M5",
            direction="short", trigger_price=100.0,
            trigger_time=datetime(2024, 1, 2, 9, 0),
            target_price=95.0,
            step1_price=99.0,
            step2_price=97.0,
            status="step1_confirmed",
        )
        bar = _make_bar(close=98.0)  # above step2

        status, reason = engine.validate_order_flow(of, bar)
        assert status == "invalidated"
        assert "step2" in reason

    def test_already_invalidated_returns_same(self):
        engine = _make_engine()
        of = OrderFlow(
            id="of_1", instrument="GER40", timeframe="M5",
            direction="long", trigger_price=100.0,
            trigger_time=datetime(2024, 1, 2, 9, 0),
            target_price=105.0,
            status="invalidated",
            invalidation_reason="test",
        )
        bar = _make_bar()

        status, reason = engine.validate_order_flow(of, bar)
        assert status == "invalidated"
        assert reason == "test"

    def test_pending_no_steps_stays_pending(self):
        engine = _make_engine()
        of = OrderFlow(
            id="of_1", instrument="GER40", timeframe="M5",
            direction="long", trigger_price=100.0,
            trigger_time=datetime(2024, 1, 2, 9, 0),
            target_price=105.0,
        )
        bar = _make_bar()

        status, reason = engine.validate_order_flow(of, bar)
        assert status == "pending"


# ---------------------
# Entry Validation Tests
# ---------------------

class TestValidateEntry:
    """Test entry validation (Model 1 and Model 2)."""

    def test_continuation_passes_with_confirmations(self):
        """OCAE signal passes Model 1 with 2+ confirmations."""
        engine = _make_engine()
        signal = MockSignal(signal_type="OCAE", direction="long", entry_price=100.0)
        ctx = _empty_context()
        # Add supporting FVG near entry
        ctx.fvgs_m2 = [_make_fvg(direction="bullish", high=100.5, low=99.5)]
        # Add CISD confirmation
        ctx.cisd_events = [_make_cisd(direction="long", confirmation_close=100.0)]

        valid, reason = engine.validate_entry(signal, ctx)
        assert valid is True

    def test_continuation_fails_with_opposing_zone(self):
        """OCAE signal fails Model 1 when opposing zone exists."""
        engine = _make_engine()
        signal = MockSignal(signal_type="OCAE", direction="long", entry_price=100.0)
        ctx = _empty_context()
        # Bearish FVG just above entry (within 2%)
        ctx.fvgs_h1 = [_make_fvg(direction="bearish", high=102.0, low=100.5)]

        valid, reason = engine.validate_entry(signal, ctx)
        assert valid is False
        assert "Opposing" in reason

    def test_continuation_fails_insufficient_confirmations(self):
        """OCAE signal fails Model 1 with <2 confirmations."""
        engine = _make_engine()
        signal = MockSignal(signal_type="OCAE", direction="long", entry_price=100.0)
        ctx = _empty_context()
        # Only 1 confirmation (FVG but no CISD/BOS)
        ctx.fvgs_m2 = [_make_fvg(direction="bullish", high=100.5, low=99.5)]

        valid, reason = engine.validate_entry(signal, ctx)
        assert valid is False
        assert "confirmations" in reason.lower()

    def test_reversal_passes_with_one_confirmation(self):
        """Reverse signal passes Model 2 with 1 confirmation."""
        engine = _make_engine()
        signal = MockSignal(signal_type="Reverse", direction="long", entry_price=100.0)
        ctx = _empty_context()
        ctx.fvgs_m2 = [_make_fvg(direction="bullish", high=100.5, low=99.5)]

        valid, reason = engine.validate_entry(signal, ctx)
        assert valid is True
        assert "Model 2" in reason

    def test_reversal_fails_no_confirmations(self):
        """Reverse signal fails with zero confirmations."""
        engine = _make_engine()
        signal = MockSignal(signal_type="Reverse", direction="long", entry_price=100.0)
        ctx = _empty_context()

        valid, reason = engine.validate_entry(signal, ctx)
        assert valid is False
        assert "No confirmations" in reason

    def test_reversal_fails_too_many_opposing(self):
        """Reverse signal fails when >2 opposing zones."""
        engine = _make_engine()
        signal = MockSignal(signal_type="Reverse", direction="long", entry_price=100.0)
        ctx = _empty_context()
        # 1 confirmation
        ctx.fvgs_m2 = [_make_fvg(direction="bullish", high=100.5, low=99.5)]
        # 3 opposing zones (high fractals below entry within 2%)
        ctx.active_fractal_highs = [99.0, 98.5, 98.2]

        valid, reason = engine.validate_entry(signal, ctx)
        assert valid is False
        assert "opposing" in reason.lower()


class TestCountConfirmations:
    """Test confirmation counting."""

    def test_counts_supporting_fvg(self):
        engine = _make_engine()
        signal = MockSignal(direction="long", entry_price=100.0)
        ctx = _empty_context()
        ctx.fvgs_m2 = [_make_fvg(direction="bullish", high=100.5, low=99.5)]

        count = engine._count_confirmations(signal, ctx)
        assert count >= 1

    def test_counts_cisd(self):
        engine = _make_engine()
        signal = MockSignal(direction="long", entry_price=100.0)
        ctx = _empty_context()
        ctx.cisd_events = [_make_cisd(direction="long", confirmation_close=100.0)]

        count = engine._count_confirmations(signal, ctx)
        assert count >= 1

    def test_counts_bos(self):
        engine = _make_engine()
        signal = MockSignal(direction="long", entry_price=100.0)
        ctx = _empty_context()
        ctx.bos_events = [_make_bos(direction="bullish", broken_level=100.0)]

        count = engine._count_confirmations(signal, ctx)
        assert count >= 1

    def test_zero_with_empty_context(self):
        engine = _make_engine()
        signal = MockSignal(direction="long", entry_price=100.0)
        ctx = _empty_context()

        count = engine._count_confirmations(signal, ctx)
        assert count == 0


# ---------------------
# TP Validation Tests
# ---------------------

class TestValidateTPTarget:
    """Test TP target validation."""

    def test_valid_tp_at_bearish_fvg_for_long(self):
        """Long TP valid at bearish FVG (resistance)."""
        engine = _make_engine()
        ctx = _empty_context()
        ctx.fvgs_h1 = [_make_fvg(direction="bearish", high=106.0, low=104.0)]

        valid = engine.validate_tp_target(105.0, "long", ctx)
        assert valid is True

    def test_valid_tp_at_bullish_fvg_for_short(self):
        """Short TP valid at bullish FVG (support)."""
        engine = _make_engine()
        ctx = _empty_context()
        ctx.fvgs_m2 = [_make_fvg(direction="bullish", high=96.0, low=94.0)]

        valid = engine.validate_tp_target(95.0, "short", ctx)
        assert valid is True

    def test_valid_tp_at_fractal_high(self):
        """Long TP valid at unswept high fractal."""
        engine = _make_engine()
        ctx = _empty_context()
        ctx.active_fractal_highs = [105.0]

        valid = engine.validate_tp_target(105.0, "long", ctx)
        assert valid is True

    def test_invalid_tp_no_zone(self):
        """TP invalid when no zone at target."""
        engine = _make_engine()
        ctx = _empty_context()

        valid = engine.validate_tp_target(110.0, "long", ctx)
        assert valid is False


class TestFindNearestValidTP:
    """Test finding nearest valid TP target."""

    def test_finds_nearest_for_long(self):
        engine = _make_engine()
        ctx = _empty_context()
        ctx.fvgs_h1 = [
            _make_fvg(direction="bearish", high=108.0, low=106.0),
            _make_fvg(direction="bearish", high=113.0, low=111.0),
        ]
        ctx.active_fractal_highs = [105.0, 115.0]

        tp = engine.find_nearest_valid_tp(100.0, "long", ctx)
        assert tp == 105.0  # Nearest fractal high above entry

    def test_finds_nearest_for_short(self):
        engine = _make_engine()
        ctx = _empty_context()
        ctx.active_fractal_lows = [95.0, 90.0]

        tp = engine.find_nearest_valid_tp(100.0, "short", ctx)
        assert tp == 95.0

    def test_returns_none_when_no_targets(self):
        engine = _make_engine()
        ctx = _empty_context()

        tp = engine.find_nearest_valid_tp(100.0, "long", ctx)
        assert tp is None

    def test_ignores_zones_behind_entry(self):
        """Only zones beyond entry count as TP targets."""
        engine = _make_engine()
        ctx = _empty_context()
        ctx.active_fractal_highs = [95.0, 98.0]  # All below entry

        tp = engine.find_nearest_valid_tp(100.0, "long", ctx)
        assert tp is None


# ---------------------
# SL Validation Tests
# ---------------------

class TestValidateSLPlacement:
    """Test SL placement validation."""

    def test_sl_behind_poi_is_invalid_long(self):
        """Long SL placed behind bullish POI (FVG) is invalid."""
        engine = _make_engine()
        ctx = _empty_context()
        # Bullish FVG between SL and entry
        ctx.fvgs_m2 = [_make_fvg(direction="bullish", high=99.5, low=98.5)]

        valid, warning = engine.validate_sl_placement(97.0, 100.0, "long", ctx)
        assert valid is False
        assert "POI" in warning

    def test_sl_behind_poi_is_invalid_short(self):
        """Short SL placed behind bearish POI (FVG) is invalid."""
        engine = _make_engine()
        ctx = _empty_context()
        # Bearish FVG between entry and SL
        ctx.fvgs_m2 = [_make_fvg(direction="bearish", high=101.5, low=100.5)]

        valid, warning = engine.validate_sl_placement(103.0, 100.0, "short", ctx)
        assert valid is False
        assert "POI" in warning

    def test_sl_behind_cisd_poi_is_invalid(self):
        """SL behind CISD continuation point is invalid."""
        engine = _make_engine()
        ctx = _empty_context()
        ctx.cisd_events = [_make_cisd(direction="long", confirmation_close=99.0)]

        valid, warning = engine.validate_sl_placement(97.0, 100.0, "long", ctx)
        assert valid is False
        assert "CISD" in warning

    def test_sl_correctly_behind_invalidation(self):
        """Long SL below bearish FVG invalidation zone is valid."""
        engine = _make_engine()
        ctx = _empty_context()
        # Bearish FVG below entry (invalidation zone)
        ctx.fvgs_m2 = [_make_fvg(direction="bearish", high=99.0, low=97.5)]

        valid, warning = engine.validate_sl_placement(96.0, 100.0, "long", ctx)
        assert valid is True
        assert warning is None

    def test_sl_below_fractal_is_valid(self):
        """SL correctly below low fractal."""
        engine = _make_engine()
        ctx = _empty_context()
        ctx.active_fractal_lows = [98.0]

        valid, warning = engine.validate_sl_placement(97.0, 100.0, "long", ctx)
        assert valid is True

    def test_unclear_sl_returns_warning(self):
        """SL with no clear invalidation zone returns advisory warning."""
        engine = _make_engine()
        ctx = _empty_context()

        valid, warning = engine.validate_sl_placement(97.0, 100.0, "long", ctx)
        assert valid is True
        assert "unclear" in warning.lower()


# ---------------------
# BE Validation Tests
# ---------------------

class TestValidateBEMove:
    """Test breakeven move validation."""

    def test_blocked_when_entry_in_poi(self):
        """BE blocked when entry price is inside active FVG (POI)."""
        engine = _make_engine()
        ctx = _empty_context()
        ctx.fvgs_m2 = [_make_fvg(direction="bullish", high=101.0, low=99.0)]
        bar = _make_bar()

        should_be, reason = engine.validate_be_move(
            bar, datetime(2024, 1, 2, 9, 0), 100.0, "long", ctx
        )
        assert should_be is False
        assert "POI" in reason

    def test_fta_reaction_triggers_be(self):
        """Strong FTA reaction (large wick) triggers BE."""
        engine = _make_engine()
        ctx = _empty_context()
        # Large upper wick, small body -> wick/body > 2.0
        bar = _make_bar(open=100.0, high=103.0, low=99.8, close=100.2)

        should_be, reason = engine.validate_be_move(
            bar, datetime(2024, 1, 2, 9, 0), 98.0, "long", ctx
        )
        assert should_be is True
        assert "FTA" in reason

    def test_idea_invalidation_long(self):
        """Long trade: testing opposing bearish FVG triggers BE."""
        engine = _make_engine()
        ctx = _empty_context()
        ctx.fvgs_h1 = [_make_fvg(direction="bearish", high=105.0, low=103.0)]
        bar = _make_bar(open=102.0, high=104.0, low=101.5, close=103.5)

        should_be, reason = engine.validate_be_move(
            bar, datetime(2024, 1, 2, 9, 0), 98.0, "long", ctx
        )
        assert should_be is True
        assert "invalidation" in reason.lower()

    def test_idea_invalidation_short(self):
        """Short trade: testing opposing bullish FVG triggers BE."""
        engine = _make_engine()
        ctx = _empty_context()
        ctx.fvgs_m2 = [_make_fvg(direction="bullish", high=97.0, low=95.0)]
        bar = _make_bar(open=98.0, high=98.5, low=96.0, close=96.5)

        should_be, reason = engine.validate_be_move(
            bar, datetime(2024, 1, 2, 9, 0), 102.0, "short", ctx
        )
        assert should_be is True
        assert "invalidation" in reason.lower()

    def test_structural_be_on_bos(self):
        """BOS after entry triggers structural BE."""
        engine = _make_engine()
        ctx = _empty_context()
        bos = _make_bos(direction="bearish", broken_level=99.0)
        bos.break_time = datetime(2024, 1, 2, 10, 30)  # after entry
        ctx.bos_events = [bos]
        bar = _make_bar()

        should_be, reason = engine.validate_be_move(
            bar, datetime(2024, 1, 2, 9, 0), 98.0, "long", ctx
        )
        assert should_be is True
        assert "structure" in reason.lower()

    def test_no_be_conditions(self):
        """No conditions met: do not move to BE."""
        engine = _make_engine()
        ctx = _empty_context()
        # Small body, small wicks, wick/body < 2.0
        bar = _make_bar(open=100.0, high=100.8, low=99.5, close=100.5)

        should_be, reason = engine.validate_be_move(
            bar, datetime(2024, 1, 2, 9, 0), 98.0, "long", ctx
        )
        assert should_be is False
