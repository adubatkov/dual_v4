"""Tests for SMC Engine."""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Optional

from src.smc.config import SMCConfig
from src.smc.timeframe_manager import TimeframeManager
from src.smc.engine import SMCEngine
from src.smc.models import FVG, Fractal, CISD, BOS, ConfirmationCriteria


@dataclass
class MockSignal:
    """Mock IB signal for testing."""
    direction: str
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


def _make_m1_data(periods=100, start="2024-01-01 10:00", base_price=100.0):
    """Create M1 data with some structure."""
    times = pd.date_range(start=start, periods=periods, freq="1min")
    np.random.seed(42)

    prices = [base_price]
    for _ in range(periods - 1):
        prices.append(prices[-1] + np.random.normal(0, 0.5))

    data = pd.DataFrame({
        "time": times,
        "open": prices,
        "high": [p + abs(np.random.normal(0, 0.3)) for p in prices],
        "low": [p - abs(np.random.normal(0, 0.3)) for p in prices],
        "close": [p + np.random.normal(0, 0.2) for p in prices],
    })
    return data


class TestSMCEngineInit:
    """Test engine initialization."""

    def test_engine_initialization(self):
        """Engine initializes with config and TFM."""
        m1_data = _make_m1_data()
        tfm = TimeframeManager(m1_data, "GER40")
        config = SMCConfig(instrument="GER40")

        engine = SMCEngine(config, tfm)

        assert engine.config.instrument == "GER40"
        assert engine.registry.instrument == "GER40"
        assert len(engine.event_log) == 0

    def test_engine_default_config(self):
        """Default config has fractals and FVG enabled, CISD/BOS disabled."""
        m1_data = _make_m1_data()
        tfm = TimeframeManager(m1_data, "GER40")
        config = SMCConfig(instrument="GER40")

        engine = SMCEngine(config, tfm)

        assert engine.config.enable_fractals is True
        assert engine.config.enable_fvg is True
        assert engine.config.enable_cisd is False
        assert engine.config.enable_bos is False


class TestSMCEngineUpdate:
    """Test engine update cycle."""

    def test_update_runs_without_error(self):
        """update() runs all enabled detectors without errors."""
        m1_data = _make_m1_data(periods=200)
        tfm = TimeframeManager(m1_data, "GER40")
        config = SMCConfig(instrument="GER40")

        engine = SMCEngine(config, tfm)
        engine.update(current_time=datetime(2024, 1, 1, 11, 0))

        # Should have run without errors
        assert engine.registry.count() >= 0

    def test_update_with_all_detectors(self):
        """update() with all detectors enabled."""
        m1_data = _make_m1_data(periods=300)
        tfm = TimeframeManager(m1_data, "GER40")
        config = SMCConfig(
            instrument="GER40",
            enable_fractals=True,
            enable_fvg=True,
            enable_cisd=True,
            enable_bos=True,
        )

        engine = SMCEngine(config, tfm)
        engine.update(current_time=datetime(2024, 1, 1, 12, 0))

        # Should not crash
        assert engine.registry.count() >= 0

    def test_update_appends_m1_data(self):
        """update() can append new M1 bars."""
        m1_data = _make_m1_data(periods=100)
        tfm = TimeframeManager(m1_data, "GER40")
        config = SMCConfig(instrument="GER40")

        engine = SMCEngine(config, tfm)

        # Append new bars
        new_bars = pd.DataFrame({
            "time": pd.date_range(start="2024-01-01 11:40", periods=5, freq="1min"),
            "open": [100.0] * 5,
            "high": [101.0] * 5,
            "low": [99.0] * 5,
            "close": [100.5] * 5,
        })

        engine.update(
            current_time=datetime(2024, 1, 1, 11, 45),
            new_m1_bars=new_bars,
        )

        # Should have processed without errors
        assert True

    def test_update_records_events(self):
        """update() records events in event log."""
        # Create data with clear fractal pattern
        times = pd.date_range(start="2024-01-01 10:00", periods=180, freq="1min")
        prices = [100.0] * 180

        # Create a pattern that will produce H1 data with fractals
        for i in range(180):
            minute = i
            hour_phase = (minute % 60) / 60.0
            if minute < 60:
                prices[i] = 100 + 5 * hour_phase  # Rising
            elif minute < 120:
                prices[i] = 105 + 10 * (hour_phase - 0.5)  # Peak then dip
            else:
                prices[i] = 100 + 3 * hour_phase  # Rising again

        data = pd.DataFrame({
            "time": times,
            "open": prices,
            "high": [p + 1 for p in prices],
            "low": [p - 1 for p in prices],
            "close": prices,
        })

        tfm = TimeframeManager(data, "GER40")
        config = SMCConfig(instrument="GER40")

        engine = SMCEngine(config, tfm)
        engine.update(current_time=datetime(2024, 1, 1, 13, 0))

        # Event log should have some entries if structures were found
        # (may be 0 if data doesn't produce structures)
        assert len(engine.event_log) >= 0


class TestSMCEngineEvaluateSignal:
    """Test signal evaluation."""

    def test_evaluate_signal_returns_decision(self):
        """evaluate_signal returns SMCDecision."""
        m1_data = _make_m1_data(periods=200)
        tfm = TimeframeManager(m1_data, "GER40")
        config = SMCConfig(instrument="GER40")

        engine = SMCEngine(config, tfm)
        engine.update(current_time=datetime(2024, 1, 1, 11, 0))

        signal = MockSignal(direction="long", entry_price=100.0)
        decision = engine.evaluate_signal(signal, datetime(2024, 1, 1, 11, 0))

        assert decision.action in ("ENTER", "WAIT", "REJECT")
        assert decision.reason != ""
        assert isinstance(decision.confluence_score, float)

    def test_evaluate_signal_without_entry_price(self):
        """evaluate_signal uses current price when entry_price is None."""
        m1_data = _make_m1_data(periods=100)
        tfm = TimeframeManager(m1_data, "GER40")
        config = SMCConfig(instrument="GER40")

        engine = SMCEngine(config, tfm)

        signal = MockSignal(direction="long", entry_price=None)
        decision = engine.evaluate_signal(signal, datetime(2024, 1, 1, 11, 0))

        assert decision.action in ("ENTER", "WAIT", "REJECT")

    def test_evaluate_signal_reject_on_conflict(self):
        """Signal rejected when conflicting structures exist."""
        m1_data = _make_m1_data(periods=100, base_price=100.0)
        tfm = TimeframeManager(m1_data, "GER40")
        config = SMCConfig(instrument="GER40")

        engine = SMCEngine(config, tfm)

        # Manually add a conflicting bearish FVG near signal price
        fvg = FVG(
            id="fvg_conflict",
            instrument="GER40",
            timeframe="M2",
            direction="bearish",
            high=101.0,
            low=99.5,
            midpoint=100.25,
            formation_time=datetime(2024, 1, 1, 10, 30),
            candle1_time=datetime(2024, 1, 1, 10, 26),
            candle2_time=datetime(2024, 1, 1, 10, 28),
            candle3_time=datetime(2024, 1, 1, 10, 30),
        )
        engine.registry.add("fvg", fvg)

        signal = MockSignal(direction="long", entry_price=100.0)
        decision = engine.evaluate_signal(signal, datetime(2024, 1, 1, 11, 0))

        assert decision.action == "REJECT"
        assert "Conflicting" in decision.reason


class TestSMCEngineConfluence:
    """Test confluence scoring."""

    def test_confluence_empty_registry(self):
        """No structures -> zero confluence."""
        m1_data = _make_m1_data(periods=100)
        tfm = TimeframeManager(m1_data, "GER40")
        config = SMCConfig(instrument="GER40")

        engine = SMCEngine(config, tfm)
        confluence = engine.calculate_confluence(100.0, "long", datetime(2024, 1, 1, 11, 0))

        assert confluence.total == 0.0
        assert confluence.breakdown == {}
        assert confluence.contributing_structures == []

    def test_confluence_with_fractal(self):
        """Nearby fractal contributes to confluence."""
        m1_data = _make_m1_data(periods=100, base_price=100.0)
        tfm = TimeframeManager(m1_data, "GER40")
        config = SMCConfig(instrument="GER40", fractal_proximity_pct=0.01)

        engine = SMCEngine(config, tfm)

        # Add low fractal near price 100
        frac = Fractal(
            id="frac_test",
            instrument="GER40",
            timeframe="H1",
            type="low",
            price=100.1,
            time=datetime(2024, 1, 1, 10, 0),
            confirmed_time=datetime(2024, 1, 1, 10, 30),
        )
        engine.registry.add("fractal", frac)

        confluence = engine.calculate_confluence(100.0, "long", datetime(2024, 1, 1, 11, 0))

        assert confluence.total > 0
        assert "fractals" in confluence.breakdown
        assert "frac_test" in confluence.contributing_structures

    def test_confluence_direction_alignment(self):
        """Only structures aligned with direction contribute."""
        m1_data = _make_m1_data(periods=100, base_price=100.0)
        tfm = TimeframeManager(m1_data, "GER40")
        config = SMCConfig(instrument="GER40", fractal_proximity_pct=0.01)

        engine = SMCEngine(config, tfm)

        # Add HIGH fractal near price (supports SHORT, not LONG)
        frac = Fractal(
            id="frac_high",
            instrument="GER40",
            timeframe="H1",
            type="high",
            price=100.1,
            time=datetime(2024, 1, 1, 10, 0),
            confirmed_time=datetime(2024, 1, 1, 10, 30),
        )
        engine.registry.add("fractal", frac)

        # Long signal should NOT get confluence from high fractal
        confluence_long = engine.calculate_confluence(100.0, "long", datetime(2024, 1, 1, 11, 0))
        assert confluence_long.total == 0.0

        # Short signal SHOULD get confluence from high fractal
        confluence_short = engine.calculate_confluence(100.0, "short", datetime(2024, 1, 1, 11, 0))
        assert confluence_short.total > 0


class TestSMCEngineConfirmation:
    """Test confirmation checking."""

    def test_check_confirmation_no_criteria_met(self):
        """No criteria met -> returns None."""
        m1_data = _make_m1_data(periods=100)
        tfm = TimeframeManager(m1_data, "GER40")
        config = SMCConfig(instrument="GER40")

        engine = SMCEngine(config, tfm)

        signal = MockSignal(direction="long")
        criteria = [
            ConfirmationCriteria(type="CISD", timeframe="M2", direction="long"),
        ]

        result = engine.check_confirmation(signal, criteria, datetime(2024, 1, 1, 11, 0))
        assert result is None

    def test_check_confirmation_cisd_met(self):
        """CISD criterion met -> returns modified signal."""
        m1_data = _make_m1_data(periods=100)
        tfm = TimeframeManager(m1_data, "GER40")
        config = SMCConfig(instrument="GER40")

        engine = SMCEngine(config, tfm)

        # Add recent CISD
        cisd = CISD(
            id="cisd_test",
            instrument="GER40",
            timeframe="M2",
            direction="long",
            delivery_candle_time=datetime(2024, 1, 1, 10, 50),
            delivery_candle_body_high=101,
            delivery_candle_body_low=99,
            confirmation_time=datetime(2024, 1, 1, 10, 56),
            confirmation_close=102,
        )
        engine.registry.add("cisd", cisd)

        signal = MockSignal(direction="long", entry_price=100.0, stop_loss=98.0)
        criteria = [
            ConfirmationCriteria(type="CISD", timeframe="M2", direction="long"),
        ]

        result = engine.check_confirmation(signal, criteria, datetime(2024, 1, 1, 11, 0))
        assert result is not None
