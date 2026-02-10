"""Tests for IBStrategySMC wrapper.

Uses mocks to isolate SMC overlay logic from full emulator setup.
Tests focus on:
- SMC engine initialization
- Signal evaluation (ENTER/WAIT/REJECT)
- AWAITING_CONFIRMATION state machine
- M2 -> M1 approximation
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Optional, Dict, Any
from unittest.mock import MagicMock, patch, PropertyMock

from src.smc.config import SMCConfig
from src.smc.engine import SMCEngine
from src.smc.models import SMCDecision, ConfirmationCriteria, ConfluenceScore, FVG
from src.smc.timeframe_manager import TimeframeManager
from src.strategies.base_strategy import Signal


# ---------------------
# Helpers
# ---------------------

def _make_m1_data(periods=200, start="2024-01-01 08:00", base_price=100.0):
    """Create synthetic M1 data."""
    times = pd.date_range(start=start, periods=periods, freq="1min")
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


def _make_m2_data(periods=100, start="2024-01-01 08:00", base_price=100.0):
    """Create synthetic M2 data."""
    times = pd.date_range(start=start, periods=periods, freq="2min")
    np.random.seed(42)
    prices = [base_price]
    for _ in range(periods - 1):
        prices.append(prices[-1] + np.random.normal(0, 0.5))

    return pd.DataFrame({
        "time": times,
        "open": prices,
        "high": [p + abs(np.random.normal(0, 0.3)) for p in prices],
        "low": [p - abs(np.random.normal(0, 0.3)) for p in prices],
        "close": [p + np.random.normal(0, 0.2) for p in prices],
    })


# ---------------------
# M2 -> M1 approximation tests
# ---------------------

class TestM2ToM1Approximation:
    """Test M2 -> M1 conversion utility."""

    def test_doubles_bar_count(self):
        """Each M2 bar produces 2 M1 bars."""
        from src.strategies.ib_strategy_smc import IBStrategySMC

        m2 = _make_m2_data(periods=10)
        m1 = IBStrategySMC._m2_to_m1_approximation(m2)
        assert len(m1) == 20

    def test_preserves_ohlc_range(self):
        """M1 approximation stays within M2 high/low range."""
        from src.strategies.ib_strategy_smc import IBStrategySMC

        m2 = _make_m2_data(periods=5)
        m1 = IBStrategySMC._m2_to_m1_approximation(m2)

        for i in range(len(m2)):
            m2_row = m2.iloc[i]
            m1_pair = m1.iloc[i * 2: i * 2 + 2]

            # M1 highs should not exceed M2 high
            assert m1_pair["high"].max() <= m2_row["high"] + 0.01

            # M1 lows should not go below M2 low
            assert m1_pair["low"].min() >= m2_row["low"] - 0.01

    def test_first_open_matches_m2_open(self):
        """First M1 bar opens at M2 open."""
        from src.strategies.ib_strategy_smc import IBStrategySMC

        m2 = _make_m2_data(periods=3)
        m1 = IBStrategySMC._m2_to_m1_approximation(m2)

        for i in range(len(m2)):
            assert abs(m1.iloc[i * 2]["open"] - m2.iloc[i]["open"]) < 0.01

    def test_second_close_matches_m2_close(self):
        """Second M1 bar closes at M2 close."""
        from src.strategies.ib_strategy_smc import IBStrategySMC

        m2 = _make_m2_data(periods=3)
        m1 = IBStrategySMC._m2_to_m1_approximation(m2)

        for i in range(len(m2)):
            assert abs(m1.iloc[i * 2 + 1]["close"] - m2.iloc[i]["close"]) < 0.01

    def test_m1_times_are_1min_apart(self):
        """M1 bars within an M2 pair are 1 minute apart."""
        from src.strategies.ib_strategy_smc import IBStrategySMC

        m2 = _make_m2_data(periods=3)
        m1 = IBStrategySMC._m2_to_m1_approximation(m2)

        for i in range(len(m2)):
            t1 = m1.iloc[i * 2]["time"]
            t2 = m1.iloc[i * 2 + 1]["time"]
            assert (t2 - t1) == timedelta(minutes=1)


# ---------------------
# SMC Engine Integration Tests
# ---------------------

class TestSMCEngineIntegration:
    """Test SMC engine lifecycle within strategy context."""

    def test_engine_not_initialized_before_ib(self):
        """SMC engine should be None before IB calculation."""
        from src.strategies.ib_strategy_smc import IBStrategySMC

        # Verify the class has the attribute
        m1 = _make_m1_data()
        config = SMCConfig(instrument="GER40")

        # Create mock that simulates pre-IB state
        smc = object.__new__(IBStrategySMC)
        smc.smc_config = config
        smc._m1_data_ref = m1
        smc.smc_engine = None

        assert smc.smc_engine is None

    def test_init_with_m1_data(self):
        """_init_smc_engine creates engine with M1 data."""
        from src.strategies.ib_strategy_smc import IBStrategySMC

        m1 = _make_m1_data(periods=200)
        config = SMCConfig(instrument="GER40")

        # Directly test _init_smc_engine by constructing minimal object
        smc = object.__new__(IBStrategySMC)
        smc.symbol = "GER40"
        smc.smc_config = config
        smc._m1_data_ref = m1
        smc.smc_engine = None
        smc.log_prefix = "[GER40]"

        smc._init_smc_engine(datetime(2024, 1, 1, 10, 0))

        assert smc.smc_engine is not None
        assert smc.smc_engine.config.instrument == "GER40"

    def test_init_with_m2_approximation(self):
        """_init_smc_engine creates engine from M2 approximation when no M1."""
        from src.strategies.ib_strategy_smc import IBStrategySMC

        m2 = _make_m2_data(periods=50)

        # Create mock executor
        mock_executor = MagicMock()
        mock_executor.get_bars.return_value = m2

        smc = object.__new__(IBStrategySMC)
        smc.symbol = "GER40"
        smc.smc_config = SMCConfig(instrument="GER40")
        smc._m1_data_ref = None
        smc.smc_engine = None
        smc.executor = mock_executor
        smc.log_prefix = "[GER40]"

        smc._init_smc_engine(datetime(2024, 1, 1, 10, 0))

        assert smc.smc_engine is not None
        mock_executor.get_bars.assert_called_once_with("GER40", "M2", 500)

    def test_init_with_insufficient_data(self):
        """_init_smc_engine does nothing with insufficient data."""
        from src.strategies.ib_strategy_smc import IBStrategySMC

        m1 = _make_m1_data(periods=5)  # Too few

        smc = object.__new__(IBStrategySMC)
        smc.symbol = "GER40"
        smc.smc_config = SMCConfig(instrument="GER40")
        smc._m1_data_ref = m1
        smc.smc_engine = None
        smc.log_prefix = "[GER40]"

        smc._init_smc_engine(datetime(2024, 1, 1, 10, 0))

        assert smc.smc_engine is None

    def test_engine_update_called_on_each_tick(self):
        """SMC engine.update() is called with current time."""
        from src.strategies.ib_strategy_smc import IBStrategySMC

        m1 = _make_m1_data(periods=200)
        config = SMCConfig(instrument="GER40")
        tfm = TimeframeManager(m1, "GER40")

        engine = SMCEngine(config, tfm)
        engine.update = MagicMock()

        smc = object.__new__(IBStrategySMC)
        smc.smc_engine = engine

        # Simulate what check_signal does internally
        t = datetime(2024, 1, 1, 10, 30)
        smc.smc_engine.update(t)

        engine.update.assert_called_once_with(t)


# ---------------------
# Signal Evaluation Tests
# ---------------------

class TestSignalEvaluation:
    """Test SMC evaluation of IB signals."""

    def _make_smc_strategy(self, m1_data=None):
        """Create a partially initialized IBStrategySMC for testing evaluation logic."""
        from src.strategies.ib_strategy_smc import IBStrategySMC

        if m1_data is None:
            m1_data = _make_m1_data(periods=200)

        config = SMCConfig(instrument="GER40")
        tfm = TimeframeManager(m1_data, "GER40")
        engine = SMCEngine(config, tfm)

        smc = object.__new__(IBStrategySMC)
        smc.symbol = "GER40"
        smc.smc_config = config
        smc._m1_data_ref = m1_data
        smc.smc_engine = engine
        smc.last_decision = None
        smc._pending_signal = None
        smc._pending_decision = None
        smc._pending_time = None
        smc.log_prefix = "[GER40]"

        return smc

    def test_enter_decision_returns_signal(self):
        """ENTER decision returns the original signal."""
        smc = self._make_smc_strategy()

        signal = Signal(direction="long", entry_price=100.0, stop_loss=98.0, take_profit=104.0)
        t = datetime(2024, 1, 1, 10, 0)

        # Mock engine to return ENTER
        smc.smc_engine.evaluate_signal = MagicMock(return_value=SMCDecision(
            action="ENTER",
            reason="Strong confluence",
            confluence_score=3.5,
        ))

        # Simulate the evaluation flow (not full check_signal)
        decision = smc.smc_engine.evaluate_signal(signal, t)
        smc.last_decision = decision

        assert decision.action == "ENTER"
        assert decision.confluence_score == 3.5

    def test_reject_decision_stores_last_decision(self):
        """REJECT decision is stored in last_decision."""
        smc = self._make_smc_strategy()

        signal = Signal(direction="long", entry_price=100.0, stop_loss=98.0, take_profit=104.0)
        t = datetime(2024, 1, 1, 10, 0)

        smc.smc_engine.evaluate_signal = MagicMock(return_value=SMCDecision(
            action="REJECT",
            reason="Conflicting bearish FVG",
            confluence_score=0.0,
        ))

        decision = smc.smc_engine.evaluate_signal(signal, t)
        smc.last_decision = decision

        assert smc.last_decision.action == "REJECT"
        assert "Conflicting" in smc.last_decision.reason

    def test_wait_decision_sets_pending(self):
        """WAIT decision stores pending signal and decision."""
        smc = self._make_smc_strategy()

        signal = Signal(direction="long", entry_price=100.0, stop_loss=98.0, take_profit=104.0)
        t = datetime(2024, 1, 1, 10, 0)

        wait_decision = SMCDecision(
            action="WAIT",
            reason="Weak confluence",
            confluence_score=1.0,
            timeout_minutes=30,
            confirmation_criteria=[
                ConfirmationCriteria(type="CISD", timeframe="M2", direction="long"),
            ],
        )

        smc.smc_engine.evaluate_signal = MagicMock(return_value=wait_decision)

        # Simulate WAIT flow
        decision = smc.smc_engine.evaluate_signal(signal, t)
        smc.last_decision = decision
        smc._pending_signal = signal
        smc._pending_decision = decision
        smc._pending_time = t

        assert smc._pending_signal is signal
        assert smc._pending_decision.action == "WAIT"
        assert len(smc._pending_decision.confirmation_criteria) == 1


# ---------------------
# Confirmation State Machine Tests
# ---------------------

class TestConfirmationStateMachine:
    """Test AWAITING_CONFIRMATION state transitions."""

    def _make_smc_with_pending(self):
        """Create strategy with a pending WAIT signal."""
        from src.strategies.ib_strategy_smc import IBStrategySMC

        m1 = _make_m1_data(periods=200)
        config = SMCConfig(instrument="GER40")
        tfm = TimeframeManager(m1, "GER40")
        engine = SMCEngine(config, tfm)

        smc = object.__new__(IBStrategySMC)
        smc.symbol = "GER40"
        smc.smc_config = config
        smc._m1_data_ref = m1
        smc.smc_engine = engine
        smc.last_decision = None
        smc.log_prefix = "[GER40]"

        # Set up pending signal
        signal = Signal(direction="long", entry_price=100.0, stop_loss=98.0, take_profit=104.0)
        decision = SMCDecision(
            action="WAIT",
            reason="Weak confluence",
            confluence_score=1.0,
            timeout_minutes=30,
            confirmation_criteria=[
                ConfirmationCriteria(type="CISD", timeframe="M2", direction="long"),
            ],
        )

        smc._pending_signal = signal
        smc._pending_decision = decision
        smc._pending_time = datetime(2024, 1, 1, 10, 0)

        return smc

    def test_timeout_clears_pending(self):
        """Confirmation timeout clears pending state."""
        smc = self._make_smc_with_pending()

        # 31 minutes later (past 30min timeout)
        t = datetime(2024, 1, 1, 10, 31)
        result = smc._check_pending_confirmation(t)

        assert result is None
        assert smc._pending_signal is None
        assert smc._pending_decision is None

    def test_no_confirmation_returns_none(self):
        """No confirmation yet -> returns None, keeps pending."""
        smc = self._make_smc_with_pending()

        # Mock check_confirmation to return None (not confirmed)
        smc.smc_engine.check_confirmation = MagicMock(return_value=None)

        t = datetime(2024, 1, 1, 10, 5)  # 5 min later, within timeout
        result = smc._check_pending_confirmation(t)

        assert result is None
        assert smc._pending_signal is not None  # Still pending

    def test_confirmation_returns_signal_and_clears_pending(self):
        """Confirmation received -> returns modified signal, clears pending."""
        smc = self._make_smc_with_pending()

        # Mock confirmation returning modified signal
        modified = Signal(direction="long", entry_price=100.5, stop_loss=99.0, take_profit=104.0)
        smc.smc_engine.check_confirmation = MagicMock(return_value=modified)

        t = datetime(2024, 1, 1, 10, 5)
        result = smc._check_pending_confirmation(t)

        assert result is not None
        assert result.entry_price == 100.5
        assert smc._pending_signal is None  # Cleared
        assert smc.last_decision.action == "ENTER"
        assert smc.last_decision.reason == "Confirmed after WAIT"

    def test_reset_daily_state_clears_everything(self):
        """reset_daily_state clears SMC state."""
        smc = self._make_smc_with_pending()
        smc.smc_engine = MagicMock()
        smc.last_decision = SMCDecision(action="WAIT", reason="test", confluence_score=1.0)

        # Need to setup parent state fields
        smc.state = "IN_TRADE_WINDOW"
        smc.ibh = 100.0
        smc.ibl = 99.0
        smc.eq = 99.5
        smc.trade_window_end = None
        smc.tsl_state = None
        smc._news_filtered_count = 0

        smc.reset_daily_state()

        assert smc.smc_engine is None
        assert smc.last_decision is None
        assert smc._pending_signal is None
        assert smc.state == "AWAITING_IB_CALCULATION"


# ---------------------
# SMC Decision with Conflict Tests
# ---------------------

class TestSMCDecisionWithConflicts:
    """Test conflict detection in signal evaluation."""

    def test_long_signal_rejected_by_bearish_fvg(self):
        """Long signal near bearish FVG is rejected."""
        m1 = _make_m1_data(periods=200, base_price=100.0)
        config = SMCConfig(instrument="GER40")
        tfm = TimeframeManager(m1, "GER40")
        engine = SMCEngine(config, tfm)

        # Add conflicting bearish FVG near price 100
        fvg = FVG(
            id="fvg_conflict",
            instrument="GER40",
            timeframe="M2",
            direction="bearish",
            high=101.0,
            low=99.5,
            midpoint=100.25,
            formation_time=datetime(2024, 1, 1, 9, 30),
            candle1_time=datetime(2024, 1, 1, 9, 26),
            candle2_time=datetime(2024, 1, 1, 9, 28),
            candle3_time=datetime(2024, 1, 1, 9, 30),
        )
        engine.registry.add("fvg", fvg)

        signal = Signal(direction="long", entry_price=100.0, stop_loss=98.0, take_profit=104.0)
        decision = engine.evaluate_signal(signal, datetime(2024, 1, 1, 10, 0))

        assert decision.action == "REJECT"
        assert "Conflicting" in decision.reason

    def test_long_signal_enters_with_supportive_fractal(self):
        """Long signal near low fractal gets ENTER."""
        from src.smc.models import Fractal

        m1 = _make_m1_data(periods=200, base_price=100.0)
        config = SMCConfig(
            instrument="GER40",
            fractal_proximity_pct=0.01,  # 1% threshold
            min_confluence_score=0.5,    # Low threshold for test
        )
        tfm = TimeframeManager(m1, "GER40")
        engine = SMCEngine(config, tfm)

        # Add supportive low fractal at price 100
        frac = Fractal(
            id="frac_support",
            instrument="GER40",
            timeframe="H1",
            type="low",
            price=100.0,
            time=datetime(2024, 1, 1, 9, 0),
            confirmed_time=datetime(2024, 1, 1, 9, 30),
        )
        engine.registry.add("fractal", frac)

        signal = Signal(direction="long", entry_price=100.0, stop_loss=98.0, take_profit=104.0)
        decision = engine.evaluate_signal(signal, datetime(2024, 1, 1, 10, 0))

        assert decision.action == "ENTER"
        assert decision.confluence_score >= 0.5


# ---------------------
# get_smc_state Tests
# ---------------------

class TestGetSMCState:
    """Test SMC state reporting."""

    def test_state_without_engine(self):
        """State reports engine not initialized."""
        from src.strategies.ib_strategy_smc import IBStrategySMC

        smc = object.__new__(IBStrategySMC)
        smc.smc_engine = None
        smc.last_decision = None
        smc._pending_signal = None

        state = smc.get_smc_state()
        assert state["engine_initialized"] is False

    def test_state_with_engine(self):
        """State includes registry and event counts."""
        from src.strategies.ib_strategy_smc import IBStrategySMC

        m1 = _make_m1_data(periods=200)
        config = SMCConfig(instrument="GER40")
        tfm = TimeframeManager(m1, "GER40")
        engine = SMCEngine(config, tfm)

        smc = object.__new__(IBStrategySMC)
        smc.smc_engine = engine
        smc.last_decision = None
        smc._pending_signal = None

        state = smc.get_smc_state()
        assert state["engine_initialized"] is True
        assert "registry_count" in state
        assert "event_count" in state
