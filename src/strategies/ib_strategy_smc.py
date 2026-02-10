"""
IBStrategySMC - IB Strategy with Smart Money Concepts overlay.

Wraps IBStrategy to evaluate signals through SMC context (FVG, fractals,
BOS/CHoCH, CISD) before allowing entry.

Adds AWAITING_CONFIRMATION state for WAIT decisions.
Does NOT modify production IBStrategy code.
"""
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd
import pytz

from .base_strategy import Signal
from .ib_strategy import IBStrategy
from src.smc.config import SMCConfig
from src.smc.engine import SMCEngine
from src.smc.models import SMCDecision
from src.smc.timeframe_manager import TimeframeManager

logger = logging.getLogger(__name__)


class IBStrategySMC(IBStrategy):
    """
    IB Strategy with SMC (Smart Money Concepts) overlay.

    Extends IBStrategy to evaluate every signal through SMC confluence
    and confirmation checks before entry. Three possible outcomes:
    - ENTER: Signal passes SMC checks, return to caller for execution
    - WAIT: Signal lacks confluence, enter AWAITING_CONFIRMATION state
    - REJECT: Conflicting SMC structures, suppress signal entirely

    Usage:
        # In SingleDayBacktestSMC:
        strategy = IBStrategySMC(
            symbol="GER40",
            params=GER40_PARAMS_PROD,
            executor=executor,
            magic_number=1001,
            strategy_label="Debug_SMC",
            smc_config=SMCConfig(instrument="GER40"),
            m1_data=m1_data,   # pass full M1 data for accurate multi-TF
        )

        # In live trading (without M1):
        strategy = IBStrategySMC(
            symbol="GER40",
            params=GER40_PARAMS_PROD,
            executor=executor,
            magic_number=1001,
            smc_config=SMCConfig(instrument="GER40"),
        )
    """

    def __init__(
        self,
        symbol: str,
        params: Dict[str, Any],
        executor: Any,
        magic_number: int,
        strategy_label: str = "",
        news_filter_enabled: bool = False,
        smc_config: Optional[SMCConfig] = None,
        m1_data: Optional[pd.DataFrame] = None,
    ):
        """
        Args:
            symbol: Trading symbol
            params: Strategy parameters dict
            executor: MT5Executor or BacktestExecutor
            magic_number: Unique magic number
            strategy_label: Label for logging
            news_filter_enabled: Enable news filter
            smc_config: SMC configuration (defaults to SMCConfig(instrument=symbol))
            m1_data: Full M1 data for accurate multi-TF analysis.
                     If None, falls back to M2 approximation.
        """
        super().__init__(
            symbol=symbol,
            params=params,
            executor=executor,
            magic_number=magic_number,
            strategy_label=strategy_label,
            news_filter_enabled=news_filter_enabled,
        )

        self.smc_config = smc_config or SMCConfig(instrument=symbol)
        self._m1_data_ref = m1_data
        self.smc_engine: Optional[SMCEngine] = None
        self.last_decision: Optional[SMCDecision] = None

        # AWAITING_CONFIRMATION state
        self._pending_signal: Optional[Signal] = None
        self._pending_decision: Optional[SMCDecision] = None
        self._pending_time: Optional[datetime] = None

    def reset_daily_state(self) -> None:
        """Reset all daily state including SMC state."""
        super().reset_daily_state()
        self.smc_engine = None
        self.last_decision = None
        self._pending_signal = None
        self._pending_decision = None
        self._pending_time = None

    def check_signal(self, current_time_utc: datetime) -> Optional[Signal]:
        """
        FSM logic with SMC overlay.

        Flow:
        1. Ensure SMC engine is initialized once IB is calculated
        2. If AWAITING_CONFIRMATION, check confirmation (skip parent)
        3. Otherwise, call parent's check_signal
        4. If parent returns a signal, evaluate through SMC
        5. Based on SMC decision: return, store pending, or reject
        """
        # Initialize SMC engine once we're past IB calculation
        if self.state in ("IN_TRADE_WINDOW", "AWAITING_TRADE_WINDOW") and self.smc_engine is None:
            self._init_smc_engine(current_time_utc)

        # Update SMC engine structures on each tick
        if self.smc_engine is not None:
            self.smc_engine.update(current_time_utc)

        # Handle AWAITING_CONFIRMATION: check for confirmation
        if self._pending_signal is not None:
            return self._check_pending_confirmation(current_time_utc)

        # Normal flow: call parent
        signal = super().check_signal(current_time_utc)

        if signal is None:
            return None

        # No SMC engine -> pass through
        if self.smc_engine is None:
            return signal

        # Evaluate signal through SMC
        decision = self.smc_engine.evaluate_signal(signal, current_time_utc)
        self.last_decision = decision

        self.smc_engine.event_log.record_simple(
            timestamp=current_time_utc,
            instrument=self.symbol,
            event_type=f"SIGNAL_EVALUATED_{decision.action}",
            timeframe="M2",
            direction=signal.direction,
            price=signal.entry_price,
        )

        if decision.action == "ENTER":
            logger.info(
                f"{self.log_prefix} SMC ENTER: {decision.reason} "
                f"(score={decision.confluence_score:.1f})"
            )
            return signal

        elif decision.action == "WAIT":
            logger.info(
                f"{self.log_prefix} SMC WAIT: {decision.reason} "
                f"(score={decision.confluence_score:.1f}, "
                f"timeout={decision.timeout_minutes}min)"
            )
            self._pending_signal = signal
            self._pending_decision = decision
            self._pending_time = current_time_utc
            # Prevent parent from re-detecting the same signal
            # We stay in IN_TRADE_WINDOW state so parent keeps scanning,
            # but we intercept above before calling super()
            return None

        else:  # REJECT
            logger.info(
                f"{self.log_prefix} SMC REJECT: {decision.reason} "
                f"(score={decision.confluence_score:.1f})"
            )
            return None

    def _check_pending_confirmation(self, current_time_utc: datetime) -> Optional[Signal]:
        """Check if pending WAIT signal has been confirmed."""
        timeout_minutes = self._pending_decision.timeout_minutes
        elapsed = (current_time_utc - self._pending_time).total_seconds() / 60

        if elapsed >= timeout_minutes:
            logger.info(
                f"{self.log_prefix} SMC confirmation timeout "
                f"({timeout_minutes}min elapsed), dropping pending signal"
            )
            self.smc_engine.event_log.record_simple(
                timestamp=current_time_utc,
                instrument=self.symbol,
                event_type="CONFIRMATION_TIMEOUT",
                timeframe="M2",
                direction=self._pending_signal.direction,
            )
            self._clear_pending()
            return None

        # Check confirmation criteria through SMC engine
        if self.smc_engine is not None:
            confirmed = self.smc_engine.check_confirmation(
                self._pending_signal,
                self._pending_decision.confirmation_criteria,
                current_time_utc,
            )

            if confirmed is not None:
                logger.info(
                    f"{self.log_prefix} SMC confirmed! "
                    f"Modified entry={getattr(confirmed, 'entry_price', '?')}"
                )
                self.last_decision = SMCDecision(
                    action="ENTER",
                    reason="Confirmed after WAIT",
                    confluence_score=self._pending_decision.confluence_score,
                )
                self.smc_engine.event_log.record_simple(
                    timestamp=current_time_utc,
                    instrument=self.symbol,
                    event_type="SIGNAL_CONFIRMED",
                    timeframe="M2",
                    direction=self._pending_signal.direction,
                    price=getattr(confirmed, "entry_price", 0),
                )
                self._clear_pending()
                return confirmed

        # Still waiting
        return None

    def _clear_pending(self) -> None:
        """Clear pending confirmation state."""
        self._pending_signal = None
        self._pending_decision = None
        self._pending_time = None

    def _init_smc_engine(self, current_time_utc: datetime) -> None:
        """
        Initialize SMC engine with available M1 data.

        If m1_data was provided at construction, uses it directly.
        Otherwise, approximates M1 from M2 bars.

        The TimeframeManager receives all available data but the engine's
        detectors use up_to filtering, which correctly simulates progressive
        data arrival.
        """
        if self._m1_data_ref is not None:
            # Use real M1 data (best accuracy)
            if len(self._m1_data_ref) < 10:
                return
            tfm = TimeframeManager(self._m1_data_ref, self.symbol)
        else:
            # Fall back to M2 approximation
            m2_bars = self.executor.get_bars(self.symbol, "M2", 500)
            if m2_bars is None or len(m2_bars) < 5:
                return
            m1_approx = self._m2_to_m1_approximation(m2_bars)
            tfm = TimeframeManager(m1_approx, self.symbol)

        self.smc_engine = SMCEngine(self.smc_config, tfm)
        logger.info(
            f"{self.log_prefix} SMC engine initialized "
            f"(fractals={self.smc_config.enable_fractals}, "
            f"fvg={self.smc_config.enable_fvg}, "
            f"cisd={self.smc_config.enable_cisd}, "
            f"bos={self.smc_config.enable_bos})"
        )

    @staticmethod
    def _m2_to_m1_approximation(m2_bars: pd.DataFrame) -> pd.DataFrame:
        """
        Convert M2 bars to approximate M1 bars.

        Each M2 bar becomes two synthetic M1 bars:
        - First M1: open -> midpoint
        - Second M1: midpoint -> close

        This is a lossy approximation. Prefer passing real M1 data.
        """
        m1_rows = []
        for _, row in m2_bars.iterrows():
            t = row["time"]
            o, h, l, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
            mid = (o + c) / 2.0

            if c >= o:
                # Bullish M2 candle
                h1 = max(o, mid)
                l1 = l
                h2 = h
                l2 = min(mid, c)
            else:
                # Bearish M2 candle
                h1 = h
                l1 = min(o, mid)
                h2 = max(mid, c)
                l2 = l

            m1_rows.append({
                "time": t,
                "open": o,
                "high": h1,
                "low": l1,
                "close": mid,
            })
            m1_rows.append({
                "time": t + timedelta(minutes=1),
                "open": mid,
                "high": h2,
                "low": l2,
                "close": c,
            })

        return pd.DataFrame(m1_rows)

    def get_smc_state(self) -> Dict[str, Any]:
        """Get current SMC state for debugging/charting."""
        state = {
            "engine_initialized": self.smc_engine is not None,
            "last_decision": self.last_decision,
            "pending_signal": self._pending_signal is not None,
        }

        if self.smc_engine is not None:
            state["registry_count"] = self.smc_engine.registry.count()
            state["event_count"] = len(self.smc_engine.event_log.events)
            state["market_structures"] = {
                tf: ms.current_trend
                for tf, ms in self.smc_engine.market_structures.items()
            }

        return state
