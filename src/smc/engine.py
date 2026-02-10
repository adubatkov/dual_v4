"""
SMC Engine - Main orchestrator for Smart Money Concepts analysis.

Responsibilities:
- Update all SMC structures on each bar
- Evaluate IB signals with SMC context
- Calculate confluence scores
- Check confirmation criteria
- Manage structure lifecycle
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .config import SMCConfig
from .detectors.cisd_detector import check_cisd_invalidation, detect_cisd
from .detectors.fvg_detector import check_fvg_fill, check_fvg_rebalance, detect_fvg
from .detectors.fractal_detector import (
    check_fractal_sweep,
    detect_fractals,
    find_unswept_fractals,
)
from .detectors.market_structure_detector import detect_bos_choch, detect_swing_points
from .event_log import SMCEventLog
from .models import (
    BOS,
    CISD,
    FVG,
    ConfirmationCriteria,
    ConfluenceScore,
    Fractal,
    MarketStructure,
    SMCDecision,
)
from .registry import SMCRegistry
from .timeframe_manager import TimeframeManager


class SMCEngine:
    """
    Main SMC orchestration engine.

    Manages all SMC detectors, registry, and decision logic.
    """

    def __init__(
        self,
        config: SMCConfig,
        timeframe_manager: TimeframeManager,
    ):
        self.config = config
        self.tfm = timeframe_manager
        self.registry = SMCRegistry(config.instrument)
        self.event_log = SMCEventLog()

        # Track market structure per timeframe
        self.market_structures: Dict[str, MarketStructure] = {}

    def update(
        self,
        current_time: datetime,
        new_m1_bars: Optional[pd.DataFrame] = None,
    ) -> None:
        """
        Update all SMC structures.

        Called on each M1 bar in slow backtest / live trading.
        For fast backtest, call once per day with full day data.
        """
        if new_m1_bars is not None and not new_m1_bars.empty:
            self.tfm.append_m1(new_m1_bars)

        if self.config.enable_fractals:
            self._update_fractals(current_time)

        if self.config.enable_fvg:
            self._update_fvgs(current_time)

        if self.config.enable_bos:
            self._update_market_structure(current_time)

        if self.config.enable_cisd:
            self._update_cisds(current_time)

        # Cleanup old structures
        cleanup_before = current_time - timedelta(hours=self.config.max_structure_age_hours)
        removed = self.registry.cleanup(cleanup_before)
        if removed > 0:
            self.event_log.record_simple(
                timestamp=current_time,
                instrument=self.config.instrument,
                event_type="CLEANUP",
                timeframe="all",
                removed_count=removed,
            )

    def _update_fractals(self, current_time: datetime) -> None:
        """Update fractal structures."""
        tf = self.config.fractal_timeframe
        ohlc = self.tfm.get_data(tf, up_to=current_time)

        if len(ohlc) < 3:
            return

        candle_duration = 1.0 if tf == "H1" else 4.0 if tf == "H4" else 1.0
        fractals = detect_fractals(
            ohlc,
            self.config.instrument,
            tf,
            candle_duration_hours=candle_duration,
        )

        m1_data = self.tfm.get_data("M1", up_to=current_time)
        for frac in fractals:
            if frac.confirmed_time > current_time:
                continue

            existing = self.registry.get_by_id("fractal", frac.id)
            if existing:
                continue

            sweep_time = check_fractal_sweep(
                frac,
                m1_data,
                frac.confirmed_time,
                current_time,
            )

            if sweep_time:
                frac.swept = True
                frac.sweep_time = sweep_time
                frac.status = "swept"

            self.registry.add("fractal", frac)

            self.event_log.record_simple(
                timestamp=frac.confirmed_time,
                instrument=self.config.instrument,
                event_type="FRACTAL_CONFIRMED",
                timeframe=tf,
                direction="high" if frac.type == "high" else "low",
                price=frac.price,
                structure_id=frac.id,
            )

    def _update_fvgs(self, current_time: datetime) -> None:
        """Update FVG structures."""
        for tf in self.config.fvg_timeframes:
            ohlc = self.tfm.get_data(tf, up_to=current_time)

            if len(ohlc) < 3:
                continue

            fvgs = detect_fvg(
                ohlc,
                self.config.instrument,
                tf,
                min_size_points=self.config.fvg_min_size_points,
            )

            m1_data = self.tfm.get_data("M1", up_to=current_time)

            for fvg in fvgs:
                if fvg.formation_time > current_time:
                    continue

                existing = self.registry.get_by_id("fvg", fvg.id)
                if existing:
                    continue

                fill_result = check_fvg_fill(
                    fvg,
                    m1_data,
                    fvg.formation_time,
                    current_time,
                )

                if fill_result:
                    fvg.fill_pct = fill_result["fill_pct"]
                    fvg.fill_time = fill_result["fill_time"]
                    fvg.status = fill_result["fill_type"] + "_fill"

                self.registry.add("fvg", fvg)

                self.event_log.record_simple(
                    timestamp=fvg.formation_time,
                    instrument=self.config.instrument,
                    event_type="FVG_FORMED",
                    timeframe=tf,
                    direction=fvg.direction,
                    price=fvg.midpoint,
                    structure_id=fvg.id,
                )

    def _update_market_structure(self, current_time: datetime) -> None:
        """Update market structure and BOS/CHoCH."""
        for tf in self.config.bos_timeframes:
            ohlc = self.tfm.get_data(tf, up_to=current_time)

            if len(ohlc) < 10:
                continue

            ms = detect_swing_points(
                ohlc,
                self.config.instrument,
                tf,
                lookback=3,
            )

            self.market_structures[tf] = ms

            fvgs = self.registry.get_active("fvg", timeframe=tf)
            bos_list = detect_bos_choch(
                ms,
                ohlc,
                self.config.instrument,
                tf,
                fvgs=fvgs,
            )

            for bos in bos_list:
                existing = self.registry.get_by_id("bos", bos.id)
                if existing:
                    continue

                self.registry.add("bos", bos)

                self.event_log.record_simple(
                    timestamp=bos.break_time,
                    instrument=self.config.instrument,
                    event_type=f"{bos.bos_type.upper()}_DETECTED",
                    timeframe=tf,
                    direction=bos.direction,
                    price=bos.broken_level,
                    structure_id=bos.id,
                    bias_impact=1.0 if bos.direction == "bullish" else -1.0,
                )

    def _update_cisds(self, current_time: datetime) -> None:
        """Update CISD structures."""
        for tf in self.config.cisd_timeframes:
            ohlc = self.tfm.get_data(tf, up_to=current_time)

            if len(ohlc) < 5:
                continue

            # Build POI zones from FVGs and fractals
            poi_zones = []

            fvgs = self.registry.get_active("fvg", timeframe=tf, before_time=current_time)
            for fvg in fvgs:
                poi_zones.append({
                    "type": "fvg",
                    "direction": fvg.direction,
                    "high": fvg.high,
                    "low": fvg.low,
                    "time": fvg.formation_time,
                })

            fractals = self.registry.get_unswept_fractals(
                self.config.fractal_timeframe,
                before_time=current_time,
            )
            for frac in fractals:
                poi_zones.append({
                    "type": "fractal",
                    "direction": "bearish" if frac.type == "high" else "bullish",
                    "high": frac.price + 2,
                    "low": frac.price - 2,
                    "time": frac.confirmed_time,
                })

            cisds = detect_cisd(
                ohlc,
                self.config.instrument,
                tf,
                poi_zones,
            )

            for cisd in cisds:
                existing = self.registry.get_by_id("cisd", cisd.id)
                if existing:
                    continue

                if check_cisd_invalidation(cisd, ohlc, fvgs, fractals):
                    cisd.status = "invalidated"

                self.registry.add("cisd", cisd)

                self.event_log.record_simple(
                    timestamp=cisd.confirmation_time,
                    instrument=self.config.instrument,
                    event_type="CISD_DETECTED",
                    timeframe=tf,
                    direction=cisd.direction,
                    price=cisd.confirmation_close,
                    structure_id=cisd.id,
                    bias_impact=2.0 if cisd.direction == "long" else -2.0,
                )

    def evaluate_signal(
        self,
        signal: Any,
        current_time: datetime,
    ) -> SMCDecision:
        """
        Evaluate an IB signal with SMC context.

        Args:
            signal: IB signal object (must have .direction attribute)
            current_time: Current time

        Returns:
            SMCDecision with action (ENTER/WAIT/REJECT) and reasoning
        """
        signal_direction = signal.direction
        signal_price = getattr(signal, "entry_price", None)

        if signal_price is None:
            m1_data = self.tfm.get_data("M1", up_to=current_time)
            signal_price = float(m1_data.iloc[-1]["close"]) if not m1_data.empty else 0

        confluence = self.calculate_confluence(
            signal_price,
            signal_direction,
            current_time,
        )

        conflicts = self._check_conflicts(signal_direction, signal_price, current_time)

        if conflicts:
            return SMCDecision(
                action="REJECT",
                reason=f"Conflicting structures: {conflicts}",
                confluence_score=confluence.total,
            )

        if confluence.total >= self.config.min_confluence_score:
            return SMCDecision(
                action="ENTER",
                reason=f"Strong confluence ({confluence.total:.1f}): {', '.join(confluence.contributing_structures)}",
                confluence_score=confluence.total,
            )

        criteria = self._build_confirmation_criteria(signal_direction)

        return SMCDecision(
            action="WAIT",
            reason=f"Weak confluence ({confluence.total:.1f}), waiting for confirmation",
            confirmation_criteria=criteria,
            confluence_score=confluence.total,
            timeout_minutes=self.config.max_wait_minutes,
        )

    def check_confirmation(
        self,
        pending_signal: Any,
        criteria: List[ConfirmationCriteria],
        current_time: datetime,
    ) -> Optional[Any]:
        """
        Check if confirmation criteria are met.

        Returns:
            Modified signal if confirmed, None otherwise
        """
        for criterion in criteria:
            if criterion.type == "CISD":
                if self._check_cisd_criterion(criterion, current_time):
                    return self._create_modified_signal(pending_signal, criterion, current_time)

            elif criterion.type == "FVG_REBALANCE":
                if self._check_fvg_rebalance_criterion(criterion, current_time):
                    return self._create_modified_signal(pending_signal, criterion, current_time)

            elif criterion.type == "BOS":
                if self._check_bos_criterion(criterion, current_time):
                    return self._create_modified_signal(pending_signal, criterion, current_time)

        return None

    def calculate_confluence(
        self,
        price: float,
        direction: str,
        current_time: datetime,
    ) -> ConfluenceScore:
        """
        Calculate weighted confluence score from all active structures.
        """
        score = 0.0
        breakdown = {}
        contributing = []

        if self.config.enable_fractals:
            frac_score, frac_ids = self._score_fractals(price, direction, current_time)
            score += frac_score
            if frac_score > 0:
                breakdown["fractals"] = frac_score
                contributing.extend(frac_ids)

        if self.config.enable_fvg:
            fvg_score, fvg_ids = self._score_fvgs(price, direction, current_time)
            score += fvg_score
            if fvg_score > 0:
                breakdown["fvgs"] = fvg_score
                contributing.extend(fvg_ids)

        if self.config.enable_cisd:
            cisd_score, cisd_ids = self._score_cisds(direction, current_time)
            score += cisd_score
            if cisd_score > 0:
                breakdown["cisds"] = cisd_score
                contributing.extend(cisd_ids)

        if self.config.enable_bos:
            bos_score, bos_ids = self._score_bos(direction, current_time)
            score += bos_score
            if bos_score > 0:
                breakdown["bos"] = bos_score
                contributing.extend(bos_ids)

        return ConfluenceScore(
            total=score,
            breakdown=breakdown,
            direction_bias=direction,
            contributing_structures=contributing,
        )

    def _score_fractals(
        self,
        price: float,
        direction: str,
        current_time: datetime,
    ) -> Tuple[float, List[str]]:
        """Score fractals near price."""
        fractals = self.registry.get_unswept_fractals(
            self.config.fractal_timeframe,
            before_time=current_time,
        )

        score = 0.0
        ids = []
        threshold = price * self.config.fractal_proximity_pct

        for frac in fractals:
            distance = abs(price - frac.price)
            if distance > threshold:
                continue

            if direction == "long" and frac.type == "low":
                score += self.config.weight_fractal
                ids.append(frac.id)

            if direction == "short" and frac.type == "high":
                score += self.config.weight_fractal
                ids.append(frac.id)

        return score, ids

    def _score_fvgs(
        self,
        price: float,
        direction: str,
        current_time: datetime,
    ) -> Tuple[float, List[str]]:
        """Score FVGs near price."""
        score = 0.0
        ids = []

        for tf in self.config.fvg_timeframes:
            fvgs = self.registry.get_fvgs_near_price(
                tf,
                price,
                max_distance_pct=0.005,
            )

            for fvg in fvgs:
                if direction == "long" and fvg.direction == "bullish":
                    score += self.config.weight_fvg
                    ids.append(fvg.id)

                if direction == "short" and fvg.direction == "bearish":
                    score += self.config.weight_fvg
                    ids.append(fvg.id)

        return score, ids

    def _score_cisds(
        self,
        direction: str,
        current_time: datetime,
    ) -> Tuple[float, List[str]]:
        """Score recent CISDs."""
        score = 0.0
        ids = []

        recent_threshold = current_time - timedelta(minutes=30)

        for tf in self.config.cisd_timeframes:
            cisds = self.registry.get_active("cisd", timeframe=tf, direction=direction)

            for cisd in cisds:
                if cisd.confirmation_time >= recent_threshold:
                    score += self.config.weight_cisd
                    ids.append(cisd.id)

        return score, ids

    def _score_bos(
        self,
        direction: str,
        current_time: datetime,
    ) -> Tuple[float, List[str]]:
        """Score recent BOS in signal direction."""
        score = 0.0
        ids = []

        recent_threshold = current_time - timedelta(minutes=30)

        for tf in self.config.bos_timeframes:
            bos_dir = "bullish" if direction == "long" else "bearish"
            bos_list = self.registry.get_active("bos", timeframe=tf, direction=bos_dir)

            for bos in bos_list:
                if bos.break_time >= recent_threshold:
                    weight = self.config.weight_bos if bos.bos_type == "bos" else self.config.weight_bos * 0.5
                    score += weight
                    ids.append(bos.id)

        return score, ids

    def _check_conflicts(
        self,
        direction: str,
        price: float,
        current_time: datetime,
    ) -> List[str]:
        """Check for conflicting structures."""
        conflicts = []

        if direction == "long":
            fvgs = self.registry.get_fvgs_near_price(
                self.config.fvg_timeframes[0] if self.config.fvg_timeframes else "M2",
                price,
                max_distance_pct=0.01,
                direction="bearish",
            )
            if fvgs:
                conflicts.append("bearish_fvg_above")

        else:  # short
            fvgs = self.registry.get_fvgs_near_price(
                self.config.fvg_timeframes[0] if self.config.fvg_timeframes else "M2",
                price,
                max_distance_pct=0.01,
                direction="bullish",
            )
            if fvgs:
                conflicts.append("bullish_fvg_below")

        return conflicts

    def _build_confirmation_criteria(
        self,
        direction: str,
    ) -> List[ConfirmationCriteria]:
        """Build confirmation criteria for WAIT decision."""
        criteria = []

        if self.config.enable_cisd:
            criteria.append(
                ConfirmationCriteria(
                    type="CISD",
                    timeframe=self.config.cisd_timeframes[0] if self.config.cisd_timeframes else "M2",
                    direction=direction,
                )
            )

        if self.config.enable_fvg:
            criteria.append(
                ConfirmationCriteria(
                    type="FVG_REBALANCE",
                    timeframe=self.config.fvg_timeframes[0] if self.config.fvg_timeframes else "M2",
                    direction="bullish" if direction == "long" else "bearish",
                )
            )

        return criteria

    def _check_cisd_criterion(
        self,
        criterion: ConfirmationCriteria,
        current_time: datetime,
    ) -> bool:
        """Check if CISD criterion is met."""
        recent_threshold = current_time - timedelta(minutes=10)

        cisds = self.registry.get_active(
            "cisd",
            timeframe=criterion.timeframe,
            direction=criterion.direction,
        )

        for cisd in cisds:
            if cisd.confirmation_time >= recent_threshold:
                return True

        return False

    def _check_fvg_rebalance_criterion(
        self,
        criterion: ConfirmationCriteria,
        current_time: datetime,
    ) -> bool:
        """Check if FVG rebalance criterion is met."""
        m1_data = self.tfm.get_data("M1", up_to=current_time)
        if m1_data.empty:
            return False

        last_candle = m1_data.iloc[-1]

        fvgs = self.registry.get_active(
            "fvg",
            timeframe=criterion.timeframe,
            direction=criterion.direction,
        )

        for fvg in fvgs:
            if check_fvg_rebalance(fvg, last_candle):
                return True

        return False

    def _check_bos_criterion(
        self,
        criterion: ConfirmationCriteria,
        current_time: datetime,
    ) -> bool:
        """Check if BOS criterion is met."""
        recent_threshold = current_time - timedelta(minutes=15)

        bos_list = self.registry.get_active(
            "bos",
            timeframe=criterion.timeframe,
            direction=criterion.direction,
        )

        for bos in bos_list:
            if bos.break_time >= recent_threshold:
                return True

        return False

    def _create_modified_signal(
        self,
        original_signal: Any,
        confirming_criterion: ConfirmationCriteria,
        current_time: datetime,
    ) -> Any:
        """Create modified signal with SMC-adjusted parameters."""
        m1_data = self.tfm.get_data("M1", up_to=current_time)
        if m1_data.empty:
            return original_signal

        conf_candle = m1_data.iloc[-1]

        modified_signal = original_signal

        modified_signal.entry_price = float(conf_candle["close"])

        if original_signal.direction == "long":
            modified_signal.stop_loss = float(conf_candle["low"])
        else:
            modified_signal.stop_loss = float(conf_candle["high"])

        return modified_signal
