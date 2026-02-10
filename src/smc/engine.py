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
    OrderFlow,
    SMCDayContext,
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

    # ================================
    # Phase 5: Validation Methods
    # ================================

    def validate_fvg_status(
        self,
        fvg: FVG,
        current_bar: pd.Series,
        recent_bos: Optional[List[BOS]] = None,
    ) -> Tuple[str, Optional[str]]:
        """
        Validate FVG status according to 6 invalidation rules.

        Implements 4 of 6 variants (inducement and zone_to_lq deferred to Phase 6).

        Returns:
            (new_status, invalidation_reason) tuple
        """
        if fvg.status != "active":
            return (fvg.status, fvg.invalidation_reason)

        # Variant 1: Structural Update (BOS/CHoCH after FVG)
        if recent_bos:
            for bos in recent_bos:
                if bos.break_time > fvg.formation_time:
                    if (fvg.direction == "bullish" and bos.direction == "bullish") or \
                       (fvg.direction == "bearish" and bos.direction == "bearish"):
                        return ("structural_update", f"Led to {bos.bos_type} at {bos.break_time}")

        # Variant 3: Zone-to-Zone Mitigation (pass-through)
        bar_in_fvg = (current_bar["low"] <= fvg.high and current_bar["high"] >= fvg.low)
        if bar_in_fvg:
            body_range = abs(current_bar["close"] - current_bar["open"])
            fvg_size = fvg.high - fvg.low
            if fvg_size > 0 and body_range > fvg_size * 1.5:
                if fvg.direction == "bullish" and current_bar["close"] > fvg.high:
                    return ("zone_to_zone", f"Pass-through at {current_bar.get('time', 'unknown')}")
                elif fvg.direction == "bearish" and current_bar["close"] < fvg.low:
                    return ("zone_to_zone", f"Pass-through at {current_bar.get('time', 'unknown')}")

        # Variant 6: Inversion (checked before full_fill - more specific condition)
        fvg_size = fvg.high - fvg.low
        if fvg_size > 0:
            if fvg.direction == "bullish" and current_bar["close"] < fvg.low:
                if current_bar["open"] > current_bar["close"]:
                    body = current_bar["open"] - current_bar["close"]
                    if body > fvg_size * 0.8:
                        return ("inverted", f"Inverted at {current_bar.get('time', 'unknown')}")
            elif fvg.direction == "bearish" and current_bar["close"] > fvg.high:
                if current_bar["close"] > current_bar["open"]:
                    body = current_bar["close"] - current_bar["open"]
                    if body > fvg_size * 0.8:
                        return ("inverted", f"Inverted at {current_bar.get('time', 'unknown')}")

        # Variant 5: Full Fill
        fill_pct = self._calculate_fvg_fill(fvg, current_bar)
        if fill_pct >= 1.0:
            return ("full_fill", f"Completely filled at {current_bar.get('time', 'unknown')}")
        fvg.fill_pct = max(fvg.fill_pct, fill_pct)

        return ("active", None)

    def _calculate_fvg_fill(self, fvg: FVG, current_bar: pd.Series) -> float:
        """Calculate how much of FVG has been filled (0.0 to 1.0)."""
        gap_size = fvg.high - fvg.low
        if gap_size <= 0:
            return 0.0

        if fvg.direction == "bullish":
            if current_bar["low"] <= fvg.high:
                fill_depth = fvg.high - current_bar["low"]
                return min(fill_depth / gap_size, 1.0)
        else:
            if current_bar["high"] >= fvg.low:
                fill_depth = current_bar["high"] - fvg.low
                return min(fill_depth / gap_size, 1.0)

        return 0.0

    def validate_bos(
        self,
        bos: BOS,
        smc_context: SMCDayContext,
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate if BOS is a true structure break.

        Rule: BOS invalid if zone exists below (bullish) or above (bearish)
        that could continue movement.
        """
        proximity_pct = 0.02

        if bos.direction == "bullish":
            for fvg in smc_context.fvgs_h1 + smc_context.fvgs_m2:
                if fvg.status == "active" and fvg.direction == "bearish":
                    if fvg.midpoint < bos.broken_level and fvg.midpoint > bos.broken_level * (1 - proximity_pct):
                        return (False, f"Bearish FVG below BOS at {fvg.midpoint:.2f}")

            for f_low in smc_context.active_fractal_lows:
                if f_low < bos.broken_level and f_low > bos.broken_level * (1 - proximity_pct):
                    return (False, f"Unswept low fractal below BOS at {f_low:.2f}")

        else:  # bearish
            for fvg in smc_context.fvgs_h1 + smc_context.fvgs_m2:
                if fvg.status == "active" and fvg.direction == "bullish":
                    if fvg.midpoint > bos.broken_level and fvg.midpoint < bos.broken_level * (1 + proximity_pct):
                        return (False, f"Bullish FVG above BOS at {fvg.midpoint:.2f}")

            for f_high in smc_context.active_fractal_highs:
                if f_high > bos.broken_level and f_high < bos.broken_level * (1 + proximity_pct):
                    return (False, f"Unswept high fractal above BOS at {f_high:.2f}")

        return (True, None)

    def validate_cisd(
        self,
        cisd: CISD,
        smc_context: SMCDayContext,
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate if CISD is a true state change.

        Rule: CISD invalid if zone exists that could continue movement against direction.
        """
        proximity_pct = 0.02
        ref_price = cisd.confirmation_close

        if cisd.direction == "long":
            for fvg in smc_context.fvgs_h1 + smc_context.fvgs_m2:
                if fvg.status == "active" and fvg.direction == "bearish":
                    if fvg.midpoint < ref_price and fvg.midpoint > ref_price * (1 - proximity_pct):
                        return (False, f"Bearish FVG below CISD at {fvg.midpoint:.2f}")

            for f_low in smc_context.active_fractal_lows:
                if f_low < ref_price and f_low > ref_price * (1 - proximity_pct):
                    return (False, f"Unswept low fractal below CISD at {f_low:.2f}")

        else:  # short
            for fvg in smc_context.fvgs_h1 + smc_context.fvgs_m2:
                if fvg.status == "active" and fvg.direction == "bullish":
                    if fvg.midpoint > ref_price and fvg.midpoint < ref_price * (1 + proximity_pct):
                        return (False, f"Bullish FVG above CISD at {fvg.midpoint:.2f}")

            for f_high in smc_context.active_fractal_highs:
                if f_high > ref_price and f_high < ref_price * (1 + proximity_pct):
                    return (False, f"Unswept high fractal above CISD at {f_high:.2f}")

        return (True, None)

    def validate_order_flow(
        self,
        order_flow: OrderFlow,
        current_bar: pd.Series,
    ) -> Tuple[str, Optional[str]]:
        """
        Validate order flow status.

        Rules:
            - Needs 2 confirmations (steps) to be validated
            - Invalidated by close below/above last confirmed step
        """
        if order_flow.status == "invalidated":
            return (order_flow.status, order_flow.invalidation_reason)

        bar_close = current_bar["close"]

        if order_flow.direction == "long":
            if order_flow.step2_price is not None:
                if bar_close < order_flow.step2_price:
                    return ("invalidated", f"Close below step2 at {current_bar.get('time', 'unknown')}")
            elif order_flow.step1_price is not None:
                if bar_close < order_flow.step1_price:
                    return ("invalidated", f"Close below step1 at {current_bar.get('time', 'unknown')}")
        else:  # short
            if order_flow.step2_price is not None:
                if bar_close > order_flow.step2_price:
                    return ("invalidated", f"Close above step2 at {current_bar.get('time', 'unknown')}")
            elif order_flow.step1_price is not None:
                if bar_close > order_flow.step1_price:
                    return ("invalidated", f"Close above step1 at {current_bar.get('time', 'unknown')}")

        if order_flow.status == "pending" and order_flow.step1_price is not None:
            return ("step1_confirmed", None)

        if order_flow.status == "step1_confirmed" and order_flow.step2_price is not None:
            return ("validated", "Two steps confirmed")

        return (order_flow.status, None)

    def validate_entry(
        self,
        signal: Any,
        smc_context: SMCDayContext,
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate entry conditions (Model 1 or Model 2).

        Model 1: Full conviction (continuation) - needs 2+ confirmations, no opposing zones
        Model 2: Lower conviction (reversal) - needs 1+ confirmation, allows some opposing
        """
        signal_type = getattr(signal, "signal_type", "")
        is_reversal = signal_type in ("Reverse", "REV_RB")

        opposing_zones = self._find_opposing_zones(signal, smc_context)
        confirmations = self._count_confirmations(signal, smc_context)

        if not is_reversal:
            if len(opposing_zones) > 0:
                return (False, f"Opposing zones: {', '.join(opposing_zones)}")
            if confirmations < 2:
                return (False, f"Insufficient confirmations ({confirmations}/2)")
            return (True, None)
        else:
            if confirmations < 1:
                return (False, "No confirmations for reversal")
            if len(opposing_zones) > 2:
                return (False, f"Too many opposing zones ({len(opposing_zones)})")
            return (True, "Model 2 (reversal entry)")

    def _find_opposing_zones(
        self,
        signal: Any,
        smc_context: SMCDayContext,
    ) -> List[str]:
        """Find zones that oppose signal direction."""
        opposing = []
        entry_price = getattr(signal, "entry_price", 0)
        direction = getattr(signal, "direction", "long")
        proximity_pct = 0.02

        if direction == "long":
            for fvg in smc_context.fvgs_h1 + smc_context.fvgs_m2:
                if fvg.status == "active" and fvg.direction == "bearish":
                    if fvg.low > entry_price and fvg.low < entry_price * (1 + proximity_pct):
                        opposing.append(f"Bearish FVG at {fvg.midpoint:.2f}")

            for f_high in smc_context.active_fractal_highs:
                if f_high < entry_price and f_high > entry_price * (1 - proximity_pct):
                    opposing.append(f"High fractal at {f_high:.2f}")
        else:
            for fvg in smc_context.fvgs_h1 + smc_context.fvgs_m2:
                if fvg.status == "active" and fvg.direction == "bullish":
                    if fvg.high < entry_price and fvg.high > entry_price * (1 - proximity_pct):
                        opposing.append(f"Bullish FVG at {fvg.midpoint:.2f}")

            for f_low in smc_context.active_fractal_lows:
                if f_low > entry_price and f_low < entry_price * (1 + proximity_pct):
                    opposing.append(f"Low fractal at {f_low:.2f}")

        return opposing

    def _count_confirmations(
        self,
        signal: Any,
        smc_context: SMCDayContext,
    ) -> int:
        """Count confirmations for signal."""
        count = 0
        entry_price = getattr(signal, "entry_price", 0)
        direction = getattr(signal, "direction", "long")
        proximity_pct = 0.01

        # Confirmation 1: Supporting FVG
        for fvg in smc_context.fvgs_h1 + smc_context.fvgs_m2:
            if fvg.status == "active":
                if direction == "long" and fvg.direction == "bullish":
                    if entry_price > 0 and abs(fvg.midpoint - entry_price) / entry_price < proximity_pct:
                        count += 1
                        break
                elif direction == "short" and fvg.direction == "bearish":
                    if entry_price > 0 and abs(fvg.midpoint - entry_price) / entry_price < proximity_pct:
                        count += 1
                        break

        # Confirmation 2: CISD in signal direction
        for cisd in smc_context.cisd_events:
            if cisd.direction == direction:
                if entry_price > 0 and abs(cisd.confirmation_close - entry_price) / entry_price < proximity_pct:
                    count += 1
                    break

        # Confirmation 3: BOS in signal direction
        bos_dir = "bullish" if direction == "long" else "bearish"
        for bos in smc_context.bos_events:
            if bos.direction == bos_dir:
                if entry_price > 0 and abs(bos.broken_level - entry_price) / entry_price < 0.02:
                    count += 1
                    break

        return count

    def validate_tp_target(
        self,
        tp_price: float,
        direction: str,
        smc_context: SMCDayContext,
    ) -> bool:
        """
        Validate if TP target is a valid open zone.

        Valid targets: active FVG (opposite direction), unswept fractal.
        """
        proximity_pct = 0.005

        for fvg in smc_context.fvgs_h1 + smc_context.fvgs_m2:
            if fvg.status != "active":
                continue
            if direction == "long" and fvg.direction == "bearish":
                if tp_price > 0 and abs(fvg.midpoint - tp_price) / tp_price < proximity_pct:
                    return True
            elif direction == "short" and fvg.direction == "bullish":
                if tp_price > 0 and abs(fvg.midpoint - tp_price) / tp_price < proximity_pct:
                    return True

        if direction == "long":
            for f_high in smc_context.active_fractal_highs:
                if tp_price > 0 and abs(f_high - tp_price) / tp_price < proximity_pct:
                    return True
        else:
            for f_low in smc_context.active_fractal_lows:
                if tp_price > 0 and abs(f_low - tp_price) / tp_price < proximity_pct:
                    return True

        return False

    def find_nearest_valid_tp(
        self,
        entry_price: float,
        direction: str,
        smc_context: SMCDayContext,
    ) -> Optional[float]:
        """Find nearest valid TP target (open zone beyond entry)."""
        candidates = []

        for fvg in smc_context.fvgs_h1 + smc_context.fvgs_m2:
            if fvg.status != "active":
                continue
            if direction == "long" and fvg.direction == "bearish" and fvg.midpoint > entry_price:
                candidates.append(fvg.midpoint)
            elif direction == "short" and fvg.direction == "bullish" and fvg.midpoint < entry_price:
                candidates.append(fvg.midpoint)

        if direction == "long":
            for f_high in smc_context.active_fractal_highs:
                if f_high > entry_price:
                    candidates.append(f_high)
        else:
            for f_low in smc_context.active_fractal_lows:
                if f_low < entry_price:
                    candidates.append(f_low)

        if not candidates:
            return None

        if direction == "long":
            candidates.sort()
        else:
            candidates.sort(reverse=True)

        return candidates[0]

    def validate_sl_placement(
        self,
        sl_price: float,
        entry_price: float,
        direction: str,
        smc_context: SMCDayContext,
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate SL placement behind idea invalidation.

        Returns warning (not rejection) - SL placement is advisory.
        """
        if direction == "long":
            for fvg in smc_context.fvgs_m2 + smc_context.fvgs_h1:
                if fvg.status == "active" and fvg.direction == "bullish":
                    if sl_price < fvg.midpoint < entry_price:
                        return (False, f"SL behind bullish FVG POI at {fvg.midpoint:.2f}")

            for cisd in smc_context.cisd_events:
                if cisd.direction == "long":
                    if sl_price < cisd.confirmation_close < entry_price:
                        return (False, f"SL behind CISD POI at {cisd.confirmation_close:.2f}")

            for fvg in smc_context.fvgs_m2 + smc_context.fvgs_h1:
                if fvg.status == "active" and fvg.direction == "bearish":
                    if fvg.low < entry_price and sl_price < fvg.low:
                        return (True, None)

            for f_low in smc_context.active_fractal_lows:
                if f_low < entry_price and sl_price < f_low:
                    return (True, None)

        else:  # short
            for fvg in smc_context.fvgs_m2 + smc_context.fvgs_h1:
                if fvg.status == "active" and fvg.direction == "bearish":
                    if entry_price < fvg.midpoint < sl_price:
                        return (False, f"SL behind bearish FVG POI at {fvg.midpoint:.2f}")

            for cisd in smc_context.cisd_events:
                if cisd.direction == "short":
                    if entry_price < cisd.confirmation_close < sl_price:
                        return (False, f"SL behind CISD POI at {cisd.confirmation_close:.2f}")

            for fvg in smc_context.fvgs_m2 + smc_context.fvgs_h1:
                if fvg.status == "active" and fvg.direction == "bullish":
                    if fvg.high > entry_price and sl_price > fvg.high:
                        return (True, None)

            for f_high in smc_context.active_fractal_highs:
                if f_high > entry_price and sl_price > f_high:
                    return (True, None)

        return (True, "SL placement unclear (no invalidation zone identified)")

    def validate_be_move(
        self,
        current_bar: pd.Series,
        entry_time: datetime,
        entry_price: float,
        direction: str,
        smc_context: SMCDayContext,
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate if moving to breakeven is appropriate.

        7 conditions (5 implemented, 2 deferred to Phase 6: LQ zone, SMT).
        """
        # Check 7 first: NEVER if BE point (entry price) is in potential POI
        for fvg in smc_context.fvgs_m2 + smc_context.fvgs_h1:
            if fvg.status == "active":
                if fvg.low <= entry_price <= fvg.high:
                    return (False, "BE point is inside active FVG (potential POI)")

        # Check 1: Strong FTA reaction (large wick relative to body)
        body_size = abs(current_bar["close"] - current_bar["open"])
        upper_wick = current_bar["high"] - max(current_bar["open"], current_bar["close"])
        lower_wick = min(current_bar["open"], current_bar["close"]) - current_bar["low"]
        wick_size = max(upper_wick, lower_wick)

        if body_size > 0 and wick_size / body_size > 2.0:
            return (True, "Strong FTA reaction (large wick)")

        # Check 3: Idea invalidation (price tested opposing zone)
        if direction == "long":
            for fvg in smc_context.fvgs_m2 + smc_context.fvgs_h1:
                if fvg.status == "active" and fvg.direction == "bearish":
                    if current_bar["high"] >= fvg.low and current_bar["high"] <= fvg.high:
                        return (True, "Idea invalidation: tested opposing FVG")
        else:
            for fvg in smc_context.fvgs_m2 + smc_context.fvgs_h1:
                if fvg.status == "active" and fvg.direction == "bullish":
                    if current_bar["low"] <= fvg.high and current_bar["low"] >= fvg.low:
                        return (True, "Idea invalidation: tested opposing FVG")

        # Check 6: Structural BE via LTF structure close (BOS after entry)
        for bos in smc_context.bos_events:
            if bos.break_time > entry_time:
                return (True, f"LTF structure close at {bos.break_time}")

        return (False, None)
