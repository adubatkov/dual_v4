"""
TimeManager - Virtual clock and SL/TP management.

Manages time progression in backtest and checks for SL/TP hits.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, List, Callable, Tuple
from dataclasses import dataclass

import pandas as pd

from .mt5_emulator import MT5Emulator
from .models import Position, PositionType, TradeLog

logger = logging.getLogger(__name__)


@dataclass
class SLTPCheckResult:
    """Result of SL/TP check for a position."""
    ticket: int
    hit_type: str  # 'sl', 'tp', or None
    hit_price: float
    hit_time: datetime


class TimeManager:
    """
    Manages virtual time progression and SL/TP checking.

    Responsibilities:
        - Advance time by tick, candle, or arbitrary jumps
        - Check SL/TP levels against price during time intervals
        - Trigger position closes when SL/TP hit
    """

    def __init__(self, emulator: MT5Emulator):
        """
        Initialize TimeManager.

        Args:
            emulator: MT5Emulator instance to control
        """
        self.emulator = emulator
        self.current_time: Optional[datetime] = None
        self._tick_interval = timedelta(seconds=5)
        self._candle_interval = timedelta(minutes=2)  # M2 timeframe

        # Callbacks
        self._on_sltp_hit: Optional[Callable[[SLTPCheckResult], None]] = None

    def set_start_time(self, start_time: datetime) -> None:
        """
        Set initial time for backtest.

        Args:
            start_time: Starting datetime
        """
        self.current_time = start_time
        self.emulator.set_time(start_time)
        logger.info(f"TimeManager start time set to {start_time}")

    def tick(self) -> datetime:
        """
        Advance time by one tick interval (5 seconds).

        Returns:
            New current time
        """
        if self.current_time is None:
            raise ValueError("Start time not set. Call set_start_time first.")

        self.current_time += self._tick_interval
        self.emulator.set_time(self.current_time)

        # Check SL/TP for open positions
        self._check_all_sltp()

        return self.current_time

    def next_candle(self, timeframe_minutes: int = 2) -> datetime:
        """
        Advance time to next candle close.

        Args:
            timeframe_minutes: Candle timeframe in minutes

        Returns:
            New current time (at candle close)
        """
        if self.current_time is None:
            raise ValueError("Start time not set. Call set_start_time first.")

        # Calculate next candle boundary
        interval = timedelta(minutes=timeframe_minutes)

        # Round up to next interval
        minutes_since_midnight = (
            self.current_time.hour * 60 +
            self.current_time.minute +
            self.current_time.second / 60
        )
        current_interval = int(minutes_since_midnight // timeframe_minutes)
        next_interval = current_interval + 1
        next_minutes = next_interval * timeframe_minutes

        next_hour = next_minutes // 60
        next_minute = next_minutes % 60

        # Handle day rollover
        if next_hour >= 24:
            next_day = self.current_time.date() + timedelta(days=1)
            next_hour = next_hour % 24
            self.current_time = datetime.combine(
                next_day,
                datetime.min.time().replace(hour=next_hour, minute=next_minute),
            )
            if self.current_time.tzinfo is None and hasattr(self.emulator.current_time, 'tzinfo'):
                self.current_time = self.current_time.replace(tzinfo=self.emulator.current_time.tzinfo)
        else:
            self.current_time = self.current_time.replace(
                hour=next_hour,
                minute=next_minute,
                second=0,
                microsecond=0,
            )

        self.emulator.set_time(self.current_time)

        # Check SL/TP
        self._check_all_sltp()

        return self.current_time

    def jump_to(self, target_time: datetime) -> datetime:
        """
        Jump directly to a specific time.

        WARNING: Does not check SL/TP for intermediate time.
        Use advance_to() for SL/TP checking during jumps.

        Args:
            target_time: Target datetime

        Returns:
            New current time
        """
        if self.current_time and target_time < self.current_time:
            logger.warning(f"Jumping backwards in time: {self.current_time} -> {target_time}")

        self.current_time = target_time
        self.emulator.set_time(target_time)

        return self.current_time

    def advance_to(
        self,
        target_time: datetime,
        check_sltp: bool = True,
        tick_data: Optional[pd.DataFrame] = None,
    ) -> List[SLTPCheckResult]:
        """
        Advance to target time while checking SL/TP.

        Args:
            target_time: Target datetime to reach
            check_sltp: Whether to check SL/TP (requires tick_data)
            tick_data: Tick data for price checking

        Returns:
            List of SL/TP hits that occurred during advance
        """
        if self.current_time is None:
            raise ValueError("Start time not set. Call set_start_time first.")

        hits = []

        if not check_sltp or tick_data is None:
            self.jump_to(target_time)
            return hits

        # Filter tick data for the time range
        mask = (tick_data["time"] > self.current_time) & (tick_data["time"] <= target_time)
        relevant_ticks = tick_data[mask].sort_values("time")

        # Process each tick
        for _, tick_row in relevant_ticks.iterrows():
            self.current_time = tick_row["time"]
            if hasattr(tick_row["time"], "to_pydatetime"):
                self.current_time = tick_row["time"].to_pydatetime()
            self.emulator.set_time(self.current_time)

            # Check SL/TP against this tick's high/low
            tick_hits = self._check_all_sltp_with_price(
                bid=tick_row["bid"],
                ask=tick_row["ask"],
            )
            hits.extend(tick_hits)

        # Ensure we end at target time
        if self.current_time != target_time:
            self.current_time = target_time
            self.emulator.set_time(target_time)

        return hits

    def advance_with_candle_check(
        self,
        candle_high: float,
        candle_low: float,
        candle_time: datetime,
    ) -> List[SLTPCheckResult]:
        """
        Advance time and check SL/TP against candle high/low.

        Used for faster backtesting when tick data is not available.
        Checks if price touched SL/TP during the candle.

        Args:
            candle_high: Candle high price
            candle_low: Candle low price
            candle_time: Candle timestamp

        Returns:
            List of SL/TP hits
        """
        self.jump_to(candle_time)

        hits = []
        positions = self.emulator.get_open_positions_list()

        for position in positions:
            hit = self._check_sltp_for_candle(
                position, candle_high, candle_low, candle_time
            )
            if hit:
                hits.append(hit)
                # Close position at SL/TP price
                self.emulator.close_position_by_ticket(
                    position.ticket,
                    price=hit.hit_price,
                    exit_reason=hit.hit_type,
                )

        return hits

    def _check_all_sltp(self) -> List[SLTPCheckResult]:
        """
        Check SL/TP for all open positions at current price.

        Returns:
            List of SL/TP hits
        """
        hits = []
        positions = self.emulator.get_open_positions_list()

        for position in positions:
            tick = self.emulator.symbol_info_tick(position.symbol)
            if not tick:
                continue

            hit = self._check_sltp_for_position(position, tick.bid, tick.ask)
            if hit:
                hits.append(hit)
                # Close position
                self.emulator.close_position_by_ticket(
                    position.ticket,
                    exit_reason=hit.hit_type,
                )

        return hits

    def _check_all_sltp_with_price(
        self, bid: float, ask: float
    ) -> List[SLTPCheckResult]:
        """
        Check SL/TP for all positions against specific bid/ask.

        Args:
            bid: Bid price
            ask: Ask price

        Returns:
            List of SL/TP hits
        """
        hits = []
        positions = self.emulator.get_open_positions_list()

        for position in positions:
            hit = self._check_sltp_for_position(position, bid, ask)
            if hit:
                hits.append(hit)
                self.emulator.close_position_by_ticket(
                    position.ticket,
                    price=hit.hit_price,
                    exit_reason=hit.hit_type,
                )

        return hits

    def _check_sltp_for_position(
        self,
        position: Position,
        bid: float,
        ask: float,
    ) -> Optional[SLTPCheckResult]:
        """
        Check if position's SL or TP was hit.

        Args:
            position: Position to check
            bid: Current bid price
            ask: Current ask price

        Returns:
            SLTPCheckResult if hit, None otherwise
        """
        sl = position.sl
        tp = position.tp

        if position.type == PositionType.POSITION_TYPE_BUY:
            # Long position: closes at bid price
            close_price = bid

            # SL hit if bid <= SL
            if sl > 0 and close_price <= sl:
                return SLTPCheckResult(
                    ticket=position.ticket,
                    hit_type="sl",
                    hit_price=sl,  # Execute at SL price
                    hit_time=self.current_time,
                )

            # TP hit if bid >= TP
            if tp > 0 and close_price >= tp:
                return SLTPCheckResult(
                    ticket=position.ticket,
                    hit_type="tp",
                    hit_price=tp,  # Execute at TP price
                    hit_time=self.current_time,
                )

        else:  # POSITION_TYPE_SELL
            # Short position: closes at ask price
            close_price = ask

            # SL hit if ask >= SL
            if sl > 0 and close_price >= sl:
                return SLTPCheckResult(
                    ticket=position.ticket,
                    hit_type="sl",
                    hit_price=sl,
                    hit_time=self.current_time,
                )

            # TP hit if ask <= TP
            if tp > 0 and close_price <= tp:
                return SLTPCheckResult(
                    ticket=position.ticket,
                    hit_type="tp",
                    hit_price=tp,
                    hit_time=self.current_time,
                )

        return None

    def _check_sltp_for_candle(
        self,
        position: Position,
        candle_high: float,
        candle_low: float,
        candle_time: datetime,
    ) -> Optional[SLTPCheckResult]:
        """
        Check if position's SL or TP was hit during a candle.

        Determines which was hit first based on position direction
        and candle shape.

        Args:
            position: Position to check
            candle_high: Candle high price
            candle_low: Candle low price
            candle_time: Candle timestamp

        Returns:
            SLTPCheckResult if hit, None otherwise
        """
        sl = position.sl
        tp = position.tp

        if position.type == PositionType.POSITION_TYPE_BUY:
            # Long: SL below entry, TP above entry
            sl_hit = sl > 0 and candle_low <= sl
            tp_hit = tp > 0 and candle_high >= tp

            if sl_hit and tp_hit:
                # Both hit - assume SL hit first (conservative)
                # In reality would need tick data to determine order
                return SLTPCheckResult(
                    ticket=position.ticket,
                    hit_type="sl",
                    hit_price=sl,
                    hit_time=candle_time,
                )
            elif sl_hit:
                return SLTPCheckResult(
                    ticket=position.ticket,
                    hit_type="sl",
                    hit_price=sl,
                    hit_time=candle_time,
                )
            elif tp_hit:
                return SLTPCheckResult(
                    ticket=position.ticket,
                    hit_type="tp",
                    hit_price=tp,
                    hit_time=candle_time,
                )

        else:  # Short
            # Short: SL above entry, TP below entry
            sl_hit = sl > 0 and candle_high >= sl
            tp_hit = tp > 0 and candle_low <= tp

            if sl_hit and tp_hit:
                # Both hit - assume SL hit first (conservative)
                return SLTPCheckResult(
                    ticket=position.ticket,
                    hit_type="sl",
                    hit_price=sl,
                    hit_time=candle_time,
                )
            elif sl_hit:
                return SLTPCheckResult(
                    ticket=position.ticket,
                    hit_type="sl",
                    hit_price=sl,
                    hit_time=candle_time,
                )
            elif tp_hit:
                return SLTPCheckResult(
                    ticket=position.ticket,
                    hit_type="tp",
                    hit_price=tp,
                    hit_time=candle_time,
                )

        return None

    def set_sltp_callback(
        self, callback: Callable[[SLTPCheckResult], None]
    ) -> None:
        """
        Set callback for SL/TP hit events.

        Args:
            callback: Function to call when SL/TP is hit
        """
        self._on_sltp_hit = callback

    def get_current_time(self) -> Optional[datetime]:
        """Get current virtual time."""
        return self.current_time

    def is_market_open(
        self,
        symbol: str,
        time: Optional[datetime] = None,
    ) -> bool:
        """
        Check if market is open for a symbol.

        Simple implementation - can be extended for specific market hours.

        Args:
            symbol: Symbol to check
            time: Time to check (uses current_time if None)

        Returns:
            True if market is open
        """
        check_time = time or self.current_time
        if check_time is None:
            return False

        # Skip weekends
        weekday = check_time.weekday()
        if weekday >= 5:  # Saturday = 5, Sunday = 6
            return False

        # For simplicity, assume markets are open during weekdays
        # Can be extended with specific market hours per symbol
        return True
