"""
IB Strategy - Finite State Machine implementation

Implements the Initial Balance strategy with 4 signal variations:
- Reverse (REV)
- OCAE (Open Close After Equilibrium)
- TCWE (Two Candles Without Equilibrium)
- REV_RB (Reverse Blocked - limit orders)

FSM States:
- AWAITING_IB_CALCULATION
- AWAITING_TRADE_WINDOW
- IN_TRADE_WINDOW
- POSITION_OPEN
- DAY_ENDED
"""
import logging
from datetime import datetime, date, time as datetime_time, timedelta
from typing import Optional, Dict, Any
import pytz
import pandas as pd
import MetaTrader5 as mt5

from .base_strategy import BaseStrategy, Signal
from src.news_filter import NewsFilter
from src.utils.strategy_logic import (
    compute_ib,
    process_reverse_signal_fixed,
    eq_touched_before_idx,
    first_breakout_bar,
    tcwe_second_further_idx,
    find_cisd_level,
    initial_stop_price,
    place_sl_tp_with_min_size,
    simulate_after_entry,
    simulate_reverse_limit_both_sides,
    ib_window_on_date,
    trade_window_on_date,
    get_trade_window
)

logger = logging.getLogger(__name__)


class IBStrategy(BaseStrategy):
    """
    Initial Balance Strategy with FSM implementation

    This strategy implements the complex IB logic as a Finite State Machine,
    "stretching" the batch processing logic from the backtest into real-time execution.
    """

    def __init__(self, symbol: str, params: Dict[str, Any], executor, magic_number: int, strategy_label: str = "",
                 news_filter_enabled: bool = False):
        """
        Initialize IB Strategy

        Args:
            symbol: Trading symbol
            params: Strategy parameters (GER40_PARAMS_PROD or XAUUSD_PARAMS_PROD)
            executor: MT5Executor instance
            magic_number: Unique magic number
            strategy_label: Optional label for logging (e.g., "IB09:00-10:00")
            news_filter_enabled: Enable news filter for 5ers compliance
        """
        super().__init__(symbol, magic_number)

        self.params = params
        self.executor = executor

        # News filter for 5ers compliance
        self.news_filter: Optional[NewsFilter] = None
        self._news_filtered_count: int = 0
        if news_filter_enabled:
            try:
                self.news_filter = NewsFilter(symbol=symbol)
                logger.info(f"[{symbol}] News filter enabled with {self.news_filter.event_count} events")
            except Exception as e:
                logger.warning(f"[{symbol}] Failed to initialize news filter: {e}")

        # Timezone and IB parameters (assuming same across variations)
        self.ib_tz = pytz.timezone(params["Reverse"]["IB_TZ"])
        self.ib_start = params["Reverse"]["IB_START"]
        self.ib_end = params["Reverse"]["IB_END"]

        # Generate logging prefix
        if strategy_label:
            self.log_prefix = f"[{symbol}_M{magic_number}_{strategy_label}]"
        else:
            self.log_prefix = f"[{symbol}_M{magic_number}_IB{self.ib_start}-{self.ib_end}]"

        # FSM state
        self.state = "AWAITING_IB_CALCULATION"
        self.current_local_date: Optional[date] = None

        # IB values (calculated at IB_END)
        self.ibh: Optional[float] = None
        self.ibl: Optional[float] = None
        self.eq: Optional[float] = None

        # Trade window
        self.trade_window_end: Optional[datetime] = None

        # TSL tracking for open positions
        self.tsl_state: Optional[Dict[str, Any]] = None

        logger.info(f"{self.log_prefix} IBStrategy initialized (TZ: {self.ib_tz})")

    def reset_daily_state(self) -> None:
        """Reset all daily state variables"""
        self.state = "AWAITING_IB_CALCULATION"
        self.ibh = None
        self.ibl = None
        self.eq = None
        self.trade_window_end = None
        self.tsl_state = None
        # Don't reset _news_filtered_count - it's a cumulative counter
        logger.info(f"{self.log_prefix} Daily state reset")

    def get_news_filter_stats(self) -> Dict[str, Any]:
        """Get news filter statistics for logging/monitoring."""
        return {
            "news_filter_enabled": self.news_filter is not None,
            "trades_blocked_by_news": self._news_filtered_count,
            "events_loaded": self.news_filter.event_count if self.news_filter else 0,
        }

    def check_signal(self, current_time_utc: datetime) -> Optional[Signal]:
        """
        FSM logic for signal detection

        Implements the state machine as described in Architecture Spec Section 4.2

        States flow:
        AWAITING_IB_CALCULATION -> AWAITING_TRADE_WINDOW -> IN_TRADE_WINDOW -> DAY_ENDED

        Args:
            current_time_utc: Current UTC time

        Returns:
            Signal if detected, None otherwise
        """
        local_time = current_time_utc.astimezone(self.ib_tz)

        # 0. Check for new day and reset state
        if self.current_local_date is None or local_time.date() != self.current_local_date:
            self.reset_daily_state()
            self.current_local_date = local_time.date()
            logger.info(f"{self.log_prefix} New day {self.current_local_date}, state reset")

        # 1. FSM State: AWAITING_IB_CALCULATION
        if self.state == "AWAITING_IB_CALCULATION":
            # Check if IB period ended
            ib_end_time = datetime_time.fromisoformat(self.ib_end)

            if local_time.time() >= ib_end_time:
                logger.info(f"{self.log_prefix} IB period ended, calculating IB...")

                # Get bars for the entire day up to now (M2 timeframe)
                bars = self.executor.get_bars(self.symbol, "M2", 500)

                if bars is not None and not bars.empty:
                    # Calculate IB
                    ib = compute_ib(bars, local_time.date(), self.ib_start, self.ib_end, str(self.ib_tz))

                    if ib:
                        self.ibh = ib["IBH"]
                        self.ibl = ib["IBL"]
                        self.eq = ib["EQ"]
                        # Use more decimal places for Forex pairs (5 decimals)
                        logger.info(f"{self.log_prefix} IB calculated - H:{self.ibh:.5f}, L:{self.ibl:.5f}, EQ:{self.eq:.5f}")
                        self.state = "AWAITING_TRADE_WINDOW"
                    else:
                        logger.warning(f"{self.log_prefix} Failed to calculate IB")
                        self.state = "DAY_ENDED"
                else:
                    logger.warning(f"{self.log_prefix} No bars available for IB calculation")

        # 2. FSM State: AWAITING_TRADE_WINDOW
        if self.state == "AWAITING_TRADE_WINDOW":
            # For now, we'll use Reverse variation params for trade window timing
            reverse_params = self.params["Reverse"]

            # Calculate trade window start time: IB_END + IB_WAIT
            ib_end_time = datetime_time.fromisoformat(reverse_params["IB_END"])
            ib_wait_minutes = reverse_params["IB_WAIT"]

            # Convert to datetime and add wait period (in IB timezone, then convert to UTC)
            today_ib_end = datetime.combine(local_time.date(), ib_end_time)
            today_ib_end_local = self.ib_tz.localize(today_ib_end)
            trade_start_time_local = today_ib_end_local + timedelta(minutes=ib_wait_minutes)
            # Convert to UTC for consistent comparisons
            trade_start_time = trade_start_time_local.astimezone(pytz.utc)

            if current_time_utc >= trade_start_time:
                # Store trade window start time for variation-specific window calculation
                self.trade_window_start = trade_start_time_local
                # Will calculate trade_window_end when signal is detected (variation-specific)
                # For now, use MAX window just for checking if window expired for signal detection
                max_window = max(
                    self.params["Reverse"]["TRADE_WINDOW"],
                    self.params["OCAE"]["TRADE_WINDOW"],
                    self.params["TCWE"]["TRADE_WINDOW"],
                    self.params.get("REV_RB", {}).get("TRADE_WINDOW", 0)
                )
                # Calculate end time in local timezone first, then convert to UTC
                trade_window_end_local = trade_start_time_local + timedelta(minutes=max_window)
                # Store as UTC time for signal detection timeout
                self.trade_window_end = trade_window_end_local.astimezone(pytz.utc)

                # Log times in both UTC and local timezone for clarity
                logger.info(f"{self.log_prefix} Trade window opened for signal detection until {trade_window_end_local.strftime('%H:%M:%S')} ({self.ib_tz})")
                logger.info(f"{self.log_prefix} Trade window start: {trade_start_time_local.strftime('%H:%M:%S')} ({self.ib_tz}) / {trade_start_time.strftime('%H:%M:%S UTC')}")
                logger.info(f"{self.log_prefix} Signal detection end: {trade_window_end_local.strftime('%H:%M:%S')} ({self.ib_tz}) / {self.trade_window_end.strftime('%H:%M:%S UTC')}")
                logger.info(f"{self.log_prefix} Note: Position window will be set based on variation when signal is detected")
                self.state = "IN_TRADE_WINDOW"
                # CRITICAL FIX: Immediately check for signals after opening trade window
                # Don't wait for next poll cycle - this prevents 5-15 second delays

        # 3. FSM State: IN_TRADE_WINDOW
        # NOTE: Using 'if' instead of 'elif' allows immediate signal checking
        # after state transition from AWAITING_TRADE_WINDOW to IN_TRADE_WINDOW
        if self.state == "IN_TRADE_WINDOW":
            # Check if window expired
            if current_time_utc >= self.trade_window_end:
                logger.info(f"{self.log_prefix} Trade window closed, no signal found")
                self.state = "DAY_ENDED"
                return None

            # NEWS FILTER CHECK (5ers compliance)
            # Block trades 2 minutes before/after high-impact news
            if self.news_filter is not None:
                allowed, blocking_event = self.news_filter.is_trade_allowed(current_time_utc)
                if not allowed:
                    self._news_filtered_count += 1
                    logger.info(f"{self.log_prefix} Trade blocked by news filter: {blocking_event.title} ({blocking_event.country})")
                    logger.info(f"  Event time: {blocking_event.datetime_utc.strftime('%Y-%m-%d %H:%M:%S UTC')}")
                    return None

            # Get current bars
            bars = self.executor.get_bars(self.symbol, "M2", 500)

            if bars is None or bars.empty:
                return None

            # Try each variation in priority order: Reverse -> OCAE -> TCWE -> REV_RB

            # 1. Try Reverse
            signal = self._check_reverse_signal(bars, local_time.date(), current_time_utc)
            if signal:
                return signal

            # 2. Try OCAE
            signal = self._check_ocae_signal(bars, local_time.date(), current_time_utc)
            if signal:
                return signal

            # 3. Try TCWE
            signal = self._check_tcwe_signal(bars, local_time.date(), current_time_utc)
            if signal:
                return signal

            # 4. Try REV_RB (if enabled)
            if self.params.get("REV_RB", {}).get("REV_RB_ENABLED", False):
                signal = self._check_rev_rb_signal(bars, local_time.date(), current_time_utc)
                if signal:
                    return signal

        return None

    def _is_last_candle_closed(self, df_trade: pd.DataFrame, current_time_utc: datetime) -> bool:
        """
        Check if the last candle in df_trade is fully formed (closed)

        CRITICAL: In live trading, we must wait for candle to close before checking signal.
        Entering during candle formation (intrabar) leads to false signals.

        Args:
            df_trade: DataFrame with OHLC data (M2 timeframe)
            current_time_utc: Current UTC time

        Returns:
            True if last candle is closed (fully formed), False if still forming
        """
        if df_trade.empty:
            return False

        last_candle_time = df_trade["time"].iloc[-1]

        # M2 candles - next candle starts 2 minutes later
        next_candle_time = last_candle_time + timedelta(minutes=2)

        # Last candle is closed if current time >= next candle time
        is_closed = current_time_utc >= next_candle_time

        return is_closed

    def _check_reverse_signal(self, bars: pd.DataFrame, day_date: date, current_time_utc: datetime) -> Optional[Signal]:
        """Check for Reverse variation signal"""
        params = self.params["Reverse"]

        # Get trade window data
        df_trade = get_trade_window(bars, day_date, params["IB_END"], params["IB_TZ"],
                                     params["IB_WAIT"], params["TRADE_WINDOW"])

        if df_trade.empty:
            return None

        # Get context before trade window
        ib_start_ts, ib_end_ts = ib_window_on_date(day_date, params["IB_START"],
                                                   params["IB_END"], params["IB_TZ"])
        first_trade_ts = df_trade["time"].iat[0]
        df_pre_context = bars[
            (bars["time"] >= ib_start_ts) & (bars["time"] < first_trade_ts)
        ][["time", "open", "high", "low", "close"]].copy()

        # Get new parameters (with defaults for backward compatibility)
        ib_buffer_pct = params.get("IB_BUFFER_PCT", 0.0)

        # Process signal
        reverse_signal = process_reverse_signal_fixed(
            df_trade, self.ibh, self.ibl, self.eq, params, df_pre_context,
            ib_buffer_pct=ib_buffer_pct
        )

        if reverse_signal is None:
            return None

        cisd_idx = reverse_signal["cisd_idx"]
        direction = reverse_signal["direction"]
        sweep_extreme = reverse_signal["sweep_extreme"]

        # Entry on NEXT candle after CISD
        entry_idx = cisd_idx + 1

        if entry_idx >= len(df_trade):
            return None  # Entry candle not available yet

        # CRITICAL: Check if ENTRY CANDLE has started (opened)
        # For Reverse, we enter on OPEN of next candle after CISD
        entry_candle = df_trade.iloc[entry_idx]
        entry_candle_time = entry_candle["time"]

        if current_time_utc < entry_candle_time:
            # Entry candle hasn't started yet
            return None

        entry_price = float(entry_candle["open"])
        stop_price = float(sweep_extreme)

        # Calculate SL and TP
        stop, tp, adjusted = place_sl_tp_with_min_size(direction, entry_price, stop_price,
                                                       params["RR_TARGET"], params["MIN_SL_PCT"])

        logger.info(f"{self.log_prefix} Reverse signal - {direction.upper()} at {entry_price:.5f}, SL:{stop:.5f}, TP:{tp:.5f}")

        # Calculate variation-specific position window end
        variation_window = params["TRADE_WINDOW"]
        position_window_end_local = self.trade_window_start + timedelta(minutes=variation_window)
        position_window_end_utc = position_window_end_local.astimezone(pytz.utc)

        # Store TSL parameters for later use
        self.tsl_state = {
            "variation": "Reverse",
            "tsl_target": params["TSL_TARGET"],
            "tsl_sl": params["TSL_SL"],
            "initial_sl": stop,
            "initial_tp": tp,
            "current_tp": tp,  # Virtual TP for TSL logic
            "entry_price": entry_price,
            "tsl_triggered": False,
            "position_window_end": position_window_end_utc,
            "variation_window_minutes": variation_window,
            "tsl_history": [],  # List of (time, sl, tp) for chart visualization
        }

        logger.info(f"{self.log_prefix} Position window: {variation_window} min, closes at {position_window_end_local.strftime('%H:%M:%S')} ({self.ib_tz})")

        return Signal(
            direction=direction,
            entry_price=entry_price,
            stop_loss=stop,
            take_profit=tp,
            comment=f"IBStrategy_Reverse_{direction}",
            variation="Reverse",
            use_virtual_tp=(params["TSL_TARGET"] > 0)  # Use actual TP when TSL disabled
        )

    def _check_ocae_signal(self, bars: pd.DataFrame, day_date: date, current_time_utc: datetime) -> Optional[Signal]:
        """Check for OCAE (Open Close After Equilibrium) signal"""
        params = self.params["OCAE"]

        df_trade = get_trade_window(bars, day_date, params["IB_END"], params["IB_TZ"],
                                     params["IB_WAIT"], params["TRADE_WINDOW"])

        if df_trade.empty:
            return None

        # Find OCAE signal first
        trade_start_price = float(df_trade["open"].iat[0])

        # Get new parameters (with defaults for backward compatibility)
        ib_buffer_pct = params.get("IB_BUFFER_PCT", 0.0)
        max_distance_pct = params.get("MAX_DISTANCE_PCT", 1.0)

        br_result = first_breakout_bar(df_trade, self.ibh, self.ibl, ib_buffer_pct, max_distance_pct)

        if br_result is None:
            return None

        br_idx, direction = br_result

        if not eq_touched_before_idx(df_trade, self.eq, br_idx):
            return None

        entry_candle = df_trade.iloc[br_idx]

        # CRITICAL: Check if SIGNAL CANDLE is fully formed (closed)
        # We must wait for THIS candle to close, not the last candle in window
        signal_candle_time = entry_candle["time"]
        next_candle_time = signal_candle_time + timedelta(minutes=2)

        if current_time_utc < next_candle_time:
            # Signal candle not closed yet, wait
            return None

        entry_price = float(entry_candle["close"])

        # Calculate stop
        cisd_stop = find_cisd_level(df_trade, direction, br_idx, entry_price) if params[
                                                                                      "STOP_MODE"].lower() == "cisd" else None
        stop_price = initial_stop_price(trade_start_price, self.eq, cisd_stop, params["STOP_MODE"])
        stop, tp, adjusted = place_sl_tp_with_min_size(direction, entry_price, stop_price,
                                                       params["RR_TARGET"], params["MIN_SL_PCT"])

        logger.info(f"{self.log_prefix} OCAE signal - {direction.upper()} at {entry_price:.5f}, SL:{stop:.5f}, TP:{tp:.5f}")

        # Calculate variation-specific position window end
        variation_window = params["TRADE_WINDOW"]
        position_window_end_local = self.trade_window_start + timedelta(minutes=variation_window)
        position_window_end_utc = position_window_end_local.astimezone(pytz.utc)

        self.tsl_state = {
            "variation": "OCAE",
            "tsl_target": params["TSL_TARGET"],
            "tsl_sl": params["TSL_SL"],
            "initial_sl": stop,
            "initial_tp": tp,
            "current_tp": tp,  # Virtual TP for TSL logic
            "entry_price": entry_price,
            "tsl_triggered": False,
            "position_window_end": position_window_end_utc,
            "variation_window_minutes": variation_window,
            "tsl_history": [],  # List of (time, sl, tp) for chart visualization
        }

        logger.info(f"{self.log_prefix} Position window: {variation_window} min, closes at {position_window_end_local.strftime('%H:%M:%S')} ({self.ib_tz})")

        return Signal(
            direction=direction,
            entry_price=entry_price,
            stop_loss=stop,
            take_profit=tp,
            comment=f"IBStrategy_OCAE_{direction}",
            variation="OCAE",
            use_virtual_tp=(params["TSL_TARGET"] > 0)  # Use actual TP when TSL disabled
        )

    def _check_tcwe_signal(self, bars: pd.DataFrame, day_date: date, current_time_utc: datetime) -> Optional[Signal]:
        """Check for TCWE (Two Candles Without Equilibrium) signal"""
        params = self.params["TCWE"]

        df_trade = get_trade_window(bars, day_date, params["IB_END"], params["IB_TZ"],
                                     params["IB_WAIT"], params["TRADE_WINDOW"])

        if df_trade.empty:
            return None

        # Find TCWE signal first
        trade_start_price = float(df_trade["open"].iat[0])

        # Get new parameters (with defaults for backward compatibility)
        ib_buffer_pct = params.get("IB_BUFFER_PCT", 0.0)
        max_distance_pct = params.get("MAX_DISTANCE_PCT", 1.0)

        tc = tcwe_second_further_idx(df_trade, self.ibh, self.ibl, self.eq, ib_buffer_pct, max_distance_pct)

        if tc is None:
            return None

        idx2, direction = tc
        entry_candle = df_trade.iloc[idx2]

        # CRITICAL: Check if SIGNAL CANDLE is fully formed (closed)
        # We must wait for THIS candle to close, not the last candle in window
        signal_candle_time = entry_candle["time"]
        next_candle_time = signal_candle_time + timedelta(minutes=2)

        if current_time_utc < next_candle_time:
            # Signal candle not closed yet, wait
            return None

        entry_price = float(entry_candle["close"])

        cisd_stop = find_cisd_level(df_trade, direction, idx2, entry_price) if params[
                                                                                    "STOP_MODE"].lower() == "cisd" else None
        stop_price = initial_stop_price(trade_start_price, self.eq, cisd_stop, params["STOP_MODE"])
        stop, tp, adjusted = place_sl_tp_with_min_size(direction, entry_price, stop_price,
                                                       params["RR_TARGET"], params["MIN_SL_PCT"])

        logger.info(f"{self.log_prefix} TCWE signal - {direction.upper()} at {entry_price:.5f}, SL:{stop:.5f}, TP:{tp:.5f}")

        # Calculate variation-specific position window end
        variation_window = params["TRADE_WINDOW"]
        position_window_end_local = self.trade_window_start + timedelta(minutes=variation_window)
        position_window_end_utc = position_window_end_local.astimezone(pytz.utc)

        self.tsl_state = {
            "variation": "TCWE",
            "tsl_target": params["TSL_TARGET"],
            "tsl_sl": params["TSL_SL"],
            "initial_sl": stop,
            "initial_tp": tp,
            "current_tp": tp,  # Virtual TP for TSL logic
            "entry_price": entry_price,
            "tsl_triggered": False,
            "position_window_end": position_window_end_utc,
            "variation_window_minutes": variation_window,
            "tsl_history": [],  # List of (time, sl, tp) for chart visualization
        }

        logger.info(f"{self.log_prefix} Position window: {variation_window} min, closes at {position_window_end_local.strftime('%H:%M:%S')} ({self.ib_tz})")

        return Signal(
            direction=direction,
            entry_price=entry_price,
            stop_loss=stop,
            take_profit=tp,
            comment=f"IBStrategy_TCWE_{direction}",
            variation="TCWE",
            use_virtual_tp=(params["TSL_TARGET"] > 0)  # Use actual TP when TSL disabled
        )

    def _check_rev_rb_signal(self, bars: pd.DataFrame, day_date: date, current_time_utc: datetime) -> Optional[Signal]:
        """Check for REV_RB (Reverse Blocked - limit orders) signal"""
        params = self.params.get("REV_RB")
        if not params or not params.get("REV_RB_ENABLED"):
            return None

        df_trade = get_trade_window(bars, day_date, params["IB_END"], params["IB_TZ"],
                                     params["IB_WAIT"], params["TRADE_WINDOW"])

        if df_trade.empty:
            return None

        # CRITICAL: Check if last candle is fully formed (closed)
        # We must wait for candle to close before checking signal
        if not self._is_last_candle_closed(df_trade, current_time_utc):
            return None

        # For REV_RB, we use first trade window bar as fake_block_time
        # (as per original backtest logic)
        fake_block_time = df_trade["time"].iat[0]

        # Process REV_RB signal using simulate_reverse_limit_both_sides
        result = simulate_reverse_limit_both_sides(
            df_trade, self.ibh, self.ibl, self.eq, fake_block_time, params
        )

        if result is None or result.get("status") != "done":
            return None

        direction = result["direction"]
        entry_price = result["entry_price"]
        stop = result["stop"]
        tp = result["tp"]

        logger.info(f"{self.log_prefix} REV_RB signal - {direction.upper()} at {entry_price:.5f}, SL:{stop:.5f}, TP:{tp:.5f}")

        # Calculate variation-specific position window end
        variation_window = params["TRADE_WINDOW"]
        position_window_end_local = self.trade_window_start + timedelta(minutes=variation_window)
        position_window_end_utc = position_window_end_local.astimezone(pytz.utc)

        self.tsl_state = {
            "variation": "REV_RB",
            "tsl_target": params["TSL_TARGET"],
            "tsl_sl": params["TSL_SL"],
            "initial_sl": stop,
            "initial_tp": tp,
            "current_tp": tp,  # Virtual TP for TSL logic
            "entry_price": entry_price,
            "tsl_triggered": False,
            "position_window_end": position_window_end_utc,
            "variation_window_minutes": variation_window,
            "tsl_history": [],  # List of (time, sl, tp) for chart visualization
        }

        logger.info(f"{self.log_prefix} Position window: {variation_window} min, closes at {position_window_end_local.strftime('%H:%M:%S')} ({self.ib_tz})")

        return Signal(
            direction=direction,
            entry_price=entry_price,
            stop_loss=stop,
            take_profit=tp,
            comment=f"IBStrategy_REV_RB_{direction}",
            variation="REV_RB",
            use_virtual_tp=(params["TSL_TARGET"] > 0)  # Use actual TP when TSL disabled
        )

    def _restore_tsl_state_from_position(self, position: Any) -> None:
        """
        Restore TSL state from existing open position

        This is needed when bot restarts with open positions.
        We reconstruct TSL parameters based on position data.

        Args:
            position: MT5 position object
        """
        direction = "long" if position.type == mt5.POSITION_TYPE_BUY else "short"

        # Parse comment to determine variation
        comment = position.comment if hasattr(position, 'comment') else ""
        variation = None
        if "Reverse" in comment:
            variation = "Reverse"
        elif "OCAE" in comment:
            variation = "OCAE"
        elif "TCWE" in comment:
            variation = "TCWE"
        elif "REV_RB" in comment:
            variation = "REV_RB"

        if variation is None:
            logger.warning(f"{self.log_prefix} Cannot determine variation from comment: {comment}")
            # Default to Reverse
            variation = "Reverse"

        # Get TSL parameters for this variation
        params = self.params.get(variation, self.params["Reverse"])

        # CRITICAL ISSUE: Cannot reliably restore TSL state after bot restart!
        # Problem: MT5 only stores current SL, not original SL
        # After TSL triggers, current SL != initial SL
        # This makes it impossible to calculate correct initial_risk
        #
        # Example:
        #   Original: Entry=157.24, Initial SL=157.038, Initial Risk=0.202
        #   After TSL #2: Entry=157.24, Current SL=157.24 (breakeven)
        #   Calculated "risk": 157.24 - 157.24 = 0.0 ❌ WRONG!
        #
        # SOLUTION: Do NOT restore TSL state. Let position be managed by MT5 SL only.
        # Position will close at current SL level set by MT5.

        logger.warning(f"{self.log_prefix} Cannot restore TSL state reliably after bot restart")
        logger.warning(f"{self.log_prefix} Position {position.ticket} will be managed by MT5 SL only")
        logger.warning(f"{self.log_prefix} Current SL: {position.sl:.5f}, Entry: {position.price_open:.5f}")
        logger.warning(f"{self.log_prefix} TSL tracking disabled for this position")

        # Set a dummy tsl_state to prevent repeated restoration attempts
        # This signals that we already tried and failed to restore
        self.tsl_state = {
            "restoration_failed": True,
            "ticket": position.ticket,
            "note": "TSL state could not be restored after bot restart. Position managed by MT5 SL."
        }
        return

        # Try to calculate position window end
        # If we don't have trade_window_start, use a fallback
        if hasattr(self, 'trade_window_start') and self.trade_window_start is not None:
            variation_window = params["TRADE_WINDOW"]
            position_window_end_local = self.trade_window_start + timedelta(minutes=variation_window)
            position_window_end_utc = position_window_end_local.astimezone(pytz.utc)
        else:
            # Fallback: position stays open indefinitely (bot restart scenario)
            position_window_end_utc = None
            logger.warning(f"{self.log_prefix} Cannot determine position window end (bot restart?), position will not auto-close")

        self.tsl_state = {
            "variation": variation,
            "tsl_target": params["TSL_TARGET"],
            "tsl_sl": params["TSL_SL"],
            "initial_sl": current_sl,  # Use current SL as we don't know original
            "initial_tp": virtual_tp,
            "current_tp": virtual_tp,
            "entry_price": entry_price,
            "tsl_triggered": True,  # Assume TSL already active if position exists
            "position_window_end": position_window_end_utc,
            "variation_window_minutes": params.get("TRADE_WINDOW", 0)
        }

        logger.info(f"{self.log_prefix} TSL state restored from position {position.ticket} ({variation})")

    def update_position_state(self, position: Any, tick: dict, current_time_utc: datetime = None) -> None:
        """
        TSL (Trailing Stop Loss) logic

        Implements the trailing stop logic from simulate_after_entry in backtest.

        This method is called on every tick for open positions to:
        1. Check if TP was reached
        2. If yes, move SL and TP according to TSL parameters
        3. Update position via MT5

        Args:
            position: Open position object
            tick: Current tick data (dict with 'bid', 'ask', 'last')
            current_time_utc: Current UTC time (for backtest - emulator time, for live - uses real time if None)
        """
        # Use provided time or fall back to real time (for live trading)
        if current_time_utc is None:
            current_time_utc = datetime.now(pytz.utc)
        if self.tsl_state is None:
            # Try to restore TSL state from existing position
            logger.info(f"{self.log_prefix} No TSL state, attempting to restore from position {position.ticket}")
            self._restore_tsl_state_from_position(position)

            if self.tsl_state is None:
                logger.warning(f"{self.log_prefix} Failed to restore TSL state for position {position.ticket}")
                return

        # Check if restoration failed (dummy state set)
        if self.tsl_state.get("restoration_failed", False):
            # Don't spam logs - just silently skip TSL logic
            return

        # CRITICAL: Check if position window expired BEFORE TSL logic
        # This matches backtest behavior: close position at trade window end
        position_window_end = self.tsl_state.get("position_window_end")
        if position_window_end is not None:
            # current_time_utc is already set (parameter or real time)
            if current_time_utc >= position_window_end:
                logger.info(f"{self.log_prefix} Position window expired, closing position {position.ticket}")
                logger.info(f"  Window end: {position_window_end.strftime('%Y-%m-%d %H:%M:%S UTC')}")
                logger.info(f"  Current time: {current_time_utc.strftime('%Y-%m-%d %H:%M:%S UTC')}")

                # Close position at current market price
                result = self.executor.close_position(position.ticket)
                if result:
                    logger.info(f"{self.log_prefix} Position {position.ticket} closed at window end (exit_reason: time)")
                    self.tsl_state = None
                    return
                else:
                    logger.error(f"{self.log_prefix} Failed to close position {position.ticket} at window end")
                    # Continue with TSL logic anyway (position will still be protected by SL)

        # Extract TSL parameters
        tsl_target = self.tsl_state.get("tsl_target")
        tsl_sl = self.tsl_state.get("tsl_sl")

        # If no trailing (TSL_TARGET <= 0), do nothing
        if tsl_target is None or tsl_target <= 0:
            return

        # Get current price based on position direction
        direction = "long" if position.type == mt5.POSITION_TYPE_BUY else "short"
        current_price = tick.get("bid") if direction == "short" else tick.get("ask")

        if current_price is None:
            logger.warning(f"{self.log_prefix} No price in tick data")
            return

        # Get current position parameters
        current_sl = position.sl
        # Use VIRTUAL TP from tsl_state (not from position.tp which is 0)
        virtual_tp = self.tsl_state.get("current_tp")
        if virtual_tp is None or virtual_tp == 0:
            logger.warning(f"{self.log_prefix} No virtual TP in tsl_state")
            return

        entry_price = self.tsl_state["entry_price"]
        initial_sl = self.tsl_state["initial_sl"]

        # Calculate risk
        risk = abs(entry_price - initial_sl)
        if risk <= 0:
            logger.error(f"{self.log_prefix} Invalid risk calculation")
            return

        # Check if TP was hit
        tp_hit = False

        if direction == "long":
            # Log TP check for debugging
            logger.debug(f"{self.log_prefix} TSL Check LONG: price={current_price:.5f}, virtual_tp={virtual_tp:.5f}, hit={current_price >= virtual_tp}")

            # For long: check if price >= VIRTUAL TP
            if current_price >= virtual_tp:
                tp_hit = True
                logger.info(f"{self.log_prefix} LONG virtual TP hit at {current_price:.5f}, adjusting TSL...")

                # CORRECT TSL FORMULAS:
                # SL moves incrementally from current position
                # TP moves incrementally from current position

                # New SL = current SL + (TSL_SL * initial_risk)
                new_sl = current_sl + (tsl_sl * risk)
                # Ensure SL only moves up
                new_sl = max(current_sl, new_sl)

                # New virtual TP = current TP + (TSL_TARGET * initial_risk)
                new_virtual_tp = virtual_tp + (tsl_target * risk)

                # Log detailed TSL calculation
                logger.info(f"{self.log_prefix} TSL Calculation (LONG):")
                logger.info(f"  Entry Price:     {entry_price:.5f}")
                logger.info(f"  Initial Risk:    {risk:.5f} ({risk * 10000:.1f} pips)")
                logger.info(f"  Current SL:      {current_sl:.5f}")
                logger.info(f"  Virtual TP Hit:  {virtual_tp:.5f}")
                logger.info(f"  Current Price:   {current_price:.5f}")
                logger.info(f"  TSL_SL:          {tsl_sl}")
                logger.info(f"  TSL_TARGET:      {tsl_target}")
                logger.info(f"  New SL:          {new_sl:.5f} (moved +{(new_sl - current_sl) * 10000:.1f} pips)")
                logger.info(f"  New Virtual TP:  {new_virtual_tp:.5f} (moved +{(new_virtual_tp - virtual_tp) * 10000:.1f} pips)")

                # Update position: move SL, keep TP=0 (virtual)
                result = self.executor.modify_position(position.ticket, new_sl, 0.0)

                if result:
                    logger.info(f"{self.log_prefix} TSL updated successfully")
                    # Update virtual TP in tsl_state
                    self.tsl_state["current_tp"] = new_virtual_tp
                    self.tsl_state["tsl_triggered"] = True
                    # Record TSL change for chart visualization
                    if "tsl_history" not in self.tsl_state:
                        self.tsl_state["tsl_history"] = []
                    self.tsl_state["tsl_history"].append({
                        "time": current_time_utc,
                        "sl": new_sl,
                        "tp": new_virtual_tp,
                    })
                else:
                    logger.error(f"{self.log_prefix} Failed to update TSL")

        else:  # short
            # Log TP check for debugging
            logger.debug(f"{self.log_prefix} TSL Check SHORT: price={current_price:.5f}, virtual_tp={virtual_tp:.5f}, hit={current_price <= virtual_tp}")

            # For short: check if price <= VIRTUAL TP
            if current_price <= virtual_tp:
                tp_hit = True
                logger.info(f"{self.log_prefix} SHORT virtual TP hit at {current_price:.5f}, adjusting TSL...")

                # CORRECT TSL FORMULAS:
                # SL moves incrementally from current position
                # TP moves incrementally from current position

                # New SL = current SL - (TSL_SL * initial_risk)
                new_sl = current_sl - (tsl_sl * risk)
                # Ensure SL only moves down
                new_sl = min(current_sl, new_sl)

                # New virtual TP = current TP - (TSL_TARGET * initial_risk)
                new_virtual_tp = virtual_tp - (tsl_target * risk)

                # Log detailed TSL calculation
                logger.info(f"{self.log_prefix} TSL Calculation (SHORT):")
                logger.info(f"  Entry Price:     {entry_price:.5f}")
                logger.info(f"  Initial Risk:    {risk:.5f} ({risk * 10000:.1f} pips)")
                logger.info(f"  Current SL:      {current_sl:.5f}")
                logger.info(f"  Virtual TP Hit:  {virtual_tp:.5f}")
                logger.info(f"  Current Price:   {current_price:.5f}")
                logger.info(f"  TSL_SL:          {tsl_sl}")
                logger.info(f"  TSL_TARGET:      {tsl_target}")
                logger.info(f"  New SL:          {new_sl:.5f} (moved -{(current_sl - new_sl) * 10000:.1f} pips)")
                logger.info(f"  New Virtual TP:  {new_virtual_tp:.5f} (moved -{(virtual_tp - new_virtual_tp) * 10000:.1f} pips)")

                # Update position: move SL, keep TP=0 (virtual)
                result = self.executor.modify_position(position.ticket, new_sl, 0.0)

                if result:
                    logger.info(f"{self.log_prefix} TSL updated successfully")
                    # Update virtual TP in tsl_state
                    self.tsl_state["current_tp"] = new_virtual_tp
                    self.tsl_state["tsl_triggered"] = True
                    # Record TSL change for chart visualization
                    if "tsl_history" not in self.tsl_state:
                        self.tsl_state["tsl_history"] = []
                    self.tsl_state["tsl_history"].append({
                        "time": current_time_utc,
                        "sl": new_sl,
                        "tp": new_virtual_tp,
                    })
                else:
                    logger.error(f"{self.log_prefix} Failed to update TSL")
