"""
MT5Emulator - Mock implementation of MetaTrader5 API.

Provides a drop-in replacement for the MetaTrader5 library,
allowing the trading bot to run in backtesting mode without modifications.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any, Union
from pathlib import Path
import threading

import pandas as pd
import numpy as np

from .models import (
    TickData,
    SymbolInfo,
    AccountInfo,
    Position,
    Deal,
    Order,
    OrderResult,
    OrderSendRequest,
    TradeLog,
    OrderType,
    PositionType,
    TradeAction,
    TradeRetcode,
    Timeframe,
)
from ..config import BacktestConfig, SymbolConfig

logger = logging.getLogger(__name__)


class MT5Emulator:
    """
    Singleton MT5 Emulator.

    Emulates the MetaTrader5 Python API for backtesting purposes.
    Stores virtual terminal state including positions, account, and market data.

    Thread-safe singleton pattern ensures only one instance exists.
    """

    _instance: Optional["MT5Emulator"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "MT5Emulator":
        """Singleton pattern implementation."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize emulator state (only runs once due to singleton)."""
        if self._initialized:
            return

        # Configuration
        self.config: Optional[BacktestConfig] = None

        # Virtual time
        self.current_time: Optional[datetime] = None

        # Account state
        self._account: AccountInfo = AccountInfo()
        self._initial_balance: float = 50000.0

        # Market data
        self._tick_data: Dict[str, pd.DataFrame] = {}  # Symbol -> tick DataFrame
        self._m1_data: Dict[str, pd.DataFrame] = {}  # Symbol -> M1 DataFrame
        self._symbol_configs: Dict[str, SymbolConfig] = {}

        # Positions and orders
        self._positions: Dict[int, Position] = {}  # ticket -> Position
        self._orders_history: List[Order] = []
        self._deals_history: List[Deal] = []
        self._trade_log: List[TradeLog] = []

        # Internal state
        self._next_ticket: int = 1000000
        self._connected: bool = False

        self._initialized = True
        logger.info("MT5Emulator singleton initialized")

    def reset(self) -> None:
        """Reset emulator to initial state."""
        self.current_time = None
        self._account = AccountInfo(
            balance=self._initial_balance,
            equity=self._initial_balance,
            margin_free=self._initial_balance,
        )
        self._positions.clear()
        self._orders_history.clear()
        self._deals_history.clear()
        self._trade_log.clear()
        self._next_ticket = 1000000
        self._connected = False
        logger.info("MT5Emulator reset to initial state")

    def configure(
        self,
        config: BacktestConfig,
        tick_data: Optional[Dict[str, pd.DataFrame]] = None,
        m1_data: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> None:
        """
        Configure emulator with backtest settings and data.

        Args:
            config: Backtest configuration
            tick_data: Dict of symbol -> tick DataFrame
            m1_data: Dict of symbol -> M1 DataFrame
        """
        self.config = config
        self._initial_balance = config.initial_balance
        self._account.balance = config.initial_balance
        self._account.equity = config.initial_balance
        self._account.margin_free = config.initial_balance
        self._account.leverage = config.leverage

        if tick_data:
            self._tick_data = tick_data

        if m1_data:
            self._m1_data = m1_data

        self._symbol_configs = config.symbols
        self._connected = True  # Mark as connected after configuration

        logger.info(f"Emulator configured: balance={config.initial_balance}, "
                   f"symbols={list(self._symbol_configs.keys())}")

    def load_tick_data(self, symbol: str, data: pd.DataFrame) -> None:
        """
        Load tick data for a symbol.

        Args:
            symbol: Symbol name
            data: DataFrame with columns [time, bid, ask, mid]
        """
        # Ensure time is datetime with UTC
        if not pd.api.types.is_datetime64_any_dtype(data["time"]):
            data["time"] = pd.to_datetime(data["time"], utc=True)

        # Sort by time and set index for fast lookup
        data = data.sort_values("time").reset_index(drop=True)
        self._tick_data[symbol] = data

        logger.info(f"Loaded {len(data)} ticks for {symbol}: "
                   f"{data['time'].min()} to {data['time'].max()}")

    def load_m1_data(self, symbol: str, data: pd.DataFrame) -> None:
        """
        Load M1 candlestick data for a symbol.

        Args:
            symbol: Symbol name
            data: DataFrame with columns [time, open, high, low, close, volume]
        """
        # Ensure time is datetime with UTC
        if not pd.api.types.is_datetime64_any_dtype(data["time"]):
            data["time"] = pd.to_datetime(data["time"], utc=True)

        data = data.sort_values("time").reset_index(drop=True)
        self._m1_data[symbol] = data

        logger.info(f"Loaded {len(data)} M1 candles for {symbol}")

    def set_time(self, time: datetime) -> None:
        """
        Set virtual current time.

        Args:
            time: New current time
        """
        self.current_time = time

    # ============================================================
    # MT5 API Methods (mirror real MT5 API signatures)
    # ============================================================

    def initialize(self, path: str = "", login: int = 0, password: str = "",
                   server: str = "", timeout: int = 60000, portable: bool = False) -> bool:
        """
        Initialize MT5 connection (emulated).

        Returns:
            True always (emulation mode)
        """
        self._connected = True
        logger.debug("MT5Emulator: initialize() called")
        return True

    def shutdown(self) -> None:
        """Shutdown MT5 connection (emulated)."""
        self._connected = False
        logger.debug("MT5Emulator: shutdown() called")

    def login(self, login: int, password: str = "", server: str = "",
              timeout: int = 60000) -> bool:
        """
        Login to MT5 account (emulated).

        Returns:
            True always (emulation mode)
        """
        self._account.login = login
        self._account.server = server or "BacktestServer"
        self._connected = True
        logger.debug(f"MT5Emulator: login({login}, server={server})")
        return True

    def account_info(self) -> Optional[AccountInfo]:
        """
        Get account information.

        Returns:
            AccountInfo object with current account state
        """
        if not self._connected:
            return None

        # Update equity based on floating P&L
        floating_pnl = sum(p.profit for p in self._positions.values())
        self._account.equity = self._account.balance + floating_pnl
        self._account.profit = floating_pnl

        return self._account

    def symbol_info(self, symbol: str) -> Optional[SymbolInfo]:
        """
        Get symbol information.

        Args:
            symbol: Symbol name

        Returns:
            SymbolInfo object or None if not found
        """
        if symbol not in self._symbol_configs:
            logger.warning(f"Symbol not found: {symbol}")
            return None

        cfg = self._symbol_configs[symbol]
        return SymbolInfo(
            name=cfg.name,
            digits=cfg.digits,
            point=cfg.point,
            spread=int(cfg.spread_points / cfg.point),
            volume_min=cfg.volume_min,
            volume_max=cfg.volume_max,
            volume_step=cfg.volume_step,
            trade_tick_size=cfg.trade_tick_size,
            trade_tick_value=cfg.trade_tick_value,
            trade_contract_size=cfg.trade_contract_size,
            filling_mode=2,  # IOC
        )

    def symbol_info_tick(self, symbol: str) -> Optional[TickData]:
        """
        Get current tick for symbol.

        Uses current_time to find the appropriate tick from loaded data.
        If exact time not found, uses most recent tick before current_time.

        Args:
            symbol: Symbol name

        Returns:
            TickData object or None
        """
        if self.current_time is None:
            logger.error("current_time not set")
            return None

        if symbol not in self._tick_data:
            # Try to generate from M1 data
            if symbol in self._m1_data:
                return self._tick_from_m1(symbol)
            logger.warning(f"No tick data for {symbol}")
            return None

        df = self._tick_data[symbol]

        # Find tick at or before current_time (asof merge)
        mask = df["time"] <= self.current_time
        if not mask.any():
            logger.warning(f"No tick data before {self.current_time} for {symbol}")
            return None

        idx = mask.idxmax() if mask.iloc[-1] else df[mask].index[-1]
        row = df.iloc[idx] if mask.any() else df.iloc[0]

        # Get the most recent tick at or before current time
        valid_ticks = df[df["time"] <= self.current_time]
        if valid_ticks.empty:
            return None

        row = valid_ticks.iloc[-1]

        return TickData(
            time=row["time"].to_pydatetime() if hasattr(row["time"], "to_pydatetime") else row["time"],
            bid=float(row["bid"]),
            ask=float(row["ask"]),
            last=float(row.get("mid", (row["bid"] + row["ask"]) / 2)),
        )

    def _tick_from_m1(self, symbol: str) -> Optional[TickData]:
        """
        Generate tick from M1 data when tick data not available.

        Uses the last CLOSED candle (strictly before current_time).
        At 01:06:00, the candle starting at 01:06 is just opening,
        so we use close of M1 01:05 as the price reference.
        """
        if symbol not in self._m1_data:
            return None

        df = self._m1_data[symbol]
        # Use strictly < to get last CLOSED candle
        valid_candles = df[df["time"] < self.current_time]

        if valid_candles.empty:
            return None

        row = valid_candles.iloc[-1]
        cfg = self._symbol_configs.get(symbol)
        spread = cfg.spread_points if cfg else 0.0

        mid_price = float(row["close"])
        half_spread = spread / 2

        return TickData(
            time=self.current_time,
            bid=mid_price - half_spread,
            ask=mid_price + half_spread,
            last=mid_price,
        )

    def copy_rates_from_pos(
        self,
        symbol: str,
        timeframe: int,
        start_pos: int,
        count: int,
    ) -> Optional[np.ndarray]:
        """
        Get historical rates starting from position.

        IMPORTANT: Only returns data up to current_time (no future data).

        Args:
            symbol: Symbol name
            timeframe: Timeframe constant (e.g., TIMEFRAME_M1, TIMEFRAME_M2)
            start_pos: Starting position (0 = most recent)
            count: Number of bars to retrieve

        Returns:
            NumPy array with dtype matching MT5 format, or None
        """
        if self.current_time is None:
            logger.error("current_time not set - cannot get rates")
            return None

        if symbol not in self._m1_data:
            logger.warning(f"No M1 data for {symbol}")
            return None

        df = self._m1_data[symbol]

        # Filter to only data before or at current_time (prevent look-ahead)
        df_filtered = df[df["time"] <= self.current_time].copy()

        if df_filtered.empty:
            logger.warning(f"No data before {self.current_time} for {symbol}")
            return None

        # Resample to requested timeframe if needed
        if timeframe > 1:
            df_filtered = self._resample_to_timeframe(df_filtered, timeframe)

        # Get requested range from the end
        if start_pos > 0:
            df_filtered = df_filtered.iloc[:-start_pos] if start_pos < len(df_filtered) else pd.DataFrame()

        if df_filtered.empty:
            return None

        df_result = df_filtered.tail(count)

        # Convert to numpy structured array (MT5 format)
        dtype = np.dtype([
            ("time", "i8"),
            ("open", "f8"),
            ("high", "f8"),
            ("low", "f8"),
            ("close", "f8"),
            ("tick_volume", "i8"),
            ("spread", "i4"),
            ("real_volume", "i8"),
        ])

        result = np.zeros(len(df_result), dtype=dtype)
        result["time"] = df_result["time"].astype(np.int64) // 10**9  # Convert to timestamp
        result["open"] = df_result["open"].values
        result["high"] = df_result["high"].values
        result["low"] = df_result["low"].values
        result["close"] = df_result["close"].values
        # Get volume from available column (tick_volume or volume)
        if "tick_volume" in df_result.columns:
            result["tick_volume"] = df_result["tick_volume"].values
        elif "volume" in df_result.columns:
            result["tick_volume"] = df_result["volume"].values
        else:
            result["tick_volume"] = np.zeros(len(df_result), dtype=np.int64)

        return result

    def _resample_to_timeframe(self, df: pd.DataFrame, timeframe_minutes: int) -> pd.DataFrame:
        """Resample M1 data to higher timeframe."""
        df = df.set_index("time")

        agg_dict = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
        }

        # Add volume aggregation only if column exists
        if "volume" in df.columns:
            agg_dict["volume"] = "sum"
        elif "tick_volume" in df.columns:
            agg_dict["tick_volume"] = "sum"

        resampled = df.resample(f"{timeframe_minutes}min").agg(agg_dict).dropna()

        resampled = resampled.reset_index()
        return resampled

    def positions_get(
        self,
        symbol: Optional[str] = None,
        ticket: Optional[int] = None,
    ) -> Optional[tuple]:
        """
        Get open positions.

        Args:
            symbol: Filter by symbol (optional)
            ticket: Filter by ticket (optional)

        Returns:
            Tuple of Position objects or None
        """
        positions = list(self._positions.values())

        if symbol:
            positions = [p for p in positions if p.symbol == symbol]

        if ticket:
            positions = [p for p in positions if p.ticket == ticket]

        # Update current prices and profit
        for pos in positions:
            tick = self.symbol_info_tick(pos.symbol)
            if tick:
                pos.price_current = tick.bid if pos.type == PositionType.POSITION_TYPE_BUY else tick.ask
                pos.profit = self._calculate_profit(pos)

        return tuple(positions) if positions else None

    def _calculate_profit(self, position: Position) -> float:
        """Calculate floating profit for a position."""
        cfg = self._symbol_configs.get(position.symbol)
        if not cfg:
            return 0.0

        tick = self.symbol_info_tick(position.symbol)
        if not tick:
            return 0.0

        if position.type == PositionType.POSITION_TYPE_BUY:
            price_diff = tick.bid - position.price_open
        else:
            price_diff = position.price_open - tick.ask

        # Profit = (price_diff / tick_size) * tick_value * volume * contract_size
        ticks = price_diff / cfg.trade_tick_size
        profit = ticks * cfg.trade_tick_value * position.volume

        return round(profit, 2)

    def order_send(self, request: Union[Dict, OrderSendRequest]) -> OrderResult:
        """
        Send trading order.

        Supports:
        - TRADE_ACTION_DEAL: Market order (open/close position)
        - TRADE_ACTION_SLTP: Modify SL/TP

        Args:
            request: Order request dict or OrderSendRequest

        Returns:
            OrderResult with retcode and details
        """
        # Convert dict to OrderSendRequest if needed
        if isinstance(request, dict):
            req = OrderSendRequest(
                action=request.get("action", TradeAction.TRADE_ACTION_DEAL),
                symbol=request.get("symbol", ""),
                volume=request.get("volume", 0.0),
                type=request.get("type", OrderType.ORDER_TYPE_BUY),
                price=request.get("price", 0.0),
                sl=request.get("sl", 0.0),
                tp=request.get("tp", 0.0),
                deviation=request.get("deviation", 20),
                magic=request.get("magic", 0),
                comment=request.get("comment", ""),
                type_filling=request.get("type_filling", 1),
                position=request.get("position", 0),
                initial_tp=request.get("initial_tp", 0.0),
            )
        else:
            req = request

        # Route to appropriate handler
        if req.action == TradeAction.TRADE_ACTION_DEAL:
            if req.position > 0:
                # Close position
                return self._close_position(req)
            else:
                # Open position
                return self._open_position(req)
        elif req.action == TradeAction.TRADE_ACTION_SLTP:
            return self._modify_sltp(req)
        else:
            return OrderResult(
                retcode=TradeRetcode.TRADE_RETCODE_INVALID,
                comment=f"Unsupported action: {req.action}",
            )

    def _open_position(self, req: OrderSendRequest) -> OrderResult:
        """Open a new position."""
        symbol = req.symbol
        tick = self.symbol_info_tick(symbol)

        if not tick:
            return OrderResult(
                retcode=TradeRetcode.TRADE_RETCODE_PRICE_OFF,
                comment="No price available",
            )

        # Determine execution price
        if req.type == OrderType.ORDER_TYPE_BUY:
            exec_price = tick.ask
        else:
            exec_price = tick.bid

        # Check margin (simplified)
        cfg = self._symbol_configs.get(symbol)
        if cfg:
            required_margin = exec_price * req.volume * cfg.trade_contract_size / self._account.leverage
            if required_margin > self._account.margin_free:
                return OrderResult(
                    retcode=TradeRetcode.TRADE_RETCODE_NO_MONEY,
                    comment="Insufficient margin",
                )

        # Create position
        ticket = self._next_ticket
        self._next_ticket += 1

        position = Position(
            ticket=ticket,
            symbol=symbol,
            type=PositionType.POSITION_TYPE_BUY if req.type == OrderType.ORDER_TYPE_BUY else PositionType.POSITION_TYPE_SELL,
            volume=req.volume,
            price_open=exec_price,
            sl=req.sl,
            tp=req.tp,
            magic=req.magic,
            comment=req.comment,
            time=self.current_time,
            price_current=exec_price,
        )

        self._positions[ticket] = position

        # Update margin
        self._account.margin += required_margin if cfg else 0
        self._account.margin_free = self._account.equity - self._account.margin

        # Log deal
        deal = Deal(
            ticket=ticket,
            order=ticket,
            symbol=symbol,
            type=0 if req.type == OrderType.ORDER_TYPE_BUY else 1,
            volume=req.volume,
            price=exec_price,
            magic=req.magic,
            comment=req.comment,
            time=self.current_time,
            position_id=ticket,
            entry=0,  # DEAL_ENTRY_IN
        )
        self._deals_history.append(deal)

        # Create trade log entry
        direction = "long" if req.type == OrderType.ORDER_TYPE_BUY else "short"

        # Extract variation from comment (format: "Label_VARIATION")
        variation = None
        if req.comment and "_" in req.comment:
            parts = req.comment.split("_")
            if len(parts) >= 2:
                variation = parts[1]  # e.g., "Backtest_OCAE" -> "OCAE"

        # Use initial_tp for tracking if set, otherwise use tp
        tracking_tp = req.initial_tp if req.initial_tp > 0 else req.tp
        trade_log = TradeLog(
            ticket=ticket,
            symbol=symbol,
            direction=direction,
            entry_time=self.current_time,
            entry_price=exec_price,
            volume=req.volume,
            sl=req.sl,
            tp=tracking_tp,
            magic=req.magic,
            comment=req.comment,
            variation=variation,
        )
        self._trade_log.append(trade_log)

        logger.info(f"Position opened: {ticket} {symbol} {direction} {req.volume} @ {exec_price}")

        return OrderResult(
            retcode=TradeRetcode.TRADE_RETCODE_DONE,
            order=ticket,
            deal=ticket,
            volume=req.volume,
            price=exec_price,
            bid=tick.bid,
            ask=tick.ask,
            comment="Position opened",
        )

    def _close_position(self, req: OrderSendRequest) -> OrderResult:
        """Close an existing position."""
        ticket = req.position

        if ticket not in self._positions:
            return OrderResult(
                retcode=TradeRetcode.TRADE_RETCODE_INVALID,
                comment=f"Position {ticket} not found",
            )

        position = self._positions[ticket]
        tick = self.symbol_info_tick(position.symbol)

        if not tick:
            return OrderResult(
                retcode=TradeRetcode.TRADE_RETCODE_PRICE_OFF,
                comment="No price available",
            )

        # Determine close price
        if position.type == PositionType.POSITION_TYPE_BUY:
            close_price = tick.bid
        else:
            close_price = tick.ask

        # Calculate final profit
        profit = self._calculate_profit(position)

        # Update account
        self._account.balance += profit
        self._account.equity = self._account.balance

        # Release margin
        cfg = self._symbol_configs.get(position.symbol)
        if cfg:
            margin_release = position.price_open * position.volume * cfg.trade_contract_size / self._account.leverage
            self._account.margin -= margin_release
            self._account.margin_free = self._account.equity - self._account.margin

        # Log deal
        deal = Deal(
            ticket=self._next_ticket,
            order=self._next_ticket,
            symbol=position.symbol,
            type=1 if position.type == PositionType.POSITION_TYPE_BUY else 0,  # Opposite
            volume=position.volume,
            price=close_price,
            profit=profit,
            magic=position.magic,
            comment=req.comment,
            time=self.current_time,
            position_id=ticket,
            entry=1,  # DEAL_ENTRY_OUT
        )
        self._deals_history.append(deal)
        self._next_ticket += 1

        # Update trade log
        for log in self._trade_log:
            if log.ticket == ticket and log.exit_time is None:
                log.exit_time = self.current_time
                log.exit_price = close_price
                log.profit = profit
                log.exit_reason = req.comment or "closed"
                break

        # Remove position
        del self._positions[ticket]

        logger.info(f"Position closed: {ticket} @ {close_price}, profit={profit:.2f}")

        return OrderResult(
            retcode=TradeRetcode.TRADE_RETCODE_DONE,
            order=self._next_ticket - 1,
            deal=self._next_ticket - 1,
            volume=position.volume,
            price=close_price,
            bid=tick.bid,
            ask=tick.ask,
            comment=f"Position closed, profit={profit:.2f}",
        )

    def _modify_sltp(self, req: OrderSendRequest) -> OrderResult:
        """Modify SL/TP for a position."""
        ticket = req.position

        if ticket not in self._positions:
            return OrderResult(
                retcode=TradeRetcode.TRADE_RETCODE_INVALID,
                comment=f"Position {ticket} not found",
            )

        position = self._positions[ticket]
        position.sl = req.sl
        position.tp = req.tp

        logger.debug(f"Position {ticket} modified: SL={req.sl}, TP={req.tp}")

        return OrderResult(
            retcode=TradeRetcode.TRADE_RETCODE_DONE,
            comment="SL/TP modified",
        )

    def _close_position_at_price(
        self,
        position,  # Position object
        close_price: float,
        exit_reason: str,
    ) -> bool:
        """
        Close position at exact price (for SL/TP hits).

        This bypasses spread - when SL/TP is hit, we close at exact SL/TP price.
        """
        cfg = self._symbol_configs.get(position.symbol)
        if not cfg:
            return False

        # Calculate profit at exact close price (no spread adjustment)
        if position.type == PositionType.POSITION_TYPE_BUY:
            price_diff = close_price - position.price_open
        else:
            price_diff = position.price_open - close_price

        ticks = price_diff / cfg.trade_tick_size
        profit = round(ticks * cfg.trade_tick_value * position.volume, 2)

        # Update account
        self._account.balance += profit
        self._account.equity = self._account.balance

        # Release margin
        margin_release = position.price_open * position.volume * cfg.trade_contract_size / self._account.leverage
        self._account.margin -= margin_release
        self._account.margin_free = self._account.equity - self._account.margin

        # Log deal
        deal = Deal(
            ticket=self._next_ticket,
            order=self._next_ticket,
            symbol=position.symbol,
            type=1 if position.type == PositionType.POSITION_TYPE_BUY else 0,
            volume=position.volume,
            price=close_price,
            profit=profit,
            magic=position.magic,
            comment=exit_reason,
            time=self.current_time,
            position_id=position.ticket,
            entry=1,  # DEAL_ENTRY_OUT
        )
        self._deals_history.append(deal)
        self._next_ticket += 1

        # Update trade log
        for log in self._trade_log:
            if log.ticket == position.ticket and log.exit_time is None:
                log.exit_time = self.current_time
                log.exit_price = close_price
                log.profit = profit
                log.exit_reason = exit_reason
                break

        # Remove position
        del self._positions[position.ticket]

        logger.info(f"Position closed at exact price: {position.ticket} @ {close_price}, profit={profit:.2f}")

        return True

    def close_position_by_ticket(
        self,
        ticket: int,
        price: Optional[float] = None,
        exit_reason: str = "closed",
    ) -> bool:
        """
        Close position by ticket (helper method).

        Args:
            ticket: Position ticket
            price: Close price (uses exact price for SL/TP, current tick if None)
            exit_reason: Reason for closing

        Returns:
            True if closed successfully
        """
        if ticket not in self._positions:
            return False

        position = self._positions[ticket]

        # If exact price provided (SL/TP hit), calculate profit directly
        if price is not None:
            return self._close_position_at_price(position, price, exit_reason)

        # Otherwise use standard close via order_send
        close_type = OrderType.ORDER_TYPE_SELL if position.type == PositionType.POSITION_TYPE_BUY else OrderType.ORDER_TYPE_BUY

        request = OrderSendRequest(
            action=TradeAction.TRADE_ACTION_DEAL,
            symbol=position.symbol,
            volume=position.volume,
            type=close_type,
            position=ticket,
            comment=exit_reason,
        )

        result = self.order_send(request)
        return result.retcode == TradeRetcode.TRADE_RETCODE_DONE

    def last_error(self) -> tuple:
        """
        Get last error (emulated - always returns success).

        Returns:
            Tuple of (error_code, error_description)
        """
        return (1, "Success")

    def history_deals_get(
        self,
        date_from: Union[datetime, int],
        date_to: Union[datetime, int],
    ) -> Optional[tuple]:
        """
        Get deals history.

        Args:
            date_from: Start date/timestamp
            date_to: End date/timestamp

        Returns:
            Tuple of Deal objects
        """
        if isinstance(date_from, int):
            date_from = datetime.fromtimestamp(date_from)
        if isinstance(date_to, int):
            date_to = datetime.fromtimestamp(date_to)

        deals = [
            d for d in self._deals_history
            if d.time and date_from <= d.time <= date_to
        ]

        return tuple(deals) if deals else None

    # ============================================================
    # Backtest-specific methods
    # ============================================================

    def get_trade_log(self) -> List[TradeLog]:
        """Get complete trade log for analysis."""
        return self._trade_log.copy()

    def get_open_positions_list(self) -> List[Position]:
        """Get list of open positions."""
        return list(self._positions.values())

    def force_close_all_positions(self, reason: str = "backtest_end") -> None:
        """Force close all open positions (for backtest end)."""
        tickets = list(self._positions.keys())
        for ticket in tickets:
            self.close_position_by_ticket(ticket, exit_reason=reason)

    def get_equity_at_time(self) -> float:
        """Get current equity (balance + floating P&L)."""
        floating = sum(self._calculate_profit(p) for p in self._positions.values())
        return self._account.balance + floating


# Create module-level constants to match MT5 API
TIMEFRAME_M1 = Timeframe.TIMEFRAME_M1
TIMEFRAME_M2 = Timeframe.TIMEFRAME_M2
TIMEFRAME_M3 = Timeframe.TIMEFRAME_M3
TIMEFRAME_M5 = Timeframe.TIMEFRAME_M5
TIMEFRAME_M15 = Timeframe.TIMEFRAME_M15
TIMEFRAME_M30 = Timeframe.TIMEFRAME_M30
TIMEFRAME_H1 = Timeframe.TIMEFRAME_H1
TIMEFRAME_H4 = Timeframe.TIMEFRAME_H4
TIMEFRAME_D1 = Timeframe.TIMEFRAME_D1

ORDER_TYPE_BUY = OrderType.ORDER_TYPE_BUY
ORDER_TYPE_SELL = OrderType.ORDER_TYPE_SELL

POSITION_TYPE_BUY = PositionType.POSITION_TYPE_BUY
POSITION_TYPE_SELL = PositionType.POSITION_TYPE_SELL

TRADE_ACTION_DEAL = TradeAction.TRADE_ACTION_DEAL
TRADE_ACTION_SLTP = TradeAction.TRADE_ACTION_SLTP

TRADE_RETCODE_DONE = TradeRetcode.TRADE_RETCODE_DONE
TRADE_RETCODE_DONE_PARTIAL = TradeRetcode.TRADE_RETCODE_DONE_PARTIAL

ORDER_FILLING_FOK = 0
ORDER_FILLING_IOC = 1
ORDER_FILLING_RETURN = 2

ORDER_TIME_GTC = 0


# Module-level functions to match MT5 API
_emulator = MT5Emulator()


def initialize(path: str = "", login: int = 0, password: str = "",
               server: str = "", timeout: int = 60000, portable: bool = False) -> bool:
    """Module-level initialize function."""
    return _emulator.initialize(path, login, password, server, timeout, portable)


def shutdown() -> None:
    """Module-level shutdown function."""
    _emulator.shutdown()


def login(login: int, password: str = "", server: str = "",
          timeout: int = 60000) -> bool:
    """Module-level login function."""
    return _emulator.login(login, password, server, timeout)


def account_info() -> Optional[AccountInfo]:
    """Module-level account_info function."""
    return _emulator.account_info()


def symbol_info(symbol: str) -> Optional[SymbolInfo]:
    """Module-level symbol_info function."""
    return _emulator.symbol_info(symbol)


def symbol_info_tick(symbol: str) -> Optional[TickData]:
    """Module-level symbol_info_tick function."""
    return _emulator.symbol_info_tick(symbol)


def copy_rates_from_pos(symbol: str, timeframe: int, start_pos: int,
                        count: int) -> Optional[np.ndarray]:
    """Module-level copy_rates_from_pos function."""
    return _emulator.copy_rates_from_pos(symbol, timeframe, start_pos, count)


def positions_get(symbol: str = None, ticket: int = None) -> Optional[tuple]:
    """Module-level positions_get function."""
    return _emulator.positions_get(symbol, ticket)


def order_send(request: Dict) -> OrderResult:
    """Module-level order_send function."""
    return _emulator.order_send(request)


def last_error() -> tuple:
    """Module-level last_error function."""
    return _emulator.last_error()


def history_deals_get(date_from, date_to) -> Optional[tuple]:
    """Module-level history_deals_get function."""
    return _emulator.history_deals_get(date_from, date_to)
