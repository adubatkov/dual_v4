"""
Backtest Adapter - Provides MT5Executor-compatible interface for backtesting.

This module allows IBStrategy to run unchanged during backtest by providing:
1. BacktestExecutor - Drop-in replacement for MT5Executor
2. Patch function to replace mt5 module import in IBStrategy
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import pandas as pd
import pytz

from .emulator import mt5_emulator as mt5_emu
from .emulator.models import (
    OrderType,
    PositionType,
    TradeAction,
    TradeRetcode,
    Timeframe,
)
from .config import BacktestConfig

logger = logging.getLogger(__name__)


class BacktestExecutor:
    """
    Drop-in replacement for MT5Executor for backtesting.

    Provides the same interface as MT5Executor but uses the MT5Emulator
    instead of the real MetaTrader5 library.
    """

    def __init__(self, emulator: mt5_emu.MT5Emulator, config: BacktestConfig):
        """
        Initialize backtest executor.

        Args:
            emulator: MT5Emulator instance
            config: Backtest configuration
        """
        self.emulator = emulator
        self.config = config
        self.connected = False
        self.symbol_info_cache = {}
        self.dry_run = False  # Always false for backtest

        logger.info("BacktestExecutor initialized")

    def connect(self, login: int, password: str, server: str,
                retries: int = 3, delay: int = 1) -> bool:
        """
        Connect to emulator (always succeeds).

        Args:
            login: Account login (ignored)
            password: Account password (ignored)
            server: Server name (ignored)
            retries: Number of retries (ignored)
            delay: Delay between retries (ignored)

        Returns:
            True always
        """
        self.emulator.initialize()
        self.emulator.login(login, password, server)
        self.connected = True

        account = self.emulator.account_info()
        if account:
            logger.info(f"BacktestExecutor connected")
            logger.info(f"  Account: {account.login}")
            logger.info(f"  Balance: {account.balance}")
            logger.info(f"  Leverage: {account.leverage}")

        return True

    def disconnect(self) -> None:
        """Disconnect from emulator."""
        self.emulator.shutdown()
        self.connected = False
        logger.info("BacktestExecutor disconnected")

    def get_bars(self, symbol: str, timeframe: str, count: int) -> Optional[pd.DataFrame]:
        """
        Fetch OHLC bars for symbol.

        IMPORTANT: IBStrategy uses M2 (2-minute) timeframe.

        Args:
            symbol: Symbol name (e.g., "GER40")
            timeframe: Timeframe string ("M1", "M2", etc.)
            count: Number of bars to fetch

        Returns:
            DataFrame with columns [time, open, high, low, close, tick_volume]
            or None if failed
        """
        if not self.connected:
            logger.error("Cannot get bars: not connected")
            return None

        # Map timeframe string to emulator constant
        timeframe_map = {
            "M1": Timeframe.TIMEFRAME_M1,
            "M2": Timeframe.TIMEFRAME_M2,
            "M5": Timeframe.TIMEFRAME_M5,
            "M15": Timeframe.TIMEFRAME_M15,
            "M30": Timeframe.TIMEFRAME_M30,
            "H1": Timeframe.TIMEFRAME_H1,
            "H4": Timeframe.TIMEFRAME_H4,
            "D1": Timeframe.TIMEFRAME_D1,
        }

        if timeframe not in timeframe_map:
            logger.error(f"Invalid timeframe: {timeframe}")
            return None

        tf = timeframe_map[timeframe]

        # Get rates from emulator
        rates = self.emulator.copy_rates_from_pos(symbol, tf, 0, count)

        if rates is None or len(rates) == 0:
            logger.warning(f"No bars available for {symbol} ({timeframe})")
            return None

        # Convert structured array to DataFrame
        df = pd.DataFrame({
            "time": pd.to_datetime(rates["time"], unit="s", utc=True),
            "open": rates["open"],
            "high": rates["high"],
            "low": rates["low"],
            "close": rates["close"],
            "tick_volume": rates["tick_volume"],
        })

        logger.debug(f"Fetched {len(df)} bars for {symbol} ({timeframe})")
        return df

    def get_tick(self, symbol: str) -> Optional[Dict[str, float]]:
        """
        Get current tick for symbol.

        Args:
            symbol: Symbol name

        Returns:
            Dict with 'bid', 'ask', 'last', 'time' or None
        """
        if not self.connected:
            logger.error("Cannot get tick: not connected")
            return None

        tick = self.emulator.symbol_info_tick(symbol)
        if tick is None:
            logger.warning(f"No tick for {symbol}")
            return None

        return {
            "bid": tick.bid,
            "ask": tick.ask,
            "last": tick.last,
            "time": tick.time,
        }

    def get_open_positions(self) -> List[Any]:
        """
        Get all open positions.

        Returns:
            List of position objects
        """
        if not self.connected:
            logger.error("Cannot get positions: not connected")
            return []

        positions = self.emulator.positions_get()
        if positions is None:
            return []

        return list(positions)

    def get_positions_history_today(self, magic_number: Optional[int] = None) -> List[Any]:
        """
        Get positions history for today (closed positions).

        Args:
            magic_number: Filter by magic number if specified

        Returns:
            List of closed deals from today
        """
        if not self.connected:
            return []

        now = self.emulator.current_time or datetime.now(pytz.utc)
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        today_end = now.replace(hour=23, minute=59, second=59, microsecond=999999)

        deals = self.emulator.history_deals_get(today_start, today_end)
        if deals is None:
            return []

        if magic_number is not None:
            return [d for d in deals if d.magic == magic_number]

        return list(deals)

    def get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get symbol information (for lot sizing).

        Args:
            symbol: Symbol name

        Returns:
            Dict with volume_step, volume_min, volume_max, etc.
        """
        if symbol in self.symbol_info_cache:
            return self.symbol_info_cache[symbol]

        if not self.connected:
            logger.error("Cannot get symbol info: not connected")
            return None

        info = self.emulator.symbol_info(symbol)
        if info is None:
            logger.error(f"Symbol not found: {symbol}")
            return None

        symbol_data = {
            "volume_step": info.volume_step,
            "volume_min": info.volume_min,
            "volume_max": info.volume_max,
            "trade_tick_value": info.trade_tick_value,
            "trade_tick_size": info.trade_tick_size,
            "trade_contract_size": info.trade_contract_size,
            "point": info.point,
            "digits": info.digits,
            "spread": info.spread,
            "filling_mode": info.filling_mode,
        }

        self.symbol_info_cache[symbol] = symbol_data
        return symbol_data

    def place_order(self, symbol: str, signal, lots: float, magic_number: int) -> Dict[str, Any]:
        """
        Place order based on signal.

        Args:
            symbol: Trading symbol
            signal: Signal object with direction, entry_price, stop_loss, take_profit
            lots: Lot size
            magic_number: Magic number for identification

        Returns:
            Dict with success, retcode, ticket, comment
        """
        if not self.connected:
            return {"success": False, "retcode": None, "comment": "Not connected"}

        direction = signal.direction
        sl = signal.stop_loss
        tp = signal.take_profit

        # Determine order type
        if direction == "long":
            order_type = OrderType.ORDER_TYPE_BUY
        else:
            order_type = OrderType.ORDER_TYPE_SELL

        # Get variation from signal
        variation = getattr(signal, "variation", "Unknown")

        # Prepare request
        request = {
            "action": TradeAction.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lots,
            "type": order_type,
            "price": signal.entry_price,
            "sl": sl,
            "tp": 0.0 if signal.use_virtual_tp else tp,  # Virtual TP when TSL active, actual TP when TSL disabled
            "initial_tp": tp,  # Original TP for tracking/charting
            "magic": magic_number,
            "comment": f"IBStrategy_{variation}_{direction}",
            "type_filling": 1,  # IOC
        }

        logger.info(f"PLACING ORDER: {symbol} {direction} {lots} lots @ {signal.entry_price}, "
                   f"SL={sl}, TP={tp} (virtual), Magic={magic_number}")

        result = self.emulator.order_send(request)

        if result.retcode == TradeRetcode.TRADE_RETCODE_DONE:
            logger.info(f"ORDER SUCCESSFUL: Ticket={result.order}")
            return {
                "success": True,
                "retcode": result.retcode,
                "ticket": result.order,
                "comment": "Order placed",
                "price": result.price,
                "volume": result.volume,
            }
        else:
            logger.error(f"ORDER FAILED: {result.comment}")
            return {
                "success": False,
                "retcode": result.retcode,
                "comment": result.comment,
            }

    def modify_position(self, ticket: int, sl: float, tp: float) -> bool:
        """
        Modify stop loss and take profit for open position.

        Args:
            ticket: Position ticket number
            sl: New stop loss price
            tp: New take profit price

        Returns:
            True if successful
        """
        if not self.connected:
            return False

        positions = self.emulator.positions_get(ticket=ticket)
        if not positions:
            logger.error(f"Position {ticket} not found")
            return False

        request = {
            "action": TradeAction.TRADE_ACTION_SLTP,
            "symbol": positions[0].symbol,
            "position": ticket,
            "sl": sl,
            "tp": tp,
        }

        result = self.emulator.order_send(request)

        if result.retcode == TradeRetcode.TRADE_RETCODE_DONE:
            logger.debug(f"Position {ticket} modified: SL={sl}, TP={tp}")
            return True
        else:
            logger.error(f"Failed to modify position {ticket}: {result.comment}")
            return False

    def close_position(self, ticket: int) -> bool:
        """
        Close an open position at current market price.

        Args:
            ticket: Position ticket number

        Returns:
            True if successful
        """
        if not self.connected:
            return False

        success = self.emulator.close_position_by_ticket(ticket, exit_reason="time")
        if success:
            logger.info(f"Position {ticket} closed")
        else:
            logger.error(f"Failed to close position {ticket}")

        return success


def create_mt5_patch_module(emulator: mt5_emu.MT5Emulator):
    """
    Create a module-like object that can replace 'MetaTrader5' import.

    This allows IBStrategy to use the emulator transparently by patching
    the MetaTrader5 import at runtime.

    Usage:
        import sys
        from backtest.adapter import create_mt5_patch_module
        sys.modules['MetaTrader5'] = create_mt5_patch_module(emulator)

        # Now importing MetaTrader5 will use the emulator
        import MetaTrader5 as mt5

    Args:
        emulator: MT5Emulator instance

    Returns:
        Module-like object with MT5 API
    """
    from types import ModuleType

    # Create a new module
    mt5_patch = ModuleType("MetaTrader5")

    # Copy all module-level functions and constants
    mt5_patch.initialize = lambda **kwargs: emulator.initialize()
    mt5_patch.shutdown = emulator.shutdown
    mt5_patch.login = lambda login, password="", server="", **kwargs: emulator.login(login, password, server)
    mt5_patch.account_info = emulator.account_info
    mt5_patch.symbol_info = emulator.symbol_info
    mt5_patch.symbol_info_tick = emulator.symbol_info_tick
    mt5_patch.copy_rates_from_pos = emulator.copy_rates_from_pos
    mt5_patch.positions_get = lambda **kwargs: emulator.positions_get(
        symbol=kwargs.get("symbol"),
        ticket=kwargs.get("ticket")
    )
    mt5_patch.order_send = emulator.order_send
    mt5_patch.last_error = emulator.last_error
    mt5_patch.history_deals_get = emulator.history_deals_get

    # Constants
    mt5_patch.TIMEFRAME_M1 = Timeframe.TIMEFRAME_M1
    mt5_patch.TIMEFRAME_M2 = Timeframe.TIMEFRAME_M2
    mt5_patch.TIMEFRAME_M5 = Timeframe.TIMEFRAME_M5
    mt5_patch.TIMEFRAME_M15 = Timeframe.TIMEFRAME_M15
    mt5_patch.TIMEFRAME_M30 = Timeframe.TIMEFRAME_M30
    mt5_patch.TIMEFRAME_H1 = Timeframe.TIMEFRAME_H1
    mt5_patch.TIMEFRAME_H4 = Timeframe.TIMEFRAME_H4
    mt5_patch.TIMEFRAME_D1 = Timeframe.TIMEFRAME_D1

    mt5_patch.ORDER_TYPE_BUY = OrderType.ORDER_TYPE_BUY
    mt5_patch.ORDER_TYPE_SELL = OrderType.ORDER_TYPE_SELL

    mt5_patch.POSITION_TYPE_BUY = PositionType.POSITION_TYPE_BUY
    mt5_patch.POSITION_TYPE_SELL = PositionType.POSITION_TYPE_SELL

    mt5_patch.TRADE_ACTION_DEAL = TradeAction.TRADE_ACTION_DEAL
    mt5_patch.TRADE_ACTION_SLTP = TradeAction.TRADE_ACTION_SLTP

    mt5_patch.TRADE_RETCODE_DONE = TradeRetcode.TRADE_RETCODE_DONE
    mt5_patch.TRADE_RETCODE_DONE_PARTIAL = TradeRetcode.TRADE_RETCODE_DONE_PARTIAL
    mt5_patch.TRADE_RETCODE_ERROR = TradeRetcode.TRADE_RETCODE_ERROR
    mt5_patch.TRADE_RETCODE_INVALID = TradeRetcode.TRADE_RETCODE_INVALID
    mt5_patch.TRADE_RETCODE_INVALID_VOLUME = TradeRetcode.TRADE_RETCODE_INVALID_VOLUME
    mt5_patch.TRADE_RETCODE_NO_MONEY = TradeRetcode.TRADE_RETCODE_NO_MONEY
    mt5_patch.TRADE_RETCODE_PRICE_OFF = TradeRetcode.TRADE_RETCODE_PRICE_OFF

    mt5_patch.ORDER_FILLING_FOK = 0
    mt5_patch.ORDER_FILLING_IOC = 1
    mt5_patch.ORDER_FILLING_RETURN = 2

    mt5_patch.ORDER_TIME_GTC = 0

    return mt5_patch
