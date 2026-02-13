"""
Data models for MT5 Emulator.

Defines dataclasses that mirror MT5 API structures for seamless integration.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Any
from enum import IntEnum


class OrderType(IntEnum):
    """MT5 order types."""
    ORDER_TYPE_BUY = 0
    ORDER_TYPE_SELL = 1
    ORDER_TYPE_BUY_LIMIT = 2
    ORDER_TYPE_SELL_LIMIT = 3
    ORDER_TYPE_BUY_STOP = 4
    ORDER_TYPE_SELL_STOP = 5


class PositionType(IntEnum):
    """MT5 position types."""
    POSITION_TYPE_BUY = 0
    POSITION_TYPE_SELL = 1


class TradeAction(IntEnum):
    """MT5 trade actions."""
    TRADE_ACTION_DEAL = 1
    TRADE_ACTION_PENDING = 5
    TRADE_ACTION_SLTP = 6
    TRADE_ACTION_MODIFY = 7
    TRADE_ACTION_REMOVE = 8
    TRADE_ACTION_CLOSE_BY = 10


class TradeRetcode(IntEnum):
    """MT5 trade return codes."""
    TRADE_RETCODE_REQUOTE = 10004
    TRADE_RETCODE_REJECT = 10006
    TRADE_RETCODE_CANCEL = 10007
    TRADE_RETCODE_PLACED = 10008
    TRADE_RETCODE_DONE = 10009
    TRADE_RETCODE_DONE_PARTIAL = 10010
    TRADE_RETCODE_ERROR = 10011
    TRADE_RETCODE_TIMEOUT = 10012
    TRADE_RETCODE_INVALID = 10013
    TRADE_RETCODE_INVALID_VOLUME = 10014
    TRADE_RETCODE_INVALID_PRICE = 10015
    TRADE_RETCODE_INVALID_STOPS = 10016
    TRADE_RETCODE_TRADE_DISABLED = 10017
    TRADE_RETCODE_MARKET_CLOSED = 10018
    TRADE_RETCODE_NO_MONEY = 10019
    TRADE_RETCODE_PRICE_CHANGED = 10020
    TRADE_RETCODE_PRICE_OFF = 10021
    TRADE_RETCODE_INVALID_EXPIRATION = 10022
    TRADE_RETCODE_ORDER_CHANGED = 10023
    TRADE_RETCODE_TOO_MANY_REQUESTS = 10024
    TRADE_RETCODE_NO_CHANGES = 10025
    TRADE_RETCODE_SERVER_DISABLES_AT = 10026
    TRADE_RETCODE_CLIENT_DISABLES_AT = 10027
    TRADE_RETCODE_LOCKED = 10028
    TRADE_RETCODE_FROZEN = 10029
    TRADE_RETCODE_INVALID_FILL = 10030


class FillingMode(IntEnum):
    """MT5 filling modes."""
    ORDER_FILLING_FOK = 0
    ORDER_FILLING_IOC = 1
    ORDER_FILLING_RETURN = 2


class Timeframe(IntEnum):
    """MT5 timeframes."""
    TIMEFRAME_M1 = 1
    TIMEFRAME_M2 = 2
    TIMEFRAME_M3 = 3
    TIMEFRAME_M5 = 5
    TIMEFRAME_M15 = 15
    TIMEFRAME_M30 = 30
    TIMEFRAME_H1 = 60
    TIMEFRAME_H4 = 240
    TIMEFRAME_D1 = 1440


@dataclass
class TickData:
    """
    Tick data structure (mirrors mt5.symbol_info_tick).

    Attributes:
        time: Tick timestamp
        bid: Bid price
        ask: Ask price
        last: Last deal price
        volume: Volume of last deal
        flags: Tick flags
    """
    time: datetime
    bid: float
    ask: float
    last: float = 0.0
    volume: float = 0.0
    flags: int = 0

    def __getattr__(self, name: str) -> Any:
        """Allow attribute-style access for MT5 compatibility."""
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


@dataclass
class SymbolInfo:
    """
    Symbol information structure (mirrors mt5.symbol_info).

    Attributes:
        name: Symbol name
        digits: Price decimal places
        point: Minimum price change
        spread: Current spread
        volume_min: Minimum volume
        volume_max: Maximum volume
        volume_step: Volume step
        trade_tick_size: Tick size
        trade_tick_value: Tick value in account currency
        trade_contract_size: Contract size
        filling_mode: Supported filling modes bitmask
    """
    name: str
    digits: int = 2
    point: float = 0.01
    spread: int = 0
    volume_min: float = 0.01
    volume_max: float = 100.0
    volume_step: float = 0.01
    trade_tick_size: float = 0.01
    trade_tick_value: float = 1.0
    trade_contract_size: float = 1.0
    filling_mode: int = 2  # IOC by default
    description: str = ""


@dataclass
class AccountInfo:
    """
    Account information structure (mirrors mt5.account_info).

    Attributes:
        login: Account login
        server: Server name
        balance: Account balance
        equity: Account equity
        margin: Used margin
        margin_free: Free margin
        margin_level: Margin level percentage
        profit: Current floating profit
        currency: Account currency
        leverage: Account leverage
        company: Broker company name
    """
    login: int = 0
    server: str = "BacktestServer"
    balance: float = 50000.0
    equity: float = 50000.0
    margin: float = 0.0
    margin_free: float = 50000.0
    margin_level: float = 0.0
    profit: float = 0.0
    currency: str = "USD"
    leverage: int = 100
    company: str = "Backtest Emulator"


@dataclass
class Position:
    """
    Open position structure (mirrors mt5.positions_get element).

    Attributes:
        ticket: Position ticket (unique ID)
        symbol: Trading symbol
        type: Position type (BUY/SELL)
        volume: Position volume in lots
        price_open: Open price
        sl: Stop loss price
        tp: Take profit price
        price_current: Current price
        profit: Floating profit
        swap: Accumulated swap
        magic: Magic number
        comment: Position comment
        time: Open time
        identifier: Position identifier
    """
    ticket: int
    symbol: str
    type: int  # PositionType
    volume: float
    price_open: float
    sl: float = 0.0
    tp: float = 0.0
    price_current: float = 0.0
    profit: float = 0.0
    swap: float = 0.0
    magic: int = 0
    comment: str = ""
    time: Optional[datetime] = None
    identifier: int = 0

    def __post_init__(self):
        if self.identifier == 0:
            self.identifier = self.ticket


@dataclass
class Order:
    """
    Order structure for history tracking.

    Attributes:
        ticket: Order ticket
        symbol: Trading symbol
        type: Order type
        volume: Order volume
        price: Execution price
        sl: Stop loss
        tp: Take profit
        magic: Magic number
        comment: Order comment
        time_setup: Order setup time
        time_done: Order execution time
        state: Order state
    """
    ticket: int
    symbol: str
    type: int
    volume: float
    price: float
    sl: float = 0.0
    tp: float = 0.0
    magic: int = 0
    comment: str = ""
    time_setup: Optional[datetime] = None
    time_done: Optional[datetime] = None
    state: int = 0


@dataclass
class Deal:
    """
    Deal (trade) structure for history.

    Attributes:
        ticket: Deal ticket
        order: Related order ticket
        symbol: Trading symbol
        type: Deal type (0=buy, 1=sell)
        volume: Deal volume
        price: Deal price
        profit: Deal profit
        swap: Swap
        commission: Commission
        magic: Magic number
        comment: Deal comment
        time: Deal time
        position_id: Position identifier
        entry: Entry type (0=in, 1=out, 2=inout)
    """
    ticket: int
    order: int
    symbol: str
    type: int
    volume: float
    price: float
    profit: float = 0.0
    swap: float = 0.0
    commission: float = 0.0
    magic: int = 0
    comment: str = ""
    time: Optional[datetime] = None
    position_id: int = 0
    entry: int = 0  # 0=DEAL_ENTRY_IN, 1=DEAL_ENTRY_OUT


@dataclass
class OrderSendRequest:
    """
    Order send request structure (mirrors mt5.order_send request dict).

    Attributes:
        action: Trade action
        symbol: Trading symbol
        volume: Volume in lots
        type: Order type
        price: Order price
        sl: Stop loss
        tp: Take profit
        deviation: Max price deviation
        magic: Magic number
        comment: Order comment
        type_time: Order lifetime type
        type_filling: Order filling type
        position: Position ticket (for close/modify)
        initial_tp: Initial take profit for tracking (not set on position)
    """
    action: int
    symbol: str
    volume: float
    type: int
    price: float = 0.0
    sl: float = 0.0
    tp: float = 0.0
    deviation: int = 20
    magic: int = 0
    comment: str = ""
    type_time: int = 0
    type_filling: int = FillingMode.ORDER_FILLING_IOC
    position: int = 0
    initial_tp: float = 0.0  # Initial TP for tracking (not set on position)


@dataclass
class OrderResult:
    """
    Order send result structure (mirrors mt5.order_send result).

    Attributes:
        retcode: Return code
        deal: Deal ticket
        order: Order ticket
        volume: Executed volume
        price: Execution price
        bid: Bid at execution
        ask: Ask at execution
        comment: Result comment
        request_id: Request ID
    """
    retcode: int
    deal: int = 0
    order: int = 0
    volume: float = 0.0
    price: float = 0.0
    bid: float = 0.0
    ask: float = 0.0
    comment: str = ""
    request_id: int = 0


@dataclass
class TradeLog:
    """
    Trade log entry for analytics.

    Attributes:
        ticket: Position ticket
        symbol: Trading symbol
        direction: 'long' or 'short'
        entry_time: Entry timestamp
        entry_price: Entry price
        exit_time: Exit timestamp
        exit_price: Exit price
        volume: Position volume
        sl: Stop loss
        tp: Take profit
        profit: Realized profit
        exit_reason: Reason for exit (sl, tp, time, manual)
        magic: Magic number
        comment: Trade comment
        variation: Strategy variation name
    """
    ticket: int
    symbol: str
    direction: str
    entry_time: datetime
    entry_price: float
    exit_time: Optional[datetime] = None
    exit_price: float = 0.0
    volume: float = 0.0
    sl: float = 0.0
    tp: float = 0.0
    profit: float = 0.0
    exit_reason: str = ""
    magic: int = 0
    comment: str = ""
    variation: str = ""
