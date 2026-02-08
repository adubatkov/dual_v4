"""
MT5 Emulator Module

Provides mock implementation of MetaTrader5 API for backtesting.
"""

from .mt5_emulator import MT5Emulator
from .time_manager import TimeManager
from .models import Position, Order, AccountInfo, TickData, OrderResult

__all__ = [
    "MT5Emulator",
    "TimeManager",
    "Position",
    "Order",
    "AccountInfo",
    "TickData",
    "OrderResult",
]
