"""
Base Strategy - Abstract base class for all strategies

Defines the interface that all strategies must implement:
- check_signal(current_time_utc) -> Optional[Signal]
- update_position_state(position, tick)
"""
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Any

logger = logging.getLogger(__name__)


@dataclass
class Signal:
    """
    Signal data structure

    Represents a trading signal with all necessary information
    for order placement.
    """
    direction: str  # "long" or "short"
    entry_price: float
    stop_loss: float
    take_profit: float
    comment: str = ""
    variation: str = ""  # "Reverse", "OCAE", "TCWE", "REV_RB"
    use_virtual_tp: bool = True  # False when TSL_TARGET <= 0 (use actual TP in MT5)


class BaseStrategy(ABC):
    """
    Abstract base class for trading strategies

    All strategy implementations must inherit from this class
    and implement the required methods.
    """

    def __init__(self, symbol: str, magic_number: int):
        """
        Initialize base strategy

        Args:
            symbol: Trading symbol
            magic_number: Unique magic number for this strategy instance
        """
        self.symbol = symbol
        self.magic_number = magic_number
        self.strategy_id = f"{self.__class__.__name__}_{symbol}_{magic_number}"
        logger.info(f"Strategy initialized: {self.strategy_id}")

    @abstractmethod
    def check_signal(self, current_time_utc: datetime) -> Optional[Signal]:
        """
        Check for trading signal

        This method is called every second by the bot controller
        to check if there's a new trading signal.

        Args:
            current_time_utc: Current time in UTC

        Returns:
            Signal object if signal detected, None otherwise
        """
        pass

    @abstractmethod
    def update_position_state(self, position: Any, tick: dict) -> None:
        """
        Update position state (TSL logic)

        This method is called every second when there's an open position
        for this strategy. It should implement trailing stop loss logic.

        Args:
            position: Open position object from MT5
            tick: Current tick data (bid, ask, last)
        """
        pass
