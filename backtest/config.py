"""
Backtest Configuration

All configurable parameters for the backtest engine.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional
from pathlib import Path


@dataclass
class SymbolConfig:
    """Configuration for a trading symbol."""

    name: str
    spread_points: float  # Fixed spread in price points
    digits: int  # Price decimal places
    volume_step: float = 0.01
    volume_min: float = 0.01
    volume_max: float = 100.0
    trade_tick_size: float = 0.01
    trade_tick_value: float = 1.0
    trade_contract_size: float = 1.0
    timezone: str = "UTC"

    @property
    def point(self) -> float:
        """Minimum price change."""
        return 10 ** (-self.digits)


@dataclass
class BacktestConfig:
    """Main backtest configuration."""

    # Account settings
    initial_balance: float = 50000.0
    currency: str = "USD"
    leverage: int = 100

    # Data paths
    data_base_path: Path = field(default_factory=lambda: Path("C:/Trading/ib_trading_bot/dual_v4/data"))
    output_path: Path = field(default_factory=lambda: Path("C:/Trading/ib_trading_bot/dual_v4/backtest/output"))

    # Symbol configurations
    symbols: Dict[str, SymbolConfig] = field(default_factory=dict)

    # Tick generation settings
    ticks_per_minute: int = 12  # 5-second intervals
    jitter_factor: float = 0.1  # Price noise factor (0-1)
    random_seed: Optional[int] = 42  # For reproducibility, None for random

    # Execution settings
    slippage_points: float = 0.0  # Simulated slippage
    commission_per_lot: float = 0.0  # Commission per lot traded

    def __post_init__(self):
        """Initialize default symbol configs if not provided."""
        if not self.symbols:
            self.symbols = {
                "GER40": SymbolConfig(
                    name="GER40",
                    spread_points=1.5,  # ~1.5 index points
                    digits=1,
                    volume_step=0.1,
                    volume_min=0.1,
                    volume_max=50.0,
                    trade_tick_size=0.1,
                    trade_tick_value=0.1,  # EUR per tick per lot
                    trade_contract_size=1.0,
                    timezone="Europe/Berlin",
                ),
                "XAUUSD": SymbolConfig(
                    name="XAUUSD",
                    spread_points=0.25,  # ~25 cents
                    digits=2,
                    volume_step=0.01,
                    volume_min=0.01,
                    volume_max=100.0,
                    trade_tick_size=0.01,
                    trade_tick_value=1.0,  # USD per tick per lot
                    trade_contract_size=100.0,  # 100 oz per lot
                    timezone="Asia/Tokyo",
                ),
            }


# Default configuration instance
DEFAULT_CONFIG = BacktestConfig()


# Data folder mapping
DATA_FOLDERS = {
    "GER40": "GER40 1m 01_01_2023-04_11_2025",
    "XAUUSD": "XAUUSD 1m 01_01_2023-04_11_2025",
    # Control period (Nov 2025 - Jan 2026) for out-of-sample validation
    "GER40_CONTROL": "ger40+pepperstone_0411-2001",
    "XAUUSD_CONTROL": "xauusd_oanda_0411-2001",
}
