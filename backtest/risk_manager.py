"""
Backtest Risk Manager - Position sizing for backtesting

Responsible for:
- Calculating position size based on risk percentage OR fixed amount
- Checking margin usage (max 40% of equity)
- Respecting volume_step, volume_min, volume_max

This is a simplified version of src/risk_manager.py adapted for backtesting.
"""
import logging
from typing import Dict, Any, Optional, Union

logger = logging.getLogger(__name__)


class BacktestRiskManager:
    """
    Risk Manager for backtest position sizing.

    Supports two risk modes:
    1. Percentage mode: risk_pct=1.0 means 1% of equity per trade
    2. Fixed amount mode: risk_amount=1000 means $1000 max loss per trade
    """

    def __init__(
        self,
        emulator,
        risk_pct: Optional[float] = None,
        risk_amount: Optional[float] = None,
        max_margin_pct: float = 40.0,
    ):
        """
        Initialize backtest risk manager.

        Args:
            emulator: MT5Emulator instance
            risk_pct: Risk percentage per trade (e.g., 1.0 = 1%). Mutually exclusive with risk_amount.
            risk_amount: Fixed risk amount per trade in account currency (e.g., 1000 = $1000).
            max_margin_pct: Maximum margin usage percentage (e.g., 40.0 = 40%)

        Note: Either risk_pct or risk_amount must be provided, not both.
              If neither is provided, defaults to 1% risk.
        """
        self.emulator = emulator
        self.max_margin_pct = max_margin_pct

        # Validate risk parameters
        if risk_pct is not None and risk_amount is not None:
            raise ValueError("Cannot specify both risk_pct and risk_amount. Choose one.")

        if risk_pct is not None:
            self.risk_mode = "percent"
            self.risk_pct = risk_pct
            self.risk_amount = None
            logger.info(f"BacktestRiskManager initialized: risk={risk_pct}% (percent mode), max_margin={max_margin_pct}%")
        elif risk_amount is not None:
            self.risk_mode = "fixed"
            self.risk_pct = None
            self.risk_amount = risk_amount
            logger.info(f"BacktestRiskManager initialized: risk=${risk_amount} (fixed mode), max_margin={max_margin_pct}%")
        else:
            # Default to 1% if nothing specified
            self.risk_mode = "percent"
            self.risk_pct = 1.0
            self.risk_amount = None
            logger.info(f"BacktestRiskManager initialized: risk=1.0% (default), max_margin={max_margin_pct}%")

    def calculate_position_size(
        self,
        symbol: str,
        entry_price: float,
        stop_loss: float,
    ) -> float:
        """
        Calculate lot size based on risk parameters.

        Supports two modes:
        - Percent mode: risk_amount = equity * (risk_pct / 100)
        - Fixed mode: risk_amount = self.risk_amount (fixed value)

        Formula:
        1. risk_amount = equity * (risk_pct / 100) OR fixed amount
        2. sl_distance = abs(entry_price - stop_loss)
        3. risk_per_lot = (sl_distance / tick_size) * tick_value
        4. raw_lots = risk_amount / risk_per_lot
        5. Check margin constraint: max_lots = (equity * max_margin_pct/100 * leverage) / (price * contract_size)
        6. final_lots = min(raw_lots, max_lots)
        7. Adjust to volume_step and apply min/max limits

        Args:
            symbol: Symbol name (e.g., "GER40")
            entry_price: Entry price
            stop_loss: Stop loss price

        Returns:
            Lot size (adjusted for volume_step and min/max limits)
        """
        # Get symbol info from emulator
        symbol_info = self.emulator.symbol_info(symbol)
        if not symbol_info:
            logger.error(f"Could not get symbol info for {symbol}")
            return 0.0

        volume_step = symbol_info.volume_step
        volume_min = symbol_info.volume_min
        volume_max = symbol_info.volume_max
        contract_size = symbol_info.trade_contract_size

        # Get account info from emulator
        account = self.emulator.account_info()
        equity = account.equity
        leverage = account.leverage

        # Calculate risk amount based on mode
        if self.risk_mode == "percent":
            risk_amount = equity * (self.risk_pct / 100.0)
            risk_desc = f"Risk {self.risk_pct}%"
        else:  # fixed mode
            risk_amount = self.risk_amount
            risk_desc = f"Risk ${self.risk_amount}"

        # Calculate SL distance
        sl_distance = abs(entry_price - stop_loss)
        if sl_distance <= 0:
            logger.error(f"Invalid SL distance: {sl_distance}")
            return 0.0

        # Calculate risk per 1 lot
        # For CFD: risk_per_lot = (sl_distance / tick_size) * tick_value
        # This converts price movement to monetary value per lot
        tick_size = symbol_info.trade_tick_size
        tick_value = symbol_info.trade_tick_value
        risk_per_lot = (sl_distance / tick_size) * tick_value

        if risk_per_lot <= 0:
            logger.error(f"Invalid risk_per_lot: {risk_per_lot}")
            return 0.0

        # Calculate raw lot size based on risk
        raw_lots = risk_amount / risk_per_lot

        # Calculate maximum lots based on margin constraint (40% of equity)
        # Margin required = (price * volume * contract_size) / leverage
        # max_margin = equity * max_margin_pct / 100
        # max_volume = max_margin * leverage / (price * contract_size)
        max_margin = equity * (self.max_margin_pct / 100.0)
        margin_per_lot = (entry_price * contract_size) / leverage
        max_lots_by_margin = max_margin / margin_per_lot if margin_per_lot > 0 else float('inf')

        # Take the minimum of risk-based and margin-based lot size
        constrained_lots = min(raw_lots, max_lots_by_margin)

        # Adjust to volume_step
        adjusted_lots = round(constrained_lots / volume_step) * volume_step

        # Apply min/max limits
        final_lots = min(max(adjusted_lots, volume_min), volume_max)

        # Log calculation details
        logger.info(f"Position size calculation for {symbol}:")
        logger.info(f"  Equity: {equity:.2f}, {risk_desc}: {risk_amount:.2f}")
        logger.info(f"  Entry: {entry_price:.2f}, SL: {stop_loss:.2f}, Distance: {sl_distance:.2f}")
        logger.info(f"  Risk per lot: {risk_per_lot:.2f}")
        logger.info(f"  Raw lots (risk-based): {raw_lots:.4f}")
        logger.info(f"  Max lots (margin {self.max_margin_pct}%): {max_lots_by_margin:.4f}")
        logger.info(f"  Constrained lots: {constrained_lots:.4f}")
        logger.info(f"  Adjusted lots (step {volume_step}): {adjusted_lots:.2f}")
        logger.info(f"  Final lots (min={volume_min}, max={volume_max}): {final_lots:.2f}")

        return final_lots

    def check_margin_available(self, symbol: str, lots: float, price: float) -> Dict[str, Any]:
        """
        Check if margin is available for trade

        Args:
            symbol: Symbol name
            lots: Proposed lot size
            price: Entry price

        Returns:
            Dict with 'allowed': bool, 'required_margin': float, 'current_margin_pct': float
        """
        symbol_info = self.emulator.symbol_info(symbol)
        if not symbol_info:
            return {"allowed": False, "reason": f"Symbol {symbol} not found"}

        account = self.emulator.account_info()
        equity = account.equity
        current_margin = account.margin
        leverage = account.leverage
        contract_size = symbol_info.trade_contract_size

        # Calculate required margin for this trade
        required_margin = (price * lots * contract_size) / leverage

        # Calculate total margin if trade opens
        total_margin = current_margin + required_margin

        # Calculate margin percentage
        margin_pct = (total_margin / equity * 100) if equity > 0 else 0

        allowed = margin_pct <= self.max_margin_pct

        return {
            "allowed": allowed,
            "required_margin": required_margin,
            "total_margin": total_margin,
            "margin_pct": margin_pct,
            "max_margin_pct": self.max_margin_pct,
            "equity": equity,
        }

    def validate_trade(self, symbol: str, lots: float, entry_price: float) -> Dict[str, Any]:
        """
        Complete pre-trade validation

        Checks:
        1. Lot size is valid (>= min, <= max, correct step)
        2. Margin usage won't exceed limit

        Args:
            symbol: Symbol name
            lots: Proposed lot size
            entry_price: Entry price

        Returns:
            Dict with 'valid': bool, 'reason': str
        """
        symbol_info = self.emulator.symbol_info(symbol)
        if not symbol_info:
            return {"valid": False, "reason": f"Symbol {symbol} not found"}

        volume_min = symbol_info.volume_min
        volume_max = symbol_info.volume_max
        volume_step = symbol_info.volume_step

        # Validate lot size
        if lots < volume_min:
            return {"valid": False, "reason": f"Lot size {lots} < minimum {volume_min}"}

        if lots > volume_max:
            return {"valid": False, "reason": f"Lot size {lots} > maximum {volume_max}"}

        # Check if lots match volume_step
        steps = round(lots / volume_step)
        expected = steps * volume_step
        if abs(lots - expected) > 0.0001:
            return {"valid": False, "reason": f"Lot size {lots} doesn't match volume_step {volume_step}"}

        # Check margin
        margin_check = self.check_margin_available(symbol, lots, entry_price)
        if not margin_check["allowed"]:
            return {
                "valid": False,
                "reason": f"Margin limit exceeded: {margin_check['margin_pct']:.1f}% > {self.max_margin_pct}%",
            }

        return {"valid": True, "reason": "OK"}
