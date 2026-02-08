"""
Backtest Wrapper for Parameter Optimization.

Wraps the existing backtest engine to run with different parameter combinations.
Uses actual IBStrategy for reproducibility with live trading bot.
"""

import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import pandas as pd
import pytz

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backtest.config import BacktestConfig, SymbolConfig
from backtest.adapter import BacktestExecutor, create_mt5_patch_module
from backtest.emulator.mt5_emulator import MT5Emulator
from backtest.risk_manager import BacktestRiskManager

from params_optimizer.config import SYMBOL_CONFIGS, print_status


# Reduce logging noise from backtest engine
logging.getLogger("backtest").setLevel(logging.WARNING)
logging.getLogger("backtest.emulator").setLevel(logging.WARNING)


class BacktestWrapper:
    """
    Wraps existing backtest engine for parameter optimization.

    Uses actual IBStrategy for reproducibility with live trading bot.
    Optimized for speed: no chart generation, minimal logging, no Excel reports.
    """

    def __init__(
        self,
        symbol: str,
        m1_data: pd.DataFrame,
        initial_balance: float = 100000.0,
        risk_pct: float = 1.0,
        max_margin_pct: float = 40.0,
    ):
        """
        Initialize BacktestWrapper.

        Args:
            symbol: Trading symbol ("GER40" or "XAUUSD")
            m1_data: Pre-loaded M1 candlestick data (from optimized Parquet)
            initial_balance: Starting balance
            risk_pct: Risk percentage per trade
            max_margin_pct: Maximum margin usage percentage
        """
        self.symbol = symbol
        self.m1_data = m1_data
        self.initial_balance = initial_balance
        self.risk_pct = risk_pct
        self.max_margin_pct = max_margin_pct

        # Get symbol config
        sym_cfg = SYMBOL_CONFIGS.get(symbol)
        if sym_cfg is None:
            raise ValueError(f"Unknown symbol: {symbol}")

        self.symbol_config = sym_cfg
        self._emulator: Optional[MT5Emulator] = None
        self._mt5_patched = False

    def run_with_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run single backtest with given parameters.

        Args:
            params: Strategy parameters dict with keys:
                - ib_start: IB start time (e.g., "08:00")
                - ib_end: IB end time (e.g., "09:00")
                - ib_timezone: IB timezone (e.g., "Europe/Berlin")
                - ib_wait_minutes: Wait time after IB end
                - trade_window_minutes: Trade window duration
                - rr_target: Risk/Reward target
                - stop_mode: "ib_start" or "eq"
                - tsl_target: TSL activation R target (0 = disabled)
                - tsl_sl: TSL move SL to R level
                - min_sl_pct: Minimum SL percentage
                - rev_rb_enabled: Reverse rollback enabled
                - rev_rb_pct: Reverse rollback percentage
                - ib_buffer_pct: IB buffer percentage
                - max_distance_pct: Max distance from IB percentage

        Returns:
            Dict with results:
                - total_r: Total profit in R units
                - total_profit: Absolute profit
                - total_trades: Number of trades
                - winning_trades: Number of winning trades
                - losing_trades: Number of losing trades
                - winrate: Win rate percentage
                - sharpe_ratio: Sharpe ratio (if enough trades)
                - profit_factor: Profit factor
                - max_drawdown: Maximum drawdown percentage
                - avg_trade_r: Average R per trade
                - params: Original parameters (for tracking)
                - error: Error message if failed (None if success)
        """
        try:
            return self._run_backtest(params)
        except Exception as e:
            return {
                "total_r": 0.0,
                "total_profit": 0.0,
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "winrate": 0.0,
                "sharpe_ratio": 0.0,
                "profit_factor": 0.0,
                "max_drawdown": 0.0,
                "avg_trade_r": 0.0,
                "params": params,
                "error": str(e),
            }

    def _run_backtest(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Internal backtest execution.

        Args:
            params: Strategy parameters

        Returns:
            Results dict
        """
        # Create backtest config
        config = BacktestConfig(
            initial_balance=self.initial_balance,
            leverage=100,
            symbols={
                self.symbol: SymbolConfig(
                    name=self.symbol,
                    spread_points=self.symbol_config.spread_points,
                    digits=self.symbol_config.digits,
                    volume_step=self.symbol_config.volume_step,
                    trade_tick_size=self.symbol_config.trade_tick_size,
                    trade_tick_value=self.symbol_config.trade_tick_value,
                    trade_contract_size=self.symbol_config.trade_contract_size,
                )
            }
        )

        # Create and configure emulator
        emulator = MT5Emulator()
        emulator.reset()
        emulator.configure(config)

        # Patch MT5 module (critical for IBStrategy)
        mt5_patch = create_mt5_patch_module(emulator)
        sys.modules["MetaTrader5"] = mt5_patch

        # Import strategy AFTER patching MT5
        from src.strategies.ib_strategy import IBStrategy

        # Build strategy params dict
        strategy_params = self._build_strategy_params(params)

        # Create executor
        executor = BacktestExecutor(emulator, config)
        executor.connect(12345, "password", "BacktestServer")

        # Load M1 data into emulator
        emulator.load_m1_data(self.symbol, self.m1_data)

        # Create risk manager
        risk_manager = BacktestRiskManager(
            emulator=emulator,
            risk_pct=self.risk_pct,
            max_margin_pct=self.max_margin_pct,
        )

        # Create strategy
        magic_number = 1001 if self.symbol == "GER40" else 1002
        strategy = IBStrategy(
            symbol=self.symbol,
            params=strategy_params,
            executor=executor,
            magic_number=magic_number,
            strategy_label="Optimizer"
        )

        # Get timezone for this symbol
        tz = pytz.timezone(params.get("ib_timezone", self.symbol_config.timezone))

        # Run backtest loop
        trades_executed = 0

        for idx, row in self.m1_data.iterrows():
            current_time_utc = row["time"].to_pydatetime()

            # Set emulator time
            emulator.set_time(current_time_utc)

            # Check for signal only if no open position
            positions = executor.get_open_positions()
            has_position = any(p.magic == magic_number for p in positions)

            if not has_position:
                signal = strategy.check_signal(current_time_utc)

                if signal:
                    # Calculate position size
                    lots = risk_manager.calculate_position_size(
                        symbol=self.symbol,
                        entry_price=signal.entry_price,
                        stop_loss=signal.stop_loss,
                    )

                    if lots <= 0:
                        continue

                    # Validate trade
                    validation = risk_manager.validate_trade(self.symbol, lots, signal.entry_price)
                    if not validation["valid"]:
                        continue

                    # Place order
                    result = executor.place_order(self.symbol, signal, lots, magic_number)

                    if result["success"]:
                        trades_executed += 1
                        strategy.state = "POSITION_OPEN"

            # Update TSL for open positions
            positions = executor.get_open_positions()
            for position in positions:
                simulated_tick = {
                    "bid": float(row["low"]),
                    "ask": float(row["high"]),
                }
                strategy.update_position_state(position, simulated_tick, current_time_utc)

            # Check SL/TP hits
            self._check_sltp_hits(emulator, row, config.symbols[self.symbol])

        # Close remaining positions
        emulator.force_close_all_positions(reason="backtest_end")

        # Calculate metrics
        trade_log = emulator.get_trade_log()
        metrics = self._calculate_metrics(trade_log, params)

        return metrics

    def _build_strategy_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build strategy params dict from optimization params.

        Args:
            params: Optimization parameters

        Returns:
            Strategy params dict compatible with IBStrategy
            Format: {"OCAE": {...}, "TCWE": {...}, "Reverse": {...}, "REV_RB": {...}}
        """
        # Build base variation params
        base_variation = {
            "IB_START": params.get("ib_start", "08:00"),
            "IB_END": params.get("ib_end", "09:00"),
            "IB_TZ": params.get("ib_timezone", self.symbol_config.timezone),
            "IB_WAIT": params.get("ib_wait_minutes", 0),
            "TRADE_WINDOW": params.get("trade_window_minutes", 120),
            "RR_TARGET": params.get("rr_target", 1.0),
            "STOP_MODE": params.get("stop_mode", "ib_start"),
            "TSL_TARGET": params.get("tsl_target", 0.0),
            "TSL_SL": params.get("tsl_sl", 1.0),
            "MIN_SL_PCT": params.get("min_sl_pct", 0.001),
            "IB_BUFFER_PCT": params.get("ib_buffer_pct", 0.0),
            "MAX_DISTANCE_PCT": params.get("max_distance_pct", 1.0),
            "REV_RB_ENABLED": params.get("rev_rb_enabled", False),
            "REV_RB_PCT": params.get("rev_rb_pct", 0.5),  # FIX: was missing
        }

        # Build params dict with all variations
        # Each variation gets the same optimized parameters
        strategy_params = {
            "OCAE": base_variation.copy(),
            "TCWE": base_variation.copy(),
            "Reverse": base_variation.copy(),
            "REV_RB": base_variation.copy(),
        }

        return strategy_params

    def _check_sltp_hits(
        self,
        emulator: MT5Emulator,
        candle: pd.Series,
        symbol_cfg: SymbolConfig
    ) -> None:
        """
        Check if any position's SL/TP was hit during this candle.

        Args:
            emulator: MT5Emulator instance
            candle: Current candle data
            symbol_cfg: Symbol configuration
        """
        positions = emulator.get_open_positions_list()

        for position in positions:
            hit = False
            exit_reason = ""
            exit_price = 0.0

            high = float(candle["high"])
            low = float(candle["low"])

            if position.type == 0:  # POSITION_TYPE_BUY (long)
                if position.sl > 0 and low <= position.sl:
                    hit = True
                    exit_reason = "sl"
                    exit_price = position.sl
                elif position.tp > 0 and high >= position.tp:
                    hit = True
                    exit_reason = "tp"
                    exit_price = position.tp
            else:  # POSITION_TYPE_SELL (short)
                if position.sl > 0 and high >= position.sl:
                    hit = True
                    exit_reason = "sl"
                    exit_price = position.sl
                elif position.tp > 0 and low <= position.tp:
                    hit = True
                    exit_reason = "tp"
                    exit_price = position.tp

            if hit:
                emulator.close_position_by_ticket(
                    position.ticket,
                    price=exit_price,
                    exit_reason=exit_reason
                )

    def _calculate_metrics(
        self,
        trade_log: List,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate performance metrics from trade log.

        Args:
            trade_log: List of TradeLog objects
            params: Original parameters

        Returns:
            Metrics dict
        """
        if not trade_log:
            return {
                "total_r": 0.0,
                "total_profit": 0.0,
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "winrate": 0.0,
                "sharpe_ratio": 0.0,
                "profit_factor": 0.0,
                "max_drawdown": 0.0,
                "avg_trade_r": 0.0,
                "params": params,
                "error": None,
            }

        # Calculate metrics
        total_trades = len(trade_log)
        profits = [t.profit for t in trade_log if t.exit_time]

        winning_trades = sum(1 for p in profits if p > 0)
        losing_trades = sum(1 for p in profits if p < 0)

        total_profit = sum(profits)
        winrate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0

        # Calculate R values (profit relative to risk)
        # R = profit / initial_risk (approximated as 1% of balance per trade)
        risk_per_trade = self.initial_balance * self.risk_pct / 100
        r_values = [p / risk_per_trade for p in profits]
        total_r = sum(r_values)
        avg_trade_r = total_r / total_trades if total_trades > 0 else 0.0

        # Profit factor
        gross_profit = sum(p for p in profits if p > 0)
        gross_loss = abs(sum(p for p in profits if p < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else (
            float("inf") if gross_profit > 0 else 0.0
        )

        # Sharpe ratio (simplified: annualized from daily returns)
        if len(r_values) >= 2:
            import numpy as np
            returns = np.array(r_values)
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0.0
        else:
            sharpe_ratio = 0.0

        # Max drawdown
        equity = self.initial_balance
        peak = equity
        max_dd = 0.0

        for profit in profits:
            equity += profit
            peak = max(peak, equity)
            dd = (peak - equity) / peak if peak > 0 else 0.0
            max_dd = max(max_dd, dd)

        return {
            "total_r": round(total_r, 4),
            "total_profit": round(total_profit, 2),
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "winrate": round(winrate, 2),
            "sharpe_ratio": round(sharpe_ratio, 4),
            "profit_factor": round(profit_factor, 4) if profit_factor != float("inf") else 999.0,
            "max_drawdown": round(max_dd * 100, 2),
            "avg_trade_r": round(avg_trade_r, 4),
            "params": params,
            "error": None,
        }
