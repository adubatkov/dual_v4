"""
BacktestRunner - Main backtest simulation loop.

Orchestrates the backtest process: loads data, runs simulation,
and generates results.
"""

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Callable, Any
from dataclasses import dataclass, field

import pandas as pd
import numpy as np

from .config import BacktestConfig, DATA_FOLDERS, DEFAULT_CONFIG
from .data_processor.data_ingestor import DataIngestor
from .data_processor.tick_generator import TickGenerator, build_tick_data
from .emulator.mt5_emulator import MT5Emulator
from .emulator.time_manager import TimeManager
from .emulator.models import TradeLog

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Results from a backtest run."""

    # Run parameters
    symbol: str
    start_date: datetime
    end_date: datetime
    initial_balance: float

    # Performance
    final_balance: float
    total_profit: float
    total_trades: int
    winning_trades: int
    losing_trades: int

    # Trade log
    trades: List[TradeLog] = field(default_factory=list)

    # Equity curve
    equity_curve: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Timing
    run_duration_seconds: float = 0.0

    @property
    def win_rate(self) -> float:
        """Calculate win rate percentage."""
        if self.total_trades == 0:
            return 0.0
        return (self.winning_trades / self.total_trades) * 100

    @property
    def profit_factor(self) -> float:
        """Calculate profit factor."""
        gross_profit = sum(t.profit for t in self.trades if t.profit > 0)
        gross_loss = abs(sum(t.profit for t in self.trades if t.profit < 0))
        if gross_loss == 0:
            return float("inf") if gross_profit > 0 else 0.0
        return gross_profit / gross_loss

    @property
    def return_pct(self) -> float:
        """Calculate return percentage."""
        if self.initial_balance == 0:
            return 0.0
        return ((self.final_balance - self.initial_balance) / self.initial_balance) * 100


class BacktestRunner:
    """
    Main backtest orchestrator.

    Manages the complete backtest lifecycle:
    1. Load and prepare data
    2. Configure emulator
    3. Run simulation loop
    4. Collect and return results
    """

    def __init__(
        self,
        config: Optional[BacktestConfig] = None,
        strategy_callback: Optional[Callable] = None,
    ):
        """
        Initialize BacktestRunner.

        Args:
            config: Backtest configuration (uses DEFAULT_CONFIG if None)
            strategy_callback: Strategy function to call on each tick/candle
        """
        self.config = config or DEFAULT_CONFIG
        self.strategy_callback = strategy_callback

        # Components
        self.emulator = MT5Emulator()
        self.time_manager = TimeManager(self.emulator)

        # Data storage
        self._m1_data: Dict[str, pd.DataFrame] = {}
        self._tick_data: Dict[str, pd.DataFrame] = {}

        # State
        self._is_prepared = False
        self._equity_snapshots: List[Dict] = []

        logger.info("BacktestRunner initialized")

    def prepare_data(
        self,
        symbols: List[str],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        use_tick_data: bool = True,
        regenerate_ticks: bool = False,
    ) -> None:
        """
        Load and prepare data for backtesting.

        Args:
            symbols: List of symbols to load
            start_date: Optional start date filter
            end_date: Optional end date filter
            use_tick_data: Whether to generate/load tick data
            regenerate_ticks: Force regeneration of tick data
        """
        logger.info(f"Preparing data for symbols: {symbols}")

        for symbol in symbols:
            # Determine data folder
            folder_name = DATA_FOLDERS.get(symbol)
            if not folder_name:
                logger.warning(f"No data folder mapping for {symbol}")
                continue

            data_path = self.config.data_base_path / folder_name

            if not data_path.exists():
                logger.error(f"Data path not found: {data_path}")
                continue

            # Load M1 data
            logger.info(f"Loading M1 data for {symbol} from {data_path}")
            ingestor = DataIngestor(data_path)
            m1_df = ingestor.load_all()

            # Filter date range
            if start_date:
                m1_df = m1_df[m1_df["time"] >= start_date]
            if end_date:
                m1_df = m1_df[m1_df["time"] <= end_date]

            self._m1_data[symbol] = m1_df
            logger.info(f"Loaded {len(m1_df)} M1 candles for {symbol}")

            # Generate or load tick data
            if use_tick_data:
                self._prepare_tick_data(symbol, m1_df, regenerate_ticks)

        # Configure emulator with data
        self.emulator.reset()
        self.emulator.configure(
            config=self.config,
            m1_data=self._m1_data,
            tick_data=self._tick_data if use_tick_data else None,
        )

        # Load individual symbol data into emulator
        for symbol, df in self._m1_data.items():
            self.emulator.load_m1_data(symbol, df)

        if use_tick_data:
            for symbol, df in self._tick_data.items():
                self.emulator.load_tick_data(symbol, df)

        self._is_prepared = True
        logger.info("Data preparation complete")

    def _prepare_tick_data(
        self,
        symbol: str,
        m1_df: pd.DataFrame,
        regenerate: bool = False,
    ) -> None:
        """
        Prepare tick data for a symbol.

        Args:
            symbol: Symbol name
            m1_df: M1 candlestick DataFrame
            regenerate: Force regeneration even if parquet exists
        """
        # Check for existing parquet file
        output_dir = self.config.output_path / "tick_data"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename based on date range
        start_str = m1_df["time"].min().strftime("%Y%m%d")
        end_str = m1_df["time"].max().strftime("%Y%m%d")
        parquet_file = output_dir / f"{symbol}_5s_{start_str}_{end_str}.parquet"

        if parquet_file.exists() and not regenerate:
            logger.info(f"Loading existing tick data from {parquet_file}")
            generator = TickGenerator()
            self._tick_data[symbol] = generator.load_from_parquet(parquet_file)
            return

        # Generate tick data
        logger.info(f"Generating tick data for {symbol}...")
        cfg = self.config.symbols.get(symbol)
        spread = cfg.spread_points if cfg else 0.0

        generator = TickGenerator(
            ticks_per_minute=self.config.ticks_per_minute,
            jitter_factor=self.config.jitter_factor,
            random_seed=self.config.random_seed,
        )

        tick_df = generator.generate_from_m1(m1_df, spread=spread)
        generator.save_to_parquet(tick_df, parquet_file)

        self._tick_data[symbol] = tick_df

    def run(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        mode: str = "candle",  # 'candle' or 'tick'
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> BacktestResult:
        """
        Run backtest simulation.

        Args:
            symbol: Symbol to backtest
            start_date: Backtest start date
            end_date: Backtest end date
            mode: 'candle' for M2 candle-based, 'tick' for 5s tick-based
            progress_callback: Optional callback(current, total) for progress

        Returns:
            BacktestResult with performance metrics
        """
        if not self._is_prepared:
            raise RuntimeError("Data not prepared. Call prepare_data() first.")

        if symbol not in self._m1_data:
            raise ValueError(f"No data loaded for symbol {symbol}")

        logger.info(f"Starting backtest for {symbol} in {mode} mode")
        run_start = datetime.now()

        # Get data
        m1_df = self._m1_data[symbol]

        # Filter date range
        if start_date:
            m1_df = m1_df[m1_df["time"] >= start_date]
        if end_date:
            m1_df = m1_df[m1_df["time"] <= end_date]

        if m1_df.empty:
            raise ValueError("No data in specified date range")

        # Convert M1 to M2 for strategy
        m2_df = self._resample_to_m2(m1_df)

        # Initialize time
        first_time = m2_df["time"].iloc[0]
        self.time_manager.set_start_time(first_time)

        # Reset equity tracking
        self._equity_snapshots = []
        initial_balance = self.config.initial_balance

        # Main simulation loop
        total_candles = len(m2_df)
        logger.info(f"Processing {total_candles} M2 candles...")

        for idx, row in m2_df.iterrows():
            candle_time = row["time"]
            if hasattr(candle_time, "to_pydatetime"):
                candle_time = candle_time.to_pydatetime()

            # Advance time with SL/TP checking
            if mode == "tick" and symbol in self._tick_data:
                # Use tick data for precise SL/TP checking
                tick_df = self._tick_data[symbol]
                next_time = candle_time + timedelta(minutes=2)
                self.time_manager.advance_to(
                    next_time,
                    check_sltp=True,
                    tick_data=tick_df,
                )
            else:
                # Use candle data for SL/TP checking
                self.time_manager.advance_with_candle_check(
                    candle_high=row["high"],
                    candle_low=row["low"],
                    candle_time=candle_time,
                )

            # Call strategy callback if provided
            if self.strategy_callback:
                try:
                    self.strategy_callback(
                        symbol=symbol,
                        current_time=candle_time,
                        candle=row,
                    )
                except Exception as e:
                    logger.error(f"Strategy callback error: {e}")

            # Record equity snapshot
            equity = self.emulator.get_equity_at_time()
            self._equity_snapshots.append({
                "time": candle_time,
                "equity": equity,
                "balance": self.emulator._account.balance,
            })

            # Progress callback
            if progress_callback and idx % 100 == 0:
                progress_callback(idx, total_candles)

        # Force close any remaining positions
        self.emulator.force_close_all_positions(reason="backtest_end")

        # Compile results
        run_end = datetime.now()
        run_duration = (run_end - run_start).total_seconds()

        trades = self.emulator.get_trade_log()
        final_balance = self.emulator._account.balance

        winning = [t for t in trades if t.profit > 0]
        losing = [t for t in trades if t.profit < 0]

        result = BacktestResult(
            symbol=symbol,
            start_date=m2_df["time"].iloc[0],
            end_date=m2_df["time"].iloc[-1],
            initial_balance=initial_balance,
            final_balance=final_balance,
            total_profit=final_balance - initial_balance,
            total_trades=len(trades),
            winning_trades=len(winning),
            losing_trades=len(losing),
            trades=trades,
            equity_curve=pd.DataFrame(self._equity_snapshots),
            run_duration_seconds=run_duration,
        )

        logger.info(f"Backtest complete: {result.total_trades} trades, "
                   f"profit={result.total_profit:.2f}, "
                   f"win_rate={result.win_rate:.1f}%")

        return result

    def _resample_to_m2(self, m1_df: pd.DataFrame) -> pd.DataFrame:
        """Resample M1 data to M2 timeframe."""
        df = m1_df.set_index("time")

        m2 = df.resample("2min").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }).dropna()

        return m2.reset_index()

    def run_with_bot_integration(
        self,
        symbol: str,
        strategy_instance: Any,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> BacktestResult:
        """
        Run backtest with actual bot strategy integration.

        This method integrates with the real IBStrategy class by:
        1. Replacing mt5 module import with emulator
        2. Calling strategy's check_signal() method
        3. Managing positions through the strategy

        Args:
            symbol: Symbol to backtest
            strategy_instance: Initialized strategy instance
            start_date: Backtest start date
            end_date: Backtest end date
            progress_callback: Progress callback

        Returns:
            BacktestResult
        """
        if not self._is_prepared:
            raise RuntimeError("Data not prepared. Call prepare_data() first.")

        logger.info(f"Starting integrated backtest for {symbol}")
        run_start = datetime.now()

        m1_df = self._m1_data[symbol]

        if start_date:
            m1_df = m1_df[m1_df["time"] >= start_date]
        if end_date:
            m1_df = m1_df[m1_df["time"] <= end_date]

        m2_df = self._resample_to_m2(m1_df)

        first_time = m2_df["time"].iloc[0]
        self.time_manager.set_start_time(first_time)

        self._equity_snapshots = []
        initial_balance = self.config.initial_balance
        total_candles = len(m2_df)

        for idx, row in m2_df.iterrows():
            candle_time = row["time"]
            if hasattr(candle_time, "to_pydatetime"):
                candle_time = candle_time.to_pydatetime()

            # Advance time with SL/TP check
            self.time_manager.advance_with_candle_check(
                candle_high=row["high"],
                candle_low=row["low"],
                candle_time=candle_time,
            )

            # Call strategy's check_signal
            try:
                signal = strategy_instance.check_signal(candle_time)

                if signal:
                    # Execute signal through emulator
                    self._execute_signal(symbol, signal, strategy_instance)

                # Update TSL for open positions
                positions = self.emulator.positions_get(symbol=symbol)
                if positions:
                    tick = self.emulator.symbol_info_tick(symbol)
                    if tick:
                        tick_dict = {
                            "bid": tick.bid,
                            "ask": tick.ask,
                            "last": tick.last,
                        }
                        for pos in positions:
                            strategy_instance.update_position_state(pos, tick_dict)

            except Exception as e:
                logger.error(f"Strategy error at {candle_time}: {e}")

            # Record equity
            equity = self.emulator.get_equity_at_time()
            self._equity_snapshots.append({
                "time": candle_time,
                "equity": equity,
                "balance": self.emulator._account.balance,
            })

            if progress_callback and idx % 100 == 0:
                progress_callback(idx, total_candles)

        # Close remaining positions
        self.emulator.force_close_all_positions(reason="backtest_end")

        run_end = datetime.now()
        run_duration = (run_end - run_start).total_seconds()

        trades = self.emulator.get_trade_log()
        final_balance = self.emulator._account.balance

        winning = [t for t in trades if t.profit > 0]
        losing = [t for t in trades if t.profit < 0]

        return BacktestResult(
            symbol=symbol,
            start_date=m2_df["time"].iloc[0],
            end_date=m2_df["time"].iloc[-1],
            initial_balance=initial_balance,
            final_balance=final_balance,
            total_profit=final_balance - initial_balance,
            total_trades=len(trades),
            winning_trades=len(winning),
            losing_trades=len(losing),
            trades=trades,
            equity_curve=pd.DataFrame(self._equity_snapshots),
            run_duration_seconds=run_duration,
        )

    def _execute_signal(
        self,
        symbol: str,
        signal: Any,
        strategy_instance: Any,
    ) -> None:
        """
        Execute a signal through the emulator.

        Args:
            symbol: Trading symbol
            signal: Signal object from strategy
            strategy_instance: Strategy instance for risk calculation
        """
        from .emulator.models import OrderType, TradeAction

        # Calculate lot size (simplified)
        # In production, would use RiskManager
        lot_size = 0.1  # Default lot size

        # Create order request
        order_type = (
            OrderType.ORDER_TYPE_BUY
            if signal.direction == "long"
            else OrderType.ORDER_TYPE_SELL
        )

        request = {
            "action": TradeAction.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot_size,
            "type": order_type,
            "price": signal.entry_price,
            "sl": signal.stop_loss,
            "tp": signal.take_profit,
            "magic": strategy_instance.magic_number,
            "comment": signal.comment,
        }

        result = self.emulator.order_send(request)

        if result.retcode == 10009:  # TRADE_RETCODE_DONE
            logger.info(f"Signal executed: {signal.direction} {symbol} @ {signal.entry_price}")
        else:
            logger.warning(f"Signal execution failed: {result.comment}")


def run_quick_backtest(
    symbol: str,
    start_date: datetime,
    end_date: datetime,
    initial_balance: float = 50000.0,
) -> BacktestResult:
    """
    Quick backtest function for simple testing.

    Args:
        symbol: Symbol to backtest
        start_date: Start date
        end_date: End date
        initial_balance: Starting balance

    Returns:
        BacktestResult
    """
    config = BacktestConfig(initial_balance=initial_balance)
    runner = BacktestRunner(config=config)

    runner.prepare_data(
        symbols=[symbol],
        start_date=start_date,
        end_date=end_date,
        use_tick_data=False,  # Faster without tick generation
    )

    return runner.run(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        mode="candle",
    )
