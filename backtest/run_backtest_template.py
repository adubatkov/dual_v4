"""
Parameterized Backtest Template for IB Strategy.

Unified backtest runner supporting GER40 and XAUUSD with configurable parameters,
date ranges, risk modes, and output options via command-line arguments.

Usage Examples:
    # GER40 with PROD params, default date range, fixed $1000 risk:
    python run_backtest_template.py --symbol GER40

    # XAUUSD with V8 params, custom dates:
    python run_backtest_template.py --symbol XAUUSD --params v8 --start 2024-01-01 --end 2025-06-30

    # GER40 with custom JSON params, percentage-based risk:
    python run_backtest_template.py --symbol GER40 --params-file my_params.json --risk-mode pct --risk-value 2.0

    # XAUUSD with PROD params, skip chart generation, custom output name:
    python run_backtest_template.py --symbol XAUUSD --params prod --no-charts --output-name xauusd_test_run

    # GER40 with V8 params and custom risk amount:
    python run_backtest_template.py --symbol GER40 --params v8 --risk-value 500
"""

import sys
import json
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pytz
import pandas as pd

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backtest.backtest_runner import BacktestRunner
from backtest.config import BacktestConfig, SymbolConfig
from backtest.adapter import BacktestExecutor, create_mt5_patch_module
from backtest.emulator.mt5_emulator import MT5Emulator
from backtest.analysis.metrics import PerformanceMetrics
from backtest.analysis.visualizer import Visualizer
from backtest.risk_manager import BacktestRiskManager
from backtest.reporting import BacktestReportManager, ExcelReportGenerator
from backtest.reporting.trade_charts import generate_all_trade_charts

# Import PROD parameters from strategy_logic
from src.utils.strategy_logic import GER40_PARAMS_PROD, XAUUSD_PARAMS_PROD
from src.smc.detectors.fractal_detector import detect_fractals, find_unswept_fractals

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _resample_m1(m1_data: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Resample M1 data to a higher timeframe (e.g., '1h', '4h')."""
    df = m1_data[["time", "open", "high", "low", "close"]].copy()
    df = df.set_index("time")
    resampled = df.resample(freq).agg({
        "open": "first", "high": "max", "low": "min", "close": "last"
    }).dropna().reset_index()
    return resampled


# ============================================================
# V8 Parameter Sets (inline, from DB Optimizer - 2026-01-17)
# ============================================================

GER40_PARAMS_V8 = {
    "REV_RB": {
        "IB_START": "08:00",
        "IB_END": "08:30",
        "IB_TZ": "Europe/Berlin",
        "IB_WAIT": 20,
        "TRADE_WINDOW": 210,
        "RR_TARGET": 1.5,
        "STOP_MODE": "eq",
        "TSL_TARGET": 2.0,
        "TSL_SL": 1.5,
        "MIN_SL_PCT": 0.0015,
        "REV_RB_PCT": 1.0,
        "REV_RB_ENABLED": True,
        "IB_BUFFER_PCT": 0.2,
        "MAX_DISTANCE_PCT": 0.5,
    },
    "Reverse": {
        "IB_START": "08:00",
        "IB_END": "08:30",
        "IB_TZ": "Europe/Berlin",
        "IB_WAIT": 20,
        "TRADE_WINDOW": 210,
        "RR_TARGET": 1.0,
        "STOP_MODE": "ib_start",
        "TSL_TARGET": 1.5,
        "TSL_SL": 0.75,
        "MIN_SL_PCT": 0.0015,
        "REV_RB_ENABLED": False,
        "IB_BUFFER_PCT": 0.2,
        "MAX_DISTANCE_PCT": 0.5,
    },
    "TCWE": {
        "IB_START": "08:00",
        "IB_END": "08:30",
        "IB_TZ": "Europe/Berlin",
        "IB_WAIT": 20,
        "TRADE_WINDOW": 60,
        "RR_TARGET": 0.75,
        "STOP_MODE": "ib_start",
        "TSL_TARGET": 2.0,
        "TSL_SL": 2.0,
        "MIN_SL_PCT": 0.0015,
        "REV_RB_ENABLED": False,
        "IB_BUFFER_PCT": 0.2,
        "MAX_DISTANCE_PCT": 0.5,
    },
    "OCAE": {
        "IB_START": "08:00",
        "IB_END": "08:30",
        "IB_TZ": "Europe/Berlin",
        "IB_WAIT": 20,
        "TRADE_WINDOW": 210,
        "RR_TARGET": 0.5,
        "STOP_MODE": "eq",
        "TSL_TARGET": 2.0,
        "TSL_SL": 2.0,
        "MIN_SL_PCT": 0.0015,
        "REV_RB_ENABLED": False,
        "IB_BUFFER_PCT": 0.2,
        "MAX_DISTANCE_PCT": 0.5,
    },
}

XAUUSD_PARAMS_V8 = {
    "REV_RB": {
        "IB_START": "09:00",
        "IB_END": "09:30",
        "IB_TZ": "Asia/Tokyo",
        "IB_WAIT": 20,
        "TRADE_WINDOW": 90,
        "RR_TARGET": 1.0,
        "STOP_MODE": "ib_start",
        "TSL_TARGET": 2.0,
        "TSL_SL": 2.0,
        "MIN_SL_PCT": 0.001,
        "REV_RB_PCT": 1.0,
        "REV_RB_ENABLED": False,
        "IB_BUFFER_PCT": 0.05,
        "MAX_DISTANCE_PCT": 1.5,
    },
    "Reverse": {
        "IB_START": "09:00",
        "IB_END": "09:30",
        "IB_TZ": "Asia/Tokyo",
        "IB_WAIT": 20,
        "TRADE_WINDOW": 180,
        "RR_TARGET": 1.5,
        "STOP_MODE": "ib_start",
        "TSL_TARGET": 1.0,
        "TSL_SL": 1.0,
        "MIN_SL_PCT": 0.001,
        "REV_RB_ENABLED": False,
        "IB_BUFFER_PCT": 0.05,
        "MAX_DISTANCE_PCT": 1.5,
    },
    "TCWE": {
        "IB_START": "09:00",
        "IB_END": "09:30",
        "IB_TZ": "Asia/Tokyo",
        "IB_WAIT": 20,
        "TRADE_WINDOW": 240,
        "RR_TARGET": 0.5,
        "STOP_MODE": "ib_start",
        "TSL_TARGET": 2.0,
        "TSL_SL": 2.0,
        "MIN_SL_PCT": 0.001,
        "REV_RB_ENABLED": False,
        "IB_BUFFER_PCT": 0.05,
        "MAX_DISTANCE_PCT": 1.5,
    },
    "OCAE": {
        "IB_START": "09:00",
        "IB_END": "09:30",
        "IB_TZ": "Asia/Tokyo",
        "IB_WAIT": 20,
        "TRADE_WINDOW": 120,
        "RR_TARGET": 0.75,
        "STOP_MODE": "eq",
        "TSL_TARGET": 2.0,
        "TSL_SL": 2.0,
        "MIN_SL_PCT": 0.001,
        "REV_RB_ENABLED": False,
        "IB_BUFFER_PCT": 0.05,
        "MAX_DISTANCE_PCT": 1.5,
    },
}

# ============================================================
# Symbol Configuration Registry
# ============================================================

SYMBOL_CONFIGS: Dict[str, dict] = {
    "GER40": {
        "symbol_config": SymbolConfig(
            name="GER40",
            spread_points=1.5,
            digits=2,
            volume_step=0.01,
            trade_tick_size=0.5,
            trade_tick_value=0.5,
            trade_contract_size=1.0,
        ),
        "timezone": "Europe/Berlin",
        "magic_number_base": 1000,
    },
    "XAUUSD": {
        "symbol_config": SymbolConfig(
            name="XAUUSD",
            spread_points=0.30,
            digits=2,
            volume_step=0.01,
            trade_tick_size=0.01,
            trade_tick_value=1.0,
            trade_contract_size=100.0,
        ),
        "timezone": "Asia/Tokyo",
        "magic_number_base": 2000,
    },
}

# Parameter set version suffix for magic number differentiation
PARAMS_MAGIC_SUFFIX = {
    "prod": 9,
    "v8": 8,
    "custom": 7,
}

# Built-in parameter registry: (symbol, params_name) -> params dict
BUILTIN_PARAMS = {
    ("GER40", "v8"): GER40_PARAMS_V8,
    ("GER40", "prod"): GER40_PARAMS_PROD,
    ("XAUUSD", "v8"): XAUUSD_PARAMS_V8,
    ("XAUUSD", "prod"): XAUUSD_PARAMS_PROD,
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Parameterized IB Strategy Backtest Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_backtest_template.py --symbol GER40
  python run_backtest_template.py --symbol XAUUSD --params v8 --start 2024-01-01
  python run_backtest_template.py --symbol GER40 --params-file custom.json --risk-mode pct --risk-value 2.0
  python run_backtest_template.py --symbol XAUUSD --no-charts --output-name quick_test
        """,
    )

    parser.add_argument(
        "--symbol",
        type=str,
        required=True,
        choices=["GER40", "XAUUSD"],
        help="Trading symbol (required)",
    )
    parser.add_argument(
        "--params",
        type=str,
        default="prod",
        choices=["prod", "v8"],
        help="Built-in parameter set to use (default: prod)",
    )
    parser.add_argument(
        "--params-file",
        type=str,
        default=None,
        help="Path to custom params JSON file (overrides --params)",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2023-01-01",
        help="Backtest start date, YYYY-MM-DD (default: 2023-01-01)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default="2025-10-31",
        help="Backtest end date, YYYY-MM-DD (default: 2025-10-31)",
    )
    parser.add_argument(
        "--risk-mode",
        type=str,
        default="fixed",
        choices=["fixed", "pct"],
        help="Risk mode: 'fixed' for fixed dollar amount, 'pct' for percentage of balance (default: fixed)",
    )
    parser.add_argument(
        "--risk-value",
        type=float,
        default=1000.0,
        help="Risk value: dollar amount if fixed, percentage if pct (default: 1000)",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Custom output directory name (auto-generated if not given)",
    )
    parser.add_argument(
        "--no-charts",
        action="store_true",
        default=False,
        help="Skip trade chart generation (faster runs)",
    )
    parser.add_argument(
        "--initial-balance",
        type=float,
        default=100000.0,
        help="Starting account balance (default: 100000)",
    )
    parser.add_argument(
        "--max-margin-pct",
        type=float,
        default=40.0,
        help="Maximum margin usage percentage (default: 40)",
    )

    return parser.parse_args()


def load_params_from_file(filepath: str) -> dict:
    """
    Load strategy parameters from a JSON file.

    Expected JSON structure (same as built-in params):
    {
        "REV_RB": {"IB_START": "08:00", "IB_END": "08:30", ...},
        "Reverse": {...},
        "TCWE": {...},
        "OCAE": {...}
    }

    Args:
        filepath: Path to the JSON params file.

    Returns:
        Dictionary of variation parameters.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Params file not found: {filepath}")

    with open(path, "r", encoding="utf-8") as f:
        params = json.load(f)

    logger.info(f"Loaded custom params from: {filepath}")
    logger.info(f"  Variations found: {list(params.keys())}")
    return params


def resolve_params(symbol: str, params_name: str, params_file: Optional[str]) -> tuple:
    """
    Resolve which parameter set to use.

    Args:
        symbol: Trading symbol.
        params_name: Built-in params name ('prod' or 'v8').
        params_file: Optional path to custom JSON params file.

    Returns:
        Tuple of (params_dict, params_label) where params_label is used for output naming.
    """
    if params_file is not None:
        params = load_params_from_file(params_file)
        label = Path(params_file).stem
        return params, label

    key = (symbol, params_name)
    if key not in BUILTIN_PARAMS:
        raise ValueError(f"No built-in params for {symbol}/{params_name}")

    return BUILTIN_PARAMS[key], params_name


def generate_output_name(
    symbol: str,
    params_label: str,
    risk_mode: str,
    risk_value: float,
    custom_name: Optional[str] = None,
) -> str:
    """
    Generate output directory name with timestamp and run metadata.

    Args:
        symbol: Trading symbol.
        params_label: Parameter set label (e.g., 'prod', 'v8', custom filename stem).
        risk_mode: Risk mode ('fixed' or 'pct').
        risk_value: Risk value.
        custom_name: User-provided custom name (used as-is if given).

    Returns:
        Output directory name string.
    """
    if custom_name:
        return custom_name

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    risk_tag = f"fixed{int(risk_value)}" if risk_mode == "fixed" else f"pct{risk_value}"
    return f"{symbol}_{params_label}_{risk_tag}_{timestamp}"


def patch_mt5_module(emulator: MT5Emulator) -> None:
    """Patch MetaTrader5 module to use emulator."""
    mt5_patch = create_mt5_patch_module(emulator)
    sys.modules["MetaTrader5"] = mt5_patch
    logger.info("MetaTrader5 module patched to use emulator")


def _check_sltp_hits(
    emulator: MT5Emulator,
    candle: pd.Series,
    symbol_cfg: SymbolConfig,
    sl_override: Optional[dict] = None,
) -> None:
    """Check if any position's SL/TP was hit during this candle."""
    if sl_override is None:
        sl_override = {}

    positions = emulator.get_open_positions_list()

    for position in positions:
        hit = False
        exit_reason = ""
        exit_price = 0.0

        high = float(candle["high"])
        low = float(candle["low"])
        check_sl = sl_override.get(position.ticket, position.sl)

        if position.type == 0:  # Long
            if check_sl > 0 and low <= check_sl:
                hit = True
                exit_reason = "sl"
                exit_price = check_sl if high >= check_sl else high
            elif position.tp > 0 and high >= position.tp:
                hit = True
                exit_reason = "tp"
                exit_price = position.tp
        else:  # Short
            if check_sl > 0 and high >= check_sl:
                hit = True
                exit_reason = "sl"
                exit_price = check_sl if low <= check_sl else low
            elif position.tp > 0 and low <= position.tp:
                hit = True
                exit_reason = "tp"
                exit_price = position.tp

        if hit:
            emulator.close_position_by_ticket(
                position.ticket, price=exit_price, exit_reason=exit_reason
            )


def _calculate_equity_curve(trade_log: List, initial_balance: float) -> pd.DataFrame:
    """Calculate equity curve from trade log."""
    first_entry = None
    for trade in trade_log:
        if trade.entry_time:
            first_entry = trade.entry_time
            break

    equity_points = [{"time": first_entry, "equity": initial_balance, "drawdown": 0.0}]
    balance = initial_balance
    peak = initial_balance

    for trade in trade_log:
        if trade.exit_time:
            balance += trade.profit
            peak = max(peak, balance)
            drawdown = (peak - balance) / peak if peak > 0 else 0
            equity_points.append(
                {"time": trade.exit_time, "equity": balance, "drawdown": drawdown}
            )

    return pd.DataFrame(equity_points)


def run_backtest(
    symbol: str,
    params: dict,
    params_label: str,
    start_date: datetime,
    end_date: datetime,
    risk_mode: str = "fixed",
    risk_value: float = 1000.0,
    initial_balance: float = 100000.0,
    max_margin_pct: float = 40.0,
    output_name: Optional[str] = None,
    generate_charts: bool = True,
) -> None:
    """
    Run a parameterized backtest for the given symbol and configuration.

    Args:
        symbol: Trading symbol ('GER40' or 'XAUUSD').
        params: Strategy parameter dictionary (variation -> settings).
        params_label: Human-readable label for the parameter set.
        start_date: Backtest start date (UTC).
        end_date: Backtest end date (UTC).
        risk_mode: 'fixed' for fixed dollar risk, 'pct' for percentage of balance.
        risk_value: Risk amount ($) or percentage depending on risk_mode.
        initial_balance: Starting account balance.
        max_margin_pct: Maximum margin usage percentage.
        output_name: Custom output directory name (auto-generated if None).
        generate_charts: Whether to generate individual trade charts.
    """
    # Resolve symbol configuration
    sym_info = SYMBOL_CONFIGS[symbol]
    symbol_config = sym_info["symbol_config"]
    local_tz_name = sym_info["timezone"]
    magic_number = sym_info["magic_number_base"] + PARAMS_MAGIC_SUFFIX.get(params_label, 7)

    # Determine risk description
    if risk_mode == "fixed":
        risk_desc = f"${risk_value:,.0f} (FIXED)"
        risk_amount = risk_value
        risk_pct = None
    else:
        risk_desc = f"{risk_value}% of balance"
        risk_amount = initial_balance * (risk_value / 100.0)
        risk_pct = risk_value

    logger.info("=" * 60)
    logger.info(f"Starting {symbol} Backtest - params={params_label}")
    logger.info(f"Period: {start_date.date()} to {end_date.date()}")
    logger.info(f"Initial Balance: ${initial_balance:,.0f}")
    logger.info(f"Risk per trade: {risk_desc}")
    logger.info(f"Max margin: {max_margin_pct}%")
    logger.info(f"Magic number: {magic_number}")
    logger.info(f"Charts: {'enabled' if generate_charts else 'disabled'}")
    logger.info("=" * 60)

    # Configure backtest
    config = BacktestConfig(
        initial_balance=initial_balance,
        leverage=100,
        symbols={symbol: symbol_config},
    )

    # Create and configure emulator
    emulator = MT5Emulator()
    emulator.reset()
    emulator.configure(config)

    # Patch MT5 module BEFORE importing IBStrategy
    patch_mt5_module(emulator)

    # Now import strategy (will use patched MT5)
    from src.strategies.ib_strategy import IBStrategy

    # Create executor
    executor = BacktestExecutor(emulator, config)
    executor.connect(12345, "password", "BacktestServer")

    # Create runner and prepare data
    runner = BacktestRunner(config=config)

    logger.info("Preparing data...")
    runner.prepare_data(
        symbols=[symbol],
        start_date=start_date,
        end_date=end_date,
        use_tick_data=False,
    )

    # Load M1 data into emulator
    if runner._m1_data and symbol in runner._m1_data:
        emulator.load_m1_data(symbol, runner._m1_data[symbol])
        logger.info(
            f"Loaded {len(runner._m1_data[symbol])} M1 candles into emulator"
        )
    else:
        logger.error(f"No M1 data available for {symbol}")
        return

    # Create risk manager
    risk_manager = BacktestRiskManager(
        emulator=emulator,
        risk_amount=risk_amount,
        max_margin_pct=max_margin_pct,
    )

    # Create strategy
    strategy = IBStrategy(
        symbol=symbol,
        params=params,
        executor=executor,
        magic_number=magic_number,
        strategy_label=f"Backtest_{params_label}",
        news_filter_enabled=True,
    )

    # Log parameter summary
    logger.info(f"Parameters ({params_label}):")
    for var_name, var_params in params.items():
        ib_info = f"IB: {var_params.get('IB_START', '?')}-{var_params.get('IB_END', '?')} ({var_params.get('IB_TZ', '?')})"
        rr_info = f"RR={var_params.get('RR_TARGET', '?')}"
        stop_info = f"Stop={var_params.get('STOP_MODE', '?')}"
        rev_rb = var_params.get("REV_RB_ENABLED", False)
        logger.info(f"  {var_name}: {ib_info}, {rr_info}, {stop_info}, REV_RB={rev_rb}")

    if strategy.news_filter is not None:
        logger.info(
            f"  NEWS_FILTER: ENABLED ({strategy.news_filter.event_count} events loaded)"
        )

    # Run backtest loop
    logger.info("Running backtest loop...")

    m1_data = runner._m1_data[symbol]
    local_tz = pytz.timezone(local_tz_name)

    trades_executed = 0
    signals_detected = 0
    days_processed = set()
    ib_data_by_date = {}

    m1_data["date"] = m1_data["time"].dt.date

    for idx, row in m1_data.iterrows():
        current_time_utc = row["time"].to_pydatetime()
        emulator.set_time(current_time_utc)

        local_time = current_time_utc.astimezone(local_tz)
        current_date = local_time.date()

        if current_date not in days_processed:
            days_processed.add(current_date)
            if len(days_processed) % 50 == 0:
                logger.info(f"Processing day {len(days_processed)}: {current_date}")

        positions = executor.get_open_positions()
        has_position = any(p.magic == magic_number for p in positions)

        if not has_position:
            signal = strategy.check_signal(current_time_utc)

            if signal:
                signals_detected += 1

                date_key = current_date.strftime("%Y-%m-%d")
                if date_key not in ib_data_by_date and strategy.ibh is not None:
                    ib_data_by_date[date_key] = {
                        "ibh": strategy.ibh,
                        "ibl": strategy.ibl,
                        "eq": strategy.eq,
                        "ib_start": strategy.ib_start,
                        "ib_end": strategy.ib_end,
                        "ib_tz": str(strategy.ib_tz),
                    }

                lots = risk_manager.calculate_position_size(
                    symbol=symbol,
                    entry_price=signal.entry_price,
                    stop_loss=signal.stop_loss,
                )

                if lots <= 0:
                    continue

                validation = risk_manager.validate_trade(symbol, lots, signal.entry_price)
                if not validation["valid"]:
                    logger.warning(f"Trade validation failed: {validation['reason']}")
                    continue

                result = executor.place_order(symbol, signal, lots, magic_number)

                if result["success"]:
                    trades_executed += 1
                    strategy.state = "POSITION_OPEN"

        # TSL update
        positions = executor.get_open_positions()
        sl_before_tsl = {p.ticket: p.sl for p in positions}

        for position in positions:
            tick = executor.get_tick(symbol)
            if tick:
                strategy.update_position_state(position, tick, current_time_utc)

        _check_sltp_hits(emulator, row, config.symbols[symbol], sl_override=sl_before_tsl)

    # Close remaining positions
    emulator.force_close_all_positions(reason="backtest_end")

    # Get results
    trade_log = emulator.get_trade_log()

    logger.info("=" * 60)
    logger.info("Backtest Complete")
    logger.info(f"Days Processed: {len(days_processed)}")
    logger.info(f"Signals Detected: {signals_detected}")
    logger.info(f"Trades Executed: {trades_executed}")
    logger.info(f"Total Trades: {len(trade_log)}")
    logger.info("=" * 60)

    if trade_log:
        equity_curve = _calculate_equity_curve(trade_log, initial_balance)

        metrics = PerformanceMetrics(
            trades=trade_log,
            equity_curve=equity_curve,
            initial_balance=initial_balance,
        )
        report = metrics.calculate_all()

        # Calculate R values
        total_r = sum(t.profit for t in trade_log if t.exit_time) / risk_amount
        wins = sum(1 for t in trade_log if t.exit_time and t.profit > 0)
        losses = sum(1 for t in trade_log if t.exit_time and t.profit < 0)
        winrate = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0

        print("\n" + "=" * 60)
        print(f"PERFORMANCE REPORT - {symbol} {params_label} ({risk_desc})")
        print("=" * 60)
        print(f"Total Trades: {len(trade_log)}")
        print(f"Wins: {wins}, Losses: {losses}")
        print(f"Win Rate: {winrate:.1f}%")
        print(f"Total R: {total_r:.2f} (at ${risk_amount:,.0f}/trade)")
        print(f"Total P/L: ${sum(t.profit for t in trade_log if t.exit_time):,.2f}")
        print("=" * 60)
        print(report.summary())

        # Generate output name
        out_name = generate_output_name(
            symbol=symbol,
            params_label=params_label,
            risk_mode=risk_mode,
            risk_value=risk_value,
            custom_name=output_name,
        )

        # Create report
        report_manager = BacktestReportManager(
            base_output_path=Path(__file__).parent / "output",
            symbol=symbol,
            backtest_name=out_name,
        )

        report_manager.save_config({
            "symbol": symbol,
            "config_version": f"{params_label} (template runner)",
            "start_date": start_date,
            "end_date": end_date,
            "initial_balance": initial_balance,
            "risk_amount": risk_amount,
            "risk_mode": risk_mode.upper(),
            "risk_value": risk_value,
            "max_margin_pct": max_margin_pct,
            "total_trades": len(trade_log),
            "total_r": total_r,
            "magic_number": magic_number,
            "params": params,
        })

        report_manager.save_summary(
            metrics_report=report,
            trade_count=len(trade_log),
            start_date=start_date,
            end_date=end_date,
            initial_balance=initial_balance,
            risk_pct=risk_pct,
            max_margin_pct=max_margin_pct,
        )

        excel_gen = ExcelReportGenerator()
        excel_gen.generate_report(
            trade_log=trade_log,
            metrics_report=report,
            output_path=report_manager.get_output_dir(),
            symbol=symbol,
            initial_balance=initial_balance,
            ib_data=ib_data_by_date,
        )

        viz = Visualizer()
        viz.plot_equity_and_drawdown(
            equity_curve,
            title=f"{symbol} Equity Curve - {params_label} ({risk_desc})",
            save_path=report_manager.get_output_dir() / "equity_drawdown.png",
        )

        if generate_charts:
            # Compute H1/H4 fractals for chart overlay
            fractal_data_by_date = {}
            if ib_data_by_date:
                logger.info("Computing H1/H4 fractals for chart overlay...")
                h1_data = _resample_m1(m1_data, "1h")
                h4_data = _resample_m1(m1_data, "4h")

                all_h1 = detect_fractals(h1_data, symbol, "H1", candle_duration_hours=1.0)
                all_h4 = detect_fractals(h4_data, symbol, "H4", candle_duration_hours=4.0)
                logger.info(f"Detected {len(all_h1)} H1 and {len(all_h4)} H4 fractals")

                for date_key, ib_info in ib_data_by_date.items():
                    ib_start_str = ib_info.get("ib_start", "08:00")
                    ib_tz_str = ib_info.get("ib_tz", "Europe/Berlin")
                    trade_date = datetime.strptime(date_key, "%Y-%m-%d").date()
                    ib_tz = pytz.timezone(ib_tz_str)
                    ib_start_local = ib_tz.localize(
                        datetime.combine(trade_date, datetime.strptime(ib_start_str, "%H:%M").time())
                    )
                    ib_start_utc = ib_start_local.astimezone(pytz.UTC)

                    unswept_h1 = find_unswept_fractals(all_h1, m1_data, ib_start_utc, lookback_hours=48)
                    unswept_h4 = find_unswept_fractals(all_h4, m1_data, ib_start_utc, lookback_hours=96)

                    # Deduplicate: remove H1 fractals that overlap with H4 at same price+type
                    h4_prices = {(f.type, round(f.price, 2)) for f in unswept_h4}
                    filtered_h1 = [f for f in unswept_h1 if (f.type, round(f.price, 2)) not in h4_prices]

                    fractal_data_by_date[date_key] = {"h1": filtered_h1, "h4": unswept_h4}

            logger.info("Generating trade charts...")
            charts_count = generate_all_trade_charts(
                trade_log=trade_log,
                m1_data=m1_data,
                output_dir=report_manager.get_trades_dir(),
                timezone=local_tz_name,
                ib_data_by_date=ib_data_by_date,
                fractal_data_by_date=fractal_data_by_date,
            )
            logger.info(f"Generated {charts_count} trade charts")
        else:
            logger.info("Trade chart generation skipped (--no-charts)")

        print(f"\nFull report saved to: {report_manager.get_output_dir()}")
    else:
        logger.warning("No trades executed during backtest period")


def main() -> None:
    """Entry point: parse CLI arguments and run backtest."""
    args = parse_args()

    # Parse dates
    try:
        start_date = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=pytz.UTC)
    except ValueError:
        logger.error(f"Invalid start date format: {args.start} (expected YYYY-MM-DD)")
        sys.exit(1)

    try:
        end_date = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=pytz.UTC)
    except ValueError:
        logger.error(f"Invalid end date format: {args.end} (expected YYYY-MM-DD)")
        sys.exit(1)

    if start_date >= end_date:
        logger.error(f"Start date ({args.start}) must be before end date ({args.end})")
        sys.exit(1)

    # Resolve parameters
    params, params_label = resolve_params(args.symbol, args.params, args.params_file)

    # Run backtest
    run_backtest(
        symbol=args.symbol,
        params=params,
        params_label=params_label,
        start_date=start_date,
        end_date=end_date,
        risk_mode=args.risk_mode,
        risk_value=args.risk_value,
        initial_balance=args.initial_balance,
        max_margin_pct=args.max_margin_pct,
        output_name=args.output_name,
        generate_charts=not args.no_charts,
    )


if __name__ == "__main__":
    main()
