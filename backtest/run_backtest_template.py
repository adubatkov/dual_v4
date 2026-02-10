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
from datetime import datetime, timedelta
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
from src.smc.detectors.fractal_detector import detect_fractals

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

    # Pre-compute H1/H4 fractals for the entire period (used for BE logic + chart overlay)
    logger.info("Computing H1/H4 fractals...")
    h1_data = _resample_m1(m1_data, "1h")
    h4_data = _resample_m1(m1_data, "4h")
    all_h1 = detect_fractals(h1_data, symbol, "H1", candle_duration_hours=1.0)
    all_h4 = detect_fractals(h4_data, symbol, "H4", candle_duration_hours=4.0)
    logger.info(f"Detected {len(all_h1)} H1 and {len(all_h4)} H4 fractals")

    # Deduplicate: H4 overrides H1 at same (type, price)
    h4_keys = {(f.type, round(f.price, 2)) for f in all_h4}
    filtered_h1 = [f for f in all_h1 if (f.type, round(f.price, 2)) not in h4_keys]

    # Sort all fractals by confirmed_time for incremental activation
    all_fractals_sorted = sorted(filtered_h1 + all_h4, key=lambda f: f.confirmed_time)

    # Pre-compute M2 fractals for fractal TSL
    logger.info("Computing M2 fractals...")
    m2_data = _resample_m1(m1_data, "2min")
    all_m2 = detect_fractals(m2_data, symbol, "M2", candle_duration_hours=2/60)
    logger.info(f"Detected {len(all_m2)} M2 fractals")
    all_m2_sorted = sorted(all_m2, key=lambda f: f.confirmed_time)

    # Fractal BE tracking state
    fractal_ptr = 0
    active_fractals = []
    fractal_be_count = 0
    tsl_organic_sl = {}        # ticket -> SL as computed by TSL only (no fractal BE)
    fractal_be_active = {}     # ticket -> entry_price (once fractal BE triggered for this position)

    # Fractal TSL tracking state
    m2_fractal_ptr = 0
    last_m2_high = None          # (price, confirmed_time) of most recent M2 high
    last_m2_low = None           # (price, confirmed_time) of most recent M2 low
    fractal_tsl_active = {}      # ticket -> True
    fractal_tsl_prev_sl = {}     # ticket -> previous fractal TSL SL (for logging changes)
    fractal_tsl_count = 0
    fractal_tsl_updates = 0

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

        # --- TSL update (must use TSL-organic SL, not fractal-BE-modified) ---
        positions = executor.get_open_positions()

        # Capture effective SL from end of previous candle (before restoring organic)
        # Used to detect same-candle SL changes and prevent false _check_sltp_hits triggers
        effective_sl_prev = {p.ticket: p.sl for p in positions if p.magic == magic_number}

        # Restore TSL-organic SL before TSL runs (undo fractal BE overlay from prev candle)
        for pos in positions:
            if pos.magic == magic_number and pos.ticket in tsl_organic_sl:
                organic = tsl_organic_sl[pos.ticket]
                if pos.sl != organic:
                    executor.modify_position(pos.ticket, sl=organic, tp=pos.tp)

        # Re-fetch after restore
        positions = executor.get_open_positions()

        # Run TSL
        for position in positions:
            tick = executor.get_tick(symbol)
            if tick:
                strategy.update_position_state(position, tick, current_time_utc)

        # Capture TSL-organic SL (TSL's own result, uncontaminated)
        positions = executor.get_open_positions()
        for pos in positions:
            if pos.magic == magic_number:
                tsl_organic_sl[pos.ticket] = pos.sl

        # --- Fractal activation: add newly confirmed fractals ---
        while (fractal_ptr < len(all_fractals_sorted) and
               all_fractals_sorted[fractal_ptr].confirmed_time <= current_time_utc):
            active_fractals.append(all_fractals_sorted[fractal_ptr])
            fractal_ptr += 1

        # Expire stale fractals (match chart lookback: H1=48h, H4=96h)
        if active_fractals:
            expiry_cutoff_h1 = current_time_utc - timedelta(hours=48)
            expiry_cutoff_h4 = current_time_utc - timedelta(hours=96)
            active_fractals = [
                f for f in active_fractals
                if (f.timeframe == "H1" and f.time >= expiry_cutoff_h1)
                or (f.timeframe == "H4" and f.time >= expiry_cutoff_h4)
            ]

        # --- M2 fractal pointer: activate newly confirmed M2 fractals ---
        while (m2_fractal_ptr < len(all_m2_sorted) and
               all_m2_sorted[m2_fractal_ptr].confirmed_time <= current_time_utc):
            m2f = all_m2_sorted[m2_fractal_ptr]
            if m2f.type == "high":
                last_m2_high = (m2f.price, m2f.confirmed_time)
            else:
                last_m2_low = (m2f.price, m2f.confirmed_time)
            m2_fractal_ptr += 1

        # --- Fractal sweep check + BE trigger ---
        swept_indices = []
        for i, frac in enumerate(active_fractals):
            touched = False
            if frac.type == "high" and row["high"] >= frac.price:
                touched = True
            elif frac.type == "low" and row["low"] <= frac.price:
                touched = True

            if touched:
                swept_indices.append(i)

                our_pos = next((p for p in positions if p.magic == magic_number), None)
                if our_pos is not None:
                    variation = strategy.tsl_state.get("variation") if strategy.tsl_state else None
                    var_params = strategy.params.get(variation, {}) if variation else {}
                    is_long = our_pos.type == 0
                    direction_str = "LONG" if is_long else "SHORT"

                    # --- Fractal BE: move SL to entry (if SL in negative zone) ---
                    if our_pos.ticket not in fractal_be_active:
                        fractal_be_enabled = var_params.get("FRACTAL_BE_ENABLED", False)
                        if fractal_be_enabled:
                            entry = our_pos.price_open
                            organic_sl = tsl_organic_sl.get(our_pos.ticket, our_pos.sl)
                            sl_negative = (is_long and organic_sl < entry) or (not is_long and organic_sl > entry)
                            if sl_negative:
                                fractal_be_active[our_pos.ticket] = entry
                                logger.info(
                                    f"[FRACTAL BE] {current_time_utc.strftime('%Y-%m-%d %H:%M')} UTC | "
                                    f"{symbol} {direction_str} #{our_pos.ticket} | "
                                    f"{frac.timeframe} {frac.type} {frac.price:.2f} swept | "
                                    f"Fractal BE activated (entry={entry:.2f})"
                                )
                                fractal_be_count += 1

                    # --- Fractal TSL: start M2 fractal trailing ---
                    if our_pos.ticket not in fractal_tsl_active:
                        fractal_tsl_enabled = var_params.get("FRACTAL_TSL_ENABLED", False)
                        if fractal_tsl_enabled:
                            m2_ref = last_m2_low if is_long else last_m2_high
                            m2_sl = m2_ref[0] if m2_ref else None
                            fractal_tsl_active[our_pos.ticket] = True
                            fractal_tsl_prev_sl[our_pos.ticket] = m2_sl
                            m2_type = "low" if is_long else "high"
                            m2_sl_str = f"{m2_sl:.2f}" if m2_sl else "none"
                            logger.info(
                                f"[FRACTAL TSL] {current_time_utc.strftime('%Y-%m-%d %H:%M')} UTC | "
                                f"{symbol} {direction_str} #{our_pos.ticket} | "
                                f"{frac.timeframe} {frac.type} {frac.price:.2f} swept | "
                                f"Fractal TSL activated (M2 {m2_type} SL={m2_sl_str})"
                            )
                            fractal_tsl_count += 1

        # Remove swept fractals (reverse order to preserve indices)
        for i in sorted(swept_indices, reverse=True):
            active_fractals.pop(i)

        # --- Apply effective SL: best of (TSL organic, fractal BE, fractal TSL) ---
        positions = executor.get_open_positions()
        for pos in positions:
            if pos.magic == magic_number:
                organic = tsl_organic_sl.get(pos.ticket, pos.sl)
                frac_be = fractal_be_active.get(pos.ticket, None)
                is_long = pos.type == 0

                # Fractal TSL: current M2 fractal SL
                frac_tsl_sl = None
                if pos.ticket in fractal_tsl_active:
                    m2_ref = last_m2_low if is_long else last_m2_high
                    if m2_ref is not None:
                        frac_tsl_sl = m2_ref[0]
                        prev_frac_sl = fractal_tsl_prev_sl.get(pos.ticket)
                        if prev_frac_sl is not None and abs(frac_tsl_sl - prev_frac_sl) > 0.001:
                            m2_type = "low" if is_long else "high"
                            logger.info(
                                f"[FRACTAL TSL] {current_time_utc.strftime('%Y-%m-%d %H:%M')} UTC | "
                                f"{symbol} #{pos.ticket} | M2 {m2_type} SL "
                                f"{prev_frac_sl:.2f} -> {frac_tsl_sl:.2f}"
                            )
                            fractal_tsl_updates += 1
                        fractal_tsl_prev_sl[pos.ticket] = frac_tsl_sl

                # Effective = most favorable of all active systems
                candidates = [organic]
                if frac_be is not None:
                    candidates.append(frac_be)
                if frac_tsl_sl is not None:
                    candidates.append(frac_tsl_sl)

                effective = max(candidates) if is_long else min(candidates)

                if effective != pos.sl:
                    executor.modify_position(pos.ticket, sl=effective, tp=pos.tp)

        # Build sl_override: only for tickets where SL changed THIS candle.
        # For unchanged tickets, _check_sltp_hits uses position.sl (= effective SL).
        # This prevents same-candle exits when BE or TSL just activated.
        sl_override_for_check = {}
        for pos in executor.get_open_positions():
            if pos.magic == magic_number:
                prev = effective_sl_prev.get(pos.ticket)
                if prev is not None and abs(prev - pos.sl) > 0.001:
                    # SL changed this candle (TSL and/or BE) - use previous effective
                    sl_override_for_check[pos.ticket] = prev

        _check_sltp_hits(emulator, row, config.symbols[symbol], sl_override=sl_override_for_check)

        # Clean up tracking for closed positions
        current_tickets = {p.ticket for p in executor.get_open_positions() if p.magic == magic_number}
        for ticket in list(tsl_organic_sl.keys()):
            if ticket not in current_tickets:
                tsl_organic_sl.pop(ticket, None)
                fractal_be_active.pop(ticket, None)
                fractal_tsl_active.pop(ticket, None)
                fractal_tsl_prev_sl.pop(ticket, None)

    # Close remaining positions
    emulator.force_close_all_positions(reason="backtest_end")
    logger.info(f"[FRACTAL BE SUMMARY] {fractal_be_count} fractal BE activations "
                f"out of {trades_executed} total trades")
    logger.info(f"[FRACTAL TSL SUMMARY] {fractal_tsl_count} fractal TSL activations, "
                f"{fractal_tsl_updates} M2 SL updates out of {trades_executed} total trades")

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
            # Reuse pre-computed fractal lists for chart overlay
            fractal_lists = {"all_h1": all_h1, "all_h4": all_h4}

            logger.info("Generating trade charts...")
            charts_count = generate_all_trade_charts(
                trade_log=trade_log,
                m1_data=m1_data,
                output_dir=report_manager.get_trades_dir(),
                timezone=local_tz_name,
                ib_data_by_date=ib_data_by_date,
                fractal_lists=fractal_lists,
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
