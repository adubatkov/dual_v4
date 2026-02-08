"""
Single Group Backtest Worker

Runs slow backtest for a single parameter group.
Returns metrics without generating trade charts.

Used by run_parallel_backtest.py as worker function.
"""

import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

import pytz
import pandas as pd

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backtest.config import BacktestConfig, SymbolConfig
from backtest.adapter import BacktestExecutor, create_mt5_patch_module
from backtest.emulator.mt5_emulator import MT5Emulator
from backtest.analysis.visualizer import Visualizer
from backtest.risk_manager import BacktestRiskManager

# Data paths (relative to this file, works on both Windows and Linux)
DATA_BASE_PATH = Path(__file__).parent.parent / "data"
DATA_PATHS_OPTIMIZED = {
    "GER40": DATA_BASE_PATH / "optimized" / "GER40_m1.parquet",
    "XAUUSD": DATA_BASE_PATH / "optimized" / "XAUUSD_m1.parquet",
}

# Suppress verbose logging in worker
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Symbol configurations
SYMBOL_CONFIGS = {
    "GER40": SymbolConfig(
        name="GER40",
        spread_points=1.5,
        digits=2,
        volume_step=0.01,
        trade_tick_size=0.5,
        trade_tick_value=0.5,
        trade_contract_size=1.0,
    ),
    "XAUUSD": SymbolConfig(
        name="XAUUSD",
        spread_points=0.30,
        digits=2,
        volume_step=0.01,
        trade_tick_size=0.01,
        trade_tick_value=1.0,
        trade_contract_size=100.0,
    ),
}

SYMBOL_TIMEZONES = {
    "GER40": "Europe/Berlin",
    "XAUUSD": "Asia/Tokyo",
}


def run_single_group_backtest(
    group: Dict[str, Any],
    start_date: datetime,
    end_date: datetime,
    output_dir: Path,
    initial_balance: float = 100000.0,
    risk_amount: float = 1000.0,
    skip_charts: bool = False,
) -> Dict[str, Any]:
    """
    Run backtest for a single parameter group.

    Args:
        group: Group dict with params from generate_backtest_groups.py
        start_date: Start date (UTC)
        end_date: End date (UTC)
        output_dir: Directory to save output
        initial_balance: Starting balance
        risk_amount: Fixed risk per trade
        skip_charts: If True, skip equity chart generation

    Returns:
        Dict with results metrics
    """
    group_id = group["id"]
    symbol = group["symbol"]
    params = group["params"]

    # Create group-specific output directory
    group_output_dir = output_dir / group_id
    group_output_dir.mkdir(parents=True, exist_ok=True)

    # Configure backtest
    config = BacktestConfig(
        initial_balance=initial_balance,
        leverage=100,
        symbols={symbol: SYMBOL_CONFIGS[symbol]}
    )

    # Create fresh emulator for this worker
    emulator = MT5Emulator()
    emulator.reset()
    emulator.configure(config)

    # Patch MT5 module
    mt5_patch = create_mt5_patch_module(emulator)
    sys.modules["MetaTrader5"] = mt5_patch

    # Import strategy after patching
    from src.strategies.ib_strategy import IBStrategy

    # Create executor
    executor = BacktestExecutor(emulator, config)
    executor.connect(12345, "password", "BacktestServer")

    # Load data directly from optimized parquet (works on both Windows and Linux)
    data_path = DATA_PATHS_OPTIMIZED.get(symbol)
    if not data_path or not data_path.exists():
        return {"error": f"Data file not found: {data_path}", "group_id": group_id}

    m1_data = pd.read_parquet(data_path)

    # Ensure time column is datetime with timezone
    if "time" in m1_data.columns:
        if m1_data["time"].dt.tz is None:
            m1_data["time"] = pd.to_datetime(m1_data["time"]).dt.tz_localize("UTC")
        else:
            m1_data["time"] = pd.to_datetime(m1_data["time"]).dt.tz_convert("UTC")

    # Filter date range
    if start_date:
        m1_data = m1_data[m1_data["time"] >= start_date]
    if end_date:
        m1_data = m1_data[m1_data["time"] <= end_date]

    if m1_data.empty:
        return {"error": f"No M1 data for {symbol} in date range", "group_id": group_id}

    emulator.load_m1_data(symbol, m1_data)

    # Create risk manager
    risk_manager = BacktestRiskManager(
        emulator=emulator,
        risk_amount=risk_amount,
        max_margin_pct=40.0,
    )

    # Create strategy
    magic_number = hash(group_id) % 100000 + 10000
    strategy = IBStrategy(
        symbol=symbol,
        params=params,
        executor=executor,
        magic_number=magic_number,
        strategy_label=group_id,
        news_filter_enabled=True,
    )

    # Run backtest loop
    local_tz = pytz.timezone(SYMBOL_TIMEZONES[symbol])

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

        _check_sltp_hits(emulator, row, SYMBOL_CONFIGS[symbol], sl_override=sl_before_tsl)

    # Close remaining positions
    emulator.force_close_all_positions(reason="backtest_end")

    # Get results
    trade_log = emulator.get_trade_log()

    # Calculate metrics
    total_r = sum(t.profit for t in trade_log if t.exit_time) / risk_amount if trade_log else 0
    wins = sum(1 for t in trade_log if t.exit_time and t.profit > 0)
    losses = sum(1 for t in trade_log if t.exit_time and t.profit < 0)
    winrate = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0

    # Calculate equity curve
    equity_curve = _calculate_equity_curve(trade_log, initial_balance)

    # Calculate max drawdown
    max_dd = 0
    if not equity_curve.empty:
        equity_curve["peak"] = equity_curve["equity"].cummax()
        equity_curve["drawdown"] = (equity_curve["peak"] - equity_curve["equity"]) / equity_curve["peak"]
        max_dd = equity_curve["drawdown"].max() * 100

    # Calculate Sharpe ratio (simplified)
    if trade_log:
        returns = [t.profit / risk_amount for t in trade_log if t.exit_time]
        if returns:
            mean_r = sum(returns) / len(returns)
            std_r = (sum((r - mean_r) ** 2 for r in returns) / len(returns)) ** 0.5
            sharpe = (mean_r / std_r) * (252 ** 0.5) if std_r > 0 else 0
        else:
            sharpe = 0
    else:
        sharpe = 0

    # Build result
    result = {
        "group_id": group_id,
        "symbol": symbol,
        "source_category": group.get("source_category", "unknown"),
        "total_trades": len(trade_log),
        "wins": wins,
        "losses": losses,
        "winrate": round(winrate, 2),
        "total_r": round(total_r, 2),
        "sharpe": round(sharpe, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "total_pnl": round(sum(t.profit for t in trade_log if t.exit_time), 2),
        "expected_total_r": group.get("combined_total_r", 0),
        "expected_trades": group.get("total_trades", 0),
        "r_difference": round(total_r - group.get("combined_total_r", 0), 2),
    }

    # Save trade log to CSV (for combined analysis)
    if trade_log:
        trades_data = []
        for t in trade_log:
            if t.exit_time:
                trades_data.append({
                    "entry_time": t.entry_time.isoformat() if t.entry_time else None,
                    "exit_time": t.exit_time.isoformat() if t.exit_time else None,
                    "symbol": symbol,
                    "direction": t.direction,
                    "profit": round(t.profit, 2),
                    "r": round(t.profit / risk_amount, 4),
                })
        if trades_data:
            trades_df = pd.DataFrame(trades_data)
            trades_df.to_csv(group_output_dir / "trades.csv", index=False)

    # Save config
    config_path = group_output_dir / "config.json"
    import json
    with open(config_path, "w") as f:
        json.dump({
            "group_id": group_id,
            "symbol": symbol,
            "start_date": str(start_date.date()),
            "end_date": str(end_date.date()),
            "params": params,
            "results": result,
        }, f, indent=2)

    # Save summary
    summary_path = group_output_dir / "summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"Backtest Results: {group_id}\n")
        f.write("=" * 50 + "\n")
        f.write(f"Symbol: {symbol}\n")
        f.write(f"Period: {start_date.date()} to {end_date.date()}\n")
        f.write(f"Source: {group.get('source_category', 'unknown')}\n")
        f.write("\n")
        f.write("Results:\n")
        f.write(f"  Total Trades: {result['total_trades']}\n")
        f.write(f"  Wins/Losses: {wins}/{losses}\n")
        f.write(f"  Win Rate: {winrate:.1f}%\n")
        f.write(f"  Total R: {total_r:.2f}\n")
        f.write(f"  Sharpe: {sharpe:.2f}\n")
        f.write(f"  Max DD: {max_dd:.1f}%\n")
        f.write(f"  Total P/L: ${result['total_pnl']:,.2f}\n")
        f.write("\n")
        f.write("Comparison with Fast Backtest:\n")
        f.write(f"  Expected R: {group.get('combined_total_r', 0):.2f}\n")
        f.write(f"  Actual R: {total_r:.2f}\n")
        f.write(f"  Difference: {result['r_difference']:.2f}\n")

    # Generate Excel report with detailed trades
    if trade_log:
        try:
            from backtest.reporting import ExcelReportGenerator
            from backtest.analysis.metrics import PerformanceMetrics

            metrics = PerformanceMetrics(
                trades=trade_log,
                equity_curve=equity_curve,
                initial_balance=initial_balance,
            )
            metrics_report = metrics.calculate_all()

            excel_gen = ExcelReportGenerator()
            excel_gen.generate_report(
                trade_log=trade_log,
                metrics_report=metrics_report,
                output_path=group_output_dir,
                symbol=symbol,
                initial_balance=initial_balance,
                ib_data=ib_data_by_date,
            )
        except Exception as e:
            logger.warning(f"Failed to generate Excel report for {group_id}: {e}")

    # Generate equity chart (optional)
    if not skip_charts and not equity_curve.empty:
        try:
            viz = Visualizer()
            viz.plot_equity_and_drawdown(
                equity_curve,
                title=f"{group_id} Equity Curve",
                save_path=group_output_dir / "equity_drawdown.png",
            )
        except Exception as e:
            logger.warning(f"Failed to generate chart for {group_id}: {e}")

    return result


def _check_sltp_hits(emulator: MT5Emulator, candle: pd.Series, symbol_cfg: SymbolConfig,
                     sl_override: dict = None) -> None:
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
            emulator.close_position_by_ticket(position.ticket, price=exit_price, exit_reason=exit_reason)


def _calculate_equity_curve(trade_log: List, initial_balance: float) -> pd.DataFrame:
    """Calculate equity curve from trade log."""
    if not trade_log:
        return pd.DataFrame()

    first_entry = None
    for trade in trade_log:
        if trade.entry_time:
            first_entry = trade.entry_time
            break

    if first_entry is None:
        return pd.DataFrame()

    equity_points = [{"time": first_entry, "equity": initial_balance, "drawdown": 0.0}]
    balance = initial_balance
    peak = initial_balance

    for trade in trade_log:
        if trade.exit_time:
            balance += trade.profit
            peak = max(peak, balance)
            drawdown = (peak - balance) / peak if peak > 0 else 0
            equity_points.append({"time": trade.exit_time, "equity": balance, "drawdown": drawdown})

    return pd.DataFrame(equity_points)


if __name__ == "__main__":
    # Test run with first group
    import json

    analyze_dir = Path(__file__).parent.parent / "analyze"
    groups_file = analyze_dir / "backtest_groups_GER40.json"

    if groups_file.exists():
        with open(groups_file) as f:
            groups = json.load(f)

        if groups:
            print(f"Testing with group: {groups[0]['id']}")

            result = run_single_group_backtest(
                group=groups[0],
                start_date=datetime(2024, 1, 1, tzinfo=pytz.UTC),
                end_date=datetime(2024, 3, 31, tzinfo=pytz.UTC),
                output_dir=Path(__file__).parent / "output" / "test_parallel",
                skip_charts=False,
            )

            print(f"\nResult: {json.dumps(result, indent=2)}")
    else:
        print(f"Groups file not found: {groups_file}")
