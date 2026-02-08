"""
Report Manager - Orchestrates backtest report generation

Handles:
- Output folder structure creation
- Coordinating Excel and chart generation
- Summary file creation
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import json

import pandas as pd

from ..emulator.models import TradeLog

logger = logging.getLogger(__name__)


class BacktestReportManager:
    """
    Manages backtest report generation and folder structure.

    Creates timestamped output folders with:
    - results.xlsx - Detailed Excel report
    - summary.txt - Text summary
    - equity_drawdown.png - Equity curve chart
    - trades/ - Individual trade charts
    """

    def __init__(
        self,
        base_output_path: Path,
        symbol: str,
        backtest_name: Optional[str] = None,
    ):
        """
        Initialize report manager.

        Args:
            base_output_path: Base path for backtest outputs
            symbol: Trading symbol (e.g., "GER40")
            backtest_name: Optional custom name for this backtest run
        """
        self.base_output_path = Path(base_output_path)
        self.symbol = symbol
        self.backtest_name = backtest_name

        # Create timestamped folder
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        if backtest_name:
            folder_name = f"{timestamp}_{symbol}_{backtest_name}"
        else:
            folder_name = f"{timestamp}_{symbol}"

        self.output_dir = self.base_output_path / folder_name
        self.trades_dir = self.output_dir / "trades"

        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.trades_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Report output directory: {self.output_dir}")

    def get_output_dir(self) -> Path:
        """Get main output directory."""
        return self.output_dir

    def get_trades_dir(self) -> Path:
        """Get trades charts directory."""
        return self.trades_dir

    def save_config(self, config: Dict[str, Any]) -> None:
        """
        Save backtest configuration to JSON file.

        Args:
            config: Configuration dictionary
        """
        config_path = self.output_dir / "config.json"

        # Convert non-serializable objects
        serializable_config = {}
        for key, value in config.items():
            if isinstance(value, (str, int, float, bool, list, dict, type(None))):
                serializable_config[key] = value
            elif isinstance(value, datetime):
                serializable_config[key] = value.isoformat()
            elif isinstance(value, Path):
                serializable_config[key] = str(value)
            else:
                serializable_config[key] = str(value)

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(serializable_config, f, indent=2, ensure_ascii=False)

        logger.info(f"Config saved to {config_path}")

    def save_summary(
        self,
        metrics_report,
        trade_count: int,
        start_date: datetime,
        end_date: datetime,
        initial_balance: float,
        risk_pct: float,
        max_margin_pct: float,
    ) -> None:
        """
        Save text summary of backtest results.

        Args:
            metrics_report: MetricsReport object
            trade_count: Number of trades
            start_date: Backtest start date
            end_date: Backtest end date
            initial_balance: Starting balance
            risk_pct: Risk percentage per trade
            max_margin_pct: Maximum margin percentage
        """
        summary_path = self.output_dir / "summary.txt"

        lines = [
            "=" * 60,
            f"BACKTEST REPORT: {self.symbol}",
            "=" * 60,
            "",
            "CONFIGURATION",
            f"  Period: {start_date.date()} to {end_date.date()}",
            f"  Initial Balance: ${initial_balance:,.2f}",
            f"  Risk per Trade: {risk_pct}%",
            f"  Max Margin: {max_margin_pct}%",
            "",
            "TRADE STATISTICS",
            f"  Total Trades: {metrics_report.total_trades}",
            f"  Winning Trades: {metrics_report.winning_trades}",
            f"  Losing Trades: {metrics_report.losing_trades}",
            f"  Win Rate: {metrics_report.win_rate:.1f}%",
            "",
            "PROFIT METRICS",
            f"  Total Profit: ${metrics_report.total_profit:,.2f}",
            f"  Gross Profit: ${metrics_report.gross_profit:,.2f}",
            f"  Gross Loss: ${metrics_report.gross_loss:,.2f}",
            f"  Profit Factor: {metrics_report.profit_factor:.2f}",
            f"  Avg Win: ${metrics_report.avg_win:,.2f}",
            f"  Avg Loss: ${metrics_report.avg_loss:,.2f}",
            "",
            "RISK METRICS",
            f"  Max Drawdown: ${metrics_report.max_drawdown:,.2f} ({metrics_report.max_drawdown_pct:.1f}%)",
            f"  Sharpe Ratio: {metrics_report.sharpe_ratio:.2f}",
            f"  Sortino Ratio: {metrics_report.sortino_ratio:.2f}",
            "",
            "RETURN METRICS",
            f"  Total Return: {metrics_report.total_return_pct:.1f}%",
            f"  Annualized Return: {metrics_report.annualized_return_pct:.1f}%",
            "",
            "STREAK ANALYSIS",
            f"  Max Consecutive Wins: {metrics_report.max_consecutive_wins}",
            f"  Max Consecutive Losses: {metrics_report.max_consecutive_losses}",
            "",
        ]

        # Add variation breakdown if available
        if metrics_report.by_variation:
            lines.append("BY VARIATION")
            for var, stats in metrics_report.by_variation.items():
                lines.append(f"  {var}:")
                lines.append(f"    Trades: {stats['total_trades']}, Win Rate: {stats['win_rate']:.1f}%, P/L: ${stats['total_profit']:.2f}")
            lines.append("")

        lines.append("=" * 60)
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Output: {self.output_dir}")

        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        logger.info(f"Summary saved to {summary_path}")

    def get_trade_chart_path(self, trade: TradeLog, index: int) -> Path:
        """
        Generate path for individual trade chart.

        Args:
            trade: Trade log entry
            index: Trade index

        Returns:
            Path for trade chart file
        """
        if trade.entry_time:
            date_str = trade.entry_time.strftime("%Y-%m-%d")
        else:
            date_str = "unknown"

        direction = trade.direction or "unknown"
        variation = trade.variation or "unknown"

        filename = f"{index:03d}_{date_str}_{variation}_{direction}.jpg"
        return self.trades_dir / filename
