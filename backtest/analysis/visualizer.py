"""
Visualizer - Trading performance visualization.

Generates charts and reports for backtest analysis.
"""

import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

import pandas as pd
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.figure import Figure

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from ..emulator.models import TradeLog
from .metrics import PerformanceMetrics, MetricsReport

logger = logging.getLogger(__name__)


class Visualizer:
    """
    Generate visualizations for backtest results.

    Creates:
    - Equity curve charts
    - Drawdown charts
    - Trade distribution charts
    - Monthly returns heatmap
    - Entry/exit markers on price chart
    """

    def __init__(self, style: str = "seaborn-v0_8-darkgrid"):
        """
        Initialize Visualizer.

        Args:
            style: Matplotlib style to use
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available. Visualization disabled.")
            return

        try:
            plt.style.use(style)
        except Exception:
            plt.style.use("default")

        self.fig_size = (12, 6)
        self.colors = {
            "equity": "#2ecc71",
            "drawdown": "#e74c3c",
            "buy": "#27ae60",
            "sell": "#c0392b",
            "balance": "#3498db",
        }

    def plot_equity_curve(
        self,
        equity_curve: pd.DataFrame,
        title: str = "Equity Curve",
        show_balance: bool = True,
        save_path: Optional[Path] = None,
    ) -> Optional[Figure]:
        """
        Plot equity curve.

        Args:
            equity_curve: DataFrame with time, equity, balance columns
            title: Chart title
            show_balance: Whether to show balance line
            save_path: Path to save figure

        Returns:
            Matplotlib Figure or None
        """
        if not MATPLOTLIB_AVAILABLE:
            return None

        fig, ax = plt.subplots(figsize=self.fig_size)

        times = pd.to_datetime(equity_curve["time"])

        # Plot equity
        ax.plot(
            times,
            equity_curve["equity"],
            color=self.colors["equity"],
            linewidth=1.5,
            label="Equity",
        )

        # Plot balance if available
        if show_balance and "balance" in equity_curve.columns:
            ax.plot(
                times,
                equity_curve["balance"],
                color=self.colors["balance"],
                linewidth=1,
                linestyle="--",
                label="Balance",
                alpha=0.7,
            )

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Date")
        ax.set_ylabel("Value ($)")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)

        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.xticks(rotation=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved equity curve to {save_path}")

        return fig

    def plot_drawdown(
        self,
        equity_curve: pd.DataFrame,
        title: str = "Drawdown",
        save_path: Optional[Path] = None,
    ) -> Optional[Figure]:
        """
        Plot drawdown chart.

        Args:
            equity_curve: DataFrame with time and equity columns
            title: Chart title
            save_path: Path to save figure

        Returns:
            Matplotlib Figure or None
        """
        if not MATPLOTLIB_AVAILABLE:
            return None

        fig, ax = plt.subplots(figsize=self.fig_size)

        times = pd.to_datetime(equity_curve["time"])
        equity = equity_curve["equity"].values

        # Calculate drawdown
        running_max = np.maximum.accumulate(equity)
        drawdown_pct = (equity - running_max) / running_max * 100

        # Fill area
        ax.fill_between(
            times,
            drawdown_pct,
            0,
            color=self.colors["drawdown"],
            alpha=0.3,
        )
        ax.plot(
            times,
            drawdown_pct,
            color=self.colors["drawdown"],
            linewidth=1,
        )

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Date")
        ax.set_ylabel("Drawdown (%)")
        ax.grid(True, alpha=0.3)

        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.xticks(rotation=45)

        # Set y-axis to always show negative (below zero)
        ax.set_ylim(top=0)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def plot_equity_and_drawdown(
        self,
        equity_curve: pd.DataFrame,
        title: str = "Performance Overview",
        save_path: Optional[Path] = None,
    ) -> Optional[Figure]:
        """
        Plot equity curve and drawdown together.

        Args:
            equity_curve: DataFrame with time, equity columns
            title: Chart title
            save_path: Path to save figure

        Returns:
            Matplotlib Figure or None
        """
        if not MATPLOTLIB_AVAILABLE:
            return None

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[2, 1])

        times = pd.to_datetime(equity_curve["time"])
        equity = equity_curve["equity"].values

        # Top: Equity curve
        ax1.plot(times, equity, color=self.colors["equity"], linewidth=1.5)
        ax1.set_title(title, fontsize=14, fontweight="bold")
        ax1.set_ylabel("Equity ($)")
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

        # Bottom: Drawdown
        running_max = np.maximum.accumulate(equity)
        drawdown_pct = (equity - running_max) / running_max * 100

        ax2.fill_between(times, drawdown_pct, 0, color=self.colors["drawdown"], alpha=0.3)
        ax2.plot(times, drawdown_pct, color=self.colors["drawdown"], linewidth=1)
        ax2.set_ylabel("Drawdown (%)")
        ax2.set_xlabel("Date")
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(top=0)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.xticks(rotation=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def plot_trade_distribution(
        self,
        trades: List[TradeLog],
        title: str = "Trade Profit Distribution",
        save_path: Optional[Path] = None,
    ) -> Optional[Figure]:
        """
        Plot histogram of trade profits.

        Args:
            trades: List of TradeLog entries
            title: Chart title
            save_path: Path to save figure

        Returns:
            Matplotlib Figure or None
        """
        if not MATPLOTLIB_AVAILABLE or not trades:
            return None

        profits = [t.profit for t in trades]

        fig, ax = plt.subplots(figsize=self.fig_size)

        # Histogram
        n, bins, patches = ax.hist(profits, bins=30, edgecolor="black", alpha=0.7)

        # Color bars by profit/loss
        for i, patch in enumerate(patches):
            if bins[i] >= 0:
                patch.set_facecolor(self.colors["buy"])
            else:
                patch.set_facecolor(self.colors["sell"])

        ax.axvline(x=0, color="black", linestyle="--", linewidth=1)
        ax.axvline(x=np.mean(profits), color="blue", linestyle="-", linewidth=2, label=f"Mean: ${np.mean(profits):.2f}")

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Profit ($)")
        ax.set_ylabel("Frequency")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def plot_monthly_returns(
        self,
        trades: List[TradeLog],
        title: str = "Monthly Returns",
        save_path: Optional[Path] = None,
    ) -> Optional[Figure]:
        """
        Plot monthly returns as bar chart.

        Args:
            trades: List of TradeLog entries
            title: Chart title
            save_path: Path to save figure

        Returns:
            Matplotlib Figure or None
        """
        if not MATPLOTLIB_AVAILABLE or not trades:
            return None

        # Group by month
        df = pd.DataFrame([
            {"month": t.exit_time.strftime("%Y-%m") if t.exit_time else None, "profit": t.profit}
            for t in trades
        ])
        df = df.dropna()
        monthly = df.groupby("month")["profit"].sum()

        fig, ax = plt.subplots(figsize=self.fig_size)

        colors = [self.colors["buy"] if v >= 0 else self.colors["sell"] for v in monthly.values]
        ax.bar(monthly.index, monthly.values, color=colors, edgecolor="black", alpha=0.7)

        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Month")
        ax.set_ylabel("Profit ($)")
        ax.grid(True, alpha=0.3, axis="y")

        plt.xticks(rotation=45)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def plot_trades_on_price(
        self,
        price_data: pd.DataFrame,
        trades: List[TradeLog],
        title: str = "Trades on Price Chart",
        save_path: Optional[Path] = None,
    ) -> Optional[Figure]:
        """
        Plot price chart with trade entry/exit markers.

        Args:
            price_data: DataFrame with time, open, high, low, close columns
            trades: List of TradeLog entries
            title: Chart title
            save_path: Path to save figure

        Returns:
            Matplotlib Figure or None
        """
        if not MATPLOTLIB_AVAILABLE:
            return None

        fig, ax = plt.subplots(figsize=(14, 7))

        times = pd.to_datetime(price_data["time"])

        # Plot close price as line
        ax.plot(times, price_data["close"], color="gray", linewidth=0.5, alpha=0.7)

        # Add trade markers
        for trade in trades:
            if trade.entry_time and trade.entry_price:
                marker = "^" if trade.direction == "long" else "v"
                color = self.colors["buy"] if trade.direction == "long" else self.colors["sell"]
                ax.scatter(
                    trade.entry_time,
                    trade.entry_price,
                    marker=marker,
                    color=color,
                    s=100,
                    zorder=5,
                    edgecolors="black",
                )

            if trade.exit_time and trade.exit_price:
                ax.scatter(
                    trade.exit_time,
                    trade.exit_price,
                    marker="x",
                    color="black",
                    s=80,
                    zorder=5,
                )

                # Draw line connecting entry to exit
                if trade.entry_time and trade.entry_price:
                    line_color = self.colors["buy"] if trade.profit >= 0 else self.colors["sell"]
                    ax.plot(
                        [trade.entry_time, trade.exit_time],
                        [trade.entry_price, trade.exit_price],
                        color=line_color,
                        linewidth=1,
                        alpha=0.5,
                    )

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.grid(True, alpha=0.3)

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        plt.xticks(rotation=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def generate_report(
        self,
        metrics: MetricsReport,
        equity_curve: pd.DataFrame,
        trades: List[TradeLog],
        output_dir: Path,
        symbol: str = "",
    ) -> None:
        """
        Generate complete visual report.

        Args:
            metrics: MetricsReport instance
            equity_curve: Equity curve DataFrame
            trades: List of trades
            output_dir: Directory to save figures
            symbol: Symbol name for titles
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Generating visual report in {output_dir}")

        # Equity and drawdown
        self.plot_equity_and_drawdown(
            equity_curve,
            title=f"{symbol} Performance Overview" if symbol else "Performance Overview",
            save_path=output_dir / "equity_drawdown.png",
        )

        # Trade distribution
        self.plot_trade_distribution(
            trades,
            title=f"{symbol} Trade Distribution" if symbol else "Trade Distribution",
            save_path=output_dir / "trade_distribution.png",
        )

        # Monthly returns
        self.plot_monthly_returns(
            trades,
            title=f"{symbol} Monthly Returns" if symbol else "Monthly Returns",
            save_path=output_dir / "monthly_returns.png",
        )

        # Save metrics summary to text file
        summary_path = output_dir / "metrics_summary.txt"
        with open(summary_path, "w") as f:
            f.write(metrics.summary())

        logger.info(f"Report generated: {len(list(output_dir.glob('*.png')))} charts saved")

    @staticmethod
    def show():
        """Display all figures."""
        if MATPLOTLIB_AVAILABLE:
            plt.show()
