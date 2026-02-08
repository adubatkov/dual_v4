"""
PerformanceMetrics - Trading performance analytics.

Calculates comprehensive trading metrics from backtest results.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from datetime import datetime

import pandas as pd
import numpy as np

from ..emulator.models import TradeLog

logger = logging.getLogger(__name__)


@dataclass
class MetricsReport:
    """Complete metrics report."""

    # Basic stats
    total_trades: int
    winning_trades: int
    losing_trades: int
    breakeven_trades: int

    # Profit metrics
    total_profit: float
    gross_profit: float
    gross_loss: float
    net_profit: float

    # Ratios
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    avg_trade: float
    win_loss_ratio: float

    # Risk metrics
    max_drawdown: float
    max_drawdown_pct: float
    max_drawdown_duration_days: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float

    # Trade stats
    avg_trade_duration_hours: float
    max_consecutive_wins: int
    max_consecutive_losses: int
    longest_win_streak: int
    longest_loss_streak: int

    # Return metrics
    total_return_pct: float
    annualized_return_pct: float
    monthly_returns: Dict[str, float] = None

    # By variation (if available)
    by_variation: Dict[str, Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": round(self.win_rate, 2),
            "total_profit": round(self.total_profit, 2),
            "profit_factor": round(self.profit_factor, 2),
            "sharpe_ratio": round(self.sharpe_ratio, 2),
            "sortino_ratio": round(self.sortino_ratio, 2),
            "max_drawdown_pct": round(self.max_drawdown_pct, 2),
            "avg_win": round(self.avg_win, 2),
            "avg_loss": round(self.avg_loss, 2),
            "total_return_pct": round(self.total_return_pct, 2),
        }

    def summary(self) -> str:
        """Generate text summary."""
        lines = [
            "=" * 50,
            "BACKTEST PERFORMANCE REPORT",
            "=" * 50,
            "",
            "TRADE STATISTICS",
            f"  Total Trades:      {self.total_trades}",
            f"  Winning Trades:    {self.winning_trades}",
            f"  Losing Trades:     {self.losing_trades}",
            f"  Win Rate:          {self.win_rate:.1f}%",
            "",
            "PROFIT METRICS",
            f"  Total Profit:      ${self.total_profit:,.2f}",
            f"  Gross Profit:      ${self.gross_profit:,.2f}",
            f"  Gross Loss:        ${self.gross_loss:,.2f}",
            f"  Profit Factor:     {self.profit_factor:.2f}",
            f"  Avg Win:           ${self.avg_win:,.2f}",
            f"  Avg Loss:          ${self.avg_loss:,.2f}",
            "",
            "RISK METRICS",
            f"  Max Drawdown:      ${self.max_drawdown:,.2f} ({self.max_drawdown_pct:.1f}%)",
            f"  Sharpe Ratio:      {self.sharpe_ratio:.2f}",
            f"  Sortino Ratio:     {self.sortino_ratio:.2f}",
            "",
            "RETURN METRICS",
            f"  Total Return:      {self.total_return_pct:.1f}%",
            f"  Annualized Return: {self.annualized_return_pct:.1f}%",
            "",
            "STREAK ANALYSIS",
            f"  Max Consecutive Wins:   {self.max_consecutive_wins}",
            f"  Max Consecutive Losses: {self.max_consecutive_losses}",
            "=" * 50,
        ]
        return "\n".join(lines)


class PerformanceMetrics:
    """
    Calculate trading performance metrics.

    Provides comprehensive analysis of backtest results including:
    - Win rate and profit factor
    - Sharpe and Sortino ratios
    - Maximum drawdown analysis
    - Monthly/yearly breakdowns
    - Strategy variation analysis
    """

    def __init__(
        self,
        trades: List[TradeLog],
        equity_curve: pd.DataFrame,
        initial_balance: float = 50000.0,
        risk_free_rate: float = 0.02,  # 2% annual
    ):
        """
        Initialize PerformanceMetrics.

        Args:
            trades: List of TradeLog entries
            equity_curve: DataFrame with time and equity columns
            initial_balance: Starting account balance
            risk_free_rate: Annual risk-free rate for Sharpe calculation
        """
        self.trades = trades
        self.equity_curve = equity_curve
        self.initial_balance = initial_balance
        self.risk_free_rate = risk_free_rate

        # Convert trades to DataFrame for easier analysis
        self._trades_df = self._trades_to_dataframe()

    def _trades_to_dataframe(self) -> pd.DataFrame:
        """Convert trades list to DataFrame."""
        if not self.trades:
            return pd.DataFrame()

        records = []
        for t in self.trades:
            records.append({
                "ticket": t.ticket,
                "symbol": t.symbol,
                "direction": t.direction,
                "entry_time": t.entry_time,
                "entry_price": t.entry_price,
                "exit_time": t.exit_time,
                "exit_price": t.exit_price,
                "volume": t.volume,
                "profit": t.profit,
                "exit_reason": t.exit_reason,
                "variation": t.variation,
            })

        df = pd.DataFrame(records)
        if not df.empty and "entry_time" in df.columns:
            df["entry_time"] = pd.to_datetime(df["entry_time"])
            df["exit_time"] = pd.to_datetime(df["exit_time"])
            df["duration"] = (df["exit_time"] - df["entry_time"]).dt.total_seconds() / 3600
        return df

    def calculate_all(self) -> MetricsReport:
        """
        Calculate all performance metrics.

        Returns:
            MetricsReport with comprehensive metrics
        """
        df = self._trades_df

        # Basic counts
        total_trades = len(df)
        winning_trades = len(df[df["profit"] > 0]) if not df.empty else 0
        losing_trades = len(df[df["profit"] < 0]) if not df.empty else 0
        breakeven_trades = len(df[df["profit"] == 0]) if not df.empty else 0

        # Profit calculations
        profits = df["profit"].values if not df.empty else np.array([0])
        gross_profit = float(np.sum(profits[profits > 0])) if len(profits) > 0 else 0.0
        gross_loss = float(np.abs(np.sum(profits[profits < 0]))) if len(profits) > 0 else 0.0
        total_profit = float(np.sum(profits))

        # Ratios
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)

        avg_win = (gross_profit / winning_trades) if winning_trades > 0 else 0.0
        avg_loss = (gross_loss / losing_trades) if losing_trades > 0 else 0.0
        avg_trade = (total_profit / total_trades) if total_trades > 0 else 0.0
        win_loss_ratio = (avg_win / avg_loss) if avg_loss > 0 else (float("inf") if avg_win > 0 else 0.0)

        # Drawdown analysis
        dd_result = self._calculate_drawdown()

        # Risk-adjusted returns
        sharpe = self._calculate_sharpe_ratio()
        sortino = self._calculate_sortino_ratio()
        calmar = self._calculate_calmar_ratio(dd_result["max_drawdown_pct"])

        # Trade duration
        avg_duration = float(df["duration"].mean()) if not df.empty and "duration" in df.columns else 0.0

        # Streaks
        streaks = self._calculate_streaks()

        # Returns
        final_balance = self.initial_balance + total_profit
        total_return_pct = ((final_balance - self.initial_balance) / self.initial_balance * 100) if self.initial_balance > 0 else 0.0

        # Annualized return
        if not self.equity_curve.empty:
            days = (self.equity_curve["time"].max() - self.equity_curve["time"].min()).days
            years = max(days / 365.25, 0.01)
            annualized_return = ((1 + total_return_pct / 100) ** (1 / years) - 1) * 100
        else:
            annualized_return = 0.0

        # Monthly returns
        monthly_returns = self._calculate_monthly_returns()

        # By variation
        by_variation = self._calculate_by_variation()

        return MetricsReport(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            breakeven_trades=breakeven_trades,
            total_profit=total_profit,
            gross_profit=gross_profit,
            gross_loss=gross_loss,
            net_profit=total_profit,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            avg_trade=avg_trade,
            win_loss_ratio=win_loss_ratio,
            max_drawdown=dd_result["max_drawdown"],
            max_drawdown_pct=dd_result["max_drawdown_pct"],
            max_drawdown_duration_days=dd_result["max_drawdown_duration_days"],
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            avg_trade_duration_hours=avg_duration,
            max_consecutive_wins=streaks["max_wins"],
            max_consecutive_losses=streaks["max_losses"],
            longest_win_streak=streaks["max_wins"],
            longest_loss_streak=streaks["max_losses"],
            total_return_pct=total_return_pct,
            annualized_return_pct=annualized_return,
            monthly_returns=monthly_returns,
            by_variation=by_variation,
        )

    def _calculate_drawdown(self) -> Dict[str, float]:
        """Calculate maximum drawdown metrics."""
        if self.equity_curve.empty:
            return {
                "max_drawdown": 0.0,
                "max_drawdown_pct": 0.0,
                "max_drawdown_duration_days": 0.0,
            }

        equity = self.equity_curve["equity"].values
        times = pd.to_datetime(self.equity_curve["time"])

        # Calculate running maximum
        running_max = np.maximum.accumulate(equity)

        # Drawdown in dollars
        drawdown = running_max - equity

        # Drawdown percentage
        drawdown_pct = np.where(running_max > 0, drawdown / running_max * 100, 0)

        max_dd = float(np.max(drawdown))
        max_dd_pct = float(np.max(drawdown_pct))

        # Drawdown duration
        max_dd_duration = 0.0
        in_drawdown = False
        dd_start = None

        for i, (eq, peak) in enumerate(zip(equity, running_max)):
            if eq < peak:
                if not in_drawdown:
                    in_drawdown = True
                    dd_start = times.iloc[i]
            else:
                if in_drawdown and dd_start is not None:
                    duration = (times.iloc[i] - dd_start).days
                    max_dd_duration = max(max_dd_duration, duration)
                    in_drawdown = False

        return {
            "max_drawdown": max_dd,
            "max_drawdown_pct": max_dd_pct,
            "max_drawdown_duration_days": max_dd_duration,
        }

    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio."""
        if self.equity_curve.empty or len(self.equity_curve) < 2:
            return 0.0

        equity = self.equity_curve["equity"].values

        # Daily returns
        returns = np.diff(equity) / equity[:-1]

        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0

        # Annualize
        avg_return = np.mean(returns)
        std_return = np.std(returns)

        # Assume daily data, annualize
        trading_days = 252
        annualized_return = avg_return * trading_days
        annualized_std = std_return * np.sqrt(trading_days)

        if annualized_std == 0:
            return 0.0

        sharpe = (annualized_return - self.risk_free_rate) / annualized_std
        return float(sharpe)

    def _calculate_sortino_ratio(self) -> float:
        """Calculate Sortino ratio (uses downside deviation)."""
        if self.equity_curve.empty or len(self.equity_curve) < 2:
            return 0.0

        equity = self.equity_curve["equity"].values
        returns = np.diff(equity) / equity[:-1]

        if len(returns) == 0:
            return 0.0

        # Downside deviation (only negative returns)
        negative_returns = returns[returns < 0]
        if len(negative_returns) == 0:
            return float("inf") if np.mean(returns) > 0 else 0.0

        downside_std = np.std(negative_returns)

        if downside_std == 0:
            return 0.0

        avg_return = np.mean(returns)
        trading_days = 252
        annualized_return = avg_return * trading_days
        annualized_downside = downside_std * np.sqrt(trading_days)

        sortino = (annualized_return - self.risk_free_rate) / annualized_downside
        return float(sortino)

    def _calculate_calmar_ratio(self, max_drawdown_pct: float) -> float:
        """Calculate Calmar ratio (return / max drawdown)."""
        if max_drawdown_pct == 0:
            return 0.0

        if self.equity_curve.empty:
            return 0.0

        equity = self.equity_curve["equity"].values
        total_return_pct = ((equity[-1] - equity[0]) / equity[0]) * 100

        return total_return_pct / max_drawdown_pct

    def _calculate_streaks(self) -> Dict[str, int]:
        """Calculate win/loss streaks."""
        if self._trades_df.empty:
            return {"max_wins": 0, "max_losses": 0}

        profits = self._trades_df["profit"].values

        max_wins = 0
        max_losses = 0
        current_wins = 0
        current_losses = 0

        for p in profits:
            if p > 0:
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            elif p < 0:
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)
            else:
                current_wins = 0
                current_losses = 0

        return {"max_wins": max_wins, "max_losses": max_losses}

    def _calculate_monthly_returns(self) -> Dict[str, float]:
        """Calculate returns by month."""
        if self._trades_df.empty:
            return {}

        df = self._trades_df.copy()
        df["month"] = df["exit_time"].dt.to_period("M")

        monthly = df.groupby("month")["profit"].sum()
        return {str(k): float(v) for k, v in monthly.items()}

    def _calculate_by_variation(self) -> Dict[str, Dict[str, Any]]:
        """Calculate metrics breakdown by strategy variation."""
        if self._trades_df.empty or "variation" not in self._trades_df.columns:
            return {}

        result = {}
        for var in self._trades_df["variation"].unique():
            if pd.isna(var) or var == "":
                continue

            var_trades = self._trades_df[self._trades_df["variation"] == var]

            winning = len(var_trades[var_trades["profit"] > 0])
            losing = len(var_trades[var_trades["profit"] < 0])
            total = len(var_trades)
            profit = float(var_trades["profit"].sum())

            result[var] = {
                "total_trades": total,
                "winning_trades": winning,
                "losing_trades": losing,
                "win_rate": (winning / total * 100) if total > 0 else 0.0,
                "total_profit": profit,
                "avg_profit": profit / total if total > 0 else 0.0,
            }

        return result

    def get_equity_curve(self) -> pd.DataFrame:
        """Get equity curve DataFrame."""
        return self.equity_curve.copy()

    def get_trades_dataframe(self) -> pd.DataFrame:
        """Get trades as DataFrame."""
        return self._trades_df.copy()

    def get_drawdown_series(self) -> pd.Series:
        """Get drawdown series."""
        if self.equity_curve.empty:
            return pd.Series()

        equity = self.equity_curve["equity"].values
        running_max = np.maximum.accumulate(equity)
        drawdown = (running_max - equity) / running_max * 100

        return pd.Series(drawdown, index=self.equity_curve["time"])
