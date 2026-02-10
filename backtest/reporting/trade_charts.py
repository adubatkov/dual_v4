"""
Trade Chart Generator

Creates individual trade charts with:
- Candlestick chart
- IB zone highlighting
- IBH/IBL/EQ levels
- Entry/Exit markers
- Profit/Stop zones (TradingView style)
"""

import logging
from datetime import datetime, date, time as datetime_time, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

import pandas as pd
import numpy as np
import pytz

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.patches import Rectangle
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from ..emulator.models import TradeLog

logger = logging.getLogger(__name__)


# Chart colors (inspired by graph_multi_asset.py)
COLORS = {
    "ib_zone": "#7e57c2",
    "ib_zone_alpha": 0.12,
    "candle_up": "#26a69a",
    "candle_down": "#ef5350",
    "wick": "#000000",
    "profit_zone": "#4caf50",
    "stop_zone": "#f44336",
    "entry_marker": "#2196f3",
    "exit_marker": "#ff9800",
    "ibh_ibl": "#666666",
    "eq": "#9c27b0",
    "grid": "#e0e0e0",
}

FIGSIZE = (16, 10)
DPI = 150


class TradeChartGenerator:
    """
    Generates individual trade charts.

    Creates professional-looking charts similar to TradingView with:
    - OHLC candlesticks
    - IB zone highlighting
    - Entry/exit markers
    - Profit/stop zones
    """

    def __init__(self, timezone: str = "Europe/Berlin"):
        """
        Initialize trade chart generator.

        Args:
            timezone: Timezone for chart display
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("matplotlib not available, trade charts will be disabled")

        self.timezone = timezone
        self.tz = pytz.timezone(timezone)

    def generate_trade_chart(
        self,
        trade: TradeLog,
        m1_data: pd.DataFrame,
        output_path: Path,
        ib_data: Optional[Dict[str, float]] = None,
        plot_window_hours: float = 5.0,
        fractal_lists: Optional[Dict] = None,
    ) -> Optional[Path]:
        """
        Generate chart for a single trade.

        Args:
            trade: TradeLog entry
            m1_data: M1 candlestick data (must have time, open, high, low, close)
            output_path: Path to save chart
            ib_data: Dict with ibh, ibl, eq values
            plot_window_hours: Hours to show on chart
            fractal_lists: Dict with all_h1/all_h4 raw fractal lists

        Returns:
            Path to saved chart or None if failed
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Cannot generate trade chart - matplotlib not installed")
            return None

        if trade.entry_time is None:
            logger.warning("Trade has no entry time, skipping chart")
            return None

        try:
            # Get chart window
            chart_start, chart_end = self._get_chart_window(trade, plot_window_hours)

            # Filter candle data
            df_chart = self._filter_candles(m1_data, chart_start, chart_end)
            if df_chart.empty:
                logger.warning(f"No candle data for trade on {trade.entry_time.date()}")
                return None

            # Create figure
            fig, ax = plt.subplots(figsize=FIGSIZE)

            # Draw components
            self._draw_candlesticks(ax, df_chart)
            self._draw_ib_zone(ax, df_chart, ib_data, chart_start, chart_end)
            self._draw_levels(ax, ib_data, chart_start, chart_end)
            self._draw_trade(ax, trade, chart_start, chart_end)

            # Configure axes first so y-limits are set
            self._configure_axes(ax, df_chart, chart_start, chart_end)

            # Prepare and draw fractals (per-trade sweep computation)
            fractal_draw_data = self._prepare_fractal_data(fractal_lists, m1_data, chart_start, chart_end)
            self._draw_fractals(ax, fractal_draw_data, chart_start, chart_end)

            # Title
            self._add_title(ax, trade, ib_data)

            # Watermark
            self._add_watermark(ax)

            # Save
            plt.savefig(output_path, dpi=DPI, bbox_inches="tight", facecolor="white")
            plt.close(fig)

            logger.debug(f"Trade chart saved: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Failed to generate trade chart: {e}")
            return None

    def _get_chart_window(
        self,
        trade: TradeLog,
        window_hours: float,
    ) -> Tuple[datetime, datetime]:
        """Calculate chart time window around trade."""
        entry_time = trade.entry_time

        # Convert to local timezone
        if entry_time.tzinfo is None:
            entry_time = pytz.utc.localize(entry_time)
        entry_local = entry_time.astimezone(self.tz)

        # Get trade date
        trade_date = entry_local.date()

        # Chart window: typically 07:00 - 12:00 for GER40
        # Adjust based on timezone
        chart_start = self.tz.localize(datetime.combine(trade_date, datetime_time(7, 0)))
        chart_end = self.tz.localize(datetime.combine(trade_date, datetime_time(12, 0)))

        return chart_start, chart_end

    def _filter_candles(
        self,
        m1_data: pd.DataFrame,
        chart_start: datetime,
        chart_end: datetime,
    ) -> pd.DataFrame:
        """Filter candles to chart window and convert to local time."""
        df = m1_data.copy()

        # Ensure time is datetime with UTC
        if df["time"].dt.tz is None:
            df["time"] = df["time"].dt.tz_localize("UTC")

        # Convert to local timezone (naive for matplotlib)
        df["time_local"] = df["time"].apply(
            lambda x: x.astimezone(self.tz).replace(tzinfo=None)
        )

        # Convert bounds to naive
        start_naive = chart_start.replace(tzinfo=None)
        end_naive = chart_end.replace(tzinfo=None)

        # Filter
        mask = (df["time_local"] >= start_naive) & (df["time_local"] <= end_naive)
        return df.loc[mask].copy()

    def _draw_candlesticks(self, ax, df: pd.DataFrame) -> None:
        """Draw OHLC candlesticks."""
        if df.empty:
            return

        df = df.sort_values("time_local")
        xs = mdates.date2num(df["time_local"])

        # Calculate candle width
        if len(xs) > 1:
            cw = (xs.max() - xs.min()) / max(len(xs), 1) * 0.6
        else:
            cw = 0.005

        for _, row in df.iterrows():
            t = mdates.date2num(row["time_local"])
            o, h, l, c = row["open"], row["high"], row["low"], row["close"]

            # Wick
            ax.plot([t, t], [l, h], color=COLORS["wick"], linewidth=0.8, zorder=2)

            # Body
            if c >= o:
                lower, height = o, c - o
                fc = COLORS["candle_up"]
            else:
                lower, height = c, o - c
                fc = COLORS["candle_down"]

            if height == 0:
                height = 0.01  # Minimum height for doji

            rect = Rectangle(
                (t - cw / 2, lower),
                cw,
                height,
                facecolor=fc,
                edgecolor=COLORS["wick"],
                linewidth=0.8,
                zorder=3
            )
            ax.add_patch(rect)

    def _draw_ib_zone(
        self,
        ax,
        df: pd.DataFrame,
        ib_data: Optional[Dict[str, float]],
        chart_start: datetime,
        chart_end: datetime,
    ) -> None:
        """Draw IB zone highlighting."""
        if ib_data is None:
            return

        ibh = ib_data.get("ibh")
        ibl = ib_data.get("ibl")

        if ibh is None or ibl is None or ibh <= ibl:
            return

        # IB time window from actual params (fallback to 08:00-09:00)
        trade_date = chart_start.date()
        ib_start_str = ib_data.get("ib_start", "08:00")
        ib_end_str = ib_data.get("ib_end", "09:00")
        ib_tz_str = ib_data.get("ib_tz")

        ib_start_time = datetime_time.fromisoformat(ib_start_str)
        ib_end_time = datetime_time.fromisoformat(ib_end_str)

        if ib_tz_str:
            ib_tz = pytz.timezone(ib_tz_str)
            ib_start = ib_tz.localize(datetime.combine(trade_date, ib_start_time)).replace(tzinfo=None)
            ib_end = ib_tz.localize(datetime.combine(trade_date, ib_end_time)).replace(tzinfo=None)
        else:
            ib_start = datetime.combine(trade_date, ib_start_time)
            ib_end = datetime.combine(trade_date, ib_end_time)

        # Clip to chart window
        start_naive = chart_start.replace(tzinfo=None)
        end_naive = chart_end.replace(tzinfo=None)

        ib_start = max(ib_start, start_naive)
        ib_end = min(ib_end, end_naive)

        if ib_end <= ib_start:
            return

        x1 = mdates.date2num(ib_start)
        x2 = mdates.date2num(ib_end)

        rect = Rectangle(
            (x1, ibl),
            x2 - x1,
            ibh - ibl,
            facecolor=COLORS["ib_zone"],
            edgecolor=COLORS["ib_zone"],
            alpha=COLORS["ib_zone_alpha"],
            linewidth=0.8,
            zorder=1
        )
        ax.add_patch(rect)

    def _draw_levels(
        self,
        ax,
        ib_data: Optional[Dict[str, float]],
        chart_start: datetime,
        chart_end: datetime,
    ) -> None:
        """Draw horizontal levels (IBH, IBL, EQ)."""
        if ib_data is None:
            return

        start_naive = chart_start.replace(tzinfo=None)
        end_naive = chart_end.replace(tzinfo=None)

        ibh = ib_data.get("ibh")
        ibl = ib_data.get("ibl")
        eq = ib_data.get("eq")

        # IBH and IBL
        if ibh is not None and ibl is not None:
            ax.hlines(
                [ibh, ibl],
                xmin=start_naive,
                xmax=end_naive,
                colors=COLORS["ibh_ibl"],
                linewidth=1.0,
                linestyles=":",
                label="IBH/IBL"
            )
            # Labels
            ax.text(end_naive, ibh, f" IBH: {ibh:.2f}", va="center", fontsize=8, color=COLORS["ibh_ibl"])
            ax.text(end_naive, ibl, f" IBL: {ibl:.2f}", va="center", fontsize=8, color=COLORS["ibh_ibl"])

        # EQ
        if eq is not None:
            ax.hlines(
                [eq],
                xmin=start_naive,
                xmax=end_naive,
                colors=COLORS["eq"],
                linewidth=1.0,
                linestyles="--",
                label="EQ"
            )
            ax.text(end_naive, eq, f" EQ: {eq:.2f}", va="center", fontsize=8, color=COLORS["eq"])

    def _prepare_fractal_data(
        self,
        fractal_lists: Optional[Dict],
        m1_data: pd.DataFrame,
        chart_start: datetime,
        chart_end: datetime,
    ) -> List[Dict]:
        """Find unswept fractals at chart_start and compute sweep times within window.

        Returns flat list of dicts ready for drawing:
        {price, type, timeframe, frac_time, sweep_time}
        """
        if not fractal_lists:
            return []

        from src.smc.detectors.fractal_detector import find_unswept_fractals, check_fractal_sweep

        start_utc = chart_start.astimezone(pytz.UTC)
        end_utc = chart_end.astimezone(pytz.UTC)

        result = []
        for tf_key, lookback in [("all_h1", 48), ("all_h4", 96)]:
            all_fracs = fractal_lists.get(tf_key, [])
            unswept = find_unswept_fractals(all_fracs, m1_data, start_utc, lookback_hours=lookback)
            for frac in unswept:
                sweep_time = check_fractal_sweep(frac, m1_data, start_utc, end_utc)
                result.append({
                    "price": frac.price,
                    "type": frac.type,
                    "timeframe": frac.timeframe,
                    "frac_time": frac.time,
                    "sweep_time": sweep_time,
                })

        # Deduplicate: if H1 and H4 at same (type, price) -> keep only H4
        h4_keys = {(r["type"], round(r["price"], 2)) for r in result if r["timeframe"] == "H4"}
        result = [
            r for r in result
            if r["timeframe"] == "H4" or (r["type"], round(r["price"], 2)) not in h4_keys
        ]

        return result

    def _draw_fractals(
        self,
        ax,
        fractal_draw_data: List[Dict],
        chart_start: datetime,
        chart_end: datetime,
    ) -> None:
        """Draw fractal lines ending at sweep point (or dashed to chart edge if unswept).

        H1 fractals: thin black lines.
        H4 fractals: thicker red lines.
        Swept: solid line ending at sweep point.
        Unswept: dashed line extending to chart right edge.
        """
        if not fractal_draw_data:
            return

        start_naive = chart_start.replace(tzinfo=None)
        end_naive = chart_end.replace(tzinfo=None)
        ymin, ymax = ax.get_ylim()

        for frac in fractal_draw_data:
            price = frac["price"]
            if price < ymin or price > ymax:
                continue

            # Convert fractal formation time to local naive
            frac_time = frac["frac_time"]
            if frac_time.tzinfo is None:
                frac_time = pytz.utc.localize(frac_time)
            frac_local = frac_time.astimezone(self.tz).replace(tzinfo=None)

            # Line start: max(chart left edge, fractal formation time)
            line_start = max(start_naive, frac_local)

            # Line end: sweep time or chart right edge
            swept = frac["sweep_time"] is not None
            if swept:
                sweep_time = frac["sweep_time"]
                if sweep_time.tzinfo is None:
                    sweep_time = pytz.utc.localize(sweep_time)
                line_end = sweep_time.astimezone(self.tz).replace(tzinfo=None)
            else:
                line_end = end_naive

            if line_end <= line_start:
                continue

            # Style: H4=red, H1=black; swept=solid, unswept=dashed
            is_h4 = frac["timeframe"] == "H4"
            color = "#d32f2f" if is_h4 else "black"
            linewidth = 1.5 if is_h4 else 0.8
            linestyle = "-" if swept else "--"
            alpha = 0.8 if is_h4 else 0.6

            ax.hlines(
                y=[price], xmin=line_start, xmax=line_end,
                colors=color, linewidth=linewidth, linestyles=linestyle,
                zorder=1, alpha=alpha,
            )

            # Label: "4h H 05:00" or "1h L 12:00" at line_start
            tf_label = "4h" if is_h4 else "1h"
            type_label = "H" if frac["type"] == "high" else "L"
            va = "bottom" if frac["type"] == "high" else "top"
            label_utc = frac_time.strftime("%H:%M")

            ax.text(
                line_start, price, f" {tf_label} {type_label} {label_utc}",
                fontsize=7 if is_h4 else 6, va=va, ha="left",
                color=color, alpha=alpha,
                fontweight="bold" if is_h4 else "normal",
            )

        # Re-apply y-limits to prevent hlines/text from expanding axes
        ax.set_ylim(ymin, ymax)

    def _draw_trade(
        self,
        ax,
        trade: TradeLog,
        chart_start: datetime,
        chart_end: datetime,
    ) -> None:
        """Draw trade entry/exit and zones."""
        entry_time = trade.entry_time
        exit_time = trade.exit_time
        entry_price = trade.entry_price
        exit_price = trade.exit_price
        stop_price = trade.sl
        tp_price = trade.tp
        direction = trade.direction

        if entry_time is None or entry_price is None:
            return

        # Convert times to local naive
        if entry_time.tzinfo is None:
            entry_time = pytz.utc.localize(entry_time)
        entry_local = entry_time.astimezone(self.tz).replace(tzinfo=None)

        if exit_time is not None:
            if exit_time.tzinfo is None:
                exit_time = pytz.utc.localize(exit_time)
            exit_local = exit_time.astimezone(self.tz).replace(tzinfo=None)
        else:
            exit_local = chart_end.replace(tzinfo=None)

        start_naive = chart_start.replace(tzinfo=None)
        end_naive = chart_end.replace(tzinfo=None)

        # Clip to chart window
        entry_local = max(entry_local, start_naive)
        exit_local = min(exit_local, end_naive)

        if exit_local <= entry_local:
            return

        x1 = mdates.date2num(entry_local)
        x2 = mdates.date2num(exit_local)

        # Draw profit/stop zones (TradingView style)
        if stop_price is not None and tp_price is not None:
            if direction == "long":
                # Profit zone (entry to TP)
                profit_lo, profit_hi = sorted([entry_price, tp_price])
                # Stop zone (SL to entry)
                stop_lo, stop_hi = sorted([stop_price, entry_price])
            else:  # short
                # Profit zone (TP to entry)
                profit_lo, profit_hi = sorted([tp_price, entry_price])
                # Stop zone (entry to SL)
                stop_lo, stop_hi = sorted([entry_price, stop_price])

            # Profit zone
            if profit_hi > profit_lo:
                ax.add_patch(Rectangle(
                    (x1, profit_lo),
                    x2 - x1,
                    profit_hi - profit_lo,
                    facecolor=COLORS["profit_zone"],
                    edgecolor="none",
                    alpha=0.15,
                    zorder=0
                ))

            # Stop zone
            if stop_hi > stop_lo:
                ax.add_patch(Rectangle(
                    (x1, stop_lo),
                    x2 - x1,
                    stop_hi - stop_lo,
                    facecolor=COLORS["stop_zone"],
                    edgecolor="none",
                    alpha=0.15,
                    zorder=0
                ))

        # Entry/Exit/SL/TP lines
        ax.hlines([entry_price], xmin=entry_local, xmax=exit_local,
                  colors=COLORS["entry_marker"], linewidth=1.5, linestyles="-", zorder=4)

        if stop_price is not None:
            ax.hlines([stop_price], xmin=entry_local, xmax=exit_local,
                      colors=COLORS["stop_zone"], linewidth=1.0, linestyles="--", zorder=4)

        if tp_price is not None:
            ax.hlines([tp_price], xmin=entry_local, xmax=exit_local,
                      colors=COLORS["profit_zone"], linewidth=1.0, linestyles="--", zorder=4)

        # Entry marker
        ax.scatter([entry_local], [entry_price], color=COLORS["entry_marker"],
                   s=100, marker="^" if direction == "long" else "v", zorder=6,
                   edgecolors="white", linewidth=1)

        # Exit marker
        if exit_price is not None and trade.exit_time is not None:
            exit_local_actual = trade.exit_time
            if exit_local_actual.tzinfo is None:
                exit_local_actual = pytz.utc.localize(exit_local_actual)
            exit_local_actual = exit_local_actual.astimezone(self.tz).replace(tzinfo=None)

            if start_naive <= exit_local_actual <= end_naive:
                ax.scatter([exit_local_actual], [exit_price], color=COLORS["exit_marker"],
                           s=100, marker="x", zorder=6, linewidth=2)

    def _configure_axes(
        self,
        ax,
        df: pd.DataFrame,
        chart_start: datetime,
        chart_end: datetime,
    ) -> None:
        """Configure chart axes."""
        start_naive = chart_start.replace(tzinfo=None)
        end_naive = chart_end.replace(tzinfo=None)

        ax.set_xlim(start_naive, end_naive)

        # Y limits with padding
        if not df.empty:
            ymin = df["low"].min()
            ymax = df["high"].max()
            padding = (ymax - ymin) * 0.05
            ax.set_ylim(ymin - padding, ymax + padding)

        # X axis format
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))

        # Grid
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.3, color=COLORS["grid"])

        # Labels
        ax.set_xlabel(f"Time ({self.timezone})", fontsize=10)
        ax.set_ylabel("Price", fontsize=10)

    def _add_title(self, ax, trade: TradeLog, ib_data: Optional[Dict[str, float]]) -> None:
        """Add chart title."""
        date_str = trade.entry_time.strftime("%Y-%m-%d") if trade.entry_time else "Unknown"
        direction = trade.direction.upper() if trade.direction else "Unknown"
        variation = trade.variation or "Unknown"
        profit = trade.profit or 0
        exit_reason = trade.exit_reason or ""

        # Calculate R (in price units, not USD)
        initial_risk = abs(trade.entry_price - trade.sl) if trade.sl else 0
        if initial_risk > 0:
            # R = price movement / initial risk
            if direction == "LONG":
                price_movement = trade.exit_price - trade.entry_price
            else:  # SHORT
                price_movement = trade.entry_price - trade.exit_price
            r_value = price_movement / initial_risk
        else:
            r_value = 0

        title = f"{trade.symbol} - {date_str} - {variation} {direction}"
        subtitle = f"P/L: ${profit:+.2f} | R: {r_value:+.2f} | Exit: {exit_reason}"

        ax.set_title(f"{title}\n{subtitle}", fontsize=12, fontweight="bold")

    def _add_watermark(self, ax) -> None:
        """Add watermark to chart."""
        ax.text(
            0.995, 0.01,
            "IB Strategy Backtest",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            color="#999999",
            alpha=0.6
        )


def generate_all_trade_charts(
    trade_log: List[TradeLog],
    m1_data: pd.DataFrame,
    output_dir: Path,
    timezone: str,
    ib_data_by_date: Optional[Dict[str, Dict[str, float]]] = None,
    fractal_lists: Optional[Dict] = None,
) -> int:
    """
    Generate charts for all trades.

    Args:
        trade_log: List of TradeLog entries
        m1_data: M1 candlestick data
        output_dir: Directory to save charts
        timezone: Timezone for display
        ib_data_by_date: Dict of date_str -> {ibh, ibl, eq}
        fractal_lists: Dict with all_h1/all_h4 raw fractal lists

    Returns:
        Number of charts generated
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("matplotlib not available, skipping trade charts")
        return 0

    generator = TradeChartGenerator(timezone=timezone)
    charts_generated = 0

    for i, trade in enumerate(trade_log):
        if not trade.exit_time:
            continue  # Skip open trades

        # Get IB data for this trade's date
        ib_data = None
        if trade.entry_time:
            date_key = trade.entry_time.strftime("%Y-%m-%d")
            if ib_data_by_date:
                ib_data = ib_data_by_date.get(date_key)

        # Generate output path
        output_path = output_dir / _get_chart_filename(trade, i)

        # Generate chart
        result = generator.generate_trade_chart(
            trade=trade,
            m1_data=m1_data,
            output_path=output_path,
            ib_data=ib_data,
            fractal_lists=fractal_lists,
        )

        if result:
            charts_generated += 1

    logger.info(f"Generated {charts_generated} trade charts")
    return charts_generated


def _get_chart_filename(trade: TradeLog, index: int) -> str:
    """Generate filename for trade chart."""
    if trade.entry_time:
        date_str = trade.entry_time.strftime("%Y-%m-%d")
    else:
        date_str = "unknown"

    direction = trade.direction or "unknown"
    variation = trade.variation or "unknown"

    return f"{index + 1:03d}_{date_str}_{variation}_{direction}.jpg"
