"""
Generate trade charts with 1H fractal sweep lines.

Reads trades from backtest results and regenerates charts with:
- Original chart elements (candles, IB zone, trade markers)
- Black horizontal lines for swept 1H fractals
- Timestamp labels on fractal lines

Usage:
    python generate_fractal_charts.py
"""

import sys
import json
import logging
from datetime import datetime, time as datetime_time, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

import pandas as pd
import pytz

# Add paths for imports
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle

from backtest.reporting.trade_charts import TradeChartGenerator, COLORS, FIGSIZE, DPI
from strategy_optimization.fractals.fractals import get_swept_fractals_for_trade

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Paths
BACKTEST_OUTPUT = ROOT / "backtest" / "output" / "parallel_control"
DATA_DIR = ROOT / "data" / "control"
OUTPUT_DIR = ROOT / "strategy_optimization" / "fractals" / "results"

# Groups to process
GROUPS = ["GER40_055", "XAUUSD_059", "GER40_006", "XAUUSD_010"]

SYMBOL_TIMEZONES = {
    "GER40": "Europe/Berlin",
    "XAUUSD": "Asia/Tokyo",
}


class FractalChartGenerator(TradeChartGenerator):
    """Extended chart generator with fractal sweep lines."""

    def __init__(self, timezone: str = "Europe/Berlin"):
        super().__init__(timezone)

    def generate_chart_with_fractals(
        self,
        trade_info: Dict[str, Any],
        m1_data: pd.DataFrame,
        output_path: Path,
        ib_data: Optional[Dict[str, float]] = None,
        ib_start_time: str = "09:00",
        ib_end_time: str = "09:30",
        trade_window_minutes: int = 150,
        lookback_hours: int = 48,
    ) -> Optional[Path]:
        """
        Generate chart with fractal sweep lines.

        Args:
            trade_info: Dict with entry_time, exit_time, direction, etc.
            m1_data: Full M1 data (UTC)
            output_path: Where to save chart
            ib_data: Dict with ibh, ibl, eq
            ib_start_time: IB start time string (e.g., "09:00")
            ib_end_time: IB end time string (e.g., "09:30")
            trade_window_minutes: Trade window length in minutes
            lookback_hours: How far back to look for fractals

        Returns:
            Path to saved chart or None
        """
        entry_time = trade_info["entry_time"]
        exit_time = trade_info.get("exit_time")
        direction = trade_info.get("direction", "unknown")
        profit = trade_info.get("profit", 0)
        r_value = trade_info.get("r", 0)
        variation = trade_info.get("variation", "unknown")
        symbol = trade_info.get("symbol", "unknown")

        if entry_time is None:
            logger.warning("Trade has no entry time, skipping")
            return None

        # Parse entry_time if string
        if isinstance(entry_time, str):
            entry_time = pd.to_datetime(entry_time)
        if entry_time.tzinfo is None:
            entry_time = pytz.utc.localize(entry_time)

        # Calculate IB start and trade window end
        entry_local = entry_time.astimezone(self.tz)
        trade_date = entry_local.date()

        ib_start_local = self.tz.localize(datetime.combine(
            trade_date,
            datetime.strptime(ib_start_time, "%H:%M").time()
        ))
        ib_end_local = self.tz.localize(datetime.combine(
            trade_date,
            datetime.strptime(ib_end_time, "%H:%M").time()
        ))

        ib_start_utc = ib_start_local.astimezone(pytz.utc)
        ib_end_utc = ib_end_local.astimezone(pytz.utc)

        # Trade window end = IB end + TRADE_WINDOW minutes
        window_end_utc = ib_end_utc + timedelta(minutes=trade_window_minutes)

        # Chart window: 1 hour before IB start to 4.5 hours after IB start
        chart_start = ib_start_local - timedelta(hours=1)
        chart_end = ib_start_local + timedelta(hours=4, minutes=30)

        # Chart start in UTC (for fractal sweep detection)
        chart_start_utc = chart_start.astimezone(pytz.utc)

        # Get swept fractals - from chart start to window end
        swept_fractals = get_swept_fractals_for_trade(
            m1_data=m1_data,
            ib_start=chart_start_utc,
            window_end=window_end_utc,
            lookback_hours=lookback_hours,
        )

        # Resample M1 to M2 candles and filter to chart window
        m2_data = self._resample_to_m2(m1_data)
        df_chart = self._filter_candles(m2_data, chart_start, chart_end)
        if df_chart.empty:
            logger.warning(f"No candle data for trade on {trade_date}")
            return None

        # Create figure
        fig, ax = plt.subplots(figsize=FIGSIZE)

        # Draw original components
        self._draw_candlesticks(ax, df_chart)
        self._draw_ib_zone_custom(ax, df_chart, ib_data, chart_start, chart_end, ib_start_time, ib_end_time)
        self._draw_levels(ax, ib_data, chart_start, chart_end)
        self._draw_trade_custom(ax, trade_info, chart_start, chart_end)

        # Draw fractal sweeps
        self._draw_fractal_sweeps(ax, swept_fractals, chart_start, chart_end)

        # Configure axes
        self._configure_axes(ax, df_chart, chart_start, chart_end)

        # Title
        self._add_title_custom(ax, trade_info, symbol, trade_date, variation, direction, profit, r_value)

        # Watermark
        self._add_watermark(ax)

        # Save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=DPI, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        logger.debug(f"Fractal chart saved: {output_path}")
        return output_path

    def _resample_to_m2(self, m1_data: pd.DataFrame) -> pd.DataFrame:
        """Resample M1 data to M2 (2-minute) candles."""
        df = m1_data.copy()

        if df["time"].dt.tz is None:
            df["time"] = df["time"].dt.tz_localize("UTC")

        df = df.set_index("time")

        agg_dict = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
        }

        if "tick_volume" in df.columns:
            agg_dict["tick_volume"] = "sum"
        elif "volume" in df.columns:
            agg_dict["volume"] = "sum"

        m2 = df.resample("2min").agg(agg_dict).dropna()
        m2 = m2.reset_index().rename(columns={"time": "time"})

        return m2

    def _draw_ib_zone_custom(
        self,
        ax,
        df: pd.DataFrame,
        ib_data: Optional[Dict[str, float]],
        chart_start: datetime,
        chart_end: datetime,
        ib_start_time: str,
        ib_end_time: str,
    ) -> None:
        """Draw IB zone with custom times."""
        if ib_data is None:
            return

        ibh = ib_data.get("ibh")
        ibl = ib_data.get("ibl")

        if ibh is None or ibl is None or ibh <= ibl:
            return

        trade_date = chart_start.date()
        ib_start = datetime.combine(trade_date, datetime.strptime(ib_start_time, "%H:%M").time())
        ib_end = datetime.combine(trade_date, datetime.strptime(ib_end_time, "%H:%M").time())

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

    def _draw_trade_custom(
        self,
        ax,
        trade_info: Dict[str, Any],
        chart_start: datetime,
        chart_end: datetime,
    ) -> None:
        """Draw trade markers from trade_info dict."""
        entry_time = trade_info.get("entry_time")
        exit_time = trade_info.get("exit_time")
        entry_price = trade_info.get("entry_price")
        exit_price = trade_info.get("exit_price")
        stop_price = trade_info.get("sl")
        tp_price = trade_info.get("tp")
        direction = trade_info.get("direction", "long")

        if entry_time is None:
            return

        # Convert to datetime if string
        if isinstance(entry_time, str):
            entry_time = pd.to_datetime(entry_time)
        if entry_time.tzinfo is None:
            entry_time = pytz.utc.localize(entry_time)

        if isinstance(exit_time, str):
            exit_time = pd.to_datetime(exit_time)
        if exit_time is not None and exit_time.tzinfo is None:
            exit_time = pytz.utc.localize(exit_time)

        entry_local = entry_time.astimezone(self.tz).replace(tzinfo=None)
        exit_local = exit_time.astimezone(self.tz).replace(tzinfo=None) if exit_time else chart_end.replace(tzinfo=None)

        start_naive = chart_start.replace(tzinfo=None)
        end_naive = chart_end.replace(tzinfo=None)

        entry_local = max(entry_local, start_naive)
        exit_local = min(exit_local, end_naive)

        if exit_local <= entry_local:
            return

        # If we don't have price info from trade_info, skip detailed drawing
        if entry_price is None:
            return

        x1 = mdates.date2num(entry_local)
        x2 = mdates.date2num(exit_local)

        # Draw zones if we have SL/TP
        if stop_price is not None and tp_price is not None:
            if direction == "long":
                profit_lo, profit_hi = sorted([entry_price, tp_price])
                stop_lo, stop_hi = sorted([stop_price, entry_price])
            else:
                profit_lo, profit_hi = sorted([tp_price, entry_price])
                stop_lo, stop_hi = sorted([entry_price, stop_price])

            if profit_hi > profit_lo:
                ax.add_patch(Rectangle(
                    (x1, profit_lo), x2 - x1, profit_hi - profit_lo,
                    facecolor=COLORS["profit_zone"], edgecolor="none", alpha=0.15, zorder=0
                ))

            if stop_hi > stop_lo:
                ax.add_patch(Rectangle(
                    (x1, stop_lo), x2 - x1, stop_hi - stop_lo,
                    facecolor=COLORS["stop_zone"], edgecolor="none", alpha=0.15, zorder=0
                ))

        # Entry line
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
        if exit_price is not None:
            ax.scatter([exit_local], [exit_price], color=COLORS["exit_marker"],
                       s=100, marker="x", zorder=6, linewidth=2)

    def _draw_fractal_sweeps(
        self,
        ax,
        swept_fractals: List[Dict],
        chart_start: datetime,
        chart_end: datetime,
    ) -> None:
        """
        Draw black horizontal lines for swept fractals.

        Lines go from chart left edge (or fractal time if visible) to sweep time.
        """
        if not swept_fractals:
            return

        start_naive = chart_start.replace(tzinfo=None)
        end_naive = chart_end.replace(tzinfo=None)

        for frac in swept_fractals:
            frac_time = frac["fractal_time"]
            sweep_time = frac["sweep_time"]
            price = frac["fractal_price"]
            frac_type = frac["fractal_type"]

            # Convert to local naive
            if frac_time.tzinfo is not None:
                frac_local = frac_time.astimezone(self.tz).replace(tzinfo=None)
            else:
                frac_local = pytz.utc.localize(frac_time).astimezone(self.tz).replace(tzinfo=None)

            if sweep_time.tzinfo is not None:
                sweep_local = sweep_time.astimezone(self.tz).replace(tzinfo=None)
            else:
                sweep_local = pytz.utc.localize(sweep_time).astimezone(self.tz).replace(tzinfo=None)

            # Clip to chart window
            if sweep_local < start_naive or sweep_local > end_naive:
                # Sweep not visible
                continue

            # Line starts at chart left or fractal time (whichever is later)
            line_start = max(start_naive, frac_local)
            line_end = sweep_local

            if line_end <= line_start:
                continue

            # Draw black horizontal line
            ax.hlines(
                y=[price],
                xmin=line_start,
                xmax=line_end,
                colors="black",
                linewidth=1.5,
                linestyles="-",
                zorder=5,
            )

            # Add label with fractal formation time (UTC)
            label_time = frac_time
            if label_time.tzinfo is None:
                label_time = pytz.utc.localize(label_time)
            label = label_time.strftime("%H:%M UTC")

            # Position label at line start
            ax.text(
                line_start,
                price,
                f" {label}",
                fontsize=8,
                va="bottom" if frac_type == "high" else "top",
                ha="left",
                color="black",
                fontweight="bold",
            )

    def _add_title_custom(
        self, ax, trade_info, symbol, trade_date, variation, direction, profit, r_value
    ) -> None:
        """Add chart title."""
        date_str = trade_date.strftime("%Y-%m-%d") if trade_date else "Unknown"
        direction_str = direction.upper() if direction else "Unknown"
        exit_reason = trade_info.get("exit_reason", "")

        title = f"{symbol} - {date_str} - {variation} {direction_str}"
        subtitle = f"P/L: ${profit:+.2f} | R: {r_value:+.2f} | Exit: {exit_reason}"

        ax.set_title(f"{title}\n{subtitle}", fontsize=12, fontweight="bold")


def load_ib_data_from_results(group_dir: Path) -> Dict[str, Dict[str, float]]:
    """Load IB data from results.xlsx."""
    xlsx_path = group_dir / "results.xlsx"
    if not xlsx_path.exists():
        return {}

    try:
        # Sheet name is "Trades" with capital T
        df = pd.read_excel(xlsx_path, sheet_name="Trades")
        ib_data = {}

        for _, row in df.iterrows():
            # Use "Date" column for date (Entry Time only contains time, not date)
            date_val = row.get("Date")
            if pd.notna(date_val):
                date_str = pd.to_datetime(date_val).strftime("%Y-%m-%d")
                if date_str not in ib_data:
                    # Columns are capitalized: IBH, IBL, EQ
                    ibh = row.get("IBH")
                    ibl = row.get("IBL")
                    eq = row.get("EQ")

                    if pd.notna(ibh) and pd.notna(ibl):
                        ib_data[date_str] = {
                            "ibh": float(ibh),
                            "ibl": float(ibl),
                            "eq": float(eq) if pd.notna(eq) else (float(ibh) + float(ibl)) / 2,
                        }

        return ib_data
    except Exception as e:
        logger.warning(f"Could not load IB data from {xlsx_path}: {e}")
        return {}


def load_trades_from_excel(group_dir: Path) -> Optional[pd.DataFrame]:
    """Load full trade info from results.xlsx Trades sheet."""
    xlsx_path = group_dir / "results.xlsx"
    if not xlsx_path.exists():
        return None

    try:
        df = pd.read_excel(xlsx_path, sheet_name="Trades")
        return df
    except Exception as e:
        logger.warning(f"Could not load trades from {xlsx_path}: {e}")
        return None


def parse_trade_from_filename(filename: str, trades_df: pd.DataFrame, excel_df: Optional[pd.DataFrame] = None) -> Optional[Dict]:
    """
    Parse trade info from filename and match with trades data.

    Filename format: 051_2026-01-30_TCWE_long.jpg
    """
    name = Path(filename).stem
    parts = name.split("_")

    if len(parts) < 4:
        return None

    try:
        index = int(parts[0]) - 1  # 1-indexed to 0-indexed
        date_str = parts[1]
        variation = parts[2]
        direction = parts[3]

        if index < 0 or index >= len(trades_df):
            return None

        row = trades_df.iloc[index]

        result = {
            "entry_time": row["entry_time"],
            "exit_time": row.get("exit_time"),
            "symbol": row.get("symbol", ""),
            "direction": row.get("direction", direction),
            "profit": row.get("profit", 0),
            "r": row.get("r", 0),
            "variation": variation,
            "index": index + 1,
        }

        # If we have Excel data, get entry/exit prices and SL/TP
        if excel_df is not None and index < len(excel_df):
            excel_row = excel_df.iloc[index]
            result["entry_price"] = excel_row.get("Entry Price")
            result["exit_price"] = excel_row.get("Exit Price")
            result["sl"] = excel_row.get("Stop Loss")
            result["tp"] = excel_row.get("Take Profit")
            result["exit_reason"] = excel_row.get("Exit Reason", "")

        return result
    except Exception as e:
        logger.warning(f"Could not parse {filename}: {e}")
        return None


def process_group(group_id: str, m1_data: pd.DataFrame) -> int:
    """Process all trades for a single group."""
    group_dir = BACKTEST_OUTPUT / group_id
    if not group_dir.exists():
        logger.warning(f"Group directory not found: {group_dir}")
        return 0

    trades_csv = group_dir / "trades.csv"
    config_json = group_dir / "config.json"
    trades_dir = group_dir / "trades"

    if not trades_csv.exists():
        logger.warning(f"trades.csv not found for {group_id}")
        return 0

    # Load trades
    trades_df = pd.read_csv(trades_csv)

    # Load config to get IB times and trade windows
    config = {}
    if config_json.exists():
        with open(config_json) as f:
            config = json.load(f)

    symbol = config.get("symbol", group_id.split("_")[0])
    params = config.get("params", {})

    # Get timezone
    tz_str = SYMBOL_TIMEZONES.get(symbol, "UTC")

    # Load IB data from results.xlsx
    ib_data_by_date = load_ib_data_from_results(group_dir)

    # Load full trade info from Excel
    excel_df = load_trades_from_excel(group_dir)

    # Output directory
    output_group_dir = OUTPUT_DIR / group_id / "trades"
    output_group_dir.mkdir(parents=True, exist_ok=True)

    # Create generator
    generator = FractalChartGenerator(timezone=tz_str)

    charts_generated = 0

    # List existing trade chart files
    if not trades_dir.exists():
        logger.warning(f"trades directory not found for {group_id}")
        return 0

    chart_files = sorted(trades_dir.glob("*.jpg"))

    for chart_file in chart_files:
        trade_info = parse_trade_from_filename(chart_file.name, trades_df, excel_df)
        if trade_info is None:
            continue

        variation = trade_info.get("variation", "OCAE")
        var_params = params.get(variation, params.get("OCAE", {}))

        ib_start = var_params.get("IB_START", "09:00")
        ib_end = var_params.get("IB_END", "09:30")
        trade_window = var_params.get("TRADE_WINDOW", 150)

        # Get IB data for this date
        entry_time = trade_info["entry_time"]
        if isinstance(entry_time, str):
            entry_time = pd.to_datetime(entry_time)
        date_str = entry_time.strftime("%Y-%m-%d")
        ib_data = ib_data_by_date.get(date_str)

        trade_info["symbol"] = symbol

        # Output path
        output_path = output_group_dir / chart_file.name

        result = generator.generate_chart_with_fractals(
            trade_info=trade_info,
            m1_data=m1_data,
            output_path=output_path,
            ib_data=ib_data,
            ib_start_time=ib_start,
            ib_end_time=ib_end,
            trade_window_minutes=trade_window,
            lookback_hours=48,
        )

        if result:
            charts_generated += 1

        if charts_generated % 10 == 0:
            logger.info(f"  {group_id}: Generated {charts_generated} charts...")

    return charts_generated


def main():
    """Main entry point."""
    logger.info("=" * 60)
    logger.info("Generating fractal sweep charts")
    logger.info("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load M1 data for each symbol
    m1_data_cache = {}

    for symbol in ["GER40", "XAUUSD"]:
        parquet_path = DATA_DIR / f"{symbol}_m1.parquet"
        if parquet_path.exists():
            logger.info(f"Loading {symbol} M1 data...")
            df = pd.read_parquet(parquet_path)
            if df["time"].dt.tz is None:
                df["time"] = df["time"].dt.tz_localize("UTC")
            m1_data_cache[symbol] = df
            logger.info(f"  Loaded {len(df):,} candles")
        else:
            logger.warning(f"M1 data not found: {parquet_path}")

    total_charts = 0

    for group_id in GROUPS:
        logger.info(f"\nProcessing {group_id}...")

        symbol = group_id.split("_")[0]
        m1_data = m1_data_cache.get(symbol)

        if m1_data is None:
            logger.warning(f"No M1 data for {symbol}, skipping {group_id}")
            continue

        count = process_group(group_id, m1_data)
        total_charts += count
        logger.info(f"  {group_id}: Generated {count} charts")

    logger.info("=" * 60)
    logger.info(f"Total charts generated: {total_charts}")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
