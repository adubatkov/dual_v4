"""
Market Activity Analysis by Time of Day.

Analyzes price movement (volatility) for 15-min, 30-min and 1-hour intervals.
Shows when the market is most active based on sum of |Close[i] - Close[i-1]|.

Usage:
    python -m strategy_optimization.volatility.analyze_volatility
"""

import sys
from datetime import time as datetime_time
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd
import pytz

# Add paths for imports
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

# Paths
DATA_DIR = ROOT / "data" / "optimized"
OUTPUT_DIR = ROOT / "strategy_optimization" / "volatility" / "results"

# Symbol configurations
SYMBOL_CONFIG = {
    "GER40": {
        "timezone": "Europe/Berlin",
        "parquet": "GER40_m1.parquet",
    },
    "XAUUSD": {
        "timezone": "Asia/Tokyo",
        "parquet": "XAUUSD_m1.parquet",
    },
    "NAS100": {
        "timezone": "America/New_York",
        "parquet": "NAS100_m1.parquet",
    },
    "UK100": {
        "timezone": "Europe/London",
        "parquet": "UK100_m1.parquet",
    },
}

# Timeframes to analyze
TIMEFRAMES = [
    {"rule": "15min", "label": "15min", "x_tick_step": 4},   # show every hour
    {"rule": "30min", "label": "30min", "x_tick_step": 2},   # show every hour
    {"rule": "1h",    "label": "1h",    "x_tick_step": 1},   # show every hour
]


def load_m1_data(symbol: str) -> pd.DataFrame:
    """Load M1 data for a symbol."""
    config = SYMBOL_CONFIG[symbol]
    parquet_path = DATA_DIR / config["parquet"]

    df = pd.read_parquet(parquet_path)

    # Ensure UTC timezone
    if df["time"].dt.tz is None:
        df["time"] = df["time"].dt.tz_localize("UTC")

    return df


def calculate_activity(m1_data: pd.DataFrame, local_tz: str, rule: str = "15min") -> pd.DataFrame:
    """
    Calculate price movement for each time interval.

    Returns DataFrame with columns:
    - time_slot: time of day (e.g., 09:00, 09:15, ...)
    - total_movement_pct: sum of |Close[i] - Close[i-1]| / Close * 100
    - avg_movement_pct: average per trading day
    - trading_days: number of days with data for this interval
    """
    df = m1_data.copy()

    # Convert to local timezone
    tz = pytz.timezone(local_tz)
    df["local_time"] = df["time"].dt.tz_convert(tz)

    # Calculate price change between consecutive candles
    df["price_change"] = df["close"].diff().abs()
    df["price_change_pct"] = df["price_change"] / df["close"] * 100

    # Remove NaN from first row
    df = df.dropna(subset=["price_change_pct"])

    # Floor to given interval
    df["time_slot"] = df["local_time"].dt.floor(rule).dt.time
    df["date"] = df["local_time"].dt.date

    # Group by time slot and aggregate
    grouped = df.groupby("time_slot").agg(
        total_movement_pct=("price_change_pct", "sum"),
        trading_days=("date", "nunique"),
        candle_count=("price_change_pct", "count"),
    ).reset_index()

    # Calculate average per trading day
    grouped["avg_movement_pct"] = grouped["total_movement_pct"] / grouped["trading_days"]

    # Sort by time
    grouped = grouped.sort_values("time_slot").reset_index(drop=True)

    return grouped


def create_activity_chart(
    activity_data: pd.DataFrame,
    symbol: str,
    local_tz: str,
    date_range: Tuple[str, str],
    output_path: Path,
    tf_label: str = "15min",
    x_tick_step: int = 4,
) -> None:
    """Create bar chart showing activity by time of day."""

    fig, ax = plt.subplots(figsize=(16, 8))

    # Prepare data for plotting
    time_labels = [t.strftime("%H:%M") for t in activity_data["time_slot"]]
    values = activity_data["avg_movement_pct"].values

    # Create bar colors based on value (higher = more red)
    max_val = values.max()
    colors = plt.cm.YlOrRd(values / max_val * 0.8 + 0.1)

    # Plot bars
    ax.bar(range(len(time_labels)), values, color=colors, edgecolor="none", width=0.8)

    # Set x-axis labels
    ax.set_xticks(range(0, len(time_labels), x_tick_step))
    ax.set_xticklabels(
        [time_labels[i] for i in range(0, len(time_labels), x_tick_step)],
        rotation=45, ha="right",
    )

    # Labels and title
    ax.set_xlabel(f"Time of Day ({local_tz})", fontsize=12)
    ax.set_ylabel("Average Price Movement (%)", fontsize=12)

    trading_days = activity_data["trading_days"].max()
    ax.set_title(
        f"{symbol} - Market Activity ({tf_label}) by Time of Day\n"
        f"Data: {date_range[0]} to {date_range[1]} ({trading_days} trading days)",
        fontsize=14,
        fontweight="bold",
    )

    # Grid
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    # Session markers
    if symbol == "GER40":
        _add_session_line(ax, activity_data, datetime_time(9, 0), "green", "EU Open (09:00)")
        _add_session_line(ax, activity_data, datetime_time(14, 30), "blue", "US Open (14:30)")
        _add_session_line(ax, activity_data, datetime_time(17, 30), "red", "EU Close (17:30)")
        ax.legend(loc="upper right")
    elif symbol == "XAUUSD":
        _add_session_line(ax, activity_data, datetime_time(9, 0), "red", "Tokyo Open (09:00)")
        _add_session_line(ax, activity_data, datetime_time(16, 0), "green", "London Open (16:00)")
        _add_session_line(ax, activity_data, datetime_time(21, 0), "blue", "NY Open (21:00)")
        ax.legend(loc="upper right")
    elif symbol == "NAS100":
        _add_session_line(ax, activity_data, datetime_time(9, 30), "green", "NYSE Open (09:30)")
        _add_session_line(ax, activity_data, datetime_time(16, 0), "red", "NYSE Close (16:00)")
        ax.legend(loc="upper right")
    elif symbol == "UK100":
        _add_session_line(ax, activity_data, datetime_time(8, 0), "green", "London Open (08:00)")
        _add_session_line(ax, activity_data, datetime_time(14, 30), "blue", "US Open (14:30)")
        _add_session_line(ax, activity_data, datetime_time(16, 30), "red", "London Close (16:30)")
        ax.legend(loc="upper right")

    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"  Chart saved: {output_path}")


def _add_session_line(ax, data, session_time, color, label):
    """Add vertical session marker line."""
    idx = next(
        (i for i, t in enumerate(data["time_slot"]) if t >= session_time),
        None,
    )
    if idx is not None:
        ax.axvline(x=idx - 0.5, color=color, linestyle="--", alpha=0.7, label=label)


def analyze_symbol(symbol: str) -> None:
    """Analyze market activity for a single symbol across all timeframes."""
    print(f"\n{'='*60}")
    print(f"Analyzing {symbol}")
    print("=" * 60)

    config = SYMBOL_CONFIG[symbol]
    local_tz = config["timezone"]

    # Load data once
    print("Loading M1 data...")
    m1_data = load_m1_data(symbol)
    print(f"  Loaded {len(m1_data):,} candles")

    date_range = (
        m1_data["time"].min().strftime("%Y-%m-%d"),
        m1_data["time"].max().strftime("%Y-%m-%d"),
    )
    print(f"  Date range: {date_range[0]} to {date_range[1]}")

    for tf in TIMEFRAMES:
        rule = tf["rule"]
        label = tf["label"]
        step = tf["x_tick_step"]

        print(f"\n--- {label} ---")

        activity = calculate_activity(m1_data, local_tz, rule)
        print(f"  Time slots: {len(activity)}")

        # Save CSV
        csv_path = OUTPUT_DIR / f"{symbol}_activity_{label}.csv"
        activity.to_csv(csv_path, index=False)
        print(f"  Data saved: {csv_path}")

        # Create chart
        chart_path = OUTPUT_DIR / f"{symbol}_activity_{label}.png"
        create_activity_chart(activity, symbol, local_tz, date_range, chart_path, label, step)

        # Summary
        print(f"  Top 3 ({local_tz}):")
        top_3 = activity.nlargest(3, "avg_movement_pct")
        for _, row in top_3.iterrows():
            print(f"    {row['time_slot'].strftime('%H:%M')}: {row['avg_movement_pct']:.4f}%")


def main():
    """Main entry point."""
    print("=" * 60)
    print("Market Activity Analysis (15min, 30min, 1h)")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for symbol in SYMBOL_CONFIG.keys():
        try:
            analyze_symbol(symbol)
        except FileNotFoundError as e:
            print(f"  Skipping {symbol}: {e}")

    print("\n" + "=" * 60)
    print(f"Results saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
