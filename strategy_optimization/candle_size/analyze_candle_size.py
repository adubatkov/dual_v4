"""
Candle Size Analysis by Time of Day.

Resamples M1 data to 15-min, 30-min and 1-hour candles, calculates
directional movement |Open - Close| in %, and averages by time of day.

Usage:
    python -m strategy_optimization.candle_size.analyze_candle_size
"""

import sys
from datetime import time as datetime_time
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import pytz

# Add paths for imports
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

# Paths
DATA_DIR = ROOT / "data" / "optimized"
OUTPUT_DIR = ROOT / "strategy_optimization" / "candle_size" / "results"

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


def load_m1_local(symbol: str) -> pd.DataFrame:
    """Load M1 data and convert to local timezone index."""
    config = SYMBOL_CONFIG[symbol]
    local_tz = config["timezone"]

    df = pd.read_parquet(DATA_DIR / config["parquet"])

    if df["time"].dt.tz is None:
        df["time"] = df["time"].dt.tz_localize("UTC")

    df["local_time"] = df["time"].dt.tz_convert(local_tz)
    df = df.set_index("local_time")
    return df


def resample_candles(m1_local: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample M1 data to given timeframe and calculate body size."""
    candles = m1_local.resample(rule).agg({
        "open": "first",
        "close": "last",
    }).dropna()

    candles["body_pct"] = ((candles["close"] - candles["open"]).abs()
                           / candles["open"] * 100)

    return candles


def calculate_avg_by_time(candles_15m: pd.DataFrame) -> pd.DataFrame:
    """
    Average candle body size by time of day.

    Returns DataFrame with columns: time_slot, avg_body_pct, trading_days, candle_count
    """
    df = candles_15m.copy()
    df["time_slot"] = df.index.time
    df["date"] = df.index.date

    grouped = df.groupby("time_slot").agg(
        total_body_pct=("body_pct", "sum"),
        trading_days=("date", "nunique"),
        candle_count=("body_pct", "count"),
    ).reset_index()

    grouped["avg_body_pct"] = grouped["total_body_pct"] / grouped["trading_days"]
    grouped = grouped.sort_values("time_slot").reset_index(drop=True)

    return grouped


def create_chart(
    data: pd.DataFrame,
    symbol: str,
    local_tz: str,
    date_range: tuple,
    output_path: Path,
    tf_label: str = "15min",
    x_tick_step: int = 4,
) -> None:
    """Create bar chart showing average candle body by time of day."""
    fig, ax = plt.subplots(figsize=(16, 8))

    time_labels = [t.strftime("%H:%M") for t in data["time_slot"]]
    values = data["avg_body_pct"].values

    # Color by value
    max_val = values.max()
    colors = plt.cm.YlOrRd(values / max_val * 0.8 + 0.1)

    ax.bar(range(len(time_labels)), values, color=colors, edgecolor="none", width=0.8)

    # X-axis labels
    ax.set_xticks(range(0, len(time_labels), x_tick_step))
    ax.set_xticklabels(
        [time_labels[i] for i in range(0, len(time_labels), x_tick_step)],
        rotation=45, ha="right",
    )

    ax.set_xlabel(f"Time of Day ({local_tz})", fontsize=12)
    ax.set_ylabel("Average |Open - Close| (%)", fontsize=12)

    trading_days = data["trading_days"].max()
    ax.set_title(
        f"{symbol} - Average {tf_label} Candle Body Size\n"
        f"Data: {date_range[0]} to {date_range[1]} ({trading_days} trading days)",
        fontsize=14,
        fontweight="bold",
    )

    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    # Session markers
    if symbol == "GER40":
        _add_session_line(ax, data, datetime_time(9, 0), "green", "EU Open (09:00)")
        _add_session_line(ax, data, datetime_time(14, 30), "blue", "US Open (14:30)")
        _add_session_line(ax, data, datetime_time(17, 30), "red", "EU Close (17:30)")
        ax.legend(loc="upper right")
    elif symbol == "XAUUSD":
        _add_session_line(ax, data, datetime_time(9, 0), "red", "Tokyo Open (09:00)")
        _add_session_line(ax, data, datetime_time(16, 0), "green", "London Open (16:00)")
        _add_session_line(ax, data, datetime_time(21, 0), "blue", "NY Open (21:00)")
        ax.legend(loc="upper right")
    elif symbol == "NAS100":
        _add_session_line(ax, data, datetime_time(9, 30), "green", "NYSE Open (09:30)")
        _add_session_line(ax, data, datetime_time(16, 0), "red", "NYSE Close (16:00)")
        ax.legend(loc="upper right")
    elif symbol == "UK100":
        _add_session_line(ax, data, datetime_time(8, 0), "green", "London Open (08:00)")
        _add_session_line(ax, data, datetime_time(14, 30), "blue", "US Open (14:30)")
        _add_session_line(ax, data, datetime_time(16, 30), "red", "London Close (16:30)")
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
    """Analyze candle size for a single symbol across all timeframes."""
    config = SYMBOL_CONFIG[symbol]
    local_tz = config["timezone"]

    print(f"\n{'='*60}")
    print(f"Analyzing {symbol}")
    print("=" * 60)

    # Load M1 data once
    print("Loading M1 data...")
    m1_local = load_m1_local(symbol)
    print(f"  M1 candles: {len(m1_local):,}")

    for tf in TIMEFRAMES:
        rule = tf["rule"]
        label = tf["label"]
        step = tf["x_tick_step"]

        print(f"\n--- {label} ---")

        # Resample
        candles = resample_candles(m1_local, rule)
        print(f"  {label} candles: {len(candles):,}")

        date_range = (
            candles.index.min().strftime("%Y-%m-%d"),
            candles.index.max().strftime("%Y-%m-%d"),
        )

        # Calculate averages
        avg_data = calculate_avg_by_time(candles)
        print(f"  Time slots: {len(avg_data)}")

        # Save CSV
        csv_path = OUTPUT_DIR / f"{symbol}_candle_size_{label}.csv"
        avg_data.to_csv(csv_path, index=False)
        print(f"  Data saved: {csv_path}")

        # Create chart
        chart_path = OUTPUT_DIR / f"{symbol}_candle_size_{label}.png"
        create_chart(avg_data, symbol, local_tz, date_range, chart_path, label, step)

        # Summary
        print(f"  Top 3 ({local_tz}):")
        top_3 = avg_data.nlargest(3, "avg_body_pct")
        for _, row in top_3.iterrows():
            print(f"    {row['time_slot'].strftime('%H:%M')}: {row['avg_body_pct']:.4f}%")


def main():
    """Main entry point."""
    print("=" * 60)
    print("Candle Size Analysis by Time of Day (15min, 30min, 1h)")
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
