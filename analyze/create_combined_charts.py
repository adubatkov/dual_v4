"""
Create combined equity/drawdown charts for GER40 + XAUUSD combinations.

1. Re-run backtests for selected groups to get trade logs
2. Combine trades from each pair, sort by date
3. Calculate combined equity curve and drawdown
4. Generate charts
"""

import sys
import json
from datetime import datetime
from pathlib import Path
from itertools import product

import pytz
import pandas as pd
import matplotlib.pyplot as plt

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backtest.backtest_single_group import run_single_group_backtest

# Selected groups
GER40_GROUPS = [1, 5, 7, 15, 22, 36, 48]
XAUUSD_GROUPS = [1, 12, 28]

# Paths
ANALYZE_DIR = Path(__file__).parent
GROUPS_DIR = ANALYZE_DIR / "parallel_results"
OUTPUT_DIR = ANALYZE_DIR / "parallel_results" / "charts_combined"
TRADES_DIR = ANALYZE_DIR / "parallel_results" / "trades_for_combined"

# Backtest period
START_DATE = datetime(2023, 1, 1, tzinfo=pytz.UTC)
END_DATE = datetime(2025, 10, 31, tzinfo=pytz.UTC)


def load_group_config(symbol: str, group_num: int) -> dict:
    """Load group config from parallel results."""
    group_id = f"{symbol}_{group_num:03d}"
    config_path = GROUPS_DIR / symbol / group_id / "config.json"

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path) as f:
        config = json.load(f)

    return {
        "id": group_id,
        "symbol": symbol,
        "params": config["params"],
        "source_category": config["results"].get("source_category", "unknown"),
        "combined_total_r": config["results"].get("expected_total_r", 0),
        "total_trades": config["results"].get("expected_trades", 0),
    }


def run_backtest_with_trades(symbol: str, group_num: int) -> Path:
    """Run backtest for a group and return path to trades.csv."""
    group_id = f"{symbol}_{group_num:03d}"
    output_dir = TRADES_DIR / group_id
    trades_path = output_dir / "trades.csv"

    # Skip if already exists
    if trades_path.exists():
        print(f"  [SKIP] {group_id} - trades.csv already exists")
        return trades_path

    print(f"  [RUN] {group_id}...")

    # Load group config
    group = load_group_config(symbol, group_num)

    # Run backtest
    result = run_single_group_backtest(
        group=group,
        start_date=START_DATE,
        end_date=END_DATE,
        output_dir=TRADES_DIR,
        skip_charts=True,
    )

    print(f"       -> R={result.get('total_r', 0):.2f}, Trades={result.get('total_trades', 0)}")

    return trades_path


def load_trades(trades_path: Path) -> pd.DataFrame:
    """Load trades from CSV."""
    df = pd.read_csv(trades_path)
    df["exit_time"] = pd.to_datetime(df["exit_time"])
    return df


def create_combined_chart(
    ger40_group: int,
    xauusd_group: int,
    ger40_trades: pd.DataFrame,
    xauusd_trades: pd.DataFrame,
    output_path: Path,
):
    """Create combined equity/drawdown chart."""
    # Combine trades
    combined = pd.concat([ger40_trades, xauusd_trades], ignore_index=True)
    combined = combined.sort_values("exit_time").reset_index(drop=True)

    # Calculate cumulative R
    combined["cumulative_r"] = combined["r"].cumsum()

    # Calculate drawdown
    combined["peak"] = combined["cumulative_r"].cummax()
    combined["drawdown"] = combined["peak"] - combined["cumulative_r"]
    combined["drawdown_pct"] = (combined["drawdown"] / combined["peak"].replace(0, 1)) * 100

    # Stats
    total_r = combined["cumulative_r"].iloc[-1]
    max_dd = combined["drawdown"].max()
    max_dd_pct = combined["drawdown_pct"].max()
    total_trades = len(combined)
    ger40_trades_count = len(ger40_trades)
    xauusd_trades_count = len(xauusd_trades)

    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Equity curve
    ax1.plot(combined["exit_time"], combined["cumulative_r"], "b-", linewidth=1.5)
    ax1.fill_between(combined["exit_time"], 0, combined["cumulative_r"], alpha=0.3)
    ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax1.set_ylabel("Cumulative R", fontsize=12)
    ax1.set_title(
        f"Combined: GER40_{ger40_group:03d} + XAUUSD_{xauusd_group:03d}\n"
        f"Total R: {total_r:.2f} | Trades: {total_trades} (GER40: {ger40_trades_count}, XAUUSD: {xauusd_trades_count})",
        fontsize=14,
    )
    ax1.grid(True, alpha=0.3)
    ax1.legend([f"Equity (R={total_r:.2f})"], loc="upper left")

    # Drawdown
    ax2.fill_between(combined["exit_time"], 0, -combined["drawdown"], color="red", alpha=0.5)
    ax2.set_ylabel("Drawdown (R)", fontsize=12)
    ax2.set_xlabel("Date", fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend([f"Max DD: {max_dd:.2f} R ({max_dd_pct:.1f}%)"], loc="lower left")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return {
        "ger40_group": f"GER40_{ger40_group:03d}",
        "xauusd_group": f"XAUUSD_{xauusd_group:03d}",
        "total_r": round(total_r, 2),
        "max_dd_r": round(max_dd, 2),
        "max_dd_pct": round(max_dd_pct, 2),
        "total_trades": total_trades,
        "ger40_trades": ger40_trades_count,
        "xauusd_trades": xauusd_trades_count,
    }


def main():
    # Create output directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TRADES_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("STEP 1: Running backtests to get trade logs")
    print("=" * 60)

    # Run backtests for GER40 groups
    print("\nGER40 groups:")
    ger40_trades = {}
    for g in GER40_GROUPS:
        trades_path = run_backtest_with_trades("GER40", g)
        ger40_trades[g] = load_trades(trades_path)

    # Run backtests for XAUUSD groups
    print("\nXAUUSD groups:")
    xauusd_trades = {}
    for g in XAUUSD_GROUPS:
        trades_path = run_backtest_with_trades("XAUUSD", g)
        xauusd_trades[g] = load_trades(trades_path)

    print("\n" + "=" * 60)
    print("STEP 2: Creating combined charts")
    print("=" * 60)

    # Create all combinations
    results = []
    for ger40_g, xauusd_g in product(GER40_GROUPS, XAUUSD_GROUPS):
        chart_name = f"GER40_{ger40_g:03d}_XAUUSD_{xauusd_g:03d}.png"
        output_path = OUTPUT_DIR / chart_name

        print(f"  Creating {chart_name}...")

        result = create_combined_chart(
            ger40_group=ger40_g,
            xauusd_group=xauusd_g,
            ger40_trades=ger40_trades[ger40_g],
            xauusd_trades=xauusd_trades[xauusd_g],
            output_path=output_path,
        )
        results.append(result)

    # Save summary
    summary_df = pd.DataFrame(results)
    summary_df = summary_df.sort_values("total_r", ascending=False)
    summary_df.to_csv(OUTPUT_DIR / "combined_summary.csv", index=False)
    summary_df.to_excel(OUTPUT_DIR / "combined_summary.xlsx", index=False)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\nCreated {len(results)} combined charts in: {OUTPUT_DIR}")
    print("\nTop 5 combinations by Total R:")
    print(summary_df[["ger40_group", "xauusd_group", "total_r", "max_dd_r", "total_trades"]].head().to_string(index=False))


if __name__ == "__main__":
    main()
