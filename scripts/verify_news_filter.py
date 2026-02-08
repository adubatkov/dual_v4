#!/usr/bin/env python3
"""
Verification script for News Filter integration.

Compares backtest results with and without news filter.
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from datetime import datetime, date

from src.news_filter import NewsFilter, NewsStorage
from params_optimizer.engine.fast_backtest_optimized import FastBacktestOptimized
from params_optimizer.config import DATA_PATHS_OPTIMIZED


def run_comparison(symbol: str = "GER40", num_days: int = 30):
    """Run backtest comparison with and without news filter."""

    print("=" * 60)
    print(f"News Filter Verification for {symbol}")
    print("=" * 60)

    # Load data
    data_path = DATA_PATHS_OPTIMIZED.get(symbol)
    if not data_path or not data_path.exists():
        print(f"[ERROR] Data not found for {symbol}")
        return

    print(f"\n[INFO] Loading data from {data_path}")
    m1_data = pd.read_parquet(data_path)

    # Ensure timezone aware
    if m1_data["time"].dt.tz is None:
        m1_data["time"] = m1_data["time"].dt.tz_localize("UTC")

    print(f"[INFO] Loaded {len(m1_data):,} M1 candles")

    # Limit to recent data for faster testing
    if num_days > 0:
        cutoff_date = m1_data["time"].max() - pd.Timedelta(days=num_days)
        m1_data = m1_data[m1_data["time"] >= cutoff_date]
        print(f"[INFO] Using last {num_days} days: {len(m1_data):,} candles")

    # Test parameters
    params = {
        "ib_start": "08:00",
        "ib_end": "08:30",
        "ib_timezone": "Europe/Berlin",
        "ib_wait_minutes": 0,
        "trade_window_minutes": 60,
        "rr_target": 1.0,
        "stop_mode": "ib_start",
        "tsl_target": 0.0,
        "tsl_sl": 0.0,
        "min_sl_pct": 0.001,
        "rev_rb_enabled": False,
        "rev_rb_pct": 0.5,
        "ib_buffer_pct": 0.0,
        "max_distance_pct": 1.0,
    }

    # Run WITHOUT news filter
    print("\n[1] Running backtest WITHOUT news filter...")
    backtest_no_filter = FastBacktestOptimized(
        symbol=symbol,
        m1_data=m1_data,
        news_filter=None,
    )
    results_no_filter = backtest_no_filter.run_with_params(params)

    print(f"    Total trades: {results_no_filter.get('total_trades', 0)}")
    print(f"    Total R: {results_no_filter.get('total_r', 0):.2f}")
    print(f"    Winrate: {results_no_filter.get('winrate', 0):.1f}%")

    # Create news filter
    print("\n[2] Creating NewsFilter...")
    storage = NewsStorage()
    stats = storage.get_stats()
    print(f"    Years available: {stats['years']}")
    print(f"    High-impact events: {stats['high_impact_events']}")

    news_filter = NewsFilter(
        symbol=symbol,
        storage=storage,
        before_minutes=2,
        after_minutes=2,
    )
    print(f"    Loaded {news_filter.event_count} relevant events for {symbol}")

    # Run WITH news filter
    print("\n[3] Running backtest WITH news filter...")
    backtest_with_filter = FastBacktestOptimized(
        symbol=symbol,
        m1_data=m1_data,
        news_filter=news_filter,
    )
    results_with_filter = backtest_with_filter.run_with_params(params)

    print(f"    Total trades: {results_with_filter.get('total_trades', 0)}")
    print(f"    Total R: {results_with_filter.get('total_r', 0):.2f}")
    print(f"    Winrate: {results_with_filter.get('winrate', 0):.1f}%")

    # Compare results
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)

    trades_diff = results_no_filter.get('total_trades', 0) - results_with_filter.get('total_trades', 0)
    r_diff = results_no_filter.get('total_r', 0) - results_with_filter.get('total_r', 0)

    print(f"Trades filtered by news: {trades_diff}")
    print(f"R difference: {r_diff:+.2f}")
    print(f"News filter stats: {backtest_with_filter.get_news_filter_stats()}")

    if trades_diff > 0:
        print(f"\n[OK] News filter blocked {trades_diff} trades during high-impact news events")
    elif trades_diff == 0:
        print("\n[INFO] No trades were filtered (no news events in test period)")
    else:
        print("\n[WARNING] Unexpected: more trades with filter than without")

    return {
        "without_filter": results_no_filter,
        "with_filter": results_with_filter,
        "trades_filtered": trades_diff,
    }


def show_news_events_in_period(symbol: str = "GER40", days: int = 30):
    """Show news events in the test period."""
    storage = NewsStorage()

    end_date = date.today()
    start_date = date(end_date.year, end_date.month - 1 if end_date.month > 1 else 12, 1)

    print(f"\n[INFO] High-impact news events for {symbol}:")
    from src.news_filter.filter import get_relevant_currencies
    currencies = get_relevant_currencies(symbol)
    print(f"    Relevant currencies: {currencies}")

    events = storage.load_high_impact(
        start_date=start_date,
        end_date=end_date,
        countries=currencies,
    )

    for event in events[:10]:  # Show first 10
        print(f"    - {event}")

    if len(events) > 10:
        print(f"    ... and {len(events) - 10} more events")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Verify news filter integration")
    parser.add_argument("--symbol", default="GER40", help="Symbol to test")
    parser.add_argument("--days", type=int, default=30, help="Number of days to test")

    args = parser.parse_args()

    show_news_events_in_period(args.symbol, args.days)
    run_comparison(args.symbol, args.days)
