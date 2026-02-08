#!/usr/bin/env python3
"""
IB Pre-compute Module for Parameter Optimization.

Pre-calculates Initial Balance (IBH, IBL, EQ) for all days and all IB time configs.
This eliminates redundant IB calculations across millions of backtests.

Usage:
    python -m params_optimizer.data.ib_precompute --symbol GER40
    python -m params_optimizer.data.ib_precompute --symbol XAUUSD
"""

import argparse
import pickle
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import pandas as pd
import pytz

from params_optimizer.config import (
    IB_TIME_CONFIGS,
    DATA_PATHS_OPTIMIZED,
    print_status,
)
from params_optimizer.data.loader import load_data


def parse_session_time(t_str: str) -> time:
    """Parse time string 'HH:MM' to time object."""
    return datetime.strptime(t_str, "%H:%M").time()


def ib_window_on_date(
    local_date: datetime.date,
    start_str: str,
    end_str: str,
    timezone_str: str
) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Get IB window timestamps for given date."""
    tz = pytz.timezone(timezone_str)
    start_naive = datetime.combine(local_date, parse_session_time(start_str))
    end_naive = datetime.combine(local_date, parse_session_time(end_str))
    start = tz.localize(start_naive)
    end = tz.localize(end_naive)
    return (pd.Timestamp(start), pd.Timestamp(end))


def compute_ib_for_day(
    day_df: pd.DataFrame,
    day_date: datetime.date,
    start_str: str,
    end_str: str,
    timezone_str: str
) -> Optional[Dict[str, float]]:
    """
    Compute Initial Balance for a specific day and IB config.

    Args:
        day_df: M1 data for the day
        day_date: Date to compute IB for
        start_str: IB start time "HH:MM"
        end_str: IB end time "HH:MM"
        timezone_str: Timezone for IB times

    Returns:
        Dict with IBH, IBL, EQ or None if no data
    """
    start_ib, end_ib = ib_window_on_date(day_date, start_str, end_str, timezone_str)
    df_ib = day_df[(day_df["time"] >= start_ib) & (day_df["time"] < end_ib)]

    if df_ib.empty:
        return None

    ib_high = float(df_ib["high"].max())
    ib_low = float(df_ib["low"].min())
    eq = (ib_high + ib_low) / 2.0

    return {"IBH": ib_high, "IBL": ib_low, "EQ": eq}


def precompute_ib_cache(
    symbol: str,
    m1_data: pd.DataFrame,
    ib_configs: list
) -> Dict[Tuple[str, str, str], Dict[datetime.date, Dict[str, float]]]:
    """
    Pre-compute IB cache for all days and all IB configs.

    Args:
        symbol: Trading symbol
        m1_data: M1 candle data
        ib_configs: List of (start, end, timezone) tuples

    Returns:
        Nested dict: {(start, end, tz): {date: {IBH, IBL, EQ}}}
    """
    cache = {}

    print_status(f"Pre-computing IB cache for {symbol}", "HEADER")
    print_status(f"IB configs: {len(ib_configs)}", "INFO")

    # Process each IB config
    for ib_start, ib_end, ib_tz in ib_configs:
        config_key = (ib_start, ib_end, ib_tz)
        print_status(f"Processing {ib_start}-{ib_end} {ib_tz}...", "INFO")

        cache[config_key] = {}

        # Compute IB date column for this timezone
        tz = pytz.timezone(ib_tz)
        m1_data["_ib_date"] = m1_data["time"].apply(lambda x: x.astimezone(tz).date())

        # Process each day
        days_processed = 0
        days_with_ib = 0

        for day_date, day_df in m1_data.groupby("_ib_date"):
            days_processed += 1
            day_df = day_df.sort_values("time")

            ib = compute_ib_for_day(day_df, day_date, ib_start, ib_end, ib_tz)
            if ib:
                cache[config_key][day_date] = ib
                days_with_ib += 1

        print_status(f"  Days: {days_processed}, with IB: {days_with_ib}", "SUCCESS")

        # Clean up temp column
        m1_data.drop(columns=["_ib_date"], inplace=True, errors="ignore")

    return cache


def save_cache(cache: Dict, output_path: Path) -> None:
    """Save IB cache to pickle file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    print_status(f"Cache saved: {output_path}", "SUCCESS")


def load_cache(cache_path: Path) -> Optional[Dict]:
    """Load IB cache from pickle file."""
    if not cache_path.exists():
        return None

    with open(cache_path, "rb") as f:
        return pickle.load(f)


def get_cache_path(symbol: str) -> Path:
    """Get path to IB cache file for symbol."""
    return Path(__file__).parent / "optimized" / f"{symbol}_ib_cache.pkl"


def main():
    parser = argparse.ArgumentParser(description="Pre-compute IB cache for optimization")
    parser.add_argument("--symbol", choices=["GER40", "XAUUSD"], required=True,
                        help="Symbol to pre-compute IB for")
    parser.add_argument("--force", action="store_true",
                        help="Force re-computation even if cache exists")
    args = parser.parse_args()

    symbol = args.symbol
    cache_path = get_cache_path(symbol)

    # Check if cache already exists
    if cache_path.exists() and not args.force:
        print_status(f"Cache already exists: {cache_path}", "WARNING")
        print_status("Use --force to re-compute", "INFO")
        return

    # Load M1 data
    print_status(f"Loading M1 data for {symbol}...", "INFO")
    m1_data = load_data(symbol)
    print_status(f"Loaded {len(m1_data):,} candles", "SUCCESS")

    # Get IB configs for symbol
    if symbol not in IB_TIME_CONFIGS:
        print_status(f"No IB configs defined for {symbol}", "ERROR")
        return

    ib_configs = IB_TIME_CONFIGS[symbol]

    # Pre-compute cache
    cache = precompute_ib_cache(symbol, m1_data, ib_configs)

    # Calculate stats
    total_configs = len(cache)
    total_days = sum(len(days) for days in cache.values())

    print_status("=" * 50, "HEADER")
    print_status(f"IB Cache Complete for {symbol}", "HEADER")
    print_status(f"Configs: {total_configs}", "INFO")
    print_status(f"Total cached IB values: {total_days}", "INFO")

    # Save cache
    save_cache(cache, cache_path)

    # Report estimated speedup
    # IB calculation is ~30% of backtest time for a single combo
    # With 2.5M combos, this saves ~30% * 2.5M * 4sec / 90 workers = ~9 hours
    print_status("Estimated speedup: ~30% (IB lookup vs compute)", "SUCCESS")


if __name__ == "__main__":
    main()
