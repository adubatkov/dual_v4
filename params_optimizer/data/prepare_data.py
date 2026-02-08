"""
Data Preparation Module.

Converts raw CSV data to optimized Parquet format with trading hours filter.
"""

import glob
from pathlib import Path
from typing import Optional
from datetime import datetime

import pandas as pd
import pytz

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from params_optimizer.config import (
    DATA_PATHS_RAW,
    DATA_PATHS_OPTIMIZED,
    TRADING_HOURS,
    print_status,
)


def load_csv_files(data_folder: Path) -> pd.DataFrame:
    """
    Load all CSV files from folder and merge into single DataFrame.

    Args:
        data_folder: Path to folder containing CSV files

    Returns:
        Merged and sorted DataFrame with time, open, high, low, close columns
    """
    print_status(f"Loading CSV files from {data_folder}", "INFO")

    all_files = list(data_folder.glob("*.csv"))
    if not all_files:
        raise FileNotFoundError(f"No CSV files found in {data_folder}")

    print_status(f"Found {len(all_files)} CSV files", "INFO")

    df_list = []
    for i, file in enumerate(all_files):
        if (i + 1) % 10 == 0:
            print_status(f"Loading file {i + 1}/{len(all_files)}", "PROGRESS")

        try:
            tmp = pd.read_csv(file)

            # Parse time column
            tmp["time"] = pd.to_datetime(
                tmp["time"].astype(str).str.strip(),
                errors="coerce",
                utc=True
            )
            tmp = tmp.dropna(subset=["time"])

            # Convert OHLC to numeric
            for col in ("open", "high", "low", "close"):
                if col in tmp.columns:
                    tmp[col] = pd.to_numeric(tmp[col], errors="coerce")

            # Add volume column if not present (some CSV files don't have it)
            if "volume" not in tmp.columns and "tick_volume" not in tmp.columns:
                tmp["tick_volume"] = 1  # Default placeholder value

            # Select columns (tick_volume or volume)
            vol_col = "tick_volume" if "tick_volume" in tmp.columns else "volume"
            if vol_col in tmp.columns:
                df_list.append(tmp[["time", "open", "high", "low", "close", vol_col]].rename(columns={vol_col: "tick_volume"}))
            else:
                tmp["tick_volume"] = 1
                df_list.append(tmp[["time", "open", "high", "low", "close", "tick_volume"]])

        except Exception as e:
            print_status(f"Error loading {file.name}: {e}", "WARNING")
            continue

    if not df_list:
        raise ValueError("No valid data loaded from CSV files")

    # Merge all files
    df = pd.concat(df_list, ignore_index=True)
    df = df.sort_values("time").reset_index(drop=True)

    # Remove duplicates
    original_len = len(df)
    df = df.drop_duplicates(subset=["time"], keep="first")
    if len(df) < original_len:
        print_status(f"Removed {original_len - len(df)} duplicate timestamps", "INFO")

    print_status(f"Loaded {len(df):,} total candles", "SUCCESS")
    return df


def filter_trading_hours_ger40(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter GER40 data to trading hours only.

    Keeps only candles between 07:00-23:00 Europe/Berlin.

    Args:
        df: DataFrame with UTC timestamps

    Returns:
        Filtered DataFrame
    """
    print_status("Filtering GER40 trading hours (07:00-23:00 Berlin)", "INFO")

    config = TRADING_HOURS["GER40"]
    tz = pytz.timezone(config["timezone"])
    start_hour = config["start_hour"]
    end_hour = config["end_hour"]

    # Convert to local timezone
    df = df.copy()
    df["local_time"] = df["time"].dt.tz_convert(tz)
    df["hour"] = df["local_time"].dt.hour

    # Filter by hour
    mask = (df["hour"] >= start_hour) & (df["hour"] < end_hour)
    df_filtered = df[mask].drop(columns=["local_time", "hour"])

    print_status(f"Filtered from {len(df):,} to {len(df_filtered):,} candles ({len(df_filtered)/len(df)*100:.1f}%)", "SUCCESS")
    return df_filtered


def filter_trading_hours_xauusd(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter XAUUSD data to exclude weekends.

    XAUUSD trades 24/5, so we only exclude Saturday and Sunday.

    Args:
        df: DataFrame with UTC timestamps

    Returns:
        Filtered DataFrame
    """
    print_status("Filtering XAUUSD weekends", "INFO")

    df = df.copy()
    df["weekday"] = df["time"].dt.dayofweek  # 0=Monday, 6=Sunday

    # Exclude Saturday (5) and Sunday (6)
    mask = df["weekday"] < 5
    df_filtered = df[mask].drop(columns=["weekday"])

    print_status(f"Filtered from {len(df):,} to {len(df_filtered):,} candles ({len(df_filtered)/len(df)*100:.1f}%)", "SUCCESS")
    return df_filtered


def prepare_optimized_data(
    symbol: str,
    output_path: Optional[Path] = None,
    force: bool = False
) -> Path:
    """
    Prepare optimized Parquet data for symbol.

    1. Load all CSV files
    2. Filter to trading hours
    3. Save as Parquet

    Args:
        symbol: "GER40" or "XAUUSD"
        output_path: Custom output path (default from config)
        force: Overwrite existing file if True

    Returns:
        Path to created Parquet file
    """
    if symbol not in DATA_PATHS_RAW:
        raise ValueError(f"Unknown symbol: {symbol}. Must be GER40 or XAUUSD")

    # Determine paths
    raw_path = DATA_PATHS_RAW[symbol]
    if output_path is None:
        output_path = DATA_PATHS_OPTIMIZED[symbol]
    output_path = Path(output_path)

    # Check if already exists
    if output_path.exists() and not force:
        print_status(f"Optimized data already exists: {output_path}", "WARNING")
        print_status("Use --force to overwrite", "INFO")
        return output_path

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print_status(f"Preparing optimized data for {symbol}", "HEADER")
    print_status("=" * 60, "HEADER")

    # Load raw CSV data
    df = load_csv_files(raw_path)

    # Filter trading hours
    if symbol == "GER40":
        df = filter_trading_hours_ger40(df)
    elif symbol == "XAUUSD":
        df = filter_trading_hours_xauusd(df)

    # Save as Parquet
    print_status(f"Saving to {output_path}", "INFO")
    df.to_parquet(output_path, index=False, compression="snappy")

    # Report file size
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print_status(f"Created {output_path.name} ({file_size_mb:.1f} MB)", "SUCCESS")

    # Summary
    print_status("=" * 60, "HEADER")
    print_status(f"Data range: {df['time'].min()} to {df['time'].max()}", "INFO")
    print_status(f"Total candles: {len(df):,}", "INFO")
    print_status(f"Unique days: {df['time'].dt.date.nunique()}", "INFO")

    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare optimized data for parameter optimization")
    parser.add_argument("--symbol", choices=["GER40", "XAUUSD", "all"], default="all",
                        help="Symbol to prepare (default: all)")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing files")

    args = parser.parse_args()

    symbols = ["GER40", "XAUUSD"] if args.symbol == "all" else [args.symbol]

    for sym in symbols:
        try:
            prepare_optimized_data(sym, force=args.force)
            print()
        except Exception as e:
            print_status(f"Error preparing {sym}: {e}", "ERROR")
