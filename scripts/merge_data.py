"""
Merge M1 data from multiple sources into single parquet files per symbol.

Combines existing optimized parquet (training period) with control period CSVs.
Deduplicates by timestamp, sorts chronologically, saves to data/optimized/.

Usage:
    python scripts/merge_data.py
"""

import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from backtest.data_processor.data_ingestor import DataIngestor

DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR = DATA_DIR / "optimized"

# Sources per symbol: (parquet_file, [csv_directories])
SOURCES = {
    "GER40": {
        "parquet": OUTPUT_DIR / "GER40_m1.parquet",
        "csv_dirs": [
            DATA_DIR / "ger40+pepperstone_0411-2001",
        ],
        "output": OUTPUT_DIR / "GER40_m1.parquet",
    },
    "XAUUSD": {
        "parquet": OUTPUT_DIR / "XAUUSD_m1.parquet",
        "csv_dirs": [
            DATA_DIR / "xauusd_oanda_0411-2001",
        ],
        "output": OUTPUT_DIR / "XAUUSD_m1.parquet",
    },
}


def merge_symbol(symbol: str, config: dict) -> None:
    """Merge all data sources for a symbol into one parquet file."""
    frames = []

    # Load existing parquet
    parquet_path = config["parquet"]
    if parquet_path.exists():
        df_existing = pd.read_parquet(parquet_path)
        print(f"  [{symbol}] Existing parquet: {len(df_existing)} rows, "
              f"{df_existing['time'].min()} to {df_existing['time'].max()}")
        frames.append(df_existing)
    else:
        print(f"  [{symbol}] No existing parquet at {parquet_path}")

    # Load CSV directories
    for csv_dir in config["csv_dirs"]:
        if not csv_dir.exists():
            print(f"  [{symbol}] CSV dir not found: {csv_dir}")
            continue

        ingestor = DataIngestor(csv_dir)
        df_csv = ingestor.load_all()

        # Ensure tick_volume column matches existing data
        if "tick_volume" not in df_csv.columns and "volume" in df_csv.columns:
            df_csv["tick_volume"] = df_csv["volume"]
        if "volume" in df_csv.columns and "tick_volume" in df_csv.columns:
            df_csv = df_csv.drop(columns=["volume"])

        print(f"  [{symbol}] CSV dir {csv_dir.name}: {len(df_csv)} rows, "
              f"{df_csv['time'].min()} to {df_csv['time'].max()}")
        frames.append(df_csv)

    if not frames:
        print(f"  [{symbol}] No data found, skipping")
        return

    # Concat all frames
    merged = pd.concat(frames, ignore_index=True)
    before_dedup = len(merged)

    # Ensure consistent columns
    # Existing parquet has: time, open, high, low, close, tick_volume
    # CSVs may have: time, open, high, low, close, volume
    if "tick_volume" not in merged.columns:
        merged["tick_volume"] = 0
    keep_cols = ["time", "open", "high", "low", "close", "tick_volume"]
    merged = merged[[c for c in keep_cols if c in merged.columns]]

    # Deduplicate by time (keep last = prefer newer data)
    merged = merged.drop_duplicates(subset=["time"], keep="last")
    after_dedup = len(merged)

    # Sort
    merged = merged.sort_values("time").reset_index(drop=True)

    print(f"  [{symbol}] Merged: {before_dedup} -> {after_dedup} rows "
          f"({before_dedup - after_dedup} duplicates removed)")
    print(f"  [{symbol}] Range: {merged['time'].min()} to {merged['time'].max()}")

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(config["output"], index=False)
    size_mb = config["output"].stat().st_size / (1024 * 1024)
    print(f"  [{symbol}] Saved: {config['output']} ({size_mb:.1f} MB)")


def main():
    print("Merging M1 data into unified parquet files...")
    print()

    for symbol, config in SOURCES.items():
        print(f"Processing {symbol}:")
        merge_symbol(symbol, config)
        print()

    print("Done.")


if __name__ == "__main__":
    main()
