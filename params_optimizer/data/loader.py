"""
Data Loader Module.

Provides optimized data loading with caching for multiprocessing workers.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple
from datetime import datetime

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from params_optimizer.config import (
    DATA_PATHS_OPTIMIZED,
    DATA_PATHS_RAW,
    print_status,
)


def load_data(
    symbol: str,
    data_path: Optional[Path] = None,
    use_optimized: bool = True
) -> pd.DataFrame:
    """
    Load M1 data for symbol.

    Args:
        symbol: "GER40" or "XAUUSD"
        data_path: Custom data path (optional)
        use_optimized: Try optimized Parquet first (default True)

    Returns:
        DataFrame with time, open, high, low, close columns
    """
    # Determine data path
    if data_path is not None:
        path = Path(data_path)
    elif use_optimized and symbol in DATA_PATHS_OPTIMIZED:
        path = DATA_PATHS_OPTIMIZED[symbol]
        if not path.exists():
            print_status(f"Optimized data not found at {path}", "WARNING")
            print_status("Run: python -m params_optimizer.data.prepare_data", "INFO")
            # Fall back to raw CSV
            path = DATA_PATHS_RAW.get(symbol)
    else:
        path = DATA_PATHS_RAW.get(symbol)

    if path is None or not path.exists():
        raise FileNotFoundError(f"Data not found for {symbol}")

    print_status(f"Loading data from {path}", "INFO")

    # Load based on file type
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.is_dir():
        # Load from CSV directory
        from .prepare_data import load_csv_files
        df = load_csv_files(path)
    else:
        raise ValueError(f"Unsupported data format: {path}")

    # Ensure time column is datetime with UTC
    if df["time"].dt.tz is None:
        df["time"] = df["time"].dt.tz_localize("UTC")

    # Ensure tick_volume column exists (required by backtest engine)
    if "tick_volume" not in df.columns and "volume" not in df.columns:
        df["tick_volume"] = 1  # Default placeholder
    elif "volume" in df.columns and "tick_volume" not in df.columns:
        df["tick_volume"] = df["volume"]

    print_status(f"Loaded {len(df):,} candles for {symbol}", "SUCCESS")
    return df


def get_data_info(symbol: str) -> Dict:
    """
    Get information about available data for symbol.

    Args:
        symbol: "GER40" or "XAUUSD"

    Returns:
        Dict with data info (exists, path, size, candles, date_range)
    """
    info = {
        "symbol": symbol,
        "optimized_exists": False,
        "raw_exists": False,
        "path": None,
        "size_mb": 0,
        "candles": 0,
        "date_range": None,
    }

    # Check optimized data
    opt_path = DATA_PATHS_OPTIMIZED.get(symbol)
    if opt_path and opt_path.exists():
        info["optimized_exists"] = True
        info["path"] = str(opt_path)
        info["size_mb"] = opt_path.stat().st_size / (1024 * 1024)

        # Load to get stats
        df = pd.read_parquet(opt_path)
        info["candles"] = len(df)
        info["date_range"] = f"{df['time'].min()} to {df['time'].max()}"

    # Check raw data
    raw_path = DATA_PATHS_RAW.get(symbol)
    if raw_path and raw_path.exists():
        info["raw_exists"] = True
        if not info["optimized_exists"]:
            info["path"] = str(raw_path)
            # Count CSV files
            csv_files = list(raw_path.glob("*.csv"))
            info["csv_files"] = len(csv_files)

    return info


# ================================
# WORKER CACHE FOR MULTIPROCESSING
# ================================
# Module-level cache (loaded once per worker process)
_CACHED_DATA: Optional[pd.DataFrame] = None
_CACHED_SYMBOL: Optional[str] = None


def init_worker_cache(symbol: str, data_path: Optional[Path] = None) -> None:
    """
    Initialize worker cache with data.

    Call this in multiprocessing pool initializer.

    Args:
        symbol: Symbol to load
        data_path: Custom data path (optional)
    """
    global _CACHED_DATA, _CACHED_SYMBOL

    _CACHED_DATA = load_data(symbol, data_path)
    _CACHED_SYMBOL = symbol

    print_status(f"Worker cache initialized: {len(_CACHED_DATA):,} candles for {symbol}", "SUCCESS")


def get_cached_data() -> Tuple[str, pd.DataFrame]:
    """
    Get cached data for current worker.

    Returns:
        Tuple of (symbol, DataFrame)

    Raises:
        RuntimeError if cache not initialized
    """
    if _CACHED_DATA is None or _CACHED_SYMBOL is None:
        raise RuntimeError("Worker cache not initialized. Call init_worker_cache first.")

    return _CACHED_SYMBOL, _CACHED_DATA


def clear_worker_cache() -> None:
    """Clear worker cache (free memory)."""
    global _CACHED_DATA, _CACHED_SYMBOL
    _CACHED_DATA = None
    _CACHED_SYMBOL = None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Check available data")
    parser.add_argument("--symbol", choices=["GER40", "XAUUSD", "all"], default="all")

    args = parser.parse_args()

    symbols = ["GER40", "XAUUSD"] if args.symbol == "all" else [args.symbol]

    print("Data Status")
    print("=" * 60)

    for sym in symbols:
        info = get_data_info(sym)
        print(f"\n{sym}:")
        print(f"  Optimized: {'Yes' if info['optimized_exists'] else 'No'}")
        print(f"  Raw: {'Yes' if info['raw_exists'] else 'No'}")
        if info['path']:
            print(f"  Path: {info['path']}")
        if info['size_mb']:
            print(f"  Size: {info['size_mb']:.1f} MB")
        if info['candles']:
            print(f"  Candles: {info['candles']:,}")
        if info['date_range']:
            print(f"  Range: {info['date_range']}")
