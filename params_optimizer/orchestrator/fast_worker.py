#!/usr/bin/env python3
"""
Fast Worker for Parameter Optimization.

Uses FastBacktest engine with module-level caching for maximum performance.
Data is loaded ONCE per worker process and reused for all parameter combinations.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import pandas as pd

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from params_optimizer.engine.fast_backtest import FastBacktest


# Module-level cache (loaded ONCE per worker process)
_WORKER_DATA: Dict[str, Any] = {
    "symbol": None,
    "m1_data": None,
    "backtest": None,
    "initialized": False,
}


def init_worker(symbol: str, data_path: str):
    """
    Initialize worker process with cached data.

    Called once when worker process starts.
    Loads data and creates FastBacktest instance.

    Args:
        symbol: Trading symbol (e.g., "GER40")
        data_path: Path to Parquet data file
    """
    global _WORKER_DATA

    # Load data
    _WORKER_DATA["symbol"] = symbol
    _WORKER_DATA["m1_data"] = pd.read_parquet(data_path)
    _WORKER_DATA["backtest"] = FastBacktest(symbol, _WORKER_DATA["m1_data"])
    _WORKER_DATA["initialized"] = True


def process_params(params_tuple: Tuple) -> Dict[str, Any]:
    """
    Process single parameter combination using cached data.

    Args:
        params_tuple: Tuple of parameter values in order:
            (ib_start, ib_end, ib_timezone, ib_wait_minutes, trade_window_minutes,
             rr_target, stop_mode, tsl_target, tsl_sl, min_sl_pct,
             rev_rb_enabled, rev_rb_pct, ib_buffer_pct, max_distance_pct)

    Returns:
        Dict with backtest results including the original params
    """
    if not _WORKER_DATA["initialized"]:
        raise RuntimeError("Worker not initialized. Call init_worker first.")

    # Convert tuple to dict
    params = tuple_to_dict(params_tuple)

    # Run backtest
    results = _WORKER_DATA["backtest"].run_with_params(params)

    # Add params to results for tracking
    results["params"] = params
    results["params_tuple"] = params_tuple

    return results


def tuple_to_dict(params_tuple: Tuple) -> Dict[str, Any]:
    """
    Convert parameter tuple to dict.

    Args:
        params_tuple: Tuple of parameter values

    Returns:
        Dict with named parameters
    """
    # Define parameter order (must match parameter_grid.py)
    param_names = [
        "ib_start",
        "ib_end",
        "ib_timezone",
        "ib_wait_minutes",
        "trade_window_minutes",
        "rr_target",
        "stop_mode",
        "tsl_target",
        "tsl_sl",
        "min_sl_pct",
        "rev_rb_enabled",
        "rev_rb_pct",
        "ib_buffer_pct",
        "max_distance_pct",
    ]

    if len(params_tuple) != len(param_names):
        raise ValueError(
            f"Expected {len(param_names)} params, got {len(params_tuple)}"
        )

    return dict(zip(param_names, params_tuple))


def dict_to_tuple(params: Dict[str, Any]) -> Tuple:
    """
    Convert parameter dict to tuple.

    Args:
        params: Dict with named parameters

    Returns:
        Tuple of parameter values
    """
    param_names = [
        "ib_start",
        "ib_end",
        "ib_timezone",
        "ib_wait_minutes",
        "trade_window_minutes",
        "rr_target",
        "stop_mode",
        "tsl_target",
        "tsl_sl",
        "min_sl_pct",
        "rev_rb_enabled",
        "rev_rb_pct",
        "ib_buffer_pct",
        "max_distance_pct",
    ]

    return tuple(params[name] for name in param_names)


def process_params_dict(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process parameter dict directly (for single-threaded testing).

    Args:
        params: Dict with named parameters

    Returns:
        Dict with backtest results
    """
    if not _WORKER_DATA["initialized"]:
        raise RuntimeError("Worker not initialized. Call init_worker first.")

    results = _WORKER_DATA["backtest"].run_with_params(params)
    results["params"] = params

    return results


# Standalone test
if __name__ == "__main__":
    from params_optimizer.config import print_status, DATA_PATHS

    print_status("Testing fast_worker...", "INFO")

    # Initialize worker
    symbol = "GER40"
    data_path = DATA_PATHS[symbol]
    init_worker(symbol, str(data_path))

    print_status(f"Worker initialized with {len(_WORKER_DATA['m1_data']):,} candles", "SUCCESS")

    # Test with V2 params
    test_params = (
        "08:00",      # ib_start
        "09:00",      # ib_end
        "Europe/Berlin",  # ib_timezone
        15,           # ib_wait_minutes
        60,           # trade_window_minutes
        1.0,          # rr_target
        "ib_start",   # stop_mode
        0.0,          # tsl_target
        0.5,          # tsl_sl
        0.001,        # min_sl_pct
        False,        # rev_rb_enabled
        0.5,          # rev_rb_pct
        0.01,         # ib_buffer_pct
        0.5,          # max_distance_pct
    )

    import time
    start = time.perf_counter()
    results = process_params(test_params)
    elapsed = time.perf_counter() - start

    print_status(f"Time: {elapsed:.2f}s", "INFO")
    print_status(f"Total R: {results['total_r']:.2f}", "INFO")
    print_status(f"Trades: {results['total_trades']}", "INFO")
    print_status(f"Win Rate: {results['winrate']:.1f}%", "INFO")
