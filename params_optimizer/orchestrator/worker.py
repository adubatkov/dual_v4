"""
Worker Process for Parameter Optimization.

Handles individual parameter combination testing with cached data.
Designed for multiprocessing Pool workers.

Uses FastBacktestOptimized engine for speed (~4s per combo vs ~300s with BacktestWrapper).
Optimizations: vectorized timezone conversion, pre-computed EQ mask.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

import pandas as pd

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# ================================
# MODULE-LEVEL CACHE
# ================================
# Loaded once per worker process via initializer
_WORKER_DATA: Optional[pd.DataFrame] = None
_WORKER_SYMBOL: Optional[str] = None
_WORKER_BACKTEST = None
_WORKER_IB_CACHE = None
_WORKER_NEWS_FILTER = None


def init_worker_data(
    symbol: str,
    data_path: Path,
    initial_balance: float = 100000.0,
    risk_pct: float = 1.0,
    max_margin_pct: float = 40.0,
    ib_cache_path: Optional[Path] = None,
    news_filter_enabled: bool = True,
    news_before_minutes: int = 2,
    news_after_minutes: int = 2,
) -> None:
    """
    Initialize worker with cached data and FastBacktest.

    Called once per worker process via Pool initializer.

    Args:
        symbol: Trading symbol
        data_path: Path to Parquet data file
        initial_balance: Starting balance (not used in FastBacktest)
        risk_pct: Risk percentage per trade (not used in FastBacktest)
        max_margin_pct: Maximum margin usage percentage (not used in FastBacktest)
        ib_cache_path: Path to pre-computed IB cache (loaded by worker to avoid memory copy)
        news_filter_enabled: Enable news filter for 5ers compliance (default: True)
        news_before_minutes: Minutes before news to block trades (default: 2)
        news_after_minutes: Minutes after news to block trades (default: 2)
    """
    global _WORKER_DATA, _WORKER_SYMBOL, _WORKER_BACKTEST, _WORKER_IB_CACHE, _WORKER_NEWS_FILTER

    from params_optimizer.engine.fast_backtest_optimized import FastBacktestOptimized
    from params_optimizer.data.ib_precompute import load_cache

    # Load data
    if data_path.suffix == ".parquet":
        _WORKER_DATA = pd.read_parquet(data_path)
    else:
        raise ValueError(f"Expected Parquet file, got: {data_path}")

    # Ensure time column is datetime with UTC
    if _WORKER_DATA["time"].dt.tz is None:
        _WORKER_DATA["time"] = _WORKER_DATA["time"].dt.tz_localize("UTC")

    _WORKER_SYMBOL = symbol

    # Load IB cache from file (each worker loads its own copy - uses OS page cache)
    _WORKER_IB_CACHE = None
    if ib_cache_path and Path(ib_cache_path).exists():
        _WORKER_IB_CACHE = load_cache(Path(ib_cache_path))

    # Initialize news filter if enabled
    _WORKER_NEWS_FILTER = None
    if news_filter_enabled:
        try:
            from src.news_filter import NewsFilter
            _WORKER_NEWS_FILTER = NewsFilter(
                symbol=symbol,
                before_minutes=news_before_minutes,
                after_minutes=news_after_minutes,
            )
        except Exception as e:
            print(f"[Worker] Warning: Failed to initialize news filter: {e}")

    # Create FastBacktestOptimized instance (~1.6x faster than original FastBacktest)
    # Optimizations: vectorized TZ conversion, pre-computed EQ mask
    _WORKER_BACKTEST = FastBacktestOptimized(
        symbol=symbol,
        m1_data=_WORKER_DATA,
        ib_cache=_WORKER_IB_CACHE,
        news_filter=_WORKER_NEWS_FILTER,
    )

    # Log worker initialization (minimal)
    import os
    pid = os.getpid()
    cache_status = "with IB cache" if _WORKER_IB_CACHE else "without IB cache"
    news_status = f", news filter ({_WORKER_NEWS_FILTER.event_count} events)" if _WORKER_NEWS_FILTER else ""
    print(f"[Worker {pid}] Initialized FastBacktestOptimized with {len(_WORKER_DATA):,} candles for {symbol} ({cache_status}{news_status})")


def process_combination(params_tuple: Tuple) -> Dict[str, Any]:
    """
    Process single parameter combination.

    Uses cached FastBacktest for ~4s per combo (vs ~300s with BacktestWrapper).

    Args:
        params_tuple: Tuple of parameter values in fixed order

    Returns:
        Dict with results:
            - params: Original parameters dict
            - All metrics from FastBacktest (total_r, winrate, etc.)
            - error: Error message if failed
    """
    global _WORKER_BACKTEST

    if _WORKER_BACKTEST is None:
        return {
            "params": None,
            "total_r": 0.0,
            "total_trades": 0,
            "winrate": 0.0,
            "sharpe_ratio": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "avg_trade_r": 0.0,
            "error": "Worker not initialized",
        }

    from params_optimizer.engine.parameter_grid import ParameterGrid

    # Convert tuple back to dict
    grid = ParameterGrid(_WORKER_SYMBOL)
    params = grid.from_tuple(params_tuple)

    try:
        # Run fast backtest
        results = _WORKER_BACKTEST.run_with_params(params)

        # Add params to results
        results["params"] = params
        return results

    except Exception as e:
        return {
            "params": params,
            "total_r": 0.0,
            "total_trades": 0,
            "winrate": 0.0,
            "sharpe_ratio": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "avg_trade_r": 0.0,
            "error": str(e),
        }


def process_combination_dict(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process single parameter combination (dict version).

    Alternative to process_combination that takes dict directly.

    Args:
        params: Parameter dict

    Returns:
        Dict with results
    """
    global _WORKER_BACKTEST

    if _WORKER_BACKTEST is None:
        return {
            "params": params,
            "total_r": 0.0,
            "total_trades": 0,
            "winrate": 0.0,
            "sharpe_ratio": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "avg_trade_r": 0.0,
            "error": "Worker not initialized",
        }

    try:
        results = _WORKER_BACKTEST.run_with_params(params)
        results["params"] = params
        return results

    except Exception as e:
        return {
            "params": params,
            "total_r": 0.0,
            "total_trades": 0,
            "winrate": 0.0,
            "sharpe_ratio": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "avg_trade_r": 0.0,
            "error": str(e),
        }


def get_worker_info() -> Dict[str, Any]:
    """
    Get information about current worker state.

    Returns:
        Dict with worker info
    """
    import os

    return {
        "pid": os.getpid(),
        "symbol": _WORKER_SYMBOL,
        "data_loaded": _WORKER_DATA is not None,
        "candles": len(_WORKER_DATA) if _WORKER_DATA is not None else 0,
        "backtest_ready": _WORKER_BACKTEST is not None,
        "ib_cache_loaded": _WORKER_IB_CACHE is not None,
        "ib_cache_configs": len(_WORKER_IB_CACHE) if _WORKER_IB_CACHE else 0,
    }


if __name__ == "__main__":
    # Test worker initialization
    import argparse

    parser = argparse.ArgumentParser(description="Test worker initialization")
    parser.add_argument("--symbol", choices=["GER40", "XAUUSD", "NAS100", "UK100"], required=True)
    parser.add_argument("--data-path", type=Path, required=True)

    args = parser.parse_args()

    print(f"Initializing worker for {args.symbol}...")
    init_worker_data(args.symbol, args.data_path)

    info = get_worker_info()
    print(f"Worker info: {info}")

    # Test with sample params
    from params_optimizer.engine.parameter_grid import ParameterGrid

    grid = ParameterGrid(args.symbol)
    all_combos = grid.generate_all()

    print(f"\nTesting first combination...")
    first_params = all_combos[0]
    params_tuple = grid.to_tuple(first_params)

    result = process_combination(params_tuple)
    print(f"Result: {result}")
