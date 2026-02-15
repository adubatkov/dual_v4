#!/usr/bin/env python3
"""
Benchmark script for params_optimizer.

Measures per-combo execution time, memory usage, and throughput
for a small batch of combinations to estimate VM requirements.

Usage:
    python temp/benchmark_optimizer.py --symbol GER40 --combos 100 --workers 4
    python temp/benchmark_optimizer.py --symbol GER40 --combos 50 --workers 1  # single-threaded
"""

import argparse
import os
import sys
import time
import random
import pickle
from pathlib import Path
from multiprocessing import Pool, cpu_count
from typing import List, Dict, Any, Tuple

import psutil
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from params_optimizer.config import (
    DATA_PATHS_OPTIMIZED,
    print_status,
)
from params_optimizer.engine.parameter_grid import ParameterGrid
from params_optimizer.data.fractal_precompute import get_cache_path as get_fractal_cache_path
from params_optimizer.data.ib_precompute import get_cache_path as get_ib_cache_path


def measure_memory_mb() -> float:
    """Get current process memory in MB."""
    return psutil.Process().memory_info().rss / (1024 * 1024)


def benchmark_single_threaded(
    symbol: str,
    combinations: List[Dict[str, Any]],
    fractal_cache_path: Path = None,
) -> Dict[str, Any]:
    """
    Run combinations single-threaded, measuring per-combo time.

    Returns dict with timing stats.
    """
    from params_optimizer.engine.fast_backtest import FastBacktest

    # Load data
    data_path = DATA_PATHS_OPTIMIZED[symbol]
    print_status(f"Loading M1 data from {data_path}...", "INFO")
    t0 = time.perf_counter()
    m1_data = pd.read_parquet(data_path)
    if m1_data["time"].dt.tz is None:
        m1_data["time"] = m1_data["time"].dt.tz_localize("UTC")
    t_load = time.perf_counter() - t0
    print_status(f"Loaded {len(m1_data):,} candles in {t_load:.1f}s", "SUCCESS")

    # Create engine
    engine = FastBacktest(symbol, m1_data)
    mem_after_init = measure_memory_mb()
    print_status(f"Engine initialized. Memory: {mem_after_init:.0f} MB", "INFO")

    # Load fractal cache if available
    fractal_cache = None
    if fractal_cache_path and fractal_cache_path.exists():
        print_status(f"Loading fractal cache from {fractal_cache_path}...", "INFO")
        with open(fractal_cache_path, "rb") as f:
            fractal_cache = pickle.load(f)
        print_status("Fractal cache loaded", "SUCCESS")

    # Run combinations
    times = []
    results_summary = []
    mem_peak = mem_after_init

    print_status(f"Running {len(combinations)} combinations single-threaded...", "HEADER")

    for i, params in enumerate(combinations):
        t0 = time.perf_counter()
        result = engine.run_with_params(params, fractal_cache=fractal_cache)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)

        results_summary.append({
            "combo_idx": i,
            "time_s": round(elapsed, 2),
            "total_r": result.get("total_r", 0),
            "total_trades": result.get("total_trades", 0),
            "winrate": result.get("winrate", 0),
        })

        curr_mem = measure_memory_mb()
        mem_peak = max(mem_peak, curr_mem)

        if (i + 1) % 10 == 0 or i == 0:
            avg_so_far = sum(times) / len(times)
            print_status(
                f"  [{i+1}/{len(combinations)}] "
                f"last={elapsed:.2f}s avg={avg_so_far:.2f}s "
                f"mem={curr_mem:.0f}MB "
                f"trades={result.get('total_trades', 0)} "
                f"R={result.get('total_r', 0):.1f}",
                "PROGRESS"
            )

    # Stats
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    median_time = sorted(times)[len(times) // 2]
    total_time = sum(times)

    return {
        "mode": "single_threaded",
        "symbol": symbol,
        "combos": len(combinations),
        "total_time_s": round(total_time, 1),
        "avg_time_s": round(avg_time, 2),
        "median_time_s": round(median_time, 2),
        "min_time_s": round(min_time, 2),
        "max_time_s": round(max_time, 2),
        "mem_init_mb": round(mem_after_init, 0),
        "mem_peak_mb": round(mem_peak, 0),
        "fractal_cache": fractal_cache is not None,
        "candles": len(m1_data),
        "results": results_summary,
    }


# Module-level worker state for multiprocessing
_BENCH_ENGINE = None
_BENCH_FRACTAL_CACHE = None


def _init_bench_worker(symbol: str, data_path: str, fractal_cache_path: str = None):
    """Initialize worker for benchmark."""
    global _BENCH_ENGINE, _BENCH_FRACTAL_CACHE

    from params_optimizer.engine.fast_backtest import FastBacktest

    m1_data = pd.read_parquet(data_path)
    if m1_data["time"].dt.tz is None:
        m1_data["time"] = m1_data["time"].dt.tz_localize("UTC")

    _BENCH_ENGINE = FastBacktest(symbol, m1_data)

    if fractal_cache_path and Path(fractal_cache_path).exists():
        with open(fractal_cache_path, "rb") as f:
            _BENCH_FRACTAL_CACHE = pickle.load(f)

    pid = os.getpid()
    mem = psutil.Process(pid).memory_info().rss / (1024 * 1024)
    print(f"[Worker {pid}] Initialized: {len(m1_data):,} candles, {mem:.0f} MB")


def _bench_process(params: Dict[str, Any]) -> Dict[str, Any]:
    """Process single combo in worker."""
    t0 = time.perf_counter()
    result = _BENCH_ENGINE.run_with_params(params, fractal_cache=_BENCH_FRACTAL_CACHE)
    elapsed = time.perf_counter() - t0
    result["bench_time_s"] = elapsed
    result["params"] = params
    return result


def benchmark_parallel(
    symbol: str,
    combinations: List[Dict[str, Any]],
    num_workers: int,
    fractal_cache_path: Path = None,
) -> Dict[str, Any]:
    """
    Run combinations in parallel with Pool, measuring throughput.
    """
    data_path = str(DATA_PATHS_OPTIMIZED[symbol])
    fc_path = str(fractal_cache_path) if fractal_cache_path and fractal_cache_path.exists() else None

    print_status(f"Starting parallel benchmark: {len(combinations)} combos, {num_workers} workers", "HEADER")

    t_start = time.perf_counter()

    with Pool(
        processes=num_workers,
        initializer=_init_bench_worker,
        initargs=(symbol, data_path, fc_path),
    ) as pool:
        results = list(pool.imap_unordered(_bench_process, combinations, chunksize=5))

    t_total = time.perf_counter() - t_start

    # Extract per-combo times
    times = [r["bench_time_s"] for r in results]
    avg_time = sum(times) / len(times)
    throughput = len(combinations) / t_total

    print_status(f"Parallel benchmark complete: {t_total:.1f}s total", "SUCCESS")

    return {
        "mode": "parallel",
        "symbol": symbol,
        "workers": num_workers,
        "combos": len(combinations),
        "wall_time_s": round(t_total, 1),
        "avg_combo_time_s": round(avg_time, 2),
        "throughput_per_s": round(throughput, 2),
        "fractal_cache": fc_path is not None,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark params_optimizer")
    parser.add_argument("--symbol", choices=["GER40", "XAUUSD", "NAS100", "UK100"],
                        default="GER40", help="Symbol to benchmark")
    parser.add_argument("--combos", type=int, default=100,
                        help="Number of random combinations to test (default: 100)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of workers for parallel test (default: CPU-2)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--skip-single", action="store_true",
                        help="Skip single-threaded benchmark")
    parser.add_argument("--skip-parallel", action="store_true",
                        help="Skip parallel benchmark")
    args = parser.parse_args()

    symbol = args.symbol
    num_combos = args.combos
    num_workers = args.workers or max(1, cpu_count() - 2)

    # Check data exists
    data_path = DATA_PATHS_OPTIMIZED.get(symbol)
    if not data_path or not data_path.exists():
        print_status(f"No optimized data for {symbol}. Run prepare_data first.", "ERROR")
        sys.exit(1)

    # Check fractal cache
    fractal_cache_path = get_fractal_cache_path(symbol)
    if fractal_cache_path.exists():
        print_status(f"Fractal cache found: {fractal_cache_path}", "SUCCESS")
    else:
        print_status(f"No fractal cache for {symbol}. Benchmark will be slower.", "WARNING")
        fractal_cache_path = None

    # Generate random combinations
    print_status(f"Generating {num_combos} random combinations for {symbol}...", "INFO")
    grid = ParameterGrid(symbol)
    all_combos = grid.generate_all()
    print_status(f"Total grid: {len(all_combos):,} combinations", "INFO")

    random.seed(args.seed)
    sample = random.sample(all_combos, min(num_combos, len(all_combos)))
    print_status(f"Sampled {len(sample)} combinations", "SUCCESS")

    # Print system info
    print_status("=" * 60, "HEADER")
    print_status("SYSTEM INFO", "HEADER")
    print_status(f"CPU cores: {cpu_count()}", "INFO")
    print_status(f"RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB", "INFO")
    print_status(f"Symbol: {symbol}", "INFO")
    print_status(f"Data: {data_path} ({data_path.stat().st_size / (1024**2):.1f} MB)", "INFO")
    print_status("=" * 60, "HEADER")

    # Single-threaded benchmark
    if not args.skip_single:
        print_status("\n--- SINGLE-THREADED BENCHMARK ---", "HEADER")
        st_results = benchmark_single_threaded(symbol, sample[:20], fractal_cache_path)

        print_status("\n--- SINGLE-THREADED RESULTS ---", "HEADER")
        print(f"  Combos tested: {st_results['combos']}")
        print(f"  Total time:    {st_results['total_time_s']}s")
        print(f"  Avg per combo: {st_results['avg_time_s']}s")
        print(f"  Median:        {st_results['median_time_s']}s")
        print(f"  Min/Max:       {st_results['min_time_s']}s / {st_results['max_time_s']}s")
        print(f"  Memory init:   {st_results['mem_init_mb']:.0f} MB")
        print(f"  Memory peak:   {st_results['mem_peak_mb']:.0f} MB")
        print(f"  Fractal cache: {st_results['fractal_cache']}")

    # Parallel benchmark
    if not args.skip_parallel:
        print_status(f"\n--- PARALLEL BENCHMARK ({num_workers} workers) ---", "HEADER")
        par_results = benchmark_parallel(symbol, sample, num_workers, fractal_cache_path)

        print_status("\n--- PARALLEL RESULTS ---", "HEADER")
        print(f"  Combos tested:  {par_results['combos']}")
        print(f"  Workers:        {par_results['workers']}")
        print(f"  Wall time:      {par_results['wall_time_s']}s")
        print(f"  Avg per combo:  {par_results['avg_combo_time_s']}s")
        print(f"  Throughput:     {par_results['throughput_per_s']} combos/s")
        print(f"  Fractal cache:  {par_results['fractal_cache']}")

    # Projections
    print_status("\n--- VM PROJECTIONS ---", "HEADER")
    total_combos = len(all_combos)

    if not args.skip_single:
        avg_s = st_results["avg_time_s"]
        mem_per_worker = st_results["mem_peak_mb"]
    elif not args.skip_parallel:
        avg_s = par_results["avg_combo_time_s"]
        mem_per_worker = 500  # estimate
    else:
        return

    for vm_workers in [10, 15, 20]:
        time_hours = (total_combos / vm_workers * avg_s) / 3600
        mem_total_gb = (vm_workers * mem_per_worker) / 1024
        print(f"  {vm_workers} workers: {time_hours:.1f}h for {total_combos:,} combos, ~{mem_total_gb:.1f} GB RAM")

    print(f"\n  4 VMs x 20 workers: {total_combos:,} combos per symbol")
    print(f"  1 symbol per VM: {(total_combos / 20 * avg_s) / 3600:.1f}h")
    print(f"  Total across 4 symbols: {4 * total_combos:,} combos")


if __name__ == "__main__":
    main()
