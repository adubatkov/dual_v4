#!/usr/bin/env python3
"""
Fractal Pre-compute Module for Parameter Optimization.

Pre-calculates H1/H4/M2 fractals and pre-sweep state for all M1 data.
Fractals depend ONLY on price data, not on strategy params, so they
can be computed once and reused across millions of param combinations.

Usage:
    python -m params_optimizer.data.fractal_precompute --symbol GER40
    python -m params_optimizer.data.fractal_precompute --symbol XAUUSD
"""

import argparse
import pickle
import time
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd

from params_optimizer.config import print_status
from params_optimizer.data.loader import load_data
from params_optimizer.engine.fast_backtest import FastBacktest


def precompute_fractal_cache(
    symbol: str,
    m1_data: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Pre-compute fractal cache for all M1 data.

    Performs:
    1. Resample M1 -> H1, H4, M2
    2. Detect fractals on each timeframe
    3. H4 dedup + sort by confirmed_time
    4. Pre-sweep: mark which fractals were swept by price action
    5. Sort M2 fractals by confirmed_time

    Args:
        symbol: Trading symbol
        m1_data: Full M1 candle data

    Returns:
        Dict with h1h4_fractals, m2_fractals, and metadata
    """
    from src.smc.detectors.fractal_detector import detect_fractals

    engine = FastBacktest(symbol, m1_data)

    # Resample
    t0 = time.perf_counter()
    h1_data = engine._resample_full("1h")
    h4_data = engine._resample_full("4h")
    m2_data = engine._resample_full("2min")
    t_resample = time.perf_counter() - t0
    print_status(f"Resample: {t_resample:.2f}s "
                 f"(H1={len(h1_data)}, H4={len(h4_data)}, M2={len(m2_data)} candles)", "INFO")

    # Detect fractals
    t0 = time.perf_counter()
    h1_fractals = detect_fractals(h1_data, symbol, "H1", candle_duration_hours=1.0)
    t_h1 = time.perf_counter() - t0

    t0 = time.perf_counter()
    h4_fractals = detect_fractals(h4_data, symbol, "H4", candle_duration_hours=4.0)
    t_h4 = time.perf_counter() - t0

    t0 = time.perf_counter()
    m2_fractals = detect_fractals(m2_data, symbol, "M2", candle_duration_hours=2 / 60)
    t_m2 = time.perf_counter() - t0

    print_status(f"Detect H1: {t_h1:.2f}s ({len(h1_fractals)} fractals)", "INFO")
    print_status(f"Detect H4: {t_h4:.2f}s ({len(h4_fractals)} fractals)", "INFO")
    print_status(f"Detect M2: {t_m2:.2f}s ({len(m2_fractals)} fractals)", "INFO")

    # H4 dedup + sort
    h4_keys = {(f.type, round(f.price, 2)) for f in h4_fractals}
    filtered_h1 = [f for f in h1_fractals if (f.type, round(f.price, 2)) not in h4_keys]
    h1h4_sorted = sorted(filtered_h1 + h4_fractals, key=lambda f: f.confirmed_time)
    m2_sorted = sorted(m2_fractals, key=lambda f: f.confirmed_time)

    print_status(f"H1H4 combined: {len(h1h4_sorted)} (H1={len(filtered_h1)}, H4={len(h4_fractals)})", "INFO")

    # Pre-sweep
    t0 = time.perf_counter()
    engine._presweep_fractals(h1h4_sorted)
    t_presweep = time.perf_counter() - t0

    swept_count = sum(1 for f in h1h4_sorted if f.swept)
    print_status(f"Pre-sweep: {t_presweep:.2f}s ({swept_count}/{len(h1h4_sorted)} swept)", "INFO")

    # FVG detection for FVG BE logic (reuses h1_data/h4_data already resampled)
    from src.smc.detectors.fvg_detector import detect_fvg

    t0 = time.perf_counter()
    h1_fvgs = detect_fvg(h1_data, symbol, "H1")
    h4_fvgs = detect_fvg(h4_data, symbol, "H4")
    htf_fvgs = sorted(h1_fvgs + h4_fvgs, key=lambda f: f.formation_time)
    engine._premitigate_fvgs(htf_fvgs)
    t_fvg = time.perf_counter() - t0

    mitigated_count = sum(1 for f in htf_fvgs if f.fill_time is not None)
    print_status(f"FVG detect+premitigate: {t_fvg:.2f}s "
                 f"(H1={len(h1_fvgs)}, H4={len(h4_fvgs)}, "
                 f"{mitigated_count}/{len(htf_fvgs)} mitigated)", "INFO")

    return {
        "h1h4_fractals": h1h4_sorted,
        "m2_fractals": m2_sorted,
        "htf_fvgs": htf_fvgs,
        "h1_count": len(h1_fractals),
        "h4_count": len(h4_fractals),
        "m2_count": len(m2_fractals),
        "h1_fvg_count": len(h1_fvgs),
        "h4_fvg_count": len(h4_fvgs),
        "symbol": symbol,
        "m1_candles": len(m1_data),
    }


def save_cache(cache: Dict, output_path: Path) -> None:
    """Save fractal cache to pickle file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print_status(f"Cache saved: {output_path} ({size_mb:.1f} MB)", "SUCCESS")


def load_cache(cache_path: Path) -> Optional[Dict]:
    """Load fractal cache from pickle file."""
    if not cache_path.exists():
        return None
    with open(cache_path, "rb") as f:
        return pickle.load(f)


def get_cache_path(symbol: str) -> Path:
    """Get path to fractal cache file for symbol."""
    return Path(__file__).parent / "optimized" / f"{symbol}_fractal_cache.pkl"


def main():
    parser = argparse.ArgumentParser(description="Pre-compute fractal cache for optimization")
    parser.add_argument("--symbol", choices=["GER40", "XAUUSD", "NAS100", "UK100"],
                        required=True, help="Symbol to pre-compute fractals for")
    parser.add_argument("--force", action="store_true",
                        help="Force re-computation even if cache exists")
    args = parser.parse_args()

    symbol = args.symbol
    cache_path = get_cache_path(symbol)

    if cache_path.exists() and not args.force:
        print_status(f"Cache already exists: {cache_path}", "WARNING")
        print_status("Use --force to re-compute", "INFO")
        return

    # Load M1 data
    print_status(f"Loading M1 data for {symbol}...", "INFO")
    t0 = time.perf_counter()
    m1_data = load_data(symbol)
    t_load = time.perf_counter() - t0
    print_status(f"Loaded {len(m1_data):,} candles in {t_load:.2f}s", "SUCCESS")

    # Pre-compute cache
    print_status("=" * 50, "HEADER")
    print_status(f"Pre-computing fractal cache for {symbol}", "HEADER")
    t0 = time.perf_counter()
    cache = precompute_fractal_cache(symbol, m1_data)
    t_total = time.perf_counter() - t0

    print_status("=" * 50, "HEADER")
    print_status(f"Total precompute time: {t_total:.2f}s", "SUCCESS")

    # Save cache
    save_cache(cache, cache_path)


if __name__ == "__main__":
    main()
