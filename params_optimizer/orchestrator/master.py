"""
Master Orchestrator for Parameter Optimization.

Coordinates parallel parameter search using multiprocessing Pool.
Enhanced with:
- Pre-computed IB cache for ~30% speedup
- Variation-level analytics (OCAE, TCWE, Reverse, REV_RB)
- Memory monitoring
"""

import gc
import os
import sys
import time
import psutil
from datetime import datetime, timedelta
from pathlib import Path
from multiprocessing import Pool, cpu_count
from typing import Optional, List, Set, Tuple, Dict, Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from params_optimizer.config import (
    OptimizerConfig,
    DATA_PATHS_OPTIMIZED,
    print_status,
    print_progress_bar,
)
from params_optimizer.engine.parameter_grid import ParameterGrid
from params_optimizer.engine.metrics_calculator import MetricsCalculator
from params_optimizer.orchestrator.checkpoint import CheckpointManager
from params_optimizer.orchestrator.worker import init_worker_data, process_combination
from params_optimizer.analytics.variation_aggregator import VariationAggregator
from params_optimizer.data.ib_precompute import load_cache, get_cache_path


class OptimizerMaster:
    """
    Master orchestrator for parallel parameter optimization.

    Coordinates:
    - Parameter grid generation
    - Worker pool management
    - Progress tracking
    - Checkpoint/resume
    - Results aggregation
    - Variation-level analytics
    """

    def __init__(self, config: OptimizerConfig):
        """
        Initialize OptimizerMaster.

        Args:
            config: Optimizer configuration
        """
        self.config = config
        self.symbol = config.symbol

        # Determine data path
        self.data_path = config.data_path
        if self.data_path is None:
            self.data_path = DATA_PATHS_OPTIMIZED.get(self.symbol)

        if self.data_path is None or not self.data_path.exists():
            raise FileNotFoundError(
                f"Data not found for {self.symbol}. "
                f"Run: python -m params_optimizer.data.prepare_data --symbol {self.symbol}"
            )

        # Initialize components
        self.grid = ParameterGrid(self.symbol, mode=config.grid_mode)
        self.checkpoint = CheckpointManager(
            config.output_dir,
            config.checkpoint_file,
        )
        self.metrics = MetricsCalculator(
            weight_total_r=config.weight_total_r,
            weight_sharpe=config.weight_sharpe,
            weight_winrate=config.weight_winrate,
        )

        # Variation aggregator for on-the-fly analytics
        self.variation_agg = VariationAggregator()

        # Load pre-computed IB cache if available
        self.ib_cache = self._load_ib_cache()

        # State
        self._results: List[dict] = []
        self._completed: Set[Tuple] = set()
        self._start_time: Optional[datetime] = None
        self._combinations_processed: int = 0

    def _load_ib_cache(self) -> Optional[Dict[Any, Any]]:
        """Load pre-computed IB cache if available."""
        cache_path = get_cache_path(self.symbol)
        if cache_path.exists():
            print_status(f"Loading IB cache from {cache_path}...", "INFO")
            cache = load_cache(cache_path)
            if cache:
                configs = len(cache)
                total_days = sum(len(days) for days in cache.values())
                print_status(f"IB cache loaded: {configs} configs, {total_days:,} days", "SUCCESS")
                return cache
        else:
            print_status(
                f"IB cache not found. Run: python -m params_optimizer.data.ib_precompute --symbol {self.symbol}",
                "WARNING"
            )
        return None

    def _get_memory_usage(self) -> str:
        """Get current memory usage."""
        try:
            process = psutil.Process()
            mem = process.memory_info().rss / (1024 * 1024 * 1024)  # GB
            return f"{mem:.2f} GB"
        except Exception:
            return "N/A"

    def run(self, resume: bool = True) -> pd.DataFrame:
        """
        Run parameter optimization.

        Args:
            resume: Resume from checkpoint if available

        Returns:
            DataFrame with ranked results
        """
        print_status("=" * 60, "HEADER")
        print_status(f"Parameter Optimization for {self.symbol}", "HEADER")
        print_status("=" * 60, "HEADER")

        self._start_time = datetime.now()

        # Generate all combinations
        print_status("Generating parameter combinations...", "INFO")
        all_combinations = self.grid.generate_all()
        total_combinations = len(all_combinations)
        print_status(f"Total combinations: {total_combinations:,}", "SUCCESS")

        # Load checkpoint if resuming
        if resume and self.checkpoint.exists():
            print_status("Loading checkpoint...", "INFO")
            self._completed, self._results, var_agg = self.checkpoint.load()
            if var_agg is not None:
                self.variation_agg = var_agg
                print_status(f"Variation aggregator restored: {self.variation_agg.total_results:,} results", "SUCCESS")
            print_status(f"Resuming from {len(self._completed):,} completed", "SUCCESS")
        else:
            self._completed = set()
            self._results = []
            self.variation_agg.clear()

        # Set checkpoint metadata
        self.checkpoint.set_metadata(self.symbol, total_combinations)

        # Shuffle ALL combinations first (seed=42 for reproducibility)
        # This ensures the same subset is selected regardless of resume state
        all_shuffled = self.grid.shuffle(all_combinations, seed=42)

        # Limit to target subset BEFORE filtering completed
        # This ensures resumed runs continue the SAME target set
        if self.config.max_combos and len(all_shuffled) > self.config.max_combos:
            target_combos = all_shuffled[:self.config.max_combos]
            print_status(f"Target: {self.config.max_combos:,} combinations (--max-combos)", "INFO")
        else:
            target_combos = all_shuffled

        # Filter out completed combinations from the target set
        remaining = self.grid.filter_completed(target_combos, self._completed)
        print_status(f"Remaining combinations: {len(remaining):,} (of {len(target_combos):,} target)", "INFO")

        if len(remaining) == 0:
            print_status("All combinations already processed!", "SUCCESS")
            return self._finalize_results()

        # Convert to tuples for multiprocessing
        remaining_tuples = [self.grid.to_tuple(p) for p in remaining]

        # Estimate time (realistic: ~42s/combo single-process on Xeon E5-2650)
        time_est = self.grid.estimate_time(
            len(remaining),
            self.config.num_workers,
            seconds_per_combo=42.0
        )
        print_status(f"Estimated time: ~{time_est} with {self.config.num_workers} workers", "INFO")

        # Run optimization
        print_status("Starting optimization...", "HEADER")
        self._run_parallel(remaining_tuples)

        # Finalize
        return self._finalize_results()

    def _run_parallel(self, combinations_tuples: List[Tuple]) -> None:
        """
        Run parallel optimization using multiprocessing Pool.

        Args:
            combinations_tuples: List of parameter tuples to process
        """
        total = len(combinations_tuples)
        processed = 0
        checkpoint_count = 0

        # Create worker pool with IB cache and fractal cache
        print_status(f"Creating pool with {self.config.num_workers} workers...", "INFO")
        if self.ib_cache:
            print_status("Workers will use pre-computed IB cache", "INFO")
        if self.config.news_filter_enabled:
            print_status("Workers will use news filter (5ers compliance)", "INFO")

        # Get cache paths (workers load themselves to avoid pickle/copy overhead)
        ib_cache_path = get_cache_path(self.symbol) if self.ib_cache else None

        # Get fractal cache path (~8x speedup: ~145s -> ~17s per combo)
        from params_optimizer.data.fractal_precompute import get_cache_path as get_fractal_cache_path
        fractal_cache_path = get_fractal_cache_path(self.symbol)
        if fractal_cache_path.exists():
            print_status("Workers will use pre-computed fractal cache", "INFO")
        else:
            fractal_cache_path = None
            print_status("Fractal cache not found - fractals computed per combo (slow)", "WARNING")

        with Pool(
            processes=self.config.num_workers,
            initializer=init_worker_data,
            initargs=(
                self.symbol,
                self.data_path,
                self.config.initial_balance,
                self.config.risk_pct,
                self.config.max_margin_pct,
                ib_cache_path,  # Pass path, not cache object (avoids pickle/copy to each worker)
                self.config.news_filter_enabled,  # Pass news filter flag
                self.config.news_before_minutes,  # Minutes before news to block
                self.config.news_after_minutes,   # Minutes after news to block
                fractal_cache_path,  # Fractal cache for ~8x speedup
            )
        ) as pool:
            print_status("Worker pool ready", "SUCCESS")

            # Use imap_unordered for best performance
            results_iter = pool.imap_unordered(
                process_combination,
                combinations_tuples,
                chunksize=self.config.combinations_per_chunk,
            )

            # Process results as they complete
            for result in results_iter:
                processed += 1
                self._combinations_processed += 1

                # Extract params tuple from result
                if result.get("params"):
                    params_tuple = self.grid.to_tuple(result["params"])
                    self._completed.add(params_tuple)
                    self._results.append(result)

                    # Add to variation aggregator for on-the-fly analytics
                    self.variation_agg.add_result(params_tuple, result)

                # Progress update
                if processed % 10 == 0 or processed == total:
                    self._print_progress(processed, total)

                # Checkpoint (including variation aggregator) - ASYNC for performance
                checkpoint_count += 1
                if checkpoint_count >= self.config.checkpoint_interval:
                    self.checkpoint.save_async(
                        self._completed,
                        self._results,
                        self.variation_agg,
                    )
                    checkpoint_count = 0
                    gc.collect()

        # Final checkpoint - SYNC to ensure completion
        self.checkpoint.wait_for_pending()  # Wait for any async saves
        self.checkpoint.save(self._completed, self._results, self.variation_agg)
        self.checkpoint.shutdown()  # Clean up executor

    def _print_progress(self, current: int, total: int) -> None:
        """Print progress update with memory info."""
        elapsed = datetime.now() - self._start_time
        rate = current / elapsed.total_seconds() if elapsed.total_seconds() > 0 else 0

        remaining = total - current
        eta_seconds = remaining / rate if rate > 0 else 0
        eta = timedelta(seconds=int(eta_seconds))

        pct = current / total * 100
        mem = self._get_memory_usage()

        print_progress_bar(
            current, total,
            prefix=f"Progress",
            suffix=f"| {current:,}/{total:,} | {rate:.1f}/s | ETA: {eta} | Mem: {mem}"
        )

    def _finalize_results(self) -> pd.DataFrame:
        """
        Finalize and save results including variation analytics.

        Returns:
            Ranked results DataFrame
        """
        print_status("\nFinalizing results...", "HEADER")

        # Convert to DataFrame
        df = self.checkpoint.get_results_df()

        if len(df) == 0:
            print_status("No results to process", "WARNING")
            return pd.DataFrame()

        # Calculate rankings
        ranked_df = self.metrics.rank_results(df, min_trades=10)

        # Save outputs
        output_dir = self.config.output_dir

        # Full results
        full_results_path = output_dir / f"{self.symbol}_full_results.parquet"
        df.to_parquet(full_results_path, index=False)
        print_status(f"Full results saved: {full_results_path}", "SUCCESS")

        # Top 100 CSV (human readable)
        if len(ranked_df) > 0:
            top_100_path = output_dir / f"{self.symbol}_top_100.csv"
            top_100 = ranked_df.head(100)

            # Flatten params for CSV
            top_100_flat = self._flatten_params_for_csv(top_100)
            top_100_flat.to_csv(top_100_path, index=False)
            print_status(f"Top 100 saved: {top_100_path}", "SUCCESS")

            # Best params JSON
            best_params = self.metrics.get_best_params(df, min_trades=10)
            if best_params:
                best_path = output_dir / f"{self.symbol}_best_params.json"
                import json
                with open(best_path, "w") as f:
                    json.dump(best_params, f, indent=2)
                print_status(f"Best params saved: {best_path}", "SUCCESS")

            # Summary report
            report = self.metrics.generate_summary_report(df, top_n=20, min_trades=10)
            report_path = output_dir / f"{self.symbol}_summary.txt"
            with open(report_path, "w") as f:
                f.write(report)
            print_status(f"Summary report saved: {report_path}", "SUCCESS")

            # Print summary to console
            print("\n" + report)

        # Save variation analytics (like anal.py output)
        self._save_variation_analytics(output_dir)

        # Clear checkpoint (move to final)
        self.checkpoint.clear()

        # Final statistics
        elapsed = datetime.now() - self._start_time
        print_status("=" * 60, "HEADER")
        print_status("OPTIMIZATION COMPLETE", "HEADER")
        print_status("=" * 60, "HEADER")
        print_status(f"Total time: {elapsed}", "INFO")
        print_status(f"Combinations processed: {len(self._completed):,}", "INFO")
        print_status(f"Memory usage: {self._get_memory_usage()}", "INFO")
        print_status(f"Results saved to: {output_dir}", "SUCCESS")

        return ranked_df

    def _save_variation_analytics(self, output_dir: Path) -> None:
        """Save variation-level analytics (like anal.py output)."""
        from params_optimizer.analytics.variation_aggregator import generate_variation_report

        if self.variation_agg.total_results == 0:
            print_status("No variation data to save", "WARNING")
            return

        print_status("Saving variation analytics...", "INFO")

        # Save variation aggregator state (for later analysis)
        agg_path = output_dir / f"{self.symbol}_variation_agg.pkl"
        self.variation_agg.save_state(agg_path)
        print_status(f"Variation aggregator saved: {agg_path}", "SUCCESS")

        # Variation summary CSV
        summary = self.variation_agg.get_variation_summary()
        summary_path = output_dir / f"{self.symbol}_variations_summary.csv"
        summary.to_csv(summary_path, index=False)
        print_status(f"Variation summary saved: {summary_path}", "SUCCESS")

        # Best params per variation JSON
        best_by_var = self.variation_agg.get_best_params_per_variation()
        param_keys = ParameterGrid.PARAM_KEYS
        best_dict = {}
        for var, params_tuple in best_by_var.items():
            best_dict[var] = dict(zip(param_keys, params_tuple))

        import json
        best_var_path = output_dir / f"{self.symbol}_best_by_variation.json"
        with open(best_var_path, "w") as f:
            json.dump(best_dict, f, indent=2)
        print_status(f"Best params by variation saved: {best_var_path}", "SUCCESS")

        # Human-readable variation report (like anal.py output)
        report = generate_variation_report(self.variation_agg)
        report_path = output_dir / f"{self.symbol}_variation_report.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        print_status(f"Variation report saved: {report_path}", "SUCCESS")

        # Detailed top 500 per variation
        for var in self.variation_agg.variations:
            details = self.variation_agg.get_variation_details(var, top_n=500)
            if len(details) > 0:
                # Flatten params_tuple
                if "params_tuple" in details.columns:
                    for i, key in enumerate(param_keys):
                        details[f"param_{key}"] = details["params_tuple"].apply(
                            lambda t, idx=i: t[idx] if len(t) > idx else ""
                        )
                    details = details.drop(columns=["params_tuple"])

                var_path = output_dir / f"{self.symbol}_variation_{var}_top500.csv"
                details.to_csv(var_path, index=False)
                print_status(f"Variation {var} details saved: {var_path}", "SUCCESS")

        # Print variation summary to console
        stats = self.variation_agg.get_stats()
        print_status(f"\nVariation Stats: {stats['total_results']:,} results, "
                    f"{stats['results_with_trades']:,} with trades", "INFO")
        print("\n" + summary.to_string())

    def _flatten_params_for_csv(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Flatten params dict into separate columns for CSV export.

        Args:
            df: DataFrame with 'params' column containing dicts

        Returns:
            DataFrame with flattened params columns
        """
        result = df.copy()

        if "params" not in result.columns:
            return result

        # Extract param keys
        param_keys = ParameterGrid.PARAM_KEYS

        for key in param_keys:
            result[f"param_{key}"] = result["params"].apply(
                lambda p: p.get(key, "") if isinstance(p, dict) else ""
            )

        # Drop original params column
        result = result.drop(columns=["params"])

        return result


def calculate_safe_workers(symbol: str, available_ram_gb: float = None) -> int:
    """
    Calculate safe number of workers based on available RAM.

    Per-worker memory from benchmarks:
        GER40: ~500MB, XAUUSD: ~620MB, NAS100: ~615MB, UK100: ~530MB
    """
    per_worker_mb = {"GER40": 500, "XAUUSD": 620, "NAS100": 615, "UK100": 530}

    if available_ram_gb is None:
        available_ram_gb = psutil.virtual_memory().total / (1024 ** 3)

    reserve_gb = 3  # OS + master process
    usable_mb = (available_ram_gb - reserve_gb) * 1024
    worker_mb = per_worker_mb.get(symbol, 600)

    max_by_ram = int(usable_mb / worker_mb)
    max_by_cpu = max(1, cpu_count() - 4)

    safe = min(max_by_ram, max_by_cpu)
    return max(1, safe)


def run_optimization(
    symbol: str,
    num_workers: Optional[int] = None,
    resume: bool = True,
    output_dir: Optional[Path] = None,
    grid_mode: str = "standard",
    news_filter_enabled: bool = True,
    news_before_minutes: int = 2,
    news_after_minutes: int = 2,
) -> pd.DataFrame:
    """
    Convenience function to run optimization.

    Args:
        symbol: Trading symbol ("GER40" or "XAUUSD")
        num_workers: Number of parallel workers (default: CPU count - 6)
        resume: Resume from checkpoint
        output_dir: Output directory
        news_filter_enabled: Enable news filter for 5ers compliance (default: True)
        news_before_minutes: Minutes before news to block trades (default: 2)
        news_after_minutes: Minutes after news to block trades (default: 2)

    Returns:
        Ranked results DataFrame
    """
    if num_workers is None:
        num_workers = calculate_safe_workers(symbol)

    config = OptimizerConfig(
        symbol=symbol,
        num_workers=num_workers,
        grid_mode=grid_mode,
        output_dir=output_dir or Path(__file__).parent.parent / "output",
        news_filter_enabled=news_filter_enabled,
        news_before_minutes=news_before_minutes,
        news_after_minutes=news_after_minutes,
    )

    master = OptimizerMaster(config)
    return master.run(resume=resume)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run parameter optimization")
    parser.add_argument("--symbol", choices=["GER40", "XAUUSD", "NAS100", "UK100"], required=True)
    parser.add_argument("--workers", type=int, default=None,
                       help="Number of workers (default: CPU count - 6)")
    parser.add_argument("--no-resume", action="store_true",
                       help="Start fresh, ignore checkpoint")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--no-news-filter", action="store_true",
                       help="Disable news filter (enabled by default for 5ers compliance)")
    parser.add_argument("--news-before-minutes", type=int, default=2,
                       help="Minutes before news to block trades (default: 2)")
    parser.add_argument("--news-after-minutes", type=int, default=2,
                       help="Minutes after news to block trades (default: 2)")

    args = parser.parse_args()

    results = run_optimization(
        symbol=args.symbol,
        num_workers=args.workers,
        resume=not args.no_resume,
        output_dir=args.output_dir,
        news_filter_enabled=not args.no_news_filter,
        news_before_minutes=args.news_before_minutes,
        news_after_minutes=args.news_after_minutes,
    )

    print(f"\nOptimization complete. {len(results)} ranked results.")
