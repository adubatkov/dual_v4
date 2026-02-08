"""
Parallel Slow Backtest Runner

Runs multiple backtest groups in parallel using multiprocessing.
Designed for execution on VMs with 24 cores each.

Usage:
    python run_parallel_backtest.py --symbol GER40 --workers 20
    python run_parallel_backtest.py --groups-file groups_vm11.json --workers 20
"""

import sys
import json
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
from multiprocessing import Pool, cpu_count
import time

import pytz
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def worker_init():
    """Initialize worker process (suppress logging)."""
    logging.getLogger().setLevel(logging.WARNING)


def run_backtest_wrapper(args: tuple) -> Dict[str, Any]:
    """
    Wrapper function for multiprocessing.

    Args:
        args: Tuple of (group, start_date, end_date, output_dir, risk_amount, skip_charts)

    Returns:
        Result dict from run_single_group_backtest
    """
    group, start_date, end_date, output_dir, risk_amount, skip_charts = args

    try:
        from backtest.backtest_single_group import run_single_group_backtest

        result = run_single_group_backtest(
            group=group,
            start_date=start_date,
            end_date=end_date,
            output_dir=output_dir,
            risk_amount=risk_amount,
            skip_charts=skip_charts,
        )
        result["status"] = "success"
        return result

    except Exception as e:
        import traceback
        return {
            "group_id": group.get("id", "unknown"),
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


def run_parallel_backtests(
    groups: List[Dict],
    start_date: datetime,
    end_date: datetime,
    output_dir: Path,
    num_workers: int = 20,
    risk_amount: float = 1000.0,
    skip_charts: bool = False,
) -> List[Dict]:
    """
    Run backtests for multiple groups in parallel.

    Args:
        groups: List of group dicts from generate_backtest_groups.py
        start_date: Start date (UTC)
        end_date: End date (UTC)
        output_dir: Base output directory
        num_workers: Number of parallel workers
        risk_amount: Fixed risk per trade
        skip_charts: If True, skip equity chart generation

    Returns:
        List of result dicts
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting parallel backtest")
    logger.info(f"  Groups: {len(groups)}")
    logger.info(f"  Workers: {num_workers}")
    logger.info(f"  Period: {start_date.date()} to {end_date.date()}")
    logger.info(f"  Output: {output_dir}")

    # Prepare arguments for each group
    args_list = [
        (group, start_date, end_date, output_dir, risk_amount, skip_charts)
        for group in groups
    ]

    # Run in parallel
    start_time = time.time()
    results = []

    with Pool(processes=num_workers, initializer=worker_init) as pool:
        for i, result in enumerate(pool.imap_unordered(run_backtest_wrapper, args_list)):
            results.append(result)

            # Progress update
            elapsed = time.time() - start_time
            completed = i + 1
            remaining = len(groups) - completed
            avg_time = elapsed / completed if completed > 0 else 0
            eta = avg_time * remaining

            status = result.get("status", "unknown")
            group_id = result.get("group_id", "unknown")

            if status == "success":
                logger.info(
                    f"[{completed}/{len(groups)}] {group_id}: "
                    f"R={result.get('total_r', 0):.2f}, "
                    f"Trades={result.get('total_trades', 0)}, "
                    f"ETA: {eta/60:.1f}min"
                )
            else:
                logger.error(f"[{completed}/{len(groups)}] {group_id}: FAILED - {result.get('error', 'unknown')}")

    total_time = time.time() - start_time
    logger.info(f"Completed in {total_time/60:.1f} minutes")

    return results


def save_results_summary(results: List[Dict], output_dir: Path, symbol: str = None):
    """Save results to CSV and JSON summary."""
    output_dir = Path(output_dir)

    # Filter successful results
    successful = [r for r in results if r.get("status") == "success"]
    failed = [r for r in results if r.get("status") != "success"]

    logger.info(f"Results: {len(successful)} successful, {len(failed)} failed")

    # Save to CSV
    if successful:
        df = pd.DataFrame(successful)
        df = df.sort_values("total_r", ascending=False)

        csv_path = output_dir / f"results_summary{'_' + symbol if symbol else ''}.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved CSV: {csv_path}")

        # Print top results
        logger.info("\nTop 10 Results by Total R:")
        for i, row in df.head(10).iterrows():
            logger.info(
                f"  {row['group_id']}: R={row['total_r']:.2f}, "
                f"Sharpe={row['sharpe']:.2f}, "
                f"Trades={row['total_trades']}, "
                f"R_diff={row['r_difference']:.2f}"
            )

    # Save full results to JSON
    json_path = output_dir / f"results_full{'_' + symbol if symbol else ''}.json"
    with open(json_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "total_groups": len(results),
            "successful": len(successful),
            "failed": len(failed),
            "results": results,
        }, f, indent=2)
    logger.info(f"Saved JSON: {json_path}")

    # Save failed groups for retry
    if failed:
        failed_path = output_dir / f"failed_groups{'_' + symbol if symbol else ''}.json"
        with open(failed_path, "w") as f:
            json.dump(failed, f, indent=2)
        logger.warning(f"Saved failed groups for retry: {failed_path}")


def main():
    parser = argparse.ArgumentParser(description="Run parallel slow backtests")
    parser.add_argument("--symbol", type=str, help="Symbol to backtest (GER40 or XAUUSD)")
    parser.add_argument("--groups-file", type=str, help="Path to groups JSON file")
    parser.add_argument("--workers", type=int, default=20, help="Number of parallel workers")
    parser.add_argument("--start-date", type=str, default="2023-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default="2025-10-31", help="End date (YYYY-MM-DD)")
    parser.add_argument("--risk", type=float, default=1000.0, help="Risk amount per trade")
    parser.add_argument("--skip-charts", action="store_true", help="Skip equity chart generation")
    parser.add_argument("--output-dir", type=str, help="Output directory")

    args = parser.parse_args()

    # Load groups
    if args.groups_file:
        groups_path = Path(args.groups_file)
    elif args.symbol:
        groups_path = Path(__file__).parent.parent / "analyze" / f"backtest_groups_{args.symbol}.json"
    else:
        parser.error("Either --symbol or --groups-file is required")

    if not groups_path.exists():
        logger.error(f"Groups file not found: {groups_path}")
        sys.exit(1)

    with open(groups_path) as f:
        groups = json.load(f)

    logger.info(f"Loaded {len(groups)} groups from {groups_path}")

    # Filter by symbol if specified
    if args.symbol:
        groups = [g for g in groups if g.get("symbol") == args.symbol]
        logger.info(f"Filtered to {len(groups)} groups for {args.symbol}")

    if not groups:
        logger.error("No groups to process")
        sys.exit(1)

    # Parse dates
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d").replace(tzinfo=pytz.UTC)
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d").replace(tzinfo=pytz.UTC)

    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        symbol_suffix = f"_{args.symbol}" if args.symbol else ""
        output_dir = Path(__file__).parent / "output" / f"parallel_{timestamp}{symbol_suffix}"

    # Run backtests
    results = run_parallel_backtests(
        groups=groups,
        start_date=start_date,
        end_date=end_date,
        output_dir=output_dir,
        num_workers=min(args.workers, cpu_count()),
        risk_amount=args.risk,
        skip_charts=args.skip_charts,
    )

    # Save summary
    save_results_summary(results, output_dir, args.symbol)

    logger.info(f"\nAll results saved to: {output_dir}")


if __name__ == "__main__":
    main()
