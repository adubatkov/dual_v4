#!/usr/bin/env python3
"""
Parameter Optimizer CLI for IB Trading Bot.

Usage:
    # Prepare data (one-time, converts CSV to optimized Parquet)
    python -m params_optimizer.run_optimizer --prepare-data --symbol GER40

    # Run optimization
    python -m params_optimizer.run_optimizer --symbol GER40 --workers 90

    # Resume from checkpoint
    python -m params_optimizer.run_optimizer --symbol GER40 --resume

    # Show parameter grid info
    python -m params_optimizer.run_optimizer --info --symbol GER40
"""

import argparse
import sys
from datetime import datetime
from multiprocessing import cpu_count
from pathlib import Path
from typing import Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from params_optimizer.config import (
    OptimizerConfig,
    DEFAULT_OUTPUT_DIR,
    DATA_PATHS_OPTIMIZED,
    DATA_PATHS_RAW,
    print_status,
)


def prepare_data(symbol: str, force: bool = False) -> None:
    """
    Prepare optimized data for symbol.

    Converts raw CSV to Parquet with trading hours filter.
    """
    from params_optimizer.data.prepare_data import prepare_optimized_data

    print_status(f"Preparing data for {symbol}...", "HEADER")
    output_path = prepare_optimized_data(symbol, force=force)
    print_status(f"Data ready: {output_path}", "SUCCESS")


def show_info(symbol: str) -> None:
    """Show parameter grid and data information."""
    from params_optimizer.data.loader import get_data_info
    from params_optimizer.engine.parameter_grid import print_grid_info

    # Data info
    print_status("DATA STATUS", "HEADER")
    print_status("=" * 50, "HEADER")

    info = get_data_info(symbol)
    print(f"\n{symbol}:")
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

    # Grid info
    print("\n")
    print_grid_info(symbol)


def run_optimization(
    symbol: str,
    workers: int,
    resume: bool,
    output_dir: Path,
    run_dir: str = None,
    news_filter_enabled: bool = True,
    news_before_minutes: int = 2,
    news_after_minutes: int = 2,
) -> Path:
    """Run parameter optimization.

    Args:
        symbol: Trading symbol
        workers: Number of parallel workers
        resume: Resume from checkpoint
        output_dir: Base output directory
        run_dir: Specific run directory name (for resume)
        news_filter_enabled: Enable news filter (default: True for 5ers compliance)
        news_before_minutes: Minutes before news to block trades
        news_after_minutes: Minutes after news to block trades

    Returns:
        Path to output directory used for this run
    """
    from params_optimizer.orchestrator.master import OptimizerMaster
    from params_optimizer.engine.parameter_grid import ParameterGrid

    # Check data exists
    data_path = DATA_PATHS_OPTIMIZED.get(symbol)
    if data_path is None or not data_path.exists():
        print_status(f"Optimized data not found for {symbol}", "ERROR")
        print_status(f"Run: python -m params_optimizer.run_optimizer --prepare-data --symbol {symbol}", "INFO")
        sys.exit(1)

    # Calculate number of combinations
    grid = ParameterGrid(symbol)
    num_combos = len(grid.generate_all())

    # Create run-specific subdirectory
    if run_dir:
        # Resume from specific run directory
        run_output_dir = output_dir / run_dir
    else:
        # New run - create timestamped directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir_name = f"{symbol}_{num_combos}combos_{timestamp}"
        run_output_dir = output_dir / run_dir_name

    run_output_dir.mkdir(parents=True, exist_ok=True)
    print_status(f"Output directory: {run_output_dir}", "INFO")

    # Create config
    config = OptimizerConfig(
        symbol=symbol,
        num_workers=workers,
        output_dir=run_output_dir,
        news_filter_enabled=news_filter_enabled,
        news_before_minutes=news_before_minutes,
        news_after_minutes=news_after_minutes,
    )

    # Run
    master = OptimizerMaster(config)
    results = master.run(resume=resume)

    if len(results) > 0:
        print_status(f"\nOptimization complete. {len(results)} ranked results.", "SUCCESS")
    else:
        print_status("No results generated.", "WARNING")

    return run_output_dir


def find_latest_run_dir(base_dir: Path, symbol: str) -> Optional[Path]:
    """Find the latest run directory for a symbol."""
    pattern = f"{symbol}_*combos_*"
    matching_dirs = sorted(
        [d for d in base_dir.glob(pattern) if d.is_dir()],
        key=lambda d: d.stat().st_mtime,
        reverse=True
    )
    return matching_dirs[0] if matching_dirs else None


def generate_excel(symbol: str, output_dir: Path, top_n: int = 5000, run_dir: str = None) -> None:
    """
    Generate Excel report from existing parquet results.

    Args:
        symbol: Trading symbol
        output_dir: Base output directory
        top_n: Top N results per variation sheet
        run_dir: Specific run directory name (optional)
    """
    # Determine run directory
    if run_dir:
        run_output_dir = output_dir / run_dir
    else:
        # Find latest run directory for this symbol
        run_output_dir = find_latest_run_dir(output_dir, symbol)
        if run_output_dir is None:
            print_status(f"No run directories found for {symbol} in {output_dir}", "ERROR")
            print_status("Run optimization first or specify --run-dir", "INFO")
            sys.exit(1)
        print_status(f"Using latest run: {run_output_dir.name}", "INFO")

    # Find results file
    results_path = run_output_dir / "results_final.parquet"
    if not results_path.exists():
        # Try partial results
        results_path = run_output_dir / "results_partial.parquet"

    if not results_path.exists():
        # Try full results
        results_path = run_output_dir / f"{symbol}_full_results.parquet"

    if not results_path.exists():
        print_status(f"No results found in {run_output_dir}", "ERROR")
        print_status("Run optimization first or check the run directory.", "INFO")
        sys.exit(1)

    # Check openpyxl
    try:
        import openpyxl  # noqa: F401
    except ImportError:
        print_status("openpyxl is required for Excel generation", "ERROR")
        print_status("Install with: pip install openpyxl", "INFO")
        sys.exit(1)

    from params_optimizer.reports.excel_generator import generate_excel_report

    print_status(f"Generating Excel report for {symbol}...", "HEADER")
    print_status(f"Source: {results_path}", "INFO")

    output_path = generate_excel_report(
        results_path=results_path,
        output_dir=run_output_dir,
        symbol=symbol,
        top_n=top_n,
    )

    print_status(f"Excel report generated: {output_path}", "SUCCESS")


def check_requirements() -> bool:
    """Check that all requirements are installed."""
    required = ["pandas", "numpy", "pytz"]

    missing = []
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        print_status(f"Missing packages: {', '.join(missing)}", "ERROR")
        print_status("Install with: pip install " + " ".join(missing), "INFO")
        return False

    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Parameter Optimizer for IB Trading Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Prepare data (one-time)
  python -m params_optimizer.run_optimizer --prepare-data --symbol GER40

  # Run optimization (creates new run directory)
  python -m params_optimizer.run_optimizer --symbol GER40 --workers 4

  # Resume from specific run directory
  python -m params_optimizer.run_optimizer --symbol GER40 --run-dir GER40_144combos_20240107_220100

  # Show info
  python -m params_optimizer.run_optimizer --info --symbol GER40

  # Generate Excel from latest run
  python -m params_optimizer.run_optimizer --generate-excel --symbol GER40

  # Generate Excel from specific run
  python -m params_optimizer.run_optimizer --generate-excel --symbol GER40 --run-dir GER40_144combos_20240107_220100
        """
    )

    # Required
    parser.add_argument(
        "--symbol",
        choices=["GER40", "XAUUSD", "NAS100", "UK100"],
        required=True,
        help="Trading symbol to optimize"
    )

    # Actions
    parser.add_argument(
        "--prepare-data",
        action="store_true",
        help="Prepare optimized data (CSV to Parquet)"
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Show parameter grid and data info"
    )
    parser.add_argument(
        "--generate-excel",
        action="store_true",
        help="Generate Excel report from existing parquet results"
    )

    # Options
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=f"Number of parallel workers (default: CPU count - 6 = {max(1, cpu_count() - 6)})"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint (default: True)"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh, ignore checkpoint"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force overwrite existing data/results"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Base output directory (default: {DEFAULT_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Specific run directory name (for resume or generate-excel)"
    )

    # News filter options (5ers compliance)
    parser.add_argument(
        "--no-news-filter",
        action="store_true",
        help="Disable news filter (enabled by default for 5ers compliance)"
    )
    parser.add_argument(
        "--news-before-minutes",
        type=int,
        default=2,
        help="Minutes before high-impact news to block trades (default: 2)"
    )
    parser.add_argument(
        "--news-after-minutes",
        type=int,
        default=2,
        help="Minutes after high-impact news to block trades (default: 2)"
    )

    args = parser.parse_args()

    # Check requirements
    if not check_requirements():
        sys.exit(1)

    # Handle actions
    if args.prepare_data:
        prepare_data(args.symbol, force=args.force)
        return

    if args.info:
        show_info(args.symbol)
        return

    if args.generate_excel:
        generate_excel(args.symbol, args.output_dir, run_dir=args.run_dir)
        return

    # Default action: run optimization
    workers = args.workers
    if workers is None:
        workers = max(1, cpu_count() - 6)

    # If run_dir specified, always resume
    resume = args.run_dir is not None or not args.no_resume

    # News filter settings (enabled by default)
    news_filter_enabled = not args.no_news_filter

    print_status(f"Starting optimization for {args.symbol}", "HEADER")
    print_status(f"Workers: {workers}", "INFO")
    print_status(f"Resume: {resume}", "INFO")
    print_status(f"Base output: {args.output_dir}", "INFO")
    print_status(f"News filter: {'enabled' if news_filter_enabled else 'disabled'} ({args.news_before_minutes}min before, {args.news_after_minutes}min after)", "INFO")

    run_optimization(
        symbol=args.symbol,
        workers=workers,
        resume=resume,
        output_dir=args.output_dir,
        run_dir=args.run_dir,
        news_filter_enabled=news_filter_enabled,
        news_before_minutes=args.news_before_minutes,
        news_after_minutes=args.news_after_minutes,
    )


if __name__ == "__main__":
    main()
