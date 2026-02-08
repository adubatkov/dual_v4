#!/usr/bin/env python3
"""
Report Generator for Parameter Optimization.

Generates final reports from optimization results:
1. Top N combinations (CSV)
2. Variation-level analytics (CSV)
3. Best params per variation (JSON)
4. Human-readable summary (TXT)

Usage:
    python -m params_optimizer.analytics.generate_reports --symbol GER40
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd

from params_optimizer.config import print_status
from params_optimizer.analytics.variation_aggregator import (
    VariationAggregator,
    generate_variation_report,
)
from params_optimizer.engine.parameter_grid import ParameterGrid


def load_full_results(symbol: str, output_dir: Path) -> Optional[pd.DataFrame]:
    """Load full results from Parquet file."""
    results_path = output_dir / f"{symbol}_full_results.parquet"
    if not results_path.exists():
        print_status(f"Results file not found: {results_path}", "ERROR")
        return None

    df = pd.read_parquet(results_path)
    print_status(f"Loaded {len(df):,} results from {results_path}", "SUCCESS")
    return df


def load_variation_aggregator(symbol: str, output_dir: Path) -> Optional[VariationAggregator]:
    """Load variation aggregator state."""
    agg_path = output_dir / f"{symbol}_variation_agg.pkl"
    if not agg_path.exists():
        print_status(f"Variation aggregator not found: {agg_path}", "WARNING")
        return None

    agg = VariationAggregator()
    if agg.load_state(agg_path):
        print_status(f"Loaded variation aggregator from {agg_path}", "SUCCESS")
        return agg
    return None


def generate_top_n_csv(
    df: pd.DataFrame,
    symbol: str,
    output_dir: Path,
    top_n: int = 1000,
    min_trades: int = 10
) -> None:
    """Generate CSV with top N results."""
    # Filter by minimum trades
    df_filtered = df[df["total_trades"] >= min_trades].copy()

    if len(df_filtered) == 0:
        print_status(f"No results with >={min_trades} trades", "WARNING")
        return

    # Sort by total_r descending
    df_sorted = df_filtered.sort_values("total_r", ascending=False)

    # Take top N
    df_top = df_sorted.head(top_n)

    # Flatten params for CSV if needed
    if "params" in df_top.columns:
        param_keys = ParameterGrid.PARAM_KEYS
        for key in param_keys:
            df_top[f"param_{key}"] = df_top["params"].apply(
                lambda p: p.get(key, "") if isinstance(p, dict) else ""
            )
        df_top = df_top.drop(columns=["params"])

    # Save
    output_path = output_dir / f"{symbol}_top_{top_n}.csv"
    df_top.to_csv(output_path, index=False)
    print_status(f"Saved top {len(df_top)} results to {output_path}", "SUCCESS")


def generate_variation_csv(
    aggregator: VariationAggregator,
    symbol: str,
    output_dir: Path
) -> None:
    """Generate CSV with variation-level analytics."""
    # Summary
    summary = aggregator.get_variation_summary()
    summary_path = output_dir / f"{symbol}_variations_summary.csv"
    summary.to_csv(summary_path, index=False)
    print_status(f"Saved variation summary to {summary_path}", "SUCCESS")

    # Details per variation
    for var in aggregator.variations:
        details = aggregator.get_variation_details(var, top_n=500)
        if len(details) > 0:
            # Flatten params_tuple
            if "params_tuple" in details.columns:
                param_keys = ParameterGrid.PARAM_KEYS
                for i, key in enumerate(param_keys):
                    details[f"param_{key}"] = details["params_tuple"].apply(
                        lambda t: t[i] if len(t) > i else ""
                    )
                details = details.drop(columns=["params_tuple"])

            var_path = output_dir / f"{symbol}_variation_{var}_top500.csv"
            details.to_csv(var_path, index=False)
            print_status(f"Saved {var} details ({len(details)} rows) to {var_path}", "SUCCESS")


def generate_best_params_json(
    aggregator: VariationAggregator,
    symbol: str,
    output_dir: Path
) -> None:
    """Generate JSON with best params per variation."""
    best_params = aggregator.get_best_params_per_variation()
    param_keys = ParameterGrid.PARAM_KEYS

    # Convert tuples to dicts
    best_dict = {}
    for var, params_tuple in best_params.items():
        best_dict[var] = dict(zip(param_keys, params_tuple))

    output_path = output_dir / f"{symbol}_best_by_variation.json"
    with open(output_path, "w") as f:
        json.dump(best_dict, f, indent=2)
    print_status(f"Saved best params per variation to {output_path}", "SUCCESS")


def generate_summary_txt(
    df: pd.DataFrame,
    aggregator: Optional[VariationAggregator],
    symbol: str,
    output_dir: Path,
    min_trades: int = 10
) -> None:
    """Generate human-readable summary report."""
    lines = [
        "=" * 70,
        f"PARAMETER OPTIMIZATION REPORT - {symbol}",
        "=" * 70,
        "",
    ]

    # Overall stats
    valid = df[df["total_trades"] >= min_trades]
    lines.append("OVERALL STATISTICS")
    lines.append("-" * 40)
    lines.append(f"Total combinations tested: {len(df):,}")
    lines.append(f"Valid combinations (>={min_trades} trades): {len(valid):,}")
    lines.append("")

    if len(valid) > 0:
        lines.append(f"Total R range: {valid['total_r'].min():.2f} to {valid['total_r'].max():.2f}")
        lines.append(f"Winrate range: {valid['winrate'].min():.1f}% to {valid['winrate'].max():.1f}%")
        lines.append(f"Sharpe range: {valid['sharpe_ratio'].min():.2f} to {valid['sharpe_ratio'].max():.2f}")
        lines.append("")

        # Top 10 results
        lines.append("TOP 10 COMBINATIONS")
        lines.append("-" * 40)
        top10 = valid.nlargest(10, "total_r")

        for rank, (_, row) in enumerate(top10.iterrows(), 1):
            lines.append(f"\n#{rank}: Total_R={row['total_r']:.2f}")
            lines.append(f"   Winrate: {row['winrate']:.1f}% | Sharpe: {row['sharpe_ratio']:.2f}")
            lines.append(f"   Trades: {row['total_trades']} | PF: {row['profit_factor']:.2f}")

            if "params" in row and isinstance(row["params"], dict):
                p = row["params"]
                lines.append(f"   IB: {p.get('ib_start')}-{p.get('ib_end')} | "
                            f"RR: {p.get('rr_target')} | Stop: {p.get('stop_mode')}")

    # Variation report
    if aggregator:
        lines.append("\n")
        lines.append(generate_variation_report(aggregator))

    # Write to file
    output_path = output_dir / f"{symbol}_summary.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print_status(f"Saved summary to {output_path}", "SUCCESS")


def generate_all_reports(symbol: str, output_dir: Path) -> None:
    """Generate all reports for symbol."""
    print_status("=" * 60, "HEADER")
    print_status(f"Generating Reports for {symbol}", "HEADER")
    print_status("=" * 60, "HEADER")

    # Load data
    df = load_full_results(symbol, output_dir)
    aggregator = load_variation_aggregator(symbol, output_dir)

    if df is None:
        print_status("Cannot generate reports without results", "ERROR")
        return

    # Generate reports
    generate_top_n_csv(df, symbol, output_dir, top_n=1000)
    generate_summary_txt(df, aggregator, symbol, output_dir)

    if aggregator:
        generate_variation_csv(aggregator, symbol, output_dir)
        generate_best_params_json(aggregator, symbol, output_dir)
    else:
        print_status("Variation reports skipped (no aggregator data)", "WARNING")

    print_status("=" * 60, "HEADER")
    print_status("Report generation complete!", "SUCCESS")


def main():
    parser = argparse.ArgumentParser(description="Generate optimization reports")
    parser.add_argument("--symbol", choices=["GER40", "XAUUSD"], required=True,
                        help="Symbol to generate reports for")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory (default: params_optimizer/output)")
    args = parser.parse_args()

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "output"

    generate_all_reports(args.symbol, output_dir)


if __name__ == "__main__":
    main()
