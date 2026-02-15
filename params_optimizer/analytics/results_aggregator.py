"""
Results Aggregator for Parameter Optimization.

Combines results from all workers and generates reports.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

import pandas as pd

from params_optimizer.config import print_status
from params_optimizer.engine.metrics_calculator import MetricsCalculator
from params_optimizer.engine.parameter_grid import ParameterGrid


class ResultsAggregator:
    """
    Aggregates optimization results and generates reports.

    Handles:
    - Converting raw results to DataFrame
    - Ranking by combined score
    - Generating various output formats
    """

    def __init__(
        self,
        symbol: str,
        output_dir: Path,
        metrics_calculator: Optional[MetricsCalculator] = None,
    ):
        """
        Initialize ResultsAggregator.

        Args:
            symbol: Trading symbol
            output_dir: Output directory for reports
            metrics_calculator: Custom metrics calculator (optional)
        """
        self.symbol = symbol
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.metrics = metrics_calculator or MetricsCalculator()
        self.grid = ParameterGrid(symbol)

    def aggregate(self, results: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Convert raw results to DataFrame.

        Args:
            results: List of result dicts from workers

        Returns:
            DataFrame with all results
        """
        if not results:
            return pd.DataFrame()

        flat_results = []

        for r in results:
            if r.get("results"):
                row = r["results"].copy()
                flat_results.append(row)
            elif r.get("params"):
                # Error case
                row = {
                    "total_r": 0.0,
                    "total_profit": 0.0,
                    "total_trades": 0,
                    "winning_trades": 0,
                    "losing_trades": 0,
                    "winrate": 0.0,
                    "sharpe_ratio": 0.0,
                    "profit_factor": 0.0,
                    "max_drawdown": 0.0,
                    "avg_trade_r": 0.0,
                    "params": r["params"],
                    "error": r.get("error", "Unknown error"),
                }
                flat_results.append(row)

        return pd.DataFrame(flat_results)

    def rank(
        self,
        df: pd.DataFrame,
        min_trades: int = 10,
        top_n: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Rank results by combined score.

        Args:
            df: Results DataFrame
            min_trades: Minimum trades to include
            top_n: Return only top N results

        Returns:
            Ranked DataFrame
        """
        return self.metrics.rank_results(df, min_trades, top_n)

    def save_reports(
        self,
        df: pd.DataFrame,
        run_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Path]:
        """
        Generate and save all reports.

        Args:
            df: Results DataFrame
            run_info: Optional run metadata

        Returns:
            Dict of report names to file paths
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        paths = {}

        # 1. Full results Parquet
        full_path = self.output_dir / f"{self.symbol}_full_results_{timestamp}.parquet"
        df_for_parquet = df.copy()

        # Convert params dict to JSON for Parquet storage
        if "params" in df_for_parquet.columns:
            df_for_parquet["params_json"] = df_for_parquet["params"].apply(
                lambda p: json.dumps(p) if isinstance(p, dict) else str(p)
            )
            df_for_parquet = df_for_parquet.drop(columns=["params"])

        df_for_parquet.to_parquet(full_path, index=False)
        paths["full_results"] = full_path
        print_status(f"Full results: {full_path}", "SUCCESS")

        # 2. Top 100 CSV
        ranked = self.rank(df, min_trades=10, top_n=100)
        if len(ranked) > 0:
            top_path = self.output_dir / f"{self.symbol}_top_100_{timestamp}.csv"
            top_flat = self._flatten_params(ranked)
            top_flat.to_csv(top_path, index=False)
            paths["top_100"] = top_path
            print_status(f"Top 100: {top_path}", "SUCCESS")

        # 3. Best params JSON
        best_params = self.metrics.get_best_params(df, min_trades=10)
        if best_params:
            best_path = self.output_dir / f"{self.symbol}_best_params_{timestamp}.json"
            with open(best_path, "w") as f:
                json.dump(best_params, f, indent=2)
            paths["best_params"] = best_path
            print_status(f"Best params: {best_path}", "SUCCESS")

        # 4. Summary report
        report = self.metrics.generate_summary_report(df, top_n=20, min_trades=10)
        report_path = self.output_dir / f"{self.symbol}_summary_{timestamp}.txt"

        # Add run info header if provided
        if run_info:
            header_lines = [
                "RUN INFORMATION",
                "-" * 40,
                f"Symbol: {run_info.get('symbol', self.symbol)}",
                f"Start Time: {run_info.get('start_time', 'N/A')}",
                f"End Time: {run_info.get('end_time', datetime.now().isoformat())}",
                f"Duration: {run_info.get('duration', 'N/A')}",
                f"Workers: {run_info.get('workers', 'N/A')}",
                f"Data Period: {run_info.get('data_period', 'N/A')}",
                "",
            ]
            report = "\n".join(header_lines) + report

        with open(report_path, "w") as f:
            f.write(report)
        paths["summary"] = report_path
        print_status(f"Summary: {report_path}", "SUCCESS")

        # 5. Latest symlinks (for easy access)
        self._create_latest_symlinks(paths, timestamp)

        return paths

    def _flatten_params(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Flatten params dict into separate columns.

        Args:
            df: DataFrame with 'params' column

        Returns:
            DataFrame with flattened params
        """
        result = df.copy()

        if "params" not in result.columns:
            return result

        # Extract each param key
        for key in ParameterGrid.PARAM_KEYS:
            result[f"param_{key}"] = result["params"].apply(
                lambda p: p.get(key, "") if isinstance(p, dict) else ""
            )

        result = result.drop(columns=["params"])
        return result

    def _create_latest_symlinks(self, paths: Dict[str, Path], timestamp: str) -> None:
        """
        Create 'latest' symlinks to most recent files.

        Args:
            paths: Dict of report type to file path
            timestamp: Current timestamp string
        """
        for report_type, path in paths.items():
            suffix = path.suffix
            latest_path = self.output_dir / f"{self.symbol}_{report_type}_latest{suffix}"

            # Remove existing symlink
            if latest_path.exists() or latest_path.is_symlink():
                latest_path.unlink()

            # Create new symlink (or copy on Windows)
            try:
                latest_path.symlink_to(path.name)
            except OSError:
                # Windows may not support symlinks, just copy
                import shutil
                shutil.copy2(path, latest_path)

    def load_results(self, results_path: Path) -> pd.DataFrame:
        """
        Load results from Parquet file.

        Args:
            results_path: Path to results Parquet file

        Returns:
            DataFrame with results
        """
        df = pd.read_parquet(results_path)

        # Convert params_json back to dict if present
        if "params_json" in df.columns:
            df["params"] = df["params_json"].apply(json.loads)
            df = df.drop(columns=["params_json"])

        return df

    def compare_runs(
        self,
        results_paths: List[Path],
        labels: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Compare top results across multiple optimization runs.

        Args:
            results_paths: List of results Parquet file paths
            labels: Optional labels for each run

        Returns:
            DataFrame comparing top results
        """
        if labels is None:
            labels = [f"Run_{i}" for i in range(len(results_paths))]

        comparisons = []

        for path, label in zip(results_paths, labels):
            df = self.load_results(path)
            ranked = self.rank(df, min_trades=10, top_n=5)

            if len(ranked) > 0:
                for idx, row in ranked.iterrows():
                    comparisons.append({
                        "run": label,
                        "rank": int(row["rank"]),
                        "total_r": row["total_r"],
                        "sharpe_ratio": row["sharpe_ratio"],
                        "winrate": row["winrate"],
                        "total_trades": int(row["total_trades"]),
                        "combined_score": row["combined_score"],
                    })

        return pd.DataFrame(comparisons)


if __name__ == "__main__":
    # Test with sample data
    import tempfile
    import random

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # Generate sample results
        sample_results = []
        for i in range(100):
            params = {
                "ib_start": "08:00",
                "ib_end": "09:00",
                "ib_timezone": "Europe/Berlin",
                "ib_wait_minutes": random.choice([0, 15, 30]),
                "trade_window_minutes": random.choice([60, 90, 120]),
                "rr_target": random.choice([0.75, 1.0, 1.25]),
                "stop_mode": random.choice(["ib_start", "eq"]),
                "tsl_target": random.choice([0, 1.0, 1.5]),
                "tsl_sl": 1.0,
                "min_sl_pct": 0.001,
                "ib_buffer_pct": random.choice([0, 0.05, 0.1]),
                "max_distance_pct": random.choice([0.5, 0.75, 1.0]),
                "analysis_tf": "2min",
                "fractal_be_enabled": True,
                "fractal_tsl_enabled": True,
                "fvg_be_enabled": False,
                "rev_rb_enabled": random.choice([True, False]),
                "btib_enabled": False,
                "btib_sl_mode": "fractal_2m",
                "btib_core_cutoff_min": 40,
                "btib_extension_pct": 1.0,
                "btib_rr_target": 1.0,
                "btib_tsl_target": 0.0,
                "btib_tsl_sl": 0.0,
            }

            sample_results.append({
                "params": params,
                "results": {
                    "total_r": random.uniform(-10, 50),
                    "total_profit": random.uniform(-5000, 20000),
                    "total_trades": random.randint(5, 100),
                    "winning_trades": random.randint(0, 50),
                    "losing_trades": random.randint(0, 50),
                    "winrate": random.uniform(30, 70),
                    "sharpe_ratio": random.uniform(-1, 3),
                    "profit_factor": random.uniform(0.5, 3),
                    "max_drawdown": random.uniform(5, 30),
                    "avg_trade_r": random.uniform(-0.5, 1),
                    "params": params,
                    "error": None,
                },
                "error": None,
            })

        # Test aggregator
        aggregator = ResultsAggregator("GER40", output_dir)

        df = aggregator.aggregate(sample_results)
        print(f"Aggregated {len(df)} results")

        ranked = aggregator.rank(df, min_trades=10, top_n=10)
        print(f"Ranked {len(ranked)} results")

        paths = aggregator.save_reports(df, {
            "symbol": "GER40",
            "start_time": datetime.now().isoformat(),
            "workers": 4,
        })

        print(f"\nSaved reports to:")
        for name, path in paths.items():
            print(f"  {name}: {path}")
