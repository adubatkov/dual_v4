"""
Metrics Calculator for Parameter Optimization.

Calculates combined scores and rankings for optimization results.
"""

from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd

from params_optimizer.config import RANKING_WEIGHTS


class MetricsCalculator:
    """
    Calculates combined scores and rankings for optimization results.

    Uses weighted combination of:
    - Total R (profit in risk units)
    - Sharpe Ratio (risk-adjusted return)
    - Win Rate (stability)
    """

    def __init__(
        self,
        weight_total_r: float = 0.40,
        weight_sharpe: float = 0.35,
        weight_winrate: float = 0.25,
    ):
        """
        Initialize MetricsCalculator.

        Args:
            weight_total_r: Weight for Total R (default 0.40)
            weight_sharpe: Weight for Sharpe Ratio (default 0.35)
            weight_winrate: Weight for Win Rate (default 0.25)
        """
        self.weights = {
            "total_r": weight_total_r,
            "sharpe_ratio": weight_sharpe,
            "winrate": weight_winrate,
        }

        # Validate weights sum to 1.0
        total_weight = sum(self.weights.values())
        if abs(total_weight - 1.0) > 0.001:
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")

    def calculate_combined_score(
        self,
        results: pd.DataFrame,
        min_trades: int = 10,
    ) -> pd.DataFrame:
        """
        Calculate combined score for all results.

        Args:
            results: DataFrame with columns [total_r, sharpe_ratio, winrate, ...]
            min_trades: Minimum trades to include in ranking

        Returns:
            DataFrame with added columns [normalized_*, combined_score]
        """
        df = results.copy()

        # Filter by minimum trades
        df = df[df["total_trades"] >= min_trades].copy()

        if len(df) == 0:
            return pd.DataFrame()

        # Normalize metrics using min-max scaling
        for metric in ["total_r", "sharpe_ratio", "winrate"]:
            if metric not in df.columns:
                raise ValueError(f"Missing column: {metric}")

            col = df[metric]
            min_val = col.min()
            max_val = col.max()

            if max_val > min_val:
                df[f"norm_{metric}"] = (col - min_val) / (max_val - min_val)
            else:
                df[f"norm_{metric}"] = 0.5  # All same value

        # Calculate combined score
        df["combined_score"] = (
            df["norm_total_r"] * self.weights["total_r"] +
            df["norm_sharpe_ratio"] * self.weights["sharpe_ratio"] +
            df["norm_winrate"] * self.weights["winrate"]
        )

        return df

    def rank_results(
        self,
        results: pd.DataFrame,
        min_trades: int = 10,
        top_n: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Rank results by combined score.

        Args:
            results: DataFrame with results
            min_trades: Minimum trades to include
            top_n: Return only top N results (None for all)

        Returns:
            Ranked DataFrame with 'rank' column
        """
        df = self.calculate_combined_score(results, min_trades)

        if len(df) == 0:
            return pd.DataFrame()

        # Sort by combined score (descending)
        df = df.sort_values("combined_score", ascending=False)

        # Add rank
        df["rank"] = range(1, len(df) + 1)

        # Limit to top N if specified
        if top_n is not None:
            df = df.head(top_n)

        return df

    def get_best_params(self, results: pd.DataFrame, min_trades: int = 10) -> Optional[Dict[str, Any]]:
        """
        Get best parameter combination.

        Args:
            results: DataFrame with results
            min_trades: Minimum trades to include

        Returns:
            Best params dict or None if no valid results
        """
        ranked = self.rank_results(results, min_trades, top_n=1)

        if len(ranked) == 0:
            return None

        best_row = ranked.iloc[0]

        # Extract params from the row
        if "params" in best_row and isinstance(best_row["params"], dict):
            return best_row["params"]

        return None

    def calculate_percentile_scores(
        self,
        results: pd.DataFrame,
        min_trades: int = 10,
    ) -> pd.DataFrame:
        """
        Calculate percentile scores for each metric.

        Args:
            results: DataFrame with results
            min_trades: Minimum trades to include

        Returns:
            DataFrame with percentile columns
        """
        df = results[results["total_trades"] >= min_trades].copy()

        if len(df) == 0:
            return pd.DataFrame()

        for metric in ["total_r", "sharpe_ratio", "winrate", "profit_factor"]:
            if metric in df.columns:
                df[f"pct_{metric}"] = df[metric].rank(pct=True) * 100

        return df

    def get_statistics(self, results: pd.DataFrame) -> Dict[str, Any]:
        """
        Get summary statistics for results.

        Args:
            results: DataFrame with results

        Returns:
            Dict with statistics
        """
        if len(results) == 0:
            return {
                "total_combinations": 0,
                "valid_combinations": 0,
                "errors": 0,
            }

        # Handle case when "error" column doesn't exist (all results are valid)
        if "error" not in results.columns:
            valid = results
            errors = results.iloc[0:0]  # Empty DataFrame with same structure
        else:
            valid = results[results["error"].isna() | (results["error"] == "")]
            errors = results[results["error"].notna() & (results["error"] != "")]

        stats = {
            "total_combinations": len(results),
            "valid_combinations": len(valid),
            "errors": len(errors),
            "min_trades": int(valid["total_trades"].min()) if len(valid) > 0 else 0,
            "max_trades": int(valid["total_trades"].max()) if len(valid) > 0 else 0,
            "avg_trades": float(valid["total_trades"].mean()) if len(valid) > 0 else 0,
            "total_r_range": (
                float(valid["total_r"].min()) if len(valid) > 0 else 0,
                float(valid["total_r"].max()) if len(valid) > 0 else 0,
            ),
            "winrate_range": (
                float(valid["winrate"].min()) if len(valid) > 0 else 0,
                float(valid["winrate"].max()) if len(valid) > 0 else 0,
            ),
            "sharpe_range": (
                float(valid["sharpe_ratio"].min()) if len(valid) > 0 else 0,
                float(valid["sharpe_ratio"].max()) if len(valid) > 0 else 0,
            ),
        }

        return stats

    def generate_summary_report(
        self,
        results: pd.DataFrame,
        top_n: int = 10,
        min_trades: int = 10,
    ) -> str:
        """
        Generate human-readable summary report.

        Args:
            results: DataFrame with results
            top_n: Number of top results to show
            min_trades: Minimum trades for ranking

        Returns:
            Formatted report string
        """
        stats = self.get_statistics(results)
        ranked = self.rank_results(results, min_trades, top_n)

        lines = [
            "=" * 70,
            "PARAMETER OPTIMIZATION RESULTS",
            "=" * 70,
            "",
            "STATISTICS",
            "-" * 40,
            f"Total Combinations: {stats['total_combinations']:,}",
            f"Valid Combinations: {stats['valid_combinations']:,}",
            f"Errors: {stats['errors']:,}",
            "",
            f"Trades Range: {stats['min_trades']} - {stats['max_trades']}",
            f"Avg Trades: {stats['avg_trades']:.1f}",
            "",
            f"Total R Range: {stats['total_r_range'][0]:.2f} to {stats['total_r_range'][1]:.2f}",
            f"Win Rate Range: {stats['winrate_range'][0]:.1f}% to {stats['winrate_range'][1]:.1f}%",
            f"Sharpe Range: {stats['sharpe_range'][0]:.2f} to {stats['sharpe_range'][1]:.2f}",
            "",
            "=" * 70,
            f"TOP {min(top_n, len(ranked))} PARAMETER SETS (min {min_trades} trades)",
            "=" * 70,
            "",
        ]

        for idx, row in ranked.iterrows():
            params = row.get("params", {})
            lines.append(f"Rank #{int(row['rank'])}: Score={row['combined_score']:.4f}")
            lines.append(f"  Total R: {row['total_r']:.2f} | Sharpe: {row['sharpe_ratio']:.2f} | "
                        f"Win Rate: {row['winrate']:.1f}%")
            lines.append(f"  Trades: {int(row['total_trades'])} | PF: {row['profit_factor']:.2f} | "
                        f"Max DD: {row['max_drawdown']:.1f}%")
            lines.append(f"  Params: IB {params.get('ib_start', '?')}-{params.get('ib_end', '?')} | "
                        f"Wait: {params.get('ib_wait_minutes', '?')}m | "
                        f"Window: {params.get('trade_window_minutes', '?')}m")
            lines.append(f"          RR: {params.get('rr_target', '?')} | "
                        f"Stop: {params.get('stop_mode', '?')} | "
                        f"TSL: {params.get('tsl_target', '?')}/{params.get('tsl_sl', '?')}")
            lines.append(f"          Buffer: {params.get('ib_buffer_pct', '?')} | "
                        f"MaxDist: {params.get('max_distance_pct', '?')} | "
                        f"RevRB: {params.get('rev_rb_enabled', '?')}")
            lines.append("")

        return "\n".join(lines)


if __name__ == "__main__":
    # Test with sample data
    import random

    sample_data = []
    for i in range(100):
        sample_data.append({
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
            "params": {
                "ib_start": "08:00",
                "ib_end": "09:00",
                "ib_wait_minutes": random.choice([0, 15, 30]),
                "trade_window_minutes": random.choice([60, 90, 120]),
                "rr_target": random.choice([0.75, 1.0, 1.25]),
                "stop_mode": random.choice(["ib_start", "eq"]),
                "tsl_target": random.choice([0, 1.0, 1.5]),
                "tsl_sl": 1.0,
                "ib_buffer_pct": random.choice([0, 0.05, 0.1]),
                "max_distance_pct": random.choice([0.5, 0.75, 1.0]),
                "rev_rb_enabled": random.choice([True, False]),
            },
            "error": None if random.random() > 0.05 else "Test error",
        })

    df = pd.DataFrame(sample_data)

    calc = MetricsCalculator()
    report = calc.generate_summary_report(df, top_n=5, min_trades=10)
    print(report)
