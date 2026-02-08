"""
Parameter Grid Generator for Parameter Optimization.

Generates and manages all valid parameter combinations for grid search.
"""

import itertools
import random
from typing import Dict, List, Any, Tuple, Set, Optional

from params_optimizer.config import (
    IB_TIME_CONFIGS,
    IB_WAIT_MINUTES_LIST,
    TRADE_WINDOW_MINUTES_LIST,
    RR_TARGET_LIST,
    STOP_MODE_LIST,
    TSL_TARGET_LIST,
    TSL_SL_LIST,
    MIN_SL_PCT_LIST,
    REV_RB_ENABLED_LIST,
    REV_RB_PCT_LIST,
    IB_BUFFER_PCT_LIST,
    MAX_DISTANCE_PCT_LIST,
    print_status,
)


class ParameterGrid:
    """
    Generates and manages parameter combinations for optimization.

    Handles:
    - Generation of all valid combinations
    - Filtering invalid combinations (e.g., REV_RB_PCT when REV_RB disabled)
    - Shuffling for random order testing
    - Conversion to/from hashable tuples for tracking
    """

    # Fixed keys order for tuple conversion (must be consistent)
    PARAM_KEYS = [
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

    def __init__(self, symbol: str):
        """
        Initialize ParameterGrid.

        Args:
            symbol: Trading symbol ("GER40" or "XAUUSD")
        """
        self.symbol = symbol

        if symbol not in IB_TIME_CONFIGS:
            raise ValueError(f"Unknown symbol: {symbol}. Must be GER40 or XAUUSD")

        self.ib_configs = IB_TIME_CONFIGS[symbol]
        self._combinations: Optional[List[Dict[str, Any]]] = None
        self._total_count: Optional[int] = None

    def generate_all(self, skip_invalid: bool = True) -> List[Dict[str, Any]]:
        """
        Generate all valid parameter combinations.

        Args:
            skip_invalid: Skip invalid combinations (default True)

        Returns:
            List of parameter dicts
        """
        if self._combinations is not None:
            return self._combinations

        combinations = []

        # Iterate through all IB time configs
        for ib_start, ib_end, ib_tz in self.ib_configs:
            # Generate base combinations
            base_combos = itertools.product(
                IB_WAIT_MINUTES_LIST,
                TRADE_WINDOW_MINUTES_LIST,
                RR_TARGET_LIST,
                STOP_MODE_LIST,
                TSL_TARGET_LIST,
                MIN_SL_PCT_LIST,
                REV_RB_ENABLED_LIST,
                IB_BUFFER_PCT_LIST,
                MAX_DISTANCE_PCT_LIST,
            )

            for (
                ib_wait,
                trade_window,
                rr_target,
                stop_mode,
                tsl_target,
                min_sl_pct,
                rev_rb_enabled,
                ib_buffer_pct,
                max_distance_pct,
            ) in base_combos:

                # Handle TSL_SL variations
                if tsl_target == 0.0 or tsl_target == 0:
                    # TSL disabled, only one TSL_SL value needed
                    tsl_sl_values = [TSL_SL_LIST[0]]  # Use first value as placeholder
                else:
                    # TSL enabled, test only TSL_SL values <= TSL_TARGET
                    # Filter: TSL_SL > TSL_TARGET creates instant SL hit on first TP
                    tsl_sl_values = [v for v in TSL_SL_LIST if v <= tsl_target]
                    if not tsl_sl_values:
                        # Fallback if no valid values (shouldn't happen with proper config)
                        tsl_sl_values = [min(TSL_SL_LIST)]

                # Handle REV_RB_PCT variations
                if not rev_rb_enabled:
                    # REV_RB disabled, only one REV_RB_PCT value needed
                    rev_rb_pct_values = [REV_RB_PCT_LIST[0]]  # Use first value as placeholder
                else:
                    # REV_RB enabled, test all REV_RB_PCT values
                    rev_rb_pct_values = REV_RB_PCT_LIST

                # Generate combinations with TSL_SL and REV_RB_PCT
                for tsl_sl in tsl_sl_values:
                    for rev_rb_pct in rev_rb_pct_values:
                        params = {
                            "ib_start": ib_start,
                            "ib_end": ib_end,
                            "ib_timezone": ib_tz,
                            "ib_wait_minutes": ib_wait,
                            "trade_window_minutes": trade_window,
                            "rr_target": rr_target,
                            "stop_mode": stop_mode,
                            "tsl_target": tsl_target,
                            "tsl_sl": tsl_sl,
                            "min_sl_pct": min_sl_pct,
                            "rev_rb_enabled": rev_rb_enabled,
                            "rev_rb_pct": rev_rb_pct,
                            "ib_buffer_pct": ib_buffer_pct,
                            "max_distance_pct": max_distance_pct,
                        }
                        combinations.append(params)

        self._combinations = combinations
        self._total_count = len(combinations)

        return combinations

    def get_total_count(self) -> int:
        """Get total number of combinations."""
        if self._total_count is None:
            self.generate_all()
        return self._total_count

    def shuffle(
        self,
        combinations: List[Dict[str, Any]],
        seed: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Shuffle combinations for random order testing.

        Args:
            combinations: List of parameter dicts
            seed: Random seed for reproducibility (None for random)

        Returns:
            Shuffled list
        """
        shuffled = combinations.copy()
        if seed is not None:
            random.seed(seed)
        random.shuffle(shuffled)
        return shuffled

    def to_tuple(self, params: Dict[str, Any]) -> Tuple:
        """
        Convert parameter dict to hashable tuple.

        Args:
            params: Parameter dict

        Returns:
            Tuple of parameter values in fixed order
        """
        return tuple(params[key] for key in self.PARAM_KEYS)

    def from_tuple(self, t: Tuple) -> Dict[str, Any]:
        """
        Convert tuple back to parameter dict.

        Args:
            t: Tuple of parameter values

        Returns:
            Parameter dict
        """
        return dict(zip(self.PARAM_KEYS, t))

    def filter_completed(
        self,
        combinations: List[Dict[str, Any]],
        completed: Set[Tuple]
    ) -> List[Dict[str, Any]]:
        """
        Filter out already completed combinations.

        Args:
            combinations: All combinations
            completed: Set of completed tuples

        Returns:
            List of remaining combinations
        """
        return [
            params for params in combinations
            if self.to_tuple(params) not in completed
        ]

    def estimate_time(
        self,
        total_combinations: int,
        num_workers: int,
        seconds_per_combo: float = 5.0
    ) -> str:
        """
        Estimate total run time.

        Args:
            total_combinations: Number of combinations
            num_workers: Number of parallel workers
            seconds_per_combo: Estimated time per combination

        Returns:
            Human-readable time estimate
        """
        total_seconds = (total_combinations / num_workers) * seconds_per_combo
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)

        if hours > 0:
            return f"{hours}h {minutes}m"
        else:
            return f"{minutes}m"


def print_grid_info(symbol: str) -> None:
    """
    Print parameter grid information for symbol.

    Args:
        symbol: Trading symbol
    """
    grid = ParameterGrid(symbol)
    combinations = grid.generate_all()

    print_status(f"Parameter Grid for {symbol}", "HEADER")
    print_status("=" * 50, "HEADER")

    print_status(f"IB Time Configs: {len(IB_TIME_CONFIGS[symbol])}", "INFO")
    for start, end, tz in IB_TIME_CONFIGS[symbol]:
        print(f"  {start}-{end} {tz}")

    print_status(f"IB Wait Minutes: {IB_WAIT_MINUTES_LIST}", "INFO")
    print_status(f"Trade Window Minutes: {TRADE_WINDOW_MINUTES_LIST}", "INFO")
    print_status(f"RR Target: {RR_TARGET_LIST}", "INFO")
    print_status(f"Stop Mode: {STOP_MODE_LIST}", "INFO")
    print_status(f"TSL Target: {TSL_TARGET_LIST}", "INFO")
    print_status(f"TSL SL: {TSL_SL_LIST}", "INFO")
    print_status(f"Min SL PCT: {MIN_SL_PCT_LIST}", "INFO")
    print_status(f"REV RB Enabled: {REV_RB_ENABLED_LIST}", "INFO")
    print_status(f"REV RB PCT: {REV_RB_PCT_LIST}", "INFO")
    print_status(f"IB Buffer PCT: {IB_BUFFER_PCT_LIST}", "INFO")
    print_status(f"Max Distance PCT: {MAX_DISTANCE_PCT_LIST}", "INFO")

    print_status("=" * 50, "HEADER")
    print_status(f"Total Combinations: {len(combinations):,}", "SUCCESS")

    # Estimate time
    for workers in [1, 10, 50, 90]:
        time_est = grid.estimate_time(len(combinations), workers)
        print_status(f"  With {workers} workers: ~{time_est}", "INFO")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Print parameter grid info")
    parser.add_argument("--symbol", choices=["GER40", "XAUUSD", "all"], default="all")

    args = parser.parse_args()

    symbols = ["GER40", "XAUUSD"] if args.symbol == "all" else [args.symbol]

    for sym in symbols:
        print_grid_info(sym)
        print()
