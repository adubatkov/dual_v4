"""
Parameter Grid Generator for Parameter Optimization.

Generates and manages all valid parameter combinations for grid search.
Supports multiple grid modes: standard, features, btib.
"""

import itertools
import random
from typing import Dict, List, Any, Tuple, Set, Optional

from params_optimizer.config import (
    IB_TIME_CONFIGS,
    get_grid_for_symbol,
    print_status,
)


class ParameterGrid:
    """
    Generates and manages parameter combinations for optimization.

    Handles:
    - Generation of all valid combinations from per-asset grids
    - Smart filtering (skip BTIB sub-params when disabled, etc.)
    - Shuffling for random order testing
    - Conversion to/from hashable tuples for tracking
    """

    # Fixed keys order for tuple conversion (must be consistent).
    # 24 keys: 3 IB session + 21 grid params.
    # NOTE: Changing this order breaks checkpoint compatibility.
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
        "ib_buffer_pct",
        "max_distance_pct",
        "analysis_tf",
        "fractal_be_enabled",
        "fractal_tsl_enabled",
        "fvg_be_enabled",
        "rev_rb_enabled",
        "btib_enabled",
        "btib_sl_mode",
        "btib_core_cutoff_min",
        "btib_extension_pct",
        "btib_rr_target",
        "btib_tsl_target",
        "btib_tsl_sl",
    ]

    def __init__(self, symbol: str, mode: str = "standard"):
        """
        Initialize ParameterGrid.

        Args:
            symbol: Trading symbol (GER40, XAUUSD, NAS100, UK100)
            mode: Grid mode - "standard", "features", or "btib"
        """
        self.symbol = symbol
        self.mode = mode

        if symbol not in IB_TIME_CONFIGS:
            raise ValueError(f"Unknown symbol: {symbol}. Supported: {list(IB_TIME_CONFIGS.keys())}")

        self.ib_configs = IB_TIME_CONFIGS[symbol]
        self._combinations: Optional[List[Dict[str, Any]]] = None
        self._total_count: Optional[int] = None

    def generate_all(self, skip_invalid: bool = True) -> List[Dict[str, Any]]:
        """
        Generate all valid parameter combinations.

        Smart filtering rules:
        - TSL_TARGET <= RR_TARGET + 1 (skip unreachable TSL targets)
        - TSL_SL: placeholder when TSL_TARGET=0; filter TSL_SL > TSL_TARGET
        - BTIB sub-params: placeholder when BTIB_ENABLED=False
        - BTIB_TSL_SL: placeholder when BTIB_TSL_TARGET=0

        Args:
            skip_invalid: Skip invalid combinations (default True)

        Returns:
            List of parameter dicts
        """
        if self._combinations is not None:
            return self._combinations

        grid = get_grid_for_symbol(self.symbol, self.mode)
        combinations = []

        for ib_start, ib_end, ib_tz in self.ib_configs:
            # Base product: all non-conditional params
            base_combos = itertools.product(
                grid["IB_WAIT_MINUTES"],
                grid["TRADE_WINDOW_MINUTES"],
                grid["RR_TARGET"],
                grid["STOP_MODE"],
                grid["TSL_TARGET"],
                grid["MIN_SL_PCT"],
                grid["IB_BUFFER_PCT"],
                grid["MAX_DISTANCE_PCT"],
                grid["ANALYSIS_TF"],
                grid["FRACTAL_BE_ENABLED"],
                grid["FRACTAL_TSL_ENABLED"],
                grid["FVG_BE_ENABLED"],
                grid["REV_RB_ENABLED"],
                grid["BTIB_ENABLED"],
            )

            for (
                ib_wait, trade_window, rr_target, stop_mode,
                tsl_target, min_sl_pct, ib_buffer_pct, max_distance_pct,
                analysis_tf, fractal_be, fractal_tsl, fvg_be,
                rev_rb_enabled, btib_enabled,
            ) in base_combos:

                # --- Filter: TSL_TARGET <= RR_TARGET + 1 ---
                # TSL can't exceed RR+1 (first step is at RR, so max useful is RR+1)
                if tsl_target > 0 and tsl_target > rr_target + 1:
                    continue

                # --- Conditional: TSL_SL ---
                if tsl_target == 0.0 or tsl_target == 0:
                    tsl_sl_values = [grid["TSL_SL"][0]]
                else:
                    tsl_sl_values = [v for v in grid["TSL_SL"] if v <= tsl_target]
                    if not tsl_sl_values:
                        tsl_sl_values = [min(grid["TSL_SL"])]

                # --- Conditional: BTIB sub-params ---
                if not btib_enabled:
                    btib_sl_modes = [grid["BTIB_SL_MODE"][0]]
                    btib_cutoffs = [grid["BTIB_CORE_CUTOFF_MIN"][0]]
                    btib_extensions = [grid["BTIB_EXTENSION_PCT"][0]]
                    btib_rr_targets = [grid["BTIB_RR_TARGET"][0]]
                    btib_tsl_targets = [grid["BTIB_TSL_TARGET"][0]]
                    btib_tsl_sls = [grid["BTIB_TSL_SL"][0]]
                else:
                    btib_sl_modes = grid["BTIB_SL_MODE"]
                    btib_cutoffs = grid["BTIB_CORE_CUTOFF_MIN"]
                    btib_extensions = grid["BTIB_EXTENSION_PCT"]
                    btib_rr_targets = grid["BTIB_RR_TARGET"]
                    btib_tsl_targets = grid["BTIB_TSL_TARGET"]
                    btib_tsl_sls = grid["BTIB_TSL_SL"]

                # Generate conditional combinations
                for tsl_sl in tsl_sl_values:
                    for btib_combo in itertools.product(
                        btib_sl_modes, btib_cutoffs, btib_extensions,
                        btib_rr_targets, btib_tsl_targets,
                    ):
                        (btib_sl_mode, btib_cutoff, btib_extension,
                         btib_rr, btib_tsl_target_val) = btib_combo

                        # Conditional: BTIB_TSL_SL
                        if btib_tsl_target_val == 0 or btib_tsl_target_val == 0.0:
                            btib_tsl_sl_vals = [btib_tsl_sls[0]]
                        else:
                            btib_tsl_sl_vals = [v for v in btib_tsl_sls if v <= btib_tsl_target_val]
                            if not btib_tsl_sl_vals:
                                btib_tsl_sl_vals = [min(btib_tsl_sls)]

                        for btib_tsl_sl_val in btib_tsl_sl_vals:
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
                                "ib_buffer_pct": ib_buffer_pct,
                                "max_distance_pct": max_distance_pct,
                                "analysis_tf": analysis_tf,
                                "fractal_be_enabled": fractal_be,
                                "fractal_tsl_enabled": fractal_tsl,
                                "fvg_be_enabled": fvg_be,
                                "rev_rb_enabled": rev_rb_enabled,
                                "btib_enabled": btib_enabled,
                                "btib_sl_mode": btib_sl_mode,
                                "btib_core_cutoff_min": btib_cutoff,
                                "btib_extension_pct": btib_extension,
                                "btib_rr_target": btib_rr,
                                "btib_tsl_target": btib_tsl_target_val,
                                "btib_tsl_sl": btib_tsl_sl_val,
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


def print_grid_info(symbol: str, mode: str = "standard") -> None:
    """
    Print parameter grid information for symbol and mode.

    Args:
        symbol: Trading symbol
        mode: Grid mode
    """
    grid = ParameterGrid(symbol, mode=mode)
    combinations = grid.generate_all()

    print_status(f"Parameter Grid for {symbol} (mode={mode})", "HEADER")
    print_status("=" * 50, "HEADER")

    print_status(f"IB Time Configs: {len(IB_TIME_CONFIGS[symbol])}", "INFO")
    for start, end, tz in IB_TIME_CONFIGS[symbol]:
        print(f"  {start}-{end} {tz}")

    grid_dict = get_grid_for_symbol(symbol, mode)
    print_status(f"\nGrid parameters ({len(grid_dict)} params):", "INFO")
    for key, values in grid_dict.items():
        if len(values) > 1:
            print(f"  {key}: {values} ({len(values)} values)")
        else:
            print(f"  {key}: {values[0]} (fixed)")

    print_status("=" * 50, "HEADER")
    print_status(f"Total Combinations: {len(combinations):,}", "SUCCESS")

    for workers in [1, 10, 15, 20]:
        time_est = grid.estimate_time(len(combinations), workers)
        print_status(f"  With {workers} workers: ~{time_est}", "INFO")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Print parameter grid info")
    parser.add_argument("--symbol", choices=["GER40", "XAUUSD", "NAS100", "UK100", "all"], default="all")
    parser.add_argument("--mode", choices=["standard", "features", "btib"], default="standard")

    args = parser.parse_args()

    symbols = list(IB_TIME_CONFIGS.keys()) if args.symbol == "all" else [args.symbol]

    for sym in symbols:
        print_grid_info(sym, mode=args.mode)
        print()
