"""
Generate Backtest Groups for Parallel Slow Backtest Validation

Creates unique parameter groups from SQLite optimization DBs using:
- 3 grouping modes: IB Only, IB+Buffer, IB+Buffer+MaxDist
- 4 ranking strategies: Total R, Sharpe-Weighted, Calmar, Multi-Criteria

Output: backtest_groups_{symbol}.json with unique parameter sets
"""

import sqlite3
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any

# Configuration
ANALYZE_DIR = Path(__file__).parent
MAX_DRAWDOWN_FILTER = 12.0  # Max allowed drawdown in R per variation
TOP_N = 40  # Number of top groups per ranking strategy (18 categories × 40 = 720 candidates → ~100+ unique after dedup)

DB_PATHS = {
    "GER40": ANALYZE_DIR / "GER40_optimization.db",
    "XAUUSD": ANALYZE_DIR / "XAUUSD_optimization.db",
}

VARIATIONS = ["ocae", "tcwe", "reverse", "rev_rb"]

# Default MIN_SL_PCT by symbol
MIN_SL_PCT = {
    "GER40": 0.0015,
    "XAUUSD": 0.001,
}


def load_data_from_db(symbol: str) -> pd.DataFrame:
    """Load data from SQLite database."""
    db_path = DB_PATHS[symbol]
    print(f"[INFO] Loading {symbol} from {db_path.name}...")

    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM results", conn)
    conn.close()

    print(f"  -> Loaded {len(df):,} rows")
    return df


def parse_params(df: pd.DataFrame) -> pd.DataFrame:
    """Parse params_json column into separate columns."""
    def safe_parse(x):
        if isinstance(x, dict):
            return x
        elif isinstance(x, str):
            return json.loads(x)
        return {}

    params_list = df["params_json"].apply(safe_parse).tolist()
    params_df = pd.DataFrame(params_list)
    result = pd.concat([df.drop(columns=["params_json"]), params_df], axis=1)
    return result


# =============================================================================
# Grouping Functions
# =============================================================================

def get_group_key_ib_only(row) -> tuple:
    """Group key: IB params only (no buffer/maxdist constraint)."""
    return (
        row["ib_start"],
        row["ib_end"],
        row["ib_timezone"],
        int(row["ib_wait_minutes"]),
    )


def get_group_key_ib_buffer(row) -> tuple:
    """Group key: IB params + Buffer."""
    return (
        row["ib_start"],
        row["ib_end"],
        row["ib_timezone"],
        int(row["ib_wait_minutes"]),
        float(row["ib_buffer_pct"]),
    )


def get_group_key_ib_buffer_maxdist(row) -> tuple:
    """Group key: IB params + Buffer + MaxDist."""
    return (
        row["ib_start"],
        row["ib_end"],
        row["ib_timezone"],
        int(row["ib_wait_minutes"]),
        float(row["ib_buffer_pct"]),
        float(row["max_distance_pct"]),
    )


# =============================================================================
# Ranking Strategies
# =============================================================================

def rank_by_total_r(groups: List[dict]) -> List[dict]:
    """Rank groups by combined_total_r (highest first)."""
    return sorted(groups, key=lambda x: x["combined_total_r"], reverse=True)


def rank_by_sharpe_weighted(groups: List[dict]) -> List[dict]:
    """Rank groups by 50% Total R + 50% Sharpe (normalized)."""
    if not groups:
        return groups

    max_r = max(g["combined_total_r"] for g in groups) or 1
    max_sharpe = max(g["weighted_sharpe"] for g in groups) or 1

    for g in groups:
        norm_r = g["combined_total_r"] / max_r
        norm_sharpe = g["weighted_sharpe"] / max_sharpe
        g["_score"] = (norm_r * 0.5) + (norm_sharpe * 0.5)

    return sorted(groups, key=lambda x: x["_score"], reverse=True)


def rank_by_calmar(groups: List[dict]) -> List[dict]:
    """Rank groups by Calmar ratio (Total R / Max DD)."""
    for g in groups:
        max_dd = g.get("max_group_dd", 1) or 1
        g["_score"] = g["combined_total_r"] / (max_dd + 1)

    return sorted(groups, key=lambda x: x["_score"], reverse=True)


def rank_by_multi_criteria(groups: List[dict]) -> List[dict]:
    """Rank by multi-criteria: 35% R + 35% Sharpe + 15% DD penalty + 15% trades."""
    if not groups:
        return groups

    max_r = max(g["combined_total_r"] for g in groups) or 1
    max_sharpe = max(g["weighted_sharpe"] for g in groups) or 1
    max_trades = max(g["total_trades"] for g in groups) or 1

    for g in groups:
        norm_r = g["combined_total_r"] / max_r
        norm_sharpe = g["weighted_sharpe"] / max_sharpe
        norm_trades = g["total_trades"] / max_trades

        # DD penalty: lower is better, max penalty at 12R
        max_dd = min(g.get("max_group_dd", 0), 12)
        dd_penalty = (12 - max_dd) / 12

        g["_score"] = (norm_r * 0.35) + (norm_sharpe * 0.35) + (dd_penalty * 0.15) + (norm_trades * 0.15)

    return sorted(groups, key=lambda x: x["_score"], reverse=True)


def rank_by_winrate_focus(groups: List[dict]) -> List[dict]:
    """Rank groups with winrate focus: 50% winrate + 30% R + 20% Sharpe."""
    if not groups:
        return groups

    # Calculate average winrate per group
    for g in groups:
        total_winrate = 0
        count = 0
        for var_data in g["variations"].values():
            if var_data["trades"] > 0:
                total_winrate += var_data["winrate"]
                count += 1
        g["_avg_winrate"] = total_winrate / count if count > 0 else 0

    max_r = max(g["combined_total_r"] for g in groups) or 1
    max_sharpe = max(g["weighted_sharpe"] for g in groups) or 1
    max_wr = max(g["_avg_winrate"] for g in groups) or 1

    for g in groups:
        norm_r = g["combined_total_r"] / max_r
        norm_sharpe = g["weighted_sharpe"] / max_sharpe
        norm_wr = g["_avg_winrate"] / max_wr
        g["_score"] = (norm_wr * 0.5) + (norm_r * 0.3) + (norm_sharpe * 0.2)

    return sorted(groups, key=lambda x: x["_score"], reverse=True)


def rank_by_r_sharpe_70_30(groups: List[dict]) -> List[dict]:
    """Rank groups by 70% R + 30% Sharpe."""
    if not groups:
        return groups

    max_r = max(g["combined_total_r"] for g in groups) or 1
    max_sharpe = max(g["weighted_sharpe"] for g in groups) or 1

    for g in groups:
        norm_r = g["combined_total_r"] / max_r
        norm_sharpe = g["weighted_sharpe"] / max_sharpe
        g["_score"] = (norm_r * 0.7) + (norm_sharpe * 0.3)

    return sorted(groups, key=lambda x: x["_score"], reverse=True)


RANKING_STRATEGIES = {
    "total_r": rank_by_total_r,
    "sharpe_weighted": rank_by_sharpe_weighted,
    "calmar": rank_by_calmar,
    "multi_criteria": rank_by_multi_criteria,
    "winrate_focus": rank_by_winrate_focus,
    "r_sharpe_70_30": rank_by_r_sharpe_70_30,
}


# =============================================================================
# Group Analysis
# =============================================================================

def find_best_variation_params(group_df: pd.DataFrame, variation: str) -> dict:
    """Find best parameters for a variation within a group."""
    prefix = variation

    total_r_col = f"{prefix}_total_r"
    max_dd_col = f"{prefix}_max_drawdown"
    sharpe_col = f"{prefix}_sharpe_ratio"
    winrate_col = f"{prefix}_winrate"
    trades_col = f"{prefix}_trades"

    # Filter by max drawdown
    filtered = group_df[group_df[max_dd_col] <= MAX_DRAWDOWN_FILTER].copy()

    if filtered.empty:
        filtered = group_df.copy()

    # Filter by TSL constraint: TSL_SL < RR_TARGET + 1
    # This ensures TSL can actually execute on live bot (price must reach target before SL adjustment)
    if "tsl_target" in filtered.columns and "tsl_sl" in filtered.columns and "rr_target" in filtered.columns:
        valid_tsl_mask = (filtered["tsl_target"] <= 0) | (filtered["tsl_sl"] < filtered["rr_target"] + 1)
        filtered = filtered[valid_tsl_mask]

    if filtered.empty:
        return None

    best_idx = filtered[total_r_col].idxmax()
    best_row = filtered.loc[best_idx]

    result = {
        "variation": variation.upper(),
        "total_r": float(best_row[total_r_col]),
        "sharpe": float(best_row[sharpe_col]) if pd.notna(best_row[sharpe_col]) else 0,
        "winrate": float(best_row[winrate_col]) if pd.notna(best_row[winrate_col]) else 0,
        "trades": int(best_row[trades_col]) if pd.notna(best_row[trades_col]) else 0,
        "max_drawdown": float(best_row[max_dd_col]) if pd.notna(best_row[max_dd_col]) else 0,
        # Common params
        "ib_start": best_row["ib_start"],
        "ib_end": best_row["ib_end"],
        "ib_timezone": best_row["ib_timezone"],
        "ib_wait_minutes": int(best_row["ib_wait_minutes"]),
        "ib_buffer_pct": float(best_row["ib_buffer_pct"]),
        "max_distance_pct": float(best_row["max_distance_pct"]),
        # Variation-specific params
        "trade_window_minutes": int(best_row["trade_window_minutes"]),
        "rr_target": float(best_row["rr_target"]),
        "stop_mode": best_row["stop_mode"],
        "tsl_target": float(best_row["tsl_target"]),
        "tsl_sl": float(best_row["tsl_sl"]),
        "min_sl_pct": float(best_row["min_sl_pct"]),
        "rev_rb_enabled": bool(best_row.get("rev_rb_enabled", False)),
        "rev_rb_pct": float(best_row.get("rev_rb_pct", 1.0)),
    }

    return result


def analyze_with_grouping(df: pd.DataFrame, group_key_func, mode_name: str) -> List[dict]:
    """Analyze data with specified grouping function."""
    print(f"\n[INFO] Analyzing with grouping: {mode_name}")

    df["group_key"] = df.apply(group_key_func, axis=1)
    groups = df["group_key"].unique()
    print(f"  -> Found {len(groups)} unique groups")

    group_results = []

    for group_key in groups:
        group_df = df[df["group_key"] == group_key]

        variation_results = {}
        combined_total_r = 0
        total_trades = 0
        weighted_sharpe_sum = 0
        max_group_dd = 0

        for var in VARIATIONS:
            best = find_best_variation_params(group_df, var)
            if best:
                variation_results[var] = best
                combined_total_r += best["total_r"]
                trades = best["trades"]
                if trades > 0:
                    weighted_sharpe_sum += best["sharpe"] * trades
                    total_trades += trades
                max_group_dd = max(max_group_dd, best["max_drawdown"])

        if not variation_results:
            continue

        weighted_avg_sharpe = weighted_sharpe_sum / total_trades if total_trades > 0 else 0

        first_var = list(variation_results.values())[0]

        result = {
            "group_key": str(group_key),
            "mode": mode_name,
            "ib_start": first_var["ib_start"],
            "ib_end": first_var["ib_end"],
            "ib_timezone": first_var["ib_timezone"],
            "ib_wait": first_var["ib_wait_minutes"],
            "ib_buffer_pct": first_var["ib_buffer_pct"],
            "max_distance_pct": first_var["max_distance_pct"],
            "combined_total_r": combined_total_r,
            "weighted_sharpe": weighted_avg_sharpe,
            "max_group_dd": max_group_dd,
            "total_trades": total_trades,
            "variations": variation_results,
        }

        group_results.append(result)

    return group_results


def generate_groups_for_symbol(symbol: str) -> Dict[str, List[dict]]:
    """Generate all groups for a symbol with all ranking strategies."""
    print(f"\n{'='*60}")
    print(f"GENERATING GROUPS FOR {symbol}")
    print(f"{'='*60}")

    df = load_data_from_db(symbol)
    df = parse_params(df)

    # Apply 3 grouping modes
    grouping_modes = {
        "ib_only": (get_group_key_ib_only, "IB Only"),
        "ib_buffer": (get_group_key_ib_buffer, "IB + Buffer"),
        "ib_buffer_maxdist": (get_group_key_ib_buffer_maxdist, "IB + Buffer + MaxDist"),
    }

    all_groups_by_mode = {}
    for mode_key, (func, mode_name) in grouping_modes.items():
        all_groups_by_mode[mode_key] = analyze_with_grouping(df, func, mode_name)

    # Apply 4 ranking strategies to each mode and collect TOP N
    result = {}

    for mode_key, groups in all_groups_by_mode.items():
        for rank_key, rank_func in RANKING_STRATEGIES.items():
            key = f"{mode_key}_{rank_key}"
            ranked = rank_func(groups.copy())[:TOP_N]
            result[key] = ranked
            print(f"  {key}: {len(ranked)} groups")

    return result


def deduplicate_groups(groups_by_category: Dict[str, List[dict]], symbol: str) -> List[dict]:
    """Deduplicate groups across categories, keeping unique parameter sets."""
    seen_keys = set()
    unique_groups = []

    for category, groups in groups_by_category.items():
        for i, group in enumerate(groups):
            # Create unique key from all variation parameters
            params_key = []
            for var_name, var_data in sorted(group["variations"].items()):
                params_key.append((
                    var_name,
                    var_data["ib_start"],
                    var_data["ib_end"],
                    var_data["ib_wait_minutes"],
                    var_data["ib_buffer_pct"],
                    var_data["max_distance_pct"],
                    var_data["trade_window_minutes"],
                    var_data["rr_target"],
                    var_data["stop_mode"],
                    var_data["tsl_target"],
                    var_data["tsl_sl"],
                ))

            key = tuple(params_key)

            if key not in seen_keys:
                seen_keys.add(key)

                # Create backtest-ready group
                backtest_group = {
                    "id": f"{symbol}_{len(unique_groups)+1:03d}",
                    "symbol": symbol,
                    "source_category": category,
                    "rank_in_category": i + 1,
                    "combined_total_r": group["combined_total_r"],
                    "weighted_sharpe": group["weighted_sharpe"],
                    "max_group_dd": group["max_group_dd"],
                    "total_trades": group["total_trades"],
                    "params": {}
                }

                # Convert variations to backtest params format
                # Strategy expects: "Reverse", "OCAE", "TCWE", "REV_RB"
                VAR_NAME_MAP = {
                    "ocae": "OCAE",
                    "tcwe": "TCWE",
                    "reverse": "Reverse",
                    "rev_rb": "REV_RB",
                }
                for var_name, var_data in group["variations"].items():
                    var_key = VAR_NAME_MAP.get(var_name, var_name.upper())
                    backtest_group["params"][var_key] = {
                        "IB_START": var_data["ib_start"],
                        "IB_END": var_data["ib_end"],
                        "IB_TZ": var_data["ib_timezone"],
                        "IB_WAIT": var_data["ib_wait_minutes"],
                        "TRADE_WINDOW": var_data["trade_window_minutes"],
                        "RR_TARGET": var_data["rr_target"],
                        "STOP_MODE": var_data["stop_mode"],
                        "TSL_TARGET": var_data["tsl_target"],
                        "TSL_SL": var_data["tsl_sl"],
                        "MIN_SL_PCT": MIN_SL_PCT[symbol],
                        "REV_RB_ENABLED": var_data["rev_rb_enabled"] if var_name == "rev_rb" else False,
                        "REV_RB_PCT": var_data["rev_rb_pct"] if var_name == "rev_rb" else 1.0,
                        "IB_BUFFER_PCT": var_data["ib_buffer_pct"],
                        "MAX_DISTANCE_PCT": var_data["max_distance_pct"],
                    }

                    # Add expected metrics for comparison
                    backtest_group["params"][var_key]["_expected"] = {
                        "total_r": var_data["total_r"],
                        "sharpe": var_data["sharpe"],
                        "winrate": var_data["winrate"],
                        "trades": var_data["trades"],
                        "max_dd": var_data["max_drawdown"],
                    }

                unique_groups.append(backtest_group)

    return unique_groups


def main():
    print("="*60)
    print("BACKTEST GROUPS GENERATOR")
    print(f"TOP {TOP_N} groups per ranking strategy")
    print(f"3 grouping modes x 4 ranking strategies = 12 categories")
    print("="*60)

    all_groups = {}

    for symbol in ["GER40", "XAUUSD"]:
        try:
            groups_by_category = generate_groups_for_symbol(symbol)
            unique_groups = deduplicate_groups(groups_by_category, symbol)

            all_groups[symbol] = unique_groups

            # Save to JSON
            output_path = ANALYZE_DIR / f"backtest_groups_{symbol}.json"
            with open(output_path, "w") as f:
                json.dump(unique_groups, f, indent=2)

            print(f"\n[INFO] {symbol}: {len(unique_groups)} unique groups saved to {output_path.name}")

        except Exception as e:
            print(f"[ERROR] Failed to process {symbol}: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for symbol, groups in all_groups.items():
        print(f"{symbol}: {len(groups)} unique groups")
    print(f"Total: {sum(len(g) for g in all_groups.values())} groups")

    # Print first few groups for verification
    for symbol, groups in all_groups.items():
        print(f"\n{symbol} first 3 groups:")
        for g in groups[:3]:
            print(f"  {g['id']}: R={g['combined_total_r']:.2f}, Sharpe={g['weighted_sharpe']:.2f}, "
                  f"Trades={g['total_trades']}, Source={g['source_category']}")


if __name__ == "__main__":
    main()
