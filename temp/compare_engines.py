#!/usr/bin/env python3
"""
Compare Fast vs Slow backtest engines trade-by-trade.

Usage:
    cd dual_v4
    python temp/compare_engines.py --symbol GER40 --start 2025-03-01 --end 2025-04-01

Runs both engines with identical params, matches trades, reports divergences.
"""

import sys
import os
import logging
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from copy import deepcopy

import pandas as pd
import pytz

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from backtest.config import BacktestConfig, DATA_FILES
from backtest.run_backtest_template import run_backtest
from params_optimizer.engine.fast_backtest import FastBacktest
from src.utils.strategy_logic import GER40_PARAMS_PROD, XAUUSD_PARAMS_PROD
from src.news_filter import NewsFilter

logger = logging.getLogger(__name__)


def prod_to_flat(nested: dict, variation: str = "Reverse") -> dict:
    """Convert nested PROD params to flat fast-engine params.

    Extracts base params from `variation` (default Reverse) and adds
    per-variation overrides for OCAE/TCWE buffer/distance params since
    these may differ from Reverse.
    """
    v = nested[variation]
    flat = {
        "ib_start": v["IB_START"],
        "ib_end": v["IB_END"],
        "ib_timezone": v["IB_TZ"],
        "ib_wait_minutes": v["IB_WAIT"],
        "trade_window_minutes": v["TRADE_WINDOW"],
        "rr_target": v["RR_TARGET"],
        "stop_mode": v.get("STOP_MODE", "ib_start"),
        "tsl_target": v["TSL_TARGET"],
        "tsl_sl": v["TSL_SL"],
        "min_sl_pct": v["MIN_SL_PCT"],
        "ib_buffer_pct": v.get("IB_BUFFER_PCT", 0.0),
        "max_distance_pct": v.get("MAX_DISTANCE_PCT", 1.0),
        "rev_rb_enabled": nested.get("REV_RB", {}).get("REV_RB_ENABLED", False),
        "fractal_be_enabled": v.get("FRACTAL_BE_ENABLED", False),
        "fractal_tsl_enabled": v.get("FRACTAL_TSL_ENABLED", False),
        "news_skip_events": nested.get("NEWS_SKIP_EVENTS", []),
    }

    # Per-variation buffer/distance overrides (OCAE and TCWE may differ from Reverse)
    ocae = nested.get("OCAE", {})
    flat["ocae_ib_buffer_pct"] = ocae.get("IB_BUFFER_PCT", flat["ib_buffer_pct"])
    flat["ocae_max_distance_pct"] = ocae.get("MAX_DISTANCE_PCT", flat["max_distance_pct"])

    tcwe = nested.get("TCWE", {})
    flat["tcwe_ib_buffer_pct"] = tcwe.get("IB_BUFFER_PCT", flat["ib_buffer_pct"])
    flat["tcwe_max_distance_pct"] = tcwe.get("MAX_DISTANCE_PCT", flat["max_distance_pct"])

    return flat


def load_m1_data(symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """Load M1 parquet data for given symbol and date range."""
    parquet_path = DATA_FILES.get(symbol)
    if parquet_path is None or not parquet_path.exists():
        raise FileNotFoundError(f"No data file for {symbol}: {parquet_path}")

    df = pd.read_parquet(parquet_path)

    # Ensure time column is datetime UTC
    if not pd.api.types.is_datetime64_any_dtype(df["time"]):
        df["time"] = pd.to_datetime(df["time"], utc=True)
    if df["time"].dt.tz is None:
        df["time"] = df["time"].dt.tz_localize("UTC")

    # Filter date range
    df = df[(df["time"] >= start_date) & (df["time"] < end_date)].copy()
    df = df.sort_values("time").reset_index(drop=True)

    return df


def run_slow_engine(symbol: str, params: dict, start_date: datetime, end_date: datetime):
    """Run slow backtest engine, return trade log."""
    trade_log = run_backtest(
        symbol=symbol,
        params=params,
        params_label="compare_slow",
        start_date=start_date,
        end_date=end_date,
        risk_mode="fixed",
        risk_value=1000.0,
        initial_balance=100000.0,
        generate_charts=False,
        output_name="compare_slow",
    )
    return trade_log or []


def run_fast_engine(symbol: str, m1_data: pd.DataFrame, flat_params: dict, news_filter=None):
    """Run fast backtest engine, return trades detail list."""
    engine = FastBacktest(
        symbol=symbol,
        m1_data=m1_data,
        news_filter=news_filter,
    )
    result = engine.run_with_params(flat_params)
    return result.get("trades_detail", [])


def normalize_time(t):
    """Convert various time formats to comparable string."""
    if t is None:
        return ""
    if isinstance(t, str):
        return t
    if hasattr(t, "strftime"):
        return t.strftime("%Y-%m-%d %H:%M")
    return str(t)


def match_trades(slow_trades, fast_trades):
    """Match trades between engines by date + variation + direction."""
    # Build slow trade list
    slow_list = []
    for t in slow_trades:
        if not t.exit_time:
            continue
        entry_date = t.entry_time.strftime("%Y-%m-%d") if t.entry_time else ""
        initial_risk = abs(t.entry_price - t.sl) if t.sl else 0
        if initial_risk > 0:
            if t.direction and t.direction.lower() == "long":
                r_val = (t.exit_price - t.entry_price) / initial_risk
            else:
                r_val = (t.entry_price - t.exit_price) / initial_risk
        else:
            r_val = 0
        slow_list.append({
            "date": entry_date,
            "variation": t.variation or "",
            "direction": (t.direction or "").lower(),
            "entry_time": normalize_time(t.entry_time),
            "exit_time": normalize_time(t.exit_time),
            "entry_price": t.entry_price,
            "exit_price": t.exit_price,
            "exit_reason": t.exit_reason or "",
            "sl": t.sl,
            "tp": t.tp,
            "R": round(r_val, 4),
        })

    # Build fast trade list
    fast_list = []
    for t in fast_trades:
        entry_time = t.get("entry_time")
        if hasattr(entry_time, "strftime"):
            entry_date = entry_time.strftime("%Y-%m-%d")
        elif hasattr(t.get("date"), "strftime"):
            entry_date = t["date"].strftime("%Y-%m-%d")
        else:
            entry_date = str(t.get("date", ""))
        fast_list.append({
            "date": entry_date,
            "variation": t.get("variation", ""),
            "direction": t.get("direction", "").lower(),
            "entry_time": normalize_time(t.get("entry_time")),
            "exit_time": normalize_time(t.get("exit_time")),
            "entry_price": t.get("entry_price", 0),
            "exit_price": t.get("exit_price", 0),
            "exit_reason": t.get("exit_reason", ""),
            "sl": t.get("stop", 0),
            "tp": t.get("tp", 0),
            "R": round(t.get("R", 0), 4),
        })

    # Match by date + variation + direction
    matched = []
    fast_used = set()

    for si, s in enumerate(slow_list):
        key = (s["date"], s["variation"], s["direction"])
        for fi, f in enumerate(fast_list):
            if fi in fast_used:
                continue
            fkey = (f["date"], f["variation"], f["direction"])
            if key == fkey:
                matched.append({"slow": s, "fast": f})
                fast_used.add(fi)
                break
        else:
            matched.append({"slow": s, "fast": None})

    # Unmatched fast trades
    for fi, f in enumerate(fast_list):
        if fi not in fast_used:
            matched.append({"slow": None, "fast": f})

    return matched


def print_comparison(matches):
    """Print comparison table."""
    header = (
        f"{'Date':<12} {'Var':<9} {'Dir':<6} "
        f"{'Entry_S':<18} {'Entry_F':<18} "
        f"{'Exit_S':<18} {'Exit_F':<18} "
        f"{'ExR_S':<6} {'ExR_F':<6} "
        f"{'R_S':>7} {'R_F':>7} {'dR':>7} {'Status':<10}"
    )
    print("\n" + "=" * len(header))
    print("TRADE-BY-TRADE COMPARISON: SLOW vs FAST")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    total_matched = 0
    total_r_delta = 0.0
    r_deltas = []
    slow_only = 0
    fast_only = 0
    r_mismatch = 0

    for m in matches:
        s = m["slow"]
        f = m["fast"]

        if s is None:
            fast_only += 1
            date = f["date"]
            var = f["variation"]
            dir_ = f["direction"]
            print(f"{date:<12} {var:<9} {dir_:<6} "
                  f"{'---':<18} {f['entry_time']:<18} "
                  f"{'---':<18} {f['exit_time']:<18} "
                  f"{'---':<6} {f['exit_reason']:<6} "
                  f"{'---':>7} {f['R']:>7.2f} {'---':>7} {'FAST_ONLY':<10}")
            continue

        if f is None:
            slow_only += 1
            date = s["date"]
            var = s["variation"]
            dir_ = s["direction"]
            print(f"{date:<12} {var:<9} {dir_:<6} "
                  f"{s['entry_time']:<18} {'---':<18} "
                  f"{s['exit_time']:<18} {'---':<18} "
                  f"{s['exit_reason']:<6} {'---':<6} "
                  f"{s['R']:>7.2f} {'---':>7} {'---':>7} {'SLOW_ONLY':<10}")
            continue

        total_matched += 1
        delta_r = f["R"] - s["R"]
        r_deltas.append(abs(delta_r))
        total_r_delta += delta_r

        # Determine status
        if abs(delta_r) <= 0.001:
            status = "OK"
        elif abs(delta_r) <= abs(s["R"]) * 0.10 + 0.05:
            status = "OK (~)"
        else:
            status = "MISMATCH"
            r_mismatch += 1

        date = s["date"]
        var = s["variation"]
        dir_ = s["direction"]

        print(f"{date:<12} {var:<9} {dir_:<6} "
              f"{s['entry_time']:<18} {f['entry_time']:<18} "
              f"{s['exit_time']:<18} {f['exit_time']:<18} "
              f"{s['exit_reason']:<6} {f['exit_reason']:<6} "
              f"{s['R']:>7.2f} {f['R']:>7.2f} {delta_r:>+7.2f} {status:<10}")

    # Summary
    print("-" * len(header))
    total_slow = sum(1 for m in matches if m["slow"] is not None)
    total_fast = sum(1 for m in matches if m["fast"] is not None)
    mean_abs_delta = sum(r_deltas) / len(r_deltas) if r_deltas else 0

    print(f"\nSUMMARY:")
    print(f"  Slow engine trades:  {total_slow}")
    print(f"  Fast engine trades:  {total_fast}")
    print(f"  Matched:             {total_matched}")
    print(f"  Slow-only:           {slow_only}")
    print(f"  Fast-only:           {fast_only}")
    print(f"  R mismatches (>10%): {r_mismatch}")
    print(f"  Mean |dR|:           {mean_abs_delta:.4f}")
    print(f"  Total dR:            {total_r_delta:+.4f}")

    # Pass/fail
    if slow_only == 0 and fast_only == 0 and r_mismatch == 0:
        print(f"\n  [PASS] All trades match within tolerance")
    else:
        print(f"\n  [FAIL] Divergences detected")

    return {
        "total_slow": total_slow,
        "total_fast": total_fast,
        "matched": total_matched,
        "slow_only": slow_only,
        "fast_only": fast_only,
        "r_mismatch": r_mismatch,
        "mean_abs_delta": mean_abs_delta,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Compare fast vs slow backtest engines")
    parser.add_argument("--symbol", type=str, default="GER40", help="Trading symbol")
    parser.add_argument("--start", type=str, required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--params", type=str, default="prod",
                        help="Param set: 'prod' or path to JSON")
    parser.add_argument("--fractal-be", action="store_true", default=False,
                        help="Enable FRACTAL_BE_ENABLED")
    parser.add_argument("--fractal-tsl", action="store_true", default=False,
                        help="Enable FRACTAL_TSL_ENABLED")
    parser.add_argument("--rr", type=float, default=None, help="Override RR_TARGET")
    parser.add_argument("--tsl-target", type=float, default=None, help="Override TSL_TARGET")
    parser.add_argument("--tsl-sl", type=float, default=None, help="Override TSL_SL")
    parser.add_argument("--stop-mode", type=str, default=None, help="Override STOP_MODE (ib_start|eq)")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    return parser.parse_args()


def main():
    args = parse_args()

    log_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")
    # Suppress noisy loggers
    logging.getLogger("backtest").setLevel(logging.WARNING)
    logging.getLogger("src").setLevel(logging.WARNING)

    start_date = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=pytz.UTC)
    end_date = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=pytz.UTC)

    # Load params
    PARAMS_MAP = {
        "GER40": GER40_PARAMS_PROD,
        "XAUUSD": XAUUSD_PARAMS_PROD,
    }
    nested_params = deepcopy(PARAMS_MAP.get(args.symbol, GER40_PARAMS_PROD))

    # Override fractal flags and param overrides
    for var_name in ["OCAE", "TCWE", "Reverse", "REV_RB"]:
        if var_name in nested_params:
            nested_params[var_name]["FRACTAL_BE_ENABLED"] = args.fractal_be
            nested_params[var_name]["FRACTAL_TSL_ENABLED"] = args.fractal_tsl
            if args.rr is not None:
                nested_params[var_name]["RR_TARGET"] = args.rr
            if args.tsl_target is not None:
                nested_params[var_name]["TSL_TARGET"] = args.tsl_target
            if args.tsl_sl is not None:
                nested_params[var_name]["TSL_SL"] = args.tsl_sl
            if args.stop_mode is not None:
                nested_params[var_name]["STOP_MODE"] = args.stop_mode

    rr_str = f"RR={args.rr}" if args.rr else "RR=PROD"
    tsl_str = f"TSL={args.tsl_target}/{args.tsl_sl}" if args.tsl_target is not None else "TSL=PROD"
    sm_str = f"STOP={args.stop_mode}" if args.stop_mode else "STOP=PROD"
    print(f"\nComparing engines: {args.symbol} | {args.start} to {args.end}")
    print(f"  FRACTAL_BE={args.fractal_be} | FRACTAL_TSL={args.fractal_tsl}")
    print(f"  {rr_str} | {tsl_str} | {sm_str}")
    print(f"  NEWS_SKIP={nested_params.get('NEWS_SKIP_EVENTS', [])}")

    # Load M1 data
    print("\nLoading M1 data...")
    m1_data = load_m1_data(args.symbol, start_date, end_date)
    print(f"  Loaded {len(m1_data)} M1 candles")

    # Initialize news filter
    news_filter = NewsFilter(symbol=args.symbol, preload_years=[2025])

    # Run slow engine
    print("\nRunning SLOW engine...")
    slow_trades = run_slow_engine(args.symbol, nested_params, start_date, end_date)
    print(f"  Slow: {len(slow_trades)} trades")

    # Convert to flat params and run fast engine
    flat_params = prod_to_flat(nested_params)
    print("\nRunning FAST engine...")
    fast_trades = run_fast_engine(args.symbol, m1_data, flat_params, news_filter)
    print(f"  Fast: {len(fast_trades)} trades")

    # Match and compare
    matches = match_trades(slow_trades, fast_trades)
    print_comparison(matches)


if __name__ == "__main__":
    main()
