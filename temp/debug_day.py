#!/usr/bin/env python3
"""Debug a specific day's signal detection in both engines."""

import sys
from pathlib import Path
from datetime import datetime, date, timedelta
import pandas as pd
import pytz
from copy import deepcopy

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from backtest.config import DATA_FILES
from params_optimizer.engine.fast_backtest import FastBacktest
from src.utils.strategy_logic import GER40_PARAMS_PROD


def load_m1(symbol, start, end):
    df = pd.read_parquet(DATA_FILES[symbol])
    if not pd.api.types.is_datetime64_any_dtype(df["time"]):
        df["time"] = pd.to_datetime(df["time"], utc=True)
    if df["time"].dt.tz is None:
        df["time"] = df["time"].dt.tz_localize("UTC")
    df = df[(df["time"] >= start) & (df["time"] < end)].copy()
    return df.sort_values("time").reset_index(drop=True)


def debug_fast_day(symbol, target_date_str, params_nested):
    """Debug fast engine signal detection for a specific day."""
    target_date = datetime.strptime(target_date_str, "%Y-%m-%d").date()

    # Load 3 days of data (day before, target day, day after)
    start = datetime(target_date.year, target_date.month, target_date.day, tzinfo=pytz.UTC) - timedelta(days=1)
    end = start + timedelta(days=3)
    m1 = load_m1(symbol, start, end)

    # Flat params
    v = params_nested["Reverse"]
    flat = {
        "ib_start": v["IB_START"], "ib_end": v["IB_END"],
        "ib_timezone": v["IB_TZ"], "ib_wait_minutes": v["IB_WAIT"],
        "trade_window_minutes": v["TRADE_WINDOW"], "rr_target": v["RR_TARGET"],
        "stop_mode": v.get("STOP_MODE", "ib_start"),
        "tsl_target": v["TSL_TARGET"], "tsl_sl": v["TSL_SL"],
        "min_sl_pct": v["MIN_SL_PCT"],
        "ib_buffer_pct": v.get("IB_BUFFER_PCT", 0.0),
        "max_distance_pct": v.get("MAX_DISTANCE_PCT", 1.0),
        "rev_rb_enabled": False,
    }
    # Per-variation buffer overrides
    ocae = params_nested.get("OCAE", {})
    flat["ocae_ib_buffer_pct"] = ocae.get("IB_BUFFER_PCT", flat["ib_buffer_pct"])
    flat["ocae_max_distance_pct"] = ocae.get("MAX_DISTANCE_PCT", flat["max_distance_pct"])
    tcwe = params_nested.get("TCWE", {})
    flat["tcwe_ib_buffer_pct"] = tcwe.get("IB_BUFFER_PCT", flat["ib_buffer_pct"])
    flat["tcwe_max_distance_pct"] = tcwe.get("MAX_DISTANCE_PCT", flat["max_distance_pct"])

    engine = FastBacktest(symbol, m1)

    # Compute IB date column
    tz = pytz.timezone(flat["ib_timezone"])
    engine.m1_data["ib_date"] = engine.m1_data["time"].apply(lambda x: x.astimezone(tz).date())

    # Get target day data
    day_df = engine.m1_data[engine.m1_data["ib_date"] == target_date].copy()
    day_df = day_df.sort_values("time").reset_index(drop=True)

    if day_df.empty:
        print(f"No data for {target_date}")
        return

    # Get IB
    ib = engine._compute_ib(day_df, target_date, flat)
    if not ib:
        print(f"No IB for {target_date}")
        return

    ibh, ibl, eq = ib["IBH"], ib["IBL"], ib["EQ"]
    print(f"\nDay: {target_date} | IBH={ibh:.1f} IBL={ibl:.1f} EQ={eq:.1f}")
    print(f"  IB Range: {ibh - ibl:.1f} points")

    # Get trade window M1
    df_trade_m1 = engine._get_trade_window(day_df, target_date, flat)
    if df_trade_m1.empty:
        print("  No trade window data")
        return

    print(f"  Trade window: {df_trade_m1['time'].iat[0]} to {df_trade_m1['time'].iat[-1]} ({len(df_trade_m1)} M1 candles)")

    # Resample to M2
    df_trade_m2 = engine._resample_m1_to_m2(df_trade_m1)
    print(f"  M2 candles: {len(df_trade_m2)}")

    # Get pre-context
    ib_start_ts, ib_end_ts = engine._ib_window_on_date(
        target_date, flat["ib_start"], flat["ib_end"], flat["ib_timezone"]
    )
    first_trade_ts = df_trade_m1["time"].iat[0]
    df_pre_m1 = day_df[(day_df["time"] >= ib_start_ts) & (day_df["time"] < first_trade_ts)][
        ["time", "open", "high", "low", "close"]
    ].copy()
    df_pre_m2 = engine._resample_m1_to_m2(df_pre_m1) if not df_pre_m1.empty else pd.DataFrame()

    print(f"  Pre-context M2: {len(df_pre_m2)} candles")

    # Print M2 candles with IB annotations
    print(f"\n  {'idx':<4} {'Time':<20} {'Open':>10} {'High':>10} {'Low':>10} {'Close':>10} {'Flags'}")
    print(f"  {'---':<4} {'----':<20} {'----':>10} {'----':>10} {'---':>10} {'-----':>10} {'-----'}")

    for idx, row in df_trade_m2.iterrows():
        t = row["time"]
        o, h, l, c = row["open"], row["high"], row["low"], row["close"]
        flags = []
        if h > ibh:
            flags.append(f"BREAK_UP(h>{ibh:.1f})")
        if l < ibl:
            flags.append(f"BREAK_DN(l<{ibl:.1f})")
        if l <= eq <= h:
            flags.append("EQ_TOUCH")
        flag_str = " ".join(flags)
        print(f"  {idx:<4} {str(t):<20} {o:>10.1f} {h:>10.1f} {l:>10.1f} {c:>10.1f} {flag_str}")

    # Check signals
    trade_start_price = float(df_trade_m2["open"].iat[0])

    rev_sig = engine._check_reverse(df_trade_m2, df_pre_m2, ibh, ibl, eq, flat)
    ocae_sig = engine._check_ocae(df_trade_m2, ibh, ibl, eq, trade_start_price, flat)
    tcwe_sig = engine._check_tcwe(df_trade_m2, ibh, ibl, eq, trade_start_price, flat)

    print(f"\n  Signals:")
    print(f"    Reverse: {rev_sig.signal_type + ' ' + rev_sig.direction + ' idx=' + str(rev_sig.entry_idx) + ' t=' + str(rev_sig.entry_time) if rev_sig else 'None'}")
    print(f"    OCAE:    {ocae_sig.signal_type + ' ' + ocae_sig.direction + ' idx=' + str(ocae_sig.entry_idx) + ' t=' + str(ocae_sig.entry_time) if ocae_sig else 'None'}")
    print(f"    TCWE:    {tcwe_sig.signal_type + ' ' + tcwe_sig.direction + ' idx=' + str(tcwe_sig.entry_idx) + ' t=' + str(tcwe_sig.entry_time) if tcwe_sig else 'None'}")

    # Priority selection
    primary = []
    if rev_sig:
        primary.append(rev_sig)
    if ocae_sig:
        primary.append(ocae_sig)
    if tcwe_sig:
        primary.append(tcwe_sig)

    if primary:
        priority = {"Reverse": 0, "OCAE": 1, "TCWE": 2}
        winner = min(primary, key=lambda s: (s.entry_idx, priority.get(s.signal_type, 99)))
        print(f"\n  Winner: {winner.signal_type} {winner.direction} at idx={winner.entry_idx}")
    else:
        print(f"\n  No primary signal")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python temp/debug_day.py YYYY-MM-DD [YYYY-MM-DD ...]")
        sys.exit(1)

    params = deepcopy(GER40_PARAMS_PROD)
    # Disable fractal features for baseline
    for var_name in ["OCAE", "TCWE", "Reverse", "REV_RB"]:
        if var_name in params:
            params[var_name]["FRACTAL_BE_ENABLED"] = False
            params[var_name]["FRACTAL_TSL_ENABLED"] = False

    for day_str in sys.argv[1:]:
        debug_fast_day("GER40", day_str, params)
