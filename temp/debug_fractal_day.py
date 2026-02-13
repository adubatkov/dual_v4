#!/usr/bin/env python3
"""Debug fractal state for a specific trade day in the fast engine."""

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
from src.smc.detectors.fractal_detector import detect_fractals
from temp.compare_engines import prod_to_flat


def load_m1(symbol, start, end):
    df = pd.read_parquet(DATA_FILES[symbol])
    if not pd.api.types.is_datetime64_any_dtype(df["time"]):
        df["time"] = pd.to_datetime(df["time"], utc=True)
    if df["time"].dt.tz is None:
        df["time"] = df["time"].dt.tz_localize("UTC")
    df = df[(df["time"] >= start) & (df["time"] < end)].copy()
    return df.sort_values("time").reset_index(drop=True)


def main():
    target_date_str = sys.argv[1] if len(sys.argv) > 1 else "2025-10-28"
    target_date = datetime.strptime(target_date_str, "%Y-%m-%d").date()

    symbol = "GER40"
    params = deepcopy(GER40_PARAMS_PROD)
    for var_name in ["OCAE", "TCWE", "Reverse", "REV_RB"]:
        if var_name in params:
            params[var_name]["FRACTAL_BE_ENABLED"] = True
            params[var_name]["FRACTAL_TSL_ENABLED"] = True

    flat = prod_to_flat(params)

    # Load same date range as October comparison
    start = datetime(2025, 10, 1, tzinfo=pytz.UTC)
    end = datetime(2025, 11, 1, tzinfo=pytz.UTC)
    m1 = load_m1(symbol, start, end)
    print(f"Loaded {len(m1)} M1 candles")

    engine = FastBacktest(symbol, m1)

    # Manually run fractal pre-computation to inspect state
    h1_data = engine._resample_full("1h")
    h4_data = engine._resample_full("4h")
    m2_data = engine._resample_full("2min")

    h1_fractals = detect_fractals(h1_data, symbol, "H1", candle_duration_hours=1.0)
    h4_fractals = detect_fractals(h4_data, symbol, "H4", candle_duration_hours=4.0)
    m2_fractals = detect_fractals(m2_data, symbol, "M2", candle_duration_hours=2 / 60)

    h4_keys = {(f.type, round(f.price, 2)) for f in h4_fractals}
    filtered_h1 = [f for f in h1_fractals if (f.type, round(f.price, 2)) not in h4_keys]
    h1h4_sorted = sorted(filtered_h1 + h4_fractals, key=lambda f: f.confirmed_time)

    print(f"\nTotal H1/H4 fractals: {len(h1h4_sorted)} (H1: {len(filtered_h1)}, H4: {len(h4_fractals)})")

    # Entry time is approximately 2025-10-28 08:02 UTC (OCAE short)
    tz = pytz.timezone(flat["ib_timezone"])
    entry_approx = pd.Timestamp("2025-10-28 06:02:00", tz="UTC")  # UTC (Berlin is UTC+2 in Oct)
    # Actually let's check the IB window for this day
    ib_start_str = flat["ib_start"]  # e.g. "08:00"
    print(f"IB timezone: {flat['ib_timezone']}, IB start: {ib_start_str}, IB end: {flat['ib_end']}")

    # For Berlin in Oct (CEST ends late Oct), offset is UTC+2 (summer) or UTC+1 (winter)
    # Oct 28 2025 is after DST change (Oct 26 2025), so UTC+1
    # IB start 08:00 Berlin = 07:00 UTC
    # Trade window starts after IB end (08:30 Berlin = 07:30 UTC) + wait

    # Let me just look at which fractals would be active near entry
    entry_approx = pd.Timestamp("2025-10-28 07:02:00", tz="UTC")

    print(f"\nFractals confirmed before ~{entry_approx} and not yet expired:")
    expiry_h1 = entry_approx - timedelta(hours=48)
    expiry_h4 = entry_approx - timedelta(hours=96)

    for f in h1h4_sorted:
        if f.confirmed_time > entry_approx:
            break
        expired = False
        if f.timeframe == "H1" and f.time < expiry_h1:
            expired = True
        if f.timeframe == "H4" and f.time < expiry_h4:
            expired = True
        if expired:
            continue
        print(f"  {f.timeframe:3s} {f.type:5s} price={f.price:.1f} "
              f"time={f.time} confirmed={f.confirmed_time} "
              f"swept={f.swept} sweep_time={f.sweep_time}")

    # Now run pre-sweep
    print("\n--- Running pre-sweep ---")
    from copy import deepcopy as dc
    h1h4_copy = []
    for f in h1h4_sorted:
        # Reset sweep state since detect_fractals returns clean fractals
        f.swept = False
        f.sweep_time = None

    engine._presweep_fractals(h1h4_sorted)

    print(f"\nFractals confirmed before ~{entry_approx}, not expired, AFTER pre-sweep:")
    active_count = 0
    presswept_count = 0
    for f in h1h4_sorted:
        if f.confirmed_time > entry_approx:
            break
        expired = False
        if f.timeframe == "H1" and f.time < expiry_h1:
            expired = True
        if f.timeframe == "H4" and f.time < expiry_h4:
            expired = True
        if expired:
            continue

        if f.swept and f.sweep_time <= entry_approx:
            presswept_count += 1
            # Show pre-swept fractals near the entry price range (within 200 points)
            if abs(f.price - 19500) < 500:  # approximate GER40 level in Oct 2025
                print(f"  [PRE-SWEPT] {f.timeframe:3s} {f.type:5s} price={f.price:.1f} "
                      f"time={f.time} sweep_time={f.sweep_time}")
        else:
            active_count += 1
            print(f"  [ACTIVE]    {f.timeframe:3s} {f.type:5s} price={f.price:.1f} "
                  f"time={f.time} confirmed={f.confirmed_time}")

    print(f"\n  Active: {active_count}, Pre-swept: {presswept_count}")

    # Now let's see what the slow engine does. Let's check if the slow engine had a fractal
    # that triggered BE. The slow engine entry is 08:02 and exit is 08:03 at R=0.00.
    # This means: a fractal was swept on the 08:03 candle (or the entry candle itself),
    # the SL was negative (short, so SL > entry price), fractal_be moved SL to entry_price,
    # and then the exit happened.

    # Let me check the actual entry/IB data for this day
    print("\n--- Running full engine to get trade details ---")
    flat["fractal_be_enabled"] = True
    flat["fractal_tsl_enabled"] = True
    result = engine.run_with_params(flat)
    trades = result.get("trades_detail", [])

    for t in trades:
        entry_time = t.get("entry_time")
        if hasattr(entry_time, "strftime"):
            edate = entry_time.strftime("%Y-%m-%d")
        elif hasattr(t.get("date"), "strftime"):
            edate = t["date"].strftime("%Y-%m-%d")
        else:
            edate = str(t.get("date", ""))

        if edate == target_date_str:
            print(f"\n  FAST trade on {edate}:")
            for k, v in t.items():
                print(f"    {k}: {v}")


if __name__ == "__main__":
    main()
