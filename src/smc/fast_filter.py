"""
SMC Fast Filter - batch SMC pre-computation and signal filtering for fast backtest.

Designed for integration with params_optimizer/engine/fast_backtest.py.

Usage:
    from src.smc.fast_filter import build_smc_day_context, apply_smc_filter

    # In _process_day(), after IB computation, before signal detection:
    ctx = build_smc_day_context(day_df, lookback_df, instrument, smc_params)

    # After signal detection:
    filtered_signal = apply_smc_filter(signal, ctx, df_trade_m1)
"""
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd

from .config import SMCConfig
from .detectors.fractal_detector import detect_fractals, find_unswept_fractals
from .detectors.fvg_detector import detect_fvg, check_fvg_fill
from .detectors.market_structure_detector import detect_swing_points, detect_bos_choch
from .detectors.cisd_detector import detect_cisd
from .models import (
    BOS, CISD, FVG, Fractal, SMCDayContext,
)
from .timeframe_manager import TimeframeManager


def build_smc_day_context(
    day_m1: pd.DataFrame,
    lookback_m1: pd.DataFrame,
    instrument: str,
    smc_params: Optional[Dict[str, Any]] = None,
) -> SMCDayContext:
    """
    Pre-compute all SMC structures for a single trading day.

    Called once per day BEFORE signal detection.

    Args:
        day_m1: M1 data for current day
        lookback_m1: M1 data for lookback period (e.g., prior 48h for H1 fractals)
        instrument: "GER40" / "XAUUSD"
        smc_params: Optional dict with keys:
            - enable_fractals: bool (default True)
            - enable_fvg: bool (default True)
            - enable_bos: bool (default False)
            - enable_cisd: bool (default False)
            - fractal_timeframe: str (default "H1")
            - fvg_min_size_points: float (default 0.0)

    Returns:
        SMCDayContext with all pre-computed structures
    """
    params = smc_params or {}
    enable_fractals = params.get("enable_fractals", True)
    enable_fvg = params.get("enable_fvg", True)
    enable_bos = params.get("enable_bos", False)
    enable_cisd = params.get("enable_cisd", False)
    fractal_tf = params.get("fractal_timeframe", "H1")
    fvg_min_size = params.get("fvg_min_size_points", 0.0)

    day_date = None
    if not day_m1.empty:
        t = day_m1["time"].iloc[0]
        day_date = t.date() if hasattr(t, "date") else t

    ctx = SMCDayContext(
        instrument=instrument,
        day_date=day_date,
        build_time=datetime.utcnow(),
    )

    # Combine lookback + day for multi-day analysis
    if not lookback_m1.empty:
        combined = pd.concat([lookback_m1, day_m1], ignore_index=True)
        combined = combined.drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)
    else:
        combined = day_m1.copy()

    if len(combined) < 3:
        return ctx

    # Build TimeframeManager for resampling
    tfm = TimeframeManager(combined, instrument)

    # 1. Fractals (H1)
    if enable_fractals:
        h1_data = tfm.get_data(fractal_tf)
        if len(h1_data) >= 3:
            candle_hours = tfm.get_candle_duration_hours(fractal_tf)
            fractals = detect_fractals(h1_data, instrument, fractal_tf, candle_hours)

            for f in fractals:
                if f.type == "high":
                    ctx.fractals_high.append(f)
                    if not f.swept:
                        ctx.active_fractal_highs.append(f.price)
                else:
                    ctx.fractals_low.append(f)
                    if not f.swept:
                        ctx.active_fractal_lows.append(f.price)

            ctx.num_fractals = len(fractals)

    # Sort for binary search
    ctx.active_fractal_highs.sort()
    ctx.active_fractal_lows.sort()

    # 2. FVGs (M2 and H1)
    if enable_fvg:
        for tf in ["M2", "H1"]:
            tf_data = tfm.get_data(tf)
            if len(tf_data) >= 3:
                fvgs = detect_fvg(tf_data, instrument, tf, min_size_points=fvg_min_size)
                active_fvgs = [f for f in fvgs if f.status == "active"]

                if tf == "M2":
                    ctx.fvgs_m2 = active_fvgs
                else:
                    ctx.fvgs_h1 = active_fvgs

                for fvg in active_fvgs:
                    ctx.active_fvg_zones.append({
                        "high": fvg.high,
                        "low": fvg.low,
                        "midpoint": fvg.midpoint,
                        "direction": fvg.direction,
                        "timeframe": tf,
                    })

        ctx.num_fvgs = len(ctx.fvgs_m2) + len(ctx.fvgs_h1)

    # 3. BOS/CHoCH (M5, M15)
    if enable_bos:
        for tf in ["M5", "M15"]:
            tf_data = tfm.get_data(tf)
            if len(tf_data) >= 10:
                ms = detect_swing_points(tf_data, instrument, tf, lookback=3)
                bos_list = detect_bos_choch(ms, tf_data, instrument, tf)
                ctx.bos_events.extend(bos_list)

        ctx.num_bos = len(ctx.bos_events)

    # 4. CISD (M2)
    if enable_cisd and ctx.active_fvg_zones:
        m2_data = tfm.get_data("M2")
        if len(m2_data) >= 5:
            # Build POI zones from FVGs
            poi_zones = [
                {
                    "type": "fvg",
                    "direction": z["direction"],
                    "high": z["high"],
                    "low": z["low"],
                    "time": datetime.min,
                }
                for z in ctx.active_fvg_zones
            ]

            cisds = detect_cisd(m2_data, instrument, "M2", poi_zones)
            ctx.cisd_events = cisds
            ctx.num_cisds = len(cisds)

    return ctx


def apply_smc_filter(
    signal: Any,
    ctx: SMCDayContext,
    df_trade_m1: pd.DataFrame,
    smc_params: Optional[Dict[str, Any]] = None,
) -> Optional[Any]:
    """
    Apply SMC filtering to a detected signal.

    Checks for conflicts between signal and active SMC structures.
    If conflict found, scans forward for confirmation. If no confirmation,
    rejects the signal.

    Args:
        signal: Signal object (must have .direction, .entry_price, .entry_time, .stop_price, .extra)
        ctx: Pre-computed SMCDayContext for this day
        df_trade_m1: M1 data for trade window (for forward scanning)
        smc_params: Optional config dict with:
            - smc_confirmation_max_bars: int (default 30)

    Returns:
        Modified signal if passed/confirmed, None if rejected
    """
    params = smc_params or {}
    max_scan_bars = params.get("smc_confirmation_max_bars", 30)

    direction = signal.direction
    entry_price = signal.entry_price

    # 1. Check for opposing fractals
    has_conflict = False

    if direction == "long" and ctx.active_fractal_highs:
        # Bearish: unswept high fractals near/above entry
        opposing = [p for p in ctx.active_fractal_highs if p <= entry_price * 1.005]
        if opposing:
            has_conflict = True
            signal.extra["smc_conflict"] = "opposing_high_fractal"

    elif direction == "short" and ctx.active_fractal_lows:
        # Bullish: unswept low fractals near/below entry
        opposing = [p for p in ctx.active_fractal_lows if p >= entry_price * 0.995]
        if opposing:
            has_conflict = True
            signal.extra["smc_conflict"] = "opposing_low_fractal"

    # 2. Check for FVGs opposing signal direction
    if not has_conflict:
        for zone in ctx.active_fvg_zones:
            if zone["low"] <= entry_price <= zone["high"]:
                # Entry inside FVG
                if direction == "long" and zone["direction"] == "bearish":
                    has_conflict = True
                    signal.extra["smc_conflict"] = "bearish_fvg_at_entry"
                    break
                elif direction == "short" and zone["direction"] == "bullish":
                    has_conflict = True
                    signal.extra["smc_conflict"] = "bullish_fvg_at_entry"
                    break

    # 3. If no conflict, pass signal through
    if not has_conflict:
        signal.extra["smc_checked"] = True
        signal.extra["smc_result"] = "pass"
        return signal

    # 4. Conflict found: scan forward for confirmation
    confirmation = _scan_for_confirmation(
        signal, ctx, df_trade_m1, max_scan_bars
    )

    if confirmation is not None:
        # Modify signal with confirmed entry
        signal.entry_price = confirmation["entry_price"]
        signal.stop_price = confirmation["stop_price"]
        signal.entry_time = confirmation["entry_time"]
        signal.extra["smc_confirmed"] = True
        signal.extra["smc_confirmation_type"] = confirmation["type"]
        signal.extra["smc_result"] = "confirmed"
        return signal

    # 5. No confirmation: reject signal
    signal.extra["smc_result"] = "rejected"
    return None


def _scan_for_confirmation(
    signal: Any,
    ctx: SMCDayContext,
    df_trade_m1: pd.DataFrame,
    max_bars: int = 30,
) -> Optional[Dict[str, Any]]:
    """
    Scan forward in M1 data for SMC confirmation after conflict.

    Checks for:
    - FVG rebalance (price enters FVG, closes beyond midpoint)
    - Fractal sweep + reversal (wick beyond fractal, close back)
    - CISD pattern at nearby level

    Returns confirmation dict or None.
    """
    if signal.entry_time is None or df_trade_m1.empty:
        return None

    # Find start index
    mask = df_trade_m1["time"] >= signal.entry_time
    if not mask.any():
        return None

    start_idx = mask.idxmax()
    end_idx = min(start_idx + max_bars, len(df_trade_m1) - 1)

    direction = signal.direction

    for i in range(start_idx + 1, end_idx + 1):
        bar = df_trade_m1.iloc[i]

        # Early termination: price moved too far against signal
        if direction == "long" and bar["low"] < signal.stop_price * 0.995:
            return None
        if direction == "short" and bar["high"] > signal.stop_price * 1.005:
            return None

        # Check FVG rebalance
        for zone in ctx.active_fvg_zones:
            bar_in_fvg = bar["low"] <= zone["high"] and bar["high"] >= zone["low"]
            if not bar_in_fvg:
                continue

            if direction == "long" and zone["direction"] == "bullish":
                if bar["low"] <= zone["midpoint"] and bar["close"] > zone["midpoint"]:
                    return {
                        "type": "FVG_REBALANCE",
                        "entry_price": float(bar["close"]),
                        "stop_price": float(zone["low"]) * 0.999,
                        "entry_time": bar["time"],
                    }

            elif direction == "short" and zone["direction"] == "bearish":
                if bar["high"] >= zone["midpoint"] and bar["close"] < zone["midpoint"]:
                    return {
                        "type": "FVG_REBALANCE",
                        "entry_price": float(bar["close"]),
                        "stop_price": float(zone["high"]) * 1.001,
                        "entry_time": bar["time"],
                    }

        # Check fractal sweep + reversal
        if direction == "long":
            for frac_price in ctx.active_fractal_lows:
                if bar["low"] < frac_price * 0.999 and bar["close"] > frac_price:
                    return {
                        "type": "FRACTAL_SWEEP",
                        "entry_price": float(bar["close"]),
                        "stop_price": float(bar["low"]) * 0.998,
                        "entry_time": bar["time"],
                    }

        elif direction == "short":
            for frac_price in ctx.active_fractal_highs:
                if bar["high"] > frac_price * 1.001 and bar["close"] < frac_price:
                    return {
                        "type": "FRACTAL_SWEEP",
                        "entry_price": float(bar["close"]),
                        "stop_price": float(bar["high"]) * 1.002,
                        "entry_time": bar["time"],
                    }

    return None
