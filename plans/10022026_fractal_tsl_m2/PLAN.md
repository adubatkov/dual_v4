# Plan: Fractal TSL -- 2-Minute Fractal Trailing Stop

## Context

Fractal BE (breakeven) logic is implemented and working (v3). It moves SL to entry when
an H1/H4 fractal is swept. This plan extends the concept: when an H1/H4 fractal is swept
during an open trade, start a **fractal-based trailing stop** using 2-minute fractals.
SL follows the latest confirmed M2 fractal of the protective type (low for LONG, high
for SHORT). This competes with TSL -- the more favorable SL wins.

**New parameter**: `FRACTAL_TSL_ENABLED` -- independent of `FRACTAL_BE_ENABLED`.
Both can be on/off independently. When both are on, all three SL sources compete:
`effective = max(TSL_organic, fractal_BE, fractal_TSL)` for LONG.

## Requirements

1. **Trigger**: Any H1/H4 fractal sweep during an open position (no SL zone check)
2. **M2 fractal detection**: Resample M1 -> 2min, detect Williams 3-bar fractals
3. **SL source**: For LONG -> last confirmed M2 low fractal price; for SHORT -> M2 high
4. **Trailing**: As new M2 fractals form, SL moves to their level
5. **Competition**: `effective = max(TSL_organic, fractal_BE, fractal_TSL)` for LONG,
   `min()` for SHORT -- most protective SL wins
6. **Parameter**: `FRACTAL_TSL_ENABLED` per-variation in strategy_logic.py
7. **Same-candle protection**: Handled by existing `effective_sl_prev` pattern

## Files Modified

1. **`backtest/run_backtest_template.py`** -- M2 detection, tracking, activation, SL computation
2. **`src/utils/strategy_logic.py`** -- `FRACTAL_TSL_ENABLED: True` in all 8 variations

## Key Design Decisions

### Incremental M2 pointer
Pre-compute all M2 fractals sorted by `confirmed_time`. Use `m2_fractal_ptr` to
incrementally advance as M1 candles iterate. Maintain `last_m2_high` and `last_m2_low`
as running values -- no backward scan needed.

### Three-way SL competition
```python
candidates = [organic]                    # TSL from ib_strategy
if frac_be is not None:
    candidates.append(frac_be)            # entry price from fractal BE
if frac_tsl_sl is not None:
    candidates.append(frac_tsl_sl)        # M2 fractal price
effective = max(candidates) if is_long else min(candidates)
```

### Independent activation
Fractal TSL activates on ANY H1/H4 fractal sweep, regardless of SL position.
Fractal BE still requires SL in negative zone. They are separate parameters.

## Verification

```bash
cd C:\Trading\ib_trading_bot\dual_v4
python backtest/run_backtest_template.py --symbol GER40 --params prod --start 2025-11-04 --end 2026-02-02 --output-name Fractal_TSL_v1
```

## SMC Code Reused

| Function | File | Purpose |
|----------|------|---------|
| `detect_fractals()` | `src/smc/detectors/fractal_detector.py:16` | Williams 3-bar fractal detection |
| `_resample_m1()` | `backtest/run_backtest_template.py:61` | M1 -> 2min resampling |
