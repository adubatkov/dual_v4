# Plan: Fractal-Based Breakeven (BE) Logic in Backtest

## Context

After implementing fractal chart overlay (commit `3f73a95`), the next step is to test
whether fractals can improve trade management. The idea: when price touches an unswept
fractal during an open trade, move SL to entry price (breakeven) -- but only if SL is
still in the "negative zone" (below entry for LONG, above entry for SHORT).

This protects trades that have already shown favorable price action (fractal touch =
structure interaction) without cutting them short via premature TSL.

## Requirements

1. **Fractal BE trigger**: When an M1 candle touches an unswept H1/H4 fractal during an
   open position, move SL to entry price if SL is in negative zone
2. **TSL/BE decoupling**: TSL must track its own organic SL independently from fractal BE
   overlay to prevent contamination (TSL using entry as base -> oversized jumps)
3. **Lookback expiry**: H1 fractals expire after 48h, H4 after 96h (match chart overlay)
4. **Configurable toggle**: `FRACTAL_BE_ENABLED` per-variation parameter in strategy_logic.py
5. **Same-candle protection**: SL modifications (TSL or fractal BE) only take effect from
   the NEXT candle, not the current one (prevents false BE exits on entry candle)

## Key Design Decisions

### tsl_organic_sl pattern
Track TSL's own SL progression separately. Before TSL runs each candle: restore organic SL
(undo fractal BE overlay). After TSL: capture organic. Effective SL = max(organic, fractal_be)
for LONG, min() for SHORT.

### effective_sl_prev pattern
Capture `position.sl` at start of candle (before any modifications). When passing to
`_check_sltp_hits`, only override tickets where SL changed THIS candle. For unchanged
tickets, `position.sl` (effective SL) is used directly. This prevents same-candle exits.

### Incremental fractal tracking
Pre-compute all H1/H4 fractals sorted by `confirmed_time`. Activate incrementally as time
passes. Remove on sweep. This avoids per-candle lookback queries.

## Files Modified

1. **`backtest/run_backtest_template.py`** -- Main changes:
   - Pre-loop: fractal detection, sorting, deduplication
   - Main loop: TSL/BE decoupling, fractal activation/expiry, sweep check, BE trigger,
     effective SL computation, same-candle protection via `effective_sl_prev`
   - Post-loop: fractal BE summary log

2. **`src/utils/strategy_logic.py`** -- Added `FRACTAL_BE_ENABLED: True` to all 8 variations
   (4 in GER40_PARAMS_PROD, 4 in XAUUSD_PARAMS_PROD)

## Verification

```bash
cd C:\Trading\ib_trading_bot\dual_v4
python backtest/run_backtest_template.py --symbol GER40 --params prod --start 2025-11-04 --end 2026-02-02 --output-name Fractal_BE_v3
```

Check specific dates:
- **14.11.2025**: SHORT should NOT close at BE on entry candle
- **19.01.2026**: LONG should continue past entry, BE protects on subsequent candles
- **30.01.2026**: LONG should NOT close at BE on entry candle

## SMC Code Reused

| Function | File | Purpose |
|----------|------|---------|
| `detect_fractals()` | `src/smc/detectors/fractal_detector.py:16` | Williams 3-bar fractal detection |
| `find_unswept_fractals()` | `src/smc/detectors/fractal_detector.py:76` | Filter to unswept at given time |
| `check_fractal_sweep()` | `src/smc/detectors/fractal_detector.py:128` | Find sweep time in window |
