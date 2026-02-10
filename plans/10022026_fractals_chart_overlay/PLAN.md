# Plan: Fix Fractal Lines -- End at Sweep Point

## Context

First implementation (commit `0c87294`) drew full-width horizontal lines for unswept fractals.
User correction: "Lines must END at the place where they get 'swept' (where price first
interacts with the level after formation)." Reference: `dual_v3/strategy_optimization/fractals/
generate_fractal_charts.py:339` (`_draw_fractal_sweeps` method).

**Key behavioral change:**
- Line starts at `max(chart_left_edge, fractal_formation_time)`
- Line ends at `sweep_time` (first M1 bar touching the level within chart window)
- Unswept fractals (no price interaction during window): dashed line to chart right edge
- Swept fractals: solid line ending at sweep point
- Labels: formation time in UTC, positioned at line_start, va=bottom for highs / top for lows

## Files to Modify

1. `backtest/run_backtest_template.py` -- simplify: pass raw fractal lists instead of per-date
2. `backtest/reporting/trade_charts.py` -- rewrite `_draw_fractals()` with sweep computation

## Existing Code to Reuse

- `fractal_detector.py:16` `detect_fractals()` -- already used, keep as-is
- `fractal_detector.py:76` `find_unswept_fractals(fractals, m1_data, before_time, lookback_hours)` -- move into chart gen (per-trade)
- `fractal_detector.py:128` `check_fractal_sweep(fractal, m1_data, window_start, window_end) -> Optional[datetime]` -- NEW usage, returns first M1 bar that touches the fractal level
- Reference: `dual_v3/.../generate_fractal_charts.py:339-413` `_draw_fractal_sweeps()` pattern

## Implementation

### Step 1: Simplify `run_backtest_template.py` (line 807-846)

Remove the per-date loop that pre-computes unswept fractals. Instead, pass raw fractal lists
to the chart generator and let it handle per-trade filtering.

**Before** (current, lines 807-846): per-date loop calling `find_unswept_fractals`, dedup, building `fractal_data_by_date` dict.

**After:**
```python
if generate_charts:
    fractal_lists = None
    logger.info("Computing H1/H4 fractals for chart overlay...")
    h1_data = _resample_m1(m1_data, "1h")
    h4_data = _resample_m1(m1_data, "4h")
    all_h1 = detect_fractals(h1_data, symbol, "H1", candle_duration_hours=1.0)
    all_h4 = detect_fractals(h4_data, symbol, "H4", candle_duration_hours=4.0)
    logger.info(f"Detected {len(all_h1)} H1 and {len(all_h4)} H4 fractals")
    fractal_lists = {"all_h1": all_h1, "all_h4": all_h4}

    charts_count = generate_all_trade_charts(
        ...,
        fractal_lists=fractal_lists,  # RENAMED from fractal_data_by_date
    )
```

Also update import (line 51): add `check_fractal_sweep` (or move import into trade_charts.py).

### Step 2: Update `generate_all_trade_charts()` signature (trade_charts.py:581)

Rename param `fractal_data_by_date` -> `fractal_lists`. Pass same dict to every trade
(no per-date lookup needed -- per-trade filtering happens inside chart generator).

```python
def generate_all_trade_charts(
    ...,
    fractal_lists: Optional[Dict] = None,   # {"all_h1": List[Fractal], "all_h4": List[Fractal]}
) -> int:
```

Inside the loop: pass `fractal_lists=fractal_lists` to each `generate_trade_chart()` call.

### Step 3: Update `generate_trade_chart()` signature (trade_charts.py:78)

Rename param `fractal_data` -> `fractal_lists`. Also need `m1_data` accessible for sweep
checks -- it's already a parameter.

### Step 4: Add `_prepare_fractal_data()` method (NEW, in TradeChartGenerator)

This method takes raw fractal lists + m1_data + chart window and returns a flat list of
draw-ready dicts with computed sweep times.

```python
def _prepare_fractal_data(self, fractal_lists, m1_data, chart_start, chart_end):
    """Find unswept fractals at chart_start and compute sweep times within window."""
    if not fractal_lists:
        return []

    from src.smc.detectors.fractal_detector import find_unswept_fractals, check_fractal_sweep

    start_utc = chart_start.astimezone(pytz.UTC)
    end_utc = chart_end.astimezone(pytz.UTC)

    result = []
    for tf_key, lookback in [("all_h1", 48), ("all_h4", 96)]:
        all_fracs = fractal_lists.get(tf_key, [])
        unswept = find_unswept_fractals(all_fracs, m1_data, start_utc, lookback_hours=lookback)
        for frac in unswept:
            sweep_time = check_fractal_sweep(frac, m1_data, start_utc, end_utc)
            result.append({
                "price": frac.price,
                "type": frac.type,
                "timeframe": frac.timeframe,   # "H1" or "H4"
                "frac_time": frac.time,         # UTC datetime
                "sweep_time": sweep_time,       # UTC datetime or None
            })

    # Deduplicate: if H1 and H4 at same (type, price) -> keep only H4
    h4_keys = {(r["type"], round(r["price"], 2)) for r in result if r["timeframe"] == "H4"}
    result = [r for r in result if r["timeframe"] == "H4" or
              (r["type"], round(r["price"], 2)) not in h4_keys]

    return result
```

### Step 5: Rewrite `_draw_fractals()` (trade_charts.py:349)

Replace current full-width line drawing with sweep-aware drawing.

```python
def _draw_fractals(self, ax, fractal_draw_data, chart_start, chart_end):
    """Draw fractal lines ending at sweep point (or dashed to chart edge if unswept)."""
    if not fractal_draw_data:
        return

    start_naive = chart_start.replace(tzinfo=None)
    end_naive = chart_end.replace(tzinfo=None)
    ymin, ymax = ax.get_ylim()

    for frac in fractal_draw_data:
        price = frac["price"]
        if price < ymin or price > ymax:
            continue

        # Convert fractal formation time to local naive
        frac_time = frac["frac_time"]
        if frac_time.tzinfo is None:
            frac_time = pytz.utc.localize(frac_time)
        frac_local = frac_time.astimezone(self.tz).replace(tzinfo=None)

        # Line start: max(chart left edge, fractal formation time)
        line_start = max(start_naive, frac_local)

        # Line end: sweep time or chart right edge
        swept = frac["sweep_time"] is not None
        if swept:
            sweep_time = frac["sweep_time"]
            if sweep_time.tzinfo is None:
                sweep_time = pytz.utc.localize(sweep_time)
            line_end = sweep_time.astimezone(self.tz).replace(tzinfo=None)
        else:
            line_end = end_naive

        if line_end <= line_start:
            continue

        # Style: H4=red, H1=black; swept=solid, unswept=dashed
        is_h4 = frac["timeframe"] == "H4"
        color = "#d32f2f" if is_h4 else "black"
        linewidth = 1.5 if is_h4 else 0.8
        linestyle = "-" if swept else "--"
        alpha = 0.8 if is_h4 else 0.6

        ax.hlines(y=[price], xmin=line_start, xmax=line_end,
                  colors=color, linewidth=linewidth, linestyles=linestyle,
                  zorder=1, alpha=alpha)

        # Label: "4h H 05:00" or "1h L 12:00" at line_start
        tf_label = "4h" if is_h4 else "1h"
        type_label = "H" if frac["type"] == "high" else "L"
        va = "bottom" if frac["type"] == "high" else "top"
        label_utc = frac_time.strftime("%H:%M")

        ax.text(line_start, price, f" {tf_label} {type_label} {label_utc}",
                fontsize=7 if is_h4 else 6, va=va, ha="left",
                color=color, alpha=alpha,
                fontweight="bold" if is_h4 else "normal")

    ax.set_ylim(ymin, ymax)
```

### Step 6: Wire it up in `generate_trade_chart()` (line ~126-132)

Replace current call:
```python
# BEFORE:
self._draw_fractals(ax, fractal_data, chart_start, chart_end)

# AFTER:
fractal_draw_data = self._prepare_fractal_data(fractal_lists, m1_data, chart_start, chart_end)
self._draw_fractals(ax, fractal_draw_data, chart_start, chart_end)
```

## Verification

```bash
cd C:\Trading\ib_trading_bot\dual_v4
python backtest/run_backtest_template.py --symbol GER40 --params prod --start 2025-11-04 --end 2026-02-02 --output-name Fractals_Test_v2
```

Check trade charts:
- Swept fractals: solid lines ending at the price interaction point (NOT chart edge)
- Unswept fractals: dashed lines extending to chart right edge
- H1 = black (thin), H4 = red (thick)
- Labels show formation time in UTC at line start
- No H1/H4 duplicates at same price
- Y-axis not stretched by out-of-range fractals
