# Fractals Chart Overlay -- Changelog

**Date**: 2026-02-10
**Scope**: H1/H4 fractal lines on backtest trade charts
**Status**: DONE

---

## Original Request

> Мы реализовали планы C:\Trading\ib_trading_bot\dual_v4\plans и провели контрольный бэктест
> C:\Trading\ib_trading_bot\dual_v4\backtest\output\Control_2026-02-10_185925_GER40_GER40_prod_fixed1000_20260210_185925
> с помощью C:\Trading\ib_trading_bot\dual_v4\backtest\run_backtest_template.py.
> Теперь нам надо постепенно протестировать работоспособность наших новых инструментов.
> Начнем с фракталов. Задача - провести бэктест этого же периода, но теперь расчитать
> еще и фракталы 1h и 4h и отобразить их на графиках, как мы это делали в старой папке
> C:\Trading\ib_trading_bot\dual_v3\strategy_optimization\fractals. Обратить внимание -
> 4 часовые фракталы скорее всего будут и 1 часовыми, поэтому нужно отобразить только
> фрактал старшего таймфрейма на графике. 1 часовой фрактал - черная линия, 4 часовой
> фрактал - красная линия. Результат должен быть в папке
> C:\Trading\ib_trading_bot\dual_v4\backtest\output точно также, как и контрольный бэктест.
> Есть вопросы?

---

## Iteration 1: Basic fractal overlay (commit 0c87294)

**Commit**: `0c87294874620050708a63d54dc9f756209286ef`
**Message**: feat: add H1/H4 fractal overlay on trade charts
**Date**: 2026-02-10 19:25 UTC+5

**What was done:**
- Added `_resample_m1()` helper in `run_backtest_template.py` for M1 -> H1/H4 resampling
- Used `detect_fractals()` from `src/smc/detectors/fractal_detector.py` for H1 and H4
- Pre-computed unswept fractals per-date using `find_unswept_fractals()` at IB start
- H1/H4 deduplication: H4 takes priority at same price+type
- Drew full-width horizontal lines on trade charts (H1=black, H4=red)
- Filtered fractals to visible y-range to prevent axis stretching

**Files changed:**
- `backtest/run_backtest_template.py` (+42 lines)
- `backtest/reporting/trade_charts.py` (+68 lines)

**Problem found:**
Lines were stretched across the entire chart width. Correct behavior (per reference
`dual_v3/strategy_optimization/fractals/generate_fractal_charts.py`) requires lines to
END at the sweep point where price first interacts with the fractal level.

---

## Iteration 2: Lines end at sweep point (commit 3f73a95)

**Commit**: `3f73a95ec31599d6cb91f5cc4cc8acc0402af59c`
**Message**: fix: fractal lines end at sweep point, not full chart width
**Date**: 2026-02-10 19:45 UTC+5

**What was changed:**
- Removed per-date fractal pre-computation from runner -> pass raw lists to chart gen
- Added `_prepare_fractal_data()` method in `TradeChartGenerator`:
  - Per-trade: finds unswept fractals at chart start (`find_unswept_fractals`)
  - Computes sweep time per fractal (`check_fractal_sweep`)
  - Deduplicates H1/H4 overlaps
- Rewrote `_draw_fractals()`:
  - Line starts at `max(chart_left, fractal_formation_time)`
  - Line ends at `sweep_time` (solid) or chart right edge (dashed if unswept)
  - Labels show fractal formation time in UTC at line start
  - va=bottom for high fractals, va=top for low fractals

**Files changed:**
- `backtest/run_backtest_template.py` (-20 lines net)
- `backtest/reporting/trade_charts.py` (+45 lines net)

**Visual result:**
- Swept fractals: solid lines ending at the price interaction point
- Unswept fractals: dashed lines extending to chart right edge
- H1 = thin black (0.8px), H4 = thick red (1.5px)
- No Y-axis stretching from out-of-range fractals

---

## Backtest Output

```
Output: backtest/output/2026-02-10_194255_GER40_Fractals_Test_v2/
Symbol: GER40
Params: PROD
Period: 2025-11-04 to 2026-02-02
Fractals detected: 551 H1 + 163 H4
Charts generated: 55
```

---

## Key Files Modified

| File | Role |
|------|------|
| `backtest/run_backtest_template.py` | Fractal detection (resample + detect), pass to charts |
| `backtest/reporting/trade_charts.py` | Per-trade sweep computation + drawing |

## SMC Code Reused

| Function | File | Purpose |
|----------|------|---------|
| `detect_fractals()` | `src/smc/detectors/fractal_detector.py:16` | Williams 3-bar fractal detection |
| `find_unswept_fractals()` | `src/smc/detectors/fractal_detector.py:76` | Filter to unswept at given time |
| `check_fractal_sweep()` | `src/smc/detectors/fractal_detector.py:128` | Find sweep time in window |

## Reference Implementation

`dual_v3/strategy_optimization/fractals/generate_fractal_charts.py:339` -- `_draw_fractal_sweeps()`
