# Fractal TSL (M2 Fractal Trailing Stop) -- Changelog

**Date**: 2026-02-11
**Scope**: Trailing stop based on 2-minute fractals, triggered by H1/H4 fractal sweep
**Status**: DONE

---

## Original Request

> Текущее задание -- расширить функционал стратегии с помощью логики 2 минутных фракталов
> и провести бэктест следующим образом:
> Используется логика FRACTAL_BE_ENABLED
> При касании 1h | 4h фрактала должна запускаться новая система установки SL параллельно
> с TSL - Установка SL на уровне последнего образовавшегося 2м фрактала (если сделка в
> Long, то фрактал шортовый (L), если сделка в short, то фрактал лонговый (h)).
> В дальнейшем, при образовании новых 2м фракталов, SL переносится на их уровень.
> При выборе уровня SL между фрактальным и TSL выбирается более "выгодный".
> Это должен быть отдельный параметр в strategy_logic.py не зависящий от FRACTAL_BE_ENABLED

### Clarification

> Fractal TSL должен активироваться при свипе ЛЮБОГО H1/H4 фрактала, или только при
> условии что SL в отрицательной зоне (как у FRACTAL_BE)?

**Answer**: Любой свип (без проверки зоны SL).

---

## Iteration 1: Fractal TSL v1 (uncommitted)

**Output**: `backtest/output/2026-02-11_020554_GER40_Fractal_TSL_v1/`
**Date**: 2026-02-11 ~02:05 UTC+5

**What was done:**

1. **M2 fractal detection** (pre-loop):
   - Resample M1 -> 2min via `_resample_m1(m1_data, "2min")`
   - `detect_fractals(m2_data, symbol, "M2", candle_duration_hours=2/60)`
   - Sort by `confirmed_time`, use incremental pointer

2. **State tracking**:
   - `m2_fractal_ptr` + `last_m2_high` / `last_m2_low` -- running latest M2 fractal
   - `fractal_tsl_active` -- ticket -> True when activated
   - `fractal_tsl_prev_sl` -- ticket -> previous M2 SL (for change logging)

3. **Activation** (in H1/H4 sweep check block):
   - When H1/H4 fractal swept + position open + ticket not yet tracked
   - Read `FRACTAL_TSL_ENABLED` from variation params
   - Log: `[FRACTAL TSL] ... Fractal TSL activated (M2 {type} SL={price})`

4. **SL computation** (effective SL block):
   - For LONG: `frac_tsl_sl = last_m2_low[0]`; for SHORT: `last_m2_high[0]`
   - Log M2 SL changes: `[FRACTAL TSL] ... M2 {type} SL {prev} -> {new}`
   - Three-way competition: `effective = max(organic, frac_be, frac_tsl)` for LONG

5. **Parameter**: `FRACTAL_TSL_ENABLED: True` added to all 8 variations in `strategy_logic.py`

**Files changed:**
- `backtest/run_backtest_template.py` (~+60 lines)
- `src/utils/strategy_logic.py` (+8 lines, 1 per variation)

---

## Results: Fractal TSL v1

```
Output: backtest/output/2026-02-11_020554_GER40_Fractal_TSL_v1/
Symbol: GER40
Params: PROD
Period: 2025-11-04 to 2026-02-02
Total Trades: 55 (23W / 26L)
Fractal BE activations: 20
Fractal TSL activations: 21, M2 SL updates: 29
```

### Comparison vs Fractal BE v3 (baseline)

| Metric | BE v3 | TSL v1 | Delta |
|--------|-------|--------|-------|
| Total P/L | $9,243 | $11,524 | +$2,281 (+24.7%) |
| Profit Factor | 1.41 | 1.51 | +0.10 |
| Win Rate | 38.2% | 41.8% | +3.6pp |
| Max Drawdown | 4.9% | 4.6% | -0.3pp |
| Sharpe | 1.91 | 2.38 | +0.47 |
| Sortino | 7.30 | 9.14 | +1.84 |
| Annualized Return | 44.9% | 58.1% | +13.2pp |

### Full comparison chain

| Metric | Control | BE v3 | TSL v1 |
|--------|---------|-------|--------|
| Total P/L | $6,742 | $9,243 | $11,524 |
| Profit Factor | 1.26 | 1.41 | 1.51 |
| Max Drawdown | 7.1% | 4.9% | 4.6% |
| Sharpe | - | 1.91 | 2.38 |

### Trades that changed (11 of 55)

| # | Date | Var | Dir | P/L v3 | P/L TSL | Delta |
|---|------|-----|-----|--------|---------|-------|
| 7 | 13.11 | TCWE | SHORT | 1,824 | 2,014 | +190 |
| 8 | 14.11 | TCWE | SHORT | 985 | 1,997 | +1,012 |
| 16 | 26.11 | TCWE | SHORT | 1,979 | 2,142 | +164 |
| 17 | 27.11 | OCAE | LONG | 979 | 1,802 | +823 |
| 25 | 09.12 | OCAE | LONG | 796 | 1,002 | +206 |
| 35 | 29.12 | OCAE | SHORT | 2,879 | 2,131 | -748 |
| 39 | 08.01 | OCAE | LONG | 980 | 1,315 | +335 |
| 43 | 14.01 | TCWE | LONG | 0 | 22 | +22 |
| 46 | 19.01 | OCAE | LONG | 0 | 609 | +609 |
| 50 | 23.01 | OCAE | LONG | 127 | 712 | +585 |
| 52 | 27.01 | OCAE | SHORT | 2,058 | 1,141 | -917 |

**9 improved, 2 worsened.** Net delta from changed trades: +$2,281.

---

## By Variation (TSL v1)

| Variation | Trades | Win Rate | P/L |
|-----------|--------|----------|-----|
| TCWE | 18 | 44.4% | $8,074 |
| OCAE | 33 | 42.4% | $4,562 |
| Reverse | 4 | 25.0% | -$1,113 |

---

## Console Output Sample

```
Computing H1/H4 fractals...
Detected 551 H1 and 163 H4 fractals
Computing M2 fractals...
Detected 5249 M2 fractals
Running backtest loop...
...
[FRACTAL BE]  2025-11-13 09:19 UTC | GER40 SHORT #1000012 | H1 low 24343.50 swept | Fractal BE activated (entry=24386.75)
[FRACTAL TSL] 2025-11-13 09:19 UTC | GER40 SHORT #1000012 | H1 low 24343.50 swept | Fractal TSL activated (M2 high SL=24369.30)
[FRACTAL TSL] 2025-11-13 09:38 UTC | GER40 #1000012 | M2 high SL 24369.30 -> 24313.10
...
[FRACTAL BE SUMMARY] 20 fractal BE activations out of 55 total trades
[FRACTAL TSL SUMMARY] 21 fractal TSL activations, 29 M2 SL updates out of 55 total trades
```

---

## How It Works

Three SL systems compete. The most protective (favorable) SL wins:

| System | Source | When active | Trigger condition |
|--------|--------|-------------|-------------------|
| TSL organic | `ib_strategy.update_position_state()` | After virtual TP hit | Standard TSL formula |
| Fractal BE | Entry price | After H1/H4 fractal swept | SL in negative zone |
| Fractal TSL | Latest M2 fractal price | After H1/H4 fractal swept | Any sweep (no zone check) |

**LONG**: `effective = max(TSL, BE, fractal_TSL)` -- highest SL wins
**SHORT**: `effective = min(TSL, BE, fractal_TSL)` -- lowest SL wins

Same-candle protection via `effective_sl_prev` applies to all three systems.

---

## Key Files Modified

| File | Changes |
|------|---------|
| `backtest/run_backtest_template.py` | M2 detection, pointer, activation, SL computation, logging |
| `src/utils/strategy_logic.py` | Added `FRACTAL_TSL_ENABLED: True` to all 8 variations |

## Key Code Patterns

| Pattern | Purpose |
|---------|---------|
| `m2_fractal_ptr` + `last_m2_high/low` | Incremental M2 fractal tracking with running latest values |
| `fractal_tsl_active` | Track which tickets have fractal TSL activated |
| `fractal_tsl_prev_sl` | Detect M2 SL changes for logging |
| `candidates = [organic, frac_be, frac_tsl]` | Three-way SL competition |
