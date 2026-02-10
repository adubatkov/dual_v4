# Fractal-Based Breakeven (BE) Logic -- Changelog

**Date**: 2026-02-10
**Scope**: Move SL to entry (breakeven) when price touches unswept H1/H4 fractal
**Status**: DONE

---

## Original Requests

### Request 1: Initial fractal BE implementation

> Теперь давай протестируем работоспособность фракталов. Задача - когда у нас есть открытая
> позиция, и цена в процессе движения задевает фрактал - перевести позицию в безубыток,
> т.е. переместить стоп-лосс на цену открытия позиции. Условие для активации - стоп-лосс
> должен быть в минусовой зоне (ниже цены входа для LONG, выше цены входа для SHORT).
> Если стоп лосс уже за ценой входа (например TSL уже вывел в плюс) - не трогаем.
> Фрактал считается задетым когда M1 свеча касается уровня фрактала (high >= fractal.price
> для high фрактала, low <= fractal.price для low фрактала).

### Request 2: Bug reports after v1

> Вижу решение проблемы с TSL, но не увидел комментариев и решения проблем ниже:
>
> 14.11.2025 - Какая здесь была логика? Позиция SHORT открывается на 21455.52, при
> этом H4 Low фрактал на 21408.10 свипается в рамках одной и той же M1 свечи.
> Результат: Позиция закрывается в 0 по BE. Это не правильное поведение.
>
> 19.01.2026 - неожиданное поведение: На графике видно, что H1 high fractal на 21025
> свипается ценой 21029 (разница 4 пункта). При этом позиция LONG (вход около 20980)
> закрывается по BE с profit=0. Вопросы: Почему такая маленькая разница (4 пункта)
> привела к свипу? Правильно ли, что свип на уровне, очень близком ко входу, закрыл
> позицию в 0?
>
> 30.01.2026 - Неясна причина выхода из сделки по BE: позиция LONG (вход 21135) как будто
> задела фрактал и вышла с profit=0. Но на графике не видно очевидного касания фрактала.

### Request 3: Same-candle BE exit bug investigation

> а в чем была проблема?

---

## Iteration 1: Basic fractal BE (uncommitted, superseded by v2)

**Output**: `backtest/output/2026-02-10_205259_GER40_Fractal_BE_v1/`
**Date**: 2026-02-10 ~20:50 UTC+5

**What was done:**
- Added fractal pre-computation in main loop (H1/H4 detection, deduplication)
- Added per-candle sweep check: when M1 candle touches unswept fractal and position
  has SL in negative zone, move SL to entry price
- Removed swept fractals from active set
- Added `[FRACTAL BE]` log messages and summary count

**Results:**
- 21 fractal BE activations out of 55 trades
- Total P/L: $10,591, Profit Factor: 1.47, Max DD: 5.1%
- Improvement over control ($6,742, PF 1.26, DD 7.1%)

**Problems found (3 bugs):**
1. **TSL contamination**: TSL reads `position.sl` which fractal BE modifies to entry.
   Next candle TSL uses entry as base -> oversized jumps
2. **Invisible fractals**: No lookback limit in backtest (fractals active forever) vs
   chart overlay (H1=48h, H4=96h lookback)
3. **No configurable toggle**: No way to enable/disable fractal BE per variation

---

## Iteration 2: TSL/BE decoupling + lookback + toggle (uncommitted, superseded by v3)

**Output**: `backtest/output/2026-02-10_212240_GER40_Fractal_BE_v2/`
**Date**: 2026-02-10 ~21:22 UTC+5

**What was done:**
- **TSL decoupling**: Added `tsl_organic_sl` dict to track TSL's own SL independently.
  Before TSL runs: restore organic SL (undo BE overlay). After TSL: capture organic.
  Effective SL = max(organic, fractal_be) for LONG.
- **Lookback expiry**: H1 fractals expire after 48h, H4 after 96h (match chart overlay)
- **Configurable toggle**: Added `FRACTAL_BE_ENABLED: True` to all 8 variations in
  `strategy_logic.py` (GER40_PARAMS_PROD + XAUUSD_PARAMS_PROD)
- **Import fix**: Added `timedelta` to `from datetime import datetime, timedelta`

**Results:**
- 20 fractal BE activations (1 fewer due to lookback expiry removing stale fractals)
- Total P/L: $8,331, Profit Factor: 1.37, Max DD: 5.0%, Sharpe: 1.72

**Remaining problem:**
3 specific dates showed same-candle BE exit (position closes at BE profit=0 on the
same candle where BE activates):
- 14.11.2025: SHORT #1000014, H4 Low swept -> closed at BE immediately
- 19.01.2026: LONG #1000090, H1 High swept -> closed at BE immediately
- 30.01.2026: LONG #1000108, H1 High swept -> closed at BE immediately

---

## Iteration 3: Same-candle BE exit fix (final)

**Output**: `backtest/output/2026-02-10_214634_GER40_Fractal_BE_v3/`
**Date**: 2026-02-10 ~21:46 UTC+5

**Root cause of same-candle bug:**
Lines 821-826 (v2) updated `sl_before_tsl` to include the BE level (entry price).
`_check_sltp_hits` uses `sl_before_tsl` as `sl_override`, checking SL at entry price
on the SAME candle. Entry price is always within the entry candle's range, so the
position closes immediately.

**The principle:** SL modifications (TSL or fractal BE) should only take effect from
the NEXT candle, not the current one.

**What was changed:**
- **Replaced `sl_before_tsl` with `effective_sl_prev`**: Capture `position.sl` at
  start of candle (before any modifications). This records the effective SL from the
  end of the previous candle.
- **Smart `sl_override`**: Only override `_check_sltp_hits` for tickets where SL
  actually changed THIS candle. For unchanged tickets, `position.sl` (effective SL)
  is used directly via `_check_sltp_hits` line 451 fallback.
- **Removed `sl_before_tsl` variable entirely**: No longer needed.

**How the fix works per scenario:**

| Scenario | effective_sl_prev | position.sl after | Override? | _check_sltp_hits uses |
|----------|-------------------|--------------------|-----------|-----------------------|
| Position opens + BE activates (same candle) | initial SL | entry (BE) | YES: initial SL | initial SL (no false BE exit) |
| BE was active from previous candle (no change) | entry | entry | NO | position.sl = entry (BE protects) |
| TSL triggers (no BE) | pre-TSL SL | new TSL SL | YES: pre-TSL | pre-TSL (same-candle TSL protection) |

**Results:**
- 20 fractal BE activations (unchanged)
- Total P/L: $9,243, Profit Factor: 1.41, Max DD: 4.9%, Sharpe: 1.91, Sortino: 7.30

**Date-specific fixes:**
- 14.11.2025: FIXED -- trade survived entry candle, TSL triggered twice, closed at TP
  with profit=$985.47 (was $0)
- 19.01.2026: Correct behavior -- position lasted multiple candles, closed at BE when
  price returned to entry ($0, but legitimate BE protection)
- 30.01.2026: Correct behavior -- position lasted multiple candles, closed at BE when
  price returned to entry ($0, but legitimate BE protection)

---

## Backtest Results Comparison

| Metric | Control | v1 (buggy) | v2 (TSL fix) | v3 (final) |
|--------|---------|-----------|--------------|------------|
| Total P/L | $6,742 | $10,591 | $8,331 | $9,243 |
| Profit Factor | 1.26 | 1.47 | 1.37 | 1.41 |
| Win Rate | 38.2% | - | 36.4% | 38.2% |
| Max Drawdown | 7.1% | 5.1% | 5.0% | 4.9% |
| Sharpe Ratio | - | - | 1.72 | 1.91 |
| Sortino Ratio | - | - | 7.18 | 7.30 |
| Annualized Return | - | - | 39.9% | 44.9% |
| BE activations | 0 | 21 | 20 | 20 |

**v3 vs Control improvement**: +$2,501 P/L (+37%), PF 1.26->1.41, DD 7.1%->4.9%

---

## By Variation (v3 final)

| Variation | Trades | Win Rate | P/L |
|-----------|--------|----------|-----|
| TCWE | 18 | 38.9% | $6,709 |
| OCAE | 33 | 39.4% | $3,647 |
| Reverse | 4 | 25.0% | -$1,113 |

---

## Files Modified

| File | Changes |
|------|---------|
| `backtest/run_backtest_template.py` | Fractal BE logic: pre-computation, TSL decoupling, sweep check, BE trigger, effective_sl_prev, same-candle protection |
| `src/utils/strategy_logic.py` | Added `FRACTAL_BE_ENABLED: True` to all 8 variations |

## Key Code Patterns

| Pattern | Purpose |
|---------|---------|
| `tsl_organic_sl` | Track TSL's own SL independently from fractal BE overlay |
| `fractal_be_active` | Track which tickets have fractal BE triggered (ticket -> entry_price) |
| `effective_sl_prev` | Capture position.sl before modifications to prevent same-candle exits |
| `sl_override_for_check` | Only override _check_sltp_hits for tickets with SL changes this candle |

## SMC Code Reused

| Function | File | Purpose |
|----------|------|---------|
| `detect_fractals()` | `src/smc/detectors/fractal_detector.py:16` | Williams 3-bar fractal detection |

## Backtest Output (final)

```
Output: backtest/output/2026-02-10_214634_GER40_Fractal_BE_v3/
Symbol: GER40
Params: PROD
Period: 2025-11-04 to 2026-02-02
Total Trades: 55 (21W / 26L)
BE Activations: 20
Total P/L: $9,243 (9.2%)
Profit Factor: 1.41
Max Drawdown: 4.9%
Sharpe: 1.91
```
