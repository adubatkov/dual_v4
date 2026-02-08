# TSL (Trailing Stop Loss) Logic Analysis - USDCHF Trade 11577281

**Date:** 2025-11-20
**Issue:** Position closed earlier than expected, possibly due to incorrect TSL calculation

---

## Original TSL Logic (from dual_asset_ib_strategy.py)

### Function: `simulate_after_entry()` (lines 460-534)

**TSL Logic (for LONG):**

```python
# Lines 504-516
if direction == "long":
    hit = False
    while hi >= curr_tp:  # While price reaches current TP
        hit = True
        last_tp = curr_tp

        # KEY FORMULA: New SL = last reached TP - (TSL_SL * risk)
        new_stop = last_tp - tsl_sl * risk
        curr_stop = max(curr_stop, new_stop)  # SL only moves up

        # KEY FORMULA: New target R = current + TSL_TARGET
        curr_target_R += tsl_target

        # KEY FORMULA: New TP = entry + (new target R * risk)
        curr_tp = entry_price + curr_target_R * risk

        if tsl_target <= 0:
            break

    # If TP was hit AND price hit new SL on same candle
    if hit and lo <= curr_stop:
        exit_next_open = True  # Exit on next candle open
```

**Key Points:**
1. **Incremental logic**: `curr_target_R += tsl_target` (adds each time)
2. **SL from last TP**: `new_stop = last_tp - tsl_sl * risk`
3. **TP from entry**: `curr_tp = entry_price + curr_target_R * risk`

---

## Bot Implementation (ib_strategy.py lines 634-665)

```python
if direction == "long":
    if current_price >= virtual_tp:
        tp_hit = True
        logger.info(f"LONG virtual TP hit at {current_price:.2f}, adjusting TSL...")

        # Calculate new SL and TP
        # New SL = last virtual TP - (TSL_SL * risk)
        new_sl = virtual_tp - (tsl_sl * risk)
        new_sl = max(current_sl, new_sl)

        # Calculate current target R
        current_target_R = (virtual_tp - entry_price) / risk

        # New target R = current + TSL_TARGET
        new_target_R = current_target_R + tsl_target

        # New virtual TP = entry + (new_target_R * risk)
        new_virtual_tp = entry_price + (new_target_R * risk)

        # Update position
        result = self.executor.modify_position(position.ticket, new_sl, 0.0)
```

---

## Analysis USDCHF Trade 11577281

### Trade Parameters

```
Entry:           0.80429
Initial SL:      0.80349
Risk:            0.00080 (8.0 pips)
Initial TP:      0.80550 (virtual)
RR_TARGET:       1.5
TSL_TARGET:      2.0
TSL_SL:          0.5
```

### Initial TP Calculation

```
Initial Target R = RR_TARGET = 1.5
Initial TP = entry + (RR_TARGET * risk)
           = 0.80429 + (1.5 * 0.00080)
           = 0.80429 + 0.00120
           = 0.80549
```

**In log:** Initial TP = 0.80550 (rounded) - CORRECT

### TSL Update at 21:30:25

**Condition:** Price reached 0.80550 (virtual TP)

**New SL calculation:**
```
new_sl = virtual_tp - (tsl_sl * risk)
       = 0.80550 - (0.5 * 0.00080)
       = 0.80550 - 0.00040
       = 0.80510
```

**Rounded:** 0.80510
**In log:** New SL = 0.80509
**Difference:** 0.1 pip (rounding)

- SL CALCULATION CORRECT (within rounding error)

---

## Comparison: Bot vs Original Strategy

| Aspect | Original (dual_asset_ib_strategy.py) | Bot (ib_strategy.py) | Status |
|--------|---------------------------------------|----------------------|--------|
| **SL Formula** | `new_stop = last_tp - tsl_sl * risk` | `new_sl = virtual_tp - (tsl_sl * risk)` | Identical |
| **TP Formula** | `curr_tp = entry_price + curr_target_R * risk` | `new_virtual_tp = entry_price + (new_target_R * risk)` | Identical |
| **Target R Update** | `curr_target_R += tsl_target` | `new_target_R = current_target_R + tsl_target` | Identical |
| **Incremental** | Yes (while loop, multiple hits) | Yes (check each tick) | Equivalent |
| **SL only up** | `curr_stop = max(curr_stop, new_stop)` | `new_sl = max(current_sl, new_sl)` | Identical |

---

## Conclusions

### TSL Logic Works CORRECTLY

1. **SL and TP formulas identical to original strategy**
2. **Incremental logic (curr_target_R += tsl_target) implemented correctly**
3. **SL only moves up for LONG / down for SHORT - correct**

### Position Closed CORRECTLY

**USDCHF 11577281:**
- Entry: 0.80429
- TP reached: 0.80550 (+12.1 pips)
- New SL: 0.80510 (breakeven + 8.1 pips protected profit)
- Exit: between 21:30 and 21:32 (price hit SL 0.80510)
- **Result: Profit +8 pips** (instead of -8 pips risk)

**This is normal TSL behavior** - profit protection after reaching first TP.

### Logging Issues Only

1. **Virtual TP displayed as .2f instead of .5f** - misleading
2. **Current price displayed as .2f instead of .5f** - hard to analyze

### Recommendations

1. **Fix log formatting** (lines 638, 659, 670, 690 in ib_strategy.py):
   - Change `.2f` to `.5f` for Forex pairs

2. **DO NOT CHANGE TSL calculation logic** - it works correctly!
