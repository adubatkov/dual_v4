# TSL Monitoring Guide

**Quick reference for monitoring bot behavior during demo/production testing**

---

## What to Look For in Logs

### 1. Position Opening (Should See)

```
[USDJPY_M2003_IB09:00-10:00] OCAE signal - LONG at 157.24000, SL:157.03800, TP:157.34100
Position window: 90 min, closes at 11:30:00 (Asia/Tokyo)
Order sent to MT5: LONG 0.03 lots USDJPY at 157.24000
```

**Check:**
- [x] Price shows .5f precision (5 decimals for JPY pairs)
- [x] Position window matches variation (TCWE=60, OCAE=90, REV=120, REV_RB=150)
- [x] Lot size reasonable (~0.02-0.05 for 1% risk on $50k)
- [x] Window end time shown in exchange timezone

---

### 2. TSL Trigger (Should See)

```
[USDJPY_M2003_IB09:00-10:00] TSL Check LONG: price=157.34700, virtual_tp=157.34100
[USDJPY_M2003_IB09:00-10:00] LONG virtual TP hit at 157.34700

TSL Calculation (LONG):
  Entry Price:     157.24000
  Initial Risk:    0.20200 (2020.0 pips)
  Current SL:      157.03800
  Virtual TP Hit:  157.34100
  Current Price:   157.34700
  TSL_SL:          0.5
  TSL_TARGET:      0.5
  New SL:          157.13900 (moved +1010.0 pips)
  New Virtual TP:  157.44200 (moved +1010.0 pips)

Modifying position 11580505: SL 157.03800 -> 157.13900
```

**Check:**
- [x] Initial Risk stays CONSTANT (never changes between TSL triggers)
- [x] SL movement = TSL_SL * Initial Risk (e.g., 0.5 * 0.202 = 0.101)
- [x] TP movement = TSL_TARGET * Initial Risk (e.g., 0.5 * 0.202 = 0.101)
- [x] Pip movements consistent between steps
- [x] Current Price >= Virtual TP (for LONG) or <= Virtual TP (for SHORT)

---

### 3. Position Window Expiration (Should See)

```
[USDJPY_M2003_IB09:00-10:00] Position window expired, closing position 11580505
  Window end: 2025-11-20 02:30:00 UTC
Closing position 11580505 at market price
Position 11580505 closed successfully
```

**Check:**
- [x] Closes within ~5-10 seconds of window end
- [x] Closure happens BEFORE TSL checks
- [x] Success message received

---

### 4. Bot Restart with Open Position (Should See)

```
[USDJPY_M2003_IB09:00-10:00] Found open position 11580505 for this strategy
[USDJPY_M2003_IB09:00-10:00] Cannot restore TSL state reliably after bot restart
[USDJPY_M2003_IB09:00-10:00] Position 11580505 will be managed by MT5 SL only
[USDJPY_M2003_IB09:00-10:00] Current SL: 157.24000, Entry: 157.24000
[USDJPY_M2003_IB09:00-10:00] TSL tracking disabled for this position
```

**Check:**
- [x] Warning messages appear
- [x] TSL does NOT trigger after restart
- [x] Position remains open with current SL
- [x] No spam of TSL triggers (would indicate bug)

---

## Red Flags (Should NOT See)

### TSL Triggering Too Often
```
06:50:23 - TSL triggered
06:50:28 - TSL triggered  # ONLY 5 SECONDS LATER!
06:50:33 - TSL triggered  # SPAM!
```

**Means:** Initial Risk calculated incorrectly (likely restoration bug)
**Action:** STOP BOT, check logs, verify fix is in place

---

### TSL Steps Inconsistent
```
TSL #1: moved +1010.0 pips
TSL #2: moved +142.0 pips   # DIFFERENT SIZE!
```

**Means:** Formula using wrong base or initial_risk changed
**Action:** STOP BOT, review TSL calculation logs

---

### Initial Risk Changes Between TSL Triggers
```
TSL #1: Initial Risk: 0.20200 (2020.0 pips)
TSL #2: Initial Risk: 0.02800 (280.0 pips)   # CHANGED!
```

**Means:** Critical bug - initial_risk must NEVER change for a position
**Action:** STOP BOT IMMEDIATELY

---

### Prices with .2f Precision
```
LONG at 157.24, SL:157.04  # ONLY 2 DECIMALS
```

**Means:** Old code still running, fix not applied
**Action:** Check that correct version is deployed

---

### Wrong Position Window
```
[USDJPY_M2003_IB09:00-10:00] OCAE signal...
Position window: 150 min  # SHOULD BE 90 for OCAE!
```

**Means:** Using MAX window instead of variation-specific
**Action:** Check that fix is deployed

---

### Position Never Closes at Window End
```
Window end: 11:30:00 Tokyo
11:35:00 - Position still open  # SHOULD HAVE CLOSED!
```

**Means:** Time-based exit not working
**Action:** Check close_position() method exists

---

## Manual Verification Formulas

### For LONG Positions:

**TSL SL Step:**
```
New SL = Current SL + (TSL_SL x Initial Risk)
       = Current SL + (0.5 x Initial Risk)

Example:
  Current SL:  157.03800
  Initial Risk: 0.20200
  New SL = 157.03800 + (0.5 x 0.20200)
         = 157.03800 + 0.10100
         = 157.13900
```

**TSL TP Step:**
```
New TP = Current TP + (TSL_TARGET x Initial Risk)
       = Current TP + (0.5 x Initial Risk)

Example:
  Current TP:   157.34100
  Initial Risk:  0.20200
  New TP = 157.34100 + (0.5 x 0.20200)
         = 157.34100 + 0.10100
         = 157.44200
```

**Initial Risk (NEVER changes):**
```
Initial Risk = Entry - Initial SL

Example:
  Entry:      157.24000
  Initial SL: 157.03800
  Risk = 157.24000 - 157.03800 = 0.20200
```

### For SHORT Positions:

**TSL SL Step:**
```
New SL = Current SL - (TSL_SL x Initial Risk)
       = Current SL - (0.5 x Initial Risk)
```

**TSL TP Step:**
```
New TP = Current TP - (TSL_TARGET x Initial Risk)
       = Current TP - (0.5 x Initial Risk)
```

**Initial Risk (NEVER changes):**
```
Initial Risk = Initial SL - Entry

Example:
  Entry:      0.80710
  Initial SL: 0.80910
  Risk = 0.80910 - 0.80710 = 0.00200
```

---

## Position Window Quick Reference

| Variation | Trade Window | Position Window | Example Close Time (Tokyo) |
|-----------|--------------|-----------------|---------------------------|
| TCWE      | 60 min       | 60 min          | 11:00:00 (if entry 10:00) |
| OCAE      | 90 min       | 90 min          | 11:30:00 (if entry 10:00) |
| REV       | 120 min      | 120 min         | 12:00:00 (if entry 10:00) |
| REV_RB    | 150 min      | 150 min         | 12:30:00 (if entry 10:00) |

**Note:** Times are in exchange timezone, bot converts to UTC internally.

---

## Testing Checklist

### Before Production:
- [ ] Run bot 3-5 days on demo
- [ ] See at least 3-5 positions opened
- [ ] See at least 1-2 positions with TSL triggers
- [ ] See at least 1 position closed by time window
- [ ] Verify no "Initial Risk changed" errors
- [ ] Verify no TSL spam (>3 triggers per minute)
- [ ] Verify .5f precision in all logs
- [ ] Verify variation-specific windows used

### Optional (Restart Test):
- [ ] Open position manually in demo
- [ ] Let TSL trigger once
- [ ] Restart bot
- [ ] Verify warning messages appear
- [ ] Verify TSL doesn't continue triggering
- [ ] Verify position remains with MT5 SL

---

**Remember:** If anything looks wrong, STOP the bot and review logs before continuing!
