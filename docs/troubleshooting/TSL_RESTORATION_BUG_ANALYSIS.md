# TSL State Restoration Bug - Critical Analysis

**Date:** 2025-11-20
**Issue:** Bot restart causes incorrect TSL behavior
**Severity:** CRITICAL
**Status:** FIXED

---

## The Problem

When bot restarts while position is open, `_restore_tsl_state_from_position()` tries to reconstruct TSL state from MT5 position data.

**CRITICAL FLAW:**
```python
entry_price = position.price_open  # ✓ Correct
current_sl = position.sl           # ✓ Correct

# ❌ WRONG! After TSL triggers, current_sl != initial_sl
sl_distance = abs(entry_price - current_sl)
```

**Why This is Wrong:**

MT5 only stores:
- `position.price_open` (entry) - never changes
- `position.sl` (current SL) - changes with TSL
- `position.tp` (current TP) - we use virtual TP, so this is 0

MT5 does NOT store:
- Initial SL (the original SL when position opened)
- Initial risk (entry - initial SL)
- TSL history (how many times TSL triggered)

---

## Real Example from Logs

### Original Trade (06:44:03)
```
Entry:           157.24000
Initial SL:      157.03800
Initial Risk:    0.20200 (202.0 pips)
Initial TP:      157.34100 (virtual)

TSL Parameters:
  TSL_SL:        0.5
  TSL_TARGET:    0.5
```

### After TSL Trigger #1 (06:50:23)
```
Price reached:   157.34700 (TP hit)

New SL:          157.13900 (+101 pips from 157.03800)
New Virtual TP:  157.44200 (+101 pips from 157.34100)

MT5 Position:
  price_open:    157.24000 (unchanged)
  sl:            157.13900 (NEW VALUE)
  tp:            0.0 (virtual TP not in MT5)
```

### After TSL Trigger #2 (06:59:09)
```
Price reached:   157.44300 (TP hit again)

New SL:          157.24000 (+101 pips, now at BREAKEVEN!)
New Virtual TP:  157.54300 (+101 pips)

MT5 Position:
  price_open:    157.24000 (unchanged)
  sl:            157.24000 (NEW VALUE, = entry!)
  tp:            0.0
```

### Bot Restart at 10:59:24

**Bot tries to restore TSL state:**
```python
entry_price = position.price_open  # = 157.24000 ✓
current_sl = position.sl           # = 157.24000 ✓

# Calculate "initial risk"
sl_distance = abs(entry_price - current_sl)
          = abs(157.24000 - 157.24000)
          = 0.00000  ❌ COMPLETELY WRONG!

# Should be: 0.20200 (original risk)
```

**But in actual logs we see 0.02800... why?**

Looking closer at the log:
```
Entry: 157.26800
Initial SL: 157.24000
Initial Risk: 0.02800
```

This means `position.sl` was 157.26800 (slightly higher than entry after more TSL steps).

**Calculated risk:**
```
sl_distance = abs(157.26800 - 157.24000) = 0.02800
```

**Real original risk:** 0.20200

**Error factor:** 0.20200 / 0.02800 = 7.2x too small!

---

## Consequences

### 1. TSL Steps Too Small

**Correct TSL step size:**
```
TSL_SL = 0.5 * 0.20200 = 0.10100 (101 pips)
```

**Incorrect TSL step size (with wrong risk):**
```
TSL_SL = 0.5 * 0.02800 = 0.01400 (14 pips)
```

**Result:** TSL moves in 14-pip steps instead of 101-pip steps!

### 2. Constant TSL Triggering

Price volatility: ~20-30 pips normal movement
TSL step: 14 pips (too small)

**Result:** TP gets hit every 5 seconds!

**Evidence from logs:**
```
10:59:24 - TSL triggered
10:59:29 - TSL triggered (5 sec later)
10:59:34 - TSL triggered (5 sec later)
10:59:40 - TSL triggered (6 sec later)
... 20+ times in 2 minutes!
```

### 3. Excessive MT5 API Calls

Each TSL trigger = 1 `modify_position()` call to MT5

20 triggers in 2 minutes = **10 API calls per minute**

This is:
- Unnecessary server load
- Potential rate limiting
- Wasted network bandwidth
- Increased slippage risk

### 4. Virtual TP Incorrect

**Correct initial virtual TP:**
```
virtual_tp = entry + (RR_TARGET * initial_risk)
           = 157.24000 + (1.5 * 0.20200)
           = 157.54300
```

**Incorrect (with wrong risk):**
```
virtual_tp = 157.26800 + (1.5 * 0.02800)
           = 157.31000
```

**Difference:** 23.3 pips (157.54300 - 157.31000)

This means bot thinks TP is much closer than it really should be.

### 5. Position Risk Exposure

If bot tried to open a NEW position with wrong risk:
```
Correct lot size:
  Risk: 1% of 53090.70 = 530.91
  SL distance: 0.20200 (202 pips)
  Risk per lot: 20200
  Lots: 530.91 / 20200 = 0.026 ≈ 0.03 lots

Incorrect (with wrong risk):
  Risk: 1% of 53090.70 = 530.91
  SL distance: 0.02800 (28 pips)
  Risk per lot: 2800
  Lots: 530.91 / 2800 = 0.19 ≈ 0.2 lots

ERROR: 6.7x MORE LOTS! Massive over-leverage!
```

Fortunately, position was already open, so lot size didn't change.

---

## Why Restoration is Impossible

**What we need to restore TSL state:**
- entry_price ✓ (MT5 stores)
- initial_sl ❌ (MT5 does NOT store)
- initial_risk ❌ (can't calculate without initial_sl)
- current_tp ❌ (virtual TP not in MT5)
- tsl_target ✓ (from strategy params)
- tsl_sl ✓ (from strategy params)

**Missing critical data:** initial_sl, initial_risk

**Cannot be calculated from:**
- current SL (changes with TSL)
- entry price alone
- current price
- MT5 history (doesn't store original SL)

---

## The Fix

**SOLUTION: Do NOT restore TSL state after bot restart.**

```python
def _restore_tsl_state_from_position(self, position):
    # CRITICAL ISSUE: Cannot reliably restore TSL state after bot restart!
    # MT5 only stores current SL, not original SL
    # After TSL triggers, current SL != initial SL

    logger.warning("Cannot restore TSL state reliably after bot restart")
    logger.warning(f"Position {position.ticket} will be managed by MT5 SL only")
    logger.warning("TSL tracking disabled for this position")

    # Do NOT set tsl_state - this prevents TSL from triggering
    # Position will remain protected by current MT5 SL
    return
```

**Why this is safe:**

1. Position is protected by current MT5 SL (set before restart)
2. If price hits SL, MT5 will close position automatically
3. No risk of incorrect lot sizing (position already open)
4. No excessive API calls
5. Clean, simple, predictable behavior

**Trade-off:**

- TSL won't continue after restart
- Position won't trail further
- But position is still protected by last SL before restart
- This is acceptable for rare restart scenarios

---

## Alternative Solution (Future Enhancement)

**Persist TSL state to file:**

When TSL triggers, save state to:
```
dual_v3/state/tsl_state_11580505.json
{
  "ticket": 11580505,
  "entry_price": 157.24000,
  "initial_sl": 157.03800,
  "initial_risk": 0.20200,
  "current_sl": 157.24000,
  "current_tp": 157.54300,
  "tsl_target": 0.5,
  "tsl_sl": 0.5,
  "variation": "OCAE",
  "position_window_end": "2025-11-20T02:30:00Z"
}
```

On restart:
1. Check if file exists for position
2. Load TSL state from file
3. Validate data (entry matches, ticket matches)
4. Continue TSL tracking

**Benefits:**
- Full TSL tracking after restart
- Correct risk calculations
- No loss of trailing functionality

**Drawbacks:**
- Added complexity
- File I/O overhead
- Need to clean up old files
- Potential file corruption issues

**Recommendation:** Implement this ONLY if bot restarts are frequent. For now, simple solution is sufficient.

---

## Testing

**Test Case 1: Fresh position (no restart)**
- Entry: 100, SL: 98, Risk: 2
- TSL triggers at 103
- New SL: 98 + (0.5 * 2) = 99 ✓
- New TP: 103 + (2.0 * 2) = 107 ✓
- **PASS**

**Test Case 2: Restart after TSL #1**
- Original: Entry=100, Initial SL=98, Risk=2
- After TSL: Entry=100, Current SL=99
- Bot restart
- Old behavior: Calculate risk = 100 - 99 = 1 ❌
- New behavior: Don't restore TSL, log warning ✓
- **PASS**

**Test Case 3: Restart at breakeven**
- Original: Entry=100, Initial SL=98, Risk=2
- After TSL #2: Entry=100, Current SL=100 (breakeven)
- Bot restart
- Old behavior: Calculate risk = 100 - 100 = 0 ❌ CATASTROPHIC
- New behavior: Don't restore TSL, log warning ✓
- **PASS**

---

## Files Changed

**`dual_v3/src/strategies/ib_strategy.py`:**
- Lines 584-604: Replaced restoration logic with warning and return
- Prevents incorrect TSL behavior after restart
- Leaves position managed by MT5 SL

**`dual_v3/CHANGELOG.md`:**
- Added entry documenting the bug and fix

---

## Recommendations

1. **DO NOT restart bot** while positions are open (if possible)
2. **If restart needed:**
   - Positions will remain open
   - Protected by last SL before restart
   - TSL won't continue tracking
   - Manually monitor if needed
3. **For production:**
   - Consider implementing file-based state persistence
   - Add monitoring/alerts for restarts with open positions
   - Log warnings prominently

---

**Summary:** Critical bug fixed. Bot will no longer attempt impossible TSL state restoration, preventing catastrophic miscalculations and excessive API calls.
