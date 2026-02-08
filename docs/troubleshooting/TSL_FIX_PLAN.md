# TSL Implementation Fix Plan - USDCHF Trade 11577281 Issues

**Date:** 2025-11-20
**Ticket:** 11577281 (USDCHF TCWE Demo Test)
**Status:** PLAN - AWAITING APPROVAL BEFORE IMPLEMENTATION

---

## Executive Summary

Analysis of USDCHF trade 11577281 revealed **critical bugs** in TSL implementation:

1. **CRITICAL: False TP Detection** - Bot logged "virtual TP hit at 0.81" when price never reached 0.80550
2. **TSL SL Formula Wrong** - Calculates from TP downward instead of from current SL upward
3. **TSL TP Formula Overly Complex** - Works but accumulates rounding errors
4. **Trade Window Bug** - Uses MAX window (150 min) instead of variation-specific (60 min for TCWE)
5. **Missing Time-Based Exit** - Doesn't close positions when trade window expires
6. **Logging Issues** - .2f formatting hides actual values, misleading analysis

---

## Issue #1: CRITICAL - False TP Detection

### Problem

**Log Entry:**
```
2025-11-19 21:30:25 - LONG virtual TP hit at 0.81, adjusting TSL...
2025-11-19 21:30:25 - New SL:0.80509, New virtual TP:0.81
```

**Reality:**
```
Price at 21:30:25: ~0.80536 (BID), ~0.80538 (ASK with 0.2 pip spread)
Virtual TP:        0.80550 (should be)
Result:            Price NEVER reached 0.80550 on Nov 19!
```

**Evidence:**
- Script `check_usdchf_price_movement.py` proved max price was 0.80536
- Even with spread, ASK would be ~0.80538, still below 0.80550
- Price actually reached 0.80550 around 00:00 on Nov 20

### Root Cause Investigation Needed

**Hypothesis 1: Virtual TP Value Wrong**
```python
# Initial TP calculation might be wrong
initial_tp = entry_price + (rr_target * risk)
# Expected: 0.80429 + (1.5 * 0.00080) = 0.80549 (rounds to 0.80550)
# Actual in tsl_state: ???
```

**Hypothesis 2: Price Check Logic Bug**
```python
# Line 635 in ib_strategy.py
if current_price >= virtual_tp:
    # Is current_price actually >= 0.80550 at 21:30?
    # Or is virtual_tp stored as something lower?
```

**Hypothesis 3: Logging Masks Real Values**
```python
# Line 638
logger.info(f"LONG virtual TP hit at {current_price:.2f}")
# Shows "0.81" but actual value might be 0.80540 or similar
```

### Investigation Steps

1. **Add detailed logging BEFORE TP check:**
   ```python
   logger.info(f"TSL Check: price={current_price:.5f}, virtual_tp={virtual_tp:.5f}, hit={current_price >= virtual_tp}")
   ```

2. **Log tsl_state contents when initialized:**
   ```python
   logger.info(f"TSL State initialized: {json.dumps(self.tsl_state, indent=2)}")
   ```

3. **Verify initial TP calculation:**
   ```python
   calculated_tp = entry_price + (rr_target * risk)
   logger.info(f"Initial TP calculation: {entry_price:.5f} + ({rr_target} * {risk:.5f}) = {calculated_tp:.5f}")
   ```

4. **Run test script to simulate exact conditions:**
   - Entry: 0.80429 at 20:16
   - Check: What should virtual_tp be?
   - Check: What was ASK price at 21:30:25?

### Fix

**Once root cause identified, implement appropriate fix:**

If virtual_tp calculation is wrong:
```python
# Ensure proper precision in initial TP calculation
virtual_tp = round(entry_price + (rr_target * risk), 5)  # 5 decimals for Forex
```

If price check logic is wrong:
```python
# Ensure using correct price (ASK for LONG, BID for SHORT)
if direction == "long":
    current_price = tick.get("ask")  # Exit price for LONG
else:
    current_price = tick.get("bid")  # Exit price for SHORT
```

If it's just logging:
```python
# Fix all price logging to .5f
logger.info(f"LONG virtual TP hit at {current_price:.5f}, virtual_tp={virtual_tp:.5f}")
```

---

## Issue #2: TSL SL Formula Wrong

### Current (WRONG) Implementation

**File:** `dual_v3/src/strategies/ib_strategy.py` (Lines 656-658)

```python
# WRONG: Calculates SL from TP downward
new_sl = virtual_tp - (tsl_sl * risk)
new_sl = max(current_sl, new_sl)
```

**Example (USDCHF 11577281):**
```
Entry:          0.80429
Current SL:     0.80349
Virtual TP:     0.80550 (when hit)
Risk:           0.00080
TSL_SL:         0.5

Current formula:
  new_sl = 0.80550 - (0.5 * 0.00080)
         = 0.80550 - 0.00040
         = 0.80510

Expected formula:
  new_sl = 0.80349 + (0.5 * 0.00080)
         = 0.80349 + 0.00040
         = 0.80389

Difference: 12.1 pips!
```

### Correct Implementation

**File:** `dual_v3/src/strategies/ib_strategy.py` (Lines 656-658)

```python
# CORRECT: Move SL incrementally from current position
if direction == "long":
    new_sl = current_sl + (tsl_sl * risk)
    new_sl = max(current_sl, new_sl)  # Ensure SL only moves up
else:  # short
    new_sl = current_sl - (tsl_sl * risk)
    new_sl = min(current_sl, new_sl)  # Ensure SL only moves down
```

### Testing

After fix, verify with same example:
```
Entry:          0.80429
Current SL:     0.80349
Virtual TP:     0.80550
Risk:           0.00080
TSL_SL:         0.5

New formula:
  new_sl = 0.80349 + (0.5 * 0.00080)
         = 0.80389

This protects:
  Profit = 0.80389 - 0.80429 = -0.00040 = -4 pips (still small loss)
  But better than original -8 pips risk!
```

---

## Issue #3: TSL TP Formula Overly Complex

### Current (WORKS BUT COMPLEX) Implementation

**File:** `dual_v3/src/strategies/ib_strategy.py` (Lines 660-665)

```python
# Calculate current target R
current_target_R = (virtual_tp - entry_price) / risk

# New target R = current + TSL_TARGET
new_target_R = current_target_R + tsl_target

# New virtual TP = entry + (new_target_R * risk)
new_virtual_tp = entry_price + (new_target_R * risk)
```

**Why Complex?**
- Requires 3 steps instead of 1
- Calculates cumulative R from entry
- Accumulates floating-point rounding errors
- Conceptually confusing (why calculate from entry when moving from current position?)

### Correct (SIMPLE) Implementation

```python
# CORRECT: Move TP incrementally from current position
if direction == "long":
    new_virtual_tp = virtual_tp + (tsl_target * risk)
else:  # short
    new_virtual_tp = virtual_tp - (tsl_target * risk)
```

**Why Better?**
- Single calculation
- Conceptually clear: "move TP by X from where it is now"
- No cumulative rounding errors
- Matches TSL SL formula style

### Mathematical Equivalence Proof

**Current approach:**
```
Step 0:
  virtual_tp = entry + (1.5 * risk) = entry + 0.0012
  current_R = (entry + 0.0012 - entry) / risk = 1.5
  new_R = 1.5 + 2.0 = 3.5
  new_tp = entry + (3.5 * risk) = entry + 0.0028

Step 1:
  virtual_tp = entry + 0.0028
  current_R = (entry + 0.0028 - entry) / risk = 3.5
  new_R = 3.5 + 2.0 = 5.5
  new_tp = entry + (5.5 * risk) = entry + 0.0044
```

**Simplified approach:**
```
Step 0:
  virtual_tp = entry + 0.0012
  new_tp = virtual_tp + (2.0 * risk) = entry + 0.0012 + 0.0016 = entry + 0.0028

Step 1:
  virtual_tp = entry + 0.0028
  new_tp = virtual_tp + (2.0 * risk) = entry + 0.0028 + 0.0016 = entry + 0.0044
```

Result: **Identical!** But simpler approach is clearer and faster.

---

## Issue #4: Trade Window Bug - Uses MAX Instead of Variation-Specific

### Current (WRONG) Implementation

**File:** `dual_v3/src/strategies/ib_strategy.py` (Lines 173-182)

```python
# Calculate max trade window across ALL variations
max_window = max(
    self.params["Reverse"]["TRADE_WINDOW"],  # 40
    self.params["OCAE"]["TRADE_WINDOW"],      # 90
    self.params["TCWE"]["TRADE_WINDOW"],      # 60
    self.params.get("REV_RB", {}).get("TRADE_WINDOW", 0)  # 150
)
# Uses 150 minutes for ALL trades!
self.trade_window_end = trade_start_time_local + timedelta(minutes=max_window)
```

**Problem:**
- TCWE trade should use 60-minute window
- Bot uses 150-minute window (from REV_RB variation)
- Results in trades staying open 90 minutes longer than intended

**Impact on USDCHF 11577281:**
```
Expected window end: 21:00 Almaty (10:00 NY + 60 min)
Actual window end:   22:30 Almaty (10:00 NY + 150 min)
Trade closed:        21:30 Almaty (SL hit)

Trade should have been force-closed at 21:00, not waited until 22:30!
```

### Correct Implementation

**Strategy:** Store variation when signal detected, use variation-specific window

```python
# When signal detected, store variation
signal = self._check_tcwe_signal(...)
if signal:
    self.active_variation = signal["variation"]  # "TCWE"
    self.trade_window_end = trade_start_time_local + timedelta(
        minutes=self.params[self.active_variation]["TRADE_WINDOW"]
    )
```

**Alternative:** Calculate window end when entering position

```python
def enter_position(self, signal: dict) -> bool:
    # Get variation-specific window
    variation = signal["variation"]
    trade_window = self.params[variation]["TRADE_WINDOW"]

    # Calculate window end for THIS specific trade
    self.trade_window_end = datetime.now(pytz.utc) + timedelta(minutes=trade_window)

    # Store variation in tsl_state
    self.tsl_state["variation"] = variation
    self.tsl_state["window_end"] = self.trade_window_end
```

---

## Issue #5: Missing Time-Based Exit Logic

### Current Implementation

**File:** `dual_v3/src/strategies/ib_strategy.py` (Lines 574-696)

```python
def update_position_state(self, position: Any, tick: dict) -> None:
    # Only implements TSL logic
    # NO check for trade window expiration!

    if direction == "long":
        if current_price >= virtual_tp:
            # TSL logic...
        # Missing: if current_time >= window_end: close position
```

**Problem:**
- Bot never closes positions when trade window expires
- Positions stay open indefinitely until TP/SL hit
- Contradicts original backtest strategy which closes at window end

### Original Backtest Logic

**File:** `dual_v3/project_info/dual_asset_ib_strategy.py` (Lines 531-534)

```python
# Close position at end of trade window if still open
if exit_reason is None:
    last_close = float(df_trade["close"].iat[-1])
    return {
        "exit_reason": "time",
        "exit_time": exit_time,
        "exit_price": last_close,
        # ...
    }
```

### Correct Implementation

**File:** `dual_v3/src/strategies/ib_strategy.py` (Add to update_position_state)

```python
def update_position_state(self, position: Any, tick: dict) -> None:
    """Update TSL state and check for exit conditions"""

    # Check if trade window expired
    current_time_utc = datetime.now(pytz.utc)
    if current_time_utc >= self.trade_window_end:
        logger.info(f"{self.log_prefix} Trade window expired, closing position {position.ticket}")

        # Close position at current price
        result = self.executor.close_position(position.ticket)
        if result:
            logger.info(f"{self.log_prefix} Position {position.ticket} closed at window end")
            self.tsl_state = None
            return
        else:
            logger.error(f"{self.log_prefix} Failed to close position {position.ticket} at window end")
            return

    # Continue with TSL logic...
    if self.tsl_state is None:
        self._restore_tsl_state_from_position(position)
    # ... rest of TSL logic ...
```

**Important:** Check window expiration BEFORE TSL logic to ensure positions close on time.

---

## Issue #6: Logging Issues - .2f Formatting Hides Values

### Current (MISLEADING) Implementation

**File:** `dual_v3/src/strategies/ib_strategy.py` (Multiple locations)

```python
# Line 638
logger.info(f"LONG virtual TP hit at {current_price:.2f}, adjusting TSL...")

# Line 659
logger.info(f"New SL:{new_sl:.2f}, New virtual TP:{new_virtual_tp:.2f}")

# Line 670
logger.info(f"SHORT virtual TP hit at {current_price:.2f}, adjusting TSL...")

# Line 690
logger.info(f"New SL:{new_sl:.2f}, New virtual TP:{new_virtual_tp:.2f}")
```

**Problem:**
```
Real value:  0.80710
Logged as:   0.81
Difference:  29 pips!

For Forex with 5 decimal places, .2f is completely inadequate.
```

### Correct Implementation

**Change all price logging to .5f:**

```python
# CORRECT: Show full precision
logger.info(f"LONG virtual TP hit at {current_price:.5f}, adjusting TSL...")
logger.info(f"New SL:{new_sl:.5f}, New virtual TP:{new_virtual_tp:.5f}")
logger.info(f"SHORT virtual TP hit at {current_price:.5f}, adjusting TSL...")
logger.info(f"New SL:{new_sl:.5f}, New virtual TP:{new_virtual_tp:.5f}")
```

**Also add detailed TSL calculation logging:**

```python
logger.info(f"{self.log_prefix} TSL Triggered:")
logger.info(f"  Entry Price:     {entry_price:.5f}")
logger.info(f"  Initial Risk:    {risk:.5f} ({risk * 10000:.1f} pips)")
logger.info(f"  Current SL:      {current_sl:.5f}")
logger.info(f"  Virtual TP Hit:  {virtual_tp:.5f}")
logger.info(f"  Current Price:   {current_price:.5f}")
logger.info(f"  TSL_SL:          {tsl_sl}")
logger.info(f"  TSL_TARGET:      {tsl_target}")
logger.info(f"  New SL:          {new_sl:.5f} (moved {(new_sl - current_sl) * 10000:.1f} pips)")
logger.info(f"  New Virtual TP:  {new_virtual_tp:.5f} (moved {(new_virtual_tp - virtual_tp) * 10000:.1f} pips)")
```

---

## Implementation Plan

### Phase 1: Investigation (CRITICAL - Do First)

**Objective:** Understand why bot logged "TP hit" when price never reached TP

**Tasks:**
1. Add detailed logging to TP check logic
2. Run simulation script for USDCHF on Nov 19 at 21:30:25
3. Check actual tsl_state values at that time
4. Verify ASK price vs virtual_tp
5. Determine root cause (calculation bug, price bug, or just logging)

**Files to Modify:**
- `dual_v3/src/strategies/ib_strategy.py` (add debug logging)

**Test Script:**
- `dual_v3/_temp/investigate_false_tp_hit.py` (create new)

**Success Criteria:**
- Understand exactly why bot thought TP was hit
- Have evidence (logs, calculations) proving root cause

### Phase 2: Fix TSL Calculation Formulas

**Objective:** Correct TSL SL and TP calculation logic

**Tasks:**
1. Fix TSL SL formula: `current_sl + (tsl_sl * risk)` for LONG
2. Fix TSL TP formula: `virtual_tp + (tsl_target * risk)` for LONG
3. Add mirror logic for SHORT positions
4. Ensure initial_risk is stored and never changes

**Files to Modify:**
- `dual_v3/src/strategies/ib_strategy.py` (lines 634-696)

**Before:**
```python
new_sl = virtual_tp - (tsl_sl * risk)
current_target_R = (virtual_tp - entry_price) / risk
new_target_R = current_target_R + tsl_target
new_virtual_tp = entry_price + (new_target_R * risk)
```

**After:**
```python
new_sl = current_sl + (tsl_sl * risk)
new_virtual_tp = virtual_tp + (tsl_target * risk)
```

**Test Cases:**
- Entry: 100, SL: 98, TP: 103, TSL_SL: 0.5, TSL_TARGET: 2.0
  - After 1st hit: SL=99, TP=107
  - After 2nd hit: SL=100, TP=111
  - After 3rd hit: SL=101, TP=115

### Phase 3: Fix Trade Window Logic

**Objective:** Use variation-specific window, close positions at window end

**Tasks:**
1. Store active variation when signal detected
2. Use variation-specific TRADE_WINDOW value
3. Add time-based exit check in update_position_state
4. Close position when trade window expires

**Files to Modify:**
- `dual_v3/src/strategies/ib_strategy.py` (lines 173-182, 574-696)

**Changes:**
```python
# When signal detected
self.active_variation = signal["variation"]
variation_window = self.params[signal["variation"]]["TRADE_WINDOW"]
self.trade_window_end = trade_start_time_local + timedelta(minutes=variation_window)

# In update_position_state (BEFORE TSL logic)
if current_time_utc >= self.trade_window_end:
    self.executor.close_position(position.ticket)
    self.tsl_state = None
    return
```

**Test Cases:**
- TCWE: window = 60 min, closes at IB_end + 60
- OCAE: window = 90 min, closes at IB_end + 90
- REV_RB: window = 150 min, closes at IB_end + 150

### Phase 4: Fix Logging

**Objective:** Show accurate prices in logs for analysis

**Tasks:**
1. Change all price formatting from .2f to .5f
2. Add detailed TSL calculation logs
3. Log tsl_state contents on initialization
4. Add pip movement calculations to logs

**Files to Modify:**
- `dual_v3/src/strategies/ib_strategy.py` (lines 638, 659, 670, 690, and others)

**Changes:**
```python
# Replace all .2f with .5f for prices
logger.info(f"LONG virtual TP hit at {current_price:.5f}")
logger.info(f"New SL:{new_sl:.5f}, New virtual TP:{new_virtual_tp:.5f}")

# Add detailed logging
logger.info(f"TSL: SL moved from {current_sl:.5f} to {new_sl:.5f} (+{(new_sl - current_sl) * 10000:.1f} pips)")
```

### Phase 5: Testing

**Objective:** Verify all fixes work correctly

**Test 1: TSL Calculation**
- Script: `dual_v3/_temp/test_tsl_formulas.py`
- Verify: SL and TP move correctly through multiple steps
- Check: Values match reference examples

**Test 2: Trade Window**
- Script: `dual_v3/_temp/test_trade_window.py`
- Verify: Variation-specific windows used
- Check: Positions close at window end

**Test 3: USDCHF Replay**
- Script: `dual_v3/_temp/replay_usdchf_11577281.py`
- Replay: Nov 19 trade with fixes
- Compare: New behavior vs old behavior
- Verify: No false TP detection, correct SL levels

**Test 4: Live Demo**
- Run bot in demo mode for 3-5 days
- Monitor: TSL behavior on real trades
- Check: Logs show correct prices, timing, exits

---

## Files to Create

1. **`dual_v3/_temp/investigate_false_tp_hit.py`** - Debug script for Phase 1
2. **`dual_v3/_temp/test_tsl_formulas.py`** - Unit tests for TSL calculations
3. **`dual_v3/_temp/test_trade_window.py`** - Tests for window logic
4. **`dual_v3/_temp/replay_usdchf_11577281.py`** - Replay actual trade with fixes

---

## Files to Modify

1. **`dual_v3/src/strategies/ib_strategy.py`**
   - Lines 173-182: Fix trade window calculation
   - Lines 574-696: Fix TSL logic and add time-based exit
   - Multiple lines: Fix logging (.2f → .5f)

2. **`dual_v3/CHANGELOG.md`**
   - Document all changes

---

## Risk Assessment

### High Risk Changes
- **TSL SL Formula**: Changes actual SL placement, affects profit protection
- **Time-Based Exit**: New behavior, closes positions that previously stayed open

### Medium Risk Changes
- **TSL TP Formula**: Simplified but mathematically equivalent
- **Trade Window Logic**: Changes timing but matches backtest

### Low Risk Changes
- **Logging**: Cosmetic, doesn't affect trading logic

### Mitigation Strategy
1. Test all changes in demo account first
2. Run for minimum 5 days before considering live
3. Monitor logs closely for any anomalies
4. Keep backup of current code
5. Be ready to revert if issues found

---

## Success Criteria

### Fix is Successful If:
- [ ] No false TP detections in logs
- [ ] TSL SL values match hand calculations
- [ ] TSL TP values match hand calculations
- [ ] Positions close at variation-specific window end
- [ ] Logs show full precision prices (.5f)
- [ ] All test scripts pass
- [ ] 5-day demo run shows expected behavior

### Fix is Failed If:
- [ ] Still seeing false TP hits
- [ ] SL values don't match expected
- [ ] Positions don't close at window end
- [ ] Any position enters loss when it should be breakeven+
- [ ] Bot crashes or errors

---

## Timeline

- **Phase 1 (Investigation):** 2-4 hours
- **Phase 2 (TSL Fix):** 2-3 hours
- **Phase 3 (Window Fix):** 1-2 hours
- **Phase 4 (Logging Fix):** 1 hour
- **Phase 5 (Testing):** 4-6 hours
- **Demo Run:** 5-7 days

**Total:** 1-2 days coding + 5-7 days validation

---

## Approval Required

**Before proceeding with implementation:**
1. User reviews this plan
2. User approves investigation approach (Phase 1)
3. User confirms fixes align with trading strategy goals
4. User agrees to demo testing timeline

**Critical Questions for User:**
1. Should we investigate false TP hit first (Phase 1) before fixing formulas?
2. Is it acceptable to close positions at trade window end (matches backtest but changes current bot behavior)?
3. Are there any trades currently open that would be affected by these changes?
4. Should we implement fixes incrementally (phase by phase) or all at once?

---

## Notes

- TSL logic reference saved to: `dual_v3/project_info/TSL_LOGIC_REFERENCE.md`
- All analysis files saved to: `dual_v3/_temp/`
- This plan document: `dual_v3/_temp/TSL_FIX_PLAN.md`
