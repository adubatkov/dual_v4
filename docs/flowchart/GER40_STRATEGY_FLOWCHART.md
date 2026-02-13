# GER40 IB TRADING STRATEGY -- ALGORITHMIC FLOWCHART

**Instrument**: GER40 (DAX Index)
**Timeframe**: M2 (2-minute) for signals, M1 (1-minute) for position management
**Strategy Type**: Initial Balance (IB) breakout/reversal with fractal overlay
**Session**: European (Frankfurt/Berlin)
**Baseline**: dual_v4 PROD parameters (uniform across all variations)

This document describes the complete algorithm for the GER40 IB trading strategy.
It is self-contained: a trader who has never seen the source code should be able to
reconstruct the full decision logic from this document alone.

Maximum one trade per day. All times in Europe/Berlin unless stated otherwise.

---

## TABLE OF CONTENTS

1. [Phase 0: Daily Reset](#phase-0-daily-reset)
2. [Phase 1: IB Formation](#phase-1-ib-initial-balance-formation)
3. [Phase 2: Trade Window](#phase-2-trade-window-activation)
4. [Phase 3: Signal Detection](#phase-3-signal-detection-priority-cascade)
5. [Phase 4: Trade Execution](#phase-4-trade-execution)
6. [Phase 5: Position Management](#phase-5-position-management)
7. [Phase 6: Fractal Overlay](#phase-6-fractal-overlay-additional-sl-layer)
8. [Phase 7: Exit Conditions](#phase-7-exit-conditions)
9. [Parameter Reference](#parameter-reference-table)

---

## PHASE 0: DAILY RESET

```
TRIGGER: Midnight (00:00) Europe/Berlin

    |
    v
    Clear IB values:
        IBH  = None
        IBL  = None
        EQ   = None
    |
    v
    Clear trade window state
    |
    v
    Clear TSL (trailing stop-loss) state
    |
    v
    State -> AWAITING_IB_CALCULATION
    |
    v
    [Wait for Phase 1]
```

All tracking variables from the previous day are wiped. The bot enters a clean
state ready to observe the new session's Initial Balance formation.

---

## PHASE 1: IB (INITIAL BALANCE) FORMATION

The Initial Balance is the price range established during the first hour of the
regular trading session. It is the foundation for all subsequent signal logic.

```
IB Window: 08:00 - 09:00 Europe/Berlin
Candle size: M2 (2-minute bars)

    |
    v
    Wait until local time >= 09:00 Berlin
    |
    v
    Collect all M2 candles with timestamps in [08:00, 09:00)
    |
    v
    IF no candles found:
    |   -> State = DAY_ENDED (skip this day entirely)
    |   -> [END]
    |
    ELSE:
    |
    v
    Calculate IB metrics:
        IBH      = max(high) across all IB candles
        IBL      = min(low) across all IB candles
        EQ       = (IBH + IBL) / 2
        IB_Range = IBH - IBL
    |
    v
    IF IB_Range <= 0:
    |   -> State = DAY_ENDED (degenerate range, skip day)
    |   -> [END]
    |
    ELSE:
    |   -> State = AWAITING_TRADE_WINDOW
    |   -> [Proceed to Phase 2]
```

### Key Definitions

| Term | Definition |
|------|------------|
| IBH | IB High -- the highest price reached during 08:00-09:00 |
| IBL | IB Low -- the lowest price reached during 08:00-09:00 |
| EQ | Equilibrium -- the midpoint of the IB range |
| IB Range | The total width of the IB in price units (IBH - IBL) |

---

## PHASE 2: TRADE WINDOW ACTIVATION

The trade window defines the time period during which the bot actively scans for
entry signals. No signals are generated outside this window.

```
    trade_window_start = IB_END + IB_WAIT
                       = 09:00 + 0 min
                       = 09:00 Berlin

    trade_window_end   = trade_window_start + max(TRADE_WINDOW across all variations)
                       = 09:00 + 120 min
                       = 11:00 Berlin
    |
    v
    Wait until current_time >= trade_window_start
    |
    v
    State -> IN_TRADE_WINDOW
    |
    v
    [Proceed to Phase 3 -- scan for signals]
```

The trade window runs from 09:00 to 11:00 Berlin time (2 hours). If no signal
is found by 11:00, the day ends with no trade.

---

## PHASE 3: SIGNAL DETECTION (PRIORITY CASCADE)

Signal detection runs on each new M1 (1-minute) candle while the trade window
is open. A priority cascade determines which variation is evaluated first.

### 3.0 Pre-Checks (Every M1 Candle)

```
    New M1 candle arrives
    |
    v
    CHECK 1: Is current_time >= trade_window_end (11:00)?
    |   YES -> State = DAY_ENDED, no trade today -> [END]
    |   NO  -> continue
    |
    v
    CHECK 2: News filter -- is current time within 2 minutes
             of a high-impact EUR or USD news event?
    |   YES -> Block this candle (no retry on this candle)
    |           Wait for next M1 candle -> [loop back to top]
    |   NO  -> continue
    |
    v
    Fetch 500 most recent M2 candles (for sweep/breakout analysis)
    |
    v
    [Enter Priority Cascade]
```

### Priority Cascade Order

```
    Priority 1: REVERSE   (highest)
        |
        IF signal found -> [Phase 4]
        IF no signal:
        |
        v
    Priority 2: OCAE
        |
        IF signal found -> [Phase 4]
        IF no signal:
        |
        v
    Priority 3: TCWE
        |
        IF signal found -> [Phase 4]
        IF no signal:
        |
        v
    Priority 4: REV_RB   [DISABLED -- never executes]
        |
        v
    No signal this candle -> wait for next M1 candle -> [loop]
```

The first variation that produces a valid signal wins. Lower-priority variations
are NOT evaluated once a higher-priority signal is found.

---

### 3.1 REVERSE (Priority 1 -- Highest)

The Reverse variation detects IB boundary sweeps (false breakouts) followed by
a Change in State of Delivery (CISD), signaling a reversal back into the range.

```
    buffer = IB_BUFFER_PCT * IB_Range = 0.0 * IB_Range = 0
```

#### STEP A -- Scan for IB Sweeps

Iterate through M2 candles inside the trade window. For each candle (open, high, low, close):

```
    FOR each M2 candle in trade_window:
    |
    v
    INVALIDATION CHECK (upper):
        IF open > IBH + buffer AND close > IBH + buffer:
        |   -> Both body endpoints above IBH
        |   -> Stop scanning for upper sweeps
        |      (all prior upper sweeps remain valid)
    |
    INVALIDATION CHECK (lower):
        IF open < IBL - buffer AND close < IBL - buffer:
        |   -> Both body endpoints below IBL
        |   -> Stop scanning for lower sweeps
    |
    v
    UPPER SWEEP detection (SHORT setup):
        IF high > IBH
           AND open <= IBH + buffer
           AND close <= IBH + buffer:
        |   -> Wick pierced above IBH, but body stayed at or below IBH
        |   -> Record as UPPER sweep
    |
    LOWER SWEEP detection (LONG setup):
        IF low < IBL
           AND open >= IBL - buffer
           AND close >= IBL - buffer:
        |   -> Wick pierced below IBL, but body stayed at or above IBL
        |   -> Record as LOWER sweep
```

#### STEP B -- Group Consecutive Sweeps

```
    Group consecutive sweeps in the same direction:
    |
    Upper sweep groups:
    |   ext = max(high) of all candles in the group
    |
    Lower sweep groups:
    |   ext = min(low) of all candles in the group
```

#### STEP C -- Find CISD (Change in State of Delivery)

For each sweep group, search for confirmation that price has reversed:

```
    FOR each sweep group:
    |
    v
    STEP C.1 -- Find the reference candle (last opposite candle before sweep):
    |
    |   IF upper sweep (SHORT setup):
    |       Find last BEARISH candle (close < open) before the sweep group
    |       -> Record its LOW as the reference level
    |
    |   IF lower sweep (LONG setup):
    |       Find last BULLISH candle (close > open) before the sweep group
    |       -> Record its HIGH as the reference level
    |
    v
    STEP C.2 -- Scan candles AFTER the sweep group for CISD:
    |
    |   IF SHORT setup:
    |       First candle where close < reference_candle.low
    |       -> CISD confirmed
    |
    |   IF LONG setup:
    |       First candle where close > reference_candle.high
    |       -> CISD confirmed
    |
    v
    STEP C.3 -- Distance check:
    |
    |   IF CISD candle close is >= 50% of IB_Range away from the IB boundary:
    |       -> REJECT this CISD (too far from IB -- likely not a reversal)
    |       -> Stop searching in this sweep group
    |
    |   IF distance < 50% of IB_Range:
    |       -> CISD is VALID
    |       -> Return signal
```

#### REVERSE Entry Calculation

```
    IF valid CISD found:
    |
    v
    Entry price = OPEN of the candle immediately AFTER the CISD candle
    |
    v
    SL (stop-loss):
        SHORT -> SL = max(high) of sweep group (the sweep extreme)
        LONG  -> SL = min(low) of sweep group (the sweep extreme)
    |
    v
    SL VALIDATION:
        IF SL is on wrong side of entry (e.g., SL < entry for SHORT):
        |   -> Flip SL to correct side at same distance
        |      distance = |entry - SL|
        |      SHORT: SL = entry + distance
        |      LONG:  SL = entry - distance
    |
    v
    MIN_SL CHECK:
        IF |entry - SL| < entry * MIN_SL_PCT (0.0015):
        |   -> Widen SL to entry * MIN_SL_PCT on the correct side
    |
    v
    TP (take-profit):
        raw_tp = entry +/- RR_TARGET * |entry - SL|
               = entry +/- 1.0 * risk
    |
    v
    VIRTUAL TP MODE (TSL_TARGET = 1.0 > 0):
        -> Set broker TP = 0 (no hard TP in broker)
        -> Track TP internally as virtual_tp = raw_tp
    |
    v
    -> SIGNAL FOUND. Proceed to Phase 4.
    -> DO NOT evaluate OCAE or TCWE.
```

---

### 3.2 OCAE (Priority 2 -- Open-Close Above/Below EQ)

The OCAE variation detects a clean IB breakout that occurs AFTER price has
touched the equilibrium (EQ) during the trade window. The EQ touch confirms
that the range was explored before the breakout.

```
    buffer   = IB_BUFFER_PCT * IB_Range = 0
    max_dist = MAX_DISTANCE_PCT * IB_Range = 1.0 * IB_Range
```

Only evaluated if REVERSE returned no signal.

#### STEP A -- Find First Breakout Bar

```
    FOR each M2 candle in trade_window (chronological):
    |
    v
    LONG breakout:
        IF close > IBH + buffer
           AND (close - IBH) <= max_dist:
        |   -> First valid upward breakout found
    |
    SHORT breakout:
        IF close < IBL - buffer
           AND (IBL - close) <= max_dist:
        |   -> First valid downward breakout found
    |
    v
    Return the FIRST candle that meets either condition
    |
    IF no breakout candle found -> OCAE INVALID -> [skip to TCWE]
```

#### STEP B -- EQ Touch Validation

```
    Check ALL M2 candles from trade_window_start up to and including
    the breakout candle:
    |
    v
    EQ_touched = any candle where (low <= EQ AND high >= EQ)?
    |
    IF EQ NOT touched:
    |   -> OCAE INVALID (breakout occurred without EQ exploration)
    |   -> [Skip to TCWE]
    |
    IF EQ touched:
    |   -> OCAE VALID
    |   -> Continue
```

#### STEP C -- Wait for Candle Close

```
    The breakout candle must be fully closed (M2 candle complete).
    Do not act on partial/forming candles.
```

#### OCAE Entry Calculation

```
    IF valid OCAE breakout found:
    |
    v
    Entry price = CLOSE of the breakout candle
    |
    v
    SL:
        STOP_MODE = "ib_start"
        -> SL = open price of the FIRST candle in the trade window
           (i.e., the open of the 09:00 M2 candle)
    |
    v
    SL VALIDATION:
        IF SL on wrong side of entry -> flip to correct side at same distance
    |
    v
    MIN_SL CHECK:
        IF |entry - SL| < entry * 0.0015 -> widen to minimum
    |
    v
    TP:
        raw_tp = entry +/- RR_TARGET * |entry - SL|
               = entry +/- 1.0 * risk
    |
    v
    VIRTUAL TP MODE (TSL_TARGET = 1.0 > 0):
        -> Broker TP = 0
        -> Internal virtual_tp = raw_tp
    |
    v
    -> SIGNAL FOUND. Proceed to Phase 4.
    -> DO NOT evaluate TCWE.
```

---

### 3.3 TCWE (Priority 3 -- Two-Candle Weighted Extension)

The TCWE variation detects a double breakout -- two successive candles breaking
further beyond the IB boundary -- specifically when EQ has NOT been touched.
This captures momentum breakouts where the market never returned to equilibrium.

```
    buffer   = IB_BUFFER_PCT * IB_Range = 0
    max_dist = MAX_DISTANCE_PCT * IB_Range = 1.0 * IB_Range
```

Only evaluated if REVERSE and OCAE both returned no signal.

#### STEP A -- Find First Breakout Bar (Without EQ Touch)

```
    FOR each M2 candle in trade_window (chronological):
    |
    v
    Check EQ touch from trade_window_start up to this candle:
        IF EQ was touched at ANY point -> TCWE CANCELLED entirely -> [END]
    |
    v
    LONG: close > IBH + buffer AND (close - IBH) <= max_dist
    SHORT: close < IBL - buffer AND (IBL - close) <= max_dist
    |
    IF valid first breakout found (and EQ never touched) -> record it
    IF no breakout found -> TCWE INVALID -> [END]
```

#### STEP B -- Find Second (Further) Breakout Bar

```
    Starting from the candle AFTER the first breakout:
    |
    FOR each subsequent M2 candle:
    |
    v
    CHECK: Has EQ been touched since trade_window_start?
    |   YES -> TCWE CANCELLED entirely -> [END]
    |   NO  -> continue
    |
    v
    LONG second breakout:
        IF close > IBH + buffer
           AND close > first_breakout.close  (further than first)
           AND (close - IBH) <= max_dist:
        |   -> Second breakout found
    |
    SHORT second breakout:
        IF close < IBL - buffer
           AND close < first_breakout.close  (further than first)
           AND (IBL - close) <= max_dist:
        |   -> Second breakout found
    |
    IF no second breakout found -> TCWE INVALID -> [END]
```

#### STEP C -- Wait for Candle Close

```
    The second breakout candle must be fully closed (M2 complete).
```

#### TCWE Entry Calculation

```
    IF valid second breakout found:
    |
    v
    Entry price = CLOSE of the second breakout candle
    |
    v
    SL:
        STOP_MODE = "ib_start"
        -> SL = open of first candle in trade window (same as OCAE)
    |
    v
    SL VALIDATION + MIN_SL CHECK (same rules as above)
    |
    v
    TP:
        raw_tp = entry +/- 1.0 * |entry - SL|
    |
    v
    VIRTUAL TP MODE:
        -> Broker TP = 0, virtual_tp = raw_tp
    |
    v
    -> SIGNAL FOUND. Proceed to Phase 4.
```

---

### 3.4 REV_RB (Priority 4 -- DISABLED)

```
    REV_RB_ENABLED = False

    This variation is DISABLED for all GER40 configurations.
    It is never evaluated and never produces a signal.
```

---

### Signal Detection Summary

```
    IF no variation produced a signal on this M1 candle:
    |   -> Wait for next M1 candle
    |   -> Re-enter Phase 3 pre-checks
    |   -> Repeat until trade_window_end (11:00) or signal found
    |
    IF trade_window_end reached with no signal:
    |   -> State = DAY_ENDED
    |   -> No trade today
```

---

## PHASE 4: TRADE EXECUTION

Once a signal is generated by any variation, the bot prepares and places the order.

```
    Signal received: {direction, entry_price, SL, TP, variation}
    |
    v
    STEP 1 -- Position sizing:
        lots = risk_amount / (|entry_price - SL| * point_value_per_lot)
    |
    v
    STEP 2 -- Margin validation:
        required_margin = lots * margin_per_lot
        IF required_margin > 40% of account_balance:
        |   -> REJECT trade (margin too high)
        |   -> [END -- no trade today]
    |
    v
    STEP 3 -- Place order:
        Order type:  MARKET (at entry_price)
        Stop-loss:   SL (set in broker -- visible to broker)
        Take-profit: 0  (virtual mode -- NOT set in broker)
    |
    v
    STEP 4 -- Update state:
        State -> POSITION_OPEN
    |
    v
    STEP 5 -- Initialize TSL tracking:
        tsl_state = {
            variation:           <which variation triggered>
            tsl_target:          1.0   (TP step size in R)
            tsl_sl:              1.0   (SL step size in R)
            initial_sl:          SL
            initial_tp:          virtual_tp
            current_tp:          virtual_tp
            entry_price:         entry_price
            tsl_triggered:       False
            position_window_end: trade_window_start + TRADE_WINDOW (120 min)
        }
    |
    v
    [Proceed to Phase 5 -- position management]
```

---

## PHASE 5: POSITION MANAGEMENT

Position management runs on every new M1 candle while a position is open.
Three checks are performed in strict order on each candle.

### 5.1 Time Exit Check (HIGHEST PRIORITY -- checked first)

```
    IF current_time >= position_window_end:
    |   -> Close position at MARKET (last close price)
    |   -> Exit reason = "time"
    |   -> [Proceed to Phase 7 cleanup]
    |
    ELSE:
    |   -> Continue to 5.2
```

### 5.2 Virtual TP Check (Trailing Stop-Loss Ratchet)

The virtual TP is not placed in the broker. Instead, the bot checks each candle
to see if price has reached the virtual TP level. When hit, both SL and TP are
ratcheted by fixed R-unit increments.

```
    IF TSL_TARGET > 0 (= 1.0 in PROD -- virtual mode is ACTIVE):
    |
    v
    Has virtual TP been hit this candle?
    |
    |   LONG:  current_price >= virtual_tp ?
    |   SHORT: current_price <= virtual_tp ?
    |
    IF NOT hit -> skip to 5.3
    |
    IF HIT:
    |
    v
    risk = |entry_price - initial_sl|    (FIXED -- never recalculated)
    |
    v
    LONG RATCHET:
        new_sl = current_sl + (TSL_SL * risk) = current_sl + (1.0 * risk)
        new_sl = max(current_sl, new_sl)       -- SL only moves UP (tighter)
        new_virtual_tp = virtual_tp + (TSL_TARGET * risk) = virtual_tp + (1.0 * risk)
    |
    SHORT RATCHET:
        new_sl = current_sl - (TSL_SL * risk) = current_sl - (1.0 * risk)
        new_sl = min(current_sl, new_sl)       -- SL only moves DOWN (tighter)
        new_virtual_tp = virtual_tp - (TSL_TARGET * risk) = virtual_tp - (1.0 * risk)
    |
    v
    Update broker SL to new_sl
    Update internal virtual_tp to new_virtual_tp
    |
    v
    TRAIL CONFLICT CHECK:
        IF this same candle ALSO hits the new SL after ratchet:
        |   (LONG: candle low <= new_sl)
        |   (SHORT: candle high >= new_sl)
        |
        |   -> DO NOT exit on this candle
        |   -> Defer exit to OPEN of NEXT candle
        |   -> Exit reason = "trail_conflict"
        |   -> [Phase 7]
    |
    v
    Continue to 5.3
```

### 5.3 SL/TP Hit Check

```
    LONG position:
        IF candle low <= effective_SL:
        |   -> Exit at SL level
        |   -> Exit reason = "sl"
        |   -> [Phase 7]
    |
    SHORT position:
        IF candle high >= effective_SL:
        |   -> Exit at SL level
        |   -> Exit reason = "sl"
        |   -> [Phase 7]
    |
    (Broker TP = 0 in virtual mode, so TP is never hit via broker)
    |
    v
    No exit this candle -> wait for next M1 candle -> [loop to 5.1]
```

NOTE: The effective_SL used in 5.3 may be modified by the fractal overlay
(Phase 6). See Phase 6 for the three-way SL competition logic.

---

## PHASE 6: FRACTAL OVERLAY (ADDITIONAL SL LAYER)

The fractal overlay is an independent SL management layer that runs alongside
the organic TSL ratchet from Phase 5. It uses Williams fractals on multiple
timeframes to detect structural levels, and can tighten the SL beyond what
the organic TSL alone would achieve.

### 6.0 Fractal Pre-Computation

```
    Fractals are pre-computed from historical data:
    |
    |   H1 fractals:  Williams 3-bar fractals on H1 (1-hour) candles
    |                  Expire after 48 hours from formation
    |
    |   H4 fractals:  Williams 3-bar fractals on H4 (4-hour) candles
    |                  Expire after 96 hours from formation
    |
    |   M2 fractals:  Williams 3-bar fractals on M2 (2-minute) candles
    |                  Used for trailing (no fixed expiry -- updated live)
    |
    Williams 3-bar fractal definition:
        High fractal: bar[i].high > bar[i-1].high AND bar[i].high > bar[i+1].high
        Low fractal:  bar[i].low  < bar[i-1].low  AND bar[i].low  < bar[i+1].low
```

### 6.1 Fractal Sweep Detection

Checked on every M1 candle while a position is open:

```
    FOR each active (non-expired) H1 and H4 fractal:
    |
    v
    High fractal sweep:
        IF candle high >= fractal.price -> fractal is SWEPT
    |
    Low fractal sweep:
        IF candle low <= fractal.price -> fractal is SWEPT
```

### 6.2 Fractal BE (Breakeven)

When an H1 or H4 fractal is swept, the SL can be moved to breakeven as a
protective measure.

```
    IF H1/H4 fractal swept
       AND position is open
       AND FRACTAL_BE_ENABLED = True:
    |
    v
    IF position is NOT already in fractal_be_active state:
    |
    v
    CHECK: Is current organic SL in negative territory?
    |
    |   LONG:  organic_sl < entry_price?  (SL below entry = unrealized loss zone)
    |   SHORT: organic_sl > entry_price?  (SL above entry = unrealized loss zone)
    |
    IF SL IS in negative zone:
    |   -> Set fractal_be = entry_price (breakeven level)
    |   -> Mark position as fractal_be_active
    |   -> This is a ONE-TIME move (does not repeat)
    |
    IF SL is NOT in negative zone (already in profit):
    |   -> No action needed (organic SL already protects profit)
```

### 6.3 Fractal TSL (M2 Trailing Stop)

When an H1 or H4 fractal is swept, an M2-based trailing stop can be activated.
This trails more aggressively than the organic TSL using the micro-structure
of M2 fractals.

```
    IF H1/H4 fractal swept
       AND position is open
       AND FRACTAL_TSL_ENABLED = True:
    |
    v
    IF position is NOT already in fractal_tsl_active state:
    |
    v
    Activate fractal TSL.
    |
    v
    Set initial M2 SL:
    |   LONG:  latest confirmed M2 LOW fractal price
    |   SHORT: latest confirmed M2 HIGH fractal price
    |
    v
    On each subsequent candle:
    |   Update M2 SL as new M2 fractals form
    |   (M2 fractals update frequently on 2-minute data)
```

### 6.4 Three-Way SL Competition

When multiple SL mechanisms are active simultaneously, the most protective
SL wins.

```
    candidates = [organic_sl]     -- from TSL ratchet (Phase 5.2)
    |
    IF fractal_be_active:
    |   candidates.append(entry_price)    -- breakeven level
    |
    IF fractal_tsl_active:
    |   candidates.append(m2_fractal_sl)  -- latest M2 fractal SL
    |
    v
    LONG position:
        effective_sl = max(candidates)    -- highest value = tightest/most protective
    |
    SHORT position:
        effective_sl = min(candidates)    -- lowest value = tightest/most protective
    |
    v
    Use effective_sl for hit detection in Phase 5.3
```

### 6.5 Same-Candle Protection

Prevents false SL triggers caused by a sudden jump in the effective SL level
on the same candle where the SL change occurred.

```
    IF effective_sl changed on THIS candle compared to the PREVIOUS candle:
    |
    |   -> Use the PREVIOUS candle's effective_sl for hit detection
    |   -> The new SL takes effect starting from the NEXT candle
    |
    This prevents a scenario where:
        1. Fractal TSL activates mid-candle and sets a new SL
        2. The same candle's price data already breached that new SL
        3. Without protection, this would cause a false stop-out
```

---

## PHASE 7: EXIT CONDITIONS

### Exit Reason Summary

| Exit Reason | Trigger | Exit Price |
|---|---|---|
| sl | Price hits effective SL (organic TSL, fractal BE, or fractal TSL -- whichever is most protective) | The SL level |
| tp | Price hits hard TP (only when TSL_TARGET = 0, fixed mode -- NOT used in PROD) | The TP level |
| time | Position window expires (trade_window_start + 120 min) | Market price (last candle close) |
| trail_conflict | Same candle hits both new virtual TP and new SL after a TSL ratchet | Open of next candle |

### Post-Exit Cleanup

```
    After exit (any reason):
    |
    v
    Clear all tracking state:
        - TSL state
        - Fractal BE state
        - Fractal TSL state
        - Virtual TP tracking
    |
    v
    State -> IN_TRADE_WINDOW
    |
    HOWEVER: Only 1 trade per day is allowed.
    -> No further signals will be generated today.
    -> Effectively equivalent to DAY_ENDED for trading purposes.
```

---

## PARAMETER REFERENCE TABLE

All variations (Reverse, OCAE, TCWE) use identical parameters in the
dual_v4 PROD baseline configuration.

| Parameter | Value | Description |
|---|---|---|
| IB_START | 08:00 | IB period start time (Europe/Berlin) |
| IB_END | 09:00 | IB period end time (Europe/Berlin) |
| IB_TZ | Europe/Berlin | Timezone for all session calculations |
| IB_WAIT | 0 min | Delay after IB end before trade window opens |
| TRADE_WINDOW | 120 min | Maximum duration of the trade window (09:00-11:00) |
| RR_TARGET | 1.0 | Risk-reward target (TP distance = 1.0 x risk) |
| STOP_MODE | ib_start | SL placement method: open of first trade window candle |
| TSL_TARGET | 1.0 | Virtual TP ratchet step size (in R units) |
| TSL_SL | 1.0 | SL ratchet step size (in R units) |
| MIN_SL_PCT | 0.15% | Minimum SL distance as percentage of entry price |
| IB_BUFFER_PCT | 0.0% | Buffer zone around IBH/IBL (no buffer) |
| MAX_DISTANCE_PCT | 100% | Maximum breakout distance (100% of IB Range) |
| REV_RB_ENABLED | False | REV_RB variation is disabled |
| FRACTAL_BE_ENABLED | True | Fractal breakeven mechanism is active |
| FRACTAL_TSL_ENABLED | True | M2 fractal trailing stop is active |

### Derived Values (Calculated Daily)

| Derived Value | Formula | Example (IB Range = 100 pts) |
|---|---|---|
| buffer | IB_BUFFER_PCT * IB_Range | 0.0 * 100 = 0 pts |
| max_dist | MAX_DISTANCE_PCT * IB_Range | 1.0 * 100 = 100 pts |
| min_sl | entry * MIN_SL_PCT | 23000 * 0.0015 = 34.5 pts |
| trade_window_start | IB_END + IB_WAIT | 09:00 + 0 = 09:00 |
| trade_window_end | trade_window_start + TRADE_WINDOW | 09:00 + 120min = 11:00 |
| risk (R) | |entry - SL| | Varies per trade |
| virtual_tp | entry +/- RR_TARGET * R | entry +/- 1.0R |
| news_buffer | 2 minutes | Before and after high-impact EUR/USD events |

---

## COMPLETE DAILY LIFECYCLE

```
    00:00 Berlin    Phase 0: Daily Reset
         |
         v
    08:00 Berlin    Phase 1: IB Formation begins (observe M2 candles)
         |
    09:00 Berlin    Phase 1: IB Formation ends -> calculate IBH, IBL, EQ
         |
         v
    09:00 Berlin    Phase 2: Trade window opens
         |
         v
    09:00-11:00     Phase 3: Signal detection (M1 loop)
         |              Priority: Reverse > OCAE > TCWE
         |
         v
    Signal found    Phase 4: Trade execution (1 trade max)
         |
         v
    Position open   Phase 5+6: Position management + fractal overlay (M1 loop)
         |              - Time exit check
         |              - Virtual TP / TSL ratchet
         |              - Fractal BE / Fractal TSL
         |              - Three-way SL competition
         |              - SL hit check
         |
         v
    Exit            Phase 7: Close position, record reason, cleanup
         |
         v
    Day complete    Wait for next midnight -> Phase 0
```

```
    IF no signal found by 11:00:
         -> DAY_ENDED (no trade)
         -> Wait for next midnight -> Phase 0
```

---

*Document generated for dual_v4 PROD baseline parameters.*
*Instrument: GER40. All times: Europe/Berlin.*
