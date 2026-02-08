# IB (Initial Balance) Intraday Trading Strategy -- Manual

**Version**: V9 Production (GER40_055 + XAUUSD_059)
**Last Updated**: 2026-02-08
**Timeframe**: M2 (2-minute candles)
**Rule**: One trade per instrument per day. Take only the first valid signal.

---

## Table of Contents

1. [Strategy Overview](#1-strategy-overview)
2. [Instruments and Trading Sessions](#2-instruments-and-trading-sessions)
3. [Step 1: Calculate the Initial Balance](#3-step-1-calculate-the-initial-balance-ib)
4. [Step 2: Apply Filters](#4-step-2-apply-filters)
5. [Step 3: Wait for Signal](#5-step-3-wait-for-signal)
6. [Step 4: Place the Trade](#6-step-4-place-the-trade)
7. [Step 5: Manage the Position (Trailing Stop Loss)](#7-step-5-manage-the-position-trailing-stop-loss)
8. [Production Parameter Tables](#8-production-parameter-tables)
9. [Risk Management Rules](#9-risk-management-rules)
10. [News Filter](#10-news-filter)
11. [Important Notes and Edge Cases](#11-important-notes-and-edge-cases)

---

## 1. Strategy Overview

This is an **Initial Balance (IB) based intraday strategy**. It trades two instruments -- GER40 (DAX 40 index) and XAUUSD (Gold) -- once per day each.

The core idea is:

1. At the start of each trading session, observe the **first 30 minutes** of price action to establish a price range called the Initial Balance (IB).
2. After a waiting period, watch for one of four predefined price patterns (signals) that occur relative to the IB range.
3. Enter a single trade based on the first valid signal found.
4. Manage the trade with a fixed Stop Loss and Take Profit, optionally trailing the stop if the trade goes in your favor.
5. Close the trade when Stop Loss is hit, Take Profit is hit, or the trade window expires.

The strategy checks for signals in a fixed priority order:

1. **Reverse** (highest priority)
2. **OCAE** (Open Close After Equilibrium)
3. **TCWE** (Two Candles Without Equilibrium)
4. **REV_RB** (Reverse Blocked Rebound) -- currently DISABLED in production

Only the **first** signal found is taken. If Reverse triggers, OCAE and TCWE are never checked. If no signal triggers during the trade window, no trade is placed for that instrument on that day.

---

## 2. Instruments and Trading Sessions

### GER40 (DAX 40 Index)

| Parameter          | Value                    |
|--------------------|--------------------------|
| IB Period          | 09:00 -- 09:30           |
| Timezone           | Europe/Berlin (CET/CEST) |
| IB Wait            | 15 minutes               |
| Trade Window Start | 09:45 Berlin time        |

The **Trade Window** duration varies by signal variation (90 to 240 minutes; see parameter tables in Section 8).

### XAUUSD (Gold)

| Parameter          | Value                    |
|--------------------|--------------------------|
| IB Period          | 09:00 -- 09:30           |
| Timezone           | Asia/Tokyo (JST, UTC+9)  |
| IB Wait            | 20 minutes               |
| Trade Window Start | 09:50 Tokyo time         |

The **Trade Window** duration varies by signal variation (120 to 240 minutes; see parameter tables in Section 8).

**What "IB Wait" means**: After the IB period ends at 09:30, you must wait an additional 15 minutes (GER40) or 20 minutes (XAUUSD) before you begin scanning for signals. No trades are placed during the wait.

**What "Trade Window" means**: The maximum time after the wait period during which you are allowed to scan for signals AND hold a position. If the trade window is 240 minutes, your position must close by Trade Window Start + 240 minutes at the latest.

---

## 3. Step 1: Calculate the Initial Balance (IB)

On your 2-minute chart, mark the **High** and **Low** of the IB period.

### Procedure

1. Open the M2 (2-minute) chart for the instrument.
2. Identify all candles between 09:00 and 09:30 in the instrument's timezone.
3. Record the following:
   - **IBH** (IB High) = the highest high of any candle within the IB period.
   - **IBL** (IB Low) = the lowest low of any candle within the IB period.
4. Calculate:
   - **EQ** (Equilibrium) = (IBH + IBL) / 2
   - **IB Range** = IBH - IBL

### Example: GER40

Suppose during 09:00-09:30 Berlin time, the M2 candles produce:
- Highest high = 20,100
- Lowest low = 20,000

Then:
- **IBH** = 20,100
- **IBL** = 20,000
- **EQ** = (20,100 + 20,000) / 2 = **20,050**
- **IB Range** = 20,100 - 20,000 = **100 points**

Draw horizontal lines on your chart at IBH (20,100), IBL (20,000), and EQ (20,050).

### Example: XAUUSD

Suppose during 09:00-09:30 Tokyo time, the M2 candles produce:
- Highest high = 2,650.50
- Lowest low = 2,645.00

Then:
- **IBH** = 2,650.50
- **IBL** = 2,645.00
- **EQ** = (2,650.50 + 2,645.00) / 2 = **2,647.75**
- **IB Range** = 2,650.50 - 2,645.00 = **5.50 points**

---

## 4. Step 2: Apply Filters

Before any signal can be valid, two filters must be satisfied. These filters apply to all signals except Reverse (which has its own sweep validation).

### 4.1 IB Buffer Filter

The IB Buffer ensures that the breakout beyond the IB level is meaningful, not just noise touching the level by a few ticks.

**How it works**: When checking if a candle has broken out above IBH (for a long signal) or below IBL (for a short signal), the candle's close must exceed the IB boundary by a minimum buffer.

- **Long breakout**: Close > IBH + (IB_BUFFER_PCT x IB Range)
- **Short breakout**: Close < IBL - (IB_BUFFER_PCT x IB Range)

**Production values**:
- GER40: IB_BUFFER_PCT = 0.20 (20% of IB Range)
- XAUUSD: IB_BUFFER_PCT = 0.05 (5% of IB Range)

**Example** (GER40, IBH = 20,100, IBL = 20,000, IB Range = 100):
- Buffer = 0.20 x 100 = 20 points
- For a valid long breakout, the candle's close must be above 20,100 + 20 = **20,120**
- For a valid short breakout, the candle's close must be below 20,000 - 20 = **19,980**

**Example** (XAUUSD, IBH = 2,650.50, IBL = 2,645.00, IB Range = 5.50):
- Buffer = 0.05 x 5.50 = 0.275 points
- For a valid long breakout, close must be above 2,650.50 + 0.275 = **2,650.775**
- For a valid short breakout, close must be below 2,645.00 - 0.275 = **2,644.725**

### 4.2 Max Distance Filter

The Max Distance filter prevents entries when price has already moved too far from the IB boundary. A very distant entry means the Stop Loss would be extremely large, making the risk-reward unfavorable.

**How it works**: When a breakout candle is detected, the distance from the IB boundary to the close must not exceed a maximum.

- **Long**: Distance = Close - IBH. Reject if Distance > MAX_DISTANCE_PCT x IB Range
- **Short**: Distance = IBL - Close. Reject if Distance > MAX_DISTANCE_PCT x IB Range

**Production values**:
- GER40: MAX_DISTANCE_PCT = 0.50 to 0.75 (varies by signal, see parameter tables)
- XAUUSD: MAX_DISTANCE_PCT = 0.75 (all signals)

**Example** (GER40 OCAE, IBH = 20,100, IB Range = 100):
- MAX_DISTANCE_PCT = 0.50
- Max allowed distance = 0.50 x 100 = 50 points
- If a candle closes at 20,160 (distance = 60), **reject** -- too far
- If a candle closes at 20,130 (distance = 30), **accept** -- within range

---

## 5. Step 3: Wait for Signal

Starting at the trade window open time (09:45 Berlin for GER40, 09:50 Tokyo for XAUUSD), scan M2 candles in real-time. Check for signals in priority order. Stop checking once any signal triggers.

### 5.1 Signal: Reverse (Highest Priority)

The Reverse signal detects a **failed breakout** (sweep) of the IB level, followed by a reversal back into the range.

#### What to look for on the chart

**Concept**: Price briefly pokes through an IB level (shadow/wick exceeds the level), but the candle body stays inside or near the range. This "sweep" suggests liquidity was grabbed above/below the IB. Then price reverses and confirms the reversal by breaking through a key structure level (CISD).

#### Step-by-step detection

1. **Identify sweeps**: Look at each M2 candle in the trade window. A sweep occurs when:
   - **Upper sweep** (potential SHORT): The candle's high goes above IBH, but both the open and close remain at or below IBH + Buffer. This means the wick pierced above IBH but the body did not close above it (or only barely above).
   - **Lower sweep** (potential LONG): The candle's low goes below IBL, but both the open and close remain at or above IBL - Buffer. The wick pierced below IBL but the body stayed inside.

2. **Invalidation rule**: If any candle has BOTH its open AND close above IBH + Buffer (or both below IBL - Buffer), the sweep is invalidated. All sweeps found before this candle remain valid; no further sweeps are searched.

3. **Group consecutive sweeps**: If multiple consecutive candles form sweeps in the same direction, group them together. For upper sweeps, record the highest high across the group. For lower sweeps, record the lowest low.

4. **Find the last opposite candle** (context candle): For each sweep group, look backward in time (including the IB period candles) and find the most recent candle whose close is on the opposite side of its open:
   - If the sweep is an upper sweep (SHORT setup), find the last bearish candle (close < open). Record its low.
   - If the sweep is a lower sweep (LONG setup), find the last bullish candle (close > open). Record its high.

5. **Wait for CISD (Change in State of Delivery)**: After the sweep group ends, watch subsequent candles for a close that breaks through the context candle's level:
   - **Upper sweep (SHORT)**: A candle closes below the last bearish candle's low. This confirms the reversal downward.
   - **Lower sweep (LONG)**: A candle closes above the last bullish candle's high. This confirms the reversal upward.

6. **Distance check**: The CISD candle must not be too far from the IB boundary. If the distance from the IB level to the CISD close is >= 50% of the IB Range, reject this signal.

7. **Entry**: Enter on the **open of the very next candle** after the CISD candle.

8. **Direction**:
   - Upper sweep produces a **SHORT** trade (price swept above IBH, then reversed down).
   - Lower sweep produces a **LONG** trade (price swept below IBL, then reversed up).

#### Stop Loss for Reverse

The Stop Loss is placed at the **sweep extreme**:
- For a SHORT (from upper sweep): SL = the highest high of the sweep group.
- For a LONG (from lower sweep): SL = the lowest low of the sweep group.

If the calculated risk (distance from entry to SL) is smaller than the minimum SL size (entry x MIN_SL_PCT), the SL is widened to meet the minimum.

#### Example: Reverse SHORT on GER40

Given: IBH = 20,100, IBL = 20,000, IB Range = 100, Buffer = 20% x 100 = 20

1. At 09:48, a candle has: Open = 20,095, High = 20,115, Low = 20,088, Close = 20,098.
   - High (20,115) > IBH (20,100) -- wick above IBH.
   - Open (20,095) and Close (20,098) are both <= IBH + Buffer (20,120).
   - **Upper sweep detected**. Sweep extreme = 20,115.

2. At 09:50, the next candle stays inside the range. No more sweeps.

3. Look backward: the last bearish candle (close < open) before the sweep had Low = 20,080.

4. At 09:54, a candle closes at 20,075. This is below 20,080 (the context candle's low).
   - Distance = IBH - Close = 20,100 - 20,075 = 25. This is < 50 (0.5 x 100). Valid.
   - **CISD confirmed**.

5. Enter SHORT at the open of the 09:56 candle, say at 20,078.
   - SL = 20,115 (sweep extreme)
   - Risk = 20,115 - 20,078 = 37 points
   - TP = 20,078 - (0.5 x 37) = 20,078 - 18.5 = **20,059.5** (RR_TARGET = 0.5 for GER40 Reverse)

---

### 5.2 Signal: OCAE (Open Close After Equilibrium)

OCAE stands for "Open Close After Equilibrium." This signal requires that price **touches the EQ level first** and then breaks out of the IB range. Touching EQ first suggests the price has tested the midpoint and gathered momentum for a genuine breakout.

#### Step-by-step detection

1. **Scan candles** in the trade window from the beginning, looking for the **first breakout candle** -- a candle whose close exceeds IBH + Buffer (long) or falls below IBL - Buffer (short).

2. **Check the EQ touch condition**: Before this breakout candle (at any point from the start of the trade window up to and including the breakout candle), at least one candle must have had its price range cross the EQ level. Specifically, any candle where Low <= EQ AND High >= EQ satisfies this condition. If EQ has NOT been touched, this is not an OCAE signal (it may be a TCWE signal instead).

3. **Apply the Max Distance filter**: The distance from the IB boundary to the breakout close must be <= MAX_DISTANCE_PCT x IB Range.

4. **Entry**: Enter at the **close of the breakout candle** (the same candle that confirmed the breakout). Wait for the candle to fully close (2 minutes) before entering.

5. **Direction**:
   - Close above IBH + Buffer: **LONG**
   - Close below IBL - Buffer: **SHORT**

#### Stop Loss for OCAE

The Stop Loss placement depends on the STOP_MODE parameter:

| Instrument | STOP_MODE | SL Placement                                      |
|------------|-----------|---------------------------------------------------|
| GER40      | ib_start  | SL = opening price of the first candle in the trade window |
| XAUUSD     | eq        | SL = EQ level (midpoint of IB range)              |

**"ib_start" mode explained**: The SL is placed at the opening price of the very first M2 candle of the trade window (the candle at Trade Window Start time). This is a reference price at the moment signal scanning begins.

**"eq" mode explained**: The SL is placed at the EQ level. For a LONG trade, EQ will be below the entry. For a SHORT trade, EQ will be above the entry. If EQ happens to be on the wrong side (e.g., above entry for a LONG), the SL is flipped to the correct side at the same distance.

#### Example: OCAE LONG on XAUUSD

Given: IBH = 2,650.50, IBL = 2,645.00, EQ = 2,647.75, IB Range = 5.50, Buffer = 0.05 x 5.50 = 0.275

1. At 09:52 Tokyo time, a candle's range crosses EQ (Low = 2,647.00, High = 2,648.50 -- EQ = 2,647.75 is within this range). EQ is "touched."

2. At 10:04, a candle closes at 2,651.10.
   - 2,651.10 > IBH + Buffer (2,650.775). Breakout confirmed.
   - Distance = 2,651.10 - 2,650.50 = 0.60. Max allowed = 0.75 x 5.50 = 4.125. Valid.

3. EQ was touched before this candle. OCAE signal is valid.

4. Enter LONG at 2,651.10 (close of breakout candle).
   - STOP_MODE = "eq", so SL = EQ = 2,647.75
   - Risk = 2,651.10 - 2,647.75 = 3.35
   - RR_TARGET = 1.25, so TP = 2,651.10 + (1.25 x 3.35) = 2,651.10 + 4.1875 = **2,655.2875**

---

### 5.3 Signal: TCWE (Two Candles Without Equilibrium)

TCWE stands for "Two Candles Without Equilibrium." This signal triggers when two consecutive breakout candles appear WITHOUT price ever touching the EQ level. The idea is that price broke out of the IB range directly without testing the midpoint, and the second candle going further confirms momentum.

#### Step-by-step detection

1. **Find the first breakout candle**: A candle whose close exceeds IBH + Buffer (long) or falls below IBL - Buffer (short). **Critically, EQ must NOT have been touched at any point before or during this candle.**

2. **Find the second breakout candle**: After the first breakout candle, find a subsequent candle that closes further in the same direction than the first:
   - For long: second close > first close, AND second close > IBH + Buffer
   - For short: second close < first close, AND second close < IBL - Buffer

3. **EQ must still not be touched**: Between the first and second breakout candle, no candle should touch EQ. If EQ is touched at any point before the second candle, the TCWE signal is cancelled entirely (returns no signal, not a fallback to OCAE).

4. **Max Distance filter**: The second breakout candle must also pass the max distance check.

5. **Entry**: Enter at the **close of the second breakout candle**. Wait for it to fully close before entering.

6. **Direction**: Same as the breakout direction.

#### Stop Loss for TCWE

| Instrument | STOP_MODE | SL Placement                                      |
|------------|-----------|---------------------------------------------------|
| GER40      | ib_start  | SL = opening price of the first candle in the trade window |
| XAUUSD     | ib_start  | SL = opening price of the first candle in the trade window |

#### Example: TCWE LONG on GER40

Given: IBH = 20,100, IBL = 20,000, EQ = 20,050, IB Range = 100, Buffer = 0.20 x 100 = 20

1. Trade window opens at 09:45. First candle opens at 20,090.

2. Throughout the early candles, price stays above EQ (20,050). EQ is **not touched** at any point.

3. At 09:51, a candle closes at 20,125.
   - 20,125 > IBH + Buffer (20,120). First breakout candle found.
   - EQ has not been touched. Good.

4. At 09:55, a candle closes at 20,132.
   - 20,132 > 20,125 (further than first breakout). Second breakout confirmed.
   - Distance = 20,132 - 20,100 = 32. Max allowed = 0.75 x 100 = 75. Valid.
   - EQ still not touched. TCWE confirmed.

5. Enter LONG at 20,132 (close of second breakout candle).
   - STOP_MODE = "ib_start", so SL = 20,090 (trade window first candle's open)
   - Risk = 20,132 - 20,090 = 42 points
   - RR_TARGET = 0.75, so TP = 20,132 + (0.75 x 42) = 20,132 + 31.5 = **20,163.5**

---

### 5.4 Signal: REV_RB (Reverse Blocked Rebound) -- CURRENTLY DISABLED

**IMPORTANT: REV_RB is DISABLED in production for both GER40 and XAUUSD** (REV_RB_ENABLED = False). It is documented here for completeness but should NOT be traded until explicitly re-enabled.

This signal fires when price extends significantly beyond the IB range (by a full IB Range width) and then snaps back to the IB boundary. It is a mean-reversion counter-trend entry using a limit order.

#### How it works (when enabled)

1. Calculate extension levels:
   - Upper extension = IBH + (REV_RB_PCT x IB Range)
   - Lower extension = IBL - (REV_RB_PCT x IB Range)
   - With REV_RB_PCT = 1.0, the extensions are one full IB range beyond the boundaries.

2. Watch for price to reach one of these extension levels:
   - If upper extension is reached first: prepare a **LONG** trade (buying the pullback to IBH).
   - If lower extension is reached first: prepare a **SHORT** trade (selling the bounce to IBL).

3. Place a limit order at the IB boundary:
   - For LONG: limit buy at IBH. Wait for price to pull back to IBH.
   - For SHORT: limit sell at IBL. Wait for price to bounce back to IBL.

4. SL = EQ (midpoint of IB range) for all REV_RB trades.

---

## 6. Step 4: Place the Trade

### 6.1 Entry Price Rules by Signal

| Signal   | Entry Price                                        |
|----------|----------------------------------------------------|
| Reverse  | **Open** of the candle following the CISD candle   |
| OCAE     | **Close** of the breakout candle                    |
| TCWE     | **Close** of the second breakout candle             |
| REV_RB   | Limit order at IBH (long) or IBL (short)           |

### 6.2 Stop Loss Calculation

The Stop Loss depends on both the signal type and the STOP_MODE parameter.

**For Reverse**: SL = sweep extreme (the most extreme high/low of the sweep candle group).

**For OCAE and TCWE**: SL depends on STOP_MODE:

| STOP_MODE  | SL Placement                                                       |
|------------|--------------------------------------------------------------------|
| `ib_start` | Opening price of the first candle in the trade window              |
| `eq`       | EQ level (midpoint of the IB range)                                |
| `cisd`     | Low of the last bearish candle (LONG) or High of the last bullish candle (SHORT) before entry |

**For REV_RB**: SL = EQ level.

**SL Validation Rules**:
1. SL must be on the correct side of the entry: below entry for LONG, above entry for SHORT.
2. If SL ends up on the wrong side, it is flipped to the correct side at the same distance.
3. If the resulting risk (distance from entry to SL) is smaller than the minimum SL size, the SL is widened to meet the minimum.
   - GER40: Minimum SL = 0.15% of entry price
   - XAUUSD: Minimum SL = 0.10% of entry price

### 6.3 Take Profit Calculation

TP is always calculated as:

- **LONG**: TP = Entry + (RR_TARGET x Risk)
- **SHORT**: TP = Entry - (RR_TARGET x Risk)

Where Risk = absolute distance from Entry to SL.

### 6.4 Worked Example: Full OCAE Trade Placement on GER40

Given:
- IBH = 20,100, IBL = 20,000, EQ = 20,050, IB Range = 100
- First candle of trade window opens at 20,085
- OCAE LONG breakout confirmed. Entry at close = 20,125

Calculation:
- STOP_MODE = "ib_start", so SL candidate = 20,085
- Risk = 20,125 - 20,085 = 40 points
- Minimum SL = 0.0015 x 20,125 = 30.19 points. Since 40 > 30.19, no adjustment needed.
- RR_TARGET = 0.75
- TP = 20,125 + (0.75 x 40) = 20,125 + 30 = **20,155**

**Summary**: BUY at 20,125, SL at 20,085, TP at 20,155.

---

## 7. Step 5: Manage the Position (Trailing Stop Loss)

Once a position is open, it is managed by the Trailing Stop Loss (TSL) system. The TSL system steps the SL and TP forward each time price reaches the current TP level.

### 7.1 When TSL Is Active vs. Inactive

- **TSL_TARGET > 0**: Trailing is active. Each time TP is hit, SL is moved closer to the market and a new TP is set further out.
- **TSL_TARGET = 0**: Fixed TP mode. When price reaches TP, the trade is closed immediately. No trailing occurs.

### 7.2 How Trailing Works (Step by Step)

Define:
- **Risk** = |Entry Price - Initial SL| (this value never changes)
- **Accumulated_R** = starts at RR_TARGET (the initial TP expressed in multiples of risk)
- **Current TP** = Entry + Accumulated_R x Risk (for LONG)
- **Current SL** = the Initial SL (starts here)

**When price reaches the Current TP**:

1. **Move SL**: New SL = Current TP (the TP just hit) - TSL_SL x Risk (for LONG).
   - Only move if the new SL is better (closer to market) than the current SL.
   - For LONG: New SL = max(Current SL, TP_just_hit - TSL_SL x Risk)
   - For SHORT: New SL = min(Current SL, TP_just_hit + TSL_SL x Risk)

2. **Set new TP**: Accumulated_R = Accumulated_R + TSL_TARGET
   - New TP = Entry + Accumulated_R x Risk (for LONG)
   - New TP = Entry - Accumulated_R x Risk (for SHORT)

3. **Repeat**: The process repeats each time the new TP is reached.

4. **Exit**: The trade closes when:
   - Price hits the (trailed) SL.
   - The trade window expires (position closes at the last available close price).

### 7.3 Full Numerical Example: GER40 Reverse with Trailing

**Setup**:
- Entry = 20,078 (SHORT after upper sweep)
- Initial SL = 20,115 (sweep extreme)
- Risk = |20,078 - 20,115| = 37 points
- RR_TARGET = 0.5
- TSL_TARGET = 1.5
- TSL_SL = 0.5

**Step 0 -- Initial Position**:
- Accumulated_R = 0.5
- TP = 20,078 - (0.5 x 37) = 20,078 - 18.5 = 20,059.5
- SL = 20,115

**Step 1 -- Price drops to 20,059.5 (TP hit)**:
- Move SL: New SL = 20,059.5 + (0.5 x 37) = 20,059.5 + 18.5 = 20,078.0
  - Current SL was 20,115. New SL (20,078) is better (lower for SHORT). Use 20,078.
  - Note: SL has moved to breakeven (entry price).
- Set new TP: Accumulated_R = 0.5 + 1.5 = 2.0
  - New TP = 20,078 - (2.0 x 37) = 20,078 - 74 = 20,004

**Step 2 -- Price drops to 20,004 (TP hit again)**:
- Move SL: New SL = 20,004 + (0.5 x 37) = 20,004 + 18.5 = 20,022.5
  - Current SL was 20,078. New SL (20,022.5) is better (lower). Use 20,022.5.
- Set new TP: Accumulated_R = 2.0 + 1.5 = 3.5
  - New TP = 20,078 - (3.5 x 37) = 20,078 - 129.5 = 19,948.5

**Step 3 -- Price bounces back up to 20,022.5**:
- SL is hit. Trade closes at 20,022.5.
- Final P/L in R-multiples: (20,078 - 20,022.5) / 37 = 55.5 / 37 = **+1.5R**

The trader captured 1.5R of profit through the trailing mechanism, compared to only 0.5R if the trade had been closed at the initial TP.

### 7.4 Example: Fixed TP (No Trailing) -- XAUUSD OCAE

**Setup**:
- XAUUSD OCAE has TSL_TARGET = 0 (trailing disabled)
- Entry = 2,651.10 (LONG)
- SL = 2,647.75 (EQ mode)
- Risk = 3.35
- RR_TARGET = 1.25
- TP = 2,651.10 + (1.25 x 3.35) = 2,655.2875

**Outcome**: When price reaches 2,655.29, the trade closes immediately at TP. There is no trailing. Result = +1.25R.

### 7.5 Conflict Scenario

If a single M2 candle hits both the new SL and the new TP simultaneously (i.e., the candle range is very wide), the trade is closed at the **open of the next candle** at whatever price is available. This is called a "trail_conflict" exit.

### 7.6 Time Exit

If neither SL nor TP is hit before the trade window expires, the position is closed at the **closing price of the last M2 candle** in the trade window. The resulting P/L may be positive or negative.

---

## 8. Production Parameter Tables

### 8.1 GER40 Production Parameters (V9 -- GER40_055)

| Parameter         | Reverse    | OCAE       | TCWE       | REV_RB (DISABLED) |
|-------------------|------------|------------|------------|---------------------|
| IB Start          | 09:00      | 09:00      | 09:00      | 09:00               |
| IB End            | 09:30      | 09:30      | 09:30      | 09:30               |
| IB Timezone       | Europe/Berlin | Europe/Berlin | Europe/Berlin | Europe/Berlin |
| IB Wait (min)     | 15         | 15         | 15         | 15                  |
| Trade Window (min)| 180        | 240        | 90         | 240                 |
| RR Target         | 0.5        | 0.75       | 0.75       | 2.0                 |
| Stop Mode         | ib_start*  | ib_start   | ib_start   | ib_start            |
| TSL Target        | 1.5        | 0.75       | 2.0        | 0.0 (fixed TP)      |
| TSL SL            | 0.5        | 0.5        | 1.5        | 0.5                 |
| Min SL (%)        | 0.15%      | 0.15%      | 0.15%      | 0.15%               |
| IB Buffer (%)     | 20%        | 20%        | 20%        | 20%                 |
| Max Distance (%)  | 75%        | 50%        | 75%        | 75%                 |
| REV_RB Enabled    | --         | --         | --         | Yes (but DISABLED)** |
| REV_RB PCT        | --         | --         | --         | 1.0                 |

(*) For Reverse, the actual SL is at the sweep extreme, not at ib_start. The STOP_MODE parameter is technically "ib_start" in the config but the Reverse signal logic overrides it with the sweep extreme.

(**) REV_RB_ENABLED is True in the GER40 REV_RB config block, but with only 8 historical trades and 2.32R total, it is considered too unreliable for production use. It may be enabled in the future after further validation.

### 8.2 XAUUSD Production Parameters (V9 -- XAUUSD_059)

| Parameter         | Reverse    | OCAE       | TCWE       | REV_RB (DISABLED) |
|-------------------|------------|------------|------------|---------------------|
| IB Start          | 09:00      | 09:00      | 09:00      | 09:00               |
| IB End            | 09:30      | 09:30      | 09:30      | 09:30               |
| IB Timezone       | Asia/Tokyo | Asia/Tokyo | Asia/Tokyo | Asia/Tokyo          |
| IB Wait (min)     | 20         | 20         | 20         | 20                  |
| Trade Window (min)| 240        | 240        | 240        | 120                 |
| RR Target         | 1.5        | 1.25       | 0.75       | 1.75                |
| Stop Mode         | eq*        | eq         | ib_start   | ib_start            |
| TSL Target        | 1.0        | 0.0 (fixed)| 1.5        | 2.0                 |
| TSL SL            | 1.0        | 0.5        | 1.5        | 1.0                 |
| Min SL (%)        | 0.10%      | 0.10%      | 0.10%      | 0.10%               |
| IB Buffer (%)     | 5%         | 5%         | 5%         | 5%                  |
| Max Distance (%)  | 75%        | 75%        | 75%        | 75%                 |
| REV_RB Enabled    | --         | --         | --         | No                  |
| REV_RB PCT        | --         | --         | --         | 1.0                 |

(*) For Reverse, the actual SL is at the sweep extreme regardless of the STOP_MODE setting. The "eq" STOP_MODE applies to the stop level used for non-Reverse signals on this instrument.

### 8.3 Summary of Active Configurations

| Instrument | Active Signals          | TSL Behavior                                   |
|------------|-------------------------|-------------------------------------------------|
| GER40      | Reverse, OCAE, TCWE     | All three use trailing                          |
| XAUUSD     | Reverse, OCAE, TCWE     | OCAE uses fixed TP; Reverse and TCWE use trailing |

---

## 9. Risk Management Rules

These rules are enforced by the trading system and must also be followed manually.

| Rule                        | Value     | Description                                                      |
|-----------------------------|-----------|------------------------------------------------------------------|
| Risk per trade              | 0.2%      | Maximum account equity risked on any single trade                |
| Max daily loss              | 3.0%      | If daily P/L reaches -3% of account, stop trading for the day   |
| Max margin usage            | 40%       | Total margin used must not exceed 40% of account equity          |
| Trades per instrument/day   | 1         | Maximum one trade per instrument per day                         |

### Lot Size Calculation

Given:
- Account equity = $100,000
- Risk per trade = 0.2% = $200
- Risk in points = |Entry - SL| = 37 points (example from GER40 Reverse)
- Point value per lot for DAX40 (GER40) = varies by broker

Formula: **Lot Size = Risk in Money / (Risk in Points x Point Value per Lot)**

Consult your broker's contract specifications for the exact point value per lot for each instrument.

---

## 10. News Filter

The news filter prevents trade entries during periods of high-impact economic news releases. This is required for compliance with The5ers prop firm rules.

### Rules

- **No new orders** from 2 minutes before until 2 minutes after any high-impact economic news event.
- Only **high-impact** events are filtered (red-flag events on ForexFactory).
- The filter blocks **new entries only**. Existing positions are NOT closed by the news filter.

### Currency Relevance

| Instrument | Affected by News From |
|------------|----------------------|
| GER40      | EUR, USD             |
| XAUUSD     | USD                  |

This means:
- A high-impact EUR news event (e.g., ECB Interest Rate Decision) will block GER40 entries.
- A high-impact USD news event (e.g., Non-Farm Payrolls) will block BOTH GER40 and XAUUSD entries.
- A high-impact JPY news event will NOT block either instrument.

### Practical Procedure

1. Before the trading session, check the economic calendar (ForexFactory or similar) for the day.
2. Note any high-impact (red) events related to EUR or USD.
3. If any such event falls within your trade window, avoid entering trades from 2 minutes before the event time to 2 minutes after.
4. If a signal fires during a blocked period, skip it entirely. Do NOT wait for the news period to end and then enter -- the signal is forfeited.

---

## 11. Important Notes and Edge Cases

### 11.1 Timeframe

All signal detection uses **M2 (2-minute) candles**. This is critical -- using a different timeframe will produce different IB levels, different breakout candles, and different results. Always use M2.

### 11.2 Candle Close Confirmation

In live trading, you must wait for a candle to fully close before acting on it.

- For **OCAE** and **TCWE**: The signal candle must be fully closed (the next M2 candle has started) before entering. Do not enter mid-candle based on a wick that is currently above/below the breakout level.
- For **Reverse**: The CISD candle must be fully closed. Entry occurs on the open of the next candle, which by definition means the CISD candle has already closed.

### 11.3 Sweep Validation Details (Reverse)

The IB Buffer parameter serves a dual purpose for the Reverse signal:
- It defines the **buffer zone** around IB levels. A sweep is only valid if the candle's open and close are both within IBH + Buffer (upper) or IBL - Buffer (lower).
- A larger buffer means more "room" for the candle body to still count as a sweep. With GER40's 20% buffer and a 100-point IB range, a candle closing up to 20 points above IBH can still be a valid upper sweep.

### 11.4 Distance Check for Reverse CISD

The Reverse signal has its own distance check separate from the Max Distance filter used by OCAE/TCWE. For the CISD to be valid:
- **Upper sweep (SHORT)**: The distance from IBH to the CISD close (IBH - close) must be less than 50% of the IB Range.
- **Lower sweep (LONG)**: The distance from the CISD close to IBL (close - IBL) must be less than 50% of the IB Range.

This is a fixed 50% threshold, not configurable per variation.

### 11.5 One Signal Per Day Rule

The strategy evaluates signals in strict priority order: Reverse, then OCAE, then TCWE, then REV_RB. Once ANY signal is found, the remaining signals are never checked. This means:
- If a Reverse signal fires but results in a losing trade, you do NOT then check for OCAE or TCWE that day.
- If no Reverse signal is found but an OCAE signal fires, you do NOT then check for TCWE.
- You get at most one trade per instrument per day.

### 11.6 Position Window (Time Exit)

The Trade Window parameter determines the maximum duration for both signal scanning AND position holding. If you enter a trade at 09:55 Berlin time and the trade window is 180 minutes (Reverse on GER40), your trade must close by 09:45 + 180 = 12:45 Berlin time at the latest (09:45 being the trade window start after the 15-minute wait).

If price has not hit SL or TP by the end of the window, the position is closed at whatever the current market price is. This can result in either a profit or a loss.

### 11.7 Minimum SL Size

To prevent extremely tight stops that would cause premature stop-outs from normal market noise, a minimum SL distance is enforced:
- **GER40**: MIN_SL_PCT = 0.15% of entry price. At an entry price of 20,000 this equals 30 points.
- **XAUUSD**: MIN_SL_PCT = 0.10% of entry price. At an entry price of 2,650 this equals 2.65 points.

If the calculated SL is tighter than this minimum, it is widened (moved further from entry) to meet the minimum. The TP is then recalculated based on the new (wider) risk.

### 11.8 SL Flip Safety

If a calculation produces a Stop Loss on the wrong side of the entry (e.g., SL above entry for a LONG trade), the SL is automatically flipped to the correct side at the same distance. This is a safety mechanism and should rarely occur with correct IB calculations, but it can happen when using "eq" or "ib_start" stop modes in unusual market conditions.

### 11.9 Multiple TP Hits in One Candle (Trailing)

During trailing, if a single M2 candle is extremely large and passes through multiple TP levels, the TSL system processes all TP levels in sequence within that candle. The SL and TP are stepped forward multiple times. After processing, if the candle also hit the (newly moved) SL, the trade is marked for exit on the open of the next candle (trail_conflict exit).

### 11.10 Timezone Considerations

- **GER40**: All times are in Europe/Berlin (CET in winter = UTC+1, CEST in summer = UTC+2). Daylight saving time shifts occur in March and October.
- **XAUUSD**: All times are in Asia/Tokyo (JST = UTC+9). Japan does not observe daylight saving time, so this timezone is always UTC+9.
- **Bot/User**: The bot operates in Asia/Almaty (UTC+5/UTC+6).
- **MT5 Server**: The5ers MT5 server uses Asia/Jerusalem (UTC+2/UTC+3).

When trading manually, always verify which timezone your charting platform is displaying and convert accordingly.

---

*End of Strategy Manual.*
