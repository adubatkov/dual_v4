# EURUSD OCAE SHORT - Detailed SL Calculation Breakdown

**Trade:** EURUSD SHORT OCAE
**Time:** 2025-11-20 13:10:03
**Ticket:** 11582079

---

## Input Data from Logs

### IB (Initial Balance) Parameters
Calculated at 12:30:02 (Almaty time = 08:30 Berlin time):
```
IB High:  1.15218
IB Low:   1.15149
EQ (Mid): 1.15183
```

**IB period:** 08:00-08:30 (Europe/Berlin)

### OCAE SHORT Signal
Detected at 13:10:03 (Almaty time = 09:10 Berlin time):
```
Direction:    SHORT
Entry Price:  1.15137
SL:           1.15310
TP:           1.14835 (virtual)
```

---

## Step 1: OCAE Variation Parameters

From `src/utils/strategy_logic.py` lines 187-199:

```python
"OCAE": {
    "IB_START": "08:00",
    "IB_END": "08:30",
    "IB_TZ": "Europe/Berlin",
    "IB_WAIT": 15,                # Wait 15 minutes after IB_END
    "TRADE_WINDOW": 180,          # Signal detection window: 180 minutes
    "RR_TARGET": 1.75,            # Risk:Reward = 1:1.75
    "STOP_MODE": "ib_start",      # SL based on IB_START price
    "TSL_TARGET": 2.0,            # TSL TP step
    "TSL_SL": 1.5,                # TSL SL step
    "MIN_SL_PCT": 0.0015,         # Minimum SL = 0.15% of entry
    "REV_RB_ENABLED": True
}
```

**Key parameter:** `STOP_MODE = "ib_start"`

---

## Step 2: Direction Determination

From `src/strategies/ib_strategy.py` line 378:

```python
direction = "long" if float(entry_candle["close"]) > self.ibh else "short"
```

**Logic:**
- If close > IB High (1.15218) -> LONG
- If close < IB High -> SHORT

**In our case:**
```
Entry close: 1.15137
IB High:     1.15218

1.15137 < 1.15218 -> SHORT
```

---

## Step 3: trade_start_price Calculation (IB_START price)

From `src/utils/strategy_logic.py` function `get_trade_window()`:

```python
# For OCAE: trade starts at first bar AFTER IB_WAIT period
# IB_END = 08:30, IB_WAIT = 15 min
# Trade window starts at 08:45 Berlin time

# trade_start_price = OPEN price of first trade window bar (08:45)
```

**Meaning:**

1. IB ended: 08:30 Berlin
2. Wait IB_WAIT: 15 minutes
3. Trade window starts: 08:45 Berlin
4. `trade_start_price` = OPEN price of M2 candle at 08:45:00

Based on IB data, approximately:
```
trade_start_price = 1.15183 (around EQ)
```

---

## Step 4: Initial Stop Price Calculation

From `src/utils/strategy_logic.py` lines 663-670:

```python
def initial_stop_price(trade_start_price: float, eq_price: float,
                       cisd_price: Optional[float], stop_mode: str) -> float:
    """Initial stop price based on STOP_MODE"""
    if stop_mode.lower() == "eq":
        return float(eq_price)
    elif stop_mode.lower() == "cisd" and cisd_price is not None:
        return float(cisd_price)
    else:  # "ib_start"
        return float(trade_start_price)
```

**In our case:**
```python
stop_mode = "ib_start"  # From OCAE parameters

# So:
stop_price = trade_start_price = 1.15183
```

---

## Step 5: Minimum SL Check

From `src/utils/strategy_logic.py` lines 672-690:

```python
def place_sl_tp_with_min_size(direction: str, entry_price: float,
                               stop_price: float, rr_target: float,
                               min_sl_pct: float) -> Tuple[float, float, bool]:
    """Place SL and TP considering minimum SL size"""

    # 1. Calculate current risk
    risk = abs(entry_price - stop_price)

    # 2. Calculate minimum allowed SL
    min_sl_size = entry_price * min_sl_pct

    # 3. If risk less than minimum - increase
    adjusted = False
    if risk < min_sl_size:
        adjusted = True
        if direction == "long":
            stop_price = entry_price - min_sl_size
        else:  # SHORT
            stop_price = entry_price + min_sl_size
        risk = min_sl_size

    # 4. Calculate TP
    if direction == "long":
        tp = entry_price + rr_target * risk
    else:  # SHORT
        tp = entry_price - rr_target * risk

    return float(stop_price), float(tp), adjusted
```

**Applying to our trade:**

### Step 5.1: Calculate initial risk

```python
entry_price = 1.15137
stop_price = 1.15183  # (trade_start_price)

risk = abs(1.15137 - 1.15183)
     = abs(-0.00046)
     = 0.00046  # 4.6 pips
```

### Step 5.2: Calculate minimum SL

```python
min_sl_pct = 0.0015  # 0.15% from OCAE parameters

min_sl_size = entry_price * min_sl_pct
            = 1.15137 * 0.0015
            = 0.00172705  # 17.3 pips
```

### Step 5.3: Compare

```python
risk = 0.00046  # 4.6 pips
min_sl_size = 0.00172705  # 17.3 pips

0.00046 < 0.00172705  # TRUE! Adjustment needed!
```

**CRITICAL:** Initial SL is too small (4.6 pips)!

### Step 5.4: SL Adjustment for SHORT

```python
adjusted = True
direction = "short"

# For SHORT: SL must be ABOVE entry
stop_price = entry_price + min_sl_size
           = 1.15137 + 0.00172705
           = 1.15309705

# Rounded to 5 decimals:
stop_price = 1.15310
```

### Step 5.5: Final risk

```python
risk = min_sl_size = 0.00172705  # 17.3 pips
```

---

## Step 6: TP Calculation

```python
direction = "short"
rr_target = 1.75  # From OCAE parameters
risk = 0.00172705

# For SHORT: TP below entry
tp = entry_price - (rr_target * risk)
   = 1.15137 - (1.75 * 0.00172705)
   = 1.15137 - 0.00302233
   = 1.14834767

# Rounded:
tp = 1.14835
```

---

## Final Values

```
Entry:  1.15137
SL:     1.15310  (entry + 0.00173, above entry for SHORT)
TP:     1.14835  (entry - 0.00302, below entry for SHORT)

Risk:   0.00173  (17.3 pips)
Reward: 0.00302  (30.2 pips)

RR Ratio: 30.2 / 17.3 = 1.75 (matches RR_TARGET)
```

---

## Visualization for SHORT

```
        SL: 1.15310  <--- Stop ABOVE entry (protection from price rise)
          |
       +17.3 pips
          |
     Entry: 1.15137  <--- Entry point
          |
       -30.2 pips
          |
        TP: 1.14835  <--- Take-profit BELOW entry (target on price fall)
```

---

## Why SL = 1.15310?

### Reason: Minimum SL Size

**Without MIN_SL_PCT:**
- SL would be = trade_start_price = 1.15183
- Risk = 0.00046 (4.6 pips)
- This is too small! Spread can eat the entire risk!

**With MIN_SL_PCT = 0.0015:**
- Minimum risk = 1.15137 * 0.0015 = 0.00173 (17.3 pips)
- SL adjusted: 1.15137 + 0.00173 = 1.15310
- This ensures sufficient distance for normal trade operation

---

## Summary

### How bot calculated SL = 1.15310:

1. **IB calculated:** High=1.15218, Low=1.15149, EQ=1.15183
2. **OCAE SHORT signal detected:** Entry=1.15137
3. **STOP_MODE = "ib_start":** Initial stop_price = trade_start_price = 1.15183
4. **Initial risk calculated:** risk = abs(1.15137 - 1.15183) = 0.00046 (4.6 pips)
5. **MIN_SL_PCT = 0.0015 check:** min_sl_size = 1.15137 * 0.0015 = 0.00173 (17.3 pips)
6. **0.00046 < 0.00173 -> ADJUSTMENT NEEDED**
7. **SHORT adjustment:** stop_price = entry + min_sl_size = 1.15137 + 0.00173 = **1.15310**
8. **TP calculation:** tp = entry - (1.75 * 0.00173) = **1.14835**

---

**Conclusion:** SL was adjusted from too small value (4.6 pips) to minimum allowed (17.3 pips) to ensure normal trade operation and protection from spread/market noise.
