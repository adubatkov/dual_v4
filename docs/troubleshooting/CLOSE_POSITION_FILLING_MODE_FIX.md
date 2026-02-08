# Close Position Filling Mode Fix

**Date:** 2025-11-20 18:30
**Issue:** retcode=10030 when closing positions
**Status:** FIXED

---

## Problem

When bot tried to close position at window expiration:

```
2025-11-20 18:25:14 - INFO - [EURUSD_M2001] Position window expired, closing position 11582079
2025-11-20 18:25:14 - INFO - CLOSING POSITION 11582079: EURUSD 3.07 lots at market price 1.15144
2025-11-20 18:25:14 - ERROR - Failed to close position 11582079: retcode=10030 (Unknown error 10030)
2025-11-20 18:25:14 - ERROR -   Request: Symbol=EURUSD, Volume=3.07, Price=1.15144
```

**Error Code 10030:** Invalid order filling type

---

## Root Cause

### Original Code (WRONG)

`dual_v3/src/mt5_executor.py:677` (before fix):

```python
# Prepare close request
request = {
    "action": mt5.TRADE_ACTION_DEAL,
    "symbol": symbol,
    "volume": volume,
    "type": trade_type,
    "position": ticket,
    "price": price,
    "deviation": 20,
    "magic": position.magic,
    "comment": "Close by time-based exit",
    "type_time": mt5.ORDER_TIME_GTC,
    "type_filling": mt5.ORDER_FILLING_IOC,  # ❌ HARDCODED!
}
```

**Problem:** Hardcoded `ORDER_FILLING_IOC` but broker doesn't support it for this symbol.

### Why This Happened

When `close_position()` was added (2025-11-20 ~07:30), we copied the request structure but forgot to add the filling_mode auto-detection logic that was already implemented in `open_position()`.

---

## Solution

### Auto-detect Filling Mode (Like open_position)

`dual_v3/src/mt5_executor.py:651-670` (after fix):

```python
# Get symbol info (including filling_mode)
symbol_info_dict = self.get_symbol_info(symbol)
if symbol_info_dict is None:
    logger.error(f"Failed to get symbol info for {symbol}")
    return False

# Determine best filling type based on broker support
filling_mode = symbol_info_dict.get('filling_mode', 0)
if filling_mode & 2:  # SYMBOL_FILLING_IOC (bit 1)
    type_filling = mt5.ORDER_FILLING_IOC
    filling_name = "IOC"
elif filling_mode & 1:  # SYMBOL_FILLING_FOK (bit 0)
    type_filling = mt5.ORDER_FILLING_FOK
    filling_name = "FOK"
else:  # SYMBOL_FILLING_RETURN (bit 2) - market execution
    type_filling = mt5.ORDER_FILLING_RETURN
    filling_name = "RETURN"

logger.debug(f"Using filling type: {filling_name} (mode={filling_mode})")

# ... rest of code ...

request = {
    # ...
    "type_filling": type_filling,  # ✓ AUTO-DETECTED!
}
```

---

## How Filling Mode Detection Works

### Filling Mode Bitmask

MT5 symbol_info contains `filling_mode` field which is a bitmask:

```
Bit 0 (value 1): SYMBOL_FILLING_FOK (Fill or Kill)
Bit 1 (value 2): SYMBOL_FILLING_IOC (Immediate or Cancel)
Bit 2 (value 4): SYMBOL_FILLING_RETURN (Market execution)
```

**Examples:**

| filling_mode | Binary | Supported Types |
|--------------|--------|-----------------|
| 1            | 001    | FOK only |
| 2            | 010    | IOC only |
| 3            | 011    | FOK + IOC |
| 4            | 100    | RETURN only |
| 6            | 110    | IOC + RETURN |
| 7            | 111    | All three |

### Detection Logic

```python
if filling_mode & 2:  # Check bit 1
    # IOC supported (preferred)
    type_filling = mt5.ORDER_FILLING_IOC
elif filling_mode & 1:  # Check bit 0
    # FOK supported (fallback)
    type_filling = mt5.ORDER_FILLING_FOK
else:
    # Neither IOC nor FOK, use RETURN (market execution)
    type_filling = mt5.ORDER_FILLING_RETURN
```

**Priority Order:** IOC > FOK > RETURN

---

## Additional Improvements

### 1. Error Code Documentation

Added 10030 to known error codes (line 720):

```python
retcode_desc = {
    10027: "Autotrading disabled by client terminal or broker",
    10015: "Invalid price (too close to market)",
    10019: "No money (insufficient margin)",
    10025: "Trade disabled (symbol locked)",
    10030: "Invalid order filling type (auto-detection may have failed)",  # ✓ ADDED
    10004: "Requote",
}.get(result.retcode, f"Unknown error {result.retcode}")
```

### 2. Debug Logging

Added debug log to see which filling mode was selected:

```python
logger.debug(f"Using filling type: {filling_name} (mode={filling_mode})")
```

**Example output:**
```
DEBUG - Using filling type: IOC (mode=2)
```

or

```
DEBUG - Using filling type: RETURN (mode=4)
```

---

## Testing

### Before Fix

```
❌ Position close fails with retcode=10030
❌ Position remains open after window expiration
❌ No visibility into why close failed
```

### After Fix

```
✓ Position close will use correct filling_mode
✓ Debug log shows which filling_mode was selected
✓ Better error message if 10030 still occurs
✓ Consistent with open_position() logic
```

---

## Files Modified

**`dual_v3/src/mt5_executor.py`:**
- Lines 651-670: Added filling_mode auto-detection
- Line 686: Changed from hardcoded IOC to auto-detected `type_filling`
- Line 720: Added retcode 10030 to error descriptions

**`dual_v3/CHANGELOG.md`:**
- Lines 3-42: New entry documenting this fix

---

## Comparison: open_position vs close_position

### open_position (Already Correct)

Lines 421-431:
```python
filling_mode = symbol_info.get('filling_mode', 0)

if filling_mode & 2:  # SYMBOL_FILLING_IOC (bit 1)
    type_filling = mt5.ORDER_FILLING_IOC
elif filling_mode & 1:  # SYMBOL_FILLING_FOK (bit 0)
    type_filling = mt5.ORDER_FILLING_FOK
else:  # SYMBOL_FILLING_RETURN (bit 2)
    type_filling = mt5.ORDER_FILLING_RETURN
```

### close_position (Now Fixed)

Lines 657-667:
```python
filling_mode = symbol_info_dict.get('filling_mode', 0)

if filling_mode & 2:  # SYMBOL_FILLING_IOC (bit 1)
    type_filling = mt5.ORDER_FILLING_IOC
elif filling_mode & 1:  # SYMBOL_FILLING_FOK (bit 0)
    type_filling = mt5.ORDER_FILLING_FOK
else:  # SYMBOL_FILLING_RETURN (bit 2)
    type_filling = mt5.ORDER_FILLING_RETURN
```

**Now both methods use identical logic!**

---

## Related Issues

This is the same issue that was fixed for `open_position` on 2025-11-18 22:00 (see CHANGELOG.md line 678-708).

**Lesson Learned:** When adding new methods that send orders to MT5, ALWAYS use filling_mode auto-detection, never hardcode filling type!

---

## Why retcode=10030 Occurs

**MT5 Error 10030:** `TRADE_RETCODE_INVALID_FILL` - "Invalid order filling type"

**Causes:**
1. Broker doesn't support the requested filling type for this symbol
2. Different symbols may have different supported filling types
3. Forex.com demo account specifically doesn't support IOC for some pairs

**Solution:** Always query `symbol_info.filling_mode` and use appropriate type.

---

## Summary

**Before:** close_position used hardcoded ORDER_FILLING_IOC → fails with retcode=10030

**After:** close_position auto-detects filling_mode from symbol info → works with any broker/symbol

**Status:** ✅ FIXED and ready for testing

---

**Next Step:** Continue running demo bot, verify that position window expiration now closes positions successfully without errors.
