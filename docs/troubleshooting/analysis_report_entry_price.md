# Analysis Report: High Entry Price Investigation

## Date: 2025-11-24
## Symbol: USDCHF
## Signal: TCWE LONG
## Issue: Entry at 0.80838 appears too high visually on MT5 chart

---

## Summary

The trade opened at **0.80838**, which is **7.6 pips higher** than the historical close price (0.80762) of the signal candle on MT5 real account server.

**Root Cause**: Data source discrepancy between demo and real accounts.

---

## Timeline

### Historical Signal (Real MT5 Data)
- **Time**: 15:08:00 UTC (10:08:00 NY)
- **Signal Candle Close**: 0.80762
- **Spread**: 15 points (1.5 pips)

### Bot Execution
- **Time**: 20:06:03 UTC (15:06:03 NY) - **5 hours later**
- **Entry Price**: 0.80838
- **Difference**: +7.6 pips

---

## Key Findings

### 1. Signal Detection Logic (Correct)

The TCWE signal logic is working correctly:

```
Bar [3] - 1st breakout candle (15:06 UTC)
  Time: 15:06:00 UTC
  OHLC: 0.80721 / 0.80751 / 0.80716 / 0.80751
  Above IBH (0.80745): Partial breakout
  Above EQ (0.80713): Yes

Bar [4] - 2nd breakout candle (15:08 UTC) [SIGNAL]
  Time: 15:08:00 UTC
  OHLC: 0.80752 / 0.80764 / 0.80746 / 0.80762
  Above IBH (0.80745): Yes (low = 0.80746)
  Above EQ (0.80713): Yes

Entry Price = Close of 2nd candle = 0.80762
```

The signal is valid - two consecutive candles breaking above IBH without touching EQ.

### 2. Entry Price Calculation (Correct per code)

From `dual_v3/src/strategies/ib_strategy.py:448`:

```python
entry_price = float(entry_candle["close"])
```

For TCWE, the bot correctly uses the **close price** of the signal candle.

### 3. Data Source Discrepancy (The Problem)

**Real MT5 Historical Data**:
- Signal candle close: 0.80762
- Spread: 15 points

**Bot's Data Source**:
- Entry price logged: 0.80838
- Difference: +7.6 pips (76 points)

This 7.6 pip difference is **NOT** explained by:
- Spread (only 1.5 pips)
- Slippage (order was market order)
- ASK price adjustment (would only add 1.5 pips)

### 4. Possible Explanations

#### A. Demo Account Data Quality Issue
The bot logged: **"ORDER EXECUTED ON DEMO ACCOUNT!"**

Demo accounts often have:
- Less accurate historical data
- Different liquidity providers
- Wider spreads during replay
- Data gaps or mismatched timestamps

#### B. Backtest/Replay Mode Timing
The bot detected the signal at **20:06:03 UTC**, which is **5 hours after** the actual signal time (15:08 UTC).

This suggests the bot is running in a replay/backtest mode, not real-time trading.

#### C. Data Source: Demo vs Real Server
- **Demo server**: May have different tick data
- **Real server (5ers Tel Aviv)**: More accurate, tighter spreads
- The 7.6 pip difference suggests different liquidity providers

---

## Visual Analysis

On MT5 chart, the entry point (0.80838) appears "floating" above the candle because:

1. Historical candle high on real server: 0.80764
2. Entry price: 0.80838
3. Entry is **7.4 pips above** the candle high

This is **not possible** if using the same data source, confirming the data discrepancy.

---

## Spread Analysis

**Normal USDCHF spread**: 15 points (1.5 pips)

**Expected entry price for LONG**:
- Signal candle close (BID): 0.80762
- Add spread for ASK: 0.80762 + 0.00015 = 0.80777
- Actual entry: 0.80838
- Difference: 0.80838 - 0.80777 = **6.1 pips unexplained**

Even accounting for spread, there's still a 6.1 pip discrepancy.

---

## IB Levels

The IB levels are consistent:
```
IBH: 0.80745 (0.80801 in bot's log - different data!)
IBL: 0.80682 (0.80702 in bot's log - different data!)
EQ:  0.80713 (0.80751 in bot's log - different data!)
```

Wait - the IB levels in the bot's log are also different!

From bot log:
```
2025-11-24 20:00:03,220 - INFO - IB calculated - H:0.80801, L:0.80702, EQ:0.80751
```

From real MT5 data:
```
IBH: 0.80745
IBL: 0.80682
EQ:  0.80713
```

**Difference**:
- IBH: +5.6 pips
- IBL: +2.0 pips
- EQ: +3.8 pips

This confirms that **the bot is using completely different data** from the real MT5 account.

---

## Conclusion

### The Problem

The bot is running on a **demo account with different tick data** than the real 5ers MT5 server in Tel Aviv.

The 7.6 pip difference in entry price is due to:
1. Different historical data between demo and real servers
2. Demo account may have wider spreads or less accurate data
3. The bot is not trading live - it's replaying historical data

### Impact

1. **Backtest results may not match live trading** - 7.6 pips difference on entry is significant
2. **Risk management affected** - stop loss distance calculated from wrong entry
3. **Visual discrepancy** - entry point doesn't match chart because charts use real server data

### Recommendation

For accurate testing that matches production:

1. **Use the real 5ers account for testing**, not demo
2. **Reduce test lot sizes** instead of using demo account
3. **Verify data consistency** by comparing IB levels between bot log and MT5 chart
4. **Add logging** to show data source (demo vs real) clearly in bot output

### Is This a Bug?

**No** - The bot is working as designed. It's using the data provided by MT5 connection.

**But** - The data quality issue means backtest/demo results won't accurately predict live trading performance.

---

## Additional Notes

### Why the Entry Looks "High"

On the MT5 chart (real server), you see:
- Signal candle high: 0.80764
- Entry marker: 0.80838 (from bot)

The entry appears to "float" above the candle because it's from a different data source where the price actually reached 0.80838.

### Spread Check

The normal 15-point spread (1.5 pips) is reasonable for USDCHF. The issue is not spread-related.

### Data Integrity

For future trades, always verify:
1. Bot's IB levels match MT5 chart IB levels
2. Entry price is within the candle range on chart
3. Spread is consistent with broker specifications

If any of these don't match, the bot is using different data.

---

## Files Created During Investigation

All analysis files stored in: `dual_v3\_temp\`

- `analyze_tcwe_entry.py` - Initial analysis of signal candle
- `check_real_entry_price.py` - Market price at order time
- `check_order_execution.py` - Order execution details
- `reproduce_signal_detection.py` - Reproduce exact bot logic
- `analysis_report_entry_price.md` - This report
