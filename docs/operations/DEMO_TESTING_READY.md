# Demo Account Testing - Ready to Run

**Date**: 2025-11-18
**Status**: READY FOR TESTING

## Summary

Bot configured for full testing on Forex.com demo account with 4 Forex pairs.
All critical bugs fixed, reconnect logic implemented.

## Configuration

### Demo Account
- **Server**: Forex.comGlobal-Demo
- **Login**: 22597078
- **Balance**: $50,000 (demo)
- **Risk**: 1% per trade

### Trading Pairs
1. **EURUSD** - Magic 2001 - IB 08:00-08:30 Europe/Berlin
2. **GBPUSD** - Magic 2002 - IB 08:00-08:30 Europe/London
3. **USDJPY** - Magic 2003 - IB 09:00-10:00 Asia/Tokyo
4. **USDCHF** - Magic 2004 - IB 09:00-10:00 America/New_York

## Critical Fixes Applied

### 1. Signal AttributeError Fixed
**Issue**: `'Signal' object has no attribute 'get'`
**Fix**: Changed `mt5_executor.place_order()` to use Signal object attributes instead of dictionary `.get()`

**Files modified**:
- `dual_v3/src/mt5_executor.py` - signature changed
- `dual_v3/src/bot_controller.py` - caller updated

### 2. MT5 IPC Error Fixed
**Issue**: `(-10001, 'IPC send failed')` after ~18 minutes
**Fix**: Added automatic reconnect logic

**Changes**:
- Store credentials in MT5Executor for reconnect
- Detect IPC error in `get_open_positions()`
- Call `_handle_connection_loss()` to reconnect automatically
- Clear symbol cache after reconnect

### 3. Logging Improvements
Added strategy prefixes for easy identification:
- Format: `[{symbol}_M{magic}_{label}]`
- Example: `[EURUSD_M2001_IB08:00-08:30]`

### 4. State Persistence Fixed
**Issue**: Bot opens duplicate trades after restart
**Fix**: Restore state from MT5 positions/history on startup

**Changes**:
- Added `mt5_executor.get_positions_history_today()` - query MT5 history
- Added `state_manager.restore_state_from_mt5()` - restore traded state
- Bot checks both open positions AND today's history on startup
- Prevents duplicate trades even after restart

## How to Run

### Start Bot:
```bash
cd C:\Trading\ib_trading_bot\dual_v3
python run_demo.py
```

### Stop Bot:
Press **Ctrl+C**

### View Logs:
- **Console**: Real-time output
- **File**: `logs/demo_test.log`

## What to Test

1. **Connection**: Verify connects to demo account
2. **IB Calculation**: Check all 4 pairs calculate IB correctly
3. **Signal Detection**: Monitor for Reverse, OCAE, TCWE, REV_RB signals
4. **Order Execution**: Verify real orders placed on demo account
5. **TSL Logic**: Check trailing stop loss works with open positions
6. **Reconnect**: Verify auto-reconnect if IPC error occurs
7. **Multi-pair**: Ensure all 4 pairs work simultaneously

## Expected Behavior

### Startup
```
INFO - Connecting to MT5 DEMO account...
INFO - MT5 connected successfully
INFO - Strategy: EURUSD - Magic 2001
INFO - Strategy: GBPUSD - Magic 2002
INFO - Strategy: USDJPY - Magic 2003
INFO - Strategy: USDCHF - Magic 2004
INFO - RESTORING STATE FROM MT5
INFO - STATE RESTORATION COMPLETE
INFO - STARTING MAIN LOOP (INFINITE)
```

### IB Calculation
```
INFO - [EURUSD_M2001_IB08:00-08:30] IB calculated - H:1.05234, L:1.05123, EQ:1.05179
INFO - [EURUSD_M2001_IB08:00-08:30] Trade window opened until 11:45:00
```

### Signal Detection
```
INFO - SIGNAL DETECTED: EURUSD
INFO - Variation: OCAE
INFO - Direction: LONG
INFO - Entry: 1.05234
INFO - SL: 1.05123
INFO - TP: 1.05456
INFO - Trade validation PASSED: 0.98 lots
INFO - ORDER EXECUTED ON DEMO ACCOUNT!
```

### Reconnect (if IPC error)
```
ERROR - MT5 IPC Error: (-10001, 'IPC send failed'). Attempting reconnect...
WARNING - MT5 CONNECTION LOST - Attempting to reconnect
INFO - MT5 reconnected successfully
```

## Monitoring

### Progress Updates
Bot logs progress every 5 minutes:
```
INFO - [300s / 0.08h] Running... | Open positions: 0
INFO - [600s / 0.17h] Running... | Open positions: 1
```

### Position Status
Final stats shown on exit:
```
FINAL DEMO ACCOUNT STATUS:
  Balance: 50000.00
  Equity: 50123.45
  Profit: 123.45
  Margin Used: 256.78
```

## Switching Back to Prop Account

When ready to use real prop account:

1. **Edit `.env`**:
   - Comment demo credentials
   - Uncomment 5ers credentials

2. **Edit `config.py`**:
   - Set `RISK_PER_TRADE_PCT = 0.2`
   - Set `MAX_DAILY_LOSS_PCT = 3.0`
   - Set `MAX_MARGIN_USAGE_PCT = 40.0`
   - Uncomment original `SYMBOL_MAPPING` (DAX40, XAUUSD)
   - Uncomment original magic numbers (1001, 1002)

3. **Update `bot_controller.py`**:
   - Modify `_initialize_strategies()` to use GER40/XAUUSD

## Important Notes

- This is DEMO account - no real money at risk
- Orders ARE executed on demo account
- Test thoroughly before switching to prop account
- Monitor for at least 24-48 hours
- Check all 4 pairs work correctly
- Verify TSL logic with real positions

## Known Issues

None currently. All critical issues fixed.

## Files Modified

**Configuration**:
- `dual_v3/.env` - demo credentials
- `dual_v3/config.py` - forex pairs, 1% risk
- `dual_v3/src/utils/strategy_logic.py` - forex params

**Core Bot** (permanent changes):
- `dual_v3/src/mt5_executor.py` - IPC fix, reconnect logic, place_order signature
- `dual_v3/src/state_manager.py` - state restoration from MT5
- `dual_v3/src/bot_controller.py` - place_order caller, state restoration call
- `dual_v3/src/strategies/ib_strategy.py` - logging prefixes

**Testing**:
- `dual_v3/run_demo.py` - demo test script

## Ready to Test!

Bot is fully configured and ready for demo testing.
All critical issues fixed.
Run `python run_demo.py` to start.
