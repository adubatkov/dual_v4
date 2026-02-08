# XAUUSD V8_STRICT parameters - DB Optimizer (2026-01-20)
# Mode: IB + Buffer + MaxDist (same for all variations)
# Best: IB 09:00-09:30 (Asia/Tokyo) Wait 20m
# Buffer: 5%, MaxDist: 150%
# Combined Total R: 98.68, Weighted Sharpe: 1.81, Total Trades: 668
XAUUSD_PARAMS_V8_STRICT = {
    "REV_RB": {
        "IB_START": "09:00",
        "IB_END": "09:30",
        "IB_TZ": "Asia/Tokyo",
        "IB_WAIT": 20,
        "TRADE_WINDOW": 90,
        "RR_TARGET": 2.0,
        "STOP_MODE": "eq",
        "TSL_TARGET": 1.25,
        "TSL_SL": 0.75,
        "MIN_SL_PCT": 0.001,
        "REV_RB_PCT": 1.0,
        "REV_RB_ENABLED": False,
        "IB_BUFFER_PCT": 0.05,
        "MAX_DISTANCE_PCT": 1.5,
        # R: 0.00, Sharpe: 0.00, WR: 0.0%, Trades: 0, MaxDD: 0.0
    },
    "Reverse": {
        "IB_START": "09:00",
        "IB_END": "09:30",
        "IB_TZ": "Asia/Tokyo",
        "IB_WAIT": 20,
        "TRADE_WINDOW": 240,
        "RR_TARGET": 1.5,
        "STOP_MODE": "ib_start",
        "TSL_TARGET": 1.0,
        "TSL_SL": 1.0,
        "MIN_SL_PCT": 0.001,
        "REV_RB_ENABLED": False,
        "IB_BUFFER_PCT": 0.05,
        "MAX_DISTANCE_PCT": 1.5,
        # R: 44.06, Sharpe: 3.53, WR: 47.6%, Trades: 97, MaxDD: 11.5
    },
    "TCWE": {
        "IB_START": "09:00",
        "IB_END": "09:30",
        "IB_TZ": "Asia/Tokyo",
        "IB_WAIT": 20,
        "TRADE_WINDOW": 210,
        "RR_TARGET": 0.5,
        "STOP_MODE": "ib_start",
        "TSL_TARGET": 2.0,
        "TSL_SL": 1.25,
        "MIN_SL_PCT": 0.001,
        "REV_RB_ENABLED": False,
        "IB_BUFFER_PCT": 0.05,
        "MAX_DISTANCE_PCT": 1.5,
        # R: 25.10, Sharpe: 1.20, WR: 68.3%, Trades: 344, MaxDD: 9.3
    },
    "OCAE": {
        "IB_START": "09:00",
        "IB_END": "09:30",
        "IB_TZ": "Asia/Tokyo",
        "IB_WAIT": 20,
        "TRADE_WINDOW": 240,
        "RR_TARGET": 1.25,
        "STOP_MODE": "eq",
        "TSL_TARGET": 0.0,
        "TSL_SL": 0.5,
        "MIN_SL_PCT": 0.001,
        "REV_RB_ENABLED": False,
        "IB_BUFFER_PCT": 0.05,
        "MAX_DISTANCE_PCT": 1.5,
        # R: 29.52, Sharpe: 1.99, WR: 53.7%, Trades: 227, MaxDD: 7.2
    },
}