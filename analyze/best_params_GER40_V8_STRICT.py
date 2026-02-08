# GER40 V8_STRICT parameters - DB Optimizer (2026-01-20)
# Mode: IB + Buffer + MaxDist (same for all variations)
# Best: IB 08:00-08:30 (Europe/Berlin) Wait 15m
# Buffer: 20%, MaxDist: 50%
# Combined Total R: 148.09, Weighted Sharpe: 1.99, Total Trades: 666
GER40_PARAMS_V8_STRICT = {
    "REV_RB": {
        "IB_START": "08:00",
        "IB_END": "08:30",
        "IB_TZ": "Europe/Berlin",
        "IB_WAIT": 15,
        "TRADE_WINDOW": 240,
        "RR_TARGET": 1.5,
        "STOP_MODE": "ib_start",
        "TSL_TARGET": 2.0,
        "TSL_SL": 1.5,
        "MIN_SL_PCT": 0.0015,
        "REV_RB_PCT": 1.0,
        "REV_RB_ENABLED": True,
        "IB_BUFFER_PCT": 0.2,
        "MAX_DISTANCE_PCT": 0.5,
        # R: 35.46, Sharpe: 7.16, WR: 61.0%, Trades: 41, MaxDD: 4.0
    },
    "Reverse": {
        "IB_START": "08:00",
        "IB_END": "08:30",
        "IB_TZ": "Europe/Berlin",
        "IB_WAIT": 15,
        "TRADE_WINDOW": 240,
        "RR_TARGET": 0.5,
        "STOP_MODE": "ib_start",
        "TSL_TARGET": 1.5,
        "TSL_SL": 0.75,
        "MIN_SL_PCT": 0.0015,
        "REV_RB_ENABLED": False,
        "IB_BUFFER_PCT": 0.2,
        "MAX_DISTANCE_PCT": 0.5,
        # R: 54.04, Sharpe: 2.43, WR: 26.2%, Trades: 160, MaxDD: 11.5
    },
    "TCWE": {
        "IB_START": "08:00",
        "IB_END": "08:30",
        "IB_TZ": "Europe/Berlin",
        "IB_WAIT": 15,
        "TRADE_WINDOW": 90,
        "RR_TARGET": 0.5,
        "STOP_MODE": "ib_start",
        "TSL_TARGET": 1.25,
        "TSL_SL": 0.75,
        "MIN_SL_PCT": 0.0015,
        "REV_RB_ENABLED": False,
        "IB_BUFFER_PCT": 0.2,
        "MAX_DISTANCE_PCT": 0.5,
        # R: 12.50, Sharpe: 0.81, WR: 32.8%, Trades: 177, MaxDD: 11.4
    },
    "OCAE": {
        "IB_START": "08:00",
        "IB_END": "08:30",
        "IB_TZ": "Europe/Berlin",
        "IB_WAIT": 15,
        "TRADE_WINDOW": 210,
        "RR_TARGET": 1.0,
        "STOP_MODE": "ib_start",
        "TSL_TARGET": 0.75,
        "TSL_SL": 0.75,
        "MIN_SL_PCT": 0.0015,
        "REV_RB_ENABLED": False,
        "IB_BUFFER_PCT": 0.2,
        "MAX_DISTANCE_PCT": 0.5,
        # R: 46.09, Sharpe: 1.74, WR: 42.0%, Trades: 288, MaxDD: 10.5
    },
}