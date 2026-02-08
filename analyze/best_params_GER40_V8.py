# GER40 V8 parameters - DB Optimizer (2026-01-20)
# Mode: IB + Buffer (same for all variations)
# Best: IB 08:00-08:30 (Europe/Berlin) Wait 15m
# Buffer: 20%, MaxDist: 150%
# Combined Total R: 154.26, Weighted Sharpe: 1.90, Total Trades: 740
GER40_PARAMS_V8 = {
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
        "MAX_DISTANCE_PCT": 1.0,
        # R: 16.77, Sharpe: 0.79, WR: 28.5%, Trades: 242, MaxDD: 11.8
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
        "MAX_DISTANCE_PCT": 1.5,
        # R: 47.99, Sharpe: 1.79, WR: 41.4%, Trades: 297, MaxDD: 12.0
    },
}