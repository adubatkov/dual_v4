"""
Configuration file for dual_v4 Strategy R&D Workspace

This workspace is for strategy evolution and backtesting only.
No live trading capabilities - credentials removed.
"""
import os

# ================================
# SYMBOL MAPPING (MT5 vs Strategy Names)
# ================================
SYMBOL_MAPPING = {
    "GER40": "DAX40",     # 5ers uses DAX40 for German index
    "XAUUSD": "XAUUSD"    # Gold vs USD (standard name)
}

# ================================
# MAGIC NUMBERS
# ================================
MAGIC_NUMBER_GER40 = 1001
MAGIC_NUMBER_XAUUSD = 1002

# ================================
# STRATEGY PARAMETERS
# ================================
# GER40_PARAMS_PROD and XAUUSD_PARAMS_PROD are imported from src.utils.strategy_logic

# ================================
# RISK MANAGEMENT (backtest defaults)
# ================================
RISK_PER_TRADE_PCT = 0.9   # 0.9% per trade
MAX_DAILY_LOSS_PCT = 3.0   # 3% max daily loss
MAX_MARGIN_USAGE_PCT = 40.0  # 40% max margin

# ================================
# LOGGING SETTINGS
# ================================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("LOG_FILE", "bot.log")
