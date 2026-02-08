"""
Configuration file for dual_v4 strategy R&D workspace

ВАЖНО: Временные зоны и их роль:
- BOT_TIMEZONE: Локальное время бота (Алматы, UTC+5)
- MT5_SERVER_TIMEZONE: Время сервера MT5 (5ers Тель-Авив, UTC+2)
- STRATEGY_TIMEZONES: Временные зоны для стратегий (Berlin для DAX, Tokyo для XAU)

Логика работы:
1. MT5 возвращает время баров в виде Unix timestamp (UTC)
2. Мы конвертируем его в локальное время бота (Алматы)
3. Затем преобразуем в нужную временную зону стратегии (Berlin/Tokyo)
"""

# ================================
# ВРЕМЕННЫЕ ЗОНЫ
# ================================

# Локальная временная зона бота (Алматы, Казахстан)
BOT_TIMEZONE = "Asia/Almaty"  # UTC+5 (зимой) / UTC+6 (летом)

# Временная зона сервера MT5 (5ers - Тель-Авив)
MT5_SERVER_TIMEZONE = "Asia/Jerusalem"  # UTC+2 (зимой) / UTC+3 (летом)

# Временные зоны для стратегий
STRATEGY_TIMEZONES = {
    "DAX40": "Europe/Berlin",  # UTC+1 (зимой) / UTC+2 (летом)
    "XAUUSD": "Asia/Tokyo"     # UTC+9
}

# ================================
# MT5 CONNECTION (not used in R&D, kept for reference)
# ================================

# MT5 credentials removed for dual_v4 (R&D workspace, no live trading)
# For live trading, see dual_v3/src/config.py or .env

# Symbol mapping (GER40 в бэктесте -> DAX40 на 5ers)
SYMBOL_MAPPING = {
    "GER40": "DAX40",
    "XAUUSD": "XAUUSD"
}

# ================================
# RISK MANAGEMENT
# ================================

# Risk per trade (0.2% for prop account)
RISK_PER_TRADE = 0.002  # 0.2%

# Maximum margin usage
MAX_MARGIN = 0.40  # 40%

# Maximum daily loss
MAX_DAILY_LOSS = 0.03  # 3%

# ================================
# DRY RUN MODE
# ================================

# DRY_RUN mode: if True, orders are logged but NOT executed
import os
DRY_RUN = os.environ.get('DRY_RUN', 'false').lower() == 'true'

# ================================
# MAGIC NUMBERS
# ================================

MAGIC_NUMBERS = {
    "DAX40": 1001,
    "XAUUSD": 1002
}

# ================================
# STRATEGY PARAMETERS
# ================================

# Imported from strategy_logic.py
# See src/utils/strategy_logic.py for GER40_PARAMS_PROD and XAUUSD_PARAMS_PROD
