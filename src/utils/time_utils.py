"""
Time utilities for dual_v4 trading bot

КРИТИЧЕСКИ ВАЖНО: Правильная работа со временем

Проблема:
- MT5 сервер 5ers находится в Тель-Авиве (UTC+2)
- MT5 API возвращает время в виде Unix timestamp, который мы конвертируем в UTC
- Но сервер MT5 МОЖЕТ возвращать локальное серверное время, а не UTC!

Решение:
- MT5 copy_rates_from_pos возвращает Unix timestamp (seconds since epoch)
- Pandas конвертирует его в UTC: pd.to_datetime(timestamp, unit='s', utc=True)
- Но ВНИМАНИЕ: MT5 может возвращать локальное время сервера в timestamp!
- Мы должны ВСЕГДА явно конвертировать в нужную временную зону

Архитектура:
1. BOT_TIMEZONE = "Asia/Almaty" (UTC+5) - локальное время бота
2. MT5_SERVER_TIMEZONE = "Asia/Jerusalem" (UTC+2) - время сервера MT5
3. STRATEGY_TIMEZONE - "Europe/Berlin" для DAX, "Asia/Tokyo" для XAU

Процесс:
1. MT5 возвращает timestamp
2. Мы предполагаем, что это UTC (как документировано в MT5 API)
3. Конвертируем в нужную временную зону стратегии для расчетов IB
"""

import pandas as pd
import pytz
from datetime import datetime, date, time as dt_time
from typing import Tuple
import logging

logger = logging.getLogger(__name__)

# ================================
# КОНСТАНТЫ ВРЕМЕННЫХ ЗОН
# ================================

# ВАЖНО: Эти значения импортируются из config.py
BOT_TIMEZONE = "Asia/Almaty"  # UTC+5
MT5_SERVER_TIMEZONE = "Asia/Jerusalem"  # UTC+2 (Тель-Авив)

# ================================
# КОНВЕРТАЦИЯ ВРЕМЕНИ MT5
# ================================

def convert_mt5_time_to_utc(timestamp: int) -> pd.Timestamp:
    """
    Конвертирует MT5 timestamp в UTC pandas Timestamp

    ВАЖНО: MT5 API документирует, что copy_rates_from_pos возвращает
    Unix timestamp в UTC. Но на практике может быть локальное время сервера!

    Решение: Мы предполагаем, что MT5 возвращает локальное время сервера,
    и конвертируем его из MT5_SERVER_TIMEZONE в UTC.

    Args:
        timestamp: Unix timestamp из MT5

    Returns:
        UTC pandas Timestamp
    """
    # Создаем naive datetime из timestamp
    naive_dt = datetime.fromtimestamp(timestamp)

    # Локализуем как серверное время MT5
    server_tz = pytz.timezone(MT5_SERVER_TIMEZONE)
    server_dt = server_tz.localize(naive_dt)

    # Конвертируем в UTC
    utc_dt = server_dt.astimezone(pytz.UTC)

    return pd.Timestamp(utc_dt)


def get_current_time_in_timezone(timezone_str: str) -> datetime:
    """
    Получить текущее время в указанной временной зоне

    Args:
        timezone_str: Строка временной зоны (например, "Europe/Berlin")

    Returns:
        datetime с указанной временной зоной
    """
    tz = pytz.timezone(timezone_str)
    return datetime.now(tz)


def get_current_date_in_timezone(timezone_str: str) -> date:
    """
    Получить текущую дату в указанной временной зоне

    Args:
        timezone_str: Строка временной зоны

    Returns:
        date объект
    """
    return get_current_time_in_timezone(timezone_str).date()


def localize_time_to_timezone(dt: datetime, timezone_str: str) -> datetime:
    """
    Локализовать naive datetime в указанную временную зону

    Args:
        dt: Naive datetime
        timezone_str: Строка временной зоны

    Returns:
        Aware datetime
    """
    tz = pytz.timezone(timezone_str)
    if dt.tzinfo is None:
        return tz.localize(dt)
    else:
        return dt.astimezone(tz)


def create_datetime_in_timezone(
    local_date: date,
    time_str: str,
    timezone_str: str
) -> pd.Timestamp:
    """
    Создать datetime для указанной даты, времени и временной зоны

    Args:
        local_date: Дата (например, date(2025, 11, 17))
        time_str: Время в формате "HH:MM" (например, "08:00")
        timezone_str: Строка временной зоны (например, "Europe/Berlin")

    Returns:
        pd.Timestamp с временной зоной
    """
    tz = pytz.timezone(timezone_str)

    # Парсим время
    hour, minute = map(int, time_str.split(':'))
    time_obj = dt_time(hour, minute)

    # Создаем naive datetime
    naive_dt = datetime.combine(local_date, time_obj)

    # Локализуем в нужную временную зону
    aware_dt = tz.localize(naive_dt)

    return pd.Timestamp(aware_dt)


# ================================
# ЛОГИРОВАНИЕ ВРЕМЕНИ
# ================================

def log_time_comparison(label: str, timestamp: pd.Timestamp):
    """
    Логировать время в разных временных зонах для отладки

    Args:
        label: Метка для лога
        timestamp: pd.Timestamp для сравнения
    """
    if timestamp.tzinfo is None:
        logger.warning(f"{label}: Timestamp is naive (no timezone info)")
        return

    utc_time = timestamp.astimezone(pytz.UTC)
    almaty_time = timestamp.astimezone(pytz.timezone(BOT_TIMEZONE))
    server_time = timestamp.astimezone(pytz.timezone(MT5_SERVER_TIMEZONE))

    logger.debug(f"{label}:")
    logger.debug(f"  UTC: {utc_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    logger.debug(f"  Almaty: {almaty_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    logger.debug(f"  MT5 Server: {server_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
