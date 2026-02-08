"""
Timezone utilities for News Filter.

Handles conversions between:
- Eastern Time (ET) - ForexFactory's native timezone
- UTC - Standard storage format
- Instrument timezones (Europe/Berlin for GER40, Asia/Tokyo for XAUUSD)

ForexFactory uses Eastern Time (ET):
- EST (Winter): UTC-5 (November - March)
- EDT (Summer): UTC-4 (March - November)

DST transitions are handled automatically by pytz.
"""

from datetime import datetime, date, time
from typing import Union, Optional

import pytz


# Timezone constants
TZ_EASTERN = pytz.timezone("America/New_York")  # ET (EST/EDT)
TZ_UTC = pytz.UTC
TZ_BERLIN = pytz.timezone("Europe/Berlin")  # CET/CEST
TZ_TOKYO = pytz.timezone("Asia/Tokyo")  # JST (no DST)
TZ_JERUSALEM = pytz.timezone("Asia/Jerusalem")  # MT5 server timezone

# Instrument timezone mapping
INSTRUMENT_TIMEZONES = {
    "GER40": TZ_BERLIN,
    "XAUUSD": TZ_TOKYO,
    "EURUSD": TZ_BERLIN,
    "GBPUSD": TZ_BERLIN,
    "USDJPY": TZ_TOKYO,
}


def get_instrument_timezone(symbol: str) -> pytz.BaseTzInfo:
    """
    Get the timezone for a trading instrument.

    Args:
        symbol: Trading symbol (GER40, XAUUSD, etc.)

    Returns:
        pytz timezone object
    """
    return INSTRUMENT_TIMEZONES.get(symbol.upper(), TZ_UTC)


def et_to_utc(dt: datetime) -> datetime:
    """
    Convert Eastern Time to UTC.

    Handles DST automatically (EST/EDT).

    Args:
        dt: datetime in Eastern Time (can be naive or aware)

    Returns:
        datetime in UTC with tzinfo
    """
    if dt.tzinfo is None:
        # Naive datetime - localize to ET
        dt_et = TZ_EASTERN.localize(dt)
    elif dt.tzinfo == TZ_EASTERN:
        dt_et = dt
    else:
        # Already has timezone - convert to ET first
        dt_et = dt.astimezone(TZ_EASTERN)

    return dt_et.astimezone(TZ_UTC)


def utc_to_et(dt: datetime) -> datetime:
    """
    Convert UTC to Eastern Time.

    Args:
        dt: datetime in UTC (can be naive or aware)

    Returns:
        datetime in Eastern Time with tzinfo
    """
    if dt.tzinfo is None:
        # Naive datetime - assume UTC
        dt_utc = TZ_UTC.localize(dt)
    elif dt.tzinfo == TZ_UTC:
        dt_utc = dt
    else:
        # Already has timezone - convert to UTC first
        dt_utc = dt.astimezone(TZ_UTC)

    return dt_utc.astimezone(TZ_EASTERN)


def utc_to_instrument_tz(dt: datetime, symbol: str) -> datetime:
    """
    Convert UTC to instrument's local timezone.

    Args:
        dt: datetime in UTC
        symbol: Trading symbol

    Returns:
        datetime in instrument's local timezone
    """
    if dt.tzinfo is None:
        dt_utc = TZ_UTC.localize(dt)
    else:
        dt_utc = dt.astimezone(TZ_UTC)

    instrument_tz = get_instrument_timezone(symbol)
    return dt_utc.astimezone(instrument_tz)


def parse_forexfactory_datetime(
    date_str: str,
    time_str: str,
    year: Optional[int] = None,
) -> datetime:
    """
    Parse ForexFactory date and time strings to UTC datetime.

    ForexFactory format:
    - Date: "Jan 10" or "01-10-2026" or "Jan 10, 2026"
    - Time: "8:30am" or "8:30pm" or "Tentative" or "All Day"

    All times are in Eastern Time.

    Args:
        date_str: Date string from ForexFactory
        time_str: Time string from ForexFactory
        year: Year to use if not in date_str (default: current year)

    Returns:
        datetime in UTC
    """
    import re
    from datetime import date as dt_date

    if year is None:
        year = datetime.now().year

    # Parse date
    date_str = date_str.strip()

    # Format: "01-10-2026" or "01-10-26"
    if re.match(r"\d{2}-\d{2}-\d{2,4}", date_str):
        parts = date_str.split("-")
        month = int(parts[0])
        day = int(parts[1])
        if len(parts) > 2:
            y = int(parts[2])
            year = y if y > 100 else 2000 + y
    # Format: "Jan 10" or "Jan 10, 2026"
    elif re.match(r"[A-Za-z]{3}\s+\d{1,2}", date_str):
        month_names = {
            "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
            "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
        }
        parts = date_str.replace(",", "").split()
        month = month_names.get(parts[0][:3], 1)
        day = int(parts[1])
        if len(parts) > 2:
            year = int(parts[2])
    else:
        # Fallback
        month = 1
        day = 1

    # Parse time
    time_str = time_str.strip().lower()

    # Handle special cases
    if time_str in ("tentative", "all day", "", "tba"):
        # Default to market open time (9:30 AM ET)
        hour = 9
        minute = 30
    else:
        # Format: "8:30am" or "2:00pm"
        time_match = re.match(r"(\d{1,2}):(\d{2})\s*(am|pm)?", time_str)
        if time_match:
            hour = int(time_match.group(1))
            minute = int(time_match.group(2))
            ampm = time_match.group(3)

            if ampm == "pm" and hour != 12:
                hour += 12
            elif ampm == "am" and hour == 12:
                hour = 0
        else:
            # Fallback
            hour = 12
            minute = 0

    # Create ET datetime and convert to UTC
    try:
        dt_et = datetime(year, month, day, hour, minute)
        return et_to_utc(dt_et)
    except ValueError:
        # Invalid date - return epoch
        return datetime(1970, 1, 1, tzinfo=TZ_UTC)


def is_dst_in_et(dt: datetime) -> bool:
    """
    Check if DST is active in Eastern Time for a given datetime.

    Args:
        dt: datetime (any timezone or naive)

    Returns:
        True if DST is active (EDT), False if not (EST)
    """
    if dt.tzinfo is None:
        dt = TZ_UTC.localize(dt)

    dt_et = dt.astimezone(TZ_EASTERN)
    return bool(dt_et.dst())


def get_utc_offset_et(dt: datetime) -> int:
    """
    Get the UTC offset for Eastern Time at a given datetime.

    Args:
        dt: datetime

    Returns:
        UTC offset in hours (-5 for EST, -4 for EDT)
    """
    if dt.tzinfo is None:
        dt = TZ_UTC.localize(dt)

    dt_et = dt.astimezone(TZ_EASTERN)
    offset = dt_et.utcoffset()
    if offset:
        return int(offset.total_seconds() / 3600)
    return -5  # Default to EST
