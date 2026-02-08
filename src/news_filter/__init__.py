"""
News Filter Module for ForexFactory Economic Calendar.

Provides filtering of trades based on high-impact economic news events
according to 5ers prop firm rules:
- No new orders 2 minutes before until 2 minutes after high-impact news

Usage:
    from src.news_filter import NewsFilter, NewsEvent

    filter = NewsFilter(symbol="GER40")
    allowed, event = filter.is_trade_allowed(entry_time_utc)
"""

from .models import NewsEvent
from .filter import NewsFilter, is_trade_allowed, get_relevant_currencies
from .storage import NewsStorage
from .forexfactory_client import ForexFactoryClient
from .timezone_utils import (
    et_to_utc,
    utc_to_et,
    utc_to_instrument_tz,
    get_instrument_timezone,
)

__all__ = [
    "NewsEvent",
    "NewsFilter",
    "NewsStorage",
    "ForexFactoryClient",
    "is_trade_allowed",
    "get_relevant_currencies",
    "et_to_utc",
    "utc_to_et",
    "utc_to_instrument_tz",
    "get_instrument_timezone",
]
