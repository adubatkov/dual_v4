"""
News Filter for trade decisions.

Implements 5ers prop firm rules:
- No new orders 2 minutes before until 2 minutes after high-impact news

Usage:
    from src.news_filter import NewsFilter

    filter = NewsFilter(symbol="GER40")
    allowed, event = filter.is_trade_allowed(entry_time_utc)
"""

from datetime import datetime, date, timedelta
from typing import List, Optional, Tuple, Set

import pytz

from .models import NewsEvent
from .storage import NewsStorage
from .timezone_utils import TZ_UTC


# Currency relevance map for trading instruments
# Maps instrument symbols to currencies that affect them
CURRENCY_RELEVANCE_MAP = {
    # European indices
    "GER40": ["EUR", "USD"],  # DAX affected by EUR and USD news
    "UK100": ["GBP", "USD"],
    "FRA40": ["EUR", "USD"],
    "EU50": ["EUR", "USD"],

    # US indices
    "US30": ["USD"],
    "US500": ["USD"],
    "NAS100": ["USD"],

    # Forex pairs
    "EURUSD": ["EUR", "USD"],
    "GBPUSD": ["GBP", "USD"],
    "USDJPY": ["USD", "JPY"],
    "USDCHF": ["USD", "CHF"],
    "AUDUSD": ["AUD", "USD"],
    "NZDUSD": ["NZD", "USD"],
    "USDCAD": ["USD", "CAD"],
    "EURGBP": ["EUR", "GBP"],
    "EURJPY": ["EUR", "JPY"],
    "GBPJPY": ["GBP", "JPY"],

    # Commodities
    "XAUUSD": ["USD"],  # Gold primarily affected by USD
    "XAGUSD": ["USD"],  # Silver
    "USOIL": ["USD"],
    "UKOIL": ["USD"],
}

# Default window for news blocking (in minutes)
DEFAULT_BEFORE_MINUTES = 2
DEFAULT_AFTER_MINUTES = 2

# Category mapping for day-level news skip (NEWS_SKIP_EVENTS param)
# Keys = short category codes used in strategy_logic.py
# Values = substrings to match against ForexFactory event titles
NEWS_EVENT_CATEGORIES = {
    "NFP": ["Non-Farm Employment Change"],
    "CPI": ["CPI m/m", "CPI y/y", "Core CPI", "German Prelim CPI"],
    "FOMC": ["FOMC Statement", "FOMC Meeting Minutes", "FOMC Press Conference",
             "Federal Funds Rate", "FOMC Economic Projections"],
    "ECB": ["ECB Press Conference", "ECB Interest Rate Decision"],
    "GDP": ["GDP"],
    "ISM_PMI": ["ISM Manufacturing PMI", "ISM Services PMI"],
    "RETAIL_SALES": ["Retail Sales"],
}


def categorize_event(title: str) -> str:
    """Map event title to category code. Returns 'OTHER' if no match."""
    for cat, patterns in NEWS_EVENT_CATEGORIES.items():
        if any(p.lower() in title.lower() for p in patterns):
            return cat
    return "OTHER"


def get_relevant_currencies(symbol: str) -> List[str]:
    """
    Get list of currencies relevant to a trading instrument.

    Args:
        symbol: Trading instrument symbol (e.g., "GER40", "EURUSD")

    Returns:
        List of currency codes that affect this instrument
    """
    symbol_upper = symbol.upper()

    # Check direct mapping
    if symbol_upper in CURRENCY_RELEVANCE_MAP:
        return CURRENCY_RELEVANCE_MAP[symbol_upper]

    # For forex pairs, extract currencies from symbol
    if len(symbol_upper) == 6:
        base = symbol_upper[:3]
        quote = symbol_upper[3:]
        return [base, quote]

    # Default: USD affects everything
    return ["USD"]


def is_trade_allowed(
    entry_time_utc: datetime,
    symbol: str,
    news_events: List[NewsEvent],
    before_minutes: int = DEFAULT_BEFORE_MINUTES,
    after_minutes: int = DEFAULT_AFTER_MINUTES,
    high_impact_only: bool = True,
) -> Tuple[bool, Optional[NewsEvent]]:
    """
    Check if a trade is allowed at the given time.

    Implements 5ers rule:
    - No new orders 2 minutes before until 2 minutes after high-impact news

    Args:
        entry_time_utc: Proposed entry time in UTC
        symbol: Trading instrument symbol
        news_events: List of news events to check against
        before_minutes: Minutes before news to block trades
        after_minutes: Minutes after news to block trades
        high_impact_only: Only check high-impact events (default: True)

    Returns:
        Tuple of (is_allowed, blocking_event):
        - (True, None) if trade is allowed
        - (False, NewsEvent) if blocked by a news event
    """
    # Ensure entry_time_utc has timezone info
    if entry_time_utc.tzinfo is None:
        entry_time_utc = pytz.UTC.localize(entry_time_utc)
    else:
        entry_time_utc = entry_time_utc.astimezone(pytz.UTC)

    # Get relevant currencies for this instrument
    relevant_currencies = set(get_relevant_currencies(symbol))

    # Check each news event
    for event in news_events:
        # Skip non-high-impact if filtering
        if high_impact_only and not event.is_high_impact():
            continue

        # Skip if currency not relevant
        if event.country.upper() not in relevant_currencies:
            continue

        # Get blocking window
        window_start, window_end = event.get_blocking_window(
            before_minutes=before_minutes,
            after_minutes=after_minutes,
        )

        # Normalize timezone (window times inherit from event.datetime_utc which may be naive)
        if window_start.tzinfo is None:
            window_start = pytz.UTC.localize(window_start)
        if window_end.tzinfo is None:
            window_end = pytz.UTC.localize(window_end)

        # Check if entry time falls within blocking window
        if window_start <= entry_time_utc <= window_end:
            return (False, event)

    return (True, None)


def get_next_blocking_window(
    from_time_utc: datetime,
    symbol: str,
    news_events: List[NewsEvent],
    before_minutes: int = DEFAULT_BEFORE_MINUTES,
    after_minutes: int = DEFAULT_AFTER_MINUTES,
) -> Optional[Tuple[datetime, datetime, NewsEvent]]:
    """
    Get the next blocking window after a given time.

    Useful for knowing when to wait before entering a trade.

    Args:
        from_time_utc: Time to search from (UTC)
        symbol: Trading instrument symbol
        news_events: List of news events
        before_minutes: Minutes before news to block
        after_minutes: Minutes after news to block

    Returns:
        Tuple of (window_start, window_end, event) or None if no upcoming blocking window
    """
    if from_time_utc.tzinfo is None:
        from_time_utc = pytz.UTC.localize(from_time_utc)

    relevant_currencies = set(get_relevant_currencies(symbol))

    # Filter and sort events
    upcoming_events = []
    for event in news_events:
        if not event.is_high_impact():
            continue
        if event.country.upper() not in relevant_currencies:
            continue

        window_start, window_end = event.get_blocking_window(
            before_minutes=before_minutes,
            after_minutes=after_minutes,
        )

        # Normalize timezone
        if window_start.tzinfo is None:
            window_start = pytz.UTC.localize(window_start)
        if window_end.tzinfo is None:
            window_end = pytz.UTC.localize(window_end)

        # Only consider events where window hasn't ended yet
        if window_end > from_time_utc:
            upcoming_events.append((window_start, window_end, event))

    if not upcoming_events:
        return None

    # Return the nearest upcoming window
    upcoming_events.sort(key=lambda x: x[0])
    return upcoming_events[0]


class NewsFilter:
    """
    News filter for trading decisions.

    Caches news events and provides efficient filtering.

    Usage:
        filter = NewsFilter(symbol="GER40")

        # Check if trade allowed
        allowed, event = filter.is_trade_allowed(entry_time)

        # Get next blocking window
        window = filter.get_next_blocking_window(current_time)
    """

    def __init__(
        self,
        symbol: str,
        storage: Optional[NewsStorage] = None,
        before_minutes: int = DEFAULT_BEFORE_MINUTES,
        after_minutes: int = DEFAULT_AFTER_MINUTES,
        preload_years: Optional[List[int]] = None,
    ):
        """
        Initialize NewsFilter.

        Args:
            symbol: Trading instrument symbol
            storage: NewsStorage instance (created if None)
            before_minutes: Minutes before news to block trades
            after_minutes: Minutes after news to block trades
            preload_years: Years to preload (default: current and previous year)
        """
        self.symbol = symbol.upper()
        self.storage = storage or NewsStorage()
        self.before_minutes = before_minutes
        self.after_minutes = after_minutes

        self._events: List[NewsEvent] = []
        self._events_by_date: dict = {}

        # Preload events
        if preload_years is None:
            current_year = datetime.now().year
            preload_years = [current_year - 1, current_year, current_year + 1]

        self._preload_events(preload_years)

    def _preload_events(self, years: List[int]) -> None:
        """Preload events for given years."""
        relevant_currencies = get_relevant_currencies(self.symbol)

        for year in years:
            year_events = self.storage.load_high_impact(
                start_date=date(year, 1, 1),
                end_date=date(year, 12, 31),
                countries=relevant_currencies,
            )
            self._events.extend(year_events)

        # Sort by datetime
        self._events.sort(key=lambda e: e.datetime_utc)

        # Index by date for faster lookups
        for event in self._events:
            event_date = event.datetime_utc.date()
            if event_date not in self._events_by_date:
                self._events_by_date[event_date] = []
            self._events_by_date[event_date].append(event)

        print(f"[INFO] NewsFilter loaded {len(self._events)} high-impact events for {self.symbol}")

    def is_trade_allowed(
        self,
        entry_time_utc: datetime,
    ) -> Tuple[bool, Optional[NewsEvent]]:
        """
        Check if a trade is allowed at the given time.

        Args:
            entry_time_utc: Proposed entry time in UTC

        Returns:
            Tuple of (is_allowed, blocking_event)
        """
        # Get events for the relevant date range (±1 day for safety)
        if entry_time_utc.tzinfo is None:
            entry_time_utc = pytz.UTC.localize(entry_time_utc)

        entry_date = entry_time_utc.date()
        nearby_events: List[NewsEvent] = []

        for delta in [-1, 0, 1]:
            check_date = entry_date + timedelta(days=delta)
            if check_date in self._events_by_date:
                nearby_events.extend(self._events_by_date[check_date])

        return is_trade_allowed(
            entry_time_utc=entry_time_utc,
            symbol=self.symbol,
            news_events=nearby_events,
            before_minutes=self.before_minutes,
            after_minutes=self.after_minutes,
            high_impact_only=True,  # Already filtered
        )

    def get_next_blocking_window(
        self,
        from_time_utc: datetime,
    ) -> Optional[Tuple[datetime, datetime, NewsEvent]]:
        """
        Get the next blocking window after a given time.

        Args:
            from_time_utc: Time to search from (UTC)

        Returns:
            Tuple of (window_start, window_end, event) or None
        """
        return get_next_blocking_window(
            from_time_utc=from_time_utc,
            symbol=self.symbol,
            news_events=self._events,
            before_minutes=self.before_minutes,
            after_minutes=self.after_minutes,
        )

    def should_skip_day(
        self,
        trade_date: date,
        skip_categories: List[str],
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if trading day should be skipped due to scheduled news.

        Args:
            trade_date: Date to check
            skip_categories: List of category codes (e.g. ["NFP", "CPI", "ECB"])

        Returns:
            Tuple of (should_skip, reason_string)
        """
        if not skip_categories:
            return (False, None)
        events = self._events_by_date.get(trade_date, [])
        for event in events:
            cat = categorize_event(event.title)
            if cat in skip_categories:
                return (True, f"{cat}: {event.title} ({event.country})")
        return (False, None)

    def get_blocking_events_for_day(
        self,
        target_date: date,
    ) -> List[NewsEvent]:
        """
        Get all blocking events for a specific day.

        Args:
            target_date: Date to check

        Returns:
            List of NewsEvent objects for that day
        """
        return self._events_by_date.get(target_date, [])

    def refresh_current_week(self) -> int:
        """
        Refresh events for current week from ForexFactory API.

        Returns:
            Number of new events added
        """
        from .forexfactory_client import ForexFactoryClient

        client = ForexFactoryClient()
        new_events = client.fetch_current_week()

        if not new_events:
            return 0

        # Save to storage
        self.storage.add_events(new_events)

        # Add high-impact events to local cache
        relevant_currencies = set(get_relevant_currencies(self.symbol))
        added = 0

        for event in new_events:
            if not event.is_high_impact():
                continue
            if event.country.upper() not in relevant_currencies:
                continue

            # Check if already in cache
            event_date = event.datetime_utc.date()
            existing_keys = {
                (e.title, e.datetime_utc.isoformat())
                for e in self._events_by_date.get(event_date, [])
            }

            if (event.title, event.datetime_utc.isoformat()) not in existing_keys:
                self._events.append(event)
                if event_date not in self._events_by_date:
                    self._events_by_date[event_date] = []
                self._events_by_date[event_date].append(event)
                added += 1

        # Re-sort
        self._events.sort(key=lambda e: e.datetime_utc)

        return added

    @property
    def event_count(self) -> int:
        """Get total number of cached events."""
        return len(self._events)

    def get_stats(self) -> dict:
        """Get filter statistics."""
        return {
            "symbol": self.symbol,
            "event_count": len(self._events),
            "date_range": (
                self._events[0].datetime_utc.date() if self._events else None,
                self._events[-1].datetime_utc.date() if self._events else None,
            ),
            "blocking_window_minutes": (self.before_minutes, self.after_minutes),
            "relevant_currencies": get_relevant_currencies(self.symbol),
        }
