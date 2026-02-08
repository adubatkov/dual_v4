# News Filter Module

Last updated: 2026-02-08

## Table of Contents

- [Purpose](#purpose)
- [Architecture](#architecture)
- [How the Filter Works](#how-the-filter-works)
- [Currency-to-Instrument Mapping](#currency-to-instrument-mapping)
- [ForexFactory Integration](#forexfactory-integration)
- [Data Storage](#data-storage)
- [Configuration Parameters](#configuration-parameters)
- [Scripts](#scripts)
- [Integration with Live Bot](#integration-with-live-bot)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)

---

## Purpose

The News Filter module enforces The5ers prop firm compliance rule that prohibits
opening new trades in the vicinity of high-impact economic news events.

**The5ers Rule**: No new orders may be placed 2 minutes before or 2 minutes after
a high-impact news release. Violation of this rule can result in account
disqualification.

The module covers both instruments traded by the bot:

| Instrument | Relevant Currencies |
|------------|-------------------|
| GER40      | EUR, USD          |
| XAUUSD     | USD               |

High-impact events tracked include Non-Farm Payrolls (NFP), FOMC Statements,
ECB Interest Rate Decisions, CPI releases, and other major economic
announcements as classified by ForexFactory.

---

## Architecture

The module is located at `dual_v3/src/news_filter/` and consists of six files:

```
src/news_filter/
    __init__.py              # Public API exports
    models.py                # NewsEvent dataclass
    filter.py                # Core filtering logic + NewsFilter class
    forexfactory_client.py   # ForexFactory XML API + CSV parser
    storage.py               # JSON file persistence (by year)
    timezone_utils.py        # ET/UTC/instrument timezone conversions
```

### Module Dependency Graph

```
__init__.py
    |-- models.py            (NewsEvent)
    |-- filter.py            (NewsFilter, is_trade_allowed, get_relevant_currencies)
    |       |-- models.py
    |       |-- storage.py
    |       |-- timezone_utils.py
    |-- storage.py           (NewsStorage)
    |       |-- models.py
    |-- forexfactory_client.py (ForexFactoryClient)
    |       |-- models.py
    |       |-- timezone_utils.py
    |-- timezone_utils.py    (et_to_utc, utc_to_et, utc_to_instrument_tz, ...)
```

---

## How the Filter Works

### Core Algorithm

1. When a trade signal is generated, `is_trade_allowed(entry_time_utc, symbol, news_events)` is called.
2. The function identifies which currencies are relevant to the given instrument using `CURRENCY_RELEVANCE_MAP`.
3. For each high-impact news event in the provided list:
   - Skip if `high_impact_only=True` and the event is not high-impact.
   - Skip if the event's currency is not relevant to the instrument.
   - Compute the blocking window: `[event_time - before_minutes, event_time + after_minutes]`.
   - If the proposed entry time falls within the blocking window, return `(False, blocking_event)`.
4. If no blocking event is found, return `(True, None)`.

### NewsFilter Class

The `NewsFilter` class wraps the stateless `is_trade_allowed` function with:

- **Preloading**: On initialization, loads high-impact events for the previous year, current year, and next year from JSON storage.
- **Date indexing**: Events are indexed by date (`_events_by_date` dict) for O(1) lookup of nearby events.
- **Efficient lookups**: When checking a trade, only events within +/-1 day of the entry time are considered.
- **Live refresh**: `refresh_current_week()` fetches the latest week from the ForexFactory XML API and merges new events into the cache.

```python
from src.news_filter import NewsFilter

# Initialize for GER40 (preloads EUR + USD high-impact events)
nf = NewsFilter(symbol="GER40", before_minutes=2, after_minutes=2)

# Check if trade is allowed
allowed, blocking_event = nf.is_trade_allowed(entry_time_utc)

if not allowed:
    print(f"Trade blocked: {blocking_event.title} ({blocking_event.country})")
```

---

## Currency-to-Instrument Mapping

Defined in `filter.py` as `CURRENCY_RELEVANCE_MAP`:

| Instrument | Relevant Currencies | Rationale |
|------------|-------------------|-----------|
| GER40      | EUR, USD          | DAX is an EUR-denominated index; USD news (NFP, FOMC) causes cross-market volatility |
| XAUUSD     | USD               | Gold is priced in USD; only USD news is relevant |
| UK100      | GBP, USD          | FTSE 100 |
| US30       | USD               | Dow Jones |
| US500      | USD               | S&P 500 |
| NAS100     | USD               | NASDAQ 100 |
| EURUSD     | EUR, USD          | Direct currency pair |
| GBPUSD     | GBP, USD          | Direct currency pair |
| XAGUSD     | USD               | Silver |

For forex pairs not explicitly listed, the module extracts currencies from the
6-character symbol (e.g., `AUDCAD` -> `["AUD", "CAD"]`). For unknown instruments,
the fallback is `["USD"]`.

---

## ForexFactory Integration

### Data Sources

The `ForexFactoryClient` class supports three data ingestion methods:

#### 1. XML API (Current Week)

- **Endpoint**: `https://nfs.faireconomy.media/ff_calendar_thisweek.xml`
- **Scope**: Current week only
- **Format**: XML with `<event>` elements containing `<title>`, `<country>`, `<date>`, `<time>`, `<impact>`, `<forecast>`, `<previous>`
- **Timezone**: All times are in Eastern Time (ET), converted to UTC during parsing

```xml
<weeklyevents>
    <event>
        <title>Non-Farm Payrolls</title>
        <country>USD</country>
        <date>01-10-2026</date>
        <time>8:30am</time>
        <impact>High</impact>
        <forecast>180K</forecast>
        <previous>256K</previous>
    </event>
</weeklyevents>
```

#### 2. HTML Scraping (Historical)

The `scrape_forexfactory.py` script scrapes the ForexFactory calendar website
week by week. It parses embedded JSON data from the HTML, extracting events
using the `dateline` Unix timestamp field. This approach captures events for
any historical period.

- **URL pattern**: `https://www.forexfactory.com/calendar?week=jan1.2023`
- **Rate limiting**: 1-second delay between requests (configurable)
- **Retry logic**: 3 attempts with exponential backoff

#### 3. Kaggle CSV (Bulk Historical)

The `parse_kaggle_csv()` method can ingest the Kaggle ForexFactory historical
dataset. It handles multiple CSV column naming conventions (e.g., `event`/`title`/`name`,
`currency`/`country`/`ccy`).

### Impact Level Normalization

Raw impact levels from various sources are normalized:

| Raw Value                           | Normalized |
|-------------------------------------|-----------|
| `high`, `red`, `3`, `important`, `major` | `High`    |
| `medium`, `orange`, `2`, `moderate`      | `Medium`  |
| `low`, `yellow`, `1`, `minor`            | `Low`     |

Only `High` impact events trigger trade blocking by default.

---

## Data Storage

### File Location

```
dual_v3/data/news/
    forex_factory_2023.json
    forex_factory_2024.json
    forex_factory_2025.json
    forex_factory_2026.json
```

### JSON Format

Each file stores events for a single year:

```json
{
    "version": "1.0",
    "source": "forexfactory",
    "last_updated": "2026-01-12T10:00:00Z",
    "year": 2026,
    "event_count": 42,
    "events": [
        {
            "title": "Non-Farm Payrolls",
            "country": "USD",
            "datetime_utc": "2026-01-10T13:30:00+00:00",
            "impact": "High",
            "forecast": "180K",
            "previous": "256K",
            "actual": null
        }
    ]
}
```

### Storage Class (NewsStorage)

Key operations:

| Method                   | Description |
|--------------------------|------------|
| `load_year(year)`        | Load all events for a year (with in-memory cache) |
| `save_year(year, events)`| Save events for a year to JSON |
| `load_range(start, end)` | Load events across a date range with optional impact/country filters |
| `load_high_impact(start, end, countries)` | Convenience method for high-impact events only |
| `add_events(events)`     | Add events with automatic deduplication (by title + country + datetime) |
| `get_stats()`            | Return counts by year, impact level, and country |

The storage layer uses in-memory caching (`_cache` dict keyed by year) to avoid
repeated disk reads during a single session.

---

## Configuration Parameters

### Blocking Window

| Parameter        | Default | Description |
|-----------------|---------|-------------|
| `before_minutes` | 2       | Minutes before the news event to start blocking trades |
| `after_minutes`  | 2       | Minutes after the news event to stop blocking trades |

These defaults enforce the 5ers rule of 2 minutes before and 2 minutes after.
The values are configurable per-instance:

```python
# More conservative: 5 min before, 10 min after
nf = NewsFilter(symbol="GER40", before_minutes=5, after_minutes=10)
```

### Optimizer Configuration

In `params_optimizer/config.py`, the news filter is configured via:

```python
# News filter - 5ers compliance
news_filter_enabled: bool = True    # from news_filter.enabled in optimizer_config.json
news_before_minutes: int = 2        # from news_filter.before_minutes
news_after_minutes: int = 2         # from news_filter.after_minutes
```

### Preloaded Years

By default, `NewsFilter.__init__` preloads events for `[current_year - 1, current_year, current_year + 1]`.
Override with the `preload_years` parameter:

```python
nf = NewsFilter(symbol="GER40", preload_years=[2023, 2024, 2025, 2026])
```

---

## Scripts

### 1. load_news_data.py

**Location**: `dual_v3/scripts/load_news_data.py`

Generates high-impact events programmatically and optionally fetches the current
week from the ForexFactory XML API.

```bash
# Generate historical + fetch current week
python scripts/load_news_data.py

# Only fetch current week from API
python scripts/load_news_data.py --fetch-only

# Only generate historical data (no network)
python scripts/load_news_data.py --generate-only

# Specify years
python scripts/load_news_data.py --years 2023,2024,2025,2026
```

**Generated Events** (per year):
- Non-Farm Payrolls (NFP): 12 events (first Friday of each month, 8:30 AM ET)
- FOMC Statement: ~8 events (scheduled Fed meetings, 2:00 PM ET)
- ECB Interest Rate Decision: ~8 events (third Thursday of select months, 8:15 AM ET)
- CPI m/m: 12 events (around the 12th of each month, 8:30 AM ET)

### 2. scrape_forexfactory.py

**Location**: `dual_v3/scripts/scrape_forexfactory.py`

Scrapes ForexFactory website for real historical data (not generated approximations).
Iterates through every week of the specified year(s), parsing embedded JSON from
the HTML.

```bash
# Scrape a single year
python scripts/scrape_forexfactory.py --year 2024

# Scrape a range
python scripts/scrape_forexfactory.py --start-year 2023 --end-year 2026

# Only use XML API (current week)
python scripts/scrape_forexfactory.py --xml-only

# Custom rate limiting (2 seconds between requests)
python scripts/scrape_forexfactory.py --delay 2.0
```

### 3. verify_news_filter.py

**Location**: `dual_v3/scripts/verify_news_filter.py`

Runs a comparative backtest with and without the news filter to validate that it
correctly blocks trades during news events and measure its impact on performance.

```bash
# Verify for GER40 (last 30 days)
python scripts/verify_news_filter.py

# Verify for XAUUSD (last 60 days)
python scripts/verify_news_filter.py --symbol XAUUSD --days 60
```

Output includes:
- Number of trades blocked by the news filter
- Total R difference (with vs. without filter)
- List of high-impact events in the test period

---

## Integration with Live Bot

The news filter is integrated into the IB Strategy at
`dual_v3/src/strategies/ib_strategy.py`.

### Initialization

```python
class IBStrategy(BaseStrategy):
    def __init__(self, symbol, params, executor, magic_number,
                 strategy_label="", news_filter_enabled=False):
        # ...
        self.news_filter: Optional[NewsFilter] = None
        self._news_filtered_count: int = 0
        if news_filter_enabled:
            try:
                self.news_filter = NewsFilter(symbol=symbol)
                logger.info(f"[{symbol}] News filter enabled with "
                           f"{self.news_filter.event_count} events")
            except Exception as e:
                logger.warning(f"[{symbol}] Failed to initialize news filter: {e}")
```

### Signal Check

Inside `check_signal()`, after a breakout is detected but before the trade is
submitted:

```python
# NEWS FILTER CHECK (5ers compliance)
# Block trades 2 minutes before/after high-impact news
if self.news_filter is not None:
    allowed, blocking_event = self.news_filter.is_trade_allowed(current_time_utc)
    if not allowed:
        self._news_filtered_count += 1
        logger.info(f"{self.log_prefix} Trade blocked by news filter: "
                    f"{blocking_event.title} ({blocking_event.country})")
        return None
```

### Monitoring

The `get_news_filter_stats()` method returns runtime statistics:

```python
{
    "news_filter_enabled": True,
    "trades_blocked_by_news": 3,
    "events_loaded": 156,
}
```

The `_news_filtered_count` counter is cumulative and is not reset on daily state
reset.

---

## Testing

### Test Suite

**Location**: `dual_v3/tests/test_news_filter.py`

Run with:

```bash
cd dual_v3
pytest tests/test_news_filter.py -v
```

### Test Classes

| Class              | Tests | Coverage |
|--------------------|-------|----------|
| `TestNewsEvent`    | 4     | NewsEvent creation, blocking window calculation, custom window, serialization round-trip |
| `TestTimezoneUtils`| 7     | ET-to-UTC (EST/EDT), UTC-to-ET, ForexFactory datetime parsing (full date, PM times, text month), DST detection, instrument timezone mapping |
| `TestFilter`       | 8     | Trade allowed outside window, blocked during window, blocked before/after, allowed after window ends, irrelevant currency not blocked, low-impact ignored, next blocking window |
| `TestStorage`      | 3     | Save/load round-trip, high-impact-only filtering, deduplication on add |
| `TestIntegration`  | 1     | Full NewsFilter with storage, preloading, and filtering |

### Key Test Scenarios

**Blocking window boundary tests** (using NFP at 2026-01-10 13:30 UTC):

| Entry Time (UTC) | Expected | Reason |
|-------------------|----------|--------|
| 12:30             | Allowed  | 60 minutes before event |
| 13:28             | Blocked  | Exactly at window start (2 min before) |
| 13:29             | Blocked  | 1 minute before event |
| 13:30             | Blocked  | Exactly at event time |
| 13:31             | Blocked  | 1 minute after event |
| 13:32             | Blocked  | Exactly at window end (2 min after) |
| 13:33             | Allowed  | 1 second after window end |

**Currency relevance tests**:
- ECB Rate Decision (EUR) blocks GER40 trades (GER40 is affected by EUR).
- ECB Rate Decision (EUR) does NOT block XAUUSD trades (XAUUSD only cares about USD).

---

## Troubleshooting

### No events loaded

If `NewsFilter` reports 0 events, verify:

1. JSON files exist in `dual_v3/data/news/`.
2. Run `python scripts/load_news_data.py` to generate/fetch data.
3. Check that the preloaded years overlap with your backtest or trading date range.

### Timezone mismatches

ForexFactory publishes times in Eastern Time (ET). The module handles
EST (UTC-5) and EDT (UTC-4) automatically via `pytz`. If you observe
off-by-one-hour errors near DST transitions (March/November), verify
that `pytz` is up to date.

### API fetch failures

The XML API at `nfs.faireconomy.media` may return HTTP errors or timeout.
The client retries 3 times with exponential backoff. If scraping fails
consistently, the delay between requests may need to be increased
(`--delay` flag in `scrape_forexfactory.py`).
