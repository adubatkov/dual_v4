#!/usr/bin/env python3
"""
Scraper for ForexFactory Historical Calendar Data.

Downloads high-impact news events from ForexFactory website.
The XML API only provides current week, so we need to scrape for historical data.

Usage:
    python scripts/scrape_forexfactory.py --start-year 2023 --end-year 2026
    python scripts/scrape_forexfactory.py --year 2024
"""

import sys
import time
import json
import re
from pathlib import Path
from datetime import datetime, date, timedelta
from typing import List, Optional, Dict, Any
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.news_filter import NewsEvent, NewsStorage
from src.news_filter.timezone_utils import parse_forexfactory_datetime


# ForexFactory calendar URL pattern
FF_CALENDAR_URL = "https://www.forexfactory.com/calendar"

# Headers to mimic browser request
HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Connection": "keep-alive",
}


def get_week_mondays(year: int) -> List[date]:
    """Get all Monday dates for a given year."""
    mondays = []
    d = date(year, 1, 1)

    # Find first Monday
    while d.weekday() != 0:  # 0 = Monday
        d += timedelta(days=1)

    # Collect all Mondays of the year
    while d.year == year:
        mondays.append(d)
        d += timedelta(weeks=1)

    return mondays


def format_ff_week_param(monday: date) -> str:
    """
    Format date for ForexFactory URL parameter.

    Example: jan1.2023, feb5.2024
    """
    month_abbr = monday.strftime("%b").lower()
    return f"{month_abbr}{monday.day}.{monday.year}"


def fetch_page(url: str, timeout: int = 30) -> Optional[str]:
    """Fetch page content with retry logic."""
    max_retries = 3

    for attempt in range(max_retries):
        try:
            request = Request(url, headers=HTTP_HEADERS)
            with urlopen(request, timeout=timeout) as response:
                return response.read().decode("utf-8")
        except (HTTPError, URLError) as e:
            print(f"  [WARNING] Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                return None
        except Exception as e:
            print(f"  [ERROR] Unexpected error: {e}")
            return None

    return None


def parse_calendar_html(html: str, week_start: date) -> List[NewsEvent]:
    """
    Parse ForexFactory calendar HTML to extract events.

    ForexFactory embeds JSON data in the HTML for their React components.
    We extract this JSON data using the "dateline" Unix timestamp field.

    JSON structure found in HTML:
    "name":"CPI m/m","currency":"AUD","dateline":1767745800,
    "impactClass":"icon--ff-impact-red","timeLabel":"5:30am"
    """
    events = []

    # Pattern to find high-impact event blocks with dateline (Unix timestamp)
    # Look for: "name":"...", "currency":"...", "dateline":..., "impactClass":"icon--ff-impact-red"
    # The order may vary, so we find context around each high-impact marker

    # Find all contexts containing high-impact events
    # Each event has: name, currency, dateline (Unix timestamp), impactClass
    high_impact_pattern = re.compile(
        r'.{0,1000}"impactClass"\s*:\s*"icon--ff-impact-red".{0,200}',
        re.DOTALL
    )

    for match in high_impact_pattern.finditer(html):
        context = match.group(0)

        # Extract name (event title)
        name_match = re.search(r'"name"\s*:\s*"([^"]+)"', context)
        if not name_match:
            continue
        name = name_match.group(1).replace(r'\/', '/')  # Unescape

        # Extract currency
        currency_match = re.search(r'"currency"\s*:\s*"([^"]+)"', context)
        if not currency_match:
            continue
        currency = currency_match.group(1)

        # Extract dateline (Unix timestamp) - this is key!
        dateline_match = re.search(r'"dateline"\s*:\s*(\d+)', context)
        if not dateline_match:
            continue

        try:
            # Convert Unix timestamp to UTC datetime
            dateline = int(dateline_match.group(1))
            datetime_utc = datetime.utcfromtimestamp(dateline)
            datetime_utc = datetime_utc.replace(tzinfo=None)

            # Extract actual/forecast/previous if available
            actual_match = re.search(r'"actual"\s*:\s*"([^"]*)"', context)
            forecast_match = re.search(r'"forecast"\s*:\s*"([^"]*)"', context)
            previous_match = re.search(r'"previous"\s*:\s*"([^"]*)"', context)

            actual = actual_match.group(1) if actual_match else None
            forecast = forecast_match.group(1) if forecast_match else None
            previous = previous_match.group(1) if previous_match else None

            # Clean up empty strings
            actual = actual if actual else None
            forecast = forecast if forecast else None
            previous = previous if previous else None

            event = NewsEvent(
                title=name,
                country=currency,
                datetime_utc=datetime_utc,
                impact="High",
                forecast=forecast,
                previous=previous,
                actual=actual,
            )
            events.append(event)

        except Exception as e:
            print(f"  [WARNING] Failed to parse event {name}: {e}")

    return events


def scrape_year(year: int, storage: NewsStorage, delay: float = 1.0) -> int:
    """
    Scrape all high-impact events for a given year.

    Args:
        year: Year to scrape
        storage: NewsStorage instance to save events
        delay: Delay between requests in seconds

    Returns:
        Number of events scraped
    """
    print(f"\n[INFO] Scraping ForexFactory calendar for {year}...")

    mondays = get_week_mondays(year)
    all_events = []

    for i, monday in enumerate(mondays):
        week_param = format_ff_week_param(monday)
        url = f"{FF_CALENDAR_URL}?week={week_param}"

        print(f"  [{i+1}/{len(mondays)}] Fetching week of {monday}...", end="", flush=True)

        html = fetch_page(url)

        if html:
            events = parse_calendar_html(html, monday)
            high_impact = [e for e in events if e.impact == "High"]
            all_events.extend(high_impact)
            print(f" {len(high_impact)} high-impact events")
        else:
            print(" FAILED")

        # Rate limiting
        if i < len(mondays) - 1:
            time.sleep(delay)

    # Save to storage
    if all_events:
        # Deduplicate by (title, country, datetime)
        unique_events = {}
        for e in all_events:
            key = (e.title, e.country, e.datetime_utc.isoformat())
            unique_events[key] = e

        final_events = sorted(unique_events.values(), key=lambda e: e.datetime_utc)
        storage.save_year(year, final_events)
        print(f"[OK] Saved {len(final_events)} unique high-impact events for {year}")
        return len(final_events)
    else:
        print(f"[WARNING] No events found for {year}")
        return 0


def try_xml_api_first(storage: NewsStorage) -> None:
    """Try to get current week data from XML API first."""
    from src.news_filter import ForexFactoryClient

    print("[INFO] Fetching current week from ForexFactory XML API...")
    client = ForexFactoryClient()
    events = client.fetch_current_week()

    if events:
        high_impact = [e for e in events if e.impact == "High"]
        print(f"[OK] Got {len(high_impact)} high-impact events from XML API")

        # Add to storage
        storage.add_events(high_impact)
    else:
        print("[WARNING] XML API returned no events")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Scrape ForexFactory historical calendar")
    parser.add_argument("--year", type=int, help="Single year to scrape")
    parser.add_argument("--start-year", type=int, default=2023, help="Start year (default: 2023)")
    parser.add_argument("--end-year", type=int, default=2026, help="End year (default: 2026)")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between requests in seconds")
    parser.add_argument("--xml-only", action="store_true", help="Only fetch from XML API (current week)")

    args = parser.parse_args()

    storage = NewsStorage()

    print("=" * 60)
    print("ForexFactory Calendar Scraper")
    print("=" * 60)

    # Always try XML API first for current data
    try_xml_api_first(storage)

    if args.xml_only:
        print("\n[INFO] XML-only mode, skipping historical scrape")
        return

    # Determine years to scrape
    if args.year:
        years = [args.year]
    else:
        years = list(range(args.start_year, args.end_year + 1))

    total_events = 0

    for year in years:
        count = scrape_year(year, storage, delay=args.delay)
        total_events += count

    # Final stats
    print("\n" + "=" * 60)
    print("SCRAPING COMPLETE")
    print("=" * 60)

    stats = storage.get_stats()
    print(f"Years: {stats['years']}")
    print(f"Total events: {stats['total_events']}")
    print(f"High-impact: {stats['high_impact_events']}")
    print(f"Countries: {stats['countries']}")


if __name__ == "__main__":
    main()
