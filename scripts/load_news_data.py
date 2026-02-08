#!/usr/bin/env python3
"""
Script to load and populate news data from ForexFactory.

This script:
1. Fetches current week from ForexFactory XML API
2. Generates comprehensive historical high-impact events (2023-2026)
3. Saves all data to JSON files

Usage:
    python scripts/load_news_data.py
    python scripts/load_news_data.py --fetch-only  # Only fetch current week
    python scripts/load_news_data.py --generate-only  # Only generate historical
"""

import sys
from pathlib import Path
from datetime import date, timedelta
from typing import List

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.news_filter import NewsEvent, NewsStorage, ForexFactoryClient
from src.news_filter.timezone_utils import parse_forexfactory_datetime


def get_nfp_dates(year: int) -> List[date]:
    """
    Get Non-Farm Payrolls release dates for a year.

    NFP is released on the first Friday of each month at 8:30 ET.
    """
    dates = []
    for month in range(1, 13):
        # Find first Friday of month
        first_day = date(year, month, 1)
        # Find days until Friday (4 = Friday in weekday())
        days_until_friday = (4 - first_day.weekday()) % 7
        first_friday = first_day + timedelta(days=days_until_friday)
        dates.append(first_friday)
    return dates


def get_fomc_dates(year: int) -> List[date]:
    """
    Get FOMC meeting statement dates for a year.

    FOMC meets 8 times per year, announcements at 14:00 ET.
    """
    # Approximate FOMC meeting dates (based on Fed schedule)
    fomc_months = [
        (1, 3),   # Late January
        (3, 2),   # Mid March
        (5, 1),   # Early May
        (6, 2),   # Mid June
        (7, 4),   # Late July
        (9, 2),   # Mid September
        (11, 1),  # Early November
        (12, 2),  # Mid December
    ]

    dates = []
    for month, week in fomc_months:
        # Find Wednesday of specified week
        first_day = date(year, month, 1)
        days_until_wednesday = (2 - first_day.weekday()) % 7
        first_wednesday = first_day + timedelta(days=days_until_wednesday)
        target_wednesday = first_wednesday + timedelta(weeks=week-1)

        # Ensure date is valid
        try:
            if target_wednesday.month == month:
                dates.append(target_wednesday)
        except:
            pass

    return dates


def get_ecb_dates(year: int) -> List[date]:
    """
    Get ECB Interest Rate Decision dates for a year.

    ECB meets ~6-8 times per year, announcements at 12:15 ET.
    """
    # ECB typically meets on Thursdays
    ecb_months = [1, 3, 4, 6, 7, 9, 10, 12]

    dates = []
    for month in ecb_months:
        # Find third Thursday of month (typical ECB meeting)
        first_day = date(year, month, 1)
        days_until_thursday = (3 - first_day.weekday()) % 7
        first_thursday = first_day + timedelta(days=days_until_thursday)
        third_thursday = first_thursday + timedelta(weeks=2)

        if third_thursday.month == month:
            dates.append(third_thursday)

    return dates


def get_cpi_dates(year: int) -> List[date]:
    """
    Get US CPI release dates for a year.

    CPI is released around the 10th-15th of each month at 8:30 ET.
    """
    dates = []
    for month in range(1, 13):
        # CPI typically released around the 12th, on a Tuesday or Wednesday
        target_day = 12
        try:
            cpi_date = date(year, month, target_day)
            # Adjust to nearest business day
            while cpi_date.weekday() > 4:  # Skip weekend
                cpi_date += timedelta(days=1)
            dates.append(cpi_date)
        except:
            pass

    return dates


def generate_high_impact_events(year: int) -> List[NewsEvent]:
    """
    Generate high-impact news events for a given year.

    Includes:
    - NFP (Non-Farm Payrolls) - USD - monthly
    - FOMC Statement - USD - 8x/year
    - ECB Interest Rate Decision - EUR - 6-8x/year
    - CPI (Consumer Price Index) - USD - monthly
    """
    events = []

    # Non-Farm Payrolls (first Friday each month, 8:30 ET)
    for nfp_date in get_nfp_dates(year):
        date_str = nfp_date.strftime("%m-%d-%Y")
        datetime_utc = parse_forexfactory_datetime(date_str, "8:30am")
        events.append(NewsEvent(
            title="Non-Farm Payrolls",
            country="USD",
            datetime_utc=datetime_utc,
            impact="High",
        ))

    # FOMC Statement (8x/year, 14:00 ET)
    for fomc_date in get_fomc_dates(year):
        date_str = fomc_date.strftime("%m-%d-%Y")
        datetime_utc = parse_forexfactory_datetime(date_str, "2:00pm")
        events.append(NewsEvent(
            title="FOMC Statement",
            country="USD",
            datetime_utc=datetime_utc,
            impact="High",
        ))

    # ECB Interest Rate Decision (6-8x/year, 12:15 ET / 8:15 ET summer)
    for ecb_date in get_ecb_dates(year):
        date_str = ecb_date.strftime("%m-%d-%Y")
        # ECB announces at 13:45 CET = 7:45 ET winter / 8:15 ET summer
        datetime_utc = parse_forexfactory_datetime(date_str, "8:15am")
        events.append(NewsEvent(
            title="ECB Interest Rate Decision",
            country="EUR",
            datetime_utc=datetime_utc,
            impact="High",
        ))

    # US CPI (monthly, 8:30 ET)
    for cpi_date in get_cpi_dates(year):
        date_str = cpi_date.strftime("%m-%d-%Y")
        datetime_utc = parse_forexfactory_datetime(date_str, "8:30am")
        events.append(NewsEvent(
            title="CPI m/m",
            country="USD",
            datetime_utc=datetime_utc,
            impact="High",
        ))

    # Sort by datetime
    events.sort(key=lambda e: e.datetime_utc)

    return events


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Load news data from ForexFactory")
    parser.add_argument("--fetch-only", action="store_true",
                       help="Only fetch current week from API")
    parser.add_argument("--generate-only", action="store_true",
                       help="Only generate historical data")
    parser.add_argument("--years", type=str, default="2023,2024,2025,2026",
                       help="Years to generate (comma-separated)")

    args = parser.parse_args()

    storage = NewsStorage()
    client = ForexFactoryClient()

    print("=" * 60)
    print("News Data Loader")
    print("=" * 60)

    # Generate historical data
    if not args.fetch_only:
        years = [int(y.strip()) for y in args.years.split(",")]

        for year in years:
            print(f"\n[INFO] Generating high-impact events for {year}...")
            events = generate_high_impact_events(year)

            # Save to storage
            storage.save_year(year, events)
            print(f"[OK] Saved {len(events)} events for {year}")

            # Show summary
            titles = {}
            for e in events:
                titles[e.title] = titles.get(e.title, 0) + 1

            for title, count in sorted(titles.items()):
                print(f"    - {title}: {count} events")

    # Fetch current week from API
    if not args.generate_only:
        print("\n[INFO] Fetching current week from ForexFactory API...")
        try:
            current_events = client.fetch_current_week()

            if current_events:
                print(f"[OK] Fetched {len(current_events)} events from API")

                # Filter to high-impact only
                high_impact = [e for e in current_events if e.impact == "High"]
                print(f"[INFO] High-impact events this week: {len(high_impact)}")

                for event in high_impact:
                    print(f"    - {event.datetime_utc.strftime('%Y-%m-%d %H:%M')} UTC: "
                          f"{event.title} ({event.country})")

                # Merge with existing year data
                if high_impact:
                    year = high_impact[0].datetime_utc.year
                    existing = storage.load_year(year)

                    # Add only new events (by datetime+title)
                    existing_keys = {(e.datetime_utc, e.title) for e in existing}
                    new_events = [e for e in high_impact
                                  if (e.datetime_utc, e.title) not in existing_keys]

                    if new_events:
                        all_events = existing + new_events
                        all_events.sort(key=lambda e: e.datetime_utc)
                        storage.save_year(all_events, year)
                        print(f"[OK] Added {len(new_events)} new events to {year} data")
            else:
                print("[WARNING] No events fetched from API")

        except Exception as e:
            print(f"[ERROR] Failed to fetch from API: {e}")

    # Show final stats
    print("\n" + "=" * 60)
    print("Final Statistics")
    print("=" * 60)

    stats = storage.get_stats()
    print(f"Years available: {stats['years']}")
    print(f"Total events: {stats['total_events']}")
    print(f"High-impact events: {stats['high_impact_events']}")

    print("\n[OK] News data loading complete!")


if __name__ == "__main__":
    main()
