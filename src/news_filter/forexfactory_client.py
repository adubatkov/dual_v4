"""
ForexFactory Client for fetching economic calendar data.

Supports:
1. XML API for current week: https://nfs.faireconomy.media/ff_calendar_thisweek.xml
2. CSV parsing for Kaggle historical dataset
3. Manual data entry for gaps
"""

import csv
import re
import xml.etree.ElementTree as ET
from datetime import datetime, date
from pathlib import Path
from typing import List, Optional, Dict, Any
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

from .models import NewsEvent
from .timezone_utils import parse_forexfactory_datetime, et_to_utc


# ForexFactory XML API endpoint
FF_XML_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.xml"

# HTTP headers for requests
HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/xml, text/xml, */*",
}


class ForexFactoryClient:
    """
    Client for fetching ForexFactory economic calendar data.

    Usage:
        client = ForexFactoryClient()

        # Fetch current week from XML API
        events = client.fetch_current_week()

        # Parse Kaggle CSV file
        events = client.parse_kaggle_csv("forex_factory_calendar.csv")
    """

    def __init__(self, timeout: int = 30):
        """
        Initialize ForexFactoryClient.

        Args:
            timeout: HTTP request timeout in seconds
        """
        self.timeout = timeout

    def fetch_current_week(self) -> List[NewsEvent]:
        """
        Fetch current week's events from ForexFactory XML API.

        Returns:
            List of NewsEvent objects for current week
        """
        try:
            request = Request(FF_XML_URL, headers=HTTP_HEADERS)
            with urlopen(request, timeout=self.timeout) as response:
                xml_data = response.read().decode("utf-8")

            return self._parse_xml(xml_data)

        except HTTPError as e:
            print(f"[ERROR] HTTP error fetching ForexFactory: {e.code} {e.reason}")
            return []
        except URLError as e:
            print(f"[ERROR] URL error fetching ForexFactory: {e.reason}")
            return []
        except Exception as e:
            print(f"[ERROR] Failed to fetch ForexFactory data: {e}")
            return []

    def _parse_xml(self, xml_data: str) -> List[NewsEvent]:
        """
        Parse ForexFactory XML data.

        XML format:
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
            ...
        </weeklyevents>
        """
        events: List[NewsEvent] = []

        try:
            root = ET.fromstring(xml_data)

            for event_elem in root.findall(".//event"):
                try:
                    title = event_elem.findtext("title", "").strip()
                    country = event_elem.findtext("country", "").strip()
                    date_str = event_elem.findtext("date", "").strip()
                    time_str = event_elem.findtext("time", "").strip()
                    impact = event_elem.findtext("impact", "").strip()
                    forecast = event_elem.findtext("forecast", "").strip() or None
                    previous = event_elem.findtext("previous", "").strip() or None

                    if not all([title, country, date_str, impact]):
                        continue

                    # Parse datetime (ET -> UTC)
                    datetime_utc = parse_forexfactory_datetime(date_str, time_str)

                    event = NewsEvent(
                        title=title,
                        country=country,
                        datetime_utc=datetime_utc,
                        impact=self._normalize_impact(impact),
                        forecast=forecast,
                        previous=previous,
                    )
                    events.append(event)

                except Exception as e:
                    print(f"[WARNING] Failed to parse event: {e}")
                    continue

        except ET.ParseError as e:
            print(f"[ERROR] Failed to parse XML: {e}")

        return events

    def parse_kaggle_csv(
        self,
        csv_path: Path,
        year_filter: Optional[int] = None,
    ) -> List[NewsEvent]:
        """
        Parse Kaggle ForexFactory historical dataset.

        CSV format (expected columns):
        - date: "2023-01-01" or "Jan 1, 2023"
        - time: "8:30am" or "8:30 AM"
        - currency: "USD"
        - impact: "High" or "Medium" or "Low"
        - event: "Non-Farm Payrolls"
        - actual, forecast, previous (optional)

        Args:
            csv_path: Path to CSV file
            year_filter: Only load events for specific year

        Returns:
            List of NewsEvent objects
        """
        events: List[NewsEvent] = []
        csv_path = Path(csv_path)

        if not csv_path.exists():
            print(f"[ERROR] CSV file not found: {csv_path}")
            return []

        try:
            with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
                # Try to detect delimiter
                sample = f.read(4096)
                f.seek(0)

                dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
                reader = csv.DictReader(f, dialect=dialect)

                for row in reader:
                    try:
                        event = self._parse_csv_row(row)
                        if event is None:
                            continue

                        # Apply year filter
                        if year_filter and event.datetime_utc.year != year_filter:
                            continue

                        events.append(event)

                    except Exception as e:
                        # Skip invalid rows silently
                        continue

        except Exception as e:
            print(f"[ERROR] Failed to parse CSV: {e}")

        print(f"[INFO] Parsed {len(events)} events from {csv_path.name}")
        return events

    def _parse_csv_row(self, row: Dict[str, str]) -> Optional[NewsEvent]:
        """
        Parse a single CSV row to NewsEvent.

        Handles various column naming conventions.
        """
        # Normalize column names (lowercase, strip)
        row = {k.lower().strip(): v.strip() for k, v in row.items() if v}

        # Extract fields (try multiple column name variants)
        title = (
            row.get("event") or
            row.get("title") or
            row.get("name") or
            row.get("event_name") or
            ""
        )

        country = (
            row.get("currency") or
            row.get("country") or
            row.get("ccy") or
            ""
        )

        date_str = (
            row.get("date") or
            row.get("datetime") or
            row.get("time") or  # Sometimes date is in time column
            ""
        )

        time_str = (
            row.get("time") or
            row.get("datetime") or
            "12:00pm"  # Default to noon if no time
        )

        impact = (
            row.get("impact") or
            row.get("importance") or
            row.get("priority") or
            "Medium"
        )

        forecast = row.get("forecast")
        previous = row.get("previous")
        actual = row.get("actual")

        if not all([title, country, date_str]):
            return None

        # Parse datetime
        try:
            # Handle various date formats
            if "-" in date_str and len(date_str) >= 10:
                # ISO format: 2023-01-01
                year = int(date_str[:4])
            else:
                year = None

            datetime_utc = parse_forexfactory_datetime(date_str, time_str, year)

        except Exception:
            return None

        return NewsEvent(
            title=title,
            country=country.upper(),
            datetime_utc=datetime_utc,
            impact=self._normalize_impact(impact),
            forecast=forecast,
            previous=previous,
            actual=actual,
        )

    def _normalize_impact(self, impact: str) -> str:
        """Normalize impact string to High/Medium/Low."""
        impact_lower = impact.lower().strip()

        if impact_lower in ("high", "red", "3", "important", "major"):
            return "High"
        elif impact_lower in ("medium", "orange", "2", "moderate"):
            return "Medium"
        elif impact_lower in ("low", "yellow", "1", "minor"):
            return "Low"
        else:
            return "Medium"  # Default to Medium

    def create_manual_event(
        self,
        title: str,
        country: str,
        date_obj: date,
        time_str: str = "8:30am",
        impact: str = "High",
        forecast: Optional[str] = None,
        previous: Optional[str] = None,
    ) -> NewsEvent:
        """
        Create a manual news event.

        Useful for adding events not in the API/CSV.

        Args:
            title: Event title (e.g., "Non-Farm Payrolls")
            country: Currency code (e.g., "USD")
            date_obj: Date of the event
            time_str: Time string in ET (e.g., "8:30am")
            impact: Impact level
            forecast: Forecast value
            previous: Previous value

        Returns:
            NewsEvent object
        """
        date_str = date_obj.strftime("%m-%d-%Y")
        datetime_utc = parse_forexfactory_datetime(date_str, time_str)

        return NewsEvent(
            title=title,
            country=country,
            datetime_utc=datetime_utc,
            impact=self._normalize_impact(impact),
            forecast=forecast,
            previous=previous,
        )


# Convenience function for common use
def fetch_forexfactory_events() -> List[NewsEvent]:
    """
    Fetch current week's events from ForexFactory.

    Returns:
        List of NewsEvent objects
    """
    client = ForexFactoryClient()
    return client.fetch_current_week()
