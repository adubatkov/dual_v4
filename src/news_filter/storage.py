"""
Storage module for News Filter.

Handles loading and saving news events to JSON files.
Events are stored by year for efficient querying.
"""

import json
from datetime import datetime, date
from pathlib import Path
from typing import List, Optional, Dict, Any

from .models import NewsEvent


# Default storage directory
DEFAULT_DATA_DIR = Path(__file__).parent.parent.parent / "data" / "news"


class NewsStorage:
    """
    Manages storage and retrieval of news events.

    Events are stored in JSON files organized by year:
    - forex_factory_2023.json
    - forex_factory_2024.json
    - etc.

    Format:
    {
        "version": "1.0",
        "source": "forexfactory",
        "last_updated": "2026-01-12T10:00:00Z",
        "events": [...]
    }
    """

    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize NewsStorage.

        Args:
            data_dir: Directory for JSON files (default: dual_v3/data/news/)
        """
        self.data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[int, List[NewsEvent]] = {}

    def _get_file_path(self, year: int) -> Path:
        """Get file path for a specific year."""
        return self.data_dir / f"forex_factory_{year}.json"

    def load_year(self, year: int, use_cache: bool = True) -> List[NewsEvent]:
        """
        Load all events for a specific year.

        Args:
            year: Year to load
            use_cache: Use in-memory cache if available

        Returns:
            List of NewsEvent objects
        """
        if use_cache and year in self._cache:
            return self._cache[year]

        file_path = self._get_file_path(year)
        if not file_path.exists():
            return []

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            events = [NewsEvent.from_dict(e) for e in data.get("events", [])]

            if use_cache:
                self._cache[year] = events

            return events

        except (json.JSONDecodeError, KeyError) as e:
            print(f"[WARNING] Failed to load {file_path}: {e}")
            return []

    def save_year(self, year: int, events: List[NewsEvent]) -> None:
        """
        Save events for a specific year.

        Args:
            year: Year to save
            events: List of NewsEvent objects
        """
        file_path = self._get_file_path(year)

        data = {
            "version": "1.0",
            "source": "forexfactory",
            "last_updated": datetime.utcnow().isoformat() + "Z",
            "year": year,
            "event_count": len(events),
            "events": [e.to_dict() for e in events],
        }

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        # Update cache
        self._cache[year] = events

        print(f"[INFO] Saved {len(events)} events to {file_path}")

    def load_range(
        self,
        start_date: date,
        end_date: date,
        impact_filter: Optional[str] = None,
        country_filter: Optional[List[str]] = None,
    ) -> List[NewsEvent]:
        """
        Load events for a date range.

        Args:
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            impact_filter: Filter by impact level ("High", "Medium", "Low")
            country_filter: Filter by countries (e.g., ["USD", "EUR"])

        Returns:
            List of NewsEvent objects sorted by datetime
        """
        events: List[NewsEvent] = []

        # Load all years in range
        for year in range(start_date.year, end_date.year + 1):
            year_events = self.load_year(year)
            events.extend(year_events)

        # Filter by date range
        start_dt = datetime.combine(start_date, datetime.min.time())
        end_dt = datetime.combine(end_date, datetime.max.time())

        filtered = [
            e for e in events
            if start_dt <= e.datetime_utc.replace(tzinfo=None) <= end_dt
        ]

        # Filter by impact
        if impact_filter:
            filtered = [e for e in filtered if e.impact.lower() == impact_filter.lower()]

        # Filter by country
        if country_filter:
            country_set = {c.upper() for c in country_filter}
            filtered = [e for e in filtered if e.country.upper() in country_set]

        # Sort by datetime
        filtered.sort(key=lambda e: e.datetime_utc)

        return filtered

    def load_high_impact(
        self,
        start_date: date,
        end_date: date,
        countries: Optional[List[str]] = None,
    ) -> List[NewsEvent]:
        """
        Load only high-impact events.

        Convenience method for common use case.

        Args:
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            countries: Filter by countries

        Returns:
            List of high-impact NewsEvent objects
        """
        return self.load_range(
            start_date=start_date,
            end_date=end_date,
            impact_filter="High",
            country_filter=countries,
        )

    def add_events(self, events: List[NewsEvent]) -> None:
        """
        Add events to storage.

        Events are automatically sorted into correct year files.

        Args:
            events: List of NewsEvent objects to add
        """
        # Group by year
        by_year: Dict[int, List[NewsEvent]] = {}
        for event in events:
            year = event.datetime_utc.year
            if year not in by_year:
                by_year[year] = []
            by_year[year].append(event)

        # Merge with existing and save
        for year, new_events in by_year.items():
            existing = self.load_year(year, use_cache=False)

            # Create set of existing event keys for deduplication
            existing_keys = {
                (e.title, e.country, e.datetime_utc.isoformat())
                for e in existing
            }

            # Add only non-duplicate events
            for event in new_events:
                key = (event.title, event.country, event.datetime_utc.isoformat())
                if key not in existing_keys:
                    existing.append(event)
                    existing_keys.add(key)

            # Sort and save
            existing.sort(key=lambda e: e.datetime_utc)
            self.save_year(year, existing)

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about stored events."""
        stats = {
            "years": [],
            "total_events": 0,
            "high_impact_events": 0,
            "countries": set(),
        }

        # Check all files in data_dir
        for file_path in self.data_dir.glob("forex_factory_*.json"):
            try:
                year = int(file_path.stem.split("_")[-1])
                events = self.load_year(year)

                stats["years"].append(year)
                stats["total_events"] += len(events)
                stats["high_impact_events"] += sum(1 for e in events if e.is_high_impact())
                stats["countries"].update(e.country for e in events)

            except (ValueError, Exception):
                continue

        stats["years"].sort()
        stats["countries"] = sorted(stats["countries"])

        return stats

    def clear_cache(self) -> None:
        """Clear in-memory cache."""
        self._cache.clear()
