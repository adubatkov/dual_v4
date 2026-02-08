"""
Data models for News Filter.

Contains NewsEvent dataclass representing economic calendar events.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Tuple


@dataclass
class NewsEvent:
    """
    Represents a single economic news event from ForexFactory.

    All times are stored in UTC for consistency.
    Impact levels: "High", "Medium", "Low"
    """

    title: str
    country: str  # Currency code: USD, EUR, GBP, JPY, etc.
    datetime_utc: datetime
    impact: str  # "High", "Medium", "Low"
    forecast: Optional[str] = None
    previous: Optional[str] = None
    actual: Optional[str] = None

    def get_blocking_window(
        self,
        before_minutes: int = 2,
        after_minutes: int = 2,
    ) -> Tuple[datetime, datetime]:
        """
        Get the blocking window for this news event.

        According to 5ers rules:
        - No new orders 2 minutes before until 2 minutes after high-impact news

        Args:
            before_minutes: Minutes before event to start blocking (default: 2)
            after_minutes: Minutes after event to end blocking (default: 2)

        Returns:
            Tuple of (window_start, window_end) in UTC
        """
        window_start = self.datetime_utc - timedelta(minutes=before_minutes)
        window_end = self.datetime_utc + timedelta(minutes=after_minutes)
        return window_start, window_end

    def is_high_impact(self) -> bool:
        """Check if this is a high-impact news event."""
        return self.impact.lower() == "high"

    def __str__(self) -> str:
        return f"{self.country} {self.title} ({self.impact}) @ {self.datetime_utc.strftime('%Y-%m-%d %H:%M')} UTC"

    def __repr__(self) -> str:
        return (
            f"NewsEvent(title={self.title!r}, country={self.country!r}, "
            f"datetime_utc={self.datetime_utc!r}, impact={self.impact!r})"
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "title": self.title,
            "country": self.country,
            "datetime_utc": self.datetime_utc.isoformat(),
            "impact": self.impact,
            "forecast": self.forecast,
            "previous": self.previous,
            "actual": self.actual,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "NewsEvent":
        """Create from dictionary (JSON deserialization)."""
        datetime_utc = data["datetime_utc"]
        if isinstance(datetime_utc, str):
            # Parse ISO format datetime
            datetime_utc = datetime.fromisoformat(datetime_utc.replace("Z", "+00:00"))

        return cls(
            title=data["title"],
            country=data["country"],
            datetime_utc=datetime_utc,
            impact=data["impact"],
            forecast=data.get("forecast"),
            previous=data.get("previous"),
            actual=data.get("actual"),
        )


# Common high-impact events for reference
HIGH_IMPACT_EVENTS = [
    "Non-Farm Payrolls",
    "NFP",
    "FOMC Statement",
    "Fed Interest Rate Decision",
    "ECB Interest Rate Decision",
    "ECB Press Conference",
    "CPI",
    "Core CPI",
    "GDP",
    "Retail Sales",
    "Employment Change",
    "Unemployment Rate",
    "BOE Interest Rate Decision",
    "BOJ Interest Rate Decision",
    "RBA Interest Rate Decision",
    "RBNZ Interest Rate Decision",
    "SNB Interest Rate Decision",
    "Trade Balance",
    "ISM Manufacturing PMI",
    "ISM Services PMI",
]
