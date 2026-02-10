"""
SMC Event Log - chronological audit trail of all SMC events.
"""

from dataclasses import asdict
from datetime import datetime
from typing import List, Optional

import pandas as pd

from .models import SMCEvent


class SMCEventLog:
    """Chronological audit trail. Append-only during execution.
    Queryable for debugging. Exportable to CSV/DataFrame.
    """

    def __init__(self):
        self.events: List[SMCEvent] = []

    def record(self, event: SMCEvent) -> None:
        """Record a new event."""
        self.events.append(event)

    def record_simple(
        self,
        timestamp: datetime,
        instrument: str,
        event_type: str,
        timeframe: str,
        direction: Optional[str] = None,
        price: Optional[float] = None,
        structure_id: Optional[str] = None,
        bias_impact: float = 0.0,
        **details,
    ) -> None:
        """Convenience method to record an event."""
        self.record(SMCEvent(
            timestamp=timestamp,
            instrument=instrument,
            event_type=event_type,
            timeframe=timeframe,
            direction=direction,
            price=price,
            structure_id=structure_id,
            details=details,
            bias_impact=bias_impact,
        ))

    def get_events(
        self,
        event_type: Optional[str] = None,
        after: Optional[datetime] = None,
        before: Optional[datetime] = None,
        structure_id: Optional[str] = None,
    ) -> List[SMCEvent]:
        """Query events with filters."""
        result = self.events
        if event_type:
            result = [e for e in result if e.event_type == event_type]
        if after:
            result = [e for e in result if e.timestamp >= after]
        if before:
            result = [e for e in result if e.timestamp < before]
        if structure_id:
            result = [e for e in result if e.structure_id == structure_id]
        return result

    def get_bias_at_time(self, time: datetime) -> float:
        """Sum of bias_impact of all events up to given time.
        Positive = long bias, negative = short bias.
        """
        return sum(e.bias_impact for e in self.events if e.timestamp <= time)

    def to_dataframe(self) -> pd.DataFrame:
        """Export as DataFrame."""
        if not self.events:
            return pd.DataFrame()
        return pd.DataFrame([asdict(e) for e in self.events])

    def to_csv(self, path: str) -> None:
        """Export to CSV file."""
        self.to_dataframe().to_csv(path, index=False)

    def clear(self) -> None:
        """Clear all events."""
        self.events.clear()

    def __len__(self) -> int:
        return len(self.events)
