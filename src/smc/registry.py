"""
SMC Registry - in-memory container for all active/invalidated structures.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
from dataclasses import asdict


class SMCRegistry:
    """In-memory registry of all SMC structures for one instrument.

    Supports:
    - Add new structures
    - Query active structures by type/timeframe/direction
    - Update status (active -> partial_fill -> invalidated, etc.)
    - Cleanup old structures
    """

    STRUCTURE_TYPES = ("fvg", "fractal", "cisd", "bos")

    def __init__(self, instrument: str):
        self.instrument = instrument
        self._store: Dict[str, Dict[str, Any]] = {
            stype: {} for stype in self.STRUCTURE_TYPES
        }

    def add(self, structure_type: str, structure: Any) -> None:
        """Add a structure to the registry."""
        if structure_type not in self._store:
            raise ValueError(f"Unknown structure type: {structure_type}")
        self._store[structure_type][structure.id] = structure

    def get_by_id(self, structure_type: str, structure_id: str) -> Optional[Any]:
        """Get structure by ID."""
        return self._store.get(structure_type, {}).get(structure_id)

    def get_active(
        self,
        structure_type: str,
        timeframe: Optional[str] = None,
        direction: Optional[str] = None,
        before_time: Optional[datetime] = None,
    ) -> List[Any]:
        """Query active structures with optional filters."""
        results = []
        for s in self._store.get(structure_type, {}).values():
            if s.status != "active":
                continue
            if timeframe and s.timeframe != timeframe:
                continue
            if direction and hasattr(s, "direction") and s.direction != direction:
                continue
            if before_time:
                t = getattr(s, "formation_time", None) or getattr(s, "confirmed_time", None)
                if t and t > before_time:
                    continue
            results.append(s)
        return results

    def update_status(self, structure_type: str, structure_id: str,
                      new_status: str, **kwargs) -> bool:
        """Update a structure's status and optional fields.

        Returns True if found and updated.
        """
        s = self.get_by_id(structure_type, structure_id)
        if s is None:
            return False
        object.__setattr__(s, "status", new_status)
        for key, value in kwargs.items():
            if hasattr(s, key):
                object.__setattr__(s, key, value)
        return True

    def get_unswept_fractals(
        self,
        timeframe: Optional[str] = None,
        before_time: Optional[datetime] = None,
    ) -> List[Any]:
        """Get all active (unswept) fractals."""
        return [
            f for f in self.get_active("fractal", timeframe=timeframe, before_time=before_time)
            if not f.swept
        ]

    def get_fvgs_near_price(
        self,
        timeframe: str,
        price: float,
        max_distance_pct: float = 0.005,
        direction: Optional[str] = None,
    ) -> List[Any]:
        """Get active FVGs within distance of price."""
        threshold = price * max_distance_pct
        results = []
        for fvg in self.get_active("fvg", timeframe=timeframe, direction=direction):
            if fvg.low <= price <= fvg.high:
                dist = 0.0
            else:
                dist = min(abs(price - fvg.high), abs(price - fvg.low))
            if dist <= threshold:
                results.append(fvg)
        return results

    def cleanup(self, before_time: datetime) -> int:
        """Remove structures formed before given time. Returns count removed."""
        removed = 0
        for stype in self._store:
            to_remove = []
            for sid, s in self._store[stype].items():
                t = (
                    getattr(s, "formation_time", None)
                    or getattr(s, "confirmed_time", None)
                    or getattr(s, "break_time", None)
                )
                if t and t < before_time:
                    to_remove.append(sid)
            for sid in to_remove:
                del self._store[stype][sid]
                removed += 1
        return removed

    def count(self, structure_type: Optional[str] = None) -> int:
        """Count structures in registry."""
        if structure_type:
            return len(self._store.get(structure_type, {}))
        return sum(len(v) for v in self._store.values())

    def clear(self) -> None:
        """Remove all structures."""
        for stype in self._store:
            self._store[stype].clear()

    def to_dataframe(self, structure_type: str) -> pd.DataFrame:
        """Export structures as DataFrame for analysis."""
        records = [asdict(s) for s in self._store.get(structure_type, {}).values()]
        if not records:
            return pd.DataFrame()
        return pd.DataFrame(records)
