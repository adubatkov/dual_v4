"""
SMC configuration - per-instrument parameters.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class SMCConfig:
    """Configuration for SMC engine. One per instrument."""

    instrument: str                       # "GER40" / "XAUUSD"

    # --- Feature flags ---
    enable_fractals: bool = True
    enable_fvg: bool = True
    enable_cisd: bool = False             # Phase 2
    enable_bos: bool = False              # Phase 2

    # --- Fractal params ---
    fractal_timeframe: str = "H1"
    fractal_lookback_hours: int = 48
    fractal_proximity_pct: float = 0.002  # 0.2% of price

    # --- FVG params ---
    fvg_timeframes: List[str] = field(default_factory=lambda: ["M2", "H1"])
    fvg_min_size_points: float = 0.0      # 0 = any size

    # --- CISD params (Phase 2) ---
    cisd_timeframes: List[str] = field(default_factory=lambda: ["M2"])

    # --- BOS params (Phase 2) ---
    bos_timeframes: List[str] = field(default_factory=lambda: ["M5", "M15"])

    # --- Confluence weights (Phase 2) ---
    weight_fractal: float = 1.0
    weight_fvg: float = 1.5
    weight_cisd: float = 2.0
    weight_bos: float = 1.5
    min_confluence_score: float = 2.0

    # --- Confirmation (Phase 3) ---
    max_wait_minutes: int = 30

    # --- Timeframes for multi-TF context ---
    context_tfs: List[str] = field(default_factory=lambda: ["M2", "M5", "H1"])

    # --- Registry cleanup ---
    max_structure_age_hours: int = 72
