"""
SMC Analyzer - Standalone tool for analyzing SMC structures on historical data.

Usage:
    analyzer = SMCAnalyzer("GER40")
    analyzer.load_data("2024-06-01", "2024-06-30")
    report = analyzer.analyze_day("2024-06-15")
    analyzer.print_report(report)
"""

import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# Add parent path for imports
DUAL_V4_PATH = Path("C:/Trading/ib_trading_bot/dual_v4")
sys.path.insert(0, str(DUAL_V4_PATH))

from src.smc.config import SMCConfig
from src.smc.detectors.fractal_detector import detect_fractals, find_unswept_fractals
from src.smc.detectors.fvg_detector import check_fvg_fill, detect_fvg
from src.smc.models import FVG, Fractal
from src.smc.timeframe_manager import TimeframeManager


class SMCAnalyzer:
    """Standalone SMC structure analysis on historical data."""

    def __init__(self, instrument: str, config: Optional[SMCConfig] = None):
        self.instrument = instrument
        self.config = config or SMCConfig(instrument=instrument)
        self.m1_data: Optional[pd.DataFrame] = None
        self.tfm: Optional[TimeframeManager] = None

    def load_data(self, start_date: str, end_date: str) -> None:
        """Load M1 data from CSV files for date range.

        Args:
            start_date: "YYYY-MM-DD"
            end_date: "YYYY-MM-DD"
        """
        from backtest.config import DATA_FOLDERS, DEFAULT_CONFIG

        data_folder = DEFAULT_CONFIG.data_base_path / DATA_FOLDERS[self.instrument]

        csv_files = sorted(data_folder.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files in {data_folder}")

        frames = []
        for f in csv_files:
            try:
                tmp = pd.read_csv(f)
                tmp["time"] = pd.to_datetime(tmp["time"].astype(str).str.strip(), errors="coerce", utc=True)
                tmp = tmp.dropna(subset=["time"])
                for col in ("open", "high", "low", "close"):
                    tmp[col] = pd.to_numeric(tmp[col], errors="coerce")
                frames.append(tmp[["time", "open", "high", "low", "close"]])
            except Exception as e:
                print(f"[WARNING] Skipping {f.name}: {e}")

        if not frames:
            raise ValueError("No data loaded")

        df = pd.concat(frames, ignore_index=True).sort_values("time").drop_duplicates(subset=["time"])
        df = df.dropna()

        # Filter date range
        start_dt = pd.Timestamp(start_date, tz="UTC")
        end_dt = pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(days=1)
        df = df[(df["time"] >= start_dt) & (df["time"] < end_dt)].reset_index(drop=True)

        if df.empty:
            raise ValueError(f"No data for {start_date} to {end_date}")

        self.m1_data = df
        self.tfm = TimeframeManager(m1_data=df, instrument=self.instrument)

        print(f"[OK] Loaded {len(df)} M1 bars: {df['time'].iloc[0]} to {df['time'].iloc[-1]}")

    def analyze_day(self, day_date: str) -> Dict:
        """Run SMC analysis for a single day.

        Args:
            day_date: "YYYY-MM-DD"

        Returns:
            dict with fractals, FVGs, and summary stats.
        """
        if self.tfm is None:
            raise RuntimeError("Call load_data() first")

        day_start = pd.Timestamp(day_date, tz="UTC")
        day_end = day_start + pd.Timedelta(days=1)

        # H1 fractals (include lookback before this day)
        lookback_start = day_start - pd.Timedelta(hours=self.config.fractal_lookback_hours)
        h1_data = self.tfm.get_data("H1", up_to=day_end)
        h1_lookback = h1_data[h1_data["time"] >= lookback_start] if not h1_data.empty else h1_data

        fractals_h1 = detect_fractals(
            h1_lookback, self.instrument, "H1", candle_duration_hours=1.0
        )

        # Unswept fractals at day start (approximate IB time)
        unswept_at_start = find_unswept_fractals(
            fractals_h1,
            self.m1_data,
            before_time=day_start.to_pydatetime(),
            lookback_hours=self.config.fractal_lookback_hours,
        )

        # FVGs on each configured timeframe
        fvgs_by_tf = {}
        for tf in self.config.fvg_timeframes:
            tf_data = self.tfm.get_data(tf, up_to=day_end)
            if not tf_data.empty:
                tf_day = tf_data[(tf_data["time"] >= day_start) & (tf_data["time"] < day_end)]
                fvgs_by_tf[tf] = detect_fvg(
                    tf_day,
                    self.instrument,
                    tf,
                    min_size_points=self.config.fvg_min_size_points,
                )
            else:
                fvgs_by_tf[tf] = []

        # Summary
        total_fvgs = sum(len(v) for v in fvgs_by_tf.values())
        bullish_fvgs = sum(1 for fvgs in fvgs_by_tf.values() for f in fvgs if f.direction == "bullish")
        bearish_fvgs = total_fvgs - bullish_fvgs

        return {
            "day": day_date,
            "instrument": self.instrument,
            "fractals_h1": fractals_h1,
            "unswept_fractals": unswept_at_start,
            "fvgs_by_tf": fvgs_by_tf,
            "summary": {
                "h1_fractals_total": len(fractals_h1),
                "h1_fractals_high": len([f for f in fractals_h1 if f.type == "high"]),
                "h1_fractals_low": len([f for f in fractals_h1 if f.type == "low"]),
                "unswept_at_start": len(unswept_at_start),
                "fvgs_total": total_fvgs,
                "fvgs_bullish": bullish_fvgs,
                "fvgs_bearish": bearish_fvgs,
            },
        }

    def analyze_range(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Run analysis for date range. Returns summary DataFrame.

        Args:
            start_date: "YYYY-MM-DD"
            end_date: "YYYY-MM-DD"
        """
        start = date.fromisoformat(start_date)
        end = date.fromisoformat(end_date)

        rows = []
        current = start
        while current <= end:
            day_str = current.isoformat()
            try:
                report = self.analyze_day(day_str)
                row = {"day": day_str, **report["summary"]}
                rows.append(row)
            except Exception as e:
                print(f"[WARNING] {day_str}: {e}")
            current += timedelta(days=1)

        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows)

    def print_report(self, report: Dict) -> None:
        """Print human-readable analysis report."""
        s = report["summary"]
        print(f"\n{'='*60}")
        print(f"SMC Analysis: {report['instrument']} | {report['day']}")
        print(f"{'='*60}")

        print(f"\nH1 Fractals: {s['h1_fractals_total']} (high: {s['h1_fractals_high']}, low: {s['h1_fractals_low']})")
        print(f"Unswept at day start: {s['unswept_at_start']}")

        if report["unswept_fractals"]:
            print("\n  Unswept fractal levels:")
            for f in report["unswept_fractals"]:
                print(f"    {f.type:>4} @ {f.price:.2f}  (formed {f.time})")

        print(f"\nFVGs: {s['fvgs_total']} (bullish: {s['fvgs_bullish']}, bearish: {s['fvgs_bearish']})")
        for tf, fvgs in report["fvgs_by_tf"].items():
            if fvgs:
                print(f"\n  {tf} FVGs:")
                for fvg in fvgs:
                    size = fvg.high - fvg.low
                    print(f"    {fvg.direction:>7} | {fvg.low:.2f} - {fvg.high:.2f} (size: {size:.2f}) @ {fvg.formation_time}")

        print(f"\n{'='*60}\n")
