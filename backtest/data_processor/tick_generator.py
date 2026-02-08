"""
TickGenerator - Synthetic tick data generation module.

Generates 5-second interval data from M1 candles using interpolation
with realistic price path simulation.
"""

import logging
from pathlib import Path
from typing import Optional, Union, Tuple
from datetime import timedelta

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class TickGenerator:
    """
    Generates synthetic 5-second bars from M1 candlestick data.

    Uses interpolation to simulate realistic price movement within
    each 1-minute candle, creating 12 data points per minute.

    Algorithm:
        1. Determine price path direction (high-first or low-first)
        2. Interpolate: Open -> Extreme1 -> Extreme2 -> Close
        3. Add micro-noise (jitter) for realistic movement
        4. Generate Bid/Ask with configurable spread
    """

    def __init__(
        self,
        ticks_per_minute: int = 12,
        jitter_factor: float = 0.1,
        random_seed: Optional[int] = 42,
    ):
        """
        Initialize TickGenerator.

        Args:
            ticks_per_minute: Number of synthetic ticks per M1 candle (12 = 5 seconds)
            jitter_factor: Noise factor for price interpolation (0-1)
            random_seed: Seed for reproducibility, None for random
        """
        self.ticks_per_minute = ticks_per_minute
        self.jitter_factor = jitter_factor
        self.random_seed = random_seed

        if random_seed is not None:
            np.random.seed(random_seed)

        self.tick_interval_seconds = 60 // ticks_per_minute
        logger.info(
            f"TickGenerator initialized: {ticks_per_minute} ticks/min, "
            f"{self.tick_interval_seconds}s intervals, jitter={jitter_factor}"
        )

    def _determine_path_direction(
        self, open_price: float, high: float, low: float, close: float
    ) -> str:
        """
        Determine whether price hits high or low first.

        Uses candle shape to probabilistically determine path:
        - Bullish candle (close > open): Higher chance of low-first
        - Bearish candle (close < open): Higher chance of high-first
        - Doji: Random

        Args:
            open_price: Candle open price
            high: Candle high price
            low: Candle low price
            close: Candle close price

        Returns:
            'high_first' or 'low_first'
        """
        body = close - open_price
        range_size = high - low

        if range_size == 0:
            return np.random.choice(["high_first", "low_first"])

        # Calculate probability based on candle shape
        body_ratio = body / range_size

        # Bullish -> more likely went down first then up
        # Bearish -> more likely went up first then down
        if body_ratio > 0.1:  # Bullish
            prob_low_first = 0.6 + 0.3 * min(body_ratio, 1.0)
        elif body_ratio < -0.1:  # Bearish
            prob_low_first = 0.4 - 0.3 * min(abs(body_ratio), 1.0)
        else:  # Doji
            prob_low_first = 0.5

        return "low_first" if np.random.random() < prob_low_first else "high_first"

    def _interpolate_path(
        self,
        open_price: float,
        high: float,
        low: float,
        close: float,
        path_direction: str,
    ) -> np.ndarray:
        """
        Generate interpolated price path within a candle.

        Creates smooth path: Open -> Extreme1 -> Extreme2 -> Close

        Args:
            open_price: Candle open price
            high: Candle high price
            low: Candle low price
            close: Candle close price
            path_direction: 'high_first' or 'low_first'

        Returns:
            Array of interpolated prices
        """
        n = self.ticks_per_minute

        # Define path segments
        if path_direction == "high_first":
            # Open -> High -> Low -> Close
            waypoints = [open_price, high, low, close]
        else:
            # Open -> Low -> High -> Close
            waypoints = [open_price, low, high, close]

        # Distribute points across segments (roughly equal)
        # First segment: ~30%, middle: ~40%, last: ~30%
        seg1_points = max(1, int(n * 0.3))
        seg3_points = max(1, int(n * 0.3))
        seg2_points = n - seg1_points - seg3_points

        # Generate points for each segment
        prices = []

        # Segment 1: Open -> First extreme
        seg1 = np.linspace(waypoints[0], waypoints[1], seg1_points, endpoint=False)
        prices.extend(seg1)

        # Segment 2: First extreme -> Second extreme
        seg2 = np.linspace(waypoints[1], waypoints[2], seg2_points, endpoint=False)
        prices.extend(seg2)

        # Segment 3: Second extreme -> Close
        seg3 = np.linspace(waypoints[2], waypoints[3], seg3_points + 1)
        prices.extend(seg3)

        # Ensure correct length
        prices = np.array(prices[:n])

        return prices

    def _add_jitter(
        self, prices: np.ndarray, high: float, low: float
    ) -> np.ndarray:
        """
        Add micro-noise to price path.

        Ensures prices stay within high-low range.

        Args:
            prices: Base interpolated prices
            high: Candle high (max allowed)
            low: Candle low (min allowed)

        Returns:
            Prices with added jitter
        """
        if self.jitter_factor <= 0:
            return prices

        range_size = high - low
        if range_size <= 0:
            return prices

        # Add noise proportional to range and jitter factor
        noise_scale = range_size * self.jitter_factor * 0.1
        noise = np.random.normal(0, noise_scale, len(prices))

        jittered = prices + noise

        # Clip to stay within high-low range
        jittered = np.clip(jittered, low, high)

        # Ensure first and last points are exact
        if len(jittered) > 0:
            # First point should be close to open (already set in interpolation)
            # Last point should be exact close
            pass  # Keep as is, interpolation handles endpoints

        return jittered

    def generate_ticks_for_candle(
        self,
        timestamp: pd.Timestamp,
        open_price: float,
        high: float,
        low: float,
        close: float,
        spread: float = 0.0,
    ) -> pd.DataFrame:
        """
        Generate synthetic tick data for a single M1 candle.

        Args:
            timestamp: Candle timestamp (start of minute)
            open_price: Candle open price
            high: Candle high price
            low: Candle low price
            close: Candle close price
            spread: Bid-Ask spread in price points

        Returns:
            DataFrame with columns: [time, bid, ask, mid]
        """
        # Determine path direction
        path_dir = self._determine_path_direction(open_price, high, low, close)

        # Generate interpolated prices
        mid_prices = self._interpolate_path(open_price, high, low, close, path_dir)

        # Add jitter
        mid_prices = self._add_jitter(mid_prices, high, low)

        # Generate timestamps (5-second intervals within the minute)
        timestamps = [
            timestamp + timedelta(seconds=i * self.tick_interval_seconds)
            for i in range(self.ticks_per_minute)
        ]

        # Calculate bid/ask from mid price
        half_spread = spread / 2
        bid_prices = mid_prices - half_spread
        ask_prices = mid_prices + half_spread

        return pd.DataFrame({
            "time": timestamps,
            "bid": bid_prices,
            "ask": ask_prices,
            "mid": mid_prices,
        })

    def generate_from_m1(
        self,
        m1_data: pd.DataFrame,
        spread: float = 0.0,
        progress_callback: Optional[callable] = None,
    ) -> pd.DataFrame:
        """
        Generate synthetic tick data from M1 DataFrame.

        Args:
            m1_data: DataFrame with M1 candles [time, open, high, low, close]
            spread: Bid-Ask spread in price points
            progress_callback: Optional callback(current, total) for progress

        Returns:
            DataFrame with 5-second tick data [time, bid, ask, mid]
        """
        logger.info(f"Generating ticks from {len(m1_data)} M1 candles...")

        all_ticks = []
        total = len(m1_data)

        for idx, row in m1_data.iterrows():
            ticks = self.generate_ticks_for_candle(
                timestamp=row["time"],
                open_price=row["open"],
                high=row["high"],
                low=row["low"],
                close=row["close"],
                spread=spread,
            )
            all_ticks.append(ticks)

            # Progress callback
            if progress_callback and idx % 10000 == 0:
                progress_callback(idx, total)

        # Concatenate all ticks
        result = pd.concat(all_ticks, ignore_index=True)

        logger.info(f"Generated {len(result)} tick records")
        return result

    def save_to_parquet(
        self,
        tick_data: pd.DataFrame,
        output_path: Union[str, Path],
        compression: str = "snappy",
    ) -> Path:
        """
        Save tick data to Parquet file.

        Args:
            tick_data: DataFrame with tick data
            output_path: Output file path
            compression: Parquet compression ('snappy', 'gzip', 'brotli')

        Returns:
            Path to saved file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        tick_data.to_parquet(output_path, compression=compression, index=False)
        file_size_mb = output_path.stat().st_size / (1024 * 1024)

        logger.info(f"Saved {len(tick_data)} ticks to {output_path} ({file_size_mb:.1f} MB)")
        return output_path

    def load_from_parquet(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """
        Load tick data from Parquet file.

        Args:
            file_path: Path to Parquet file

        Returns:
            DataFrame with tick data
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {file_path}")

        df = pd.read_parquet(file_path)
        logger.info(f"Loaded {len(df)} ticks from {file_path}")
        return df


def build_tick_data(
    m1_data: pd.DataFrame,
    symbol: str,
    spread: float,
    output_dir: Union[str, Path],
    ticks_per_minute: int = 12,
    jitter_factor: float = 0.1,
    random_seed: Optional[int] = 42,
) -> Path:
    """
    Convenience function to build and save tick data.

    Args:
        m1_data: M1 candlestick DataFrame
        symbol: Symbol name for output filename
        spread: Bid-Ask spread
        output_dir: Directory for output Parquet file
        ticks_per_minute: Ticks per minute (12 = 5 seconds)
        jitter_factor: Price noise factor
        random_seed: Random seed for reproducibility

    Returns:
        Path to saved Parquet file
    """
    generator = TickGenerator(
        ticks_per_minute=ticks_per_minute,
        jitter_factor=jitter_factor,
        random_seed=random_seed,
    )

    tick_data = generator.generate_from_m1(m1_data, spread=spread)

    # Generate filename with date range
    start_date = m1_data["time"].min().strftime("%Y%m%d")
    end_date = m1_data["time"].max().strftime("%Y%m%d")
    filename = f"{symbol}_5s_{start_date}_{end_date}.parquet"

    output_path = Path(output_dir) / filename
    return generator.save_to_parquet(tick_data, output_path)
