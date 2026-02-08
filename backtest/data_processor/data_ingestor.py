"""
DataIngestor - CSV aggregation and cleaning module.

Loads M1 candlestick data from multiple CSV files, validates, cleans,
and merges into a single DataFrame.
"""

import logging
from pathlib import Path
from typing import Optional, List, Union
from datetime import datetime

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class DataIngestor:
    """
    Aggregates and cleans M1 candlestick data from CSV files.

    Responsibilities:
        - Recursive search for CSV files in a directory
        - Validation of required columns
        - Removal of duplicate timestamps
        - Timezone normalization to UTC
        - Merging into a single sorted DataFrame
    """

    REQUIRED_COLUMNS = {"time", "open", "high", "low", "close"}
    OPTIONAL_COLUMNS = {"volume", "tick_volume"}

    def __init__(self, base_path: Union[str, Path]):
        """
        Initialize DataIngestor.

        Args:
            base_path: Root directory containing CSV files
        """
        self.base_path = Path(base_path)
        if not self.base_path.exists():
            raise FileNotFoundError(f"Data path does not exist: {self.base_path}")

        logger.info(f"DataIngestor initialized with path: {self.base_path}")

    def find_csv_files(self, pattern: str = "*.csv") -> List[Path]:
        """
        Recursively find all CSV files in the base path.

        Args:
            pattern: Glob pattern for CSV files

        Returns:
            List of Path objects to CSV files
        """
        csv_files = list(self.base_path.rglob(pattern))
        logger.info(f"Found {len(csv_files)} CSV files in {self.base_path}")
        return sorted(csv_files)

    def _validate_columns(self, df: pd.DataFrame, file_path: Path) -> bool:
        """
        Validate that DataFrame has required columns.

        Args:
            df: DataFrame to validate
            file_path: Path to source file (for error messages)

        Returns:
            True if valid, False otherwise
        """
        # Normalize column names to lowercase
        df.columns = df.columns.str.lower().str.strip()

        missing = self.REQUIRED_COLUMNS - set(df.columns)
        if missing:
            logger.error(f"Missing columns in {file_path.name}: {missing}")
            return False

        return True

    def _parse_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Parse time column to datetime with UTC timezone.

        Handles multiple datetime formats:
        - ISO 8601 with Z suffix (2025-10-06T00:15:00Z)
        - ISO 8601 with timezone offset
        - Standard datetime formats

        Args:
            df: DataFrame with 'time' column

        Returns:
            DataFrame with parsed datetime index
        """
        # Try parsing as datetime
        try:
            # First try ISO format (most common in our data)
            df["time"] = pd.to_datetime(df["time"], utc=True)
        except Exception:
            # Try alternative formats
            try:
                df["time"] = pd.to_datetime(df["time"], format="mixed", utc=True)
            except Exception as e:
                logger.error(f"Failed to parse datetime: {e}")
                raise

        return df

    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate DataFrame.

        Operations:
        - Remove rows with NaN in OHLC columns
        - Validate OHLC relationships (high >= low, etc.)
        - Add volume column if missing (set to 0)

        Args:
            df: Raw DataFrame

        Returns:
            Cleaned DataFrame
        """
        initial_len = len(df)

        # Remove NaN in OHLC
        ohlc_cols = ["open", "high", "low", "close"]
        df = df.dropna(subset=ohlc_cols)

        # Validate OHLC relationships
        valid_mask = (
            (df["high"] >= df["low"]) &
            (df["high"] >= df["open"]) &
            (df["high"] >= df["close"]) &
            (df["low"] <= df["open"]) &
            (df["low"] <= df["close"])
        )

        invalid_count = (~valid_mask).sum()
        if invalid_count > 0:
            logger.warning(f"Found {invalid_count} rows with invalid OHLC relationships")
            df = df[valid_mask]

        # Add volume if missing
        if "volume" not in df.columns and "tick_volume" not in df.columns:
            df["volume"] = 0
        elif "tick_volume" in df.columns and "volume" not in df.columns:
            df["volume"] = df["tick_volume"]

        cleaned_len = len(df)
        if cleaned_len < initial_len:
            logger.info(f"Cleaned {initial_len - cleaned_len} invalid rows")

        return df

    def load_single_csv(self, file_path: Path) -> Optional[pd.DataFrame]:
        """
        Load and validate a single CSV file.

        Args:
            file_path: Path to CSV file

        Returns:
            Cleaned DataFrame or None if invalid
        """
        try:
            # Read CSV
            df = pd.read_csv(file_path)

            # Validate columns
            if not self._validate_columns(df, file_path):
                return None

            # Parse datetime
            df = self._parse_datetime(df)

            # Clean data
            df = self._clean_dataframe(df)

            # Select only needed columns
            columns = ["time", "open", "high", "low", "close", "volume"]
            df = df[[c for c in columns if c in df.columns]]

            logger.debug(f"Loaded {len(df)} rows from {file_path.name}")
            return df

        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return None

    def load_all(self, symbol: Optional[str] = None) -> pd.DataFrame:
        """
        Load and merge all CSV files into a single DataFrame.

        Args:
            symbol: Optional symbol name to filter files

        Returns:
            Merged and sorted DataFrame with columns:
            [time, open, high, low, close, volume]
        """
        csv_files = self.find_csv_files()

        if not csv_files:
            raise ValueError(f"No CSV files found in {self.base_path}")

        # Filter by symbol if specified
        if symbol:
            csv_files = [f for f in csv_files if symbol.upper() in f.name.upper()]
            logger.info(f"Filtered to {len(csv_files)} files for symbol {symbol}")

        # Load all files
        dataframes = []
        for file_path in csv_files:
            df = self.load_single_csv(file_path)
            if df is not None and not df.empty:
                dataframes.append(df)

        if not dataframes:
            raise ValueError("No valid data loaded from CSV files")

        # Merge all DataFrames
        logger.info(f"Merging {len(dataframes)} DataFrames...")
        merged_df = pd.concat(dataframes, ignore_index=True)

        # Remove duplicates by timestamp
        initial_len = len(merged_df)
        merged_df = merged_df.drop_duplicates(subset=["time"], keep="last")
        duplicates_removed = initial_len - len(merged_df)
        if duplicates_removed > 0:
            logger.info(f"Removed {duplicates_removed} duplicate timestamps")

        # Sort by time
        merged_df = merged_df.sort_values("time").reset_index(drop=True)

        # Final statistics
        logger.info(f"Loaded {len(merged_df)} M1 candles")
        logger.info(f"Date range: {merged_df['time'].min()} to {merged_df['time'].max()}")

        return merged_df

    def get_date_range(self, df: pd.DataFrame) -> tuple:
        """
        Get the date range of the DataFrame.

        Args:
            df: DataFrame with 'time' column

        Returns:
            Tuple of (start_date, end_date)
        """
        return df["time"].min(), df["time"].max()

    def filter_date_range(
        self,
        df: pd.DataFrame,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Filter DataFrame to a specific date range.

        Args:
            df: DataFrame to filter
            start_date: Start date (inclusive)
            end_date: End date (inclusive)

        Returns:
            Filtered DataFrame
        """
        if start_date:
            if start_date.tzinfo is None:
                start_date = start_date.replace(tzinfo=pd.Timestamp.now().tz)
            df = df[df["time"] >= start_date]

        if end_date:
            if end_date.tzinfo is None:
                end_date = end_date.replace(tzinfo=pd.Timestamp.now().tz)
            df = df[df["time"] <= end_date]

        return df.reset_index(drop=True)


def load_symbol_data(
    symbol: str,
    data_path: Union[str, Path],
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
) -> pd.DataFrame:
    """
    Convenience function to load data for a specific symbol.

    Args:
        symbol: Symbol name (GER40, XAUUSD)
        data_path: Path to data directory
        start_date: Optional start date filter
        end_date: Optional end date filter

    Returns:
        DataFrame with M1 candlestick data
    """
    ingestor = DataIngestor(data_path)
    df = ingestor.load_all(symbol=symbol)

    if start_date or end_date:
        df = ingestor.filter_date_range(df, start_date, end_date)

    return df
