# Data Sources and Directory Structure

Last updated: 2026-02-08

## Table of Contents

- [Overview](#overview)
- [Directory Structure](#directory-structure)
- [Original Data (In-Sample)](#original-data-in-sample)
- [Extended Data (Control Period)](#extended-data-control-period)
- [Optimized Data (Parquet)](#optimized-data-parquet)
- [Control Period Data](#control-period-data)
- [News Data](#news-data)
- [CSV Format Specification](#csv-format-specification)
- [Data Ingestion Pipeline](#data-ingestion-pipeline)
- [Timezone Considerations](#timezone-considerations)
- [Configuration References](#configuration-references)

---

## Overview

The dual_v3 project uses M1 (1-minute) candlestick data for two instruments:

| Instrument | Source Broker | Data Provider | Primary Timezone |
|------------|--------------|---------------|-----------------|
| GER40 (DAX) | Pepperstone | TradingView CSV export | Europe/Berlin (CET/CEST) |
| XAUUSD (Gold) | Oanda | TradingView CSV export | Asia/Tokyo (JST) |

Data is organized into four categories:
1. **Original** -- Raw CSV files for in-sample optimization (Jan 2023 - Apr 2025)
2. **Extended** -- Raw CSV files for control/validation period (Nov 2025 - Jan 2026)
3. **Optimized** -- Parquet files derived from original CSVs for fast backtest execution
4. **News** -- ForexFactory economic calendar JSON files (2023-2026)

---

## Directory Structure

```
dual_v3/data/
|
|-- GER40 1m 01_01_2023-04_11_2025/       # Original GER40 M1 data
|   |-- PEPPERSTONE_GER40, 1.csv          # 37 CSV files total
|   |-- PEPPERSTONE_GER40, 1 (1).csv
|   |-- ...
|   |-- PEPPERSTONE_GER40, 1 (36).csv
|
|-- XAUUSD 1m 01_01_2023-04_11_2025/      # Original XAUUSD M1 data
|   |-- OANDA_XAUUSD, 1.csv              # 48 CSV files total
|   |-- OANDA_XAUUSD, 1 (1).csv
|   |-- ...
|   |-- OANDA_XAUUSD, 1 (47).csv
|
|-- ger40+pepperstone_0411-2001/           # Extended GER40 data (control period)
|   |-- PEPPERSTONE_GER40, 1.csv          # 3 CSV files + 1 merged file
|   |-- PEPPERSTONE_GER40, 1 (1).csv
|   |-- PEPPERSTONE_GER40, 1 (2).csv
|   |-- PEPPERSTONE_GER40, 111.csv
|
|-- xauusd_oanda_0411-2001/                # Extended XAUUSD data (control period)
|   |-- OANDA_XAUUSD, 1.csv              # 5 CSV files
|   |-- OANDA_XAUUSD, 1 (1).csv
|   |-- ...
|   |-- OANDA_XAUUSD, 1 (4).csv
|
|-- optimized/                             # Parquet files for fast backtest
|   |-- GER40_m1.parquet
|   |-- XAUUSD_m1.parquet
|
|-- control/                               # Parquet files for control period
|   |-- GER40_m1.parquet
|   |-- XAUUSD_m1.parquet
|
|-- news/                                  # ForexFactory economic calendar
    |-- forex_factory_2023.json
    |-- forex_factory_2024.json
    |-- forex_factory_2025.json
    |-- forex_factory_2026.json
```

---

## Original Data (In-Sample)

### GER40 (DAX Index)

- **Source**: Pepperstone via TradingView export
- **Path**: `data/GER40 1m 01_01_2023-04_11_2025/`
- **Files**: 37 CSV files (`PEPPERSTONE_GER40, 1.csv` through `PEPPERSTONE_GER40, 1 (36).csv`)
- **Date range**: 2023-01-01 to 2025-04-11
- **Timeframe**: M1 (1-minute candles)
- **Usage**: In-sample optimization period for parameter grid search

### XAUUSD (Gold)

- **Source**: Oanda via TradingView export
- **Path**: `data/XAUUSD 1m 01_01_2023-04_11_2025/`
- **Files**: 48 CSV files (`OANDA_XAUUSD, 1.csv` through `OANDA_XAUUSD, 1 (47).csv`)
- **Date range**: 2023-01-01 to 2025-04-11
- **Timeframe**: M1 (1-minute candles)
- **Usage**: In-sample optimization period for parameter grid search

### How Data Was Obtained

1. Open TradingView with the Pepperstone (GER40) or Oanda (XAUUSD) data feed.
2. Set timeframe to 1 minute.
3. Use TradingView's "Export chart data" feature.
4. TradingView splits large exports into multiple CSV files due to row limits.
5. Files are placed in the respective directories without modification.

---

## Extended Data (Control Period)

### GER40 Extended

- **Source**: Pepperstone via TradingView export
- **Path**: `data/ger40+pepperstone_0411-2001/`
- **Files**: 3 CSV files + 1 merged file
- **Date range**: 2025-04-11 (approximately, continuing from original) to 2026-01-20
- **Usage**: Out-of-sample validation (control period)

### XAUUSD Extended

- **Source**: Oanda via TradingView export
- **Path**: `data/xauusd_oanda_0411-2001/`
- **Files**: 5 CSV files
- **Date range**: 2025-04-11 (approximately, continuing from original) to 2026-01-20
- **Usage**: Out-of-sample validation (control period)

### Purpose of Control Period

The control period (approximately November 2025 through January 2026) serves as
out-of-sample data for validating optimized parameter sets. This prevents
overfitting: parameters are optimized on in-sample data (2023-01-01 to
2025-10-31), then validated on control data that was not used during
optimization.

---

## Optimized Data (Parquet)

### Location

```
data/optimized/
    GER40_m1.parquet
    XAUUSD_m1.parquet
```

### Purpose

The raw CSV files (37+ files for GER40, 48+ for XAUUSD) are slow to load
individually. The optimized Parquet files are pre-processed, merged, deduplicated,
and stored in Apache Parquet format with Snappy compression for fast reads.

### How They Are Generated

The `DataIngestor` class (`backtest/data_processor/data_ingestor.py`) handles
conversion:

1. Recursively finds all CSV files in the source directory.
2. Validates required columns (`time`, `open`, `high`, `low`, `close`).
3. Parses datetime to UTC timezone-aware format.
4. Validates OHLC relationships (high >= low, high >= open, etc.).
5. Removes duplicate timestamps (keeps last occurrence).
6. Sorts by time and exports to Parquet.

### Path Configuration

In `params_optimizer/config.py`:

```python
DATA_BASE_PATH = Path(__file__).parent.parent / "data"

DATA_PATHS_RAW = {
    "GER40": DATA_BASE_PATH / "GER40 1m 01_01_2023-04_11_2025",
    "XAUUSD": DATA_BASE_PATH / "XAUUSD 1m 01_01_2023-04_11_2025",
}

DATA_PATHS_OPTIMIZED = {
    "GER40": DATA_BASE_PATH / "optimized" / "GER40_m1.parquet",
    "XAUUSD": DATA_BASE_PATH / "optimized" / "XAUUSD_m1.parquet",
}
```

In `backtest/config.py`:

```python
DATA_FOLDERS = {
    "GER40": "GER40 1m 01_01_2023-04_11_2025",
    "XAUUSD": "XAUUSD 1m 01_01_2023-04_11_2025",
    "GER40_CONTROL": "ger40+pepperstone_0411-2001",
    "XAUUSD_CONTROL": "xauusd_oanda_0411-2001",
}
```

---

## Control Period Data

### Location

```
data/control/
    GER40_m1.parquet
    XAUUSD_m1.parquet
```

### Preparation Script

**File**: `backtest/prepare_control_data.py`

This script reads the extended CSV data, normalizes timezones, filters trading
hours, and exports to Parquet:

```bash
cd dual_v3
python backtest/prepare_control_data.py
```

**Processing steps**:

1. Loads all CSVs from the control data directory.
2. Parses timestamps to UTC-aware datetime.
3. Removes duplicates by timestamp (keeps first occurrence).
4. Drops rows with NaN in OHLC columns.
5. Applies instrument-specific filters:
   - **GER40**: Filters to 07:00-23:00 Europe/Berlin (trading hours only).
   - **XAUUSD**: Filters to weekdays only (24/5 market).
6. Backs up existing Parquet files to `_trash/` before overwriting.
7. Saves as Parquet with Snappy compression.

### Date Range

| Instrument | Control Period Start | Control Period End |
|------------|---------------------|-------------------|
| GER40      | 2025-11-04          | 2026-01-20        |
| XAUUSD     | 2025-11-04          | 2026-01-20        |

The control period backtest runner (`run_parallel_backtest_control.py`) defaults
to `--start-date 2025-11-04 --end-date 2026-01-20`.

---

## News Data

### Location

```
data/news/
    forex_factory_2023.json
    forex_factory_2024.json
    forex_factory_2025.json
    forex_factory_2026.json
```

### Content

Each JSON file contains economic calendar events from ForexFactory for one year.
See [NEWS_FILTER.md](NEWS_FILTER.md) for full documentation on the data format,
storage schema, and how events are fetched.

### Generation

```bash
# Generate programmatic events + fetch current week
python scripts/load_news_data.py

# Scrape real historical data from ForexFactory website
python scripts/scrape_forexfactory.py --start-year 2023 --end-year 2026
```

---

## CSV Format Specification

### Standard Format (TradingView Export)

All CSV files follow the TradingView export format:

| Column       | Type     | Description | Example |
|-------------|----------|-------------|---------|
| `time`      | string   | ISO 8601 datetime with timezone | `2025-10-06T00:15:00Z` or `2025-10-06T09:15:00+09:00` |
| `open`      | float    | Opening price of the candle | `20150.3` |
| `high`      | float    | Highest price during the candle | `20155.7` |
| `low`       | float    | Lowest price during the candle | `20148.1` |
| `close`     | float    | Closing price of the candle | `20153.2` |
| `volume`    | int/float | Tick volume (may be named `tick_volume`) | `142` |

### Notes on the CSV Data

- The `time` column may use `Z` suffix (UTC) or timezone offset (`+09:00` for JST, `+01:00` for CET).
- Volume is tick volume (number of price changes), not real volume.
- Some files may not include a volume column; the ingestion pipeline assigns a default value of 0 or 1.
- Column names are case-insensitive in the ingestion pipeline (normalized to lowercase).
- OHLC values must satisfy: `low <= open,close <= high`. Rows violating this are discarded during cleaning.

### Example CSV Content

```csv
time,open,high,low,close,volume
2025-10-06T00:15:00Z,20150.3,20155.7,20148.1,20153.2,142
2025-10-06T00:16:00Z,20153.2,20154.8,20151.0,20152.5,98
```

---

## Data Ingestion Pipeline

### DataIngestor Class

**Location**: `backtest/data_processor/data_ingestor.py`

The `DataIngestor` class is responsible for:

1. **Discovery**: Recursively finds all `*.csv` files in a given directory.
2. **Validation**: Checks for required columns (`time`, `open`, `high`, `low`, `close`).
3. **Parsing**: Converts `time` column to timezone-aware UTC datetime using `pd.to_datetime(..., utc=True)`.
4. **Cleaning**: Removes NaN rows, validates OHLC relationships, adds volume column if missing.
5. **Deduplication**: Removes duplicate timestamps (keeps last occurrence).
6. **Merging**: Concatenates all DataFrames and sorts by time.

```python
from backtest.data_processor.data_ingestor import DataIngestor

ingestor = DataIngestor("data/GER40 1m 01_01_2023-04_11_2025")
df = ingestor.load_all(symbol="GER40")
# df.columns: ['time', 'open', 'high', 'low', 'close', 'volume']
# df['time'] is timezone-aware UTC
```

### Convenience Function

```python
from backtest.data_processor.data_ingestor import load_symbol_data

df = load_symbol_data(
    symbol="GER40",
    data_path="data/GER40 1m 01_01_2023-04_11_2025",
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 12, 31),
)
```

---

## Timezone Considerations

Timezone handling is a critical aspect of this project due to the involvement of
multiple time zones across data sources, instruments, and the MT5 server.

### Summary Table

| Context             | Timezone                    | UTC Offset | Notes |
|--------------------|-----------------------------|-----------|-------|
| ForexFactory data   | America/New_York (ET)       | -5 (EST) / -4 (EDT) | DST transitions in March/November |
| GER40 trading hours | Europe/Berlin (CET/CEST)    | +1 (CET) / +2 (CEST) | Trading 07:00-23:00 local |
| XAUUSD trading hours| Asia/Tokyo (JST)            | +9 (fixed, no DST) | 24/5, weekdays only |
| MT5 server          | Asia/Jerusalem              | +2 (IST) / +3 (IDT) | The5ers MT5 server timezone |
| Internal storage    | UTC                         | 0         | All datetimes stored in UTC |
| Bot/User            | Asia/Almaty                 | +5 (fixed, no DST) | Development machine timezone |

### Key Rules

1. **All internal timestamps are stored in UTC.** This includes Parquet files, JSON news data, and in-memory DataFrames.
2. **ForexFactory times are published in Eastern Time.** The `timezone_utils.py` module converts ET to UTC during parsing, handling EST/EDT automatically.
3. **Trading hour filters are applied in instrument-local timezone.** GER40 data is filtered to 07:00-23:00 Europe/Berlin. XAUUSD data is filtered to weekdays only.
4. **CSV files from TradingView may contain timezone offsets.** The ingestion pipeline handles both `Z` (UTC) and explicit offsets (`+09:00`, `+01:00`).

### Timezone Conversion Utilities

Located in `src/news_filter/timezone_utils.py`:

```python
from src.news_filter.timezone_utils import (
    et_to_utc,          # Eastern Time -> UTC
    utc_to_et,          # UTC -> Eastern Time
    utc_to_instrument_tz,  # UTC -> instrument's local timezone
    get_instrument_timezone,  # Symbol -> pytz timezone object
)
```

---

## Configuration References

### backtest/config.py

- `data_base_path`: `C:/Trading/ib_trading_bot/dual_v3/data` (default)
- `DATA_FOLDERS`: Maps symbol names to directory names for raw CSV data
- Symbol-specific configuration: spread, digits, tick size, contract size, timezone

### params_optimizer/config.py

- `DATA_BASE_PATH`: Relative to `params_optimizer/` parent directory
- `DATA_PATHS_RAW`: Maps symbols to raw CSV directories
- `DATA_PATHS_OPTIMIZED`: Maps symbols to Parquet file paths
- Trading hours filter configuration per instrument
