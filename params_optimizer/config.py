"""
Configuration and Parameter Grids for Parameter Optimizer.

This module contains all configurable parameters for grid search optimization.
Configuration can be loaded from optimizer_config.json file.
"""

import os
import json
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path

# ================================
# BLAS THREAD LIMITS (for multiprocessing)
# ================================
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


# ================================
# SYMBOL CONFIGURATION
# ================================
@dataclass
class SymbolConfig:
    """Configuration for a trading symbol."""
    name: str
    spread_points: float
    digits: int
    volume_step: float = 0.01
    volume_min: float = 0.01
    volume_max: float = 100.0
    trade_tick_size: float = 0.01
    trade_tick_value: float = 1.0
    trade_contract_size: float = 1.0
    timezone: str = "UTC"

    @property
    def point(self) -> float:
        """Minimum price change."""
        return 10 ** (-self.digits)


SYMBOL_CONFIGS: Dict[str, SymbolConfig] = {
    "GER40": SymbolConfig(
        name="GER40",
        spread_points=1.5,
        digits=1,
        volume_step=0.1,
        volume_min=0.1,
        volume_max=50.0,
        trade_tick_size=0.1,
        trade_tick_value=0.1,
        trade_contract_size=1.0,
        timezone="Europe/Berlin",
    ),
    "XAUUSD": SymbolConfig(
        name="XAUUSD",
        spread_points=0.30,
        digits=2,
        volume_step=0.01,
        volume_min=0.01,
        volume_max=100.0,
        trade_tick_size=0.01,
        trade_tick_value=1.0,
        trade_contract_size=100.0,
        timezone="Asia/Tokyo",
    ),
    "NAS100": SymbolConfig(
        name="NAS100",
        spread_points=1.5,
        digits=1,
        volume_step=0.1,
        volume_min=0.1,
        volume_max=50.0,
        trade_tick_size=0.1,
        trade_tick_value=0.1,
        trade_contract_size=1.0,
        timezone="US/Eastern",
    ),
    "UK100": SymbolConfig(
        name="UK100",
        spread_points=1.0,
        digits=1,
        volume_step=0.1,
        volume_min=0.1,
        volume_max=50.0,
        trade_tick_size=0.1,
        trade_tick_value=0.1,
        trade_contract_size=1.0,
        timezone="Europe/London",
    ),
}


# ================================
# DATA PATHS
# ================================
# Base path for data
DATA_BASE_PATH = Path(__file__).parent.parent / "data"

# Raw CSV data paths
DATA_PATHS_RAW = {
    "GER40": DATA_BASE_PATH / "GER40 1m 01_01_2023-04_11_2025",
    "XAUUSD": DATA_BASE_PATH / "XAUUSD 1m 01_01_2023-04_11_2025",
    "NAS100": DATA_BASE_PATH / "NAS100_2023-2026_forexcom",
    "UK100": DATA_BASE_PATH / "UK100_2023-2026_forexcom",
}

# Optimized Parquet data paths (created by prepare_data.py)
DATA_PATHS_OPTIMIZED = {
    "GER40": DATA_BASE_PATH / "optimized" / "GER40_m1.parquet",
    "XAUUSD": DATA_BASE_PATH / "optimized" / "XAUUSD_m1.parquet",
    "NAS100": DATA_BASE_PATH / "optimized" / "NAS100_m1.parquet",
    "UK100": DATA_BASE_PATH / "optimized" / "UK100_m1.parquet",
}

# Default output directory
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "output"

# JSON config file path
CONFIG_FILE = Path(__file__).parent / "optimizer_config.json"


def load_json_config() -> Dict[str, Any]:
    """Load configuration from JSON file if it exists."""
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    return {}


# Load JSON config once at module load
_JSON_CONFIG = load_json_config()


# ================================
# OPTIMIZER CONFIGURATION
# ================================
def _get_json_default(section: str, key: str, default: Any) -> Any:
    """Get value from JSON config with fallback to default."""
    return _JSON_CONFIG.get(section, {}).get(key, default)


@dataclass
class OptimizerConfig:
    """
    Configuration for parameter optimization run.

    Values are loaded from optimizer_config.json if present,
    with CLI arguments taking precedence.
    """
    symbol: str
    data_path: Optional[Path] = None
    output_dir: Path = DEFAULT_OUTPUT_DIR

    # Parallelization (from JSON: parallelization.num_workers, parallelization.combinations_per_chunk)
    num_workers: int = field(default_factory=lambda: _get_json_default("parallelization", "num_workers", None) or 90)
    combinations_per_chunk: int = field(default_factory=lambda: _get_json_default("parallelization", "combinations_per_chunk", 1))

    # Checkpoint (from JSON: checkpoint.interval, checkpoint.file)
    checkpoint_interval: int = field(default_factory=lambda: _get_json_default("checkpoint", "interval", 100))
    checkpoint_file: str = field(default_factory=lambda: _get_json_default("checkpoint", "file", "checkpoint.json"))

    # Ranking weights (from JSON: ranking_weights.*)
    weight_total_r: float = field(default_factory=lambda: _get_json_default("ranking_weights", "total_r", 0.40))
    weight_sharpe: float = field(default_factory=lambda: _get_json_default("ranking_weights", "sharpe_ratio", 0.35))
    weight_winrate: float = field(default_factory=lambda: _get_json_default("ranking_weights", "winrate", 0.25))

    # Backtest settings (from JSON: backtest.*)
    initial_balance: float = field(default_factory=lambda: _get_json_default("backtest", "initial_balance", 100000.0))
    risk_pct: float = field(default_factory=lambda: _get_json_default("backtest", "risk_pct", 1.0))
    max_margin_pct: float = field(default_factory=lambda: _get_json_default("backtest", "max_margin_pct", 40.0))

    # Grid mode (standard / features / btib)
    grid_mode: str = "standard"

    # Max combinations to test (None = all)
    max_combos: Optional[int] = None

    # News filter - 5ers compliance (from JSON: news_filter.*)
    news_filter_enabled: bool = field(default_factory=lambda: _get_json_default("news_filter", "enabled", True))
    news_before_minutes: int = field(default_factory=lambda: _get_json_default("news_filter", "before_minutes", 2))
    news_after_minutes: int = field(default_factory=lambda: _get_json_default("news_filter", "after_minutes", 2))

    def __post_init__(self):
        """Set default data path if not provided."""
        if self.data_path is None:
            self.data_path = DATA_PATHS_OPTIMIZED.get(self.symbol)

        # Ensure output directory exists
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)


# ================================
# UTILITY FUNCTIONS
# ================================
def frange(start: float, stop: float, step: float, inclusive: bool = True, ndigits: int = 6) -> List[float]:
    """Generate a list of floats with given step."""
    if step == 0:
        return [round(start, ndigits)]
    vals = []
    x = start
    comp = (lambda a, b: a <= b + 1e-9) if step > 0 else (lambda a, b: a >= b - 1e-9)
    while comp(x, stop) if inclusive else comp(x, stop - step):
        vals.append(round(x, ndigits))
        x += step
    return vals


# ================================
# PARAMETER GRIDS (Per-Asset)
# ================================
#
# Each asset has its own grid because MIN_SL_PCT, IB_TIME_CONFIGS,
# and sweep ranges differ per symbol.
#
# Grid modes:
#   "standard"  - Core IB/RR/TSL/STOP sweep per asset
#   "features"  - Feature switch sweep, PROD defaults for rest
#   "btib"      - BTIB parameter sweep, PROD defaults for rest


# IB Time Configurations: (start, end, timezone)
IB_TIME_CONFIGS = {
    "GER40": [
        ("08:00", "08:30", "Europe/Berlin"),
        ("08:00", "09:00", "Europe/Berlin"),
        ("09:00", "09:30", "Europe/Berlin"),
    ],
    "XAUUSD": [
        ("09:00", "09:30", "Asia/Tokyo"),
        ("09:00", "10:00", "Asia/Tokyo"),
        ("10:00", "10:30", "Asia/Tokyo"),
    ],
    "NAS100": [
        ("08:00", "08:30", "Europe/Berlin"),
        ("08:00", "09:00", "Europe/Berlin"),
        ("09:00", "09:30", "Europe/Berlin"),
    ],
    "UK100": [
        ("08:00", "08:30", "Europe/Berlin"),
        ("08:00", "09:00", "Europe/Berlin"),
        ("09:00", "09:30", "Europe/Berlin"),
    ],
}

# ----------------
# PER-ASSET STANDARD GRIDS
# ----------------
GER40_GRID = {
    "IB_WAIT_MINUTES": [0, 15],
    "TRADE_WINDOW_MINUTES": [60, 90, 120],
    "RR_TARGET": [0.5, 1.0, 1.5, 2.0],
    "STOP_MODE": ["ib_start", "eq"],
    "TSL_TARGET": [0, 0.5, 1.0, 1.5],
    "TSL_SL": [0.5, 1.0],
    "MIN_SL_PCT": [0.0015],
    "IB_BUFFER_PCT": [0.05, 0.10, 0.15, 0.20, 0.25],
    "MAX_DISTANCE_PCT": [0.5, 0.75, 1.0, 1.25],
    "ANALYSIS_TF": ["2min"],
    "FRACTAL_BE_ENABLED": [True, False],
    "FRACTAL_TSL_ENABLED": [True, False],
    "FVG_BE_ENABLED": [True, False],
    "REV_RB_ENABLED": [True, False],
    "BTIB_ENABLED": [True, False],
    "BTIB_SL_MODE": ["fractal_2m", "cisd"],
    "BTIB_CORE_CUTOFF_MIN": [40],
    "BTIB_EXTENSION_PCT": [1.0],
    "BTIB_RR_TARGET": [1.0],
    "BTIB_TSL_TARGET": [0.0],
    "BTIB_TSL_SL": [0.0],
}

XAUUSD_GRID = {
    "IB_WAIT_MINUTES": [0, 15, 20],
    "TRADE_WINDOW_MINUTES": [90, 120, 180, 240],
    "RR_TARGET": [0.5, 0.75, 1.0, 1.25, 1.5],
    "STOP_MODE": ["ib_start", "eq"],
    "TSL_TARGET": [0, 0.5, 1.0, 1.5],
    "TSL_SL": [0.5, 1.0, 1.5],
    "MIN_SL_PCT": [0.001],
    "IB_BUFFER_PCT": [0.0, 0.025, 0.05, 0.075, 0.10],
    "MAX_DISTANCE_PCT": [0.5, 0.75, 1.0],
    "ANALYSIS_TF": ["2min"],
    "FRACTAL_BE_ENABLED": [True, False],
    "FRACTAL_TSL_ENABLED": [True, False],
    "FVG_BE_ENABLED": [True, False],
    "REV_RB_ENABLED": [True, False],
    "BTIB_ENABLED": [True, False],
    "BTIB_SL_MODE": ["fractal_2m", "cisd"],
    "BTIB_CORE_CUTOFF_MIN": [40],
    "BTIB_EXTENSION_PCT": [1.0],
    "BTIB_RR_TARGET": [1.0],
    "BTIB_TSL_TARGET": [0.0],
    "BTIB_TSL_SL": [0.0],
}

NAS100_GRID = {
    "IB_WAIT_MINUTES": [0, 15],
    "TRADE_WINDOW_MINUTES": [60, 90, 120],
    "RR_TARGET": [0.5, 1.0, 1.5, 2.0],
    "STOP_MODE": ["ib_start", "eq"],
    "TSL_TARGET": [0, 0.5, 1.0, 1.5],
    "TSL_SL": [0.5, 1.0],
    "MIN_SL_PCT": [0.0015],
    "IB_BUFFER_PCT": [0.05, 0.10, 0.15, 0.20, 0.25],
    "MAX_DISTANCE_PCT": [0.5, 0.75, 1.0, 1.25],
    "ANALYSIS_TF": ["2min"],
    "FRACTAL_BE_ENABLED": [True, False],
    "FRACTAL_TSL_ENABLED": [True, False],
    "FVG_BE_ENABLED": [True, False],
    "REV_RB_ENABLED": [True, False],
    "BTIB_ENABLED": [True, False],
    "BTIB_SL_MODE": ["fractal_2m", "cisd"],
    "BTIB_CORE_CUTOFF_MIN": [40],
    "BTIB_EXTENSION_PCT": [1.0],
    "BTIB_RR_TARGET": [1.0],
    "BTIB_TSL_TARGET": [0.0],
    "BTIB_TSL_SL": [0.0],
}

UK100_GRID = {
    "IB_WAIT_MINUTES": [0, 15],
    "TRADE_WINDOW_MINUTES": [60, 90, 120],
    "RR_TARGET": [0.5, 1.0, 1.5, 2.0],
    "STOP_MODE": ["ib_start", "eq"],
    "TSL_TARGET": [0, 0.5, 1.0, 1.5],
    "TSL_SL": [0.5, 1.0],
    "MIN_SL_PCT": [0.0015],
    "IB_BUFFER_PCT": [0.05, 0.10, 0.15, 0.20, 0.25],
    "MAX_DISTANCE_PCT": [0.5, 0.75, 1.0, 1.25],
    "ANALYSIS_TF": ["2min"],
    "FRACTAL_BE_ENABLED": [True, False],
    "FRACTAL_TSL_ENABLED": [True, False],
    "FVG_BE_ENABLED": [True, False],
    "REV_RB_ENABLED": [True, False],
    "BTIB_ENABLED": [True, False],
    "BTIB_SL_MODE": ["fractal_2m", "cisd"],
    "BTIB_CORE_CUTOFF_MIN": [40],
    "BTIB_EXTENSION_PCT": [1.0],
    "BTIB_RR_TARGET": [1.0],
    "BTIB_TSL_TARGET": [0.0],
    "BTIB_TSL_SL": [0.0],
}

# Per-asset standard grid map
_STANDARD_GRIDS: Dict[str, Dict[str, List]] = {
    "GER40": GER40_GRID,
    "XAUUSD": XAUUSD_GRID,
    "NAS100": NAS100_GRID,
    "UK100": UK100_GRID,
}


# ----------------
# FEATURES SWEEP GRID
# Fix IB/RR/TSL at PROD values, sweep only feature switches.
# ~48 combos per symbol (2 * 2 * 2 * 2 * 3 IB configs)
# ----------------
_FEATURES_GRID = {
    "ANALYSIS_TF": ["2min", "5min"],
    "FRACTAL_BE_ENABLED": [True, False],
    "FRACTAL_TSL_ENABLED": [True, False],
    "FVG_BE_ENABLED": [True, False],
}


# ----------------
# BTIB SWEEP GRID
# Fix core params at PROD values, sweep only BTIB params.
# ~2.5K combos per symbol (after smart filtering)
# ----------------
_BTIB_GRID = {
    "BTIB_ENABLED": [True],
    "BTIB_SL_MODE": ["fractal_2m", "cisd"],
    "BTIB_CORE_CUTOFF_MIN": [30, 40, 50, 60],
    "BTIB_EXTENSION_PCT": [0.5, 1.0, 1.5],
    "BTIB_RR_TARGET": [0.5, 1.0, 1.5],
    "BTIB_TSL_TARGET": [0, 0.5, 1.0],
    "BTIB_TSL_SL": [0, 0.5, 1.0],
}


def _get_prod_defaults(symbol: str) -> Dict[str, List]:
    """
    Get PROD defaults for a symbol (single-value lists).

    Used by features/btib modes to fix non-swept params at PROD values.
    Values sourced from strategy_logic.py *_PARAMS_PROD.
    """
    defaults = {
        "IB_WAIT_MINUTES": [0],
        "TRADE_WINDOW_MINUTES": [120],
        "RR_TARGET": [1.0],
        "STOP_MODE": ["ib_start"],
        "TSL_TARGET": [1.0],
        "TSL_SL": [0.5],
        "MIN_SL_PCT": [0.0015],
        "IB_BUFFER_PCT": [0.10],
        "MAX_DISTANCE_PCT": [0.5],
        "ANALYSIS_TF": ["2min"],
        "FRACTAL_BE_ENABLED": [True],
        "FRACTAL_TSL_ENABLED": [True],
        "FVG_BE_ENABLED": [False],
        "REV_RB_ENABLED": [False],
        "BTIB_ENABLED": [False],
        "BTIB_SL_MODE": ["fractal_2m"],
        "BTIB_CORE_CUTOFF_MIN": [40],
        "BTIB_EXTENSION_PCT": [1.0],
        "BTIB_RR_TARGET": [1.0],
        "BTIB_TSL_TARGET": [0.0],
        "BTIB_TSL_SL": [0.0],
    }

    # Per-symbol overrides
    overrides = {
        "XAUUSD": {
            "IB_WAIT_MINUTES": [20],
            "TRADE_WINDOW_MINUTES": [240],
            "MIN_SL_PCT": [0.001],
            "IB_BUFFER_PCT": [0.05],
            "BTIB_ENABLED": [True],
        },
    }

    if symbol in overrides:
        defaults.update(overrides[symbol])

    return defaults


def get_grid_for_symbol(symbol: str, mode: str = "standard") -> Dict[str, List]:
    """
    Get parameter grid for a symbol and mode.

    Args:
        symbol: Trading symbol (GER40, XAUUSD, NAS100, UK100)
        mode: Grid mode - "standard", "features", or "btib"

    Returns:
        Dict mapping parameter names (UPPERCASE) to lists of values to sweep
    """
    if mode == "standard":
        if symbol not in _STANDARD_GRIDS:
            raise ValueError(f"Unknown symbol: {symbol}. Supported: {list(_STANDARD_GRIDS.keys())}")
        return _STANDARD_GRIDS[symbol].copy()

    elif mode == "features":
        grid = _get_prod_defaults(symbol)
        grid.update(_FEATURES_GRID)
        return grid

    elif mode == "btib":
        grid = _get_prod_defaults(symbol)
        grid.update(_BTIB_GRID)
        return grid

    else:
        raise ValueError(f"Unknown mode: {mode}. Supported: standard, features, btib")


# ================================
# RANKING WEIGHTS
# ================================
RANKING_WEIGHTS: Dict[str, float] = {
    "Total_R": 0.40,       # Primary: total profit in R
    "Sharpe_Ratio": 0.35,  # Risk-adjusted return
    "Winrate_pct": 0.25,   # Stability
}


# ================================
# TRADING HOURS FILTER
# ================================
# Hours to keep when filtering data (in local timezone)
TRADING_HOURS = {
    "GER40": {
        "start_hour": 7,   # 07:00 local
        "end_hour": 23,    # 23:00 local
        "timezone": "Europe/Berlin",
    },
    "XAUUSD": {
        # XAUUSD trades 24/5, filter weekends only
        "filter_weekends": True,
        "timezone": "Asia/Tokyo",
    },
    "NAS100": {
        # NAS100 trades extended hours, filter weekends only
        "filter_weekends": True,
        "timezone": "US/Eastern",
    },
    "UK100": {
        "start_hour": 7,   # 07:00 local
        "end_hour": 23,    # 23:00 local
        "timezone": "Europe/London",
    },
}


# ================================
# CONSOLE OUTPUT HELPERS
# ================================
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_status(message: str, status: str = "INFO") -> None:
    """Print status message with color."""
    from datetime import datetime
    colors = {
        "INFO": Colors.OKBLUE,
        "SUCCESS": Colors.OKGREEN,
        "WARNING": Colors.WARNING,
        "ERROR": Colors.FAIL,
        "HEADER": Colors.HEADER,
        "PROGRESS": Colors.OKCYAN,
    }
    color = colors.get(status, Colors.ENDC)
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"{color}[{timestamp}] {status}: {message}{Colors.ENDC}")


def print_progress_bar(current: int, total: int, prefix: str = '', suffix: str = '', length: int = 50) -> None:
    """Display progress bar."""
    percent = current / total if total > 0 else 0
    filled = int(length * percent)
    bar = '#' * filled + '-' * (length - filled)
    print(f'\r{Colors.OKCYAN}{prefix} |{bar}| {percent:.1%} {suffix}{Colors.ENDC}', end='', flush=True)
    if current == total:
        print()


def estimate_combinations(symbol: str = "GER40", mode: str = "standard") -> int:
    """
    Estimate total number of parameter combinations for a symbol.

    This is an approximation. Actual count from ParameterGrid.generate_all()
    may differ due to smart filtering (TSL_SL <= TSL_TARGET, TSL_TARGET <= RR_TARGET+1, BTIB sub-params).
    """
    grid = get_grid_for_symbol(symbol, mode)
    ib_configs = len(IB_TIME_CONFIGS.get(symbol, []))

    # Naive product of all grid dimensions (upper bound)
    total = ib_configs
    for key, values in grid.items():
        total *= len(values)

    return total


if __name__ == "__main__":
    print("Parameter Optimizer Configuration")
    print("=" * 60)

    for mode in ["standard", "features", "btib"]:
        print(f"\nMode: {mode}")
        print("-" * 40)
        for sym in SYMBOL_CONFIGS:
            grid = get_grid_for_symbol(sym, mode)
            est = estimate_combinations(sym, mode)
            swept = sum(1 for v in grid.values() if len(v) > 1)
            print(f"  {sym}: ~{est:,} combos ({swept} params swept)")

    print("\n\nPer-asset grid details (standard mode):")
    print("=" * 60)
    for sym in SYMBOL_CONFIGS:
        grid = get_grid_for_symbol(sym, "standard")
        print(f"\n{sym}:")
        for key, values in grid.items():
            if len(values) > 1:
                print(f"  {key}: {values} ({len(values)} values)")
            else:
                print(f"  {key}: {values[0]} (fixed)")

        print(f"\n  IB Time Configs:")
        for start, end, tz in IB_TIME_CONFIGS[sym]:
            print(f"    {start}-{end} {tz}")
