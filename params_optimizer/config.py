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
}

# Optimized Parquet data paths (created by prepare_data.py)
DATA_PATHS_OPTIMIZED = {
    "GER40": DATA_BASE_PATH / "optimized" / "GER40_m1.parquet",
    "XAUUSD": DATA_BASE_PATH / "optimized" / "XAUUSD_m1.parquet",
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
    combinations_per_chunk: int = field(default_factory=lambda: _get_json_default("parallelization", "combinations_per_chunk", 10))

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
# PARAMETER GRIDS
# ================================
#
# Grid modes:
#   "standard" - ~65K combinations, good for quick tests
#   "expanded" - ~2.5M combinations, for exhaustive AWS optimization
#
# Change GRID_MODE to switch between them.
#
GRID_MODE = "standard"  # "standard" or "expanded"


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
}

# # ----------------
# # STANDARD GRIDS (~65K combinations)
# # ----------------
# _STANDARD_GRIDS = {
#     "IB_WAIT_MINUTES": [0, 15, 30],
#     "TRADE_WINDOW_MINUTES": [40, 60, 90, 120, 150, 180],
#     "RR_TARGET": frange(0.5, 2.0, 0.25),  # 7 values
#     "STOP_MODE": ["ib_start", "eq"],
#     "TSL_TARGET": [0.0, 0.5, 1.0, 1.5, 2.0],  # 5 values, 0.0 = disabled
#     "TSL_SL": [0.5, 1.0, 1.5],
#     "MIN_SL_PCT": [0.001, 0.0015],
#     "REV_RB_ENABLED": [True, False],
#     "REV_RB_PCT": [0.25, 0.5, 0.75, 1.0],
#     "IB_BUFFER_PCT": [0.0, 0.01, 0.05, 0.075, 0.10, 0.125, 0.15],  # 7 values
#     "MAX_DISTANCE_PCT": [0.50, 0.65, 0.80, 1.0],  # 4 values
# }

# ----------------
# STANDARD GRIDS (~20K combinations, ~2 hours on 8-core PC)
# ----------------
_STANDARD_GRIDS = {
    "IB_WAIT_MINUTES": [0, 15],              # 3 values
    "TRADE_WINDOW_MINUTES": [60, 90, 120],       # 3 values
    "RR_TARGET": frange(0.5, 2.0, 0.5),          # 4 values: [0.5, 1.0, 1.5, 2.0]
    "STOP_MODE": ["ib_start", "eq"],             # 2 values
    "TSL_TARGET": [0, 0.5, 1.0, 1.5],               # 3 values
    "TSL_SL": [0.5, 1.0],                        # 2 values
    "MIN_SL_PCT": [0.001],                       # 1 value
    "REV_RB_ENABLED": [True, False],             # 2 values
    "REV_RB_PCT": [0.5],                         # 1 value
    "IB_BUFFER_PCT": frange(0.05, 0.25, 0.05),   # 5 values: [0.05, 0.10, 0.15, 0.20, 0.25]
    "MAX_DISTANCE_PCT": frange(0.5, 1.0, 0.25), # 4 values: [0.5, 0.75, 1.0, 1.25]
}


# ----------------
# EXPANDED GRIDS (~2.5M combinations)
# For exhaustive AWS optimization with 96 CPUs
# ----------------
_EXPANDED_GRIDS = {
    "IB_WAIT_MINUTES": [0, 10, 15, 20],
    "TRADE_WINDOW_MINUTES": [40, 60, 90, 120, 150, 180, 210, 240],
    "RR_TARGET": frange(0.5, 2.0, 0.25),
    "STOP_MODE": ["ib_start", "eq"],
    "TSL_TARGET": [0.0, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
    "TSL_SL": [0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
    "MIN_SL_PCT": [0.0015],  # 0.0015 (GER40) / 0.001 (XAUUSD)
    "REV_RB_ENABLED": [True, False],
    "REV_RB_PCT": [1.0],
    "IB_BUFFER_PCT": frange(0.0, 0.20, 0.05),
    "MAX_DISTANCE_PCT": frange(0.5, 1.5, 0.25),
}

# Select active grids based on mode
_ACTIVE_GRIDS = _EXPANDED_GRIDS if GRID_MODE == "expanded" else _STANDARD_GRIDS

# Export individual lists for backward compatibility
IB_WAIT_MINUTES_LIST: List[int] = _ACTIVE_GRIDS["IB_WAIT_MINUTES"]
TRADE_WINDOW_MINUTES_LIST: List[int] = _ACTIVE_GRIDS["TRADE_WINDOW_MINUTES"]
RR_TARGET_LIST: List[float] = _ACTIVE_GRIDS["RR_TARGET"]
STOP_MODE_LIST: List[str] = _ACTIVE_GRIDS["STOP_MODE"]
TSL_TARGET_LIST: List[float] = _ACTIVE_GRIDS["TSL_TARGET"]
TSL_SL_LIST: List[float] = _ACTIVE_GRIDS["TSL_SL"]
MIN_SL_PCT_LIST: List[float] = _ACTIVE_GRIDS["MIN_SL_PCT"]
REV_RB_ENABLED_LIST: List[bool] = _ACTIVE_GRIDS["REV_RB_ENABLED"]
REV_RB_PCT_LIST: List[float] = _ACTIVE_GRIDS["REV_RB_PCT"]
IB_BUFFER_PCT_LIST: List[float] = _ACTIVE_GRIDS["IB_BUFFER_PCT"]
MAX_DISTANCE_PCT_LIST: List[float] = _ACTIVE_GRIDS["MAX_DISTANCE_PCT"]


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


def estimate_combinations(symbol: str = "GER40") -> int:
    """Estimate total number of parameter combinations for a symbol."""
    ib_configs = len(IB_TIME_CONFIGS.get(symbol, []))

    # Base combinations (no REV_RB conditional)
    base = (
        ib_configs *
        len(IB_WAIT_MINUTES_LIST) *
        len(TRADE_WINDOW_MINUTES_LIST) *
        len(RR_TARGET_LIST) *
        len(STOP_MODE_LIST) *
        len(TSL_TARGET_LIST) *
        len(TSL_SL_LIST) *
        len(MIN_SL_PCT_LIST) *
        len(IB_BUFFER_PCT_LIST) *
        len(MAX_DISTANCE_PCT_LIST)
    )

    # REV_RB adds combinations when enabled
    # When REV_RB=False: 1 combination (no PCT needed)
    # When REV_RB=True: len(REV_RB_PCT_LIST) combinations
    rev_rb_factor = 1 + len(REV_RB_PCT_LIST)

    return base * rev_rb_factor


if __name__ == "__main__":
    # Test: show configuration
    print("Parameter Optimizer Configuration")
    print("=" * 60)

    print(f"\nGRID MODE: {GRID_MODE}")
    print(f"Estimated combinations (GER40): {estimate_combinations('GER40'):,}")
    print(f"Estimated combinations (XAUUSD): {estimate_combinations('XAUUSD'):,}")

    for sym in ["GER40", "XAUUSD"]:
        print(f"\n{sym} Symbol Config:")
        cfg = SYMBOL_CONFIGS[sym]
        print(f"  Spread: {cfg.spread_points} points")
        print(f"  Digits: {cfg.digits}")
        print(f"  Timezone: {cfg.timezone}")

        print(f"\n{sym} IB Time Configs:")
        for start, end, tz in IB_TIME_CONFIGS[sym]:
            print(f"  {start}-{end} {tz}")

    print("\nParameter Grid Values:")
    print(f"  IB_WAIT: {IB_WAIT_MINUTES_LIST} ({len(IB_WAIT_MINUTES_LIST)} values)")
    print(f"  TRADE_WINDOW: {TRADE_WINDOW_MINUTES_LIST} ({len(TRADE_WINDOW_MINUTES_LIST)} values)")
    print(f"  RR_TARGET: {len(RR_TARGET_LIST)} values ({RR_TARGET_LIST[0]}..{RR_TARGET_LIST[-1]})")
    print(f"  TSL_TARGET: {len(TSL_TARGET_LIST)} values ({TSL_TARGET_LIST[0]}..{TSL_TARGET_LIST[-1]})")
    print(f"  TSL_SL: {len(TSL_SL_LIST)} values")
    print(f"  IB_BUFFER_PCT: {len(IB_BUFFER_PCT_LIST)} values")
    print(f"  MAX_DISTANCE_PCT: {len(MAX_DISTANCE_PCT_LIST)} values")
