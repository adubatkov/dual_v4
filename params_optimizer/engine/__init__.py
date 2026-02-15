"""
Engine module for Parameter Optimizer.

Provides backtest wrapper, parameter grid generation, and metrics calculation.
"""

from .parameter_grid import ParameterGrid
from .metrics_calculator import MetricsCalculator

# Lazy import: BacktestWrapper depends on backtest module (not available on VMs)
try:
    from .backtest_wrapper import BacktestWrapper
except ImportError:
    BacktestWrapper = None

__all__ = ["BacktestWrapper", "ParameterGrid", "MetricsCalculator"]
