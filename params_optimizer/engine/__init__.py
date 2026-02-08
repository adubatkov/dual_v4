"""
Engine module for Parameter Optimizer.

Provides backtest wrapper, parameter grid generation, and metrics calculation.
"""

from .backtest_wrapper import BacktestWrapper
from .parameter_grid import ParameterGrid
from .metrics_calculator import MetricsCalculator

__all__ = ["BacktestWrapper", "ParameterGrid", "MetricsCalculator"]
