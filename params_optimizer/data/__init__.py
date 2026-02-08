"""
Data handling module for Parameter Optimizer.

Provides data loading, preprocessing, and caching functionality.
"""

from .prepare_data import prepare_optimized_data
from .loader import load_data, get_data_info

__all__ = ["prepare_optimized_data", "load_data", "get_data_info"]
