"""
SMC Detectors - pure functions for detecting market structures.

Each detector takes a DataFrame + params, returns list of structures.
No side effects, no state.
"""
from .fractal_detector import detect_fractals, find_unswept_fractals, check_fractal_sweep
from .fvg_detector import detect_fvg, check_fvg_fill, check_fvg_rebalance
