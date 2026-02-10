"""
SMC Detectors - pure functions for detecting market structures.

Each detector takes a DataFrame + params, returns list of structures.
No side effects, no state.
"""
from .fractal_detector import detect_fractals, find_unswept_fractals, check_fractal_sweep
from .fvg_detector import detect_fvg, check_fvg_fill, check_fvg_rebalance
from .cisd_detector import detect_cisd, check_cisd_invalidation
from .market_structure_detector import detect_swing_points, detect_bos_choch
