"""
SMC (Smart Money Concepts) Engine for dual_v4.

Three-layer architecture:
- Detectors: pure functions that scan DataFrames for SMC structures
- Registry: in-memory state tracking for active/invalidated structures
- Engine: orchestrator that ties detectors + registry + decision logic
"""
from .engine import SMCEngine
