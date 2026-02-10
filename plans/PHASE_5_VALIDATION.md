# Phase 5: SMC Validation Rules Implementation

## Goal

Implement validation rules from `notion_export/InValidation.md` to determine when SMC structures (FVGs, blocks, levels) become invalid and should no longer be used for trading decisions.

**CRITICAL SCOPE LIMITATION**: Phase 5 implements ONLY validation rules from InValidation.md. The following are explicitly OUT OF SCOPE for Phase 5:
- Order Blocks (OB) detection
- Breaker Blocks (BB) detection
- Mitigation Blocks (MB) detection
- Fair Value Levels (FVL)
- Liquidity pools detection
- Inducement (IDM) detection

These will be implemented in Phase 6 (Advanced SMC Structures).

**Success Criteria**:
- FVG validation: all 6 variants from InValidation.md implemented
- Block validation: CISD-based validation (using existing CISD detector)
- Instrument validation: BOS and CISD invalidation rules
- Entry validation: Model 1 and Model 2 logic
- TP validation: open zone targeting logic
- SL validation: idea invalidation placement logic
- BE validation: all 7 conditions from InValidation.md
- Parity test: validation decisions match manual analysis on 10 test days

---

## 1. Validation Architecture

### 1.1 ValidationEngine Location

Validation logic will be implemented as methods within `SMCEngine` class (not a separate class). This keeps all SMC decision logic in one place.

```python
# src/smc/engine.py

class SMCEngine:
    """
    High-level orchestrator for SMC analysis.

    Responsibilities:
        - Call detectors on each timeframe update
        - Store results in SMCRegistry
        - Evaluate signals against SMC context
        - Manage confirmation logic
        - VALIDATE structures and entries (Phase 5)
    """

    def __init__(self, registry: SMCRegistry, config: SMCConfig):
        self.registry = registry
        self.config = config
        self.event_log = SMCEventLog()

    # ... existing methods (evaluate_signal, check_confirmation, etc.) ...

    # PHASE 5: Validation methods
    def validate_fvg_status(self, fvg: FVG, current_bar: pd.Series) -> Tuple[str, Optional[str]]:
        """Validate FVG status (6 variants from InValidation.md)."""
        ...

    def validate_entry(self, signal: Signal, smc_context: SMCDayContext) -> Tuple[bool, Optional[str]]:
        """Validate if entry conditions are met (Model 1/2)."""
        ...

    def validate_tp_target(self, tp_price: float, direction: str, smc_context: SMCDayContext) -> bool:
        """Validate if TP target is a valid open zone."""
        ...

    def validate_sl_placement(self, sl_price: float, entry_price: float, direction: str, smc_context: SMCDayContext) -> Tuple[bool, Optional[str]]:
        """Validate if SL is correctly placed behind idea invalidation."""
        ...

    def validate_be_move(self, current_bar: pd.Series, entry_time: datetime, entry_price: float, direction: str, smc_context: SMCDayContext) -> Tuple[bool, Optional[str]]:
        """Validate if moving to breakeven is appropriate."""
        ...
```

### 1.2 Integration Points

**Fast Backtest Integration** (builds on Phase 4):

```python
# In fast_backtest.py _process_day()

# After building SMC context
smc_context = self._build_smc_context(day_df, day_date, smc_config)

# Update FVG statuses based on validation rules
for fvg in smc_context.fvgs_h1 + smc_context.fvgs_m2:
    # Check each bar in trade window to update FVG status
    for bar in df_trade_m1.itertuples():
        new_status, reason = self.smc_engine.validate_fvg_status(fvg, bar)
        if new_status != fvg.status:
            fvg.status = new_status
            fvg.invalidation_reason = reason
            fvg.invalidation_time = bar.time

# After signal detection and SMC filter
signal = self._apply_smc_filter_batch(signal, smc_context, df_trade_m1, params)

# Validate entry
entry_valid, rejection_reason = self.smc_engine.validate_entry(signal, smc_context)
if not entry_valid:
    return None  # Reject trade

# Validate TP target
tp_valid = self.smc_engine.validate_tp_target(signal.tp_price, signal.direction, smc_context)
if not tp_valid:
    # Adjust TP to nearest valid zone
    signal.tp_price = self._find_nearest_valid_tp(signal, smc_context)

# Validate SL placement
sl_valid, sl_warning = self.smc_engine.validate_sl_placement(signal.sl_price, signal.entry_price, signal.direction, smc_context)
if not sl_valid:
    # Log warning but don't reject trade (SL placement is advisory)
    logger.warning(f"SL placement warning: {sl_warning}")
```

---

## 2. FVG Validation (6 Variants)

### 2.1 Overview

FVGs become invalid after ANY of these 6 conditions:

1. **Structural Update**: FVG leads to BOS/CHoCH/cBOS
2. **Inducement Formation**: FVG leads to IDM (OUT OF SCOPE - will implement in Phase 6)
3. **Zone-to-Zone Mitigation**: Price moves from FVG to another zone (90% of time, first zone ignored)
4. **Zone-to-LQ/SMT Mitigation**: After liquidity sweep or SMT, ignore zone below
5. **Fulfill (Full Fill)**: FVG completely filled
6. **Inversion**: FVG changes direction properties

### 2.2 Implementation

```python
# src/smc/engine.py

from typing import Tuple, Optional
import pandas as pd
from src.smc.models import FVG, BOS

def validate_fvg_status(
    self,
    fvg: FVG,
    current_bar: pd.Series,
    recent_bos: List[BOS] = None
) -> Tuple[str, Optional[str]]:
    """
    Validate FVG status according to 6 invalidation rules.

    Args:
        fvg: FVG structure to validate
        current_bar: Current price bar (with columns: time, open, high, low, close)
        recent_bos: List of recent BOS events (for structural update check)

    Returns:
        (new_status, invalidation_reason) tuple
        Statuses: "active", "structural_update", "zone_to_zone", "full_fill", "inverted"

    Note: "inducement" and "zone_to_lq" variants deferred to Phase 6 (require LQ/IDM detection)
    """
    if fvg.status != "active":
        # Already invalidated, keep current status
        return (fvg.status, fvg.invalidation_reason)

    # Variant 1: Structural Update (BOS/CHoCH/cBOS)
    if recent_bos:
        for bos in recent_bos:
            # Check if BOS occurred after FVG formation and price tested FVG first
            if bos.break_time > fvg.formation_time:
                # Check if price tested FVG zone before BOS
                # (This requires checking bars between FVG formation and BOS)
                # Simplified: if BOS is in same direction as FVG, FVG likely contributed
                if (fvg.direction == "bullish" and bos.direction == "bullish") or \
                   (fvg.direction == "bearish" and bos.direction == "bearish"):
                    return ("structural_update", f"Led to {bos.bos_type} at {bos.break_time}")

    # Variant 2: Inducement Formation
    # OUT OF SCOPE for Phase 5 (requires IDM detector)
    # Will implement in Phase 6

    # Variant 3: Zone-to-Zone Mitigation
    # Check if price tested FVG and then moved to another zone
    # Algorithm: if price entered FVG zone and then closed beyond it toward another zone
    bar_in_fvg = (current_bar["low"] <= fvg.high and current_bar["high"] >= fvg.low)

    if bar_in_fvg:
        # Price is testing FVG
        # Check if this is a "pass-through" (price entered and exited without reaction)
        if fvg.direction == "bullish":
            # Bullish FVG: if bar enters and closes above high, it's passing through
            if current_bar["close"] > fvg.high:
                # Check if there's another zone above (deferred: requires zone registry)
                # For now, mark as potential zone-to-zone if price aggressively breaks through
                body_range = abs(current_bar["close"] - current_bar["open"])
                if body_range > (fvg.high - fvg.low) * 1.5:  # Strong candle through FVG
                    return ("zone_to_zone", f"Pass-through at {current_bar['time']}")
        else:  # bearish
            if current_bar["close"] < fvg.low:
                body_range = abs(current_bar["close"] - current_bar["open"])
                if body_range > (fvg.high - fvg.low) * 1.5:
                    return ("zone_to_zone", f"Pass-through at {current_bar['time']}")

    # Variant 4: Zone-to-LQ/SMT Mitigation
    # OUT OF SCOPE for Phase 5 (requires LQ pool detection)
    # Will implement in Phase 6

    # Variant 5: Fulfill (Full Fill)
    # Calculate fill percentage
    fill_pct = self._calculate_fvg_fill(fvg, current_bar)

    if fill_pct >= 1.0:  # 100% filled
        return ("full_fill", f"Completely filled at {current_bar['time']}")

    # Update fill_pct in FVG object (side effect, but acceptable for state tracking)
    fvg.fill_pct = fill_pct

    if fill_pct >= 0.9:  # 90% filled (near-complete)
        # Mark as partial_fill but still active for now
        # Full invalidation only at 100%
        pass

    # Variant 6: Inversion
    # FVG inverts if price closes on opposite side and forms opposing FVG
    if fvg.direction == "bullish":
        # Bullish FVG inverts if price closes below low and forms bearish FVG
        if current_bar["close"] < fvg.low:
            # Check if there's a bearish FVG forming at this level
            # (Requires checking next few bars - simplified check)
            if current_bar["open"] > current_bar["close"]:  # Bearish candle
                wick_range = current_bar["open"] - current_bar["close"]
                if wick_range > (fvg.high - fvg.low) * 0.8:  # Strong bearish move
                    return ("inverted", f"Inverted at {current_bar['time']}")
    else:  # bearish
        if current_bar["close"] > fvg.high:
            if current_bar["close"] > current_bar["open"]:  # Bullish candle
                wick_range = current_bar["close"] - current_bar["open"]
                if wick_range > (fvg.high - fvg.low) * 0.8:
                    return ("inverted", f"Inverted at {current_bar['time']}")

    # No invalidation conditions met
    return ("active", None)


def _calculate_fvg_fill(self, fvg: FVG, current_bar: pd.Series) -> float:
    """
    Calculate how much of FVG has been filled.

    Returns:
        Fill percentage (0.0 to 1.0+)
    """
    gap_size = fvg.high - fvg.low

    if fvg.direction == "bullish":
        # Bullish FVG: fill comes from above (price retraces down into gap)
        # Track lowest price that entered the gap
        if current_bar["low"] <= fvg.high:
            fill_depth = fvg.high - current_bar["low"]
            fill_pct = fill_depth / gap_size
            return min(fill_pct, 1.0)
    else:  # bearish
        # Bearish FVG: fill comes from below (price retraces up into gap)
        if current_bar["high"] >= fvg.low:
            fill_depth = current_bar["high"] - fvg.low
            fill_pct = fill_depth / gap_size
            return min(fill_pct, 1.0)

    return 0.0
```

### 2.3 FVG Status Update in Registry

```python
# src/smc/registry.py

class SMCRegistry:
    """
    Tracks all SMC structures (active and invalidated).
    """

    def __init__(self):
        self.fvgs: Dict[str, FVG] = {}  # ID -> FVG
        self.fractals: Dict[str, Fractal] = {}
        self.bos_events: List[BOS] = []
        # ... other structures

    def update_fvg_status(self, fvg_id: str, new_status: str, reason: Optional[str], time: datetime):
        """Update FVG status and log event."""
        if fvg_id not in self.fvgs:
            return

        fvg = self.fvgs[fvg_id]
        old_status = fvg.status

        fvg.status = new_status
        fvg.invalidation_reason = reason
        fvg.invalidation_time = time

        # Log event
        self.event_log.add_event(
            event_type="FVG_STATUS_CHANGE",
            time=time,
            details={
                "fvg_id": fvg_id,
                "old_status": old_status,
                "new_status": new_status,
                "reason": reason
            }
        )

    def get_active_fvgs(self, instrument: str, timeframe: str, direction: Optional[str] = None) -> List[FVG]:
        """Get all active FVGs for instrument/timeframe."""
        return [
            fvg for fvg in self.fvgs.values()
            if fvg.instrument == instrument
            and fvg.timeframe == timeframe
            and fvg.status == "active"
            and (direction is None or fvg.direction == direction)
        ]
```

---

## 3. Block Validation

### 3.1 Overview

**CRITICAL**: Phase 5 does NOT implement Order Block, Breaker Block, or Mitigation Block DETECTION. These are Phase 6 features.

Phase 5 implements ONLY the validation logic that will be used once blocks are detected:

- **Rejection Block (RB)**: Validated via liquidity sweep + X-1 TF trigger
- **Mitigation Block (MB)**: Validated via MSS (Market Structure Shift)
- **Breaker Block (BB)**: Validated via structure + FVG at level (Unicorn setup)
- **All blocks**: Use CISD as main reaction indicator

Since we don't have block detection yet, this section provides the FUNCTION SIGNATURES and ALGORITHM PSEUDOCODE that will be called in Phase 6.

### 3.2 Implementation (Placeholder for Phase 6)

```python
# src/smc/engine.py

def validate_block(
    self,
    block: 'OrderBlock',  # Will be defined in Phase 6
    current_bar: pd.Series,
    recent_cisd: List[CISD]
) -> Tuple[bool, Optional[str]]:
    """
    Validate if block is valid for entry.

    Block types and validation rules:
        - RB (Rejection Block): Requires liquidity sweep + X-1 TF trigger
        - MB (Mitigation Block): Requires MSS (Market Structure Shift)
        - BB (Breaker Block): Requires structure inversion + FVG at level (best)
        - All: Use CISD as main reaction indicator

    Args:
        block: Order block to validate (RB, MB, or BB)
        current_bar: Current price bar
        recent_cisd: List of recent CISD events near block level

    Returns:
        (is_valid, reason) tuple

    Note: This is a PLACEHOLDER for Phase 6. Block detection not yet implemented.
    """
    # Algorithm Pseudocode:

    # 1. Check for CISD at block level (main validation)
    cisd_at_block = [
        cisd for cisd in recent_cisd
        if abs(cisd.confirmation_price - block.price) / block.price < 0.002  # Within 0.2%
    ]

    if not cisd_at_block:
        return (False, "No CISD confirmation at block level")

    # 2. Block-type specific validation
    if block.block_type == "RB":  # Rejection Block
        # Requires liquidity sweep (check if recent fractal swept)
        # AND X-1 timeframe trigger (check lower TF for entry trigger)
        # (Deferred to Phase 6 - requires LQ detection and multi-TF analysis)
        pass

    elif block.block_type == "MB":  # Mitigation Block
        # Requires MSS (Market Structure Shift)
        # Check if recent BOS qualifies as MSS (major structure break)
        # (Deferred to Phase 6 - requires MSS detection)
        pass

    elif block.block_type == "BB":  # Breaker Block
        # Best validation: FVG at level (Unicorn setup)
        # Check if active FVG exists at block level
        fvg_at_block = [
            fvg for fvg in self.registry.get_active_fvgs(block.instrument, block.timeframe)
            if abs(fvg.midpoint - block.price) / block.price < 0.001
        ]

        if fvg_at_block:
            return (True, "Unicorn setup: BB + FVG at level")
        else:
            # BB valid even without FVG, but less reliable
            return (True, "BB validated via structure inversion (no FVG)")

    # Default: valid if CISD present
    return (True, f"CISD confirmed at {cisd_at_block[0].confirmation_time}")
```

**Implementation Status**: DEFERRED to Phase 6 (requires Order Block detection).

---

## 4. Instrument Validation

### 4.1 BOS Validation

**Rule**: BOS is invalid if there's a zone below/above that can continue movement.

```python
# src/smc/engine.py

def validate_bos(
    self,
    bos: BOS,
    smc_context: SMCDayContext
) -> Tuple[bool, Optional[str]]:
    """
    Validate if BOS is a true structure break.

    Rule: BOS invalid if zone exists below (for bullish BOS) or above (for bearish BOS)
          that could continue movement.

    Args:
        bos: BOS event to validate
        smc_context: Current SMC context (with active FVGs, fractals, etc.)

    Returns:
        (is_valid, reason) tuple
    """
    if bos.direction == "bullish":
        # Bullish BOS: check for zones below that could reject price
        # Zones to check: bearish FVGs, low fractals, support levels

        # Check for active bearish FVGs below BOS level
        opposing_fvgs = [
            fvg for fvg in smc_context.fvgs_h1 + smc_context.fvgs_m2
            if fvg.status == "active"
            and fvg.direction == "bearish"
            and fvg.midpoint < bos.broken_level
            and fvg.midpoint > bos.broken_level * 0.98  # Within 2% below
        ]

        if opposing_fvgs:
            return (False, f"Bearish FVG below BOS at {opposing_fvgs[0].midpoint}")

        # Check for unswept low fractals below BOS level
        opposing_fractals = [
            f for f in smc_context.active_fractal_lows
            if f < bos.broken_level
            and f > bos.broken_level * 0.98
        ]

        if opposing_fractals:
            return (False, f"Unswept low fractal below BOS at {opposing_fractals[0]}")

    else:  # bearish BOS
        # Check for zones above that could reject price
        opposing_fvgs = [
            fvg for fvg in smc_context.fvgs_h1 + smc_context.fvgs_m2
            if fvg.status == "active"
            and fvg.direction == "bullish"
            and fvg.midpoint > bos.broken_level
            and fvg.midpoint < bos.broken_level * 1.02
        ]

        if opposing_fvgs:
            return (False, f"Bullish FVG above BOS at {opposing_fvgs[0].midpoint}")

        opposing_fractals = [
            f for f in smc_context.active_fractal_highs
            if f > bos.broken_level
            and f < bos.broken_level * 1.02
        ]

        if opposing_fractals:
            return (False, f"Unswept high fractal above BOS at {opposing_fractals[0]}")

    # No opposing zones found, BOS is valid
    return (True, None)
```

### 4.2 CISD Validation

**Rule**: CISD invalid if zone below/above can continue movement (same as BOS).

```python
def validate_cisd(
    self,
    cisd: CISD,
    smc_context: SMCDayContext
) -> Tuple[bool, Optional[str]]:
    """
    Validate if CISD is a true state change.

    Rule: CISD invalid if zone exists that could continue movement against CISD direction.

    Args:
        cisd: CISD event to validate
        smc_context: Current SMC context

    Returns:
        (is_valid, reason) tuple
    """
    # Same logic as BOS validation
    # CISD direction "long" means bullish delivery, check for zones below
    # CISD direction "short" means bearish delivery, check for zones above

    if cisd.direction == "long":
        # Check for bearish FVGs below CISD confirmation price
        opposing_fvgs = [
            fvg for fvg in smc_context.fvgs_h1 + smc_context.fvgs_m2
            if fvg.status == "active"
            and fvg.direction == "bearish"
            and fvg.midpoint < cisd.confirmation_price
            and fvg.midpoint > cisd.confirmation_price * 0.98
        ]

        if opposing_fvgs:
            return (False, f"Bearish FVG below CISD at {opposing_fvgs[0].midpoint}")

        opposing_fractals = [
            f for f in smc_context.active_fractal_lows
            if f < cisd.confirmation_price
            and f > cisd.confirmation_price * 0.98
        ]

        if opposing_fractals:
            return (False, f"Unswept low fractal below CISD at {opposing_fractals[0]}")

    else:  # short
        opposing_fvgs = [
            fvg for fvg in smc_context.fvgs_h1 + smc_context.fvgs_m2
            if fvg.status == "active"
            and fvg.direction == "bullish"
            and fvg.midpoint > cisd.confirmation_price
            and fvg.midpoint < cisd.confirmation_price * 1.02
        ]

        if opposing_fvgs:
            return (False, f"Bullish FVG above CISD at {opposing_fvgs[0].midpoint}")

        opposing_fractals = [
            f for f in smc_context.active_fractal_highs
            if f > cisd.confirmation_price
            and f < cisd.confirmation_price * 1.02
        ]

        if opposing_fractals:
            return (False, f"Unswept high fractal above CISD at {opposing_fractals[0]}")

    return (True, None)
```

### 4.3 Order Flow Validation

**Rule**: Order flow needs 2 confirmations (steps). Invalidated by close below last confirmed step.

```python
# src/smc/models.py

@dataclass
class OrderFlow:
    """
    Order flow from trigger point (A) to target (B).

    Requires 2 confirmations (steps) to be valid.
    """
    id: str
    instrument: str
    timeframe: str
    direction: str  # "long" / "short"
    trigger_price: float  # Point A
    trigger_time: datetime
    target_price: float   # Point B (estimated)

    # Confirmation steps
    step1_price: Optional[float] = None
    step1_time: Optional[datetime] = None
    step1_type: Optional[str] = None  # "liquidity_sweep", "zone_test", "cisd", etc.

    step2_price: Optional[float] = None
    step2_time: Optional[datetime] = None
    step2_type: Optional[str] = None

    status: str = "pending"  # "pending", "step1_confirmed", "validated", "invalidated"
    invalidation_time: Optional[datetime] = None


# src/smc/engine.py

def validate_order_flow(
    self,
    order_flow: OrderFlow,
    current_bar: pd.Series
) -> Tuple[str, Optional[str]]:
    """
    Validate order flow status.

    Rules:
        - Needs 2 confirmations (steps) to be validated
        - Invalidated by close below last confirmed step

    Args:
        order_flow: Order flow to validate
        current_bar: Current price bar

    Returns:
        (new_status, reason) tuple
    """
    if order_flow.status == "invalidated":
        return (order_flow.status, order_flow.invalidation_reason)

    # Check for invalidation: close below last confirmed step
    if order_flow.direction == "long":
        if order_flow.step2_price:
            # Two steps confirmed, check close below step2
            if current_bar["close"] < order_flow.step2_price:
                return ("invalidated", f"Close below step2 at {current_bar['time']}")
        elif order_flow.step1_price:
            # One step confirmed, check close below step1
            if current_bar["close"] < order_flow.step1_price:
                return ("invalidated", f"Close below step1 at {current_bar['time']}")

    else:  # short
        if order_flow.step2_price:
            if current_bar["close"] > order_flow.step2_price:
                return ("invalidated", f"Close above step2 at {current_bar['time']}")
        elif order_flow.step1_price:
            if current_bar["close"] > order_flow.step1_price:
                return ("invalidated", f"Close above step1 at {current_bar['time']}")

    # Check for step confirmations (logic depends on what qualifies as a "step")
    # For now, placeholder - will be refined based on specific criteria
    # (e.g., step = CISD + zone test + price advancement)

    if order_flow.status == "pending" and order_flow.step1_price:
        return ("step1_confirmed", None)

    if order_flow.status == "step1_confirmed" and order_flow.step2_price:
        return ("validated", "Two steps confirmed")

    return (order_flow.status, None)
```

**Implementation Status**: PARTIAL (structure defined, step detection logic TBD based on Phase 6 features).

---

## 5. Entry Validation

### 5.1 Model 1 vs Model 2

**Model 1** (full conviction):
- Clear variables
- Sufficient confirmations
- No opposing open zones
- Works per narrative

**Model 2** (lower conviction, used for reversals or continuation after TP1):
- For reversals
- Insufficient confirming factors but clear idea
- After first TP hit on continuation

### 5.2 Implementation

```python
# src/smc/engine.py

def validate_entry(
    self,
    signal: Signal,
    smc_context: SMCDayContext
) -> Tuple[bool, Optional[str]]:
    """
    Validate if entry conditions are met (Model 1 or Model 2).

    Model 1: Full conviction (requires all factors)
    Model 2: Lower conviction (for reversals or post-TP1 continuation)

    Args:
        signal: Detected signal (Reverse, OCAE, TCWE, REV_RB)
        smc_context: Current SMC context

    Returns:
        (is_valid, rejection_reason) tuple
    """
    # Determine if this is a reversal trade
    is_reversal = (signal.signal_type in ["Reverse", "REV_RB"])

    # Check for opposing open zones
    opposing_zones = self._find_opposing_zones(signal, smc_context)

    # Count confirmations
    confirmations = self._count_confirmations(signal, smc_context)

    # Model 1: Full conviction entry
    if not is_reversal:
        # Continuation trade - use Model 1 criteria
        if len(opposing_zones) > 0:
            return (False, f"Opposing zones: {opposing_zones}")

        if confirmations < 2:
            return (False, f"Insufficient confirmations ({confirmations}/2)")

        # Check narrative alignment (placeholder - requires narrative tracking)
        # For now, assume narrative OK

        return (True, None)

    # Model 2: Reversal entry (lower bar)
    else:
        # Allow entry even with fewer confirmations if idea is clear
        if confirmations < 1:
            return (False, "No confirmations for reversal")

        # Allow opposing zones for reversals (we're betting on reversal)
        # But warn if opposing zones are strong
        if len(opposing_zones) > 2:
            # Too many opposing zones, even for reversal
            return (False, f"Too many opposing zones ({len(opposing_zones)})")

        return (True, "Model 2 (reversal entry)")


def _find_opposing_zones(self, signal: Signal, smc_context: SMCDayContext) -> List[str]:
    """Find zones that oppose signal direction."""
    opposing = []

    if signal.direction == "long":
        # For long, opposing zones are bearish FVGs above entry
        for fvg in smc_context.fvgs_h1 + smc_context.fvgs_m2:
            if fvg.status == "active" and fvg.direction == "bearish":
                if fvg.low > signal.entry_price and fvg.low < signal.entry_price * 1.02:
                    opposing.append(f"Bearish FVG at {fvg.midpoint}")

        # Unswept high fractals below entry (could reject)
        for f_high in smc_context.active_fractal_highs:
            if f_high < signal.entry_price and f_high > signal.entry_price * 0.98:
                opposing.append(f"High fractal at {f_high}")

    else:  # short
        for fvg in smc_context.fvgs_h1 + smc_context.fvgs_m2:
            if fvg.status == "active" and fvg.direction == "bullish":
                if fvg.high < signal.entry_price and fvg.high > signal.entry_price * 0.98:
                    opposing.append(f"Bullish FVG at {fvg.midpoint}")

        for f_low in smc_context.active_fractal_lows:
            if f_low > signal.entry_price and f_low < signal.entry_price * 1.02:
                opposing.append(f"Low fractal at {f_low}")

    return opposing


def _count_confirmations(self, signal: Signal, smc_context: SMCDayContext) -> int:
    """Count confirmations for signal."""
    count = 0

    # Confirmation 1: Supporting FVG
    for fvg in smc_context.fvgs_h1 + smc_context.fvgs_m2:
        if fvg.status == "active":
            if signal.direction == "long" and fvg.direction == "bullish":
                if abs(fvg.midpoint - signal.entry_price) / signal.entry_price < 0.01:
                    count += 1
                    break
            elif signal.direction == "short" and fvg.direction == "bearish":
                if abs(fvg.midpoint - signal.entry_price) / signal.entry_price < 0.01:
                    count += 1
                    break

    # Confirmation 2: CISD in signal direction
    for cisd in smc_context.cisd_events:
        if cisd.direction == signal.direction:
            if abs(cisd.confirmation_price - signal.entry_price) / signal.entry_price < 0.01:
                count += 1
                break

    # Confirmation 3: BOS in signal direction
    for bos in smc_context.bos_events:
        if bos.direction == ("bullish" if signal.direction == "long" else "bearish"):
            if abs(bos.broken_level - signal.entry_price) / signal.entry_price < 0.02:
                count += 1
                break

    return count
```

---

## 6. TP (Take Profit) Validation

### 6.1 Rules

- Aim for **open zones** (OB, FVG, S/R, LQ pools)
- If LQ pool already worked (swept/tested), it's **closed** -> aim for first zone before it
- In reversals, set TPs at points theoretically continuing main movement

### 6.2 Implementation

```python
# src/smc/engine.py

def validate_tp_target(
    self,
    tp_price: float,
    direction: str,
    smc_context: SMCDayContext
) -> bool:
    """
    Validate if TP target is a valid open zone.

    Valid TP targets:
        - Active FVG (opposite direction)
        - Unswept fractal
        - S/R level (Phase 6)
        - Open LQ pool (Phase 6)

    Invalid TP targets:
        - Already filled FVG
        - Already swept fractal
        - Closed LQ pool (Phase 6)

    Args:
        tp_price: Proposed TP price
        direction: Trade direction ("long" / "short")
        smc_context: Current SMC context

    Returns:
        True if TP target is valid, False otherwise
    """
    # Check for active FVG near TP price
    for fvg in smc_context.fvgs_h1 + smc_context.fvgs_m2:
        if fvg.status != "active":
            continue

        # TP should target opposite-direction FVG (where price might react)
        if direction == "long":
            # Long trade: TP should be at bearish FVG (resistance)
            if fvg.direction == "bearish":
                if abs(fvg.midpoint - tp_price) / tp_price < 0.005:  # Within 0.5%
                    return True
        else:  # short
            # Short trade: TP should be at bullish FVG (support)
            if fvg.direction == "bullish":
                if abs(fvg.midpoint - tp_price) / tp_price < 0.005:
                    return True

    # Check for unswept fractals near TP price
    if direction == "long":
        # Long trade: TP at high fractal (resistance)
        for f_high in smc_context.active_fractal_highs:
            if abs(f_high - tp_price) / tp_price < 0.005:
                return True
    else:  # short
        # Short trade: TP at low fractal (support)
        for f_low in smc_context.active_fractal_lows:
            if abs(f_low - tp_price) / tp_price < 0.005:
                return True

    # No valid zone found at TP price
    return False


def find_nearest_valid_tp(
    self,
    entry_price: float,
    direction: str,
    smc_context: SMCDayContext,
    min_rr: float = 2.0
) -> Optional[float]:
    """
    Find nearest valid TP target (open zone).

    Args:
        entry_price: Entry price
        direction: Trade direction
        smc_context: Current SMC context
        min_rr: Minimum R:R ratio for TP

    Returns:
        TP price or None if no valid target found
    """
    candidates = []

    # Collect all potential TP targets (active zones)
    for fvg in smc_context.fvgs_h1 + smc_context.fvgs_m2:
        if fvg.status != "active":
            continue

        if direction == "long" and fvg.direction == "bearish":
            if fvg.midpoint > entry_price:
                candidates.append(("FVG", fvg.midpoint))
        elif direction == "short" and fvg.direction == "bullish":
            if fvg.midpoint < entry_price:
                candidates.append(("FVG", fvg.midpoint))

    # Add fractals
    if direction == "long":
        for f_high in smc_context.active_fractal_highs:
            if f_high > entry_price:
                candidates.append(("Fractal", f_high))
    else:
        for f_low in smc_context.active_fractal_lows:
            if f_low < entry_price:
                candidates.append(("Fractal", f_low))

    # Sort by distance from entry
    if direction == "long":
        candidates.sort(key=lambda x: x[1])  # Ascending (nearest first)
    else:
        candidates.sort(key=lambda x: -x[1])  # Descending

    # Return first candidate that meets min R:R
    # (Simplified: assumes SL is at standard distance, calculate R:R properly in real implementation)
    for zone_type, price in candidates:
        # Placeholder R:R calculation (need actual SL to compute)
        # For now, just return nearest valid zone
        return price

    return None
```

---

## 7. SL (Stop Loss) Validation

### 7.1 Rules

- SL behind **idea invalidation**
- If exiting from zone, SL behind zone test
- **MISTAKE**: SL behind new POI (Point of Interest) or continuation point

### 7.2 Implementation

```python
# src/smc/engine.py

def validate_sl_placement(
    self,
    sl_price: float,
    entry_price: float,
    direction: str,
    smc_context: SMCDayContext
) -> Tuple[bool, Optional[str]]:
    """
    Validate if SL is correctly placed behind idea invalidation.

    Valid SL placement:
        - Behind invalidation zone (FVG, fractal, CISD level)
        - NOT behind new POI or continuation point

    Args:
        sl_price: Proposed SL price
        entry_price: Entry price
        direction: Trade direction
        smc_context: Current SMC context

    Returns:
        (is_valid, warning_message) tuple

    Note: Returns warning instead of rejection (SL placement is advisory)
    """
    # Check if SL is behind a new POI (Point of Interest)
    # New POI = recent FVG or CISD that could support continuation

    if direction == "long":
        # Long trade: check for bullish POIs below SL
        # If bullish POI exists between SL and entry, SL is incorrectly placed
        for fvg in smc_context.fvgs_m2 + smc_context.fvgs_h1:
            if fvg.status == "active" and fvg.direction == "bullish":
                if sl_price < fvg.midpoint < entry_price:
                    return (False, f"SL behind bullish FVG POI at {fvg.midpoint}")

        for cisd in smc_context.cisd_events:
            if cisd.direction == "long":
                if sl_price < cisd.confirmation_price < entry_price:
                    return (False, f"SL behind CISD POI at {cisd.confirmation_price}")

    else:  # short
        for fvg in smc_context.fvgs_m2 + smc_context.fvgs_h1:
            if fvg.status == "active" and fvg.direction == "bearish":
                if entry_price < fvg.midpoint < sl_price:
                    return (False, f"SL behind bearish FVG POI at {fvg.midpoint}")

        for cisd in smc_context.cisd_events:
            if cisd.direction == "short":
                if entry_price < cisd.confirmation_price < sl_price:
                    return (False, f"SL behind CISD POI at {cisd.confirmation_price}")

    # Check if SL is behind invalidation zone (GOOD)
    # Invalidation zone = where idea is proven wrong

    if direction == "long":
        # For long, invalidation = bearish FVG or low fractal below entry
        # SL should be below these
        for fvg in smc_context.fvgs_m2 + smc_context.fvgs_h1:
            if fvg.status == "active" and fvg.direction == "bearish":
                if fvg.low < entry_price and sl_price < fvg.low:
                    # SL correctly placed below bearish FVG (invalidation zone)
                    return (True, None)

        for f_low in smc_context.active_fractal_lows:
            if f_low < entry_price and sl_price < f_low:
                # SL correctly placed below low fractal
                return (True, None)

    else:  # short
        for fvg in smc_context.fvgs_m2 + smc_context.fvgs_h1:
            if fvg.status == "active" and fvg.direction == "bullish":
                if fvg.high > entry_price and sl_price > fvg.high:
                    return (True, None)

        for f_high in smc_context.active_fractal_highs:
            if f_high > entry_price and sl_price > f_high:
                return (True, None)

    # No clear invalidation zone found, issue warning
    return (True, "SL placement unclear (no invalidation zone identified)")
```

---

## 8. BE (Breakeven) Validation

### 8.1 Rules (7 conditions from InValidation.md)

Move to breakeven when:
1. Strong FTA reaction
2. Important liquidity zone arrival
3. Idea invalidation
4. Time variables (session timing, news)
5. SMT (can BE but better full exit)
6. Structural BE via LTF structure close
7. **NEVER** if BE point is in potential POI (with exceptions)

### 8.2 Implementation

```python
# src/smc/engine.py

def validate_be_move(
    self,
    current_bar: pd.Series,
    entry_time: datetime,
    entry_price: float,
    direction: str,
    smc_context: SMCDayContext
) -> Tuple[bool, Optional[str]]:
    """
    Validate if moving to breakeven is appropriate.

    7 conditions for BE:
        1. Strong FTA (First Trouble Area) reaction
        2. Important LQ zone arrival (Phase 6)
        3. Idea invalidation
        4. Time variables (session, news)
        5. SMT (Phase 6 - recommend full exit instead)
        6. Structural BE via LTF structure close
        7. NEVER if BE point in potential POI

    Args:
        current_bar: Current price bar
        entry_time: When trade was entered
        entry_price: Entry price
        direction: Trade direction
        smc_context: Current SMC context

    Returns:
        (should_move_to_be, reason) tuple
    """
    # Check 7: NEVER if BE point (entry price) is in potential POI
    for fvg in smc_context.fvgs_m2 + smc_context.fvgs_h1:
        if fvg.status == "active":
            if fvg.low <= entry_price <= fvg.high:
                # Entry price is inside active FVG (POI)
                return (False, "BE point is inside active FVG (potential POI)")

    # Check 1: Strong FTA reaction
    # FTA = first significant level after entry
    # Strong reaction = large wick at FTA level
    # (Simplified: check if current bar has large wick relative to body)
    wick_size = max(
        abs(current_bar["high"] - max(current_bar["open"], current_bar["close"])),
        abs(current_bar["low"] - min(current_bar["open"], current_bar["close"]))
    )
    body_size = abs(current_bar["close"] - current_bar["open"])

    if body_size > 0 and wick_size / body_size > 2.0:
        # Large wick relative to body (strong rejection)
        return (True, "Strong FTA reaction (large wick)")

    # Check 2: Important LQ zone arrival
    # OUT OF SCOPE for Phase 5 (requires LQ pool detection)
    # Will implement in Phase 6

    # Check 3: Idea invalidation
    # Check if opposing zone has been activated
    # (Simplified: check if price tested opposing FVG)
    if direction == "long":
        for fvg in smc_context.fvgs_m2 + smc_context.fvgs_h1:
            if fvg.status == "active" and fvg.direction == "bearish":
                if current_bar["high"] >= fvg.low and current_bar["high"] <= fvg.high:
                    # Price entered bearish FVG (opposing zone)
                    return (True, "Idea invalidation: tested opposing FVG")
    else:  # short
        for fvg in smc_context.fvgs_m2 + smc_context.fvgs_h1:
            if fvg.status == "active" and fvg.direction == "bullish":
                if current_bar["low"] <= fvg.high and current_bar["low"] >= fvg.low:
                    return (True, "Idea invalidation: tested opposing FVG")

    # Check 4: Time variables
    # News events: check if high-impact news coming up
    # (Requires news filter integration - placeholder)
    time_to_news_minutes = self._time_to_next_news(current_bar["time"])
    if time_to_news_minutes and time_to_news_minutes < 15:
        return (True, f"High-impact news in {time_to_news_minutes} minutes")

    # Session timing: check if session ending soon
    # (Simplified: check if close to end of trade window)
    # (Requires trade window end time - placeholder)

    # Check 5: SMT
    # OUT OF SCOPE for Phase 5 (requires SMT detection)
    # Will recommend full exit instead of BE

    # Check 6: Structural BE via LTF structure close
    # Check if LTF (lower timeframe, e.g., M1) closed beyond structure level
    # (Requires BOS on M1 - simplified check)
    recent_bos = [bos for bos in smc_context.bos_events if bos.break_time > entry_time]
    if recent_bos:
        # Structural break occurred after entry
        return (True, f"LTF structure close at {recent_bos[0].break_time}")

    # No BE conditions met
    return (False, None)


def _time_to_next_news(self, current_time: datetime) -> Optional[int]:
    """
    Calculate minutes to next high-impact news event.

    Returns:
        Minutes to news or None if no news in next 60 minutes
    """
    # Placeholder: requires news filter integration
    # For now, return None (no news)
    return None
```

---

## 9. Execution Checklist

### Phase 5A: FVG Validation (Week 1)

- [ ] Implement `validate_fvg_status()` in `SMCEngine`
- [ ] Implement helper `_calculate_fvg_fill()`
- [ ] Add FVG status update logic to `SMCRegistry.update_fvg_status()`
- [ ] Test FVG validation on synthetic data (all 6 variants)
- [ ] Integrate FVG validation into fast backtest `_build_smc_context()`

### Phase 5B: Instrument Validation (Week 2)

- [ ] Implement `validate_bos()` in `SMCEngine`
- [ ] Implement `validate_cisd()` in `SMCEngine`
- [ ] Define `OrderFlow` dataclass in `models.py`
- [ ] Implement `validate_order_flow()` in `SMCEngine` (partial, step detection TBD)
- [ ] Test BOS/CISD validation on historical data

### Phase 5C: Entry Validation (Week 3)

- [ ] Implement `validate_entry()` in `SMCEngine`
- [ ] Implement `_find_opposing_zones()` helper
- [ ] Implement `_count_confirmations()` helper
- [ ] Test entry validation on 10 historical signal days (Model 1 and Model 2)
- [ ] Integrate entry validation into fast backtest after SMC filter

### Phase 5D: TP/SL Validation (Week 4)

- [ ] Implement `validate_tp_target()` in `SMCEngine`
- [ ] Implement `find_nearest_valid_tp()` helper
- [ ] Implement `validate_sl_placement()` in `SMCEngine`
- [ ] Test TP/SL validation on historical trades
- [ ] Integrate TP/SL validation into fast backtest trade setup

### Phase 5E: BE Validation (Week 5)

- [ ] Implement `validate_be_move()` in `SMCEngine`
- [ ] Implement `_time_to_next_news()` helper (integrate with news filter)
- [ ] Test BE validation on live trades (paper trading mode)
- [ ] Integrate BE validation into trade management logic
- [ ] Document BE conditions in `docs/SMC_VALIDATION_RULES.md`

### Phase 5F: Parity Testing (Week 6)

- [ ] Create test dataset (10 days with known SMC events)
- [ ] Manually analyze test days: mark expected validation outcomes
- [ ] Run fast backtest with validation enabled
- [ ] Compare: automated validation decisions vs manual analysis
- [ ] Debug mismatches, refine validation logic
- [ ] Achieve >90% parity on test dataset

### Phase 5G: Documentation (Week 7)

- [ ] Create `docs/SMC_VALIDATION_RULES.md` with all rules explained
- [ ] Update `PHASE_1_FOUNDATION.md` with validation references
- [ ] Document validation params (if any added)
- [ ] Create validation decision flowcharts (FVG, BOS, entry, TP, SL, BE)
- [ ] Add validation examples to `notebooks/debug_smc.ipynb`

---

## 10. Integration with Fast Backtest (Phase 4)

**Modified _process_day() with Phase 5 validation**:

```python
def _process_day(self, day_df: pd.DataFrame, day_date: datetime.date, params: Dict) -> Optional[Dict[str, Any]]:
    """
    Process single trading day (PHASE 4 + PHASE 5).

    Workflow:
        1. Get IB
        2. Build SMC context (Phase 4)
        3. Update FVG statuses (Phase 5)
        4. Get trade window data
        5. Detect signals
        6. Apply SMC filter (Phase 4)
        7. Validate entry (Phase 5)
        8. Validate TP/SL (Phase 5)
        9. Simulate trade with BE validation (Phase 5)
    """
    # ... IB computation ...

    # Build SMC context (Phase 4)
    smc_context = self._build_smc_context(day_df, day_date, smc_config)

    # Update FVG statuses (Phase 5)
    recent_bos = smc_context.bos_events[-5:] if len(smc_context.bos_events) > 0 else []
    for fvg in smc_context.fvgs_h1 + smc_context.fvgs_m2:
        # Check each bar in trade window to update FVG status
        for bar in df_trade_m1.itertuples():
            new_status, reason = self.smc_engine.validate_fvg_status(fvg, bar._asdict(), recent_bos)
            if new_status != fvg.status:
                fvg.status = new_status
                fvg.invalidation_reason = reason
                fvg.invalidation_time = bar.time
                break  # Stop checking once invalidated

    # ... Signal detection (Phase 4) ...

    # Apply SMC filter (Phase 4)
    signal = self._apply_smc_filter_batch(signal, smc_context, df_trade_m1, params)
    if signal is None:
        return None

    # Validate entry (Phase 5)
    entry_valid, rejection_reason = self.smc_engine.validate_entry(signal, smc_context)
    if not entry_valid:
        logger.info(f"Entry rejected: {rejection_reason}")
        return None

    # Validate TP target (Phase 5)
    tp_valid = self.smc_engine.validate_tp_target(signal.tp_price, signal.direction, smc_context)
    if not tp_valid:
        # Adjust TP to nearest valid zone
        new_tp = self.smc_engine.find_nearest_valid_tp(signal.entry_price, signal.direction, smc_context)
        if new_tp:
            logger.info(f"TP adjusted: {signal.tp_price} -> {new_tp}")
            signal.tp_price = new_tp
        else:
            logger.warning("No valid TP target found, using original TP")

    # Validate SL placement (Phase 5)
    sl_valid, sl_warning = self.smc_engine.validate_sl_placement(
        signal.sl_price, signal.entry_price, signal.direction, smc_context
    )
    if not sl_valid:
        logger.warning(f"SL placement issue: {sl_warning}")
        # Don't reject trade, but log warning

    # Simulate trade with BE validation (Phase 5)
    return self._simulate_trade_with_be(df_trade_m1, signal, ib, day_date, params, smc_context)
```

**New method: _simulate_trade_with_be()**:

```python
def _simulate_trade_with_be(
    self,
    df_trade_m1: pd.DataFrame,
    signal: Signal,
    ib: Dict,
    day_date: datetime.date,
    params: Dict,
    smc_context: SMCDayContext
) -> Optional[Dict[str, Any]]:
    """
    Simulate trade execution with BE (breakeven) validation.

    Enhanced version of _simulate_trade_on_m1() with BE logic.
    """
    # ... existing trade simulation logic ...

    be_moved = False
    current_sl = signal.sl_price

    for i, bar in enumerate(df_trade_m1.itertuples()):
        # Check TP hit
        # ... existing TP logic ...

        # Check SL hit
        # ... existing SL logic ...

        # Check BE conditions (Phase 5)
        if not be_moved and i > 0:  # Only check after entry bar
            should_be, be_reason = self.smc_engine.validate_be_move(
                bar._asdict(),
                signal.entry_time,
                signal.entry_price,
                signal.direction,
                smc_context
            )

            if should_be:
                current_sl = signal.entry_price
                be_moved = True
                logger.info(f"Moved to BE: {be_reason}")

        # ... continue simulation ...
```

---

## 11. Parity Testing Approach

### 11.1 Test Dataset Preparation

```python
# tests/test_validation_parity.py

TEST_DAYS = [
    {
        "date": "2024-01-15",
        "expected_fvg_invalidations": [
            {"fvg_id": "fvg_abc123", "status": "structural_update", "time": "10:30"},
            {"fvg_id": "fvg_def456", "status": "full_fill", "time": "14:15"},
        ],
        "expected_entry_rejections": [
            {"signal_type": "OCAE", "reason": "Opposing zones: Bearish FVG at 16500"},
        ],
        "expected_be_moves": [
            {"trade_id": 1, "time": "11:00", "reason": "Strong FTA reaction"},
        ],
    },
    # ... 9 more test days ...
]
```

### 11.2 Parity Test Script

```python
def test_validation_parity():
    """
    Test that validation decisions match manual analysis.

    For each test day:
        1. Run fast backtest with validation enabled
        2. Extract validation decisions (FVG status, entry rejections, BE moves, etc.)
        3. Compare with expected outcomes from manual analysis
    """
    for test_case in TEST_DAYS:
        # Load data
        day_df = load_m1_data("GER40", date=test_case["date"])

        # Run fast backtest
        fast_bt = FastBacktest(symbol="GER40", m1_data=day_df)
        results = fast_bt.run_with_params(GER40_PARAMS_PROD)

        # Extract validation events from event log
        validation_events = fast_bt.smc_engine.event_log.get_events(
            event_types=["FVG_STATUS_CHANGE", "ENTRY_REJECTED", "BE_MOVED"]
        )

        # Compare FVG invalidations
        for expected_fvg in test_case["expected_fvg_invalidations"]:
            actual_fvg = next(
                (e for e in validation_events
                 if e["event_type"] == "FVG_STATUS_CHANGE"
                 and e["details"]["fvg_id"] == expected_fvg["fvg_id"]),
                None
            )
            assert actual_fvg is not None, f"FVG {expected_fvg['fvg_id']} not invalidated"
            assert actual_fvg["details"]["new_status"] == expected_fvg["status"]
            # Time tolerance: within 5 minutes
            assert abs((actual_fvg["time"] - pd.Timestamp(expected_fvg["time"])).total_seconds()) < 300

        # Compare entry rejections
        # ... similar checks ...

        # Compare BE moves
        # ... similar checks ...

    print("Parity test PASSED: All validation decisions match manual analysis")
```

---

## 12. Performance Considerations

**Validation overhead**:
- FVG status update: O(n_bars * n_fvgs) per day (~100 bars * 5 FVGs = 500 checks)
- Entry validation: O(n_zones) per signal (~10 zones per signal)
- TP/SL validation: O(n_zones) per signal
- BE validation: O(n_bars) per trade (~100 bars per trade)

**Optimization strategies**:
- Pre-compute zone lists in `_build_smc_context()` (already done in Phase 4)
- Use spatial indexing for zone lookups (future optimization)
- Early termination in validation checks
- Cache validation results per bar (if multiple checks needed)

**Target overhead**: <500ms per day (compared to Phase 4 target of ~50ms per day for SMC context build).

---

## 13. Out of Scope for Phase 5

The following features are explicitly **OUT OF SCOPE** for Phase 5 and will be implemented in **Phase 6 (Advanced SMC Structures)**:

- [ ] Order Block (OB) detection
- [ ] Breaker Block (BB) detection
- [ ] Mitigation Block (MB) detection
- [ ] Rejection Block (RB) detection
- [ ] Fair Value Levels (FVL)
- [ ] Liquidity pool detection
- [ ] Inducement (IDM) detection
- [ ] SMT (Smart Money Tool) detection
- [ ] Support/Resistance (S/R) level detection
- [ ] Fibonacci level integration

---

## 14. Success Metrics

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| FVG validation accuracy | >90% | Parity test vs manual analysis |
| Entry rejection accuracy | >85% | Parity test (avoid bad entries) |
| TP adjustment quality | >80% | Backtest R improvement vs unadjusted |
| SL warning accuracy | >75% | Manual review of warnings |
| BE timing accuracy | >70% | Parity test (avoid premature BE) |
| Performance overhead | <500ms/day | Benchmark on full dataset |
| Code coverage | >80% | pytest coverage report |

---

## 15. Next Steps After Phase 5

Once Phase 5 is complete and tested:
- **Phase 6**: Implement advanced SMC structures (OB, BB, MB, LQ, IDM, SMT)
- **Phase 7**: Integrate SMC into live bot (`src/strategies/ib_strategy_smc.py`)
- **Phase 8**: Production deployment on VPS with full monitoring

---

## Appendix A: Validation Decision Flowchart (FVG)

```
[New FVG Detected]
       |
       v
[Status: active]
       |
       v
[On each new bar]
       |
       +--[Check 1: BOS occurred after FVG?]--YES--> [Status: structural_update]
       |
       +--[Check 2: IDM formed? (Phase 6)]--YES--> [Status: inducement]
       |
       +--[Check 3: Pass-through to zone?]--YES--> [Status: zone_to_zone]
       |
       +--[Check 4: LQ/SMT mitigation? (Phase 6)]--YES--> [Status: zone_to_lq]
       |
       +--[Check 5: 100% filled?]--YES--> [Status: full_fill]
       |
       +--[Check 6: Inverted?]--YES--> [Status: inverted]
       |
       v
[Status: active] (continue monitoring)
```

---

## Appendix B: File Modifications Summary

| File | Change Type | Description |
|------|-------------|-------------|
| `src/smc/engine.py` | MODIFY | Add validation methods: `validate_fvg_status()`, `validate_entry()`, `validate_tp_target()`, `validate_sl_placement()`, `validate_be_move()`, `validate_bos()`, `validate_cisd()`, `validate_order_flow()` |
| `src/smc/models.py` | MODIFY | Add `OrderFlow` dataclass |
| `src/smc/registry.py` | MODIFY | Add `update_fvg_status()`, `get_active_fvgs()` |
| `params_optimizer/engine/fast_backtest.py` | MODIFY | Add validation integration in `_process_day()`, add `_simulate_trade_with_be()` |
| `tests/test_validation_parity.py` | NEW | Parity testing script for validation |
| `docs/SMC_VALIDATION_RULES.md` | NEW | Comprehensive validation rules documentation |

---

**END OF PHASE_5_VALIDATION.md**
