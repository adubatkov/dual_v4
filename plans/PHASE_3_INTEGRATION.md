# Phase 3: Integration with Strategy and Debug Tools

## Goal

Integrate SMC engine into the strategy execution layer and debug tooling. After this phase, the bot can evaluate signals through SMC filters in both slow backtest and debug notebook environments.

**Deliverables:**
1. `IBStrategySMC` class - SMC-aware wrapper around existing `IBStrategy`
2. `debug_smc.ipynb` - Jupyter notebook for SMC-aware debugging
3. `generate_chart_with_smc()` - Chart overlays for SMC structures
4. Integration pattern documentation for fast backtest

---

## Dependencies

**Phase 1 (must be complete):**
- `src/smc/models.py` - FVG, Fractal, CISD, BOS, SMCDecision dataclasses
- `src/smc/config.py` - SMCConfig per-instrument parameters
- `src/smc/timeframe_manager.py` - M1 resampling to all timeframes
- `src/smc/registry.py` - SMCRegistry for structure tracking
- `src/smc/event_log.py` - SMCEventLog for audit trail
- `src/smc/detectors/fractal_detector.py` - detect_fractals()
- `src/smc/detectors/fvg_detector.py` - detect_fvgs()

**Phase 2 (must be complete):**
- `src/smc/detectors/cisd_detector.py` - detect_cisd()
- `src/smc/detectors/market_structure_detector.py` - detect_bos(), detect_market_structure()
- `src/smc/engine.py` - SMCEngine with evaluate_signal() and check_confirmation()

**Existing infrastructure:**
- `src/strategies/ib_strategy.py` - Production strategy (MUST NOT be modified)
- `notebooks/debug_strategy.ipynb` - Existing debug notebook (reference pattern)
- `notebooks/debug_tools.py` - SingleDayBacktest, generate_chart()
- `backtest/adapter.py` - BacktestExecutor, create_mt5_patch_module()

---

## 1. IBStrategySMC: SMC-Aware Strategy Wrapper

### 1.1 Design Philosophy

**CRITICAL**: Do NOT modify the original `IBStrategy`. It is production-validated code (~40KB, complex FSM) used by live bot.

`IBStrategySMC` is a wrapper class that:
- Inherits from `IBStrategy`
- Overrides only signal checking methods
- Adds new FSM state: `AWAITING_CONFIRMATION`
- Calls `super()` for all unchanged states
- Initializes and manages `SMCEngine`

### 1.2 File Location

```
dual_v4/src/strategies/ib_strategy_smc.py
```

### 1.3 Class Interface

```python
"""
SMC-Aware IB Strategy Wrapper.

Extends IBStrategy with Smart Money Concepts overlay:
- Evaluates signals through SMC filter (FVG, BOS, CISD, fractals)
- Can delay entry for confirmation
- Can modify SL/TP based on active structures
- Adds AWAITING_CONFIRMATION state to FSM

FSM States:
- AWAITING_IB_CALCULATION (inherited)
- AWAITING_TRADE_WINDOW (inherited)
- IN_TRADE_WINDOW (modified - adds SMC filter)
- AWAITING_CONFIRMATION (new)
- POSITION_OPEN (inherited)
- DAY_ENDED (inherited)
"""
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import pandas as pd

from .base_strategy import Signal
from .ib_strategy import IBStrategy
from src.smc.engine import SMCEngine
from src.smc.config import SMCConfig
from src.smc.timeframe_manager import TimeframeManager
from src.smc.models import SMCDecision

logger = logging.getLogger(__name__)


class IBStrategySMC(IBStrategy):
    """
    IB Strategy with Smart Money Concepts overlay.

    Minimal extension of IBStrategy - only overrides signal checking logic
    to add SMC evaluation layer.
    """

    def __init__(
        self,
        symbol: str,
        params: Dict[str, Any],
        executor,
        magic_number: int,
        strategy_label: str = "",
        news_filter_enabled: bool = False,
        smc_config: Optional[SMCConfig] = None,
    ):
        """
        Initialize SMC-aware strategy.

        Args:
            symbol: Trading symbol
            params: Strategy parameters (GER40_PARAMS_PROD or XAUUSD_PARAMS_PROD)
            executor: MT5Executor or BacktestExecutor
            magic_number: Unique magic number
            strategy_label: Optional label for logging
            news_filter_enabled: Enable news filter for 5ers compliance
            smc_config: SMC configuration (auto-generated if None)
        """
        # Initialize parent
        super().__init__(
            symbol=symbol,
            params=params,
            executor=executor,
            magic_number=magic_number,
            strategy_label=strategy_label,
            news_filter_enabled=news_filter_enabled,
        )

        # Update log prefix
        if strategy_label:
            self.log_prefix = f"[{symbol}_M{magic_number}_{strategy_label}_SMC]"
        else:
            self.log_prefix = f"[{symbol}_M{magic_number}_IB{self.ib_start}-{self.ib_end}_SMC]"

        # SMC configuration
        if smc_config is None:
            smc_config = self._create_default_smc_config(symbol)

        # Initialize timeframe manager
        self.timeframe_manager = TimeframeManager(
            base_timeframe="M1",
            target_timeframes=smc_config.timeframes,
        )

        # Initialize SMC engine
        self.smc_engine = SMCEngine(
            instrument=symbol,
            config=smc_config,
            timeframe_manager=self.timeframe_manager,
        )

        # Confirmation tracking (for AWAITING_CONFIRMATION state)
        self.pending_signal: Optional[Signal] = None
        self.confirmation_start_time: Optional[datetime] = None
        self.confirmation_timeout_minutes: int = smc_config.confirmation_timeout_minutes

        logger.info(f"{self.log_prefix} SMC engine initialized (TFs: {smc_config.timeframes})")

    def _create_default_smc_config(self, symbol: str) -> SMCConfig:
        """
        Create default SMC configuration based on symbol.

        Args:
            symbol: Trading symbol

        Returns:
            SMCConfig with instrument-specific defaults
        """
        from src.smc.config import SMCConfig

        if symbol == "GER40":
            return SMCConfig(
                instrument="GER40",
                timeframes=["M5", "M15", "H1"],
                rules={
                    "require_fvg_alignment": True,
                    "require_bos_confirmation": True,
                    "allow_entry_into_fvg": True,
                    "min_fvg_size_points": 5.0,
                    "cisd_required": False,
                },
                confirmation_timeout_minutes=30,
                filter_enabled=True,
            )
        else:  # XAUUSD
            return SMCConfig(
                instrument="XAUUSD",
                timeframes=["M5", "M15", "H1"],
                rules={
                    "require_fvg_alignment": True,
                    "require_bos_confirmation": True,
                    "allow_entry_into_fvg": True,
                    "min_fvg_size_points": 0.50,
                    "cisd_required": False,
                },
                confirmation_timeout_minutes=30,
                filter_enabled=True,
            )

    def reset_daily_state(self) -> None:
        """Reset all daily state variables including SMC engine."""
        super().reset_daily_state()

        # Reset SMC engine
        self.smc_engine.reset()

        # Reset confirmation tracking
        self.pending_signal = None
        self.confirmation_start_time = None

        logger.info(f"{self.log_prefix} Daily state + SMC engine reset")

    def check_signal(self, current_time_utc: datetime) -> Optional[Signal]:
        """
        FSM logic for signal detection with SMC overlay.

        States flow:
        AWAITING_IB_CALCULATION -> AWAITING_TRADE_WINDOW -> IN_TRADE_WINDOW
        -> (optional) AWAITING_CONFIRMATION -> POSITION_OPEN/DAY_ENDED

        Args:
            current_time_utc: Current UTC time

        Returns:
            Signal if detected and SMC-approved, None otherwise
        """
        # Handle states 0-2 (IB calculation, trade window) via parent
        if self.state in ["AWAITING_IB_CALCULATION", "AWAITING_TRADE_WINDOW"]:
            return super().check_signal(current_time_utc)

        # State 3: IN_TRADE_WINDOW - modified with SMC filter
        if self.state == "IN_TRADE_WINDOW":
            return self._check_signals_with_smc(current_time_utc)

        # State 3b: AWAITING_CONFIRMATION (new state)
        if self.state == "AWAITING_CONFIRMATION":
            return self._check_confirmation(current_time_utc)

        # States 4-5: POSITION_OPEN, DAY_ENDED - unchanged
        return super().check_signal(current_time_utc)

    def _check_signals_with_smc(self, current_time_utc: datetime) -> Optional[Signal]:
        """
        Check for signals with SMC evaluation.

        Flow:
        1. Call parent's signal detection (uses original IB logic)
        2. If signal detected, update SMC engine with M1 data
        3. Evaluate signal through SMC filter
        4. Decision outcomes:
           - ENTER: return signal (possibly modified SL/TP)
           - WAIT: store signal, transition to AWAITING_CONFIRMATION
           - REJECT: return None

        Args:
            current_time_utc: Current UTC time

        Returns:
            Signal if SMC approved, None otherwise
        """
        # Get signal from parent (original IB logic)
        signal = super().check_signal(current_time_utc)

        if signal is None:
            return None

        # Update SMC engine with latest M1 data
        # Get M1 bars from executor (M2 is minimum granularity in executor)
        # WORKAROUND: Fetch M2 and convert to M1 approximation
        bars_m2 = self.executor.get_bars(self.symbol, "M2", 500)

        if bars_m2 is None or bars_m2.empty:
            logger.warning(f"{self.log_prefix} No M1 data for SMC update, passing signal")
            return signal

        # Convert M2 to M1 approximation (duplicate each bar)
        # NOTE: This is a WORKAROUND for backtest. In live bot, use true M1 feed.
        bars_m1 = self._m2_to_m1_approximation(bars_m2)

        # Update SMC engine
        self.smc_engine.update(bars_m1, current_time_utc)

        # Evaluate signal through SMC filter
        decision = self.smc_engine.evaluate_signal(
            direction=signal.direction,
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            current_time=current_time_utc,
        )

        logger.info(
            f"{self.log_prefix} SMC Decision: {decision.action} "
            f"(reason: {decision.reason}, score: {decision.score:.2f})"
        )

        # Handle decision
        if decision.action == "ENTER":
            # Apply modifications if any
            if decision.modified_stop_loss is not None:
                signal.stop_loss = decision.modified_stop_loss
                logger.info(f"{self.log_prefix} SL modified by SMC: {decision.modified_stop_loss:.2f}")

            if decision.modified_take_profit is not None:
                signal.take_profit = decision.modified_take_profit
                logger.info(f"{self.log_prefix} TP modified by SMC: {decision.modified_take_profit:.2f}")

            return signal

        elif decision.action == "WAIT":
            # Store signal for confirmation checks
            self.pending_signal = signal
            self.confirmation_start_time = current_time_utc
            self.state = "AWAITING_CONFIRMATION"

            logger.info(
                f"{self.log_prefix} Signal held for confirmation. "
                f"Criteria: {decision.confirmation_criteria}"
            )
            return None

        else:  # REJECT
            logger.info(f"{self.log_prefix} Signal rejected by SMC: {decision.reason}")
            return None

    def _check_confirmation(self, current_time_utc: datetime) -> Optional[Signal]:
        """
        Check if pending signal meets confirmation criteria.

        Called repeatedly while in AWAITING_CONFIRMATION state.

        Flow:
        1. Check timeout (if exceeded, return to IN_TRADE_WINDOW)
        2. Update SMC engine with latest data
        3. Check confirmation criteria via engine
        4. If confirmed: return signal, transition to POSITION_OPEN (via parent)
        5. If not confirmed: return None, stay in AWAITING_CONFIRMATION

        Args:
            current_time_utc: Current UTC time

        Returns:
            Signal if confirmed, None otherwise
        """
        # Check timeout
        elapsed_minutes = (current_time_utc - self.confirmation_start_time).total_seconds() / 60

        if elapsed_minutes > self.confirmation_timeout_minutes:
            logger.info(
                f"{self.log_prefix} Confirmation timeout ({self.confirmation_timeout_minutes}m). "
                f"Returning to IN_TRADE_WINDOW."
            )
            self.pending_signal = None
            self.confirmation_start_time = None
            self.state = "IN_TRADE_WINDOW"
            return None

        # Update SMC engine
        bars_m2 = self.executor.get_bars(self.symbol, "M2", 500)
        if bars_m2 is not None and not bars_m2.empty:
            bars_m1 = self._m2_to_m1_approximation(bars_m2)
            self.smc_engine.update(bars_m1, current_time_utc)

        # Check confirmation
        confirmed = self.smc_engine.check_confirmation(current_time_utc)

        if confirmed:
            logger.info(f"{self.log_prefix} Signal CONFIRMED by SMC")
            signal = self.pending_signal
            self.pending_signal = None
            self.confirmation_start_time = None
            # State will transition to POSITION_OPEN when order is placed
            return signal

        # Not yet confirmed
        return None

    def _m2_to_m1_approximation(self, bars_m2: pd.DataFrame) -> pd.DataFrame:
        """
        Convert M2 bars to M1 approximation by duplicating each bar.

        WORKAROUND for backtest where M1 data is not directly available from executor.
        In live bot, use true M1 feed.

        Args:
            bars_m2: M2 OHLCV DataFrame

        Returns:
            M1-like DataFrame (each M2 bar duplicated with 1-minute offsets)
        """
        rows = []
        for idx, row in bars_m2.iterrows():
            # First M1 bar (same as M2)
            rows.append({
                "time": row["time"],
                "open": row["open"],
                "high": row["high"],
                "low": row["low"],
                "close": row["close"],
                "tick_volume": row["tick_volume"] // 2,
                "real_volume": row.get("real_volume", 0) // 2,
            })

            # Second M1 bar (1 minute later, same OHLC approximation)
            rows.append({
                "time": row["time"] + pd.Timedelta(minutes=1),
                "open": row["close"],
                "high": row["high"],
                "low": row["low"],
                "close": row["close"],
                "tick_volume": row["tick_volume"] // 2,
                "real_volume": row.get("real_volume", 0) // 2,
            })

        return pd.DataFrame(rows)
```

### 1.4 Testing in Slow Backtest

`IBStrategySMC` can be used as a drop-in replacement for `IBStrategy` in `backtest_runner.py`:

```python
# In backtest_runner.py or debug notebook:

from src.strategies.ib_strategy_smc import IBStrategySMC
from src.smc.config import SMCConfig

# Create SMC config (optional - uses defaults if None)
smc_config = SMCConfig(
    instrument="GER40",
    timeframes=["M5", "M15", "H1"],
    rules={
        "require_fvg_alignment": True,
        "require_bos_confirmation": False,  # More permissive
        "allow_entry_into_fvg": True,
    },
    confirmation_timeout_minutes=20,
    filter_enabled=True,
)

# Create strategy (same constructor as IBStrategy + smc_config)
strategy = IBStrategySMC(
    symbol="GER40",
    params=GER40_PARAMS_PROD,
    executor=executor,
    magic_number=1001,
    smc_config=smc_config,
)

# Use exactly like IBStrategy
signal = strategy.check_signal(current_time)
```

---

## 2. Debug Notebook: debug_smc.ipynb

### 2.1 Purpose

Jupyter notebook for SMC-aware single-day debugging. Allows side-by-side comparison of:
- Baseline trade (without SMC)
- SMC-filtered trade (with SMC)

### 2.2 File Location

```
dual_v4/notebooks/debug_smc.ipynb
```

### 2.3 Cell Structure

#### Cell 0: Title (Markdown)

```markdown
# IB Strategy SMC Debug Notebook (V4)

This notebook is used to:
1. Run single-day backtests with and without SMC filter
2. Compare baseline vs SMC-filtered results
3. Visualize trades with SMC structure overlays (FVG, BOS, fractals)
4. Inspect SMC event log and registry state at signal time

**PROD Parameters** (from AWS optimization):
- **GER40**: IB 08:00-09:00 (Europe/Berlin), RR=1.0, TSL_TARGET=0.5, TSL_SL=1.5
- **XAUUSD**: IB 09:00-09:30 (Asia/Tokyo), RR=0.5, TSL_TARGET=0.5, TSL_SL=1.5

**SMC Overlay**: FVG detection, BOS/CISD confirmation, fractal stops
```

#### Cell 1: Setup

```python
# Setup - run this first
import sys
import os
import logging
from datetime import datetime, timedelta
from pathlib import Path
import importlib

import pandas as pd
import matplotlib.pyplot as plt

# Set path to project root
NOTEBOOK_DIR = Path(os.getcwd())
DUAL_V4_PATH = NOTEBOOK_DIR.parent if NOTEBOOK_DIR.name == 'notebooks' else NOTEBOOK_DIR
sys.path.insert(0, str(DUAL_V4_PATH))

# Configure logging
logging.basicConfig(level=logging.WARNING, force=True)

# Force reload of modified modules
import notebooks.debug_tools
importlib.reload(notebooks.debug_tools)

from notebooks.debug_tools import (
    SingleDayBacktest,
    SingleDayBacktestSMC,  # NEW
    generate_chart_with_smc,  # NEW
)

# Enable inline plots
%matplotlib inline
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 150

print(f"Project path: {DUAL_V4_PATH}")
print("Setup complete! (PROD parameters + SMC overlay)")
```

#### Cell 2: Configuration (Markdown)

```markdown
## Configuration

Set your backtest parameters here:
```

#### Cell 3: Configuration

```python
# ============================================================
# CONFIGURATION
# ============================================================

# Symbol: "GER40" or "XAUUSD"
SYMBOL = "GER40"

# Initial balance
INITIAL_BALANCE = 100000.0

# Risk mode
RISK_PCT = 1.0  # 1% of equity per trade
RISK_AMOUNT = None

# Max margin
MAX_MARGIN_PCT = 40.0

# SMC Configuration
SMC_CONFIG = {
    "timeframes": ["M5", "M15", "H1"],
    "require_fvg_alignment": True,
    "require_bos_confirmation": True,
    "allow_entry_into_fvg": True,
    "confirmation_timeout_minutes": 30,
    "filter_enabled": True,
}

print(f"Symbol: {SYMBOL}")
print(f"Initial Balance: ${INITIAL_BALANCE:,.2f}")
print(f"Risk: {RISK_PCT}% per trade")
print(f"SMC Timeframes: {SMC_CONFIG['timeframes']}")
```

#### Cell 4: Initialize Engines

```python
# Initialize BASELINE engine (no SMC)
backtest_baseline = SingleDayBacktest(
    symbol=SYMBOL,
    initial_balance=INITIAL_BALANCE,
    risk_pct=RISK_PCT,
    risk_amount=RISK_AMOUNT,
    max_margin_pct=MAX_MARGIN_PCT,
)

# Initialize SMC engine
backtest_smc = SingleDayBacktestSMC(
    symbol=SYMBOL,
    initial_balance=INITIAL_BALANCE,
    risk_pct=RISK_PCT,
    risk_amount=RISK_AMOUNT,
    max_margin_pct=MAX_MARGIN_PCT,
    smc_config=SMC_CONFIG,
)

print(f"Baseline engine ready for {SYMBOL}")
print(f"SMC engine ready for {SYMBOL}")
print(f"Timezone: {backtest_baseline.timezone}")
```

#### Cell 5: Run Baseline (Markdown)

```markdown
## Run Baseline (No SMC)

First, run the baseline trade without SMC filter to see original behavior:
```

#### Cell 6: Run Baseline

```python
# ============================================================
# RUN BASELINE (NO SMC)
# ============================================================

TEST_DATE = "2025-05-29"

result_baseline = backtest_baseline.run_day(TEST_DATE)

if result_baseline["error"]:
    print(f"Error: {result_baseline['error']}")
else:
    print(f"\n{'='*50}")
    print(f"BASELINE (NO SMC) | DATE: {TEST_DATE}")
    print(f"{'='*50}")

    # IB Data
    if result_baseline["ib_data"]:
        ib = result_baseline["ib_data"]
        print(f"\nIB Levels:")
        print(f"  IBH: {ib['ibh']:.2f}")
        print(f"  IBL: {ib['ibl']:.2f}")
        print(f"  EQ:  {ib['eq']:.2f}")
        print(f"  IB Range: {ib['ibh'] - ib['ibl']:.2f} points")

    # Signals
    if result_baseline["signals"]:
        sig = result_baseline["signals"][0]
        print(f"\nSignal detected:")
        print(f"  Time: {sig['time'].strftime('%H:%M')} UTC")
        print(f"  Variation: {sig['variation']}")
        print(f"  Direction: {sig['direction'].upper()}")
        print(f"  Entry: {sig['entry_price']:.2f}")
        print(f"  SL: {sig['stop_loss']:.2f}")
        print(f"  TP: {sig['take_profit']:.2f}")

    # Trade
    if result_baseline["trade"]:
        trade = result_baseline["trade"]
        print(f"\nTrade Executed:")
        print(f"  Entry: {trade.entry_price:.2f} @ {trade.entry_time.strftime('%H:%M:%S')}")
        print(f"  Volume: {trade.volume:.2f} lots")
        print(f"  Exit: {trade.exit_price:.2f} @ {trade.exit_time.strftime('%H:%M:%S')}")
        print(f"  Exit Reason: {trade.exit_reason}")
        print(f"  P/L: ${trade.profit:.2f}")

        risk_money = RISK_AMOUNT or (INITIAL_BALANCE * RISK_PCT / 100)
        r_value = trade.profit / risk_money if risk_money > 0 else 0
        print(f"  R: {r_value:+.2f}")
    else:
        print("\nNo trade executed")
```

#### Cell 7: Run SMC (Markdown)

```markdown
## Run With SMC Filter

Now run the same date with SMC filter enabled:
```

#### Cell 8: Run SMC

```python
# ============================================================
# RUN WITH SMC FILTER
# ============================================================

result_smc = backtest_smc.run_day(TEST_DATE)

if result_smc["error"]:
    print(f"Error: {result_smc['error']}")
else:
    print(f"\n{'='*50}")
    print(f"WITH SMC FILTER | DATE: {TEST_DATE}")
    print(f"{'='*50}")

    # IB Data (same as baseline)
    if result_smc["ib_data"]:
        ib = result_smc["ib_data"]
        print(f"\nIB Levels:")
        print(f"  IBH: {ib['ibh']:.2f}")
        print(f"  IBL: {ib['ibl']:.2f}")
        print(f"  EQ:  {ib['eq']:.2f}")

    # Signals
    if result_smc["signals"]:
        sig = result_smc["signals"][0]
        print(f"\nSignal detected:")
        print(f"  Time: {sig['time'].strftime('%H:%M')} UTC")
        print(f"  Variation: {sig['variation']}")
        print(f"  Direction: {sig['direction'].upper()}")
        print(f"  Entry: {sig['entry_price']:.2f}")
        print(f"  SL: {sig['stop_loss']:.2f}")
        print(f"  TP: {sig['take_profit']:.2f}")

    # SMC Decision
    if result_smc["smc_decision"]:
        decision = result_smc["smc_decision"]
        print(f"\nSMC Decision:")
        print(f"  Action: {decision.action}")
        print(f"  Reason: {decision.reason}")
        print(f"  Score: {decision.score:.2f}")

        if decision.modified_stop_loss or decision.modified_take_profit:
            print(f"  Modifications:")
            if decision.modified_stop_loss:
                print(f"    SL: {sig['stop_loss']:.2f} -> {decision.modified_stop_loss:.2f}")
            if decision.modified_take_profit:
                print(f"    TP: {sig['take_profit']:.2f} -> {decision.modified_take_profit:.2f}")

    # Trade
    if result_smc["trade"]:
        trade = result_smc["trade"]
        print(f"\nTrade Executed:")
        print(f"  Entry: {trade.entry_price:.2f} @ {trade.entry_time.strftime('%H:%M:%S')}")
        print(f"  Volume: {trade.volume:.2f} lots")
        print(f"  Exit: {trade.exit_price:.2f} @ {trade.exit_time.strftime('%H:%M:%S')}")
        print(f"  Exit Reason: {trade.exit_reason}")
        print(f"  P/L: ${trade.profit:.2f}")

        risk_money = RISK_AMOUNT or (INITIAL_BALANCE * RISK_PCT / 100)
        r_value = trade.profit / risk_money if risk_money > 0 else 0
        print(f"  R: {r_value:+.2f}")
    else:
        print("\nNo trade executed (signal rejected or awaiting confirmation)")
```

#### Cell 9: Compare Results (Markdown)

```markdown
## Compare Baseline vs SMC

Side-by-side comparison:
```

#### Cell 10: Compare Results

```python
# ============================================================
# COMPARE BASELINE vs SMC
# ============================================================

def format_trade_summary(result, label):
    """Format trade result as dict for comparison."""
    if result["trade"]:
        trade = result["trade"]
        risk_money = RISK_AMOUNT or (INITIAL_BALANCE * RISK_PCT / 100)
        r_value = trade.profit / risk_money if risk_money > 0 else 0

        return {
            "Label": label,
            "Signal Time": result["signals"][0]["time"].strftime("%H:%M") if result["signals"] else "-",
            "Variation": trade.variation,
            "Direction": trade.direction.upper(),
            "Entry": f"{trade.entry_price:.2f}",
            "Exit": f"{trade.exit_price:.2f}",
            "Exit Reason": trade.exit_reason,
            "P/L": f"${trade.profit:.2f}",
            "R": f"{r_value:+.2f}",
        }
    else:
        return {
            "Label": label,
            "Signal Time": result["signals"][0]["time"].strftime("%H:%M") if result["signals"] else "-",
            "Variation": "-",
            "Direction": "-",
            "Entry": "-",
            "Exit": "-",
            "Exit Reason": result.get("error") or "No signal/Rejected",
            "P/L": "$0.00",
            "R": "0.00",
        }

# Create comparison table
comparison = pd.DataFrame([
    format_trade_summary(result_baseline, "BASELINE (No SMC)"),
    format_trade_summary(result_smc, "WITH SMC"),
])

print(f"\nComparison for {TEST_DATE}:")
display(comparison)

# Calculate difference
baseline_pl = result_baseline["trade"].profit if result_baseline["trade"] else 0
smc_pl = result_smc["trade"].profit if result_smc["trade"] else 0
diff_pl = smc_pl - baseline_pl

print(f"\nP/L Difference: ${diff_pl:+.2f}")

if diff_pl > 0:
    print("SMC filter IMPROVED the result")
elif diff_pl < 0:
    print("SMC filter DEGRADED the result (rejected profitable trade)")
else:
    print("SMC filter had NO IMPACT (same result)")
```

#### Cell 11: Chart Baseline (Markdown)

```markdown
## Chart: Baseline (No SMC)
```

#### Cell 12: Chart Baseline

```python
# Generate baseline chart (existing function)
fig_baseline = backtest_baseline.generate_chart(
    result_baseline,
    figsize=(16, 9),
    hours_after_ib=4.0
)

if fig_baseline:
    plt.show()
```

#### Cell 13: Chart SMC (Markdown)

```markdown
## Chart: WITH SMC Overlays

Enhanced chart showing FVG zones, fractals, BOS/CISD markers, and SMC decision:
```

#### Cell 14: Chart SMC

```python
# Generate SMC-enhanced chart (new function)
fig_smc = generate_chart_with_smc(
    result=result_smc,
    smc_registry=backtest_smc.strategy.smc_engine.registry,
    smc_decision=result_smc.get("smc_decision"),
    figsize=(16, 10),
    hours_after_ib=4.0,
)

if fig_smc:
    plt.show()
```

#### Cell 15: SMC Event Log (Markdown)

```markdown
## SMC Event Log

Chronological log of all SMC events (FVG formations, BOS detections, invalidations):
```

#### Cell 16: SMC Event Log

```python
# ============================================================
# SMC EVENT LOG
# ============================================================

if result_smc["smc_event_log"]:
    events = result_smc["smc_event_log"]

    events_df = pd.DataFrame([
        {
            "Time": e.timestamp.strftime("%H:%M:%S"),
            "Event Type": e.event_type,
            "Structure Type": e.structure_type or "-",
            "Timeframe": e.timeframe or "-",
            "Message": e.message,
        }
        for e in events
    ])

    print(f"\nSMC Event Log ({len(events)} events):")
    display(events_df)
else:
    print("No SMC events logged")
```

#### Cell 17: Registry Dump (Markdown)

```markdown
## Registry State at Signal Time

Snapshot of all active SMC structures when signal was detected:
```

#### Cell 18: Registry Dump

```python
# ============================================================
# REGISTRY DUMP AT SIGNAL TIME
# ============================================================

if result_smc["smc_registry_snapshot"]:
    snapshot = result_smc["smc_registry_snapshot"]

    print(f"\nActive Structures at Signal Time:")
    print(f"{'='*60}")

    for timeframe, structures in snapshot.items():
        print(f"\n{timeframe}:")

        # FVGs
        fvgs = structures.get("fvgs", [])
        if fvgs:
            print(f"  FVGs ({len(fvgs)}):")
            for fvg in fvgs:
                print(f"    - {fvg.direction} @ {fvg.low:.2f}-{fvg.high:.2f} (formed {fvg.formation_time.strftime('%H:%M')})")

        # BOS
        bos_list = structures.get("bos", [])
        if bos_list:
            print(f"  BOS ({len(bos_list)}):")
            for bos in bos_list:
                print(f"    - {bos.direction} @ {bos.price:.2f} (time {bos.time.strftime('%H:%M')})")

        # Fractals
        fractals = structures.get("fractals", [])
        if fractals:
            print(f"  Fractals ({len(fractals)}):")
            for frac in fractals:
                print(f"    - {frac.type} @ {frac.price:.2f} (time {frac.time.strftime('%H:%M')})")
else:
    print("No registry snapshot available")
```

---

## 3. Chart Overlays: generate_chart_with_smc()

### 3.1 Purpose

Extend existing `generate_chart()` function to overlay SMC structures on the price chart.

### 3.2 File Location

Add to existing file:
```
dual_v4/notebooks/debug_tools.py
```

### 3.3 Function Signature

```python
def generate_chart_with_smc(
    result: Dict[str, Any],
    smc_registry,
    smc_decision: Optional[Any] = None,
    figsize: Tuple[int, int] = (16, 10),
    hours_after_ib: float = 4.0,
) -> Optional[Any]:
    """
    Generate enhanced chart with SMC structure overlays.

    Overlays:
    - FVGs: shaded rectangles (green=bullish, red=bearish, alpha=0.2)
    - Fractals: triangle markers (up=high fractal, down=low fractal)
    - BOS/CISD: vertical dashed lines with labels
    - SMC decision: text box at signal time showing ENTER/WAIT/REJECT + reason
    - Modified SL/TP: if SMC changed entry/SL/TP, show both original and modified

    Args:
        result: Result dict from SingleDayBacktestSMC.run_day()
        smc_registry: SMCRegistry instance from SMCEngine
        smc_decision: SMCDecision instance (from result["smc_decision"])
        figsize: Figure size
        hours_after_ib: Limit chart to N hours after IB end

    Returns:
        matplotlib Figure or None
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.patches import Rectangle, FancyBboxPatch
    from matplotlib.lines import Line2D
    import pytz

    # Start with base chart (same as generate_chart)
    candles = result["candles"]
    trade = result["trade"]
    ib_data = result["ib_data"]
    signals = result["signals"]

    if candles.empty:
        print("No candles to chart")
        return None

    # Filter candles by time window (same as original)
    # ... (copy filtering logic from generate_chart)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Draw candlesticks (same as original)
    # ... (copy candlestick drawing logic)

    # Draw IB box and levels (same as original)
    # ... (copy IB drawing logic)

    # ============================================================
    # NEW: Draw SMC Structures
    # ============================================================

    # 1. Draw FVGs (shaded rectangles)
    for timeframe in smc_registry.instruments.get(result["symbol"], {}).keys():
        fvgs = smc_registry.get_active_structures(
            result["symbol"],
            timeframe,
            "fvg",
            current_time=candles["time"].iloc[-1]
        )

        for fvg in fvgs:
            # Calculate rectangle coordinates
            # X: from formation_time to end of chart
            # Y: from low to high

            x_start = fvg.formation_time
            x_end = candles["time"].iloc[-1]
            y_bottom = fvg.low
            y_height = fvg.high - fvg.low

            # Color: green for bullish, red for bearish
            color = "green" if fvg.direction == "bullish" else "red"

            # Draw rectangle
            rect = Rectangle(
                (mdates.date2num(x_start), y_bottom),
                mdates.date2num(x_end) - mdates.date2num(x_start),
                y_height,
                alpha=0.15,
                facecolor=color,
                edgecolor=color,
                linewidth=0.5,
                linestyle="--",
                label=f"FVG {timeframe} {fvg.direction}"
            )
            ax.add_patch(rect)

            # Add midpoint line (Fair Value Level)
            ax.axhline(
                y=fvg.midpoint,
                color=color,
                linestyle=":",
                linewidth=0.8,
                alpha=0.5,
            )

    # 2. Draw Fractals (triangle markers)
    for timeframe in smc_registry.instruments.get(result["symbol"], {}).keys():
        fractals = smc_registry.get_active_structures(
            result["symbol"],
            timeframe,
            "fractal",
            current_time=candles["time"].iloc[-1]
        )

        for frac in fractals:
            marker = "^" if frac.type == "low" else "v"
            color = "blue" if frac.type == "low" else "purple"

            ax.plot(
                frac.time,
                frac.price,
                marker=marker,
                color=color,
                markersize=8,
                markeredgecolor="black",
                markeredgewidth=0.5,
                label=f"Fractal {timeframe} {frac.type}"
            )

    # 3. Draw BOS/CISD (vertical dashed lines)
    for timeframe in smc_registry.instruments.get(result["symbol"], {}).keys():
        bos_list = smc_registry.get_active_structures(
            result["symbol"],
            timeframe,
            "bos",
            current_time=candles["time"].iloc[-1]
        )

        for bos in bos_list:
            ax.axvline(
                x=bos.time,
                color="orange",
                linestyle="--",
                linewidth=1.2,
                alpha=0.7,
                label=f"BOS {timeframe} {bos.direction}"
            )

            # Add text label
            ax.text(
                bos.time,
                bos.price,
                f"BOS {bos.direction.upper()}",
                rotation=90,
                verticalalignment="bottom",
                fontsize=8,
                color="orange",
                weight="bold"
            )

    # 4. Draw SMC Decision Box (at signal time)
    if smc_decision and signals:
        signal_time = signals[0]["time"]
        signal_price = signals[0]["entry_price"]

        # Decision text
        decision_text = f"SMC: {smc_decision.action}\n{smc_decision.reason}\nScore: {smc_decision.score:.2f}"

        # Color based on action
        if smc_decision.action == "ENTER":
            box_color = "green"
        elif smc_decision.action == "WAIT":
            box_color = "yellow"
        else:  # REJECT
            box_color = "red"

        # Draw fancy text box
        bbox_props = dict(
            boxstyle="round,pad=0.5",
            facecolor=box_color,
            alpha=0.3,
            edgecolor="black",
            linewidth=1.5
        )

        ax.text(
            signal_time,
            signal_price,
            decision_text,
            fontsize=9,
            weight="bold",
            bbox=bbox_props,
            verticalalignment="bottom",
            horizontalalignment="left"
        )

    # 5. Draw Modified SL/TP (if SMC changed them)
    if smc_decision and signals:
        original_sl = signals[0]["stop_loss"]
        original_tp = signals[0]["take_profit"]

        if smc_decision.modified_stop_loss:
            # Draw original SL (dashed)
            ax.axhline(
                y=original_sl,
                color="red",
                linestyle=":",
                linewidth=1.0,
                alpha=0.5,
                label="Original SL"
            )

            # Draw modified SL (solid)
            ax.axhline(
                y=smc_decision.modified_stop_loss,
                color="red",
                linestyle="-",
                linewidth=1.5,
                alpha=0.8,
                label="SMC Modified SL"
            )

        if smc_decision.modified_take_profit:
            # Draw original TP (dashed)
            ax.axhline(
                y=original_tp,
                color="green",
                linestyle=":",
                linewidth=1.0,
                alpha=0.5,
                label="Original TP"
            )

            # Draw modified TP (solid)
            ax.axhline(
                y=smc_decision.modified_take_profit,
                color="green",
                linestyle="-",
                linewidth=1.5,
                alpha=0.8,
                label="SMC Modified TP"
            )

    # Draw entry/exit markers (same as original)
    # ... (copy from generate_chart)

    # Legend (deduplicate labels)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="upper left", fontsize=8)

    # Formatting
    ax.set_title(f"{result['symbol']} - {candles['time'].iloc[0].strftime('%Y-%m-%d')} (WITH SMC)", fontsize=14, weight="bold")
    ax.set_xlabel("Time (UTC)", fontsize=10)
    ax.set_ylabel("Price", fontsize=10)
    ax.grid(True, alpha=0.3)

    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.autofmt_xdate()

    plt.tight_layout()
    return fig
```

### 3.4 Supporting Class: SingleDayBacktestSMC

Add to `debug_tools.py`:

```python
class SingleDayBacktestSMC(SingleDayBacktest):
    """
    Single-day backtest with SMC overlay.

    Extends SingleDayBacktest to use IBStrategySMC instead of IBStrategy.
    """

    def __init__(
        self,
        symbol: str = "GER40",
        initial_balance: float = 50000.0,
        risk_pct: Optional[float] = None,
        risk_amount: Optional[float] = None,
        max_margin_pct: float = 40.0,
        data_path: Optional[str] = None,
        timezone: Optional[str] = None,
        smc_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize SMC-aware single-day backtest.

        Args:
            symbol: Trading symbol
            initial_balance: Starting balance
            risk_pct: Risk percentage
            risk_amount: Fixed risk amount
            max_margin_pct: Max margin percentage
            data_path: Path to M1 data folder
            timezone: Instrument timezone
            smc_config: SMC configuration dict
        """
        super().__init__(
            symbol=symbol,
            initial_balance=initial_balance,
            risk_pct=risk_pct,
            risk_amount=risk_amount,
            max_margin_pct=max_margin_pct,
            data_path=data_path,
            timezone=timezone,
        )

        self.smc_config = smc_config

    def _setup(self):
        """Initialize emulator and load data with SMC strategy."""
        # Call parent setup (creates emulator, loads data, creates executor)
        super()._setup()

        # Replace strategy with SMC version
        import importlib
        import src.strategies.ib_strategy_smc as ib_strategy_smc_module
        importlib.reload(ib_strategy_smc_module)
        from src.strategies.ib_strategy_smc import IBStrategySMC
        from src.smc.config import SMCConfig
        from src.utils.strategy_logic import GER40_PARAMS_PROD, XAUUSD_PARAMS_PROD

        # Get params
        params = GER40_PARAMS_PROD if self.symbol == "GER40" else XAUUSD_PARAMS_PROD

        # Create SMC config
        if self.smc_config:
            smc_cfg = SMCConfig(
                instrument=self.symbol,
                timeframes=self.smc_config.get("timeframes", ["M5", "M15", "H1"]),
                rules=self.smc_config.get("rules", {}),
                confirmation_timeout_minutes=self.smc_config.get("confirmation_timeout_minutes", 30),
                filter_enabled=self.smc_config.get("filter_enabled", True),
            )
        else:
            smc_cfg = None  # Use defaults

        # Create SMC strategy
        self.strategy = IBStrategySMC(
            symbol=self.symbol,
            params=params,
            executor=self.executor,
            magic_number=1001,
            strategy_label="Debug_SMC",
            smc_config=smc_cfg,
        )

    def run_day(self, date) -> Dict[str, Any]:
        """
        Run backtest for a single day with SMC overlay.

        Returns dict with additional SMC fields:
        - smc_decision: SMCDecision instance
        - smc_event_log: List of SMCEvent
        - smc_registry_snapshot: Dict of active structures
        """
        # Run parent logic
        result = super().run_day(date)

        # Add SMC-specific data
        if hasattr(self.strategy, "smc_engine"):
            result["smc_decision"] = getattr(self.strategy.smc_engine, "last_decision", None)
            result["smc_event_log"] = list(self.strategy.smc_engine.event_log.events)

            # Snapshot registry at signal time
            if result["signals"]:
                signal_time = result["signals"][0]["time"]
                result["smc_registry_snapshot"] = self._capture_registry_snapshot(signal_time)
            else:
                result["smc_registry_snapshot"] = None

        return result

    def _capture_registry_snapshot(self, signal_time: datetime) -> Dict[str, Any]:
        """Capture snapshot of registry state at signal time."""
        snapshot = {}

        for timeframe in self.strategy.smc_engine.config.timeframes:
            snapshot[timeframe] = {
                "fvgs": self.strategy.smc_engine.registry.get_active_structures(
                    self.symbol, timeframe, "fvg", signal_time
                ),
                "bos": self.strategy.smc_engine.registry.get_active_structures(
                    self.symbol, timeframe, "bos", signal_time
                ),
                "fractals": self.strategy.smc_engine.registry.get_active_structures(
                    self.symbol, timeframe, "fractal", signal_time
                ),
            }

        return snapshot
```

---

## 4. Fast Backtest Integration Pattern

### 4.1 Architecture Difference

**Slow Backtest (IBStrategySMC):**
- Tick-by-tick iteration
- SMCEngine.update() called every bar
- Registry updated in real-time
- Used for: single-day debugging, live bot

**Fast Backtest (Vectorized):**
- Batch processing of entire date range
- Pre-compute SMC structures once per day
- Store in daily cache
- Evaluate signals against pre-computed structures
- Used for: parameter optimization (1000s of combinations)

### 4.2 Integration Approach

DO NOT modify `fast_backtest.py` in Phase 3. Instead, document the pattern:

```python
# In params_optimizer/engine/fast_backtest.py (future modification)

class FastBacktestEngineSMC(FastBacktestEngine):
    """
    Fast backtest with SMC pre-computation.

    Pre-computes SMC structures once per day, then evaluates signals against cache.
    """

    def __init__(self, m1_data, params, smc_config):
        super().__init__(m1_data, params)
        self.smc_config = smc_config
        self.smc_cache = {}  # date -> SMCRegistry snapshot

    def _precompute_smc_structures(self):
        """
        Pre-compute SMC structures for all days in dataset.

        For each day:
        1. Initialize TimeframeManager
        2. Run all detectors on all timeframes
        3. Store results in smc_cache[date]
        """
        pass

    def _evaluate_signal_with_smc(self, signal, date, time):
        """
        Evaluate signal against pre-computed SMC structures.

        Lookup smc_cache[date], apply SMC rules, return ENTER/WAIT/REJECT.
        """
        pass
```

### 4.3 Migration Path

1. Phase 3: Use slow backtest (IBStrategySMC) for debugging and validation
2. Phase 4: Implement fast backtest SMC integration for parameter optimization
3. Validate: Run same date range on both engines, verify results match

---

## 5. Execution Checklist

### 5.1 Development Order

**Step 1: IBStrategySMC** (Priority: HIGH)
- [ ] Create `src/strategies/ib_strategy_smc.py`
- [ ] Implement `__init__` with SMC engine initialization
- [ ] Implement `reset_daily_state()` override
- [ ] Implement `_check_signals_with_smc()` method
- [ ] Implement `_check_confirmation()` method
- [ ] Implement `_m2_to_m1_approximation()` helper
- [ ] Add logging at key decision points

**Step 2: Debug Tools Extensions** (Priority: HIGH)
- [ ] Add `SingleDayBacktestSMC` class to `debug_tools.py`
- [ ] Implement `_setup()` override to use IBStrategySMC
- [ ] Implement `run_day()` override to capture SMC data
- [ ] Implement `_capture_registry_snapshot()` helper
- [ ] Add `generate_chart_with_smc()` function
- [ ] Implement FVG overlay rendering
- [ ] Implement fractal marker rendering
- [ ] Implement BOS/CISD line rendering
- [ ] Implement SMC decision box rendering
- [ ] Implement modified SL/TP rendering

**Step 3: Debug Notebook** (Priority: MEDIUM)
- [ ] Create `notebooks/debug_smc.ipynb`
- [ ] Add setup cell (imports, path config)
- [ ] Add configuration cell
- [ ] Add initialization cells (baseline + SMC)
- [ ] Add baseline execution cells
- [ ] Add SMC execution cells
- [ ] Add comparison cell
- [ ] Add chart cells (baseline + SMC)
- [ ] Add event log cell
- [ ] Add registry dump cell

**Step 4: Testing** (Priority: HIGH)
- [ ] Test IBStrategySMC in isolation (single day, known signal)
- [ ] Verify SMC decision logic (ENTER/WAIT/REJECT)
- [ ] Verify AWAITING_CONFIRMATION state transitions
- [ ] Verify SL/TP modifications apply correctly
- [ ] Test debug notebook end-to-end
- [ ] Verify chart overlays render correctly
- [ ] Verify event log captures all events
- [ ] Compare baseline vs SMC on 10 test dates

**Step 5: Documentation** (Priority: MEDIUM)
- [ ] Add docstrings to all new classes/methods
- [ ] Add usage examples to IBStrategySMC docstring
- [ ] Add notebook cell comments
- [ ] Update CHANGELOG.md with Phase 3 completion
- [ ] Create example output screenshots

### 5.2 Testing Dates

Use these dates for validation (known behavior from V3):

**GER40:**
- 2023-01-05 (Reverse signal, profitable)
- 2023-01-06 (TCWE signal, SL hit)
- 2023-02-14 (OCAE signal, TP hit)
- 2023-03-22 (REV_RB signal, time exit)
- 2025-05-29 (TCWE signal from notebook example)

**XAUUSD:**
- 2023-01-09 (Reverse signal, profitable)
- 2023-01-12 (OCAE signal, SL hit)
- 2023-02-16 (TCWE signal, TP hit)

### 5.3 Success Criteria

**IBStrategySMC:**
- [ ] Can run full day without errors
- [ ] SMC decision logged for every signal
- [ ] AWAITING_CONFIRMATION state works (delays entry)
- [ ] Modified SL/TP applied when appropriate
- [ ] Parent IBStrategy logic unchanged (verified by baseline comparison)

**Debug Notebook:**
- [ ] Side-by-side comparison works (baseline vs SMC)
- [ ] Charts render with all SMC overlays
- [ ] Event log shows chronological SMC activity
- [ ] Registry snapshot shows active structures at signal time

**Integration:**
- [ ] Can swap IBStrategy -> IBStrategySMC with zero code changes (same constructor)
- [ ] Fast backtest integration pattern documented
- [ ] No modifications to production IBStrategy code

### 5.4 Performance Targets

**Slow Backtest (Single Day):**
- Execution time: < 5 seconds per day
- Memory usage: < 200MB for single day

**Debug Notebook:**
- Full notebook execution: < 30 seconds
- Chart generation: < 5 seconds per chart

---

## 6. Known Limitations and Workarounds

### 6.1 M1 Data Availability

**Issue:** BacktestExecutor provides M2 as minimum granularity. SMC engine needs M1.

**Workaround:** `_m2_to_m1_approximation()` duplicates each M2 bar into two M1 bars.

**Impact:** FVG detection may be less accurate (misses intra-M2 gaps).

**Future Fix:** Modify emulator to load and provide true M1 data.

### 6.2 TSL Interaction

**Issue:** IBStrategy uses virtual TP for TSL logic. SMC may modify TP.

**Workaround:** IBStrategySMC applies SMC modifications BEFORE TSL initialization.

**Impact:** TSL will use SMC-modified TP as baseline.

**Validation:** Verify TSL history in debug notebook.

### 6.3 Confirmation Timeout

**Issue:** AWAITING_CONFIRMATION state may expire during low-volatility periods.

**Workaround:** Configurable timeout (default 30 minutes).

**Impact:** May miss valid entries if confirmation takes too long.

**Tuning:** Adjust `confirmation_timeout_minutes` in SMCConfig.

---

## 7. File Manifest

**New Files:**
```
dual_v4/src/strategies/ib_strategy_smc.py           (~400 lines)
dual_v4/notebooks/debug_smc.ipynb                   (18 cells)
```

**Modified Files:**
```
dual_v4/notebooks/debug_tools.py                    (+300 lines)
  - Add: SingleDayBacktestSMC class
  - Add: generate_chart_with_smc() function
```

**Unchanged Files (Reference Only):**
```
dual_v4/src/strategies/ib_strategy.py               (MUST NOT modify)
dual_v4/notebooks/debug_strategy.ipynb              (MUST NOT modify)
dual_v4/params_optimizer/engine/fast_backtest.py    (Phase 4 integration)
```

---

## 8. Next Steps (Phase 4)

After Phase 3 completion:

1. **Validate SMC Rules**: Run debug notebook on 100+ historical dates, analyze SMC decisions
2. **Tune SMC Parameters**: Adjust FVG size thresholds, BOS requirements, etc.
3. **Fast Backtest Integration**: Implement SMC pre-computation for vectorized backtest
4. **Parameter Optimization**: Run fast backtest with SMC on/off, compare win rates
5. **Live Bot Integration**: Test IBStrategySMC in demo environment with real MT5 feed

---

## 9. References

**Architecture:**
- `plans/SMC_ARCHITECTURE.md` - Overall SMC design
- `plans/PHASE_1_FOUNDATION.md` - Detectors and registry
- Phase 2 spec (TBD) - CISD and market structure

**Existing Code:**
- `src/strategies/ib_strategy.py` - Production strategy FSM (lines 1-40KB)
- `notebooks/debug_tools.py` - SingleDayBacktest class (lines 35-500)
- `notebooks/debug_strategy.ipynb` - Baseline debug notebook pattern

**SMC Concepts:**
- `notion_export/FVG.md` - Fair Value Gaps
- `notion_export/BOS.md` - Break of Structure
- `notion_export/CISD.md` - Change In State of Delivery

---

**Document Status:** Ready for implementation
**Last Updated:** 2026-02-10
**Phase:** 3 of 4 (Integration)
**Estimated LOC:** ~700 new lines, ~300 modified lines
