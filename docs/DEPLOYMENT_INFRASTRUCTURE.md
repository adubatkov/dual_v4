# Deployment Infrastructure

Last updated: 2026-02-08

## Table of Contents

- [Overview](#overview)
- [VM Infrastructure](#vm-infrastructure)
- [Work Distribution](#work-distribution)
- [Deployment Script](#deployment-script)
- [Parallel Backtest Execution](#parallel-backtest-execution)
- [Control Period Backtesting](#control-period-backtesting)
- [SSH Key Management](#ssh-key-management)
- [Monitoring and Progress Tracking](#monitoring-and-progress-tracking)
- [Collecting Results](#collecting-results)
- [Production Deployment](#production-deployment)
- [Troubleshooting](#troubleshooting)

---

## Overview

The parameter optimization pipeline requires running thousands of backtest
combinations across large datasets. To achieve reasonable execution times, the
workload is distributed across 4 Linux VMs, each running 20 parallel worker
processes. The deployment uses SSH/SCP for file transfer and remote execution.

---

## VM Infrastructure

### VM Specifications

| VM   | IP Address    | Role                    | Workers |
|------|--------------|-------------------------|---------|
| VM11 | 10.10.32.11  | GER40 groups 1-30       | 20      |
| VM12 | 10.10.32.12  | GER40 groups 31-60      | 20      |
| VM13 | 10.10.32.13  | XAUUSD groups 1-23      | 20      |
| VM14 | 10.10.32.14  | XAUUSD groups 24-46     | 20      |

All VMs run Ubuntu Linux and are accessed via SSH with key-based authentication.
Each VM has 24 CPU cores available; 20 are used for backtest workers to leave
headroom for the OS and monitoring.

### Remote Directory Layout

On each VM, the project is deployed to:

```
~/ib_trading_bot/dual_v3/
    backtest/
        run_parallel_backtest.py
        backtest_single_group.py
        adapter.py
        config.py
        risk_manager.py
        groups.json                  # Full groups file for the symbol
        groups_filtered.json         # Filtered subset for this VM
        emulator/
            mt5_emulator.py
        output/
            parallel_GER40_YYYYMMDD/  # Results output
    src/
        strategies/
            base_strategy.py
            ib_strategy.py
```

---

## Work Distribution

### Backtest Groups

Parameter combinations are organized into "groups." Each group represents a set
of related parameter configurations to be backtested. Groups are defined in
JSON files:

| File                              | Symbol | Total Groups |
|-----------------------------------|--------|-------------|
| `analyze/backtest_groups_GER40.json`  | GER40  | 60          |
| `analyze/backtest_groups_XAUUSD.json` | XAUUSD | 46          |

### Distribution Scheme

The groups are split evenly across VMs by index range:

```
GER40 (60 groups total):
    VM11 (10.10.32.11): groups[0:30]   -> groups 1-30
    VM12 (10.10.32.12): groups[30:60]  -> groups 31-60

XAUUSD (46 groups total):
    VM13 (10.10.32.13): groups[0:23]   -> groups 1-23
    VM14 (10.10.32.14): groups[23:46]  -> groups 24-46
```

The deploy script creates a `groups_filtered.json` on each VM containing only
that VM's assigned subset of groups.

---

## Deployment Script

### File

`dual_v3/deploy_parallel_backtest.sh`

### What It Does

The deployment script performs two sequential steps:

#### Step 1: File Transfer (SCP)

For each VM, the script copies the following files via SCP:

| Files Transferred | Remote Path |
|------------------|-------------|
| `backtest/backtest_single_group.py`, `run_parallel_backtest.py`, `adapter.py`, `config.py`, `risk_manager.py` | `~/ib_trading_bot/dual_v3/backtest/` |
| `src/strategies/base_strategy.py`, `src/strategies/ib_strategy.py` | `~/ib_trading_bot/dual_v3/src/strategies/` |
| `backtest/emulator/mt5_emulator.py` | `~/ib_trading_bot/dual_v3/backtest/emulator/` |
| `analyze/backtest_groups_{SYMBOL}.json` | `~/ib_trading_bot/dual_v3/backtest/groups.json` |

#### Step 2: Remote Execution (SSH)

For each VM, the script:

1. SSHs into the VM.
2. Filters the groups JSON to the VM's assigned index range using a Python one-liner.
3. Launches `run_parallel_backtest.py` in the background with `nohup`.

The launch command:

```bash
nohup python3 backtest/run_parallel_backtest.py \
    --groups-file backtest/groups_filtered.json \
    --workers 20 \
    --start-date 2023-01-01 \
    --end-date 2025-10-31 \
    --output-dir backtest/output/parallel_${SYMBOL}_$(date +%Y%m%d) \
    > run_parallel_${SYMBOL}.log 2>&1 &
```

### Usage

```bash
cd dual_v3
chmod +x deploy_parallel_backtest.sh
./deploy_parallel_backtest.sh
```

### Prerequisites

- SSH key file `vm_optimizer_key` must exist in the `dual_v3/` directory.
- The key must have proper permissions (`chmod 600` on Linux).
- VMs must be reachable on the network at 10.10.32.11-14.
- Python 3 with required packages must be installed on each VM.
- The base directory structure `~/ib_trading_bot/dual_v3/` must exist on each VM.

---

## Parallel Backtest Execution

### Main Runner

**File**: `backtest/run_parallel_backtest.py`

Orchestrates the parallel execution of backtest groups using Python's
`multiprocessing.Pool`.

#### Command-Line Arguments

| Argument         | Default       | Description |
|-----------------|---------------|-------------|
| `--symbol`      | (required)    | Symbol to backtest (GER40 or XAUUSD) |
| `--groups-file` | (alternative) | Path to groups JSON file |
| `--workers`     | 20            | Number of parallel workers |
| `--start-date`  | 2023-01-01    | Backtest start date |
| `--end-date`    | 2025-10-31    | Backtest end date |
| `--risk`        | 1000.0        | Fixed risk amount per trade |
| `--skip-charts` | False         | Skip equity chart generation (faster) |
| `--output-dir`  | auto-generated| Output directory path |

#### Execution Flow

1. Loads groups from JSON file (either `--groups-file` or auto-detected from `--symbol`).
2. Optionally filters groups by symbol.
3. Creates a `multiprocessing.Pool` with the specified number of workers.
4. Each worker receives one group and runs `run_single_group_backtest()` from `backtest_single_group.py`.
5. Results are collected via `imap_unordered` for progress reporting.
6. Progress is logged with ETA estimates based on average execution time.
7. Results are saved as CSV summary and full JSON.

#### Output Structure

```
backtest/output/parallel_GER40_20260208/
    results_summary_GER40.csv       # Top-level metrics per group
    results_full_GER40.json         # Complete results with all details
    failed_groups_GER40.json        # Groups that errored (for retry)
```

---

## Control Period Backtesting

### File

`backtest/run_parallel_backtest_control.py`

An identical runner to `run_parallel_backtest.py` but configured for the control
(out-of-sample) period:

| Parameter     | Main Runner         | Control Runner      |
|--------------|--------------------|--------------------|
| Start date   | 2023-01-01         | 2025-11-04         |
| End date     | 2025-10-31         | 2026-01-20         |
| Data source  | Original Parquet   | Control Parquet    |
| Import       | `backtest_single_group` | `backtest_single_group_control` |
| Output dir   | `parallel_{timestamp}_{symbol}` | `parallel_control_{symbol}` |

### Usage

```bash
python backtest/run_parallel_backtest_control.py \
    --groups-file backtest/groups_filtered.json \
    --workers 20 \
    --output-dir backtest/output/parallel_control_GER40
```

---

## SSH Key Management

### Key Files

| File                    | Location            | Type    |
|------------------------|---------------------|---------|
| `vm_optimizer_key`     | `dual_v3/`          | Private key |
| `vm_optimizer_key.pub` | `dual_v3/`          | Public key  |

Both files exist in the repository root (`dual_v3/`). The deploy script
references the private key with `-i "$BASE_DIR/$KEY"` where `KEY="vm_optimizer_key"`.

### Security Considerations

**[WARNING]** SSH private keys are currently stored in the repository directory.
This is a security risk. Recommended mitigations:

1. **Move keys to a secure location** outside the repository:
   ```
   ~/.ssh/vm_optimizer_key
   ~/.ssh/vm_optimizer_key.pub
   ```

2. **Ensure keys are in .gitignore** to prevent accidental commits:
   ```
   vm_optimizer_key
   vm_optimizer_key.pub
   ```

3. **Set proper permissions** on the private key:
   ```bash
   chmod 600 vm_optimizer_key
   ```

4. **Rotate keys periodically** and after any suspected compromise.

5. **Consider using SSH agent** instead of direct key file references.

The deploy script uses `StrictHostKeyChecking=no` to avoid interactive prompts.
This is acceptable for an internal network but should not be used in
production-facing environments.

---

## Monitoring and Progress Tracking

### View Real-Time Logs

```bash
# GER40 on VM11
ssh -i vm_optimizer_key ubuntu@10.10.32.11 \
    'tail -f ~/ib_trading_bot/dual_v3/run_parallel_GER40.log'

# XAUUSD on VM13
ssh -i vm_optimizer_key ubuntu@10.10.32.13 \
    'tail -f ~/ib_trading_bot/dual_v3/run_parallel_XAUUSD.log'
```

### Check if Processes Are Running

```bash
for ip in 11 12 13 14; do
    echo "=== VM$ip ==="
    ssh -i vm_optimizer_key ubuntu@10.10.32.$ip 'ps aux | grep python'
done
```

### Log Output Format

The parallel runner logs progress in this format:

```
2026-02-08 10:15:00 - INFO - [15/60] group_GER40_015: R=12.50, Trades=342, ETA: 45.2min
```

Fields:
- `[completed/total]`: Progress counter
- `group_id`: Identifier of the completed group
- `R`: Total R (profit in risk units)
- `Trades`: Number of trades executed
- `ETA`: Estimated time remaining based on average per-group time

---

## Collecting Results

### Download Results from VMs

```bash
# GER40 results from VM11
scp -i vm_optimizer_key -r \
    ubuntu@10.10.32.11:~/ib_trading_bot/dual_v3/backtest/output/parallel_GER40_* \
    ./backtest/output/

# GER40 results from VM12
scp -i vm_optimizer_key -r \
    ubuntu@10.10.32.12:~/ib_trading_bot/dual_v3/backtest/output/parallel_GER40_* \
    ./backtest/output/

# XAUUSD results from VM13
scp -i vm_optimizer_key -r \
    ubuntu@10.10.32.13:~/ib_trading_bot/dual_v3/backtest/output/parallel_XAUUSD_* \
    ./backtest/output/

# XAUUSD results from VM14
scp -i vm_optimizer_key -r \
    ubuntu@10.10.32.14:~/ib_trading_bot/dual_v3/backtest/output/parallel_XAUUSD_* \
    ./backtest/output/
```

### Result Files

Each VM produces:

| File                          | Content |
|-------------------------------|---------|
| `results_summary_{SYMBOL}.csv` | CSV with top-level metrics (total_r, sharpe, trades, winrate) sorted by total_r descending |
| `results_full_{SYMBOL}.json`   | Full JSON with all details, timestamps, and metadata |
| `failed_groups_{SYMBOL}.json`  | Groups that failed (if any), for retry |

### Retrying Failed Groups

If `failed_groups_{SYMBOL}.json` is non-empty, re-run with that file:

```bash
python backtest/run_parallel_backtest.py \
    --groups-file backtest/output/parallel_GER40_20260208/failed_groups_GER40.json \
    --workers 20
```

---

## Production Deployment

### Production Bot Location

The live trading bot runs at a separate location from the development workspace:

```
C:\Trading\5ers_ger_xau_trading_bot\
```

This is distinct from the development/optimization workspace at
`C:\Trading\ib_trading_bot\dual_v3\`.

### Key Differences

| Aspect | Development | Production |
|--------|------------|-----------|
| Path | `C:\Trading\ib_trading_bot\dual_v3\` | `C:\Trading\5ers_ger_xau_trading_bot\` |
| Purpose | Optimization, backtesting, analysis | Live trading on The5ers prop firm |
| Account | DEMO (for testing) | REAL (prop firm evaluation/funded) |
| Data | Historical CSV/Parquet | Real-time MT5 feed |
| News filter | Optional (for backtest comparison) | Mandatory (5ers compliance) |

### Deployment Checklist for Production Updates

When deploying optimized parameters or code changes to production:

1. Verify all backtests pass on both in-sample and control periods.
2. Run on DEMO account first and validate for at least one trading session.
3. Check `.env` file to confirm account configuration (DEMO vs REAL).
4. Verify news filter is enabled and news data is up to date.
5. Confirm stop-loss and risk parameters are within The5ers limits.
6. Monitor the first live session manually after deployment.

---

## Troubleshooting

### SSH Connection Refused

```
ssh: connect to host 10.10.32.11 port 22: Connection refused
```

Verify that the VM is running and SSH service is active. Check firewall rules
on the VM and network.

### Permission Denied (Public Key)

```
Permission denied (publickey)
```

1. Verify the private key file exists: `ls -la vm_optimizer_key`
2. Check permissions: `chmod 600 vm_optimizer_key`
3. Verify the public key is in `~/.ssh/authorized_keys` on the target VM.

### Workers Crashing with OOM

If workers are killed by the OOM killer, reduce the number of workers:

```bash
python backtest/run_parallel_backtest.py --workers 10
```

Monitor memory usage:

```bash
ssh -i vm_optimizer_key ubuntu@10.10.32.11 'free -h && top -b -n 1 | head -20'
```

### Partial Results

If a run is interrupted, the results collected so far are saved. Failed groups
are written to `failed_groups_{SYMBOL}.json` and can be retried independently.

### Network Timeout During SCP

If file transfers are slow or timing out, transfer files one at a time or
compress them first:

```bash
tar czf deploy_package.tar.gz backtest/ src/strategies/ analyze/
scp -i vm_optimizer_key deploy_package.tar.gz ubuntu@10.10.32.11:~/
ssh -i vm_optimizer_key ubuntu@10.10.32.11 \
    'cd ~/ib_trading_bot/dual_v3 && tar xzf ~/deploy_package.tar.gz'
```
