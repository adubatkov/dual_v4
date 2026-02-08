#!/bin/bash
# Deploy and run parallel backtest on VMs
# Usage: ./deploy_parallel_backtest.sh

set -e

KEY="vm_optimizer_key"
BASE_DIR="$(dirname "$0")"
REMOTE_DIR="~/ib_trading_bot/dual_v4"

VMS=(
    "10.10.32.11:GER40:1:30"    # VM11: GER40 groups 1-30 (30 groups)
    "10.10.32.12:GER40:31:60"   # VM12: GER40 groups 31-60 (30 groups)
    "10.10.32.13:XAUUSD:1:23"   # VM13: XAUUSD groups 1-23 (23 groups)
    "10.10.32.14:XAUUSD:24:46"  # VM14: XAUUSD groups 24-46 (23 groups)
)

echo "=== Deploying Parallel Backtest ==="
echo "Key: $KEY"
echo ""

# Step 1: Copy updated files to all VMs
echo "=== Step 1: Copying files to VMs ==="

for VM in "${VMS[@]}"; do
    IFS=':' read -r IP SYMBOL START END <<< "$VM"
    echo "Copying to $IP..."

    # Copy backtest files
    scp -i "$BASE_DIR/$KEY" -o StrictHostKeyChecking=no \
        "$BASE_DIR/backtest/backtest_single_group.py" \
        "$BASE_DIR/backtest/run_parallel_backtest.py" \
        "$BASE_DIR/backtest/adapter.py" \
        "$BASE_DIR/backtest/config.py" \
        "$BASE_DIR/backtest/risk_manager.py" \
        "ubuntu@$IP:$REMOTE_DIR/backtest/"

    # Copy strategy files (with TSL fix and Signal class with use_virtual_tp)
    scp -i "$BASE_DIR/$KEY" -o StrictHostKeyChecking=no \
        "$BASE_DIR/src/strategies/base_strategy.py" \
        "$BASE_DIR/src/strategies/ib_strategy.py" \
        "ubuntu@$IP:$REMOTE_DIR/src/strategies/"

    # Copy emulator files
    scp -i "$BASE_DIR/$KEY" -o StrictHostKeyChecking=no \
        "$BASE_DIR/backtest/emulator/mt5_emulator.py" \
        "ubuntu@$IP:$REMOTE_DIR/backtest/emulator/"

    # Copy groups JSON
    scp -i "$BASE_DIR/$KEY" -o StrictHostKeyChecking=no \
        "$BASE_DIR/analyze/backtest_groups_$SYMBOL.json" \
        "ubuntu@$IP:$REMOTE_DIR/backtest/groups.json"

    echo "  -> Done"
done

echo ""
echo "=== Step 2: Launching backtests on VMs ==="

for VM in "${VMS[@]}"; do
    IFS=':' read -r IP SYMBOL START END <<< "$VM"
    echo "Launching on $IP for $SYMBOL (groups $START-$END)..."

    # Create filter script to select specific groups
    ssh -i "$BASE_DIR/$KEY" -o StrictHostKeyChecking=no "ubuntu@$IP" << EOF
cd $REMOTE_DIR
# Filter groups by index range
python3 -c "
import json
with open('backtest/groups.json') as f:
    groups = json.load(f)
selected = groups[$((START-1)):$END]
print(f'Selected {len(selected)} groups for $SYMBOL')
with open('backtest/groups_filtered.json', 'w') as f:
    json.dump(selected, f)
"
# Launch parallel backtest in background
nohup python3 backtest/run_parallel_backtest.py \
    --groups-file backtest/groups_filtered.json \
    --workers 20 \
    --start-date 2023-01-01 \
    --end-date 2025-10-31 \
    --output-dir backtest/output/parallel_${SYMBOL}_$(date +%Y%m%d) \
    > run_parallel_${SYMBOL}.log 2>&1 &
echo "Started with PID \$!"
EOF
    echo "  -> Launched"
done

echo ""
echo "=== Deployment Complete ==="
echo ""
echo "Monitor progress:"
echo "  ssh -i $KEY ubuntu@10.10.32.11 'tail -f ~/ib_trading_bot/dual_v4/run_parallel_GER40.log'"
echo "  ssh -i $KEY ubuntu@10.10.32.13 'tail -f ~/ib_trading_bot/dual_v4/run_parallel_XAUUSD.log'"
echo ""
echo "Check if running:"
echo "  for ip in 11 12 13 14; do ssh -i $KEY ubuntu@10.10.32.\$ip 'ps aux | grep python'; done"
