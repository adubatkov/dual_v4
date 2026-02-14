#!/usr/bin/env python3
"""
Run all 8 test configs through compare_engines.py and print summary table.

Usage:
    cd dual_v4
    python temp/run_all_configs.py
    python temp/run_all_configs.py --config 02  # run single config
"""

import sys
import os
import json
import re
import subprocess
import argparse
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).resolve().parent.parent
configs_dir = project_root / "temp" / "test_configs"


def find_configs(filter_num=None):
    """Find all JSON test configs, optionally filtered by number prefix."""
    configs = sorted(configs_dir.glob("*.json"))
    if filter_num:
        configs = [c for c in configs if c.name.startswith(f"{filter_num:02d}_") or c.name.startswith(f"{filter_num}_")]
    return configs


def run_config(config_path):
    """Run compare_engines.py with a params-file and parse results."""
    # Read config to get metadata
    with open(config_path) as f:
        params = json.load(f)

    start_date = params.get("_start_date", "2025-01-01")
    end_date = params.get("_end_date", "2025-04-01")
    symbol = params.get("_symbol", "GER40")
    description = params.get("_description", config_path.stem)

    cmd = [
        sys.executable,
        str(project_root / "temp" / "compare_engines.py"),
        "--symbol", symbol,
        "--start", start_date,
        "--end", end_date,
        "--params-file", str(config_path),
    ]

    print(f"\n{'='*80}")
    print(f"CONFIG: {config_path.name}")
    print(f"  {description}")
    print(f"  {symbol} | {start_date} to {end_date}")
    print(f"{'='*80}")

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    # Print output
    print(result.stdout)
    if result.stderr:
        # Only print non-empty stderr
        stderr_stripped = result.stderr.strip()
        if stderr_stripped:
            print(f"STDERR: {stderr_stripped[:500]}")

    # Parse summary from stdout
    parsed = parse_summary(result.stdout)
    parsed["config"] = config_path.stem
    parsed["description"] = description
    parsed["returncode"] = result.returncode

    return parsed


def parse_summary(output):
    """Parse compare_engines.py summary from stdout."""
    result = {
        "total_slow": 0,
        "total_fast": 0,
        "matched": 0,
        "slow_only": 0,
        "fast_only": 0,
        "r_mismatch": 0,
        "mean_abs_delta": 0.0,
        "passed": False,
    }

    for line in output.split("\n"):
        line = line.strip()
        if "Slow engine trades:" in line:
            result["total_slow"] = int(re.search(r"(\d+)", line.split(":")[-1]).group(1))
        elif "Fast engine trades:" in line:
            result["total_fast"] = int(re.search(r"(\d+)", line.split(":")[-1]).group(1))
        elif "Matched:" in line and "Slow" not in line:
            result["matched"] = int(re.search(r"(\d+)", line.split(":")[-1]).group(1))
        elif "Slow-only:" in line:
            result["slow_only"] = int(re.search(r"(\d+)", line.split(":")[-1]).group(1))
        elif "Fast-only:" in line:
            result["fast_only"] = int(re.search(r"(\d+)", line.split(":")[-1]).group(1))
        elif "R mismatches" in line:
            result["r_mismatch"] = int(re.search(r"(\d+)", line.split(":")[-1]).group(1))
        elif "Mean |dR|:" in line:
            result["mean_abs_delta"] = float(re.search(r"([\d.]+)", line.split(":")[-1]).group(1))
        elif "[PASS]" in line:
            result["passed"] = True
        elif "[FAIL]" in line:
            result["passed"] = False

    return result


def print_summary_table(results):
    """Print final summary table."""
    print("\n" + "=" * 100)
    print("FINAL SUMMARY: 8-CONFIG VALIDATION")
    print("=" * 100)

    header = f"{'#':<4} {'Config':<25} {'Slow':>5} {'Fast':>5} {'Match':>5} {'S_only':>6} {'F_only':>6} {'R_mis':>5} {'|dR|':>8} {'Status':<8}"
    print(header)
    print("-" * 100)

    total_pass = 0
    total_fail = 0

    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        if r["passed"]:
            total_pass += 1
        else:
            total_fail += 1

        num = r["config"][:2]
        name = r["config"][3:][:22]
        print(f"{num:<4} {name:<25} {r['total_slow']:>5} {r['total_fast']:>5} "
              f"{r['matched']:>5} {r['slow_only']:>6} {r['fast_only']:>6} "
              f"{r['r_mismatch']:>5} {r['mean_abs_delta']:>8.4f} {status:<8}")

    print("-" * 100)
    print(f"\nResult: {total_pass}/{total_pass + total_fail} PASS")

    if total_fail == 0:
        print("\n[PASS] All configs validated successfully")
    else:
        print(f"\n[FAIL] {total_fail} config(s) failed validation")

    return total_fail == 0


def main():
    parser = argparse.ArgumentParser(description="Run all test configs through compare_engines.py")
    parser.add_argument("--config", type=int, default=None,
                        help="Run single config by number (e.g., --config 2)")
    args = parser.parse_args()

    configs = find_configs(args.config)
    if not configs:
        print(f"No configs found in {configs_dir}")
        sys.exit(1)

    print(f"Found {len(configs)} test config(s) in {configs_dir}")
    for c in configs:
        print(f"  {c.name}")

    results = []
    for config_path in configs:
        try:
            result = run_config(config_path)
            results.append(result)
        except subprocess.TimeoutExpired:
            results.append({
                "config": config_path.stem,
                "description": "TIMEOUT",
                "total_slow": 0, "total_fast": 0, "matched": 0,
                "slow_only": 0, "fast_only": 0, "r_mismatch": 0,
                "mean_abs_delta": 0.0, "passed": False, "returncode": -1,
            })
        except Exception as e:
            results.append({
                "config": config_path.stem,
                "description": f"ERROR: {e}",
                "total_slow": 0, "total_fast": 0, "matched": 0,
                "slow_only": 0, "fast_only": 0, "r_mismatch": 0,
                "mean_abs_delta": 0.0, "passed": False, "returncode": -1,
            })

    all_passed = print_summary_table(results)
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
