"""
Day-of-week performance analysis for parameter groups.
Groups:
  A: GER40_055 + XAUUSD_059
  B: GER40_006 + XAUUSD_010
Datasets: TSL (parallel_results_tsl) and Control (parallel_results_control)
"""

import os
import pandas as pd
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))

FILES = {
    ("TSL", "A"): [
        os.path.join(BASE, "parallel_results_tsl", "vm11_GER40", "GER40_055", "trades.csv"),
        os.path.join(BASE, "parallel_results_tsl", "vm14_XAUUSD", "XAUUSD_059", "trades.csv"),
    ],
    ("TSL", "B"): [
        os.path.join(BASE, "parallel_results_tsl", "vm11_GER40", "GER40_006", "trades.csv"),
        os.path.join(BASE, "parallel_results_tsl", "vm13_XAUUSD", "XAUUSD_010", "trades.csv"),
    ],
    ("Control", "A"): [
        os.path.join(BASE, "parallel_results_control", "vm11_GER40", "GER40_055", "trades.csv"),
        os.path.join(BASE, "parallel_results_control", "vm14_XAUUSD", "XAUUSD_059", "trades.csv"),
    ],
    ("Control", "B"): [
        os.path.join(BASE, "parallel_results_control", "vm11_GER40", "GER40_006", "trades.csv"),
        os.path.join(BASE, "parallel_results_control", "vm13_XAUUSD", "XAUUSD_010", "trades.csv"),
    ],
}

GROUP_LABELS = {
    "A": "GER40_055 + XAUUSD_059",
    "B": "GER40_006 + XAUUSD_010",
}

DAY_NAMES = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri"}


def load_all() -> pd.DataFrame:
    frames = []
    for (dataset, group), paths in FILES.items():
        for path in paths:
            df = pd.read_csv(path, parse_dates=["entry_time", "exit_time"])
            df["dataset"] = dataset
            df["group"] = group
            df["day"] = df["entry_time"].dt.dayofweek
            frames.append(df)
    return pd.concat(frames, ignore_index=True)


def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for day in sorted(df["day"].unique()):
        d = df[df["day"] == day]
        wins = d[d["r"] > 0]
        losses = d[d["r"] <= 0]
        total = len(d)
        win_count = len(wins)
        loss_count = len(losses)
        winrate = (win_count / total * 100) if total > 0 else 0.0
        avg_r = d["r"].mean()
        total_r = d["r"].sum()
        avg_profit = d["profit"].mean()
        avg_win_r = wins["r"].mean() if len(wins) > 0 else 0.0
        avg_loss_r = losses["r"].mean() if len(losses) > 0 else 0.0
        sum_pos = wins["r"].sum()
        sum_neg = abs(losses["r"].sum())
        pf = (sum_pos / sum_neg) if sum_neg > 0 else float("inf")

        rows.append({
            "day": DAY_NAMES.get(day, str(day)),
            "trades": total,
            "wins": win_count,
            "losses": loss_count,
            "winrate%": round(winrate, 2),
            "avg_r": round(avg_r, 4),
            "total_r": round(total_r, 2),
            "avg_profit": round(avg_profit, 2),
            "avg_win_r": round(avg_win_r, 4),
            "avg_loss_r": round(avg_loss_r, 4),
            "profit_factor": round(pf, 3),
        })
    return pd.DataFrame(rows)


def print_table(title: str, df: pd.DataFrame):
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}")
    print(df.to_string(index=False))
    print()


def main():
    data = load_all()

    # Per-dataset, per-group tables
    for dataset in ["TSL", "Control"]:
        for group in ["A", "B"]:
            subset = data[(data["dataset"] == dataset) & (data["group"] == group)]
            if subset.empty:
                continue
            metrics = compute_metrics(subset)
            label = GROUP_LABELS[group]
            print_table(f"{label}  |  {dataset}  |  trades: {len(subset)}", metrics)

    # Summary: TSL vs Control side-by-side for each group
    print(f"\n{'#' * 80}")
    print("  SUMMARY: TSL vs Control comparison (winrate% / avg_r / total_r)")
    print(f"{'#' * 80}")

    for group in ["A", "B"]:
        label = GROUP_LABELS[group]
        print(f"\n--- {label} ---")
        header = f"{'Day':<5} | {'TSL WR%':>8} {'TSL avgR':>9} {'TSL totR':>9} | {'CTL WR%':>8} {'CTL avgR':>9} {'CTL totR':>9} | {'dWR%':>6} {'dAvgR':>7}"
        print(header)
        print("-" * len(header))

        for day_num in range(5):
            day_name = DAY_NAMES[day_num]
            vals = {}
            for ds in ["TSL", "Control"]:
                subset = data[(data["dataset"] == ds) & (data["group"] == group) & (data["day"] == day_num)]
                total = len(subset)
                wins = len(subset[subset["r"] > 0])
                wr = (wins / total * 100) if total > 0 else 0.0
                ar = subset["r"].mean() if total > 0 else 0.0
                tr = subset["r"].sum() if total > 0 else 0.0
                vals[ds] = (wr, ar, tr)

            tsl = vals["TSL"]
            ctl = vals["Control"]
            dwr = ctl[0] - tsl[0]
            dar = ctl[1] - tsl[1]
            print(f"{day_name:<5} | {tsl[0]:>7.2f}% {tsl[1]:>9.4f} {tsl[2]:>9.2f} | {ctl[0]:>7.2f}% {ctl[1]:>9.4f} {ctl[2]:>9.2f} | {dwr:>+6.2f} {dar:>+7.4f}")

    print()


if __name__ == "__main__":
    main()
