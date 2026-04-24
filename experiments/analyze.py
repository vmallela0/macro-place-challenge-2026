"""experiments/analyze.py
Print per-experiment mean/std vs baseline from experiments/results.csv.

Usage:
    python experiments/analyze.py [--bench ibm01] [--budget 550]

By default shows all benches at all budgets. Highlights the delta
(exp - baseline) and whether it clears the ±σ noise floor.
"""
import argparse
import csv
import statistics
import sys
from pathlib import Path
from collections import defaultdict

CSV_PATH = Path(__file__).resolve().parent / "results.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", help="filter to one benchmark")
    ap.add_argument("--budget", type=int, help="filter to one budget_s")
    ap.add_argument("--exp", help="filter to one exp_id pattern (substring)")
    args = ap.parse_args()

    if not CSV_PATH.exists():
        print(f"No results.csv yet at {CSV_PATH}")
        sys.exit(1)

    # group by (exp_id, benchmark, budget_s)
    groups = defaultdict(list)
    with open(CSV_PATH) as f:
        for row in csv.DictReader(f):
            if args.bench and row["benchmark"] != args.bench:
                continue
            if args.budget and int(row["budget_s"]) != args.budget:
                continue
            if args.exp and args.exp not in row["exp_id"]:
                continue
            try:
                cost = float(row["final_cost"])
            except (ValueError, TypeError):
                continue  # FAIL rows
            groups[(row["exp_id"], row["benchmark"], row["budget_s"])].append({
                "seed": row["seed"], "cost": cost,
                "wl": row.get("hpwl"), "den": row.get("density"),
                "cong": row.get("congestion"),
                "wall": row.get("wall_clock_s"), "branch": row.get("branch"),
            })

    # baselines by (benchmark, budget_s)
    baselines = {}
    for (exp_id, bench, bud), rows in groups.items():
        if exp_id == "baseline":
            costs = [r["cost"] for r in rows]
            baselines[(bench, bud)] = {
                "mean": statistics.mean(costs),
                "std": statistics.pstdev(costs) if len(costs) > 1 else 0.0,
                "n": len(costs),
            }

    # print
    header = f"{'exp_id':<22} {'bench':<6} {'bud':>5} {'n':>2} {'mean':>9} {'std':>7} {'Δ vs base':>10} {'sig?':>5}"
    print(header)
    print("-" * len(header))
    for (exp_id, bench, bud), rows in sorted(groups.items()):
        costs = [r["cost"] for r in rows]
        mean = statistics.mean(costs)
        std = statistics.pstdev(costs) if len(costs) > 1 else 0.0
        base = baselines.get((bench, bud))
        if base is None or exp_id == "baseline":
            delta = ""
            sig = ""
        else:
            d = mean - base["mean"]
            delta = f"{d:+.4f}"
            # combined std (treat pooled)
            comb = (std**2 + base["std"]**2) ** 0.5
            sig = "Y" if (abs(d) > 2 * comb and abs(d) > 0.001) else ""
        print(f"{exp_id:<22} {bench:<6} {bud:>5} {len(costs):>2} {mean:>9.4f} {std:>7.4f} {delta:>10} {sig:>5}")


if __name__ == "__main__":
    main()
