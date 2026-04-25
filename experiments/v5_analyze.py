#!/usr/bin/env python3
"""experiments/v5_analyze.py

Summarize v5_* sweep results from experiments/results.csv against the
optimized_v4 baseline (from submissions/vmallela_v2/results_verified_v4/).

Usage:
    python experiments/v5_analyze.py
    python experiments/v5_analyze.py --csv experiments/results.csv

Output:
    Per-bench table (bench, v4, v5_each, delta) + per-branch mean delta.
    Highlights wins (delta < -0.005), losses (delta > +0.005), no-change.
"""
import argparse
import csv
from pathlib import Path
from collections import defaultdict
import re


REPO_ROOT = Path(__file__).resolve().parent.parent
V4_LOGS = REPO_ROOT / "submissions/vmallela_v2/results_verified_v4"
BENCHES = ["ibm01", "ibm02", "ibm03", "ibm04", "ibm06", "ibm07", "ibm08",
           "ibm09", "ibm10", "ibm11", "ibm12", "ibm13", "ibm14", "ibm15",
           "ibm16", "ibm17", "ibm18"]

V5_BRANCHES = [
    "v5_cells_skip",
    "v5_escape_v2",
    "v5_surrogate_struct",
    "v5_warmstart",
    "v5_combined",
    "v5_combined_pp",
]


def parse_v4_baseline():
    """Read v4 logs, return {bench: cost}."""
    out = {}
    for bench in BENCHES:
        log = V4_LOGS / f"{bench}.log"
        if not log.exists():
            out[bench] = None
            continue
        text = log.read_text()
        m = re.search(r"^proxy=([0-9.]+).*VALID", text, re.MULTILINE)
        out[bench] = float(m.group(1)) if m else None
    return out


def parse_csv(csv_path):
    """Return {(exp_id, bench, seed, budget): final_cost}, latest entry wins."""
    out = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                cost = float(row["final_cost"]) if row["final_cost"] else None
            except ValueError:
                cost = None
            key = (row["exp_id"], row["benchmark"], row["seed"], row["budget_s"])
            if cost is not None:
                out[key] = cost
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(REPO_ROOT / "experiments/results.csv"))
    ap.add_argument("--seed", default="42")
    ap.add_argument("--budget", default="3300")
    args = ap.parse_args()

    v4 = parse_v4_baseline()
    csv_data = parse_csv(args.csv)

    # Per-bench table.
    print(f"\nv5 sweep summary  (seed={args.seed}, budget={args.budget})\n")
    header = ["bench", "v4_baseline"] + V5_BRANCHES
    widths = [8, 12] + [14] * len(V5_BRANCHES)
    print("  ".join(h.ljust(w) for h, w in zip(header, widths)))
    print("  ".join("-" * w for w in widths))

    branch_deltas = defaultdict(list)
    branch_costs = defaultdict(list)
    for bench in BENCHES:
        v4_cost = v4.get(bench)
        row = [bench, f"{v4_cost:.4f}" if v4_cost else "—"]
        for branch in V5_BRANCHES:
            key = (branch, bench, args.seed, args.budget)
            c = csv_data.get(key)
            if c is None:
                row.append("—")
            else:
                if v4_cost is not None:
                    delta = c - v4_cost
                    sign = "+" if delta >= 0 else ""
                    marker = ""
                    if delta < -0.005: marker = " ↓"
                    elif delta > 0.005: marker = " ↑"
                    row.append(f"{c:.4f} ({sign}{delta:.3f}){marker}")
                    branch_deltas[branch].append(delta)
                    branch_costs[branch].append(c)
                else:
                    row.append(f"{c:.4f}")
                    branch_costs[branch].append(c)
        print("  ".join(s.ljust(w) for s, w in zip(row, widths)))

    print()
    print("Per-branch summary:")
    print(f"  {'branch':22s}  {'n_done':>6}  {'mean_cost':>10}  {'mean_delta':>11}  {'wins':>5}/{'loss':5}")
    v4_complete = [v4[b] for b in BENCHES if v4[b] is not None]
    v4_mean = sum(v4_complete) / len(v4_complete) if v4_complete else None
    print(f"  {'optimized_v4 (baseline)':22s}  {len(v4_complete):>6}  "
          f"{(v4_mean or 0):>10.4f}  {'(ref)':>11}  {'—':>5}/{'—':5}")
    for branch in V5_BRANCHES:
        deltas = branch_deltas[branch]
        costs = branch_costs[branch]
        if not costs:
            print(f"  {branch:22s}  {0:>6}  {'—':>10}  {'—':>11}  {'—':>5}/{'—':5}")
            continue
        mean_cost = sum(costs) / len(costs)
        mean_delta = sum(deltas) / len(deltas) if deltas else 0
        wins = sum(1 for d in deltas if d < -0.005)
        loss = sum(1 for d in deltas if d > 0.005)
        sign = "+" if mean_delta >= 0 else ""
        print(f"  {branch:22s}  {len(costs):>6}  {mean_cost:>10.4f}  "
              f"{sign}{mean_delta:>10.4f}  {wins:>5}/{loss:5}")
    print()


if __name__ == "__main__":
    main()
