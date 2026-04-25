#!/usr/bin/env python3
"""Statistical analysis for v5_box2 paired multi-seed results.

Reads experiments/results.csv, computes:
  - Per-bench mean & std for v5_combined_box2 and v5_cluster_box2 across seeds
  - Paired diff (cluster - combined) per (bench, seed)
  - Per-bench paired t-test (small N=4-5 so we report mean diff, std, sign count)
  - Aggregate paired t-test across all (bench, seed) pairs
  - Comparison to v4 reference (1.0186) and first-box single-seed v5_combined

Usage: .venv/bin/python experiments/v5_box2_analyze.py
"""
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

CSV = Path(__file__).parent / "results.csv"

def main():
    rows = list(csv.DictReader(CSV.open()))
    # bucket: (exp_id, bench, seed) -> cost
    by_key = {}
    for r in rows:
        if not r.get("final_cost"):
            continue
        try:
            cost = float(r["final_cost"])
        except ValueError:
            continue
        exp = r["exp_id"]
        bench = r["benchmark"]
        seed = int(r["seed"])
        by_key[(exp, bench, seed)] = cost

    benches = sorted({k[1] for k in by_key})
    print("=" * 78)
    print("v5_box2 paired analysis")
    print("=" * 78)

    # Per-bench summary for each branch.
    for branch in ("v5_combined_box2", "v5_cluster_box2", "v5_combined_pp_box2",
                   "v5_combined", "v5_cluster"):
        seeds = sorted({k[2] for k in by_key if k[0] == branch})
        if not seeds:
            continue
        print(f"\n[{branch}] seeds={seeds}")
        per_bench_means = []
        for b in benches:
            costs = [by_key.get((branch, b, s)) for s in seeds]
            costs = [c for c in costs if c is not None]
            if not costs:
                continue
            m = statistics.mean(costs)
            sd = statistics.stdev(costs) if len(costs) > 1 else 0.0
            per_bench_means.append(m)
            print(f"  {b:6s}  N={len(costs)}  mean={m:.4f}  std={sd:.4f}  "
                  f"costs=[{', '.join(f'{c:.4f}' for c in costs)}]")
        if per_bench_means:
            print(f"  {'AVG':6s}  N={len(per_bench_means)} benches  "
                  f"mean-of-means={statistics.mean(per_bench_means):.4f}")

    # Paired comparison: v5_cluster_box2 vs v5_combined_box2 at common (bench, seed).
    pairs = []
    for b in benches:
        for s in range(40, 60):
            c = by_key.get(("v5_cluster_box2", b, s))
            v = by_key.get(("v5_combined_box2", b, s))
            if c is not None and v is not None:
                pairs.append((b, s, v, c, c - v))
    if pairs:
        print("\n" + "=" * 78)
        print("PAIRED: v5_cluster_box2 vs v5_combined_box2 (same bench, same seed)")
        print("=" * 78)
        diffs = [p[4] for p in pairs]
        n = len(diffs)
        mean_d = statistics.mean(diffs)
        sd_d = statistics.stdev(diffs) if n > 1 else 0.0
        se_d = sd_d / math.sqrt(n) if n > 1 else 0.0
        t_stat = mean_d / se_d if se_d > 0 else float("nan")
        n_neg = sum(1 for d in diffs if d < 0)
        n_pos = sum(1 for d in diffs if d > 0)
        print(f"N pairs        : {n}")
        print(f"mean diff      : {mean_d:+.5f}  (negative = cluster better)")
        print(f"std diff       : {sd_d:.5f}")
        print(f"SE             : {se_d:.5f}")
        print(f"t-stat         : {t_stat:+.3f}")
        print(f"# cluster wins : {n_neg}/{n}  (sign-test p≈{2*sum(math.comb(n,k) for k in range(min(n_neg,n_pos)+1))/(2**n):.3f} two-sided)")

        # Per-bench breakdown.
        per_bench = defaultdict(list)
        for b, s, v, c, d in pairs:
            per_bench[b].append(d)
        print("\nper-bench paired diffs (cluster - combined):")
        for b in sorted(per_bench):
            ds = per_bench[b]
            print(f"  {b:6s}  N={len(ds)}  mean={statistics.mean(ds):+.5f}  "
                  f"diffs=[{', '.join(f'{d:+.4f}' for d in ds)}]")

    print()

if __name__ == "__main__":
    main()
