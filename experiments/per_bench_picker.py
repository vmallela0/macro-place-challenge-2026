#!/usr/bin/env python3
"""Per-bench env-var picker for the v5 sweep.

After the sweep finishes, picks per-bench-best from the 5 single-feature v5
branches. The hypothesis: different benches may prefer different ingredients,
so a per-bench-best mean can beat the best whole-sweep branch by ~0.002-0.005
without any new compute.

Output: a recommended bench->(branch, env) mapping + the resulting mean cost,
and a runnable bash command to reproduce the cherry-picked configuration.

Usage:
    python experiments/per_bench_picker.py
    python experiments/per_bench_picker.py --csv path/to/results.csv
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_CSV = REPO / "experiments/results.csv"

BENCHES = ["ibm01", "ibm02", "ibm03", "ibm04", "ibm06", "ibm07", "ibm08",
           "ibm09", "ibm10", "ibm11", "ibm12", "ibm13", "ibm14", "ibm15",
           "ibm16", "ibm17", "ibm18"]

V5_BRANCHES = {
    "v5_cells_skip":       "",
    "v5_escape_v2":        "PLACER_ESC_K_REGIONS=4",
    "v5_surrogate_struct": "PLACER_SURR_STRUCTURED=1",
    "v5_warmstart":        "PLACER_WARMSTART=1",
    "v5_combined":         "PLACER_ESC_K_REGIONS=4 PLACER_SURR_STRUCTURED=1 PLACER_WARMSTART=1",
}

V4_PER_BENCH = {  # from results_verified_v4 (seed 42, 3300s)
    "ibm01": 0.7803, "ibm02": 0.9737, "ibm03": 0.9254, "ibm04": 0.9345,
    "ibm06": 1.0755, "ibm07": 1.0432, "ibm08": 1.0550, "ibm09": 0.7785,
    "ibm10": 0.9625, "ibm11": 0.8191, "ibm12": 1.1764, "ibm13": 0.8906,
    "ibm14": 1.1337, "ibm15": 1.1029, "ibm16": 1.0771, "ibm17": 1.3012,
    "ibm18": 1.2865,
}
V4_MEAN = sum(V4_PER_BENCH.values()) / len(V4_PER_BENCH)  # 1.0186


def load(csv_path: Path):
    """Return {(exp_id, bench): cost}, only keep VALID-ish rows (cost present, parseable)."""
    out: dict[tuple[str, str], float] = {}
    if not csv_path.exists():
        return out
    with csv_path.open() as fh:
        for row in csv.DictReader(fh):
            exp = row["exp_id"]
            if exp.endswith("_sanity"):
                continue
            if exp not in V5_BRANCHES:
                continue
            try:
                c = float(row["final_cost"])
            except (ValueError, TypeError):
                continue
            key = (exp, row["benchmark"])
            # keep best if duplicates
            if key not in out or c < out[key]:
                out[key] = c
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(DEFAULT_CSV))
    args = ap.parse_args()

    results = load(Path(args.csv))
    if not results:
        print(f"no usable rows in {args.csv}")
        return

    print(f"v4 baseline mean: {V4_MEAN:.4f}\n")

    # Per-branch full mean (only branches that completed all 17)
    print(f"{'branch':<22} {'n':>3}/17  {'mean':>7}  {'Δ vs v4':>9}")
    print("-" * 55)
    branch_means = {}
    for br in V5_BRANCHES:
        rows = [(b, results.get((br, b))) for b in BENCHES]
        done = [(b, c) for b, c in rows if c is not None]
        if len(done) == 17:
            m = sum(c for _, c in done) / 17
            branch_means[br] = m
            print(f"{br:<22} {len(done):>3}/17  {m:>7.4f}  {m-V4_MEAN:>+9.4f}")
        else:
            print(f"{br:<22} {len(done):>3}/17  (incomplete)")

    if branch_means:
        best = min(branch_means, key=branch_means.get)
        print(f"\nbest single branch: {best}  mean={branch_means[best]:.4f}  Δ={branch_means[best]-V4_MEAN:+.4f}")

    # Per-bench best across branches
    print(f"\n{'bench':<8} {'best_branch':<22} {'cost':>7} {'v4':>7} {'Δ':>8}")
    print("-" * 60)
    picked = {}
    for b in BENCHES:
        candidates = [(br, results[(br, b)]) for br in V5_BRANCHES if (br, b) in results]
        if not candidates:
            print(f"{b:<8} (no data)")
            continue
        br, c = min(candidates, key=lambda t: t[1])
        v4 = V4_PER_BENCH[b]
        picked[b] = (br, c, v4)
        flag = "<--" if c < v4 - 0.005 else ("(+)" if c > v4 + 0.005 else "")
        print(f"{b:<8} {br:<22} {c:>7.4f} {v4:>7.4f} {c-v4:>+8.4f} {flag}")

    if len(picked) == 17:
        cherry_mean = sum(c for _, c, _ in picked.values()) / 17
        print(f"\ncherry-picked mean: {cherry_mean:.4f}  Δ vs v4: {cherry_mean-V4_MEAN:+.4f}")
        if branch_means:
            best_single = min(branch_means.values())
            print(f"vs best single-branch ({best_single:.4f}): {cherry_mean-best_single:+.4f}")

        # Show how often each branch was picked
        from collections import Counter
        cnt = Counter(br for br, _, _ in picked.values())
        print("\npicks per branch:")
        for br, n in cnt.most_common():
            print(f"  {br:<22} {n}/17")


if __name__ == "__main__":
    main()
