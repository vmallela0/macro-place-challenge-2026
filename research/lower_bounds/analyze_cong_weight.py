"""Analyze cong_weight sensitivity sweep on a single bench.

Reads /tmp/albania1_cong_weight_*/results.csv + per-weight log files
and computes:

1. Per-weight: proxy, Δ vs cong-off baseline, Hessian λ_min, Hessian
   step chosen by line search, Hessian lift (post-Lap → post-Hessian).
2. Correlation: does proxy vary monotonically with cong_weight?
3. Optimal weight: argmin proxy across the swept weights.
4. Saddle-depth hypothesis check: λ_min(w=0) vs optimal weight.

Run:
    .venv/bin/python research/lower_bounds/analyze_cong_weight.py
"""
from __future__ import annotations
import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def parse_log(path):
    """Extract Hessian phase signals from a placer log."""
    info = {"lambda_min": None, "step": None,
              "post_portfolio": None, "post_lap": None,
              "post_hess": None, "final": None,
              "lifted": False}
    with open(path) as f:
        for line in f:
            m = re.search(r"after portfolio\+consensus: cost=([0-9.]+)", line)
            if m: info["post_portfolio"] = float(m.group(1))
            m = re.search(r"laplacian: post-cost=([0-9.]+)", line)
            if m: info["post_lap"] = float(m.group(1))
            m = re.search(r"hessian: λ_min=([+-]?[0-9.e+-]+)", line)
            if m: info["lambda_min"] = float(m.group(1))
            m = re.search(r"adaptive\] e\d+: λ=([+-]?[0-9.e+-]+) best_step=([+-]?[0-9.]+)",
                          line)
            if m:
                info["lambda_min"] = float(m.group(1))
                info["step"] = float(m.group(2))
            m = re.search(r"HESSIAN WIN: step=[+-]?[0-9.]+ cost=([0-9.]+)", line)
            if m:
                info["post_hess"] = float(m.group(1))
                info["lifted"] = True
            m = re.search(r"DONE: cost=([0-9.]+)", line)
            if m: info["final"] = float(m.group(1))
    return info


def main():
    sweep_dirs = sorted(Path("/tmp").glob("albania1_cong_weight_*"))
    if not sweep_dirs:
        print("no sweep dir found")
        return
    out = sweep_dirs[-1]
    print(f"sweep dir: {out}")
    results_csv = out / "results.csv"
    if not results_csv.exists():
        print("no results.csv yet")
        return
    rows = list(csv.DictReader(open(results_csv)))
    print(f"\n{'weight':>8} {'proxy':>8} {'cong':>8} {'λ_min':>11} "
          f"{'step':>8} {'post-Lap':>9} {'post-Hess':>10} {'lift':>8}")
    proxies = {}
    for r in rows:
        w = r["weight"]
        log_path = out / f"w{w}.log"
        info = parse_log(log_path) if log_path.exists() else {}
        proxy = float(r["proxy_cost"]) if r["proxy_cost"] not in ("NA", "") else None
        cong = float(r["congestion_cost"]) if r["congestion_cost"] not in ("NA", "") else None
        if proxy is not None:
            proxies[float(w)] = proxy
        lam = info.get("lambda_min")
        step = info.get("step")
        plap = info.get("post_lap")
        phess = info.get("post_hess")
        lift = (plap - phess) if plap and phess else None
        def f(x, fmt=".4f"):
            return f"{x:{fmt}}" if x is not None else "—"
        print(f"{w:>8} {f(proxy):>8} {f(cong, '.3f'):>8} "
              f"{f(lam, '.2e'):>11} {f(step, '+.4f'):>8} "
              f"{f(plap, '.4f'):>9} {f(phess, '.4f'):>10} "
              f"{f(lift, '+.4f'):>8}")

    if proxies:
        # baseline (cong-off, w=0.0)
        baseline = proxies.get(0.0)
        print()
        print(f"baseline (w=0.0): {baseline}")
        for w, p in sorted(proxies.items()):
            d = (p - baseline) if baseline else None
            star = " ★" if w in proxies and p == min(proxies.values()) else ""
            print(f"  w={w}: proxy={p:.4f}  Δ={d:+.4f}{star}" if d
                  else f"  w={w}: proxy={p:.4f}{star}")
        opt_w = min(proxies, key=lambda k: proxies[k])
        print(f"\nOPTIMAL weight: {opt_w} (proxy={proxies[opt_w]:.4f})")


if __name__ == "__main__":
    main()
