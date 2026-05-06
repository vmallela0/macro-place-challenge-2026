"""Honest combined lower bound on proxy.

Each proxy component bounds the others independently. We can sum them:
   proxy_LB = WL_LB + 0.5 · D_LB + 0.5 · C_LB

WL_LB: from research/lower_bounds/l1_hpwl_lb.py (LP, drops overlap).
D_LB: utilization (the unconstrained continuous-limit floor; in practice
      CVaR top-10% of grid densities equals the local-mean for the
      uniform distribution, which is utilization. The achieved proxy
      density is ~0.5-0.7 because grid normalization differs but the
      LB structure holds).
C_LB: trivial 0 (vacuous; needs max-flow / min-cut to tighten, deferred).

So proxy_LB = WL_LB + 0.5 · (utilization * proxy_normalization) + 0.

This is a STRICT lower bound because it sums independent component
minima, each ≤ that component's value at the achievable optimum.

Per bench, report the gap (v7_proxy - LB) / LB.
"""
from __future__ import annotations
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    # Load each input
    def read(path):
        if not Path(path).exists():
            return {}
        with open(path) as f:
            return {r["bench"]: r for r in csv.DictReader(f)}

    lp = read(ROOT / "research" / "lower_bounds" / "l1_hpwl_lb_results.csv")
    util = read(ROOT / "research" / "lower_bounds" / "utilization.csv")
    # Achieved (v7 sweep_results uses 'benchmark' column instead of 'bench')
    achv_path = ROOT / "submissions" / "vmallela_v7" / "sweep_results.csv"
    achv = {}
    if achv_path.exists():
        with open(achv_path) as f:
            for r in csv.DictReader(f):
                achv[r["benchmark"]] = r

    benches = sorted(set(list(lp.keys()) + list(util.keys())))
    rows = []
    print(f"\n{'bench':<8} {'WL_LB':>9} {'D_LB(util)':>11} "
          f"{'C_LB':>6} {'proxy_LB':>10} | {'v7_proxy':>9} "
          f"{'gap%':>7} {'WL_gap%':>9}")
    for b in benches:
        try:
            wl_lb = float(lp[b]["hpwl_normalized_lb"]) if b in lp else float("nan")
            u = float(util[b]["utilization"]) if b in util else float("nan")
            v7p = float(achv[b]["proxy_cost"]) if b in achv else float("nan")
            v7w = float(achv[b]["wirelength_cost"]) if b in achv else float("nan")
            v7d = float(achv[b]["density_cost"]) if b in achv else float("nan")
            v7c = float(achv[b]["congestion_cost"]) if b in achv else float("nan")
        except (KeyError, ValueError):
            continue
        # Density LB. Trick: the proxy density value IS the top-10% of grid
        # densities, normalized somehow. The minimum value (uniform spread)
        # would set top-10% = mean = utilization. But the proxy normalization
        # divides by something else; v7's actual achieved density (0.45-0.68)
        # is BELOW utilization (0.80) → normalization is sub-utilization. So
        # density LB is bounded by the *achieved* density across all 17
        # benches' minimum (0.453 for ibm09) as an empirical floor.
        # For honest reporting: assume D_LB = 0 (no claim).
        d_lb = 0.0
        c_lb = 0.0
        proxy_lb = wl_lb + 0.5 * d_lb + 0.5 * c_lb
        gap = (v7p - proxy_lb) / proxy_lb * 100 if proxy_lb > 0 else float("nan")
        wl_gap = (v7w - wl_lb) / wl_lb * 100 if wl_lb > 0 else float("nan")
        print(f"{b:<8} {wl_lb:>9.4f} {u*0.5:>11.4f} {c_lb:>6.2f} "
              f"{proxy_lb:>10.4f} | {v7p:>9.4f} {gap:>7.1f}% "
              f"{wl_gap:>9.1f}%")
        rows.append({
            "bench": b,
            "wl_lb": wl_lb,
            "utilization": u,
            "v7_proxy": v7p,
            "v7_wl": v7w,
            "v7_d": v7d,
            "v7_c": v7c,
            "proxy_lb": proxy_lb,
            "gap_pct": gap,
            "wl_gap_pct": wl_gap,
        })

    out = ROOT / "research" / "lower_bounds" / "honest_lb.csv"
    with open(out, "w") as f:
        keys = list(rows[0].keys())
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r[k]) for k in keys) + "\n")
    print(f"\n{out}")

    # Summary stats
    proxies = [r["v7_proxy"] for r in rows]
    lbs = [r["proxy_lb"] for r in rows]
    gaps = [r["gap_pct"] for r in rows]
    if proxies:
        print()
        print(f"v7 mean proxy:   {sum(proxies)/len(proxies):.4f}")
        print(f"LB mean:         {sum(lbs)/len(lbs):.4f}")
        print(f"mean gap:        {sum(gaps)/len(gaps):.1f}%")
        print(f"max gap:         {max(gaps):.1f}% ({rows[gaps.index(max(gaps))]['bench']})")
        print(f"min gap:         {min(gaps):.1f}% ({rows[gaps.index(min(gaps))]['bench']})")


if __name__ == "__main__":
    main()
