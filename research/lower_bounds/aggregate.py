"""Aggregate per-bench lower-bound + achieved-cost calibration table.

Reads:
  research/lower_bounds/treewidth_results.csv
  research/lower_bounds/l1_hpwl_lb_results.csv
  submissions/vmallela_v7/sweep_results.csv  (achieved proxy on dev box)

Writes:
  research/lower_bounds/calibration.csv
  research/lower_bounds/calibration.md   (human-readable summary)

Columns
-------
bench, n_macros, n_nets, treewidth_ub,
hpwl_lb (LP, normalized), wl_v7 (achieved), wl_gap_pct,
proxy_lb_naive (= hpwl_lb + 0.5·utilization),
proxy_v7 (achieved), proxy_gap_pct (= (v7 − lb)/lb · 100)
"""
from __future__ import annotations
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _read_csv_dict(path):
    if not Path(path).exists():
        return {}
    out = {}
    with open(path) as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            out[row["bench"]] = row
    return out


def main():
    tw = _read_csv_dict(ROOT / "research" / "lower_bounds" /
                          "treewidth_results.csv")
    lp = _read_csv_dict(ROOT / "research" / "lower_bounds" /
                          "l1_hpwl_lb_results.csv")
    achv = _read_csv_dict(ROOT / "submissions" / "vmallela_v7" /
                            "sweep_results.csv")

    benches = sorted(set(list(tw.keys()) + list(lp.keys()) + list(achv.keys())))
    rows = []
    for b in benches:
        twr = tw.get(b, {})
        lpr = lp.get(b, {})
        ar = achv.get(b, {})
        n_macros = twr.get("n_macros", "")
        n_nets = twr.get("n_nets", "")
        treewidth_ub = twr.get("treewidth_ub", "")
        try:
            hpwl_lb = float(lpr.get("hpwl_normalized_lb", "nan"))
        except ValueError:
            hpwl_lb = float("nan")
        try:
            wl_v7 = float(ar.get("wirelength_cost", "nan"))
            den_v7 = float(ar.get("density_cost", "nan"))
            cong_v7 = float(ar.get("congestion_cost", "nan"))
            proxy_v7 = float(ar.get("proxy_cost", "nan"))
        except (ValueError, KeyError):
            wl_v7 = den_v7 = cong_v7 = proxy_v7 = float("nan")

        if hpwl_lb == hpwl_lb and wl_v7 == wl_v7 and hpwl_lb > 0:
            wl_gap = (wl_v7 - hpwl_lb) / hpwl_lb * 100.0
        else:
            wl_gap = float("nan")
        # Trivial proxy floor using LP wirelength bound.
        # density_lb is approximated by 0 (pessimistic) and 0.5·utilization
        # (loose); we report the LP-only proxy floor (just hpwl_lb) for
        # honesty and let the reader compare.
        if proxy_v7 == proxy_v7 and hpwl_lb == hpwl_lb and hpwl_lb > 0:
            proxy_gap = (proxy_v7 - hpwl_lb) / hpwl_lb * 100.0
        else:
            proxy_gap = float("nan")
        rows.append({
            "bench": b,
            "n_macros": n_macros,
            "n_nets": n_nets,
            "treewidth_ub": treewidth_ub,
            "hpwl_lb": f"{hpwl_lb:.6f}" if hpwl_lb == hpwl_lb else "",
            "wl_v7": f"{wl_v7:.4f}" if wl_v7 == wl_v7 else "",
            "wl_gap_pct": f"{wl_gap:.1f}" if wl_gap == wl_gap else "",
            "proxy_v7": f"{proxy_v7:.4f}" if proxy_v7 == proxy_v7 else "",
            "proxy_gap_vs_hpwl_lb_pct": (
                f"{proxy_gap:.1f}" if proxy_gap == proxy_gap else ""),
        })

    out_csv = ROOT / "research" / "lower_bounds" / "calibration.csv"
    keys = list(rows[0].keys()) if rows else []
    with open(out_csv, "w") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"calibration.csv → {out_csv}")

    # Human-readable summary
    out_md = ROOT / "research" / "lower_bounds" / "calibration.md"
    lines = []
    lines.append("# Calibration table — per-bench lower bounds vs v7\n")
    lines.append(
        "| bench | n_macros | n_nets | tw_ub | HPWL LP-LB | WL v7 | WL gap | "
        "proxy v7 | proxy gap (vs HPWL-LB) |")
    lines.append(
        "|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        lines.append(
            f"| {r['bench']} | {r['n_macros']} | {r['n_nets']} | "
            f"{r['treewidth_ub']} | {r['hpwl_lb']} | {r['wl_v7']} | "
            f"{r['wl_gap_pct']}% | {r['proxy_v7']} | "
            f"{r['proxy_gap_vs_hpwl_lb_pct']}% |")
    lines.append("")
    lines.append("**Interpretation.**")
    lines.append("- `tw_ub` is the min-degree heuristic upper bound on treewidth. "
                  "All > 100 → exact treewidth-DP solver intractable.")
    lines.append("- `HPWL LP-LB` is the LP optimum with overlap dropped — "
                  "a STRICT lower bound on achievable wirelength.")
    lines.append("- `WL gap` shows the wirelength slack (most of which is "
                  "overlap-induced, not algorithmic).")
    lines.append("- `proxy gap (vs HPWL-LB)` shows v7's proxy cost above "
                  "the wirelength-only LP floor. Density and congestion "
                  "components are NOT in the LB; expect this gap to be large.")
    lines.append("")
    lines.append("To get a TIGHT proxy lower bound, add density (QP) and "
                  "congestion (QP) terms to the LP formulation. Status: "
                  "follow-up convex relaxation work.")
    with open(out_md, "w") as f:
        f.write("\n".join(lines))
    print(f"calibration.md → {out_md}")
    print()
    for line in lines[:20]:
        print(line)


if __name__ == "__main__":
    main()
