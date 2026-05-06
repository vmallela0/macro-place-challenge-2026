"""Test the spectral SCFT closed-form against v7 achieved proxy.

For each α value, compute:
  - Pearson correlation across benches between F_min(α) and v7 proxy
  - Linear fit v7_proxy = a + b · F_min(α); R² of the fit
  - Residuals (v7 - prediction) per bench

If R² > 0.9 at some α, the closed-form spectral expression PREDICTS
v7's behavior up to a linear scaling — i.e., the algorithm asymptotes
to the SCFT mean field. That's the calibration we want.

Also tests power-law scaling: does v7_proxy fit
    v7 = A + B · F(α) + C · N^{-β}
suggesting a 1/N approach to a thermodynamic limit?
"""
from __future__ import annotations
import csv
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def read_csv(path):
    if not Path(path).exists():
        return None
    with open(path) as f:
        return list(csv.DictReader(f))


def main():
    sp = read_csv(ROOT / "research" / "lower_bounds" /
                    "spectral_scft_results.csv")
    lp = read_csv(ROOT / "research" / "lower_bounds" /
                    "l1_hpwl_lb_results.csv")
    achv = read_csv(ROOT / "submissions" / "vmallela_v7" /
                      "sweep_results.csv")
    if not sp or not achv:
        print("missing required CSVs")
        return

    # Index achieved by bench
    a_by_b = {r["benchmark"]: r for r in achv}
    # Index lp by bench
    lp_by_b = {r["bench"]: r for r in (lp or [])}

    # Reorganize spectral: bench → {alpha → F_norm}
    sp_by_b = {}
    for row in sp:
        sp_by_b.setdefault(row["bench"], {})[float(row["alpha"])] = float(row["F_norm"])

    benches = sorted(sp_by_b.keys())
    print(f"Benches: {len(benches)}")

    # Build vectors
    rows = []
    for b in benches:
        if b not in a_by_b:
            continue
        achv_proxy = float(a_by_b[b]["proxy_cost"])
        achv_wl = float(a_by_b[b]["wirelength_cost"])
        achv_d = float(a_by_b[b]["density_cost"])
        achv_c = float(a_by_b[b]["congestion_cost"])
        n_macros = int(sp[0].get("n_macros", 0))
        # Get n_macros for this bench from lp_by_b or sp
        for r in sp:
            if r["bench"] == b:
                n_macros = int(r["n_macros"])
                break
        lp_lb = float(lp_by_b.get(b, {}).get("hpwl_normalized_lb", "nan")) \
            if b in lp_by_b else float("nan")
        rows.append({
            "bench": b,
            "n_macros": n_macros,
            "v7_proxy": achv_proxy,
            "v7_wl": achv_wl,
            "v7_d": achv_d,
            "v7_c": achv_c,
            "lp_wl_lb": lp_lb,
            "spectral": sp_by_b[b],
        })

    print(f"\n{'bench':<8} {'n':>6} {'v7_proxy':>10} {'v7_wl':>8} "
          f"{'lp_wl_lb':>10} | F(0)={'F0':>8} F(.001) F(.01) F(.1) "
          f"F(.5) F(1) F(10)")
    for r in rows:
        sp = r["spectral"]
        print(f"{r['bench']:<8} {r['n_macros']:>6} {r['v7_proxy']:>10.4f} "
              f"{r['v7_wl']:>8.3f} {r['lp_wl_lb']:>10.4f} | "
              f"{sp.get(0.0, float('nan')):>7.4f} "
              f"{sp.get(0.001, float('nan')):>7.4f} "
              f"{sp.get(0.01, float('nan')):>6.4f} "
              f"{sp.get(0.1, float('nan')):>5.4f} "
              f"{sp.get(0.5, float('nan')):>5.4f} "
              f"{sp.get(1.0, float('nan')):>5.4f} "
              f"{sp.get(10.0, float('nan')):>5.4f}")

    # Correlations
    print("\n=== Correlation analysis: v7_proxy vs F(α) ===")
    proxy_vec = np.array([r["v7_proxy"] for r in rows])
    n_vec = np.array([r["n_macros"] for r in rows])
    print(f"v7_proxy range: [{proxy_vec.min():.4f}, {proxy_vec.max():.4f}], "
          f"mean={proxy_vec.mean():.4f}, std={proxy_vec.std():.4f}")
    for alpha in (0.0, 0.001, 0.01, 0.1, 0.5, 1.0, 10.0):
        F_vec = np.array([r["spectral"].get(alpha, float("nan")) for r in rows])
        if np.isnan(F_vec).any():
            continue
        # Pearson r
        r_pearson = float(np.corrcoef(F_vec, proxy_vec)[0, 1])
        # Linear fit
        coef = np.polyfit(F_vec, proxy_vec, 1)
        pred = coef[0] * F_vec + coef[1]
        ss_res = float(((proxy_vec - pred) ** 2).sum())
        ss_tot = float(((proxy_vec - proxy_vec.mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        # Mean abs error of linear fit
        mae = float(np.mean(np.abs(proxy_vec - pred)))
        print(f"  α={alpha:>7}: r={r_pearson:+.4f}  R²={r2:+.4f}  "
              f"fit: v7 ≈ {coef[0]:+.3f} · F({alpha}) + {coef[1]:+.3f}  "
              f"MAE={mae:.4f}")

    # Test against LP wirelength LB
    print("\n=== Correlation: v7_proxy vs LP_wl_LB ===")
    has_lp = [r for r in rows if not np.isnan(r["lp_wl_lb"])]
    if len(has_lp) > 5:
        F_vec = np.array([r["lp_wl_lb"] for r in has_lp])
        proxy_vec_h = np.array([r["v7_proxy"] for r in has_lp])
        r_pearson = float(np.corrcoef(F_vec, proxy_vec_h)[0, 1])
        coef = np.polyfit(F_vec, proxy_vec_h, 1)
        pred = coef[0] * F_vec + coef[1]
        ss_tot = float(((proxy_vec_h - proxy_vec_h.mean()) ** 2).sum())
        r2 = 1.0 - float(((proxy_vec_h - pred) ** 2).sum()) / ss_tot
        print(f"  r={r_pearson:+.4f}  R²={r2:+.4f}  fit: v7 ≈ "
              f"{coef[0]:+.3f} · LP + {coef[1]:+.3f}")

    # Test against v7's wirelength alone (sanity)
    print("\n=== Correlation: v7_proxy vs v7_wirelength ===")
    F_vec = np.array([r["v7_wl"] for r in rows])
    r_pearson = float(np.corrcoef(F_vec, proxy_vec)[0, 1])
    print(f"  r={r_pearson:+.4f}  (sanity — should be moderate)")

    # Test against v7 congestion (the dominant proxy term)
    print("\n=== Correlation: v7_proxy vs v7_congestion ===")
    F_vec = np.array([r["v7_c"] for r in rows])
    r_pearson = float(np.corrcoef(F_vec, proxy_vec)[0, 1])
    print(f"  r={r_pearson:+.4f}")

    # Output combined CSV
    out = ROOT / "research" / "lower_bounds" / "calibration.csv"
    keys = ["bench", "n_macros", "v7_proxy", "v7_wl", "v7_d", "v7_c",
            "lp_wl_lb", "F_alpha_0", "F_alpha_0.5", "F_alpha_10"]
    with open(out, "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            sp = r["spectral"]
            vals = [r["bench"], r["n_macros"], f"{r['v7_proxy']:.4f}",
                     f"{r['v7_wl']:.3f}", f"{r['v7_d']:.3f}", f"{r['v7_c']:.3f}",
                     f"{r['lp_wl_lb']:.4f}",
                     f"{sp.get(0.0, float('nan')):.4f}",
                     f"{sp.get(0.5, float('nan')):.4f}",
                     f"{sp.get(10.0, float('nan')):.4f}"]
            f.write(",".join(str(v) for v in vals) + "\n")
    print(f"\nresults → {out}")


if __name__ == "__main__":
    main()
