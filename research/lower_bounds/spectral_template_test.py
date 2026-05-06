"""Test the spectral SCFT template hypothesis.

Hypothesis: the closed-form minimizer of the L2-relaxed proxy
    x* = -(L + α I)^{-1} b
at α≈0.5 (the proxy density weight) is structurally close to the
optimum placement v7 finds. If true, this is a closed-form template
for placement and can serve as a warm start.

Test:
  1. Compute x*(α) for α ∈ {0, 0.001, 0.01, 0.1, 0.5, 1.0}
  2. Push x* through compute_proxy_cost (with overlap repair) to get
     proxy at the spectral template.
  3. Compare to v7 achieved.
  4. If proxy(x*) is ≤ 1.5× v7, spectral captures most of the structure.
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path
import numpy as np
import torch
from scipy.sparse import csr_matrix, eye
from scipy.sparse.linalg import spsolve

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "submissions" / "vmallela"))

from spectral_scft import _load_incr, build_clique_laplacian


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="ibm01")
    args = ap.parse_args()

    bench, incr = _load_incr(args.benchmark)
    n_macros = int(np.asarray(incr.macro_pos).shape[0])
    cw = float(incr.cw); ch = float(incr.ch)
    print(f"=== {args.benchmark} ===")
    print(f"  {bench}")
    print(f"  n_macros={n_macros}, canvas={cw:.2f}×{ch:.2f}")

    L, bx, by, _, _ = build_clique_laplacian(incr, n_macros)
    print(f"  Laplacian nnz {L.nnz}")

    # v7 achieved (from sweep_results.csv if available)
    achv = {}
    sweep_csv = ROOT / "submissions" / "vmallela_v7" / "sweep_results.csv"
    if sweep_csv.exists():
        import csv
        with open(sweep_csv) as f:
            for row in csv.DictReader(f):
                achv[row["benchmark"]] = row
    if args.benchmark in achv:
        r = achv[args.benchmark]
        print(f"  v7 achieved: proxy={r['proxy_cost']} "
              f"wl={r['wirelength_cost']} d={r['density_cost']} "
              f"c={r['congestion_cost']}")

    # Compute proxy via official evaluator
    from macro_place.objective import compute_proxy_cost
    import importlib.util as ilu
    v1_spec = ilu.spec_from_file_location(
        "_v1_t", str(ROOT / "submissions" / "vmallela" / "placer.py"))
    v1 = ilu.module_from_spec(v1_spec); v1_spec.loader.exec_module(v1)
    plc = v1._load_plc(bench.name)

    print(f"\n  Spectral templates:")
    for alpha in (0.001, 0.01, 0.1, 0.5, 1.0, 5.0):
        t0 = time.time()
        A = (L + alpha * eye(n_macros)).tocsc()
        x_star = spsolve(A, -0.5 * bx)
        y_star = spsolve(A, -0.5 * by)
        # Clip into canvas (the LP equivalent — soft enforcement)
        x_star = np.clip(x_star, 0.0, cw)
        y_star = np.clip(y_star, 0.0, ch)
        # Build full placement
        pos = np.stack([x_star, y_star], axis=1).astype(np.float32)
        pos_t = torch.tensor(pos)
        try:
            r = compute_proxy_cost(pos_t, bench, plc)
            proxy = float(r["proxy_cost"])
            wl = float(r["wirelength_cost"])
            d = float(r["density_cost"])
            c = float(r["congestion_cost"])
            ov = int(r["overlap_count"])
            print(f"  α={alpha:>5}: proxy={proxy:.4f} "
                  f"wl={wl:.3f} d={d:.3f} c={c:.3f} "
                  f"overlaps={ov} (solve+eval {time.time()-t0:.2f}s)",
                  flush=True)
        except Exception as e:
            print(f"  α={alpha}: eval err: {e}")


if __name__ == "__main__":
    main()
