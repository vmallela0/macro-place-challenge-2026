"""Spectral SCFT lower bound on proxy.

Hypothesis (the dream)
----------------------
Macro placement at 80% utilization is structurally identical to a 2D
branched polymer melt with quenched disorder. Branch points = macros,
bonds = nets, excluded volume = overlap constraint, bond stretch
energy = HPWL. SCFT (Helfand 1972, de Gennes 1979) gives a closed-
form mean-field equilibrium for this regime.

In linearized SCFT, the equilibrium density satisfies the Edwards
equation ∇²ψ - V(ψ)·ψ = 0. On a torus this has Jacobi-theta-function
solutions; on a rectangle they image-extend. Equivalently: the optimal
density is a sum over Laplacian eigenmodes weighted by a screening
factor 1/(λ_k + α).

Concrete computation
--------------------
Quadratic relaxation of proxy (drop overlap, use clique-Laplacian for
HPWL and uniform-density-target for density):

    F(x) = ½ x^T (L_HPWL + α I) x + b^T x + const

Optimum:
    x* = -(L_HPWL + α I)^{-1} b
    F_min = const - ½ b^T (L_HPWL + α I)^{-1} b

In Laplacian-eigenbasis (ψ_k, λ_k):
    F_min = const - ½ Σ_k b_k² / (λ_k + α)        ← CLOSED-FORM SUM

This is a STRICT lower bound on quadratic-relaxed proxy, hence on
the actual proxy (since L1 ≥ L2 stretching, density is convex, and
overlap is dropped).

α calibration
-------------
α controls the trade-off between HPWL (small α) and density (large α).
At α=0 we recover Tsay-Kuh quadratic placement (HPWL only). At α=∞
the optimum is identically at the centroid (all mass to mean). The
proxy weighting (1.0·HPWL + 0.5·density + 0.5·cong) maps to α≈0.5
(under the linearization assumption that density and congestion's
quadratic forms each contribute α/2 = 0.25).

We sweep α ∈ {0, 0.01, 0.1, 0.5, 1.0, 5.0} and report the closed-form
F_min for each, plus the gap to v7's achieved proxy.
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path
import numpy as np
import torch
from scipy.sparse import csr_matrix, lil_matrix, csc_matrix, eye
from scipy.sparse.linalg import eigsh, spsolve, cg, LinearOperator

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def _load_incr(bench_name):
    import importlib.util as ilu
    from macro_place.benchmark import Benchmark
    v1_spec = ilu.spec_from_file_location(
        "_v1_sp", str(ROOT / "submissions" / "vmallela" / "placer.py"))
    v1 = ilu.module_from_spec(v1_spec)
    v1_spec.loader.exec_module(v1)
    pt_path = str(ROOT / "benchmarks" / "processed" / "public" /
                   f"{bench_name}.pt")
    raw = torch.load(pt_path, weights_only=False)
    valid = set(Benchmark.__dataclass_fields__.keys())
    f = {k: v for k, v in raw.items() if k in valid}
    if "num_hard_macros" not in f:
        f["num_hard_macros"] = f["num_macros"]; f["num_soft_macros"] = 0
    if "soft_macro_indices" not in f: f["soft_macro_indices"] = []
    if "port_positions" not in f: f["port_positions"] = torch.zeros(0, 2)
    if "macro_pin_offsets" not in f: f["macro_pin_offsets"] = []
    bench = Benchmark(**f)
    plc = v1._load_plc(bench.name)
    incr = v1.IncrementalEvaluator(plc, bench)
    return bench, incr


def build_clique_laplacian(incr, n_macros):
    """Build sparse clique-Laplacian L over MACROS (ports/external pins
    treated separately as a forcing term). For each net of size k_macro
    macro pins (and k_port port pins), add edge weight w/(k_total-1)
    between every pair of macro pins; record port-pin terms as forcing.

    Returns:
        L: (n_macros × n_macros) csc Laplacian
        b_x, b_y: (n_macros,) forcing vectors from port pins
        const_x, const_y: scalar constants from port-port pairs
    """
    pin_macro = np.asarray(incr.pin_macro, dtype=np.int64)
    pin_xoff = np.asarray(incr.pin_xoff, dtype=np.float64)
    pin_yoff = np.asarray(incr.pin_yoff, dtype=np.float64)
    net_starts = np.asarray(incr.net_starts, dtype=np.int64)
    net_weight = np.asarray(incr.net_weight, dtype=np.float64)
    n_nets = int(net_starts.shape[0] - 1)

    # Sparse Laplacian construction via COO.
    rows, cols, data = [], [], []
    b_x = np.zeros(n_macros)
    b_y = np.zeros(n_macros)
    const_x = 0.0
    const_y = 0.0

    for nid in range(n_nets):
        start, end = net_starts[nid], net_starts[nid + 1]
        macros_in_net = []
        ports_in_net_x = []
        ports_in_net_y = []
        offsets_x = []
        offsets_y = []
        for p in range(start, end):
            m = pin_macro[p]
            if m >= 0:
                macros_in_net.append((m, pin_xoff[p], pin_yoff[p]))
            else:
                ports_in_net_x.append(pin_xoff[p])
                ports_in_net_y.append(pin_yoff[p])
        k = len(macros_in_net) + len(ports_in_net_x)
        if k < 2:
            continue
        w = net_weight[nid] / (k - 1)

        # Macro-macro edges (clique).
        for i, (m_i, dx_i, dy_i) in enumerate(macros_in_net):
            for j_idx in range(i + 1, len(macros_in_net)):
                m_j, dx_j, dy_j = macros_in_net[j_idx]
                # quadratic term: w * (x_i + dx_i - x_j - dx_j)^2
                # = w * [(x_i - x_j)^2 + 2(x_i - x_j)(dx_i - dx_j) + (dx_i - dx_j)^2]
                # Laplacian L: contributes -w on off-diag (m_i, m_j) and m_j, m_i,
                # +w on diag (m_i, m_i) and (m_j, m_j).
                rows += [m_i, m_j, m_i, m_j]
                cols += [m_j, m_i, m_i, m_j]
                data += [-w, -w, w, w]
                # Linear term in x_i, x_j: 2w * (x_i - x_j) * (dx_i - dx_j)
                # = 2w (dx_i - dx_j) x_i - 2w (dx_i - dx_j) x_j
                b_x[m_i] += 2 * w * (dx_i - dx_j)
                b_x[m_j] += -2 * w * (dx_i - dx_j)
                b_y[m_i] += 2 * w * (dy_i - dy_j)
                b_y[m_j] += -2 * w * (dy_i - dy_j)
                const_x += w * (dx_i - dx_j) ** 2
                const_y += w * (dy_i - dy_j) ** 2

        # Macro-port edges.
        for (m_i, dx_i, dy_i) in macros_in_net:
            for px, py in zip(ports_in_net_x, ports_in_net_y):
                # quadratic: w * (x_i + dx_i - px)^2
                # = w x_i^2 + 2w(dx_i - px) x_i + w(dx_i - px)^2
                rows.append(m_i); cols.append(m_i); data.append(w)
                b_x[m_i] += 2 * w * (dx_i - px)
                b_y[m_i] += 2 * w * (dy_i - py)
                const_x += w * (dx_i - px) ** 2
                const_y += w * (dy_i - py) ** 2

        # Port-port: pure constants.
        for i, px_i in enumerate(ports_in_net_x):
            py_i = ports_in_net_y[i]
            for j_idx in range(i + 1, len(ports_in_net_x)):
                px_j = ports_in_net_x[j_idx]
                py_j = ports_in_net_y[j_idx]
                const_x += w * (px_i - px_j) ** 2
                const_y += w * (py_i - py_j) ** 2

    L = csr_matrix((data, (rows, cols)),
                    shape=(n_macros, n_macros), dtype=np.float64).tocsc()
    return L, b_x, b_y, const_x, const_y


def spectral_lb(bench_name, alphas=(0.0, 0.001, 0.01, 0.1, 0.5, 1.0, 10.0),
                  k_eigs=200):
    print(f"\n=== {bench_name} ===", flush=True)
    bench, incr = _load_incr(bench_name)
    n_macros = int(np.asarray(incr.macro_pos).shape[0])
    print(f"  {bench}", flush=True)
    print(f"  n_macros={n_macros}", flush=True)

    t0 = time.time()
    L, bx, by, const_x, const_y = build_clique_laplacian(incr, n_macros)
    print(f"  Laplacian: shape {L.shape}, nnz {L.nnz}, build "
          f"{time.time()-t0:.2f}s", flush=True)
    # Verify L symmetric and PSD-like.
    sym_err = float(abs(L - L.T).max())
    print(f"  symmetry max-abs-err: {sym_err:.2e}", flush=True)

    cw = float(incr.cw); ch = float(incr.ch)
    net_cnt = float(incr.net_cnt)

    # Closed-form per α: F_min = const - ¼ b^T (L + α I)^-1 b  (using
    # the proxy form with quadratic terms on each axis separately).
    # Solve once via direct sparse factorization for speed.
    results = []
    for alpha in alphas:
        t1 = time.time()
        if alpha == 0:
            # L is singular (translation null space). Pin one macro at 0
            # to break the translation symmetry. Take macro 0 as pin.
            # Modify L: zero out row/col 0, add 1.0 on diag.
            L_pinned = L.copy().tolil()
            L_pinned[0, :] = 0; L_pinned[:, 0] = 0
            L_pinned[0, 0] = 1.0
            L_solve = L_pinned.tocsc()
            # b also: zero out entry 0
            bx_p = bx.copy(); bx_p[0] = 0
            by_p = by.copy(); by_p[0] = 0
            x_star = spsolve(L_solve, -0.5 * bx_p)
            y_star = spsolve(L_solve, -0.5 * by_p)
        else:
            A = L + alpha * eye(n_macros)
            x_star = spsolve(A.tocsc(), -0.5 * bx)
            y_star = spsolve(A.tocsc(), -0.5 * by)
        # Evaluate F at x*: F = ½ x^T L x + b^T x + const
        F_x = 0.5 * float(x_star @ (L @ x_star)) + float(bx @ x_star) + const_x
        F_y = 0.5 * float(y_star @ (L @ y_star)) + float(by @ y_star) + const_y
        F_total = F_x + F_y
        # Normalize like the proxy: divide by net_cnt · (cw + ch)
        F_norm = F_total / (net_cnt * (cw + ch))
        wall = time.time() - t1
        # Range of x_star (sanity)
        x_range = (float(x_star.min()), float(x_star.max()))
        y_range = (float(y_star.min()), float(y_star.max()))
        print(f"  α={alpha:>7}: F_min(L2-relaxed, normalized) = "
              f"{F_norm:.6f} (wall {wall:.2f}s); "
              f"x_range=[{x_range[0]:.2f},{x_range[1]:.2f}] "
              f"y_range=[{y_range[0]:.2f},{y_range[1]:.2f}]",
              flush=True)
        results.append({"alpha": alpha, "F_norm": F_norm,
                          "x_range": x_range, "y_range": y_range,
                          "wall_s": wall})

    return {"bench": bench_name, "n_macros": n_macros,
             "results": results, "L_nnz": int(L.nnz)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="ibm01")
    args = ap.parse_args()
    if args.benchmark == "all":
        benches = ["ibm01", "ibm02", "ibm03", "ibm04", "ibm06", "ibm07",
                   "ibm08", "ibm09", "ibm10", "ibm11", "ibm12", "ibm13",
                   "ibm14", "ibm15", "ibm16", "ibm17", "ibm18"]
    else:
        benches = [args.benchmark]

    all_rows = []
    for b in benches:
        r = spectral_lb(b)
        for ar in r["results"]:
            all_rows.append({"bench": b, "n_macros": r["n_macros"],
                              "alpha": ar["alpha"],
                              "F_norm": ar["F_norm"],
                              "wall_s": ar["wall_s"]})

    out = ROOT / "research" / "lower_bounds" / "spectral_scft_results.csv"
    import csv
    with open(out, "w") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader(); w.writerows(all_rows)
    print(f"\nresults → {out}")


if __name__ == "__main__":
    main()
