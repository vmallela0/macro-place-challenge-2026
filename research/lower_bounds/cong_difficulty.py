"""Netlist-intrinsic congestion difficulty estimator.

Hypothesis: v7 achieved congestion (which dominates 73% of proxy
variance) is set by NETLIST STRUCTURE, not algorithm. If we can
compute a netlist-only invariant that correlates with v7's achieved
cong across benches, we have a STRUCTURAL CONGESTION FLOOR per bench.

Several candidate invariants:
  1. Pin-density per canvas perimeter:  n_pins / (2(cw+ch))
  2. Average net Steiner length / canvas:  sum(net_extents) / canvas_area
  3. Routing demand vs supply ratio:
        Σ_n w_n · estimated_steiner_length_n  /  total_routing_supply
        where total_routing_supply = canvas_area · (h_routes + v_routes)/micron
  4. Spectral: λ_2(L)/avg_degree (Cheeger sparseness — higher = harder)

Test: compute each across the 17 benches; correlate with v7 achieved
congestion. Best correlator is the structural floor.
"""
from __future__ import annotations
import csv
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "submissions" / "vmallela"))


def _load(bench_name):
    import importlib.util as ilu
    from macro_place.benchmark import Benchmark
    v1_spec = ilu.spec_from_file_location(
        "_v1_cd", str(ROOT / "submissions" / "vmallela" / "placer.py"))
    v1 = ilu.module_from_spec(v1_spec)
    v1_spec.loader.exec_module(v1)
    bench = Benchmark.load(
        str(ROOT / "benchmarks" / "processed" / "public" / f"{bench_name}.pt"))
    plc = v1._load_plc(bench.name)
    incr = v1.IncrementalEvaluator(plc, bench)
    return bench, incr


def compute_invariants(bench, incr):
    """Compute netlist-intrinsic difficulty invariants."""
    cw = float(bench.canvas_width)
    ch = float(bench.canvas_height)
    canvas_area = cw * ch
    canvas_perim = 2 * (cw + ch)

    pin_macro = np.asarray(incr.pin_macro)
    n_pins = int(pin_macro.shape[0])
    n_macros = int(np.asarray(incr.macro_pos).shape[0])
    n_nets = int(incr.net_starts.shape[0] - 1)

    # 1. pin-density per perimeter (routing tracks per micron need to absorb pins)
    pin_per_perim = n_pins / canvas_perim
    pin_per_area = n_pins / canvas_area

    # 2. Net Steiner-length proxy: ½ × bbox of each net at .plc-init positions
    pin_xoff = np.asarray(incr.pin_xoff)
    pin_yoff = np.asarray(incr.pin_yoff)
    macro_pos = np.asarray(incr.macro_pos)
    is_port = pin_macro < 0
    abs_x = np.where(is_port, pin_xoff,
                       macro_pos[np.maximum(pin_macro, 0), 0] + pin_xoff)
    abs_y = np.where(is_port, pin_yoff,
                       macro_pos[np.maximum(pin_macro, 0), 1] + pin_yoff)

    net_starts = np.asarray(incr.net_starts)
    net_weight = np.asarray(incr.net_weight)
    total_steiner = 0.0
    for nid in range(n_nets):
        s, e = net_starts[nid], net_starts[nid + 1]
        if e - s < 2:
            continue
        xs = abs_x[s:e]; ys = abs_y[s:e]
        # Steiner tree length on Hanan grid is upper-bounded by HPWL,
        # lower-bounded by HPWL/2 (Garey-Johnson). Use HPWL as proxy.
        bbox = (xs.max() - xs.min()) + (ys.max() - ys.min())
        total_steiner += float(net_weight[nid]) * bbox

    h_per_micron = float(getattr(bench, "hroutes_per_micron", 11.285))
    v_per_micron = float(getattr(bench, "vroutes_per_micron", 12.605))
    total_supply = canvas_area * (h_per_micron + v_per_micron)
    demand_supply_ratio = total_steiner / total_supply

    # 4. Spectral: smallest non-zero Laplacian eigenvalue (Fiedler).
    # Build clique Laplacian on macros only; ports are forcing.
    from scipy.sparse import csr_matrix, csc_matrix, eye
    from scipy.sparse.linalg import eigsh

    rows, cols, data = [], [], []
    for nid in range(n_nets):
        s, e = net_starts[nid], net_starts[nid + 1]
        macros = [int(pin_macro[p]) for p in range(s, e)
                   if pin_macro[p] >= 0]
        if len(macros) < 2:
            continue
        w = float(net_weight[nid]) / (len(macros) - 1)
        for i, mi in enumerate(macros):
            for j_idx in range(i + 1, len(macros)):
                mj = macros[j_idx]
                rows += [mi, mj, mi, mj]
                cols += [mj, mi, mi, mj]
                data += [-w, -w, w, w]
    L = csr_matrix((data, (rows, cols)),
                    shape=(n_macros, n_macros), dtype=np.float64).tocsc()
    # Fiedler value (2nd smallest eigval of L). Use shift-invert-style
    # call: smallest k=2 algebraic, take 2nd one (skip zero eigval of
    # connected graph).
    try:
        eigvals, _ = eigsh(L, k=2, which="SA",
                            sigma=0.0, mode="normal", maxiter=500, tol=1e-3)
    except Exception:
        try:
            eigvals, _ = eigsh(L + 1e-6 * eye(n_macros), k=2, which="SA",
                                maxiter=500, tol=1e-3)
        except Exception:
            eigvals = np.array([0.0, 0.0])
    fiedler = float(sorted(eigvals)[1])
    avg_degree = float(L.diagonal().mean())
    fiedler_ratio = fiedler / max(avg_degree, 1e-9)

    return {
        "n_macros": n_macros,
        "n_nets": n_nets,
        "n_pins": n_pins,
        "canvas_area": canvas_area,
        "pin_per_perim": pin_per_perim,
        "pin_per_area": pin_per_area,
        "total_steiner": total_steiner,
        "demand_supply_ratio": demand_supply_ratio,
        "fiedler": fiedler,
        "avg_degree": avg_degree,
        "fiedler_ratio": fiedler_ratio,
    }


def main():
    benches = ["ibm01", "ibm02", "ibm03", "ibm04", "ibm06", "ibm07", "ibm08",
               "ibm09", "ibm10", "ibm11", "ibm12", "ibm13", "ibm14", "ibm15",
               "ibm16", "ibm17", "ibm18"]
    rows = []
    for b in benches:
        print(f"=== {b} ===", flush=True)
        bench, incr = _load(b)
        inv = compute_invariants(bench, incr)
        inv["bench"] = b
        rows.append(inv)
        print(f"  pin_per_perim={inv['pin_per_perim']:.2f}, "
              f"demand/supply={inv['demand_supply_ratio']:.4f}, "
              f"fiedler={inv['fiedler']:.4f}, "
              f"fiedler_ratio={inv['fiedler_ratio']:.4f}", flush=True)

    out = ROOT / "research" / "lower_bounds" / "cong_difficulty.csv"
    keys = ["bench"] + [k for k in rows[0] if k != "bench"]
    with open(out, "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r[k]) for k in keys) + "\n")
    print(f"\n→ {out}")

    # Correlate with v7 achieved cong
    achv = {}
    achv_path = ROOT / "submissions" / "vmallela_v7" / "sweep_results.csv"
    with open(achv_path) as f:
        for r in csv.DictReader(f):
            achv[r["benchmark"]] = r
    cong_v7 = np.array([float(achv[r["bench"]]["congestion_cost"])
                          for r in rows if r["bench"] in achv])
    print("\n=== Correlation with v7 achieved congestion ===")
    for inv_name in ["pin_per_perim", "pin_per_area", "demand_supply_ratio",
                      "fiedler", "fiedler_ratio", "n_pins", "total_steiner"]:
        x = np.array([r[inv_name] for r in rows if r["bench"] in achv])
        if len(x) == len(cong_v7) and len(x) > 5:
            r_pearson = float(np.corrcoef(x, cong_v7)[0, 1])
            print(f"  r({inv_name}, v7_cong) = {r_pearson:+.4f}")


if __name__ == "__main__":
    main()
