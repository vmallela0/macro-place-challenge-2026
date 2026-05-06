"""L1 HPWL lower bound via LP — the calibration anchor.

For each bench, formulate L1 HPWL minimization as a linear program
(no overlap constraint, with port positions fixed). Solve via scipy
HiGHS interior point. The LP optimum is a STRICT LOWER BOUND on the
true placement's L1 HPWL (and hence on proxy = HPWL + 0.5·D + 0.5·C),
since dropping the overlap constraint can only let cost go down.

LP formulation (1D, x-axis; symmetric for y)
--------------------------------------------
Decision variables:
    x_m ∈ R                  — position of each free macro m
    s_n ≥ 0                  — slack for net n (= net's HPWL_x)

Objective:
    minimize  Σ_n w_n · s_n

Constraints (per net n, per pair (p, q) ∈ pins(n) × pins(n)):
    p_pos - q_pos ≤ s_n
    q_pos - p_pos ≤ s_n
where p_pos = x_{macro(p)} + dx_p (free) or fixed port position.

Equivalent compact formulation: introduce auxiliary u_n, l_n
    u_n ≥ p_pos for all p ∈ pins(n)
    l_n ≤ p_pos for all p ∈ pins(n)
    s_n = u_n - l_n
This has 2 · n_pins constraints per net instead of pair-quadratic.
We use this form (linear in problem size).

Run:
    .venv/bin/python research/lower_bounds/l1_hpwl_lb.py --benchmark ibm01
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix, vstack, hstack, eye, csc_matrix
from scipy.optimize import linprog

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "submissions" / "vmallela"))


def _load_incr(bench_name):
    import importlib.util as ilu
    import torch
    from macro_place.benchmark import Benchmark
    v1_spec = ilu.spec_from_file_location(
        "_v1_lp", str(ROOT / "submissions" / "vmallela" / "placer.py"))
    v1 = ilu.module_from_spec(v1_spec)
    v1_spec.loader.exec_module(v1)
    pt_path = str(ROOT / "benchmarks" / "processed" / "public" /
                   f"{bench_name}.pt")
    # Some local .pt files carry extra fields (e.g. ibm15.pt has
    # net_pin_nodes). Strip unknown fields before constructing Benchmark.
    raw = torch.load(pt_path, weights_only=False)
    valid_keys = set(Benchmark.__dataclass_fields__.keys())
    filtered = {k: v for k, v in raw.items() if k in valid_keys}
    if "num_hard_macros" not in filtered:
        filtered["num_hard_macros"] = filtered["num_macros"]
        filtered["num_soft_macros"] = 0
    if "soft_macro_indices" not in filtered:
        filtered["soft_macro_indices"] = []
    if "port_positions" not in filtered:
        import torch as _t
        filtered["port_positions"] = _t.zeros(0, 2)
    if "macro_pin_offsets" not in filtered:
        filtered["macro_pin_offsets"] = []
    bench = Benchmark(**filtered)
    plc = v1._load_plc(bench.name)
    incr = v1.IncrementalEvaluator(plc, bench)
    # Compute n_total (= total macros, hard + soft) from macro_pos shape.
    if not hasattr(incr, "n_total"):
        incr.n_total = int(np.asarray(incr.macro_pos).shape[0])
    return bench, incr


def build_lp_axis(incr, axis="x"):
    """Build (c, A_ub, b_ub) for the 1D LP on the chosen axis.

    Variables: [x_0, ..., x_{n_macros-1}, s_0, ..., s_{n_nets-1},
                 u_0, ..., u_{n_nets-1}, l_0, ..., l_{n_nets-1}]
    Total: n_macros + 3·n_nets.

    Constraints:
        u_n - x_m - dx_p ≥ 0       for each pin p (in net n owned by macro m)
        l_n - x_m - dx_p ≤ 0       for each pin p
        s_n - u_n + l_n ≥ 0        (so that s_n ≥ HPWL_n; we minimize s)

    For port pins, dx_p is the absolute port position and there's no x_m
    contribution.
    """
    n_macros = int(incr.n_total)
    pin_macro = np.asarray(incr.pin_macro, dtype=np.int64)
    if axis == "x":
        pin_off = np.asarray(incr.pin_xoff, dtype=np.float64)
    else:
        pin_off = np.asarray(incr.pin_yoff, dtype=np.float64)
    net_starts = np.asarray(incr.net_starts, dtype=np.int64)
    net_weight = np.asarray(incr.net_weight, dtype=np.float64)
    n_nets = int(net_starts.shape[0] - 1)
    n_pins = int(pin_macro.shape[0])

    # Variable indexing
    x_off = 0
    s_off = n_macros
    u_off = n_macros + n_nets
    l_off = n_macros + 2 * n_nets
    n_vars = n_macros + 3 * n_nets

    # Objective: minimize sum_n w_n · s_n
    c = np.zeros(n_vars)
    c[s_off:s_off + n_nets] = net_weight

    # Build constraints in (A_ub @ x ≤ b_ub) form.
    # Inequality 1 (u_n ≥ x_m + dx_p):  x_m + dx_p - u_n ≤ 0
    # Inequality 2 (l_n ≤ x_m + dx_p):  -x_m - dx_p + l_n ≤ 0
    # Equality 3 (s_n = u_n - l_n; we make it ≥ to allow slack but the
    #            objective forces tightness):  u_n - l_n - s_n ≤ 0
    rows = []
    cols = []
    data = []
    rhs = []

    # rows are accumulated in order; row index increments
    row = 0
    for nid in range(n_nets):
        for pin_idx in range(net_starts[nid], net_starts[nid + 1]):
            m = pin_macro[pin_idx]
            d = pin_off[pin_idx]
            if m >= 0:
                # ineq 1: x_m + d - u_n ≤ 0
                rows.append(row); cols.append(x_off + m); data.append(1.0)
                rows.append(row); cols.append(u_off + nid); data.append(-1.0)
                rhs.append(-d)
                row += 1
                # ineq 2: -x_m - d + l_n ≤ 0
                rows.append(row); cols.append(x_off + m); data.append(-1.0)
                rows.append(row); cols.append(l_off + nid); data.append(1.0)
                rhs.append(d)
                row += 1
            else:
                # port pin: position is fixed at d; no x_m term
                # ineq 1:  d - u_n ≤ 0  →  -u_n ≤ -d
                rows.append(row); cols.append(u_off + nid); data.append(-1.0)
                rhs.append(-d)
                row += 1
                # ineq 2: -d + l_n ≤ 0  →   l_n ≤ d
                rows.append(row); cols.append(l_off + nid); data.append(1.0)
                rhs.append(d)
                row += 1
        # ineq 3: u_n - l_n - s_n ≤ 0
        rows.append(row); cols.append(u_off + nid); data.append(1.0)
        rows.append(row); cols.append(l_off + nid); data.append(-1.0)
        rows.append(row); cols.append(s_off + nid); data.append(-1.0)
        rhs.append(0.0)
        row += 1

    A_ub = csr_matrix((data, (rows, cols)), shape=(row, n_vars))
    b_ub = np.asarray(rhs, dtype=np.float64)

    # Bounds: x_m ∈ [half_w, canvas - half_w]; s_n ≥ 0; u_n, l_n ∈ [0, canvas]
    canvas_extent = float(incr.cw if axis == "x" else incr.ch)
    bounds = []
    macro_extents = (np.asarray(incr.macro_w) if axis == "x"
                       else np.asarray(incr.macro_h))
    for m in range(n_macros):
        half = float(macro_extents[m]) / 2
        bounds.append((half, max(half, canvas_extent - half)))
    for _ in range(n_nets):
        bounds.append((0.0, None))   # s_n ≥ 0
    for _ in range(n_nets):
        bounds.append((0.0, canvas_extent))   # u_n
    for _ in range(n_nets):
        bounds.append((0.0, canvas_extent))   # l_n

    return c, A_ub, b_ub, bounds, n_vars, n_macros, n_nets


def solve_l1_hpwl_lb(bench_name, verbose=False):
    bench, incr = _load_incr(bench_name)
    print(f"\n=== {bench_name} ===", flush=True)
    print(f"  {bench}", flush=True)
    n_macros = int(incr.n_total)
    n_nets = int(incr.net_starts.shape[0] - 1)
    n_pins = int(incr.pin_macro.shape[0])
    print(f"  n_macros (total)={n_macros}, n_nets={n_nets}, n_pins={n_pins}",
          flush=True)
    print(f"  canvas: {float(incr.cw):.2f} × {float(incr.ch):.2f}", flush=True)

    results = {}
    for axis in ("x", "y"):
        t0 = time.time()
        c, A_ub, b_ub, bounds, n_vars, _, _ = build_lp_axis(incr, axis=axis)
        print(f"  [{axis}] LP: {n_vars} vars, {A_ub.shape[0]} constraints, "
              f"{A_ub.nnz} nonzeros (build {time.time()-t0:.1f}s)",
              flush=True)
        t1 = time.time()
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds,
                       method="highs", options={"disp": verbose})
        wall = time.time() - t1
        if not res.success:
            print(f"  [{axis}] LP FAILED: {res.message}", flush=True)
            results[axis] = None
            continue
        # Recover sum of net widths in this axis
        s_off = n_macros
        s_vals = res.x[s_off:s_off + n_nets]
        net_weight = np.asarray(incr.net_weight)
        hpwl_axis = float((s_vals * net_weight).sum())
        print(f"  [{axis}] LP solved in {wall:.1f}s, "
              f"HPWL_{axis} = {hpwl_axis:.4f}",
              flush=True)
        results[axis] = (hpwl_axis, wall, res.x.copy())

    if results["x"] is None or results["y"] is None:
        return None
    hpwl_total = results["x"][0] + results["y"][0]
    # incr's normalization: divide by net_count and (cw + ch)
    cw_ch_sum = float(incr.cw) + float(incr.ch)
    net_cnt = float(incr.net_cnt)
    hpwl_normalized = hpwl_total / (net_cnt * cw_ch_sum)
    print(f"  total HPWL_LP (raw): {hpwl_total:.4f}", flush=True)
    print(f"  total HPWL_LP normalized: {hpwl_normalized:.6f}", flush=True)
    print(f"  *** L1 HPWL lower bound (proxy ≥ this): {hpwl_normalized:.6f} ***",
          flush=True)
    return {
        "bench": bench_name,
        "n_macros": n_macros,
        "n_nets": n_nets,
        "n_pins": n_pins,
        "hpwl_x_lb": results["x"][0],
        "hpwl_y_lb": results["y"][0],
        "hpwl_raw_lb": hpwl_total,
        "hpwl_normalized_lb": hpwl_normalized,
        "wall_x_s": round(results["x"][1], 2),
        "wall_y_s": round(results["y"][1], 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="ibm01")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    if args.benchmark == "all":
        benches = ["ibm01", "ibm02", "ibm03", "ibm04", "ibm06", "ibm07",
                   "ibm08", "ibm09", "ibm10", "ibm11", "ibm12", "ibm13",
                   "ibm14", "ibm15", "ibm16", "ibm17", "ibm18"]
    else:
        benches = [args.benchmark]

    results = []
    for b in benches:
        r = solve_l1_hpwl_lb(b, verbose=args.verbose)
        if r:
            results.append(r)

    out = ROOT / "research" / "lower_bounds" / "l1_hpwl_lb_results.csv"
    if results:
        with open(out, "w") as f:
            keys = list(results[0].keys())
            f.write(",".join(keys) + "\n")
            for r in results:
                f.write(",".join(str(r[k]) for k in keys) + "\n")
        print(f"\nresults → {out}")


if __name__ == "__main__":
    main()
