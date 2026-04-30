#!/usr/bin/env python3
"""Hessian negative-eigenvalue escape smoke on ibm15.

1. Run a baseline v4 pipeline on ibm15 from .plc init at 1200s.
   Capture the placement (close to v6+Lap basin).
2. At that placement, compute Hessian of smooth surrogate. Find smallest
   eigenvalue + eigenvector via Lanczos.
3. Generate K candidate perturbations along ±v_min at multiple step sizes.
4. For each candidate: run a SHORT v4 pipeline (300s) from the perturbed
   state. Validate exact cost. Strict-improvement gate.
5. Take the best across all candidates. Compare to baseline.

If any Hessian-perturbed candidate beats baseline → win, scale to multi-
worker. Otherwise → bust, move to GNC.
"""
from __future__ import annotations
import os
import sys
import time
import multiprocessing as mp
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def _baseline_worker(args):
    """Run v4 pipeline from .plc init at the given budget."""
    bench_path, budget, seed = args
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["PLACER_TOTAL_BUDGET"] = str(budget)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "submissions" / "vmallela"))
    sys.path.insert(0, str(ROOT / "submissions" / "vmallela_v2"))
    sys.path.insert(0, str(ROOT / "submissions" / "vmallela_v6"))
    sys.path.insert(0, str(ROOT / "submissions" / "vmallela_v7"))
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_v4_pl", str(ROOT / "submissions" / "vmallela_v2" / "placer.py"))
    v4 = importlib.util.module_from_spec(spec); spec.loader.exec_module(v4)
    spec1 = importlib.util.spec_from_file_location(
        "_v1_pl", str(ROOT / "submissions" / "vmallela" / "placer.py"))
    v1 = importlib.util.module_from_spec(spec1); spec1.loader.exec_module(v1)
    from macro_place.benchmark import Benchmark
    from macro_place.objective import compute_proxy_cost
    plc = v1._load_plc("ibm15")
    bench = Benchmark.load(bench_path)
    placer = v4.OptimalPlacer(seed=seed)
    pos = placer.place(bench)
    if isinstance(pos, torch.Tensor):
        pos_t = pos.detach()
    else:
        pos_t = torch.tensor(pos, dtype=torch.float32)
    r = compute_proxy_cost(pos_t, bench, plc)
    return (np.asarray(pos_t.cpu().numpy()), float(r["proxy_cost"]),
            int(r["overlap_count"]))


def _candidate_worker(args):
    """Run v4 pipeline from a given init state at short budget."""
    (bench_path, init_pos_bytes, shape, dtype_str, budget, seed,
     label) = args
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["PLACER_TOTAL_BUDGET"] = str(budget)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "submissions" / "vmallela"))
    sys.path.insert(0, str(ROOT / "submissions" / "vmallela_v2"))
    sys.path.insert(0, str(ROOT / "submissions" / "vmallela_v6"))
    sys.path.insert(0, str(ROOT / "submissions" / "vmallela_v7"))
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_v4_pl", str(ROOT / "submissions" / "vmallela_v2" / "placer.py"))
    v4 = importlib.util.module_from_spec(spec); spec.loader.exec_module(v4)
    spec1 = importlib.util.spec_from_file_location(
        "_v1_pl", str(ROOT / "submissions" / "vmallela" / "placer.py"))
    v1 = importlib.util.module_from_spec(spec1); spec1.loader.exec_module(v1)
    from macro_place.benchmark import Benchmark
    from macro_place.objective import compute_proxy_cost
    plc = v1._load_plc("ibm15")
    bench = Benchmark.load(bench_path)
    init_pos = np.frombuffer(
        init_pos_bytes, dtype=np.dtype(dtype_str)).reshape(shape).copy()
    bench.macro_positions = torch.tensor(
        init_pos, dtype=bench.macro_positions.dtype)
    placer = v4.OptimalPlacer(seed=seed)
    pos = placer.place(bench)
    if isinstance(pos, torch.Tensor):
        pos_t = pos.detach()
    else:
        pos_t = torch.tensor(pos, dtype=torch.float32)
    r = compute_proxy_cost(pos_t, bench, plc)
    return (label, float(r["proxy_cost"]), int(r["overlap_count"]))


def main():
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "submissions" / "vmallela"))
    sys.path.insert(0, str(ROOT / "submissions" / "vmallela_v7"))
    from macro_place.benchmark import Benchmark
    import importlib.util
    spec1 = importlib.util.spec_from_file_location(
        "_v1_pl_main", str(ROOT / "submissions" / "vmallela" / "placer.py"))
    v1 = importlib.util.module_from_spec(spec1); spec1.loader.exec_module(v1)

    bench_path = str(ROOT / "benchmarks" / "processed" / "public" / "ibm15.pt")
    bench = Benchmark.load(bench_path)
    plc = v1._load_plc("ibm15")
    n_hard = bench.num_hard_macros
    cw = float(bench.canvas_width)
    ch = float(bench.canvas_height)
    canvas_diag = (cw ** 2 + ch ** 2) ** 0.5

    # ── Phase A: baseline v4 from .plc init, 1200s ────────────────────
    print("=" * 60)
    print(" Phase A: baseline v4 from .plc init (1200s)")
    print("=" * 60, flush=True)
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(1) as pool:
        result = pool.map(_baseline_worker,
                           [(bench_path, 1200, 42)])[0]
    base_pos_np, base_cost, base_ov = result
    base_wall = time.time() - t0
    print(f"\n  Baseline FINAL: cost={base_cost:.6f} ov={base_ov} "
          f"wall={base_wall:.0f}s", flush=True)

    # ── Phase B: compute Hessian eigenvector at baseline placement ───
    print()
    print("=" * 60)
    print(" Phase B: Hessian min-eigvector via Lanczos")
    print("=" * 60, flush=True)
    from _hessian_escape import hessian_escape_step
    from _smooth_proxy import (lse_hpwl_vectorized, build_pin_to_net,
                                 cvar_smooth, softplus_mu)
    from _cell_window import (build_window_indices, smooth_density_grid,
                                smooth_macro_blockage)

    # Build IncrementalEvaluator on baseline state to get all needed tensors
    incr = v1.IncrementalEvaluator(plc, bench)
    incr.macro_pos[:] = base_pos_np
    incr._recompute_pin_positions()
    incr._full_recompute_wl()
    incr._full_recompute_density()
    incr._full_recompute_congestion()

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    macro_pos_t = torch.tensor(
        np.asarray(incr.macro_pos), dtype=torch.float32, device=device)

    # Constants for surrogate
    pin_macro = torch.tensor(np.asarray(incr.pin_macro), dtype=torch.long, device=device)
    pin_xoff = torch.tensor(np.asarray(incr.pin_xoff), dtype=torch.float32, device=device)
    pin_yoff = torch.tensor(np.asarray(incr.pin_yoff), dtype=torch.float32, device=device)
    net_starts = torch.tensor(np.asarray(incr.net_starts), dtype=torch.long, device=device)
    net_weight = torch.tensor(np.asarray(incr.net_weight), dtype=torch.float32, device=device)
    macro_w_t = torch.tensor(np.asarray(incr.macro_w), dtype=torch.float32, device=device)
    macro_h_t = torch.tensor(np.asarray(incr.macro_h), dtype=torch.float32, device=device)
    pin_to_net = build_pin_to_net(net_starts)
    n_nets = int(net_weight.shape[0])

    # Get cell-window indices (refresh once for this state)
    cell_idx_d, _ = build_window_indices(
        macro_pos_t.detach(), macro_w_t, macro_h_t,
        grid_col=incr.grid_col, grid_row=incr.grid_row,
        grid_w=incr.grid_width, grid_h=incr.grid_height,
        margin_cells=4)

    cw_f, ch_f = float(incr.cw), float(incr.ch)
    net_cnt = float(incr.net_cnt)

    K_d = max(1, int(0.10 * incr.n_cells))

    def smooth_proxy_call(macro_pos_var):
        """Smooth surrogate: HPWL + 0.5 · CVaR_top10%(density). No cong
        (since the cong surrogate is too far from exact). Macros are
        the only learnable variables; t is replaced by quantile."""
        # HPWL
        is_port = (pin_macro < 0)
        safe = torch.where(is_port, torch.zeros_like(pin_macro), pin_macro)
        macro_xy = macro_pos_var[safe]
        pin_x = torch.where(is_port, pin_xoff, macro_xy[:, 0] + pin_xoff)
        pin_y = torch.where(is_port, pin_yoff, macro_xy[:, 1] + pin_yoff)
        hpwl = lse_hpwl_vectorized(
            pin_x, pin_y, pin_to_net, net_weight, n_nets,
            cw=cw_f, ch=ch_f, net_cnt=net_cnt, tau_lse=50.0)
        # Density
        rho = smooth_density_grid(
            macro_pos_var, macro_w_t, macro_h_t, cell_idx_d,
            incr.grid_col, incr.grid_row, incr.grid_width, incr.grid_height,
            n_cells=incr.n_cells, cell_area=incr.grid_area, mu=100.0)
        # Set t = quantile of detached rho (so it's not a learnable var)
        with torch.no_grad():
            t_d = torch.quantile(rho, 1.0 - K_d / incr.n_cells)
        density_smooth = cvar_smooth(rho.unsqueeze(0), K_d, t_d.detach(),
                                       mu=100.0).squeeze()
        return hpwl + 0.5 * density_smooth

    # Compute Hessian min-eigvec
    t_h = time.time()
    candidates, diag = hessian_escape_step(
        macro_pos_t, smooth_proxy_call,
        step_sizes=[0.02, 0.05, 0.10, 0.20],
        canvas_diag=canvas_diag,
        n_lanczos_iters=50,
        n_hard=n_hard,
        soft_only_perturb=True,
        verbose=True)
    print(f"  Hessian eigenvector computed in {time.time()-t_h:.1f}s",
          flush=True)
    print(f"  λ_min = {diag.get('lambda_min', 'N/A'):.6f}, "
          f"||v|| pre-norm = {diag.get('v_norm_pre', 0):.3f}", flush=True)
    if not candidates:
        print("  No candidates generated. Aborting.")
        return

    # ── Phase C: run v4 from each perturbed candidate ────────────────
    print()
    print("=" * 60)
    print(f" Phase C: run v4 from {len(candidates)} candidates (300s each)")
    print("=" * 60, flush=True)

    args = []
    for s, pos_np in candidates:
        pos64 = np.ascontiguousarray(pos_np, dtype=np.float64)
        args.append((bench_path, pos64.tobytes(), pos64.shape,
                      str(pos64.dtype), 300, 42, f"step={s:+.3f}"))

    t0 = time.time()
    with ctx.Pool(min(len(args), 8)) as pool:
        results = pool.map(_candidate_worker, args)
    print(f"  All workers done in {time.time()-t0:.0f}s wall.", flush=True)

    print()
    print(f"{'config':>16} {'cost':>10} {'ov':>4}")
    print("=" * 32)
    results.sort(key=lambda r: r[1])
    for label, cost, ov in results:
        marker = " ← BEST" if cost == results[0][1] else ""
        print(f"{label:>16} {cost:>10.4f} {ov:>4}{marker}")

    print()
    print("=" * 60)
    print(" VERDICT")
    print("=" * 60)
    print(f"  Baseline (1200s): {base_cost:.4f}")
    best_pert = min(results, key=lambda r: (r[2], r[1]))
    print(f"  Best perturbed (300s): {best_pert[0]} cost={best_pert[1]:.4f}")
    print(f"  Δ vs baseline: {best_pert[1] - base_cost:+.4f}")
    if best_pert[1] < base_cost - 0.005:
        print("  WIN: Hessian escape produces a meaningfully better basin.")
    elif best_pert[1] < base_cost + 0.020:
        print("  MIXED: comparable basins but pipeline only had 300s.")
        print("  Need to retest with full 1200s budget per candidate.")
    else:
        print("  BUST: Hessian escape produces worse basins.")


if __name__ == "__main__":
    main()
