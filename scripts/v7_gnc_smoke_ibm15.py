#!/usr/bin/env python3
"""GNC (Graduated Non-Convexity) Hessian-escape smoke on ibm15.

Computes Hessian min-eigvecs at MULTIPLE smoothing scales:
  (τ_LSE, μ_softplus): (10, 50) coarse, (50, 100) medium, (200, 200) fine

Each scale gives a different escape direction (different scale of
landscape structure). Generates 12 candidates total (3 scales × 2 step
sizes × ±sign). Runs each from the same baseline placement at 600s
budget (or 1200s for stronger proof).

Compares to standalone Hessian smoke result (single scale, -0.017 lift).
If GNC's best beats single-scale Hessian's best by ≥0.005, GNC adds
real value beyond single-scale.
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
    from _gnc_hessian import gnc_hessian_escape_multi_scale
    from _smooth_proxy import (lse_hpwl_vectorized, build_pin_to_net,
                                 cvar_smooth)
    from _cell_window import build_window_indices, smooth_density_grid
    import importlib.util
    spec1 = importlib.util.spec_from_file_location(
        "_v1_pl_main", str(ROOT / "submissions" / "vmallela" / "placer.py"))
    v1 = importlib.util.module_from_spec(spec1); spec1.loader.exec_module(v1)

    bench_path = str(ROOT / "benchmarks" / "processed" / "public" / "ibm15.pt")
    bench = Benchmark.load(bench_path)
    plc = v1._load_plc("ibm15")
    n_hard = bench.num_hard_macros
    canvas_diag = (float(bench.canvas_width)**2 + float(bench.canvas_height)**2)**0.5

    # ── Phase A: baseline at 1200s ───────────────────────────────────
    print("=" * 60)
    print(" Phase A: baseline v4 (1200s)")
    print("=" * 60, flush=True)
    ctx = mp.get_context("spawn")
    with ctx.Pool(1) as pool:
        base_pos_np, base_cost, base_ov = pool.map(
            _baseline_worker, [(bench_path, 1200, 42)])[0]
    print(f"  Baseline FINAL: cost={base_cost:.6f} ov={base_ov}", flush=True)

    # ── Phase B: GNC multi-scale Hessian eigvecs ─────────────────────
    print()
    print("=" * 60)
    print(" Phase B: GNC multi-scale Hessian eigvecs")
    print("=" * 60, flush=True)
    incr = v1.IncrementalEvaluator(plc, bench)
    incr.macro_pos[:] = base_pos_np
    incr._recompute_pin_positions()
    incr._full_recompute_wl()
    incr._full_recompute_density()
    incr._full_recompute_congestion()

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    macro_pos_t = torch.tensor(np.asarray(incr.macro_pos), dtype=torch.float32, device=device)
    pin_macro_t = torch.tensor(np.asarray(incr.pin_macro), dtype=torch.long, device=device)
    pin_xoff_t = torch.tensor(np.asarray(incr.pin_xoff), dtype=torch.float32, device=device)
    pin_yoff_t = torch.tensor(np.asarray(incr.pin_yoff), dtype=torch.float32, device=device)
    net_starts_t = torch.tensor(np.asarray(incr.net_starts), dtype=torch.long, device=device)
    net_weight_t = torch.tensor(np.asarray(incr.net_weight), dtype=torch.float32, device=device)
    macro_w_t = torch.tensor(np.asarray(incr.macro_w), dtype=torch.float32, device=device)
    macro_h_t = torch.tensor(np.asarray(incr.macro_h), dtype=torch.float32, device=device)
    pin_to_net_t = build_pin_to_net(net_starts_t)
    n_nets = int(net_weight_t.shape[0])
    cell_idx_d, _ = build_window_indices(
        macro_pos_t.detach(), macro_w_t, macro_h_t,
        grid_col=incr.grid_col, grid_row=incr.grid_row,
        grid_w=incr.grid_width, grid_h=incr.grid_height,
        margin_cells=4)
    cw_f, ch_f = float(incr.cw), float(incr.ch)
    net_cnt = float(incr.net_cnt)
    K_d = max(1, int(0.10 * incr.n_cells))

    def proxy_factory(tau_lse, mu_softplus):
        def proxy_call(macro_pos_var):
            is_port = (pin_macro_t < 0)
            safe = torch.where(is_port, torch.zeros_like(pin_macro_t), pin_macro_t)
            macro_xy = macro_pos_var[safe]
            pin_x = torch.where(is_port, pin_xoff_t, macro_xy[:, 0] + pin_xoff_t)
            pin_y = torch.where(is_port, pin_yoff_t, macro_xy[:, 1] + pin_yoff_t)
            hpwl = lse_hpwl_vectorized(
                pin_x, pin_y, pin_to_net_t, net_weight_t, n_nets,
                cw=cw_f, ch=ch_f, net_cnt=net_cnt, tau_lse=tau_lse)
            rho = smooth_density_grid(
                macro_pos_var, macro_w_t, macro_h_t, cell_idx_d,
                incr.grid_col, incr.grid_row, incr.grid_width, incr.grid_height,
                n_cells=incr.n_cells, cell_area=incr.grid_area, mu=mu_softplus)
            with torch.no_grad():
                t_d = torch.quantile(rho, 1.0 - K_d / incr.n_cells)
            density_smooth = cvar_smooth(rho.unsqueeze(0), K_d, t_d.detach(),
                                           mu=mu_softplus).squeeze()
            return hpwl + 0.5 * density_smooth
        return proxy_call

    candidates, diag = gnc_hessian_escape_multi_scale(
        macro_pos_t, proxy_factory,
        tau_scales=[(10.0, 50.0), (50.0, 100.0), (200.0, 200.0)],
        step_sizes=[0.02, 0.05],
        canvas_diag=canvas_diag,
        n_lanczos_iters=50,
        n_hard=n_hard,
        soft_only_perturb=True,
        verbose=True)
    print(f"\n  Generated {len(candidates)} candidates across "
          f"{len(diag['per_scale'])} scales:", flush=True)
    for s in diag['per_scale']:
        print(f"    τ={s.get('tau_lse')} μ={s.get('mu')} → "
              f"λ_min={s.get('lambda_min', 'N/A'):.6f}", flush=True)

    # ── Phase C: run candidates at 600s each in parallel ─────────────
    print()
    print("=" * 60)
    print(f" Phase C: run {len(candidates)} candidates × 600s in parallel")
    print("=" * 60, flush=True)
    args = []
    for label, pos_np in candidates:
        pos64 = np.ascontiguousarray(pos_np, dtype=np.float64)
        args.append((bench_path, pos64.tobytes(), pos64.shape,
                      str(pos64.dtype), 600, 42, label))
    t0 = time.time()
    with ctx.Pool(min(len(args), 12)) as pool:
        results = pool.map(_candidate_worker, args)
    print(f"  All {len(results)} done in {time.time()-t0:.0f}s wall.\n",
          flush=True)
    results.sort(key=lambda r: r[1])
    print(f"{'config':>25} {'cost':>10} {'ov':>4}")
    print("=" * 41)
    for label, cost, ov in results:
        marker = " ← BEST" if cost == results[0][1] else ""
        print(f"{label:>25} {cost:>10.4f} {ov:>4}{marker}")

    print()
    print("=" * 60)
    print(" VERDICT")
    print("=" * 60)
    print(f"  Baseline (1200s):   {base_cost:.4f}")
    print(f"  Best GNC (600s):    {results[0][1]:.4f} ({results[0][0]})")
    print(f"  Δ vs baseline:      {results[0][1] - base_cost:+.4f}")
    if results[0][1] < base_cost - 0.005:
        print("  WIN: GNC produces meaningfully better basin.")
    elif results[0][1] < base_cost + 0.005:
        print("  TIE: comparable. Did 600s budget bottleneck?")
    else:
        print("  BUST: GNC didn't help.")


if __name__ == "__main__":
    main()
