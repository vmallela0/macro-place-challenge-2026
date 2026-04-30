#!/usr/bin/env python3
"""Confirmation smoke for Hessian escape at full 1200s budget.

Initial smoke at 300s showed all 8 step sizes beat 1200s baseline.
Re-run the top 3 step sizes at full 1200s to confirm:
- The win isn't a fluke of the short-budget pipeline lucking out
- The full v4 pipeline at 1200s from Hessian-perturbed init lands
  meaningfully below the 1200s .plc baseline

Top 3 from prior smoke:
  step=+0.020   1.1069  (BEST at 300s)
  step=-0.050   1.1074
  step=-0.020   1.1076

Plus the same baseline (.plc init, step=0.0) for control.
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


def _candidate_worker(args):
    """Run v4 pipeline from a given init state at the given budget."""
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
    if init_pos_bytes is not None:
        init_pos = np.frombuffer(
            init_pos_bytes, dtype=np.dtype(dtype_str)).reshape(shape).copy()
        bench.macro_positions = torch.tensor(
            init_pos, dtype=bench.macro_positions.dtype)
    placer = v4.OptimalPlacer(seed=seed)
    t0 = time.time()
    pos = placer.place(bench)
    wall = time.time() - t0
    if isinstance(pos, torch.Tensor):
        pos_t = pos.detach()
    else:
        pos_t = torch.tensor(pos, dtype=torch.float32)
    r = compute_proxy_cost(pos_t, bench, plc)
    return (label, float(r["proxy_cost"]), int(r["overlap_count"]), wall)


def main():
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "submissions" / "vmallela"))
    sys.path.insert(0, str(ROOT / "submissions" / "vmallela_v7"))
    from macro_place.benchmark import Benchmark
    from _hessian_escape import hessian_escape_step
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
    cw = float(bench.canvas_width)
    ch = float(bench.canvas_height)
    canvas_diag = (cw ** 2 + ch ** 2) ** 0.5

    # ── Phase A: 1200s baseline (from .plc) ─────────────────────────
    print("Phase A: baseline v4 from .plc init (1200s)", flush=True)
    ctx = mp.get_context("spawn")

    # ── Phase B: regenerate Hessian eigvec ───────────────────────────
    # First we need a placement to compute Hessian at. Use the prior
    # smoke's baseline result. To keep this self-contained, re-run baseline.
    # Phase A also lets us compute Hessian at the same state.
    # For efficiency, run A and the perturbed runs in PARALLEL, but
    # we need the baseline result for Hessian computation. So:
    # 1. Run baseline v4 1200s (single Pool worker)
    # 2. Compute Hessian eigvec at baseline result
    # 3. Generate perturbed candidates
    # 4. Run candidates 1200s in parallel

    t0 = time.time()
    with ctx.Pool(1) as pool:
        baseline_result = pool.map(_candidate_worker,
                                    [(bench_path, None, None, None,
                                       1200, 42, "baseline")])[0]
    label, base_cost, base_ov, base_wall = baseline_result
    print(f"\n  Baseline FINAL: cost={base_cost:.6f} ov={base_ov} "
          f"wall={base_wall:.0f}s", flush=True)

    # Re-run baseline locally to get the placement (Pool result didn't include it)
    # Actually we need the position. Let me run again locally with the same
    # seed — should give same result (deterministic).
    # Actually that's wasteful. Let me get the position via a second
    # _candidate_worker call that returns the position.

    # Simpler: run baseline single-thread here to get position
    print("\n  Re-running baseline locally to capture placement...", flush=True)
    sys.path.insert(0, str(ROOT / "submissions" / "vmallela_v2"))
    spec_v4 = importlib.util.spec_from_file_location(
        "_v4_pl_main", str(ROOT / "submissions" / "vmallela_v2" / "placer.py"))
    v4 = importlib.util.module_from_spec(spec_v4); spec_v4.loader.exec_module(v4)
    os.environ["PLACER_TOTAL_BUDGET"] = "1200"
    bench_local = Benchmark.load(bench_path)
    placer = v4.OptimalPlacer(seed=42)
    t1 = time.time()
    base_pos = placer.place(bench_local)
    if isinstance(base_pos, torch.Tensor):
        base_pos_np = np.asarray(base_pos.detach().cpu().numpy())
    else:
        base_pos_np = np.asarray(base_pos)
    print(f"  Local baseline placement captured in {time.time()-t1:.0f}s.",
          flush=True)

    # Compute Hessian eigvec at baseline placement
    print("\nPhase B: Hessian min-eigvector via Lanczos", flush=True)
    incr = v1.IncrementalEvaluator(plc, bench)
    incr.macro_pos[:] = base_pos_np
    incr._recompute_pin_positions()
    incr._full_recompute_wl()
    incr._full_recompute_density()
    incr._full_recompute_congestion()

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    macro_pos_t = torch.tensor(np.asarray(incr.macro_pos), dtype=torch.float32, device=device)
    pin_macro = torch.tensor(np.asarray(incr.pin_macro), dtype=torch.long, device=device)
    pin_xoff = torch.tensor(np.asarray(incr.pin_xoff), dtype=torch.float32, device=device)
    pin_yoff = torch.tensor(np.asarray(incr.pin_yoff), dtype=torch.float32, device=device)
    net_starts = torch.tensor(np.asarray(incr.net_starts), dtype=torch.long, device=device)
    net_weight = torch.tensor(np.asarray(incr.net_weight), dtype=torch.float32, device=device)
    macro_w_t = torch.tensor(np.asarray(incr.macro_w), dtype=torch.float32, device=device)
    macro_h_t = torch.tensor(np.asarray(incr.macro_h), dtype=torch.float32, device=device)
    pin_to_net = build_pin_to_net(net_starts)
    n_nets = int(net_weight.shape[0])
    cell_idx_d, _ = build_window_indices(
        macro_pos_t.detach(), macro_w_t, macro_h_t,
        grid_col=incr.grid_col, grid_row=incr.grid_row,
        grid_w=incr.grid_width, grid_h=incr.grid_height,
        margin_cells=4)
    cw_f, ch_f = float(incr.cw), float(incr.ch)
    net_cnt = float(incr.net_cnt)
    K_d = max(1, int(0.10 * incr.n_cells))

    def smooth_proxy_call(macro_pos_var):
        is_port = (pin_macro < 0)
        safe = torch.where(is_port, torch.zeros_like(pin_macro), pin_macro)
        macro_xy = macro_pos_var[safe]
        pin_x = torch.where(is_port, pin_xoff, macro_xy[:, 0] + pin_xoff)
        pin_y = torch.where(is_port, pin_yoff, macro_xy[:, 1] + pin_yoff)
        hpwl = lse_hpwl_vectorized(
            pin_x, pin_y, pin_to_net, net_weight, n_nets,
            cw=cw_f, ch=ch_f, net_cnt=net_cnt, tau_lse=50.0)
        rho = smooth_density_grid(
            macro_pos_var, macro_w_t, macro_h_t, cell_idx_d,
            incr.grid_col, incr.grid_row, incr.grid_width, incr.grid_height,
            n_cells=incr.n_cells, cell_area=incr.grid_area, mu=100.0)
        with torch.no_grad():
            t_d = torch.quantile(rho, 1.0 - K_d / incr.n_cells)
        density_smooth = cvar_smooth(rho.unsqueeze(0), K_d, t_d.detach(),
                                       mu=100.0).squeeze()
        return hpwl + 0.5 * density_smooth

    candidates, diag = hessian_escape_step(
        macro_pos_t, smooth_proxy_call,
        step_sizes=[0.020, 0.050],   # top step sizes from prior smoke
        canvas_diag=canvas_diag,
        n_lanczos_iters=50,
        n_hard=n_hard, soft_only_perturb=True,
        verbose=True)
    # Filter to top 3 step sizes from prior smoke: +0.020, -0.050, -0.020.
    target_steps = {0.020, -0.050, -0.020}
    selected = [(s, p) for s, p in candidates if s in target_steps]
    print(f"  Selected {len(selected)} candidates: "
          f"{[s for s, _ in selected]}", flush=True)

    # ── Phase C: run candidates at full 1200s ───────────────────────
    print("\nPhase C: 3 candidates × 1200s in parallel", flush=True)
    args = []
    for s, pos_np in selected:
        pos64 = np.ascontiguousarray(pos_np, dtype=np.float64)
        args.append((bench_path, pos64.tobytes(), pos64.shape,
                      str(pos64.dtype), 1200, 42,
                      f"hessian_step={s:+.3f}"))
    t0 = time.time()
    with ctx.Pool(len(args)) as pool:
        results = pool.map(_candidate_worker, args)
    print(f"  All 3 done in {time.time()-t0:.0f}s wall.\n", flush=True)

    print(f"{'config':>22} {'cost':>10} {'ov':>4} {'wall':>6}")
    print("=" * 46)
    print(f"{'baseline (1200s)':>22} {base_cost:>10.4f} {base_ov:>4} "
          f"{base_wall:>6.0f}")
    results.sort(key=lambda r: r[1])
    for label_, cost, ov, wall in results:
        print(f"{label_:>22} {cost:>10.4f} {ov:>4} {wall:>6.0f}")

    print("\nVERDICT")
    print("=" * 60)
    best = results[0]
    print(f"  Best Hessian-perturbed: {best[0]} cost={best[1]:.4f}")
    print(f"  Δ vs baseline: {best[1] - base_cost:+.4f}")
    if best[1] < base_cost - 0.005:
        print("  WIN: confirmed at full budget. SCALE TO MULTI-WORKER.")
    elif best[1] < base_cost + 0.005:
        print("  TIE: within noise. Direction is correct but signal weak.")
    else:
        print("  REGRESS: 300s win was a fluke. Investigate.")


if __name__ == "__main__":
    main()
