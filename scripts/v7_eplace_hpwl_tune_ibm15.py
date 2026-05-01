#!/usr/bin/env python3
"""HPWL-aware ePlace tuning sweep on ibm15.

ePlace force = β · ∇ψ_density + α · ∇HPWL (centroid pull).

Density mean force on ibm15 ≈ 2.4 per macro.
HPWL mean force ≈ 107 per macro (46× larger).
So balanced α ≈ 0.02. Test α ∈ {0.0, 0.01, 0.02, 0.05, 0.1} at n_steps=30
to find the sweet spot.

Each config run on ibm15 with 600s v4 budget. Parallel via mp.Pool.
ETA ~12 min wall.
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


def _worker(args):
    (bench_path, init_pos_bytes, shape, dtype_str, budget, seed, label) = args
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
    t0 = time.time()
    pos = placer.place(bench)
    wall = time.time() - t0
    if isinstance(pos, torch.Tensor):
        pos = pos.detach()
    else:
        pos = torch.tensor(pos, dtype=torch.float32)
    r = compute_proxy_cost(pos, bench, plc)
    return (label, float(r["proxy_cost"]),
            int(r["overlap_count"]), wall)


def main():
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "submissions" / "vmallela"))
    sys.path.insert(0, str(ROOT / "submissions" / "vmallela_v7"))
    from macro_place.benchmark import Benchmark
    from _eplace import (_density_grid, _solve_poisson_fft,
                          _grad_psi, _interpolate_force,
                          _hpwl_centroid_force)
    import importlib.util
    spec1 = importlib.util.spec_from_file_location(
        "_v1_pl_main", str(ROOT / "submissions" / "vmallela" / "placer.py"))
    v1 = importlib.util.module_from_spec(spec1); spec1.loader.exec_module(v1)

    bench_path = str(ROOT / "benchmarks" / "processed" / "public" / "ibm15.pt")
    bench = Benchmark.load(bench_path)
    plc = v1._load_plc("ibm15")
    ie = v1.IncrementalEvaluator(plc, bench)

    n_hard = bench.num_hard_macros
    sizes = bench.macro_sizes.cpu().numpy().astype(np.float64)
    init_centers = bench.macro_positions.cpu().numpy().astype(np.float64)
    init_ll = init_centers.copy()
    init_ll[:, 0] -= sizes[:, 0] / 2.0
    init_ll[:, 1] -= sizes[:, 1] / 2.0
    canvas_w = float(bench.canvas_width)
    canvas_h = float(bench.canvas_height)
    grid_col = plc.grid_col
    grid_row = plc.grid_row
    grid_w = canvas_w / grid_col
    grid_h = canvas_h / grid_row

    pin_macro = np.asarray(ie.pin_macro)
    pin_xoff = np.asarray(ie.pin_xoff)
    pin_yoff = np.asarray(ie.pin_yoff)
    net_starts = np.asarray(ie.net_starts)
    net_weight = np.asarray(ie.net_weight)

    # ── Test α values ─────────────────────────────────────────────────
    alphas = [0.0, 0.01, 0.02, 0.05, 0.1]
    n_steps = 30
    snapshots = {0.0: init_ll.copy()}   # baseline = .plc init

    import math
    canvas_diag = math.hypot(canvas_w, canvas_h)
    lr0 = 0.005 * canvas_diag

    for alpha in alphas[1:]:   # 0.0 already saved
        print(f"\nRunning HPWL-aware ePlace α={alpha} for {n_steps} steps...",
              flush=True)
        pos = init_ll.copy().astype(np.float64)
        velocity = np.zeros_like(pos)
        for step in range(n_steps):
            density = _density_grid(pos, sizes[:, 0], sizes[:, 1],
                                      grid_col, grid_row, grid_w, grid_h)
            rhs = density - density.mean()
            psi = _solve_poisson_fft(rhs, grid_w, grid_h)
            gx, gy = _grad_psi(psi, grid_w, grid_h)
            fd = _interpolate_force(gx, gy, pos, sizes[:, 0], sizes[:, 1],
                                      grid_w, grid_h)
            pos_centers = pos.copy()
            pos_centers[:, 0] += sizes[:, 0] / 2.0
            pos_centers[:, 1] += sizes[:, 1] / 2.0
            fh = _hpwl_centroid_force(
                pos_centers, pin_macro, pin_xoff, pin_yoff,
                net_starts, net_weight)
            force = fd + alpha * fh
            if n_hard > 0:
                spring = -(pos[:n_hard] - init_ll[:n_hard]) * 10.0
                force[:n_hard] = spring
            lr_step = lr0 * 0.5 * (1 + math.cos(math.pi * step / n_steps))
            velocity = 0.9 * velocity + lr_step * force
            pos = pos + velocity
            np.clip(pos[:, 0], 0.0, canvas_w - sizes[:, 0], out=pos[:, 0])
            np.clip(pos[:, 1], 0.0, canvas_h - sizes[:, 1], out=pos[:, 1])
        max_dens_final = density.max()
        print(f"  α={alpha} step {n_steps}: max_dens={max_dens_final:.2f}",
              flush=True)
        snapshots[alpha] = pos.copy()

    # ── Run v4 from each snapshot in parallel ────────────────────────
    budget = 600
    args = []
    for alpha, pos_ll in snapshots.items():
        centers = pos_ll.copy()
        centers[:, 0] += sizes[:, 0] / 2.0
        centers[:, 1] += sizes[:, 1] / 2.0
        c64 = np.ascontiguousarray(centers, dtype=np.float64)
        args.append((bench_path, c64.tobytes(), c64.shape, str(c64.dtype),
                      budget, 42, f"alpha={alpha}"))

    print(f"\nRunning {len(args)} v4 pipelines in parallel ({budget}s each)...",
          flush=True)
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(len(args)) as pool:
        results = pool.map(_worker, args)
    print(f"\nAll workers done in {time.time()-t0:.0f}s wall.\n", flush=True)

    results.sort(key=lambda r: float(r[0].split("=")[1]))
    print(f"{'config':>12} {'cost':>10} {'ov':>4} {'wall(s)':>8}")
    print("=" * 38)
    for label, cost, ov, wall in results:
        print(f"{label:>12} {cost:>10.4f} {ov:>4} {wall:>8.0f}")

    baseline = next((r for r in results if "alpha=0.0" in r[0]), None)
    if baseline:
        b_cost = baseline[1]
        print(f"\nBaseline (α=0): {b_cost:.4f}")
        wins = [r for r in results
                 if r[1] < b_cost - 0.005 and "alpha=0.0" not in r[0]]
        if wins:
            print(f"WINS: {len(wins)} config(s) beat baseline by ≥0.005")
            for w in wins:
                print(f"  {w[0]}: {w[1]:.4f} (Δ {w[1] - b_cost:+.4f})")
            print("VERDICT: HPWL-aware ePlace WORKS. Sweet spot found.")
        else:
            best = min(results, key=lambda r: r[1])
            print(f"BEST: {best[0]} cost={best[1]:.4f} "
                  f"(Δ vs baseline {best[1] - b_cost:+.4f})")
            print("VERDICT: HPWL-aware ePlace doesn't help. Move on.")


if __name__ == "__main__":
    main()
