#!/usr/bin/env python3
"""ePlace n_steps tuning on ibm15.

Run ePlace once for 100 steps, save snapshots at {0, 10, 30, 60, 100}.
For each snapshot, run the v4 pipeline (single worker, 600s budget) from
that warm-start state. Compare final costs to find the sweet spot.

Snapshots run in parallel (mp.Pool with 5 workers — small enough not to
saturate the machine). Wall time ≈ max(single v4 run) ≈ 12 min.

Hypothesis: n_steps=0 (no ePlace) is the .plc baseline. As n_steps grows,
ePlace spreads softs more aggressively. Initially spreading might help
(less density pressure on softs). Eventually too much spreading kills
HPWL. Sweet spot is a small n_steps where density drops without
disconnecting nets too much.
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
    """Run v4 pipeline from a given init state. Pickleable.
    args: (bench_path, init_pos_bytes, shape, dtype_str, budget, seed,
           n_steps_label).
    """
    (bench_path, init_pos_bytes, shape, dtype_str, budget, seed,
     n_steps_label) = args

    # Determinism
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
    v4 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v4)

    from macro_place.benchmark import Benchmark
    from macro_place.objective import compute_proxy_cost
    spec1 = importlib.util.spec_from_file_location(
        "_v1_pl", str(ROOT / "submissions" / "vmallela" / "placer.py"))
    v1 = importlib.util.module_from_spec(spec1)
    spec1.loader.exec_module(v1)
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
    return (n_steps_label, float(r["proxy_cost"]),
            int(r["overlap_count"]), wall)


def main():
    sys.path.insert(0, str(ROOT / "submissions" / "vmallela_v7"))
    sys.path.insert(0, str(ROOT))
    from macro_place.benchmark import Benchmark
    from _eplace import (eplace_warmstart, _density_grid, _solve_poisson_fft,
                          _grad_psi, _interpolate_force)

    bench_path = str(
        ROOT / "benchmarks" / "processed" / "public" / "ibm15.pt")
    bench = Benchmark.load(bench_path)
    n_hard = bench.num_hard_macros
    canvas_w = float(bench.canvas_width)
    canvas_h = float(bench.canvas_height)
    sizes = bench.macro_sizes.cpu().numpy().astype(np.float64)
    init_centers = bench.macro_positions.cpu().numpy().astype(np.float64)
    init_ll = init_centers.copy()
    init_ll[:, 0] -= sizes[:, 0] / 2.0
    init_ll[:, 1] -= sizes[:, 1] / 2.0

    # Get grid from .plc
    sys.path.insert(0, str(ROOT / "submissions" / "vmallela"))
    import importlib.util
    spec1 = importlib.util.spec_from_file_location(
        "_v1_pl_main", str(ROOT / "submissions" / "vmallela" / "placer.py"))
    v1 = importlib.util.module_from_spec(spec1)
    spec1.loader.exec_module(v1)
    plc = v1._load_plc("ibm15")
    grid_col = plc.grid_col
    grid_row = plc.grid_row

    # ── Run ePlace, capture snapshots ─────────────────────────────────
    print("Running ePlace 100 steps with snapshots at {0, 10, 30, 60, 100}…",
          flush=True)
    snapshots = {0: init_ll.copy()}
    snapshot_steps = {10, 30, 60, 100}

    # Reproduce the eplace loop manually to capture snapshots.
    grid_w = canvas_w / grid_col
    grid_h = canvas_h / grid_row
    canvas_diag = (canvas_w ** 2 + canvas_h ** 2) ** 0.5
    lr0 = 0.005 * canvas_diag

    pos = init_ll.copy().astype(np.float64)
    init_const = init_ll.copy()
    velocity = np.zeros_like(pos)
    n_steps_total = 100
    import math
    for step in range(n_steps_total):
        density = _density_grid(pos, sizes[:, 0], sizes[:, 1],
                                  grid_col, grid_row, grid_w, grid_h)
        rhs = density - density.mean()
        psi = _solve_poisson_fft(rhs, grid_w, grid_h)
        gx, gy = _grad_psi(psi, grid_w, grid_h)
        force = _interpolate_force(gx, gy, pos, sizes[:, 0], sizes[:, 1],
                                     grid_w, grid_h)
        if n_hard > 0:
            spring = -(pos[:n_hard] - init_const[:n_hard]) * 10.0
            force[:n_hard] = spring
        lr_step = lr0 * 0.5 * (1.0 + math.cos(math.pi * step / n_steps_total))
        velocity = 0.9 * velocity + lr_step * force
        pos = pos + velocity
        np.clip(pos[:, 0], 0.0, canvas_w - sizes[:, 0], out=pos[:, 0])
        np.clip(pos[:, 1], 0.0, canvas_h - sizes[:, 1], out=pos[:, 1])
        if (step + 1) in snapshot_steps:
            snapshots[step + 1] = pos.copy()
            print(f"  snapshot at step {step + 1}: max_dens={density.max():.2f}",
                  flush=True)

    # ── Build worker args ─────────────────────────────────────────────
    budget = 600
    args = []
    for n_steps_label, pos_ll in snapshots.items():
        # Convert lower-left back to center coords
        centers = pos_ll.copy()
        centers[:, 0] += sizes[:, 0] / 2.0
        centers[:, 1] += sizes[:, 1] / 2.0
        c64 = np.ascontiguousarray(centers, dtype=np.float64)
        args.append((bench_path, c64.tobytes(), c64.shape, str(c64.dtype),
                      budget, 42, n_steps_label))

    # ── Run all in parallel via spawn-context Pool ───────────────────
    print(f"\nRunning {len(args)} v4 pipelines in parallel ({budget}s each)...",
          flush=True)
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(len(args)) as pool:
        results = pool.map(_worker, args)
    print(f"\nAll workers done in {time.time()-t0:.0f}s wall.\n", flush=True)

    # ── Print results ────────────────────────────────────────────────
    results.sort(key=lambda r: r[0])
    print(f"{'n_steps':>10} {'cost':>10} {'ov':>4} {'wall(s)':>8}")
    print("=" * 36)
    for n_steps_label, cost, ov, wall in results:
        print(f"{n_steps_label:>10} {cost:>10.4f} {ov:>4} {wall:>8.0f}")

    best = min(results, key=lambda r: (r[2], r[1]))
    print(f"\nBest: n_steps={best[0]} cost={best[1]:.4f} ov={best[2]}")
    baseline = next((r for r in results if r[0] == 0), None)
    if baseline:
        print(f"Baseline (n_steps=0, .plc init): cost={baseline[1]:.4f}")
        print(f"Δ vs baseline: {best[1] - baseline[1]:+.4f}")
        if best[0] != 0 and best[1] < baseline[1] - 0.005:
            print("VERDICT: ePlace warm-start helps. Sweet spot found.")
        else:
            print("VERDICT: ePlace warm-start does not help meaningfully.")


if __name__ == "__main__":
    main()
