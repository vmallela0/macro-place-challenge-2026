#!/usr/bin/env python3
"""ePlace warm-start smoke: ibm15 only, single worker.

Compares two runs of the v4 pipeline:
  Run A: standard (init from .plc)
  Run B: ePlace warm-start (200 steps electrostatic spreading from .plc,
         then push-apart + legalize + full v4 pipeline)

If Run B's final cost is meaningfully different from Run A (in either
direction), ePlace generates a different basin and we proceed with a
multi-worker version. If Run B == Run A, ePlace doesn't change the
downstream basin and we abandon.

Per the rapid-experiments protocol, this smoke runs each at 1800s budget.
~60-90 min total wall-clock.
"""
from __future__ import annotations
import os
import sys
import time
from pathlib import Path
import numpy as np
import torch

# Locked env (matches submission run.sh)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("PYTHONHASHSEED", "42")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PLACER_TOTAL_BUDGET", "1200")
# Single-worker smoke; budget 1200s/run × 2 runs = ~40-50min total.

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "submissions" / "vmallela"))
sys.path.insert(0, str(ROOT / "submissions" / "vmallela_v2"))
sys.path.insert(0, str(ROOT / "submissions" / "vmallela_v6"))
sys.path.insert(0, str(ROOT / "submissions" / "vmallela_v7"))

from macro_place.benchmark import Benchmark
from macro_place.objective import compute_proxy_cost
import importlib.util


def load_v4_placer():
    spec = importlib.util.spec_from_file_location(
        "_v4_pl", str(ROOT / "submissions" / "vmallela_v2" / "placer.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def load_v1_placer():
    spec = importlib.util.spec_from_file_location(
        "_v1_pl", str(ROOT / "submissions" / "vmallela" / "placer.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def run_placer(benchmark, seed):
    """Run the v4 OptimalPlacer on the (possibly modified) benchmark."""
    v4 = load_v4_placer()
    placer = v4.OptimalPlacer(seed=seed)
    t0 = time.time()
    pos = placer.place(benchmark)
    return pos, time.time() - t0


def evaluate(pos, bench, plc):
    r = compute_proxy_cost(pos, bench, plc)
    return float(r["proxy_cost"]), int(r["overlap_count"])


def main():
    bench_path = ROOT / "benchmarks" / "processed" / "public" / "ibm15.pt"
    bench = Benchmark.load(str(bench_path))
    v1 = load_v1_placer()
    plc = v1._load_plc(bench.name)

    n_hard = bench.num_hard_macros
    n_total = bench.macro_positions.shape[0]
    canvas_w = float(bench.canvas_width)
    canvas_h = float(bench.canvas_height)
    grid_col = plc.grid_col
    grid_row = plc.grid_row
    macro_sizes = bench.macro_sizes.cpu().numpy().astype(np.float64)

    # The benchmark stores positions as CENTERS; ePlace operates on
    # lower-left corners. Convert.
    init_centers = bench.macro_positions.cpu().numpy().astype(np.float64)
    init_ll = init_centers.copy()
    init_ll[:, 0] -= macro_sizes[:, 0] / 2.0
    init_ll[:, 1] -= macro_sizes[:, 1] / 2.0

    # Validate the init via official scorer
    init_proxy, init_ov = evaluate(
        bench.macro_positions, bench, plc)
    print(f"\n=== ibm15 baseline init: cost={init_proxy:.4f} ov={init_ov} ===\n",
          flush=True)

    # ── Run A: standard v4 pipeline from .plc init ────────────────────
    print("=" * 60)
    print(" RUN A: v4 pipeline from .plc init")
    print("=" * 60, flush=True)
    bench_a = Benchmark.load(str(bench_path))
    posA, wallA = run_placer(bench_a, seed=42)
    if isinstance(posA, torch.Tensor):
        posA = posA.detach()
    else:
        posA = torch.tensor(posA, dtype=torch.float32)
    costA, ovA = evaluate(posA, bench_a, plc)
    print(f"\n  Run A FINAL: cost={costA:.6f} ov={ovA} wall={wallA:.0f}s",
          flush=True)

    # ── ePlace warm-start ─────────────────────────────────────────────
    print()
    print("=" * 60)
    print(" ePlace warm-start (200 steps)")
    print("=" * 60, flush=True)
    from _eplace import eplace_warmstart
    t_ep = time.time()
    eplace_pos_ll, hist = eplace_warmstart(
        init_ll, macro_sizes[:, 0], macro_sizes[:, 1],
        canvas_w, canvas_h, grid_col, grid_row,
        n_steps=200, lr_frac_canvas=0.005,
        n_hard=n_hard, hard_inertia=10.0,
        nesterov=True, verbose=True)
    print(f"  ePlace: {time.time()-t_ep:.1f}s; "
          f"max-dens {hist['max_density'][0]:.2f} → "
          f"{hist['max_density'][-1]:.2f}", flush=True)

    # Convert ePlace lower-left back to centers
    eplace_centers = eplace_pos_ll.copy()
    eplace_centers[:, 0] += macro_sizes[:, 0] / 2.0
    eplace_centers[:, 1] += macro_sizes[:, 1] / 2.0

    # Validate ePlace placement (likely overlapping; that's fine, we'll
    # let the v4 pipeline's push-apart + legalize fix it).
    eplace_tensor = torch.tensor(eplace_centers, dtype=torch.float32)
    ePlace_proxy, ePlace_ov = evaluate(eplace_tensor, bench, plc)
    print(f"  ePlace standalone proxy: {ePlace_proxy:.4f} ov={ePlace_ov} "
          f"(expected high; will be cleaned by v4 pipeline)", flush=True)

    # ── Run B: v4 pipeline from ePlace warm-start ─────────────────────
    print()
    print("=" * 60)
    print(" RUN B: v4 pipeline from ePlace warm-start")
    print("=" * 60, flush=True)
    bench_b = Benchmark.load(str(bench_path))
    bench_b.macro_positions = torch.tensor(
        eplace_centers, dtype=bench_b.macro_positions.dtype)
    posB, wallB = run_placer(bench_b, seed=42)
    if isinstance(posB, torch.Tensor):
        posB = posB.detach()
    else:
        posB = torch.tensor(posB, dtype=torch.float32)
    costB, ovB = evaluate(posB, bench_b, plc)
    print(f"\n  Run B FINAL: cost={costB:.6f} ov={ovB} wall={wallB:.0f}s",
          flush=True)

    # ── Verdict ──────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print(" VERDICT")
    print("=" * 60)
    print(f"  v4 ibm15 baseline:       1.1029")
    print(f"  v6+Lap recent baselines: 1.1218 - 1.1340")
    print(f"  Run A (.plc init):       {costA:.6f}")
    print(f"  Run B (ePlace init):     {costB:.6f}")
    print(f"  Δ (B - A):               {costB - costA:+.6f}")
    print()
    if abs(costB - costA) < 0.005:
        print("  ePlace doesn't change downstream basin meaningfully.")
        print("  Recommendation: ABANDON ePlace.")
    elif costB < costA:
        print(f"  ePlace warm-start WINS by {costA - costB:.4f}.")
        print("  Recommendation: SCALE TO MULTI-WORKER + 17-bench.")
    else:
        print(f"  ePlace warm-start LOSES by {costB - costA:.4f}.")
        print("  Different basin but worse. Tune ePlace (lr, n_steps, hard_inertia).")


if __name__ == "__main__":
    main()
