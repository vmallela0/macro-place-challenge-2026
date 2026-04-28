"""Equivalence test: MLXBatchEvaluator's HPWL+density+congestion vs the CPU
IncrementalEvaluator.

For a single-macro candidate move, the GPU evaluator should reproduce the CPU
HPWL and density terms exactly, and the congestion term up to the documented
"frozen routing demand" approximation (within a few %).
"""
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
# v6 first so its placer doesn't shadow vmallela's at module name `placer`
sys.path.insert(0, str(ROOT / "submissions" / "vmallela_v6"))

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "_v1_placer", str(ROOT / "submissions" / "vmallela" / "placer.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
_load_plc = _mod._load_plc
IncrementalEvaluator = _mod.IncrementalEvaluator

from macro_place.benchmark import Benchmark
import mlx.core as mx
from _mlx_eval import MLXBatchEvaluator


def _set_pos_in_incr(incr, macro_idx, x, y):
    """Apply move via incremental eval, return new exact proxy."""
    return incr.move_macro(macro_idx, x, y)


def _build():
    bench = Benchmark.load(str(ROOT / "benchmarks" / "processed" / "public" / "ibm01.pt"))
    plc = _load_plc("ibm01")
    incr = IncrementalEvaluator(plc, bench)
    init_pos = bench.macro_positions[:bench.num_hard_macros].numpy().copy().astype(np.float64)
    incr.sync_positions(init_pos)
    gpu = MLXBatchEvaluator(incr, bench)
    return bench, incr, gpu


def test_hpwl_only_matches_exactly():
    bench, incr, gpu = _build()
    rng = np.random.RandomState(0)
    n_hard = bench.num_hard_macros
    sizes = bench.macro_sizes[:n_hard].numpy()

    max_err_wl = 0.0
    n_trials = 60
    for _ in range(n_trials):
        m = int(rng.randint(0, n_hard))
        cw = float(bench.canvas_width)
        ch = float(bench.canvas_height)
        nx = float(np.clip(rng.uniform(sizes[m, 0] / 2, cw - sizes[m, 0] / 2),
                           sizes[m, 0] / 2, cw - sizes[m, 0] / 2))
        ny = float(np.clip(rng.uniform(sizes[m, 1] / 2, ch - sizes[m, 1] / 2),
                           sizes[m, 1] / 2, ch - sizes[m, 1] / 2))

        cands = mx.array(np.array([[nx, ny]], dtype=np.float32))
        wl, _d, _c = gpu.score_components(m, cands,
                                          skip_density=True, skip_congestion=True)
        gpu_wl = float(wl[0])

        _ = _set_pos_in_incr(incr, m, nx, ny)
        cpu_wl = float(incr.wirelength_cost)
        incr.undo_move()

        err = abs(gpu_wl - cpu_wl)
        max_err_wl = max(max_err_wl, err)
        assert err < 5e-4, (
            f"HPWL mismatch: macro={m} pos=({nx:.4f},{ny:.4f}) "
            f"gpu_wl={gpu_wl:.6f} cpu_wl={cpu_wl:.6f} err={err:.2e}")

    print(f"  HPWL: {n_trials} trials, max abs err = {max_err_wl:.2e}")


def test_density_matches_exactly():
    bench, incr, gpu = _build()
    rng = np.random.RandomState(1)
    n_hard = bench.num_hard_macros
    sizes = bench.macro_sizes[:n_hard].numpy()

    max_err = 0.0
    n_trials = 30
    for _ in range(n_trials):
        m = int(rng.randint(0, n_hard))
        cw = float(bench.canvas_width)
        ch = float(bench.canvas_height)
        nx = float(np.clip(rng.uniform(sizes[m, 0] / 2, cw - sizes[m, 0] / 2),
                           sizes[m, 0] / 2, cw - sizes[m, 0] / 2))
        ny = float(np.clip(rng.uniform(sizes[m, 1] / 2, ch - sizes[m, 1] / 2),
                           sizes[m, 1] / 2, ch - sizes[m, 1] / 2))

        cands = mx.array(np.array([[nx, ny]], dtype=np.float32))
        _wl, dens, _c = gpu.score_components(m, cands,
                                             skip_density=False, skip_congestion=True)
        # gpu's `dens` is already 0.5 * top_avg, matching incr.density_cost.
        density_gpu = float(dens[0])

        _ = _set_pos_in_incr(incr, m, nx, ny)
        density_cpu_native = float(incr.density_cost)
        incr.undo_move()

        err = abs(density_gpu - density_cpu_native)
        max_err = max(max_err, err)
        assert err < 5e-4, (
            f"Density mismatch: macro={m} pos=({nx:.4f},{ny:.4f}) "
            f"gpu={density_gpu:.6f} cpu={density_cpu_native:.6f} err={err:.2e}")

    print(f"  Density: {n_trials} trials, max abs err = {max_err:.2e}")


def test_proxy_total_close():
    """End-to-end: gpu's full score (wl + 0.5*density + 0.5*cong-approx) vs
    the CPU exact proxy. The congestion is approximate (frozen-routing): the
    CPU evaluator reroutes the macro's nets at the new position; the GPU
    evaluator only updates the macro's blockage contribution. Expect error
    bounded by 0.5 * (max change in V/H_routing_smooth from rerouting), which
    on ibm01 is ~1.5e-2 absolute. Tolerance ~2e-2."""
    bench, incr, gpu = _build()
    rng = np.random.RandomState(2)
    n_hard = bench.num_hard_macros
    sizes = bench.macro_sizes[:n_hard].numpy()

    max_err = 0.0
    n_trials = 20
    for _ in range(n_trials):
        m = int(rng.randint(0, n_hard))
        cw = float(bench.canvas_width)
        ch = float(bench.canvas_height)
        nx = float(np.clip(rng.uniform(sizes[m, 0] / 2, cw - sizes[m, 0] / 2),
                           sizes[m, 0] / 2, cw - sizes[m, 0] / 2))
        ny = float(np.clip(rng.uniform(sizes[m, 1] / 2, ch - sizes[m, 1] / 2),
                           sizes[m, 1] / 2, ch - sizes[m, 1] / 2))

        cands = mx.array(np.array([[nx, ny]], dtype=np.float32))
        gpu_score = float(gpu.score_candidates(m, cands)[0])

        cpu_cost = _set_pos_in_incr(incr, m, nx, ny)
        incr.undo_move()

        err = abs(gpu_score - cpu_cost)
        max_err = max(max_err, err)
        # Frozen-routing congestion approximation: tolerance 2e-2 absolute.
        assert err < 2e-2, (
            f"Total proxy mismatch: macro={m} pos=({nx:.4f},{ny:.4f}) "
            f"gpu={gpu_score:.6f} cpu={cpu_cost:.6f} err={err:.2e}")

    print(f"  Proxy total: {n_trials} trials, max abs err = {max_err:.2e}")


if __name__ == "__main__":
    test_hpwl_only_matches_exactly()
    test_density_matches_exactly()
    test_proxy_total_close()
    print("ALL OK")
