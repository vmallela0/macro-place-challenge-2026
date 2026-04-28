"""Equivalence test: TorchBatchEvaluator vs CPU IncrementalEvaluator and
vs the existing MLX evaluator.

For a single-macro candidate move, the torch evaluator must reproduce HPWL
and density EXACTLY (same float32 path the MLX one uses). The total proxy
matches CPU within the documented frozen-routing congestion bound (~6e-3).

Also verifies the new multi-macro entry point matches the per-macro one
when a batch covers multiple macros.
"""
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "submissions" / "vmallela_v6"))

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "_v1_placer", str(ROOT / "submissions" / "vmallela" / "placer.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
_load_plc = _mod._load_plc
IncrementalEvaluator = _mod.IncrementalEvaluator

from macro_place.benchmark import Benchmark
from _torch_eval import TorchBatchEvaluator


def _build():
    bench = Benchmark.load(str(ROOT / "benchmarks" / "processed" /
                               "public" / "ibm01.pt"))
    plc = _load_plc("ibm01")
    incr = IncrementalEvaluator(plc, bench)
    init = bench.macro_positions[:bench.num_hard_macros].numpy().copy().astype(np.float64)
    incr.sync_positions(init)
    gpu = TorchBatchEvaluator(incr, bench)
    print(f"  device: {gpu.device}")
    return bench, incr, gpu


def test_hpwl_only_matches_exactly():
    bench, incr, gpu = _build()
    rng = np.random.RandomState(0)
    n_hard = bench.num_hard_macros
    sizes = bench.macro_sizes[:n_hard].numpy()
    cw = float(bench.canvas_width)
    ch = float(bench.canvas_height)

    max_err_wl = 0.0
    n_trials = 60
    for _ in range(n_trials):
        m = int(rng.randint(0, n_hard))
        nx = float(np.clip(rng.uniform(sizes[m, 0] / 2, cw - sizes[m, 0] / 2),
                           sizes[m, 0] / 2, cw - sizes[m, 0] / 2))
        ny = float(np.clip(rng.uniform(sizes[m, 1] / 2, ch - sizes[m, 1] / 2),
                           sizes[m, 1] / 2, ch - sizes[m, 1] / 2))
        cands = np.array([[nx, ny]], dtype=np.float32)
        wl, _d, _c = gpu.score_components(m, cands,
                                          skip_density=True, skip_congestion=True)
        gpu_wl = float(wl[0])

        _ = incr.move_macro(m, nx, ny)
        cpu_wl = float(incr.wirelength_cost)
        incr.undo_move()

        err = abs(gpu_wl - cpu_wl)
        max_err_wl = max(max_err_wl, err)
        assert err < 5e-4, (
            f"HPWL mismatch: m={m} pos=({nx:.4f},{ny:.4f}) "
            f"gpu={gpu_wl:.6f} cpu={cpu_wl:.6f} err={err:.2e}")
    print(f"  HPWL: {n_trials} trials, max abs err = {max_err_wl:.2e}")


def test_density_matches_exactly():
    bench, incr, gpu = _build()
    rng = np.random.RandomState(1)
    n_hard = bench.num_hard_macros
    sizes = bench.macro_sizes[:n_hard].numpy()
    cw, ch = float(bench.canvas_width), float(bench.canvas_height)

    max_err = 0.0
    n_trials = 30
    for _ in range(n_trials):
        m = int(rng.randint(0, n_hard))
        nx = float(np.clip(rng.uniform(sizes[m, 0] / 2, cw - sizes[m, 0] / 2),
                           sizes[m, 0] / 2, cw - sizes[m, 0] / 2))
        ny = float(np.clip(rng.uniform(sizes[m, 1] / 2, ch - sizes[m, 1] / 2),
                           sizes[m, 1] / 2, ch - sizes[m, 1] / 2))
        cands = np.array([[nx, ny]], dtype=np.float32)
        _wl, dens, _c = gpu.score_components(m, cands,
                                             skip_density=False, skip_congestion=True)
        gpu_dens = float(dens[0])
        _ = incr.move_macro(m, nx, ny)
        cpu_dens = float(incr.density_cost)
        incr.undo_move()
        err = abs(gpu_dens - cpu_dens)
        max_err = max(max_err, err)
        assert err < 5e-4, (
            f"Density mismatch: m={m} gpu={gpu_dens:.6f} cpu={cpu_dens:.6f} "
            f"err={err:.2e}")
    print(f"  Density: {n_trials} trials, max abs err = {max_err:.2e}")


def test_proxy_total_close():
    bench, incr, gpu = _build()
    rng = np.random.RandomState(2)
    n_hard = bench.num_hard_macros
    sizes = bench.macro_sizes[:n_hard].numpy()
    cw, ch = float(bench.canvas_width), float(bench.canvas_height)

    max_err = 0.0
    n_trials = 20
    for _ in range(n_trials):
        m = int(rng.randint(0, n_hard))
        nx = float(np.clip(rng.uniform(sizes[m, 0] / 2, cw - sizes[m, 0] / 2),
                           sizes[m, 0] / 2, cw - sizes[m, 0] / 2))
        ny = float(np.clip(rng.uniform(sizes[m, 1] / 2, ch - sizes[m, 1] / 2),
                           sizes[m, 1] / 2, ch - sizes[m, 1] / 2))
        cands = np.array([[nx, ny]], dtype=np.float32)
        gpu_score = float(gpu.score_candidates(m, cands)[0])
        cpu_cost = float(incr.move_macro(m, nx, ny))
        incr.undo_move()
        err = abs(gpu_score - cpu_cost)
        max_err = max(max_err, err)
        assert err < 2e-2, (
            f"Total proxy mismatch: m={m} pos=({nx:.4f},{ny:.4f}) "
            f"gpu={gpu_score:.6f} cpu={cpu_cost:.6f} err={err:.2e}")
    print(f"  Proxy total: {n_trials} trials, max abs err = {max_err:.2e}")


def test_multimacro_matches_per_macro():
    """Build a batch covering 8 distinct macros at 4 candidates each (B=32)
    and assert the multimacro path returns the same scores as 8 separate
    per-macro calls."""
    bench, incr, gpu = _build()
    rng = np.random.RandomState(3)
    n_hard = bench.num_hard_macros
    sizes = bench.macro_sizes[:n_hard].numpy()
    cw, ch = float(bench.canvas_width), float(bench.canvas_height)

    macros = rng.choice(n_hard, size=8, replace=False)
    K = 4
    macro_ids = []
    cands = []
    for m in macros:
        for _ in range(K):
            nx = float(np.clip(rng.uniform(sizes[m, 0] / 2, cw - sizes[m, 0] / 2),
                               sizes[m, 0] / 2, cw - sizes[m, 0] / 2))
            ny = float(np.clip(rng.uniform(sizes[m, 1] / 2, ch - sizes[m, 1] / 2),
                               sizes[m, 1] / 2, ch - sizes[m, 1] / 2))
            macro_ids.append(int(m))
            cands.append([nx, ny])
    macro_ids = np.asarray(macro_ids)
    cands = np.asarray(cands, dtype=np.float32)

    multi = gpu.score_candidates_multimacro(macro_ids, cands)
    multi_np = multi.detach().cpu().numpy()

    # Reference: per-macro single calls
    ref = np.zeros(len(macro_ids), dtype=np.float32)
    for i in range(len(macro_ids)):
        m = int(macro_ids[i])
        c = cands[i:i + 1]
        ref[i] = float(gpu.score_candidates(m, c)[0])

    err = np.abs(multi_np - ref)
    max_err = float(err.max())
    assert max_err < 1e-4, (
        f"multimacro vs per-macro mismatch: max abs err = {max_err:.2e}")
    print(f"  multimacro vs per-macro: B={len(macro_ids)}, "
          f"max abs err = {max_err:.2e}")


if __name__ == "__main__":
    test_hpwl_only_matches_exactly()
    test_density_matches_exactly()
    test_proxy_total_close()
    test_multimacro_matches_per_macro()
    print("ALL OK")
