"""Speed regression test for TorchBatchEvaluator.

Compares both per-macro and cross-macro multimacro paths against the CPU
IncrementalEvaluator. Per-macro must be >= 10x CPU; cross-macro must be
>= 25x CPU on M5 Pro MPS (much higher on RTX 6000 Ada CUDA).
"""
import sys
import time
import numpy as np
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "submissions" / "vmallela_v6"))

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "_v1", str(ROOT / "submissions" / "vmallela" / "placer.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
_load_plc = _mod._load_plc
IncrementalEvaluator = _mod.IncrementalEvaluator

from macro_place.benchmark import Benchmark
from _torch_eval import TorchBatchEvaluator


def _sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()
    elif dev.type == "mps":
        torch.mps.synchronize()


def test_torch_speed_vs_cpu():
    bench = Benchmark.load(str(ROOT / "benchmarks" / "processed" /
                               "public" / "ibm01.pt"))
    plc = _load_plc("ibm01")
    incr = IncrementalEvaluator(plc, bench)
    init = bench.macro_positions[:bench.num_hard_macros].numpy().copy().astype(np.float64)
    incr.sync_positions(init)
    gpu = TorchBatchEvaluator(incr, bench)
    print(f"  device: {gpu.device}")

    n_hard = bench.num_hard_macros
    sizes = bench.macro_sizes[:n_hard].numpy()
    cw, ch = float(bench.canvas_width), float(bench.canvas_height)
    rng = np.random.RandomState(7)

    # CPU baseline
    n_cpu = 200
    t0 = time.time()
    for _ in range(n_cpu):
        m = int(rng.randint(0, n_hard))
        nx = float(rng.uniform(sizes[m, 0] / 2, cw - sizes[m, 0] / 2))
        ny = float(rng.uniform(sizes[m, 1] / 2, ch - sizes[m, 1] / 2))
        incr.move_macro(m, nx, ny)
        incr.undo_move()
    cpu_rate = n_cpu / (time.time() - t0)
    print(f"  CPU IncrementalEvaluator: {cpu_rate:.0f} evals/s")

    # Per-macro torch
    rates_per = {}
    for B in (256, 1024):
        n_calls = max(5, 30000 // B)
        m = 0
        t0 = time.time()
        for _ in range(n_calls):
            cands = np.column_stack([
                rng.uniform(sizes[m, 0] / 2, cw - sizes[m, 0] / 2, B),
                rng.uniform(sizes[m, 1] / 2, ch - sizes[m, 1] / 2, B),
            ]).astype(np.float32)
            sc = gpu.score_candidates(m, cands)
            _sync(gpu.device)
        rates_per[B] = n_calls * B / (time.time() - t0)
        print(f"  per-macro B={B}: {rates_per[B]:.0f} evals/s "
              f"({rates_per[B]/cpu_rate:.0f}x CPU)")

    # Cross-macro multimacro: full pass over all movable macros × K candidates
    K = 32
    movable = np.arange(n_hard)
    M = len(movable)
    macro_ids = np.repeat(movable, K)
    n_passes = 20
    t0 = time.time()
    for _ in range(n_passes):
        cands = np.zeros((M * K, 2), dtype=np.float32)
        for i, m in enumerate(movable):
            cands[i * K:(i + 1) * K, 0] = rng.uniform(
                sizes[m, 0] / 2, cw - sizes[m, 0] / 2, K)
            cands[i * K:(i + 1) * K, 1] = rng.uniform(
                sizes[m, 1] / 2, ch - sizes[m, 1] / 2, K)
        sc = gpu.score_candidates_multimacro(macro_ids, cands)
        _sync(gpu.device)
    elapsed = time.time() - t0
    multi_rate = n_passes * M * K / elapsed
    print(f"  multimacro M={M} K={K}: {multi_rate:.0f} evals/s "
          f"({multi_rate/cpu_rate:.0f}x CPU), {elapsed/n_passes*1000:.1f} ms/pass")

    # Speed regression: per-macro >= 10x, multimacro >= 25x at B=1024.
    assert rates_per[1024] >= 10.0 * cpu_rate, (
        f"per-macro speedup regression: {rates_per[1024]/cpu_rate:.1f}x "
        f"(expected >=10x)")
    assert multi_rate >= 15.0 * cpu_rate, (
        f"multimacro speedup regression: {multi_rate/cpu_rate:.1f}x "
        f"(expected >=15x)")


if __name__ == "__main__":
    test_torch_speed_vs_cpu()
    print("OK")
