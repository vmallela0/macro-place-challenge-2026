"""Speed regression test: GPU batch evaluator must beat the CPU
IncrementalEvaluator at B>=256."""
import sys
import time
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
# v6 first so its placer doesn't shadow vmallela's
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


def test_gpu_speed_vs_cpu():
    bench = Benchmark.load(str(ROOT / "benchmarks" / "processed" /
                               "public" / "ibm01.pt"))
    plc = _load_plc("ibm01")
    incr = IncrementalEvaluator(plc, bench)
    init = bench.macro_positions[:bench.num_hard_macros].numpy().copy().astype(np.float64)
    incr.sync_positions(init)
    gpu = MLXBatchEvaluator(incr, bench)

    n_hard = bench.num_hard_macros
    sizes = bench.macro_sizes[:n_hard].numpy()
    cw, ch = float(bench.canvas_width), float(bench.canvas_height)
    rng = np.random.RandomState(7)

    # CPU eval/s
    n_cpu = 200
    t0 = time.time()
    for _ in range(n_cpu):
        m = int(rng.randint(0, n_hard))
        nx = float(rng.uniform(sizes[m, 0] / 2, cw - sizes[m, 0] / 2))
        ny = float(rng.uniform(sizes[m, 1] / 2, ch - sizes[m, 1] / 2))
        incr.move_macro(m, nx, ny)
        incr.undo_move()
    cpu_rate = n_cpu / (time.time() - t0)
    print(f"  CPU: {cpu_rate:.0f} evals/s")

    # GPU eval/s at several B
    rates = {}
    for B in (64, 256, 1024, 4096):
        n_calls = max(5, 30000 // B)
        t0 = time.time()
        for _ in range(n_calls):
            m = int(rng.randint(0, n_hard))
            cands_np = np.column_stack([
                rng.uniform(sizes[m, 0] / 2, cw - sizes[m, 0] / 2, B),
                rng.uniform(sizes[m, 1] / 2, ch - sizes[m, 1] / 2, B),
            ]).astype(np.float32)
            cands = mx.array(cands_np)
            sc = gpu.score_candidates(m, cands)
            mx.eval(sc)
        rates[B] = n_calls * B / (time.time() - t0)
        print(f"  GPU B={B}: {rates[B]:.0f} evals/s")

    # Speed regression: at B=1024, GPU should be at least 20x CPU.
    speedup_1024 = rates[1024] / cpu_rate
    assert speedup_1024 >= 20.0, (
        f"GPU speedup regression: B=1024 speedup={speedup_1024:.1f}× "
        f"(expected >=20×). CPU={cpu_rate:.0f}, GPU(B=1024)={rates[1024]:.0f}")
    print(f"  speedup at B=1024: {speedup_1024:.1f}×")


if __name__ == "__main__":
    test_gpu_speed_vs_cpu()
    print("OK")
