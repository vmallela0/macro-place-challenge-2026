"""Smoke + invariant tests for Hungarian LNS repair.

Verifies:
- Cost is non-increasing (best_cost <= initial after the phase).
- Returned placement has zero overlaps (validated via the IncrementalEvaluator's
  internal sync — no separate overlap check needed).
- Multiple iterations are exercised (the phase actually accepts moves).
"""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "submissions" / "vmallela_v6"))

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "_v1", str(ROOT / "submissions" / "vmallela" / "placer.py"))
_v1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_v1)
_load_plc = _v1._load_plc
IncrementalEvaluator = _v1.IncrementalEvaluator
_push_apart = _v1._push_apart
_legalize = _v1._legalize
_refine_toward_initial = _v1._refine_toward_initial

from macro_place.benchmark import Benchmark
from _torch_eval import TorchBatchEvaluator
from _hungarian_lns import hungarian_lns_phase


def _legalized_start(name):
    bench = Benchmark.load(str(ROOT / "benchmarks" / "processed" /
                               "public" / f"{name}.pt"))
    plc = _load_plc(name)
    init = bench.macro_positions[:bench.num_hard_macros].numpy().copy().astype(np.float64)
    pushed = _push_apart(init, bench, max_iters=300, damping=0.4)
    legal = _legalize(pushed, bench, order_type=0, step_mult=0.05)
    refined = _refine_toward_initial(legal, init, bench)
    return bench, plc, refined


def _has_overlap(pos, sizes, gap=0.0):
    n = pos.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if (abs(pos[i, 0] - pos[j, 0]) < (sizes[i, 0] + sizes[j, 0]) / 2 + gap and
                    abs(pos[i, 1] - pos[j, 1]) < (sizes[i, 1] + sizes[j, 1]) / 2 + gap):
                return (i, j)
    return None


def test_hungarian_lns_invariants():
    bench, plc, start = _legalized_start("ibm01")
    incr = IncrementalEvaluator(_load_plc("ibm01"), bench)
    incr.sync_positions(start.copy())
    init_cost = float(incr.get_proxy_cost())
    print(f"  ibm01 init cost: {init_cost:.6f}")
    gpu = TorchBatchEvaluator(incr, bench)
    print(f"  device: {gpu.device}")

    pos, cost = hungarian_lns_phase(
        start.copy(), bench, incr, gpu,
        max_time=10.0, n_destroy=8, K=64, verbose=False)
    print(f"  ibm01 Hungarian-LNS 10s: cost={cost:.6f}  Δ={init_cost - cost:+.6f}")

    # Cost monotonically improves (or stays the same).
    assert cost <= init_cost + 1e-9, \
        f"cost regressed: {cost} > {init_cost}"

    # Returned placement is overlap-free vs the OFFICIAL overlap criterion
    # (matching macro_place.objective.compute_overlap_metrics).
    n_hard = bench.num_hard_macros
    sizes = bench.macro_sizes[:n_hard].numpy().astype(np.float64)
    ov = _has_overlap(pos, sizes, gap=0.0)
    assert ov is None, \
        f"Hungarian LNS produced overlap: macros {ov} at " \
        f"({pos[ov[0]]}, {pos[ov[1]]}), sizes ({sizes[ov[0]]}, {sizes[ov[1]]})"


def test_hungarian_lns_makes_progress_on_ibm10():
    """ibm10 has 786 hard macros so the destroy set is a small fraction;
    verify the phase actually accepts moves."""
    bench, plc, start = _legalized_start("ibm10")
    incr = IncrementalEvaluator(_load_plc("ibm10"), bench)
    incr.sync_positions(start.copy())
    init_cost = float(incr.get_proxy_cost())
    print(f"  ibm10 init cost: {init_cost:.6f}")
    gpu = TorchBatchEvaluator(incr, bench)

    pos, cost = hungarian_lns_phase(
        start.copy(), bench, incr, gpu,
        max_time=20.0, n_destroy=8, K=128, verbose=True)
    delta = init_cost - cost
    print(f"  ibm10 Hungarian-LNS 20s: cost={cost:.6f}  Δ={delta:+.6f}")
    # Bar: at least -0.005 on ibm10 in 20s (greedy v4 LNS makes no progress
    # in this short window because of the huge fixed-macro forest).
    assert delta >= 0.005, \
        f"Hungarian LNS made too little progress: Δ={delta:.6f}"


if __name__ == "__main__":
    test_hungarian_lns_invariants()
    test_hungarian_lns_makes_progress_on_ibm10()
    print("ALL OK")
