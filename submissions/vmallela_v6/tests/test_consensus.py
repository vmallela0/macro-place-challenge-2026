"""Unit tests for the trimmed-mean consensus.

Doesn't run the full portfolio (too slow). Instead synthesizes N placements
near a "true" target with per-seed Gaussian jitter and verifies:
- trimmed_mean_per_macro recovers the true target within a tight tolerance
- Outliers (one extreme per-seed pathology) get trimmed
- consensus_warm_start composes correctly with the v4 legalize+refine
  pipeline on a real benchmark (ibm01) with synthetic portfolio inputs.
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

from macro_place.benchmark import Benchmark
from _consensus import trimmed_mean_per_macro, consensus_warm_start


def test_trimmed_mean_recovers_target():
    """Synthetic: 16 placements = target + N(0, sigma) per macro,
    plus 2 outliers stuck in extreme positions. Trimmed-mean should
    recover target within ~3 sigma / sqrt(10) (since 6 of 16 are trimmed)."""
    rng = np.random.RandomState(0)
    n_macros = 100
    target = rng.uniform(0, 10, (n_macros, 2))
    sigma = 0.3
    n_workers = 16
    placements = []
    for w in range(n_workers):
        placements.append(target + rng.normal(0, sigma, target.shape))
    # Inject 2 outliers (extreme positions for one specific macro).
    placements[14][7] = np.array([99.0, 99.0])
    placements[15][7] = np.array([-99.0, -99.0])

    consensus = trimmed_mean_per_macro(placements, k_best=16, trim_frac=0.2)
    assert consensus.shape == target.shape
    err = np.abs(consensus - target).max()
    assert err < 3 * sigma, f"trimmed-mean drift > 3 sigma: {err}"
    # The outlier macro (index 7) should be at target ± epsilon, not 99.
    macro7_err = np.abs(consensus[7] - target[7]).max()
    assert macro7_err < 1.0, \
        f"outlier survived trimming: macro 7 consensus={consensus[7]}, target={target[7]}"
    print(f"  trimmed-mean: max err = {err:.4f}, macro 7 (with outliers) "
          f"err = {macro7_err:.4f}")


def test_consensus_on_real_benchmark():
    """Build 8 synthetic placements on ibm01 by perturbing the legalized
    starting point with different RNG seeds; verify consensus_warm_start
    runs end-to-end and returns a valid (overlap-free) placement."""
    bench = Benchmark.load(str(ROOT / "benchmarks" / "processed" /
                               "public" / "ibm01.pt"))
    plc = _v1._load_plc("ibm01")
    init = bench.macro_positions[:bench.num_hard_macros].numpy().copy().astype(np.float64)
    pushed = _v1._push_apart(init, bench, max_iters=300, damping=0.4)
    legal = _v1._legalize(pushed, bench, order_type=0, step_mult=0.05)
    refined = _v1._refine_toward_initial(legal, init, bench)

    # Synthesize 8 perturbed placements + their costs via the incremental eval.
    placements = []
    costs = []
    rng = np.random.RandomState(42)
    n_hard = bench.num_hard_macros
    sizes = bench.macro_sizes[:n_hard].numpy()
    for w in range(8):
        perturbed = refined.copy()
        # Per-worker random small displacement.
        for i in range(n_hard):
            sigma = 0.1 * max(sizes[i, 0], sizes[i, 1])
            perturbed[i, 0] += rng.normal(0, sigma)
            perturbed[i, 1] += rng.normal(0, sigma)
        # Re-push-apart so each is overlap-free.
        perturbed = _v1._push_apart(perturbed, bench, max_iters=200, damping=0.5)
        perturbed = _v1._legalize(perturbed, bench, order_type=w % 5, step_mult=0.05)
        incr = _v1.IncrementalEvaluator(_v1._load_plc("ibm01"), bench)
        incr.sync_positions(perturbed)
        c = float(incr.get_proxy_cost())
        placements.append(perturbed)
        costs.append(c)
    print(f"  synthetic costs: {[f'{c:.4f}' for c in sorted(costs)]}")

    # Run consensus with very short refine to keep test fast.
    cons_pos, cons_cost, source = consensus_warm_start(
        placements, costs, bench, plc,
        k_best=8, trim_frac=0.25,
        refine_max_time=10.0,
        use_gpu_refine=False,
        verbose=True)

    # Validate shape
    assert cons_pos.shape[0] >= n_hard, \
        f"consensus pos shape: {cons_pos.shape}"
    # Validate overlap-free (use the same gap=0.0 official criterion as compute_overlap_metrics)
    pos_hard = cons_pos[:n_hard]
    has_overlap = False
    for i in range(n_hard):
        for j in range(i + 1, n_hard):
            if (abs(pos_hard[i, 0] - pos_hard[j, 0]) < (sizes[i, 0] + sizes[j, 0]) / 2 and
                    abs(pos_hard[i, 1] - pos_hard[j, 1]) < (sizes[i, 1] + sizes[j, 1]) / 2):
                has_overlap = True
                break
        if has_overlap:
            break
    assert not has_overlap, "consensus produced an overlap"

    # Cost is finite and source is one of {graft, trimmed_mean, portfolio_min}
    assert np.isfinite(cons_cost), f"non-finite consensus cost: {cons_cost}"
    assert source in ("graft", "trimmed_mean", "portfolio_min"), source
    print(f"  consensus on ibm01: source={source}, cost={cons_cost:.6f}")


if __name__ == "__main__":
    test_trimmed_mean_recovers_target()
    test_consensus_on_real_benchmark()
    print("ALL OK")
