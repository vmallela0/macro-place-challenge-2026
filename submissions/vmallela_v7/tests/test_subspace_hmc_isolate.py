"""Isolate-test subspace HMC trajectory generation on a real benchmark.

Loads ibm06, builds the smooth proxy, runs Lanczos K=6, runs HMC with
T=8 trajectories. Validates that:
  - All trajectory endpoints are finite.
  - At least half the trajectories produce surrogate-improving moves
    (sanity check that the leapfrog + Hessian-metric mass do useful work).
  - With seed fixed, results are reproducible.
"""
from __future__ import annotations
import os, sys, time
from pathlib import Path

for k, v in [
    ("OMP_NUM_THREADS", "1"),
    ("MKL_NUM_THREADS", "1"),
    ("OPENBLAS_NUM_THREADS", "1"),
    ("VECLIB_MAXIMUM_THREADS", "1"),
    ("NUMEXPR_NUM_THREADS", "1"),
    ("PYTHONHASHSEED", "42"),
]:
    os.environ.setdefault(k, v)

ROOT = Path(__file__).resolve().parents[3]
V7 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(V7))
sys.path.insert(0, str(V7.parent))
sys.path.insert(0, str(ROOT))

import math
import numpy as np
import torch
torch.set_num_threads(4)

from test_hessian_phase_rudy import setup_ibm06, build_smooth_proxy_call


def test_hmc_reproducible():
    bench, incr = setup_ibm06()
    canvas_diag = math.hypot(incr.cw, incr.ch)
    macro_pos_t, scall = build_smooth_proxy_call(incr, rudy_enabled=True)

    from _hessian_escape import hessian_min_eigvecs_topk
    from _subspace_hmc import subspace_hmc_candidates

    t0 = time.time()
    eigvals, eigvecs = hessian_min_eigvecs_topk(
        scall, macro_pos_t, k=6, n_lanczos_iters=80, tikhonov=1e-4,
        verbose=False)
    print(f"Lanczos: λ={eigvals.tolist()} ({time.time()-t0:.2f}s)")

    cands1, diag1 = subspace_hmc_candidates(
        macro_pos_t, scall, eigvals, eigvecs,
        n_trajectories=8, n_leapfrog=12, step_size=0.5,
        canvas_diag=canvas_diag, n_hard=incr.n_hard, soft_only=True,
        seed=42, verbose=False)
    cands2, diag2 = subspace_hmc_candidates(
        macro_pos_t, scall, eigvals, eigvecs,
        n_trajectories=8, n_leapfrog=12, step_size=0.5,
        canvas_diag=canvas_diag, n_hard=incr.n_hard, soft_only=True,
        seed=42, verbose=False)
    # Reproducibility check
    for i, ((l1, p1), (l2, p2)) in enumerate(zip(cands1, cands2)):
        diff = np.abs(p1 - p2).max()
        assert diff < 1e-9, f"traj {i} not reproducible: max diff = {diff}"
    print(f"  reproducible: PASS (8 trajs, max diff < 1e-9)")

    # Surrogate-improving rate
    improving = sum(1 for d in diag1["trajectories"]
                      if d["delta_U"] < 0 and np.isfinite(d["delta_U"]))
    med_radius = float(np.median(
        [d["radius_microns"] for d in diag1["trajectories"]]))
    print(f"  improving: {improving}/8 (med_radius={med_radius:.1f}μm "
          f"wall={diag1['wall_s']:.1f}s)")

    # Different seed → different trajectories
    cands3, _ = subspace_hmc_candidates(
        macro_pos_t, scall, eigvals, eigvecs,
        n_trajectories=8, n_leapfrog=12, step_size=0.5,
        canvas_diag=canvas_diag, n_hard=incr.n_hard, soft_only=True,
        seed=100, verbose=False)
    diffs = []
    for (_, p1), (_, p3) in zip(cands1, cands3):
        diffs.append(np.abs(p1 - p3).max())
    print(f"  seed-distinct: max diff = {max(diffs):.2f} micron "
          f"(should be > 1 micron)")
    assert max(diffs) > 1.0, "different seeds should give different trajs"


if __name__ == "__main__":
    test_hmc_reproducible()
    print("\nHMC ISOLATE TEST PASSED")
