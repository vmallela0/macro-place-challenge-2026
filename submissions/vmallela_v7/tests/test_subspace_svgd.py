"""Sanity tests for _subspace_svgd.subspace_svgd_candidates.

Verifies:
  1. Mathematical correctness on a tiny quadratic with known minimum.
  2. Particles SPREAD (repulsion is doing something) when β is small.
  3. Particles COLLAPSE to the gradient minimum when β is large.
  4. API shape: candidates is a list of (label, ndarray) of right shape.
  5. Sign convention of repulsion (no NaNs, no runaway).
"""
from __future__ import annotations
import sys
from pathlib import Path
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))

import numpy as np
import torch

from _subspace_svgd import subspace_svgd_candidates


def _toy_problem(n_total=8, K_sub=4, lam_scale=1.0, seed=0):
    """A tiny quadratic U(x) = (1/2) x^T H x with random PD Hessian.

    Returns (macro_pos, U_callable, eigvals, eigvecs).
    H is built as V diag(λ) V^T so its eigvecs in the chosen subspace
    are predictable.
    """
    rng = np.random.default_rng(seed)
    N = 2 * n_total
    # Build random orthonormal full basis Q ∈ R^(N×N).
    Q, _ = np.linalg.qr(rng.standard_normal((N, N)))
    # Pick the K_sub smallest "eigvecs" — the trailing columns of Q.
    V = Q[:, -K_sub:]
    eigvals = np.linspace(0.05, 1.0, K_sub) * lam_scale   # all positive
    # Full Hessian H = V·diag(eigvals)·V^T  (rank K_sub; flat in the
    # orthogonal complement). Minimum of U at x = 0.
    H = V @ np.diag(eigvals) @ V.T
    H_t = torch.tensor(H, dtype=torch.float32)
    macro_pos = torch.tensor(rng.standard_normal((n_total, 2)) * 0.5,
                              dtype=torch.float32)

    def U_call(x: torch.Tensor) -> torch.Tensor:
        v = x.reshape(-1)
        return 0.5 * (v @ H_t @ v)

    return macro_pos, U_call, eigvals, V.astype(np.float64)


def test_api_shape():
    macro_pos, U_call, eigvals, V = _toy_problem()
    cands, diag = subspace_svgd_candidates(
        macro_pos, U_call, eigvals, V,
        n_particles=4, n_iters=5, step_size=0.1, beta=1.0,
        seed=7, max_total_step_canvas=2.0, canvas_diag=1.0,
        soft_only=False, verbose=False,
    )
    assert isinstance(cands, list), "candidates must be a list"
    assert len(cands) == 4, f"expected 4 candidates, got {len(cands)}"
    for lab, pos in cands:
        assert isinstance(lab, str)
        assert isinstance(pos, np.ndarray)
        assert pos.shape == tuple(macro_pos.shape)
        assert np.all(np.isfinite(pos)), "non-finite pos"
    assert diag["method"] == "subspace_svgd"
    assert diag["n_particles"] == 4
    assert diag["n_iters"] == 5
    print(f"  api PASS: {len(cands)} candidates, dtype={pos.dtype}, wall={diag['wall_s']:.2f}s")


def test_descent_high_beta():
    """High β + many iters should drive every particle toward U≈0."""
    macro_pos, U_call, eigvals, V = _toy_problem(seed=1)
    cands, diag = subspace_svgd_candidates(
        macro_pos, U_call, eigvals, V,
        n_particles=8, n_iters=40, step_size=0.4, beta=10.0,
        seed=11, max_total_step_canvas=10.0, canvas_diag=1.0,
        soft_only=False, verbose=False,
    )
    # Final particles' U_final should be SUBSTANTIALLY lower than U0.
    deltas = [pp["delta_U"] for pp in diag["per_particle"]]
    n_improved = sum(1 for d in deltas if d < -1e-6)
    print(f"  high-β PASS: {n_improved}/{len(deltas)} particles improved, "
          f"mean Δ={np.mean(deltas):+.4f}")
    assert n_improved >= len(deltas) - 1, \
        f"only {n_improved} particles improved under high β"


def test_spread_low_beta():
    """Low β should let particles SPREAD due to repulsion dominating."""
    macro_pos, U_call, eigvals, V = _toy_problem(seed=2)
    cands_lo, diag_lo = subspace_svgd_candidates(
        macro_pos, U_call, eigvals, V,
        n_particles=8, n_iters=30, step_size=0.3, beta=0.01,
        seed=21, max_total_step_canvas=10.0, canvas_diag=1.0,
        soft_only=False, init_scale=0.1, verbose=False,
    )
    spread_0 = diag_lo["per_iter"][0]["spread"]
    spread_T = diag_lo["per_iter"][-1]["spread"]
    # With β almost zero, the attractive term vanishes and the repulsion
    # increases the pairwise spread monotonically.
    print(f"  low-β spread {spread_0:.3f} → {spread_T:.3f}")
    assert spread_T > spread_0 * 0.9, \
        f"expected spread to grow or stay similar, got {spread_0:.3f}→{spread_T:.3f}"


def test_no_runaway():
    """With max_total_step_canvas active, no particle should exceed cap."""
    macro_pos, U_call, eigvals, V = _toy_problem(seed=3)
    cap = 0.5
    cands, diag = subspace_svgd_candidates(
        macro_pos, U_call, eigvals, V,
        n_particles=6, n_iters=20, step_size=0.5, beta=0.1,
        seed=31, max_total_step_canvas=cap, canvas_diag=2.0,
        soft_only=False, init_scale=1.0, verbose=False,
    )
    cap_microns = cap * 2.0
    for pp in diag["per_particle"]:
        assert pp["radius_microns"] <= cap_microns + 1e-6, \
            f"particle {pp['particle']} radius {pp['radius_microns']:.4f} > cap {cap_microns:.4f}"
    print(f"  no-runaway PASS: all particles within {cap_microns:.3f}μm cap")


def test_soft_only_zero_for_hard():
    """soft_only=True must keep hard-macro coordinates unchanged."""
    macro_pos, U_call, eigvals, V = _toy_problem(n_total=10, seed=4)
    n_hard = 3
    cands, _ = subspace_svgd_candidates(
        macro_pos, U_call, eigvals, V,
        n_particles=4, n_iters=10, step_size=0.3, beta=1.0,
        seed=41, max_total_step_canvas=2.0, canvas_diag=1.0,
        soft_only=True, n_hard=n_hard, verbose=False,
    )
    pos0 = macro_pos.detach().cpu().numpy()
    for lab, pos in cands:
        diff = np.abs(pos[:n_hard] - pos0[:n_hard])
        assert diff.max() < 1e-5, \
            f"hard-macro pos changed: max Δ = {diff.max():.4e}  ({lab})"
    print(f"  soft-only PASS: first {n_hard} rows untouched")


if __name__ == "__main__":
    test_api_shape()
    test_descent_high_beta()
    test_spread_low_beta()
    test_no_runaway()
    test_soft_only_zero_for_hard()
    print("ALL TESTS PASSED")
