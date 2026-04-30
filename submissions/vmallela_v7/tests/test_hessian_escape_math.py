"""Mathematical validation of Hessian escape utilities.

Asserts:
1. On a known saddle f(x,y) = x² - y², λ_min = -2 (analytic) and v_min
   points in the y-direction.
2. On a known minimum f(x) = ||x||², λ_min ≥ 0.
3. Top-k eigvecs are orthogonal (H is symmetric).
4. Iterative termination: at a true min, the check returns True (stop).
5. Iterative termination: at a saddle, the check returns False (continue).
"""
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "submissions" / "vmallela_v7"))

from _hessian_escape import (hessian_min_eigvec, hessian_min_eigvecs_topk,
                                iterative_hessian_termination_check)


def test_saddle_function_eigvec():
    """f(x,y) = x² − y² has H = diag(2, −2). λ_min = −2 (the y-direction).

    Verifies that scipy.eigsh with our HVP correctly identifies the
    negative-curvature direction.
    """
    def proxy_call(x):
        # x is shape (1, 2) — treat as a single 2D point
        return x[0, 0] ** 2 - x[0, 1] ** 2

    # At the origin (saddle point)
    x0 = torch.zeros((1, 2), dtype=torch.float64, requires_grad=False)
    lam_min, v_min = hessian_min_eigvec(
        proxy_call, x0, n_lanczos_iters=20, verbose=False)
    # λ_min should be -2 (analytic)
    assert abs(lam_min - (-2.0)) < 1e-3, \
        f"λ_min on saddle x²−y² = {lam_min}, expected -2"
    # v_min should be aligned with y-axis (the negative-curvature dir)
    v = v_min.reshape(1, 2)
    # |v_y| should be ~1.0; |v_x| should be ~0
    cos_with_y = abs(v[0, 1]) / np.linalg.norm(v)
    assert cos_with_y > 0.99, \
        f"v_min not aligned with y-axis: |v_y|/||v|| = {cos_with_y}"
    print(f"  ✓ saddle x²-y²: λ_min = {lam_min:.4f} (≈-2), "
          f"v_min ∥ y-axis (cos {cos_with_y:.4f})")


def test_minimum_function_eigvec():
    """f(x,y) = x² + y² has H = diag(2, 2). λ_min = +2 (positive)."""
    def proxy_call(x):
        return (x ** 2).sum()
    x0 = torch.zeros((1, 2), dtype=torch.float64, requires_grad=False)
    lam_min, v_min = hessian_min_eigvec(
        proxy_call, x0, n_lanczos_iters=20, verbose=False)
    assert abs(lam_min - 2.0) < 1e-3, \
        f"λ_min on min x²+y² = {lam_min}, expected +2"
    print(f"  ✓ minimum x²+y²: λ_min = {lam_min:.4f} (≈+2, positive)")


def test_topk_eigvecs_orthogonal():
    """H = diag(1, 4, 9, 16) → eigvals 1, 4, 9, 16; eigvecs are e_i."""
    n = 4
    diag = torch.tensor([1.0, 4.0, 9.0, 16.0], dtype=torch.float64)
    def proxy_call(x):
        # x is shape (1, 2) but we want a 4-D problem; reshape internally
        # Use n=4 as 2-pin macros: x has shape (2, 2) so 4 elements.
        return 0.5 * ((x.reshape(-1) ** 2) * diag).sum()
    # 2 macros × 2 dims = 4 elements
    x0 = torch.zeros((2, 2), dtype=torch.float64, requires_grad=False)
    eigvals, eigvecs = hessian_min_eigvecs_topk(
        proxy_call, x0, k=3, n_lanczos_iters=30, verbose=False)
    # Should be 1, 4, 9 (smallest 3 eigvals of diag(1,4,9,16))
    eigvals_sorted = np.sort(eigvals)
    expected = np.array([1.0, 4.0, 9.0])
    diff = np.abs(eigvals_sorted - expected).max()
    assert diff < 1e-2, \
        f"top-3 smallest eigvals = {eigvals_sorted}, expected {expected}"
    # Eigvecs should be orthogonal
    G = eigvecs.T @ eigvecs   # 3x3 should be ~identity
    off_diag = (G - np.eye(3)).max()
    assert abs(off_diag) < 1e-6, \
        f"eigvecs not orthogonal; off-diag norm = {off_diag}"
    print(f"  ✓ topk-3 eigvals: {eigvals_sorted} (≈{expected}); "
          f"eigvecs orthogonal (off-diag {abs(off_diag):.2e})")


def test_iter_termination_at_saddle():
    """At a saddle, termination_check returns (False, λ_min<0)."""
    def proxy_call(x):
        return x[0, 0] ** 2 - x[0, 1] ** 2
    x0 = torch.zeros((1, 2), dtype=torch.float64, requires_grad=False)
    should_stop, lam = iterative_hessian_termination_check(
        proxy_call, x0, epsilon=-1e-5)
    assert not should_stop, f"shouldn't stop at saddle (lam={lam})"
    assert lam < 0, f"saddle λ_min should be negative, got {lam}"
    print(f"  ✓ termination at saddle: should_stop=False, λ={lam:.4f}")


def test_iter_termination_at_min():
    """At a minimum, termination_check returns (True, λ_min≥0)."""
    def proxy_call(x):
        return (x ** 2).sum()
    x0 = torch.zeros((1, 2), dtype=torch.float64, requires_grad=False)
    should_stop, lam = iterative_hessian_termination_check(
        proxy_call, x0, epsilon=-1e-5)
    assert should_stop, f"should stop at minimum (lam={lam})"
    assert lam >= 0, f"min λ_min should be ≥0, got {lam}"
    print(f"  ✓ termination at min: should_stop=True, λ={lam:.4f}")


if __name__ == "__main__":
    test_saddle_function_eigvec()
    test_minimum_function_eigvec()
    test_topk_eigvecs_orthogonal()
    test_iter_termination_at_saddle()
    test_iter_termination_at_min()
    print("ALL OK")
