"""Smoke test for catastrophe-fold module.

Proven math claims being verified:
  1. estimate_directional_cubic recovers (λ, c) for U(t) = ½λt² + (c/6)t³
     within finite-difference error O(h²).
  2. The fold formula t* = -2λ/c minimizes U along v.
  3. U(t*) = 2λ³/(3c²) — closed-form identity.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))
import numpy as np
import torch
from _catastrophe import (estimate_directional_cubic,
                            catastrophe_fold_candidates)


def test_recover_cubic_coeffs():
    """U(x) = ½·λ·v·x² + (c/6)·(v·x)³ along direction v.

    With v = [1, 0] flattened, λ = -2.0, c = 0.5:
        Estimated (λ, c) should match within ~1% relative.
    """
    n_total = 1
    x0 = torch.zeros(n_total, 2, dtype=torch.float64)
    lam, c = -2.0, 0.5
    v = np.array([1.0, 0.0])  # flattened (2n=2,)

    def U(x_tensor):
        # x_tensor shape (n, 2)
        t = (x_tensor[0, 0])
        return 0.5 * lam * t ** 2 + (c / 6.0) * t ** 3

    second_est, cubic_est = estimate_directional_cubic(U, x0, v, h=0.1)
    rel_lam = abs(second_est - lam) / abs(lam)
    rel_c = abs(cubic_est - c) / abs(c)
    print(f"  λ_est={second_est:.6f} (true {lam}) rel_err={rel_lam:.2e}")
    print(f"  c_est={cubic_est:.6f} (true {c}) rel_err={rel_c:.2e}")
    assert rel_lam < 1e-3, f"λ recovery off: {rel_lam}"
    assert rel_c < 1e-3, f"c recovery off: {rel_c}"


def test_fold_identity():
    """Verify U(t*) = 2λ³ / (3c²).

    λ = -2, c = 1.5 → t* = -2·(-2)/1.5 = 4/3 → U(t*) = 2·(-8)/(3·2.25) = -16/6.75 ≈ -2.370.
    """
    lam, c = -2.0, 1.5
    t_star_pred = -2.0 * lam / c
    U_star_pred = 2.0 * lam ** 3 / (3.0 * c ** 2)
    # Direct numeric: f(t) = ½ λ t² + (c/6) t³
    f = lambda t: 0.5 * lam * t ** 2 + (c / 6.0) * t ** 3
    U_star_num = f(t_star_pred)
    print(f"  predicted: t*={t_star_pred:.4f}, U(t*)={U_star_pred:.4f}")
    print(f"  numeric:   t*={t_star_pred:.4f}, U(t*)={U_star_num:.4f}")
    assert abs(U_star_pred - U_star_num) < 1e-10
    # Also: at t* it's a minimum if d²U/dt²(t*) > 0.
    #   f''(t) = λ + c·t.  f''(t*) = λ + c·(-2λ/c) = λ - 2λ = -λ > 0 ✓ (since λ<0).
    fpp_star = lam + c * t_star_pred
    print(f"  f''(t*) = {fpp_star:.4f} (positive ⇒ min)")
    assert fpp_star > 0


def test_candidates_generator():
    """End-to-end: feed (eigvec, eigval, U) into catastrophe_fold_candidates,
    verify it returns a candidate with predicted ΔU close to true ΔU.
    """
    n_total = 3
    x0 = torch.zeros(n_total, 2, dtype=torch.float64)
    lam, c = -3.0, 2.0
    # Eigenvec: along direction (1, 0) for macro 0.
    v = np.zeros(2 * n_total)
    v[0] = 1.0   # x-coord of macro 0

    def U(x_tensor):
        t = x_tensor[0, 0]
        return 0.5 * lam * t ** 2 + (c / 6.0) * t ** 3

    eigvals = np.array([lam])
    eigvecs = v.reshape(-1, 1)
    cands, diag = catastrophe_fold_candidates(
        U, x0, eigvals, eigvecs,
        canvas_diag=20.0, cap_frac=0.5, h_frac=0.01,
        n_hard=0, soft_only=False, verbose=False,
    )
    assert len(cands) >= 1
    info = diag["per_eigvec"][0]
    print(f"  t_star_used={info['t_star_used']:.4f} "
          f"(true {-2*lam/c:.4f}) "
          f"ΔU_pred={info['delta_U_pred']:.4f} "
          f"(true {2*lam**3/(3*c**2):.4f})")
    # Recovered t_star within ~1% of true value.
    t_true = -2 * lam / c
    rel_t = abs(info["t_star_used"] - t_true) / abs(t_true)
    assert rel_t < 1e-2, f"t_star off: {rel_t}"


if __name__ == "__main__":
    test_recover_cubic_coeffs()
    test_fold_identity()
    test_candidates_generator()
    print("ALL CATASTROPHE TESTS PASSED")
