"""Krylov-subspace cubic subproblem solver for ARC.

Cartis, Gould, Toint (2011), "Adaptive cubic regularisation methods for
unconstrained optimization. Part I: motivation, convergence and numerical
results", §6.

Given a Lanczos basis V_k (orthonormal columns) of dimension k built by
applying the Hessian H to a starting vector (typically the gradient g),
we have:

    T_k = V_k^T H V_k        (k×k tridiagonal)
    g_k = V_k^T g            (length k)
    s   = V_k y              for some y ∈ R^k

The full-space cubic model

    m(s) = g·s + 1/2 s^T H s + (M/6) ||s||^3

reduces in the subspace (using V_k^T V_k = I and ||s|| = ||y||) to:

    m(y) = g_k·y + 1/2 y^T T_k y + (M/6) ||y||^3

We solve the small dense problem `argmin_y m(y)` exactly. By the theory
of cubic regularisation, the optimal y satisfies:

    (T_k + λ I) y = -g_k        with    λ = M/2 · ||y||         (1)

So we have a 1D nonlinear equation in λ:

    define h(λ) = ||(T_k + λ I)^{-1} g_k||  -  2λ/M
    find λ such that h(λ) = 0,  λ > max(0, -λ_min(T_k))

When T_k has a negative eigenvalue λ_min < 0, λ must be > -λ_min for
(T_k + λI) to be positive definite. We solve via secant on the safe
interval (max(0, -λ_min) + ε, large_upper).

When T_k is positive definite (all positive eigvals), λ = 0 is feasible
if ||T_k^{-1} g_k|| · M / 2 ≤ 0, i.e. ARC equals damped Newton when M
is small.

Returns: (y, lam, predicted_decrease, info_dict).
"""
from __future__ import annotations
import numpy as np
from numpy.typing import NDArray


def solve_cubic_subproblem(
    T: NDArray,                 # (k, k) symmetric (Lanczos tridiag)
    g_k: NDArray,               # (k,) projected gradient
    M: float,                   # cubic regularisation parameter
    *,
    tol: float = 1e-10,
    max_secant_iters: int = 100,
) -> tuple[NDArray, float, float, dict]:
    """Solve y* = argmin_y g_k·y + 0.5 y^T T y + (M/6) ||y||^3.

    Returns:
        y      : (k,) optimal subspace coefficient
        lam    : Lagrange multiplier (= M/2 · ||y||)
        m_dec  : predicted decrease  -m(y) = -(g·y + 0.5 y^T T y + M/6 ||y||^3)
                 (negative of the model evaluated at y; ≥ 0 for any descent step)
        info   : diagnostics — {"lambda_min_T": ..., "secant_iters": ..., "converged": ...}
    """
    k = T.shape[0]
    if k == 0 or np.linalg.norm(g_k) < tol:
        return (np.zeros(k), 0.0, 0.0,
                {"lambda_min_T": 0.0, "secant_iters": 0, "converged": True})

    # Eigendecompose T_k once. We'll exploit T_k = Q Λ Q^T to evaluate
    # ||(T_k + λI)^{-1} g_k|| cheaply for any λ.
    eigvals, Q = np.linalg.eigh(T)   # ascending eigvals, orthonormal Q
    lam_min = float(eigvals[0])
    # Project g_k into the eigenbasis: c = Q^T g_k.
    c = Q.T @ g_k                    # (k,)

    def y_of_lam(lam: float) -> NDArray:
        """y(λ) = -(T + λI)^{-1} g_k  =  -Q diag(1/(eigval + λ)) Q^T g_k"""
        denom = eigvals + lam
        return -Q @ (c / denom)

    def y_norm_of_lam(lam: float) -> float:
        denom = eigvals + lam
        return float(np.sqrt((c / denom) ** 2 @ np.ones_like(c)))
        # equivalently sqrt(sum(c_i^2 / (eigval_i + lam)^2))

    def residual(lam: float) -> float:
        """Want lam = M/2 * ||y(lam)||  →  residual = lam - M/2 * ||y(lam)||"""
        return lam - 0.5 * M * y_norm_of_lam(lam)

    if M <= tol:
        # M → 0: cubic regularisation degenerates to Newton.
        # If T is PD: y = -T^{-1} g_k = solve_lin(T, -g_k); λ = 0.
        # If T is indefinite: still Newton-step the subspace problem
        # (caller's strict-improvement gate filters non-descent).
        if lam_min > tol:
            y = y_of_lam(0.0)
            lam = 0.0
            m_val = float(g_k @ y + 0.5 * y @ T @ y)
            return (y, lam, -m_val,
                    {"lambda_min_T": lam_min,
                     "secant_iters": 0,
                     "converged": True,
                     "branch": "newton_pd"})
        # Indefinite + tiny M: take a step along the most-negative
        # eigenvector to first order; the cubic term is what's supposed
        # to bound the step magnitude in this case, so a tiny M means
        # arbitrary length. Pick a unit step along -g (subspace).
        y = -g_k / max(np.linalg.norm(g_k), tol)
        m_val = float(g_k @ y + 0.5 * y @ T @ y)
        return (y, 0.0, -m_val,
                {"lambda_min_T": lam_min,
                 "secant_iters": 0,
                 "converged": False,
                 "branch": "indef_tiny_M"})

    # Safe lower bound for λ: must be > -lam_min so T + λI is PD.
    lam_lo = max(0.0, -lam_min) + 1e-12
    # Upper bound: when λ is huge, ||y|| ≈ ||g_k||/λ, so residual ≈ λ.
    # Find a λ_hi where residual > 0 to bracket the root.
    lam_hi = max(2.0 * abs(lam_min) + 1.0, np.linalg.norm(g_k) * M / 2.0)
    while residual(lam_hi) < 0 and lam_hi < 1e16:
        lam_hi *= 2.0

    # If even at lam_lo the residual is positive, the optimum is at the
    # boundary (lam = lam_lo). This happens when M is so big that the
    # cubic term forces ||y|| → 0.
    if residual(lam_lo) > 0:
        lam_star = lam_lo
        y_star = y_of_lam(lam_star)
    else:
        # Bisection (more robust than secant near boundary).
        a, b = lam_lo, lam_hi
        fa, fb = residual(a), residual(b)
        if fa * fb > 0:
            # No sign change — fallback to lam_lo.
            lam_star = lam_lo
            y_star = y_of_lam(lam_star)
            n_iters = 0
        else:
            n_iters = 0
            while (b - a) > tol * max(1.0, b) and n_iters < max_secant_iters:
                mid = 0.5 * (a + b)
                fmid = residual(mid)
                if fa * fmid <= 0:
                    b, fb = mid, fmid
                else:
                    a, fa = mid, fmid
                n_iters += 1
            lam_star = 0.5 * (a + b)
            y_star = y_of_lam(lam_star)

    y_norm = float(np.linalg.norm(y_star))
    m_val = float(g_k @ y_star + 0.5 * y_star @ T @ y_star
                  + (M / 6.0) * y_norm ** 3)
    info = {
        "lambda_min_T": lam_min,
        "lam_star": float(lam_star),
        "y_norm": y_norm,
        "secant_iters": int(locals().get("n_iters", 0)),
        "converged": True,
        "branch": ("boundary" if residual(lam_lo) > 0
                   else ("newton_pd" if M <= tol and lam_min > tol
                         else "interior")),
    }
    return y_star, float(lam_star), -m_val, info
