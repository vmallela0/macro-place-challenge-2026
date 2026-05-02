"""Adaptive Regularization with Cubics (ARC) — Cartis-Gould-Toint 2011.

Solve at each step:
    s* = argmin_s  g·s + 0.5 s^T H s + (M/6) ||s||^3

via Krylov projection: build a Lanczos basis V_k from H starting at -g,
project the cubic problem to the k-dim subspace, solve exactly there.
Return s = V_k y* in the full space.

M is adapted across calls based on actual-vs-predicted decrease ρ:
    ρ ≥ 0.9 → shrink M   (model was conservative; can take larger steps)
    ρ < 0.1 → grow M     (model was too optimistic; cap step size)
    otherwise → keep

This file exposes:
    arc_step(...)         — main entry; produces ONE principled step.
    update_M(M, rho)      — pure adaptation rule.
    build_lanczos_basis(hvp_fn, g, k) — Lanczos for HVP-style operators.
"""
from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from typing import Callable

from _arc_subproblem import solve_cubic_subproblem


# ── M adaptation ────────────────────────────────────────────────────────


def update_M(M: float, rho: float, *,
             eta1: float = 0.1, eta2: float = 0.9,
             gamma1: float = 2.0, gamma2: float = 0.5,
             M_min: float = 1e-8, M_max: float = 1e16) -> float:
    """Cartis-Gould-Toint Algorithm ARC §3.3 update rule.

    rho  = (f(x) - f(x+s)) / (-m(s))   where m(s) = g·s + 0.5 s^T H s + (M/6)||s||^3
    """
    if rho < eta1:
        M_new = max(M_min, gamma1 * M)
    elif rho > eta2:
        M_new = max(M_min, gamma2 * M)
    else:
        M_new = M
    return min(M_max, M_new)


# ── Lanczos basis from a HVP operator ───────────────────────────────────


def build_lanczos_basis(
    hvp_fn: Callable[[NDArray], NDArray],
    g: NDArray,
    k: int,
    *,
    reorth: bool = True,
    tol: float = 1e-12,
) -> tuple[NDArray, NDArray]:
    """k-step Lanczos. Start vector is g/||g||.

    Returns:
        V : (n, k_eff) orthonormal basis.  k_eff ≤ k (early termination
            on tiny residual).
        T : (k_eff, k_eff) symmetric tridiagonal.
    """
    n = g.shape[0]
    g_norm = float(np.linalg.norm(g))
    if g_norm < tol:
        return np.zeros((n, 0)), np.zeros((0, 0))

    V = np.zeros((n, k))
    alpha = np.zeros(k)
    beta = np.zeros(k)

    V[:, 0] = g / g_norm
    w = hvp_fn(V[:, 0])
    alpha[0] = float(V[:, 0] @ w)
    w = w - alpha[0] * V[:, 0]

    k_eff = 1
    for j in range(1, k):
        bj = float(np.linalg.norm(w))
        if bj < tol:
            break
        beta[j] = bj
        V[:, j] = w / bj
        # Optional full reorthogonalisation against prior basis
        if reorth:
            V[:, j] = V[:, j] - V[:, :j] @ (V[:, :j].T @ V[:, j])
            n_v = float(np.linalg.norm(V[:, j]))
            if n_v < tol:
                break
            V[:, j] = V[:, j] / n_v
        w = hvp_fn(V[:, j])
        alpha[j] = float(V[:, j] @ w)
        w = w - alpha[j] * V[:, j] - bj * V[:, j - 1]
        k_eff = j + 1

    V = V[:, :k_eff]
    T = np.zeros((k_eff, k_eff))
    for i in range(k_eff):
        T[i, i] = alpha[i]
        if i + 1 < k_eff:
            T[i, i + 1] = beta[i + 1]
            T[i + 1, i] = beta[i + 1]
    return V, T


# ── Top-level ARC step ──────────────────────────────────────────────────


def arc_step(
    x: NDArray,
    grad_fn: Callable[[NDArray], NDArray],
    hvp_fn: Callable[[NDArray], NDArray],
    *,
    M_init: float = 1.0,
    k_lanczos: int = 50,
    reorth: bool = True,
) -> tuple[NDArray, float, float, dict]:
    """Compute one ARC step at x.

    Args:
        x         : current point (n,)
        grad_fn   : x -> ∇f(x)            (n,)
        hvp_fn    : v -> H(x) v            (n,)   — Hessian-vector product
                                                       AT x (closure captures x)
        M_init    : starting cubic regularisation parameter
        k_lanczos : Krylov subspace dimension

    Returns:
        s         : (n,) step (so x_new = x + s)
        M         : the M used (caller updates via update_M after evaluating ρ)
        lam_min   : smallest eigenvalue of the projected Hessian (informative)
        info      : diagnostics
    """
    g = grad_fn(x)
    g_norm = float(np.linalg.norm(g))
    if g_norm < 1e-14:
        return (np.zeros_like(x), M_init, 0.0,
                {"reason": "zero_gradient", "k_eff": 0,
                 "predicted_decrease": 0.0})

    V, T = build_lanczos_basis(hvp_fn, g, k_lanczos, reorth=reorth)
    if V.shape[1] == 0:
        return (np.zeros_like(x), M_init, 0.0,
                {"reason": "lanczos_failed", "k_eff": 0,
                 "predicted_decrease": 0.0})

    g_k = V.T @ g                    # (k_eff,)
    y, lam, pred_dec, sub_info = solve_cubic_subproblem(T, g_k, M_init)
    s = V @ y
    info = {
        "k_eff": int(V.shape[1]),
        "lambda_min_T": float(sub_info.get("lambda_min_T", 0.0)),
        "lam_lagrange": float(lam),
        "y_norm": float(np.linalg.norm(y)),
        "s_norm": float(np.linalg.norm(s)),
        "predicted_decrease": float(pred_dec),
        "sub_branch": sub_info.get("branch"),
    }
    return s, M_init, info["lambda_min_T"], info
