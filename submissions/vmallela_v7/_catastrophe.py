"""Catastrophe-theory cubic unfolding escape — zeus B10.

Mathematical setup
==================
Let U(x) : ℝᴺ → ℝ be the smooth proxy. At a saddle x_0 with Hessian
H = ∇²U(x_0), let v be a unit eigenvector of H with eigenvalue λ < 0.
The 1-d slice along v is
    U(x_0 + t·v) = U(x_0) + ½·λ·t² + (c/6)·t³ + O(t⁴)
where the cubic coefficient is
    c = (vᵀ ⊗ vᵀ ⊗ vᵀ) · ∇³U(x_0)
i.e., the directional 3rd derivative.

This is the FOLD CATASTROPHE (A_2 in Thom's classification): the
generic singularity that controls the local geometry of a slowly-
unfolding saddle. The fold predicts:

    • Two critical points of U along v: t = 0 (saddle) and t* = -2λ/c.
    • If c ≠ 0, the second critical point t* exists in closed form.
    • U(t*) - U(0) = 2λ³ / (3c²).
      Since λ < 0 (saddle direction), λ³ < 0, so U(t*) < U(0).
      The fold formula gives an EXACT minimum-along-v step.

Proof of the U(t*) identity
---------------------------
Let f(t) = ½ λ t² + (c/6) t³. Then
    f'(t) = λ t + (c/2) t²  =  t · (λ + (c/2) t).
Critical points: t = 0 or t = -2λ/c.
At t* = -2λ/c:
    f(t*) = ½ λ · (2λ/c)² + (c/6) · (-2λ/c)³
          = (2 λ³ / c²) - (4 λ³ / (3 c²))
          = (6 λ³ - 4 λ³) / (3 c²)
          = 2 λ³ / (3 c²).

For our use, λ < 0 → λ³ < 0 → f(t*) < 0. ✓

Estimating c
============
Given access to U(x) but not its 3rd derivatives, we estimate c by
central finite differences along v at four points {±h, ±2h}:
    f(2h) - 2·f(h) + 2·f(-h) - f(-2h)  =  2·c·h³ + O(h⁵)
    ⇒ c ≈ [f(2h) - 2·f(h) + 2·f(-h) - f(-2h)] / (2 h³)
The O(h⁵) error comes from the 5th derivative; on smooth surrogates
this is small for h ≪ O(1).

Choice of h: too small → numerical noise from finite precision;
too large → higher-order terms (quartic, ...) corrupt the estimate.
Adaptive: start at h = 0.05 · canvas_diag, halve until |c·h³| ≫ |O(h⁵)|.

Generalization to multiple negative eigvecs (k > 1)
====================================================
For k ≥ 2, the cubic 3-tensor T_{ijk} = ∂³U/(∂v_i ∂v_j ∂v_k) restricted
to span{v_1, ..., v_k} matters. The "polynomial" critical-point system
becomes
    λ_i v_i + ½ Σ_{j,k} T_{ijk} v_j v_k = 0  ∀i
This is a system of k quadratic equations — k=2 has at most 4 critical
points (Bezout). For now we apply the fold formula to EACH eigvec
separately (treat as k independent 1-d unfoldings), and add a mixed
cross-term candidate at (t_1* v_1 + t_2* v_2) / √2. This gives k+1
candidates per Lanczos call without solving the full system.

Why this should help here
=========================
The existing adaptive line search along ±v does a backtracking search
from a fixed initial step. It scans 10 step sizes, picks the best.
Cost: 10 proxy evals/eigvec.

Catastrophe-fold instead computes the OPTIMAL step in closed form from
4 proxy evals (estimate c, derive t*). When the cubic dominates the
quartic (which it does at most saddles by genericity), this is more
accurate than line search and 2.5× cheaper.

Failure modes
-------------
- c ≈ 0: fold structure absent. Could be a cusp (A_3) — need quartic.
  Fallback: line search.
- Very large |t*|: outside trust region. Cap by canvas-fraction.
- Multiple eigvals near zero: the 1-d slice approximation is poor.
  Use the multi-eigvec mixed term as a hedge.

Implementation
==============
This module computes ONE escape candidate per eigvec using the fold
formula. Returns (label, pos_np) candidates.
"""

from __future__ import annotations
import time
import numpy as np
import torch


def estimate_directional_cubic(
    smooth_proxy_call,           # callable: x_tensor → scalar U
    x0: torch.Tensor,            # (n, 2) base placement
    v: np.ndarray,               # (2n,) unit eigvec, flattened
    *,
    h: float = 1.0,              # step size in micron units
) -> tuple[float, float]:
    """Compute (f''(0)·h², c) where c = f'''(0) along v.

    Uses 4-point central difference:
        c · h³ ≈ ½ · [f(2h) - 2 f(h) + 2 f(-h) - f(-2h)]
    Also returns λ_estimate = [f(h) + f(-h) - 2 f(0)] / h² (second
    derivative — should match Lanczos eigenvalue).
    """
    device = x0.device
    dtype = x0.dtype
    n_total = x0.shape[0]
    v_t = torch.tensor(v.reshape(n_total, 2), dtype=dtype, device=device)

    def U_at(t):
        with torch.no_grad():
            x_t = x0 + t * v_t
            return float(smooth_proxy_call(x_t).item())

    f0 = U_at(0.0)
    fp1 = U_at(+h)
    fm1 = U_at(-h)
    fp2 = U_at(+2.0 * h)
    fm2 = U_at(-2.0 * h)
    # 3rd derivative coefficient.
    cubic = (fp2 - 2.0 * fp1 + 2.0 * fm1 - fm2) / (2.0 * h ** 3)
    # 2nd derivative estimate (sanity check vs Lanczos λ).
    second = (fp1 + fm1 - 2.0 * f0) / (h ** 2)
    return float(second), float(cubic)


def catastrophe_fold_candidates(
    smooth_proxy_call,
    macro_pos: torch.Tensor,             # (n_total, 2) saddle position
    eigvals: np.ndarray,                 # (K,) negative eigvals
    eigvecs: np.ndarray,                 # (2 n_total, K) eigvecs (cols)
    *,
    canvas_diag: float = 1.0,
    cap_frac: float = 0.15,              # cap step at cap_frac · canvas_diag
    h_frac: float = 0.005,               # finite-diff h as fraction of canvas
    n_hard: int = 0,
    soft_only: bool = True,
    verbose: bool = False,
) -> tuple[list, dict]:
    """For each (λ, v) pair, predict optimal step t* = -2λ/c and emit
    one candidate at x_0 + t* · v.

    Also emits a "mixed" candidate combining the top-2 eigvecs.

    Returns (candidates, diag).
    """
    candidates: list = []
    diag = {"method": "catastrophe_fold", "per_eigvec": []}
    if eigvecs is None or eigvals is None:
        return [], {"warn": "no eigeninfo"}
    eigvecs = np.asarray(eigvecs, dtype=np.float64)
    eigvals = np.asarray(eigvals, dtype=np.float64)
    if eigvecs.ndim != 2 or eigvals.ndim != 1:
        return [], {"warn": f"bad shape {eigvecs.shape}/{eigvals.shape}"}
    N, K = eigvecs.shape
    n_total = macro_pos.shape[0]
    if N != 2 * n_total:
        return [], {"warn": f"eigvec dim {N} != 2 n_total {2*n_total}"}

    # Soft-only projection.
    if soft_only and n_hard > 0:
        view = eigvecs.reshape(n_total, 2, K).copy()
        view[:n_hard, :, :] = 0.0
        eigvecs = view.reshape(N, K)
        norms = np.linalg.norm(eigvecs, axis=0)
        norms = np.where(norms > 1e-12, norms, 1.0)
        eigvecs = eigvecs / norms[None, :]

    h = h_frac * float(canvas_diag)
    cap = cap_frac * float(canvas_diag)
    base_pos_np = macro_pos.detach().cpu().numpy().astype(np.float64)
    t_total = time.time()

    # Store t_stars for mixing.
    t_stars: list[float] = []
    valid_eigvecs: list[np.ndarray] = []

    for k in range(int(K)):
        v_k = eigvecs[:, k]
        lambda_k = float(eigvals[k])
        second_est, cubic = estimate_directional_cubic(
            smooth_proxy_call, macro_pos, v_k, h=h)
        # Skip if cubic too small to give a usable t*.
        if abs(cubic) < 1e-12 * max(abs(lambda_k), 1.0) / h:
            diag["per_eigvec"].append({
                "k": k, "lambda": lambda_k, "second_fd": second_est,
                "cubic": cubic, "t_star": None,
                "reason": "cubic_too_small"})
            continue
        t_star = -2.0 * lambda_k / cubic
        # Predicted ΔU at t*: 2 λ³ / (3 c²).
        delta_U_pred = 2.0 * lambda_k ** 3 / (3.0 * cubic ** 2)
        # Cap step.
        if abs(t_star) > cap:
            t_star_capped = np.sign(t_star) * cap
        else:
            t_star_capped = t_star
        # Build candidate position.
        delta = t_star_capped * v_k.reshape(n_total, 2)
        x_new = base_pos_np + delta
        diag["per_eigvec"].append({
            "k": k, "lambda": lambda_k, "second_fd": second_est,
            "cubic": cubic, "t_star": t_star,
            "t_star_used": t_star_capped,
            "delta_U_pred": delta_U_pred,
        })
        label = f"cata_e{k:02d}_t{t_star_capped:+.2f}_dU{delta_U_pred:+.4f}"
        candidates.append((label, x_new))
        t_stars.append(t_star_capped)
        valid_eigvecs.append(v_k)

    # Mixed candidate: top-2 eigvecs combined, each at half step.
    if len(t_stars) >= 2:
        delta_mix = (0.5 * t_stars[0] * valid_eigvecs[0]
                      + 0.5 * t_stars[1] * valid_eigvecs[1])
        # Cap the mixed step too.
        delta_mix_norm = float(np.linalg.norm(delta_mix))
        if delta_mix_norm > cap:
            delta_mix = delta_mix * (cap / delta_mix_norm)
        x_new = base_pos_np + delta_mix.reshape(n_total, 2)
        label = f"cata_mix_K{len(t_stars)}"
        candidates.append((label, x_new))

    diag["wall_s"] = time.time() - t_total
    if verbose:
        print(f"    [catastrophe] K={K} valid={len(t_stars)} "
              f"h={h:.2f}μm cap={cap:.1f}μm "
              f"({diag['wall_s']:.2f}s)", flush=True)
    return candidates, diag
