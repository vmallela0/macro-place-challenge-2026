"""Subspace Stein Variational Gradient Descent — zeus Plan B.

Why this exists
===============
Subspace HMC (`_subspace_hmc.py`) generates INDEPENDENT trajectories
with random momentum draws. Two trajectories that happen to draw
similar momenta land at similar endpoints — redundant candidates, no
mode coverage guarantee.

SVGD (Liu & Wang 2016) introduces an explicit REPULSION term between
particles. K particles {a_i} maintain a fixed-size ensemble; each
particle's update is the average of (i) the gradient of log p
weighted by kernel similarity, and (ii) the gradient of the kernel
itself which pushes particles apart. The fixed-point distribution
of SVGD is the target p — but unlike MCMC chains, every particle is
useful in every iteration.

For OPTIMIZATION (not sampling), we use the same machinery but
interpret p ∝ exp(-β U_smooth) as a sharpening of the surrogate
loss landscape. Large β → particles concentrate at local minima;
small β → particles spread to cover diverse basins.

After T iterations, return all K particles' endpoint placements as
candidates. The existing strict-improvement gate against EXACT
proxy filters out bad samples. Hyperparameters that matter:
  - n_particles K_part : ensemble size
  - n_iters T          : SVGD iterations
  - step_size η        : update rate per iteration (Adam-style)
  - β                  : inverse temperature; mediates explore/exploit
  - bandwidth h        : RBF kernel width; "median heuristic" sets
                          h² = median(||a_i-a_j||²) / log(K_part)

Math derivation
===============
Restrict motion to span{v_1, ..., v_K_sub} where {(λ_j, v_j)} are
the K_sub smallest Hessian eigenpairs. Let V ∈ ℝ^(2N × K_sub),
columns orthonormal (in the soft-only subspace). K_part-dim
particles a_i ∈ ℝ^K_sub define candidates x_i = x_0 + V a_i.

The target distribution on a-space:
    p(a) ∝ exp(-β · U_smooth(x_0 + V a))

SVGD update (Liu & Wang 2016, eq. 2):
    φ(a_i) = (1/K_part) Σ_j [k(a_j, a_i) · ∇_a_j log p(a_j)
                              + ∇_a_j k(a_j, a_i)]
    a_i  ← a_i + η · φ(a_i)

With p(a) ∝ exp(-β U):
    ∇_a_j log p(a_j) = -β · ∇_a U(a_j)
    ∇_a U(a) = V^T ∇_x U_smooth(x_0 + V a)  (chain rule)

RBF kernel: k(a, b) = exp(-||a-b||² / (2h²)).
∇_a_j k(a_j, a_i) = -k(a_j, a_i) · (a_j - a_i) / h²
                  = +k(a_j, a_i) · (a_i - a_j) / h²

The repulsion term pushes a_i away from a_j when they're close.

Median-heuristic bandwidth: h² = median_{i≠j}(||a_i - a_j||²) / log(K_part).
Recomputed each iteration so the kernel auto-adapts as particles
spread or contract.

Why subspace?
-------------
Full SVGD in 2N-dim space is well-defined but expensive: each
iteration does K_part autograd backward passes through smooth_proxy_call.
For 2N ≈ 5400 (ibm17), that's K_part × (~30 ms backward) = K_part · 30 ms
per iteration. For K_part=16 T=20 that's 9.6 s. Tractable but the
subspace formulation localizes exploration to the curvature-relevant
directions, which is where saddle-escape benefits compound.

Sign conventions (to avoid bugs)
================================
For numpy with diff[i,j,:] = A[i] - A[j]:
    repulsion_i = (1/K_part) Σ_j  +k_ij · diff[i,j,:] / h²
                = (1/K_part) Σ_j  +k_ij · (a_i - a_j) / h²

When i = j: diff = 0, repulsion contribution is zero. Good.
When i ≠ j and close: k_ij ≈ 1, repulsion pushes i away from j. Good.
When i ≠ j and far: k_ij ≈ 0, repulsion vanishes. Good.

Attraction term:
    attraction_i = (1/K_part) Σ_j  k_ij · (-β · ∇_a U(a_j))

So φ(a_i) = -β·(weighted avg of gradients at OTHER particles) + repulsion.
Each particle gets credit for the gradient signal at every other
particle (weighted by kernel similarity) — fast-mixing in the
high-density region of p.

Validation
==========
- K_part=1: SVGD reduces to standard gradient descent on a, scaled by
  β. Reproduces a simple gradient-step baseline.
- K_part=∞ (continuum limit): SVGD becomes the gradient flow of KL(q || p).
- Stein's identity: at the fixed point, E_a [∇_a log p(a) k(a, x) + ∇_a k(a, x)] = 0
  for all test functions, which uniquely determines q* = p.

Cost
====
Per iteration:
  - K_part autograd backward passes (the gradient computation).
  - O(K_part² · K_sub) for kernel matrix and pairwise distance.
  - Total: ~K_part × 30 ms = 0.5 s per iter for K_part=16.
T=20 iters × 0.5 s = 10 s wall per call. Final exact-proxy eval is
K_part × 100 ms ≈ 2 s. Total budget per SVGD call: ~12 s, comparable
to one HMC call.
"""
from __future__ import annotations
import time
import numpy as np
import torch


def subspace_svgd_candidates(
    macro_pos: torch.Tensor,             # (n_total, 2) current state
    smooth_proxy_call,                   # callable: x_tensor → scalar U
    eigvals: np.ndarray,                 # (K_sub,) Hessian eigenvalues
    eigvecs: np.ndarray,                 # (2 n_total, K_sub) eigvecs (cols)
    *,
    n_particles: int = 16,
    n_iters: int = 20,
    step_size: float = 0.3,              # SVGD η (raw a-space units)
    beta: float = 1.0,                   # inverse temperature for p ∝ exp(-βU)
    canvas_diag: float = 1.0,
    mass_floor: float = 1e-3,
    n_hard: int = 0,
    soft_only: bool = True,
    seed: int = 42,
    max_total_step_canvas: float = 0.30, # cap ||V·a_T||/canvas_diag per particle
    init_scale: float = 1.0,             # σ_0 for initial a_i ~ N(0, σ_0² · diag(1/|λ|))
    verbose: bool = False,
) -> tuple[list, dict]:
    """Generate candidate placements via subspace SVGD.

    Returns
    -------
    candidates : list of (label, perturbed_pos_np) tuples — n_particles entries
    diag       : dict with timing / per-iteration statistics
    """
    if eigvecs is None or eigvals is None:
        return [], {"warn": "no eigeninfo"}
    eigvecs = np.asarray(eigvecs, dtype=np.float64)
    eigvals = np.asarray(eigvals, dtype=np.float64)
    if eigvecs.ndim != 2 or eigvals.ndim != 1:
        return [], {"warn": f"bad shape {eigvecs.shape}/{eigvals.shape}"}
    N, K_sub = eigvecs.shape
    n_total = macro_pos.shape[0]
    if N != 2 * n_total:
        return [], {"warn": f"eigvec dim {N} != 2·n_total {2*n_total}"}
    if K_sub == 0 or n_particles < 1:
        return [], {"warn": "K_sub=0 or n_particles<1"}

    device = macro_pos.device
    dtype = macro_pos.dtype

    # Soft-only projection (same convention as HMC module).
    if soft_only and n_hard > 0:
        eigvecs_view = eigvecs.reshape(n_total, 2, K_sub).copy()
        eigvecs_view[:n_hard, :, :] = 0.0
        eigvecs = eigvecs_view.reshape(N, K_sub)
        col_norms = np.linalg.norm(eigvecs, axis=0)
        col_norms = np.where(col_norms > 1e-12, col_norms, 1.0)
        eigvecs = eigvecs / col_norms[None, :]

    V_torch = torch.tensor(eigvecs, dtype=dtype, device=device)
    abs_lam = np.maximum(np.abs(eigvals), mass_floor)
    sqrt_inv_lam = 1.0 / np.sqrt(abs_lam)

    base_xy_torch = macro_pos.detach().clone()
    n_total_int = int(n_total)
    K_part = int(n_particles)
    T = int(n_iters)
    eta = float(step_size)
    rng = np.random.default_rng(seed)
    cap_radius = float(max_total_step_canvas) * float(canvas_diag)
    t0_total = time.time()

    def _grad_at(a_np: np.ndarray) -> tuple[np.ndarray, float]:
        """Compute (∇_a U, U) at x = x_0 + V·a. Same closure shape as HMC."""
        a_t = torch.tensor(a_np, dtype=dtype, device=device,
                            requires_grad=True)
        delta = (V_torch @ a_t).reshape(n_total_int, 2)
        x_t = base_xy_torch + delta
        U_t = smooth_proxy_call(x_t)
        g_a_t = torch.autograd.grad(U_t, a_t)[0]
        return (g_a_t.detach().cpu().numpy().astype(np.float64),
                float(U_t.item()))

    # Initialize particles: a_i ~ N(0, σ²·diag(1/|λ|)) — soft directions
    # get larger initial spread, stiff directions less. Same scale as HMC's
    # momentum draw but in a-space.
    A = (rng.standard_normal((K_part, K_sub))
         * (init_scale * sqrt_inv_lam)[None, :])

    # Track best-so-far per particle (we return final endpoint, but
    # may also report which particle's BEST iterate was the lowest U).
    per_iter_stats: list = []
    best_U_per_particle = np.full(K_part, np.inf)
    best_A_per_particle = A.copy()
    U_history: list = []

    for it in range(T):
        # 1) Gradients ∇_a U(a_j) for each particle j.
        grads = np.zeros((K_part, K_sub), dtype=np.float64)
        U_vals = np.zeros(K_part, dtype=np.float64)
        for j in range(K_part):
            g_j, U_j = _grad_at(A[j])
            grads[j] = g_j
            U_vals[j] = U_j
            if U_j < best_U_per_particle[j]:
                best_U_per_particle[j] = U_j
                best_A_per_particle[j] = A[j].copy()
        U_history.append(U_vals.copy())

        # 2) Pairwise distance matrix in a-space.
        diff = A[:, None, :] - A[None, :, :]      # (K_part, K_part, K_sub)
        sq_dist = (diff ** 2).sum(-1)             # (K_part, K_part)

        # 3) Median-heuristic bandwidth h² = median(||·||²) / log(K_part).
        off_diag = sq_dist[~np.eye(K_part, dtype=bool)]
        med = float(np.median(off_diag)) if off_diag.size > 0 else 1.0
        h2 = max(med / max(np.log(K_part), 1.0), 1e-12)

        # 4) RBF kernel matrix.
        kernel = np.exp(-sq_dist / (2 * h2))      # (K_part, K_part)

        # 5) Attractive term (gradient term, weighted by kernel).
        #    score_j = -β · ∇_a U(a_j)
        score = -beta * grads                      # (K_part, K_sub)
        attraction = (kernel @ score) / K_part     # (K_part, K_sub)

        # 6) Repulsion term:
        #    repulsion_i = (1/K_part) Σ_j  k_ij · (a_i - a_j) / h²
        # We have diff[i,j,:] = a_i - a_j, so:
        repulsion = (kernel[:, :, None] * diff).sum(axis=1) / (K_part * h2)

        # 7) SVGD update.
        phi = attraction + repulsion               # (K_part, K_sub)
        A = A + eta * phi

        per_iter_stats.append({
            "iter": it,
            "h2": h2,
            "mean_U": float(U_vals.mean()),
            "median_U": float(np.median(U_vals)),
            "min_U": float(U_vals.min()),
            "max_U": float(U_vals.max()),
            "spread": float(np.sqrt(med)),
            "phi_norm": float(np.linalg.norm(phi)),
        })

    # Cap each particle's trajectory.
    candidates: list = []
    per_part_diag: list = []
    base_xy_np = macro_pos.detach().cpu().numpy().astype(np.float64)
    for i in range(K_part):
        a = A[i].copy()
        delta_xy = eigvecs @ a
        radius = float(np.linalg.norm(delta_xy))
        if radius > cap_radius > 0:
            scale = cap_radius / radius
            a = a * scale
            delta_xy = delta_xy * scale
        x_final = base_xy_np + delta_xy.reshape(n_total_int, 2)
        try:
            with torch.no_grad():
                x_t = torch.tensor(x_final, dtype=dtype, device=device)
                U_final = float(smooth_proxy_call(x_t).item())
        except Exception:
            U_final = float("nan")
        # Initial U for this particle (iteration 0).
        U0 = float(U_history[0][i]) if U_history else float("nan")
        label = f"svgd_p{i:02d}_T{T}_dU{(U_final - U0):+.4f}"
        candidates.append((label, x_final.astype(np.float64)))
        per_part_diag.append({
            "particle": i, "U0": U0, "U_final": U_final,
            "delta_U": U_final - U0,
            "best_U": float(best_U_per_particle[i]),
            "a_norm": float(np.linalg.norm(a)),
            "radius_microns": float(np.linalg.norm(delta_xy)),
        })

    diag = {
        "method": "subspace_svgd",
        "n_particles": K_part,
        "n_iters": T,
        "step_size": eta,
        "beta": beta,
        "K_sub": K_sub,
        "mass_floor": mass_floor,
        "init_scale": init_scale,
        "wall_s": time.time() - t0_total,
        "lambda_min": float(np.min(eigvals)),
        "per_iter": per_iter_stats,
        "per_particle": per_part_diag,
    }
    if verbose:
        n_improving = sum(
            1 for d in per_part_diag
            if d["delta_U"] < -1e-9 and np.isfinite(d["delta_U"]))
        spread0 = per_iter_stats[0]["spread"] if per_iter_stats else 0.0
        spread_T = per_iter_stats[-1]["spread"] if per_iter_stats else 0.0
        print(
            f"    [subspace-svgd] K_part={K_part} K_sub={K_sub} T={T} "
            f"η={eta:.3f} β={beta:.2f} surrogate-improving={n_improving} "
            f"spread {spread0:.2f}→{spread_T:.2f}  ({diag['wall_s']:.1f}s)",
            flush=True,
        )
    return candidates, diag
