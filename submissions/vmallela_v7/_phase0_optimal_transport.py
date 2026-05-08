"""albania2 Bet C: Sinkhorn-based optimal transport warm-start.

Different mathematical pathway from Bet A (Phase 0 homotopy). Where
homotopy iterates Adam on a smooth surrogate, OT computes the *exact*
minimum-displacement uniformizer in closed form via Sinkhorn iteration.

Mathematical setup:
    Each macro is a point mass at its current center z_m. Define
    source distribution μ = Σ_m δ_{z_m} / N.  Define target
    distribution ν = uniform on canvas at N point locations
    {y_1, .., y_N} (an N-point lattice / blue-noise grid).

    The Wasserstein-2 transport plan T : {z_m} → {y_n} is the
    bijection minimizing Σ_m ||z_m - T(z_m)||². Brenier (1991) proves
    this T is the gradient of a convex potential — a curl-free
    displacement field, the *minimum-disruption* uniformization.

    Sinkhorn (entropic regularization) approximates this:
        Cost matrix:  C[i,j] = ||z_i - y_j||²
        Kernel:        K[i,j] = exp(-C[i,j] / ε)
        Iterate:       u = a / (K v),  v = b / (K^T u)   (a = b = 1/N)
        Plan:          T = diag(u) · K · diag(v)

    Macro displacement: weighted average over T,
        Δz_m = Σ_n T[m, n] · (y_n - z_m) / Σ_n T[m, n]

vs Phase 0 homotopy:
    Phase 0 uses Adam on a smooth surrogate (HPWL + λ·density). The
    surrogate is differentiable but its minimum may differ from the
    *true* minimum-displacement uniformization. OT computes that
    uniformization directly, in closed form (up to entropic noise).
    Phase 0 risks overshoot/oscillation; OT does not.

vs the failed random-init Phase 0:
    Random init has no HPWL structure. OT init starts from .plc and
    minimally perturbs — preserves HPWL topology while fixing density.
"""
from __future__ import annotations
import time
import numpy as np


def sinkhorn_transport(
    source_pts: np.ndarray,                 # (N, 2)
    target_pts: np.ndarray,                 # (M, 2)
    *,
    epsilon: float = 0.01,
    n_iters: int = 200,
    a: np.ndarray | None = None,            # (N,) source weights, None = uniform
    b: np.ndarray | None = None,            # (M,) target weights, None = uniform
) -> np.ndarray:
    """Returns transport plan T of shape (N, M).

    Standard Sinkhorn-Knopp iteration. ε controls entropic regularization:
    smaller ε → closer to exact OT but slower convergence + numerical
    issues. ε = 0.01 (in normalized units of canvas-diag) typically
    converges in 100-300 iterations for placement-scale problems.
    """
    N = int(source_pts.shape[0])
    M = int(target_pts.shape[0])
    if a is None:
        a = np.ones(N, dtype=np.float64) / float(N)
    if b is None:
        b = np.ones(M, dtype=np.float64) / float(M)
    # Squared Euclidean cost
    diff = source_pts[:, None, :] - target_pts[None, :, :]
    C = np.sum(diff * diff, axis=2)             # (N, M)
    # Normalize cost to canvas scale to keep ε meaningful
    C_max = float(C.max()) if C.size else 1.0
    if C_max > 1e-12:
        C = C / C_max
    K = np.exp(-C / float(epsilon))             # (N, M)
    u = np.ones(N, dtype=np.float64)
    v = np.ones(M, dtype=np.float64)
    for _ in range(int(n_iters)):
        u = a / np.maximum(K @ v, 1e-300)
        v = b / np.maximum(K.T @ u, 1e-300)
    T = u[:, None] * K * v[None, :]              # (N, M)
    return T


def ot_warmstart(
    bench,
    incr,
    *,
    target_grid_dim: int = 0,                   # 0 → auto = ceil(sqrt(N))
    epsilon: float = 0.005,
    n_sinkhorn_iters: int = 200,
    soft_only: bool = False,
    blend: float = 1.0,                          # 0=no move, 1=full OT displacement
    verbose: bool = False,
) -> np.ndarray:
    """Compute OT-warmstart positions for all macros.

    Algorithm:
      1. Build target = N-point lattice on canvas (in-bounds with macro
         half-extents respected as a margin)
      2. Sinkhorn → transport plan T (N × N)
      3. For each macro m: weighted-mean target = Σ_n T[m,n]·y_n / Σ T[m,n]
      4. Δz_m = blend · (weighted_target_m - z_m), apply with canvas clamp

    blend < 1 lets the caller take a partial step toward the OT
    target — useful when a full step would disrupt HPWL too much.
    """
    cw, ch = float(incr.cw), float(incr.ch)
    macro_pos_now = np.asarray(incr.macro_pos).copy().astype(np.float64)
    macro_w_np = np.asarray(incr.macro_w).astype(np.float64)
    macro_h_np = np.asarray(incr.macro_h).astype(np.float64)
    n_total = int(macro_pos_now.shape[0])
    n_hard = int(bench.num_hard_macros)

    # Build target lattice (N points roughly uniform on canvas).
    if target_grid_dim <= 0:
        side = int(np.ceil(np.sqrt(n_total)))
    else:
        side = int(target_grid_dim)
    n_target = side * side
    # Account for macro half-extents (mean) when laying out the lattice.
    margin_x = 0.5 * float(np.median(macro_w_np))
    margin_y = 0.5 * float(np.median(macro_h_np))
    xs = np.linspace(margin_x, cw - margin_x, side)
    ys = np.linspace(margin_y, ch - margin_y, side)
    Xg, Yg = np.meshgrid(xs, ys)
    target_pts = np.stack([Xg.flatten(), Yg.flatten()], axis=1)   # (side², 2)

    # If lattice has fewer points than macros (shouldn't happen with
    # ceil-sqrt but just in case), pad with random target locations.
    if n_target < n_total:
        rng = np.random.RandomState(0)
        extra = n_total - n_target
        ex = rng.uniform(margin_x, cw - margin_x, size=extra)
        ey = rng.uniform(margin_y, ch - margin_y, size=extra)
        target_pts = np.concatenate(
            [target_pts, np.stack([ex, ey], axis=1)], axis=0)

    # If macro count doesn't match target count, weight accordingly.
    a = np.ones(n_total, dtype=np.float64) / float(n_total)
    b = np.ones(target_pts.shape[0], dtype=np.float64) / float(target_pts.shape[0])

    t0 = time.time()
    T = sinkhorn_transport(macro_pos_now, target_pts,
                              epsilon=epsilon,
                              n_iters=n_sinkhorn_iters,
                              a=a, b=b)
    if verbose:
        print(f"  [phase0.ot] Sinkhorn N={n_total}×{target_pts.shape[0]} "
              f"in {time.time()-t0:.2f}s, "
              f"plan_max={T.max():.6f}, plan_min={T.min():.2e}",
              flush=True)

    # Weighted-mean target per source.
    row_sum = T.sum(axis=1, keepdims=True).clip(min=1e-300)
    weighted_target = (T @ target_pts) / row_sum              # (N, 2)
    new_pos = (1.0 - blend) * macro_pos_now + blend * weighted_target

    # Soft-only: keep hards at their current positions
    if soft_only and n_hard > 0:
        new_pos[:n_hard] = macro_pos_now[:n_hard]

    # Clamp into canvas with per-macro half-extent margin
    half_w = macro_w_np / 2.0
    half_h = macro_h_np / 2.0
    new_pos[:, 0] = np.clip(new_pos[:, 0], half_w, cw - half_w)
    new_pos[:, 1] = np.clip(new_pos[:, 1], half_h, ch - half_h)

    if verbose:
        delta = np.linalg.norm(new_pos - macro_pos_now, axis=1)
        print(f"  [phase0.ot] displacements: mean={delta.mean():.2f} "
              f"max={delta.max():.2f} (canvas={cw:.1f}×{ch:.1f}, "
              f"blend={blend:.2f})", flush=True)
    return new_pos.astype(np.float32)
