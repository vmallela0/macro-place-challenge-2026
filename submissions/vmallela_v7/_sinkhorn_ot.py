"""Sinkhorn Optimal Transport for global soft-cell reassignment.

Rapid Experiment #3.

Problem statement
-----------------
Given:
  - n_soft macros at current positions (p_i)
  - m grid cells with positions (q_j) and current congestion ρ_j
  - HPWL distance |p_i − q_j| as the "cost of moving"
  - Target: spread softs uniformly across cells (minimize density top-K)

We want a transport plan T[i,j] (soft i → cell j with weight T[i,j])
that minimizes:

    Σ_{i,j} T[i,j] · ( ρ_j + α |p_i − q_j| )    s.t. T·1 = 1/n_soft, 1·T = 1/m

This is a regularized OT problem; Sinkhorn solves it in O(n*m*iter) time
with iter ≈ 30 to converge for typical problems.

Then APPLY the plan: for each soft i, move it to argmax_j T[i,j].

Why this is novel for VLSI placement
------------------------------------
OT formulations have been used for placement *partitioning* (Caldwell-
Kahng-Markov 2000) but not for soft-cell density spreading. The
key insight: density top-K minimization is mathematically a transport
problem (move mass from over-occupied to under-occupied cells), so
Sinkhorn's convex relaxation gives a globally optimal *fractional*
assignment that can be projected to integer placements.

Compared to coordinate descent: CD evicts softs one at a time, allowing
neighboring softs to drift back into the cleared cell. Sinkhorn does it
all-at-once: every soft is assigned simultaneously, so the equilibrium
is reached in one shot.

Compared to Rapid #2 (greedy eviction): Sinkhorn finds the GLOBAL
optimum of the assignment cost; greedy is myopic and misses pairings
where two softs should swap.

Implementation
--------------
1. Build cost matrix C ∈ R^(n_soft × n_cells).
2. Run Sinkhorn for `iters` iterations with regularization ε.
3. For each soft, take argmax of T to get its assigned cell.
4. Apply moves all at once; validate via compute_proxy_cost.
5. Strict accept on overall improvement.

For ibm15: n_soft × n_cells ≈ 2k × 2k = 4M entries. Sinkhorn at iter=30
costs ~30 × 4M = 120M ops. ~few seconds. Affordable.
"""
from __future__ import annotations
import time
import numpy as np
import torch


def sinkhorn(
    C: np.ndarray,
    eps: float = 0.1,
    iters: int = 50,
    a: np.ndarray | None = None,
    b: np.ndarray | None = None,
) -> np.ndarray:
    """Log-domain Sinkhorn: T = arg min_T Σ T*C + ε Σ T log T
       s.t. T·1 = a, 1·T = b.

    Numerically stable for any C, ε; no exp(-C/ε) underflow/overflow.
    Updates the log-potentials f, g via logsumexp:
        g[j] = ε log(b[j]) - ε logsumexp((f - C[:,j])/ε)
        f[i] = ε log(a[i]) - ε logsumexp((g - C[i,:])/ε)
        T[i,j] = exp((f[i] + g[j] - C[i,j]) / ε)

    Equivalent to the multiplicative form K = exp(-C/ε) but operates
    entirely in log space until the final T construction. Stable when
    max(C)/ε ≫ log(realmax) ≈ 700.
    """
    n, m = C.shape
    if a is None:
        a = np.full(n, 1.0 / n, dtype=np.float64)
    if b is None:
        b = np.full(m, 1.0 / m, dtype=np.float64)
    log_a = np.log(np.maximum(a, 1e-300))
    log_b = np.log(np.maximum(b, 1e-300))

    f = np.zeros(n, dtype=np.float64)
    g = np.zeros(m, dtype=np.float64)
    for _ in range(iters):
        # g[j] = ε log(b[j]) - ε logsumexp_i ((f[i] - C[i,j]) / ε)
        # Implemented as ε * log_b - ε * logsumexp(M_col, axis=0)
        # where M[i,j] = (f[i] - C[i,j]) / ε.
        M_col = (f[:, None] - C) / eps           # (n, m)
        g = eps * (log_b - _logsumexp(M_col, axis=0))
        M_row = (g[None, :] - C) / eps           # (n, m)
        f = eps * (log_a - _logsumexp(M_row, axis=1))
    log_T = (f[:, None] + g[None, :] - C) / eps
    T = np.exp(log_T)
    return T


def _logsumexp(X: np.ndarray, axis: int) -> np.ndarray:
    """Numerically-stable logsumexp(X, axis=axis)."""
    M = np.max(X, axis=axis, keepdims=True)
    # Guard against -inf max (all -inf row → 0 sum → log -inf)
    M_safe = np.where(np.isfinite(M), M, 0.0)
    Y = np.exp(X - M_safe)
    S = np.sum(Y, axis=axis, keepdims=True)
    out = np.log(np.maximum(S, 1e-300)) + M_safe
    return np.squeeze(out, axis=axis)


def sinkhorn_evict(
    incr_eval,
    benchmark,
    plc,
    *,
    alpha_hpwl: float = 0.5,
    eps: float = 0.05,
    iters: int = 50,
    cong_floor: float = 1e-6,
    verbose: bool = False,
) -> tuple[np.ndarray, float, dict]:
    """Solve Sinkhorn OT for soft-cell assignment, apply, validate.

    Parameters
    ----------
    alpha_hpwl : weight on the |p_i - q_j| term in the cost matrix.
        Higher = preserves current positions (less mobility).
        Lower = more aggressive spreading.
    eps : Sinkhorn regularization. Smaller = sharper plan (more
        deterministic) but slower convergence.
    iters : number of Sinkhorn iterations.
    """
    from macro_place.objective import compute_proxy_cost

    n_hard = incr_eval.n_hard
    n_total = incr_eval.macro_pos.shape[0]
    n_soft = n_total - n_hard
    grid_col = incr_eval.grid_col
    grid_row = incr_eval.grid_row
    grid_w = incr_eval.grid_width
    grid_h = incr_eval.grid_height
    n_cells = grid_col * grid_row
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)

    initial_pos = np.array(incr_eval.macro_pos).copy()
    initial_tensor = torch.tensor(initial_pos, dtype=torch.float32)
    r0 = compute_proxy_cost(initial_tensor, benchmark, plc)
    initial_cost = float(r0["proxy_cost"])
    initial_overlaps = int(r0["overlap_count"])
    diagnostics = {"sinkhorn_iters": iters, "eps": eps,
                    "alpha_hpwl": alpha_hpwl,
                    "initial_cost": initial_cost,
                    "initial_overlaps": initial_overlaps}

    if initial_overlaps != 0:
        if verbose:
            print(f"  [sinkhorn] starting state has {initial_overlaps} "
                  f"overlaps; aborting", flush=True)
        return initial_pos, initial_cost, diagnostics

    # ── Compute current per-cell congestion ──────────────────────
    V = (np.asarray(incr_eval.V_routing_smooth)
         + np.asarray(incr_eval.V_macro_raw)
           / np.maximum(np.asarray(incr_eval.vrouting_alloc), 1e-9))
    H = (np.asarray(incr_eval.H_routing_smooth)
         + np.asarray(incr_eval.H_macro_raw)
           / np.maximum(np.asarray(incr_eval.hrouting_alloc), 1e-9))
    cong = np.maximum(V, H) + cong_floor   # (n_cells,)

    # ── Build cost matrix C[i, j] = cong[j] + alpha * dist(p_i, q_j) ──
    # p_i = current soft position; q_j = cell center.
    soft_pos = initial_pos[n_hard:]   # (n_soft, 2)
    soft_w = np.asarray(incr_eval.macro_w)[n_hard:]
    soft_h = np.asarray(incr_eval.macro_h)[n_hard:]

    cell_xs = np.arange(grid_col) * grid_w + grid_w / 2.0
    cell_ys = np.arange(grid_row) * grid_h + grid_h / 2.0
    cell_centers = np.empty((n_cells, 2), dtype=np.float64)
    cell_centers[:, 0] = np.tile(cell_xs, grid_row)
    cell_centers[:, 1] = np.repeat(cell_ys, grid_col)

    if verbose:
        print(f"  [sinkhorn] building cost matrix "
              f"{n_soft} × {n_cells}...", flush=True)
    t0 = time.time()
    # |p_i - q_j| = sqrt((dx)^2 + (dy)^2). For Sinkhorn we can use squared
    # distance which is differentiable and convex.
    dx = soft_pos[:, 0:1] - cell_centers[:, 0:1].T   # (n_soft, n_cells)
    dy = soft_pos[:, 1:2] - cell_centers[:, 1:2].T
    dist2 = dx ** 2 + dy ** 2
    canvas_diag2 = cw ** 2 + ch ** 2
    # Normalize distance to [0, 1] so eps is interpretable
    dist2_n = dist2 / canvas_diag2
    # Cong is already normalized roughly to O(1) by the per-cell raw/alloc
    # ratio. Take the scaled distance as the HPWL proxy.
    C = cong[None, :] + alpha_hpwl * dist2_n
    if verbose:
        print(f"  [sinkhorn] cost matrix built in "
              f"{time.time()-t0:.2f}s; running {iters} Sinkhorn iters",
              flush=True)

    t1 = time.time()
    T = sinkhorn(C, eps=eps, iters=iters)
    if verbose:
        print(f"  [sinkhorn] OT plan in {time.time()-t1:.2f}s; "
              f"row sums [{T.sum(1).min():.4f}, {T.sum(1).max():.4f}], "
              f"col sums [{T.sum(0).min():.4f}, {T.sum(0).max():.4f}]",
              flush=True)

    # ── Apply: each soft moves to argmax cell ──────────────────────
    target_cell = np.argmax(T, axis=1)   # (n_soft,)
    new_soft_pos = np.zeros_like(soft_pos)
    for si in range(n_soft):
        cidx = int(target_cell[si])
        c_col = cidx % grid_col
        c_row = cidx // grid_col
        # Place at cell lower-left, clipped to canvas
        nx = min(max(c_col * grid_w, 0.0), cw - soft_w[si])
        ny = min(max(c_row * grid_h, 0.0), ch - soft_h[si])
        new_soft_pos[si, 0] = nx
        new_soft_pos[si, 1] = ny

    trial_pos = initial_pos.copy()
    trial_pos[n_hard:] = new_soft_pos
    trial_tensor = torch.tensor(trial_pos, dtype=torch.float32)
    r = compute_proxy_cost(trial_tensor, benchmark, plc)
    trial_cost = float(r["proxy_cost"])
    trial_overlaps = int(r["overlap_count"])

    diagnostics["trial_cost_full_apply"] = trial_cost
    diagnostics["trial_overlaps_full_apply"] = trial_overlaps

    if trial_overlaps == 0 and trial_cost < initial_cost - 1e-7:
        if verbose:
            print(f"  [sinkhorn] FULL-APPLY WIN: cost {initial_cost:.6f} "
                  f"→ {trial_cost:.6f}", flush=True)
        return trial_pos, trial_cost, diagnostics

    # If full-apply doesn't win, try a more conservative approach:
    # apply only the top-K most-confident moves (by transport mass).
    if verbose:
        print(f"  [sinkhorn] full-apply failed (cost {trial_cost:.6f}, "
              f"ov {trial_overlaps}); trying confidence-sorted partial",
              flush=True)
    confidence = T[np.arange(n_soft), target_cell]
    order = np.argsort(-confidence)   # most-confident first
    best_pos = initial_pos.copy()
    best_cost = initial_cost
    accepted = 0
    for k_try, si in enumerate(order):
        cidx = int(target_cell[si])
        s = n_hard + si
        c_col = cidx % grid_col
        c_row = cidx // grid_col
        nx = min(max(c_col * grid_w, 0.0), cw - soft_w[si])
        ny = min(max(c_row * grid_h, 0.0), ch - soft_h[si])
        if abs(nx - best_pos[s, 0]) < 1e-9 and abs(ny - best_pos[s, 1]) < 1e-9:
            continue
        trial_pos = best_pos.copy()
        trial_pos[s, 0] = nx
        trial_pos[s, 1] = ny
        trial_tensor = torch.tensor(trial_pos, dtype=torch.float32)
        r = compute_proxy_cost(trial_tensor, benchmark, plc)
        if int(r["overlap_count"]) == 0 and float(r["proxy_cost"]) < best_cost - 1e-7:
            best_pos = trial_pos
            best_cost = float(r["proxy_cost"])
            accepted += 1
        if k_try >= 200:   # cap to avoid spending all budget here
            break

    diagnostics["accepted_partial"] = accepted
    diagnostics["final_cost"] = best_cost
    if verbose:
        print(f"  [sinkhorn] partial-apply: {accepted} moves accepted; "
              f"cost {initial_cost:.6f} → {best_cost:.6f}", flush=True)
    return best_pos, best_cost, diagnostics
