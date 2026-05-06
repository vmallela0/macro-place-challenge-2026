"""Hessian negative-eigenvector escape — Candidate #1.

Inspired by transition-state theory in chemistry (Crippen-Snyder 1971,
Henkelman 2000 dimer method). At a true local minimum, all Hessian
eigenvalues are ≥ 0. At a saddle, the smallest is < 0 and its
eigenvector is the unique direction in which the cost function curves
DOWN beyond the local min — the "reaction coordinate" for crossing
the barrier.

For our placement, we compute the Hessian of the smooth surrogate
(LSE-HPWL + softplus density / cong, sans top-K because the smoothed
top-K threshold is implicit). Use Hessian-vector products via second-
order autograd; Lanczos (scipy.sparse.linalg.eigsh) to find the
smallest eigenvalue without building the full Hessian.

If λ_min < 0: we are AT a saddle (or near it). The escape direction
v_min is well-defined.

If λ_min ≥ 0: we are at a true local minimum of the SMOOTH surrogate.
But that's the smoothed landscape; in the EXACT cost, this might still
be a saddle (smoothing rounds saddles into apparent minima). v_min is
still the "least-resistance" direction. Try it anyway.

Algorithm
---------
1. Compute Hessian-vector product as a LinearOperator on the (macro_pos
   × 2) flattened space (size N = 2 × n_macros).
2. Lanczos for smallest k=1 eigenvalue + eigenvector.
3. Step in the v_min direction by step_size · v_min (tested at multiple
   step sizes via geometric search).
4. Push-apart + legalize + run downstream pipeline from each perturbed
   state. Validate exact cost. Strict-improvement gate.

Cost: HVP is O(N) autograd ops ≈ ~1 forward+backward pass = ~5 ms on
ibm15 MPS. Lanczos for k=1 typically converges in 30-100 iters. So
total Hessian-escape time per benchmark is ~0.5-1 sec for the eigvec
+ pipeline cost for downstream.
"""
from __future__ import annotations
import math
import time
import numpy as np
import torch


def hessian_min_eigvec(
    proxy_call,                # callable that returns scalar loss given macro_pos tensor
    macro_pos: torch.Tensor,   # (n_total, 2) requires_grad
    *,
    n_lanczos_iters: int = 50,
    tolerance: float = 1e-4,
    tikhonov: float = 0.0,     # add ε·I to Hessian for numerical robustness
    auto_retry: bool = True,    # on convergence failure, retry with 4× maxiter
    verbose: bool = False,
) -> tuple[float, np.ndarray]:
    """Compute the smallest eigenvalue + eigenvector of the Hessian of
    proxy_call at macro_pos.

    Uses scipy.sparse.linalg.eigsh with a Hessian-vector LinearOperator.
    eigsh's `which='SA'` returns the smallest algebraic (most negative).

    Numerical robustness:
    - `tikhonov` adds ε·I to the operator. This shifts all eigenvalues
      by +ε, so λ_min(H+εI) = λ_min(H)+ε. We subtract ε after solving
      to recover the original λ_min. Tikhonov ε > 0 helps Lanczos when
      the Hessian is rank-deficient or has tightly-clustered eigvals.
    - `auto_retry`: on convergence failure (ARPACK error -1), retry
      with 4× maxiter. Many failures are due to insufficient iters,
      not actual divergence. Costs an extra 4× HVPs only on the
      already-failed branch.

    Returns (lambda_min, v_min) where v_min is shape (n_total*2,) numpy.
    """
    from scipy.sparse.linalg import eigsh, LinearOperator
    n_total = macro_pos.shape[0]
    N = 2 * n_total

    # Compute the gradient at the current state (with create_graph=True
    # so we can take the second derivative).
    macro_pos = macro_pos.detach().clone().requires_grad_(True)
    loss = proxy_call(macro_pos)
    grad = torch.autograd.grad(loss, macro_pos, create_graph=True)[0]
    grad_flat = grad.reshape(-1)
    if verbose:
        print(f"    [hessian] loss={loss.item():.6f} "
              f"||grad||={float(grad_flat.norm()):.4f} "
              f"tikhonov={tikhonov:.2e}", flush=True)

    def hv(v_np: np.ndarray) -> np.ndarray:
        v = torch.tensor(v_np, dtype=macro_pos.dtype,
                         device=macro_pos.device).reshape(n_total, 2)
        # H @ v = ∂(grad · v)/∂x
        gv = (grad * v).sum()
        Hv = torch.autograd.grad(gv, macro_pos, retain_graph=True)[0]
        out = Hv.detach().cpu().numpy().reshape(-1)
        if tikhonov > 0:
            out = out + tikhonov * v_np
        return out

    H_op = LinearOperator(shape=(N, N), matvec=hv, dtype=np.float64)

    # eigsh wants which='SA' for smallest algebraic. k=1 most-negative eigval.
    def _solve(maxiter):
        return eigsh(H_op, k=1, which="SA",
                       maxiter=maxiter, tol=tolerance)

    try:
        eigvals, eigvecs = _solve(n_lanczos_iters)
    except Exception as e:
        if auto_retry:
            if verbose:
                print(f"    [hessian] eigsh err on {n_lanczos_iters} iters: "
                      f"{e}; retrying with {4*n_lanczos_iters} iters",
                      flush=True)
            try:
                eigvals, eigvecs = _solve(4 * n_lanczos_iters)
            except Exception as e2:
                if verbose:
                    print(f"    [hessian] eigsh err on retry: {e2}",
                          flush=True)
                return 0.0, np.zeros(N)
        else:
            if verbose:
                print(f"    [hessian] eigsh err: {e}", flush=True)
            return 0.0, np.zeros(N)

    lam_min = float(eigvals[0]) - float(tikhonov)
    v_min = eigvecs[:, 0]
    if verbose:
        print(f"    [hessian] λ_min={lam_min:.6f} "
              f"(raw={float(eigvals[0]):.6f}, tikh={tikhonov:.2e}), "
              f"||v_min||={np.linalg.norm(v_min):.4f}",
              flush=True)
    return lam_min, v_min


def hessian_min_eigvecs_topk(
    proxy_call,
    macro_pos: torch.Tensor,
    *,
    k: int = 3,
    n_lanczos_iters: int = 50,
    tolerance: float = 1e-4,
    tikhonov: float = 0.0,
    auto_retry: bool = True,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the k smallest eigenvalues + eigenvectors of the Hessian.

    Mathematical justification: H is real symmetric → orthogonal
    eigenvector basis. Top-k smallest eigenvalues identify the
    "negative curvature subspace" (when negative). Each eigenvector
    is an independent escape direction in the high-dim manifold.

    See `hessian_min_eigvec` docstring for tikhonov / auto_retry.

    Returns (lambda_array, eigvec_matrix) where eigvec_matrix has
    shape (N, k) with each column a normalized eigenvector.
    """
    from scipy.sparse.linalg import eigsh, LinearOperator
    n_total = macro_pos.shape[0]
    N = 2 * n_total

    macro_pos = macro_pos.detach().clone().requires_grad_(True)
    loss = proxy_call(macro_pos)
    grad = torch.autograd.grad(loss, macro_pos, create_graph=True)[0]
    grad_flat = grad.reshape(-1)

    def hv(v_np: np.ndarray) -> np.ndarray:
        v = torch.tensor(v_np, dtype=macro_pos.dtype,
                         device=macro_pos.device).reshape(n_total, 2)
        gv = (grad * v).sum()
        Hv = torch.autograd.grad(gv, macro_pos, retain_graph=True)[0]
        out = Hv.detach().cpu().numpy().reshape(-1)
        if tikhonov > 0:
            out = out + tikhonov * v_np
        return out

    H_op = LinearOperator(shape=(N, N), matvec=hv, dtype=np.float64)
    def _solve(maxiter):
        return eigsh(H_op, k=k, which="SA",
                      maxiter=maxiter, tol=tolerance)
    try:
        eigvals, eigvecs = _solve(n_lanczos_iters)
    except Exception as e:
        if auto_retry:
            if verbose:
                print(f"    [hessian.topk] eigsh err on {n_lanczos_iters} "
                      f"iters: {e}; retrying with {4*n_lanczos_iters}",
                      flush=True)
            try:
                eigvals, eigvecs = _solve(4 * n_lanczos_iters)
            except Exception as e2:
                if verbose:
                    print(f"    [hessian.topk] eigsh err on retry: {e2}",
                          flush=True)
                return np.zeros(k), np.zeros((N, k))
        else:
            if verbose:
                print(f"    [hessian.topk] eigsh err: {e}", flush=True)
            return np.zeros(k), np.zeros((N, k))
    if tikhonov > 0:
        eigvals = eigvals - tikhonov
    return eigvals, eigvecs


def iterative_hessian_termination_check(
    proxy_call,
    macro_pos: torch.Tensor,
    *,
    n_lanczos_iters: int = 30,
    epsilon: float = -1e-5,
) -> tuple[bool, float]:
    """Should we keep iterating Hessian escape?

    Returns (should_stop, lambda_min). Stop when lambda_min ≥ epsilon
    (we're at a 2nd-order critical point of the smooth surrogate, no
    more negative-curvature direction).
    """
    from scipy.sparse.linalg import eigsh, LinearOperator
    n_total = macro_pos.shape[0]
    N = 2 * n_total
    macro_pos_d = macro_pos.detach().clone().requires_grad_(True)
    loss = proxy_call(macro_pos_d)
    grad = torch.autograd.grad(loss, macro_pos_d, create_graph=True)[0]

    def hv(v_np):
        v = torch.tensor(v_np, dtype=macro_pos_d.dtype,
                         device=macro_pos_d.device).reshape(n_total, 2)
        Hv = torch.autograd.grad(
            (grad * v).sum(), macro_pos_d, retain_graph=True)[0]
        return Hv.detach().cpu().numpy().reshape(-1)
    H_op = LinearOperator(shape=(N, N), matvec=hv, dtype=np.float64)
    try:
        eigvals, _ = eigsh(H_op, k=1, which="SA", maxiter=n_lanczos_iters,
                            tol=1e-3)
        lam = float(eigvals[0])
    except Exception:
        return True, 0.0   # error → stop (be conservative)
    return (lam >= epsilon), lam


def hessian_escape_step(
    macro_pos: torch.Tensor,
    smooth_proxy_call,
    *,
    step_sizes: list = (0.05, 0.10, 0.20, 0.40),  # multiples of canvas_diag
    canvas_diag: float = 1.0,
    n_lanczos_iters: int = 50,
    n_hard: int = 0,
    soft_only_perturb: bool = True,
    verbose: bool = False,
) -> tuple[list, dict]:
    """Compute v_min via Lanczos, return a list of candidate perturbed
    placements (one per step_size). Caller validates each via the full
    pipeline.

    Returns:
        candidates: list of (step_size, perturbed_pos_np) tuples
        diagnostics: dict with lambda_min, ||v_min||, and other info
    """
    n_total = macro_pos.shape[0]
    lam_min, v_min = hessian_min_eigvec(
        smooth_proxy_call, macro_pos,
        n_lanczos_iters=n_lanczos_iters,
        verbose=verbose)
    v_min_xy = v_min.reshape(n_total, 2)
    # Normalize the eigenvector
    v_norm = np.linalg.norm(v_min_xy)
    if v_norm < 1e-12:
        return [], {"lambda_min": lam_min, "v_norm": v_norm,
                     "warn": "degenerate eigenvector"}
    v_min_xy = v_min_xy / v_norm
    if soft_only_perturb and n_hard > 0:
        v_min_xy[:n_hard] = 0.0
        # Renormalize after zeroing hards
        v_norm_post = np.linalg.norm(v_min_xy)
        if v_norm_post > 1e-12:
            v_min_xy = v_min_xy / v_norm_post

    # Build candidates
    base = macro_pos.detach().cpu().numpy()
    candidates = []
    for s in step_sizes:
        delta_norm = s * canvas_diag
        perturbed = base + delta_norm * v_min_xy
        candidates.append((s, perturbed))
        # Also try the OPPOSITE direction (the eigvec sign is ambiguous)
        candidates.append((-s, base - delta_norm * v_min_xy))
    diagnostics = {
        "lambda_min": lam_min,
        "v_norm_pre": float(v_norm),
        "n_candidates": len(candidates),
        "n_hard_zeroed": int(n_hard) if soft_only_perturb else 0,
    }
    return candidates, diagnostics


# ============================================================================
# istanbul branch additions: adaptive line search + feasibility filter.
#
# Critique of the v7-combinatorial baseline:
#   1. Step sizes [0.02, 0.05] are arbitrary heuristics.
#   2. No feasibility check before launching 1000s SA workers per candidate.
#
# Fixes:
#   adaptive_topk_candidates  — backtracking line search on the smooth proxy
#                                replaces fixed step_sizes; one candidate per
#                                eigvec at its optimal step.
#   feasibility_filter        — O(N²) vectorized overlap count; drops
#                                candidates with too many overlaps before
#                                spawning expensive SA workers.
# ============================================================================


def adaptive_line_search(
    proxy_call,
    x0: "torch.Tensor",
    direction: "torch.Tensor",
    *,
    initial: float = 0.10,
    n_steps: int = 10,
    shrink: float = 0.6,
) -> tuple[float, float]:
    """Backtracking line search to find the best step along ±direction.

    Geometric backtracking — start at `initial` step, halve until either we
    find a step that reduces the surrogate proxy or we exhaust `n_steps`.
    Tries both signs (±direction) at every shrink level since saddle-escape
    eigvec sign is ambiguous.

    Returns (best_step, best_proxy) where best_step has sign included.
    Returns (0.0, f0) if no improvement found within budget.
    """
    import torch
    with torch.no_grad():
        f0 = float(proxy_call(x0).item())
    best_s, best_f = 0.0, f0

    s = float(initial)
    for _ in range(int(n_steps)):
        for sign in (+1.0, -1.0):
            x_try = x0 + sign * s * direction
            try:
                with torch.no_grad():
                    f_try = float(proxy_call(x_try).item())
            except Exception:
                continue
            if f_try < best_f:
                best_s = sign * s
                best_f = f_try
        s *= float(shrink)
    return best_s, best_f


def adaptive_topk_candidates(
    macro_pos: "torch.Tensor",
    smooth_proxy_call,
    *,
    k: int = 1,                    # was k=2; k=1 converges reliably in n_lanczos_iters=50
    canvas_diag: float = 1.0,
    n_lanczos_iters: int = 50,
    tikhonov: float = 0.0,
    n_hard: int = 0,
    soft_only_perturb: bool = True,
    ls_initial: float = 0.10,
    ls_n_steps: int = 10,
    ls_shrink: float = 0.6,
    verbose: bool = False,
) -> tuple[list, dict]:
    """Adaptive variant of slj2_topk_candidates.

    For each of the top-k smallest-eigval eigvecs of the Hessian, run a
    backtracking line search on the smooth surrogate to find the *optimal*
    step size in that direction. Yields one candidate per eigvec (with the
    best step found), instead of |step_sizes| × 2 candidates per eigvec at
    arbitrary fixed step magnitudes.

    Mathematically: at a true saddle (∇U = 0, λ_min < 0), the second-order
    Taylor gives U(x + s·v) ≈ U(x) + ½ s² λ_min, monotone-decreasing in
    |s| up to where higher-order terms kick in. The optimal step is the
    one where the cubic/quartic correction starts to dominate, which is
    bench-and-state-specific — *exactly* what line search discovers.
    """
    import numpy as np
    import torch

    # Fallback ladder: try the original config; if eigvec degenerate, retry
    # with (k=1, tikhonov bumped, more iters). This handles the "cong-included
    # surrogate makes Lanczos non-convergent" failure mode.
    eigvals, eigvecs = hessian_min_eigvecs_topk(
        smooth_proxy_call, macro_pos,
        k=k, n_lanczos_iters=n_lanczos_iters, tikhonov=tikhonov,
        verbose=verbose)
    if (eigvecs is None or eigvecs.shape[1] == 0
            or float(np.linalg.norm(eigvecs)) < 1e-12):
        if verbose:
            print(f"    [hessian.adaptive] k={k} unconverged; retry k=1 "
                  f"with tikhonov=1e-3 + 4× iters", flush=True)
        eigvals, eigvecs = hessian_min_eigvecs_topk(
            smooth_proxy_call, macro_pos,
            k=1, n_lanczos_iters=4 * n_lanczos_iters,
            tikhonov=max(1e-3, tikhonov * 100), verbose=verbose)

    n_total = macro_pos.shape[0]
    base = macro_pos.detach().cpu().numpy().copy()
    candidates: list = []
    used_eigvals: list = []
    line_search_results: list = []

    for j in range(eigvecs.shape[1]):
        v_j = eigvecs[:, j].reshape(n_total, 2)
        v_norm = float(np.linalg.norm(v_j))
        if v_norm < 1e-12:
            continue
        v_j = v_j / v_norm
        if soft_only_perturb and n_hard > 0:
            v_j[:n_hard] = 0.0
            v_norm_post = float(np.linalg.norm(v_j))
            if v_norm_post < 1e-12:
                continue
            v_j = v_j / v_norm_post

        # Scale by canvas_diag so step=1 means "one canvas-diag's worth of move"
        v_j_scaled = canvas_diag * v_j
        v_t = torch.tensor(v_j_scaled, dtype=macro_pos.dtype,
                           device=macro_pos.device)

        best_s, best_f = adaptive_line_search(
            smooth_proxy_call, macro_pos, v_t,
            initial=ls_initial, n_steps=ls_n_steps, shrink=ls_shrink)

        if abs(best_s) < 1e-6:
            line_search_results.append((j, 0.0, best_f, "no_improvement"))
            continue

        perturbed = base + best_s * v_j_scaled
        used_eigvals.append(float(eigvals[j]))
        candidates.append(
            (f"e{j}_ls{best_s:+.3f}", perturbed))
        line_search_results.append((j, best_s, best_f, "kept"))
        if verbose:
            print(f"    [hessian.adaptive] e{j}: λ={eigvals[j]:+.4e} "
                  f"best_step={best_s:+.4f} surrogate_f={best_f:.6f}",
                  flush=True)

    diag = {
        "lambda_min": float(eigvals[0]) if len(eigvals) else 0.0,
        "lambda_topk": [float(x) for x in used_eigvals],
        "k_eigvecs_used": len(used_eigvals),
        "n_candidates": len(candidates),
        "method": "adaptive_line_search",
        "line_search_results": line_search_results,
    }
    return candidates, diag


def feasibility_filter(
    candidates: list,
    benchmark,
    *,
    max_overlaps: int = 200,
    gap: float = 0.05,
) -> tuple[list, list]:
    """Drop candidates with too many macro-macro overlaps before SA workers spawn.

    Vectorized O(N²) pairwise overlap count on hard macros (soft macros
    are stand-cell clusters — overlaps are allowed between them per the
    competition spec). Threshold defaults to 200 overlaps which is the
    rough boundary above which legalize+CD has historically failed to
    recover within the SA budget.

    Each dropped candidate would otherwise consume hop_budget seconds of
    SA wall time on a result that ends up INVALID anyway. Filtering at
    candidate-gen time saves k_dropped × hop_budget seconds.
    """
    import numpy as np
    n_hard = benchmark.num_hard_macros
    sizes = benchmark.macro_sizes[:n_hard].numpy().astype(np.float64)
    sep_x = (sizes[:, 0:1] + sizes[:, 0:1].T) / 2.0 - gap
    sep_y = (sizes[:, 1:2] + sizes[:, 1:2].T) / 2.0 - gap

    keep: list = []
    drop: list = []
    for label, pos in candidates:
        p = pos[:n_hard]
        dx = np.abs(p[:, 0:1] - p[:, 0:1].T)
        dy = np.abs(p[:, 1:2] - p[:, 1:2].T)
        overlap = (dx < sep_x) & (dy < sep_y)
        np.fill_diagonal(overlap, False)
        n_ov = int(np.triu(overlap, k=1).sum())
        if n_ov <= int(max_overlaps):
            keep.append((label, pos))
        else:
            drop.append((label, n_ov))
    return keep, drop
