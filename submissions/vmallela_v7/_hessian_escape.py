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
    verbose: bool = False,
) -> tuple[float, np.ndarray]:
    """Compute the smallest eigenvalue + eigenvector of the Hessian of
    proxy_call at macro_pos.

    Uses scipy.sparse.linalg.eigsh with a Hessian-vector LinearOperator.
    eigsh's `which='SA'` returns the smallest algebraic (most negative).

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
              f"||grad||={float(grad_flat.norm()):.4f}", flush=True)

    def hv(v_np: np.ndarray) -> np.ndarray:
        v = torch.tensor(v_np, dtype=macro_pos.dtype,
                         device=macro_pos.device).reshape(n_total, 2)
        # H @ v = ∂(grad · v)/∂x
        gv = (grad * v).sum()
        Hv = torch.autograd.grad(gv, macro_pos, retain_graph=True)[0]
        return Hv.detach().cpu().numpy().reshape(-1)

    H_op = LinearOperator(shape=(N, N), matvec=hv, dtype=np.float64)

    # eigsh wants which='SA' for smallest algebraic. k=1 most-negative eigval.
    try:
        eigvals, eigvecs = eigsh(
            H_op, k=1, which="SA",
            maxiter=n_lanczos_iters, tol=tolerance)
    except Exception as e:
        if verbose:
            print(f"    [hessian] eigsh err: {e}", flush=True)
        return 0.0, np.zeros(N)

    lam_min = float(eigvals[0])
    v_min = eigvecs[:, 0]
    if verbose:
        print(f"    [hessian] λ_min={lam_min:.6f}, ||v_min||={np.linalg.norm(v_min):.4f}",
              flush=True)
    return lam_min, v_min


def hessian_min_eigvecs_topk(
    proxy_call,
    macro_pos: torch.Tensor,
    *,
    k: int = 3,
    n_lanczos_iters: int = 50,
    tolerance: float = 1e-4,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the k smallest eigenvalues + eigenvectors of the Hessian.

    Mathematical justification: H is real symmetric → orthogonal
    eigenvector basis. Top-k smallest eigenvalues identify the
    "negative curvature subspace" (when negative). Each eigenvector
    is an independent escape direction in the high-dim manifold.

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
        return Hv.detach().cpu().numpy().reshape(-1)

    H_op = LinearOperator(shape=(N, N), matvec=hv, dtype=np.float64)
    try:
        eigvals, eigvecs = eigsh(
            H_op, k=k, which="SA",
            maxiter=n_lanczos_iters, tol=tolerance)
    except Exception as e:
        if verbose:
            print(f"    [hessian.topk] eigsh err: {e}", flush=True)
        return np.zeros(k), np.zeros((N, k))
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
