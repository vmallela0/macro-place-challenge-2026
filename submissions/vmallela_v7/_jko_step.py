"""JKO / Wasserstein-2 proximal step — zeus B6.

Jordan-Kinderlehrer-Otto (1998) showed that the gradient flow of a
functional U(x) under the Wasserstein-2 metric is exactly the proximal
scheme
    x_{k+1} = argmin_x  U(x) + (1 / 2τ) · W_2²(x, x_k)
This is to gradient descent what backward-Euler is to forward-Euler:
implicit-style, more stable, allows large step sizes without blowup.

Why this exists
===============
Standard gradient descent uses Euclidean distance: ||x - x_k||² treats
all macro positions as equally "easy" to move. But moving macro A by 10
microns through a dense region of other macros is very different from
moving it through empty space.

The Wasserstein-2 metric on the empirical distribution of macros knows
this: it measures the OPTIMAL TRANSPORT cost between two placements,
not the raw L2 displacement. A small transport plan = a placement where
each macro went to its "natural" destination.

For our case (n_macros = ~100-1000 per bench), the discrete Wasserstein-2
distance between two placements (treated as empirical measures) is:
    W_2²(μ_x, μ_y) = min_σ Σ_i || x_i - y_{σ(i)} ||²
where σ ranges over permutations. This is the LINEAR ASSIGNMENT problem
— solvable exactly with Hungarian / Jonker-Volgenant in O(n³).

But we don't want to PERMUTE macros (they're distinguishable by netlist
identity!). So we use a SOFT-permutation version (Sinkhorn-Knopp): treat
macros as a uniform mass distribution, allow fractional transport. The
softness is controlled by entropic regularization ε; small ε → hard
permutation, large ε → uniform softening.

Math
====
Given two configs μ_x = (1/n) Σ δ_{x_i} and μ_y = (1/n) Σ δ_{y_j},
the entropic OT problem is:
    W_2^ε(μ_x, μ_y) = min_{π ≥ 0, π·1 = 1/n, π^T·1 = 1/n}
                       Σ_ij π_ij || x_i - y_j ||² + ε · KL(π || 1/n²·1)
Solved by Sinkhorn-Knopp iterations on the cost matrix C_ij = ||x_i - y_j||².

The JKO step computes:
    x_{k+1} = argmin_x  U(x) + (1 / 2τ) · W_2^ε(μ_x, μ_{x_k})
Gradient w.r.t. x_i (via Sinkhorn dual variables u, v):
    ∇_{x_i} W_2² = 2 Σ_j π_ij (x_i - x_k,j)
which is just "displacement to assigned center of mass". This is exactly
the gradient of a regularization that pulls each macro toward its
optimal transport image in x_k. So the JKO update becomes
    x_{k+1} ≈ x_k - τ · ∇U(x_k) + (proximal correction)
where the proximal correction is small when macros are well-aligned.

Practical compromise (this implementation)
==========================================
We do NOT solve the full JKO inner problem at every step; that's
expensive and a research project in itself. Instead we do ONE
W_2-aware proximal step:
  1. Compute current ∇U(x_k).
  2. Find tentative target  y = x_k - τ · ∇U(x_k).
  3. Solve Sinkhorn between μ_{x_k} and μ_y to get transport plan π.
  4. Update x_{k+1,i} = (1-α) · x_{k,i} + α · Σ_j π_ij · y_j  (smoothed step)

This is one Sinkhorn solve per outer step. For n_macros=200, Sinkhorn
with 100 iters ≈ 5ms.  Affordable.

Usage
-----
Drop-in alongside Adam in Phase 0 by setting
    PLACER_V7_PHASE0_OPTIMIZER=jko
"""

from __future__ import annotations
import torch


def sinkhorn_log_stabilized(
    C: torch.Tensor,              # (n, m) cost matrix
    a: torch.Tensor,              # (n,) source mass (default uniform 1/n)
    b: torch.Tensor,              # (m,) target mass
    epsilon: float = 1.0,         # entropic regularizer
    n_iters: int = 50,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Log-domain Sinkhorn for entropic OT, returns (π, u, v).

    π_ij = a_i b_j exp((u_i + v_j - C_ij) / ε)

    Log-stabilization: track log u, log v instead of u, v to avoid overflow.
    """
    n, m = C.shape
    log_a = torch.log(a + 1e-30)
    log_b = torch.log(b + 1e-30)
    log_u = torch.zeros(n, device=C.device, dtype=C.dtype)
    log_v = torch.zeros(m, device=C.device, dtype=C.dtype)
    for _ in range(int(n_iters)):
        # Update log_u: a_i = exp(log_u) · Σ_j b_j exp((log_v_j - C_ij)/ε)
        K_logs = (log_v.unsqueeze(0) - C / epsilon)             # (n, m)
        lse_row = torch.logsumexp(K_logs + log_b.unsqueeze(0), dim=1)  # (n,)
        log_u = log_a - lse_row
        K_logs = (log_u.unsqueeze(1) - C / epsilon)             # (n, m)
        lse_col = torch.logsumexp(K_logs + log_a.unsqueeze(1), dim=0)  # (m,)
        log_v = log_b - lse_col
    # Reconstruct π = a_i b_j exp((u + v - C) / ε)
    log_pi = (log_u.unsqueeze(1) + log_v.unsqueeze(0)
               + log_a.unsqueeze(1) + log_b.unsqueeze(0)
               - C / epsilon)
    pi = torch.exp(log_pi)
    return pi, log_u, log_v


def jko_proximal_step(
    macro_pos: torch.Tensor,             # (n_total, 2)
    grad_U: torch.Tensor,                # (n_total, 2) — ∇U at macro_pos
    *,
    tau: float = 1.0,                    # JKO time step
    alpha: float = 1.0,                  # outer step blend
    sinkhorn_eps: float = 1.0,           # OT entropy
    sinkhorn_iters: int = 30,
    n_hard: int = 0,
    soft_only: bool = True,
) -> tuple[torch.Tensor, dict]:
    """One JKO step.

    1. Tentative gradient step: y = x - τ · ∇U.
    2. Transport x → y via entropic OT (Sinkhorn).
    3. New position: x' = (1-α)·x + α·(Σ_j π_ij y_j).

    Returns (x_new, diag).
    """
    device = macro_pos.device
    dtype = macro_pos.dtype
    n_total = macro_pos.shape[0]
    diag = {"tau": float(tau), "alpha": float(alpha),
             "sinkhorn_eps": float(sinkhorn_eps),
             "sinkhorn_iters": int(sinkhorn_iters)}

    # Gradient step (only for soft macros if soft_only).
    if soft_only and n_hard > 0:
        grad_use = grad_U.clone()
        grad_use[:n_hard, :] = 0.0
    else:
        grad_use = grad_U
    y = macro_pos.detach() - float(tau) * grad_use.detach()

    # Restrict OT to soft macros to keep cost matrix small.
    if soft_only and n_hard > 0:
        x_soft = macro_pos.detach()[n_hard:]
        y_soft = y[n_hard:]
    else:
        x_soft = macro_pos.detach()
        y_soft = y
    n_soft = x_soft.shape[0]
    if n_soft == 0:
        return macro_pos.clone(), diag

    # Pairwise squared L2 cost.
    diff = x_soft.unsqueeze(1) - y_soft.unsqueeze(0)        # (n, n, 2)
    C = (diff ** 2).sum(dim=-1)                              # (n, n)
    diag["C_min"] = float(C.min())
    diag["C_max"] = float(C.max())

    a = torch.full((n_soft,), 1.0 / n_soft, device=device, dtype=dtype)
    b = torch.full((n_soft,), 1.0 / n_soft, device=device, dtype=dtype)
    pi, log_u, log_v = sinkhorn_log_stabilized(
        C, a, b, epsilon=float(sinkhorn_eps),
        n_iters=int(sinkhorn_iters))
    diag["pi_diag_mean"] = float(torch.diag(pi).mean())
    diag["pi_max"] = float(pi.max())

    # Each soft macro's barycentric image under the plan:
    #   y_image_i = Σ_j π_ij · y_j   /   Σ_j π_ij
    pi_row_sum = pi.sum(dim=1).clamp_min(1e-30)            # (n,)
    y_image = (pi @ y_soft) / pi_row_sum.unsqueeze(-1)     # (n, 2)

    # Final step: blend with α.
    x_new_soft = (1.0 - float(alpha)) * x_soft + float(alpha) * y_image
    x_new = macro_pos.detach().clone()
    if soft_only and n_hard > 0:
        x_new[n_hard:] = x_new_soft
    else:
        x_new = x_new_soft

    diag["disp_med_microns"] = float(
        (x_new[n_hard:] - macro_pos.detach()[n_hard:]).norm(dim=-1).median())
    return x_new, diag
