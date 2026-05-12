"""Renormalization-group net-length curriculum — zeus B9.

Mathematical setup
==================
The smooth proxy is U(x) = Σ_n w_n · HPWL_n(x) + λ_d Density(x) + λ_c Cong(x).
Each net contributes proportional to its HPWL, which scales with the
bbox diagonal L_n = √(Δx_n² + Δy_n²).

In Wilson's renormalization group, one integrates out the SHORT-wavelength
(high-frequency, high-momentum) modes first to obtain an effective
Hamiltonian for the LONG-wavelength modes. We do the OPPOSITE here —
optimize the short-net coupling first, then progressively turn on
long-net coupling. Why the inversion? Because the gradient structure is:

    ∂HPWL_n / ∂x_macro_i  ≈  O(1 / L_n)      (short nets give STRONGER gradients)

Short nets (L_n small) → sharp local gradients → easy to optimize.
Long nets (L_n large) → weak diffuse gradients → globally smooth.

Strategy: at step t ∈ [0, 1] of Phase 0 Adam, multiply net_weight_n by
    γ_n(t) = exp(-L_n² / (2 σ(t)² · canvas_diag²))
where σ(t) anneals from σ_0 (small) to ∞ (or large). At t=0, only nets
with L_n ≲ σ_0 · canvas_diag contribute. At t=1, all nets contribute
with their full weight.

Convergence intuition
=====================
This is HOMOTOPY OPTIMIZATION (also called CONTINUATION METHODS,
graduated optimization, mean-field annealing). The deformation
U_t(x) = U(x; γ(t)) with U_0 = "localized" approximation and U_1 = U.
If γ(t) is smooth enough and the local minima of U_t deform continuously
into those of U_{t+δt}, then tracking x*(t) gives a path to a near-global
minimum of U_1.

Theoretical guarantee (Allgower-Georg 2003 "Introduction to Numerical
Continuation Methods"): if the homotopy U_t is "regular" (no bifurcations
in t direction), the homotopy curve is a 1-manifold and tracking it
converges. For non-convex U, bifurcations CAN happen — but each bifurcation
yields a NEW basin to explore, which is itself useful.

Why this fits "RG"
==================
At small σ (early), the optimization sees a coarse-grained problem:
only LOCAL net structure matters, like an Ising model where only
nearest-neighbor interactions are active. As σ grows, longer-range
interactions are progressively "turned on". This mirrors the RG flow
in the direction of decreasing momentum cutoff: at scale σ, we have
the effective Hamiltonian retaining modes with q ≲ 1/σ.

Failure modes
-------------
- σ_0 too small: ALL nets have γ_n(0) ≈ 0 → no gradient → no descent.
  Watch for log-loss = log(small) underflow.
- σ_0 too large: γ_n(0) ≈ 1 for all n → no curriculum, equivalent to
  plain Adam.
- Schedule too fast: optimizer can't track x*(t).
- Non-monotone bboxes: net bboxes can shrink during optimization. We
  recompute γ_n(t) at each step using the CURRENT bboxes, so this is
  handled naturally — but it makes the homotopy itself path-dependent
  rather than purely time-dependent. We accept this.

Implementation
==============
This module provides a function `apply_rg_curriculum_weights` that
takes the standard net_weight tensor and produces a step-dependent
modified version. Used by adam_warm_start in the inner loop.
"""

from __future__ import annotations
import math
import numpy as np
import torch


def compute_per_net_bbox_diag(
    macro_pos: torch.Tensor,             # (n_total, 2) current placement
    pin_macro: torch.Tensor,
    pin_xoff: torch.Tensor,
    pin_yoff: torch.Tensor,
    pin_to_net: torch.Tensor,
    n_nets: int,
) -> torch.Tensor:
    """Per-net bbox diagonal in microns. Detached (no grad — for
    weight computation only).

    Returns (n_nets,) tensor.
    """
    with torch.no_grad():
        # Pin positions from macro_pos.
        is_port = (pin_macro < 0)
        safe = torch.where(is_port, torch.zeros_like(pin_macro), pin_macro)
        macro_xy = macro_pos[safe]
        pin_x = torch.where(is_port, pin_xoff,
                             macro_xy[:, 0] + pin_xoff)
        pin_y = torch.where(is_port, pin_yoff,
                             macro_xy[:, 1] + pin_yoff)
        # Per-net min/max via scatter-reduce.
        x_min = torch.full((n_nets,), float('inf'),
                            device=macro_pos.device,
                            dtype=macro_pos.dtype)
        x_max = torch.full((n_nets,), float('-inf'),
                            device=macro_pos.device,
                            dtype=macro_pos.dtype)
        y_min = x_min.clone()
        y_max = x_max.clone()
        x_min.scatter_reduce_(0, pin_to_net.to(torch.long), pin_x,
                                reduce='amin', include_self=True)
        x_max.scatter_reduce_(0, pin_to_net.to(torch.long), pin_x,
                                reduce='amax', include_self=True)
        y_min.scatter_reduce_(0, pin_to_net.to(torch.long), pin_y,
                                reduce='amin', include_self=True)
        y_max.scatter_reduce_(0, pin_to_net.to(torch.long), pin_y,
                                reduce='amax', include_self=True)
        # Replace inf (single-pin nets) with zero.
        x_diff = torch.where(torch.isfinite(x_max - x_min),
                              x_max - x_min, torch.zeros_like(x_max))
        y_diff = torch.where(torch.isfinite(y_max - y_min),
                              y_max - y_min, torch.zeros_like(y_max))
        diag = torch.sqrt(x_diff ** 2 + y_diff ** 2)
    return diag


def apply_rg_curriculum_weights(
    base_weight: torch.Tensor,           # (n_nets,) original weights
    net_bbox_diag: torch.Tensor,         # (n_nets,) current bbox diags
    sigma_now: float,                    # σ(t) in microns
) -> torch.Tensor:
    """Multiply net weights by γ_n = exp(-L_n² / (2 σ² · canvas_diag²)).

    Note: sigma_now is in raw microns (already includes canvas_diag scaling).

    Returns (n_nets,) modified weight tensor (detached gradient-wise,
    but suitable for use as a multiplier in the smooth proxy since
    weights are constants of the optimization).
    """
    with torch.no_grad():
        # γ = exp(-L²/(2σ²))
        ratio = net_bbox_diag / max(sigma_now, 1e-12)
        gamma = torch.exp(-0.5 * ratio ** 2)
    return base_weight * gamma


def schedule_sigma(
    t: float,                            # fraction of total Adam steps, in [0, 1]
    sigma_0: float,                      # σ at t=0 (fraction of canvas_diag)
    canvas_diag: float,
    sigma_inf_mult: float = 10.0,        # σ at t=1 in units of canvas_diag
) -> float:
    """Cosine schedule from σ_0·canvas_diag to σ_inf·canvas_diag.

    σ(0) = sigma_0 · canvas_diag
    σ(1) = sigma_inf · canvas_diag (so all nets ≤ canvas_diag have γ ≈ 1)

    Cosine interpolation in log space:
        log σ(t) = log σ_0 + (log σ_∞ - log σ_0) · ½(1 - cos(πt))
    """
    if t <= 0.0:
        return float(sigma_0) * float(canvas_diag)
    if t >= 1.0:
        return float(sigma_inf_mult) * float(canvas_diag)
    log_a = math.log(max(sigma_0, 1e-9))
    log_b = math.log(max(sigma_inf_mult, 1e-9))
    blend = 0.5 * (1.0 - math.cos(math.pi * t))
    log_s = log_a + (log_b - log_a) * blend
    return math.exp(log_s) * float(canvas_diag)
