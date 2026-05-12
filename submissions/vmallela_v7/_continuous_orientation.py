"""Continuous orientation co-optimization — zeus B12.

Mathematical setup
==================
Each macro i is conventionally labeled with a discrete orientation
in {R0, R90, R180, R270, MX, MX_R90, MX_R180, MX_R270} (D₄ symmetry
group). The standard approach: optimize positions with all macros at
R0, then a sidecar tries flipping each one individually.

We replace the discrete θ_i ∈ {0, π/2, π, 3π/2} with a CONTINUOUS
parameter θ_i ∈ ℝ, jointly optimized with (x_i, y_i) under a
discretization penalty
    R(θ_i)  =  -cos(4 θ_i)
which has global minima at 4θ ≡ 0 (mod 2π) ⇔ θ ≡ 0 (mod π/2).
Maxima at θ = π/4, 3π/4, ... — these are the unstable "halfway"
points.

Pin position as function of (macro_pos, θ):
    pin_x_i  =  x_macro_i + cos(θ_i) · xoff_i - sin(θ_i) · yoff_i
    pin_y_i  =  y_macro_i + sin(θ_i) · xoff_i + cos(θ_i) · yoff_i
This is smooth in θ → HPWL gradients flow.

Why this should help here
=========================
The current orientation-flip sidecar is GREEDY: it tries flipping each
macro INDIVIDUALLY and accepts the local improvement. But macro
orientations are COUPLED — flipping macro A may make flipping macro B
beneficial that wasn't before. Greedy doesn't find this; joint
optimization does.

Continuous θ also allows gradient flow ACROSS the orientation barrier:
a macro that's "best at" θ=π/2 can move there smoothly from θ=0 under
gradient descent, without needing to "discover" the discrete option
combinatorially.

Math claims
===========
Claim 1: The discretization penalty -cos(4θ) has minima EXACTLY at
the four discrete orientations. Proof: d/dθ (-cos 4θ) = 4 sin 4θ = 0
when 4θ ≡ 0 (mod π), i.e., θ ≡ kπ/4. d²/dθ² (-cos 4θ) = 16 cos 4θ.
At θ = kπ/2: 4θ = 2kπ, cos = 1 → second deriv = 16 > 0 (min). ✓
At θ = π/4 + kπ/2: 4θ = π + 2kπ, cos = -1 → second deriv = -16 < 0 (max). ✓

Claim 2: At a discrete orientation θ* ∈ {0, π/2, π, 3π/2}, the rotated
pin positions match the standard discrete-rotation pin positions.
Proof: at θ=0, cos=1, sin=0 → pin_x = x + xoff (R0 ✓).
       at θ=π/2, cos=0, sin=1 → pin_x = x - yoff, pin_y = y + xoff (R90 ✓).
       at θ=π, cos=-1, sin=0 → pin_x = x - xoff, pin_y = y - yoff (R180 ✓).
       at θ=3π/2, cos=0, sin=-1 → pin_x = x + yoff, pin_y = y - xoff (R270 ✓).

Implementation
==============
A function `joint_xy_theta_refine(macro_pos, theta_init, U_hpwl_call,
n_steps)` that runs Adam on the joint (x, y, θ) state, with
discretization penalty annealed from low (free continuous motion) to
high (snap to discrete).

Density/cong terms are NOT affected by θ in this implementation
(would require rotated macro footprint with non-differentiable |·|).
We treat density/cong as fixed at θ=0 footprint — pessimistic but
valid: real macro footprints (square or near-square) have small
density impact from rotation.

Usage
-----
Post-pipeline refinement: takes the final placement, runs N joint
optimization steps, snaps θ to nearest discrete, returns the refined
placement and per-macro orientation choices.
"""

from __future__ import annotations
import math
import time
import numpy as np
import torch


# Discretization penalty.
def discretization_penalty(theta: torch.Tensor) -> torch.Tensor:
    """R(θ) = -cos(4θ). Minimum 0 at θ ≡ 0 mod π/2.

    Returns sum over macros.
    """
    return (-torch.cos(4.0 * theta)).sum()


def rotated_pin_positions(
    macro_pos: torch.Tensor,             # (n_total, 2)
    theta: torch.Tensor,                 # (n_total,) per-macro rotation
    pin_macro: torch.Tensor,             # (n_pins,) macro index, -1 = port
    pin_xoff: torch.Tensor,              # (n_pins,) — raw (R0) pin offsets
    pin_yoff: torch.Tensor,              # (n_pins,) — raw (R0) pin offsets
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute pin_x, pin_y given (macro_pos, theta).

    For ports (pin_macro < 0): theta has no effect, pin uses raw offset
    coords directly.
    """
    is_port = (pin_macro < 0)
    # Safe gather for ports — use index 0 then mask.
    safe = torch.where(is_port, torch.zeros_like(pin_macro), pin_macro)
    macro_xy = macro_pos[safe]
    theta_per_pin = theta[safe]
    cos_t = torch.cos(theta_per_pin)
    sin_t = torch.sin(theta_per_pin)
    pin_x_rot = macro_xy[:, 0] + cos_t * pin_xoff - sin_t * pin_yoff
    pin_y_rot = macro_xy[:, 1] + sin_t * pin_xoff + cos_t * pin_yoff
    pin_x = torch.where(is_port, pin_xoff, pin_x_rot)
    pin_y = torch.where(is_port, pin_yoff, pin_y_rot)
    return pin_x, pin_y


def joint_xy_theta_refine(
    macro_pos: torch.Tensor,             # (n_total, 2) initial
    pin_macro: torch.Tensor,
    pin_xoff: torch.Tensor,
    pin_yoff: torch.Tensor,
    pin_to_net: torch.Tensor,
    net_weight: torch.Tensor,
    n_nets: int,
    *,
    cw: float, ch: float,
    net_cnt: float,
    tau_lse: float = 50.0,
    n_steps: int = 60,
    lr_xy_frac: float = 0.01,            # in fractions of canvas_diag
    lr_theta: float = 0.05,
    disc_weight_init: float = 0.001,
    disc_weight_final: float = 1.0,
    n_hard: int = 0,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Joint Adam on (x, y, θ) minimizing LSE-HPWL + α(t)·R(θ).

    Returns (final_pos, final_theta_discrete_idx, diag).

    final_theta_discrete_idx[i] ∈ {0, 1, 2, 3} is the snapped orientation
    for macro i (0=R0, 1=R90, 2=R180, 3=R270).

    No density / congestion terms — those are macro-position-dominant.
    HPWL is the term most sensitive to pin location, hence orientation.
    """
    from _smooth_proxy import lse_hpwl_vectorized
    device = macro_pos.device
    dtype = macro_pos.dtype
    n_total = macro_pos.shape[0]
    canvas_diag = math.hypot(float(cw), float(ch))

    # Soft macros get free θ; hard macros stay at θ=0.
    theta = torch.zeros(n_total, dtype=dtype, device=device, requires_grad=True)
    pos = macro_pos.clone().detach().requires_grad_(True)
    lr_xy = lr_xy_frac * canvas_diag

    opt = torch.optim.Adam([
        {"params": [pos], "lr": lr_xy},
        {"params": [theta], "lr": float(lr_theta)},
    ])

    history = {"hpwl": [], "disc": [], "max_theta_dev": []}
    t0 = time.time()

    for step in range(int(n_steps)):
        opt.zero_grad()
        pin_x, pin_y = rotated_pin_positions(
            pos, theta, pin_macro, pin_xoff, pin_yoff)
        hpwl = lse_hpwl_vectorized(
            pin_x, pin_y, pin_to_net, net_weight, n_nets,
            cw=float(cw), ch=float(ch), net_cnt=float(net_cnt),
            tau_lse=tau_lse)
        # Anneal discretization weight from low to high (cosine).
        t_frac = step / max(n_steps - 1, 1)
        alpha = disc_weight_init + (disc_weight_final - disc_weight_init) \
                  * 0.5 * (1.0 - math.cos(math.pi * t_frac))
        disc = discretization_penalty(theta)
        loss = hpwl + alpha * disc
        loss.backward()
        # Mask grads: hard macros stay still in both pos and theta.
        if n_hard > 0:
            with torch.no_grad():
                if pos.grad is not None:
                    pos.grad[:n_hard] = 0.0
                if theta.grad is not None:
                    theta.grad[:n_hard] = 0.0
        opt.step()
        # Optional: clip pos into canvas after step.
        with torch.no_grad():
            pos.data[:, 0].clamp_(0.0, float(cw))
            pos.data[:, 1].clamp_(0.0, float(ch))
        history["hpwl"].append(float(hpwl.item()))
        history["disc"].append(float(disc.item()))
        with torch.no_grad():
            # Maximum residual from nearest discrete orientation.
            quantized = (theta / (math.pi / 2.0)).round() * (math.pi / 2.0)
            max_dev = float(((theta - quantized).abs() % (math.pi / 2.0)).max())
            history["max_theta_dev"].append(max_dev)
        if verbose and (step == 0 or step == n_steps - 1
                          or (step + 1) % 20 == 0):
            print(f"    [orient] step {step+1}/{n_steps} hpwl={hpwl:.4f} "
                  f"disc={disc:.4f} α={alpha:.3f} max|θ-θ*|={max_dev:.4f}",
                  flush=True)

    # Snap θ to nearest discrete.
    with torch.no_grad():
        theta_snapped = (theta / (math.pi / 2.0)).round().long()
        # idx 0,1,2,3 mapping to R0,R90,R180,R270 — mod 4.
        theta_snapped = theta_snapped % 4
        if n_hard > 0:
            theta_snapped[:n_hard] = 0    # keep hard macros at R0
        # Re-compute pin positions at the snapped θ.
        theta_snapped_rad = theta_snapped.to(dtype) * (math.pi / 2.0)
        pin_x_snap, pin_y_snap = rotated_pin_positions(
            pos, theta_snapped_rad, pin_macro, pin_xoff, pin_yoff)
        # Sanity-eval HPWL at snapped.
        hpwl_snapped = lse_hpwl_vectorized(
            pin_x_snap, pin_y_snap, pin_to_net, net_weight, n_nets,
            cw=float(cw), ch=float(ch), net_cnt=float(net_cnt),
            tau_lse=tau_lse).item()

    diag = {
        "method": "joint_xy_theta_refine",
        "n_steps": int(n_steps),
        "final_hpwl_continuous": history["hpwl"][-1],
        "final_hpwl_snapped": float(hpwl_snapped),
        "final_max_theta_dev": history["max_theta_dev"][-1],
        "history": history,
        "wall_s": time.time() - t0,
    }
    if verbose:
        print(f"    [orient] continuous→snapped HPWL: "
              f"{history['hpwl'][-1]:.4f} → {hpwl_snapped:.4f} "
              f"(Δ={hpwl_snapped - history['hpwl'][-1]:+.4f})", flush=True)
    return (pos.detach().cpu().numpy().astype(np.float64),
             theta_snapped.cpu().numpy().astype(np.int32),
             diag)
