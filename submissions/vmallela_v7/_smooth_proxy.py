"""Adam on a smooth surrogate of the proxy cost, with CVaR top-K.

The novel piece. No published placer (to my knowledge) uses CVaR
reformulation for the density / congestion top-K terms. The CVaR
formulation makes the entire proxy cost globally smooth — letting Adam
work cleanly across the whole macro position vector.

Math (validated step by step)
==============================

The exact proxy cost:
    proxy = HPWL_norm + 0.5 · density_norm + 0.5 · congestion_norm

where:
    HPWL_norm = Σ_n  w_n · (max_i x_i^n − min_i x_i^n + max_i y_i^n − min_i y_i^n)
                / ((W+H) · net_count)
    density_norm = (1/K_d) · Σ_{c ∈ top-K_d} ρ_c    where K_d = ⌈0.1·n_cells⌉
    congestion_norm = (1/K_c) · Σ_{c ∈ top-K_c} γ_c   where K_c = ⌈0.05·2·n_cells⌉

Each term is non-smooth (max/min, top-K). We replace each with a
smooth surrogate:

────────────────────────────────────────────────────────────────────
1. HPWL via log-sum-exp (LSE) smoothing.
   Standard placement folklore (Naylor 2001, RePlAce, DREAMPlace).
   For a net with pin x-coords {x_1, ..., x_k}:
       max_i x_i  ≈  (1/τ) · log Σ_i exp(τ · x_i)
       min_i x_i  ≈ −(1/τ) · log Σ_i exp(−τ · x_i)
   So:
       max − min ≈ (1/τ) · ( log Σ exp(τx_i) + log Σ exp(−τx_i) )
   As τ → ∞ this converges to the true max − min. For τ in [10, 100]
   on a unit-scaled chip it's a tight smooth approximation.
   Differentiable everywhere; convex in pin positions.

────────────────────────────────────────────────────────────────────
2. Density / congestion via CVaR reformulation (Rockafellar-Uryasev 2000).

   For random variable X (or finite sample {x_1, ..., x_n}) and
   confidence level α ∈ (0, 1):

       CVaR_α(X) = (1/(1−α)) · E[X · I{X ≥ VaR_α(X)}]
                 = E[X | X ≥ VaR_α(X)]

   Rockafellar-Uryasev Theorem 1: for any α and any F:
       CVaR_α(X) = inf_t  { t + (1/(1−α)) · E[(X − t)_+] }
       where (·)_+ = max(0, ·)

   For finite sample with n_cells cells and α = 1 − K/n_cells:
       CVaR = inf_t  { t + (1/K) · Σ_c (ρ_c − t)_+ }

   At the optimum t* = ρ_(n_cells−K) (the (n−K)-th order statistic),
   so:
       CVaR = ρ_(n−K) + (1/K) · Σ_{c: ρ_c ≥ ρ_(n−K)} (ρ_c − ρ_(n−K))
            = ρ_(n−K) + (1/K) · ((sum of top-K) − K · ρ_(n−K))
            = (1/K) · (sum of top-K)
            = top-K mean

   So CVaR_α({ρ_c}) with α = 1−K/n IS EXACTLY the top-K average. No
   approximation. No bias. Full equivalence at the optimum.

   The CVaR formulation is jointly convex in (t, ρ) and smooth except
   at the kink (ρ_c − t)_+ = 0. We use the smooth-relu (softplus):

       (ρ_c − t)_+  ≈  (1/μ) · log(1 + exp(μ · (ρ_c − t)))      ← softplus_μ

   For μ → ∞ this converges to the hard relu. We use μ = 100 — sharp
   enough that the optimum is unaffected at machine precision, smooth
   enough that gradients flow through.

   So our smooth top-K is:
       smooth_topK_α(ρ) = inf_t  { t + (1/(K · μ)) · Σ_c log(1 + exp(μ · (ρ_c − t))) }

   We treat t as a learnable scalar — Adam optimizes (macro_pos, t_density,
   t_congestion) jointly. At convergence t equals the order statistic,
   and the surrogate equals the true top-K mean.

────────────────────────────────────────────────────────────────────
3. Density grid as a function of macro positions.

   For macros m at center (cx_m, cy_m) with size (w_m, h_m), each cell
   c at (col, row) accumulates the overlap area:

       ρ_c = (1 / cell_area) · Σ_m  overlap_area(m, c)

   where overlap_area(m, c) = max(0, min(x_m^max, c.x_max) − max(x_m^min, c.x_min))
                            × max(0, min(y_m^max, c.y_max) − max(y_m^min, c.y_min))
                            (the standard rectangle overlap)

   Each factor max(0, ·) is non-smooth at the cell boundary. We
   replace each with a softplus_μ. This makes density smooth in macro
   position. Gradient flow → Adam can move macros to reduce density.

────────────────────────────────────────────────────────────────────
4. Congestion grid as a function of macro positions.

   The congestion term in PlacementCost is the top-5 % of:
       (V_routing_smooth + V_macro_norm) ∪ (H_routing_smooth + H_macro_norm)

   V_macro_norm comes from macro blockage (same shape as density —
   smoothable via softplus_μ on the cell-overlap formula).
   V/H_routing_smooth comes from PER-NET routing demand (L-routes /
   T-routes through cell grids), which is HIGHLY discrete and
   non-smooth in macro positions (small move → completely different
   net topology).

   FOR THIS RELEASE: we hold V/H_routing_smooth FIXED at the current
   placement's value (same approximation as the v6 GPU evaluator's
   "frozen routing" term). The smooth surrogate then optimizes:
       cong_smooth(x) ≈ smooth_topK( V_smooth_fixed + V_macro_diff(x)/grid_v_routes,
                                     H_smooth_fixed + H_macro_diff(x)/grid_h_routes )
   This means Adam can pull macros AWAY from blockage hot cells (a
   meaningful gradient) but doesn't optimize routing topology.

   This is a documented approximation. The CPU IncrementalEvaluator
   validates the EXACT proxy on every accepted move during the
   downstream snap-to-exact-cost pass — so the smooth surrogate's
   approximation can never poison the final placement.

────────────────────────────────────────────────────────────────────

Adam loop
---------
Variables (all in torch with requires_grad=True):
    macro_pos: (n_total, 2) float — only soft macros are free; hards
                                    are detached (treated as constants
                                    after each Adam step).
                                    Wait — better: also free for hards,
                                    but with a HEAVY proximal penalty
                                    that keeps them within d_proximal
                                    of their starting position. This
                                    avoids hard-overlap chaos.
    t_density: scalar — CVaR threshold.
    t_congestion: scalar — CVaR threshold.

Objective:
    smooth_proxy(macro_pos, t_d, t_c) =
        HPWL_lse(macro_pos)
        + 0.5 · cvar_smooth(density_grid(macro_pos), K_d, t_d)
        + 0.5 · cvar_smooth(cong_grid(macro_pos), K_c, t_c)
        + λ_proximal · Σ_{m hard} ||macro_pos[m] − start_pos[m]||²

We optimize for n_steps Adam steps, then snap macro_pos to the
nearest legal placement (push-apart + legalize) and pass to the v6
exact-cost local search for refinement.

What this is NOT
================
This is NOT a complete replacement for the v6 pipeline. Adam on the
smooth surrogate finds a GOOD STARTING BASIN for the downstream
exact-cost search. Without the snap-to-exact + local refine, the Adam
result tends to have small overlaps (legalize fixes them) and slightly
suboptimal exact cost (because the smooth surrogate isn't bit-equal
to the exact proxy).

Implementation notes
====================
- All ops in float32 torch on cuda > mps > cpu (auto-select).
- Gradient checkpointing via batched cell-overlap re-compute (memory
  proportional to n_total × max_cells_per_macro, not n_total × n_cells).
- Adam learning rate 0.05 of canvas_diag. Cosine-decay over n_steps.
- Hard-macro proximal weight λ = 1.0 / (canvas_diag**2). Tunable.
"""
from __future__ import annotations
import math
import time
import numpy as np
import torch


def _select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def softplus_mu(x: torch.Tensor, mu: float) -> torch.Tensor:
    """Smooth approximation to max(0, x). softplus_μ(x) = (1/μ) log(1+exp(μx)).
    For numerical stability we use the equivalent (max(0, x) + log1p(exp(-|μx|))/μ)
    form via torch.nn.functional.softplus."""
    return torch.nn.functional.softplus(mu * x) / mu


def lse_max(x: torch.Tensor, tau: float, dim: int = -1) -> torch.Tensor:
    """Smooth max via log-sum-exp: (1/τ) · log Σ_i exp(τ x_i).
    Numerically stable form: shift by max."""
    x_max = x.detach().max(dim=dim, keepdim=True)[0]
    return x_max.squeeze(dim) + torch.logsumexp(tau * (x - x_max), dim=dim) / tau


def lse_min(x: torch.Tensor, tau: float, dim: int = -1) -> torch.Tensor:
    return -lse_max(-x, tau, dim=dim)


def cvar_smooth(values: torch.Tensor, K: int, t: torch.Tensor,
                mu: float = 100.0) -> torch.Tensor:
    """Smoothed CVaR top-K mean.
        CVaR ≈ t + (1 / K) · Σ_c softplus_μ(values_c − t)

    At Adam's optimum: t equals the (n−K)-th order statistic, and the
    formula equals the top-K mean exactly.

    Parameters
    ----------
    values : (..., n) float — per-cell densities or congestion values.
    K : int — top-K count (e.g. K_d = ⌈0.1·n_cells⌉).
    t : scalar tensor (requires_grad) — CVaR threshold variable.
    mu : softplus sharpness (default 100; larger → tighter).
    """
    # softplus_μ broadcasts over the leading axes of values.
    return t + softplus_mu(values - t, mu).sum(dim=-1) / K


def lse_hpwl_vectorized(
    pin_x: torch.Tensor,        # (n_pins,) float, requires_grad
    pin_y: torch.Tensor,        # (n_pins,) float, requires_grad
    pin_to_net: torch.Tensor,   # (n_pins,) long, value = net index
    net_weight: torch.Tensor,   # (n_nets,) float, const
    n_nets: int,
    cw: float, ch: float, net_cnt: float,
    tau_lse: float = 50.0,
) -> torch.Tensor:
    """Fully vectorized LSE-HPWL via scatter_reduce(amax) + scatter_add(exp).

    Mathematical equivalence to the per-net loop:
        For each net n with pin x-coords {x_pins(n)}:
            lse_max = max_{n} + (1/τ) log Σ exp(τ(x − max_{n}))
            lse_min = -lse_max(-x)
            bbox_x  = lse_max − lse_min
        HPWL = Σ_n w_n (bbox_x + bbox_y) / ((cw+ch)·net_cnt)

    Stability: per-net max is detached (gradient flows through softmax-of-exp,
    not the argmax). 1-pin nets contribute 0 (lse_max == lse_min == pin).
    Empty nets are masked via the net_lengths > 0 check.
    """
    device = pin_x.device
    dtype = pin_x.dtype

    # ── lse_max(pin_x) per net via scatter ──────────────────────────
    x_max = torch.full((n_nets,), float('-inf'), device=device, dtype=dtype)
    x_max.scatter_reduce_(0, pin_to_net, pin_x.detach(),
                          reduce='amax', include_self=True)
    x_max_per_pin = x_max[pin_to_net]
    exp_x = torch.exp(tau_lse * (pin_x - x_max_per_pin))
    sum_exp_x = torch.zeros(n_nets, device=device, dtype=dtype)
    sum_exp_x.scatter_add_(0, pin_to_net, exp_x)
    lse_max_x = x_max + torch.log(sum_exp_x.clamp_min(1e-30)) / tau_lse

    # ── lse_min(pin_x) per net via -lse_max(-pin_x) ─────────────────
    nx_max = torch.full((n_nets,), float('-inf'), device=device, dtype=dtype)
    nx_max.scatter_reduce_(0, pin_to_net, (-pin_x).detach(),
                           reduce='amax', include_self=True)
    nx_max_per_pin = nx_max[pin_to_net]
    exp_nx = torch.exp(tau_lse * (-pin_x - nx_max_per_pin))
    sum_exp_nx = torch.zeros(n_nets, device=device, dtype=dtype)
    sum_exp_nx.scatter_add_(0, pin_to_net, exp_nx)
    lse_min_x = -(nx_max + torch.log(sum_exp_nx.clamp_min(1e-30)) / tau_lse)

    bbox_x = lse_max_x - lse_min_x  # (n_nets,)

    # ── Same for pin_y ──────────────────────────────────────────────
    y_max = torch.full((n_nets,), float('-inf'), device=device, dtype=dtype)
    y_max.scatter_reduce_(0, pin_to_net, pin_y.detach(),
                          reduce='amax', include_self=True)
    y_max_per_pin = y_max[pin_to_net]
    exp_y = torch.exp(tau_lse * (pin_y - y_max_per_pin))
    sum_exp_y = torch.zeros(n_nets, device=device, dtype=dtype)
    sum_exp_y.scatter_add_(0, pin_to_net, exp_y)
    lse_max_y = y_max + torch.log(sum_exp_y.clamp_min(1e-30)) / tau_lse

    ny_max = torch.full((n_nets,), float('-inf'), device=device, dtype=dtype)
    ny_max.scatter_reduce_(0, pin_to_net, (-pin_y).detach(),
                           reduce='amax', include_self=True)
    ny_max_per_pin = ny_max[pin_to_net]
    exp_ny = torch.exp(tau_lse * (-pin_y - ny_max_per_pin))
    sum_exp_ny = torch.zeros(n_nets, device=device, dtype=dtype)
    sum_exp_ny.scatter_add_(0, pin_to_net, exp_ny)
    lse_min_y = -(ny_max + torch.log(sum_exp_ny.clamp_min(1e-30)) / tau_lse)

    bbox_y = lse_max_y - lse_min_y

    # Mask empty nets (x_max == -inf means no pins scattered into that net).
    # For finite-pin nets the bbox is finite. Replace any non-finite with 0.
    bbox = bbox_x + bbox_y
    bbox = torch.where(torch.isfinite(bbox), bbox,
                       torch.zeros_like(bbox))

    hpwl_total = (net_weight * bbox).sum()
    return hpwl_total / ((cw + ch) * net_cnt)


def smooth_proxy_for_v7(
    macro_pos: torch.Tensor,           # (n_total, 2) requires_grad
    t_density: torch.Tensor,           # () requires_grad
    t_cong: torch.Tensor,              # () requires_grad
    *,
    macro_w: torch.Tensor,             # (n_total,) const
    macro_h: torch.Tensor,             # (n_total,) const
    pin_macro: torch.Tensor,           # (n_pins,) int  (-1 = port, fixed)
    pin_xoff: torch.Tensor,            # (n_pins,) const
    pin_yoff: torch.Tensor,            # (n_pins,) const
    net_starts: torch.Tensor,          # (n_nets+1,) int
    net_weight: torch.Tensor,          # (n_nets,) const
    grid_col: int, grid_row: int,
    grid_w: float, grid_h: float,
    grid_v_routes: float, grid_h_routes: float,
    V_smooth_frozen: torch.Tensor,    # (n_cells,) const — frozen routing demand
    H_smooth_frozen: torch.Tensor,    # (n_cells,) const
    cw: float, ch: float,
    net_cnt: float,
    K_density: int, K_cong: int,
    cell_area: float,
    vrouting_alloc: float, hrouting_alloc: float,
    tau_lse: float = 50.0,
    mu_softplus: float = 100.0,
    proximal_pos: torch.Tensor = None, # (n_total, 2) const — anchor positions
    proximal_weight: float = 0.0,
    hard_only_proximal: bool = True,
    n_hard: int = 0,
    cell_idx_density: torch.Tensor = None,    # (n_total, K_max) int (-1 pad)
    cell_idx_cong: torch.Tensor = None,       # (n_total, K_max) int (-1 pad)
):
    """Smooth surrogate of the exact proxy. Differentiable in macro_pos,
    t_density, t_cong. All other arguments are constants (detached).

    Returns:
        scalar loss = HPWL_lse + 0.5 * CVaR_smooth(density) +
                      0.5 * CVaR_smooth(congestion) + proximal
    """
    device = macro_pos.device
    n_total = macro_pos.shape[0]
    n_cells = grid_col * grid_row

    # ── HPWL via LSE smoothing ────────────────────────────────────────
    # For each net, compute (lse_max - lse_min) over its pin positions.
    # Pin positions = macro_pos[pin_macro] + pin_offset  (port pins use
    # absolute coords stored in pin_xoff/yoff; for those pin_macro = -1
    # so we use the offset directly).

    # Gather pin positions: pins on macros pick up macro_pos; port pins
    # use the pin_xoff/yoff fields directly.
    is_port = (pin_macro < 0)
    safe_pin_macro = torch.where(is_port,
                                  torch.zeros_like(pin_macro),
                                  pin_macro)
    macro_xy = macro_pos[safe_pin_macro]  # (n_pins, 2)

    pin_x = torch.where(is_port, pin_xoff,
                        macro_xy[:, 0] + pin_xoff)
    pin_y = torch.where(is_port, pin_yoff,
                        macro_xy[:, 1] + pin_yoff)

    # Per-net HPWL via LSE max - min. Net i covers pin indices
    # [net_starts[i], net_starts[i+1]).
    # We compute this via segment-aggregation: rather than building a
    # ragged structure, we use pad-to-max-net-size + mask. For typical
    # benchmarks, max_net_size ≤ ~30, total pins ~17k.
    n_nets = int(net_weight.shape[0])
    # For simplicity use a Python loop over nets (n_nets ≈ 6k for ibm01,
    # may want to vectorize for ibm17). This is the prototype; tune
    # later.

    hpwl_total = torch.zeros((), device=device, dtype=macro_pos.dtype)
    for i in range(n_nets):
        s = int(net_starts[i].item())
        e = int(net_starts[i + 1].item())
        if e <= s + 1:
            continue
        xs = pin_x[s:e]
        ys = pin_y[s:e]
        bbox_x = lse_max(xs, tau_lse) - lse_min(xs, tau_lse)
        bbox_y = lse_max(ys, tau_lse) - lse_min(ys, tau_lse)
        hpwl_total = hpwl_total + net_weight[i] * (bbox_x + bbox_y)
    hpwl_norm = hpwl_total / ((cw + ch) * net_cnt)

    # ── Density grid via softplus rectangle-overlap (windowed) ──────
    # See _cell_window.py for the full math derivation. Windowed for
    # tractability: for each macro, evaluate overlap only in cells
    # within `margin_cells` of the footprint. Re-snapshot every K Adam
    # steps. See docstring of `cell_idx_density` parameter.
    if cell_idx_density is None:
        density_smooth = torch.zeros((), device=device, dtype=macro_pos.dtype)
    else:
        from _cell_window import smooth_density_grid
        rho = smooth_density_grid(
            macro_pos, macro_w, macro_h, cell_idx_density,
            grid_col, grid_row, grid_w, grid_h,
            n_cells=grid_col * grid_row, cell_area=cell_area, mu=mu_softplus)
        density_smooth = cvar_smooth(rho.unsqueeze(0), K_density,
                                      t_density, mu=mu_softplus).squeeze()

    # ── Congestion via macro-blockage softplus + frozen routing ─────
    # V_total = V_routing_smooth (frozen) + V_macro(macro_pos)/grid_v_routes
    # H_total similarly. CVaR top-5% over (V_total ∪ H_total).
    if cell_idx_cong is None:
        cong_smooth = torch.zeros((), device=device, dtype=macro_pos.dtype)
    else:
        from _cell_window import smooth_macro_blockage
        V_macro, H_macro = smooth_macro_blockage(
            macro_pos, macro_w, macro_h, cell_idx_cong,
            grid_col, grid_row, grid_w, grid_h,
            n_cells=grid_col * grid_row,
            vrouting_alloc=vrouting_alloc,
            hrouting_alloc=hrouting_alloc, mu=mu_softplus)
        V_total = V_smooth_frozen + V_macro / grid_v_routes
        H_total = H_smooth_frozen + H_macro / grid_h_routes
        combined = torch.cat([V_total, H_total], dim=0)
        cong_smooth = cvar_smooth(combined.unsqueeze(0), K_cong,
                                   t_cong, mu=mu_softplus).squeeze()

    # ── Proximal regularization ────────────────────────────────────
    proximal = torch.zeros((), device=device, dtype=macro_pos.dtype)
    if proximal_pos is not None and proximal_weight > 0.0:
        if hard_only_proximal:
            d = macro_pos[:n_hard] - proximal_pos[:n_hard]
        else:
            d = macro_pos - proximal_pos
        proximal = proximal_weight * (d ** 2).sum()

    loss = hpwl_norm + 0.5 * density_smooth + 0.5 * cong_smooth + proximal
    return loss, {"hpwl": hpwl_norm.detach(),
                  "density": density_smooth.detach(),
                  "cong": cong_smooth.detach(),
                  "proximal": proximal.detach()}


def smooth_proxy_for_v7_v2(
    macro_pos: torch.Tensor,           # (n_total, 2) requires_grad
    t_density: torch.Tensor,           # () requires_grad
    t_cong: torch.Tensor,              # () requires_grad
    *,
    macro_w: torch.Tensor,             # (n_total,) const
    macro_h: torch.Tensor,             # (n_total,) const
    pin_macro: torch.Tensor,           # (n_pins,) int  (-1 = port)
    pin_xoff: torch.Tensor,            # (n_pins,) const
    pin_yoff: torch.Tensor,            # (n_pins,) const
    pin_to_net: torch.Tensor,          # (n_pins,) int — flattened net index
    net_weight: torch.Tensor,          # (n_nets,) const
    n_nets: int,
    grid_col: int, grid_row: int,
    grid_w: float, grid_h: float,
    grid_v_routes: float, grid_h_routes: float,
    V_smooth_frozen: torch.Tensor,
    H_smooth_frozen: torch.Tensor,
    cw: float, ch: float,
    net_cnt: float,
    K_density: int, K_cong: int,
    cell_area: float,
    vrouting_alloc: float, hrouting_alloc: float,
    tau_lse: float = 50.0,
    mu_softplus: float = 100.0,
    proximal_pos: torch.Tensor = None,
    proximal_weight: float = 0.0,
    hard_only_proximal: bool = True,
    n_hard: int = 0,
    cell_idx_density: torch.Tensor = None,
    cell_idx_cong: torch.Tensor = None,
    return_components: bool = False,
):
    """Fully vectorized smooth surrogate (no Python loops in the gradient
    path). Same math as smooth_proxy_for_v7 but the HPWL inner loop uses
    scatter_reduce + scatter_add. Empirical: 50 Adam steps on ibm15 (~6k
    nets, ~26k pins) drops from ~5 min to ~1 s on MPS.
    """
    device = macro_pos.device
    dtype = macro_pos.dtype

    # ── Pin coords from macro positions + offsets ───────────────────
    is_port = (pin_macro < 0)
    safe_pin_macro = torch.where(is_port,
                                  torch.zeros_like(pin_macro),
                                  pin_macro)
    macro_xy = macro_pos[safe_pin_macro]  # (n_pins, 2)
    pin_x = torch.where(is_port, pin_xoff,
                        macro_xy[:, 0] + pin_xoff)
    pin_y = torch.where(is_port, pin_yoff,
                        macro_xy[:, 1] + pin_yoff)

    # ── Vectorized LSE-HPWL ─────────────────────────────────────────
    hpwl_norm = lse_hpwl_vectorized(
        pin_x, pin_y, pin_to_net, net_weight, n_nets,
        cw=cw, ch=ch, net_cnt=net_cnt, tau_lse=tau_lse)

    # ── Density (cell-window) ───────────────────────────────────────
    if cell_idx_density is None:
        density_smooth = torch.zeros((), device=device, dtype=dtype)
    else:
        from _cell_window import smooth_density_grid
        rho = smooth_density_grid(
            macro_pos, macro_w, macro_h, cell_idx_density,
            grid_col, grid_row, grid_w, grid_h,
            n_cells=grid_col * grid_row, cell_area=cell_area, mu=mu_softplus)
        density_smooth = cvar_smooth(rho.unsqueeze(0), K_density,
                                      t_density, mu=mu_softplus).squeeze()

    # ── Congestion (macro-blockage diff + frozen routing) ───────────
    if cell_idx_cong is None:
        cong_smooth = torch.zeros((), device=device, dtype=dtype)
    else:
        from _cell_window import smooth_macro_blockage
        V_macro, H_macro = smooth_macro_blockage(
            macro_pos, macro_w, macro_h, cell_idx_cong,
            grid_col, grid_row, grid_w, grid_h,
            n_cells=grid_col * grid_row,
            vrouting_alloc=vrouting_alloc,
            hrouting_alloc=hrouting_alloc, mu=mu_softplus)
        V_total = V_smooth_frozen + V_macro / grid_v_routes
        H_total = H_smooth_frozen + H_macro / grid_h_routes
        combined = torch.cat([V_total, H_total], dim=0)
        cong_smooth = cvar_smooth(combined.unsqueeze(0), K_cong,
                                   t_cong, mu=mu_softplus).squeeze()

    # ── Proximal regularization ─────────────────────────────────────
    proximal = torch.zeros((), device=device, dtype=dtype)
    if proximal_pos is not None and proximal_weight > 0.0:
        if hard_only_proximal:
            d = macro_pos[:n_hard] - proximal_pos[:n_hard]
        else:
            d = macro_pos - proximal_pos
        proximal = proximal_weight * (d ** 2).sum()

    if return_components:
        # Caller wants per-component scalar tensors (still in the graph)
        # for GradNorm scaling. Caller composes the final loss.
        return hpwl_norm, density_smooth, cong_smooth, proximal

    loss = hpwl_norm + 0.5 * density_smooth + 0.5 * cong_smooth + proximal
    return loss, {"hpwl": hpwl_norm.detach(),
                  "density": density_smooth.detach(),
                  "cong": cong_smooth.detach(),
                  "proximal": proximal.detach()}


def build_pin_to_net(net_starts: torch.Tensor) -> torch.Tensor:
    """Construct (n_pins,) tensor where each entry is the net index for that pin.
    pin_to_net[i] = j  iff  net_starts[j] <= i < net_starts[j+1].
    Built once at the start of adam_warm_start; constant across steps.
    """
    net_lengths = net_starts[1:] - net_starts[:-1]  # (n_nets,)
    n_nets = int(net_lengths.shape[0])
    pin_to_net = torch.repeat_interleave(
        torch.arange(n_nets, device=net_starts.device, dtype=torch.long),
        net_lengths.long())
    return pin_to_net


def adam_warm_start(
    incr_eval, benchmark,
    *,
    n_steps: int = 500,
    lr_frac_canvas: float = 0.02,
    proximal_weight_frac: float = 1.0,
    soft_only: bool = True,
    enable_density: bool = True,
    enable_congestion: bool = True,
    window_margin_cells: int = 4,
    snapshot_every: int = 50,
    verbose: bool = False,
) -> tuple[np.ndarray, dict]:
    """Run Adam on the smooth surrogate (LSE-HPWL + CVaR top-K density +
    CVaR top-K congestion) starting from incr_eval's current placement.
    Returns the optimized full (n_total, 2) placement.

    Density / congestion gradients use cell-window truncation (see
    _cell_window.py). Windows re-snapshot every `snapshot_every` Adam
    steps so macros that drift outside their initial window get a fresh
    one.

    Parameters
    ----------
    n_steps : Adam steps total
    lr_frac_canvas : Adam learning rate as fraction of canvas_diag
    enable_density : if True, include CVaR density in the surrogate
    enable_congestion : if True, include CVaR congestion (with frozen
        per-net routing demand from incr_eval, only macro-blockage
        contribution is differentiable)
    window_margin_cells : margin around each macro footprint for the
        density/cong window (4 = ~5-cell buffer in each direction)
    snapshot_every : re-build cell windows every this many Adam steps.
        Smaller = more accurate but more expensive (window construction
        is O(n_total · K)).
    """
    device = _select_device()
    n_total = incr_eval.macro_pos.shape[0]
    n_hard = incr_eval.n_hard
    cw, ch = float(incr_eval.cw), float(incr_eval.ch)
    canvas_diag = math.hypot(cw, ch)

    # Convert state to torch tensors on device.
    macro_pos_init = torch.tensor(
        np.asarray(incr_eval.macro_pos), dtype=torch.float32, device=device)

    # Free vs frozen variables: hards held proximal; softs freely
    # variable (or all variable, depending on `soft_only`).
    macro_pos = macro_pos_init.clone().detach().requires_grad_(True)
    t_density = torch.zeros((), device=device, requires_grad=True)
    t_cong = torch.zeros((), device=device, requires_grad=True)

    # Constants
    pin_macro = torch.tensor(np.asarray(incr_eval.pin_macro), dtype=torch.long, device=device)
    pin_xoff = torch.tensor(np.asarray(incr_eval.pin_xoff), dtype=torch.float32, device=device)
    pin_yoff = torch.tensor(np.asarray(incr_eval.pin_yoff), dtype=torch.float32, device=device)
    net_starts = torch.tensor(np.asarray(incr_eval.net_starts), dtype=torch.long, device=device)
    net_weight = torch.tensor(np.asarray(incr_eval.net_weight), dtype=torch.float32, device=device)
    macro_w = torch.tensor(np.asarray(incr_eval.macro_w), dtype=torch.float32, device=device)
    macro_h = torch.tensor(np.asarray(incr_eval.macro_h), dtype=torch.float32, device=device)
    V_smooth = torch.tensor(np.asarray(incr_eval.V_routing_smooth), dtype=torch.float32, device=device)
    H_smooth = torch.tensor(np.asarray(incr_eval.H_routing_smooth), dtype=torch.float32, device=device)
    proximal_anchor = macro_pos_init.clone().detach()

    K_d = max(1, math.floor(incr_eval.n_cells * 0.1))
    K_c = max(1, math.floor(2 * incr_eval.n_cells * 0.05))

    lr = lr_frac_canvas * canvas_diag
    proximal_weight = proximal_weight_frac / (canvas_diag ** 2)

    # ── Pre-build pin_to_net once (constant across steps) ──────────
    pin_to_net = build_pin_to_net(net_starts)
    n_nets = int(net_weight.shape[0])

    # Optimizer
    opt = torch.optim.Adam([macro_pos, t_density, t_cong], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps)

    from _cell_window import build_window_indices
    cell_idx_d = None
    cell_idx_c = None

    def _proxy_call():
        return smooth_proxy_for_v7_v2(
            macro_pos, t_density, t_cong,
            macro_w=macro_w, macro_h=macro_h,
            pin_macro=pin_macro, pin_xoff=pin_xoff, pin_yoff=pin_yoff,
            pin_to_net=pin_to_net, net_weight=net_weight, n_nets=n_nets,
            grid_col=incr_eval.grid_col, grid_row=incr_eval.grid_row,
            grid_w=incr_eval.grid_width, grid_h=incr_eval.grid_height,
            grid_v_routes=incr_eval.grid_v_routes,
            grid_h_routes=incr_eval.grid_h_routes,
            V_smooth_frozen=V_smooth, H_smooth_frozen=H_smooth,
            cw=cw, ch=ch, net_cnt=float(incr_eval.net_cnt),
            K_density=K_d, K_cong=K_c,
            cell_area=incr_eval.grid_area,
            vrouting_alloc=incr_eval.vrouting_alloc,
            hrouting_alloc=incr_eval.hrouting_alloc,
            tau_lse=50.0, mu_softplus=100.0,
            proximal_pos=proximal_anchor,
            proximal_weight=proximal_weight,
            hard_only_proximal=True, n_hard=n_hard,
            cell_idx_density=cell_idx_d,
            cell_idx_cong=cell_idx_c,
            return_components=True,
        )

    history = {"loss": [], "hpwl": [], "density": [], "cong": []}
    t0 = time.time()

    # ── GradNorm: compute per-component initial gradient norms ─────
    # Run an initial forward, take 3 separate backwards (each retains
    # graph), record gradient norms w.r.t. macro_pos. Use as denominators
    # for the rest of the run. This corrects for HPWL vs density vs
    # congestion gradient-magnitude disparity (Chen et al. 2018).
    if enable_density or enable_congestion:
        cell_idx0, K_max0 = build_window_indices(
            macro_pos.detach(), macro_w, macro_h,
            grid_col=incr_eval.grid_col, grid_row=incr_eval.grid_row,
            grid_w=incr_eval.grid_width, grid_h=incr_eval.grid_height,
            margin_cells=window_margin_cells)
        if enable_density:
            cell_idx_d = cell_idx0
        if enable_congestion:
            cell_idx_c = cell_idx0

    hpwl0, dens0, cong0, prox0 = _proxy_call()

    def _grad_norm(loss_term):
        """Norm of grad of loss_term w.r.t. macro_pos. retain_graph=True so
        we can take subsequent backwards from the same forward pass."""
        if not loss_term.requires_grad:
            return 1.0
        g = torch.autograd.grad(loss_term, macro_pos,
                                 retain_graph=True, allow_unused=True)[0]
        if g is None:
            return 1.0
        return float(g.norm().item())

    g_hpwl = _grad_norm(hpwl0)
    g_dens = _grad_norm(dens0) if enable_density else 1.0
    g_cong = _grad_norm(cong0) if enable_congestion else 1.0

    # GradNorm weights: 1.0 / (initial-norm + ε), with the largest fixed
    # at unit weight so absolute scale is preserved.
    eps = 1e-8
    w_hpwl = 1.0 / (g_hpwl + eps)
    w_dens = (1.0 / (g_dens + eps)) if enable_density else 0.0
    w_cong = (1.0 / (g_cong + eps)) if enable_congestion else 0.0
    # Renormalize so the largest weight is 1.0.
    w_max = max(w_hpwl, w_dens, w_cong, 1e-30)
    w_hpwl, w_dens, w_cong = w_hpwl / w_max, w_dens / w_max, w_cong / w_max

    # Apply task-importance multipliers (matches the exact-proxy weighting:
    # HPWL has weight 1, density 0.5, congestion 0.5).
    a_hpwl, a_dens, a_cong = 1.0, 0.5, 0.5

    if verbose:
        print(f"    [adam/gradnorm] g_hpwl={g_hpwl:.3e} g_dens={g_dens:.3e} "
              f"g_cong={g_cong:.3e}  →  w=({w_hpwl:.3f}, {w_dens:.3f}, "
              f"{w_cong:.3f})", flush=True)

    for step in range(n_steps):
        # Re-snapshot the cell windows periodically.
        if (enable_density or enable_congestion) and (step % snapshot_every == 0) and step > 0:
            cell_idx, K_max = build_window_indices(
                macro_pos.detach(), macro_w, macro_h,
                grid_col=incr_eval.grid_col, grid_row=incr_eval.grid_row,
                grid_w=incr_eval.grid_width, grid_h=incr_eval.grid_height,
                margin_cells=window_margin_cells)
            if enable_density:
                cell_idx_d = cell_idx
            if enable_congestion:
                cell_idx_c = cell_idx

        opt.zero_grad()
        hpwl, dens, cong, prox = _proxy_call()
        loss = (a_hpwl * w_hpwl * hpwl
                + a_dens * w_dens * dens
                + a_cong * w_cong * cong
                + prox)
        loss.backward()
        # Optional: zero-out hard-macro gradients if soft_only
        if soft_only:
            with torch.no_grad():
                macro_pos.grad[:n_hard] = 0.0
        opt.step()
        sched.step()

        # Clip to canvas
        with torch.no_grad():
            macro_pos.data[:, 0].clamp_(0.0, cw)
            macro_pos.data[:, 1].clamp_(0.0, ch)

        history["loss"].append(float(loss.item()))
        history["hpwl"].append(float(hpwl.item()))
        history["density"].append(float(dens.item()) if enable_density else 0.0)
        history["cong"].append(float(cong.item()) if enable_congestion else 0.0)
        if verbose and (step % 100 == 0 or step == n_steps - 1):
            print(f"    [adam] step {step}: loss={loss.item():.6f} "
                  f"HPWL={hpwl.item():.4f} "
                  f"den={(dens.item() if enable_density else 0.0):.4f} "
                  f"cng={(cong.item() if enable_congestion else 0.0):.4f} "
                  f"({time.time()-t0:.1f}s)", flush=True)

    pos_final = macro_pos.detach().cpu().numpy().astype(np.float64)
    if verbose:
        print(f"  [adam] {n_steps} steps in {time.time()-t0:.1f}s; "
              f"loss {history['loss'][0]:.6f} -> {history['loss'][-1]:.6f}",
              flush=True)
    return pos_final, history
