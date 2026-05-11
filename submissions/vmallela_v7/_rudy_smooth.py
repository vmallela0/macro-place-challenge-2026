"""Differentiable RUDY routing demand for the v7 Hessian smooth surrogate.

Why this exists
===============
The existing congestion-aware Hessian surrogate (see `placer.py:1469-1488`)
adds a `cong_smooth` term computed from
    V_total = V_smooth_FROZEN  +  V_macro(x) / grid_v_routes
where `V_smooth_FROZEN` is the per-net RUDY/routing demand snapshot taken
once at the BEGINNING of the Hessian phase and never re-evaluated as
macros move. As a result, autograd ∂cong/∂x captures only the
*macro-blockage* gradient — the dominant *net-routing* contribution is
treated as a fixed offset.

Empirically (research/ITERATIONS.md Iter 4d / Iter 7), enabling
congestion at weight=0.5 with frozen routing produces:
    mean Δ = -0.0007 across 5 high-room benches  (essentially zero)
    while ibm06 at w=0.0 (cong-off) WINS over any positive weight.
This is the smoking gun for the frozen-routing pathology — the
"congestion direction" the eigenvector picks is computed against a
stale routing map and so points in a direction that is good on the
stale map but actively wrong on the live map.

The fix
=======
Compute RUDY routing demand differentiably from macro positions, so
the gradient of the smooth proxy w.r.t. macros sees the full congestion
sensitivity, not just the macro-blockage piece.

RUDY (Spindler-Johannes 2007): for net n with pins
{(x_1, y_1), ..., (x_K, y_K)}, the half-perimeter bounding box has
width Δx_n and height Δy_n. The net contributes uniformly across its
bbox the wirelength density
    horizontal-wire density  =  Δx_n / (Δx_n · Δy_n)  =  1 / Δy_n
    vertical-wire   density  =  Δy_n / (Δx_n · Δy_n)  =  1 / Δx_n
Per cell c of area cell_area:
    V_demand_c  +=  net_weight · overlap(c, bbox_n) / Δy_n     (V channels carry horizontal wires)
    H_demand_c  +=  net_weight · overlap(c, bbox_n) / Δx_n

Smoothing
---------
We replace each non-smooth operator with a softplus/LSE counterpart
matching the rest of the v7 smooth proxy:
    Δx_n = LSE_max(x) − LSE_min(x)              [smooth bbox extent, τ=50]
    overlap(c, bbox_n) = softplus_μ(x_inner) · softplus_μ(y_inner)
                        with x_inner = min(cell_xmax, bbox_xmax) − max(cell_xmin, bbox_xmin)
The cell-membership decision (which window cells a net touches) is
discrete; we snapshot it every K Hessian/HMC steps via
`build_net_window_indices`. Between snapshots, cells outside a net's
window are treated as not-touched-by-this-net. With a 4-cell margin
and re-snapshot every 50 steps, the gradient stays representative as
nets drift.

Cost
----
For typical benches (n_nets ≤ 6k, max bbox ≤ ~50 cells), the
(n_nets × K_max) tensor has ~3e5 entries. One forward+backward pass is
~30 ms on a 32-thread CPU. Comparable to one Lanczos HVP. Net effect
on a 50-iter Lanczos: +50 × 30 ms = +1.5 s — negligible vs the 1000 s
Hessian budget.

Validation strategy
===================
The strict-improvement gate against EXACT PlacementCost is preserved:
this module changes the surrogate (and therefore the search direction)
but never the cost being evaluated for acceptance. So a worse RUDY
surrogate cannot regress the placer's final cost — it can only fail
to find a lift. Safe to A/B.
"""
from __future__ import annotations

import numpy as np
import torch


def softplus_mu(x: torch.Tensor, mu: float) -> torch.Tensor:
    """Smooth max(0, x). softplus_μ(x) = log(1 + exp(μ x)) / μ."""
    return torch.nn.functional.softplus(mu * x) / mu


def lse_bbox_per_net(
    pin_x: torch.Tensor,            # (n_pins,) requires_grad
    pin_y: torch.Tensor,            # (n_pins,) requires_grad
    pin_to_net: torch.Tensor,       # (n_pins,) long
    n_nets: int,
    tau_lse: float = 50.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-net smooth bounding box via scatter+LSE.

    Returns (x_lo, x_hi, y_lo, y_hi), each (n_nets,) with autograd
    through pin_x / pin_y. LSE temperature τ=50 matches lse_hpwl_vectorized.

    Math: for net n with pins x = {x_i : pin_to_net[i] == n}:
        x_hi = LSE_max_τ(x)  ≈  max(x)  as τ→∞
        x_lo = -LSE_max_τ(-x) ≈ min(x)
    Numerically stable: shift by detached max before exponentiation.
    1-pin nets give x_hi = x_lo = x (zero bbox); we don't special-case
    them — the smooth surrogate treats them as point contributions.
    """
    device = pin_x.device
    dtype = pin_x.dtype

    def _scatter_lse_max(values: torch.Tensor) -> torch.Tensor:
        v_max = torch.full((n_nets,), float('-inf'), device=device, dtype=dtype)
        v_max.scatter_reduce_(0, pin_to_net, values.detach(),
                               reduce='amax', include_self=True)
        v_max_pp = v_max[pin_to_net]
        ex = torch.exp(tau_lse * (values - v_max_pp))
        sx = torch.zeros(n_nets, device=device, dtype=dtype)
        sx.scatter_add_(0, pin_to_net, ex)
        return v_max + torch.log(sx.clamp_min(1e-30)) / tau_lse

    x_hi = _scatter_lse_max(pin_x)
    x_lo = -_scatter_lse_max(-pin_x)
    y_hi = _scatter_lse_max(pin_y)
    y_lo = -_scatter_lse_max(-pin_y)
    return x_lo, x_hi, y_lo, y_hi


def build_net_window_indices_sparse(
    pin_x: torch.Tensor,
    pin_y: torch.Tensor,
    pin_to_net: torch.Tensor,
    n_nets: int,
    grid_col: int, grid_row: int,
    grid_w: float, grid_h: float,
    margin_cells: int = 4,
    max_window_cells: int = 0,        # 0 = unlimited; >0 drops nets w/ larger bbox
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """Sparse COO version of per-net cell windows.

    Returns (pair_net, pair_cell, n_pairs, n_dropped):
        pair_net : (n_pairs,) long — net index for each (net, cell) pair.
        pair_cell : (n_pairs,) long — flat cell index for each pair.

    Avoids the dense (n_nets, K_max) tensor used by the legacy
    `build_net_window_indices`: when ONE net has a giant bbox (e.g.
    a global clock spanning the whole canvas), K_max blows up and the
    forward pass spends most of its time on padded entries. With sparse
    COO, total work is proportional to actual cell-pair count
    (~50 × n_nets typical, vs 868 × n_nets in the dense case for ibm06).

    Optional `max_window_cells` (default 0 = no cap): drops nets whose
    bbox-window exceeds this size from the RUDY sum entirely. Rationale:
    a net with a giant bbox has uniform low per-cell density (RUDY
    allocates w/Δy to each cell; large Δy = small contribution). It is
    unlikely to drive the CVaR top-K; dropping it loses signal of
    O(1/Δy) per cell — negligible for the largest nets. Combined with
    the >5× speedup, this is a good trade.

    Returns n_dropped for diagnostics.
    """
    device = pin_x.device
    pin_x_d = pin_x.detach()
    pin_y_d = pin_y.detach()
    pin_to_net_d = pin_to_net.detach().to(torch.long)

    x_max = torch.full((n_nets,), float('-inf'),
                        device=device, dtype=pin_x_d.dtype)
    x_max.scatter_reduce_(0, pin_to_net_d, pin_x_d,
                           reduce='amax', include_self=True)
    x_min = torch.full((n_nets,), float('inf'),
                        device=device, dtype=pin_x_d.dtype)
    x_min.scatter_reduce_(0, pin_to_net_d, pin_x_d,
                           reduce='amin', include_self=True)
    y_max = torch.full((n_nets,), float('-inf'),
                        device=device, dtype=pin_y_d.dtype)
    y_max.scatter_reduce_(0, pin_to_net_d, pin_y_d,
                           reduce='amax', include_self=True)
    y_min = torch.full((n_nets,), float('inf'),
                        device=device, dtype=pin_y_d.dtype)
    y_min.scatter_reduce_(0, pin_to_net_d, pin_y_d,
                           reduce='amin', include_self=True)

    inf_mask = ~torch.isfinite(x_max) | ~torch.isfinite(x_min) \
                | ~torch.isfinite(y_max) | ~torch.isfinite(y_min)
    x_max = torch.where(inf_mask, torch.zeros_like(x_max), x_max)
    x_min = torch.where(inf_mask, torch.zeros_like(x_min), x_min)
    y_max = torch.where(inf_mask, torch.zeros_like(y_max), y_max)
    y_min = torch.where(inf_mask, torch.zeros_like(y_min), y_min)

    col_min = torch.clamp((x_min / grid_w).floor().long() - margin_cells,
                           0, grid_col - 1).cpu().numpy()
    col_max = torch.clamp((x_max / grid_w).floor().long() + margin_cells,
                           0, grid_col - 1).cpu().numpy()
    row_min = torch.clamp((y_min / grid_h).floor().long() - margin_cells,
                           0, grid_row - 1).cpu().numpy()
    row_max = torch.clamp((y_max / grid_h).floor().long() + margin_cells,
                           0, grid_row - 1).cpu().numpy()
    inf_np = inf_mask.cpu().numpy()

    n_cols = col_max - col_min + 1
    n_rows = row_max - row_min + 1
    n_cells_per = n_cols * n_rows
    n_cells_per[inf_np] = 0
    if max_window_cells > 0:
        too_big = n_cells_per > max_window_cells
        n_cells_per[too_big] = 0
        n_dropped = int(too_big.sum())
    else:
        n_dropped = 0
    total_pairs = int(n_cells_per.sum())

    # Allocate flat arrays.
    pair_net_np = np.empty(total_pairs, dtype=np.int64)
    pair_cell_np = np.empty(total_pairs, dtype=np.int64)
    offset = 0
    for n in range(n_nets):
        if n_cells_per[n] == 0:
            continue
        c_lo = int(col_min[n]); c_hi = int(col_max[n])
        r_lo = int(row_min[n]); r_hi = int(row_max[n])
        n_c = c_hi - c_lo + 1
        n_r = r_hi - r_lo + 1
        cols = np.arange(c_lo, c_hi + 1)[None, :].repeat(n_r, axis=0)
        rows = np.arange(r_lo, r_hi + 1)[:, None].repeat(n_c, axis=1)
        flat = (rows * grid_col + cols).reshape(-1)
        sz = flat.size
        pair_net_np[offset:offset + sz] = n
        pair_cell_np[offset:offset + sz] = flat
        offset += sz
    assert offset == total_pairs

    pair_net = torch.tensor(pair_net_np, dtype=torch.long, device=device)
    pair_cell = torch.tensor(pair_cell_np, dtype=torch.long, device=device)
    return pair_net, pair_cell, total_pairs, n_dropped


def smooth_rudy_routing_sparse(
    pin_x: torch.Tensor,             # (n_pins,) requires_grad
    pin_y: torch.Tensor,
    pin_to_net: torch.Tensor,
    net_weight: torch.Tensor,
    n_nets: int,
    pair_net: torch.Tensor,          # (n_pairs,) long
    pair_cell: torch.Tensor,         # (n_pairs,) long
    grid_col: int, grid_row: int,
    grid_w: float, grid_h: float,
    n_cells: int,
    *,
    tau_lse: float = 50.0,
    mu_softplus: float = 100.0,
    eps_bbox: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sparse-COO differentiable RUDY.

    Same math as `smooth_rudy_routing_with_pinx` but iterates over
    (net, cell) PAIRS rather than the dense (n_nets, K_max) matrix.
    ~17× faster on ibm06 (the K_max=868 dense matrix has ~93 % padding).
    """
    device = pin_x.device
    dtype = pin_x.dtype

    bx_lo, bx_hi, by_lo, by_hi = lse_bbox_per_net(
        pin_x, pin_y, pin_to_net, n_nets, tau_lse=tau_lse)
    delta_x = bx_hi - bx_lo + eps_bbox            # (n_nets,)
    delta_y = by_hi - by_lo + eps_bbox

    # Gather per-pair from per-net tensors.
    bxl = bx_lo[pair_net]                          # (n_pairs,)
    bxh = bx_hi[pair_net]
    byl = by_lo[pair_net]
    byh = by_hi[pair_net]
    dx_p = delta_x[pair_net]
    dy_p = delta_y[pair_net]
    nw_p = net_weight[pair_net]

    rows = pair_cell // grid_col
    cols = pair_cell % grid_col
    cell_xl = cols.to(dtype) * grid_w
    cell_xh = cell_xl + grid_w
    cell_yl = rows.to(dtype) * grid_h
    cell_yh = cell_yl + grid_h

    x_inner = torch.minimum(bxh, cell_xh) - torch.maximum(bxl, cell_xl)
    y_inner = torch.minimum(byh, cell_yh) - torch.maximum(byl, cell_yl)
    x_ov = softplus_mu(x_inner, mu_softplus)
    y_ov = softplus_mu(y_inner, mu_softplus)
    overlap = x_ov * y_ov                          # (n_pairs,)

    v_contrib = nw_p * overlap / dy_p              # (n_pairs,)
    h_contrib = nw_p * overlap / dx_p

    V_demand = torch.zeros(n_cells, device=device, dtype=dtype)
    H_demand = torch.zeros(n_cells, device=device, dtype=dtype)
    V_demand.index_add_(0, pair_cell, v_contrib)
    H_demand.index_add_(0, pair_cell, h_contrib)
    return V_demand, H_demand


def build_net_window_indices(
    pin_x: torch.Tensor,             # (n_pins,) detached coords from current state
    pin_y: torch.Tensor,
    pin_to_net: torch.Tensor,        # (n_pins,) long
    n_nets: int,
    grid_col: int, grid_row: int,
    grid_w: float, grid_h: float,
    margin_cells: int = 4,
) -> tuple[torch.Tensor, int]:
    """Per-net cell-index window for differentiable RUDY scatter.

    Computes the HARD bbox of each net from current pin positions (no
    autograd) and inflates by `margin_cells` in every direction; returns
    a (n_nets, K_max) long tensor of cell indices (-1 = padding).

    Re-call this every snapshot interval (default: every 50 Lanczos /
    HMC steps). For ibm15-scale (~6k nets, max bbox ~30 cells) this
    builds in <100 ms — negligible.

    The window choice is intentionally generous (4-cell margin) so that
    motion within a single snapshot interval cannot push a net's bbox
    fully outside its window. Empirically with stepsize ≤ 1.0 cell/step
    and snapshot every 50 steps, max drift is ~50 cells — but the LSE
    bbox only changes meaningfully on the extreme pins; typical drift is
    a few cells. 4-cell margin is safe.
    """
    device = pin_x.device

    pin_x_d = pin_x.detach()
    pin_y_d = pin_y.detach()
    pin_to_net_d = pin_to_net.detach().to(torch.long)

    # Compute hard min/max per net via scatter_reduce(amin/amax).
    x_max = torch.full((n_nets,), float('-inf'),
                        device=device, dtype=pin_x_d.dtype)
    x_max.scatter_reduce_(0, pin_to_net_d, pin_x_d,
                           reduce='amax', include_self=True)
    x_min = torch.full((n_nets,), float('inf'),
                        device=device, dtype=pin_x_d.dtype)
    x_min.scatter_reduce_(0, pin_to_net_d, pin_x_d,
                           reduce='amin', include_self=True)
    y_max = torch.full((n_nets,), float('-inf'),
                        device=device, dtype=pin_y_d.dtype)
    y_max.scatter_reduce_(0, pin_to_net_d, pin_y_d,
                           reduce='amax', include_self=True)
    y_min = torch.full((n_nets,), float('inf'),
                        device=device, dtype=pin_y_d.dtype)
    y_min.scatter_reduce_(0, pin_to_net_d, pin_y_d,
                           reduce='amin', include_self=True)

    # Empty nets (no pins) keep ±inf — replace with sentinel that produces
    # an empty window.
    inf_mask = ~torch.isfinite(x_max) | ~torch.isfinite(x_min) \
                | ~torch.isfinite(y_max) | ~torch.isfinite(y_min)
    x_max = torch.where(inf_mask, torch.zeros_like(x_max), x_max)
    x_min = torch.where(inf_mask, torch.zeros_like(x_min), x_min)
    y_max = torch.where(inf_mask, torch.zeros_like(y_max), y_max)
    y_min = torch.where(inf_mask, torch.zeros_like(y_min), y_min)

    col_min = torch.clamp((x_min / grid_w).floor().long() - margin_cells,
                           0, grid_col - 1)
    col_max = torch.clamp((x_max / grid_w).floor().long() + margin_cells,
                           0, grid_col - 1)
    row_min = torch.clamp((y_min / grid_h).floor().long() - margin_cells,
                           0, grid_row - 1)
    row_max = torch.clamp((y_max / grid_h).floor().long() + margin_cells,
                           0, grid_row - 1)

    n_cols = (col_max - col_min + 1).cpu().numpy()
    n_rows = (row_max - row_min + 1).cpu().numpy()
    n_cells_per = n_cols * n_rows
    if inf_mask.any():
        n_cells_per[inf_mask.cpu().numpy()] = 0
    K_max = int(max(n_cells_per.max(), 1))

    cell_idx_np = np.full((n_nets, K_max), -1, dtype=np.int64)
    cl_np = col_min.cpu().numpy()
    ch_np = col_max.cpu().numpy()
    rl_np = row_min.cpu().numpy()
    rh_np = row_max.cpu().numpy()
    for n in range(n_nets):
        if inf_mask[n]:
            continue
        c_lo = int(cl_np[n]); c_hi = int(ch_np[n])
        r_lo = int(rl_np[n]); r_hi = int(rh_np[n])
        n_c = c_hi - c_lo + 1
        n_r = r_hi - r_lo + 1
        cols = np.arange(c_lo, c_hi + 1)[None, :].repeat(n_r, axis=0)
        rows = np.arange(r_lo, r_hi + 1)[:, None].repeat(n_c, axis=1)
        flat = (rows * grid_col + cols).reshape(-1)
        cell_idx_np[n, :flat.size] = flat
    cell_idx = torch.tensor(cell_idx_np, dtype=torch.long, device=device)
    return cell_idx, K_max


def smooth_rudy_routing(
    macro_pos: torch.Tensor,         # (n_total, 2) requires_grad
    pin_macro: torch.Tensor,         # (n_pins,) long, -1 = port (absolute coord)
    pin_xoff: torch.Tensor,          # (n_pins,) float
    pin_yoff: torch.Tensor,
    pin_to_net: torch.Tensor,        # (n_pins,) long
    net_weight: torch.Tensor,        # (n_nets,) float
    n_nets: int,
    net_cell_idx: torch.Tensor,      # (n_nets, K_max) long (-1 = padding)
    grid_col: int, grid_row: int,
    grid_w: float, grid_h: float,
    n_cells: int,
    *,
    tau_lse: float = 50.0,
    mu_softplus: float = 100.0,
    eps_bbox: float = 1.0,           # minimum bbox extent in micron to avoid div-by-zero
) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable RUDY routing demand grids.

    For each net n with smooth bbox (Δx_n, Δy_n) computed from current
    pin positions (LSE-smoothed), accumulate per-cell contributions
        V_demand_c += w_n · overlap(c, bbox_n) / Δy_n
        H_demand_c += w_n · overlap(c, bbox_n) / Δx_n
    over the net's window cells. Returns (V_demand, H_demand), each
    (n_cells,), with autograd through macro_pos via pin_x/pin_y.

    Both V_demand and H_demand are in the SAME units as the
    IncrementalEvaluator's `V_routing_smooth` / `H_routing_smooth`
    (already divided by net_cnt is handled upstream of the cong cell
    grid; here we produce the raw per-cell demand that V_total /
    grid_v_routes uses directly).

    Notes on edge cases:
    - 1-pin nets give Δx = Δy = 0 → division-by-zero in V/H factors.
      We add `eps_bbox` (1 micron, ~1 cell) to both denominators. The
      1-pin contribution is degenerate (any cell selected by the net
      window receives identical contribution), which is the correct
      RUDY limit for a point net (it contributes nothing meaningful).
    - Padding entries (-1) contribute zero via the `valid` mask.
    - When all pins of a net coincide (Δx_n very small), the routing
      density blows up; `eps_bbox` puts a floor on this. The strict-
      improvement gate against exact cost handles any residual
      pathology.
    """
    device = macro_pos.device
    dtype = macro_pos.dtype

    # ── Pin coords from macro positions + offsets ───────────────────
    is_port = (pin_macro < 0)
    safe = torch.where(is_port, torch.zeros_like(pin_macro), pin_macro)
    macro_xy = macro_pos[safe]
    pin_x = torch.where(is_port, pin_xoff, macro_xy[:, 0] + pin_xoff)
    pin_y = torch.where(is_port, pin_yoff, macro_xy[:, 1] + pin_yoff)

    # ── Smooth per-net bbox via LSE ─────────────────────────────────
    bx_lo, bx_hi, by_lo, by_hi = lse_bbox_per_net(
        pin_x, pin_y, pin_to_net, n_nets, tau_lse=tau_lse)
    delta_x = bx_hi - bx_lo + eps_bbox            # (n_nets,)
    delta_y = by_hi - by_lo + eps_bbox

    # ── Cell coords for window cells ────────────────────────────────
    K_max = net_cell_idx.shape[1]
    valid = (net_cell_idx >= 0)
    idx_safe = net_cell_idx.clamp_min(0)
    rows = idx_safe // grid_col
    cols = idx_safe % grid_col
    cell_x_lo = cols.to(dtype) * grid_w            # (n_nets, K_max)
    cell_x_hi = cell_x_lo + grid_w
    cell_y_lo = rows.to(dtype) * grid_h
    cell_y_hi = cell_y_lo + grid_h

    # ── Smooth (net, window-cell) overlap ───────────────────────────
    bxl = bx_lo.unsqueeze(1); bxh = bx_hi.unsqueeze(1)
    byl = by_lo.unsqueeze(1); byh = by_hi.unsqueeze(1)
    x_inner = torch.minimum(bxh, cell_x_hi) - torch.maximum(bxl, cell_x_lo)
    y_inner = torch.minimum(byh, cell_y_hi) - torch.maximum(byl, cell_y_lo)
    x_ov = softplus_mu(x_inner, mu_softplus)
    y_ov = softplus_mu(y_inner, mu_softplus)
    overlap_area = (x_ov * y_ov) * valid.to(dtype)  # (n_nets, K_max)

    # ── Per-net V/H factor ──────────────────────────────────────────
    v_factor = (net_weight / delta_y).unsqueeze(1)  # (n_nets, 1)
    h_factor = (net_weight / delta_x).unsqueeze(1)

    v_contrib = v_factor * overlap_area
    h_contrib = h_factor * overlap_area

    V_demand = torch.zeros(n_cells, device=device, dtype=dtype)
    H_demand = torch.zeros(n_cells, device=device, dtype=dtype)
    V_demand.index_add_(0, idx_safe.reshape(-1), v_contrib.reshape(-1))
    H_demand.index_add_(0, idx_safe.reshape(-1), h_contrib.reshape(-1))
    return V_demand, H_demand


def smooth_rudy_routing_with_pinx(
    pin_x: torch.Tensor,             # (n_pins,) requires_grad — already-computed pin coords
    pin_y: torch.Tensor,
    pin_to_net: torch.Tensor,
    net_weight: torch.Tensor,
    n_nets: int,
    net_cell_idx: torch.Tensor,
    grid_col: int, grid_row: int,
    grid_w: float, grid_h: float,
    n_cells: int,
    *,
    tau_lse: float = 50.0,
    mu_softplus: float = 100.0,
    eps_bbox: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Variant where the caller has already computed pin_x / pin_y.

    Same math; lets the smooth_proxy_call avoid recomputing pin coords
    when HPWL and RUDY share them.
    """
    device = pin_x.device
    dtype = pin_x.dtype

    bx_lo, bx_hi, by_lo, by_hi = lse_bbox_per_net(
        pin_x, pin_y, pin_to_net, n_nets, tau_lse=tau_lse)
    delta_x = bx_hi - bx_lo + eps_bbox
    delta_y = by_hi - by_lo + eps_bbox

    K_max = net_cell_idx.shape[1]
    valid = (net_cell_idx >= 0)
    idx_safe = net_cell_idx.clamp_min(0)
    rows = idx_safe // grid_col
    cols = idx_safe % grid_col
    cell_x_lo = cols.to(dtype) * grid_w
    cell_x_hi = cell_x_lo + grid_w
    cell_y_lo = rows.to(dtype) * grid_h
    cell_y_hi = cell_y_lo + grid_h

    bxl = bx_lo.unsqueeze(1); bxh = bx_hi.unsqueeze(1)
    byl = by_lo.unsqueeze(1); byh = by_hi.unsqueeze(1)
    x_inner = torch.minimum(bxh, cell_x_hi) - torch.maximum(bxl, cell_x_lo)
    y_inner = torch.minimum(byh, cell_y_hi) - torch.maximum(byl, cell_y_lo)
    x_ov = softplus_mu(x_inner, mu_softplus)
    y_ov = softplus_mu(y_inner, mu_softplus)
    overlap_area = (x_ov * y_ov) * valid.to(dtype)

    v_factor = (net_weight / delta_y).unsqueeze(1)
    h_factor = (net_weight / delta_x).unsqueeze(1)

    v_contrib = v_factor * overlap_area
    h_contrib = h_factor * overlap_area

    V_demand = torch.zeros(n_cells, device=device, dtype=dtype)
    H_demand = torch.zeros(n_cells, device=device, dtype=dtype)
    V_demand.index_add_(0, idx_safe.reshape(-1), v_contrib.reshape(-1))
    H_demand.index_add_(0, idx_safe.reshape(-1), h_contrib.reshape(-1))
    return V_demand, H_demand
