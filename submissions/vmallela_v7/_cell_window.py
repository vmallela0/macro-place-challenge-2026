"""Cell-window truncation for the smooth-proxy density / congestion gradient.

Problem (with the naive smooth surrogate)
------------------------------------------
The exact density / congestion grid is a function of macro position via
the integer-floor cell-index map (which cells each macro footprint
covers). That map is non-smooth. To get gradients, we replace `max(0, ·)`
in the rectangle-overlap formula with `softplus_μ`, but we still need to
evaluate the overlap with EVERY cell — for ibm17 that's 6500 macros ×
30 000 cells = 195 M pairs per Adam step. Memory and compute blow up.

Solution
--------
**Cell-window truncation.** For each macro, snapshot a *window* of
cells the macro might overlap during the next K Adam steps. Compute
the soft overlap only over the window. Cells outside the window are
treated as not-touched-by-this-macro (their accumulated contribution
across all macros remains a function of every macro's window).

For typical macros and K=50 Adam steps with lr ≤ 0.5 · cell_width,
the window only needs 4-6 cells of margin in each direction. This
gives ~36 cells per macro, n_total · 36 ≈ 250 K pairs per step on
ibm17. Memory negligible, compute trivial.

Re-snapshot every K steps to allow long-range macro motion.

API
---
`build_window_indices(...)` returns a (n_total, K) int tensor of cell
indices per macro (padded with -1 for unused slots, K = max window
cells across all macros). This is computed ONCE per snapshot interval.

`smooth_density_grid(...)` and `smooth_congestion_grid(...)` then use
this to compute the per-cell density / congestion contribution from
each macro via softplus rectangle overlap. Padded slots (-1) contribute
0.

Math validation
---------------
For a macro at (cx, cy) with size (w, h), the contribution to cell c
at (col, row) is:

  area(m, c) = max(0, min(cx + w/2, c.xmax) - max(cx - w/2, c.xmin))
             × max(0, min(cy + h/2, c.ymax) - max(cy - h/2, c.ymin))

Smoothed via softplus_μ:

  area_smooth(m, c) = softplus_μ(min(cx + w/2, c.xmax) - max(cx - w/2, c.xmin))
                    × softplus_μ(min(cy + h/2, c.ymax) - max(cy - h/2, c.ymin))

`min` and `max` operations are NOT smoothed — we use the hard versions.
This is correct because:
  - When macro is fully inside cell: min returns cx + w/2; max returns
    cx - w/2; their difference is w (a constant), softplus_μ(w) ≈ w
    for w >> 0; gradient w.r.t. cx is 0. CORRECT (full coverage doesn't
    change with small motion).
  - When macro is partially in cell: min/max select the smaller bound,
    softplus_μ has a non-zero gradient, motion changes overlap. CORRECT.
  - When macro is fully outside cell: difference is negative, softplus_μ
    is ~0, gradient is ~0. CORRECT.

So `min` / `max` operating on quantities that are individually
differentiable in cx (each branch is linear in cx, and the selection is
piecewise-constant) gives the right gradient via PyTorch's autograd
(it picks the gradient of the active branch — which corresponds to the
correct subgradient at the boundary).

The softplus_μ smoothing handles the kink at zero (transition from
overlap to no-overlap), giving smooth gradient flow.
"""
from __future__ import annotations
import math
import numpy as np
import torch


def softplus_mu(x: torch.Tensor, mu: float) -> torch.Tensor:
    """Smooth max(0, x). softplus_μ(x) = log(1 + exp(μ x)) / μ."""
    return torch.nn.functional.softplus(mu * x) / mu


def build_window_indices(
    macro_pos: torch.Tensor,        # (n_total, 2) on device
    macro_w: torch.Tensor,          # (n_total,)
    macro_h: torch.Tensor,          # (n_total,)
    grid_col: int, grid_row: int,
    grid_w: float, grid_h: float,
    margin_cells: int = 4,
):
    """Return per-macro cell-index lists for the soft-overlap window.

    For each macro at (cx, cy) with size (w, h):
        window_col_min = floor((cx - w/2)/grid_w) - margin_cells
        window_col_max = floor((cx + w/2)/grid_w) + margin_cells
        window_row_min = floor((cy - h/2)/grid_h) - margin_cells
        window_row_max = floor((cy + h/2)/grid_h) + margin_cells

    All clipped to [0, grid_col-1] / [0, grid_row-1].

    Returns
    -------
    cell_idx : (n_total, K_max) int — flat cell indices (-1 = padding)
    K_max : int — the maximum window size across all macros (for buffer
        sizing in downstream ops).

    The output is detached from autograd (it's a discrete index).
    Padded entries hold -1 and should be masked in the downstream
    accumulation.
    """
    device = macro_pos.device
    n_total = int(macro_pos.shape[0])

    # Compute per-macro window bounds in cell coordinates.
    cx = macro_pos[:, 0].detach()
    cy = macro_pos[:, 1].detach()
    half_w = macro_w / 2.0
    half_h = macro_h / 2.0

    col_min = torch.clamp(((cx - half_w) / grid_w).floor().long() - margin_cells,
                          0, grid_col - 1)
    col_max = torch.clamp(((cx + half_w) / grid_w).floor().long() + margin_cells,
                          0, grid_col - 1)
    row_min = torch.clamp(((cy - half_h) / grid_h).floor().long() - margin_cells,
                          0, grid_row - 1)
    row_max = torch.clamp(((cy + half_h) / grid_h).floor().long() + margin_cells,
                          0, grid_row - 1)

    # Window dimensions
    n_cols_per = (col_max - col_min + 1)   # (n_total,)
    n_rows_per = (row_max - row_min + 1)
    n_cells_per = n_cols_per * n_rows_per
    K_max = int(n_cells_per.max().item())

    # Build the index tensor by enumerating window cells per macro.
    # We do this on CPU since the per-macro loop is fast in numpy.
    col_min_np = col_min.cpu().numpy()
    col_max_np = col_max.cpu().numpy()
    row_min_np = row_min.cpu().numpy()
    row_max_np = row_max.cpu().numpy()

    cell_idx_np = np.full((n_total, K_max), -1, dtype=np.int64)
    for m in range(n_total):
        c_lo = int(col_min_np[m])
        c_hi = int(col_max_np[m])
        r_lo = int(row_min_np[m])
        r_hi = int(row_max_np[m])
        n_c = c_hi - c_lo + 1
        n_r = r_hi - r_lo + 1
        cols = np.arange(c_lo, c_hi + 1)[None, :].repeat(n_r, axis=0)  # (n_r, n_c)
        rows = np.arange(r_lo, r_hi + 1)[:, None].repeat(n_c, axis=1)  # (n_r, n_c)
        flat = (rows * grid_col + cols).reshape(-1)
        cell_idx_np[m, :flat.size] = flat
    cell_idx = torch.tensor(cell_idx_np, dtype=torch.long, device=device)
    return cell_idx, K_max


def smooth_density_grid(
    macro_pos: torch.Tensor,        # (n_total, 2) — has gradients
    macro_w: torch.Tensor,          # (n_total,)
    macro_h: torch.Tensor,          # (n_total,)
    cell_idx: torch.Tensor,         # (n_total, K_max) int (-1 = pad)
    grid_col: int, grid_row: int,
    grid_w: float, grid_h: float,
    n_cells: int,
    cell_area: float,
    mu: float = 100.0,
):
    """Compute the density grid via softplus rectangle overlap, accumulated
    per cell via index_add. Returns a (n_cells,) tensor with gradient w.r.t.
    macro_pos.

    Implementation: for each (macro, window-cell) pair, evaluate the
    softplus overlap, accumulate into the global cell density via
    index_add_.
    """
    device = macro_pos.device
    n_total = macro_pos.shape[0]
    K_max = cell_idx.shape[1]

    # Per-(m, j-th-window-cell) cell index. Mask invalid (-1 → 0 then
    # zero-out contribution via mask).
    valid = (cell_idx >= 0)                     # (n_total, K_max)
    cell_idx_safe = cell_idx.clamp_min(0)       # (n_total, K_max)
    rows = cell_idx_safe // grid_col            # (n_total, K_max)
    cols = cell_idx_safe % grid_col

    cell_x_min = cols.to(macro_pos.dtype) * grid_w   # (n_total, K_max)
    cell_x_max = cell_x_min + grid_w
    cell_y_min = rows.to(macro_pos.dtype) * grid_h
    cell_y_max = cell_y_min + grid_h

    cx = macro_pos[:, 0:1]    # (n_total, 1) broadcast
    cy = macro_pos[:, 1:2]
    half_w = (macro_w / 2.0).unsqueeze(1)    # (n_total, 1)
    half_h = (macro_h / 2.0).unsqueeze(1)

    macro_x_min = cx - half_w   # (n_total, 1)
    macro_x_max = cx + half_w
    macro_y_min = cy - half_h
    macro_y_max = cy + half_h

    # x_overlap = min(macro_xmax, cell_xmax) - max(macro_xmin, cell_xmin)
    # smooth via softplus to handle the boundary kink.
    x_inner = torch.minimum(macro_x_max, cell_x_max) - \
              torch.maximum(macro_x_min, cell_x_min)
    y_inner = torch.minimum(macro_y_max, cell_y_max) - \
              torch.maximum(macro_y_min, cell_y_min)
    x_overlap = softplus_mu(x_inner, mu)   # (n_total, K_max)
    y_overlap = softplus_mu(y_inner, mu)

    area = x_overlap * y_overlap
    area = area * valid.to(area.dtype)
    contribution = area / cell_area   # density contribution per cell

    # Scatter into the global density vector.
    grid_density = torch.zeros(n_cells, device=device, dtype=macro_pos.dtype)
    grid_density.index_add_(0, cell_idx_safe.reshape(-1),
                             contribution.reshape(-1))
    return grid_density


def electrostatic_density_energy(grid_density, grid_row, grid_col,
                                    grid_w=1.0, grid_h=1.0):
    """DREAMPlace/ePlace-style electrostatic potential energy of the
    density distribution.

    Treats the density ρ(x) on the canvas grid as a charge distribution.
    Solves the 2D Poisson equation ∇²φ = ρ - ρ̄ with periodic boundary
    conditions via 2D FFT, then returns the integrated |φ|² as a
    scalar energy.

    Mathematical justification: at uniform density, ρ - ρ̄ ≡ 0, so
    φ ≡ 0 and energy = 0. Any density variation creates a non-zero
    potential field; the energy measures how "non-uniform" the
    placement is. Minimizing energy = pushing density toward uniform =
    the same global density structure DREAMPlace and ePlace exploit.

    Differentiable via torch.fft.fft2 + torch.fft.ifft2 — the entire
    Poisson solve flows gradients through to macro_pos via the upstream
    smooth_density_grid call.

    Why this matters: CVaR top-K density is *locally* myopic (sees
    only K worst cells). Electrostatic potential is *globally aware*
    (every cell contributes via the Green's function). The Hessian
    eigenvector under this surrogate captures DREAMPlace-class
    long-range curvature directions.

    Args:
        grid_density: (n_cells,) flat tensor with gradient w.r.t. macro_pos
        grid_row, grid_col: grid dimensions
        grid_w, grid_h: cell physical width and height (for the units of
            the Laplacian)

    Returns: scalar tensor (energy)
    """
    rho = grid_density.reshape(grid_row, grid_col)
    rho_balanced = rho - rho.mean()

    # 2D FFT (orthonormal scaling so that energy is preserved by Parseval)
    F_rho = torch.fft.fft2(rho_balanced, norm="ortho")

    # Wave-number grid. fftfreq returns cycles/sample; we scale by
    # 2π/(N·d) to get physical wave number k. Then k² = k_x² + k_y².
    # The Laplacian operator in Fourier space is -k², so Poisson is
    # F(φ) = -F(ρ) / k².
    import math
    kx = torch.fft.fftfreq(grid_col,
                              d=float(grid_w),
                              device=rho.device,
                              dtype=rho.dtype) * (2 * math.pi)
    ky = torch.fft.fftfreq(grid_row,
                              d=float(grid_h),
                              device=rho.device,
                              dtype=rho.dtype) * (2 * math.pi)
    kx2 = (kx * kx).reshape(1, -1)
    ky2 = (ky * ky).reshape(-1, 1)
    k_sq = kx2 + ky2

    # Solve in frequency domain. DC term (k=0) handled by setting it to
    # zero explicitly — ρ_balanced has zero mean, so F_rho[0,0] is
    # already ≈ 0 (up to floating point).
    inv_k_sq = torch.where(k_sq > 0,
                             1.0 / (k_sq + 1e-12),
                             torch.zeros_like(k_sq))
    F_phi = F_rho * inv_k_sq   # solving -∇²φ = ρ; sign absorbed into energy

    phi = torch.fft.ifft2(F_phi, norm="ortho").real

    # Energy = integrated potential squared, scaled by cell area.
    cell_area = float(grid_w) * float(grid_h)
    energy = (phi * phi).sum() * cell_area
    return energy


def electrostatic_density_energy_normalized(grid_density, grid_row, grid_col,
                                              grid_w=1.0, grid_h=1.0):
    """Scale-balanced electrostatic energy.

    Returns the same Poisson energy as `electrostatic_density_energy`
    but normalized by the energy of a delta-function input. The result
    is dimensionless and roughly O(1) for typical density distributions,
    making it directly comparable to HPWL_LSE-norm without scaling
    issues that can dominate the Hessian.

    Specifically: divide by (mean_density² · canvas_area) where the
    energy of a uniform offset = 0 by construction (we subtract mean
    inside the energy function), so we use the variance of density
    times canvas_area as the natural scale.
    """
    rho = grid_density.reshape(grid_row, grid_col)
    rho_var = ((rho - rho.mean()) ** 2).mean()
    cell_area = float(grid_w) * float(grid_h)
    canvas_area = float(grid_row) * float(grid_col) * cell_area
    energy = electrostatic_density_energy(grid_density, grid_row, grid_col,
                                             grid_w=grid_w, grid_h=grid_h)
    # Normalize by characteristic scale: var(ρ) · canvas_area.
    scale = rho_var * canvas_area + 1e-12
    return energy / scale


def smooth_macro_blockage(
    macro_pos: torch.Tensor,
    macro_w: torch.Tensor,
    macro_h: torch.Tensor,
    cell_idx: torch.Tensor,
    grid_col: int, grid_row: int,
    grid_w: float, grid_h: float,
    n_cells: int,
    vrouting_alloc: float,
    hrouting_alloc: float,
    mu: float = 100.0,
):
    """Smooth-overlap version of the macro-blockage routing demand.

    Each macro at (cx, cy) with size (w, h) blocks routing in cells
    that its footprint covers. The contribution to vertical-routing
    blockage per cell is (x_overlap * vrouting_alloc), and to
    horizontal-routing blockage per cell is (y_overlap * hrouting_alloc).
    These mirror the IncrementalEvaluator's `__macro_route_over_grid_cell`
    main path (we drop the partial-overlap correction terms for the
    surrogate; the CPU evaluator handles those exactly on commit).
    """
    device = macro_pos.device
    n_total = macro_pos.shape[0]
    K_max = cell_idx.shape[1]

    valid = (cell_idx >= 0)
    cell_idx_safe = cell_idx.clamp_min(0)
    rows = cell_idx_safe // grid_col
    cols = cell_idx_safe % grid_col

    cell_x_min = cols.to(macro_pos.dtype) * grid_w
    cell_x_max = cell_x_min + grid_w
    cell_y_min = rows.to(macro_pos.dtype) * grid_h
    cell_y_max = cell_y_min + grid_h

    cx = macro_pos[:, 0:1]
    cy = macro_pos[:, 1:2]
    half_w = (macro_w / 2.0).unsqueeze(1)
    half_h = (macro_h / 2.0).unsqueeze(1)

    x_inner = torch.minimum(cx + half_w, cell_x_max) - \
              torch.maximum(cx - half_w, cell_x_min)
    y_inner = torch.minimum(cy + half_h, cell_y_max) - \
              torch.maximum(cy - half_h, cell_y_min)
    x_overlap = softplus_mu(x_inner, mu)
    y_overlap = softplus_mu(y_inner, mu)

    valid_f = valid.to(macro_pos.dtype)
    v_contrib = x_overlap * vrouting_alloc * valid_f
    h_contrib = y_overlap * hrouting_alloc * valid_f

    V_macro = torch.zeros(n_cells, device=device, dtype=macro_pos.dtype)
    H_macro = torch.zeros(n_cells, device=device, dtype=macro_pos.dtype)
    V_macro.index_add_(0, cell_idx_safe.reshape(-1), v_contrib.reshape(-1))
    H_macro.index_add_(0, cell_idx_safe.reshape(-1), h_contrib.reshape(-1))
    return V_macro, H_macro
