"""ePlace-lite: electrostatic-analogy placement for warm-start generation.

Reference: Lu et al. 2014 "ePlace: Electrostatics-based Placement Using
Fast Fourier Transform and Nesterov's Method." TODAES.
DREAMPlace (Cheng et al. 2019, DAC) is a modern PyTorch port.

Mechanism
---------
Each macro is a "charged particle" on a 2D continuous canvas. Macros
exert repulsive force on each other proportional to their density
overlap. We model this via the electrostatic analogy:

    ρ(c) = density at cell c (sum of macro footprint overlap / cell_area)
    target ρ(c) = uniform spread = total_macro_area / canvas_area
    ε(c) = ρ(c) − target = "charge density" (positive = over-occupied)

Solve Poisson: ∇²ψ = -ε   (with periodic-canvas BCs for FFT efficiency)
Force on macro at (x, y) = -∇ψ at that position.

Gradient descent: x ← x - lr · ∇ψ(x). Equilibrium when ε ≡ 0
(perfectly uniform density). For real placement we don't reach
equilibrium — we run for N steps and use the result as a globally-
spread warm start that's structurally different from .plc init.

Why this might break the post-Laplacian basin trap
-----------------------------------------------------
The v6 portfolio + Laplacian gets stuck in a local minimum on hard
benches because all 8 worker seeds start from the same .plc init and
explore only nearby states. ePlace warm-start gives a globally-spread
configuration that's macroscopically different from .plc; v6 portfolio
then refines from this new basin. If ePlace+v6 lands at a different
local minimum than v6-from-.plc, we get basin diversity that single-
seed perturbations can't achieve.

Implementation
--------------
- 2D Poisson solve via numpy.fft (O(N log N) where N = grid_col × grid_row)
- Density grid: scatter-add macro footprints (each macro contributes
  fractional overlap to cells in its footprint)
- Force interpolation: bilinear sampling of -∇ψ at macro positions
- Gradient descent with cosine-annealed lr, optional Nesterov accel.
- Hards held proximal (small force, doesn't move much); softs free.
"""
from __future__ import annotations
import math
import time
import numpy as np


def _density_grid(
    macro_pos: np.ndarray,    # (n_total, 2) lower-left corners
    macro_w: np.ndarray,
    macro_h: np.ndarray,
    grid_col: int, grid_row: int,
    grid_w: float, grid_h: float,
) -> np.ndarray:
    """Build density[r, c] = sum of macro footprint overlap in cell (r,c)
    divided by cell area. Returns (grid_row, grid_col) array.

    Each macro contributes overlap_area(macro, cell) / cell_area to each
    cell in its footprint. Bilinear-style fractional contribution.
    """
    cell_area = grid_w * grid_h
    density = np.zeros((grid_row, grid_col), dtype=np.float64)
    for i in range(macro_pos.shape[0]):
        x0 = macro_pos[i, 0]
        y0 = macro_pos[i, 1]
        x1 = x0 + macro_w[i]
        y1 = y0 + macro_h[i]
        # Cells the footprint touches (inclusive)
        c0 = max(0, int(x0 / grid_w))
        c1 = min(grid_col - 1, int(x1 / grid_w))
        r0 = max(0, int(y0 / grid_h))
        r1 = min(grid_row - 1, int(y1 / grid_h))
        for c in range(c0, c1 + 1):
            cell_x0 = c * grid_w
            cell_x1 = cell_x0 + grid_w
            ow = min(x1, cell_x1) - max(x0, cell_x0)
            if ow <= 0:
                continue
            for r in range(r0, r1 + 1):
                cell_y0 = r * grid_h
                cell_y1 = cell_y0 + grid_h
                oh = min(y1, cell_y1) - max(y0, cell_y0)
                if oh <= 0:
                    continue
                density[r, c] += (ow * oh) / cell_area
    return density


def _solve_poisson_fft(
    rhs: np.ndarray,                # (R, C) right-hand side
    grid_w: float, grid_h: float,
) -> np.ndarray:
    """Solve ∇²ψ = -rhs via 2D FFT with periodic BCs.

    The 2D Laplacian in Fourier space is:
        -L̂(kx, ky) = (2 - 2 cos(2π kx / C)) / dx²
                   + (2 - 2 cos(2π ky / R)) / dy²

    So ψ̂ = rhŝ / L̂ (with the (0,0) mode set to 0 since ψ is defined
    up to a constant; the periodic-BC nullspace is the constant mode).

    Returns ψ of shape (R, C).
    """
    R, C = rhs.shape
    rhs_hat = np.fft.fft2(rhs)
    kx = np.fft.fftfreq(C) * C    # 0, 1, ..., C/2, -C/2+1, ..., -1
    ky = np.fft.fftfreq(R) * R
    KX, KY = np.meshgrid(kx, ky)
    # Discrete 5-point Laplacian eigenvalues
    Lhat = ((2.0 - 2.0 * np.cos(2 * np.pi * KX / C)) / (grid_w ** 2)
            + (2.0 - 2.0 * np.cos(2 * np.pi * KY / R)) / (grid_h ** 2))
    Lhat[0, 0] = 1.0   # avoid 0/0; (0,0) mode set to 0 below
    psi_hat = rhs_hat / Lhat
    psi_hat[0, 0] = 0.0
    psi = np.real(np.fft.ifft2(psi_hat))
    return psi


def _grad_psi(psi: np.ndarray, grid_w: float, grid_h: float) -> tuple:
    """Central-difference gradient of ψ. Returns (grad_x, grad_y),
    each shape (R, C). Uses periodic BCs to match the Poisson solve.
    """
    R, C = psi.shape
    grad_x = (np.roll(psi, -1, axis=1) - np.roll(psi, 1, axis=1)) / (2 * grid_w)
    grad_y = (np.roll(psi, -1, axis=0) - np.roll(psi, 1, axis=0)) / (2 * grid_h)
    return grad_x, grad_y


def _interpolate_force(
    grad_x: np.ndarray, grad_y: np.ndarray,
    macro_pos: np.ndarray, macro_w: np.ndarray, macro_h: np.ndarray,
    grid_w: float, grid_h: float,
) -> np.ndarray:
    """Bilinear-interpolate the force field at each macro center.
    Returns (n, 2) force vector per macro.
    """
    R, C = grad_x.shape
    cx = macro_pos[:, 0] + macro_w / 2.0
    cy = macro_pos[:, 1] + macro_h / 2.0
    fx_cell = cx / grid_w - 0.5
    fy_cell = cy / grid_h - 0.5
    fx0 = np.floor(fx_cell).astype(np.int64) % C
    fy0 = np.floor(fy_cell).astype(np.int64) % R
    fx1 = (fx0 + 1) % C
    fy1 = (fy0 + 1) % R
    wx = fx_cell - np.floor(fx_cell)
    wy = fy_cell - np.floor(fy_cell)
    # Bilinear interpolation
    g00x = grad_x[fy0, fx0]
    g01x = grad_x[fy0, fx1]
    g10x = grad_x[fy1, fx0]
    g11x = grad_x[fy1, fx1]
    g00y = grad_y[fy0, fx0]
    g01y = grad_y[fy0, fx1]
    g10y = grad_y[fy1, fx0]
    g11y = grad_y[fy1, fx1]
    gx = ((1 - wx) * (1 - wy) * g00x + wx * (1 - wy) * g01x
          + (1 - wx) * wy * g10x + wx * wy * g11x)
    gy = ((1 - wx) * (1 - wy) * g00y + wx * (1 - wy) * g01y
          + (1 - wx) * wy * g10y + wx * wy * g11y)
    # Force = -grad psi  (descend the potential)
    return -np.stack([gx, gy], axis=1)


def _hpwl_centroid_force(
    pos_centers: np.ndarray,          # (n_total, 2) center coords
    pin_macro: np.ndarray,            # (n_pins,) int (-1 = port)
    pin_xoff: np.ndarray,             # (n_pins,) float
    pin_yoff: np.ndarray,
    net_starts: np.ndarray,           # (n_nets+1,) int
    net_weight: np.ndarray,           # (n_nets,) float
) -> np.ndarray:
    """HPWL-gradient force: pull each macro toward the weighted centroid
    of its connected nets. For each net n with k pins:
        net_centroid = mean of pin_positions
    For each pin on a macro, force_pin = (net_centroid - pin_pos).
    The macro's net force = sum of pin forces × net_weight.

    This is the gradient of the quadratic HPWL surrogate (clique model):
        ½ x^T L x where L is the netlist Laplacian
        ∇ = -L x  →  per-macro force = -L_row · x = average of net pulls

    Returns (n_total, 2) force vector — the desired displacement direction
    per macro, with magnitude proportional to the net-pull pressure.
    """
    n_total = pos_centers.shape[0]
    # Compute pin world positions
    is_port = (pin_macro < 0)
    safe = np.where(is_port, 0, pin_macro)
    pin_x = np.where(is_port, pin_xoff,
                     pos_centers[safe, 0] + pin_xoff)
    pin_y = np.where(is_port, pin_yoff,
                     pos_centers[safe, 1] + pin_yoff)
    # Compute per-net centroid via segment_mean. Use repeat_interleave
    # to map pins to nets, scatter for sums.
    n_nets = int(net_weight.shape[0])
    net_lengths = (net_starts[1:] - net_starts[:-1]).astype(np.int64)
    pin_to_net = np.repeat(np.arange(n_nets, dtype=np.int64), net_lengths)
    net_count = np.zeros(n_nets, dtype=np.float64)
    net_sum_x = np.zeros(n_nets, dtype=np.float64)
    net_sum_y = np.zeros(n_nets, dtype=np.float64)
    np.add.at(net_count, pin_to_net, 1.0)
    np.add.at(net_sum_x, pin_to_net, pin_x)
    np.add.at(net_sum_y, pin_to_net, pin_y)
    net_count = np.maximum(net_count, 1.0)
    net_cx = net_sum_x / net_count
    net_cy = net_sum_y / net_count
    # Per-pin pull = (net_cx - pin_x) * net_weight
    pull_x = (net_cx[pin_to_net] - pin_x) * net_weight[pin_to_net]
    pull_y = (net_cy[pin_to_net] - pin_y) * net_weight[pin_to_net]
    # Aggregate per-macro force (skip port pins which don't act on macros)
    force = np.zeros_like(pos_centers)
    np.add.at(force[:, 0], safe[~is_port], pull_x[~is_port])
    np.add.at(force[:, 1], safe[~is_port], pull_y[~is_port])
    return force


def eplace_warmstart(
    init_pos: np.ndarray,         # (n_total, 2) initial positions, lower-left
    macro_w: np.ndarray,
    macro_h: np.ndarray,
    canvas_w: float, canvas_h: float,
    grid_col: int, grid_row: int,
    *,
    n_steps: int = 100,
    lr_frac_canvas: float = 0.005,
    n_hard: int = 0,
    hard_inertia: float = 10.0,
    nesterov: bool = True,
    # HPWL-aware mode (DREAMPlace formulation): when net data is provided,
    # add the HPWL-gradient (net-centroid pull) to the force field.
    pin_macro: np.ndarray | None = None,
    pin_xoff: np.ndarray | None = None,
    pin_yoff: np.ndarray | None = None,
    net_starts: np.ndarray | None = None,
    net_weight: np.ndarray | None = None,
    alpha_hpwl: float = 0.0,      # weight on HPWL force; 0 = HPWL-blind
    beta_density: float = 1.0,    # weight on density spreading force
    verbose: bool = False,
) -> tuple[np.ndarray, dict]:
    """Run electrostatic-analogy placement for n_steps, return updated
    positions (lower-left). Hard macros are coupled to their initial
    position with a quadratic spring of strength `hard_inertia` to
    prevent them from drifting (preserves topology of the hard floorplan).

    Parameters
    ----------
    init_pos : (n_total, 2) float, lower-left positions.
    n_steps : number of gradient-descent steps.
    lr_frac_canvas : learning rate as fraction of canvas_diag.
    n_hard : number of hard macros (first n_hard rows are hard).
    hard_inertia : multiplier on the proximal spring force for hards.
        Higher = hards barely move; lower = hards drift like softs.
    nesterov : enable Nesterov-accelerated GD.
    """
    grid_w = canvas_w / grid_col
    grid_h = canvas_h / grid_row
    canvas_diag = math.hypot(canvas_w, canvas_h)
    lr0 = lr_frac_canvas * canvas_diag

    pos = init_pos.copy().astype(np.float64)
    init_pos_const = init_pos.copy().astype(np.float64)
    n_total = pos.shape[0]
    velocity = np.zeros_like(pos)
    history = {"step": [], "max_density": [], "mean_density": [],
                "force_norm": []}
    target_density = 1.0   # uniform; absolute scale handled by lr

    have_hpwl = (alpha_hpwl > 0.0 and pin_macro is not None
                 and net_starts is not None)

    t0 = time.time()
    for step in range(n_steps):
        # Build density grid
        density = _density_grid(pos, macro_w, macro_h,
                                  grid_col, grid_row, grid_w, grid_h)
        # Center: subtract mean to get rhs with sum = 0 (compatible
        # with periodic-BC Poisson)
        rhs = density - density.mean()
        # Solve Poisson
        psi = _solve_poisson_fft(rhs, grid_w, grid_h)
        # Density-spreading force
        grad_x, grad_y = _grad_psi(psi, grid_w, grid_h)
        force_density = _interpolate_force(
            grad_x, grad_y, pos, macro_w, macro_h, grid_w, grid_h)
        # HPWL-pull force (DREAMPlace style): pull each macro toward its
        # weighted-net centroid. Computed in CENTER coords, applied to
        # lower-left coords (forces are translation-invariant).
        if have_hpwl:
            pos_centers = pos.copy()
            pos_centers[:, 0] += macro_w / 2.0
            pos_centers[:, 1] += macro_h / 2.0
            force_hpwl = _hpwl_centroid_force(
                pos_centers, pin_macro, pin_xoff, pin_yoff,
                net_starts, net_weight)
        else:
            force_hpwl = np.zeros_like(pos)

        force = beta_density * force_density + alpha_hpwl * force_hpwl

        # Hard inertia: replace force on hards with proximal spring
        if n_hard > 0:
            spring = -(pos[:n_hard] - init_pos_const[:n_hard]) * hard_inertia
            force[:n_hard] = spring

        # Cosine-anneal lr
        lr_step = lr0 * 0.5 * (1.0 + math.cos(math.pi * step / n_steps))

        if nesterov:
            momentum = 0.9
            velocity = momentum * velocity + lr_step * force
            pos = pos + velocity
        else:
            pos = pos + lr_step * force

        # Clip to canvas
        np.clip(pos[:, 0], 0.0, canvas_w - macro_w, out=pos[:, 0])
        np.clip(pos[:, 1], 0.0, canvas_h - macro_h, out=pos[:, 1])

        if step % max(1, n_steps // 10) == 0 or step == n_steps - 1:
            history["step"].append(step)
            history["max_density"].append(float(density.max()))
            history["mean_density"].append(float(density.mean()))
            history["force_norm"].append(float(np.linalg.norm(force)))
            if verbose:
                print(f"    [eplace] step {step}: max_dens="
                      f"{density.max():.3f} mean={density.mean():.3f} "
                      f"||F||={np.linalg.norm(force):.3f} "
                      f"lr={lr_step:.3f} ({time.time()-t0:.1f}s)",
                      flush=True)

    if verbose:
        print(f"  [eplace] {n_steps} steps in {time.time()-t0:.1f}s; "
              f"max_dens {history['max_density'][0]:.2f} → "
              f"{history['max_density'][-1]:.2f}", flush=True)
    return pos, history
