"""Phase 0: DREAMPlace-style electrostatic spreader (warm-start).

Replaces .plc init with a globally-density-balanced placement obtained
by running Adam on the smooth surrogate from RANDOM init.

Why this works (vs the failed Adam Phase 4.5 attempt):
- Failed Adam 4.5 tried to refine FROM .plc init using smooth-surrogate
  gradients. .plc init is already in a local minimum where surrogate
  gradients diverge from exact-cost gradients → drift into worse exact-
  cost regions.
- Phase 0 starts from RANDOM init. There's no local minimum to escape,
  just SPREADING. Under the smooth surrogate `HPWL_LSE + α·electro_norm`,
  the dynamics are globally convergent (mean-field equilibrium is
  unique; Earnshaw's theorem with confinement). Adam reliably finds
  the global density-balanced placement.
- The output is overlap-RESOLVABLE (close-packed but not collapsed),
  with HPWL roughly minimized AND density roughly uniform. v4's
  push-apart + legalize handles the residual overlaps.

Algorithm:
1. Random init: macros uniform in canvas (with half-extent margin)
2. Adam loop on:  HPWL_LSE(x) + α · electrostatic_density_energy_normalized(x)
   - Cosine LR schedule
   - Per-step canvas projection
3. Return final positions

For Tier 1: better global density structure at v4-start → cleaner
basin, better post-Lap, expected -0.02 to -0.05 mean improvement.

For Tier 2: globally-balanced density = clean routing channels =
real WNS improvement on critical paths.
"""
from __future__ import annotations
import time
import numpy as np
import torch


def _select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def electrostatic_spread(
    bench,
    incr,
    *,
    n_iters: int = 500,
    lr_frac_canvas: float = 0.01,
    electro_weight: float = 0.5,
    tau_lse: float = 50.0,
    snapshot_every: int = 20,
    seed: int = 42,
    soft_only: bool = False,
    verbose: bool = False,
):
    """Phase 0 electrostatic spreader. Returns (n_total, 2) np.float32.

    Args:
        bench: Benchmark object
        incr: IncrementalEvaluator (provides pin/net structure)
        n_iters: Adam steps. 500 is enough for convergence on ibm-scale.
        lr_frac_canvas: Adam learning rate as fraction of canvas_diag.
        electro_weight: Weight on electrostatic energy term (0.5 matches
            our verified Hessian config).
        tau_lse: LSE temperature for HPWL smoothing.
        snapshot_every: Re-snapshot density window every N steps.
        seed: random init seed.
        soft_only: If True, only optimize soft macros (hards stay at .plc init).
        verbose: print progress every 50 steps.
    """
    from _smooth_proxy import (lse_hpwl_vectorized, build_pin_to_net)
    from _cell_window import (build_window_indices, smooth_density_grid,
                                 electrostatic_density_energy_normalized)

    device = _select_device()
    n_total = int(np.asarray(incr.macro_pos).shape[0])
    n_hard = bench.num_hard_macros
    cw, ch = float(incr.cw), float(incr.ch)
    canvas_diag = float(np.hypot(cw, ch))

    # Constants
    pin_macro_t = torch.tensor(np.asarray(incr.pin_macro), dtype=torch.long,
                                  device=device)
    pin_xoff_t = torch.tensor(np.asarray(incr.pin_xoff), dtype=torch.float32,
                                 device=device)
    pin_yoff_t = torch.tensor(np.asarray(incr.pin_yoff), dtype=torch.float32,
                                 device=device)
    net_starts_t = torch.tensor(np.asarray(incr.net_starts), dtype=torch.long,
                                   device=device)
    net_weight_t = torch.tensor(np.asarray(incr.net_weight),
                                   dtype=torch.float32, device=device)
    macro_w_t = torch.tensor(np.asarray(incr.macro_w), dtype=torch.float32,
                                device=device)
    macro_h_t = torch.tensor(np.asarray(incr.macro_h), dtype=torch.float32,
                                device=device)
    pin_to_net_t = build_pin_to_net(net_starts_t)
    n_nets = int(net_weight_t.shape[0])
    net_cnt = float(incr.net_cnt)

    # Random init within canvas (respecting macro extents).
    rng = np.random.RandomState(seed)
    init_pos = np.zeros((n_total, 2), dtype=np.float32)
    macro_w_np = np.asarray(incr.macro_w)
    macro_h_np = np.asarray(incr.macro_h)
    for m in range(n_total):
        hw = macro_w_np[m] / 2.0
        hh = macro_h_np[m] / 2.0
        init_pos[m, 0] = rng.uniform(hw, max(hw + 1e-3, cw - hw))
        init_pos[m, 1] = rng.uniform(hh, max(hh + 1e-3, ch - hh))

    if soft_only:
        # Keep hards at .plc init, randomize only softs.
        plc_pos = np.asarray(incr.macro_pos).copy()
        init_pos[:n_hard] = plc_pos[:n_hard]

    macro_pos = torch.tensor(init_pos, dtype=torch.float32,
                                device=device, requires_grad=True)

    if verbose:
        print(f"  [phase0] random init: {n_total} macros in canvas "
              f"{cw:.1f}×{ch:.1f}", flush=True)

    # Density grid window (re-snapshotted periodically)
    cell_idx_d, _ = build_window_indices(
        macro_pos.detach(), macro_w_t, macro_h_t,
        grid_col=incr.grid_col, grid_row=incr.grid_row,
        grid_w=incr.grid_width, grid_h=incr.grid_height,
        margin_cells=4)

    # Adam optimizer (only on movable positions)
    if soft_only:
        # Only soft macros are trainable.
        # Use a mask on the gradient.
        params = [macro_pos]
    else:
        params = [macro_pos]
    lr = lr_frac_canvas * canvas_diag
    opt = torch.optim.Adam(params, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_iters)

    t0 = time.time()
    history = []

    for step in range(n_iters):
        opt.zero_grad()

        # Re-snapshot density window every snapshot_every steps.
        if step > 0 and step % snapshot_every == 0:
            cell_idx_d, _ = build_window_indices(
                macro_pos.detach(), macro_w_t, macro_h_t,
                grid_col=incr.grid_col, grid_row=incr.grid_row,
                grid_w=incr.grid_width, grid_h=incr.grid_height,
                margin_cells=4)

        # Compute smooth proxy
        is_port = (pin_macro_t < 0)
        safe = torch.where(is_port, torch.zeros_like(pin_macro_t), pin_macro_t)
        macro_xy = macro_pos[safe]
        pin_x = torch.where(is_port, pin_xoff_t,
                              macro_xy[:, 0] + pin_xoff_t)
        pin_y = torch.where(is_port, pin_yoff_t,
                              macro_xy[:, 1] + pin_yoff_t)
        hpwl = lse_hpwl_vectorized(
            pin_x, pin_y, pin_to_net_t, net_weight_t, n_nets,
            cw=cw, ch=ch, net_cnt=net_cnt, tau_lse=tau_lse)

        rho = smooth_density_grid(
            macro_pos, macro_w_t, macro_h_t, cell_idx_d,
            incr.grid_col, incr.grid_row,
            incr.grid_width, incr.grid_height,
            n_cells=incr.n_cells, cell_area=incr.grid_area, mu=100.0)
        density_term = electrostatic_density_energy_normalized(
            rho, incr.grid_row, incr.grid_col,
            grid_w=float(incr.grid_width),
            grid_h=float(incr.grid_height))

        loss = hpwl + electro_weight * density_term

        loss.backward()

        # Zero hard-macro gradients if soft_only
        if soft_only:
            with torch.no_grad():
                if macro_pos.grad is not None:
                    macro_pos.grad[:n_hard] = 0.0

        opt.step()
        sched.step()

        # Project onto canvas-interior (per-macro half-extent margin)
        with torch.no_grad():
            half_w = macro_w_t / 2.0
            half_h = macro_h_t / 2.0
            macro_pos[:, 0] = torch.clamp(macro_pos[:, 0],
                                              min=half_w,
                                              max=cw - half_w)
            macro_pos[:, 1] = torch.clamp(macro_pos[:, 1],
                                              min=half_h,
                                              max=ch - half_h)

        if verbose and step % 50 == 0:
            history.append((step, float(loss.item()),
                           float(hpwl.item()), float(density_term.item())))
            print(f"  [phase0] step {step:4d}/{n_iters}: loss={loss.item():.4f} "
                  f"hpwl={hpwl.item():.4f} density={density_term.item():.4f} "
                  f"({time.time()-t0:.1f}s)", flush=True)

    final_pos = macro_pos.detach().cpu().numpy().astype(np.float32)
    if verbose:
        print(f"  [phase0] DONE in {time.time()-t0:.1f}s, "
              f"{n_iters} Adam steps", flush=True)
    return final_pos
