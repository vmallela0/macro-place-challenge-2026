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


def electrostatic_spread_homotopy(
    bench,
    incr,
    *,
    n_iters: int = 500,
    n_stages: int = 20,
    lr_frac_canvas: float = 0.005,
    lambda_0: float = 0.01,
    lambda_f: float = 50.0,
    tau_lse: float = 50.0,
    seed: int = 42,
    init_from_plc: bool = True,
    soft_only: bool = False,
    snapshot_every: int = 20,
    verbose: bool = False,
):
    """albania2 Bet A: homotopy spreader with cosine λ-ramp.

    Mathematical basis: at each fixed λ, the cost
        f_λ(x) = HPWL_LSE(x) + λ · electro_norm(x)
    has a unique global minimizer x*(λ) (sum of convex terms). The
    curve {x*(λ) : λ ∈ [λ_0, λ_f]} is a continuous deformation path.
    Slow ramping along this path with Adam tracking-steps amounts to a
    homotopy method (a.k.a. continuation method) — guaranteed convergence
    to the global x*(λ_f) when ramp speed ≪ Adam tracking speed.

    vs. the previous fixed-λ Phase 0: that one tried to jump directly to
    x*(0.5) from a bad init. The basin of attraction at any fixed λ is
    narrow when initialized far from x*(λ); the ramp solves that by
    starting at λ_0 (where x*(λ_0) is wide-basin, HPWL-clustered) and
    tracking continuously to x*(λ_f).

    Schedule: cosine ramp over n_stages stages, n_iters_per_stage Adam
    steps each:
        λ(k) = λ_0 + (λ_f − λ_0) · (1 − cos(π·k/(n_stages−1))) / 2
        n_iters_per_stage = n_iters // n_stages
    """
    from _smooth_proxy import (lse_hpwl_vectorized, build_pin_to_net,
                                  cvar_smooth, softplus_mu)
    from _cell_window import (build_window_indices, smooth_density_grid)

    device = _select_device()
    n_total = int(np.asarray(incr.macro_pos).shape[0])
    n_hard = bench.num_hard_macros
    cw, ch = float(incr.cw), float(incr.ch)
    canvas_diag = float(np.hypot(cw, ch))

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

    # Init: from .plc (default). Random init fights the homotopy because
    # x(0) is far from x*(λ_0) → first stage pulls macros toward HPWL
    # minimum (= clustered) before density penalty kicks in, defeating
    # the spread purpose.
    if init_from_plc:
        init_pos = np.asarray(incr.macro_pos).copy().astype(np.float32)
    else:
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
            plc_pos = np.asarray(incr.macro_pos).copy()
            init_pos[:n_hard] = plc_pos[:n_hard]

    macro_pos = torch.tensor(init_pos, dtype=torch.float32,
                                device=device, requires_grad=True)
    # t is the CVaR threshold variable — learned jointly with positions.
    # At Adam's optimum, t equals the (n_cells - K_d)-th order statistic.
    t_dens = torch.tensor(0.0, dtype=torch.float32,
                              device=device, requires_grad=True)

    if verbose:
        print(f"  [phase0.homotopy] init={'plc' if init_from_plc else 'random'}, "
              f"n_total={n_total}, canvas={cw:.1f}×{ch:.1f}, "
              f"λ ramp [{lambda_0:.3f} → {lambda_f:.3f}] over "
              f"{n_stages} stages × {n_iters//n_stages} steps",
              flush=True)

    cell_idx_d, _ = build_window_indices(
        macro_pos.detach(), macro_w_t, macro_h_t,
        grid_col=incr.grid_col, grid_row=incr.grid_row,
        grid_w=incr.grid_width, grid_h=incr.grid_height,
        margin_cells=4)

    lr = lr_frac_canvas * canvas_diag
    opt = torch.optim.Adam([macro_pos, t_dens], lr=lr)

    n_iters_per_stage = max(1, int(n_iters // n_stages))
    total_steps = n_iters_per_stage * n_stages

    # Cosine ramp helper: stage k ∈ {0..n_stages-1} → λ_k.
    def lam_at_stage(k: int) -> float:
        if n_stages <= 1:
            return float(lambda_f)
        u = float(k) / float(n_stages - 1)        # 0 → 1
        cos_factor = 0.5 * (1.0 - float(np.cos(np.pi * u)))
        return float(lambda_0 + (lambda_f - lambda_0) * cos_factor)

    t0 = time.time()
    step = 0
    for stage in range(n_stages):
        lam = lam_at_stage(stage)
        for substep in range(n_iters_per_stage):
            opt.zero_grad()
            if step > 0 and step % snapshot_every == 0:
                cell_idx_d, _ = build_window_indices(
                    macro_pos.detach(), macro_w_t, macro_h_t,
                    grid_col=incr.grid_col, grid_row=incr.grid_row,
                    grid_w=incr.grid_width, grid_h=incr.grid_height,
                    margin_cells=4)
            is_port = (pin_macro_t < 0)
            safe = torch.where(is_port, torch.zeros_like(pin_macro_t),
                                  pin_macro_t)
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
            # CVaR top-K density — directly approximates the exact
            # leaderboard cost (which IS top-K mean of grid_density).
            # Avoids the surrogate-exact divergence from electrostatic
            # (Poisson) energy. K_d = 10% of cells matches the scorer.
            K_d = max(1, int(0.10 * incr.n_cells))
            density_term = cvar_smooth(rho, K_d, t_dens, mu=100.0)
            loss = hpwl + lam * density_term
            loss.backward()
            if soft_only and n_hard > 0:
                with torch.no_grad():
                    if macro_pos.grad is not None:
                        macro_pos.grad[:n_hard] = 0.0
            opt.step()
            with torch.no_grad():
                half_w = macro_w_t / 2.0
                half_h = macro_h_t / 2.0
                macro_pos[:, 0] = torch.clamp(macro_pos[:, 0],
                                                  min=half_w,
                                                  max=cw - half_w)
                macro_pos[:, 1] = torch.clamp(macro_pos[:, 1],
                                                  min=half_h,
                                                  max=ch - half_h)
            step += 1
        if verbose:
            print(f"  [phase0.homotopy] stage {stage:2d}/{n_stages-1}: "
                  f"λ={lam:.4f}  loss={loss.item():.4f}  "
                  f"hpwl={hpwl.item():.4f}  density={density_term.item():.4f} "
                  f"({time.time()-t0:.1f}s)", flush=True)

    final_pos = macro_pos.detach().cpu().numpy().astype(np.float32)
    if verbose:
        print(f"  [phase0.homotopy] DONE in {time.time()-t0:.1f}s, "
              f"{step} Adam steps", flush=True)
    return final_pos


def electrostatic_spread_homotopy_overlap_aware(
    bench, incr,
    *,
    n_iters: int = 500, n_stages: int = 20,
    lr_frac_canvas: float = 0.001,
    lambda_0: float = 0.05, lambda_f: float = 2.0,
    overlap_weight_0: float = 0.0,
    overlap_weight_f: float = 5.0,
    tau_lse: float = 50.0,
    seed: int = 42,
    init_from_plc: bool = True,
    soft_only: bool = False,
    snapshot_every: int = 20,
    overlap_recompute_every: int = 25,
    verbose: bool = False,
):
    """Phase 0 with overlap-aware penalty in addition to density.

    Adds soft margin penalty:
        L_overlap = Σ_{(i,j) overlapping} (max(0, sep_x_ij - |dx|))² +
                    (max(0, sep_y_ij - |dy|))²
    where sep_x_ij = (w_i + w_j)/2, sep_y_ij = (h_i + h_j)/2.

    Overlap pairs recomputed every `overlap_recompute_every` steps via
    O(N²) pairwise check on hard macros (cheap for N≤2000).

    Cosine ramp on BOTH λ (density) and overlap_weight, so the optimizer
    starts HPWL-driven, gradually adds density push-apart, finally adds
    explicit overlap repair.
    """
    from _smooth_proxy import (lse_hpwl_vectorized, build_pin_to_net,
                                  cvar_smooth)
    from _cell_window import (build_window_indices, smooth_density_grid)

    device = _select_device()
    n_total = int(np.asarray(incr.macro_pos).shape[0])
    n_hard = bench.num_hard_macros
    cw, ch = float(incr.cw), float(incr.ch)
    canvas_diag = float(np.hypot(cw, ch))

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

    if init_from_plc:
        init_pos = np.asarray(incr.macro_pos).copy().astype(np.float32)
    else:
        rng = np.random.RandomState(seed)
        init_pos = np.zeros((n_total, 2), dtype=np.float32)
        macro_w_np = np.asarray(incr.macro_w)
        macro_h_np = np.asarray(incr.macro_h)
        for m in range(n_total):
            hw = macro_w_np[m] / 2.0; hh = macro_h_np[m] / 2.0
            init_pos[m, 0] = rng.uniform(hw, max(hw + 1e-3, cw - hw))
            init_pos[m, 1] = rng.uniform(hh, max(hh + 1e-3, ch - hh))

    macro_pos = torch.tensor(init_pos, dtype=torch.float32,
                                device=device, requires_grad=True)
    t_dens = torch.tensor(0.0, dtype=torch.float32,
                              device=device, requires_grad=True)

    # Pairwise overlap detection on HARD macros (soft macros allowed to
    # overlap by competition spec).
    macro_w_np = np.asarray(incr.macro_w)[:n_hard]
    macro_h_np = np.asarray(incr.macro_h)[:n_hard]
    sep_x_full = ((macro_w_np[:, None] + macro_w_np[None, :]) / 2.0).astype(np.float32)
    sep_y_full = ((macro_h_np[:, None] + macro_h_np[None, :]) / 2.0).astype(np.float32)

    def find_overlapping_pairs(pos_np: np.ndarray) -> torch.Tensor:
        """Returns shape (n_pairs, 2) torch.long; or empty tensor if none."""
        if n_hard < 2:
            return torch.zeros((0, 2), dtype=torch.long, device=device)
        p = pos_np[:n_hard]
        dx = np.abs(p[:, 0:1] - p[:, 0:1].T)
        dy = np.abs(p[:, 1:2] - p[:, 1:2].T)
        ov = (dx < sep_x_full) & (dy < sep_y_full)
        np.fill_diagonal(ov, False)
        ov = np.triu(ov, k=1)
        pairs = np.argwhere(ov)
        if pairs.size == 0:
            return torch.zeros((0, 2), dtype=torch.long, device=device)
        return torch.tensor(pairs, dtype=torch.long, device=device)

    cell_idx_d, _ = build_window_indices(
        macro_pos.detach(), macro_w_t, macro_h_t,
        grid_col=incr.grid_col, grid_row=incr.grid_row,
        grid_w=incr.grid_width, grid_h=incr.grid_height,
        margin_cells=4)

    lr = lr_frac_canvas * canvas_diag
    opt = torch.optim.Adam([macro_pos, t_dens], lr=lr)
    n_iters_per_stage = max(1, int(n_iters // n_stages))

    def lam_at_stage(k, lo, hi):
        if n_stages <= 1: return float(hi)
        u = float(k) / float(n_stages - 1)
        cf = 0.5 * (1.0 - float(np.cos(np.pi * u)))
        return float(lo + (hi - lo) * cf)

    if verbose:
        print(f"  [phase0.ovaware] init={'plc' if init_from_plc else 'random'}, "
              f"n_total={n_total}, n_hard={n_hard}, "
              f"λ ramp [{lambda_0:.3f}→{lambda_f:.3f}], "
              f"ov ramp [{overlap_weight_0:.3f}→{overlap_weight_f:.3f}]",
              flush=True)

    macro_w_t_h = macro_w_t[:n_hard]
    macro_h_t_h = macro_h_t[:n_hard]
    overlap_pairs = find_overlapping_pairs(macro_pos.detach().cpu().numpy())

    t0 = time.time()
    step = 0
    for stage in range(n_stages):
        lam = lam_at_stage(stage, lambda_0, lambda_f)
        ov_w = lam_at_stage(stage, overlap_weight_0, overlap_weight_f)
        for substep in range(n_iters_per_stage):
            opt.zero_grad()
            if step > 0 and step % snapshot_every == 0:
                cell_idx_d, _ = build_window_indices(
                    macro_pos.detach(), macro_w_t, macro_h_t,
                    grid_col=incr.grid_col, grid_row=incr.grid_row,
                    grid_w=incr.grid_width, grid_h=incr.grid_height,
                    margin_cells=4)
            if step > 0 and step % overlap_recompute_every == 0:
                overlap_pairs = find_overlapping_pairs(
                    macro_pos.detach().cpu().numpy())

            is_port = (pin_macro_t < 0)
            safe = torch.where(is_port, torch.zeros_like(pin_macro_t),
                                  pin_macro_t)
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
            K_d = max(1, int(0.10 * incr.n_cells))
            density_term = cvar_smooth(rho, K_d, t_dens, mu=100.0)

            # Overlap repair penalty (only on hard macros).
            ov_term = torch.zeros((), dtype=torch.float32, device=device)
            if overlap_pairs.shape[0] > 0 and ov_w > 1e-6:
                a_idx = overlap_pairs[:, 0]
                b_idx = overlap_pairs[:, 1]
                pa = macro_pos[a_idx]
                pb = macro_pos[b_idx]
                dx = (pa[:, 0] - pb[:, 0]).abs()
                dy = (pa[:, 1] - pb[:, 1]).abs()
                sep_x = (macro_w_t_h[a_idx] + macro_w_t_h[b_idx]) / 2.0
                sep_y = (macro_h_t_h[a_idx] + macro_h_t_h[b_idx]) / 2.0
                # Penalize the SHORTER axis only — that's the axis along
                # which moving will resolve the overlap with minimum
                # disruption. Squared-soft-margin per pair.
                margin_x = torch.relu(sep_x - dx)
                margin_y = torch.relu(sep_y - dy)
                # min margin = the resolving axis
                margin = torch.minimum(margin_x, margin_y)
                ov_term = (margin * margin).sum()
                # Normalize by canvas² so it's scale-invariant
                ov_term = ov_term / (canvas_diag * canvas_diag)

            loss = hpwl + lam * density_term + ov_w * ov_term
            loss.backward()
            if soft_only and n_hard > 0:
                with torch.no_grad():
                    if macro_pos.grad is not None:
                        macro_pos.grad[:n_hard] = 0.0
            opt.step()
            with torch.no_grad():
                half_w = macro_w_t / 2.0
                half_h = macro_h_t / 2.0
                macro_pos[:, 0] = torch.clamp(macro_pos[:, 0],
                                                  min=half_w,
                                                  max=cw - half_w)
                macro_pos[:, 1] = torch.clamp(macro_pos[:, 1],
                                                  min=half_h,
                                                  max=ch - half_h)
            step += 1
        if verbose:
            print(f"  [phase0.ovaware] stage {stage:2d}/{n_stages-1}: "
                  f"λ={lam:.4f}  ov_w={ov_w:.4f}  "
                  f"loss={loss.item():.4f}  hpwl={hpwl.item():.4f}  "
                  f"density={density_term.item():.4f}  "
                  f"ov_pairs={overlap_pairs.shape[0]}  "
                  f"ov_term={ov_term.item():.4f} ({time.time()-t0:.1f}s)",
                  flush=True)

    final_pos = macro_pos.detach().cpu().numpy().astype(np.float32)
    if verbose:
        print(f"  [phase0.ovaware] DONE in {time.time()-t0:.1f}s, "
              f"{step} steps", flush=True)
    return final_pos
