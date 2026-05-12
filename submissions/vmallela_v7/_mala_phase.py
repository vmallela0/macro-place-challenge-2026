"""Plan-B helper: MALA refinement phase plugin for placer.py.

When invoked, runs Metropolis-Adjusted Langevin search starting from
the current macro_pos, using the smooth_proxy_call's gradient as a
proposal direction and the exact PlacementCost as the acceptance
criterion. Returns (new_pos, new_cost) — caller applies the standard
strict-improvement gate.

This is meant to be inserted into placer.py's main loop AFTER the
Hessian phase, e.g.:

    if (os.environ.get("PLACER_V7_MALA", "0") == "1" and overlaps == 0):
        from _mala_phase import run_mala_phase
        new_pos, new_cost = run_mala_phase(
            portfolio_pos, portfolio_cost, bench_path, ...)
        if new_cost < portfolio_cost - 1e-7:
            portfolio_pos, portfolio_cost = new_pos, new_cost

Wire-in code lives in placer.py; this module just provides the
self-contained orchestration so the wire-in stays small.
"""
from __future__ import annotations
import os
import time
import numpy as np
import torch

# ROOT computed for sys.path is handled by placer.py at import time.


def run_mala_phase(
    current_pos,                  # torch.Tensor (n_total, 2)
    current_cost: float,
    bench_path: str,
    *,
    n_steps: int = 1500,
    step_size_frac: float = 0.003,
    temp_init_frac: float = 0.003,
    temp_decay: float = 0.999,
    n_burn: int = 50,
    cap_disp_frac: float = 0.20,
    verbose: bool = False,
) -> tuple[torch.Tensor, float, dict]:
    """Run MALA on the smooth surrogate, gated by exact PlacementCost.

    Wraps the lower-level `mala_search` and handles the closure +
    PlacementCost setup. Returns (best_pos_tensor, best_cost, diag).
    """
    from _langevin_mala import mala_search
    from macro_place.benchmark import Benchmark
    from macro_place.objective import compute_proxy_cost
    from _smooth_proxy import (lse_hpwl_vectorized, build_pin_to_net,
                                  cvar_smooth)
    from _cell_window import (build_window_indices, smooth_density_grid,
                                smooth_macro_blockage,
                                electrostatic_density_energy_normalized)

    bench = Benchmark.load(bench_path)
    canvas_diag = float(np.hypot(bench.canvas_width, bench.canvas_height))
    n_hard = bench.num_hard_macros

    # IncrementalEvaluator at current state
    import importlib.util as ilu
    from pathlib import Path
    HERE = Path(__file__).resolve().parent
    v1_spec = ilu.spec_from_file_location(
        "_v1_mala", str(HERE.parent / "vmallela" / "placer.py"))
    v1 = ilu.module_from_spec(v1_spec); v1_spec.loader.exec_module(v1)
    plc_load = v1._load_plc(bench.name)
    incr = v1.IncrementalEvaluator(plc_load, bench)
    incr.macro_pos[:] = current_pos.cpu().numpy()
    incr._recompute_pin_positions()
    incr._full_recompute_wl()
    incr._full_recompute_density()
    incr._full_recompute_congestion()

    device = torch.device("cpu")
    pin_macro_t = torch.tensor(incr.pin_macro, dtype=torch.long)
    pin_xoff_t = torch.tensor(incr.pin_xoff, dtype=torch.float32)
    pin_yoff_t = torch.tensor(incr.pin_yoff, dtype=torch.float32)
    net_starts_t = torch.tensor(incr.net_starts, dtype=torch.long)
    net_weight_t = torch.tensor(incr.net_weight, dtype=torch.float32)
    macro_w_t = torch.tensor(incr.macro_w, dtype=torch.float32)
    macro_h_t = torch.tensor(incr.macro_h, dtype=torch.float32)
    pin_to_net_t = build_pin_to_net(net_starts_t)
    n_nets = int(net_weight_t.shape[0])
    macro_pos_t = torch.tensor(incr.macro_pos, dtype=torch.float32)
    cell_idx_d, _ = build_window_indices(
        macro_pos_t.detach(), macro_w_t, macro_h_t,
        grid_col=incr.grid_col, grid_row=incr.grid_row,
        grid_w=incr.grid_width, grid_h=incr.grid_height, margin_cells=4)

    cw_f, ch_f = float(incr.cw), float(incr.ch)
    net_cnt = float(incr.net_cnt)

    def smooth_proxy_call(x):
        is_port = (pin_macro_t < 0)
        safe = torch.where(is_port, torch.zeros_like(pin_macro_t), pin_macro_t)
        macro_xy = x[safe]
        pin_x = torch.where(is_port, pin_xoff_t, macro_xy[:, 0] + pin_xoff_t)
        pin_y = torch.where(is_port, pin_yoff_t, macro_xy[:, 1] + pin_yoff_t)
        hpwl = lse_hpwl_vectorized(
            pin_x, pin_y, pin_to_net_t, net_weight_t, n_nets,
            cw=cw_f, ch=ch_f, net_cnt=net_cnt, tau_lse=50.0)
        rho = smooth_density_grid(
            x, macro_w_t, macro_h_t, cell_idx_d,
            incr.grid_col, incr.grid_row, incr.grid_width, incr.grid_height,
            n_cells=incr.n_cells, cell_area=incr.grid_area, mu=100.0)
        density_term = electrostatic_density_energy_normalized(
            rho, incr.grid_row, incr.grid_col,
            grid_w=float(incr.grid_width),
            grid_h=float(incr.grid_height))
        return hpwl + 0.5 * density_term

    def exact_proxy_call(x_np):
        x_t = torch.tensor(x_np, dtype=torch.float32)
        r = compute_proxy_cost(x_t, bench, plc_load)
        return float(r["proxy_cost"]), int(r["overlap_count"])

    best_pos_np, best_cost, diag = mala_search(
        current_pos.detach(), smooth_proxy_call, exact_proxy_call,
        canvas_diag=canvas_diag,
        n_steps=n_steps, step_size_frac=step_size_frac,
        temp_init_frac=temp_init_frac, temp_decay=temp_decay,
        n_burn=n_burn, soft_only=True, n_hard=n_hard,
        cap_displacement_frac=cap_disp_frac,
        verbose=verbose)
    best_pos = torch.tensor(best_pos_np, dtype=torch.float32)
    return best_pos, float(best_cost), diag
