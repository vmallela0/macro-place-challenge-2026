"""Isolate-test of the Hessian phase with differentiable RUDY + subspace HMC.

Loads ibm06 at its .plc starting state, builds the smooth_proxy_call
closure, runs Lanczos top-K, then exercises:
  - smooth_rudy_routing forward + backward gradient flow
  - subspace_hmc_candidates trajectory generation

This is the cheapest possible smoke for the math: ~10-30 s. Avoids
running the full v4 SA pipeline.
"""
from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path

# Self-locked env (matches placer.py production for reproducibility).
for k, v in [
    ("OMP_NUM_THREADS", "1"),
    ("MKL_NUM_THREADS", "1"),
    ("OPENBLAS_NUM_THREADS", "1"),
    ("VECLIB_MAXIMUM_THREADS", "1"),
    ("NUMEXPR_NUM_THREADS", "1"),
    ("PYTHONHASHSEED", "42"),
]:
    os.environ.setdefault(k, v)

ROOT = Path(__file__).resolve().parents[3]
V7 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(V7))
sys.path.insert(0, str(V7.parent))  # for `vmallela.placer` import
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

torch.set_num_threads(4)


def setup_ibm06():
    """Load ibm06, build IncrementalEvaluator + smooth tensors."""
    from macro_place.benchmark import Benchmark
    from macro_place._plc import PlacementCost
    bench = Benchmark.load(str(ROOT / "benchmarks/processed/public/ibm06.pt"))
    plc_path = str(ROOT / "external/MacroPlacement/Testcases/ICCAD04/ibm06")
    plc = PlacementCost(plc_path + "/netlist.pb.txt")
    plc.set_routes_per_micron(plc.hroutes_per_micron, plc.vroutes_per_micron)

    import importlib.util as ilu
    v1_spec = ilu.spec_from_file_location("_v1",
        str(V7.parent / "vmallela" / "placer.py"))
    v1 = ilu.module_from_spec(v1_spec); v1_spec.loader.exec_module(v1)
    plc_loaded = v1._load_plc(bench.name)
    incr = v1.IncrementalEvaluator(plc_loaded, bench)
    incr._recompute_pin_positions()
    incr._full_recompute_wl()
    incr._full_recompute_density()
    incr._full_recompute_congestion()
    return bench, incr


def build_smooth_proxy_call(incr, *, rudy_enabled: bool):
    """Mirror placer._hessian_escape_phase's smooth_proxy_call construction."""
    from _smooth_proxy import (lse_hpwl_vectorized, build_pin_to_net,
                                  cvar_smooth)
    from _cell_window import (build_window_indices, smooth_density_grid,
                                smooth_macro_blockage,
                                electrostatic_density_energy_normalized)
    from _rudy_smooth import (build_net_window_indices_sparse,
                                smooth_rudy_routing_sparse)
    device = torch.device("cpu")
    macro_pos_t = torch.tensor(np.asarray(incr.macro_pos),
                                 dtype=torch.float32, device=device)
    pin_macro_t = torch.tensor(np.asarray(incr.pin_macro),
                                 dtype=torch.long, device=device)
    pin_xoff_t = torch.tensor(np.asarray(incr.pin_xoff),
                                dtype=torch.float32, device=device)
    pin_yoff_t = torch.tensor(np.asarray(incr.pin_yoff),
                                dtype=torch.float32, device=device)
    net_starts_t = torch.tensor(np.asarray(incr.net_starts),
                                  dtype=torch.long, device=device)
    net_weight_t = torch.tensor(np.asarray(incr.net_weight),
                                  dtype=torch.float32, device=device)
    macro_w_t = torch.tensor(np.asarray(incr.macro_w),
                               dtype=torch.float32, device=device)
    macro_h_t = torch.tensor(np.asarray(incr.macro_h),
                               dtype=torch.float32, device=device)
    pin_to_net_t = build_pin_to_net(net_starts_t)
    n_nets = int(net_weight_t.shape[0])
    cell_idx_d, _ = build_window_indices(
        macro_pos_t.detach(), macro_w_t, macro_h_t,
        grid_col=incr.grid_col, grid_row=incr.grid_row,
        grid_w=incr.grid_width, grid_h=incr.grid_height, margin_cells=4)
    cell_idx_c = cell_idx_d  # density and cong share windowing
    cw_f, ch_f = float(incr.cw), float(incr.ch)
    net_cnt = float(incr.net_cnt)
    K_d = max(1, int(0.10 * incr.n_cells))
    K_c = max(1, int(2 * incr.n_cells * 0.05))
    V_smooth_frozen = torch.tensor(np.asarray(incr.V_routing_smooth),
                                      dtype=torch.float32, device=device)
    H_smooth_frozen = torch.tensor(np.asarray(incr.H_routing_smooth),
                                      dtype=torch.float32, device=device)
    grid_v_routes = float(incr.grid_v_routes)
    grid_h_routes = float(incr.grid_h_routes)
    v_alloc = float(np.asarray(incr.vrouting_alloc).mean())
    h_alloc = float(np.asarray(incr.hrouting_alloc).mean())

    # Pin coords at init for RUDY window
    rudy_scale = 1.0
    pair_net_t = pair_cell_t = None
    if rudy_enabled:
        with torch.no_grad():
            _is_port = (pin_macro_t < 0)
            _safe = torch.where(_is_port, torch.zeros_like(pin_macro_t), pin_macro_t)
            _macro_xy = macro_pos_t[_safe]
            pin_x_init = torch.where(_is_port, pin_xoff_t,
                                       _macro_xy[:, 0] + pin_xoff_t)
            pin_y_init = torch.where(_is_port, pin_yoff_t,
                                       _macro_xy[:, 1] + pin_yoff_t)
        pair_net_t, pair_cell_t, n_pairs, n_dropped = \
            build_net_window_indices_sparse(
                pin_x_init, pin_y_init, pin_to_net_t, n_nets,
                incr.grid_col, incr.grid_row,
                incr.grid_width, incr.grid_height,
                margin_cells=4, max_window_cells=256)
        with torch.no_grad():
            V_init, H_init = smooth_rudy_routing_sparse(
                pin_x_init, pin_y_init, pin_to_net_t, net_weight_t,
                n_nets, pair_net_t, pair_cell_t,
                incr.grid_col, incr.grid_row,
                incr.grid_width, incr.grid_height,
                n_cells=incr.n_cells)
            Vp = V_init[V_init > 1e-9]
            Vf = V_smooth_frozen[V_smooth_frozen > 1e-9]
            v_med_rudy = float(Vp.median().item()) if Vp.numel() > 0 else 1.0
            v_med_fr = float(Vf.median().item()) if Vf.numel() > 0 else 1.0
            rudy_scale = (v_med_fr * grid_v_routes / max(v_med_rudy, 1e-9))
        print(f"  RUDY init: n_pairs={n_pairs} dropped={n_dropped} "
              f"V_init.med={v_med_rudy:.4e} V_fr.med={v_med_fr:.4e} "
              f"scale={rudy_scale:.4f}")

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
            grid_w=float(incr.grid_width), grid_h=float(incr.grid_height))
        loss = hpwl + 1.0 * density_term
        V_macro, H_macro = smooth_macro_blockage(
            x, macro_w_t, macro_h_t, cell_idx_c,
            incr.grid_col, incr.grid_row, incr.grid_width, incr.grid_height,
            n_cells=incr.n_cells, vrouting_alloc=v_alloc,
            hrouting_alloc=h_alloc, mu=100.0)
        if rudy_enabled:
            V_rudy, H_rudy = smooth_rudy_routing_sparse(
                pin_x, pin_y, pin_to_net_t, net_weight_t, n_nets,
                pair_net_t, pair_cell_t,
                incr.grid_col, incr.grid_row,
                incr.grid_width, incr.grid_height, n_cells=incr.n_cells)
            V_total = (rudy_scale * V_rudy + V_macro) / max(grid_v_routes, 1e-9)
            H_total = (rudy_scale * H_rudy + H_macro) / max(grid_h_routes, 1e-9)
        else:
            V_total = V_smooth_frozen + V_macro / max(grid_v_routes, 1e-9)
            H_total = H_smooth_frozen + H_macro / max(grid_h_routes, 1e-9)
        combined = torch.cat([V_total, H_total], dim=0)
        with torch.no_grad():
            t_c = torch.quantile(combined, 1.0 - K_c / (2 * incr.n_cells))
        cong_smooth = cvar_smooth(combined.unsqueeze(0), K_c, t_c.detach(),
                                    mu=100.0).squeeze()
        loss = loss + 0.5 * cong_smooth
        return loss

    return macro_pos_t, smooth_proxy_call


def test_grad_flow():
    bench, incr = setup_ibm06()
    print(f"ibm06: n_macros={incr.macro_pos.shape[0]} "
          f"n_hard={incr.n_hard} n_nets={int(incr.net_weight.shape[0])} "
          f"n_cells={incr.n_cells}")
    for rudy in (False, True):
        t0 = time.time()
        macro_pos_t, scall = build_smooth_proxy_call(incr, rudy_enabled=rudy)
        x = macro_pos_t.clone().detach().requires_grad_(True)
        U = scall(x)
        g = torch.autograd.grad(U, x)[0]
        elapsed = time.time() - t0
        norm = float(g.norm())
        print(f"  rudy={int(rudy)}: U={float(U):.6f} ||∇U||={norm:.4e} "
              f"setup+fwd+bwd {elapsed:.2f}s")
        assert torch.isfinite(g).all()


def test_lanczos_and_hmc():
    """Run Lanczos top-K + subspace HMC end-to-end."""
    from _hessian_escape import hessian_min_eigvecs_topk
    from _subspace_hmc import subspace_hmc_candidates
    bench, incr = setup_ibm06()
    canvas_diag = math.hypot(incr.cw, incr.ch)
    for rudy in (False, True):
        print(f"\n=== rudy={rudy} ===")
        macro_pos_t, scall = build_smooth_proxy_call(incr, rudy_enabled=rudy)
        t0 = time.time()
        eigvals, eigvecs = hessian_min_eigvecs_topk(
            scall, macro_pos_t, k=6, n_lanczos_iters=80, tikhonov=1e-4,
            verbose=False)
        t_lanczos = time.time() - t0
        print(f"  Lanczos: λ_top6={eigvals.tolist()} ({t_lanczos:.2f}s)")
        assert eigvecs.shape == (2 * macro_pos_t.shape[0], 6), eigvecs.shape

        t0 = time.time()
        cands, diag = subspace_hmc_candidates(
            macro_pos_t, scall, eigvals, eigvecs,
            n_trajectories=8, n_leapfrog=8, step_size=0.5,
            canvas_diag=canvas_diag, n_hard=incr.n_hard, soft_only=True,
            seed=42, verbose=True)
        print(f"  HMC: generated {len(cands)} candidates in "
              f"{time.time()-t0:.2f}s, wall reported {diag['wall_s']:.2f}s")
        assert len(cands) == 8
        # Surrogate-improving rate
        n_improving = sum(1 for d in diag["trajectories"]
                            if d["delta_U"] < 0 and np.isfinite(d["delta_U"]))
        med_dU = float(np.median([d["delta_U"]
                                    for d in diag["trajectories"]
                                    if np.isfinite(d["delta_U"])]))
        print(f"  surrogate Δ med={med_dU:+.4f}, "
              f"improving {n_improving}/{len(cands)}")


if __name__ == "__main__":
    test_grad_flow()
    test_lanczos_and_hmc()
    print("\nALL ISOLATE TESTS PASSED")
