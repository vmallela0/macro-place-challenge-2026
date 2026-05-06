"""Hessian eigenvector component decomposition.

Given a placement and the smooth surrogate's three components
(HPWL_LSE, CVaR_density, CVaR_congestion), compute the Hessian
v_min for several weight combinations:
  (HPWL=1, D=0, C=0)   — pure HPWL Hessian
  (HPWL=0, D=1, C=0)   — pure density Hessian
  (HPWL=0, D=0, C=1)   — pure congestion Hessian
  (1, 0.5, 0)          — verified-baseline surrogate
  (1, 0.5, 0.5)        — albania1 default cong-on
  (0, 0, 1)            — pure congestion eigvec

For each variant: λ_min, ||v_min||, plus the *projection* of v_min onto
each component's gradient direction. The dominant projection tells us
which component drives the escape direction in that variant.

This is a one-shot computation per placement (~1-2 sec). Useful for
understanding why the cong-aware Hessian DOES or DOESN'T find better
saddles than the HPWL-only baseline.

Run:
    .venv/bin/python research/lower_bounds/hessian_decomp.py --benchmark ibm01
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "submissions" / "vmallela"))
sys.path.insert(0, str(ROOT / "submissions" / "vmallela_v7"))


def _load(bench_name):
    import importlib.util as ilu
    from macro_place.benchmark import Benchmark
    v1_spec = ilu.spec_from_file_location(
        "_v1_d", str(ROOT / "submissions" / "vmallela" / "placer.py"))
    v1 = ilu.module_from_spec(v1_spec)
    v1_spec.loader.exec_module(v1)
    bench = Benchmark.load(
        str(ROOT / "benchmarks" / "processed" / "public" / f"{bench_name}.pt"))
    plc = v1._load_plc(bench.name)
    incr = v1.IncrementalEvaluator(plc, bench)
    return bench, incr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="ibm01")
    args = ap.parse_args()

    bench, incr = _load(args.benchmark)
    print(f"=== {args.benchmark} ===")
    print(f"  {bench}")

    # Build the three surrogate components separately, then test eigvec under
    # different weight combinations.
    from _smooth_proxy import (lse_hpwl_vectorized, build_pin_to_net,
                                 cvar_smooth)
    from _cell_window import (build_window_indices, smooth_density_grid,
                                smooth_macro_blockage)
    from _hessian_escape import hessian_min_eigvec

    device = torch.device("cpu")  # CPU avoids contention with running sweep
    macro_pos_t = torch.tensor(np.asarray(incr.macro_pos), dtype=torch.float32,
                                 device=device)
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
    cw_f, ch_f = float(incr.cw), float(incr.ch)
    net_cnt = float(incr.net_cnt)
    K_d = max(1, int(0.10 * incr.n_cells))
    K_c = max(1, int(2 * incr.n_cells * 0.05))
    cell_idx_d, _ = build_window_indices(
        macro_pos_t.detach(), macro_w_t, macro_h_t,
        grid_col=incr.grid_col, grid_row=incr.grid_row,
        grid_w=incr.grid_width, grid_h=incr.grid_height,
        margin_cells=4)
    V_smooth = torch.tensor(np.asarray(incr.V_routing_smooth),
                              dtype=torch.float32, device=device)
    H_smooth = torch.tensor(np.asarray(incr.H_routing_smooth),
                              dtype=torch.float32, device=device)
    v_alloc = float(np.asarray(incr.vrouting_alloc).mean())
    h_alloc = float(np.asarray(incr.hrouting_alloc).mean())
    grid_v_routes = float(incr.grid_v_routes)
    grid_h_routes = float(incr.grid_h_routes)

    def make_proxy(w_hpwl=1.0, w_dens=0.5, w_cong=0.5):
        def proxy(macro_pos_var):
            is_port = (pin_macro_t < 0)
            safe = torch.where(is_port, torch.zeros_like(pin_macro_t),
                                pin_macro_t)
            macro_xy = macro_pos_var[safe]
            pin_x = torch.where(is_port, pin_xoff_t, macro_xy[:, 0] + pin_xoff_t)
            pin_y = torch.where(is_port, pin_yoff_t, macro_xy[:, 1] + pin_yoff_t)
            loss = torch.zeros((), dtype=macro_pos_var.dtype,
                                device=macro_pos_var.device)
            if w_hpwl != 0:
                hpwl = lse_hpwl_vectorized(
                    pin_x, pin_y, pin_to_net_t, net_weight_t, n_nets,
                    cw=cw_f, ch=ch_f, net_cnt=net_cnt, tau_lse=50.0)
                loss = loss + w_hpwl * hpwl
            if w_dens != 0:
                rho = smooth_density_grid(
                    macro_pos_var, macro_w_t, macro_h_t, cell_idx_d,
                    incr.grid_col, incr.grid_row,
                    incr.grid_width, incr.grid_height,
                    n_cells=incr.n_cells, cell_area=incr.grid_area, mu=100.0)
                with torch.no_grad():
                    t_d = torch.quantile(rho, 1.0 - K_d / incr.n_cells)
                ds = cvar_smooth(rho.unsqueeze(0), K_d, t_d.detach(),
                                   mu=100.0).squeeze()
                loss = loss + w_dens * ds
            if w_cong != 0:
                V_macro, H_macro = smooth_macro_blockage(
                    macro_pos_var, macro_w_t, macro_h_t, cell_idx_d,
                    incr.grid_col, incr.grid_row,
                    incr.grid_width, incr.grid_height,
                    n_cells=incr.n_cells,
                    vrouting_alloc=v_alloc, hrouting_alloc=h_alloc,
                    mu=100.0)
                V_total = V_smooth + V_macro / max(grid_v_routes, 1e-9)
                H_total = H_smooth + H_macro / max(grid_h_routes, 1e-9)
                comb = torch.cat([V_total, H_total], dim=0)
                with torch.no_grad():
                    t_c = torch.quantile(comb, 1.0 - K_c / (2 * incr.n_cells))
                cs = cvar_smooth(comb.unsqueeze(0), K_c, t_c.detach(),
                                   mu=100.0).squeeze()
                loss = loss + w_cong * cs
            return loss
        return proxy

    # Compute eigvec for each variant
    variants = [
        ("HPWL only",    1.0, 0.0, 0.0),
        ("density only", 0.0, 1.0, 0.0),
        ("cong only",    0.0, 0.0, 1.0),
        ("verified base (1, 0.5, 0)",  1.0, 0.5, 0.0),
        ("albania1 default (1, 0.5, 0.5)", 1.0, 0.5, 0.5),
        ("cong-boost (1, 0.5, 1.0)",    1.0, 0.5, 1.0),
        ("cong-dominant (0.5, 0.5, 2.0)", 0.5, 0.5, 2.0),
    ]
    print(f"\n{'variant':<35} {'lambda_min':>14} {'||v_min||':>10} {'wall':>6}")
    eigvecs_by_variant = {}
    for name, wh, wd, wc in variants:
        proxy = make_proxy(wh, wd, wc)
        t0 = time.time()
        try:
            lam, v = hessian_min_eigvec(
                proxy, macro_pos_t, n_lanczos_iters=50, verbose=False)
        except Exception as e:
            print(f"  {name}: FAIL {e}")
            continue
        wall = time.time() - t0
        v_norm = float(np.linalg.norm(v))
        print(f"  {name:<33} {lam:>+14.6f} {v_norm:>10.4f} {wall:>5.2f}s")
        eigvecs_by_variant[name] = (lam, v)

    # Cross-projections: how aligned are eigvecs across variants?
    print(f"\n=== Eigvec alignment (cosine similarity) ===")
    names = list(eigvecs_by_variant.keys())
    print(f"{'':<35} " + " ".join(f"{n[:10]:>11}" for n in names))
    for ni in names:
        _, vi = eigvecs_by_variant[ni]
        if np.linalg.norm(vi) < 1e-12:
            continue
        vi_n = vi / np.linalg.norm(vi)
        row = []
        for nj in names:
            _, vj = eigvecs_by_variant[nj]
            if np.linalg.norm(vj) < 1e-12:
                row.append("nan")
                continue
            vj_n = vj / np.linalg.norm(vj)
            cos = float(abs(vi_n @ vj_n))   # |abs| because eigvec sign is ambiguous
            row.append(f"{cos:.3f}")
        print(f"  {ni:<33} " + " ".join(f"{r:>11}" for r in row))


if __name__ == "__main__":
    main()
