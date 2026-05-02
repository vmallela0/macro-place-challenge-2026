"""v8 phase orchestrator — wires ARC + PT + Riemannian into v7's
IncrementalEvaluator state and the official compute_proxy_cost validator.

Single entry point: `run_v8_phases(self, current_pos, current_cost,
bench_path, ...)`. Returns (best_pos, best_cost) under strict-improvement
gating against current_pos/current_cost.

Each phase is gated by the corresponding flag arg. Phases are sequential:
Phase A (ARC) → Phase B (PT from ARC's best) → Phase C (Riemannian polish).
Each phase's strict-improvement gate keeps the prior best on no-improvement
(so a failing phase never regresses the placement).
"""
from __future__ import annotations
import importlib.util as _ilu
import math
import multiprocessing as _mp
import os
import time
from pathlib import Path
import numpy as np
import torch

from _runlog import log as runlog
from _arc import arc_step, update_M
from _replica_exchange import run_pt, geometric_ladder
from _riemannian import riemannian_descent

_V8_HERE = Path(__file__).resolve().parent
_V7_HERE = _V8_HERE.parent / "vmallela_v7"


def _select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_v1_module():
    """Import the original 'vmallela' placer for IncrementalEvaluator + utils."""
    spec = _ilu.spec_from_file_location(
        "_v1_for_v8", str(_V8_HERE.parent / "vmallela" / "placer.py"))
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _build_smooth_proxy_closure(incr, current_pos_np):
    """Mirror of v7's _hessian_escape_phase smooth-proxy closure
    construction. Returns (smooth_proxy_call, macro_pos_t, n_total, n_hard,
    canvas_diag, device, info).

    The closure takes a torch tensor (n_total, 2) requires_grad, returns a
    scalar loss combining LSE-HPWL + 0.5 * CVaR_smooth(density).
    """
    import sys
    sys.path.insert(0, str(_V7_HERE))
    from _smooth_proxy import (lse_hpwl_vectorized, build_pin_to_net,
                                  cvar_smooth)
    from _cell_window import (build_window_indices, smooth_density_grid)

    device = _select_device()
    n_total = incr.macro_pos.shape[0]
    n_hard = incr.n_hard

    macro_pos_t = torch.tensor(
        np.asarray(incr.macro_pos), dtype=torch.float32, device=device)
    pin_macro_t = torch.tensor(
        np.asarray(incr.pin_macro), dtype=torch.long, device=device)
    pin_xoff_t = torch.tensor(
        np.asarray(incr.pin_xoff), dtype=torch.float32, device=device)
    pin_yoff_t = torch.tensor(
        np.asarray(incr.pin_yoff), dtype=torch.float32, device=device)
    net_starts_t = torch.tensor(
        np.asarray(incr.net_starts), dtype=torch.long, device=device)
    net_weight_t = torch.tensor(
        np.asarray(incr.net_weight), dtype=torch.float32, device=device)
    macro_w_t = torch.tensor(
        np.asarray(incr.macro_w), dtype=torch.float32, device=device)
    macro_h_t = torch.tensor(
        np.asarray(incr.macro_h), dtype=torch.float32, device=device)
    pin_to_net_t = build_pin_to_net(net_starts_t)
    n_nets = int(net_weight_t.shape[0])
    cell_idx_d, _ = build_window_indices(
        macro_pos_t.detach(), macro_w_t, macro_h_t,
        grid_col=incr.grid_col, grid_row=incr.grid_row,
        grid_w=incr.grid_width, grid_h=incr.grid_height,
        margin_cells=4)
    cw_f, ch_f = float(incr.cw), float(incr.ch)
    canvas_diag = math.hypot(cw_f, ch_f)
    net_cnt = float(incr.net_cnt)
    K_d = max(1, int(0.10 * incr.n_cells))

    def smooth_proxy_call(macro_pos_var):
        is_port = (pin_macro_t < 0)
        safe = torch.where(is_port, torch.zeros_like(pin_macro_t), pin_macro_t)
        macro_xy = macro_pos_var[safe]
        pin_x = torch.where(is_port, pin_xoff_t, macro_xy[:, 0] + pin_xoff_t)
        pin_y = torch.where(is_port, pin_yoff_t, macro_xy[:, 1] + pin_yoff_t)
        hpwl = lse_hpwl_vectorized(
            pin_x, pin_y, pin_to_net_t, net_weight_t, n_nets,
            cw=cw_f, ch=ch_f, net_cnt=net_cnt, tau_lse=50.0)
        rho = smooth_density_grid(
            macro_pos_var, macro_w_t, macro_h_t, cell_idx_d,
            incr.grid_col, incr.grid_row,
            incr.grid_width, incr.grid_height,
            n_cells=incr.n_cells, cell_area=incr.grid_area, mu=100.0)
        with torch.no_grad():
            t_d = torch.quantile(rho, 1.0 - K_d / incr.n_cells)
        density_smooth = cvar_smooth(rho.unsqueeze(0), K_d, t_d.detach(),
                                       mu=100.0).squeeze()
        return hpwl + 0.5 * density_smooth

    return (smooth_proxy_call, macro_pos_t, n_total, n_hard,
            canvas_diag, device,
            {"cw": cw_f, "ch": ch_f})


def _make_grad_hvp(smooth_proxy_call, n_total, device):
    """Return (grad_fn, hvp_fn) backed by torch autograd. They take/return
    flat float64 numpy arrays of length 2*n_total.
    """
    def grad_fn(x_flat_np):
        x_t = torch.tensor(
            x_flat_np.reshape(n_total, 2),
            dtype=torch.float32, device=device, requires_grad=True)
        loss = smooth_proxy_call(x_t)
        g = torch.autograd.grad(loss, x_t)[0]
        return g.detach().cpu().numpy().astype(np.float64).reshape(-1)

    def hvp_fn_at(x_flat_np):
        """Build a hvp closure at x_flat_np. Returns hvp(v) -> np."""
        x_t = torch.tensor(
            x_flat_np.reshape(n_total, 2),
            dtype=torch.float32, device=device, requires_grad=True)
        loss = smooth_proxy_call(x_t)
        g = torch.autograd.grad(loss, x_t, create_graph=True)[0]

        def hvp(v_np):
            v_t = torch.tensor(
                v_np.reshape(n_total, 2),
                dtype=torch.float32, device=device)
            gv = (g * v_t).sum()
            Hv = torch.autograd.grad(gv, x_t, retain_graph=True)[0]
            return Hv.detach().cpu().numpy().astype(np.float64).reshape(-1)

        return hvp

    return grad_fn, hvp_fn_at


def _validate_pos_via_v7_worker(pos_np, bench_path, hop_budget, base_seed, label):
    """Run pos through the v7 hessian_candidate_worker (reduced pipeline +
    legalize + exact-cost). Returns (pos, cost, overlaps).
    """
    import sys
    sys.path.insert(0, str(_V7_HERE))
    from _hessian_worker import hessian_candidate_worker

    pos64 = np.ascontiguousarray(pos_np, dtype=np.float64)
    args = (bench_path, pos64.tobytes(), pos64.shape, str(pos64.dtype),
            int(hop_budget), int(base_seed), str(label))
    label_out, pos_bytes, shape, dtype_str, cost, ov = hessian_candidate_worker(args)
    out_pos = np.frombuffer(pos_bytes, dtype=np.dtype(dtype_str)).reshape(shape).copy()
    return out_pos, float(cost), int(ov)


def _validate_pos_exact(pos_np, bench, plc):
    """Quick exact-cost validation without rerunning the v7 pipeline.
    Used inside PT/Riemannian where we only need the proxy of an
    in-place modification to a legal placement.
    """
    from macro_place.objective import compute_proxy_cost
    t = torch.tensor(pos_np.astype(np.float32))
    r = compute_proxy_cost(t, bench, plc)
    return float(r["proxy_cost"]), int(r["overlap_count"])


# ── Phase A: ARC ─────────────────────────────────────────────────────────


def _phase_a_arc(self_obj, current_pos_t, current_cost, bench_path,
                  hop_budget, n_lanczos):
    """Phase A: cubic-regularised step.

    Build smooth surrogate, compute ARC step, dispatch the perturbed
    candidate to the v7 reduced pipeline (legalize + downstream + exact
    cost). On rejection, retry with M *= 2 up to 3 times then give up.
    """
    import sys
    sys.path.insert(0, str(_V7_HERE))
    from macro_place.benchmark import Benchmark
    bench = Benchmark.load(bench_path)
    n_hard = bench.num_hard_macros

    # IncrementalEvaluator at current placement.
    v1 = _load_v1_module()
    plc = v1._load_plc(bench.name)
    incr = v1.IncrementalEvaluator(plc, bench)
    incr.macro_pos[:] = current_pos_t.cpu().numpy()
    incr._recompute_pin_positions()
    incr._full_recompute_wl()
    incr._full_recompute_density()
    incr._full_recompute_congestion()

    (smooth_proxy_call, macro_pos_t, n_total, n_hard_eval,
     canvas_diag, device, _info) = _build_smooth_proxy_closure(
         incr, current_pos_t.cpu().numpy())

    grad_fn, hvp_at = _make_grad_hvp(smooth_proxy_call, n_total, device)
    x0_flat = current_pos_t.cpu().numpy().astype(np.float64).reshape(-1)
    hvp_fn = hvp_at(x0_flat)

    M_init = float(os.environ.get("PLACER_V8_ARC_M_INIT", "1.0"))
    runlog("phase_a", "start",
           f"n_total={n_total} n_hard={n_hard} canvas_diag={canvas_diag:.3f} "
           f"M_init={M_init}")

    t0 = time.time()
    s_full, M_used, lam_min_T, info = arc_step(
        x0_flat, grad_fn, hvp_fn,
        M_init=M_init, k_lanczos=n_lanczos)
    runlog("phase_a", "step_computed",
           f"k_eff={info['k_eff']} lambda_min_T={info['lambda_min_T']:.6f} "
           f"s_norm={info['s_norm']:.3f} pred_dec={info['predicted_decrease']:.6f} "
           f"branch={info.get('sub_branch')} elapsed={time.time()-t0:.1f}s")

    # Zero out hard-macro components — softs only get perturbed (matches v7).
    s_xy = s_full.reshape(n_total, 2).copy()
    s_xy[:n_hard] = 0.0
    perturbed = current_pos_t.cpu().numpy().astype(np.float64) + s_xy

    # Auto-tune via M shrink/grow on rejection (per spec).
    best_pos = current_pos_t.cpu().numpy().astype(np.float64)
    best_cost = float(current_cost)
    accepted_label = None
    M = M_init
    for attempt in range(3):
        try:
            out_pos, out_cost, out_ov = _validate_pos_via_v7_worker(
                perturbed, bench_path, hop_budget,
                self_obj.seed + 8000 + attempt, f"arc_M={M:.3g}")
        except Exception as e:
            runlog("phase_a", "worker_error", f"attempt={attempt} {type(e).__name__}: {e}")
            break

        runlog("phase_a", "candidate_eval",
               f"attempt={attempt} M={M:.3g} cost={out_cost:.6f} "
               f"overlaps={out_ov}")

        if out_ov == 0 and out_cost < best_cost - 1e-7:
            best_pos = out_pos
            best_cost = out_cost
            accepted_label = f"arc_M={M:.3g}_attempt{attempt}"
            runlog("phase_a", "accept",
                   f"{float(current_cost):.6f} -> {best_cost:.6f}  "
                   f"({accepted_label})")
            break

        # Retry with larger M (more conservative step).
        M *= 2.0
        s_xy_new = s_full.reshape(n_total, 2).copy()
        # Approximate effect of M doubling: shorter step. Recompute:
        s_full_retry, _, _, info_retry = arc_step(
            x0_flat, grad_fn, hvp_at(x0_flat),
            M_init=M, k_lanczos=n_lanczos)
        s_xy_new = s_full_retry.reshape(n_total, 2).copy()
        s_xy_new[:n_hard] = 0.0
        perturbed = current_pos_t.cpu().numpy().astype(np.float64) + s_xy_new
        runlog("phase_a", "retry_with_larger_M",
               f"new_M={M:.3g} new_s_norm={info_retry['s_norm']:.3f}")

    if accepted_label is None:
        runlog("phase_a", "no_improvement",
               f"keeping current cost {best_cost:.6f}")
    return best_pos, best_cost


# ── Phase B: Replica Exchange (PT) ───────────────────────────────────────


def _phase_b_pt(self_obj, current_pos_np, current_cost, bench_path,
                  budget_seconds):
    """Phase B: parallel tempering on exact proxy from current_pos.

    n_chains = 8 (default; matches smoke pod's vCPU count). Budget is
    soft-translated to per-chain step count via a calibration: target
    O(200) steps per chain so the whole phase fits in budget.

    Proposal kernel: pick a random soft macro, perturb its (x, y) by
    Gaussian σ ∝ √T · 0.03 · canvas_diag. Hard macros never move.
    """
    import sys
    sys.path.insert(0, str(_V7_HERE))
    from macro_place.benchmark import Benchmark
    bench = Benchmark.load(bench_path)
    n_hard = bench.num_hard_macros
    cw = float(bench.canvas_width)
    ch = float(bench.canvas_height)
    canvas_diag = math.hypot(cw, ch)

    v1 = _load_v1_module()
    plc = v1._load_plc(bench.name)

    # Build energy_fn closure that calls compute_proxy_cost on the
    # whole placement (legality is preserved by the proposal kernel which
    # only perturbs softs; we still validate overlaps per-step).
    def energy_fn(pos_np):
        cost, ov = _validate_pos_exact(pos_np, bench, plc)
        if ov > 0:
            # Penalise overlapping configurations heavily; PT can still
            # explore them but the cold chain will reject on cost.
            return cost + 1e3 + ov * 10.0
        return cost

    sigma_base = 0.03 * canvas_diag

    def proposal_fn(pos, rng, T):
        out = pos.copy()
        # Pick one soft macro
        if pos.shape[0] <= n_hard:
            return out
        idx = int(rng.integers(n_hard, pos.shape[0]))
        sigma = sigma_base * math.sqrt(T)
        out[idx, 0] += rng.normal(0, sigma)
        out[idx, 1] += rng.normal(0, sigma)
        # Clip to canvas (center coords)
        out[idx, 0] = float(np.clip(out[idx, 0], 0.0, cw))
        out[idx, 1] = float(np.clip(out[idx, 1], 0.0, ch))
        return out

    n_chains = int(os.environ.get("PLACER_V8_PT_CHAINS", "8"))
    n_steps = int(os.environ.get("PLACER_V8_PT_STEPS", "200"))
    swap_interval = int(os.environ.get("PLACER_V8_PT_SWAP_INT", "20"))
    T_min = float(os.environ.get("PLACER_V8_PT_TMIN", "0.01"))
    T_max = float(os.environ.get("PLACER_V8_PT_TMAX", "1.0"))
    ladder = geometric_ladder(T_min, T_max, n_chains)

    runlog("phase_b", "start",
           f"chains={n_chains} steps={n_steps} swap_int={swap_interval} "
           f"T_min={T_min} T_max={T_max} sigma_base={sigma_base:.3f} "
           f"current_cost={float(current_cost):.6f}")

    t0 = time.time()
    best_pos, best_E, info = run_pt(
        current_pos_np.astype(np.float64),
        energy_fn=energy_fn, proposal_fn=proposal_fn,
        n_chains=n_chains, temp_ladder=ladder,
        n_steps=n_steps, swap_interval=swap_interval,
        base_seed=self_obj.seed + 9000,
        autotune=True, autotune_after_steps=max(50, n_steps // 4))
    elapsed = time.time() - t0
    runlog("phase_b", "done",
           f"best_E={best_E:.6f} chain_acc={info['chain_acceptance']} "
           f"swap_acc={info['swap_acceptance']} "
           f"autotuned={info['autotuned']} elapsed={elapsed:.1f}s")

    # Validate exactness (PT energy_fn returns penalised value if ov>0).
    cost, ov = _validate_pos_exact(best_pos, bench, plc)
    runlog("phase_b", "validate", f"exact_cost={cost:.6f} overlaps={ov}")

    if ov == 0 and cost < float(current_cost) - 1e-7:
        return best_pos, cost
    runlog("phase_b", "no_improvement",
           f"keep current {float(current_cost):.6f} (best PT {cost:.6f} ov={ov})")
    return current_pos_np, float(current_cost)


# ── Phase C: Riemannian descent ──────────────────────────────────────────


def _phase_c_riemannian(self_obj, current_pos_np, current_cost, bench_path):
    """Phase C: Riemannian descent on smooth surrogate, exact-cost gated.

    Tangent projection at each step zeroes gradient components along
    active no-overlap constraints. Retraction is windowed push-apart.
    """
    import sys
    sys.path.insert(0, str(_V7_HERE))
    from macro_place.benchmark import Benchmark
    bench = Benchmark.load(bench_path)
    n_hard = bench.num_hard_macros
    cw = float(bench.canvas_width)
    ch = float(bench.canvas_height)

    v1 = _load_v1_module()
    plc = v1._load_plc(bench.name)
    incr = v1.IncrementalEvaluator(plc, bench)
    incr.macro_pos[:] = current_pos_np
    incr._recompute_pin_positions()
    incr._full_recompute_wl()
    incr._full_recompute_density()
    incr._full_recompute_congestion()

    (smooth_proxy_call, _macro_pos_t, n_total, n_hard_eval,
     canvas_diag, device, _info) = _build_smooth_proxy_closure(
         incr, current_pos_np)

    macro_w = np.asarray(incr.macro_w, dtype=np.float64)
    macro_h = np.asarray(incr.macro_h, dtype=np.float64)
    mean_size = float(np.mean(np.concatenate([macro_w, macro_h])))
    radius_init = 2.0 * mean_size

    n_steps = int(os.environ.get("PLACER_V8_RIEM_STEPS", "200"))
    eta = float(os.environ.get("PLACER_V8_RIEM_ETA",
                                  str(0.005 * canvas_diag)))

    def grad_fn(p):
        x_t = torch.tensor(p, dtype=torch.float32, device=device,
                            requires_grad=True)
        loss = smooth_proxy_call(x_t)
        g = torch.autograd.grad(loss, x_t)[0]
        return g.detach().cpu().numpy().astype(np.float64)

    def energy_fn(p):
        cost, ov = _validate_pos_exact(p, bench, plc)
        if ov > 0:
            return cost + 1e3
        return cost

    runlog("phase_c", "start",
           f"n_total={n_total} n_hard={n_hard} eta={eta:.3f} "
           f"radius_init={radius_init:.3f} n_steps={n_steps}")

    t0 = time.time()
    best_pos, best_E, info = riemannian_descent(
        current_pos_np.astype(np.float64),
        grad_fn, energy_fn, macro_w, macro_h,
        n_hard=n_hard, eta=eta, radius_init=radius_init,
        canvas_w=cw, canvas_h=ch,
        n_steps=n_steps, autotune_radius=True)
    elapsed = time.time() - t0
    runlog("phase_c", "done",
           f"best_E={best_E:.6f} steps={info['n_steps_done']} "
           f"final_radius={info['final_radius']:.3f} "
           f"accept_rate={info['accept_rate']:.2f} elapsed={elapsed:.1f}s")

    cost, ov = _validate_pos_exact(best_pos, bench, plc)
    runlog("phase_c", "validate", f"exact_cost={cost:.6f} overlaps={ov}")

    if ov == 0 and cost < float(current_cost) - 1e-7:
        return best_pos, cost
    return current_pos_np, float(current_cost)


# ── Top-level orchestrator ────────────────────────────────────────────────


def run_v8_phases(self_obj, current_pos_t, current_cost, bench_path,
                    *, hop_budget, n_lanczos_iters,
                    v8_arc, v8_pt, v8_riem):
    """Sequentially run enabled phases. Returns (pos_tensor_or_array, cost)."""
    cur_pos_np = (current_pos_t.cpu().numpy() if isinstance(current_pos_t, torch.Tensor)
                  else current_pos_t).astype(np.float64)
    cur_cost = float(current_cost)

    if v8_arc:
        try:
            new_pos, new_cost = _phase_a_arc(
                self_obj, current_pos_t, cur_cost, bench_path,
                hop_budget, n_lanczos_iters)
            if new_cost < cur_cost - 1e-7:
                cur_pos_np = new_pos
                cur_cost = new_cost
        except Exception as e:
            runlog("phase_a", "fatal", f"{type(e).__name__}: {e}")

    if v8_pt:
        try:
            new_pos, new_cost = _phase_b_pt(
                self_obj, cur_pos_np, cur_cost, bench_path, hop_budget)
            if new_cost < cur_cost - 1e-7:
                cur_pos_np = new_pos
                cur_cost = new_cost
        except Exception as e:
            runlog("phase_b", "fatal", f"{type(e).__name__}: {e}")

    if v8_riem:
        try:
            new_pos, new_cost = _phase_c_riemannian(
                self_obj, cur_pos_np, cur_cost, bench_path)
            if new_cost < cur_cost - 1e-7:
                cur_pos_np = new_pos
                cur_cost = new_cost
        except Exception as e:
            runlog("phase_c", "fatal", f"{type(e).__name__}: {e}")

    return cur_pos_np, cur_cost
