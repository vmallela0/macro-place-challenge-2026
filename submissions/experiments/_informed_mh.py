"""Metropolis-Hastings with net-centroid-biased proposals for soft macros.

Key difference from v17 (uniform SA) and v6 (pure CD):
  - Propose: sample target from Gaussian around net-weighted centroid.
    σ = scaled by temperature, anneal σ over time.
  - Accept: Metropolis. Always accept improvement; accept degradation
    with p = exp(-ΔC / T_cost).

This is "guided" SA. Uniform SA wasted budget on junk moves; this biases
toward moves that are *likely* improving but allows the occasional
uphill move to escape local minima.
"""
import time
import math
import numpy as np


def mh_informed_soft(pos_np, benchmark, incr_eval, max_time,
                     t_cost_start=0.005, t_cost_end=5e-5,
                     sigma_start=1.0, sigma_end=0.2,
                     verbose=False):
    n_hard = benchmark.num_hard_macros
    n_total = benchmark.macro_positions.shape[0]
    cw, ch = float(benchmark.canvas_width), float(benchmark.canvas_height)
    macro_fixed = benchmark.macro_fixed.numpy() if hasattr(benchmark.macro_fixed, 'numpy') else benchmark.macro_fixed
    soft_movable = [n_hard + i for i in range(n_total - n_hard) if not macro_fixed[n_hard + i]]
    if not soft_movable:
        return pos_np, incr_eval.get_proxy_cost()

    net_count = np.array([len(incr_eval.macro_nets[i]) for i in range(n_total)])
    soft_sorted = sorted(soft_movable, key=lambda i: -net_count[i])

    current_cost = incr_eval.get_proxy_cost()
    best_cost = current_cost
    best_pos = incr_eval.macro_pos.copy()

    rng = np.random.RandomState(777)
    t0 = time.time()
    n_attempt = 0
    n_accept_down = 0
    n_accept_up = 0
    pass_num = 0

    while time.time() - t0 < max_time:
        pass_num += 1
        progress = min(1.0, (time.time() - t0) / max_time)
        T = t_cost_start * (t_cost_end / t_cost_start) ** progress
        sigma = sigma_start * (sigma_end / sigma_start) ** progress
        improved = False

        for i in soft_sorted:
            if time.time() - t0 > max_time:
                break

            # Compute net-centroid
            cx_sum, cy_sum, cnt = 0.0, 0.0, 0
            for nid in incr_eval.macro_nets[i][:8]:
                start = int(incr_eval.net_starts[nid])
                end = int(incr_eval.net_starts[nid + 1])
                for pidx in range(start, end):
                    pm = int(incr_eval.pin_macro[pidx])
                    if pm == i:
                        continue
                    if pm < 0:
                        cx_sum += float(incr_eval.pin_xoff[pidx])
                        cy_sum += float(incr_eval.pin_yoff[pidx])
                    else:
                        cx_sum += float(incr_eval.macro_pos[pm, 0])
                        cy_sum += float(incr_eval.macro_pos[pm, 1])
                    cnt += 1
            if cnt > 0:
                ncx = cx_sum / cnt
                ncy = cy_sum / cnt
            else:
                ncx = float(incr_eval.macro_pos[i, 0])
                ncy = float(incr_eval.macro_pos[i, 1])

            # Proposal: Gaussian around net centroid
            ox = float(incr_eval.macro_pos[i, 0])
            oy = float(incr_eval.macro_pos[i, 1])
            nx = ncx + rng.randn() * sigma
            ny = ncy + rng.randn() * sigma
            nx = float(max(0, min(cw, nx)))
            ny = float(max(0, min(ch, ny)))
            if abs(nx - ox) < 1e-4 and abs(ny - oy) < 1e-4:
                continue

            c = incr_eval.move_macro(i, nx, ny)
            dC = c - current_cost
            n_attempt += 1

            if dC < 0:
                current_cost = c
                if c < best_cost:
                    best_cost = c
                    best_pos = incr_eval.macro_pos.copy()
                n_accept_down += 1
                improved = True
            else:
                if rng.random() < math.exp(-dC / max(T, 1e-10)):
                    current_cost = c
                    n_accept_up += 1
                else:
                    incr_eval.undo_move()

        if not improved and n_accept_up == 0:
            break

    # Restore best
    if best_cost < current_cost - 1e-7:
        for i in soft_movable:
            incr_eval.move_macro(i, float(best_pos[i, 0]), float(best_pos[i, 1]))

    if verbose:
        print(f"    mh_informed: {n_attempt} attempts, {n_accept_down} down, "
              f"{n_accept_up} up, best {best_cost:.6f}")
    return pos_np, best_cost
