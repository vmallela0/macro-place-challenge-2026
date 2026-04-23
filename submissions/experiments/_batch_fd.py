"""Batch force-directed soft placement.

Move ALL softs simultaneously toward their net-centroid targets in each
iteration. Then do ONE full recompute of proxy cost. Accept if better,
otherwise halve step and retry.

Each batch step:
  targets[m] = weighted mean of connected pin positions (excluding m)
  new_pos[m] = pos[m] + damping * (targets[m] - pos[m])
  apply moves, do ONE sync_positions, check cost

This is O(1) cost evals per batch, vs O(N_soft) in per-macro CD.
"""
import time
import numpy as np


def batch_fd_soft(pos_np, benchmark, incr_eval, max_time,
                  damping=0.3, verbose=False):
    n_hard = benchmark.num_hard_macros
    n_total = benchmark.macro_positions.shape[0]
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    macro_fixed = benchmark.macro_fixed.numpy() if hasattr(benchmark.macro_fixed, 'numpy') else benchmark.macro_fixed
    soft_movable = np.array([n_hard + i for i in range(n_total - n_hard)
                             if not macro_fixed[n_hard + i]])
    if len(soft_movable) == 0:
        return pos_np, incr_eval.get_proxy_cost()

    incr_eval.sync_positions(pos_np)
    current_cost = incr_eval.get_proxy_cost()
    best_cost = current_cost
    best_state = incr_eval.macro_pos.copy()

    t0 = time.time()
    iteration = 0
    d = damping

    while time.time() - t0 < max_time:
        iteration += 1

        # Compute targets for all softs (vectorised)
        targets = np.zeros((len(soft_movable), 2), dtype=np.float64)
        for k, m in enumerate(soft_movable):
            cx_sum, cy_sum, cnt = 0.0, 0.0, 0
            for nid in incr_eval.macro_nets[m]:
                w = float(incr_eval.net_weight[nid])
                start = int(incr_eval.net_starts[nid])
                end = int(incr_eval.net_starts[nid + 1])
                nc_x, nc_y, pcnt = 0.0, 0.0, 0
                for pidx in range(start, end):
                    pm = int(incr_eval.pin_macro[pidx])
                    if pm == m:
                        continue
                    if pm < 0:
                        nc_x += float(incr_eval.pin_xoff[pidx])
                        nc_y += float(incr_eval.pin_yoff[pidx])
                    else:
                        nc_x += float(incr_eval.macro_pos[pm, 0] + incr_eval.pin_xoff[pidx])
                        nc_y += float(incr_eval.macro_pos[pm, 1] + incr_eval.pin_yoff[pidx])
                    pcnt += 1
                if pcnt > 0:
                    targets[k, 0] += w * nc_x / pcnt
                    targets[k, 1] += w * nc_y / pcnt
            # Normalize by total weight
            wsum = sum(float(incr_eval.net_weight[nid]) for nid in incr_eval.macro_nets[m])
            if wsum > 0:
                targets[k] /= wsum

        # Batch move
        old_positions = [(m, float(incr_eval.macro_pos[m, 0]),
                             float(incr_eval.macro_pos[m, 1])) for m in soft_movable]

        for k, m in enumerate(soft_movable):
            ox, oy = old_positions[k][1], old_positions[k][2]
            nx = ox + d * (targets[k, 0] - ox)
            ny = oy + d * (targets[k, 1] - oy)
            nx = max(0, min(cw, nx))
            ny = max(0, min(ch, ny))
            # Apply direct (bypass move_macro for speed — just update pos, resync after)
            incr_eval.macro_pos[m, 0] = np.float32(nx)
            incr_eval.macro_pos[m, 1] = np.float32(ny)

        # Full recompute
        incr_eval._recompute_pin_positions()
        incr_eval._full_recompute_wl()
        incr_eval._full_recompute_density()
        incr_eval._full_recompute_congestion()
        new_cost = incr_eval.get_proxy_cost()

        if new_cost < best_cost - 1e-7:
            best_cost = new_cost
            best_state = incr_eval.macro_pos.copy()
            current_cost = new_cost
        else:
            # Revert
            for m, ox, oy in old_positions:
                incr_eval.macro_pos[m, 0] = np.float32(ox)
                incr_eval.macro_pos[m, 1] = np.float32(oy)
            incr_eval._recompute_pin_positions()
            incr_eval._full_recompute_wl()
            incr_eval._full_recompute_density()
            incr_eval._full_recompute_congestion()
            # Halve damping
            d *= 0.5
            if d < 0.01:
                break

    if verbose:
        print(f"    batch_fd: {iteration} iters, cost {best_cost:.6f} d_end={d:.4f}")

    return pos_np, best_cost
