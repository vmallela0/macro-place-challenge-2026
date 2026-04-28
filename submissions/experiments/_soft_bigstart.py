"""One-shot: initialize all soft macros at their net centroids, then CD.

Hypothesis: starting softs at net-centroid (HPWL minimizer) gives a
better basin than the initial placement, which may have put them
uniformly. Then CD polishes.
"""
import time
import numpy as np


def soft_bigstart(pos_np, benchmark, incr_eval, verbose=False):
    n_hard = benchmark.num_hard_macros
    n_total = benchmark.macro_positions.shape[0]
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    macro_fixed = benchmark.macro_fixed.numpy() if hasattr(benchmark.macro_fixed, 'numpy') else benchmark.macro_fixed
    soft_movable = [n_hard + i for i in range(n_total - n_hard) if not macro_fixed[n_hard + i]]
    if not soft_movable:
        return pos_np, incr_eval.get_proxy_cost()

    incr_eval.sync_positions(pos_np)
    cost_before = incr_eval.get_proxy_cost()

    # Compute target for each soft: net-weighted centroid of its pins' other endpoints
    old_state = {m: (float(incr_eval.macro_pos[m, 0]), float(incr_eval.macro_pos[m, 1]))
                 for m in soft_movable}

    for m in soft_movable:
        cx_sum, cy_sum, wtotal = 0.0, 0.0, 0.0
        for nid in incr_eval.macro_nets[m]:
            w = float(incr_eval.net_weight[nid])
            start = int(incr_eval.net_starts[nid])
            end = int(incr_eval.net_starts[nid + 1])
            nc_x, nc_y, cnt = 0.0, 0.0, 0
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
                cnt += 1
            if cnt > 0:
                cx_sum += w * nc_x / cnt
                cy_sum += w * nc_y / cnt
                wtotal += w
        if wtotal < 1e-10:
            continue
        tx = cx_sum / wtotal
        ty = cy_sum / wtotal
        tx = max(0, min(cw, tx))
        ty = max(0, min(ch, ty))
        incr_eval.macro_pos[m, 0] = np.float32(tx)
        incr_eval.macro_pos[m, 1] = np.float32(ty)

    # Full recompute
    incr_eval._recompute_pin_positions()
    incr_eval._full_recompute_wl()
    incr_eval._full_recompute_density()
    incr_eval._full_recompute_congestion()
    cost_after = incr_eval.get_proxy_cost()

    if verbose:
        print(f"    bigstart: {cost_before:.6f} -> {cost_after:.6f} "
              f"({'BETTER' if cost_after < cost_before else 'WORSE — rollback'})")

    if cost_after >= cost_before:
        # Roll back
        for m, (ox, oy) in old_state.items():
            incr_eval.macro_pos[m, 0] = np.float32(ox)
            incr_eval.macro_pos[m, 1] = np.float32(oy)
        incr_eval._recompute_pin_positions()
        incr_eval._full_recompute_wl()
        incr_eval._full_recompute_density()
        incr_eval._full_recompute_congestion()
        return pos_np, cost_before
    return pos_np, cost_after
