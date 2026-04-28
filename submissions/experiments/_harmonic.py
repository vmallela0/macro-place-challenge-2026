"""Harmonic oscillator init — each soft attached by spring to its net centroid.

Solve coupled harmonic system analytically: equilibrium at centroid of net
centroids. Closed-form O(N) initialization.
"""
import numpy as np


def harmonic_init_softs(benchmark, incr_eval, iterations=50):
    n_hard = benchmark.num_hard_macros
    n_total = benchmark.macro_positions.shape[0]
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    macro_fixed = benchmark.macro_fixed.numpy() if hasattr(benchmark.macro_fixed, 'numpy') else benchmark.macro_fixed

    # Iterative Jacobi: each iteration, set soft position to weighted centroid
    # of net-connected pins. Converges as fixed point of linear system.
    soft_movable = [n_hard + i for i in range(n_total - n_hard) if not macro_fixed[n_hard + i]]
    if not soft_movable:
        return incr_eval.get_proxy_cost()

    for it in range(iterations):
        new_x = {}
        new_y = {}
        for m in soft_movable:
            cx_sum, cy_sum, wtotal = 0.0, 0.0, 0.0
            for nid in incr_eval.macro_nets[m]:
                w = float(incr_eval.net_weight[nid])
                start = int(incr_eval.net_starts[nid])
                end = int(incr_eval.net_starts[nid + 1])
                cnt = 0
                nc_x, nc_y = 0.0, 0.0
                for pidx in range(start, end):
                    pm = int(incr_eval.pin_macro[pidx])
                    if pm == m: continue
                    if pm < 0:
                        nc_x += float(incr_eval.pin_xoff[pidx])
                        nc_y += float(incr_eval.pin_yoff[pidx])
                    else:
                        nc_x += float(incr_eval.macro_pos[pm, 0])
                        nc_y += float(incr_eval.macro_pos[pm, 1])
                    cnt += 1
                if cnt > 0:
                    cx_sum += w * nc_x / cnt
                    cy_sum += w * nc_y / cnt
                    wtotal += w
            if wtotal > 1e-10:
                new_x[m] = max(0.0, min(cw, cx_sum / wtotal))
                new_y[m] = max(0.0, min(ch, cy_sum / wtotal))

        # Apply simultaneously
        for m, x in new_x.items():
            incr_eval.macro_pos[m, 0] = np.float32(x)
            incr_eval.macro_pos[m, 1] = np.float32(new_y[m])

    incr_eval._recompute_pin_positions()
    incr_eval._full_recompute_wl()
    incr_eval._full_recompute_density()
    incr_eval._full_recompute_congestion()
    return incr_eval.get_proxy_cost()
