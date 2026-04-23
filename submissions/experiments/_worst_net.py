"""Worst-net-first soft optimization — iterate nets by current HPWL."""
import time
import numpy as np


def worst_net_soft(pos_np, benchmark, incr_eval, max_time, verbose=False):
    n_hard = benchmark.num_hard_macros
    n_total = benchmark.macro_positions.shape[0]
    cw, ch = float(benchmark.canvas_width), float(benchmark.canvas_height)
    macro_fixed = benchmark.macro_fixed.numpy() if hasattr(benchmark.macro_fixed, 'numpy') else benchmark.macro_fixed

    current_cost = incr_eval.get_proxy_cost()
    best_cost = current_cost
    t0 = time.time()
    n_moves = 0
    pass_num = 0

    while time.time() - t0 < max_time:
        pass_num += 1
        improved = False
        # Sort nets by current HPWL (worst first)
        net_hpwl_arr = incr_eval.net_hpwl
        worst_nets = np.argsort(-net_hpwl_arr)[:500]  # top 500

        for nid in worst_nets:
            if time.time() - t0 > max_time: break
            nid = int(nid)
            # Get softs in this net
            softs = [m for m in incr_eval.net_macros[nid]
                     if m >= n_hard and not macro_fixed[m]]
            if not softs:
                continue

            # Compute median target (other pins)
            start = int(incr_eval.net_starts[nid])
            end = int(incr_eval.net_starts[nid + 1])

            for m in softs:
                xs, ys = [], []
                for pidx in range(start, end):
                    pm = int(incr_eval.pin_macro[pidx])
                    if pm == m: continue
                    if pm < 0:
                        xs.append(float(incr_eval.pin_xoff[pidx]))
                        ys.append(float(incr_eval.pin_yoff[pidx]))
                    else:
                        xs.append(float(incr_eval.macro_pos[pm, 0]))
                        ys.append(float(incr_eval.macro_pos[pm, 1]))
                if not xs:
                    continue
                tx = float(np.median(xs))
                ty = float(np.median(ys))
                ox = float(incr_eval.macro_pos[m, 0])
                oy = float(incr_eval.macro_pos[m, 1])

                for alpha in [1.0, 0.5, 0.2]:
                    nx = float(max(0, min(cw, ox + alpha * (tx - ox))))
                    ny = float(max(0, min(ch, oy + alpha * (ty - oy))))
                    if abs(nx - ox) < 1e-4 and abs(ny - oy) < 1e-4:
                        continue
                    c = incr_eval.move_macro(m, nx, ny)
                    if c < current_cost - 1e-7:
                        current_cost = c
                        if c < best_cost: best_cost = c
                        n_moves += 1
                        improved = True
                        break
                    incr_eval.undo_move()

        if not improved: break

    if verbose:
        print(f"    worst_net: {pass_num} passes, {n_moves} moves, cost {best_cost:.6f}")
    return pos_np, best_cost
