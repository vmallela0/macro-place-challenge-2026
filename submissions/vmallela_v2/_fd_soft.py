"""Analytical force-directed soft-macro placement.

Soft macros in ibm benchmarks are zero-size stdcell cluster centroids.
They contribute to HPWL (directly through net bboxes) and to congestion
(through routing). They DON'T contribute to density (zero area).

Key identity for HPWL gradient: for a pin at (x,y) in a net with bbox
[x_min, x_max] × [y_min, y_max]:
  d HPWL_net / dx = +1 if this pin is the unique x_max, -1 if unique x_min, 0 otherwise.

Minimizing weighted HPWL alone, the force on a macro from a net is the
vector from macro to net-centroid (if we model it as sum-of-squares) or
to the median of other pins (if HPWL).

For a soft macro m, net centroid gives good direction but not optimal. We
approximate with net-centroid (weighted mean of other pin positions) and
move with damping — iterate until no improvement via incremental evaluator.

After each FD iteration, we verify proxy cost went down. If not, undo.
"""
import time
import numpy as np


def fd_soft_place(pos_np, benchmark, incr_eval, max_time,
                  damping=0.5, check_cost=True, verbose=False):
    """Force-directed soft macro placement using net-centroid attraction.

    pos_np is hard-macro positions (unchanged). Softs live in incr_eval.macro_pos.

    check_cost: if True, verify each move reduces proxy cost; undo if not.
                if False, accept all FD moves (faster but may overshoot).
    """
    n_hard = benchmark.num_hard_macros
    n_total = benchmark.macro_positions.shape[0]
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)

    macro_fixed = benchmark.macro_fixed.numpy() if hasattr(benchmark.macro_fixed, 'numpy') else benchmark.macro_fixed
    soft_movable = [n_hard + i for i in range(n_total - n_hard) if not macro_fixed[n_hard + i]]
    if not soft_movable:
        if verbose:
            print(f"    fd_soft: no softs")
        return pos_np, incr_eval.get_proxy_cost()

    # Keep incr_eval synced to pos_np for hards (softs are already inside)
    incr_eval.sync_positions(pos_np)
    current_cost = incr_eval.get_proxy_cost()
    best_cost = current_cost
    best_pos = incr_eval.macro_pos.copy()

    net_starts = incr_eval.net_starts
    pin_macro = incr_eval.pin_macro
    pin_xoff = incr_eval.pin_xoff
    pin_yoff = incr_eval.pin_yoff
    net_weight = incr_eval.net_weight

    t0 = time.time()
    iteration = 0
    n_moves = 0
    n_undone = 0

    while time.time() - t0 < max_time:
        iteration += 1
        any_improved = False

        # Shuffle for stochasticity
        order = list(soft_movable)
        np.random.shuffle(order)

        for m in order:
            if time.time() - t0 > max_time:
                break
            # Compute net-centroid weighted target
            cx_sum, cy_sum, w_sum = 0.0, 0.0, 0.0
            for nid in incr_eval.macro_nets[m]:
                w = float(net_weight[nid])
                start = int(net_starts[nid])
                end = int(net_starts[nid + 1])
                nc_x, nc_y, cnt = 0.0, 0.0, 0
                for pidx in range(start, end):
                    pin_m = int(pin_macro[pidx])
                    if pin_m == m:
                        continue
                    if pin_m < 0:
                        # Port: pin_xoff, pin_yoff store absolute port pos
                        nc_x += float(pin_xoff[pidx])
                        nc_y += float(pin_yoff[pidx])
                    else:
                        nc_x += float(incr_eval.macro_pos[pin_m, 0] + pin_xoff[pidx])
                        nc_y += float(incr_eval.macro_pos[pin_m, 1] + pin_yoff[pidx])
                    cnt += 1
                if cnt > 0:
                    cx_sum += w * nc_x / cnt
                    cy_sum += w * nc_y / cnt
                    w_sum += w

            if w_sum < 1e-10:
                continue

            target_x = cx_sum / w_sum
            target_y = cy_sum / w_sum
            ox = float(incr_eval.macro_pos[m, 0])
            oy = float(incr_eval.macro_pos[m, 1])

            # Damped step
            new_x = ox + damping * (target_x - ox)
            new_y = oy + damping * (target_y - oy)
            new_x = max(0.0, min(cw, new_x))
            new_y = max(0.0, min(ch, new_y))

            if abs(new_x - ox) < 1e-5 and abs(new_y - oy) < 1e-5:
                continue

            if check_cost:
                c = incr_eval.move_macro(m, new_x, new_y)
                if c < current_cost - 1e-8:
                    current_cost = c
                    any_improved = True
                    n_moves += 1
                    if c < best_cost:
                        best_cost = c
                        best_pos = incr_eval.macro_pos.copy()
                else:
                    incr_eval.undo_move()
                    n_undone += 1
            else:
                incr_eval.move_macro(m, new_x, new_y)
                n_moves += 1
                any_improved = True

        if check_cost and not any_improved:
            break

    final_cost = incr_eval.get_proxy_cost()
    if verbose:
        print(f"    fd_soft: {iteration} iters, {n_moves} moves, "
              f"{n_undone} undone, cost {final_cost:.6f} (best {best_cost:.6f})")

    return pos_np, best_cost
