"""Faster soft CD — pre-compute net pin positions per pass.

Current soft CD recomputes affected net HPWL on every move. The inner
eval is bottlenecked by O(pins per net) work for each probe.

Optimization: inside soft_macro_cd, precompute net centroid (cx, cy)
and radius once per pass. For each probe, estimate cost delta via
distance-to-centroid (analytical-ish). Only full-eval for promising
probes.

Not a correctness change — we still use real cost for acceptance.
"""
import time
import numpy as np


def soft_cd_fast(pos_np, benchmark, incr_eval, max_time, verbose=False):
    """4-direction soft CD with per-pass centroid cache."""
    n_hard = benchmark.num_hard_macros
    n_total = benchmark.macro_positions.shape[0]
    cw, ch = float(benchmark.canvas_width), float(benchmark.canvas_height)
    macro_fixed = benchmark.macro_fixed.numpy() if hasattr(benchmark.macro_fixed, 'numpy') else benchmark.macro_fixed
    soft_movable = [n_hard + i for i in range(n_total - n_hard) if not macro_fixed[n_hard + i]]
    if not soft_movable:
        return pos_np, incr_eval.get_proxy_cost()

    net_count = np.array([len(incr_eval.macro_nets[i]) for i in range(n_total)])
    soft_sorted = sorted(soft_movable, key=lambda i: -net_count[i])

    dirs_4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    deltas = [1.0, 0.5, 0.25, 0.12, 0.06]

    current_cost = incr_eval.get_proxy_cost()
    best_cost = current_cost
    t0 = time.time()
    n_moves = 0
    pass_num = 0

    while time.time() - t0 < max_time:
        pass_num += 1
        improved = False
        # Precompute per-macro target direction (toward net centroid)
        target = {}
        for m in soft_movable:
            cx_sum, cy_sum, cnt = 0.0, 0.0, 0
            for nid in incr_eval.macro_nets[m]:
                start = int(incr_eval.net_starts[nid])
                end = int(incr_eval.net_starts[nid + 1])
                for pidx in range(start, end):
                    pm = int(incr_eval.pin_macro[pidx])
                    if pm == m: continue
                    if pm < 0:
                        cx_sum += float(incr_eval.pin_xoff[pidx])
                        cy_sum += float(incr_eval.pin_yoff[pidx])
                    else:
                        cx_sum += float(incr_eval.macro_pos[pm, 0])
                        cy_sum += float(incr_eval.macro_pos[pm, 1])
                    cnt += 1
            if cnt > 0:
                target[m] = (cx_sum / cnt, cy_sum / cnt)
            else:
                target[m] = (float(incr_eval.macro_pos[m, 0]),
                             float(incr_eval.macro_pos[m, 1]))

        for delta in deltas:
            if time.time() - t0 > max_time:
                break
            for i in soft_sorted:
                if time.time() - t0 > max_time:
                    break
                ox = float(incr_eval.macro_pos[i, 0])
                oy = float(incr_eval.macro_pos[i, 1])

                # Add "toward centroid" direction as a 5th probe candidate
                tx, ty = target[i]
                ddx = tx - ox
                ddy = ty - oy
                d_mag = (ddx * ddx + ddy * ddy) ** 0.5
                directions = list(dirs_4)
                if d_mag > 1e-6:
                    directions.append((ddx / d_mag, ddy / d_mag))

                best_dir_c = current_cost
                best_nx, best_ny = None, None
                for dx, dy in directions:
                    nx = float(max(0, min(cw, ox + delta * dx)))
                    ny = float(max(0, min(ch, oy + delta * dy)))
                    if abs(nx - ox) < 1e-4 and abs(ny - oy) < 1e-4:
                        continue
                    c = incr_eval.move_macro(i, nx, ny)
                    if c < best_dir_c:
                        best_dir_c = c
                        best_nx, best_ny = nx, ny
                    incr_eval.undo_move()
                if best_nx is not None:
                    incr_eval.move_macro(i, best_nx, best_ny)
                    current_cost = best_dir_c
                    if best_dir_c < best_cost:
                        best_cost = best_dir_c
                    n_moves += 1
                    improved = True
        if not improved:
            break

    if verbose:
        print(f"    fast_soft: {pass_num} passes, {n_moves} moves, cost {best_cost:.6f}")
    return pos_np, best_cost
